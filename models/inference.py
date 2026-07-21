"""
统一灾害推理引擎

提供四灾害一键预测接口，供 Gradio app 和 LLM tools 共同调用。

用法:
    from models.inference import DisasterInference

    engine = DisasterInference()                # 自动加载/训练模型
    result = engine.predict("2025-08-15", "flash_flood")
    # → {"risk": np.array, "lat": np.array, "lon": np.array,
    #    "n_high": int, "mean_risk": float, ...}

    result = engine.predict_batch("2025-08-15", ["flash_flood", "extreme_heat"])
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config.model_config import (
    DISASTER_FEATURES, DISASTER_THRESHOLDS, DISASTER_LABELS,
    LIGHTGBM_PARAMS, TRAIN_START, TRAIN_END, FEATURE_PHYSICS,
)
from config.settings import PROJECT_ROOT

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def add_latlon_features(df: pd.DataFrame) -> pd.DataFrame:
    """为 DataFrame 添加经纬度 sin/cos 周期编码（就地修改）。"""
    if "lat_sin" in df.columns:  # 已有则不重复添加
        return df
    # 兼容两种格式：MultiIndex 或 flat columns
    if "latitude" in df.columns:
        lat, lon = df["latitude"].values, df["longitude"].values
    else:
        lat = df.index.get_level_values("latitude")
        lon = df.index.get_level_values("longitude")
    df["lat_sin"] = np.sin(np.radians(lat))
    df["lat_cos"] = np.cos(np.radians(lat))
    df["lon_sin"] = np.sin(np.radians(lon))
    df["lon_cos"] = np.cos(np.radians(lon))
    return df


def _prepare_features(df: pd.DataFrame, disaster_type: str) -> pd.DataFrame:
    """准备入模特征：添加经纬度编码 + 筛选特征列 + 填 NaN。

    对于模型训练时使用但当前数据中缺失的特征，填 0 代替（丢失的信号 = 无信号）。
    """
    df = add_latlon_features(df.copy())
    feats = DISASTER_FEATURES[disaster_type]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"  ⚠ [{disaster_type}] {len(missing)} 个特征不在数据中，填 0: {missing}")
        for f in missing:
            df[f] = 0.0
    X = df[feats].fillna(0).astype(np.float32)
    return X


# ═══════════════════════════════════════════════
# 推理引擎
# ═══════════════════════════════════════════════

class DisasterInference:
    """四灾害统一推理引擎。

    - 首次使用自动训练并保存模型到 outputs/models/
    - 后续使用直接加载已保存的模型
    - 自动处理经纬度编码、特征筛选、阈值判定
    """

    def __init__(self, train_if_missing: bool = True):
        self.models: Dict[str, object] = {}
        self._ensure_models(train_if_missing)

    # ── 模型加载/训练 ──

    def _ensure_models(self, train: bool):
        missing = []
        for disaster in DISASTER_FEATURES:
            path = os.path.join(OUTPUT_DIR, f"{disaster}.pkl")
            if os.path.exists(path):
                self.models[disaster] = self._load(path, disaster)
            else:
                missing.append(disaster)
        if missing and train:
            print(f"[Inference] 缺失 {len(missing)} 个模型: {missing}，开始训练...")
            self.train_all()

    def _load(self, path: str, disaster_type: str):
        from models.lightgbm_model import LightGBMDisasterModel
        model = LightGBMDisasterModel(disaster_type)
        model.load(path)
        return model

    def train_all(self, save: bool = True):
        """训练全部四灾害模型并保存。"""
        from data.loader import load_date_range
        from data.label_builder import DisasterLabelBuilder

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 加载标签构建器所需数据
        label_vars = ["flash_flood_risk", "heatwave_day_flag", "tmax_c",
                       "wind10_speed", "rh2m", "vpd_kpa", "orography", "ivt"]
        all_feat_vars = list(set(
            f for f_list in DISASTER_FEATURES.values() for f in f_list))
        # 过滤掉 lat/lon 编码（动态添加）
        load_vars = list(set(
            [v for v in all_feat_vars + label_vars
             if v not in ("lat_sin", "lat_cos", "lon_sin", "lon_cos", "sst_celsius")]))

        print(f"[Inference] 加载训练数据 ({TRAIN_START}~{TRAIN_END})...")
        ds = load_date_range(TRAIN_START, TRAIN_END, variables=load_vars, show_progress=True)
        df = ds.to_dataframe().fillna(0)

        # 标签构建
        builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
        builder.fit(df)
        labels = builder.build_all(df)

        # 训练各灾害
        for disaster in DISASTER_FEATURES:
            print(f"\n[Inference] 训练 {disaster}...")
            X = _prepare_features(df, disaster)

            if disaster == "flash_flood":
                y = (df["flash_flood_risk"] >= 1).astype(int).values
            elif disaster == "extreme_heat":
                y = df["heatwave_day_flag"].astype(int).values
            elif disaster == "dust_wind":
                y = labels["dust_wind_label"].values
            elif disaster == "coastal_wave":
                y = labels["coastal_wave_label"].values

            from models.lightgbm_model import LightGBMDisasterModel
            model = LightGBMDisasterModel(disaster)
            model.fit(X, y)
            self.models[disaster] = model

            if save:
                path = os.path.join(OUTPUT_DIR, f"{disaster}.pkl")
                model.save(path)
                print(f"  已保存 → {path}")

        print("\n[Inference] 全部模型训练完成")

    # ── 预测接口 ──

    def predict(
        self,
        df: pd.DataFrame,
        disaster_type: str,
        return_labels: bool = False,
    ) -> Dict:
        """对单个灾害进行预测。

        Args:
            df: 包含气象变量的 DataFrame（需含 latitude/longitude 列或 MultiIndex）
            disaster_type: 灾害类型
            return_labels: 是否返回二分类标签（否则返回概率）

        Returns:
            {"risk": np.array, "lat": np.array, "lon": np.array,
             "n_high": int, "mean_risk": float, "max_risk": float}
        """
        if disaster_type not in self.models:
            raise ValueError(f"未知灾害类型: {disaster_type}")

        X = _prepare_features(df, disaster_type)
        model = self.models[disaster_type]
        proba = model.predict_proba(X)
        threshold = DISASTER_THRESHOLDS.get(disaster_type, 0.5)

        if return_labels:
            risk = (proba >= threshold).astype(int).astype(float)
        else:
            risk = proba

        # 提取坐标
        if "latitude" in df.columns:
            lat, lon = df["latitude"].values, df["longitude"].values
        else:
            lat = df.index.get_level_values("latitude")
            lon = df.index.get_level_values("longitude")

        n_high = int((proba >= threshold).sum())
        return {
            "risk": risk,
            "proba": proba,
            "lat": np.array(lat),
            "lon": np.array(lon),
            "n_total": len(risk),
            "n_high": n_high,
            "high_pct": round(n_high / len(risk) * 100, 1),
            "mean_risk": round(float(np.mean(proba)), 4),
            "max_risk": round(float(np.max(proba)), 4),
            "threshold": threshold,
            "disaster_type": disaster_type,
        }

    def explain(
        self,
        df: pd.DataFrame,
        disaster_type: str,
        top_n: int = 5,
    ) -> Dict:
        """对预测结果进行 SHAP 可解释性分析。

        针对全局平均高风险格点的特征贡献进行解释。

        Args:
            df: 包含气象变量的 DataFrame
            disaster_type: 灾害类型
            top_n: 返回 Top-N 重要特征

        Returns:
            {
                "top_features": [{"feature": "cape", "name": "对流有效位能",
                                   "contribution": 0.38, "unit": "J/kg",
                                   "description": "CAPE越高对流越旺盛"}],
                "summary": "本次高风险主要由于 CAPE 异常偏高(贡献38%)和强降水(贡献25%)",
                "n_samples_used": int
            }
        """
        import shap

        X = _prepare_features(df, disaster_type)
        model_obj = self.models[disaster_type]
        model = model_obj.model  # LightGBM Booster
        feats = DISASTER_FEATURES[disaster_type]

        # 获取高风险样本（概率 ≥ 阈值）
        threshold = DISASTER_THRESHOLDS.get(disaster_type, 0.5)
        proba = model_obj.predict_proba(X)
        high_mask = proba >= threshold
        n_high = high_mask.sum()

        if n_high == 0:
            return {
                "top_features": [],
                "summary": "无高风险格点，无法计算 SHAP 解释",
                "n_samples_used": 0,
            }

        # 对高风险样本抽样（最多 500 个，避免 SHAP 太慢）
        high_indices = np.where(high_mask)[0]
        n_sample = min(500, len(high_indices))
        sampled_idx = np.random.default_rng(42).choice(
            high_indices, size=n_sample, replace=False)
        X_high = X.iloc[sampled_idx] if hasattr(X, 'iloc') else X[sampled_idx]

        # SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_high)

        # LightGBM binary: shap_values shape = (n_samples, n_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 正类 SHAP

        # 平均绝对 SHAP 值 → 特征贡献排序
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feat_contrib = sorted(
            zip(feats, mean_abs_shap),
            key=lambda x: x[1], reverse=True
        )

        # 构建 Top-N
        total = sum(c for _, c in feat_contrib) or 1.0
        top_features = []
        for feat, contrib in feat_contrib[:top_n]:
            info = FEATURE_PHYSICS.get(feat, (feat, "", ""))
            pct = round(contrib / total * 100, 1)
            top_features.append({
                "feature": feat,
                "name": info[0],
                "contribution": pct,
                "unit": info[1],
                "description": info[2],
            })

        # 生成自然语言摘要
        parts = []
        for f in top_features[:3]:
            parts.append(f"{f['name']}(贡献{f['contribution']}%)")
        summary = f"本次{disaster_type}高风险主要由于: " + "、".join(parts)

        return {
            "top_features": top_features,
            "summary": summary,
            "n_samples_used": n_sample,
            "n_high_total": int(n_high),
        }

    def predict_trend(
        self,
        end_date: str,
        disaster_type: str,
        lookback_days: int = 7,
    ) -> Dict:
        """预测时序趋势：过去 N 天 → 今天，展示风险演变。

        Args:
            end_date: 截止日期 (YYYY-MM-DD)
            disaster_type: 灾害类型
            lookback_days: 回溯天数

        Returns:
            {
                "trend": [{"date": "2025-08-21", "n_high": 1234, "mean_risk": 0.05}, ...],
                "direction": "上升" | "下降" | "平稳",
                "change_pct": 变化百分比,
                "peak_date": 峰值日期,
            }
        """
        from data.loader import load_date_range
        from datetime import datetime, timedelta

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # 非 2025 年日期回退到 2025 年同月同日
        original_date = end_date
        if end_dt.year != 2025:
            end_dt = end_dt.replace(year=2025)
            end_date = end_dt.strftime("%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days - 1)
        start_str = start_dt.strftime("%Y-%m-%d")

        feats = DISASTER_FEATURES[disaster_type]
        label_vars = ["flash_flood_risk", "heatwave_day_flag",
                       "wind10_speed", "rh2m", "vpd_kpa", "orography"]
        load_vars = list(set(
            [f for f in feats
             if f not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")]
            + label_vars
        ))

        try:
            ds = load_date_range(start_str, end_date, variables=load_vars,
                                 show_progress=False)
        except Exception:
            return {"trend": [], "direction": "无数据", "change_pct": 0,
                    "peak_date": "", "error": "数据加载失败"}

        trend = []
        for day in sorted(ds["day"].values):
            date_str = str(day)[:10]
            day_ds = ds.sel(day=day)
            df = day_ds.to_dataframe().fillna(0).reset_index()
            r = self.predict(df, disaster_type)
            trend.append({
                "date": date_str,
                "n_total": r["n_total"],
                "n_high": r["n_high"],
                "high_pct": r["high_pct"],
                "mean_risk": r["mean_risk"],
                "max_risk": r["max_risk"],
            })

        # 判断趋势方向
        if len(trend) >= 2:
            first = trend[0]["mean_risk"]
            last = trend[-1]["mean_risk"]
            change = (last - first) / max(first, 0.001) * 100
            if change > 20:
                direction = "↑ 快速上升"
            elif change > 5:
                direction = "↗ 缓慢上升"
            elif change < -20:
                direction = "↓ 快速下降"
            elif change < -5:
                direction = "↘ 缓慢下降"
            else:
                direction = "→ 平稳"
        else:
            change = 0
            direction = "—"

        peak = max(trend, key=lambda x: x["mean_risk"]) if trend else {"date": "", "mean_risk": 0}

        return {
            "trend": trend,
            "direction": direction,
            "change_pct": round(change, 1),
            "peak_date": peak["date"],
            "peak_mean_risk": peak["mean_risk"],
            "fallback_note": f"以{end_date}为参考" if original_date != end_date else "",
        }

    def predict_from_nc(
        self,
        date: str,
        disaster_type: str,
    ) -> Dict:
        """从 NC 文件直接加载并预测。

        如果请求日期的 NC 文件不存在，自动回退到 2025 年同月同日。

        Args:
            date: 日期 (YYYY-MM-DD)
            disaster_type: 灾害类型
        """
        from data.loader import load_date_range

        feats = DISASTER_FEATURES[disaster_type]
        load_vars = [f for f in feats
                     if f not in ("lat_sin", "lat_cos", "lon_sin", "lon_cos")]

        try:
            ds = load_date_range(date, date, variables=load_vars, show_progress=False)
            fallback_note = ""
        except (FileNotFoundError, ValueError, OSError):
            # 回退到 2025 年同月同日
            from datetime import datetime
            try:
                dt = datetime.strptime(date, "%Y-%m-%d")
                fallback_date = f"2025-{dt.month:02d}-{dt.day:02d}"
            except ValueError:
                raise FileNotFoundError(f"无法解析日期: {date}，且 2025 回退也失败")

            ds = load_date_range(fallback_date, fallback_date, variables=load_vars, show_progress=False)
            fallback_note = f"（注：{date} 无数据，已使用 2025 年同日 {fallback_date} 作为参考）"

        df = ds.to_dataframe().fillna(0)
        result = self.predict(df, disaster_type)

        if fallback_note:
            result["fallback_note"] = fallback_note
            result["original_date"] = date
            result["actual_date"] = fallback_date

        return result

    def predict_batch(
        self,
        date: str,
        disaster_types: List[str],
    ) -> Dict[str, Dict]:
        """批量预测多个灾害。"""
        results = {}
        for dt in disaster_types:
            try:
                results[dt] = self.predict_from_nc(date, dt)
            except Exception as e:
                results[dt] = {"error": str(e)}
        return results

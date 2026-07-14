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
    LIGHTGBM_PARAMS, TRAIN_START, TRAIN_END,
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
    """准备入模特征：添加经纬度编码 + 筛选特征列 + 填 NaN。"""
    df = add_latlon_features(df.copy())
    feats = DISASTER_FEATURES[disaster_type]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(
            f"[{disaster_type}] 缺少特征列: {missing}\n"
            f"可用列: {list(df.columns)}\n"
            f"请确认 load_date_range 时加载了所需变量。")
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

    def predict_from_nc(
        self,
        date: str,
        disaster_type: str,
    ) -> Dict:
        """从 NC 文件直接加载并预测（便捷接口）。

        Args:
            date: 日期 (YYYY-MM-DD)
            disaster_type: 灾害类型
        """
        from data.loader import load_date_range

        feats = DISASTER_FEATURES[disaster_type]
        load_vars = [f for f in feats
                     if f not in ("lat_sin", "lat_cos", "lon_sin", "lon_cos")]

        ds = load_date_range(date, date, variables=load_vars, show_progress=False)
        df = ds.to_dataframe().fillna(0)

        return self.predict(df, disaster_type)

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

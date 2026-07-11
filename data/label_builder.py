"""
灾害标签构建器

根据物理指标和阈值，为四类灾害构建 0/1 标签。

用法:
    from data.label_builder import DisasterLabelBuilder
    from data.loader import load_date_range

    # 加载数据
    df = load_date_range("2025-06-01", "2025-08-31").to_dataframe()

    # 构建标签
    builder = DisasterLabelBuilder()
    builder.fit(df)  # 在训练集上计算阈值
    labels = builder.build_all(df)  # → DataFrame with label columns

设计原则:
- fit/transform 模式：阈值从训练集计算，避免数据泄露
- 支持 simple / standard / enhanced 三种构建模式
- flash_flood 和 extreme_heat 使用数据中已有的原生标签
- dust_wind 和 coastal_wave 从气象变量物理推理构建
"""

import numpy as np
import pandas as pd
from typing import Dict

from config.disaster_config import get_label_config


class DisasterLabelBuilder:
    """四类灾害标签构建器。

    采用 fit/transform 模式：
    - fit(): 在训练数据上计算分位数阈值
    - build_*(): 应用阈值构建标签
    - build_all(): 一次性构建全部四类标签
    """

    def __init__(
        self,
        dust_mode: str = "standard",
        coastal_mode: str = "standard",
        random_state: int = 42,
    ):
        """
        Args:
            dust_mode: 沙尘构建模式 — "simple" / "standard" / "enhanced"
            coastal_mode: 风浪构建模式 — "simple" / "standard" / "enhanced"
            random_state: 随机种子
        """
        self.dust_mode = dust_mode
        self.coastal_mode = coastal_mode
        self.random_state = random_state

        # 存储 fit 阶段计算的分位数阈值
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._is_fitted = False

        # 统计信息
        self._label_stats: Dict[str, Dict] = {}

    # ================================================================
    # fit: 在训练数据上计算所有阈值
    # ================================================================

    def fit(self, df: pd.DataFrame) -> "DisasterLabelBuilder":
        """在训练数据上计算分位数阈值。

        应对所有后续 transform 使用的数据调用 fit；
        阈值仅从训练集计算，以保持时间因果性。

        Args:
            df: 训练数据 DataFrame，需包含所有气象变量列

        Returns:
            self
        """
        print("=" * 60)
        print("  DisasterLabelBuilder.fit() — 计算分位数阈值")
        print("=" * 60)

        # ---- 沙尘强风阈值 ----
        self._fit_dust_wind(df)

        # ---- 沿海风浪阈值 ----
        self._fit_coastal_wave(df)

        # ---- 极端高温阈值（备选构建用） ----
        self._fit_extreme_heat(df)

        self._is_fitted = True
        print("fit 完成\n")
        return self

    def _fit_dust_wind(self, df: pd.DataFrame) -> None:
        """计算沙尘强风的分位数阈值。"""
        config = get_label_config("dust_wind")
        mode = self.dust_mode
        thresholds_config = config["thresholds"][mode]

        p_wind = thresholds_config["wind10_speed_percentile"]
        wind_vals = df["wind10_speed"].values
        wind_vals = wind_vals[~np.isnan(wind_vals)]
        wind_threshold = np.percentile(wind_vals, p_wind)

        self._thresholds["dust_wind"] = {
            "wind10_speed_p": wind_threshold,
            "wind_percentile_used": p_wind,
            "mode": mode,
        }

        if mode == "standard":
            # RH 使用绝对值阈值（物理常数：30% 是沙尘扬起的典型门槛）
            self._thresholds["dust_wind"]["rh2m_abs_max"] = thresholds_config.get(
                "rh2m_absolute_max", 30.0
            )
            p_vpd = thresholds_config.get("vpd_kpa_percentile", 80)
            if "vpd_kpa" in df.columns:
                vpd_vals = df["vpd_kpa"].values
                vpd_vals = vpd_vals[~np.isnan(vpd_vals)]
                self._thresholds["dust_wind"]["vpd_kpa_p"] = np.percentile(vpd_vals, p_vpd)

        if mode == "simple":
            self._thresholds["dust_wind"]["rh2m_abs_max"] = 30.0

        if mode == "enhanced":
            # RH 和 VPD 都用分位数
            p_rh = thresholds_config["rh2m_percentile"]
            rh_vals = df["rh2m"].values
            rh_vals = rh_vals[~np.isnan(rh_vals)]
            self._thresholds["dust_wind"]["rh2m_p"] = np.percentile(rh_vals, p_rh)

            p_vpd = thresholds_config["vpd_kpa_percentile"]
            vpd_vals = df["vpd_kpa"].values
            vpd_vals = vpd_vals[~np.isnan(vpd_vals)]
            self._thresholds["dust_wind"]["vpd_kpa_p"] = np.percentile(vpd_vals, p_vpd)

        if mode == "enhanced":
            p_shear = thresholds_config["wind_shear_percentile"]
            for shear_var in ["wind_shear_850_300", "wind_shear_850_200"]:
                if shear_var in df.columns:
                    sv = df[shear_var].values
                    sv = sv[~np.isnan(sv)]
                    self._thresholds["dust_wind"][f"{shear_var}_p"] = np.percentile(sv, p_shear)
                    break

        print(f"  [dust_wind / {mode}] wind10_speed P{p_wind} = {wind_threshold:.2f} m/s", end="")
        if "rh2m_abs_max" in self._thresholds["dust_wind"]:
            print(f", rh2m < {self._thresholds['dust_wind']['rh2m_abs_max']:.0f}%", end="")
        if "vpd_kpa_p" in self._thresholds["dust_wind"]:
            print(f", vpd P{thresholds_config.get('vpd_kpa_percentile', '?')} = {self._thresholds['dust_wind']['vpd_kpa_p']:.2f} kPa", end="")
        print()

    def _fit_coastal_wave(self, df: pd.DataFrame) -> None:
        """计算沿海风浪的分位数阈值。"""
        config = get_label_config("coastal_wave")
        mode = self.coastal_mode
        thresholds_config = config["thresholds"][mode]

        # 识别沿海格点
        coastal_mask = self._identify_coastal(df)

        # 风阈值 — 在沿海格点上计算
        p_wind = thresholds_config["wind10_speed_percentile"]
        coastal_wind = df.loc[coastal_mask, "wind10_speed"].values
        coastal_wind = coastal_wind[~np.isnan(coastal_wind)]
        wind_threshold = np.percentile(coastal_wind, p_wind) if len(coastal_wind) > 0 else 10.0

        self._thresholds["coastal_wave"] = {
            "wind10_speed_p": wind_threshold,
            "wind_percentile_used": p_wind,
            "mode": mode,
            "n_coastal_cells": int(coastal_mask.sum()),
            "coastal_ratio": float(coastal_mask.mean()),
        }

        if mode in ("standard", "enhanced"):
            if "sst_celsius" in df.columns:
                sst_vals = df.loc[coastal_mask, "sst_celsius"].values
                sst_vals = sst_vals[~np.isnan(sst_vals)]
                if len(sst_vals) > 0:
                    self._thresholds["coastal_wave"]["sst_median"] = float(np.median(sst_vals))

            p_ivt = thresholds_config.get("ivt_percentile", 75)
            if "ivt" in df.columns:
                ivt_vals = df.loc[coastal_mask, "ivt"].values
                ivt_vals = ivt_vals[~np.isnan(ivt_vals)]
                if len(ivt_vals) > 0:
                    self._thresholds["coastal_wave"]["ivt_p"] = np.percentile(ivt_vals, p_ivt)

        print(f"  [coastal_wave / {mode}] 沿海格点: {coastal_mask.sum()} ({coastal_mask.mean()*100:.1f}%), "
              f"沿海wind10 P{p_wind} = {wind_threshold:.2f} m/s", end="")
        if "sst_median" in self._thresholds["coastal_wave"]:
            print(f", SST median = {self._thresholds['coastal_wave']['sst_median']:.1f}°C", end="")
        print()

    def _fit_extreme_heat(self, df: pd.DataFrame) -> None:
        """为极端高温备选构建计算阈值（当原生标签不可用时）。"""
        config = get_label_config("extreme_heat")
        rules = config.get("build_rules", {})

        if "tmax_c" in df.columns:
            tmax = df["tmax_c"].values
            tmax = tmax[~np.isnan(tmax)]
            p = rules.get("tmax_percentile", 90)
            self._thresholds["extreme_heat"] = {
                "tmax_p": np.percentile(tmax, p),
                "mode": "build_fallback",
            }

    # ================================================================
    # build_*: 各类灾害标签构建
    # ================================================================

    def build_flash_flood(self, df: pd.DataFrame) -> pd.Series:
        """构建暴雨山洪标签。

        使用数据中原生的 flash_flood_risk 指标。
        flash_flood_risk: 0 (无) / 1 (低) / 2 (中) / 3 (高)
        标签: >= 1 → 1 (有风险)

        Returns:
            pd.Series, 0/1 标签
        """
        config = get_label_config("flash_flood")

        if config["label_column"] not in df.columns:
            raise KeyError(
                f"数据中缺少 {config['label_column']} 列，无法构建 flash_flood 标签"
            )

        raw = df[config["label_column"]]
        label = (raw >= config["threshold"]).astype(int)

        pos_pct = label.mean() * 100
        print(f"  [flash_flood] 正样本率: {pos_pct:.3f}% ({label.sum():,} / {len(label):,})")
        self._label_stats["flash_flood"] = {
            "positive_rate": pos_pct,
            "n_positive": int(label.sum()),
            "n_total": len(label),
        }
        return label

    def build_extreme_heat(self, df: pd.DataFrame) -> pd.Series:
        """构建极端高温标签。

        优先使用数据中原生的 heatwave_day_flag。
        如果不可用，使用 tmax > P90 代理构建。

        Returns:
            pd.Series, 0/1 标签
        """
        config = get_label_config("extreme_heat")

        if config["label_column"] in df.columns:
            # 使用原生标签
            raw = df[config["label_column"]]
            label = (raw > config["threshold"]).astype(int)
            source = "native"
        else:
            # 代理构建
            if not self._is_fitted:
                raise RuntimeError("请先调用 fit() 计算阈值")
            tmax_p = self._thresholds.get("extreme_heat", {}).get("tmax_p", 40)
            if "tmax_c" not in df.columns:
                raise KeyError("数据中缺少 tmax_c 列，无法构建 extreme_heat 标签")
            label = (df["tmax_c"] > tmax_p).astype(int)
            source = f"proxy (tmax > P90 = {tmax_p:.1f}°C)"
            print(f"  [extreme_heat] 原生标签不可用，使用代理: tmax > {tmax_p:.1f}°C")

        pos_pct = label.mean() * 100
        print(f"  [extreme_heat / {source}] 正样本率: {pos_pct:.3f}% ({label.sum():,} / {len(label):,})")
        self._label_stats["extreme_heat"] = {
            "positive_rate": pos_pct,
            "n_positive": int(label.sum()),
            "n_total": len(label),
            "source": source,
        }
        return label

    def build_dust_wind(self, df: pd.DataFrame) -> pd.Series:
        """构建沙尘强风标签。

        基于物理推理：强风 + 干燥大气 → 沙尘暴风险。
        支持三种模式：
        - simple:   wind10_speed > P95 AND rh2m < 30%
        - standard: wind10_speed > P90 AND (rh2m < 30% OR vpd_kpa > P80)
        - enhanced: wind + dryness + wind_shear 综合评分 >= 2

        Returns:
            pd.Series, 0/1 标签
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 计算阈值")

        t = self._thresholds.get("dust_wind", {})
        mode = t.get("mode", self.dust_mode)

        # 检查必要变量
        if "wind10_speed" not in df.columns:
            raise KeyError("数据中缺少 wind10_speed 列")

        wind = df["wind10_speed"].values

        if mode == "simple":
            wind_ok = wind > t["wind10_speed_p"]
            dry_ok = df["rh2m"].values < t["rh2m_abs_max"] if "rh2m" in df.columns else np.ones(len(df), dtype=bool)
            label = (wind_ok & dry_ok).astype(int)

        elif mode == "standard":
            wind_ok = wind > t["wind10_speed_p"]
            # RH 用绝对值阈值（物理常数 30%），VPD 用分位数
            rh_dry = df["rh2m"].values < t.get("rh2m_abs_max", 30.0) if "rh2m" in df.columns else np.zeros(len(df), dtype=bool)
            vpd_high = df["vpd_kpa"].values > t["vpd_kpa_p"] if "vpd_kpa" in df.columns and "vpd_kpa_p" in t else np.zeros(len(df), dtype=bool)
            dry_ok = rh_dry | vpd_high
            label = (wind_ok & dry_ok).astype(int)

        elif mode == "enhanced":
            # 综合评分
            scores = np.zeros(len(df), dtype=int)
            scores += (wind > t["wind10_speed_p"]).astype(int)

            if "rh2m" in df.columns and "vpd_kpa" in df.columns:
                dry = (df["rh2m"].values < t["rh2m_p"]) | (df["vpd_kpa"].values > t["vpd_kpa_p"])
                scores += dry.astype(int)

            # 风切变
            shear_ok = np.zeros(len(df), dtype=bool)
            for sv in ["wind_shear_850_300", "wind_shear_850_200"]:
                if sv in df.columns and f"{sv}_p" in t:
                    shear_ok = shear_ok | (df[sv].values > t[f"{sv}_p"])
            scores += shear_ok.astype(int)

            label = (scores >= 2).astype(int)

        else:
            raise ValueError(f"未知模式: {mode}")

        pos_pct = label.mean() * 100
        print(f"  [dust_wind / {mode}] 正样本率: {pos_pct:.3f}% ({label.sum():,} / {len(label):,})")
        self._label_stats["dust_wind"] = {
            "positive_rate": pos_pct,
            "n_positive": int(label.sum()),
            "n_total": len(label),
            "mode": mode,
            "thresholds": {k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()},
        }
        return pd.Series(label, index=df.index, name="dust_wind_label")

    def build_coastal_wave(self, df: pd.DataFrame) -> pd.Series:
        """构建沿海风浪标签。

        基于物理推理：沿海格点 + 强海面风 + 高海温/水汽 → 风浪灾害。
        支持三种模式：
        - simple:   coastal AND wind10_speed > P90
        - standard: coastal AND wind10 > P85 AND (SST > median OR ivt > P75)
        - enhanced: coastal + wind + SST异常 + IVT 综合评分 >= 3

        Returns:
            pd.Series, 0/1 标签
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 计算阈值")

        t = self._thresholds.get("coastal_wave", {})
        mode = t.get("mode", self.coastal_mode)

        # 识别沿海格点
        is_coastal = self._identify_coastal(df)
        wind_ok = df["wind10_speed"].values > t["wind10_speed_p"]

        if mode == "simple":
            label = (is_coastal & wind_ok).astype(int)

        elif mode == "standard":
            sst_warm = np.zeros(len(df), dtype=bool)
            if "sst_celsius" in df.columns and "sst_median" in t:
                sst_vals = df["sst_celsius"].values
                sst_warm = ~np.isnan(sst_vals) & (sst_vals > t["sst_median"])

            ivt_high = np.zeros(len(df), dtype=bool)
            if "ivt" in df.columns and "ivt_p" in t:
                ivt_high = df["ivt"].values > t["ivt_p"]

            ocean_ok = sst_warm | ivt_high
            label = (is_coastal & wind_ok & ocean_ok).astype(int)

        elif mode == "enhanced":
            scores = np.zeros(len(df), dtype=int)
            scores += is_coastal.astype(int)
            scores += wind_ok.astype(int)

            if "sst_celsius" in df.columns and "sst_median" in t:
                sst = df["sst_celsius"].values
                sst_warm = ~np.isnan(sst) & (sst > t["sst_median"])
                scores += sst_warm.astype(int)

            if "ivt" in df.columns and "ivt_p" in t:
                scores += (df["ivt"].values > t["ivt_p"]).astype(int)

            label = (scores >= 3).astype(int)

        else:
            raise ValueError(f"未知模式: {mode}")

        pos_pct = label.mean() * 100
        coastal_pct = is_coastal.mean() * 100
        print(f"  [coastal_wave / {mode}] 沿海格点: {coastal_pct:.1f}%, "
              f"正样本率: {pos_pct:.3f}% ({label.sum():,} / {len(label):,})")
        self._label_stats["coastal_wave"] = {
            "positive_rate": pos_pct,
            "n_positive": int(label.sum()),
            "n_total": len(label),
            "coastal_ratio": coastal_pct,
            "mode": mode,
            "thresholds": {k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()},
        }
        return pd.Series(label, index=df.index, name="coastal_wave_label")

    # ================================================================
    # build_all: 一次性构建全部标签
    # ================================================================

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次性构建全部四类灾害标签。

        Args:
            df: 包含气象变量的 DataFrame（通常从 loader 获取）

        Returns:
            DataFrame，列: flash_flood_label, extreme_heat_label,
                           dust_wind_label, coastal_wave_label
        """
        print("\n" + "=" * 60)
        print("  构建四类灾害标签")
        print("=" * 60)

        labels = pd.DataFrame(index=df.index)

        labels["flash_flood_label"] = self.build_flash_flood(df)
        labels["extreme_heat_label"] = self.build_extreme_heat(df)
        labels["dust_wind_label"] = self.build_dust_wind(df)
        labels["coastal_wave_label"] = self.build_coastal_wave(df)

        print("-" * 60)
        total_pos = (labels.sum(axis=0) / len(labels) * 100)
        print(f"  标签汇总 (正样本率):")
        for col in labels.columns:
            print(f"    {col}: {total_pos[col]:.3f}%")
        print("=" * 60 + "\n")

        return labels

    # ================================================================
    # 辅助方法
    # ================================================================

    def _identify_coastal(self, df: pd.DataFrame) -> np.ndarray:
        """识别沿海格点。

        策略（按优先级）:
        1. sst_celsius 非 NaN → 海洋/沿海格点（最可靠）
        2. orography < 100m → 低海拔近海
        3. 两者结合

        Returns:
            bool 数组, True = 沿海格点
        """
        coastal = np.zeros(len(df), dtype=bool)

        # 方法1: SST 非 NaN
        if "sst_celsius" in df.columns:
            sst = df["sst_celsius"].values
            # SST 可能有多维 (time, lat, lon) → 取第一个 time 步
            if sst.ndim > 1:
                sst = sst[:, 0] if sst.shape[1] > 0 else sst[:, 0]
            coastal = coastal | (~np.isnan(sst))

        # 方法2: 低海拔
        if "orography" in df.columns and coastal.sum() == 0:
            # 只在 SST 不可用时使用
            oro = df["orography"].values
            coastal = oro < self._thresholds.get("coastal_wave", {}).get(
                "orography_max", 100.0
            )
        elif "orography" in df.columns:
            oro = df["orography"].values
            coastal = coastal | (oro < 100.0)

        return coastal

    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """返回 fit 阶段计算的所有阈值。"""
        return self._thresholds.copy()

    def get_stats(self) -> Dict[str, Dict]:
        """返回各灾害标签的统计信息。"""
        return self._label_stats.copy()

    def print_summary(self) -> None:
        """打印标签构建摘要。"""
        print("\n" + "=" * 60)
        print("  DisasterLabelBuilder — 标签构建摘要")
        print("=" * 60)

        for disaster, stats in self._label_stats.items():
            config = get_label_config(disaster)
            print(f"\n  {config['name_cn']} ({disaster})")
            print(f"    构建方式: {stats.get('mode', stats.get('source', 'native'))}")
            print(f"    正样本数: {stats['n_positive']:,} / {stats['n_total']:,} "
                  f"({stats['positive_rate']:.3f}%)")

        print("\n" + "=" * 60)

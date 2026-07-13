"""
时序特征工程：滑动窗口统计、累积指标、趋势特征

对每个格点沿时间轴构建：
- N日滑动累积量（降水、辐射等）
- 连续极端天数（高温、干旱等）
- 变化趋势（CAPE升降、湿度下降率等）
- 滚动窗口统计（均值、最大值、标准差等）

设计原则：
- 对每个 (lat, lon) 格点独立计算，保证时间因果性
- 仅使用过去 N 天的信息（不含未来）
- 与 data/loader.py 的 DataFrame 产出格式兼容

用法:
    from data.loader import load_to_dataframe
    from features.temporal_features import TemporalFeatureBuilder

    df = load_to_dataframe("2025-06-01", "2025-08-31", variables=[...])

    builder = TemporalFeatureBuilder(windows=[3, 5, 7])
    df_features = builder.build(df)
    df_full = pd.concat([df, df_features], axis=1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple


class TemporalFeatureBuilder:
    """时序特征构建器。

    对每个 (lat, lon) 格点，沿 day 轴滑动计算衍生特征。
    所有计算仅使用过去信息（不含未来），保证时间因果性。
    """

    def __init__(
        self,
        windows: List[int] = [3, 5, 7],
        precip_cols: Optional[List[str]] = None,
        temp_cols: Optional[List[str]] = None,
        wind_cols: Optional[List[str]] = None,
        humidity_cols: Optional[List[str]] = None,
        instability_cols: Optional[List[str]] = None,
        include_trend: bool = True,
        include_extreme_days: bool = True,
        min_days_for_window: int = 2,
    ):
        """
        Args:
            windows: 滑动窗口大小列表（天数），如 [3, 5, 7]
            precip_cols: 降水相关变量列名，None 则自动检测
            temp_cols: 温度相关变量列名
            wind_cols: 风速相关变量列名
            humidity_cols: 湿度相关变量列名
            instability_cols: 对流不稳定相关变量列名
            include_trend: 是否计算趋势特征（N日变化量）
            include_extreme_days: 是否计算连续极端天数
            min_days_for_window: 窗口内最少有效天数（低于此值则不计算）
        """
        self.windows = sorted(windows)
        self.include_trend = include_trend
        self.include_extreme_days = include_extreme_days
        self.min_days_for_window = min_days_for_window

        # 变量分组（用于自动选择计算策略）
        self.precip_cols = precip_cols or []
        self.temp_cols = temp_cols or []
        self.wind_cols = wind_cols or []
        self.humidity_cols = humidity_cols or []
        self.instability_cols = instability_cols or []

        # 自动检测的列
        self._auto_detected: Dict[str, List[str]] = {}

        # 记录生成的特征名称
        self._generated_features: List[str] = []

        # 阈值（从数据中 fit 得到）
        self._thresholds: Dict[str, float] = {}

    # ================================================================
    # fit: 计算分位数阈值（用于连续极端天数判定）
    # ================================================================

    def fit(self, df: pd.DataFrame) -> "TemporalFeatureBuilder":
        """在训练数据上计算极端事件阈值。

        Args:
            df: 训练数据 DataFrame

        Returns:
            self
        """
        print("=" * 60)
        print("  TemporalFeatureBuilder.fit() — 计算极端阈值")
        print(f"  窗口: {self.windows} 天")
        print("=" * 60)

        # 自动检测列分组
        self._auto_detect_columns(df)

        # 计算极端事件阈值（使用 P90/P95）
        for label, cols in [
            ("high_temp", self.temp_cols),
            ("strong_wind", self.wind_cols),
        ]:
            for col in cols:
                if col in df.columns:
                    vals = df[col].dropna().values
                    if len(vals) > 100:
                        self._thresholds[f"{col}_p90"] = float(
                            np.percentile(vals, 90)
                        )
                        self._thresholds[f"{col}_p95"] = float(
                            np.percentile(vals, 95)
                        )

        print(f"  已计算 {len(self._thresholds)} 个极端阈值")
        print("  fit 完成\n")
        return self

    def _auto_detect_columns(self, df: pd.DataFrame) -> None:
        """根据列名自动分组变量。"""
        all_cols = set(df.columns)

        if not self.precip_cols:
            self.precip_cols = [
                c for c in all_cols
                if any(kw in c.lower() for kw in ["precip", "rain", "snow"])
            ]
            self._auto_detected["precip"] = self.precip_cols

        if not self.temp_cols:
            self.temp_cols = [
                c for c in all_cols
                if any(kw in c.lower() for kw in ["t2m", "tmax", "tmin", "temp", "heat"])
            ]
            self._auto_detected["temp"] = self.temp_cols

        if not self.wind_cols:
            self.wind_cols = [
                c for c in all_cols
                if any(kw in c.lower() for kw in ["wind", "jet", "gust"])
            ]
            self._auto_detected["wind"] = self.wind_cols

        if not self.humidity_cols:
            self.humidity_cols = [
                c for c in all_cols
                if any(kw in c.lower() for kw in ["rh", "vpd", "humidity", "dewpoint", "pwat"])
            ]
            self._auto_detected["humidity"] = self.humidity_cols

        if not self.instability_cols:
            self.instability_cols = [
                c for c in all_cols
                if any(kw in c.lower() for kw in ["cape", "cin", "shear", "lifted"])
            ]
            self._auto_detected["instability"] = self.instability_cols

        print("  [自动检测变量分组]")
        for group, cols in self._auto_detected.items():
            print(f"    {group}: {len(cols)} 个变量")

    # ================================================================
    # build: 构建全部时序特征
    # ================================================================

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建全部时序衍生特征。

        Args:
            df: 含 day, latitude, longitude 列的 DataFrame

        Returns:
            仅包含新特征列的 DataFrame
        """
        if "day" not in df.columns:
            raise KeyError("DataFrame 必须包含 'day' 列")

        # 确保 day 为 datetime 并按格点+时间排序
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["day"]):
            df["day"] = pd.to_datetime(df["day"])
        df = df.sort_values(["latitude", "longitude", "day"]).reset_index(drop=True)

        # 自动检测列分组（如果还没做）
        if not any([self.precip_cols, self.temp_cols, self.wind_cols,
                     self.humidity_cols, self.instability_cols]):
            self._auto_detect_columns(df)

        features_list = []

        print(f"\n  [时序特征构建] 窗口: {self.windows}")

        # 1. N日累积特征
        accum = self._build_accumulation_features(df)
        if accum is not None and len(accum.columns) > 0:
            features_list.append(accum)

        # 2. N日滚动统计
        rolling = self._build_rolling_features(df)
        if rolling is not None and len(rolling.columns) > 0:
            features_list.append(rolling)

        # 3. 变化趋势（N日变化量）
        if self.include_trend:
            trend = self._build_trend_features(df)
            if trend is not None and len(trend.columns) > 0:
                features_list.append(trend)

        # 4. 连续极端天数
        if self.include_extreme_days:
            extreme = self._build_extreme_day_features(df)
            if extreme is not None and len(extreme.columns) > 0:
                features_list.append(extreme)

        # 5. 特殊灾害相关特征
        special = self._build_special_features(df)
        if special is not None and len(special.columns) > 0:
            features_list.append(special)

        # 合并
        if features_list:
            result = pd.concat(features_list, axis=1)
        else:
            result = pd.DataFrame(index=df.index)

        self._generated_features = list(result.columns)
        print(f"  生成 {len(self._generated_features)} 个时序特征\n")

        return result

    # ================================================================
    # 累积特征
    # ================================================================

    def _build_accumulation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建 N日累积量特征。

        对降水、辐射等累积型变量，计算滑动窗口内的总和。
        """
        accum_cols = self.precip_cols + [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["radiation", "evap", "runoff"])
        ]
        accum_cols = [c for c in accum_cols if c in df.columns]
        if not accum_cols:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)

        for col in accum_cols:
            for w in self.windows:
                feat_name = f"{col}_sum_{w}d"
                features[feat_name] = (
                    df.groupby(["latitude", "longitude"])[col]
                    .transform(lambda x: x.rolling(w, min_periods=self.min_days_for_window).sum())
                    .values
                )

        return features

    # ================================================================
    # 滚动统计特征
    # ================================================================

    def _build_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建 N日滚动窗口统计量。

        对温度、风速、湿度等变量计算均值、最大、最小。
        """
        roll_cols = (
            self.temp_cols + self.wind_cols + self.humidity_cols
        )
        roll_cols = [c for c in roll_cols if c in df.columns]
        if not roll_cols:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)

        for col in roll_cols:
            for w in self.windows:
                grp = df.groupby(["latitude", "longitude"])[col]

                # 均值
                features[f"{col}_mean_{w}d"] = (
                    grp.transform(lambda x: x.rolling(w, min_periods=self.min_days_for_window).mean())
                )
                # 最大值
                features[f"{col}_max_{w}d"] = (
                    grp.transform(lambda x: x.rolling(w, min_periods=self.min_days_for_window).max())
                )
                # 最小值
                features[f"{col}_min_{w}d"] = (
                    grp.transform(lambda x: x.rolling(w, min_periods=self.min_days_for_window).min())
                )

        return features

    # ================================================================
    # 趋势特征
    # ================================================================

    def _build_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建变化趋势特征。

        对关键变量计算短期变化：
        - 1日变化量（今日-昨日）
        - N日变化量
        - 变化率
        """
        trend_cols = (
            self.temp_cols + self.instability_cols + self.humidity_cols + self.wind_cols
        )
        trend_cols = [c for c in trend_cols if c in df.columns]
        if not trend_cols:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)

        for col in trend_cols[:10]:  # 限制数量避免特征爆炸
            grp = df.groupby(["latitude", "longitude"])[col]

            # 1日变化
            features[f"{col}_diff_1d"] = grp.transform(lambda x: x.diff(1))

            # N日变化
            for w in self.windows:
                features[f"{col}_diff_{w}d"] = grp.transform(
                    lambda x: x.diff(w)
                )

            # 变化速度（N日内线性回归斜率 → 简化：用差值/天数）
            if self.windows:
                w_max = max(self.windows)
                # (当前值 - N天前值) / N
                diff = grp.transform(lambda x: x.diff(w_max)).values
                features[f"{col}_trend_{w_max}d"] = diff / w_max

        return features

    # ================================================================
    # 连续极端天数
    # ================================================================

    def _build_extreme_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建连续极端天数特征。

        - 连续高温天数（t2m > P90）
        - 连续无降水天数
        - 连续强风天数
        """
        features = pd.DataFrame(index=df.index)

        # 连续无降水天数
        precip_candidates = [c for c in self.precip_cols if "total" in c.lower()]
        precip_col = (
            precip_candidates[0] if precip_candidates
            else self.precip_cols[0] if self.precip_cols
            else None
        )
        if precip_col and precip_col in df.columns:
            features["consecutive_dry_days"] = (
                df.groupby(["latitude", "longitude"])[precip_col]
                .transform(self._count_consecutive_zeros)
            )

        # 连续高温天数（如果有阈值）
        high_temp_candidates = [c for c in self.temp_cols if "max" in c.lower()]
        high_temp_col = (
            high_temp_candidates[0] if high_temp_candidates
            else self.temp_cols[0] if self.temp_cols
            else None
        )
        if high_temp_col and high_temp_col in df.columns:
            threshold = self._thresholds.get(f"{high_temp_col}_p90", 40.0)
            features["consecutive_hot_days"] = (
                df.groupby(["latitude", "longitude"])[high_temp_col]
                .transform(lambda x: self._count_consecutive_above(x, threshold))
            )

        # 连续强风天数
        wind_col = self.wind_cols[0] if self.wind_cols else None
        if wind_col and wind_col in df.columns:
            threshold = self._thresholds.get(f"{wind_col}_p90", 15.0)
            features["consecutive_windy_days"] = (
                df.groupby(["latitude", "longitude"])[wind_col]
                .transform(lambda x: self._count_consecutive_above(x, threshold))
            )

        return features

    # ================================================================
    # 特殊特征（灾害特定）
    # ================================================================

    def _build_special_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建灾害特定的特征。

        - CAPE 日增幅（山洪）：CAPE突增意味着强对流发展
        - 湿度下降速率（沙尘）：RH快速下降意味着干燥气团入侵
        - 海温距平趋势（风浪）：SST持续偏高
        """
        features = pd.DataFrame(index=df.index)

        # CAPE 增幅（山洪）
        if "cape" in df.columns:
            for w in [1, 3]:
                diff = df.groupby(["latitude", "longitude"])["cape"].transform(
                    lambda x: x.diff(w)
                )
                features[f"cape_increase_{w}d"] = diff.clip(lower=0)  # 只看增加

        # 湿度下降率（沙尘）
        if "rh2m" in df.columns:
            for w in [1, 3]:
                diff = df.groupby(["latitude", "longitude"])["rh2m"].transform(
                    lambda x: x.diff(w)
                )
                features[f"rh_drop_{w}d"] = (-diff).clip(lower=0)  # 只看下降

        # 地表温度与露点差（沙尘：差值越大越容易起尘）
        if "t2m_c" in df.columns and "d2m_c" in df.columns:
            features["t_dewpoint_diff"] = df["t2m_c"] - df["d2m_c"]

        # 日较差（高温热浪）
        if "tmax_c" in df.columns and "tmin_c" in df.columns:
            features["diurnal_range"] = df["tmax_c"] - df["tmin_c"]

        # 高温持续累积（热浪强度）
        if "tmax_c" in df.columns:
            threshold = self._thresholds.get("tmax_c_p95", 45.0)
            exceed = (df["tmax_c"] > threshold).astype(int)
            for w in [3, 5, 7]:
                features[f"heat_degree_days_{w}d"] = (
                    df.groupby(["latitude", "longitude"])["tmax_c"]
                    .transform(lambda x: x.rolling(w, min_periods=1).apply(
                        lambda r: (r > threshold).sum()
                    ))
                )

        return features

    # ================================================================
    # 辅助方法
    # ================================================================

    @staticmethod
    def _count_consecutive_zeros(series: pd.Series) -> pd.Series:
        """计算连续为零的天数。"""
        result = pd.Series(0, index=series.index, dtype=int)
        count = 0
        for i, val in enumerate(series.values):
            if val == 0 or (isinstance(val, float) and np.isnan(val)):
                count += 1
            else:
                count = 0
            result.iloc[i] = count
        return result

    @staticmethod
    def _count_consecutive_above(series: pd.Series, threshold: float) -> pd.Series:
        """计算连续超过阈值的天数。"""
        result = pd.Series(0, index=series.index, dtype=int)
        count = 0
        for i, val in enumerate(series.values):
            if not np.isnan(val) and val > threshold:
                count += 1
            else:
                count = 0
            result.iloc[i] = count
        return result

    # ================================================================
    # 获取信息
    # ================================================================

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表。"""
        return self._generated_features

    def get_thresholds(self) -> Dict[str, float]:
        """返回极端事件阈值。"""
        return self._thresholds.copy()

    def get_column_groups(self) -> Dict[str, List[str]]:
        """返回自动检测的变量分组。"""
        return self._auto_detected.copy()

    def print_summary(self) -> None:
        """打印特征构建摘要。"""
        print("\n" + "=" * 60)
        print("  TemporalFeatureBuilder — 摘要")
        print("=" * 60)
        print(f"  窗口大小: {self.windows}")
        print(f"  生成特征数: {len(self._generated_features)}")
        if self._auto_detected:
            print(f"  变量分组:")
            for group, cols in self._auto_detected.items():
                print(f"    {group}: {cols}")
        print(f"  极端阈值数: {len(self._thresholds)}")
        if self._generated_features:
            print(f"\n  前10个特征:")
            for f in self._generated_features[:10]:
                print(f"    - {f}")
            if len(self._generated_features) > 10:
                print(f"    ... 共 {len(self._generated_features)} 个")
        print("=" * 60 + "\n")

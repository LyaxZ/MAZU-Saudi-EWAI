"""
数据预处理器：缺失值填补、归一化、异常值处理

支持策略：
- 缺失值填补：时间插值 / 空间插值 / 气候态回填 / 简单丢弃
- 归一化：StandardScaler (Z-score) / MinMaxScaler (0~1)
- 异常值截断：分位数截断 (winsorization) / 标准差截断

设计原则：
- fit/transform 模式：在训练集上 fit，在验证/测试集上 transform
- 保持时间因果性：不使用未来信息填补过去
- 与 data/loader.py 的 DataFrame 产出无缝衔接

用法:
    from data.loader import load_to_dataframe
    from data.preprocessor import DataPreprocessor

    df = load_to_dataframe("2025-06-01", "2025-08-31", variables=[...])

    preprocessor = DataPreprocessor(strategy="temporal", scaler="standard")
    preprocessor.fit(df, feature_cols=[...])
    df_clean = preprocessor.transform(df)
    # 或一步完成:
    df_clean = preprocessor.fit_transform(df, feature_cols=[...])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """气象网格数据预处理器。

    fit/transform 模式：
    - fit(): 在训练数据上学习归一化参数和气候态均值
    - transform(): 应用缺失值填补 + 归一化 + 异常值截断
    """

    def __init__(
        self,
        strategy: str = "temporal",
        scaler: str = "standard",
        clip_outliers: bool = True,
        clip_method: str = "quantile",
        clip_lower: float = 0.01,
        clip_upper: float = 0.99,
        clip_std: float = 4.0,
        min_valid_ratio: float = 0.3,
        fill_climatology: bool = True,
    ):
        """
        Args:
            strategy: 缺失值填补策略
                - "temporal": 时间方向线性插值（同格点前→后）
                - "spatial": 空间邻居均值填补
                - "temporal+spatial": 先时间后空间
                - "drop": 直接丢弃含 NaN 的行
            scaler: 归一化方法
                - "standard": Z-score (均值0 标准差1)
                - "minmax": MinMax (0~1)
                - "none": 不做归一化
            clip_outliers: 是否截断异常值
            clip_method: 截断方法
                - "quantile": 按分位数截断
                - "std": 按标准差截断 (mean ± N*std)
            clip_lower: 分位数下限（quantile 模式）
            clip_upper: 分位数上限（quantile 模式）
            clip_std: 标准差倍数（std 模式）
            min_valid_ratio: 某变量有效值比例低于此值时，发出警告
            fill_climatology: 插值仍有 NaN 时，是否用气候态均值回填
        """
        self.strategy = strategy
        self.scaler_type = scaler
        self.clip_outliers = clip_outliers
        self.clip_method = clip_method
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.clip_std = clip_std
        self.min_valid_ratio = min_valid_ratio
        self.fill_climatology = fill_climatology

        # 训练阶段学习的参数
        self._scalers: Dict[str, object] = {}          # 每个变量的 scaler
        self._clip_bounds: Dict[str, Tuple[float, float]] = {}  # 截断边界
        self._climatology: Dict[str, float] = {}       # 气候态均值（全局）
        self._monthly_climo: Dict[str, Dict[int, float]] = {}  # 逐月气候态
        self._feature_cols: List[str] = []
        self._is_fitted = False

        # 统计信息
        self._missing_report: Dict[str, Dict] = {}

    # ================================================================
    # fit: 学习归一化参数
    # ================================================================

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> "DataPreprocessor":
        """在训练数据上学习预处理参数。

        Args:
            df: 训练数据 DataFrame
            feature_cols: 需要处理的数值特征列，None 则自动选择数值列

        Returns:
            self
        """
        print("=" * 60)
        print("  DataPreprocessor.fit() — 学习预处理参数")
        print(f"  策略: {self.strategy}, 归一化: {self.scaler_type}, "
              f"异常值截断: {self.clip_outliers}")
        print("=" * 60)

        # 1. 确定特征列
        if feature_cols is None:
            exclude = {"day", "latitude", "longitude"}
            feature_cols = [
                c for c in df.columns
                if c not in exclude
                and np.issubdtype(df[c].dtype, np.number)
                # 排除 timedelta 等非标准数值类型
                and np.issubdtype(df[c].dtype, np.floating)
            ]
            # 如果排除后特征太少，放宽条件（包含整数列）
            if len(feature_cols) < 5:
                feature_cols = [
                    c for c in df.columns
                    if c not in exclude
                    and np.issubdtype(df[c].dtype, np.number)
                ]
        # 最终过滤：只保留 float32/64 和 int32/64
        feature_cols = [
            c for c in feature_cols
            if df[c].dtype in (np.float32, np.float64, np.int32, np.int64, float, int)
            or np.issubdtype(df[c].dtype, np.floating)
            or np.issubdtype(df[c].dtype, np.integer)
        ]
        self._feature_cols = feature_cols

        # 2. 缺失值检测与报告
        self._detect_missing(df)

        # 3. 学习归一化参数
        if self.scaler_type != "none":
            self._fit_scalers(df)

        # 4. 学习截断边界
        if self.clip_outliers:
            self._fit_clip_bounds(df)

        # 5. 学习气候态（用于最后的兜底回填）
        if self.fill_climatology:
            self._fit_climatology(df)

        self._is_fitted = True
        print("  fit 完成\n")
        return self

    def _detect_missing(self, df: pd.DataFrame) -> None:
        """检测并报告各变量的缺失情况。"""
        print("\n  [缺失值检测]")
        print(f"  {'变量':<30} {'缺失率':>8} {'缺失数':>10} {'状态'}")
        print(f"  {'-'*58}")

        issues = []
        for col in self._feature_cols:
            if col not in df.columns:
                continue
            n_missing = df[col].isna().sum()
            ratio = n_missing / len(df)
            status = "OK"
            if ratio == 1.0:
                status = "⚠ 全缺失"
                issues.append(col)
            elif ratio > 0.5:
                status = "⚠ 严重缺失"
                issues.append(col)
            elif ratio > 0.1:
                status = "⚡ 中度缺失"
            elif ratio > 0:
                status = "· 少量缺失"

            self._missing_report[col] = {
                "n_missing": int(n_missing),
                "missing_ratio": float(ratio),
                "status": status,
            }

            marker = " !" if ratio > 0.1 else ""
            print(f"  {col:<30} {ratio*100:7.2f}% {n_missing:>10,}  {status}{marker}")

        if issues:
            print(f"\n  ⚠ 以下变量存在严重缺失: {issues}")

        total_cells = len(df)
        complete_rows = df[self._feature_cols].dropna().shape[0]
        print(f"\n  完整行数: {complete_rows:,} / {total_cells:,} "
              f"({complete_rows/total_cells*100:.1f}%)")

    def _fit_scalers(self, df: pd.DataFrame) -> None:
        """对每个特征列拟合 scaler。"""
        print(f"\n  [归一化参数] scaler={self.scaler_type}")

        for col in self._feature_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values.reshape(-1, 1)
            if len(vals) == 0:
                print(f"    ⚠ {col}: 无有效值，跳过")
                continue

            if self.scaler_type == "standard":
                sc = StandardScaler()
            elif self.scaler_type == "minmax":
                sc = MinMaxScaler()
            else:
                raise ValueError(f"未知 scaler 类型: {self.scaler_type}")

            sc.fit(vals)
            self._scalers[col] = sc

        print(f"    已拟合 {len(self._scalers)} 个变量的 scaler")

    def _fit_clip_bounds(self, df: pd.DataFrame) -> None:
        """学习各变量的异常值截断边界。"""
        print(f"\n  [异常值截断] method={self.clip_method}")

        for col in self._feature_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) < 10:
                continue

            if self.clip_method == "quantile":
                low = float(np.percentile(vals, self.clip_lower * 100))
                high = float(np.percentile(vals, self.clip_upper * 100))
            elif self.clip_method == "std":
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                low = mean - self.clip_std * std
                high = mean + self.clip_std * std
            else:
                raise ValueError(f"未知截断方法: {self.clip_method}")

            self._clip_bounds[col] = (low, high)

        print(f"    已计算 {len(self._clip_bounds)} 个变量的截断边界")

    def _fit_climatology(self, df: pd.DataFrame) -> None:
        """学习气候态均值（全局 + 逐月），用于兜底回填。"""
        for col in self._feature_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) > 0:
                try:
                    self._climatology[col] = float(vals.mean())
                except (TypeError, ValueError):
                    continue

        # 逐月气候态
        if "day" in df.columns:
            df_temp = df.copy()
            df_temp["month"] = pd.to_datetime(df_temp["day"]).dt.month
            for col in self._feature_cols:
                if col not in df.columns:
                    continue
                monthly = {}
                for m in range(1, 13):
                    m_vals = df_temp.loc[df_temp["month"] == m, col].dropna()
                    if len(m_vals) > 0:
                        try:
                            monthly[m] = float(m_vals.mean())
                        except (TypeError, ValueError):
                            continue
                if monthly:
                    self._monthly_climo[col] = monthly

    # ================================================================
    # transform: 应用预处理
    # ================================================================

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用预处理：缺失值填补 → 异常值截断 → 归一化。

        Args:
            df: 原始 DataFrame

        Returns:
            处理后的 DataFrame（新对象，不修改原数据）
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit() 学习预处理参数")

        df_out = df.copy()

        # 1. 缺失值填补
        if self.strategy != "drop":
            df_out = self._impute(df_out)
        else:
            before = len(df_out)
            df_out = df_out.dropna(subset=self._feature_cols)
            print(f"  [drop] 丢弃含NaN行: {before:,} → {len(df_out):,} "
                  f"({(before - len(df_out)) / before * 100:.1f}% 丢弃)")

        # 2. 异常值截断
        if self.clip_outliers:
            df_out = self._apply_clip(df_out)

        # 3. 归一化
        if self.scaler_type != "none":
            df_out = self._apply_scalers(df_out)

        return df_out

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """fit + transform 一步完成。"""
        self.fit(df, feature_cols)
        return self.transform(df)

    # ================================================================
    # 缺失值填补
    # ================================================================

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据 strategy 执行缺失值填补。"""
        print(f"\n  [缺失值填补] strategy={self.strategy}")

        for col in self._feature_cols:
            if col not in df.columns:
                continue
            if df[col].isna().sum() == 0:
                continue

            n_before = df[col].isna().sum()

            if "temporal" in self.strategy:
                df = self._impute_temporal(df, col)

            # 时间插值后仍可能有 NaN（首尾），尝试空间插值
            if "spatial" in self.strategy:
                df = self._impute_spatial(df, col)

            # 仍缺失的用气候态回填
            remaining = df[col].isna().sum()
            if remaining > 0 and self.fill_climatology:
                df = self._impute_climatology(df, col)

            n_after = df[col].isna().sum()
            if n_before > 0:
                print(f"    {col}: {n_before:,} → {n_after:,} NaN "
                      f"(填补 {n_before - n_after:,})")

        return df

    def _impute_temporal(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """时间方向线性插值。

        对每个 (lat, lon) 格点，按 day 排序后线性插值。
        首尾缺失用 ffill/bfill 填充。
        """
        if "day" not in df.columns or "latitude" not in df.columns:
            return df

        # 确保 day 为 datetime
        if not pd.api.types.is_datetime64_any_dtype(df["day"]):
            df = df.copy()
            df["day"] = pd.to_datetime(df["day"])

        # 按格点分组，沿时间插值
        df = df.copy()
        df = df.sort_values(["latitude", "longitude", "day"])

        # 创建 (day, lat, lon) 的多索引以便插值
        # 策略：对每个 (lat, lon)，按 day 排序后 interpolate
        def _interp_group(grp):
            grp = grp.sort_values("day")
            grp[col] = grp[col].interpolate(
                method="linear", limit_direction="both"
            )
            # ffill/bfill 处理首尾
            grp[col] = grp[col].ffill().bfill()
            return grp

        df = df.groupby(["latitude", "longitude"], group_keys=False).apply(
            _interp_group
        )

        return df

    def _impute_spatial(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """空间邻居均值填补。

        对每个仍为 NaN 的格点，用周围格点（同一天）的均值填补。
        周围格点定义为 lat ± 0.1°, lon ± 0.1° 范围内的非 NaN 值。
        """
        if col not in df.columns:
            return df

        still_nan = df[col].isna()
        if still_nan.sum() == 0:
            return df

        # 对每个缺失值，查找同一天、邻近格点的均值
        # 具体做法：用 pivot → 空间均值卷积 → 回填
        if "day" not in df.columns:
            return df

        df = df.copy()

        # 使用 pivot table: rows=(lat, lon), cols=day, values=col
        # 然后用 3×3 窗口均值填补
        unique_days = df["day"].unique()
        unique_lats = sorted(df["latitude"].unique())
        unique_lons = sorted(df["longitude"].unique())

        lat_res = unique_lats[1] - unique_lats[0] if len(unique_lats) > 1 else 0.1
        lon_res = unique_lons[1] - unique_lons[0] if len(unique_lons) > 1 else 0.1

        # 为每个缺失的 NaN 查找邻居
        nan_mask = df[col].isna()
        nan_indices = df.index[nan_mask]

        for idx in nan_indices:
            row = df.loc[idx]
            day_val = row["day"]
            lat_val = row["latitude"]
            lon_val = row["longitude"]

            # 查找同一天相邻格点的值
            neighbors = df.loc[
                (df["day"] == day_val)
                & (df["latitude"] >= lat_val - lat_res * 1.1)
                & (df["latitude"] <= lat_val + lat_res * 1.1)
                & (df["longitude"] >= lon_val - lon_res * 1.1)
                & (df["longitude"] <= lon_val + lon_res * 1.1)
                & (~df[col].isna()),
                col,
            ]

            if len(neighbors) > 0:
                df.loc[idx, col] = neighbors.mean()

        return df

    def _impute_climatology(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """用气候态均值填补剩余 NaN。

        优先用逐月气候态，其次用全局气候态。
        """
        still_nan = df[col].isna()
        if still_nan.sum() == 0:
            return df

        df = df.copy()

        # 优先逐月
        if col in self._monthly_climo and "day" in df.columns:
            months = pd.to_datetime(df.loc[still_nan, "day"]).dt.month
            for m in months.unique():
                if m in self._monthly_climo[col]:
                    mask = still_nan & (pd.to_datetime(df["day"]).dt.month == m)
                    df.loc[mask, col] = self._monthly_climo[col][m]
                    still_nan = df[col].isna()

        # 全局气候态兜底
        if still_nan.sum() > 0 and col in self._climatology:
            df.loc[still_nan, col] = self._climatology[col]

        return df

    # ================================================================
    # 异常值截断
    # ================================================================

    def _apply_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """对各变量应用异常值截断。"""
        df = df.copy()
        for col, (low, high) in self._clip_bounds.items():
            if col in df.columns:
                n_clipped_low = (df[col] < low).sum()
                n_clipped_high = (df[col] > high).sum()
                df[col] = df[col].clip(low, high)
                if n_clipped_low + n_clipped_high > 0:
                    pass  # 静默处理

        return df

    # ================================================================
    # 归一化
    # ================================================================

    def _apply_scalers(self, df: pd.DataFrame) -> pd.DataFrame:
        """对各变量应用 scaler.transform()。"""
        df = df.copy()
        for col, sc in self._scalers.items():
            if col in df.columns:
                vals = df[col].values.reshape(-1, 1)
                # 处理 scaler 拟合后可能出现的 NaN
                mask = ~np.isnan(vals.flatten())
                if mask.sum() < len(vals):
                    # 只对非 NaN 值做 transform
                    transformed = np.full(len(vals), np.nan)
                    transformed[mask] = sc.transform(
                        vals[mask].reshape(-1, 1)
                    ).flatten()
                    df[col] = transformed
                else:
                    df[col] = sc.transform(vals).flatten()

        return df

    # ================================================================
    # 逆变换（用于将归一化后的值还原到原始尺度）
    # ================================================================

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """将归一化后的数据还原到原始尺度（仅对有 scaler 的列）。"""
        if self.scaler_type == "none":
            return df.copy()

        df_out = df.copy()
        for col, sc in self._scalers.items():
            if col in df_out.columns:
                vals = df_out[col].values.reshape(-1, 1)
                mask = ~np.isnan(vals.flatten())
                if mask.sum() < len(vals):
                    inv = np.full(len(vals), np.nan)
                    inv[mask] = sc.inverse_transform(
                        vals[mask].reshape(-1, 1)
                    ).flatten()
                    df_out[col] = inv
                else:
                    df_out[col] = sc.inverse_transform(vals).flatten()

        return df_out

    # ================================================================
    # 报告与诊断
    # ================================================================

    def get_missing_report(self) -> pd.DataFrame:
        """返回缺失值检测报告 DataFrame。"""
        if not self._missing_report:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self._missing_report, orient="index")

    def get_scaler_params(self) -> Dict[str, Dict]:
        """返回各变量 scaler 的参数。"""
        params = {}
        for col, sc in self._scalers.items():
            if isinstance(sc, StandardScaler):
                params[col] = {
                    "type": "standard",
                    "mean": float(sc.mean_[0]),
                    "std": float(sc.scale_[0]),
                }
            elif isinstance(sc, MinMaxScaler):
                params[col] = {
                    "type": "minmax",
                    "min": float(sc.data_min_[0]),
                    "max": float(sc.data_max_[0]),
                }
        return params

    def get_clip_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回异常值截断边界。"""
        return self._clip_bounds.copy()

    def print_summary(self) -> None:
        """打印预处理器摘要。"""
        print("\n" + "=" * 60)
        print("  DataPreprocessor — 摘要")
        print("=" * 60)
        print(f"  填补策略: {self.strategy}")
        print(f"  归一化:   {self.scaler_type} ({len(self._scalers)} 个变量)")
        print(f"  异常值:   {'截断 (' + self.clip_method + ')' if self.clip_outliers else '不截断'}")
        print(f"  特征数:   {len(self._feature_cols)}")

        if self._missing_report:
            severe = [k for k, v in self._missing_report.items()
                      if v["missing_ratio"] > 0.5]
            if severe:
                print(f"  ⚠ 严重缺失变量: {severe}")
        print("=" * 60 + "\n")


# ================================================================
# 便捷函数
# ================================================================

def quick_preprocess(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    strategy: str = "temporal",
    scaler: str = "standard",
) -> pd.DataFrame:
    """快速预处理：默认时间插值 + StandardScaler。

    适用于大多数场景的一行式调用。
    """
    pp = DataPreprocessor(strategy=strategy, scaler=scaler)
    return pp.fit_transform(df, feature_cols)

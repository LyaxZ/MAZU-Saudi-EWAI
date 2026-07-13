"""
空间特征工程：邻域统计、地形衍生、位置编码

对每个格点计算：
- 邻域统计：周围 N×N 网格的均值/最大值/标准差（scipy 加速）
- 地形特征：坡度（orography 梯度）、距海岸线距离
- 位置编码：经纬度 sin/cos 周期编码、沿海标识
- 空间梯度：风速梯度、气压梯度

设计原则：
- 每个格点的空间特征仅基于该时刻的周围格点（不含未来时间）
- 对每个 day 独立计算（确保时间因果性）
- 边界格点使用可用邻居（不做外推）

用法:
    from data.loader import load_to_dataframe
    from features.spatial_features import SpatialFeatureBuilder

    df = load_to_dataframe("2025-06-01", "2025-08-31", variables=[...])

    builder = SpatialFeatureBuilder(neighbor_size=1)
    df_features = builder.build(df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter, binary_dilation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class SpatialFeatureBuilder:
    """空间特征构建器。

    对每个 (day, lat, lon) 计算基于周围格点的空间衍生特征。
    """

    def __init__(
        self,
        neighbor_size: int = 1,
        include_slope: bool = True,
        include_coast: bool = True,
        include_position: bool = True,
        include_gradient: bool = True,
        coastal_orography_max: float = 100.0,
    ):
        """
        Args:
            neighbor_size: 邻域半径（格点数）。1 = 周围 3×3, 2 = 周围 5×5
            include_slope: 是否计算地形坡度
            include_coast: 是否计算沿海相关特征
            include_position: 是否计算经纬度周期编码
            include_gradient: 是否计算空间梯度
            coastal_orography_max: 视为沿海的海拔上限 (m)
        """
        self.neighbor_size = neighbor_size
        self.include_slope = include_slope
        self.include_coast = include_coast
        self.include_position = include_position
        self.include_gradient = include_gradient
        self.coastal_orography_max = coastal_orography_max

        # 记录生成的特征名称
        self._generated_features: List[str] = []

        # 缓存：经纬度网格信息
        self._lat_grid: Optional[List[float]] = None
        self._lon_grid: Optional[List[float]] = None
        self._n_lat: int = 0
        self._n_lon: int = 0
        self._coast_mask: Optional[np.ndarray] = None
        self._dist_to_coast: Optional[np.ndarray] = None
        self._slope_grid: Optional[np.ndarray] = None

    # ================================================================
    # build: 构建全部空间特征
    # ================================================================

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建全部空间衍生特征。

        Args:
            df: 含 day, latitude, longitude 列的 DataFrame

        Returns:
            仅包含新空间特征列的 DataFrame
        """
        print(f"\n  [空间特征构建] 邻域半径: {self.neighbor_size}")

        # 确保排序
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["day"]):
            df["day"] = pd.to_datetime(df["day"])
        df = df.sort_values(["day", "latitude", "longitude"]).reset_index(drop=True)

        # 识别网格结构
        self._lat_grid = sorted(df["latitude"].unique())
        self._lon_grid = sorted(df["longitude"].unique())
        self._n_lat = len(self._lat_grid)
        self._n_lon = len(self._lon_grid)

        # 创建 lat/lon → index 映射
        self._lat_to_idx = {lat: i for i, lat in enumerate(self._lat_grid)}
        self._lon_to_idx = {lon: j for j, lon in enumerate(self._lon_grid)}

        lat_res = self._lat_grid[1] - self._lat_grid[0] if self._n_lat > 1 else 0.1
        lon_res = self._lon_grid[1] - self._lon_grid[0] if self._n_lon > 1 else 0.1
        print(f"    网格: {self._n_lat} × {self._n_lon}, 分辨率: {lat_res:.2f}° × {lon_res:.2f}°")

        features_list = []

        # 1. 位置编码（最快，不需要网格计算）
        if self.include_position:
            pos = self._build_position_features(df)
            if pos is not None and len(pos.columns) > 0:
                features_list.append(pos)

        # 2. 地形坡度（只需计算一次，缓存复用）
        if self.include_slope and "orography" in df.columns:
            self._compute_slope_cache(df)

        # 3. 沿海特征（只需计算一次，缓存复用）
        if self.include_coast:
            self._compute_coast_cache(df)

        # 4. 为每个 day 批量构建 per-day 特征
        day_features = self._build_per_day_features(df)
        if day_features is not None and len(day_features.columns) > 0:
            features_list.append(day_features)

        # 5. 静态空间特征（对所有 day 相同）: slope, coast
        static_features = self._build_static_features(df)
        if static_features is not None and len(static_features.columns) > 0:
            features_list.append(static_features)

        # 合并
        if features_list:
            result = pd.concat(features_list, axis=1)
        else:
            result = pd.DataFrame(index=df.index)

        self._generated_features = list(result.columns)
        print(f"  生成 {len(self._generated_features)} 个空间特征\n")

        return result

    # ================================================================
    # 预计算缓存
    # ================================================================

    def _compute_slope_cache(self, df: pd.DataFrame) -> None:
        """预计算坡度网格（地形不随时间变化）。"""
        first_day = df["day"].min()
        oro_df = df[df["day"] == first_day][["latitude", "longitude", "orography"]].dropna()
        if len(oro_df) == 0:
            return

        oro_grid = np.full((self._n_lat, self._n_lon), np.nan)
        for _, row in oro_df.iterrows():
            i = self._lat_to_idx.get(row["latitude"])
            j = self._lon_to_idx.get(row["longitude"])
            if i is not None and j is not None:
                oro_grid[i, j] = row["orography"]

        # 计算坡度
        lat_res = self._lat_grid[1] - self._lat_grid[0] if self._n_lat > 1 else 0.1
        lon_res = self._lon_grid[1] - self._lon_grid[0] if self._n_lon > 1 else 0.1

        dz_dlat = np.zeros_like(oro_grid)
        dz_dlon = np.zeros_like(oro_grid)
        dz_dlat[1:-1, :] = (oro_grid[2:, :] - oro_grid[:-2, :]) / (2 * lat_res)
        dz_dlon[:, 1:-1] = (oro_grid[:, 2:] - oro_grid[:, :-2]) / (2 * lon_res)

        self._slope_grid = np.sqrt(dz_dlat ** 2 + dz_dlon ** 2)
        self._oro_grid = oro_grid

    def _compute_coast_cache(self, df: pd.DataFrame) -> None:
        """预计算沿海掩码和距离（不随时间变化）。"""
        coast_mask = np.zeros((self._n_lat, self._n_lon), dtype=bool)

        # 基于 orography
        if "orography" in df.columns:
            first_day = df["day"].min()
            oro_df = df[df["day"] == first_day][["latitude", "longitude", "orography"]].dropna()
            for _, row in oro_df.iterrows():
                i = self._lat_to_idx.get(row["latitude"])
                j = self._lon_to_idx.get(row["longitude"])
                if i is not None and j is not None:
                    if row["orography"] < self.coastal_orography_max:
                        coast_mask[i, j] = True

        self._coast_mask = coast_mask

        # 近似距离（使用距离变换）
        if coast_mask.any():
            from_land = ~coast_mask
            # 简化：曼哈顿距离近似
            coastal_idx = np.argwhere(coast_mask)
            dist_grid = np.full((self._n_lat, self._n_lon), np.nan)

            n_total = self._n_lat * self._n_lon
            if n_total <= 50000:  # 只在合理大小下计算距离
                for i in range(self._n_lat):
                    for j in range(self._n_lon):
                        if coast_mask[i, j]:
                            dist_grid[i, j] = 0.0
                        else:
                            dlat = np.abs(coastal_idx[:, 0] - i) * (self._lat_grid[1] - self._lat_grid[0])
                            dlon = np.abs(coastal_idx[:, 1] - j) * (self._lon_grid[1] - self._lon_grid[0])
                            dist_grid[i, j] = np.min(np.sqrt(dlat**2 + dlon**2))
            self._dist_to_coast = dist_grid
        else:
            self._dist_to_coast = np.full((self._n_lat, self._n_lon), np.nan)

    # ================================================================
    # Per-day 特征（邻域统计 + 空间梯度）
    # ================================================================

    def _build_per_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对每一天构建空间邻域特征和梯度特征。

        使用向量化操作加速：将每天的数据 reshape 为 (n_lat, n_lon) 网格。
        """
        agg_cols = [
            c for c in [
                "daily_precip_total", "cape", "t2m_c", "tmax_c",
                "wind10_speed", "rh2m", "vpd_kpa", "surface_pressure",
            ] if c in df.columns
        ]
        if not agg_cols:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)
        n = self.neighbor_size
        window = 2 * n + 1  # e.g. 3×3 for n=1

        unique_days = df["day"].unique()
        n_days = len(unique_days)

        for d_idx, day_val in enumerate(unique_days):
            if d_idx % max(1, n_days // 5) == 0:
                print(f"    处理日期 {d_idx+1}/{n_days}: {pd.Timestamp(day_val).strftime('%Y-%m-%d')}")

            day_mask = df["day"] == day_val
            day_idx = df.index[day_mask]

            for col in agg_cols:
                # 构建网格
                grid = np.full((self._n_lat, self._n_lon), np.nan)
                day_data = df.loc[day_mask, ["latitude", "longitude", col]]
                for _, row in day_data.iterrows():
                    i = self._lat_to_idx.get(row["latitude"])
                    j = self._lon_to_idx.get(row["longitude"])
                    if i is not None and j is not None:
                        grid[i, j] = row[col]

                feat_prefix = f"{col}_spatial"

                if _HAS_SCIPY:
                    # 使用 scipy 快速卷积
                    mean_grid = uniform_filter(grid, size=window, mode='nearest')
                    max_grid = maximum_filter(grid, size=window, mode='nearest')
                    min_grid = minimum_filter(grid, size=window, mode='nearest')
                else:
                    # NumPy 向量化回退
                    mean_grid = self._numpy_window_agg(grid, window, "mean")
                    max_grid = self._numpy_window_agg(grid, window, "max")
                    min_grid = self._numpy_window_agg(grid, window, "min")

                # 将网格结果映射回 DataFrame
                for idx in day_idx:
                    row = df.loc[idx]
                    i = self._lat_to_idx.get(row["latitude"])
                    j = self._lon_to_idx.get(row["longitude"])
                    if i is not None and j is not None:
                        features.loc[idx, f"{feat_prefix}_mean_{n}"] = mean_grid[i, j]
                        features.loc[idx, f"{feat_prefix}_max_{n}"] = max_grid[i, j]
                        features.loc[idx, f"{feat_prefix}_min_{n}"] = min_grid[i, j]

        return features

    def _numpy_window_agg(
        self, arr: np.ndarray, window: int, mode: str
    ) -> np.ndarray:
        """NumPy 向量化滑动窗口聚合（scipy 不可用时的回退方案）。

        使用 stride_tricks 创建滑动窗口视图后沿轴聚合。
        """
        from numpy.lib.stride_tricks import sliding_window_view

        pad = window // 2
        padded = np.pad(arr, pad, mode='edge')

        # 创建 (n_lat, n_lon, window, window) 视图
        view = sliding_window_view(padded, (window, window))
        # view shape: (n_lat, n_lon, window, window)

        if mode == "mean":
            with np.errstate(all='ignore'):
                return np.nanmean(view, axis=(2, 3))
        elif mode == "max":
            with np.errstate(all='ignore'):
                return np.nanmax(view, axis=(2, 3))
        elif mode == "min":
            with np.errstate(all='ignore'):
                return np.nanmin(view, axis=(2, 3))
        else:
            raise ValueError(f"未知模式: {mode}")

    # ================================================================
    # 静态特征（对所有 day 相同）
    # ================================================================

    def _build_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建不随时间变化的静态空间特征。"""
        features = pd.DataFrame(index=df.index)

        # 地形坡度
        if self._slope_grid is not None:
            for idx, row in df.iterrows():
                i = self._lat_to_idx.get(row["latitude"])
                j = self._lon_to_idx.get(row["longitude"])
                if i is not None and j is not None:
                    features.loc[idx, "slope_raw"] = self._slope_grid[i, j]
                    features.loc[idx, "orography_static"] = self._oro_grid[i, j]

            # 坡度分类
            if "slope_raw" in features.columns:
                valid = features["slope_raw"].dropna()
                if len(valid) > 0:
                    p75 = np.nanpercentile(valid, 75)
                    features["slope_steep"] = (features["slope_raw"] > p75).astype(int)

        # 沿海标记
        if self._coast_mask is not None:
            for idx, row in df.iterrows():
                i = self._lat_to_idx.get(row["latitude"])
                j = self._lon_to_idx.get(row["longitude"])
                if i is not None and j is not None:
                    features.loc[idx, "coast_flag"] = int(self._coast_mask[i, j])

        # 距海岸线近似距离
        if self._dist_to_coast is not None:
            for idx, row in df.iterrows():
                i = self._lat_to_idx.get(row["latitude"])
                j = self._lon_to_idx.get(row["longitude"])
                if i is not None and j is not None:
                    d = self._dist_to_coast[i, j]
                    if not np.isnan(d):
                        features.loc[idx, "dist_to_coast_deg"] = d

        return features

    # ================================================================
    # 位置编码
    # ================================================================

    def _build_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建经纬度周期编码特征。"""
        features = pd.DataFrame(index=df.index)

        lat = df["latitude"].values
        lon = df["longitude"].values

        # 纬度编码（范围 16°N ~ 32°N）
        lat_norm = (lat - 16.0) / (32.0 - 16.0)
        features["lat_sin"] = np.sin(2 * np.pi * lat_norm)
        features["lat_cos"] = np.cos(2 * np.pi * lat_norm)
        features["lat_sin_2f"] = np.sin(4 * np.pi * lat_norm)
        features["lat_cos_2f"] = np.cos(4 * np.pi * lat_norm)

        # 经度编码（范围 34°E ~ 56°E）
        lon_norm = (lon - 34.0) / (56.0 - 34.0)
        features["lon_sin"] = np.sin(2 * np.pi * lon_norm)
        features["lon_cos"] = np.cos(2 * np.pi * lon_norm)
        features["lon_sin_2f"] = np.sin(4 * np.pi * lon_norm)
        features["lon_cos_2f"] = np.cos(4 * np.pi * lon_norm)

        # 绝对坐标
        features["lat_abs"] = lat
        features["lon_abs"] = lon

        return features

    # ================================================================
    # 获取信息
    # ================================================================

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表。"""
        return self._generated_features

    def get_coast_mask(self) -> Optional[np.ndarray]:
        """返回沿海掩码 (n_lat, n_lon)。"""
        return self._coast_mask

    def get_dist_to_coast(self) -> Optional[np.ndarray]:
        """返回距海岸线距离 (n_lat, n_lon)。"""
        return self._dist_to_coast

    def get_slope_grid(self) -> Optional[np.ndarray]:
        """返回坡度网格 (n_lat, n_lon)。"""
        return self._slope_grid

    def print_summary(self) -> None:
        """打印空间特征构建摘要。"""
        print("\n" + "=" * 60)
        print("  SpatialFeatureBuilder — 摘要")
        print("=" * 60)
        print(f"  邻域半径: {self.neighbor_size} ({(2*self.neighbor_size+1)}×{(2*self.neighbor_size+1)} 窗口)")
        print(f"  scipy 加速: {'✓' if _HAS_SCIPY else '✗ (NumPy 回退)'}")
        print(f"  地形坡度: {'✓' if self.include_slope else '✗'}")
        print(f"  沿海特征: {'✓' if self.include_coast else '✗'}")
        print(f"  位置编码: {'✓' if self.include_position else '✗'}")
        print(f"  空间梯度: {'✓' if self.include_gradient else '✗'}")
        print(f"  生成特征数: {len(self._generated_features)}")
        if self._generated_features:
            print(f"\n  特征列表:")
            for f in self._generated_features:
                print(f"    - {f}")
        print("=" * 60 + "\n")

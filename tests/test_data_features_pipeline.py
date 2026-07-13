r"""
测试脚本：验证 data/preprocessor + splitter + features 完整流水线

用法:
    cd D:\Mazu\MAZU-Saudi-EWAI
    python tests/test_data_features_pipeline.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from data.preprocessor import DataPreprocessor
from data.splitter import TimeSeriesSplitter
from features.temporal_features import TemporalFeatureBuilder
from features.spatial_features import SpatialFeatureBuilder
from features.feature_registry import FeatureRegistry

# 测试用变量子集（减少加载时间）
TEST_VARS = [
    "daily_precip_total", "cape", "cin",
    "t2m_c", "tmax_c", "tmin_c", "diurnal_temp_range_c",
    "rh2m", "vpd_kpa",
    "wind10_speed", "orography",
    "flash_flood_risk", "heatwave_day_flag",
    "surface_pressure",
]


def test_preprocessor():
    """测试 DataPreprocessor 基本功能。"""
    print("\n" + "=" * 60)
    print("  测试 1: DataPreprocessor")
    print("=" * 60)

    # 加载1天数据
    print("  加载 2025-06-15 数据...")
    ds = load_date_range("2025-06-15", "2025-06-15", variables=TEST_VARS,
                         show_progress=False)
    df = ds.to_dataframe().reset_index()

    # 只选择真正的气象变量（排除标量坐标等非数据列）
    exclude_coords = {"step", "surface", "atmosphere", "heightAboveGround",
                      "highCloudLayer", "lowCloudLayer", "middleCloudLayer",
                      "atmosphereSingleLayer", "highestTroposphericFreezing",
                      "isothermZero", "maxWind", "meanSea", "sigma", "theta",
                      "tropopause", "isobaricInhPa"}
    features = [c for c in df.columns if c not in ("day", "latitude", "longitude")
                and c not in exclude_coords
                and np.issubdtype(df[c].dtype, np.number)]

    print(f"  原始数据: {len(df):,} 行, {len(features)} 个数值特征")

    # 测试 fit + transform
    pp = DataPreprocessor(strategy="temporal", scaler="standard",
                          clip_outliers=True)
    df_clean = pp.fit_transform(df, feature_cols=features)

    print(f"  处理后:   {len(df_clean):,} 行, NaN 剩余: {df_clean[features].isna().sum().sum()}")

    # 验证
    assert len(df_clean) == len(df), "行数不应变化"
    assert df_clean[features].isna().sum().sum() < len(features), "缺失值应大幅减少"

    # 测试 inverse_transform
    df_inv = pp.inverse_transform(df_clean)
    assert df_inv.shape == df_clean.shape

    pp.print_summary()
    print("  [OK] DataPreprocessor 测试通过")
    return df


def test_splitter(df):
    """测试 TimeSeriesSplitter 基本功能。"""
    print("\n" + "=" * 60)
    print("  测试 2: TimeSeriesSplitter")
    print("=" * 60)

    # 加载多天数据
    print("  加载 2025-06-01 ~ 2025-09-30 数据...")
    ds = load_date_range("2025-06-01", "2025-09-30", variables=TEST_VARS,
                         show_progress=False)
    df_multi = ds.to_dataframe().reset_index()

    print(f"  数据: {len(df_multi):,} 行, {df_multi['day'].nunique()} 天")

    # 按日期范围划分
    splitter = TimeSeriesSplitter(
        train_start="2025-06-01", train_end="2025-08-31",
        val_start="2025-09-01", val_end="2025-09-15",
        test_start="2025-09-16", test_end="2025-09-30",
        split_method="date_range",
    )

    splits = splitter.split(df_multi)
    splitter.print_split_summary()

    # 验证
    assert len(splits["train"]) > 0, "训练集非空"
    assert len(splits["val"]) > 0, "验证集非空"
    assert len(splits["test"]) > 0, "测试集非空"

    # 验证时间因果性
    train_max = splits["train"]["day"].max()
    val_min = splits["val"]["day"].min()
    assert train_max <= val_min, f"时间因果性违反: train_max={train_max} > val_min={val_min}"

    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    print(f"  总覆盖率: {total}/{len(df_multi)} ({total/len(df_multi)*100:.1f}%)")

    print("  [OK] TimeSeriesSplitter 测试通过")
    return splits


def test_temporal_features(df):
    """测试 TemporalFeatureBuilder 基本功能。"""
    print("\n" + "=" * 60)
    print("  测试 3: TemporalFeatureBuilder")
    print("=" * 60)

    # 加载5天数据用于时序特征
    print("  加载 2025-06-11 ~ 2025-06-15 数据...")
    ds = load_date_range("2025-06-11", "2025-06-15", variables=TEST_VARS,
                         show_progress=False)
    df_multi = ds.to_dataframe().reset_index()

    print(f"  数据: {len(df_multi):,} 行, {df_multi['day'].nunique()} 天")

    builder = TemporalFeatureBuilder(windows=[3, 5])
    builder.fit(df_multi)
    df_feats = builder.build(df_multi)

    print(f"  生成特征: {len(df_feats.columns)} 个")
    print(f"  特征示例: {list(df_feats.columns)[:10]}...")

    # 验证
    assert len(df_feats.columns) > 0, "应生成至少1个特征"
    assert len(df_feats) == len(df_multi), "行数应一致"

    builder.print_summary()
    print("  [OK] TemporalFeatureBuilder 测试通过")
    return df_feats


def test_spatial_features(df):
    """测试 SpatialFeatureBuilder 基本功能（小规模）。"""
    print("\n" + "=" * 60)
    print("  测试 4: SpatialFeatureBuilder (小规模)")
    print("=" * 60)

    # 使用单日数据 + 空间子集测试
    ds = load_date_range("2025-06-15", "2025-06-15", variables=TEST_VARS,
                         show_progress=False)
    df_one = ds.to_dataframe().reset_index()

    # 取空间子集加速测试（50×50区域）
    df_subset = df_one[
        (df_one["latitude"] >= df_one["latitude"].quantile(0.3)) &
        (df_one["latitude"] <= df_one["latitude"].quantile(0.5)) &
        (df_one["longitude"] >= df_one["longitude"].quantile(0.3)) &
        (df_one["longitude"] <= df_one["longitude"].quantile(0.5))
    ].copy()

    print(f"  数据: {len(df_subset):,} 行 ({df_subset['latitude'].nunique()}×{df_subset['longitude'].nunique()} 网格)")

    builder = SpatialFeatureBuilder(neighbor_size=1)
    df_feats = builder.build(df_subset)

    print(f"  生成特征: {len(df_feats.columns)} 个")
    if len(df_feats.columns) > 0:
        print(f"  特征示例: {list(df_feats.columns)[:8]}...")

    # 验证
    assert len(df_feats.columns) > 0, "应生成至少1个特征"
    assert len(df_feats) == len(df_subset), "行数应一致"

    # 验证位置编码
    for col in ["lat_sin", "lat_cos", "lon_sin", "lon_cos"]:
        if col in df_feats.columns:
            assert df_feats[col].between(-1, 1).all(), f"{col} 应在 [-1, 1] 范围"

    builder.print_summary()
    print("  [OK] SpatialFeatureBuilder 测试通过")
    return df_feats


def test_feature_registry():
    """测试 FeatureRegistry 基本功能。"""
    print("\n" + "=" * 60)
    print("  测试 5: FeatureRegistry")
    print("=" * 60)

    registry = FeatureRegistry()

    # 注册时序特征
    registry.register_temporal_features([
        "daily_precip_total_sum_3d", "cape_increase_1d",
        "consecutive_dry_days", "consecutive_hot_days",
    ])

    # 注册空间特征
    registry.register_spatial_features([
        "daily_precip_total_spatial_mean_1",
        "wind10_speed_spatial_max_1",
        "lat_sin", "lon_cos",
    ])

    # 查询
    ff_feats = registry.get_features("flash_flood",
                                      groups=["raw", "temporal", "spatial", "position", "terrain"])
    print(f"  flash_flood 特征数: {len(ff_feats)}")
    assert len(ff_feats) > 10, "flash_flood 应有足够特征"

    # 消融实验
    ablation = registry.ablation_sets("flash_flood")
    print(f"  消融实验组: {list(ablation.keys())}")
    for name, feats in ablation.items():
        print(f"    {name}: {len(feats)} 特征")

    # 重复检测
    dups = registry.check_duplicates()
    print(f"  跨组重复: {dups if dups else '无'}")

    registry.print_summary()
    print("  [OK] FeatureRegistry 测试通过")
    return registry


def test_label_builder():
    """快速验证 label_builder 仍正常工作。"""
    print("\n" + "=" * 60)
    print("  测试 6: DisasterLabelBuilder (回归)")
    print("=" * 60)

    ds = load_date_range("2025-06-15", "2025-06-15", variables=TEST_VARS,
                         show_progress=False)
    df = ds.to_dataframe().reset_index()

    builder = DisasterLabelBuilder()
    builder.fit(df)
    labels = builder.build_all(df)

    expected_cols = [
        "flash_flood_label", "extreme_heat_label",
        "dust_wind_label", "coastal_wave_label",
    ]
    for col in expected_cols:
        assert col in labels.columns, f"缺少列: {col}"
        assert set(labels[col].unique()).issubset({0, 1})

    print("  [OK] DisasterLabelBuilder 回归测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("  Data + Features 流水线测试")
    print("=" * 60)

    test_label_builder()

    df = test_preprocessor()

    splits = test_splitter(df)

    test_temporal_features(df)

    test_spatial_features(df)

    test_feature_registry()

    print("\n" + "=" * 60)
    print("  全部测试通过! ✅")
    print("=" * 60)

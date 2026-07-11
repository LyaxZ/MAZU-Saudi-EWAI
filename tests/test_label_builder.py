"""
测试脚本：验证 DisasterLabelBuilder 四类灾害标签构建

用法:
    cd D:\Mazu\MAZU-Saudi-EWAI
    python tests/test_label_builder.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder

# 标签构建需要的变量
LABEL_VARS = [
    "flash_flood_risk", "heatwave_day_flag",
    "wind10_speed", "rh2m", "vpd_kpa",
    "orography", "ivt", "tmax_c",
]


def test_all_modes():
    """测试三种构建模式在不同季节的表现。"""
    print("=" * 60)
    print("  DisasterLabelBuilder — 全模式全季节测试")
    print("=" * 60)

    # 训练数据: 夏季（沙尘+风浪高发季）
    print("\n>>> 加载训练数据 (6月15日)...")
    df_train = load_date_range(
        "2025-06-15", "2025-06-15",
        variables=LABEL_VARS, show_progress=False,
    ).to_dataframe().dropna()
    print(f"    训练样本: {len(df_train):,}")

    # 测试日期
    test_dates = {
        "冬季": "2025-01-15",
        "春季": "2025-04-15",
        "夏季": "2025-07-15",
        "秋季": "2025-09-16",
    }

    for dust_mode in ["simple", "standard"]:
        for coastal_mode in ["simple", "standard"]:
            mode_label = f"dust={dust_mode}, coastal={coastal_mode}"
            print(f"\n{'='*60}")
            print(f"  模式: {mode_label}")
            print(f"{'='*60}")

            builder = DisasterLabelBuilder(
                dust_mode=dust_mode,
                coastal_mode=coastal_mode,
            )
            builder.fit(df_train)

            print(f"\n{'Season':<10} {'flash_flood':>12} {'extreme_heat':>12} "
                  f"{'dust_wind':>12} {'coastal_wave':>12}")
            print("-" * 52)

            for season, date in test_dates.items():
                df_test = load_date_range(
                    date, date,
                    variables=LABEL_VARS, show_progress=False,
                ).to_dataframe().dropna()

                labels = builder.build_all(df_test)

                ff = labels["flash_flood_label"].mean() * 100
                eh = labels["extreme_heat_label"].mean() * 100
                dw = labels["dust_wind_label"].mean() * 100
                cw = labels["coastal_wave_label"].mean() * 100

                print(f"{season:<10} {ff:11.3f}% {eh:11.3f}% {dw:11.3f}% {cw:11.3f}%")

    print(f"\n{'='*60}")
    print("  测试完成")
    print(f"{'='*60}")


def test_build_all():
    """测试 build_all 返回格式。"""
    print("\n>>> 测试 build_all 返回格式...")

    df = load_date_range(
        "2025-06-15", "2025-06-15",
        variables=LABEL_VARS, show_progress=False,
    ).to_dataframe().dropna()

    builder = DisasterLabelBuilder()
    builder.fit(df)

    labels = builder.build_all(df)

    # 验证列名
    expected_cols = [
        "flash_flood_label", "extreme_heat_label",
        "dust_wind_label", "coastal_wave_label",
    ]
    for col in expected_cols:
        assert col in labels.columns, f"缺少列: {col}"
        assert set(labels[col].unique()).issubset({0, 1}), \
            f"{col} 包含非 0/1 值: {labels[col].unique()}"

    print(f"    [OK] 列名正确: {list(labels.columns)}")
    print(f"    [OK] 所有标签为 0/1")
    print(f"    [OK] 形状: {labels.shape}")

    # 验证统计信息
    stats = builder.get_stats()
    assert len(stats) == 4, f"应有4类灾害统计, 实际: {len(stats)}"
    print(f"    [OK] 统计信息完整 (4类灾害)")

    # 验证阈值
    thresholds = builder.get_thresholds()
    assert "dust_wind" in thresholds, "缺少 dust_wind 阈值"
    assert "coastal_wave" in thresholds, "缺少 coastal_wave 阈值"
    print(f"    [OK] 阈值信息完整")

    print("    [OK] build_all 测试通过")


if __name__ == "__main__":
    test_build_all()
    print("\n")
    test_all_modes()

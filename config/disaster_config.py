"""
灾害标签配置：四类灾害的阈值定义、判定逻辑、物理依据

每一类灾害定义了：
- 标签构建所需的核心变量
- 分位数阈值（从训练数据中 fit 得到）
- 判定逻辑（简单模式 / 增强模式）
- 物理依据说明
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. 暴雨山洪 (Flash Flood)
# ============================================================
# 物理依据: 强降水 + 对流不稳定 + 地形抬升 → 山洪
# 数据中已有 flash_flood_risk (0/1/2/3)，直接使用
# 判定: flash_flood_risk >= 1 → 正样本
FLASH_FLOOD_LABEL_CONFIG = {
    "disaster_type": "flash_flood",
    "name_cn": "暴雨山洪",
    "label_column": "flash_flood_risk",
    "mode": "native",  # 数据中已有标签，直接使用
    "threshold": 1,  # risk >= 1 视为正样本
    "description": "使用数据中内置的 flash_flood_risk 指标，>= 1 为正样本",
    "severity_levels": {
        0: "无风险",
        1: "低风险",
        2: "中风险",
        3: "高风险",
    },
}


# ============================================================
# 2. 极端高温 (Extreme Heat)
# ============================================================
# 物理依据: 日最高气温持续超过气候态 → 热浪
# 数据中已有 heatwave_day_flag (0/1)，直接使用
# 判定: heatwave_day_flag == 1 → 正样本
EXTREME_HEAT_LABEL_CONFIG = {
    "disaster_type": "extreme_heat",
    "name_cn": "极端高温",
    "label_column": "heatwave_day_flag",
    "mode": "native",  # 数据中已有标签，直接使用
    "threshold": 0.5,
    "description": "使用数据中内置的 heatwave_day_flag，基于多日高温阈值的综合判定",
    # 增强模式：如果没有内置标签，可基于以下变量构建
    "build_variables": [
        "tmax_c", "t2m_c", "heat_index_c", "diurnal_temp_range_c",
        "heatwave_duration_days",
    ],
    "build_rules": {
        # 代理方案: tmax > P90 AND 连续 >= 3天
        "tmax_percentile": 90,
        "consecutive_days": 3,
    },
}


# ============================================================
# 3. 沙尘强风 (Dust Wind)
# ============================================================
# 物理依据:
#   沙特沙尘暴的三大触发条件：
#   1. 强地表风 (> 8-10 m/s) — 提供扬尘动力
#   2. 干燥大气 (RH < 30%, VPD 极高) — 沙源区地表松散
#   3. 常伴随锋面过境或对流下击暴流 (风切变)
#
# 构建策略（三档可选）:
#   - simple:   wind10_speed > P95 AND rh2m < 30%（与吕基线一致）
#   - standard: wind10_speed > P90 AND (rh2m < 30% OR vpd_kpa > P80)
#   - enhanced: wind10 + 干燥 + 风切变综合评分 >= 2分
DUST_WIND_LABEL_CONFIG = {
    "disaster_type": "dust_wind",
    "name_cn": "沙尘强风",
    "label_column": "dust_wind_label",
    "mode": "build",  # 需要从气象变量构建
    "build_mode": "standard",  # simple / standard / enhanced
    "description": "从风速+干燥+风切变指标综合构建沙尘暴标签",

    # 核心变量
    "wind_variables": [
        "wind10_speed", "wind925_speed",
    ],
    "dryness_variables": [
        "rh2m", "vpd_kpa", "dewpoint_depression_c",
    ],
    "auxiliary_variables": [
        "wind_shear_850_300", "wind_shear_850_200",
    ],

    # 阈值配置（从训练数据 fit 得到）
    "thresholds": {
        # simple 模式
        "simple": {
            "wind10_speed_percentile": 95,
            "rh2m_absolute_max": 30.0,  # %
        },
        # standard 模式
        "standard": {
            "wind10_speed_percentile": 90,
            "rh2m_absolute_max": 30.0,  # %（物理阈值：RH < 30% 沙尘扬起条件）
            "vpd_kpa_percentile": 80,  # vpd > P80（大气蒸发需求极高）
            "logic": "wind AND (dry_rh OR dry_vpd) — RH用绝对值，VPD用分位数",
        },
        # enhanced 模式
        "enhanced": {
            "wind10_speed_percentile": 85,
            "rh2m_percentile": 25,
            "vpd_kpa_percentile": 75,
            "wind_shear_percentile": 80,
            "scoring": {
                "wind_strong": 1,      # wind10 > P85
                "very_dry": 1,         # rh2m < P25 OR vpd > P75
                "high_shear": 1,       # wind_shear > P80
                "threshold": 2,        # 总分 >= 2 → 正样本
            },
        },
    },
}


# ============================================================
# 4. 沿海风浪 (Coastal Wave)
# ============================================================
# 物理依据:
#   红海/阿拉伯湾沿岸风浪灾害的触发条件：
#   1. 沿海格点 — 通过 SST 非 NaN 或 海拔 < 阈值 判定
#   2. 强海面风 (> P90 沿海风速) — 波浪能量来源
#   3. 高 SST — 提供能量（尤其是热带气旋生成条件）
#   4. 高 IVT — 水汽输送，可能与风暴潮相关
#
# 构建策略:
#   - simple:   coastal(sst非NaN) AND wind10_speed > P90（与吕基线一致）
#   - standard: coastal AND wind10 > P85 AND (SST > median OR ivt > P75)
#   - enhanced: coastal + 风 + SST异常 + IVT 综合评分
COASTAL_WAVE_LABEL_CONFIG = {
    "disaster_type": "coastal_wave",
    "name_cn": "沿海风浪",
    "label_column": "coastal_wave_label",
    "mode": "build",  # 需要从气象变量构建
    "build_mode": "standard",  # simple / standard / enhanced
    "description": "从沿海位置+海面风+SST+IVT综合构建风浪灾害标签",

    # 沿海判定变量
    "coastal_indicators": [
        "sst_celsius",  # 非 NaN → 海洋/沿海格点
        "orography",     # 低海拔 + 近海 → 沿海
    ],
    "coastal_orography_max": 100.0,  # 海拔 < 100m 视为沿海
    "coastal_lat_lon_buffer": 0.3,  # 距海岸线 ~0.3° (~30 km) 视为沿海

    # 核心变量
    "wind_variables": [
        "wind10_speed", "wind925_speed",
    ],
    "ocean_variables": [
        "sst_celsius", "ivt", "ivt_u", "ivt_v",
    ],

    # 阈值配置
    "thresholds": {
        "simple": {
            "wind10_speed_percentile": 90,
            "coastal_method": "sst_or_orography",
        },
        "standard": {
            "wind10_speed_percentile": 85,
            "sst_percentile": 50,  # SST > median
            "ivt_percentile": 75,  # IVT > P75
            "logic": "coastal AND wind_strong AND (sst_warm OR ivt_high)",
        },
        "enhanced": {
            "wind10_speed_percentile": 80,
            "sst_anomaly_threshold": 0.5,  # SST 距平 > +0.5°C
            "ivt_percentile": 70,
            "scoring": {
                "is_coastal": 1,     # 沿海格点
                "wind_strong": 1,    # wind10 > P80
                "sst_warm": 1,       # SST > median
                "ivt_high": 1,       # IVT > P70
                "threshold": 3,      # 总分 >= 3 → 正样本
            },
        },
    },
}


# ============================================================
# 汇总
# ============================================================
DISASTER_LABEL_CONFIGS = {
    "flash_flood": FLASH_FLOOD_LABEL_CONFIG,
    "extreme_heat": EXTREME_HEAT_LABEL_CONFIG,
    "dust_wind": DUST_WIND_LABEL_CONFIG,
    "coastal_wave": COASTAL_WAVE_LABEL_CONFIG,
}


def get_label_config(disaster_type: str) -> dict:
    """获取指定灾害的标签配置。"""
    if disaster_type not in DISASTER_LABEL_CONFIGS:
        raise ValueError(
            f"未知灾害类型: {disaster_type}。可选: {list(DISASTER_LABEL_CONFIGS.keys())}"
        )
    return DISASTER_LABEL_CONFIGS[disaster_type]


def get_label_column(disaster_type: str) -> str:
    """获取指定灾害的标签列名。"""
    return get_label_config(disaster_type)["label_column"]

"""
模型配置：超参数、数据子集范围、设备选择

面向原型快速迭代：默认使用小数据集（30天）和 CPU 训练。
后续可扩大数据范围并切换到 GPU。
"""

import torch

# ============================================
# 设备选择
# ============================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[model_config] 检测到设备: {DEVICE}")

# ============================================
# 训练数据范围
# ============================================
TRAIN_START = "2025-06-01"   # 6-8月（山洪高发季），约92天
TRAIN_END = "2025-08-31"
VAL_START = "2025-09-01"     # 9月验证集（调阈值/早停），约15天
VAL_END = "2025-09-15"
TEST_START = "2025-09-16"    # 9月后半测试集（最终评估），约15天
TEST_END = "2025-09-30"

# ============================================
# 入模特征（挑选与四类灾害最相关的核心变量）
# ============================================
FLASH_FLOOD_FEATURES = [
    # 降水
    "daily_precip_total", "daily_convective_precip", "daily_large_scale_precip",
    "monthly_precip_total",
    # 卫星短历时降水（DS10）
    "ds10_max_1h",
    # 对流与不稳定
    "cape", "cin",
    # 水汽与输送
    "pwat", "ivt", "ivt_convergence", "rh2m", "vpd_kpa",
    "moisture_transport850",
    # 垂直运动与风场
    "omega500", "wind10_speed", "wind925_speed", "wind850_speed",
    # 云与辐射
    "total_cloud_cover", "net_radiation",
    # 地形
    "orography", "surface_pressure",
    # 空间位置编码
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
]  # 共24特征

EXTREME_HEAT_FEATURES = [
    # 温度异常（★ 核心特征：热浪定义的基础）
    "tmax_anomaly_c", "t2m_anomaly_c", "tmax_climatology_c",
    # 基础温度
    "t2m_c", "tmax_c", "tmin_c", "diurnal_temp_range_c",
    "heat_index_c", "apparent_temp_c", "heat_stress_index",
    # 湿度
    "vpd_kpa", "rh2m", "d2m_c",
    # 辐射
    "sw_net", "lw_net", "net_radiation", "bowen_ratio",
    # 云量
    "total_cloud_cover",
    # 地形
    "orography", "surface_pressure",
    # 空间位置编码
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
]  # 共24特征，最佳阈值=0.95

DUST_WIND_FEATURES = [
    "wind10_speed", "wind925_speed", "wind850_speed",
    "jet300_speed", "jet200_speed",
    "wind_shear_850_300", "wind_shear_850_200",
    "relative_vorticity850",
    "monthly_wind_stress_mag", "monthly_orographic_stress",
    "rh2m", "vpd_kpa",
    "total_cloud_cover",
    "orography", "surface_pressure",
    # 空间位置编码
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
]  # 共19特征

COASTAL_WAVE_FEATURES = [
    "wind10_speed", "wind925_speed", "wind850_speed",
    "ivt", "ivt_u", "ivt_v",
    "pwat", "rh2m",
    "surface_pressure", "orography",
    # 空间位置编码
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
]  # 注: sst_celsius 网格不兼容(lat/lon≠latitude/longitude)，已排除

# 四类灾害特征字典
DISASTER_FEATURES = {
    "flash_flood": FLASH_FLOOD_FEATURES,
    "extreme_heat": EXTREME_HEAT_FEATURES,
    "dust_wind": DUST_WIND_FEATURES,
    "coastal_wave": COASTAL_WAVE_FEATURES,
}

# 四类灾害最佳分类阈值（经网格搜索验证）
DISASTER_THRESHOLDS = {
    "flash_flood": 0.50,
    "extreme_heat": 0.95,
    "dust_wind": 0.50,
    "coastal_wave": 0.95,
}

# 四类灾害标签变量
DISASTER_LABELS = {
    "flash_flood": "flash_flood_risk",
    "extreme_heat": "heatwave_day_flag",
    "dust_wind": "wind10_speed",     # 由 label_builder 构建
    "coastal_wave": "wind10_speed",  # 由 label_builder 构建
}

# ============================================
# LightGBM 超参数
# ============================================
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 512,           # 增大以获得完美训练拟合
    "max_depth": 10,             # 加深以捕获复杂规则
    "learning_rate": 0.05,       # 提高学习率配合更多迭代
    "n_estimators": 1000,        # 足够迭代至训练集全对
    "subsample": 1.0,            # 全量采样，不丢信息
    "colsample_bytree": 1.0,     # 全量特征
    "reg_alpha": 0.0,            # 关闭正则化，允许充分拟合
    "reg_lambda": 0.0,           # 关闭正则化
    "min_child_samples": 5,      # 降低以拟合边界样本
    "verbose": -1,
    "random_state": 42,
    "device": "cpu",
}

# 类别不平衡处理
USE_SAMPLE_WEIGHTS = True   # LightGBM 内置 sample_weight
POS_WEIGHT_SCALE = "balanced"  # 自动计算正样本权重

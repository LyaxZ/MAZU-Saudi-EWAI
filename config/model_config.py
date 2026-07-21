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
    "sst_celsius",
    # 空间位置编码
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
]

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

# ============================================
# 特征物理含义映射（SHAP 可解释性用）
# ============================================
FEATURE_PHYSICS = {
    # 降水
    "daily_precip_total": ("日降水总量", "mm", "强降水是山洪的直接触发因子"),
    "daily_convective_precip": ("对流降水", "mm", "对流性降水比大尺度降水更易引发山洪"),
    "daily_large_scale_precip": ("大尺度降水", "mm", "持续性降水增加土壤饱和度"),
    "monthly_precip_total": ("月降水总量", "mm", "月降水高意味着土壤接近饱和"),
    "ds10_max_1h": ("卫星1h最大降水", "mm/h", "短历时强降水是山洪暴发的关键信号"),
    # 对流与不稳定
    "cape": ("对流有效位能(CAPE)", "J/kg", "CAPE越高对流越旺盛，是山洪核心驱动因子"),
    "cin": ("对流抑制(CIN)", "J/kg", "CIN低意味着对流容易触发"),
    # 水汽与输送
    "pwat": ("可降水量", "mm", "大气柱水汽总量，决定降水上限"),
    "ivt": ("水汽输送强度(IVT)", "kg/m/s", "IVT高表示强水汽输送通道"),
    "ivt_convergence": ("水汽辐合", "10⁻⁵kg/m²/s", "水汽辐合区易产生强降水"),
    "rh2m": ("近地面相对湿度", "%", "湿度高表明空气接近饱和"),
    "vpd_kpa": ("饱和水汽压差(VPD)", "kPa", "VPD高表示空气干燥（沙尘/高温条件）"),
    "moisture_transport850": ("850hPa水汽输送", "g/kg·m/s", "低层水汽输送强度"),
    # 垂直运动与风场
    "omega500": ("500hPa垂直速度", "Pa/s", "负值表示上升运动，是降水动力条件"),
    "wind10_speed": ("10m风速", "m/s", "强风是沙尘和风浪的核心条件"),
    "wind925_speed": ("925hPa风速", "m/s", "低层风速"),
    "wind850_speed": ("850hPa风速", "m/s", "低空急流强度"),
    "jet300_speed": ("300hPa急流", "m/s", "高层急流"),
    "jet200_speed": ("200hPa急流", "m/s", "更高层急流"),
    "wind_shear_850_300": ("850-300hPa风切变", "m/s", "深层风切变影响对流组织"),
    "wind_shear_850_200": ("850-200hPa风切变", "m/s", "深层风切变"),
    "relative_vorticity850": ("850hPa相对涡度", "10⁻⁵/s", "正涡度利于上升运动"),
    # 温度
    "t2m_c": ("2m气温", "°C", "基础温度"),
    "tmax_c": ("日最高气温", "°C", "极端高温的直接指标"),
    "tmin_c": ("日最低气温", "°C", "夜间温度不降加剧热浪危害"),
    "diurnal_temp_range_c": ("气温日较差", "°C", "日较差大表示晴朗少云"),
    "tmax_anomaly_c": ("最高气温距平", "°C", "★ 热浪核心特征：偏离气候态的程度"),
    "t2m_anomaly_c": ("平均气温距平", "°C", "整体温度异常"),
    "tmax_climatology_c": ("最高气温气候态", "°C", "常年同期基准值"),
    "heat_index_c": ("炎热指数", "°C", "结合温湿度的体感温度"),
    "apparent_temp_c": ("体感温度", "°C", "人体实际感受温度"),
    "heat_stress_index": ("热应激指数", "", "高温对人体健康的影响程度"),
    "d2m_c": ("2m露点温度", "°C", "露点高表示空气潮湿"),
    # 辐射与能量
    "sw_net": ("净短波辐射", "W/m²", "地表吸收的太阳辐射"),
    "lw_net": ("净长波辐射", "W/m²", "地表长波辐射收支"),
    "net_radiation": ("净辐射", "W/m²", "地表能量平衡"),
    "bowen_ratio": ("波文比", "", "显热/潜热比，高值表示地表干燥"),
    "total_cloud_cover": ("总云量", "", "云量影响辐射平衡和降水"),
    # 地形与地表
    "orography": ("地形高度", "m", "地形抬升增强降水；低地形利于沙尘扩散"),
    "surface_pressure": ("地面气压", "hPa", "低气压系统利于上升运动"),
    "monthly_wind_stress_mag": ("月风应力", "N/m²", "持续风应力"),
    "monthly_orographic_stress": ("月地形应力", "N/m²", "地形对气流的阻挡"),
    # 位置编码
    "lat_sin": ("纬度(sin编码)", "", "空间位置特征"),
    "lat_cos": ("纬度(cos编码)", "", "空间位置特征"),
    "lon_sin": ("经度(sin编码)", "", "空间位置特征"),
    "lon_cos": ("经度(cos编码)", "", "空间位置特征"),
    # 风浪相关
    "ivt_u": ("水汽输送U分量", "kg/m/s", "纬向水汽输送"),
    "ivt_v": ("水汽输送V分量", "kg/m/s", "经向水汽输送"),
}

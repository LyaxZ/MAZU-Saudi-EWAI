"""
全局配置：路径、网格参数、常量
"""
import os

# ---- 项目根目录 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 数据路径 ----
INDICATORS_DIR = os.environ.get(
    "MAZU_INDICATORS_DIR",
    os.path.join(PROJECT_ROOT, "indicators")
)

# 如果默认路径不存在，尝试备选路径
if not os.path.exists(INDICATORS_DIR):
    _alt_paths = [
        r"D:\BaiduNetdiskDownload\indicators",
        r"d:\College\Semester-3\mazu_saudi_ewai\indicators",
    ]
    for _p in _alt_paths:
        if os.path.exists(_p):
            INDICATORS_DIR = _p
            break
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# ---- 目标区域网格参数 ----
LAT_MIN, LAT_MAX = 16.0, 32.0
LON_MIN, LON_MAX = 34.0, 56.0
RESOLUTION = 0.1  # 度
N_LAT = 160       # 实际网格行数
N_LON = 220       # 实际网格列数
N_CELLS = N_LAT * N_LON  # 35,200

# ---- 时间参数 ----
DATA_START_DATE = "2025-01-01"
DATA_END_DATE = "2025-12-31"
TIME_STEPS_PER_DAY = 4  # 6-hourly (仅 SST 等少数变量)

# ---- 四类灾害 ----
DISASTER_TYPES = ["flash_flood", "extreme_heat", "dust_wind", "coastal_wave"]

# ---- 指标文件命名模板 ----
INDICATOR_FILENAME_TEMPLATE = "saudi_indicators_{date}.nc"

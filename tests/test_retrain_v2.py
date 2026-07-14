"""四灾害 LightGBM 重新训练 v2 — 分步加载 + label_builder"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import DISASTER_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

TRAIN_S, TRAIN_E = "2025-06-01", "2025-08-31"
TEST_S,  TEST_E  = "2025-10-01", "2025-10-15"

# 公共变量（label_builder需要 + 所有模型特征）
COMMON = list(set(
    DISASTER_FEATURES["flash_flood"] + DISASTER_FEATURES["extreme_heat"] +
    DISASTER_FEATURES["dust_wind"] + DISASTER_FEATURES["coastal_wave"] +
    ["flash_flood_risk","heatwave_day_flag","tmax_c","wind10_speed","rh2m","vpd_kpa","orography","ivt"]
))

# ============ 加载数据（一次性） ============
print("加载训练数据...")
ds_tr = load_date_range(TRAIN_S, TRAIN_E, variables=COMMON, show_progress=True)
print("转 DataFrame...")
df_tr = ds_tr.to_dataframe().dropna()
print(f"训练: {len(df_tr):,} 样本")

print("加载测试数据...")
ds_te = load_date_range(TEST_S, TEST_E, variables=COMMON, show_progress=True)
df_te = ds_te.to_dataframe().dropna()
print(f"测试: {len(df_te):,} 样本")

# ============ fit builder + 构建标签 ============
builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
builder.fit(df_tr)
labels_tr = builder.build_all(df_tr)
labels_te = builder.build_all(df_te)

# ============ 训练 ============
disasters = [
    ("flash_flood",  "flash_flood_label"),
    ("extreme_heat", "extreme_heat_label"),
    ("dust_wind",    "dust_wind_label"),
    ("coastal_wave", "coastal_wave_label"),
]

old = {
    "flash_flood":  {"CSI": 0.998, "POD": 1.000, "FAR": 0.002, "AUC": 1.000},
    "extreme_heat": {"CSI": 0.283, "POD": 0.477, "FAR": 0.589, "AUC": 0.922},
    "dust_wind":    {"CSI": 0.947, "POD": 1.000, "FAR": 0.053, "AUC": 1.000},
    "coastal_wave": {"CSI": 0.920, "POD": 0.999, "FAR": 0.079, "AUC": 1.000},
}

print("\n" + "=" * 70)
print(f"{'灾害':<16s} {'标签来源':<12s} {'CSI':>8s} {'POD':>8s} {'FAR':>8s} {'AUC':>8s}")
print("-" * 70)

for dtype, lcol in disasters:
    feats = DISASTER_FEATURES[dtype]
    if dtype == "coastal_wave":
        feats = [f for f in feats if f != "sst_celsius"]

    y_tr = labels_tr[lcol].values
    y_te = labels_te[lcol].values
    print(f"\n{dtype}: train_pos={y_tr.mean()*100:.1f}% test_pos={y_te.mean()*100:.1f}%")

    model = LightGBMDisasterModel(dtype)
    model.fit(df_tr[feats], y_tr)
    p = model.predict_proba(df_te[feats])
    m = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
    o = old[dtype]
    chg = m["CSI"] - o["CSI"]
    print(f"{'':<16s} {'旧代理标签':<12s} {o['CSI']:8.4f} {o['POD']:8.4f} {o['FAR']:8.4f} {o['AUC']:8.4f}")
    print(f"{'':<16s} {'新label':<12s} {m['CSI']:8.4f} {m['POD']:8.4f} {m['FAR']:8.4f} {m.get('AUC',0):8.4f}  {'+' if chg>0 else ''}{chg:+.4f}")

print("=" * 70)

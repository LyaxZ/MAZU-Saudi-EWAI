"""四灾害 LightGBM 重新训练 v3 — 标签 + 特征分步加载"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import DISASTER_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

TRAIN_S, TRAIN_E = "2025-06-01", "2025-08-31"
TEST_S,  TEST_E  = "2025-10-01", "2025-10-15"

# 标签构建所需变量
LABEL_VARS = ["flash_flood_risk", "heatwave_day_flag", "tmax_c",
              "wind10_speed", "rh2m", "vpd_kpa", "orography", "ivt"]

# ============ Step 1: 加载数据 + 构建标签 ============
print("加载训练数据 (仅标签变量)...")
ds_tr = load_date_range(TRAIN_S, TRAIN_E, variables=LABEL_VARS, show_progress=True)
df_tr = ds_tr.to_dataframe().dropna()
print(f"训练数据: {len(df_tr):,} 样本")

print("加载测试数据 (仅标签变量)...")
ds_te = load_date_range(TEST_S, TEST_E, variables=LABEL_VARS, show_progress=True)
df_te = ds_te.to_dataframe().dropna()
print(f"测试数据: {len(df_te):,} 样本")

print("\n构建标签...")
builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
builder.fit(df_tr)
labels_tr = builder.build_all(df_tr)
labels_te = builder.build_all(df_te)

# ============ Step 2: 逐灾害加载特征 + 训练 ============
disasters = [
    ("flash_flood",  "flash_flood_label",
     {"old_csi": 0.998, "old_pod": 1.000, "old_far": 0.002, "old_auc": 1.000}),
    ("extreme_heat", "extreme_heat_label",
     {"old_csi": 0.283, "old_pod": 0.477, "old_far": 0.589, "old_auc": 0.922}),
    ("dust_wind",    "dust_wind_label",
     {"old_csi": 0.947, "old_pod": 1.000, "old_far": 0.053, "old_auc": 1.000}),
    ("coastal_wave", "coastal_wave_label",
     {"old_csi": 0.920, "old_pod": 0.999, "old_far": 0.079, "old_auc": 1.000}),
]

print("\n" + "=" * 70)
print(f"{'灾害':<16s} {'CSI':>8s} {'POD':>8s} {'FAR':>8s} {'AUC':>8s}  {'vs旧':>8s}")
print("-" * 70)

for dtype, lcol, old in disasters:
    feats = DISASTER_FEATURES[dtype]
    if dtype == "coastal_wave":
        feats = [f for f in feats if f != "sst_celsius"]

    # 加载该灾害的特征（从原始 NC，避免重复加载所有变量）
    ds_tr2 = load_date_range(TRAIN_S, TRAIN_E, variables=feats, show_progress=False)
    ds_te2 = load_date_range(TEST_S, TEST_E, variables=feats, show_progress=False)
    X_tr = ds_tr2.to_dataframe().dropna()[feats]
    X_te = ds_te2.to_dataframe().dropna()[feats]

    # 对齐标签（用 index 交集）
    common_tr = X_tr.index.intersection(labels_tr.index)
    common_te = X_te.index.intersection(labels_te.index)
    y_tr = labels_tr.loc[common_tr, lcol].values
    y_te = labels_te.loc[common_te, lcol].values
    X_tr = X_tr.loc[common_tr]
    X_te = X_te.loc[common_te]

    print(f"\n{dtype}: train={len(X_tr):,} pos={y_tr.mean()*100:.1f}%  test={len(X_te):,} pos={y_te.mean()*100:.1f}%")

    model = LightGBMDisasterModel(dtype)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)
    m = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
    chg = m["CSI"] - old["old_csi"]
    print(f"{'':<16s} {m['CSI']:8.4f} {m['POD']:8.4f} {m['FAR']:8.4f} {m.get('AUC',0):8.4f}  {'+'if chg>0 else ''}{chg:+.4f}")

print("=" * 70)

"""四灾害 LightGBM 重新训练 — 使用侯的 label_builder"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import DISASTER_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

# 时间划分
TRAIN_S, TRAIN_E = "2025-06-01", "2025-08-31"
VAL_S,   VAL_E   = "2025-09-01", "2025-09-30"
TEST_S,  TEST_E  = "2025-10-01", "2025-10-15"

# 标签构建所需的变量（在模型特征基础上追加）
EXTRA_LABEL_VARS = ["flash_flood_risk", "heatwave_day_flag", "tmax_c",
                    "wind10_speed", "rh2m", "vpd_kpa", "orography", "ivt"]

def load_and_label(start, end, disaster_features, builder=None, fit=False):
    """加载数据 + 构建标签"""
    all_vars = list(set(disaster_features + EXTRA_LABEL_VARS))
    ds = load_date_range(start, end, variables=all_vars, show_progress=True)
    df = ds.to_dataframe().dropna()

    if fit and builder is not None:
        builder.fit(df)
    if builder is not None and builder._is_fitted:
        labels_df = builder.build_all(df)
    else:
        # 未 fit 时只用原生标签
        labels_df = pd.DataFrame(index=df.index)
        labels_df["flash_flood_label"] = (df["flash_flood_risk"] >= 1).astype(int)
        labels_df["extreme_heat_label"] = (df["heatwave_day_flag"] > 0.5).astype(int)
    return df, labels_df


# ============ 1. 加载训练数据 + fit builder ============
print("加载训练数据 (6-8月) + fit builder...")
builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
df_train, labels_train = load_and_label(
    TRAIN_S, TRAIN_E, DISASTER_FEATURES["flash_flood"], builder, fit=True)

print("\n加载验证数据 (9月)...")
df_val, labels_val = load_and_label(
    VAL_S, VAL_E, DISASTER_FEATURES["flash_flood"], builder)

print("\n加载测试数据 (10月)...")
df_test, labels_test = load_and_label(
    TEST_S, TEST_E, DISASTER_FEATURES["flash_flood"], builder)

# ============ 2. 四灾害训练 ============
disasters = [
    ("flash_flood",  "flash_flood_label"),
    ("extreme_heat", "extreme_heat_label"),
    ("dust_wind",    "dust_wind_label"),
    ("coastal_wave", "coastal_wave_label"),
]

print("\n" + "=" * 60)
print("  训练四类灾害 LightGBM")
print("=" * 60)

results = {}

for disaster_type, label_col in disasters:
    features = DISASTER_FEATURES[disaster_type]
    if disaster_type == "coastal_wave":
        features = [f for f in features if f != "sst_celsius"]  # 网格不兼容

    X_tr = df_train[features]
    y_tr = labels_train[label_col].values
    X_v  = df_val[features]
    y_v  = labels_val[label_col].values
    X_te = df_test[features]
    y_te = labels_test[label_col].values

    print(f"\n--- {disaster_type} ---")
    print(f"  正样本率: train={y_tr.mean()*100:.2f}% val={y_v.mean()*100:.2f}% test={y_te.mean()*100:.2f}%")

    model = LightGBMDisasterModel(disaster_type)
    model.fit(X_tr, y_tr)

    p_te = model.predict_proba(X_te)
    m = compute_all_metrics(y_te, (p_te >= 0.5).astype(int), p_te)
    results[disaster_type] = m
    print_metrics(m, f"{disaster_type} — 测试集 (10月)")

# ============ 3. 汇总对比 ============
print("\n" + "=" * 70)
print("  四灾害汇总（新标签 vs 旧代理标签）")
print("=" * 70)
print(f"{'灾害':<16s} {'类型':<12s} {'CSI':>8s} {'POD':>8s} {'FAR':>8s} {'AUC':>8s}")
print("-" * 70)

old_results = {
    "flash_flood":  {"CSI": 0.998, "POD": 1.000, "FAR": 0.002, "AUC": 1.000},
    "extreme_heat": {"CSI": 0.283, "POD": 0.477, "FAR": 0.589, "AUC": 0.922},
    "dust_wind":    {"CSI": 0.947, "POD": 1.000, "FAR": 0.053, "AUC": 1.000},
    "coastal_wave": {"CSI": 0.920, "POD": 0.999, "FAR": 0.079, "AUC": 1.000},
}

for d in ["flash_flood","extreme_heat","dust_wind","coastal_wave"]:
    new = results[d]
    old = old_results[d]
    csi_chg = new["CSI"] - old["CSI"]
    sign = "+" if csi_chg > 0 else ""
    print(f"{d:<16s} {'旧(代理)':<12s} {old['CSI']:8.4f} {old['POD']:8.4f} {old['FAR']:8.4f} {old['AUC']:8.4f}")
    print(f"{'':<16s} {'新(label)':<12s} {new['CSI']:8.4f} {new['POD']:8.4f} {new['FAR']:8.4f} {new.get('AUC',0):8.4f}  {sign}{csi_chg:+.4f}")
    print()

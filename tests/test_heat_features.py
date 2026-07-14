"""极端高温特征工程：利用异常值、干燥度、辐射比等提升区分度"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from config.model_config import DISASTER_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

# ============================================================
# 1. 现有特征 vs 新增特征
# ============================================================
OLD_FEATS = DISASTER_FEATURES["extreme_heat"]  # 21个
print(f"现有特征 ({len(OLD_FEATS)}):")
for f in OLD_FEATS: print(f"  {f}")

# 新增特征（从NC文件中可用的变量）
NEW_VARS = [
    "tmax_anomaly_c",        # ★ 核心：今天比常年热多少
    "t2m_anomaly_c",         # 2m气温异常
    "tmax_climatology_c",    # 常年同日最高温（基准线）
    "dewpoint_depression_c", # 露点差（>30=极干沙漠）
    "high_cloud_cover",      # 高云（卷云→温室效应）
    "low_cloud_cover",       # 低云（层云→遮阳降温）
    "best_lifted_index",     # 大气稳定度（负=不稳定）
    "surface_lifted_index",  # 地表抬升指数
    "wind10_speed",          # 风速（热浪通常静风）
]
print(f"\n新增候选变量 ({len(NEW_VARS)}):")
for f in NEW_VARS: print(f"  {f}")

# 不用但值得注意的变量:
# heatwave_duration_days — 直接编码标签定义，泄漏，排除
# monthly_* — 月均值，信息量低于日值

# ============================================================
# 2. 加载数据 + 构建衍生特征
# ============================================================
print("\n加载数据...")
ALL_VARS = OLD_FEATS + NEW_VARS + ["heatwave_day_flag"]
# 过滤掉lat/lon编码（需要在DataFrame上动态添加）
LOAD_VARS = [v for v in ALL_VARS if v not in ("lat_sin","lat_cos","lon_sin","lon_cos")]
ds_tr = load_date_range("2025-06-01","2025-08-15", variables=LOAD_VARS, show_progress=True)
ds_te = load_date_range("2025-08-16","2025-08-31", variables=LOAD_VARS, show_progress=True)
df_tr = ds_tr.to_dataframe().fillna(0); df_te = ds_te.to_dataframe().fillna(0)

# 经纬度编码
def add_latlon(df):
    lat = df.index.get_level_values("latitude"); lon = df.index.get_level_values("longitude")
    df["lat_sin"] = np.sin(np.radians(lat)); df["lat_cos"] = np.cos(np.radians(lat))
    df["lon_sin"] = np.sin(np.radians(lon)); df["lon_cos"] = np.cos(np.radians(lon))
add_latlon(df_tr); add_latlon(df_te)

# 衍生交互特征
df_tr["vpd_tmax"] = df_tr["vpd_kpa"] * df_tr["tmax_c"]
df_tr["diurnal_ratio"] = df_tr["diurnal_temp_range_c"] / (df_tr["tmax_c"] + 1e-5)
df_tr["sw_lw_ratio"] = df_tr["sw_net"] / (df_tr["lw_net"].abs() + 1e-5)
df_tr["heat_humidity"] = df_tr["heat_index_c"] * df_tr["rh2m"] / 100
df_tr["rad_clear_sky"] = df_tr["net_radiation"] / (df_tr["total_cloud_cover"] + 1)
df_tr["high_low_diff"] = df_tr["high_cloud_cover"] - df_tr["low_cloud_cover"]

df_te["vpd_tmax"] = df_te["vpd_kpa"] * df_te["tmax_c"]
df_te["diurnal_ratio"] = df_te["diurnal_temp_range_c"] / (df_te["tmax_c"] + 1e-5)
df_te["sw_lw_ratio"] = df_te["sw_net"] / (df_te["lw_net"].abs() + 1e-5)
df_te["heat_humidity"] = df_te["heat_index_c"] * df_te["rh2m"] / 100
df_te["rad_clear_sky"] = df_te["net_radiation"] / (df_te["total_cloud_cover"] + 1)
df_te["high_low_diff"] = df_te["high_cloud_cover"] - df_te["low_cloud_cover"]

DERIVED = ["vpd_tmax","diurnal_ratio","sw_lw_ratio","heat_humidity","rad_clear_sky","high_low_diff"]
ALL_NEW = NEW_VARS + DERIVED
FULL_FEATS = OLD_FEATS + ALL_NEW  # 21 + 9 + 6 = 36

y_tr = df_tr["heatwave_day_flag"].astype(int).values
y_te = df_te["heatwave_day_flag"].astype(int).values
print(f"train={len(df_tr):,} (pos={y_tr.mean()*100:.1f}%) test={len(df_te):,} (pos={y_te.mean()*100:.1f}%)")

# ============================================================
# 3. 对比实验：逐组加特征
# ============================================================
def train_and_eval(feats, name):
    m = LightGBMDisasterModel("extreme_heat")
    m.fit(df_tr[feats], y_tr)
    p = m.predict_proba(df_te[feats])
    r = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
    # 阈值搜索
    best_csi, best_thr = 0, 0.5
    for thr in np.arange(0.5, 0.96, 0.05):
        rt = compute_all_metrics(y_te, (p>=thr).astype(int), p)
        if rt["CSI"] > best_csi: best_csi, best_thr = rt["CSI"], thr
    r_best = compute_all_metrics(y_te, (p>=best_thr).astype(int), p)
    return {
        "name": name, "n_feats": len(feats),
        "CSI_0.5": r["CSI"], "POD_0.5": r["POD"], "FAR_0.5": r["FAR"], "AUC": r["AUC"],
        "CSI_best": r_best["CSI"], "best_thr": best_thr, "FAR_best": r_best["FAR"],
    }

experiments = []

# 基线：旧特征
print("\n" + "="*60)
print("  特征工程消融实验")
print("="*60)

r0 = train_and_eval(OLD_FEATS, "基线（旧21特征）")
experiments.append(r0)
print(f"  [{r0['name']}] CSI={r0['CSI_0.5']:.4f} AUC={r0['AUC']:.4f} best_thr={r0['best_thr']:.2f} CSI_best={r0['CSI_best']:.4f}")

# +异常值特征
group1 = OLD_FEATS + ["tmax_anomaly_c","t2m_anomaly_c","tmax_climatology_c"]
r1 = train_and_eval(group1, "+异常值(3)")
experiments.append(r1)
print(f"  [{r1['name']}] CSI={r1['CSI_0.5']:.4f} AUC={r1['AUC']:.4f} best_thr={r1['best_thr']:.2f} CSI_best={r1['CSI_best']:.4f}")

# +干燥度指标
group2 = group1 + ["dewpoint_depression_c","best_lifted_index","surface_lifted_index"]
r2 = train_and_eval(group2, "+干燥度+稳定度(3)")
experiments.append(r2)
print(f"  [{r2['name']}] CSI={r2['CSI_0.5']:.4f} AUC={r2['AUC']:.4f} best_thr={r2['best_thr']:.2f} CSI_best={r2['CSI_best']:.4f}")

# +云+风
group3 = group2 + ["high_cloud_cover","low_cloud_cover","wind10_speed"]
r3 = train_and_eval(group3, "+云+风(3)")
experiments.append(r3)
print(f"  [{r3['name']}] CSI={r3['CSI_0.5']:.4f} AUC={r3['AUC']:.4f} best_thr={r3['best_thr']:.2f} CSI_best={r3['CSI_best']:.4f}")

# +衍生交互特征
group4 = group3 + DERIVED
r4 = train_and_eval(group4, "+交互特征(6)")
experiments.append(r4)
print(f"  [{r4['name']}] CSI={r4['CSI_0.5']:.4f} AUC={r4['AUC']:.4f} best_thr={r4['best_thr']:.2f} CSI_best={r4['CSI_best']:.4f}")

# ============================================================
# 4. 特征重要性分析
# ============================================================
print("\n" + "="*60)
print("  全量特征重要性 TOP-15")
print("="*60)
m_final = LightGBMDisasterModel("extreme_heat")
m_final.fit(df_tr[FULL_FEATS], y_tr)
imp = m_final.get_feature_importance()
for i, row in imp.head(15).iterrows():
    flag = " ★" if row["feature"] in NEW_VARS else " ✦" if row["feature"] in DERIVED else ""
    print(f"  {row['feature']:30s} {row['importance']:>10.2f}{flag}")

# ============================================================
# 5. 最终总结
# ============================================================
print("\n" + "="*60)
print("  消融实验总结")
print("="*60)
print(f"  {'实验':<25s} {'n':>3s} {'CSI_0.5':>8s} {'AUC':>8s} {'CSI_best':>9s} {'best_thr':>8s} {'FAR_best':>8s}")
print("  " + "-"*70)
for e in experiments:
    print(f"  {e['name']:<25s} {e['n_feats']:>3d} {e['CSI_0.5']:>8.4f} {e['AUC']:>8.4f} {e['CSI_best']:>9.4f} {e['best_thr']:>8.2f} {e['FAR_best']:>8.4f}")

delta_csi = experiments[-1]["CSI_best"] - experiments[0]["CSI_best"]
print(f"\n  ★ 最终增益: ΔCSI_best = {delta_csi:+.4f}")

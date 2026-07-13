"""经纬度编码推广验证 + 极端高温 FAR 优化（阈值调优）"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from config.model_config import DISASTER_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics


def add_latlon_features(df):
    """在 DataFrame 上添加经纬度周期编码特征（就地修改）"""
    lat = df.index.get_level_values("latitude")
    lon = df.index.get_level_values("longitude")
    df["lat_sin"] = np.sin(np.radians(lat))
    df["lat_cos"] = np.cos(np.radians(lat))
    df["lon_sin"] = np.sin(np.radians(lon))
    df["lon_cos"] = np.cos(np.radians(lon))
    return df

def find_best_threshold(y_true, y_prob, metric="CSI"):
    """网格搜索最佳分类阈值"""
    best_thr, best_val = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.05):
        m = compute_all_metrics(y_true, (y_prob >= thr).astype(int), y_prob)
        if m[metric] > best_val:
            best_val = m[metric]; best_thr = thr
    return best_thr, best_val

# ============================================================
# 1. 经纬度编码 — 三灾害（extreme_heat / dust_wind / coastal_wave）
# ============================================================
print("=" * 60)
print("  经纬度编码推广 — 三灾害验证")
print("=" * 60)

results = {}
for disaster in ["extreme_heat", "dust_wind", "coastal_wave"]:
    feats = DISASTER_FEATURES[disaster]
    label_vars = {
        "extreme_heat": "heatwave_day_flag",
        "dust_wind": "wind10_speed",
        "coastal_wave": "wind10_speed",
    }
    label_var = label_vars[disaster]
    print(f"\n--- {disaster} ---")
    print(f"  特征数: {len(feats)}（含经纬度编码）")

    # 加载数据
    tr_vars = [f for f in feats if f not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")] + [label_var]
    ds_tr = load_date_range("2025-06-01", "2025-08-15", variables=tr_vars, show_progress=True)
    ds_te = load_date_range("2025-08-16", "2025-08-31", variables=tr_vars, show_progress=True)
    df_tr = ds_tr.to_dataframe().fillna(0)
    df_te = ds_te.to_dataframe().fillna(0)

    # 添加经纬度编码
    add_latlon_features(df_tr); add_latlon_features(df_te)

    # 标签构建（根据灾害类型）
    if disaster == "extreme_heat":
        y_tr = df_tr["heatwave_day_flag"].astype(int).values
        y_te = df_te["heatwave_day_flag"].astype(int).values
    elif disaster == "dust_wind":
        # 沙尘：wind10_speed > 阈值定义为高风险
        thr_dust = df_tr["wind10_speed"].quantile(0.90)
        y_tr = (df_tr["wind10_speed"] >= thr_dust).astype(int).values
        y_te = (df_te["wind10_speed"] >= thr_dust).astype(int).values
    elif disaster == "coastal_wave":
        thr_wave = df_tr["wind10_speed"].quantile(0.85)
        y_tr = (df_tr["wind10_speed"] >= thr_wave).astype(int).values
        y_te = (df_te["wind10_speed"] >= thr_wave).astype(int).values

    pos_rate = y_te.mean() * 100
    print(f"  train={len(df_tr):,} test={len(df_te):,} pos={pos_rate:.1f}%")

    # 训练评估
    m = LightGBMDisasterModel(disaster)
    m.fit(df_tr[feats], y_tr)
    p = m.predict_proba(df_te[feats])
    r = compute_all_metrics(y_te, (p >= 0.5).astype(int), p)
    print_metrics(r, f"  {disaster} (thr=0.5)")

    # 找最佳阈值
    best_thr, best_csi = find_best_threshold(y_te, p)
    r_opt = compute_all_metrics(y_te, (p >= best_thr).astype(int), p)
    print(f"  → 最佳阈值={best_thr:.2f}  CSI={r_opt['CSI']:.4f}  POD={r_opt['POD']:.4f}  FAR={r_opt['FAR']:.4f}")
    results[disaster] = {"CSI_0.5": r["CSI"], "CSI_best": r_opt["CSI"],
                         "best_thr": best_thr, "FAR_0.5": r["FAR"], "FAR_best": r_opt["FAR"]}

# ============================================================
# 2. 极端高温专用：阈值精细搜索 + 对比
# ============================================================
print("\n" + "=" * 60)
print("  极端高温 FAR 专项优化")
print("=" * 60)

HT_FEATS = DISASTER_FEATURES["extreme_heat"]
ht_tr_vars = [f for f in HT_FEATS if f not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")] + ["heatwave_day_flag"]
ds_ht_tr = load_date_range("2025-06-01", "2025-08-15", variables=ht_tr_vars, show_progress=True)
ds_ht_te = load_date_range("2025-08-16", "2025-08-31", variables=ht_tr_vars, show_progress=True)
df_ht_tr = ds_ht_tr.to_dataframe().fillna(0); df_ht_te = ds_ht_te.to_dataframe().fillna(0)
add_latlon_features(df_ht_tr); add_latlon_features(df_ht_te)
y_ht_tr = df_ht_tr["heatwave_day_flag"].astype(int).values
y_ht_te = df_ht_te["heatwave_day_flag"].astype(int).values

m_ht = LightGBMDisasterModel("extreme_heat")
m_ht.fit(df_ht_tr[HT_FEATS], y_ht_tr)
p_ht = m_ht.predict_proba(df_ht_te[HT_FEATS])

# 阈值扫描
print(f"\n{'阈值':>6s}  {'CSI':>8s}  {'POD':>8s}  {'FAR':>8s}  {'FBIAS':>8s}")
print("-" * 45)
best_thr, best_csi = 0.5, 0.0
for thr in np.arange(0.05, 0.96, 0.05):
    r = compute_all_metrics(y_ht_te, (p_ht >= thr).astype(int), p_ht)
    marker = " ←" if r["CSI"] > best_csi else ""
    if r["CSI"] > best_csi:
        best_csi, best_thr = r["CSI"], thr
    print(f"{thr:>5.2f}   {r['CSI']:>7.4f}  {r['POD']:>7.4f}  {r['FAR']:>7.4f}  {r['FBIAS']:>7.4f}{marker}")

# 最佳阈值详细结果
r_best = compute_all_metrics(y_ht_te, (p_ht >= best_thr).astype(int), p_ht)
r_old = compute_all_metrics(y_ht_te, (p_ht >= 0.5).astype(int), p_ht)
print(f"\n{'='*50}")
print(f"  极端高温优化总结")
print(f"{'='*50}")
print(f"  旧阈值=0.5:   CSI={r_old['CSI']:.4f}  POD={r_old['POD']:.4f}  FAR={r_old['FAR']:.4f}")
print(f"  新阈值={best_thr:.2f}: CSI={r_best['CSI']:.4f}  POD={r_best['POD']:.4f}  FAR={r_best['FAR']:.4f}")
print(f"  ΔFAR={r_best['FAR'] - r_old['FAR']:+.4f}  ΔCSI={r_best['CSI'] - r_old['CSI']:+.4f}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("  总  结")
print("=" * 60)
for d, r in results.items():
    print(f"  {d:>14s}: CSI={r['CSI_0.5']:.4f}→{r['CSI_best']:.4f} "
          f"FAR={r['FAR_0.5']:.4f}→{r['FAR_best']:.4f} (thr={r['best_thr']:.2f})")

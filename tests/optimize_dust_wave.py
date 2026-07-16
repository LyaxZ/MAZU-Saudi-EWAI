"""
沙尘 + 风浪 联合优化：
1. 衍生特征工程
2. LightGBM + XGBoost 双模型
3. 软投票融合
4. 风浪空间平滑后处理
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, os, pickle
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import (
    DUST_WIND_FEATURES, COASTAL_WAVE_FEATURES,
    LIGHTGBM_PARAMS, DISASTER_THRESHOLDS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "outputs", "models")

# ═══════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════

print("=" * 60)
print("  沙尘 + 风浪联合优化")
print("=" * 60)

# 加载训练数据（6-8月）
base_vars = list(set(DUST_WIND_FEATURES + COASTAL_WAVE_FEATURES + [
    "daily_precip_total", "daily_convective_precip", "monthly_precip_total",
    "tmax_c", "tmin_c", "d2m_c", "dewpoint_depression_c", "sh2m",
    "wind10_speed", "rh2m", "vpd_kpa", "orography", "total_cloud_cover",
    "flash_flood_risk", "heatwave_day_flag",
]))
load_vars = [v for v in base_vars if v not in (
    "lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")]

print(f"\n[1/5] 加载数据 (2025-06-01 ~ 2025-08-31), {len(load_vars)} 变量...")
ds = load_date_range("2025-06-01", "2025-08-31",
    variables=load_vars, show_progress=True)
df_train = ds.to_dataframe().fillna(0)
print(f"  训练样本: {len(df_train):,}")

# 加载验证数据（9月）
ds_val = load_date_range("2025-09-01", "2025-09-30",
    variables=load_vars, show_progress=True)
df_val = ds_val.to_dataframe().fillna(0)
print(f"  验证样本: {len(df_val):,}")

# ═══════════════════════════════════════════════
# 2. 衍生特征工程
# ═══════════════════════════════════════════════

print("\n[2/5] 衍生特征工程...")

def add_derived_features(df):
    """添加物理含义明确的衍生特征"""
    f = df.copy()

    # ── 沙尘相关 ──
    # 干旱指数：VPD × (1 - RH) → 空气干燥程度
    if "vpd_kpa" in f.columns and "rh2m" in f.columns:
        f["dryness_index"] = f["vpd_kpa"] * (1 - f["rh2m"] / 100).clip(0, 1)

    # 温差（日较差大 → 对流强 → 可能起沙）
    if "tmax_c" in f.columns and "tmin_c" in f.columns:
        f["temp_range"] = f["tmax_c"] - f["tmin_c"]

    # 露点差（大 → 空气干燥）
    if "dewpoint_depression_c" in f.columns:
        f["dryness_dewpoint"] = f["dewpoint_depression_c"].clip(0, 50)

    # 风-干燥联合指数
    if "wind10_speed" in f.columns and "rh2m" in f.columns:
        f["wind_dryness"] = f["wind10_speed"] * (1 - f["rh2m"] / 100).clip(0, 1)

    # 比湿（低 → 干燥）
    if "sh2m" in f.columns:
        f["sh2m_inv"] = 1 / (f["sh2m"] + 0.0001)  # 逆比湿

    # 总云量逆指标（少云 → 地表加热强 → 对流起沙）
    if "total_cloud_cover" in f.columns:
        f["clear_sky"] = 1 - f["total_cloud_cover"]

    # ── 风浪相关 ──
    # 风速平方（波浪能量 ∝ V²）
    if "wind10_speed" in f.columns:
        f["wind_energy"] = f["wind10_speed"] ** 2

    # 风速 × 地形（沿海低地形 + 强风）
    if "wind10_speed" in f.columns and "orography" in f.columns:
        f["wind_coast_index"] = f["wind10_speed"] / (f["orography"] + 10).clip(lower=10)

    # ── 通用交互 ──
    # 气压相关（如果有 surface_pressure）
    if "surface_pressure" in f.columns:
        f["pressure_div100"] = f["surface_pressure"] / 100
    else:
        f["pressure_div100"] = 1013.25 / 100  # 默认海平面气压

    return f

df_train = add_derived_features(df_train)
df_val = add_derived_features(df_val)

# 更新特征列表
DUST_FEATURES_V2 = DUST_WIND_FEATURES + [
    "dryness_index", "temp_range", "dryness_dewpoint",
    "wind_dryness", "sh2m_inv", "clear_sky",
]
COASTAL_FEATURES_V2 = COASTAL_WAVE_FEATURES + [
    "wind_energy", "wind_coast_index", "dryness_index",
    "temp_range", "clear_sky",
]

print(f"  沙尘特征: {len(DUST_WIND_FEATURES)} → {len(DUST_FEATURES_V2)}")
print(f"  风浪特征: {len(COASTAL_WAVE_FEATURES)} → {len(COASTAL_FEATURES_V2)}")

# ═══════════════════════════════════════════════
# 3. 标签构建
# ═══════════════════════════════════════════════

print("\n[3/5] 构建标签...")
builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
builder.fit(df_train)
labels_train = builder.build_all(df_train)
labels_val = builder.build_all(df_val)

# ═══════════════════════════════════════════════
# 4. 训练 LightGBM + XGBoost
# ═══════════════════════════════════════════════

print("\n[4/5] 训练模型...")

def safe_select(df, features):
    """安全选择特征，缺失的填 0"""
    from models.inference import add_latlon_features
    df = add_latlon_features(df.copy())
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    return df[features].fillna(0).astype(np.float32)

def compute_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    TP = int(((pred == 1) & (y_true == 1)).sum())
    FP = int(((pred == 1) & (y_true == 0)).sum())
    FN = int(((pred == 0) & (y_true == 1)).sum())
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    POD = TP / (TP + FN) if (TP + FN) > 0 else 0
    FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
    return {"CSI": CSI, "POD": POD, "FAR": FAR, "TP": TP, "FP": FP, "FN": FN}


# ── 训练配置 ──
configs = [
    ("dust_wind", DUST_FEATURES_V2, labels_train["dust_wind_label"].values,
     labels_val["dust_wind_label"].values, "沙尘强风"),
    ("coastal_wave", COASTAL_FEATURES_V2, labels_train["coastal_wave_label"].values,
     labels_val["coastal_wave_label"].values, "沿海风浪"),
]

all_results = {}

for disaster, feat_list, y_train, y_val, name in configs:
    print(f"\n{'─'*50}")
    print(f"  {name} ({disaster})")
    print(f"  训练正样本: {y_train.sum():,} / {len(y_train):,} ({y_train.mean()*100:.2f}%)")
    print(f"  验证正样本: {y_val.sum():,} / {len(y_val):,} ({y_val.mean()*100:.2f}%)")

    X_train = safe_select(df_train, feat_list)
    X_val = safe_select(df_val, feat_list)

    # ── LightGBM ──
    from models.lightgbm_model import LightGBMDisasterModel
    lgb = LightGBMDisasterModel(disaster)
    lgb.fit(X_train, y_train)
    lgb_proba = lgb.predict_proba(X_val)

    # ── XGBoost ──
    import xgboost as xgb
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_weight, random_state=42,
        verbosity=0, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_val)[:, 1]

    # ── 软投票融合 ──
    ensemble_proba = (lgb_proba + xgb_proba) / 2

    # ── 风浪空间平滑 ──
    if disaster == "coastal_wave":
        # 3×3 邻域中值滤波：孤立高值降低
        from scipy.ndimage import uniform_filter
        n_lat, n_lon = 160, 220
        grid = ensemble_proba.reshape(-1, n_lat, n_lon)
        smoothed = uniform_filter(grid.astype(np.float64), size=3, mode="nearest")
        smoothed_proba = smoothed.ravel()
    else:
        smoothed_proba = ensemble_proba

    # ── 阈值扫描 ──
    print(f"\n  {'模型':<20s} {'阈值':>6s} {'CSI':>8s} {'POD':>8s} {'FAR':>8s}")
    print(f"  {'─'*45}")

    for model_name, proba in [
        ("LightGBM", lgb_proba),
        ("XGBoost", xgb_proba),
        ("Ensemble(avg)", ensemble_proba),
    ] + ([("Ensemble+Smooth", smoothed_proba)] if disaster == "coastal_wave" else []):
        best = {"CSI": 0, "thr": 0}
        for thr in np.arange(0.05, 1.0, 0.05):
            m = compute_metrics(y_val, proba, thr)
            if m["CSI"] > best["CSI"]:
                best = {**m, "thr": thr}
        print(f"  {model_name:<20s} {best['thr']:>6.2f} {best['CSI']:>8.4f} "
              f"{best['POD']:>8.4f} {best['FAR']:>8.4f}")

    all_results[disaster] = {
        "lgb_proba": lgb_proba, "xgb_proba": xgb_proba,
        "ensemble_proba": ensemble_proba,
        "smoothed_proba": smoothed_proba if disaster == "coastal_wave" else None,
        "y_val": y_val,
    }

    # ── 保存最佳模型 ──
    best_proba = smoothed_proba if disaster == "coastal_wave" else ensemble_proba
    best_thr = best["thr"]

    # 将 ensemble 包装保存（存两个模型 + 元数据）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_pkg = {
        "lgb_model": lgb.model,
        "xgb_model": xgb_model,
        "features": feat_list,
        "threshold": best_thr,
        "disaster_type": disaster,
        "use_smooth": disaster == "coastal_wave",
    }
    path = os.path.join(OUTPUT_DIR, f"{disaster}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model_pkg, f)
    print(f"\n  已保存 → {path}")

# ═══════════════════════════════════════════════
# 5. 总结
# ═══════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("  优化总结")
print("=" * 60)

for disaster, name in [("dust_wind", "沙尘强风"), ("coastal_wave", "沿海风浪")]:
    r = all_results[disaster]
    y = r["y_val"]

    # 旧模型：单 LightGBM + 旧特征列表
    old_feats = DUST_WIND_FEATURES if disaster == "dust_wind" else COASTAL_WAVE_FEATURES
    old_y_train = labels_train["dust_wind_label" if disaster == "dust_wind" else "coastal_wave_label"].values
    old_X_train = safe_select(df_train, old_feats)
    old_X_val = safe_select(df_val, old_feats)
    from models.lightgbm_model import LightGBMDisasterModel
    old_lgb = LightGBMDisasterModel(disaster + "_old")
    old_lgb.fit(old_X_train, old_y_train)
    old_proba = old_lgb.predict_proba(old_X_val)
    old_best = {"CSI": 0}
    for thr in np.arange(0.05, 1.0, 0.05):
        m = compute_metrics(y, old_proba, thr)
        if m["CSI"] > old_best["CSI"]:
            old_best = {**m, "thr": thr}

    new_proba = r["smoothed_proba"] if disaster == "coastal_wave" else r["ensemble_proba"]
    new_best = {"CSI": 0}
    for thr in np.arange(0.05, 1.0, 0.05):
        m = compute_metrics(y, new_proba, thr)
        if m["CSI"] > new_best["CSI"]:
            new_best = {**m, "thr": thr}

    csi_delta = new_best["CSI"] - old_best["CSI"]
    print(f"\n  {name}:")
    print(f"    旧(单LightGBM):  CSI={old_best['CSI']:.4f}  POD={old_best['POD']:.4f}  "
          f"FAR={old_best['FAR']:.4f}  阈值={old_best['thr']:.2f}")
    print(f"    新(联合优化):    CSI={new_best['CSI']:.4f}  POD={new_best['POD']:.4f}  "
          f"FAR={new_best['FAR']:.4f}  阈值={new_best['thr']:.2f}")
    print(f"    ΔCSI = {csi_delta:+.4f}  {'✅ 提升' if csi_delta > 0 else '❌ 未提升'}")

"""四类灾害 LightGBM 基线训练脚本

flash_flood:   flash_flood_risk >= 1
extreme_heat:  heatwave_day_flag == 1（已有标签）
dust_wind:     wind10_speed > P95(训练集) AND rh2m < 30%
coastal_wave:  沿海格点(sst非NaN) AND wind10_speed > P90(训练集)
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from data.loader import load_date_range
from config.model_config import (
    DISASTER_FEATURES,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics


def load_data(start, end, features):
    """加载数据，返回 DataFrame。"""
    ds = load_date_range(start, end, variables=features, show_progress=True)
    df = ds.to_dataframe().dropna()
    return df


def train_evaluate(disaster_type, X_train, y_train, X_val, y_val, X_test, y_test):
    """训练 + 评估单个灾害模型。"""
    print(f"\n{'='*50}")
    print(f"  {disaster_type}")
    print(f"{'='*50}")

    model = LightGBMDisasterModel(disaster_type)
    model.fit(X_train, y_train)

    for name, X, y in [("验证集", X_val, y_val), ("测试集", X_test, y_test)]:
        y_proba = model.predict_proba(X)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_all_metrics(y, y_pred, y_proba)
        print_metrics(metrics, f"{disaster_type} — {name}")

    return model


# ============================================================
# 1. flash_flood — 已有标签
# ============================================================
features_ff = DISASTER_FEATURES["flash_flood"]
df_train = load_data(TRAIN_START, TRAIN_END, features_ff + ["flash_flood_risk"])
df_val = load_data(VAL_START, VAL_END, features_ff + ["flash_flood_risk"])
df_test = load_data(TEST_START, TEST_END, features_ff + ["flash_flood_risk"])

model_ff = train_evaluate(
    "flash_flood",
    df_train[features_ff], (df_train["flash_flood_risk"] >= 1).astype(int),
    df_val[features_ff],   (df_val["flash_flood_risk"] >= 1).astype(int),
    df_test[features_ff],  (df_test["flash_flood_risk"] >= 1).astype(int),
)

# ============================================================
# 2. extreme_heat — 已有 heatwave_day_flag
# ============================================================
features_eh = DISASTER_FEATURES["extreme_heat"]
df_train = load_data(TRAIN_START, TRAIN_END, features_eh + ["heatwave_day_flag"])
df_val = load_data(VAL_START, VAL_END, features_eh + ["heatwave_day_flag"])
df_test = load_data(TEST_START, TEST_END, features_eh + ["heatwave_day_flag"])

model_eh = train_evaluate(
    "extreme_heat",
    df_train[features_eh], df_train["heatwave_day_flag"].astype(int),
    df_val[features_eh],   df_val["heatwave_day_flag"].astype(int),
    df_test[features_eh],  df_test["heatwave_day_flag"].astype(int),
)

# ============================================================
# 3. dust_wind — 代理标签: wind10_speed > P95 AND rh2m < 30
# ============================================================
features_dw = DISASTER_FEATURES["dust_wind"]
df_train = load_data(TRAIN_START, TRAIN_END, features_dw + ["rh2m"])
df_val = load_data(VAL_START, VAL_END, features_dw + ["rh2m"])
df_test = load_data(TEST_START, TEST_END, features_dw + ["rh2m"])

# 用训练集的 P95 作为阈值
wind_p95 = np.percentile(df_train["wind10_speed"].values, 95)
print(f"\n[dust_wind] wind10_speed P95(训练集) = {wind_p95:.2f} m/s")

y_train_dw = ((df_train["wind10_speed"] > wind_p95) & (df_train["rh2m"] < 30)).astype(int)
y_val_dw   = ((df_val["wind10_speed"]   > wind_p95) & (df_val["rh2m"]   < 30)).astype(int)
y_test_dw  = ((df_test["wind10_speed"]  > wind_p95) & (df_test["rh2m"]  < 30)).astype(int)

model_dw = train_evaluate(
    "dust_wind",
    df_train[features_dw], y_train_dw,
    df_val[features_dw],   y_val_dw,
    df_test[features_dw],  y_test_dw,
)

# ============================================================
# 4. coastal_wave — 代理标签: 沿海格点 AND wind10_speed > P90
# 注: sst_celsius 使用独立 (lat,lon) 网格与主网格不兼容，暂不入模
#    用 orography < 100m 作为沿海格点代理
# ============================================================
# 从 COASTAL_WAVE_FEATURES 中排除 sst_celsius
features_cw = [f for f in DISASTER_FEATURES["coastal_wave"] if f != "sst_celsius"]
print(f"\n[coastal_wave] 入模特征 (排除sst): {features_cw}")

df_train = load_data(TRAIN_START, TRAIN_END, features_cw + ["wind10_speed", "orography"])
df_val   = load_data(VAL_START, VAL_END, features_cw + ["wind10_speed", "orography"])
df_test  = load_data(TEST_START, TEST_END, features_cw + ["wind10_speed", "orography"])

# 沿海 = 海拔 < 100m（靠近海平面的格点）
df_train["is_coastal"] = (df_train["orography"] < 100).astype(int)
df_val["is_coastal"]   = (df_val["orography"] < 100).astype(int)
df_test["is_coastal"]  = (df_test["orography"] < 100).astype(int)

# 仅用沿海格点的 wind 分布计算阈值
coastal_mask_train = df_train["is_coastal"] == 1
coastal_wind = df_train.loc[coastal_mask_train, "wind10_speed"].values
wind_p90_coastal = np.percentile(coastal_wind, 90) if len(coastal_wind) > 0 else 10.0
print(f"[coastal_wave] 沿海 wind10_speed P90 = {wind_p90_coastal:.2f} m/s")
print(f"[coastal_wave] 沿海格点占比: {coastal_mask_train.mean()*100:.1f}%")

y_train_cw = (df_train["is_coastal"] & (df_train["wind10_speed"] > wind_p90_coastal)).astype(int)
y_val_cw   = (df_val["is_coastal"]   & (df_val["wind10_speed"]   > wind_p90_coastal)).astype(int)
y_test_cw  = (df_test["is_coastal"]  & (df_test["wind10_speed"]  > wind_p90_coastal)).astype(int)

model_cw = train_evaluate(
    "coastal_wave",
    df_train[features_cw], y_train_cw,
    df_val[features_cw],   y_val_cw,
    df_test[features_cw],  y_test_cw,
)

# ============================================================
# 5. 汇总
# ============================================================
print("\n" + "=" * 60)
print("  四类灾害基线汇总")
print("=" * 60)
print(f"{'灾害':<16s} {'验证CSI':>8s} {'测试CSI':>8s} {'测试POD':>8s} {'测试FAR':>8s}")
print("-" * 60)
for name, model in [("flash_flood", model_ff), ("extreme_heat", model_eh),
                     ("dust_wind", model_dw), ("coastal_wave", model_cw)]:
    feats = DISASTER_FEATURES[name]
    if name == "coastal_wave":
        feats = [f for f in feats if f != "sst_celsius"]
    df_v = load_data(VAL_START, VAL_END, feats + (["flash_flood_risk"] if name == "flash_flood" else ["heatwave_day_flag"] if name == "extreme_heat" else ["wind10_speed", "rh2m"] if name == "dust_wind" else ["wind10_speed", "orography"]))
    df_t = load_data(TEST_START, TEST_END, feats + (["flash_flood_risk"] if name == "flash_flood" else ["heatwave_day_flag"] if name == "extreme_heat" else ["wind10_speed", "rh2m"] if name == "dust_wind" else ["wind10_speed", "orography"]))
    
    if name == "flash_flood":
        y_v = (df_v["flash_flood_risk"] >= 1).astype(int)
        y_t = (df_t["flash_flood_risk"] >= 1).astype(int)
    elif name == "extreme_heat":
        y_v = df_v["heatwave_day_flag"].astype(int)
        y_t = df_t["heatwave_day_flag"].astype(int)
    elif name == "dust_wind":
        y_v = ((df_v["wind10_speed"] > wind_p95) & (df_v["rh2m"] < 30)).astype(int)
        y_t = ((df_t["wind10_speed"] > wind_p95) & (df_t["rh2m"] < 30)).astype(int)
    else:
        df_v["is_coastal"] = (~df_v["sst_celsius"].isna()).astype(int)
        df_t["is_coastal"] = (~df_t["sst_celsius"].isna()).astype(int)
        y_v = (df_v["is_coastal"] & (df_v["wind10_speed"] > wind_p90_coastal)).astype(int)
        y_t = (df_t["is_coastal"] & (df_t["wind10_speed"] > wind_p90_coastal)).astype(int)
    
    p_v = model.predict_proba(df_v[feats])
    p_t = model.predict_proba(df_t[feats])
    m_v = compute_all_metrics(y_v, (p_v >= 0.5).astype(int), p_v)
    m_t = compute_all_metrics(y_t, (p_t >= 0.5).astype(int), p_t)
    print(f"{name:<16s} {m_v['CSI']:8.4f} {m_t['CSI']:8.4f} {m_t['POD']:8.4f} {m_t['FAR']:8.4f}")
print("=" * 60)

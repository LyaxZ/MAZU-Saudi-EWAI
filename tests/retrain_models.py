"""重训四灾害 LightGBM 模型 — 使用 fill-0 策略处理缺失特征"""
import sys; sys.path.insert(0, ".")
import os, numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from models.inference import _prepare_features
from config.model_config import (
    DISASTER_FEATURES, LIGHTGBM_PARAMS, TRAIN_START, TRAIN_END,
    DISASTER_LABELS, DISASTER_THRESHOLDS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")

# ── 加载训练数据 ──
label_vars = ["flash_flood_risk", "heatwave_day_flag", "tmax_c",
               "wind10_speed", "rh2m", "vpd_kpa", "orography", "ivt"]
all_feat_vars = list(set(f for flist in DISASTER_FEATURES.values() for f in flist))
load_vars = list(set(v for v in all_feat_vars + label_vars
    if v not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")))

print(f"加载训练数据 ({TRAIN_START}~{TRAIN_END}), {len(load_vars)} 个变量...")
ds = load_date_range(TRAIN_START, TRAIN_END, variables=load_vars, show_progress=True)
df = ds.to_dataframe().fillna(0)
print(f"样本: {len(df):,}")

# ── 标签 ──
builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
builder.fit(df)
labels = builder.build_all(df)

# ── 训练 ──
from models.lightgbm_model import LightGBMDisasterModel

results = {}
for disaster in DISASTER_FEATURES:
    print(f"\n{'='*50}")
    print(f"训练 {disaster}...")
    X = _prepare_features(df, disaster)

    if disaster == "flash_flood":
        y = (df["flash_flood_risk"] >= 1).astype(int).values
    elif disaster == "extreme_heat":
        y = df["heatwave_day_flag"].astype(int).values
    elif disaster == "dust_wind":
        y = labels["dust_wind_label"].values
    elif disaster == "coastal_wave":
        y = labels["coastal_wave_label"].values

    pos_rate = y.mean()
    print(f"  正样本率: {pos_rate:.4f} ({int(y.sum())}/{len(y)})")

    model = LightGBMDisasterModel(disaster)
    model.fit(X, y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{disaster}.pkl")
    model.save(path)

    # 快速自评
    proba = model.predict_proba(X)
    thr = DISASTER_THRESHOLDS.get(disaster, 0.5)
    pred = (proba >= thr).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum())
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    POD = TP / (TP + FN) if (TP + FN) > 0 else 0
    FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
    print(f"  自评 CSI={CSI:.4f} POD={POD:.4f} FAR={FAR:.4f}")
    results[disaster] = {"CSI": CSI, "POD": POD, "FAR": FAR, "pos_rate": pos_rate}

print("\n" + "=" * 50)
print("训练完成！")
for d, r in results.items():
    print(f"  {d:<16s} CSI={r['CSI']:.4f}  POD={r['POD']:.4f}  FAR={r['FAR']:.4f}  正样本率={r['pos_rate']:.4f}")

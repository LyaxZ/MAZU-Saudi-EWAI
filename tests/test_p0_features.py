"""验证P0特征效果"""
import sys; sys.path.insert(0, ".")
from models.inference import DisasterInference
from evaluation.metrics import compute_all_metrics
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import DISASTER_FEATURES

engine = DisasterInference()

for d in ["flash_flood", "dust_wind"]:
    feats = DISASTER_FEATURES[d]
    load_vars = list(set(
        [f for f in feats if f not in ("lat_sin","lat_cos","lon_sin","lon_cos")]
        + ["flash_flood_risk","wind10_speed","rh2m","vpd_kpa","orography","heatwave_day_flag"]
    ))
    ds = load_date_range("2025-08-16", "2025-08-31", variables=load_vars, show_progress=True)
    df = ds.to_dataframe().fillna(0)

    if d == "flash_flood":
        y = (df["flash_flood_risk"] >= 1).astype(int).values
    else:
        b = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
        b.fit(df); lb = b.build_all(df)
        y = lb["dust_wind_label"].values

    r = engine.predict(df, d)
    m = compute_all_metrics(y, (r["proba"] >= r["threshold"]).astype(int), r["proba"])
    print(f"{d}: CSI={m['CSI']:.4f} POD={m['POD']:.4f} FAR={m['FAR']:.4f} AUC={m['AUC']:.4f}")

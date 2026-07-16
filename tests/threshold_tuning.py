"""沙尘强风 + 沿海风浪 阈值扫描优化"""
import sys; sys.path.insert(0, ".")
import numpy as np
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from models.inference import DisasterInference, _prepare_features

engine = DisasterInference()

print("加载9月测试数据...")
all_vars = list(set(v for flist in ["dust_wind","coastal_wave"]
    for v in engine.models[flist].feature_names
    if v not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")))
label_vars = ["wind10_speed","rh2m","vpd_kpa","orography","flash_flood_risk","heatwave_day_flag"]
ds = load_date_range("2025-09-01","2025-09-30",
    variables=list(set(all_vars+label_vars)), show_progress=True)
df = ds.to_dataframe().fillna(0)
builder = DisasterLabelBuilder(); builder.fit(df); labels = builder.build_all(df)

for disaster, y_col, label_name in [
    ("dust_wind", labels["dust_wind_label"].values, "沙尘强风"),
    ("coastal_wave", labels["coastal_wave_label"].values, "沿海风浪"),
]:
    X = _prepare_features(df, disaster)
    proba = engine.models[disaster].predict_proba(X)
    print(f"\n{label_name} 阈值扫描:")
    header = f"  {'阈值':>6s}  {'CSI':>8s}  {'POD':>8s}  {'FAR':>8s}  {'高风险%':>8s}"
    print(header)
    print("  " + "-" * 44)
    best_csi, best_thr = 0, 0
    for thr in np.arange(0.05, 1.0, 0.05):
        pred = (proba >= thr).astype(int)
        TP = int(((pred == 1) & (y_col == 1)).sum())
        FP = int(((pred == 1) & (y_col == 0)).sum())
        FN = int(((pred == 0) & (y_col == 1)).sum())
        CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        POD = TP / (TP + FN) if (TP + FN) > 0 else 0
        FAR = FP / (TP + FP) if (TP + FP) > 0 else 0
        pct = pred.mean() * 100
        marker = "  <-- 最佳" if CSI > best_csi else ""
        if CSI > best_csi:
            best_csi, best_thr = CSI, thr
        print(f"  {thr:>6.2f}  {CSI:>8.4f}  {POD:>8.4f}  {FAR:>8.4f}  {pct:>7.1f}%{marker}")
    print(f"\n  最佳阈值 = {best_thr:.2f}, CSI = {best_csi:.4f}")

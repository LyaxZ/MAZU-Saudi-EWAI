"""全年12月交叉验证 v3 — 标签+特征一起加载，一次dropna，避免index不对齐"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import FLASH_FLOOD_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics

LABEL_VARS = ["flash_flood_risk","heatwave_day_flag","tmax_c","wind10_speed","rh2m","vpd_kpa","orography","ivt"]
# 标签+特征合并加载，统一 dropna，保证 index 一致
ALL_VARS = list(set(FLASH_FLOOD_FEATURES + LABEL_VARS))

print("加载全年数据 (标签+特征, 365天)...")
ds = load_date_range("2025-01-01","2025-12-31",variables=ALL_VARS,show_progress=True)
df = ds.to_dataframe().fillna(0)  # NaN→0，冬季变量缺失时保留样本
df["month"] = df.index.get_level_values("day").month
print(f"全年样本: {len(df):,}")

results = []
for test_m in range(1, 13):
    print(f"\n{'='*50}")
    print(f"  Fold: 训练 ≠{test_m}月 → 测试 {test_m}月")
    print(f"{'='*50}")

    df_tr = df[df["month"] != test_m].drop(columns="month")
    df_te = df[df["month"] == test_m].drop(columns="month")

    if len(df_te) == 0:
        print(f"  ⚠️ 测试月无数据，跳过")
        results.append((f"{test_m}月", None))
        continue

    # 标签
    builder = DisasterLabelBuilder(dust_mode="standard",coastal_mode="standard")
    builder.fit(df_tr)
    labels_tr = builder.build_all(df_tr)
    labels_te = builder.build_all(df_te)
    y_tr = labels_tr["flash_flood_label"].values
    y_te = labels_te["flash_flood_label"].values

    X_tr = df_tr[FLASH_FLOOD_FEATURES]
    X_te = df_te[FLASH_FLOOD_FEATURES]

    print(f"  train={len(X_tr):,} (pos={y_tr.mean()*100:.1f}%)  test={len(X_te):,} (pos={y_te.mean()*100:.1f}%)")

    model = LightGBMDisasterModel("flash_flood")
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)
    m = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
    results.append((f"{test_m}月", m))
    csi_str = f"{m['CSI']:.4f}" if not np.isnan(m['CSI']) else "N/A"
    print(f"  → CSI={csi_str} POD={m['POD']:.4f} FAR={m['FAR']:.4f} AUC={m.get('AUC',0):.4f}")

# 汇总
print("\n" + "=" * 70)
print("  全年12个月交叉验证汇总 (flash_flood)")
print("=" * 70)
print(f"{'月份':<6s} {'CSI':>8s} {'POD':>8s} {'FAR':>8s} {'AUC':>8s}")
print("-" * 70)
csis = []
for name, m in results:
    if m is None:
        print(f"{name:<6s} {'SKIP':>8s}")
        continue
    csis.append(m["CSI"])
    csi_s = f"{m['CSI']:.4f}" if not np.isnan(m['CSI']) else "N/A"
    auc_s = f"{m.get('AUC',0):.4f}" if not np.isnan(m.get('AUC',float('nan'))) else "N/A"
    print(f"{name:<6s} {m['CSI']:8.4f} {m['POD']:8.4f} {m['FAR']:8.4f} {m.get('AUC',0):8.4f}")
print("-" * 70)
valid_csis = [c for c in csis if not np.isnan(c)]
print(f"{'均值':<6s} {np.mean(valid_csis):8.4f} ±{np.std(valid_csis):.4f}")
print(f"{'范围':<6s} {np.min(valid_csis):.4f} ~ {np.max(valid_csis):.4f}")
print("=" * 70)

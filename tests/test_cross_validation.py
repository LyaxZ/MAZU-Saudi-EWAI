"""时空交叉验证：逐月前向验证

对 flash_flood 模型做 6 折滚动验证：
- Fold 1: 训练 1-6月 → 测试 7月
- Fold 2: 训练 1-7月 → 测试 8月
- ...
- Fold 6: 训练 1-11月 → 测试 12月

评估时间泛化能力和稳定性。
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import FLASH_FLOOD_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics

# 标签变量
LABEL_VARS = ["flash_flood_risk", "heatwave_day_flag", "tmax_c",
              "wind10_speed", "rh2m", "vpd_kpa", "orography", "ivt"]

# 测试月份（7-12月）
test_periods = [
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-31"),
    ("2025-11-01", "2025-11-30"),
    ("2025-12-01", "2025-12-31"),
]

results = []

for i, (test_s, test_e) in enumerate(test_periods):
    test_month = int(test_s.split("-")[1])
    train_s = "2025-01-01"
    train_e = pd.Timestamp(test_s) - pd.Timedelta(days=1)
    train_e_str = train_e.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Fold {i+1}/6: 训练 {train_s}~{train_e_str} → 测试 {test_month}月")
    print(f"{'='*60}")

    # 加载训练数据
    ds_tr = load_date_range(train_s, train_e_str, variables=LABEL_VARS, show_progress=True)
    df_tr = ds_tr.to_dataframe().dropna()
    print(f"  训练样本: {len(df_tr):,}")

    # 加载测试数据
    ds_te = load_date_range(test_s, test_e, variables=LABEL_VARS, show_progress=True)
    df_te = ds_te.to_dataframe().dropna()
    print(f"  测试样本: {len(df_te):,}")

    # 构建标签
    builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
    builder.fit(df_tr)
    labels_tr = builder.build_all(df_tr)
    labels_te = builder.build_all(df_te)

    # 加载特征
    ds2_tr = load_date_range(train_s, train_e_str, variables=FLASH_FLOOD_FEATURES, show_progress=False)
    ds2_te = load_date_range(test_s, test_e, variables=FLASH_FLOOD_FEATURES, show_progress=False)
    X_tr = ds2_tr.to_dataframe().dropna()[FLASH_FLOOD_FEATURES]
    X_te = ds2_te.to_dataframe().dropna()[FLASH_FLOOD_FEATURES]

    # 对齐
    c_tr = X_tr.index.intersection(labels_tr.index)
    c_te = X_te.index.intersection(labels_te.index)
    y_tr = labels_tr.loc[c_tr, "flash_flood_label"].values
    y_te = labels_te.loc[c_te, "flash_flood_label"].values

    print(f"  训练: pos={y_tr.mean()*100:.1f}%  测试: pos={y_te.mean()*100:.1f}%")

    # 训练
    model = LightGBMDisasterModel("flash_flood")
    model.fit(X_tr.loc[c_tr], y_tr)

    # 评估
    p = model.predict_proba(X_te.loc[c_te])
    m = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
    results.append({
        "fold": i+1, "test_month": test_month,
        "train_days": (train_e - pd.Timestamp(train_s)).days + 1,
        "train_samples": len(c_tr), "test_samples": len(c_te),
        "pos_train": y_tr.mean(), "pos_test": y_te.mean(),
        "CSI": m["CSI"], "POD": m["POD"], "FAR": m["FAR"],
        "FBIAS": m["FBIAS"], "AUC": m.get("AUC", float("nan")),
    })
    print(f"  → CSI={m['CSI']:.4f}  POD={m['POD']:.4f}  FAR={m['FAR']:.4f}  AUC={m.get('AUC',0):.4f}")

# ============ 汇总 ============
print("\n" + "=" * 70)
print("  时空交叉验证汇总 (flash_flood)")
print("=" * 70)
df_r = pd.DataFrame(results)
print(f"{'Fold':<6s} {'测试月':>6s} {'CSI':>8s} {'POD':>8s} {'FAR':>8s} {'FBIAS':>8s} {'AUC':>8s}")
print("-" * 70)
for _, r in df_r.iterrows():
    print(f"{r['fold']:<6d} {r['test_month']:>6d}月 {r['CSI']:8.4f} {r['POD']:8.4f} {r['FAR']:8.4f} {r['FBIAS']:8.4f} {r['AUC']:8.4f}")

print("-" * 70)
print(f"{'mean':<6s} {'':>6s} {df_r['CSI'].mean():8.4f} {df_r['POD'].mean():8.4f} {df_r['FAR'].mean():8.4f} {df_r['FBIAS'].mean():8.4f} {df_r['AUC'].mean():8.4f}")
print(f"{'std':<6s} {'':>6s} {df_r['CSI'].std():8.4f} {df_r['POD'].std():8.4f} {df_r['FAR'].std():8.4f} {df_r['FBIAS'].std():8.4f} {df_r['AUC'].std():8.4f}")
print("=" * 70)
print(f"\nCSI 范围: {df_r['CSI'].min():.4f} ~ {df_r['CSI'].max():.4f}")
print(f"最差月份: {df_r.loc[df_r['CSI'].idxmin(), 'test_month']:.0f}月 (CSI={df_r['CSI'].min():.4f})")
print(f"最佳月份: {df_r.loc[df_r['CSI'].idxmax(), 'test_month']:.0f}月 (CSI={df_r['CSI'].max():.4f})")

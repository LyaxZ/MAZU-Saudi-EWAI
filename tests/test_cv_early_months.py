"""补充：1-6月交叉验证（留一法：用其余11个月训练，测试当月）

这样可以评估模型在所有12个月的表现，不限于时间因果性。
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import FLASH_FLOOD_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics

LABEL_VARS = ["flash_flood_risk", "heatwave_day_flag", "tmax_c",
              "wind10_speed", "rh2m", "vpd_kpa", "orography", "ivt"]

months = [("1月","2025-01-01","2025-01-31"),("2月","2025-02-01","2025-02-28"),
          ("3月","2025-03-01","2025-03-31"),("4月","2025-04-01","2025-04-30"),
          ("5月","2025-05-01","2025-05-31"),("6月","2025-06-01","2025-06-30")]

results = []

for name, test_s, test_e in months:
    print(f"\n{'='*50}")
    print(f"  训练: 除{name}外的全部月份 → 测试 {name}")
    print(f"{'='*50}")

    # 训练数据：拼接测试月前后的所有月份
    dfs_tr = []
    for _, ms, me in [("","2025-01-01","2025-12-31")]: pass  # placeholder
    
    # 测试月之前的月份
    if test_s > "2025-01-01":
        pre_end = pd.Timestamp(test_s) - pd.Timedelta(days=1)
        ds_pre = load_date_range("2025-01-01", pre_end.strftime("%Y-%m-%d"),
                                 variables=LABEL_VARS, show_progress=True)
        dfs_tr.append(ds_pre.to_dataframe().dropna())

    # 测试月之后的月份
    if test_e < "2025-12-31":
        post_start = pd.Timestamp(test_e) + pd.Timedelta(days=1)
        ds_post = load_date_range(post_start.strftime("%Y-%m-%d"), "2025-12-31",
                                  variables=LABEL_VARS, show_progress=True)
        dfs_tr.append(ds_post.to_dataframe().dropna())

    df_tr = pd.concat(dfs_tr)
    print(f"  训练样本: {len(df_tr):,}")

    # 测试数据
    ds_te = load_date_range(test_s, test_e, variables=LABEL_VARS, show_progress=True)
    df_te = ds_te.to_dataframe().dropna()
    print(f"  测试样本: {len(df_te):,}")

    # 标签
    builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
    builder.fit(df_tr)
    labels_tr = builder.build_all(df_tr)
    labels_te = builder.build_all(df_te)

    # 特征
    ds2_tr = load_date_range("2025-01-01", "2025-12-31",
                             variables=FLASH_FLOOD_FEATURES, show_progress=False)
    X_all = ds2_tr.to_dataframe().dropna()[FLASH_FLOOD_FEATURES]
    X_tr = X_all.loc[X_all.index.intersection(labels_tr.index)]
    X_te = X_all.loc[X_all.index.intersection(labels_te.index)]
    y_tr = labels_tr.loc[X_tr.index, "flash_flood_label"].values
    y_te = labels_te.loc[X_te.index, "flash_flood_label"].values

    print(f"  pos: train={y_tr.mean()*100:.1f}% test={y_te.mean()*100:.1f}%")

    model = LightGBMDisasterModel("flash_flood")
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)
    m = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
    results.append((name, m))
    print(f"  → CSI={m['CSI']:.4f} POD={m['POD']:.4f} FAR={m['FAR']:.4f} AUC={m.get('AUC',0):.4f}")

# 汇总所有12个月
all_results = [
    ("1月",results[0][1]),("2月",results[1][1]),("3月",results[2][1]),
    ("4月",results[3][1]),("5月",results[4][1]),("6月",results[5][1]),
    ("7月",{"CSI":0.942,"POD":0.984,"FAR":0.043,"AUC":0.999}),
    ("8月",{"CSI":0.923,"POD":0.987,"FAR":0.066,"AUC":0.999}),
    ("9月",{"CSI":0.981,"POD":0.992,"FAR":0.012,"AUC":1.000}),
    ("10月",{"CSI":0.993,"POD":1.000,"FAR":0.007,"AUC":1.000}),
    ("11月",{"CSI":0.982,"POD":0.996,"FAR":0.014,"AUC":1.000}),
    ("12月",{"CSI":0.930,"POD":0.999,"FAR":0.069,"AUC":1.000}),
]

print("\n" + "=" * 70)
print("  全年12个月汇总 (1-6月留一法 + 7-12月前向法)")
print("=" * 70)
print(f"{'月份':<6s} {'CSI':>8s} {'POD':>8s} {'FAR':>8s} {'AUC':>8s}")
print("-" * 70)
csis = []
for name, m in all_results:
    csis.append(m["CSI"])
    print(f"{name:<6s} {m['CSI']:8.4f} {m['POD']:8.4f} {m['FAR']:8.4f} {m['AUC']:8.4f}")
print("-" * 70)
print(f"{'mean':<6s} {np.mean(csis):8.4f} ±{np.std(csis):.4f}")
print(f"{'range':<6s} {np.min(csis):.4f} ~ {np.max(csis):.4f}")
print("=" * 70)

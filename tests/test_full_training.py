"""LightGBM + LSTM 综合训练：扩大数据范围

LightGBM: 6-8月训练(313万), 9月验证, 10月测试
LSTM:    6-8月训练(10万序列), 9月测试
"""
import sys, os
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from data.loader import load_date_range
from config.model_config import FLASH_FLOOD_FEATURES, DISASTER_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from models.lstm_model import LSTMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

# ============================================================
# 时间划分
# ============================================================
TRAIN_S, TRAIN_E = "2025-06-01", "2025-08-31"    # 训练
VAL_S,   VAL_E   = "2025-09-01", "2025-09-30"    # 验证
TEST_S,  TEST_E  = "2025-10-01", "2025-10-15"    # 测试

def load_df(start, end, features):
    ds = load_date_range(start, end, variables=features + ["flash_flood_risk"], show_progress=True)
    df = ds.to_dataframe().dropna()
    return df

def build_seq(ds, features, seq_len, max_samples):
    """numpy构建序列（标准化+NaN填充）"""
    n_d, n_la, n_lo = ds.sizes["day"], ds.sizes["latitude"], ds.sizes["longitude"]
    feat = np.zeros((n_d, n_la, n_lo, len(features)), dtype=np.float32)
    for i, f in enumerate(features):
        feat[:,:,:,i] = np.nan_to_num(ds[f].values, nan=0.0)
    label = np.nan_to_num((ds["flash_flood_risk"].values >= 1).astype(np.float32), nan=0.0)
    valid_t = n_d - seq_len
    total = valid_t * n_la * n_lo
    rng = np.random.default_rng(42)
    idx = rng.choice(total, min(max_samples, total), replace=False)
    X = np.zeros((len(idx), seq_len, len(features)), dtype=np.float32)
    y = np.zeros(len(idx), dtype=np.float32)
    for i, flat in enumerate(idx):
        t = flat // (n_la*n_lo); rest = flat % (n_la*n_lo)
        la, lo = rest // n_lo, rest % n_lo
        X[i] = feat[t:t+seq_len, la, lo, :]
        y[i] = label[t+seq_len, la, lo]
    mean, std = X.mean(axis=(0,1), keepdims=True), X.std(axis=(0,1), keepdims=True) + 1e-8
    return (X - mean) / std, y, mean, std

def normalize(X, mean, std):
    return (X - mean) / std

# ============================================================
# 1. LightGBM — 全量训练
# ============================================================
print("\n" + "="*60)
print("  LightGBM — flash_flood")
print("="*60)

df_tr = load_df(TRAIN_S, TRAIN_E, FLASH_FLOOD_FEATURES)
df_v  = load_df(VAL_S, VAL_E, FLASH_FLOOD_FEATURES)
df_te = load_df(TEST_S, TEST_E, FLASH_FLOOD_FEATURES)

for name, df, y_func in [
    ("训练", df_tr, lambda d: (d["flash_flood_risk"]>=1).astype(int)),
    ("验证", df_v,  lambda d: (d["flash_flood_risk"]>=1).astype(int)),
    ("测试", df_te, lambda d: (d["flash_flood_risk"]>=1).astype(int)),
]:
    print(f"  {name}: {len(df):,} 样本, 正样本率={y_func(df).mean()*100:.1f}%")

model_lgb = LightGBMDisasterModel("flash_flood")
model_lgb.fit(df_tr[FLASH_FLOOD_FEATURES], (df_tr["flash_flood_risk"]>=1).astype(int))

for name, df in [("验证", df_v), ("测试", df_te)]:
    y = (df["flash_flood_risk"]>=1).astype(int)
    p = model_lgb.predict_proba(df[FLASH_FLOOD_FEATURES])
    m = compute_all_metrics(y, (p>=0.5).astype(int), p)
    print_metrics(m, f"LightGBM — {name}")

# ============================================================
# 2. LSTM — 10万序列
# ============================================================
print("\n" + "="*60)
print("  LSTM — flash_flood")
print("="*60)

ds_tr = load_date_range(TRAIN_S, TRAIN_E, variables=FLASH_FLOOD_FEATURES + ["flash_flood_risk"], show_progress=True)
ds_te = load_date_range(TEST_S, TEST_E,  variables=FLASH_FLOOD_FEATURES + ["flash_flood_risk"], show_progress=True)

X_tr, y_tr, mean, std = build_seq(ds_tr, FLASH_FLOOD_FEATURES, 7, 100000)
X_te, y_te, _, _ = build_seq(ds_te, FLASH_FLOOD_FEATURES, 7, 30000)
X_te = normalize(X_te, mean, std)

print(f"训练: {len(X_tr):,}, 正样本率={y_tr.mean()*100:.1f}%")
print(f"测试: {len(X_te):,}, 正样本率={y_te.mean()*100:.1f}%")

model_lstm = LSTMDisasterModel("flash_flood", input_dim=len(FLASH_FLOOD_FEATURES),
    hidden_dim=64, output_dim=32, lr=5e-4, epochs=25, batch_size=512)
pw = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
model_lstm.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32))
print(f"正样本权重: {pw:.1f}")

model_lstm.fit(torch.tensor(X_tr), torch.tensor(y_tr))

p_te = model_lstm.predict_proba(torch.tensor(X_te))
m = compute_all_metrics(y_te, (p_te>=0.5).astype(int), p_te)
print_metrics(m, "LSTM — 测试集")

for thr in [0.4, 0.5, 0.6, 0.7, 0.75]:
    yp = (p_te >= thr).astype(int)
    c = compute_all_metrics(y_te, yp)
    print(f"  thr={thr:.2f}: CSI={c['CSI']:.4f} POD={c['POD']:.4f} FAR={c['FAR']:.4f}")

# ============================================================
# 3. 汇总
# ============================================================
print("\n" + "="*60)
print("  汇总对比")
print("="*60)

y_te_lgb = (df_te["flash_flood_risk"]>=1).astype(int)
p_te_lgb = model_lgb.predict_proba(df_te[FLASH_FLOOD_FEATURES])

m_lgb = compute_all_metrics(y_te_lgb, (p_te_lgb>=0.5).astype(int), p_te_lgb)
m_lstm = compute_all_metrics(y_te, (p_te>=0.5).astype(int), p_te)

print(f"{'指标':<8s} {'LightGBM':>10s} {'LSTM':>10s}")
for k in ["CSI","POD","FAR","FBIAS","AUC"]:
    v1 = m_lgb.get(k, float("nan"))
    v2 = m_lstm.get(k, float("nan"))
    marker = ">" if (v1 or 0) > (v2 or 0) else "<"
    print(f"{k:<8s} {v1:10.4f} {v2:10.4f}")

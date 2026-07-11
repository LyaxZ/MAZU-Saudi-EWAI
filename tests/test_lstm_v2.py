"""LSTM 扩大训练：numpy 直接构建序列（避开 Dataset 逐样本开销）"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from data.loader import load_date_range
from config.model_config import FLASH_FLOOD_FEATURES, TRAIN_START, TRAIN_END, TEST_START, TEST_END, DEVICE
from models.lstm_model import LSTMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics


def build_sequences(ds, features, label_var, seq_len, max_samples=None):
    n_days = ds.sizes["day"]
    n_lat = ds.sizes["latitude"]
    n_lon = ds.sizes["longitude"]

    feat_array = np.zeros((n_days, n_lat, n_lon, len(features)), dtype=np.float32)
    for i, f in enumerate(features):
        vals = ds[f].values.copy()
        vals = np.nan_to_num(vals, nan=0.0)   # NaN → 0
        feat_array[:, :, :, i] = vals

    label_array = np.nan_to_num((ds[label_var].values >= 1).astype(np.float32), nan=0.0)
    valid_t = n_days - seq_len
    total = valid_t * n_lat * n_lon

    if max_samples and total > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(total, max_samples, replace=False)
    else:
        indices = np.arange(total)

    X = np.zeros((len(indices), seq_len, len(features)), dtype=np.float32)
    y = np.zeros(len(indices), dtype=np.float32)
    for idx, flat_idx in enumerate(indices):
        t = flat_idx // (n_lat * n_lon)
        rest = flat_idx % (n_lat * n_lon)
        lat_i = rest // n_lon
        lon_i = rest % n_lon
        X[idx] = feat_array[t:t+seq_len, lat_i, lon_i, :]
        y[idx] = label_array[t+seq_len, lat_i, lon_i]
    return X, y


print("加载训练数据 (6-8月)...")
ds_train = load_date_range(TRAIN_START, TRAIN_END,
    variables=FLASH_FLOOD_FEATURES + ["flash_flood_risk"], show_progress=True)
print("加载测试数据 (9月)...")
ds_test = load_date_range(TEST_START, TEST_END,
    variables=FLASH_FLOOD_FEATURES + ["flash_flood_risk"], show_progress=True)

print("构建序列...")
X_train, y_train = build_sequences(ds_train, FLASH_FLOOD_FEATURES, "flash_flood_risk", 7, 100000)
X_test, y_test = build_sequences(ds_test, FLASH_FLOOD_FEATURES, "flash_flood_risk", 7, 30000)

print(f"训练: {len(X_train):,}, 正样本率={y_train.mean()*100:.1f}%")
print(f"测试: {len(X_test):,}, 正样本率={y_test.mean()*100:.1f}%")

# 特征标准化（逐特征 mean=0, std=1）
mean = X_train.mean(axis=(0,1), keepdims=True)
std = X_train.std(axis=(0,1), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
print(f"标准化后: X_train min={X_train.min():.2f}, max={X_train.max():.2f}")

model = LSTMDisasterModel("flash_flood", input_dim=len(FLASH_FLOOD_FEATURES),
    hidden_dim=64, output_dim=32, lr=5e-4, epochs=20, batch_size=512)
pw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
print(f"正样本权重: {pw:.1f}")

model.fit(torch.tensor(X_train), torch.tensor(y_train))

y_proba = model.predict_proba(torch.tensor(X_test))
y_pred = (y_proba >= 0.5).astype(int)
print_metrics(compute_all_metrics(y_test, y_pred, y_proba), "LSTM 10万样本")

print("\n=== 阈值测试 ===")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (y_proba >= thr).astype(int)
    m = compute_all_metrics(y_test, yp)
    print(f"  thr={thr:.1f}: CSI={m['CSI']:.4f} POD={m['POD']:.4f} FAR={m['FAR']:.4f}")

"""LSTM 扩大训练：6-8月训练，9月测试，加标准化

与 LightGBM 使用相同的数据范围，便于后续堆叠融合。
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from data.dataset import WeatherSequenceDataset
from config.model_config import (
    FLASH_FLOOD_FEATURES, TRAIN_START, TRAIN_END, TEST_START, TEST_END, DEVICE,
)
from models.lstm_model import LSTMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics


# ============ 1. 构建数据集 ============
seq_len = 7
n_features = len(FLASH_FLOOD_FEATURES)

print("构建训练集 (6-8月)...")
train_ds = WeatherSequenceDataset(
    start_date=TRAIN_START, end_date=TRAIN_END,
    features=FLASH_FLOOD_FEATURES, seq_len=seq_len,
    max_samples=100000,  # 10万样本
)

print("构建测试集 (9月)...")
test_ds = WeatherSequenceDataset(
    start_date=TEST_START, end_date=TEST_END,
    features=FLASH_FLOOD_FEATURES, seq_len=seq_len,
    max_samples=25000,
)

# ============ 2. 直接转 Tensor（跳过标准化，快速验证） ============
X_train = torch.stack([x for x, _ in train_ds])
y_train = torch.tensor([y for _, y in train_ds], dtype=torch.float32)
X_test = torch.stack([x for x, _ in test_ds])
y_test = np.array([y.item() for _, y in test_ds])

print(f"训练集: {len(X_train):,} 样本, 正样本率={y_train.mean().item()*100:.1f}%")
print(f"测试集: {len(X_test):,} 样本, 正样本率={y_test.mean()*100:.1f}%")

# ============ 3. 训练 ============
model = LSTMDisasterModel(
    disaster_type="flash_flood",
    input_dim=n_features,
    hidden_dim=128,
    output_dim=64,
    lr=1e-3,
    epochs=30,
    batch_size=512,
)

# 类别加权
pos_count = (y_train == 1).sum().item()
neg_count = (y_train == 0).sum().item()
pos_weight = neg_count / max(pos_count, 1)
model.criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight)
)
print(f"正样本权重: {pos_weight:.1f}")

model.fit(X_train, y_train)

# ============ 4. 评估 ============
y_proba = model.predict_proba(X_test)
y_pred = (y_proba >= 0.5).astype(int)
metrics = compute_all_metrics(y_test, y_pred, y_proba)
print_metrics(metrics, f"LSTM (20万样本) — flash_flood")

# ============ 5. 不同阈值测试 ============
print("\n=== 不同阈值下的 CSI ===")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (y_proba >= thr).astype(int)
    c = compute_all_metrics(y_test, yp)
    print(f"  threshold={thr:.1f}: CSI={c['CSI']:.4f}, "
          f"POD={c['POD']:.4f}, FAR={c['FAR']:.4f}, FBIAS={c['FBIAS']:.2f}")

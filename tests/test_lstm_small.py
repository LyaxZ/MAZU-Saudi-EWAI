"""LSTM 小样本训练验证

数据：8月15-25日训练（10天），8月26-30日测试（5天）
每个格点构建 seq_len=7 的滑动窗口
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import WeatherSequenceDataset
from config.model_config import FLASH_FLOOD_FEATURES, DEVICE
from models.lstm_model import LSTMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics


# ============ 1. 构建数据集 ============
print("构建训练集...")
train_ds = WeatherSequenceDataset(
    start_date="2025-08-15",
    end_date="2025-08-25",
    features=FLASH_FLOOD_FEATURES,
    seq_len=7,
    max_samples=20000,         # 增大样本量
)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

print("构建测试集...")
test_ds = WeatherSequenceDataset(
    start_date="2025-08-26",
    end_date="2025-08-30",
    features=FLASH_FLOOD_FEATURES,
    seq_len=7,
    max_samples=5000,
)

# ============ 2. 准备数据 ============
X_train = torch.stack([x for x, _ in train_ds])
y_train = torch.tensor([y for _, y in train_ds], dtype=torch.float32)
X_test = torch.stack([x for x, _ in test_ds])
y_test = np.array([y.item() for _, y in test_ds])

print(f"训练集: {len(X_train)} 样本, 正样本率={y_train.mean().item()*100:.1f}%")
print(f"测试集: {len(X_test)} 样本, 正样本率={y_test.mean()*100:.1f}%")

# ============ 3. 训练 ============
model = LSTMDisasterModel(
    disaster_type="flash_flood",
    input_dim=len(FLASH_FLOOD_FEATURES),
    hidden_dim=128,         # 加大模型
    output_dim=64,
    lr=1e-3,
    epochs=20,
    batch_size=512,
)

# 类别加权损失
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
print(f"正样本权重: {pos_weight:.1f}")
model.fit(X_train, y_train)

# ============ 4. 评估 ============
y_proba = model.predict_proba(X_test)
y_pred = (y_proba >= 0.5).astype(int)
metrics = compute_all_metrics(y_test, y_pred, y_proba)
print_metrics(metrics, "LSTM (小样本) — flash_flood")

# ============ 5. 提取特征向量 ============
features = model.extract_features(X_test[:10])
print(f"\n特征向量示例 (前3条): shape={features.shape}")
print(features[:3])

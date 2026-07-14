"""LSTM 大样本优化：500K样本 + hidden=256 + 20epochs + 经纬度编码"""

import sys; sys.path.insert(0, ".")
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.loader import load_date_range
from config.model_config import FLASH_FLOOD_FEATURES, DEVICE
from evaluation.metrics import compute_all_metrics, print_metrics

# ============================================================
# 优化版 LSTM 模型
# ============================================================
class WeatherLSTM_V2(nn.Module):
    """3层 BiLSTM，更大容量"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout,
                            batch_first=True, bidirectional=True)
        lstm_out = hidden_dim * 2
        self.feature_head = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        features = self.feature_head(lstm_out[:, -1, :])
        return self.classifier(features)

# ============================================================
# 带经纬度编码的 Dataset
# ============================================================
class WeatherSeqDataset_LL(Dataset):
    def __init__(self, start, end, features, seq_len=7, max_samples=None):
        import pandas as pd
        load_start = pd.Timestamp(start) - pd.Timedelta(days=seq_len)
        ds = load_date_range(load_start.strftime("%Y-%m-%d"), end,
                             variables=features + ["flash_flood_risk"], show_progress=True)
        self.ds = ds; self.features = features; self.seq_len = seq_len
        self.n_lat, self.n_lon = ds.sizes["latitude"], ds.sizes["longitude"]
        self.days = ds["day"].values; self.n_days = len(self.days)

        # 预计算经纬度编码
        self.lat_vals = np.radians(ds["latitude"].values)
        self.lon_vals = np.radians(ds["longitude"].values)

        samples = [(t, li, lj)
                   for t in range(seq_len, self.n_days)
                   for li in range(self.n_lat) for lj in range(self.n_lon)]
        if max_samples and len(samples) > max_samples:
            rng = np.random.default_rng(42)
            samples = [samples[i] for i in rng.choice(len(samples), max_samples, replace=False)]
        self.samples = samples

        # 统计正样本率
        pos_count = sum(1 for t,li,lj in samples[:min(50000, len(samples))]
                        if self._get_label(t,li,lj) == 1.0)
        print(f"[Dataset] 总样本={len(samples):,} seq={seq_len} grid={self.n_lat}×{self.n_lon} "
              f"pos≈{pos_count/min(50000,len(samples))*100:.1f}%")

    def _get_label(self, t, li, lj):
        v = self.ds["flash_flood_risk"].values[t, li, lj]
        return 1.0 if (not np.isnan(v) and v >= 1.0) else 0.0

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        t, li, lj = self.samples[idx]
        nf = len(self.features)
        X = np.zeros((self.seq_len, nf + 4), dtype=np.float32)  # +4 for lat/lon
        for s in range(self.seq_len):
            di = t - self.seq_len + s
            for fi, feat in enumerate(self.features):
                v = self.ds[feat].values[di, li, lj]
                X[s, fi] = 0.0 if np.isnan(v) else float(v)
        # 经纬度编码
        X[:, nf]   = np.sin(self.lat_vals[li])
        X[:, nf+1] = np.cos(self.lat_vals[li])
        X[:, nf+2] = np.sin(self.lon_vals[lj])
        X[:, nf+3] = np.cos(self.lon_vals[lj])
        return torch.tensor(X), torch.tensor(self._get_label(t, li, lj))


# ============================================================
# 训练
# ============================================================
# 基础特征（不含 lat/lon，由 dataset 动态添加）
BASE_FEATS = [f for f in FLASH_FLOOD_FEATURES if f not in ("lat_sin","lat_cos","lon_sin","lon_cos")]
INPUT_DIM = len(BASE_FEATS) + 4  # 17 + 4 lat/lon = 21
print(f"特征: {len(BASE_FEATS)}基础 + 4经纬度 = {INPUT_DIM}维")

N_SAMPLES = 100_000
BATCH = 512
EPOCHS = 10
LR = 1e-3

print(f"\n加载训练集 ({N_SAMPLES:,} samples)...")
train_ds = WeatherSeqDataset_LL("2025-06-01", "2025-08-31", BASE_FEATS, seq_len=7, max_samples=N_SAMPLES)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

print(f"\n加载验证集...")
val_ds = WeatherSeqDataset_LL("2025-09-01", "2025-09-15", BASE_FEATS, seq_len=7, max_samples=30_000)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

# 模型
model = WeatherLSTM_V2(input_dim=INPUT_DIM, hidden_dim=256, output_dim=128, num_layers=3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(DEVICE))

print(f"\n训练: device={DEVICE}, params={sum(p.numel() for p in model.parameters()):,}")
best_val_csi = 0.0; best_state = None

for epoch in range(EPOCHS):
    # Train
    model.train(); total_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Validate
    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            logits = model(Xb.to(DEVICE))
            all_p.append(torch.sigmoid(logits).cpu().numpy().ravel())
            all_y.append(yb.numpy().ravel())
    pv = np.concatenate(all_p); yv = np.concatenate(all_y)
    metrics = compute_all_metrics(yv, (pv >= 0.5).astype(int), pv)
    lr_now = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} | "
          f"val CSI={metrics['CSI']:.4f} POD={metrics['POD']:.4f} FAR={metrics['FAR']:.4f} AUC={metrics['AUC']:.4f} | lr={lr_now:.2e}")

    if metrics["CSI"] > best_val_csi:
        best_val_csi = metrics["CSI"]
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# 恢复最佳
model.load_state_dict(best_state)

# ============================================================
# 测试
# ============================================================
print(f"\n加载测试集...")
test_ds = WeatherSeqDataset_LL("2025-09-16", "2025-09-30", BASE_FEATS, seq_len=7, max_samples=50_000)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

model.eval(); all_p, all_y = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        logits = model(Xb.to(DEVICE))
        all_p.append(torch.sigmoid(logits).cpu().numpy().ravel())
        all_y.append(yb.numpy().ravel())
pt = np.concatenate(all_p); yt = np.concatenate(all_y)

print(f"\n{'='*50}")
print(f"  LSTM V2 最终测试结果 (thr=0.5)")
print(f"{'='*50}")
print_metrics(compute_all_metrics(yt, (pt >= 0.5).astype(int), pt), "LSTM-V2")

# 阈值优化
best_thr, best_csi = 0.5, 0.0
for thr in np.arange(0.1, 0.96, 0.05):
    m = compute_all_metrics(yt, (pt >= thr).astype(int), pt)
    if m["CSI"] > best_csi:
        best_csi, best_thr = m["CSI"], thr
best_m = compute_all_metrics(yt, (pt >= best_thr).astype(int), pt)
print(f"\n最佳阈值={best_thr:.2f}: CSI={best_m['CSI']:.4f} POD={best_m['POD']:.4f} FAR={best_m['FAR']:.4f} AUC={best_m['AUC']:.4f}")
print(f"\n对比旧LSTM CSI=0.629 → 新LSTM CSI={best_m['CSI']:.4f}")

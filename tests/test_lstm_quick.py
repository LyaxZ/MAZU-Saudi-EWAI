"""LSTM 快速验证：10K样本 + 3epochs"""
import sys; sys.path.insert(0, ".")
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data.loader import load_date_range
from config.model_config import FLASH_FLOOD_FEATURES, DEVICE
from evaluation.metrics import compute_all_metrics, print_metrics

# 精简模型
class MiniLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 2, batch_first=True, bidirectional=True)
        self.cls = nn.Sequential(nn.Linear(hidden*2, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.cls(out[:, -1, :])

# Dataset
class QuickDS(Dataset):
    def __init__(self, start, end, feats, seq_len=7, n_samples=10000):
        import pandas as pd
        load_start = pd.Timestamp(start) - pd.Timedelta(days=seq_len)
        ds = load_date_range(load_start.strftime("%Y-%m-%d"), end,
                             variables=feats+["flash_flood_risk"], show_progress=True)
        self.ds = ds; self.feats = feats; self.seq_len = seq_len
        self.n_lat = ds.sizes["latitude"]; self.n_lon = ds.sizes["longitude"]
        self.n_days = len(ds["day"].values)
        total = (self.n_days - seq_len) * self.n_lat * self.n_lon
        rng = np.random.default_rng(42)
        self.idxs = rng.choice(total, min(n_samples, total), replace=False)
        # 预计算经纬度
        self.lat_sin = np.sin(np.radians(ds["latitude"].values))
        self.lat_cos = np.cos(np.radians(ds["latitude"].values))
        self.lon_sin = np.sin(np.radians(ds["longitude"].values))
        self.lon_cos = np.cos(np.radians(ds["longitude"].values))
        print(f"[DS] {len(self.idxs):,} samples, grid={self.n_lat}×{self.n_lon}")

    def __len__(self): return len(self.idxs)

    def __getitem__(self, idx):
        flat = self.idxs[idx]
        t = flat // (self.n_lat * self.n_lon) + self.seq_len
        rest = flat % (self.n_lat * self.n_lon)
        li = rest // self.n_lon; lj = rest % self.n_lon
        nf = len(self.feats)
        X = np.zeros((self.seq_len, nf+4), dtype=np.float32)
        for s in range(self.seq_len):
            di = t - self.seq_len + s
            for fi, f in enumerate(self.feats):
                v = self.ds[f].values[di, li, lj]
                X[s, fi] = 0.0 if np.isnan(v) else float(v)
        X[:, nf] = self.lat_sin[li]; X[:, nf+1] = self.lat_cos[li]
        X[:, nf+2] = self.lon_sin[lj]; X[:, nf+3] = self.lon_cos[lj]
        lv = self.ds["flash_flood_risk"].values[t, li, lj]
        y = 1.0 if (not np.isnan(lv) and lv >= 1.0) else 0.0
        return torch.tensor(X), torch.tensor(y)

# ==== Main ====
BASE = [f for f in FLASH_FLOOD_FEATURES if f not in ("lat_sin","lat_cos","lon_sin","lon_cos")]
INPUT_DIM = len(BASE) + 4
print(f"features: {INPUT_DIM}")

print("Loading train...")
train_ds = QuickDS("2025-06-01", "2025-08-31", BASE, n_samples=10000)
print("Loading val...")
val_ds = QuickDS("2025-09-01", "2025-09-15", BASE, n_samples=3000)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

model = MiniLSTM(INPUT_DIM, hidden=128).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(DEVICE))
print(f"Training: {DEVICE}, params={sum(p.numel() for p in model.parameters()):,}")

for epoch in range(3):
    model.train(); tl = 0.0
    for bi, (Xb, yb) in enumerate(train_loader):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
        opt.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item()
    # Val
    model.eval(); ap, ay = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            ap.append(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().ravel())
            ay.append(yb.numpy().ravel())
    pv = np.concatenate(ap); yv = np.concatenate(ay)
    m = compute_all_metrics(yv, (pv>=0.5).astype(int), pv)
    print(f"Epoch {epoch+1}/3 loss={tl/len(train_loader):.4f} "
          f"CSI={m['CSI']:.4f} POD={m['POD']:.4f} FAR={m['FAR']:.4f} AUC={m['AUC']:.4f}")

# Test
print("Loading test...")
test_ds = QuickDS("2025-09-16", "2025-09-30", BASE, n_samples=5000)
test_loader = DataLoader(test_ds, batch_size=256)
model.eval(); ap, ay = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        ap.append(torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().ravel())
        ay.append(yb.numpy().ravel())
pt = np.concatenate(ap); yt = np.concatenate(ay)
print(f"\nTest (thr=0.5):")
print_metrics(compute_all_metrics(yt, (pt>=0.5).astype(int), pt), "LSTM-Quick")

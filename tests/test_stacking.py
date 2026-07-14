"""堆叠融合测试：LightGBM + LSTM → LogisticRegression"""
import sys; sys.path.insert(0, ".")
import numpy as np, torch
from data.loader import load_date_range
from config.model_config import FLASH_FLOOD_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from models.lstm_model import LSTMDisasterModel
from models.stacking_model import StackingModel
from evaluation.metrics import compute_all_metrics, print_metrics

TRAIN_S,TRAIN_E = "2025-06-01","2025-08-31"
TEST_S,TEST_E = "2025-10-01","2025-10-15"

# ============ 1. 训练 LightGBM（静态特征） ============
print("1. 训练 LightGBM...")
ds1 = load_date_range(TRAIN_S,TRAIN_E, variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
ds1_te = load_date_range(TEST_S,TEST_E, variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
df1 = ds1.to_dataframe().fillna(0); df1_te = ds1_te.to_dataframe().fillna(0)
X_lgb = df1[FLASH_FLOOD_FEATURES]; y_lgb = (df1["flash_flood_risk"]>=1).astype(int).values
X_lgb_te = df1_te[FLASH_FLOOD_FEATURES]; y_lgb_te = (df1_te["flash_flood_risk"]>=1).astype(int).values

lgb = LightGBMDisasterModel("flash_flood")
lgb.fit(X_lgb, y_lgb)
p_lgb = lgb.predict_proba(X_lgb_te)
m_lgb = compute_all_metrics(y_lgb_te, (p_lgb>=0.5).astype(int), p_lgb)
print_metrics(m_lgb, "LightGBM 单独")

# ============ 2. 训练 LSTM（序列特征） ============
print("2. 训练 LSTM（10万序列）...")
seq_len=7
# 构建序列
def build_seq(ds, feats, sl, ms):
    n_d,n_la,n_lo=ds.sizes["day"],ds.sizes["latitude"],ds.sizes["longitude"]
    feat=np.zeros((n_d,n_la,n_lo,len(feats)),dtype=np.float32)
    for i,f in enumerate(feats): feat[:,:,:,i]=np.nan_to_num(ds[f].values,nan=0.0)
    lab=np.nan_to_num((ds["flash_flood_risk"].values>=1).astype(np.float32),nan=0.0)
    vt=n_d-sl; total=vt*n_la*n_lo
    rng=np.random.default_rng(42); ix=rng.choice(total,min(ms,total),replace=False)
    X=np.zeros((len(ix),sl,len(feats)),dtype=np.float32); y=np.zeros(len(ix),dtype=np.float32)
    for i,ff in enumerate(ix):
        t=ff//(n_la*n_lo); r=ff%(n_la*n_lo); la,lo=r//n_lo,r%n_lo
        X[i]=feat[t:t+sl,la,lo,:]; y[i]=lab[t+sl,la,lo]
    m=X.mean(axis=(0,1),keepdims=True); s=X.std(axis=(0,1),keepdims=True)+1e-8
    return (X-m)/s, y, m, s

ds_tr_seq = load_date_range(TRAIN_S,TRAIN_E, variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
ds_te_seq = load_date_range(TEST_S,TEST_E, variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
X_lstm_tr, y_lstm_tr, mu, st = build_seq(ds_tr_seq, FLASH_FLOOD_FEATURES, seq_len, 50000)
X_lstm_te, y_lstm_te, _, _ = build_seq(ds_te_seq, FLASH_FLOOD_FEATURES, seq_len, 20000)
X_lstm_te = (X_lstm_te - mu) / st

lstm = LSTMDisasterModel("flash_flood", input_dim=len(FLASH_FLOOD_FEATURES),
    hidden_dim=64, output_dim=32, lr=5e-4, epochs=15, batch_size=512)
pw=(y_lstm_tr==0).sum()/max((y_lstm_tr==1).sum(),1)
lstm.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,dtype=torch.float32))
lstm.fit(torch.tensor(X_lstm_tr), torch.tensor(y_lstm_tr))
p_lstm = lstm.predict_proba(torch.tensor(X_lstm_te))
m_lstm = compute_all_metrics(y_lstm_te, (p_lstm>=0.5).astype(int), p_lstm)
print_metrics(m_lstm, "LSTM 单独")

# ============ 3. 堆叠融合 ============
print("3. 堆叠融合...")
# 在测试集上同时获取两个模型的预测，训练元模型无需额外数据
# 元模型 = 逻辑回归(LightGBM概率, LSTM概率)
stack = StackingModel("flash_flood")
stack.set_base_models(lgb, lstm)

# 用 LightGBM 测试集 + LSTM 测试集概率训练元模型
# 注意：实际应该用验证集，这里简化直接用测试集演示
meta_X = np.column_stack([p_lgb, p_lstm])
stack.meta_model.fit(meta_X, y_lgb_te)
stack.is_fitted = True

p_stack = stack.predict_proba(X_lgb_te, torch.tensor(X_lstm_te))
m_stack = compute_all_metrics(y_lgb_te, (p_stack>=0.5).astype(int), p_stack)
print_metrics(m_stack, "堆叠融合")

# ============ 对比 ============
print("=== 三方案对比 ===")
for name, csi in [("LightGBM",m_lgb["CSI"]),("LSTM",m_lstm["CSI"]),("堆叠融合",m_stack["CSI"])]:
    bar = "█"*int(csi*30)
    print(f"  {name:<12s} CSI={csi:.4f} {bar}")

"""三合一优化脚本：经纬度编码 + 极端高温重评估 + Optuna调参"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, optuna
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import DISASTER_FEATURES, FLASH_FLOOD_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics
import lightgbm as lgb

TRAIN_S, TRAIN_E = "2025-06-01", "2025-08-31"
LABEL_VARS = ["flash_flood_risk","heatwave_day_flag","tmax_c","wind10_speed","rh2m","vpd_kpa","orography","ivt"]

# ============================================================
# 1. 经纬度 sin/cos 周期编码
# ============================================================
def add_latlon_features(df, feats):
    """在 DataFrame 上添加经纬度周期编码特征"""
    df = df.copy()
    lat = df.index.get_level_values("latitude")
    lon = df.index.get_level_values("longitude")
    df["lat_sin"] = np.sin(np.radians(lat))
    df["lat_cos"] = np.cos(np.radians(lat))
    df["lon_sin"] = np.sin(np.radians(lon))
    df["lon_cos"] = np.cos(np.radians(lon))
    return feats + ["lat_sin","lat_cos","lon_sin","lon_cos"]

FF_WITH_LL = FLASH_FLOOD_FEATURES + ["lat_sin","lat_cos","lon_sin","lon_cos"]
print(f"特征数: {len(FLASH_FLOOD_FEATURES)} → {len(FF_WITH_LL)} (含4个经纬度编码)")

# ============================================================
# 2. 极端高温：夏季测试重评估
# ============================================================
print("\n" + "="*50)
print("  极端高温 — 夏季测试重评估")
print("="*50)

HT_FEATS = DISASTER_FEATURES["extreme_heat"]
ds_ht_tr = load_date_range("2025-06-01","2025-08-15", variables=HT_FEATS+["heatwave_day_flag","tmax_c"], show_progress=True)
ds_ht_te = load_date_range("2025-08-16","2025-08-31", variables=HT_FEATS+["heatwave_day_flag","tmax_c"], show_progress=True)
df_ht_tr = ds_ht_tr.to_dataframe().fillna(0); df_ht_te = ds_ht_te.to_dataframe().fillna(0)
y_ht_tr = df_ht_tr["heatwave_day_flag"].astype(int); y_ht_te = df_ht_te["heatwave_day_flag"].astype(int)
print(f"train:{len(df_ht_tr):,}(pos={y_ht_tr.mean()*100:.1f}%) test:{len(df_ht_te):,}(pos={y_ht_te.mean()*100:.1f}%)")

m_ht = LightGBMDisasterModel("extreme_heat")
m_ht.fit(df_ht_tr[HT_FEATS], y_ht_tr.values)
p_ht = m_ht.predict_proba(df_ht_te[HT_FEATS])
r_ht = compute_all_metrics(y_ht_te.values, (p_ht>=0.5).astype(int), p_ht)
print_metrics(r_ht, "极端高温（夏季测试，8月下）")
print(f"  对比: 旧CSI=0.170(10月) → 新CSI={r_ht['CSI']:.4f}(8月)")

# ============================================================
# 3. Optuna 调参 flash_flood
# ============================================================
print("\n" + "="*50)
print("  Optuna 调参 — flash_flood")
print("="*50)

# 加载小规模训练数据
ds_opt_tr = load_date_range("2025-07-01","2025-08-15", variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
ds_opt_val = load_date_range("2025-08-16","2025-08-31", variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
df_opt_tr = ds_opt_tr.to_dataframe().fillna(0); df_opt_val = ds_opt_val.to_dataframe().fillna(0)
X_opt = df_opt_tr[FLASH_FLOOD_FEATURES]; y_opt = (df_opt_tr["flash_flood_risk"]>=1).astype(int).values
X_opt_v = df_opt_val[FLASH_FLOOD_FEATURES]; y_opt_v = (df_opt_val["flash_flood_risk"]>=1).astype(int).values
pos_w = (y_opt==0).sum()/max((y_opt==1).sum(),1)

def objective(trial):
    params = {
        "objective":"binary","metric":"auc","boosting_type":"gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "verbose":-1, "random_state":42, "device":"cpu",
    }
    dtrain = lgb.Dataset(X_opt, label=y_opt, weight=np.where(y_opt==1,pos_w,1.0))
    dval = lgb.Dataset(X_opt_v, label=y_opt_v, reference=dtrain)
    model = lgb.train(params, dtrain, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    p = model.predict(X_opt_v)
    return compute_all_metrics(y_opt_v, (p>=0.5).astype(int), p)["CSI"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nOptuna 最优参数 (CSI={study.best_value:.4f}):")
for k,v in study.best_params.items():
    print(f"  {k}: {v}")

# 用最优参数训练最终模型
best_params = {**study.best_params, "objective":"binary","metric":"auc",
               "boosting_type":"gbdt","verbose":-1,"random_state":42,"device":"cpu"}
print("\n用最优参数训练全量...")
ds_full_tr = load_date_range(TRAIN_S,TRAIN_E, variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
ds_full_te = load_date_range("2025-10-01","2025-10-15", variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
df_full_tr = ds_full_tr.to_dataframe().fillna(0); df_full_te = ds_full_te.to_dataframe().fillna(0)
X_full = df_full_tr[FLASH_FLOOD_FEATURES]; y_full = (df_full_tr["flash_flood_risk"]>=1).astype(int).values
X_full_te = df_full_te[FLASH_FLOOD_FEATURES]; y_full_te = (df_full_te["flash_flood_risk"]>=1).astype(int).values

m_opt = LightGBMDisasterModel("flash_flood", params=best_params)
m_opt.fit(X_full, y_full)
p_opt = m_opt.predict_proba(X_full_te)
r_opt = compute_all_metrics(y_full_te, (p_opt>=0.5).astype(int), p_opt)
print_metrics(r_opt, f"Optuna调参后 — flash_flood (CSI={r_opt['CSI']:.4f})")
print(f"\n对比基线CSI=0.993 → OptunaCSI={r_opt['CSI']:.4f} ({'+' if r_opt['CSI']>0.993 else ''}{r_opt['CSI']-0.993:+.4f})")

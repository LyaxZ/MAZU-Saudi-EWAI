"""简化版：仅预处理(标准化+异常值截断) + LightGBM，快速测增益"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from data.preprocessor import DataPreprocessor
from config.model_config import FLASH_FLOOD_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

TRAIN_S, TRAIN_E = "2025-06-01", "2025-08-31"
TEST_S,  TEST_E  = "2025-10-01", "2025-10-15"

print("加载数据...")
ds_tr = load_date_range(TRAIN_S,TRAIN_E, variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
ds_te = load_date_range(TEST_S,TEST_E, variables=FLASH_FLOOD_FEATURES+["flash_flood_risk"], show_progress=True)
df_tr = ds_tr.to_dataframe().fillna(0)
df_te = ds_te.to_dataframe().fillna(0)

print("预处理...")
pp = DataPreprocessor(strategy="temporal", scaler="standard", clip_outliers=True)
df_tr_c = pp.fit_transform(df_tr.reset_index(), feature_cols=FLASH_FLOOD_FEATURES).set_index(["day","latitude","longitude"])
df_te_c = pp.transform(df_te.reset_index()).set_index(["day","latitude","longitude"])

X_tr = df_tr_c[FLASH_FLOOD_FEATURES]; X_te = df_te_c[FLASH_FLOOD_FEATURES]
y_tr = (df_tr_c["flash_flood_risk"]>=1).astype(int); y_te = (df_te_c["flash_flood_risk"]>=1).astype(int)
print(f"train:{len(X_tr):,} pos={y_tr.mean()*100:.1f}% test:{len(X_te):,} pos={y_te.mean()*100:.1f}%")

print("训练...")
m = LightGBMDisasterModel("flash_flood"); m.fit(X_tr, y_tr)
p = m.predict_proba(X_te)
r = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
print_metrics(r, "预处理+LightGBM")

print("=== 对比 ===")
print("  基线(原始特征):  CSI=0.993")
print(f"  预处理(标准化):  CSI={r['CSI']:.4f}  ({'+' if r['CSI']>0.993 else ''}{r['CSI']-0.993:+.4f})")

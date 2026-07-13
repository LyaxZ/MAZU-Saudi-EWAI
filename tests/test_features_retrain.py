"""特征增强重训：preprocessor + temporal_features + LightGBM"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from data.preprocessor import DataPreprocessor
from features.temporal_features import TemporalFeatureBuilder
from features.spatial_features import SpatialFeatureBuilder
from config.model_config import FLASH_FLOOD_FEATURES
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

TRAIN_S, TRAIN_E = "2025-06-01", "2025-08-31"
TEST_S,  TEST_E  = "2025-10-01", "2025-10-15"
BASE_FEATS = FLASH_FLOOD_FEATURES
LABEL_VARS = ["flash_flood_risk","heatwave_day_flag","tmax_c","wind10_speed","rh2m","vpd_kpa","orography","ivt"]

# ============ 1. 加载数据 ============
print("加载数据...")
ds_tr = load_date_range(TRAIN_S, TRAIN_E, variables=BASE_FEATS+LABEL_VARS, show_progress=True)
ds_te = load_date_range(TEST_S, TEST_E, variables=BASE_FEATS+LABEL_VARS, show_progress=True)

df_tr = ds_tr.to_dataframe().fillna(0)
df_te = ds_te.to_dataframe().fillna(0)
print(f"训练: {len(df_tr):,}  测试: {len(df_te):,}")

# ============ 2. 预处理 ============
print("\n预处理...")
pp = DataPreprocessor(strategy="temporal", scaler="standard", clip_outliers=True)
df_tr_clean = pp.fit_transform(df_tr.reset_index(), feature_cols=BASE_FEATS).set_index(["day","latitude","longitude"])
df_te_clean = pp.transform(df_te.reset_index()).set_index(["day","latitude","longitude"])

# ============ 3. 时序特征 ============
print("构建时序特征...")
tb = TemporalFeatureBuilder(windows=[3,5,7], include_trend=True, include_extreme_days=True)
df_tr_feat = tb.build(df_tr_clean.reset_index()).set_index(["day","latitude","longitude"])
df_te_feat = tb.build(df_te_clean.reset_index()).set_index(["day","latitude","longitude"])

# 合并原始+衍生
feat_tr = pd.concat([df_tr_clean[BASE_FEATS], df_tr_feat], axis=1)
feat_te = pd.concat([df_te_clean[BASE_FEATS], df_te_feat], axis=1)
print(f"特征数: {len(feat_tr.columns)} (原始{len(BASE_FEATS)} + 衍生{len(df_tr_feat.columns)})")

# ============ 4. 标签 ============
builder = DisasterLabelBuilder(dust_mode="standard", coastal_mode="standard")
builder.fit(df_tr)
labels_tr = builder.build_all(df_tr)["flash_flood_label"]
labels_te = builder.build_all(df_te)["flash_flood_label"]

# 对齐
idx = feat_tr.index.intersection(labels_tr.index)
X_tr, y_tr = feat_tr.loc[idx], labels_tr.loc[idx].values
idx2 = feat_te.index.intersection(labels_te.index)
X_te, y_te = feat_te.loc[idx2], labels_te.loc[idx2].values
print(f"训练: {len(X_tr):,} (pos={y_tr.mean()*100:.1f}%)  测试: {len(X_te):,} (pos={y_te.mean()*100:.1f}%)")

# ============ 5. 训练（仅100棵树，快速验证） ============
model = LightGBMDisasterModel("flash_flood")
model.fit(X_tr, y_tr)
p = model.predict_proba(X_te)
m = compute_all_metrics(y_te, (p>=0.5).astype(int), p)
print_metrics(m, "新特征 LightGBM — flash_flood")

# 对比基线
print("\n=== 对比 ===")
print(f"  基线 (17特征):       CSI=0.993, AUC=1.000")
print(f"  新特征 ({len(feat_tr.columns)}特征): CSI={m['CSI']:.4f}, AUC={m.get('AUC',0):.4f}")
print(f"  变化: {'+' if m['CSI']>0.993 else ''}{m['CSI']-0.993:+.4f}")

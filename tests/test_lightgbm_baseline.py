"""快速验证：LightGBM 基线训练 + 评估"""
import sys
sys.path.insert(0, ".")

import numpy as np
from data.loader import load_date_range
from config.model_config import (
    FLASH_FLOOD_FEATURES, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
)
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics

# 1. 加载训练数据
print("Loading train data...")
ds_train = load_date_range(TRAIN_START, TRAIN_END, 
                           variables=FLASH_FLOOD_FEATURES + ["flash_flood_risk"],
                           show_progress=False)
print(f"Train: {ds_train.sizes['day']} days")

# 2. 转 DataFrame（保留列名，方便特征重要性可读）
df_train = ds_train.to_dataframe().dropna()
X_train = df_train[FLASH_FLOOD_FEATURES]  # 保持 DataFrame
y_train = df_train["flash_flood_risk"].values
y_train = (y_train >= 1).astype(int)  # 多级风险 → 二分类：>=1 为高风险
print(f"Train samples: {len(X_train)}, pos rate: {y_train.mean()*100:.2f}%")

# 3. 测试数据
print("Loading test data...")
ds_test = load_date_range(TEST_START, TEST_END,
                          variables=FLASH_FLOOD_FEATURES + ["flash_flood_risk"],
                          show_progress=False)
df_test = ds_test.to_dataframe().dropna()
X_test = df_test[FLASH_FLOOD_FEATURES]  # 保持 DataFrame
y_test = df_test["flash_flood_risk"].values
y_test = (y_test >= 1).astype(int)  # 二分类
print(f"Test samples: {len(X_test)}, pos rate: {y_test.mean()*100:.2f}%")

# 4. 训练
model = LightGBMDisasterModel("flash_flood")
model.fit(X_train, y_train)

# 5. 评估
y_proba = model.predict_proba(X_test)
y_pred = (y_proba >= 0.5).astype(int)
metrics = compute_all_metrics(y_test, y_pred, y_proba)
print_metrics(metrics, "LightGBM Baseline — flash_flood")

# 6. 特征重要性
imp = model.get_feature_importance()
print("\nTOP 10 Feature Importance:")
print(imp.head(10).to_string(index=False))

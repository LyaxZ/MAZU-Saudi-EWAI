"""LightGBM 基线训练脚本（扩大版）

训练：6-8月（~92天）
验证：9月上半月（~15天，调阈值/早停）
测试：9月下半月（~15天，最终评估）
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from data.loader import load_date_range
from config.model_config import (
    FLASH_FLOOD_FEATURES,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)
from models.lightgbm_model import LightGBMDisasterModel
from evaluation.metrics import compute_all_metrics, print_metrics


def load_and_prepare(start, end, features):
    """加载数据并转换为 (X, y)"""
    ds = load_date_range(start, end,
                         variables=features + ["flash_flood_risk"],
                         show_progress=True)
    df = ds.to_dataframe().dropna()
    X = df[features]
    y = (df["flash_flood_risk"].values >= 1).astype(int)
    return X, y


# ============ 1. 加载数据 ============
print("\n" + "=" * 50)
print("  加载数据")
print("=" * 50)

X_train, y_train = load_and_prepare(TRAIN_START, TRAIN_END, FLASH_FLOOD_FEATURES)
print(f"训练集: {TRAIN_START} ~ {TRAIN_END}, {len(X_train):,} 样本, "
      f"正样本率 {y_train.mean()*100:.2f}%")

X_val, y_val = load_and_prepare(VAL_START, VAL_END, FLASH_FLOOD_FEATURES)
print(f"验证集: {VAL_START} ~ {VAL_END}, {len(X_val):,} 样本, "
      f"正样本率 {y_val.mean()*100:.2f}%")

X_test, y_test = load_and_prepare(TEST_START, TEST_END, FLASH_FLOOD_FEATURES)
print(f"测试集: {TEST_START} ~ {TEST_END}, {len(X_test):,} 样本, "
      f"正样本率 {y_test.mean()*100:.2f}%")

# ============ 2. 训练 ============
print("\n" + "=" * 50)
print("  训练 LightGBM")
print("=" * 50)
model = LightGBMDisasterModel("flash_flood")
model.fit(X_train, y_train)

# ============ 3. 评估 ============
for name, X, y in [("验证集", X_val, y_val), ("测试集", X_test, y_test)]:
    y_proba = model.predict_proba(X)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_all_metrics(y, y_pred, y_proba)
    print_metrics(metrics, f"LightGBM Baseline — {name}")

# ============ 4. 特征重要性 ============
print("\n" + "=" * 50)
print("  特征重要性 TOP 10")
print("=" * 50)
imp = model.get_feature_importance()
for _, row in imp.head(10).iterrows():
    bar = "█" * int(row["importance"] / imp.iloc[0]["importance"] * 30)
    print(f"  {row['feature']:30s} {bar}")

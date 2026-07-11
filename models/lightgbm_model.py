"""
LightGBM 灾害预警模型

继承 BaseModel，实现四类灾害的二分类 LightGBM 基线。
支持 CPU/GPU 自动切换、样本权重（处理类别不平衡）。
"""

import os
import pickle
from typing import Optional, Union

import numpy as np
import pandas as pd
import lightgbm as lgb

from models.base_model import BaseModel
from config.model_config import LIGHTGBM_PARAMS, DEVICE


class LightGBMDisasterModel(BaseModel):
    """基于 LightGBM 的灾害二分类模型。"""

    def __init__(
        self,
        disaster_type: str,
        params: Optional[dict] = None,
        use_gpu: bool = False,
    ):
        super().__init__(disaster_type, name=f"LightGBM-{disaster_type}")
        self.params = params or LIGHTGBM_PARAMS.copy()

        # 设备选择
        if use_gpu and DEVICE == "cuda":
            self.params["device"] = "gpu"
        else:
            self.params["device"] = "cpu"

        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[list] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LightGBMDisasterModel":
        """训练 LightGBM 模型。

        Args:
            X: (n_samples, n_features)
            y: (n_samples,) 0/1 标签
            sample_weight: 样本权重，None 时自动用 balanced 权重
        """
        # 转为 numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        y = np.asarray(y).ravel()

        # 自动处理类别不平衡
        if sample_weight is None:
            pos_count = (y == 1).sum()
            neg_count = (y == 0).sum()
            if pos_count > 0:
                scale = neg_count / pos_count
                sample_weight = np.where(y == 1, scale, 1.0)
                print(f"[{self.name}] 正样本={pos_count}, 负样本={neg_count}, "
                      f"正样本权重={scale:.1f}")

        # 构建 LightGBM Dataset
        train_data = lgb.Dataset(
            X, label=y, weight=sample_weight,
            feature_name=self.feature_names,
        )

        # 训练
        print(f"[{self.name}] 开始训练 (device={self.params['device']}, "
              f"samples={len(y)}, features={X.shape[1]})")
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data],
            valid_names=["train"],
        )

        self.is_fitted = True
        print(f"[{self.name}] 训练完成")
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """输出正类概率。"""
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> pd.DataFrame:
        """获取特征重要性。

        Args:
            importance_type: "gain" 或 "split"

        Returns:
            DataFrame，列：feature, importance
        """
        self._check_fitted()
        imp = self.model.feature_importance(importance_type=importance_type)
        names = self.model.feature_name()
        df = pd.DataFrame({
            "feature": names,
            "importance": imp,
        }).sort_values("importance", ascending=False)
        return df

    def save(self, path: str) -> None:
        """保存模型到文件（pickle 格式）。"""
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "booster": self.model,
                "disaster_type": self.disaster_type,
                "feature_names": self.feature_names,
            }, f)
        print(f"[{self.name}] 模型已保存到 {path}")

    def load(self, path: str) -> None:
        """从文件加载模型。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["booster"]
        self.disaster_type = data["disaster_type"]
        self.feature_names = data.get("feature_names")
        self.is_fitted = True
        print(f"[{self.name}] 模型已从 {path} 加载")

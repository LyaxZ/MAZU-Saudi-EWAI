"""
模型抽象基类：定义 fit / predict / save / load 统一接口

所有模型（LightGBM / LSTM / 堆叠融合）均继承此基类。
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """灾害预警模型抽象基类。"""

    def __init__(self, disaster_type: str, name: str = "BaseModel"):
        self.disaster_type = disaster_type
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """训练模型。

        Args:
            X: 特征矩阵
            y: 标签 (0/1)
            sample_weight: 样本权重（用于处理类别不平衡）

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """输出正类概率。

        Args:
            X: 特征矩阵

        Returns:
            shape (n_samples,) 的概率数组
        """
        ...

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """输出二分类预测。

        Args:
            X: 特征矩阵
            threshold: 概率阈值

        Returns:
            shape (n_samples,) 的 0/1 数组
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型到文件。"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从文件加载模型。"""
        ...

    def _check_fitted(self) -> None:
        """确保模型已训练。"""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} 尚未训练，请先调用 fit()")

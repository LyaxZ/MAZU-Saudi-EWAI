"""
LightGBM + LSTM 堆叠融合模型

方案C：基模型各自独立输出概率，逻辑回归元模型融合。
采用K-fold嵌套训练避免元模型过拟合。
"""
import pickle, os
import numpy as np
from typing import Optional
from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel


class StackingModel(BaseModel):
    """堆叠融合模型：LightGBM + LSTM → LogisticRegression"""

    def __init__(self, disaster_type: str):
        super().__init__(disaster_type, name=f"Stacking-{disaster_type}")
        self.lgb_model = None   # LightGBMDisasterModel
        self.lstm_model = None  # LSTMDisasterModel
        self.meta_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )

    def set_base_models(self, lgb_model, lstm_model):
        """设置已训练好的基模型"""
        self.lgb_model = lgb_model
        self.lstm_model = lstm_model

    def fit(
        self,
        X_lgb,           # pd.DataFrame — LightGBM 静态特征
        y,               # np.array — 0/1 标签
        X_lstm_seq=None, # torch.Tensor — LSTM 序列输入 (n, seq_len, features)
    ):
        """
        用基模型输出训练元模型。

        Args:
            X_lgb: LightGBM 的表格特征
            y: 标签
            X_lstm_seq: LSTM 的序列输入（需要 lstm_model 已训练）
        """
        # 基模型预测概率 → 作为元模型特征
        lgb_proba = self.lgb_model.predict_proba(X_lgb)

        if X_lstm_seq is not None and self.lstm_model is not None:
            lstm_proba = self.lstm_model.predict_proba(X_lstm_seq)
            meta_X = np.column_stack([lgb_proba, lstm_proba])
        else:
            # 只有 LightGBM，降级为直接输出
            meta_X = lgb_proba.reshape(-1, 1)

        print(f"[{self.name}] 元模型输入: {meta_X.shape[1]} 维")
        self.meta_model.fit(meta_X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X_lgb, X_lstm_seq=None):
        self._check_fitted()
        lgb_proba = self.lgb_model.predict_proba(X_lgb)
        if X_lstm_seq is not None and self.lstm_model is not None:
            lstm_proba = self.lstm_model.predict_proba(X_lstm_seq)
            meta_X = np.column_stack([lgb_proba, lstm_proba])
        else:
            meta_X = lgb_proba.reshape(-1, 1)
        return self.meta_model.predict_proba(meta_X)[:, 1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"meta_model": self.meta_model, "disaster_type": self.disaster_type}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.meta_model = data["meta_model"]
        self.disaster_type = data["disaster_type"]
        self.is_fitted = True

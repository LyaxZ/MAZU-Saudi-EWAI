"""
LSTM 时序特征提取器

用于从气象时间序列中提取 64 维时序特征向量，
后续拼接至 LightGBM 输入中作为额外特征列。

架构：2层 BiLSTM → 64维特征 → 分类头
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

from models.base_model import BaseModel
from config.model_config import DEVICE


class WeatherLSTM(nn.Module):
    """2层双向 LSTM，输出时序特征向量 + 分类 logits。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # 将 BiLSTM 最后一层输出映射到特征向量
        lstm_out_dim = hidden_dim * 2  # bidirectional
        self.feature_head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # 分类头（训练 LSTM 单独分类时使用）
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (batch, seq_len, input_dim)
            return_features: True 时返回 64 维特征向量，用于拼接 LightGBM

        Returns:
            return_features=False: logits (batch, 1)
            return_features=True: features (batch, 64)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)                # (batch, seq, hidden*2)
        last_out = lstm_out[:, -1, :]                       # 取最后一步的输出
        features = self.feature_head(last_out)              # (batch, 64)

        if return_features:
            return features
        return self.classifier(features)                    # (batch, 1)


class LSTMDisasterModel(BaseModel):
    """LSTM 灾害预警模型（继承 BaseModel 统一接口）。"""

    def __init__(
        self,
        disaster_type: str,
        input_dim: int = 17,
        hidden_dim: int = 128,
        output_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 256,
    ):
        super().__init__(disaster_type, name=f"LSTM-{disaster_type}")
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = WeatherLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LSTMDisasterModel":
        """
        Args:
            X: (n_samples, seq_len, n_features) tensor
            y: (n_samples,) tensor
        """
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print(f"[{self.name}] 开始训练 (device={DEVICE}, epochs={self.epochs}, "
              f"samples={len(X)}, batch={self.batch_size})")

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE).unsqueeze(1)

                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} — loss={avg_loss:.4f}")

        self.is_fitted = True
        print(f"[{self.name}] 训练完成")
        return self

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """输出正类概率。"""
        self._check_fitted()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(DEVICE))
            proba = torch.sigmoid(logits).cpu().numpy().ravel()
        return proba

    def extract_features(self, X: torch.Tensor) -> np.ndarray:
        """提取 64 维时序特征向量（用于拼接 LightGBM）。"""
        self._check_fitted()
        self.model.eval()
        with torch.no_grad():
            features = self.model(X.to(DEVICE), return_features=True)
        return features.cpu().numpy()

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "disaster_type": self.disaster_type,
            "input_dim": self.input_dim,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=DEVICE)
        self.input_dim = checkpoint["input_dim"]
        self.model = WeatherLSTM(input_dim=self.input_dim).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        self.disaster_type = checkpoint["disaster_type"]
        self.is_fitted = True

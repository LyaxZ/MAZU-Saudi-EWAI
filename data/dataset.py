"""
PyTorch 序列数据集：将 NetCDF 网格数据转为 (seq_len, features) → label

每个样本：过去 seq_len 天的气象特征 → 下一天的 flash_flood_risk
对所有网格点独立构建序列。
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from data.loader import load_date_range


class WeatherSequenceDataset(Dataset):
    """气象时序数据集。

    对每个格点构建滑动窗口：
    - X: (seq_len, n_features) 过去 N 天的气象指标
    - y: 下一天的灾害标签 (0/1)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        features: list,
        label_var: str = "flash_flood_risk",
        seq_len: int = 7,
        label_threshold: float = 1.0,
        max_samples: int | None = None,
    ):
        """
        Args:
            start_date: 数据起始日期
            end_date: 数据结束日期（最后一个样本的标签日期）
            features: 入模特征名列表
            label_var: 标签变量名
            seq_len: 序列长度（回溯天数）
            label_threshold: 标签二值化阈值（>= 此值 → 1）
            max_samples: 最大样本数（小批量测试用），None 表示全部
        """
        self.features = features
        self.seq_len = seq_len
        self.label_threshold = label_threshold

        # 1. 加载数据（需多加载 seq_len 天用于构建首条序列）
        import pandas as pd
        load_start = pd.Timestamp(start_date) - pd.Timedelta(days=seq_len)
        ds = load_date_range(
            load_start.strftime("%Y-%m-%d"), end_date,
            variables=features + [label_var],
            show_progress=True,
        )
        self.ds = ds

        # 2. 提取天数、网格维度
        self.days = ds["day"].values
        self.n_lat = ds.sizes["latitude"]
        self.n_lon = ds.sizes["longitude"]
        self.n_days = len(self.days)

        # 3. 构建样本索引（每个有效窗口一个样本）
        self.samples = []
        # 跳过前 seq_len 天（用于历史窗口），从第 seq_len 天开始
        for t in range(seq_len, self.n_days):
            # 快检查标签日是否有有效数据
            label_day_str = pd.Timestamp(self.days[t]).strftime("%Y-%m-%d")
            # 遍历所有格点
            for lat_idx in range(self.n_lat):
                for lon_idx in range(self.n_lon):
                    self.samples.append((t, lat_idx, lon_idx))

        if max_samples is not None and len(self.samples) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        print(f"[Dataset] 总样本: {len(self.samples):,} (seq_len={seq_len}, "
              f"grid={self.n_lat}×{self.n_lon})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, lat_i, lon_i = self.samples[idx]

        # 提取 seq_len 天的特征序列
        X_seq = np.zeros((self.seq_len, len(self.features)), dtype=np.float32)
        for s in range(self.seq_len):
            day_idx = t - self.seq_len + s
            for f_idx, feat in enumerate(self.features):
                val = self.ds[feat].values[day_idx, lat_i, lon_i]
                X_seq[s, f_idx] = val if not np.isnan(val) else 0.0

        # 提取标签（第 t 天）
        label_val = self.ds["flash_flood_risk"].values[t, lat_i, lon_i]
        label_val = 0.0 if np.isnan(label_val) else float(label_val)
        y = 1.0 if label_val >= self.label_threshold else 0.0

        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

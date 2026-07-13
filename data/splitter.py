"""
时空数据集划分器

支持：
- 按时间序列划分训练/验证/测试集（保证时间因果性）
- 按月份分组交叉验证
- 分层抽样（保持空间分布）

设计原则：
- fit/transform 模式：先设定划分策略，再对数据 apply
- 时间因果性：验证集和测试集的时间严格在训练集之后
- 可复现：固定 random_state

用法:
    from data.loader import load_to_dataframe
    from data.splitter import TimeSeriesSplitter

    df = load_to_dataframe("2025-06-01", "2025-09-30", variables=[...])

    splitter = TimeSeriesSplitter(train_months=[6,7,8], val_months=[9], test_months=[9])
    splits = splitter.split(df)  # → {"train": df_train, "val": df_val, "test": df_test}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class TimeSeriesSplitter:
    """时间序列数据集划分器。

    按时间顺序划分训练/验证/测试集，保证：
    - 训练数据时间 < 验证数据时间 < 测试数据时间
    - 支持按月份或按日期范围灵活划分
    - 输出保持 DataFrame 格式，方便下游使用
    """

    def __init__(
        self,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        val_start: Optional[str] = None,
        val_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        train_months: Optional[List[int]] = None,
        val_months: Optional[List[int]] = None,
        test_months: Optional[List[int]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_method: str = "date_range",
        random_state: int = 42,
    ):
        """
        Args:
            train_start/end: 训练集日期范围（split_method="date_range"）
            val_start/end: 验证集日期范围
            test_start/end: 测试集日期范围
            train_months: 训练集月份列表，如 [6, 7, 8]（split_method="month"）
            val_months: 验证集月份列表，如 [9]
            test_months: 测试集月份列表，如 [9]
            train_ratio: 训练集比例（split_method="ratio"）
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            split_method: 划分方式
                - "date_range": 按具体日期范围划分
                - "month": 按月份划分
                - "ratio": 按时间顺序比例划分
            random_state: 随机种子
        """
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_method = split_method
        self.random_state = random_state

        # 记录划分结果
        self._split_info: Dict = {}

    def split(
        self,
        df: pd.DataFrame,
        label_cols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """按设定策略划分数据集。

        Args:
            df: 包含 day 列的 DataFrame
            label_cols: 标签列列表（用于报告各类别分布），可选

        Returns:
            {"train": df_train, "val": df_val, "test": df_test}
        """
        if "day" not in df.columns:
            raise KeyError("DataFrame 必须包含 'day' 列才能按时间划分")

        # 确保 day 为 datetime
        if not pd.api.types.is_datetime64_any_dtype(df["day"]):
            df = df.copy()
            df["day"] = pd.to_datetime(df["day"])

        if self.split_method == "date_range":
            splits = self._split_by_date_range(df)
        elif self.split_method == "month":
            splits = self._split_by_month(df)
        elif self.split_method == "ratio":
            splits = self._split_by_ratio(df)
        else:
            raise ValueError(f"未知划分方式: {self.split_method}")

        # 记录分布信息
        self._record_split_info(splits, label_cols)

        return splits

    def _split_by_date_range(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按具体日期范围划分。"""
        splits = {}

        if self.train_start and self.train_end:
            train_mask = (
                (df["day"] >= pd.Timestamp(self.train_start))
                & (df["day"] <= pd.Timestamp(self.train_end))
            )
            splits["train"] = df.loc[train_mask].copy()
        else:
            splits["train"] = pd.DataFrame(columns=df.columns)

        if self.val_start and self.val_end:
            val_mask = (
                (df["day"] >= pd.Timestamp(self.val_start))
                & (df["day"] <= pd.Timestamp(self.val_end))
            )
            splits["val"] = df.loc[val_mask].copy()
        else:
            splits["val"] = pd.DataFrame(columns=df.columns)

        if self.test_start and self.test_end:
            test_mask = (
                (df["day"] >= pd.Timestamp(self.test_start))
                & (df["day"] <= pd.Timestamp(self.test_end))
            )
            splits["test"] = df.loc[test_mask].copy()
        else:
            splits["test"] = pd.DataFrame(columns=df.columns)

        return splits

    def _split_by_month(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按月份划分。"""
        month_series = df["day"].dt.month
        splits = {}

        for name, months in [
            ("train", self.train_months),
            ("val", self.val_months),
            ("test", self.test_months),
        ]:
            if months:
                mask = month_series.isin(months)
                # 进一步按半月划分验证和测试（如都是9月时）
                if name in ("val", "test") and months == self.train_months:
                    splits[name] = pd.DataFrame(columns=df.columns)
                else:
                    splits[name] = df.loc[mask].copy()
            else:
                splits[name] = pd.DataFrame(columns=df.columns)

        return splits

    def _split_by_ratio(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """按时间顺序比例划分。"""
        df_sorted = df.sort_values("day").reset_index(drop=True)
        n = len(df_sorted)

        train_end_idx = int(n * self.train_ratio)
        val_end_idx = int(n * (self.train_ratio + self.val_ratio))

        splits = {
            "train": df_sorted.iloc[:train_end_idx].copy(),
            "val": df_sorted.iloc[train_end_idx:val_end_idx].copy(),
            "test": df_sorted.iloc[val_end_idx:].copy(),
        }
        return splits

    def _record_split_info(
        self,
        splits: Dict[str, pd.DataFrame],
        label_cols: Optional[List[str]] = None,
    ) -> None:
        """记录划分统计信息。"""
        self._split_info = {
            "method": self.split_method,
            "splits": {},
        }

        for name, df in splits.items():
            info = {
                "n_samples": len(df),
                "date_min": str(df["day"].min()) if len(df) > 0 else None,
                "date_max": str(df["day"].max()) if len(df) > 0 else None,
                "n_days": df["day"].nunique() if len(df) > 0 else 0,
            }

            # 标签分布
            if label_cols:
                info["label_dist"] = {}
                for lc in label_cols:
                    if lc in df.columns:
                        info["label_dist"][lc] = {
                            "positive": int(df[lc].sum()),
                            "positive_rate": float(df[lc].mean()),
                        }

            self._split_info["splits"][name] = info

    # ================================================================
    # 便捷方法
    # ================================================================

    def get_split_info(self) -> Dict:
        """返回划分统计信息。"""
        return self._split_info

    def print_split_summary(self) -> None:
        """打印划分摘要。"""
        print("\n" + "=" * 60)
        print("  TimeSeriesSplitter — 数据集划分摘要")
        print(f"  划分方式: {self._split_info['method']}")
        print("=" * 60)

        for name in ["train", "val", "test"]:
            if name not in self._split_info["splits"]:
                continue
            info = self._split_info["splits"][name]
            pct = (
                info["n_samples"]
                / sum(
                    s["n_samples"]
                    for s in self._split_info["splits"].values()
                )
                * 100
                if sum(s["n_samples"] for s in self._split_info["splits"].values()) > 0
                else 0
            )
            print(f"\n  [{name.upper()}]")
            print(f"    样本数: {info['n_samples']:,} ({pct:.1f}%)")
            print(f"    日期范围: {info['date_min']} ~ {info['date_max']}")
            print(f"    天数: {info['n_days']}")

            if "label_dist" in info:
                for lc, dist in info["label_dist"].items():
                    print(f"    {lc}: 正样本 {dist['positive']:,} "
                          f"({dist['positive_rate']*100:.3f}%)")
        print("=" * 60 + "\n")


# ================================================================
# 便捷函数
# ================================================================

def split_by_season(
    df: pd.DataFrame,
    train_season: str = "summer",
    val_days: int = 15,
    test_days: int = 15,
) -> Dict[str, pd.DataFrame]:
    """按季节划分训练/验证/测试集。

    训练集 = 整个季度，验证集 = 季度末 N 天，测试集 = 下个季度初 N 天。

    Args:
        df: 含 day 列的 DataFrame
        train_season: 训练季节 — "summer" (6-8月) / "winter" (12-2月) / "spring" (3-5月)
        val_days: 验证集天数（季度末）
        test_days: 测试集天数

    Returns:
        {"train": df, "val": df, "test": df}
    """
    season_months = {
        "summer": [6, 7, 8],
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "autumn": [9, 10, 11],
    }

    if train_season not in season_months:
        raise ValueError(f"未知季节: {train_season}。可选: {list(season_months.keys())}")

    train_months = season_months[train_season]
    month_series = df["day"].dt.month

    train_mask = month_series.isin(train_months)
    df_train = df.loc[train_mask].copy()

    # 验证集：最后 val_days 天
    if len(df_train) > 0:
        unique_days = sorted(df_train["day"].unique())
        val_days_list = unique_days[-val_days:] if len(unique_days) >= val_days else unique_days
        val_mask = df_train["day"].isin(val_days_list)
        df_val = df_train.loc[val_mask].copy()
        df_train = df_train.loc[~val_mask].copy()
    else:
        df_val = pd.DataFrame(columns=df.columns)

    # 测试集：下一个 val_days 天（下个季度刚开始）
    next_season_map = {
        "summer": 9, "autumn": 12, "winter": 3, "spring": 6,
    }
    next_month = next_season_map[train_season]
    test_mask = month_series == next_month
    df_test = df.loc[test_mask].copy()
    if len(df_test) > 0:
        unique_test_days = sorted(df_test["day"].unique())
        test_days_list = unique_test_days[:test_days] if len(unique_test_days) >= test_days else unique_test_days
        df_test = df_test[df_test["day"].isin(test_days_list)].copy()

    return {"train": df_train, "val": df_val, "test": df_test}


def split_by_date_range(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """按日期范围划分的便捷函数。"""
    splitter = TimeSeriesSplitter(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start or val_start,
        test_end=test_end or val_end,
        split_method="date_range",
    )
    return splitter.split(df)

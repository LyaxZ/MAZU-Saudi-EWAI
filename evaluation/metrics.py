"""
气象预警模型评估指标

实现 CSI、POD、FAR、FBIAS、AUC-ROC、F1-score。
所有指标基于混淆矩阵计算：TP, FP, TN, FN。
"""

import numpy as np
from typing import Dict, Optional, Tuple


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """计算混淆矩阵分量。

    Returns:
        (TP, FP, TN, FN)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Critical Success Index（临界成功指数）。

    CSI = TP / (TP + FP + FN)
    值域 [0, 1]，越高越好。同时惩罚漏报和空报。
    """
    tp, fp, _, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def pod(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Probability of Detection（命中率/召回率）。

    POD = TP / (TP + FN)
    值域 [0, 1]，越高越好。衡量"实际发生了，报出来了吗"。
    """
    tp, _, _, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Alarm Ratio（虚警率）。

    FAR = FP / (TP + FP)
    值域 [0, 1]，越低越好。衡量"报了预警，是假的多吗"。
    """
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    denom = tp + fp
    return fp / denom if denom > 0 else 0.0


def fbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Frequency Bias（频率偏差）。

    FBIAS = (TP + FP) / (TP + FN)
    = 1 无偏，>1 高估频率，<1 低估频率。
    """
    tp, fp, _, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fn
    return (tp + fp) / denom if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1-score = 2 * precision * recall / (precision + recall)。"""
    tp, fp, _, fn = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """计算全部评估指标。

    Args:
        y_true: 真实标签
        y_pred: 预测标签 (0/1)
        y_proba: 预测概率（为 None 时跳过 AUC）

    Returns:
        指标字典
    """
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)

    metrics = {
        "CSI": csi(y_true, y_pred),
        "POD": pod(y_true, y_pred),
        "FAR": far(y_true, y_pred),
        "FBIAS": fbias(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }

    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            yt = np.asarray(y_true).ravel()
            unique = np.unique(yt)
            if len(unique) >= 2:
                metrics["AUC"] = float(roc_auc_score(yt, y_proba))
            else:
                metrics["AUC"] = float("nan")
        except Exception as e:
            log.warning(f"[metrics] AUC 计算失败: {e}")
            metrics["AUC"] = float("nan")

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "评估结果") -> None:
    """格式化打印评估指标。"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k in ("TP", "FP", "TN", "FN"):
            print(f"  {k:8s}: {v:8d}")
        else:
            print(f"  {k:8s}: {v:8.4f}")
    print(f"{'='*50}\n")

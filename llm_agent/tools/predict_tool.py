"""
模型预测工具 — LLM 可调用接口

供 LLM Agent 通过 Function Calling 调用：
- 输入: 日期 + 灾害类型 + (可选) 空间范围
- 输出: 风险概率分布 + 高风险区域摘要

用法（LLM工具描述格式）:
    {
        "name": "predict_risk",
        "description": "查询指定日期和灾害类型的风险预测结果",
        "parameters": {
            "date": "日期 (YYYY-MM-DD)",
            "disaster_type": "灾害类型: flash_flood / extreme_heat / dust_wind / coastal_wave",
            "lat_min/lat_max/lon_min/lon_max": "可选空间范围"
        }
    }
"""

import sys
import os
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.loader import load_date_range
from models.inference import DisasterInference, _prepare_features, add_latlon_features
from config.model_config import DISASTER_FEATURES, DISASTER_THRESHOLDS
from config.disaster_config import get_label_config

# 全局单例推理引擎（首次加载模型，后续复用）
_inference_engine: Optional[DisasterInference] = None


def _get_engine() -> DisasterInference:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = DisasterInference()
    return _inference_engine


class PredictTool:
    """模型风险预测工具（基于 LightGBM 训练模型）。

    封装 DisasterInference 推理引擎，为 LLM Agent 提供结构化风险查询能力。
    """

    TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "predict_risk",
            "description": (
                "查询指定日期和灾害类型在沙特阿拉伯的风险预测结果。"
                "返回高风险区域数量、平均风险值、最高风险值和受影响面积估算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "查询日期，格式 YYYY-MM-DD，如 2025-06-15",
                    },
                    "disaster_type": {
                        "type": "string",
                        "enum": ["flash_flood", "extreme_heat", "dust_wind", "coastal_wave"],
                        "description": "灾害类型",
                    },
                    "lat_min": {
                        "type": "number",
                        "description": "可选，纬度下限 (16.0~32.0)",
                    },
                    "lat_max": {
                        "type": "number",
                        "description": "可选，纬度上限",
                    },
                    "lon_min": {
                        "type": "number",
                        "description": "可选，经度下限 (34.0~56.0)",
                    },
                    "lon_max": {
                        "type": "number",
                        "description": "可选，经度上限",
                    },
                },
                "required": ["date", "disaster_type"],
            },
        },
    }

    def __init__(self):
        self.engine = _get_engine()

    def __call__(
        self,
        date: str,
        disaster_type: str,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
    ) -> Dict:
        """执行风险预测。

        Returns:
            {
                "status": "success" | "error",
                "date": str,
                "disaster_type": str,
                "n_total_cells": int,
                "n_high_risk_cells": int,
                "mean_risk": float,
                "max_risk": float,
                "affected_area_km2": float,
                "top_risk_locations": [{"lat":, "lon":, "risk":}, ...],
                "message": str,
            }
        """
        try:
            # 使用 DisasterInference 统一推理
            result = self.engine.predict_from_nc(date, disaster_type)

            proba = result["proba"]
            threshold = result["threshold"]
            lat_arr = result["lat"]
            lon_arr = result["lon"]
            high_mask = proba >= threshold

            # 空间过滤
            if any(x is not None for x in [lat_min, lat_max, lon_min, lon_max]):
                mask = np.ones(len(proba), dtype=bool)
                if lat_min is not None:
                    mask &= lat_arr >= lat_min
                if lat_max is not None:
                    mask &= lat_arr <= lat_max
                if lon_min is not None:
                    mask &= lon_arr >= lon_min
                if lon_max is not None:
                    mask &= lon_arr <= lon_max
                proba = proba[mask]
                lat_arr = lat_arr[mask]
                lon_arr = lon_arr[mask]
                high_mask = high_mask[mask]

            n_high = int(high_mask.sum())

            n_total = len(proba)
            if n_total == 0:
                return {"status": "error", "message": f"日期 {date} 在指定范围内无数据"}

            mean_risk = float(np.mean(proba))
            max_risk = float(np.max(proba))

            # 前5高风险位置
            top_idx = np.argsort(proba)[-5:][::-1]
            top_locations = [
                {"lat": float(lat_arr[i]), "lon": float(lon_arr[i]),
                 "risk": round(float(proba[i]), 4)}
                for i in top_idx
            ]

            name = get_label_config(disaster_type)["name_cn"]
            message = (
                f"{date} {name}风险预测（LightGBM模型）：共分析 {n_total:,} 个格点，"
                f"其中高风险格点 {n_high:,} 个 ({n_high/max(n_total,1)*100:.1f}%)。"
                f"平均风险值 {mean_risk:.3f}，最高风险值 {max_risk:.3f}。"
            )

            return {
                "status": "success",
                "date": date,
                "disaster_type": disaster_type,
                "n_total_cells": n_total,
                "n_high_risk_cells": n_high,
                "mean_risk": round(mean_risk, 4),
                "max_risk": round(max_risk, 4),
                "affected_area_km2": n_high * 100,
                "top_risk_locations": top_locations,
                "message": message,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"预测失败: {str(e)}",
            }

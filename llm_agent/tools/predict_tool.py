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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.loader import load_to_dataframe
from data.preprocessor import DataPreprocessor


class PredictTool:
    """模型风险预测工具。

    封装模型预测接口，为 LLM Agent 提供结构化风险查询能力。
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

    def __init__(self, model=None):
        """
        Args:
            model: 训练好的模型对象（LightGBMDisasterModel）。
                   为 None 时使用启发式风险代理。
        """
        self.model = model
        self._preprocessor = None

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

        Args:
            date: 日期
            disaster_type: 灾害类型
            lat_min/max, lon_min/max: 可选空间范围

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
                "message": str,  # 人类可读摘要
            }
        """
        try:
            # 加载数据
            variables = self._get_variables_for_disaster(disaster_type)
            df = load_to_dataframe(date, date, variables=variables, show_progress=False)

            # 空间过滤
            if lat_min is not None:
                df = df[df["latitude"] >= lat_min]
            if lat_max is not None:
                df = df[df["latitude"] <= lat_max]
            if lon_min is not None:
                df = df[df["longitude"] >= lon_min]
            if lon_max is not None:
                df = df[df["longitude"] <= lon_max]

            if len(df) == 0:
                return {
                    "status": "error",
                    "message": f"日期 {date} 在指定范围内无数据",
                }

            # 预测风险
            if self.model is not None:
                # 使用训练好的模型
                feats = self.model.feature_names_
                X = df[feats].fillna(0)
                risk = self.model.predict_proba(X)
            else:
                # 启发式代理
                risk = self._heuristic_risk(df, disaster_type)

            # 统计
            n_total = len(df)
            n_high = int((risk > 0.5).sum())
            mean_risk = float(np.mean(risk))
            max_risk = float(np.max(risk))

            # 前5高风险位置
            top_idx = np.argsort(risk)[-5:][::-1]
            top_locations = []
            for i in top_idx:
                top_locations.append({
                    "lat": float(df.iloc[i]["latitude"]),
                    "lon": float(df.iloc[i]["longitude"]),
                    "risk": round(float(risk[i]), 4),
                })

            # 人类可读消息
            from config.disaster_config import get_label_config
            name = get_label_config(disaster_type)["name_cn"]
            message = (
                f"{date} {name}风险预测：共分析 {n_total:,} 个格点，"
                f"其中高风险格点 {n_high:,} 个 ({n_high/n_total*100:.1f}%)。"
                f"平均风险值 {mean_risk:.3f}，最高风险值 {max_risk:.3f}。"
                f"预估影响面积约 {n_high * 100:,.0f} km²。"
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

    def _get_variables_for_disaster(self, disaster_type: str) -> List[str]:
        """返回该灾害预测所需的最小变量集。"""
        base = ["orography"]
        disaster_vars = {
            "flash_flood": ["daily_precip_total", "cape", "cin"],
            "extreme_heat": ["tmax_c", "t2m_c", "rh2m"],
            "dust_wind": ["wind10_speed", "rh2m", "vpd_kpa"],
            "coastal_wave": ["wind10_speed", "sst_celsius", "ivt"],
        }
        return base + disaster_vars.get(disaster_type, [])

    def _heuristic_risk(self, df: pd.DataFrame, disaster_type: str) -> np.ndarray:
        """启发式风险计算（模型不可用时的代理）。"""
        n = len(df)
        risk = np.zeros(n, dtype=np.float64)

        if disaster_type == "flash_flood":
            precip = df.get("daily_precip_total", pd.Series([0]*n)).values
            cape = df.get("cape", pd.Series([0]*n)).values
            risk = np.clip(precip/50.0, 0, 1)*0.5 + np.clip(cape/2000.0, 0, 1)*0.5
        elif disaster_type == "extreme_heat":
            risk = np.clip((df.get("tmax_c", pd.Series([25]*n)).values - 35)/15.0, 0, 1)
        elif disaster_type == "dust_wind":
            wind = df.get("wind10_speed", pd.Series([0]*n)).values
            rh = df.get("rh2m", pd.Series([50]*n)).values
            risk = np.clip(wind/15.0, 0, 1)*0.6 + np.clip((30-rh)/30.0, 0, 1)*0.4
        elif disaster_type == "coastal_wave":
            wind = df.get("wind10_speed", pd.Series([0]*n)).values
            oro = df.get("orography", pd.Series([0]*n)).values
            risk = (oro < 100).astype(float) * np.clip(wind/15.0, 0, 1)

        return np.clip(risk, 0, 1)

"""
KG 查询工具 — LLM 可调用的知识图谱空间推理接口

供 LLM Agent 通过 Function Calling 调用：
- 输入: 高风险格点坐标 + 灾害类型
- 输出: 受影响区域清单、传播路径、承灾体统计

用法（LLM工具描述格式）:
    {
        "name": "query_kg_impact",
        "description": "基于知识图谱分析灾害影响范围",
        "parameters": {
            "locations": [{"lat":, "lon":, "risk":}, ...],
            "disaster_type": "灾害类型",
        }
    }
"""

import sys
import os
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator


class KGQueryTool:
    """KG 空间推理查询工具。

    封装知识图谱推理接口，为 LLM Agent 提供影响范围查询能力。
    """

    TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "query_kg_impact",
            "description": (
                "基于知识图谱分析灾害影响范围。输入高风险位置和灾害类型，"
                "返回受影响区域清单、传播路径数量和受影响承灾体统计。"
                "适用于分析山洪下游传播、沙尘扩散、风浪内陆影响等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "description": "高风险位置列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": {"type": "number", "description": "纬度"},
                                "lon": {"type": "number", "description": "经度"},
                                "risk": {"type": "number", "description": "风险值 (0~1)"},
                            },
                            "required": ["lat", "lon"],
                        },
                    },
                    "disaster_type": {
                        "type": "string",
                        "enum": ["flash_flood", "extreme_heat", "dust_wind", "coastal_wave"],
                        "description": "灾害类型",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "最大传播跳数，默认5",
                    },
                    "max_distance_km": {
                        "type": "number",
                        "description": "最大传播距离(km)，默认200",
                    },
                },
                "required": ["locations", "disaster_type"],
            },
        },
    }

    def __init__(self, G=None):
        """
        Args:
            G: NetworkX DiGraph（知识图谱），None 时需要后续通过 set_graph 设置
        """
        self.G = G
        self._builder = KnowledgeGraphBuilder()

    def set_graph(self, G) -> None:
        """设置/更新知识图谱。"""
        self.G = G

    def __call__(
        self,
        locations: List[Dict],
        disaster_type: str,
        max_hops: int = 5,
        max_distance_km: float = 200.0,
        wind_direction_deg: Optional[float] = None,
    ) -> Dict:
        """查询KG影响范围。

        Args:
            locations: [{"lat":, "lon":, "risk":?}, ...]
            disaster_type: 灾害类型
            max_hops: 最大传播跳数
            max_distance_km: 最大传播距离
            wind_direction_deg: 沙尘风向（仅 dust_wind 使用）

        Returns:
            {
                "status": "success" | "error",
                "n_source": int,
                "n_affected": int,
                "n_coastal_affected": int,
                "mean_risk": float,
                "top_affected": [{"lat":, "lon":, "risk":}, ...],
                "summary": str,
            }
        """
        if self.G is None:
            return {
                "status": "error",
                "message": "知识图谱未加载。请先构建知识图谱。",
            }

        try:
            # 查找源节点
            source_nodes = []
            source_risks = []
            for loc in locations:
                target_lat = loc["lat"]
                target_lon = loc["lon"]
                risk = loc.get("risk", 0.5)

                best_node = None
                best_dist = float("inf")
                for node, data in self.G.nodes(data=True):
                    n_lat = data.get("lat")
                    n_lon = data.get("lon")
                    if n_lat is None or n_lon is None:
                        continue
                    dist = np.sqrt(
                        (n_lat - target_lat) ** 2 + (n_lon - target_lon) ** 2
                    )
                    if dist < best_dist and dist < 0.2:  # 0.2° ≈ 20km 容差
                        best_dist = dist
                        best_node = node

                if best_node is not None:
                    source_nodes.append(best_node)
                    source_risks.append(risk)

            if not source_nodes:
                return {
                    "status": "error",
                    "message": f"在知识图谱中未找到匹配的源节点。请确认坐标在研究区域内 (16-32°N, 34-56°E)。",
                }

            # 传播
            kwargs = dict(max_hops=max_hops, max_distance_km=max_distance_km)
            if disaster_type == "dust_wind" and wind_direction_deg is not None:
                kwargs["wind_direction_deg"] = wind_direction_deg

            propagator = RiskPropagator(self.G, disaster_type=disaster_type, **kwargs)
            result = propagator.propagate(source_nodes, source_risks)

            # 构建输出
            affected_nodes = result.get("affected_nodes", [])
            risk_scores = result.get("risk_scores", {})
            summary = result.get("summary", {})

            top_affected = []
            for node_id in sorted(
                affected_nodes,
                key=lambda n: risk_scores.get(n, 0),
                reverse=True,
            )[:5]:
                data = self.G.nodes[node_id]
                top_affected.append({
                    "lat": data.get("lat"),
                    "lon": data.get("lon"),
                    "risk": round(risk_scores.get(node_id, 0), 4),
                })

            from config.disaster_config import get_label_config
            name = get_label_config(disaster_type)["name_cn"]

            summary_text = (
                f"{name} 影响分析：从 {len(source_nodes)} 个高风险源出发，"
                f"共 {len(affected_nodes):,} 个格点受到影响。"
                f"平均传播风险 {summary.get('mean_risk', 0):.3f}。"
            )
            coastal = summary.get("coastal_affected", 0)
            if coastal > 0:
                summary_text += f" 其中 {coastal} 个沿海格点受到波及。"

            # 基础设施影响分析
            try:
                from config.infrastructure import (
                    find_nearby_infrastructure, format_infrastructure_impact,
                )
                high_locs = [loc for loc in locations if loc.get("risk", 0) >= 0.5]
                if high_locs:
                    centroid_lat = np.mean([l["lat"] for l in high_locs])
                    centroid_lon = np.mean([l["lon"] for l in high_locs])
                    nearby = find_nearby_infrastructure(centroid_lat, centroid_lon, radius_km=50)
                    impact_text = format_infrastructure_impact(nearby)
                    if "受影响" in impact_text:
                        summary_text += "\n\n🏗️ 承灾体影响分析：\n" + impact_text
            except ImportError:
                pass

            return {
                "status": "success",
                "disaster_type": disaster_type,
                "n_source": len(source_nodes),
                "n_affected": len(affected_nodes),
                "n_coastal_affected": coastal,
                "mean_risk": round(summary.get("mean_risk", 0), 4),
                "max_distance_km": summary.get("max_distance_km", 0),
                "top_affected": top_affected,
                "summary": summary_text,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"KG 查询失败: {str(e)}",
            }

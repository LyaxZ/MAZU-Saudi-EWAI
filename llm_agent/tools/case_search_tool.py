"""
案例检索工具 — LLM 可调用的历史案例匹配接口

供 LLM Agent 通过 Function Calling 调用：
- 输入: 当前风险特征 + 灾害类型
- 输出: Top-K 相似历史案例 + 曾采取的应对措施

用法（LLM工具描述格式）:
    {
        "name": "search_similar_cases",
        "description": "根据当前风险特征检索历史相似案例及应对措施",
        "parameters": {
            "disaster_type": "灾害类型",
            "description_text": "当前风险情况的自然语言描述",
            "top_k": 5,
        }
    }
"""

import sys
import os
from typing import Dict, List, Optional

import numpy as np

from kg.case_retrieval import CaseRetrieval


class CaseSearchTool:
    """历史案例检索工具。

    封装案例检索接口，为 LLM Agent 提供历史经验参考能力。
    支持使用语义描述文本或特征向量进行检索。
    """

    TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "search_similar_cases",
            "description": (
                "检索与当前风险情况相似的历史灾害案例，"
                "返回相似案例的日期、位置、严重程度和曾采取的应对措施。"
                "提供历史经验参考，辅助决策。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "disaster_type": {
                        "type": "string",
                        "enum": ["flash_flood", "extreme_heat", "dust_wind", "coastal_wave"],
                        "description": "灾害类型",
                    },
                    "query_text": {
                        "type": "string",
                        "description": "当前风险情况的自然语言描述，如'强降水+高CAPE，Wadi上游区域'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回案例数，默认3",
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "最低相似度阈值 (0~1)，默认0.3",
                    },
                },
                "required": ["disaster_type"],
            },
        },
    }

    def __init__(self, case_retrieval: Optional[CaseRetrieval] = None):
        """
        Args:
            case_retrieval: CaseRetrieval 实例，None 则创建空库
        """
        self.cr = case_retrieval or CaseRetrieval()

    def __call__(
        self,
        disaster_type: str,
        query_text: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> Dict:
        """检索相似历史案例。

        Args:
            disaster_type: 灾害类型
            query_text: 自然语言查询描述
            query_vector: 特征向量查询（优先于 query_text）
            top_k: 返回数量
            min_similarity: 最低相似度

        Returns:
            {
                "status": "success" | "error",
                "n_results": int,
                "results": [{"similarity":, "date":, "location":, "severity":, "measures":}, ...],
                "summary": str,
            }
        """
        try:
            if query_vector is not None:
                results = self.cr.search(
                    query_vector,
                    top_k=top_k,
                    disaster_type=disaster_type,
                    min_similarity=min_similarity,
                )
            elif query_text is not None:
                # 将文本映射为简单特征向量（关键词匹配代理）
                # 实际部署中应使用 embedding 模型
                dummy_vec = self._text_to_vector(query_text, disaster_type)
                results = self.cr.search(
                    dummy_vec,
                    top_k=top_k,
                    disaster_type=disaster_type,
                    min_similarity=min_similarity,
                )
            else:
                # 无查询条件，返回该类型所有案例
                cases = self.cr.get_all_cases(disaster_type)
                results = []
                for c in cases[:top_k]:
                    results.append({
                        "similarity": 1.0,
                        "case_id": c["id"],
                        "disaster_type": c["disaster_type"],
                        "date": c["date"],
                        "location": c["location"],
                        "severity": c["severity"],
                        "measures": c["measures"],
                        "metadata": c.get("metadata", {}),
                    })

            # 构建摘要
            if not results:
                from config.disaster_config import get_label_config
                name = get_label_config(disaster_type)["name_cn"]
                summary = f"未找到 {name} 的相似历史案例。建议基于物理模型和专家经验做出决策。"
            else:
                summary = f"找到 {len(results)} 个相似历史案例：\n"
                for i, r in enumerate(results, 1):
                    measures = "; ".join(r["measures"][:3]) if r["measures"] else "无记录"
                    summary += (
                        f"{i}. {r['date']} @ {r['location']} "
                        f"(相似度: {r['similarity']:.1%}, 严重度: {r['severity']}/5)\n"
                        f"   措施: {measures}\n"
                    )

            return {
                "status": "success",
                "n_results": len(results),
                "results": results,
                "summary": summary,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"案例检索失败: {str(e)}",
            }

    def _text_to_vector(self, text: str, disaster_type: str) -> np.ndarray:
        """将文本描述转为简单的特征向量。

        使用关键词匹配生成代理向量（维度与案例库特征维度一致）。

        Note: 实际部署应替换为 embedding 模型（如 text-embedding-3-small）。
        """
        # 关键词 → 特征强度映射
        keywords = {
            "强降水": ["precip", "rain"], "暴雨": ["precip", "rain"],
            "高cape": ["cape"], "对流": ["cape", "cin"],
            "高温": ["temp", "heat"], "热浪": ["temp", "heat"],
            "大风": ["wind"], "强风": ["wind"],
            "干燥": ["dry", "rh"], "沙尘": ["dust"],
            "沿海": ["coast", "sst"], "风浪": ["wave", "wind"],
            "wadi": ["flood"], "山区": ["oro"],
        }

        # 获取特征维度
        if self.cr._feature_matrix is not None:
            dim = self.cr._feature_matrix.shape[1]
        else:
            dim = 10  # 默认维度

        vec = np.zeros(dim)

        # 简单关键词匹配
        text_lower = text.lower()
        for kw, indicators in keywords.items():
            if kw in text_lower:
                # 用关键词映射到向量维度（hash trick）
                for ind in indicators:
                    idx = hash(ind) % dim
                    vec[idx] += 0.3

        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

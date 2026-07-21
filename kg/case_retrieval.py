"""
历史案例检索：向量化存储 + 余弦相似度检索

功能：
- 将历史灾害事件编码为特征向量
- 基于余弦相似度检索 Top-K 相似案例
- 返回相似案例的详细信息和应对措施
- 支持案例库的增删改查和持久化

用法:
    from kg.case_retrieval import CaseRetrieval

    cr = CaseRetrieval()
    cr.add_case(
        date="2025-06-15",
        disaster_type="flash_flood",
        feature_vector=np.array([...]),
        metadata={"location": "Jeddah", "severity": 3},
        measures=["疏散低洼地区居民", "关闭Wadi通道"],
    )

    # 检索
    results = cr.search(query_vector, top_k=5)
    for r in results:
        print(f"相似度: {r['similarity']:.3f}, 措施: {r['measures']}")
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime


class CaseRetrieval:
    """历史灾害案例检索系统。

    基于特征向量相似度的案例记忆库，支持：
    - 案例添加与索引管理
    - 余弦相似度 / 欧氏距离检索
    - JSON 持久化存储
    - 按灾害类型过滤
    """

    def __init__(
        self,
        similarity_metric: str = "cosine",
    ):
        """
        Args:
            similarity_metric: 相似度度量 — "cosine" / "euclidean"
        """
        self.similarity_metric = similarity_metric

        # 案例存储
        self._cases: List[Dict] = []
        self._feature_matrix: Optional[np.ndarray] = None  # (n_cases, n_features)
        self._feature_norm: Optional[np.ndarray] = None     # 预计算的 L2 范数

        # 统计
        self._n_cases_by_type: Dict[str, int] = {}

    # ================================================================
    # 案例管理
    # ================================================================

    def add_case(
        self,
        disaster_type: str,
        feature_vector: Optional[np.ndarray] = None,
        date: Optional[str] = None,
        location: Optional[str] = None,
        severity: int = 1,
        measures: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
    ) -> int:
        """添加一个历史案例。

        Args:
            disaster_type: 灾害类型 — "flash_flood" / "extreme_heat" / "dust_wind" / "coastal_wave"
            feature_vector: 气象特征向量 (n_features,)，为None时从description自动生成
            date: 发生日期 (YYYY-MM-DD)
            location: 位置描述
            severity: 严重程度 (1-5)
            measures: 已采取的应对措施列表
            metadata: 其他元数据
            description: 案例文本描述（用于自动生成特征向量）

        Returns:
            案例 ID
        """
        # 自动生成向量
        if feature_vector is None:
            feature_vector = self._text_to_vec(description or "")
        else:
            feature_vector = np.asarray(feature_vector, dtype=np.float64)
        # 兼容 measures 传字符串的情况
        if isinstance(measures, str):
            measures = [measures]
        case_id = len(self._cases)

        case = {
            "id": case_id,
            "disaster_type": disaster_type,
            "date": date or "unknown",
            "location": location or "unknown",
            "severity": severity,
            "measures": measures or [],
            "metadata": metadata or {},
            "feature_vector": feature_vector,
            "added_at": datetime.now().isoformat(),
        }

        self._cases.append(case)

        # 更新特征矩阵
        if self._feature_matrix is None:
            self._feature_matrix = feature_vector.reshape(1, -1)
        else:
            # 确保维度一致
            if feature_vector.shape[0] != self._feature_matrix.shape[1]:
                raise ValueError(
                    f"特征维度不一致: 期望 {self._feature_matrix.shape[1]}, "
                    f"实际 {feature_vector.shape[0]}"
                )
            self._feature_matrix = np.vstack([self._feature_matrix, feature_vector])

        # 更新预计算范数
        self._feature_norm = np.linalg.norm(self._feature_matrix, axis=1)

        # 更新统计
        self._n_cases_by_type[disaster_type] = (
            self._n_cases_by_type.get(disaster_type, 0) + 1
        )

        return case_id

    def add_cases_batch(
        self,
        cases: List[Dict],
    ) -> List[int]:
        """批量添加案例。

        Args:
            cases: 案例列表，每个案例需含 disaster_type 和 feature_vector

        Returns:
            案例 ID 列表
        """
        ids = []
        for case in cases:
            cid = self.add_case(
                disaster_type=case["disaster_type"],
                feature_vector=case["feature_vector"],
                date=case.get("date"),
                location=case.get("location"),
                severity=case.get("severity", 1),
                measures=case.get("measures"),
                metadata=case.get("metadata"),
            )
            ids.append(cid)
        return ids

    def remove_case(self, case_id: int) -> bool:
        """删除指定案例。"""
        if case_id < 0 or case_id >= len(self._cases):
            return False

        disaster_type = self._cases[case_id]["disaster_type"]
        self._cases.pop(case_id)

        # 重建特征矩阵
        if self._cases:
            self._feature_matrix = np.vstack([
                c["feature_vector"] for c in self._cases
            ])
            self._feature_norm = np.linalg.norm(self._feature_matrix, axis=1)
        else:
            self._feature_matrix = None
            self._feature_norm = None

        # 更新统计
        self._n_cases_by_type[disaster_type] = max(
            0, self._n_cases_by_type.get(disaster_type, 1) - 1
        )
        return True

    # ================================================================
    # 检索
    # ================================================================

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        disaster_type: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Dict]:
        """检索与查询向量最相似的 Top-K 案例。

        Args:
            query_vector: 查询特征向量 (n_features,)
            top_k: 返回案例数
            disaster_type: 按灾害类型过滤，None 表示不限制
            min_similarity: 最低相似度阈值

        Returns:
            [
                {
                    "similarity": float,
                    "case": {...},  # 原案例数据
                    "measures": [...],
                },
                ...
            ]
        """
        if self._feature_matrix is None:
            return []

        # 计算相似度
        if self.similarity_metric == "cosine":
            similarities = self._cosine_similarity(query_vector)
        elif self.similarity_metric == "euclidean":
            similarities = self._euclidean_similarity(query_vector)
        else:
            raise ValueError(f"未知度量: {self.similarity_metric}")

        # 排序 + 过滤
        results = []
        for idx, sim in enumerate(similarities):
            if sim < min_similarity:
                continue
            case = self._cases[idx]
            if disaster_type and case["disaster_type"] != disaster_type:
                continue
            results.append({
                "similarity": float(sim),
                "case_id": case["id"],
                "disaster_type": case["disaster_type"],
                "date": case["date"],
                "location": case["location"],
                "severity": case["severity"],
                "measures": case["measures"],
                "metadata": case["metadata"],
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_by_features(
        self,
        df_features: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        top_k: int = 5,
        disaster_type: Optional[str] = None,
    ) -> List[Dict]:
        """从特征 DataFrame 构建查询向量并检索。

        Args:
            df_features: 特征 DataFrame（单行或多行取均值）
            feature_cols: 使用的特征列，None = 全部数值列
            top_k: 返回案例数
            disaster_type: 灾害类型过滤

        Returns:
            搜索结果列表
        """
        if feature_cols is None:
            feature_cols = [
                c for c in df_features.columns
                if np.issubdtype(df_features[c].dtype, np.number)
            ]

        if len(df_features) > 1:
            # 多行取均值
            query = df_features[feature_cols].mean().values
        else:
            query = df_features[feature_cols].values.flatten()

        return self.search(query, top_k=top_k, disaster_type=disaster_type)

    # ================================================================
    # 相似度计算
    # ================================================================

    def _cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        """计算查询向量与所有案例的余弦相似度。"""
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(self._cases))

        # 点积 / (||query|| * ||case_i||)
        dot = self._feature_matrix @ query
        sim = dot / (self._feature_norm * query_norm + 1e-10)
        return np.clip(sim, -1, 1)

    def _euclidean_similarity(self, query: np.ndarray) -> np.ndarray:
        """计算欧氏距离相似度（归一化到 [0, 1]）。

        similarity = 1 / (1 + distance)
        """
        diff = self._feature_matrix - query
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        return 1.0 / (1.0 + distances)

    # ================================================================
    # 生成合成案例（当无真实历史数据时）
    # ================================================================

    def generate_synthetic_cases(
        self,
        df: pd.DataFrame,
        label_cols: List[str],
        feature_cols: Optional[List[str]] = None,
        n_per_disaster: int = 20,
        random_state: int = 42,
    ) -> None:
        """从标注数据中采样生成合成案例库。

        对每种灾害，选取正样本作为案例，计算其特征均值向量入库。
        措施为占位文本。

        Args:
            df: 含特征和标签的 DataFrame
            label_cols: 四类标签列名
            feature_cols: 用于构建特征向量的列
            n_per_disaster: 每种灾害采样的案例数
            random_state: 随机种子
        """
        rng = np.random.default_rng(random_state)

        if feature_cols is None:
            exclude = {"day", "latitude", "longitude"} | set(label_cols)
            feature_cols = [
                c for c in df.columns
                if c not in exclude and np.issubdtype(df[c].dtype, np.number)
            ]

        # 预定义的措施模板（按灾害类型）
        measure_templates = {
            "flash_flood": [
                "发布山洪预警，疏散Wadi沿岸居民",
                "关闭跨Wadi公路通道",
                "启动排水泵站",
                "向低洼地区部署救援队",
                "监测水坝水位，必要时泄洪",
            ],
            "extreme_heat": [
                "发布高温红色预警",
                "开放社区避暑中心",
                "限制户外工作时段（12:00-16:00）",
                "增加医疗急救资源",
                "向脆弱人群发放饮水和降温物资",
            ],
            "dust_wind": [
                "发布沙尘暴橙色预警",
                "关闭机场和主要公路",
                "建议居民留在室内，佩戴口罩",
                "暂停港口作业",
                "向医院调配呼吸系统急救资源",
            ],
            "coastal_wave": [
                "发布沿海强风浪预警",
                "暂停所有海上作业和航行",
                "疏散低洼沿海社区居民",
                "加固海堤和港口设施",
                "部署海岸警卫队应急力量",
            ],
        }

        for label_col in label_cols:
            # 推断灾害类型
            if "flash_flood" in label_col:
                dtype = "flash_flood"
            elif "extreme_heat" in label_col:
                dtype = "extreme_heat"
            elif "dust_wind" in label_col:
                dtype = "dust_wind"
            elif "coastal_wave" in label_col:
                dtype = "coastal_wave"
            else:
                continue

            if label_col not in df.columns:
                continue

            # 取正样本
            pos = df[df[label_col] == 1]
            if len(pos) == 0:
                continue

            n_sample = min(n_per_disaster, len(pos))
            sampled = pos.sample(n=n_sample, random_state=random_state)

            for i, (_, row) in enumerate(sampled.iterrows()):
                vec = row[feature_cols].values.astype(np.float64)
                # 处理 NaN
                vec = np.nan_to_num(vec, nan=0.0)

                location = f"grid({row.get('latitude', '?')}, {row.get('longitude', '?')})"
                date = str(row.get("day", "unknown"))

                # 随机选取措施
                templates = measure_templates.get(dtype, ["启动应急响应"])
                n_measures = rng.integers(1, min(4, len(templates) + 1))
                measures = list(rng.choice(templates, size=n_measures, replace=False))

                self.add_case(
                    disaster_type=dtype,
                    feature_vector=vec,
                    date=date,
                    location=location,
                    severity=rng.integers(1, 4),
                    measures=measures,
                    metadata={
                        "source": "synthetic",
                        "label_col": label_col,
                    },
                )

        print(f"  生成 {len(self._cases)} 个合成案例 "
              f"(类型: {self._n_cases_by_type})")

    # ═══════════════════════════════════════════════
    # 特征向量生成
    # ═══════════════════════════════════════════════

    def _text_to_vec(self, text: str) -> np.ndarray:
        """从描述文本生成简单的特征向量（关键词匹配）。"""
        keywords = {
            "precip": ["强降水", "暴雨", "降雨", "洪水", "precip", "flood", "rain",
                        "洪", "雨", "水"],
            "cape":   ["对流", "CAPE", "雷暴", "冰雹", "thunder", "hail", "cape"],
            "temp":   ["高温", "热浪", "酷热", "heat", "hot", "温度", "47", "52",
                        "°C", "朝觐"],
            "wind":   ["大风", "强风", "风速", "wind", "100km", "25节", "m/s"],
            "dust":   ["沙尘", "沙暴", "Haboob", "dust", "sand", "能见度",
                        "尘", "沙墙"],
            "dry":    ["干燥", "干旱", "dry", "湿度", "VPD"],
            "coast":  ["沿海", "海岸", "港口", "风浪", "海浪", "coast", "wave",
                        "红海", "阿拉伯湾"],
            "wadi":   ["Wadi", "wadi", "河道", "山谷"],
            "urban":  ["城市", "城区", "居民", "疏散", "停课", "机场",
                        "利雅得", "吉达", "麦加"],
        }
        dim = 10
        vec = np.zeros(dim)
        for i, (key, terms) in enumerate(keywords.items()):
            if any(t in text for t in terms):
                vec[i % dim] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ═══════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════

    def save(self, filepath: str) -> None:
        """保存案例库到 JSON 文件（特征向量保存为独立 .npy 文件）。"""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # 保存特征矩阵
        npy_path = filepath.replace(".json", "_features.npy")
        if self._feature_matrix is not None:
            np.save(npy_path, self._feature_matrix)

        # 保存案例元数据（不含向量）
        cases_meta = []
        for case in self._cases:
            c = {k: v for k, v in case.items() if k != "feature_vector"}
            cases_meta.append(c)

        data = {
            "similarity_metric": self.similarity_metric,
            "n_cases_by_type": self._n_cases_by_type,
            "cases": cases_meta,
            "npy_path": npy_path,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  案例库已保存: {filepath} ({len(self._cases)} 案例)")

    @classmethod
    def load(cls, filepath: str) -> "CaseRetrieval":
        """从 JSON 文件加载案例库。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cr = cls(similarity_metric=data.get("similarity_metric", "cosine"))

        # 加载特征矩阵
        npy_path = data.get("npy_path", "")
        if os.path.exists(npy_path):
            feature_matrix = np.load(npy_path)
        else:
            feature_matrix = None

        # 重建案例
        for i, c in enumerate(data["cases"]):
            vec = feature_matrix[i] if feature_matrix is not None else np.array([])
            case = {
                "id": c["id"],
                "disaster_type": c["disaster_type"],
                "date": c.get("date", "unknown"),
                "location": c.get("location", "unknown"),
                "severity": c.get("severity", 1),
                "measures": c.get("measures", []),
                "metadata": c.get("metadata", {}),
                "feature_vector": vec,
                "added_at": c.get("added_at", ""),
            }
            cr._cases.append(case)

        # 重建特征矩阵
        if feature_matrix is not None and len(feature_matrix) > 0:
            cr._feature_matrix = feature_matrix
            cr._feature_norm = np.linalg.norm(feature_matrix, axis=1)

        cr._n_cases_by_type = data.get("n_cases_by_type", {})
        print(f"  案例库已加载: {filepath} ({len(cr._cases)} 案例)")
        return cr

    # ================================================================
    # 查询
    # ================================================================

    def get_case(self, case_id: int) -> Optional[Dict]:
        """获取指定案例。"""
        if 0 <= case_id < len(self._cases):
            case = self._cases[case_id].copy()
            case.pop("feature_vector", None)
            return case
        return None

    def get_all_cases(self, disaster_type: Optional[str] = None) -> List[Dict]:
        """获取所有案例（可选按灾害类型过滤）。"""
        cases = []
        for c in self._cases:
            if disaster_type and c["disaster_type"] != disaster_type:
                continue
            case = {k: v for k, v in c.items() if k != "feature_vector"}
            cases.append(case)
        return cases

    def get_statistics(self) -> Dict:
        """返回案例库统计信息。"""
        return {
            "total_cases": len(self._cases),
            "n_cases_by_type": self._n_cases_by_type,
            "similarity_metric": self.similarity_metric,
            "feature_dim": (self._feature_matrix.shape[1]
                            if self._feature_matrix is not None else 0),
            "has_features": self._feature_matrix is not None,
        }

    def print_summary(self) -> None:
        """打印案例库摘要。"""
        print("\n" + "=" * 60)
        print("  CaseRetrieval — 历史案例库摘要")
        print("=" * 60)
        print(f"  总案例数:   {len(self._cases)}")
        print(f"  相似度度量: {self.similarity_metric}")
        print(f"  特征维度:   {self._feature_matrix.shape[1] if self._feature_matrix is not None else 0}")
        print(f"\n  灾害分布:")
        for dtype, count in self._n_cases_by_type.items():
            print(f"    {dtype}: {count}")
        if self._cases:
            print(f"\n  最近案例:")
            recent = self._cases[-1]
            print(f"    ID: {recent['id']}, 类型: {recent['disaster_type']}, "
                  f"位置: {recent['location']}")
        print("=" * 60 + "\n")

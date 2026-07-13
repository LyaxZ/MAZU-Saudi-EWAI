"""
图统计特征提取：从知识图谱中为每个网格节点计算图特征

特征分类：
- 下游暴露度：N跳内可达的网格数、沿海节点数（山洪传播范围）
- 上游汇水面积：流向当前节点的上游网格数（集水区大小）
- 局部拓扑：节点度、最近沿海距离
- 路径特征：下游路径平均坡度、最大海拔差

这些特征作为 LightGBM 的额外入模列，捕捉空间拓扑信息。

用法:
    from kg.graph_builder import KnowledgeGraphBuilder
    from kg.graph_features import GraphFeatureExtractor

    G = KnowledgeGraphBuilder().build(df)
    extractor = GraphFeatureExtractor(max_hops=5)
    df_features = extractor.extract_all(G)  # → DataFrame, index=node_id
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


class GraphFeatureExtractor:
    """图特征提取器。

    对知识图谱中的每个网格节点，计算下游/上游图统计特征。
    """

    def __init__(
        self,
        max_hops: int = 5,
        edge_types_for_downstream: Tuple[str, ...] = ("flows_to",),
        edge_types_for_upstream: Tuple[str, ...] = ("flows_to",),
    ):
        """
        Args:
            max_hops: BFS 最大搜索深度（跳数）
            edge_types_for_downstream: 下游遍历使用的边类型
            edge_types_for_upstream: 上游遍历使用的边类型
        """
        self.max_hops = max_hops
        self.edge_types_ds = edge_types_for_downstream
        self.edge_types_us = edge_types_for_upstream

        # 缓存
        self._downstream_cache: Dict[int, Dict] = {}
        self._upstream_cache: Dict[int, Dict] = {}
        self._generated_features: List[str] = []

        # 节点属性
        self._n_lat: int = 0
        self._n_lon: int = 0

    # ================================================================
    # extract_all: 提取全部图特征
    # ================================================================

    def extract_all(self, G: nx.DiGraph) -> pd.DataFrame:
        """提取全部图特征，返回 DataFrame。

        Args:
            G: 知识图谱（NetworkX DiGraph）

        Returns:
            DataFrame，index=node_id，列为图特征
        """
        print("=" * 60)
        print("  GraphFeatureExtractor — 提取图特征")
        print(f"  最大跳数: {self.max_hops}")
        print("=" * 60)

        n_nodes = G.number_of_nodes()
        print(f"  节点总数: {n_nodes:,}")

        # 预计算 lat/lon 网格信息
        self._extract_grid_info(G)

        features = {}

        # 1. 局部拓扑特征（一次性计算）
        print("  [1/4] 局部拓扑特征...")
        local = self._extract_local_features(G)
        for k, v in local.items():
            features[k] = v

        # 2. 下游特征（BFS，对每个节点）
        print("  [2/4] 下游暴露度特征...")
        downstream = self._extract_downstream_features(G)
        for k, v in downstream.items():
            features[k] = v

        # 3. 上游特征
        print("  [3/4] 上游汇水特征...")
        upstream = self._extract_upstream_features(G)
        for k, v in upstream.items():
            features[k] = v

        # 4. 沿海/地形特征
        print("  [4/4] 沿海地形特征...")
        terrain = self._extract_terrain_features(G)
        for k, v in terrain.items():
            features[k] = v

        df = pd.DataFrame(features)
        self._generated_features = list(df.columns)

        print(f"\n  生成 {len(self._generated_features)} 个图特征\n")
        return df

    # ================================================================
    # 网格信息
    # ================================================================

    def _extract_grid_info(self, G: nx.DiGraph) -> None:
        """从图中提取网格大小信息。"""
        max_lat_idx = 0
        max_lon_idx = 0
        for _, data in G.nodes(data=True):
            if "lat_idx" in data:
                max_lat_idx = max(max_lat_idx, data["lat_idx"])
            if "lon_idx" in data:
                max_lon_idx = max(max_lon_idx, data["lon_idx"])
        self._n_lat = max_lat_idx + 1
        self._n_lon = max_lon_idx + 1

    # ================================================================
    # 局部拓扑特征
    # ================================================================

    def _extract_local_features(self, G: nx.DiGraph) -> Dict[str, np.ndarray]:
        """提取局部拓扑特征。

        Returns:
            {feature_name: (n_nodes,) array}
        """
        n = G.number_of_nodes()
        node_ids = sorted(G.nodes())

        out_degree = np.zeros(n, dtype=np.float32)
        in_degree = np.zeros(n, dtype=np.float32)
        flow_out_degree = np.zeros(n, dtype=np.float32)
        flow_in_degree = np.zeros(n, dtype=np.float32)
        adj_degree = np.zeros(n, dtype=np.float32)

        for idx, u in enumerate(node_ids):
            out_degree[idx] = G.out_degree(u)
            in_degree[idx] = G.in_degree(u)

            # 按边类型统计
            for _, _, data in G.out_edges(u, data=True):
                if data.get("type") == "flows_to":
                    flow_out_degree[idx] += 1
                elif data.get("type") == "adjacent":
                    adj_degree[idx] += 1

            for _, _, data in G.in_edges(u, data=True):
                if data.get("type") == "flows_to":
                    flow_in_degree[idx] += 1

        return {
            "kg_out_degree": out_degree,
            "kg_in_degree": in_degree,
            "kg_flow_out_degree": flow_out_degree,
            "kg_flow_in_degree": flow_in_degree,
            "kg_adj_degree": adj_degree,
        }

    # ================================================================
    # 下游暴露度特征
    # ================================================================

    def _extract_downstream_features(self, G: nx.DiGraph) -> Dict[str, np.ndarray]:
        """对每个节点计算下游 N 跳内的暴露度统计。

        下游定义：沿 flows_to 边方向（水往低处流的方向）。
        """
        n = G.number_of_nodes()
        node_ids = sorted(G.nodes())

        n_downstream = np.zeros(n, dtype=np.float32)
        n_coastal_downstream = np.zeros(n, dtype=np.float32)
        max_drop_downstream = np.zeros(n, dtype=np.float32)
        hops_to_coast = np.full(n, np.nan, dtype=np.float32)

        for idx, source in enumerate(tqdm(node_ids, desc="    下游BFS", unit="节点")):
            # BFS 沿 flows_to 边
            visited = {source: 0}  # node → hop distance
            queue = deque([source])
            coastal_hops = []

            while queue:
                u = queue.popleft()
                hop = visited[u]
                if hop >= self.max_hops:
                    continue

                for v, data in G[u].items():
                    edge_type = data.get("type", "")
                    if edge_type in self.edge_types_ds and v not in visited:
                        visited[v] = hop + 1
                        queue.append(v)

                        # 检查是否沿海节点
                        if G.nodes[v].get("type") == "coast_point":
                            coastal_hops.append(hop + 1)

            # 统计
            n_downstream[idx] = len(visited) - 1  # 不含自身
            n_coastal_downstream[idx] = len(coastal_hops)

            # 最大海拔差
            source_oro = G.nodes[source].get("orography")
            if source_oro is not None:
                max_drop = 0.0
                for v in visited:
                    v_oro = G.nodes[v].get("orography")
                    if v_oro is not None:
                        drop = source_oro - v_oro
                        if drop > max_drop:
                            max_drop = drop
                max_drop_downstream[idx] = max_drop

            # 到最近沿海节点的跳数
            if coastal_hops:
                hops_to_coast[idx] = min(coastal_hops)

        return {
            "kg_n_downstream": n_downstream,
            "kg_n_coastal_ds": n_coastal_downstream,
            "kg_max_drop_ds": max_drop_downstream,
            "kg_hops_to_coast": hops_to_coast,
        }

    # ================================================================
    # 上游汇水特征
    # ================================================================

    def _extract_upstream_features(self, G: nx.DiGraph) -> Dict[str, np.ndarray]:
        """对每个节点计算上游汇水面积特征。

        上游定义：沿 flows_to 边逆方向（所有流入该节点的上游节点）。
        汇水面积 ≈ 上游格点数 × 单格面积 (~100 km²)。
        """
        n = G.number_of_nodes()
        node_ids = sorted(G.nodes())

        n_upstream = np.zeros(n, dtype=np.float32)
        max_elev_upstream = np.zeros(n, dtype=np.float32)

        for idx, target in enumerate(tqdm(node_ids, desc="    上游BFS", unit="节点")):
            visited = {target: 0}
            queue = deque([target])

            while queue:
                v = queue.popleft()
                hop = visited[v]
                if hop >= self.max_hops:
                    continue

                # 反向查 flows_to 边
                for u in G.predecessors(v):
                    edge_data = G.get_edge_data(u, v)
                    if edge_data and edge_data.get("type") in self.edge_types_us:
                        if u not in visited:
                            visited[u] = hop + 1
                            queue.append(u)

            n_upstream[idx] = len(visited) - 1  # 不含自身

            # 上游最高海拔
            source_oro = G.nodes[target].get("orography")
            if source_oro is not None:
                max_elev = source_oro
                for u in visited:
                    u_oro = G.nodes[u].get("orography")
                    if u_oro is not None and u_oro > max_elev:
                        max_elev = u_oro
                max_elev_upstream[idx] = max_elev - source_oro  # 相对高差

        # 汇水面积 (km²)：每个格点 ~100 km²
        catchment_area = n_upstream * 100.0

        return {
            "kg_n_upstream": n_upstream,
            "kg_catchment_area_km2": catchment_area,
            "kg_relief_upstream": max_elev_upstream,
        }

    # ================================================================
    # 沿海/地形特征
    # ================================================================

    def _extract_terrain_features(self, G: nx.DiGraph) -> Dict[str, np.ndarray]:
        """提取沿海和地形相关特征。"""
        n = G.number_of_nodes()
        node_ids = sorted(G.nodes())

        is_coastal = np.zeros(n, dtype=np.float32)
        orography = np.zeros(n, dtype=np.float32)

        for idx, u in enumerate(node_ids):
            if G.nodes[u].get("type") == "coast_point":
                is_coastal[idx] = 1.0
            oro = G.nodes[u].get("orography")
            if oro is not None:
                orography[idx] = oro

        return {
            "kg_is_coastal": is_coastal,
            "kg_elevation": orography,
        }

    # ================================================================
    # 特征 Dataframe → 原始 DataFrame 映射
    # ================================================================

    def map_to_dataframe(
        self,
        graph_features: pd.DataFrame,
        df: pd.DataFrame,
        G: nx.DiGraph,
    ) -> pd.DataFrame:
        """将图特征（按 node_id 索引）映射回原始的 (day, lat, lon) DataFrame。

        Args:
            graph_features: extract_all() 的输出 (index=node_id)
            df: 原始的 (day, lat, lon) DataFrame
            G: 知识图谱

        Returns:
            与 df 行数相同、仅包含图特征的 DataFrame
        """
        result = pd.DataFrame(index=df.index, columns=graph_features.columns)

        # 建立 (lat, lon) → node_id 映射
        coord_to_node = {}
        for node_id, data in G.nodes(data=True):
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is not None and lon is not None:
                coord_to_node[(round(lat, 4), round(lon, 4))] = node_id

        # 为每行查找对应的图特征
        for idx, row in df.iterrows():
            key = (round(row["latitude"], 4), round(row["longitude"], 4))
            node_id = coord_to_node.get(key)
            if node_id is not None and node_id in graph_features.index:
                result.loc[idx] = graph_features.loc[node_id]

        return result

    # ================================================================
    # 获取信息
    # ================================================================

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表。"""
        return self._generated_features

    def print_summary(self) -> None:
        """打印图特征摘要。"""
        print("\n" + "=" * 60)
        print("  GraphFeatureExtractor — 摘要")
        print("=" * 60)
        print(f"  最大跳数: {self.max_hops}")
        print(f"  生成特征数: {len(self._generated_features)}")
        if self._generated_features:
            print(f"\n  特征列表:")
            for f in self._generated_features:
                print(f"    - {f}")
        print("=" * 60 + "\n")

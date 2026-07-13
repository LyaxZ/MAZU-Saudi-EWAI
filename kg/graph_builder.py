"""
知识图谱构建器：从气象网格数据构建 NetworkX 有向图

图结构：
- 节点类型: grid_cell (~35200个), coast_point, wadi_outlet
- 边类型: flows_to (沿地形梯度流向), adjacent (空间邻接), coastal (沿海连接)

设计原则：
- 流向边基于 orography 梯度：水往低处流（山洪沿Wadi传播）
- 支持大规模图（35200+ 节点）的高效构建
- 图结构可序列化（pickle/GML），供下游 graph_features 和 risk_propagation 使用
- API 可扩展：后续可接入真实的 Wadi 矢量数据、城市/设施 shapefile

用法:
    from data.loader import load_date_range
    from kg.graph_builder import KnowledgeGraphBuilder

    ds = load_date_range("2025-06-15", "2025-06-15", variables=["orography"])
    df = ds.to_dataframe().reset_index()

    builder = KnowledgeGraphBuilder()
    G = builder.build(df)
    builder.save(G, "outputs/models/knowledge_graph.gpickle")
"""

import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


class KnowledgeGraphBuilder:
    """知识图谱构建器。

    从气象网格 DataFrame 构建 NetworkX DiGraph，支持：
    - 网格节点与空间邻接边
    - 基于地形的流向边（山洪传播路径）
    - 沿海节点识别
    - 图统计信息
    """

    # 节点类型常量
    NODE_GRID = "grid_cell"
    NODE_COAST = "coast_point"
    NODE_WADI_OUTLET = "wadi_outlet"

    # 边类型常量
    EDGE_FLOWS_TO = "flows_to"
    EDGE_ADJACENT = "adjacent"
    EDGE_COASTAL = "coastal"

    def __init__(
        self,
        coastal_orography_max: float = 100.0,
        min_slope_for_flow: float = 0.0,
        include_diagonal: bool = True,
    ):
        """
        Args:
            coastal_orography_max: 视为沿海的最高海拔 (m)
            min_slope_for_flow: 形成流向边的最小坡度阈值
            include_diagonal: 是否包含对角邻接（8邻域 vs 4邻域）
        """
        self.coastal_orography_max = coastal_orography_max
        self.min_slope_for_flow = min_slope_for_flow
        self.include_diagonal = include_diagonal

        # 缓存
        self._lat_grid: Optional[List[float]] = None
        self._lon_grid: Optional[List[float]] = None
        self._n_lat: int = 0
        self._n_lon: int = 0
        self._oro_grid: Optional[np.ndarray] = None
        self._flow_direction: Optional[np.ndarray] = None  # (n_lat, n_lon, 2) — [dlat, dlon] 方向
        self._coast_mask: Optional[np.ndarray] = None

        # 统计
        self._stats: Dict = {}

    # ================================================================
    # build: 构建知识图谱
    # ================================================================

    def build(self, df: pd.DataFrame) -> nx.DiGraph:
        """从网格数据构建知识图谱。

        Args:
            df: 含 latitude, longitude, orography 列的 DataFrame
                至少需要一天的 orography 数据

        Returns:
            networkx.DiGraph
        """
        print("=" * 60)
        print("  KnowledgeGraphBuilder — 构建知识图谱")
        print("=" * 60)

        # 1. 提取网格信息
        self._extract_grid(df)

        # 2. 创建图
        G = nx.DiGraph()

        # 3. 添加网格节点
        print("\n  [1/4] 添加网格节点...")
        self._add_grid_nodes(G)

        # 4. 添加空间邻接边
        print("  [2/4] 添加空间邻接边...")
        self._add_adjacent_edges(G)

        # 5. 计算流向 + 添加流向边
        print("  [3/4] 计算地形流向 + 添加流向边...")
        self._compute_flow_direction()
        self._add_flow_edges(G)

        # 6. 识别沿海节点 + 添加沿海边
        print("  [4/4] 识别沿海节点 + 添加沿海边...")
        self._identify_coast()
        self._add_coastal_edges(G)

        # 记录统计
        self._stats = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_grid_cells": sum(1 for _, d in G.nodes(data=True)
                                if d.get("type") == self.NODE_GRID),
            "n_coast_points": sum(1 for _, d in G.nodes(data=True)
                                  if d.get("type") == self.NODE_COAST),
            "n_flows_to_edges": sum(1 for _, _, d in G.edges(data=True)
                                    if d.get("type") == self.EDGE_FLOWS_TO),
            "n_adjacent_edges": sum(1 for _, _, d in G.edges(data=True)
                                    if d.get("type") == self.EDGE_ADJACENT),
            "n_coastal_edges": sum(1 for _, _, d in G.edges(data=True)
                                   if d.get("type") == self.EDGE_COASTAL),
            "coastal_ratio": float(self._coast_mask.mean()) if self._coast_mask is not None else 0,
        }

        self._print_build_summary()
        return G

    # ================================================================
    # 网格提取
    # ================================================================

    def _extract_grid(self, df: pd.DataFrame) -> None:
        """从 DataFrame 提取网格结构和地形数据。"""
        self._lat_grid = sorted(df["latitude"].unique())
        self._lon_grid = sorted(df["longitude"].unique())
        self._n_lat = len(self._lat_grid)
        self._n_lon = len(self._lon_grid)

        lat_res = self._lat_grid[1] - self._lat_grid[0] if self._n_lat > 1 else 0.1
        lon_res = self._lon_grid[1] - self._lon_grid[0] if self._n_lon > 1 else 0.1
        print(f"  网格: {self._n_lat} × {self._n_lon} "
              f"(分辨率 {lat_res:.2f}° × {lon_res:.2f}°)")

        # 提取 orography
        self._oro_grid = np.full((self._n_lat, self._n_lon), np.nan)

        if "orography" in df.columns:
            # 使用第一天的 orography（地形不随时间变化）
            first_day = df["day"].min() if "day" in df.columns else None
            if first_day is not None:
                mask = df["day"] == first_day
            else:
                mask = slice(None)

            for _, row in df[mask].iterrows():
                lat_idx = self._lat_to_idx(row["latitude"])
                lon_idx = self._lon_to_idx(row["longitude"])
                if lat_idx is not None and lon_idx is not None:
                    self._oro_grid[lat_idx, lon_idx] = row["orography"]

            n_valid = (~np.isnan(self._oro_grid)).sum()
            print(f"  orography 有效格点: {n_valid:,} / {self._n_lat * self._n_lon:,}")
        else:
            print("  ⚠ 未找到 orography 列，跳过流向计算")

    # ================================================================
    # 节点
    # ================================================================

    def _add_grid_nodes(self, G: nx.DiGraph) -> None:
        """为每个有效格点创建 grid_cell 节点。"""
        node_id = 0
        for i in range(self._n_lat):
            for j in range(self._n_lon):
                lat = self._lat_grid[i]
                lon = self._lon_grid[j]
                oro = self._oro_grid[i, j] if self._oro_grid is not None else np.nan

                G.add_node(
                    node_id,
                    type=self.NODE_GRID,
                    lat=lat,
                    lon=lon,
                    lat_idx=i,
                    lon_idx=j,
                    orography=oro if not np.isnan(oro) else None,
                )
                node_id += 1

    # ================================================================
    # 空间邻接边
    # ================================================================

    def _add_adjacent_edges(self, G: nx.DiGraph) -> None:
        """添加空间邻接边（8邻域）。"""
        # 邻域偏移量
        if self.include_diagonal:
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        else:
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        edges_added = 0
        for i in range(self._n_lat):
            for j in range(self._n_lon):
                u = self._idx_to_node(i, j)
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self._n_lat and 0 <= nj < self._n_lon:
                        v = self._idx_to_node(ni, nj)
                        dist = np.sqrt(
                            ((self._lat_grid[i] - self._lat_grid[ni]) * 111.0) ** 2 +
                            ((self._lon_grid[j] - self._lon_grid[nj]) * 111.0 *
                             np.cos(np.radians(self._lat_grid[i]))) ** 2
                        )
                        G.add_edge(u, v, type=self.EDGE_ADJACENT, distance_km=dist)
                        edges_added += 1

        print(f"    添加 {edges_added:,} 条邻接边")

    # ================================================================
    # 流向计算
    # ================================================================

    def _compute_flow_direction(self) -> None:
        """基于 orography 梯度计算每个格点的流向（D8算法）。

        对每个格点，找到8个邻居中海拔最低的方向。
        流向存储为 (di, dj) 偏移量。
        """
        if self._oro_grid is None:
            print("    ⚠ 无 orography 数据，跳过流向计算")
            return

        oro = self._oro_grid.copy()
        # 填充 NaN（以最大值填充 → 水不会流向 NaN 区域）
        nan_mask = np.isnan(oro)
        if nan_mask.any():
            oro[nan_mask] = np.nanmax(oro) + 1000  # 远高于正常海拔

        if self.include_diagonal:
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            ]
        else:
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        flow = np.zeros((self._n_lat, self._n_lon, 2), dtype=int)

        for i in range(self._n_lat):
            for j in range(self._n_lon):
                current_h = oro[i, j]
                best_di, best_dj = 0, 0
                max_drop = self.min_slope_for_flow  # 需要正坡度

                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self._n_lat and 0 <= nj < self._n_lon:
                        drop = current_h - oro[ni, nj]
                        if drop > max_drop:
                            max_drop = drop
                            best_di, best_dj = di, dj

                flow[i, j] = [best_di, best_dj]

        self._flow_direction = flow

        # 统计
        has_flow = (flow[:, :, 0] != 0) | (flow[:, :, 1] != 0)
        n_flow = has_flow.sum()
        print(f"    具有有效流向的格点: {n_flow:,} / {self._n_lat * self._n_lon:,} "
              f"({n_flow / (self._n_lat * self._n_lon) * 100:.1f}%)")

    def _add_flow_edges(self, G: nx.DiGraph) -> None:
        """添加流向边（沿地形梯度方向）。"""
        if self._flow_direction is None:
            return

        edges_added = 0
        for i in range(self._n_lat):
            for j in range(self._n_lon):
                di, dj = self._flow_direction[i, j]
                if di == 0 and dj == 0:
                    continue  # 无流向（局部最低点或平地）
                ni, nj = i + di, j + dj
                if 0 <= ni < self._n_lat and 0 <= nj < self._n_lon:
                    u = self._idx_to_node(i, j)
                    v = self._idx_to_node(ni, nj)
                    # 坡度
                    drop = (self._oro_grid[i, j] - self._oro_grid[ni, nj]
                            if self._oro_grid is not None else 0)

                    dist = np.sqrt(
                        ((self._lat_grid[i] - self._lat_grid[ni]) * 111.0) ** 2 +
                        ((self._lon_grid[j] - self._lon_grid[nj]) * 111.0 *
                         np.cos(np.radians(self._lat_grid[i]))) ** 2
                    )

                    slope = drop / (dist * 1000) if dist > 0 else 0  # m/m

                    G.add_edge(u, v,
                               type=self.EDGE_FLOWS_TO,
                               elevation_drop=drop,
                               slope=slope,
                               distance_km=dist)
                    edges_added += 1

        print(f"    添加 {edges_added:,} 条流向边")

    # ================================================================
    # 沿海识别
    # ================================================================

    def _identify_coast(self) -> None:
        """识别沿海格点。

        策略：
        1. 海拔 < coastal_orography_max
        2. 位于网格边界（最南/最西 → 红海沿岸）
        """
        self._coast_mask = np.zeros((self._n_lat, self._n_lon), dtype=bool)

        if self._oro_grid is not None:
            # 低海拔格点
            low_mask = (self._oro_grid < self.coastal_orography_max) & (~np.isnan(self._oro_grid))
            self._coast_mask |= low_mask

        # 网格边界（海域方向）
        # 红海位于西侧（lon ≈ 34-44°E）和南侧
        self._coast_mask[:, 0] = True   # 最西侧
        self._coast_mask[0, :] = True   # 最南侧
        self._coast_mask[:, -1] = True  # 最东侧（阿拉伯湾）
        self._coast_mask[-1, :] = True  # 最北侧

        n_coastal = self._coast_mask.sum()
        print(f"    沿海格点: {n_coastal:,} / {self._n_lat * self._n_lon:,} "
              f"({n_coastal / (self._n_lat * self._n_lon) * 100:.1f}%)")

        # 更新节点类型
        # （在 add_coastal_edges 中处理）

    def _add_coastal_edges(self, G: nx.DiGraph) -> None:
        """为沿海格点添加沿海连接边 + 更新节点类型。"""
        edges_added = 0

        for i in range(self._n_lat):
            for j in range(self._n_lon):
                node = self._idx_to_node(i, j)
                if self._coast_mask[i, j]:
                    # 更新节点类型
                    G.nodes[node]["type"] = self.NODE_COAST

                    # 连接到相邻沿海格点
                    for di, dj in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self._n_lat and 0 <= nj < self._n_lon
                                and self._coast_mask[ni, nj]):
                            v = self._idx_to_node(ni, nj)
                            if not G.has_edge(node, v):
                                G.add_edge(node, v, type=self.EDGE_COASTAL,
                                           distance_km=10.0)
                                edges_added += 1

        print(f"    添加 {edges_added:,} 条沿海边")

    # ================================================================
    # 索引映射
    # ================================================================

    def _idx_to_node(self, lat_idx: int, lon_idx: int) -> int:
        """(lat_idx, lon_idx) → node ID。"""
        return lat_idx * self._n_lon + lon_idx

    def _node_to_idx(self, node_id: int) -> Tuple[int, int]:
        """node ID → (lat_idx, lon_idx)。"""
        return divmod(node_id, self._n_lon)

    def _lat_to_idx(self, lat: float) -> Optional[int]:
        """latitude → lat_idx。"""
        if self._lat_grid is None:
            return None
        try:
            return self._lat_grid.index(lat)
        except ValueError:
            # 找最近的
            idx = np.argmin([abs(l - lat) for l in self._lat_grid])
            if abs(self._lat_grid[idx] - lat) < 0.001:  # tolerance
                return idx
            return None

    def _lon_to_idx(self, lon: float) -> Optional[int]:
        """longitude → lon_idx。"""
        if self._lon_grid is None:
            return None
        try:
            return self._lon_grid.index(lon)
        except ValueError:
            idx = np.argmin([abs(l - lon) for l in self._lon_grid])
            if abs(self._lon_grid[idx] - lon) < 0.001:
                return idx
            return None

    # ================================================================
    # 序列化
    # ================================================================

    def save(self, G: nx.DiGraph, filepath: str) -> None:
        """保存图为 gpickle 格式。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  图已保存: {filepath}")

    @staticmethod
    def load(filepath: str) -> nx.DiGraph:
        """加载 gpickle 格式的图。"""
        with open(filepath, "rb") as f:
            G = pickle.load(f)
        print(f"  图已加载: {filepath} ({G.number_of_nodes():,} 节点, "
              f"{G.number_of_edges():,} 边)")
        return G

    # ================================================================
    # 查询
    # ================================================================

    def get_downstream_nodes(
        self, G: nx.DiGraph, source_node: int, max_hops: int = 10
    ) -> List[int]:
        """获取从 source_node 出发的下游节点列表（沿 flows_to 边）。

        Args:
            G: 知识图谱
            source_node: 源节点 ID
            max_hops: 最大跳数

        Returns:
            下游节点 ID 列表（按距离排序）
        """
        visited = {source_node: 0}
        queue = [source_node]
        while queue:
            u = queue.pop(0)
            current_hop = visited[u]
            if current_hop >= max_hops:
                continue
            for v, data in G[u].items():
                if data.get("type") == self.EDGE_FLOWS_TO and v not in visited:
                    visited[v] = current_hop + 1
                    queue.append(v)
        return list(visited.keys())

    def get_upstream_nodes(
        self, G: nx.DiGraph, target_node: int, max_hops: int = 10
    ) -> List[int]:
        """获取流向 target_node 的上游节点列表。"""
        # 反向 BFS（沿 flows_to 逆边）
        visited = {target_node: 0}
        queue = [target_node]
        while queue:
            v = queue.pop(0)
            current_hop = visited[v]
            if current_hop >= max_hops:
                continue
            for u, data in G.pred[v].items():  # predecessors
                edge_data = G.get_edge_data(u, v)
                if edge_data and edge_data.get("type") == self.EDGE_FLOWS_TO:
                    if u not in visited:
                        visited[u] = current_hop + 1
                        queue.append(u)
        return list(visited.keys())

    def get_coastal_nodes(self, G: nx.DiGraph) -> List[int]:
        """获取所有沿海节点 ID。"""
        return [n for n, d in G.nodes(data=True)
                if d.get("type") == self.NODE_COAST]

    # ================================================================
    # 报告
    # ================================================================

    def get_stats(self) -> Dict:
        """返回图统计信息。"""
        return self._stats

    def _print_build_summary(self) -> None:
        """打印构建摘要。"""
        print("\n" + "=" * 60)
        print("  KnowledgeGraphBuilder — 构建完成")
        print("=" * 60)
        s = self._stats
        print(f"  总节点数:    {s.get('n_nodes', 0):,}")
        print(f"  总边数:      {s.get('n_edges', 0):,}")
        print(f"  网格节点:    {s.get('n_grid_cells', 0):,}")
        print(f"  沿海节点:    {s.get('n_coast_points', 0):,}")
        print(f"  流向边:      {s.get('n_flows_to_edges', 0):,}")
        print(f"  邻接边:      {s.get('n_adjacent_edges', 0):,}")
        print(f"  沿海边:      {s.get('n_coastal_edges', 0):,}")
        print("=" * 60 + "\n")


# ================================================================
# 便捷函数
# ================================================================

def build_knowledge_graph(
    df: pd.DataFrame,
    coastal_orography_max: float = 100.0,
) -> nx.DiGraph:
    """一行式构建知识图谱。"""
    builder = KnowledgeGraphBuilder(coastal_orography_max=coastal_orography_max)
    return builder.build(df)

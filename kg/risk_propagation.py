"""
风险传播推理：从高风险源格点出发，按灾害物理机制传播风险

四类灾害的传播策略：
1. 暴雨山洪: 沿 flows_to 边向下游传播，衰减系数 α = exp(-β * distance)
2. 沙尘强风: 沿风向扇形传播，覆盖角 ±30°（等向性简化）
3. 极端高温: 不做图传播，汇总同一区域内的暴露度
4. 沿海风浪: 沿岸线传播 + 内陆有限距离增水推理（≤ 30 km）

输出：
- 受影响区域清单（网格/坐标）
- 受影响承灾体计数
- 风险传播路径（用于可视化）

用法:
    from kg.graph_builder import KnowledgeGraphBuilder
    from kg.risk_propagation import RiskPropagator

    G = KnowledgeGraphBuilder().build(df)

    propagator = RiskPropagator(G, disaster_type="flash_flood")
    result = propagator.propagate(high_risk_nodes=[...])
    # result: {"affected_nodes": [...], "risk_scores": {...}, "paths": {...}}
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx


class RiskPropagator:
    """风险传播推理引擎。

    输入高风险源节点列表 + 灾害类型 → 输出受影响区域及风险分数。
    """

    # 灾害类型到传播策略的映射
    DISASTER_STRATEGIES = {
        "flash_flood": "downstream_flow",
        "dust_wind": "wind_fan",
        "extreme_heat": "exposure_summary",
        "coastal_wave": "coastal_inland",
    }

    def __init__(
        self,
        G: nx.DiGraph,
        disaster_type: str,
        # 通用参数
        max_hops: int = 10,
        max_distance_km: float = 200.0,
        # 山洪参数
        flood_attenuation_beta: float = 0.05,  # 距离衰减系数 (per km)
        min_slope_for_propagation: float = 0.001,  # 最小坡度 (m/m)
        # 沙尘参数
        wind_direction_deg: float = 0.0,  # 风向（气象角度：0=N, 90=E）
        wind_fan_angle_deg: float = 30.0,  # 扇形半角
        # 风浪参数
        max_inland_distance_km: float = 30.0,  # 内陆最大影响距离
        # 衰减
        base_risk: float = 1.0,
        min_risk_threshold: float = 0.05,
    ):
        """
        Args:
            G: 知识图谱
            disaster_type: 灾害类型
            max_hops: 传播最大跳数（山洪）
            max_distance_km: 传播最大距离 (km)
            flood_attenuation_beta: 山洪距离衰减系数
            min_slope_for_propagation: 最小坡度阈值（坡度小于此值不传播）
            wind_direction_deg: 沙尘风向（气象角度）
            wind_fan_angle_deg: 沙尘扇形半角
            max_inland_distance_km: 风浪内陆影响距离
            base_risk: 源节点初始风险值
            min_risk_threshold: 风险低于此值停止传播
        """
        self.G = G
        self.disaster_type = disaster_type
        self.strategy = self.DISASTER_STRATEGIES.get(disaster_type, "downstream_flow")

        self.max_hops = max_hops
        self.max_distance_km = max_distance_km
        self.flood_attenuation_beta = flood_attenuation_beta
        self.min_slope_for_propagation = min_slope_for_propagation
        self.wind_direction_deg = wind_direction_deg
        self.wind_fan_angle_deg = wind_fan_angle_deg
        self.max_inland_distance_km = max_inland_distance_km
        self.base_risk = base_risk
        self.min_risk_threshold = min_risk_threshold

        # 结果缓存
        self._last_result: Dict = {}

    # ================================================================
    # propagate: 统一传播入口
    # ================================================================

    def propagate(
        self,
        source_nodes: List[int],
        source_risks: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict:
        """从源节点传播风险。

        Args:
            source_nodes: 高风险源节点 ID 列表
            source_risks: 各源节点的初始风险值（None → base_risk）
            **kwargs: 覆盖初始化参数

        Returns:
            {
                "disaster_type": str,
                "strategy": str,
                "source_nodes": [...],
                "affected_nodes": [...],
                "risk_scores": {node_id: float},  # 0~1
                "paths": {node_id: [source_node, ..., node_id]},
                "n_affected": int,
                "affected_coastal": [...],
                "summary": {...},
            }
        """
        # 参数覆盖
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if source_risks is None:
            source_risks = [self.base_risk] * len(source_nodes)

        print("\n" + "=" * 60)
        print(f"  RiskPropagator — {self._disaster_name()}")
        print(f"  策略: {self.strategy}, 源节点: {len(source_nodes)}")
        print("=" * 60)

        if self.strategy == "downstream_flow":
            result = self._propagate_downstream(source_nodes, source_risks)
        elif self.strategy == "wind_fan":
            result = self._propagate_wind_fan(source_nodes, source_risks)
        elif self.strategy == "exposure_summary":
            result = self._propagate_exposure(source_nodes, source_risks)
        elif self.strategy == "coastal_inland":
            result = self._propagate_coastal_inland(source_nodes, source_risks)
        else:
            raise ValueError(f"未知传播策略: {self.strategy}")

        result["disaster_type"] = self.disaster_type
        result["strategy"] = self.strategy
        result["source_nodes"] = source_nodes

        self._last_result = result
        self._print_result_summary(result)

        return result

    # ================================================================
    # 策略 1: 下游传播（山洪）
    # ================================================================

    def _propagate_downstream(
        self,
        source_nodes: List[int],
        source_risks: List[float],
    ) -> Dict:
        """沿 flows_to 边向下游传播风险。

        衰减模型: risk = source_risk * exp(-β * cumulative_distance) * slope_factor
        """
        risk_scores: Dict[int, float] = {}
        paths: Dict[int, List[int]] = {}
        cumulative_dist: Dict[int, float] = {}

        # 优先级队列：(negative_risk, node_id) — 最高风险优先
        import heapq
        pq = []

        for node, risk in zip(source_nodes, source_risks):
            risk_scores[node] = risk
            paths[node] = [node]
            cumulative_dist[node] = 0.0
            heapq.heappush(pq, (-risk, node))

        visited = set()

        while pq:
            neg_risk, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            current_risk = risk_scores[u]
            if current_risk < self.min_risk_threshold:
                continue

            # 沿 flows_to 边传播
            for v, data in self.G[u].items():
                edge_type = data.get("type", "")
                if edge_type != "flows_to":
                    continue

                dist = data.get("distance_km", 10.0)
                slope = data.get("slope", 0.01)
                new_dist = cumulative_dist[u] + dist

                if new_dist > self.max_distance_km:
                    continue

                # 坡度因子：坡度越大，传播越强
                slope_factor = min(1.0, slope / self.min_slope_for_propagation)
                if slope_factor < 0.1:  # 太平坦 → 不传播
                    continue

                # 衰减计算
                attenuated = current_risk * np.exp(-self.flood_attenuation_beta * dist)
                attenuated *= slope_factor

                if attenuated < self.min_risk_threshold:
                    continue

                if v not in risk_scores or attenuated > risk_scores[v]:
                    risk_scores[v] = attenuated
                    paths[v] = paths[u] + [v]
                    cumulative_dist[v] = new_dist
                    heapq.heappush(pq, (-attenuated, v))

        # 受影响节点（不含源节点本身，或含）
        affected = [n for n in risk_scores if n not in set(source_nodes)]
        affected_coastal = [
            n for n in affected
            if self.G.nodes[n].get("type") == "coast_point"
        ]

        return {
            "risk_scores": risk_scores,
            "paths": {k: v for k, v in paths.items() if k in affected},
            "affected_nodes": affected,
            "n_affected": len(affected),
            "affected_coastal": affected_coastal,
            "summary": {
                "total_affected": len(affected),
                "coastal_affected": len(affected_coastal),
                "max_distance_km": max(cumulative_dist.values()) if cumulative_dist else 0,
                "mean_risk": float(np.mean(list(risk_scores.values()))) if risk_scores else 0,
            },
        }

    # ================================================================
    # 策略 2: 风向扇形传播（沙尘）
    # ================================================================

    def _propagate_wind_fan(
        self,
        source_nodes: List[int],
        source_risks: List[float],
    ) -> Dict:
        """沿风向扇形传播沙尘风险。

        简化模型：
        - 从每个源节点，沿风向 ± fan_angle 扇形区域
        - 风险随距离衰减: risk = source_risk * exp(-dist / 50km)
        - 只考虑下风向格点
        """
        risk_scores: Dict[int, float] = {}
        paths: Dict[int, List[int]] = {}

        wind_rad = np.radians(self.wind_direction_deg)
        fan_cos = np.cos(np.radians(self.wind_fan_angle_deg))

        for source, src_risk in zip(source_nodes, source_risks):
            risk_scores[source] = max(risk_scores.get(source, 0), src_risk)
            paths[source] = [source]

            src_lat = self.G.nodes[source].get("lat", 0)
            src_lon = self.G.nodes[source].get("lon", 0)

            # 遍历所有节点，检查是否在下风向扇形内
            for v in self.G.nodes():
                if v == source:
                    continue

                v_lat = self.G.nodes[v].get("lat", 0)
                v_lon = self.G.nodes[v].get("lon", 0)

                # 计算距离和方向
                dlat = v_lat - src_lat
                dlon = v_lon - src_lon
                dist = np.sqrt((dlat * 111.0) ** 2 + (dlon * 111.0 * np.cos(np.radians(src_lat))) ** 2)

                if dist > self.max_distance_km or dist == 0:
                    continue

                # 风向向量
                wind_dlat = -np.sin(wind_rad)  # 气象角度 → 分量
                wind_dlon = np.cos(wind_rad)

                # 目标方向
                target_dlat = dlat / dist
                target_dlon = dlon / dist

                # 方向余弦（夹角）
                dot = wind_dlat * target_dlat + wind_dlon * target_dlon

                if dot < fan_cos:  # 不在扇形内
                    continue

                # 衰减风险
                attenuated = src_risk * np.exp(-dist / 50.0)
                if attenuated < self.min_risk_threshold:
                    continue

                if v not in risk_scores or attenuated > risk_scores[v]:
                    risk_scores[v] = attenuated
                    paths[v] = [source, v]

        affected = [n for n in risk_scores if n not in set(source_nodes)]

        return {
            "risk_scores": risk_scores,
            "paths": {k: v for k, v in paths.items() if k in affected},
            "affected_nodes": affected,
            "n_affected": len(affected),
            "affected_coastal": [],
            "summary": {
                "total_affected": len(affected),
                "wind_direction": self.wind_direction_deg,
                "fan_angle": self.wind_fan_angle_deg,
            },
        }

    # ================================================================
    # 策略 3: 暴露度汇总（高温）
    # ================================================================

    def _propagate_exposure(
        self,
        source_nodes: List[int],
        source_risks: List[float],
    ) -> Dict:
        """高温不做空间传播，仅汇总暴露度。

        输出：
        - 高风险区域的空间聚类
        - 受影响格点数
        - 各区域的暴露度汇总
        """
        risk_scores = {}
        for node, risk in zip(source_nodes, source_risks):
            risk_scores[node] = risk

        # 计算空间聚类（连通的高风险区域）
        # 简化：按邻接边聚合
        clusters = self._find_spatial_clusters(source_nodes)

        return {
            "risk_scores": risk_scores,
            "paths": {},
            "affected_nodes": list(source_nodes),
            "n_affected": len(source_nodes),
            "affected_coastal": [],
            "summary": {
                "total_affected": len(source_nodes),
                "n_clusters": len(clusters),
                "largest_cluster_size": max(len(c) for c in clusters) if clusters else 0,
                "mean_risk": float(np.mean(source_risks)) if source_risks else 0,
            },
        }

    def _find_spatial_clusters(
        self, nodes: List[int]
    ) -> List[List[int]]:
        """找出节点的空间连通分量（基于 adjacent 边）。"""
        node_set = set(nodes)
        visited = set()
        clusters = []

        for node in nodes:
            if node in visited:
                continue
            cluster = []
            queue = deque([node])
            while queue:
                u = queue.popleft()
                if u in visited:
                    continue
                visited.add(u)
                cluster.append(u)
                for v in self.G.neighbors(u):
                    edge_data = self.G.get_edge_data(u, v)
                    if (edge_data and edge_data.get("type") == "adjacent"
                            and v in node_set and v not in visited):
                        queue.append(v)
            if cluster:
                clusters.append(cluster)

        return clusters

    # ================================================================
    # 策略 4: 沿海内陆传播（风浪）
    # ================================================================

    def _propagate_coastal_inland(
        self,
        source_nodes: List[int],
        source_risks: List[float],
    ) -> Dict:
        """风浪灾害：从沿海格点向内陆有限距离传播。

        传播模型：
        - 沿海源节点 → 内陆方向传播
        - 最大内陆距离: max_inland_distance_km (默认 30 km)
        - 风险衰减: risk = source_risk * (1 - dist / max_inland_distance)
        """
        risk_scores: Dict[int, float] = {}
        paths: Dict[int, List[int]] = {}

        for source, src_risk in zip(source_nodes, source_risks):
            risk_scores[source] = max(risk_scores.get(source, 0), src_risk)
            paths[source] = [source]

            src_lat = self.G.nodes[source].get("lat", 0)
            src_lon = self.G.nodes[source].get("lon", 0)

            # BFS 向内陆传播
            visited = {source}
            queue = deque([(source, 0.0)])  # (node, cumulative_distance)

            while queue:
                u, cum_dist = queue.popleft()
                u_risk = src_risk * max(0, 1 - cum_dist / self.max_inland_distance_km)

                if u_risk < self.min_risk_threshold:
                    continue

                for v in self.G.neighbors(u):
                    if v in visited:
                        continue
                    edge_data = self.G.get_edge_data(u, v)
                    edge_dist = edge_data.get("distance_km", 10.0) if edge_data else 10.0
                    new_dist = cum_dist + edge_dist

                    if new_dist > self.max_inland_distance_km:
                        continue

                    # 确保向内陆传播（远离海岸 → 海拔应升高）
                    v_oro = self.G.nodes[v].get("orography")
                    u_oro = self.G.nodes[u].get("orography")
                    is_inland = (v_oro is None or u_oro is None or v_oro >= u_oro)

                    if is_inland or new_dist < 5.0:  # 5km内不做限制
                        visited.add(v)
                        v_risk = src_risk * max(0, 1 - new_dist / self.max_inland_distance_km)
                        if v not in risk_scores or v_risk > risk_scores[v]:
                            risk_scores[v] = v_risk
                            paths[v] = paths.get(u, [source]) + [v]
                        queue.append((v, new_dist))

        affected = [n for n in risk_scores if n not in set(source_nodes)]

        return {
            "risk_scores": risk_scores,
            "paths": {k: v for k, v in paths.items() if k in affected},
            "affected_nodes": affected,
            "n_affected": len(affected),
            "affected_coastal": [
                n for n in affected
                if self.G.nodes[n].get("type") == "coast_point"
            ],
            "summary": {
                "total_affected": len(affected),
                "max_inland_km": self.max_inland_distance_km,
                "mean_risk": float(np.mean(list(risk_scores.values()))) if risk_scores else 0,
            },
        }

    # ================================================================
    # 便捷方法
    # ================================================================

    def get_affected_dataframe(self) -> pd.DataFrame:
        """返回受影响节点的 DataFrame（含坐标和风险值）。"""
        if not self._last_result:
            raise RuntimeError("请先调用 propagate()")

        records = []
        for node, risk in self._last_result.get("risk_scores", {}).items():
            lat = self.G.nodes[node].get("lat")
            lon = self.G.nodes[node].get("lon")
            oro = self.G.nodes[node].get("orography")
            node_type = self.G.nodes[node].get("type")
            is_source = node in set(self._last_result.get("source_nodes", []))

            records.append({
                "node_id": node,
                "latitude": lat,
                "longitude": lon,
                "orography": oro,
                "node_type": node_type,
                "risk_score": risk,
                "is_source": is_source,
            })

        return pd.DataFrame(records)

    def get_top_affected(self, n: int = 10) -> List[Dict]:
        """返回风险最高的 N 个受影响节点。"""
        if not self._last_result:
            raise RuntimeError("请先调用 propagate()")

        risk_scores = self._last_result.get("risk_scores", {})
        source_nodes = set(self._last_result.get("source_nodes", []))

        # 排除源节点，按风险排序
        affected = [
            (node, risk) for node, risk in risk_scores.items()
            if node not in source_nodes
        ]
        affected.sort(key=lambda x: x[1], reverse=True)

        top = []
        for node, risk in affected[:n]:
            top.append({
                "node_id": node,
                "risk_score": risk,
                "lat": self.G.nodes[node].get("lat"),
                "lon": self.G.nodes[node].get("lon"),
            })
        return top

    # ================================================================
    # 辅助
    # ================================================================

    def _disaster_name(self) -> str:
        """灾害类型中文名。"""
        names = {
            "flash_flood": "暴雨山洪",
            "dust_wind": "沙尘强风",
            "extreme_heat": "极端高温",
            "coastal_wave": "沿海风浪",
        }
        return names.get(self.disaster_type, self.disaster_type)

    def _print_result_summary(self, result: Dict) -> None:
        """打印传播结果摘要。"""
        s = result.get("summary", {})
        print(f"\n  [传播结果]")
        print(f"  受影响节点: {result.get('n_affected', 0):,}")
        if "coastal_affected" in s:
            print(f"  其中沿海:     {s.get('coastal_affected', 0):,}")
        print(f"  总风险节点:   {len(result.get('risk_scores', {})):,}")
        print(f"  平均风险值:   {s.get('mean_risk', 0):.4f}")
        print("=" * 60 + "\n")

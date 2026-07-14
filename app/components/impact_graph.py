"""
KG 影响链可视化组件

将风险传播路径渲染为网络图：
- 源节点（高风险区域）→ 传播路径 → 受影响区域
- 节点颜色按风险等级
- 边颜色/宽度按传播强度
- 支持导出 PNG / SVG / 交互式 HTML

用法:
    from app.components.impact_graph import ImpactGraphVisualizer

    viz = ImpactGraphVisualizer()
    fig = viz.plot(G, propagation_result, disaster_type="flash_flood")
    fig.savefig("outputs/impact_chain.png")
"""

import io
import base64
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


class ImpactGraphVisualizer:
    """知识图谱影响链可视化器。"""

    # 灾害配色
    DISASTER_COLORS = {
        "flash_flood": ("#08306b", "#4292c6", "#de2d26"),
        "extreme_heat": ("#800026", "#fc4e2a", "#ff7f00"),
        "dust_wind": ("#7f2704", "#d94801", "#f16913"),
        "coastal_wave": ("#00441b", "#238b45", "#41ab5d"),
    }

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 100,
        max_nodes_display: int = 200,
        node_size_base: float = 30.0,
    ):
        """
        Args:
            figsize: 图像尺寸
            dpi: 分辨率
            max_nodes_display: 最大显示节点数（避免图过大）
            node_size_base: 节点基础大小
        """
        self.figsize = figsize
        self.dpi = dpi
        self.max_nodes_display = max_nodes_display
        self.node_size_base = node_size_base

    # ================================================================
    # plot: 主渲染方法
    # ================================================================

    def plot(
        self,
        G: nx.DiGraph,
        propagation_result: Dict,
        disaster_type: str = "flash_flood",
        title: Optional[str] = None,
        show_labels: bool = True,
        layout: str = "spatial",  # "spatial" / "spring" / "kamada_kawai"
    ) -> plt.Figure:
        """绘制风险传播影响链图。

        Args:
            G: 知识图谱
            propagation_result: RiskPropagator.propagate() 的输出
            disaster_type: 灾害类型
            title: 标题
            show_labels: 是否显示节点标签
            layout: 布局算法 — "spatial" 按地理坐标 / "spring" 力导向

        Returns:
            matplotlib Figure
        """
        risk_scores = propagation_result.get("risk_scores", {})
        source_nodes = set(propagation_result.get("source_nodes", []))
        paths = propagation_result.get("paths", {})

        if not risk_scores:
            raise ValueError("传播结果中没有 risk_scores")

        # 抽取子图
        all_nodes = set(risk_scores.keys())
        if len(all_nodes) > self.max_nodes_display:
            # 按风险排序，取前 N 个
            sorted_nodes = sorted(
                risk_scores.items(), key=lambda x: x[1], reverse=True
            )
            all_nodes = set(n for n, _ in sorted_nodes[:self.max_nodes_display]) | source_nodes

        subG = G.subgraph(all_nodes).copy()

        # 布局
        if layout == "spatial":
            pos = {
                n: (G.nodes[n].get("lon", 0), G.nodes[n].get("lat", 0))
                for n in subG.nodes()
            }
        elif layout == "spring":
            pos = nx.spring_layout(subG, seed=42, k=0.5, iterations=30)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subG)
        else:
            pos = nx.spring_layout(subG, seed=42)

        # 颜色方案
        src_color, path_color, affected_color = self.DISASTER_COLORS.get(
            disaster_type, ("#08306b", "#4292c6", "#de2d26")
        )

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 节点分类
        src_list = [n for n in subG.nodes() if n in source_nodes]
        paths_list = []
        for path in paths.values():
            paths_list.extend(path[1:-1])  # 路径中间节点
        paths_list = list(set(paths_list) & set(subG.nodes()))
        affected_list = [
            n for n in subG.nodes()
            if n not in source_nodes and n not in paths_list
        ]

        # 绘制边（传播路径）
        for u, v in subG.edges():
            if u in source_nodes and v in source_nodes:
                continue
            edge_alpha = 0.3
            edge_width = 0.5
            edge_style = "solid"
            edge_color = path_color

            if u in source_nodes:
                edge_alpha = 0.8
                edge_width = 1.5
                edge_color = src_color
            elif u in paths_list:
                edge_alpha = 0.5
                edge_width = 1.0
                edge_color = path_color

            nx.draw_networkx_edges(
                subG, pos, ax=ax,
                edgelist=[(u, v)],
                width=edge_width, alpha=edge_alpha,
                edge_color=edge_color, style=edge_style,
                arrows=True, arrowsize=8, arrowstyle="-|>",
            )

        # 绘制节点
        for label, node_list, color, size_mult in [
            ("源节点", src_list, src_color, 3.0),
            ("传播路径", paths_list, path_color, 1.5),
            ("受影响", affected_list, affected_color, 1.0),
        ]:
            if not node_list:
                continue
            node_sizes = [
                self.node_size_base * size_mult * (1 + risk_scores.get(n, 0) * 2)
                for n in node_list
            ]
            nx.draw_networkx_nodes(
                subG, pos, ax=ax,
                nodelist=node_list,
                node_color=color,
                node_size=node_sizes,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )

        # 标签
        if show_labels and len(src_list) <= 20:
            labels = {n: str(n) for n in src_list}
            nx.draw_networkx_labels(
                subG, pos, labels=labels,
                ax=ax, font_size=5, font_color="white", font_weight="bold",
            )

        # 图例
        legend_elements = [
            mpatches.Patch(color=src_color, label=f"高风险源 ({len(src_list)})"),
            mpatches.Patch(color=path_color, label=f"传播路径 ({len(paths_list)})"),
            mpatches.Patch(color=affected_color, label=f"受影响区域 ({len(affected_list)})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        # 标注
        if layout == "spatial":
            ax.set_xlabel("经度 (°E)")
            ax.set_ylabel("纬度 (°N)")
        else:
            ax.axis("off")

        if title is None:
            from config.disaster_config import get_label_config
            name = get_label_config(disaster_type)["name_cn"]
            title = f"{name} 风险传播影响链"

        ax.set_title(title, fontsize=14, fontweight="bold")

        fig.tight_layout()
        return fig

    # ================================================================
    # 导出
    # ================================================================

    def to_base64(self, fig: Optional[plt.Figure] = None) -> str:
        """Figure → base64 PNG。"""
        if fig is None:
            fig = plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

    def save(self, fig: plt.Figure, filepath: str) -> None:
        """保存为 PNG 文件。"""
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        print(f"  影响链图已保存: {filepath}")

    # ================================================================
    # 受影响资产表
    # ================================================================

    @staticmethod
    def build_asset_table(
        G: nx.DiGraph,
        propagation_result: Dict,
        top_n: int = 20,
    ) -> "pd.DataFrame":
        """从传播结果生成受影响资产清单 DataFrame。

        Args:
            G: 知识图谱
            propagation_result: RiskPropagator 输出
            top_n: 返回前 N 条

        Returns:
            DataFrame: columns=[node_id, lat, lon, risk_score, type, is_coastal]
        """
        import pandas as pd

        risk_scores = propagation_result.get("risk_scores", {})
        source_nodes = set(propagation_result.get("source_nodes", []))

        records = []
        for node, risk in risk_scores.items():
            if node in source_nodes:
                continue
            records.append({
                "node_id": node,
                "latitude": G.nodes[node].get("lat"),
                "longitude": G.nodes[node].get("lon"),
                "orography": G.nodes[node].get("orography"),
                "risk_score": round(risk, 4),
                "is_coastal": G.nodes[node].get("type") == "coast_point",
            })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values("risk_score", ascending=False).head(top_n)
        return df

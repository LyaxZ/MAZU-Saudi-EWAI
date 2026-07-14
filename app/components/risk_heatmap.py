"""
风险热力图组件

基于 matplotlib 生成沙特阿拉伯风险热力图，支持：
- 按灾害类型着色
- 风险等级分层显示
- 叠加海岸线和城市标记
- 导出 PNG / 交互式 HTML（plotly）

用法:
    from app.components.risk_heatmap import RiskHeatmap

    heatmap = RiskHeatmap()
    fig = heatmap.plot(df_risk, disaster_type="flash_flood", title="山洪风险")
    fig.savefig("outputs/heatmap.png")
"""

import io
import base64
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch


class RiskHeatmap:
    """风险热力图渲染器。"""

    # 灾害类型配色方案
    DISASTER_COLORS = {
        "flash_flood": ["#f7fbff", "#08306b"],   # 蓝 → 深蓝
        "extreme_heat": ["#ffffcc", "#800026"],   # 黄 → 深红
        "dust_wind": ["#fff5eb", "#7f2704"],      # 浅橙 → 深棕
        "coastal_wave": ["#f0f9e8", "#00441b"],   # 浅绿 → 深绿
    }

    DISASTER_NAMES = {
        "flash_flood": "暴雨山洪",
        "extreme_heat": "极端高温",
        "dust_wind": "沙尘强风",
        "coastal_wave": "沿海风浪",
    }

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        cmap: Optional[str] = None,
        lat_range: Tuple[float, float] = (16.0, 32.0),
        lon_range: Tuple[float, float] = (34.0, 56.0),
    ):
        """
        Args:
            figsize: 图像尺寸 (width, height)
            dpi: 分辨率
            cmap: 自定义 colormap，None 则按灾害类型自动选择
            lat_range: 纬度范围
            lon_range: 经度范围
        """
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.lat_range = lat_range
        self.lon_range = lon_range

    # ================================================================
    # plot: 主渲染方法
    # ================================================================

    def plot(
        self,
        df: pd.DataFrame,
        risk_col: str = "risk_score",
        disaster_type: str = "flash_flood",
        title: Optional[str] = None,
        highlight_nodes: Optional[List[Dict]] = None,
        show_colorbar: bool = True,
        alpha: float = 0.8,
        marker_size: float = 2.0,
    ) -> plt.Figure:
        """绘制风险热力图。

        Args:
            df: 含 latitude, longitude, risk_col 的 DataFrame
            risk_col: 风险值列名
            disaster_type: 灾害类型（决定配色）
            title: 图表标题
            highlight_nodes: 高亮节点列表 [{"lat":, "lon":, "label":}]
            show_colorbar: 是否显示颜色条
            alpha: 透明度
            marker_size: 散点大小

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # 配色
        if self.cmap:
            cmap = plt.cm.get_cmap(self.cmap)
        else:
            low_color, high_color = self.DISASTER_COLORS.get(
                disaster_type, ["#f7fbff", "#08306b"]
            )
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "risk", [low_color, high_color]
            )

        # 绘制风险散点
        if risk_col in df.columns and len(df) > 0:
            risk_vals = df[risk_col].values
            vmin = np.nanpercentile(risk_vals, 5) if len(risk_vals) > 0 else 0
            vmax = np.nanpercentile(risk_vals, 95) if len(risk_vals) > 0 else 1
            if vmax == vmin:
                vmax = vmin + 1

            sc = ax.scatter(
                df["longitude"], df["latitude"],
                c=risk_vals, cmap=cmap,
                s=marker_size, alpha=alpha,
                vmin=vmin, vmax=vmax,
                edgecolors="none", linewidth=0,
            )

            if show_colorbar:
                cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label("风险值", fontsize=10)

        # 高亮节点
        if highlight_nodes:
            for node in highlight_nodes:
                ax.scatter(
                    node["lon"], node["lat"],
                    c="red", s=30, marker="*",
                    edgecolors="white", linewidth=0.5,
                    zorder=5,
                )
                if "label" in node:
                    ax.annotate(
                        node["label"],
                        (node["lon"], node["lat"]),
                        fontsize=7, color="red",
                        xytext=(5, 5), textcoords="offset points",
                    )

        # 标注
        ax.set_xlim(self.lon_range)
        ax.set_ylim(self.lat_range)
        ax.set_xlabel("经度 (°E)", fontsize=11)
        ax.set_ylabel("纬度 (°N)", fontsize=11)

        name = self.DISASTER_NAMES.get(disaster_type, disaster_type)
        if title is None:
            title = f"沙特阿拉伯 {name} 风险热力图"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # 网格
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # 地名标注（关键城市）
        self._add_city_labels(ax)

        fig.tight_layout()
        return fig

    # ================================================================
    # 多灾害对比图
    # ================================================================

    def plot_multi_disaster(
        self,
        data_dict: Dict[str, pd.DataFrame],
        risk_col: str = "risk_score",
        title: str = "四类灾害风险对比",
    ) -> plt.Figure:
        """绘制四类灾害的 2×2 对比图。

        Args:
            data_dict: {disaster_type: DataFrame}
            risk_col: 风险列名
            title: 总标题

        Returns:
            matplotlib Figure
        """
        disasters = list(data_dict.keys())
        n = len(disasters)
        if n == 0:
            raise ValueError("data_dict 为空")

        cols = min(2, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), dpi=self.dpi)
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, dtype in enumerate(disasters):
            ax = axes[idx]
            df = data_dict[dtype]

            low_color, high_color = self.DISASTER_COLORS.get(
                dtype, ["#f7fbff", "#08306b"]
            )
            cmap = mcolors.LinearSegmentedColormap.from_list(
                dtype, [low_color, high_color]
            )

            if risk_col in df.columns and len(df) > 0:
                risk_vals = df[risk_col].values
                vmin = np.nanpercentile(risk_vals, 5)
                vmax = np.nanpercentile(risk_vals, 95)
                if vmax == vmin:
                    vmax = vmin + 1

                sc = ax.scatter(
                    df["longitude"], df["latitude"],
                    c=risk_vals, cmap=cmap, s=1.5, alpha=0.8,
                    vmin=vmin, vmax=vmax, edgecolors="none",
                )
                plt.colorbar(sc, ax=ax, shrink=0.8)

            name = self.DISASTER_NAMES.get(dtype, dtype)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlim(self.lon_range)
            ax.set_ylim(self.lat_range)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # 隐藏多余 subplot
        for idx in range(len(disasters), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig

    # ================================================================
    # 导出
    # ================================================================

    def to_base64(self, fig: Optional[plt.Figure] = None) -> str:
        """将 Figure 转为 base64 字符串（用于 HTML/Gradio 嵌入）。"""
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
        print(f"  热力图已保存: {filepath}")

    # ================================================================
    # 辅助
    # ================================================================

    def _add_city_labels(self, ax: plt.Axes) -> None:
        """添加沙特主要城市标注。"""
        cities = {
            "Riyadh": (24.71, 46.67),
            "Jeddah": (21.54, 39.17),
            "Mecca": (21.39, 39.86),
            "Medina": (24.47, 39.61),
            "Dammam": (26.43, 50.10),
            "Tabuk": (28.40, 36.57),
            "Abha": (18.22, 42.51),
        }
        for name, (lat, lon) in cities.items():
            if self.lat_range[0] <= lat <= self.lat_range[1] and \
               self.lon_range[0] <= lon <= self.lon_range[1]:
                ax.annotate(
                    name, (lon, lat),
                    fontsize=7, color="gray", alpha=0.8,
                    xytext=(3, 3), textcoords="offset points",
                )


# ================================================================
# 快捷函数
# ================================================================

def plot_risk_heatmap(
    df: pd.DataFrame,
    risk_col: str = "risk_score",
    disaster_type: str = "flash_flood",
    title: Optional[str] = None,
) -> plt.Figure:
    """一行式生成风险热力图。"""
    heatmap = RiskHeatmap()
    return heatmap.plot(df, risk_col=risk_col, disaster_type=disaster_type, title=title)

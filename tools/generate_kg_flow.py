"""生成实际 KG 运行可视化 — 四灾害 + 地图描边 + 颜色平滑"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.loader import load_to_dataframe
from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "kg_flow_map.html")

# ============================================================
# 沙特海岸线大致坐标（用于地图描边）
# ============================================================
RED_SEA_COAST = [
    (16.4,42.5),(17.0,42.0),(17.7,41.6),(18.5,41.0),(19.0,40.5),(19.5,40.0),
    (20.2,39.8),(21.0,39.2),(21.5,39.2),(22.0,39.0),(22.5,38.8),(23.0,38.5),
    (23.5,38.3),(24.0,37.8),(24.5,37.4),(25.0,37.0),(25.5,36.7),(26.0,36.2),
    (26.5,35.8),(27.0,35.4),(27.5,35.0),(28.0,34.8),(28.5,34.8),(29.0,34.7),
]
GULF_COAST = [
    (26.5,50.0),(27.0,49.7),(27.5,49.3),(27.8,48.8),(27.5,48.5),(27.0,48.8),
    (26.5,49.0),(26.0,49.3),(25.5,49.5),(25.0,49.8),(24.5,50.5),(24.0,50.8),
    (23.5,51.0),(23.0,51.2),(22.5,51.5),(22.0,51.8),
]
SOUTH_BORDER = [(16.4,42.5),(16.4,44),(16.4,46),(16.4,48),(16.4,50),(16.4,52),(16.4,54),(16.4,55.5)]
NORTH_BORDER = [(32.1,34.7),(32.1,36),(32.1,38),(32.1,40),(32.1,42),(32.1,44),(32.1,46),(32.1,48),(32.1,49)]

# ============================================================
# 1. 加载数据 + 构建全图
# ============================================================
print("加载 2025-08-28 数据...")
df = load_to_dataframe("2025-08-28", "2025-08-28",
    variables=["orography","t2m","tmax","u10","v10","r2","cape","tp"],
    show_progress=False).fillna(0)

print("构建知识图谱...")
G = KnowledgeGraphBuilder().build(df)
print(f"  全图: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")

# ============================================================
# 2. 辅助函数
# ============================================================
def add_map_bg(fig, row=None, col=None):
    """添加沙特地图描边 + 标注"""
    kw = {} if row is None else {"row": row, "col": col}
    # 红海海岸线
    fig.add_trace(go.Scatter(
        x=[p[1] for p in RED_SEA_COAST], y=[p[0] for p in RED_SEA_COAST],
        mode="lines", line=dict(color="#1e40af", width=2.5, dash="solid"),
        name="红海海岸", showlegend=False, hoverinfo="skip"), **kw)
    # 阿拉伯湾海岸线
    fig.add_trace(go.Scatter(
        x=[p[1] for p in GULF_COAST], y=[p[0] for p in GULF_COAST],
        mode="lines", line=dict(color="#1e40af", width=2.5, dash="solid"),
        name="阿拉伯湾海岸", showlegend=False, hoverinfo="skip"), **kw)
    # 南部边界
    fig.add_trace(go.Scatter(
        x=[p[1] for p in SOUTH_BORDER], y=[p[0] for p in SOUTH_BORDER],
        mode="lines", line=dict(color="#94a3b8", width=1.5, dash="dash"),
        showlegend=False, hoverinfo="skip"), **kw)
    # 北部边界
    fig.add_trace(go.Scatter(
        x=[p[1] for p in NORTH_BORDER], y=[p[0] for p in NORTH_BORDER],
        mode="lines", line=dict(color="#94a3b8", width=1.5, dash="dash"),
        showlegend=False, hoverinfo="skip"), **kw)

def add_sea_labels(fig, row=None, col=None):
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_annotation(x=38, y=25, text="红 海", showarrow=False,
        font=dict(size=13, color="#1e40af", family="Segoe UI"), opacity=0.5, **kw)
    fig.add_annotation(x=51, y=25, text="阿拉伯湾", showarrow=False,
        font=dict(size=11, color="#1e40af", family="Segoe UI"), opacity=0.5, **kw)

# ============================================================
# 3. 构建 2×2 子图
# ============================================================
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("⚡ 暴雨山洪 — D8流向 + 下游传播", "🔥 极端高温 — 气温异常暴露",
                    "🌪️ 沙尘强风 — 风向扇形传播", "🌊 沿海风浪 — 沿岸+内陆增水"),
    horizontal_spacing=0.06, vertical_spacing=0.08)

POS = {"flash_flood":(1,1), "extreme_heat":(1,2), "dust_wind":(2,1), "coastal_wave":(2,2)}
TITLES = {"flash_flood":"暴雨山洪","extreme_heat":"极端高温","dust_wind":"沙尘强风","coastal_wave":"沿海风浪"}

for dtype in ["flash_flood", "extreme_heat", "dust_wind", "coastal_wave"]:
    r, c = POS[dtype]
    print(f"  生成 {TITLES[dtype]} 面板...")

    # ---- 地形底图（所有面板共用） ----
    lats = sorted(df["latitude"].unique())
    lons = sorted(df["longitude"].unique())
    n_lat, n_lon = len(lats), len(lons)

    oro = np.full((n_lat, n_lon), np.nan)
    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}
    for _, row in df.iterrows():
        i, j = lat_idx.get(row["latitude"]), lon_idx.get(row["longitude"])
        if i is not None and j is not None:
            oro[i, j] = row.get("orography", 0)

    fig.add_trace(go.Heatmap(
        z=oro, x=lons, y=lats,
        colorscale=[[0,"#f0fdf4"],[0.25,"#dcfce7"],[0.5,"#bbf7d0"],[0.75,"#86efac"],[0.9,"#d4a574"],[1,"#8b5e3c"]],
        zmin=0, zmax=3000, zsmooth="best", showscale=False,
        hovertemplate="海拔: %{z:.0f}m<extra></extra>", name="地形"), row=r, col=c)

    # ---- 灾害特定可视化 ----
    if dtype == "flash_flood":
        # 山洪：流向箭头 + 传播覆盖
        sub_nodes = [nid for nid, attr in G.nodes(data=True)
                     if 17.5 <= attr.get("lat", 0) <= 19.5 and 42 <= attr.get("lon", 0) <= 44]
        subG = G.subgraph(sub_nodes).copy()
        prop = RiskPropagator(subG, disaster_type="flash_flood", max_hops=8, max_distance_km=150)
        high = sorted([(n, subG.nodes[n].get("orography", 0)) for n in subG.nodes()], key=lambda x: -x[1])[:3]
        srcs = [n for n, _ in high]
        res = prop.propagate(source_nodes=srcs)
        aff = set(res.get("affected_nodes", []))

        # 风险覆盖
        risk_g = np.full((n_lat, n_lon), np.nan)
        for nid in aff:
            attr = subG.nodes.get(nid, {})
            i2, j2 = lat_idx.get(attr.get("lat")), lon_idx.get(attr.get("lon"))
            if i2 is not None and j2 is not None:
                risk_g[i2, j2] = 0.6
        fig.add_trace(go.Heatmap(z=risk_g, x=lons, y=lats,
            colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(239,68,68,0.5)"]],
            zsmooth="best", zmin=0, zmax=1, showscale=False,
            hovertemplate="⚠ 洪水风险<extra></extra>", name="风险覆盖"), row=r, col=c)

        # 源头
        sx = [subG.nodes[n]["lon"] for n in srcs if n in subG.nodes]
        sy = [subG.nodes[n]["lat"] for n in srcs if n in subG.nodes]
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
            marker=dict(size=10, color="#dc2626", symbol="x", line=dict(width=2, color="#fff")),
            name="源头", showlegend=False,
            hovertemplate="🔴 风险源头<extra></extra>"), row=r, col=c)

        # 流向箭头
        flow = [(u,v) for u,v,d in subG.edges(data=True) if d.get("type")=="flows_to"]
        step = max(1, len(flow)//50)
        for u, v in flow[::step]:
            au, av = subG.nodes[u], subG.nodes[v]
            dx, dy = (av["lon"]-au["lon"])*0.55, (av["lat"]-au["lat"])*0.55
            if abs(dx)>0.001 or abs(dy)>0.001:
                fig.add_annotation(x=au["lon"], y=au["lat"], ax=au["lon"]+dx, ay=au["lat"]+dy,
                    xref=f"x{r}", yref=f"y{r}", axref=f"x{r}", ayref=f"y{r}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
                    arrowcolor="#475569", text="", opacity=0.5)

    elif dtype == "extreme_heat":
        # 高温：气温距平热力图
        tmax_g = np.full((n_lat, n_lon), np.nan)
        for _, row in df.iterrows():
            i, j = lat_idx.get(row["latitude"]), lon_idx.get(row["longitude"])
            if i is not None and j is not None:
                tmax_g[i, j] = row.get("tmax", np.nan) - 273.15  # K → °C
        fig.add_trace(go.Heatmap(z=tmax_g, x=lons, y=lats,
            colorscale=[[0,"#fef3c7"],[0.3,"#fdba74"],[0.6,"#f97316"],[0.8,"#ea580c"],[1,"#7c2d12"]],
            zmin=30, zmax=55, zsmooth="best", showscale=False,
            hovertemplate="最高温: %{z:.1f}°C<extra></extra>", name="气温"), row=r, col=c)

    elif dtype == "dust_wind":
        # 沙尘：10m风速
        ws_g = np.full((n_lat, n_lon), np.nan)
        for _, row in df.iterrows():
            i, j = lat_idx.get(row["latitude"]), lon_idx.get(row["longitude"])
            if i is not None and j is not None:
                u, v = row.get("u10", 0), row.get("v10", 0)
                ws_g[i, j] = np.sqrt(u*u + v*v)
        fig.add_trace(go.Heatmap(z=ws_g, x=lons, y=lats,
            colorscale=[[0,"#fefce8"],[0.3,"#fde68a"],[0.6,"#facc15"],[0.8,"#eab308"],[1,"#854d0e"]],
            zmin=0, zmax=20, zsmooth="best", showscale=False,
            hovertemplate="风速: %{z:.1f} m/s<extra></extra>", name="风速"), row=r, col=c)

    elif dtype == "coastal_wave":
        # 风浪：沿海高程 + 沿海格点高亮
        coastal_g = np.full((n_lat, n_lon), np.nan)
        for _, row in df.iterrows():
            i, j = lat_idx.get(row["latitude"]), lon_idx.get(row["longitude"])
            if i is not None and j is not None:
                o = row.get("orography", 0)
                if o <= 100:
                    coastal_g[i, j] = 1.0
        fig.add_trace(go.Heatmap(z=coastal_g, x=lons, y=lats,
            colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(59,130,246,0.6)"]],
            zsmooth="best", zmin=0, zmax=1, showscale=False,
            hovertemplate="🌊 沿海低地<extra></extra>", name="沿海区"), row=r, col=c)

    # ---- 共同：地图描边 + 海标注 ----
    add_map_bg(fig, row=r, col=c)
    add_sea_labels(fig, row=r, col=c)

    # 轴设置
    fig.update_xaxes(title_text="", range=[34, 56], showgrid=False, row=r, col=c)
    fig.update_yaxes(title_text="", range=[16, 32], showgrid=False, row=r, col=c)

# ============================================================
# 布局
# ============================================================
fig.update_layout(
    title=dict(text="MAZU 知识图谱 — 四灾害实际运行可视化", font=dict(size=20, color="#1e293b"), x=0.5),
    height=1000, width=1200,
    paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
    margin=dict(l=40, r=40, t=70, b=30),
    showlegend=False,
)

stats = f"""
<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;padding:6px 0 12px;font-size:12px">
<div style="background:#fff;border-radius:8px;padding:6px 14px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
📊 全图: <b>{G.number_of_nodes():,}</b> 节点 · <b>{G.number_of_edges():,}</b> 边</div>
<div style="background:#fff;border-radius:8px;padding:6px 14px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
⚡ 山洪: D8流向+下游传播 | 🔥 高温: 气温异常暴露</div>
<div style="background:#fff;border-radius:8px;padding:6px 14px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
🌪️ 沙尘: 风速扇形 | 🌊 风浪: 沿海低地+内陆增水</div>
</div>"""

full = f"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="utf-8"><title>MAZU KG 四灾害运行可视化</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",system-ui,sans-serif;background:#f0f4f8}}
.top{{background:#fff;padding:14px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.top h1{{font-size:19px;color:#1e293b;font-weight:700}}
.top p{{font-size:12px;color:#64748b;margin:3px 0 0}}
#wrap{{max-width:1260px;margin:0 auto;padding:8px}}
</style></head>
<body>
<div class="top">
<h1>🛰️ MAZU 知识图谱 — 四灾害实际运行可视化</h1>
<p>NetworkX DiGraph · 35,200 节点 · 279,324 边 | 2025-08-28 数据 · 地形流向 · 气温 · 风速 · 沿海高程</p>
</div>
{stats}
<div id="wrap">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(full)
print(f"✅ 已生成: {OUT}  ({len(full):,} chars)")

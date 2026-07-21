"""生成实际运行的 KG 可视化 HTML — 地形流向图 + 风险传播演示"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from data.loader import load_to_dataframe
from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "kg_flow_map.html")

# ============================================================
# 1. 加载数据 + 构建 KG
# ============================================================
print("加载 2025-08-28 数据...")
df = load_to_dataframe("2025-08-28", "2025-08-28", variables=["orography"], show_progress=False).fillna(0)

print("构建知识图谱...")
builder = KnowledgeGraphBuilder()
G = builder.build(df)
print(f"  全图: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")

# ============================================================
# 2. 采样 Asir 山地子区域（lat 17.5-19.5, lon 42-44）
# ============================================================
sub_nodes = []
for nid, attr in G.nodes(data=True):
    lat, lon = attr.get("lat", 0), attr.get("lon", 0)
    if 17.5 <= lat <= 19.5 and 42 <= lon <= 44:
        sub_nodes.append(nid)

subG = G.subgraph(sub_nodes).copy()
print(f"  子图 (Asir山区): {subG.number_of_nodes()} 节点, {subG.number_of_edges()} 边")

# ============================================================
# 3. 风险传播演示
# ============================================================
propagator = RiskPropagator(subG, disaster_type="flash_flood", max_hops=8, max_distance_km=150)
# 选海拔最高点作为源头
high_nodes = sorted([(nid, attr.get("orography", 0)) for nid, attr in subG.nodes(data=True)],
                     key=lambda x: -x[1])[:3]
source_nodes = [nid for nid, _ in high_nodes]
result = propagator.propagate(source_nodes=source_nodes)
affected = set(result.get("affected_nodes", []))
print(f"  风险传播: {len(source_nodes)} 源头 → {len(affected)} 受影响节点")

# ============================================================
# 4. 构建 plotly 可视化
# ============================================================
# 提取网格数据
lats = sorted(set(attr["lat"] for _, attr in subG.nodes(data=True)))
lons = sorted(set(attr["lon"] for _, attr in subG.nodes(data=True)))
n_lat, n_lon = len(lats), len(lons)

# 地形网格
oro_grid = np.full((n_lat, n_lon), np.nan)
risk_grid = np.full((n_lat, n_lon), np.nan)
lat_to_i = {lat: i for i, lat in enumerate(lats)}
lon_to_j = {lon: j for j, lon in enumerate(lons)}

for nid, attr in subG.nodes(data=True):
    i, j = lat_to_i.get(attr["lat"]), lon_to_j.get(attr["lon"])
    if i is not None and j is not None:
        oro_grid[i, j] = attr.get("orography", 0)

for nid in affected:
    attr = subG.nodes.get(nid)
    if attr:
        i, j = lat_to_i.get(attr["lat"]), lon_to_j.get(attr["lon"])
        if i is not None and j is not None:
            risk_grid[i, j] = 1.0

# 流向箭头
arrow_x, arrow_y, arrow_dx, arrow_dy = [], [], [], []
flow_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("type") == "flows_to"]
step = max(1, len(flow_edges) // 60)
for u, v in flow_edges[::step]:
    au, av = subG.nodes[u], subG.nodes[v]
    x0, y0 = au["lon"], au["lat"]
    x1, y1 = av["lon"], av["lat"]
    dx, dy = (x1 - x0) * 0.6, (y1 - y0) * 0.6
    if abs(dx) > 0.001 or abs(dy) > 0.001:
        arrow_x.append(x0); arrow_y.append(y0)
        arrow_dx.append(dx); arrow_dy.append(dy)

# 源头标记
src_x = [subG.nodes[n]["lon"] for n in source_nodes if n in subG.nodes]
src_y = [subG.nodes[n]["lat"] for n in source_nodes if n in subG.nodes]

fig = go.Figure()

# 地形热力图
fig.add_trace(go.Heatmap(
    z=oro_grid, x=lons, y=lats,
    colorscale=[[0,"#dcfce7"],[.3,"#86efac"],[.6,"#22c55e"],[.8,"#854d0e"],[1,"#451a03"]],
    zmin=0, zmax=3000, colorbar=dict(title="海拔 (m)", thickness=16, len=0.7),
    hovertemplate="经度: %{x:.2f}°E<br>纬度: %{y:.2f}°N<br>海拔: %{z:.0f}m<extra></extra>",
    name="地形"))

# 风险传播覆盖
risk_display = np.where(np.isfinite(risk_grid), risk_grid, np.nan)
if np.any(np.isfinite(risk_display)):
    fig.add_trace(go.Heatmap(
        z=risk_display, x=lons, y=lats,
        colorscale=[[0,"rgba(239,68,68,0)"],[1,"rgba(239,68,68,0.45)"]],
        zmin=0, zmax=1, showscale=False,
        hovertemplate="⚠ 风险传播区域<extra></extra>",
        name="风险覆盖"))

# 流向箭头
if arrow_x:
    fig.add_trace(go.Scatter(x=arrow_x, y=arrow_y, mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)"), showlegend=False,
        hoverinfo="skip"))
    for ax, ay, adx, ady in zip(arrow_x, arrow_y, arrow_dx, arrow_dy):
        fig.add_annotation(x=ax, y=ay, ax=ax+adx, ay=ay+ady,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#475569",
            text="", opacity=0.6)

# 源头标记
if src_x:
    fig.add_trace(go.Scatter(x=src_x, y=src_y, mode="markers",
        marker=dict(size=12, color="#ef4444", symbol="x", line=dict(width=2, color="#fff")),
        name="风险源头", hovertemplate="🔴 风险源头<br>经度: %{x:.2f}°E<br>纬度: %{y:.2f}°N<extra></extra>"))

fig.update_layout(
    title=dict(text="MAZU KG 实际运行 — 地形流向 & 山洪风险传播 (Asir山区)", font=dict(size=16, color="#1e293b"), x=0.5),
    xaxis=dict(title="经度 (°E)", range=[min(lons), max(lons)], showgrid=False),
    yaxis=dict(title="纬度 (°N)", range=[min(lats), max(lats)], showgrid=False, scaleanchor="x"),
    height=700, margin=dict(l=50, r=30, t=60, b=50),
    paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ============================================================
# 5. 统计面板
# ============================================================
stats_html = f"""
<div style="display:flex;gap:16px;justify-content:center;flex-wrap:wrap;padding:8px 0;font-size:13px">
<div style="background:#fff;border-radius:10px;padding:10px 18px;box-shadow:0 1px 3px rgba(0,0,0,.06)">
  📊 全图: <b>{G.number_of_nodes():,}</b> 节点 · <b>{G.number_of_edges():,}</b> 边</div>
<div style="background:#fff;border-radius:10px;padding:10px 18px;box-shadow:0 1px 3px rgba(0,0,0,.06)">
  🔍 Asir子图: <b>{subG.number_of_nodes()}</b> 节点 · <b>{subG.number_of_edges()}</b> 边</div>
<div style="background:#fff;border-radius:10px;padding:10px 18px;box-shadow:0 1px 3px rgba(0,0,0,.06)">
  🌊 流向边: <b>{sum(1 for _,_,d in subG.edges(data=True) if d.get('type')=='flows_to'):,}</b></div>
<div style="background:#fff;border-radius:10px;padding:10px 18px;box-shadow:0 1px 3px rgba(0,0,0,.06)">
  ⚠ 风险传播: <b>{len(source_nodes)}</b> 源头 → <b>{len(affected):,}</b> 受影响</div>
</div>
"""

full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>MAZU KG 实际运行可视化</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",system-ui,sans-serif;background:#f0f4f8}}
.top{{background:#fff;padding:16px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.top h1{{font-size:20px;color:#1e293b;font-weight:700}}
.top p{{font-size:12px;color:#64748b;margin:4px 0 0}}
#chart{{max-width:1100px;margin:0 auto;padding:12px}}
</style></head>
<body>
<div class="top">
<h1>🛰️ MAZU 知识图谱 — 实际运行可视化</h1>
<p>NetworkX DiGraph · 地形流向 (flows_to) · 山洪风险传播 (RiskPropagator) | Asir 山区子图</p>
</div>
{stats_html}
<div id="chart">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"✅ 已生成: {OUT}  ({len(full_html):,} chars)")

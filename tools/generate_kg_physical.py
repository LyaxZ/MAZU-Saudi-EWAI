"""生成物理层 KG 流向可视化 HTML"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, plotly.graph_objects as go
from data.loader import load_single_day, load_to_dataframe
from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator

DATE = "2025-08-28"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "kg_physical_flow.html")

ds = load_single_day(DATE)
lats = ds["latitude"].values; lons = ds["longitude"].values
n_lat, n_lon = len(lats), len(lons)
oro = ds["orography"].values.squeeze()

print("构建 KG...")
df = load_to_dataframe(DATE, DATE, variables=["orography"], show_progress=False).fillna(0)
G = KnowledgeGraphBuilder().build(df)

# Asir 子图
sns = [nid for nid, a in G.nodes(data=True) if 17.5 <= a.get("lat", 0) <= 20.0 and 41.5 <= a.get("lon", 0) <= 44.5]
sG = G.subgraph(sns).copy()
prop = RiskPropagator(sG, disaster_type="flash_flood", max_hops=10, max_distance_km=200)
high = sorted([(n, sG.nodes[n].get("orography", 0)) for n in sG.nodes()], key=lambda x: -x[1])[:3]
srcs = [n for n, _ in high]
res = prop.propagate(source_nodes=srcs)
aff = set(res.get("affected_nodes", []))

# 风险覆盖
risk = np.full((n_lat, n_lon), np.nan)
for nid in sG.nodes():
    a = sG.nodes[nid]
    i = int(round((a["lat"] - lats[0]) / (lats[1] - lats[0])))
    j = int(round((a["lon"] - lons[0]) / (lons[1] - lons[0])))
    if 0 <= i < n_lat and 0 <= j < n_lon and nid in aff:
        risk[i, j] = 0.5

fig = go.Figure()

# 风险覆盖底层
fig.add_trace(go.Heatmap(z=risk, x=lons, y=lats, zsmooth="best",
    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(239,68,68,0.35)"]],
    zmin=0, zmax=1, showscale=False, hoverinfo="skip"))

# 地形上层
fig.add_trace(go.Heatmap(z=oro, x=lons, y=lats, zsmooth="best",
    colorscale=[[0, "#f0fdf4"], [.2, "#dcfce7"], [.4, "#bbf7d0"], [.6, "#86efac"],
                [.8, "#ca8a04"], [.95, "#78350f"], [1, "#451a03"]],
    zmin=0, zmax=3000,
    colorbar=dict(title="海拔 (m)", thickness=18, len=0.75),
    hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>海拔:%{z:.0f}m<extra></extra>"))

# 源头
sx = [sG.nodes[n]["lon"] for n in srcs if n in sG.nodes]
sy = [sG.nodes[n]["lat"] for n in srcs if n in sG.nodes]
fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
    marker=dict(size=12, color="#dc2626", symbol="x-thin", line=dict(width=2.5, color="#fff")),
    hovertemplate="🔴 风险源头<br>海拔:%{text}m<extra></extra>",
    text=[f"{sG.nodes[n].get('orography', 0):.0f}" for n in srcs if n in sG.nodes],
    name="风险源头"))

# D8 流向箭头
flow = [(u, v) for u, v, d in sG.edges(data=True) if d.get("type") == "flows_to"]
seen = set(); uniq = []
for u, v in sorted(flow, key=lambda x: abs(sG.nodes[x[0]]["orography"] - sG.nodes[x[1]]["orography"]), reverse=True):
    if u not in seen: seen.add(u); uniq.append((u, v))
step = max(1, len(uniq) // 40)
for u, v in uniq[::step]:
    au, av = sG.nodes[u], sG.nodes[v]
    dx, dy = (av["lon"] - au["lon"]) * .55, (av["lat"] - au["lat"]) * .55
    if abs(dx) > .0005 or abs(dy) > .0005:
        fig.add_annotation(x=au["lon"], y=au["lat"], ax=au["lon"] + dx, ay=au["lat"] + dy,
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#475569", text="", opacity=.5)

fig.update_layout(
    title=dict(text="MAZU KG 物理层 — D8 地形流向 & 山洪风险传播 (Asir 山区)", font=dict(size=18, color="#1e293b"), x=.5),
    xaxis=dict(range=[34, 56], title="经度 (°E)", showgrid=False),
    yaxis=dict(range=[16, 32], title="纬度 (°N)", showgrid=False),
    height=700, margin=dict(l=50, r=30, t=60, b=50),
    paper_bgcolor="#fff", plot_bgcolor="#fff", dragmode="pan")

html = fig.to_html(full_html=False, include_plotlyjs="cdn")
FULL = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>MAZU KG 物理层可视化</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",system-ui,sans-serif;background:#fff}}
.top{{text-align:center;padding:14px;border-bottom:1px solid #f1f5f9}}
.top h1{{font-size:19px;color:#1e293b;font-weight:700}}
.top p{{font-size:12px;color:#64748b;margin:4px 0 0}}
.stats{{display:flex;gap:8px;justify-content:center;padding:4px 0 6px;font-size:11px;color:#64748b}}
.stats span{{background:#f8fafc;border-radius:6px;padding:2px 10px}}
</style></head><body>
<div class="top"><h1>MAZU KG 物理层 — NetworkX D8 流向图 & 山洪风险传播</h1>
<p>全图 {G.number_of_nodes():,} 节点 · {G.number_of_edges():,} 边 | Asir 子图 {len(sns)} 节点 | {DATE}</p>
</div>
<div class="stats">
<span>🟢 绿色=低海拔  🟤 棕色=高海拔</span>
<span>➡️ 箭头=D8 流向</span>
<span>🔴 红色覆盖=风险传播区域</span>
<span>✖️ 源头={len(srcs)}  →  受影响={len(aff)} 节点</span>
</div>
{html}</body></html>"""

with open(OUT, "w", encoding="utf-8") as f: f.write(FULL)
print(f"✅ {OUT} ({len(FULL):,} chars) | Asir {len(sns)}节点 | {len(srcs)}源头→{len(aff)}受影响")

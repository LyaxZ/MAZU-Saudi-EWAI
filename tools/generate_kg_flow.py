"""生成 KG 运行可视化 — 四灾害标签页 + 图例 + 可缩放"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from data.loader import load_single_day
from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "kg_flow_map.html")

# 沙特海岸线
RED_SEA = [(16.4,42.5),(17.0,42.0),(17.7,41.6),(18.5,41.0),(19.0,40.5),(19.5,40.0),
    (20.2,39.8),(21.0,39.2),(21.5,39.2),(22.0,39.0),(22.5,38.8),(23.0,38.5),
    (23.5,38.3),(24.0,37.8),(24.5,37.4),(25.0,37.0),(25.5,36.7),(26.0,36.2),
    (26.5,35.8),(27.0,35.4),(27.5,35.0),(28.0,34.8),(28.5,34.8),(29.0,34.7)]
GULF = [(26.5,50.0),(27.0,49.7),(27.5,49.3),(27.8,48.8),(27.5,48.5),(27.0,48.8),
    (26.5,49.0),(26.0,49.3),(25.5,49.5),(25.0,49.8),(24.5,50.5),(24.0,50.8),
    (23.5,51.0),(23.0,51.2),(22.5,51.5),(22.0,51.8)]
S_BORDER = [(16.4,42.5),(16.4,44),(16.4,46),(16.4,48),(16.4,50),(16.4,52),(16.4,54),(16.4,55.5)]
N_BORDER = [(32.1,34.7),(32.1,36),(32.1,38),(32.1,40),(32.1,42),(32.1,44),(32.1,46),(32.1,48),(32.1,49)]

def coast_trace():
    return [
        go.Scatter(x=[p[1] for p in RED_SEA], y=[p[0] for p in RED_SEA],
            mode="lines", line=dict(color="#1e40af", width=2.5), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=[p[1] for p in GULF], y=[p[0] for p in GULF],
            mode="lines", line=dict(color="#1e40af", width=2.5), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=[p[1] for p in S_BORDER], y=[p[0] for p in S_BORDER],
            mode="lines", line=dict(color="#94a3b8", width=1.5, dash="dash"), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=[p[1] for p in N_BORDER], y=[p[0] for p in N_BORDER],
            mode="lines", line=dict(color="#94a3b8", width=1.5, dash="dash"), showlegend=False, hoverinfo="skip"),
    ]

def sea_ann():
    return [
        dict(x=38, y=25, text="红 海", showarrow=False, font=dict(size=14, color="#1e40af"), opacity=0.45),
        dict(x=51, y=24, text="阿拉伯湾", showarrow=False, font=dict(size=12, color="#1e40af"), opacity=0.45),
    ]

def base_layout(title, cmap_title, cmin, cmax, colors):
    return go.Layout(
        title=dict(text=title, font=dict(size=18, color="#1e293b"), x=0.5),
        xaxis=dict(range=[34,56], showgrid=False, fixedrange=False),
        yaxis=dict(range=[16,32], showgrid=False, fixedrange=False, scaleanchor="x"),
        height=700, margin=dict(l=50, r=30, t=55, b=40),
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
        coloraxis=dict(colorscale=colors, cmin=cmin, cmax=cmax,
            colorbar=dict(title=cmap_title, thickness=18, len=0.75, x=1.01)),
        dragmode="pan",
        annotations=sea_ann(),
    )

# ============================================================
print("加载 2025-08-28 数据...")
ds = load_single_day("2025-08-28")
lats = ds["latitude"].values.tolist() if "latitude" in ds.coords else ds["lat"].values.tolist()
lons = ds["longitude"].values.tolist() if "longitude" in ds.coords else ds["lon"].values.tolist()
n_lat, n_lon = len(lats), len(lons)
li = {lat: i for i, lat in enumerate(lats)}
lj = {lon: j for j, lon in enumerate(lons)}

# 提取核心变量
oro_arr = ds["orography"].values.squeeze()
tmax_arr = ds["tmax_c"].values.squeeze()
ws_arr = ds["wind10_speed"].values.squeeze()

print("构建 KG...")
# 用 load_to_dataframe 构建 KG（graph_builder 需要 DataFrame）
from data.loader import load_to_dataframe
df = load_to_dataframe("2025-08-28", "2025-08-28", variables=["orography"], show_progress=False).fillna(0)
G = KnowledgeGraphBuilder().build(df)

# 辅助：从 numpy 数组填充网格
def arr_to_grid(arr):
    g = np.full((n_lat, n_lon), np.nan)
    for i in range(min(n_lat, arr.shape[0])):
        for j in range(min(n_lon, arr.shape[1])):
            g[i, j] = arr[i, j]
    return g

# ============================================================
# 面板 1: 暴雨山洪
print("生成 ⚡ 暴雨山洪...")
oro_g = arr_to_grid(oro_arr)
fig1 = go.Figure(layout=base_layout("⚡ 暴雨山洪 — D8地形流向 + 下游风险传播", "海拔 (m)", 0, 3000,
    [[0,"#f0fdf4"],[.2,"#dcfce7"],[.4,"#bbf7d0"],[.6,"#86efac"],[.8,"#d4a574"],[1,"#8b5e3c"]]))
fig1.add_trace(go.Heatmap(z=oro_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="海拔: %{z:.0f}m<extra></extra>", name="地形"))

# 子图 + 传播
sns = [nid for nid, a in G.nodes(data=True) if 17.5<=a.get("lat",0)<=19.5 and 42<=a.get("lon",0)<=44]
sG = G.subgraph(sns).copy()
prop = RiskPropagator(sG, disaster_type="flash_flood", max_hops=8, max_distance_km=150)
high = sorted([(n,sG.nodes[n].get("orography",0)) for n in sG.nodes()], key=lambda x:-x[1])[:3]
srcs = [n for n,_ in high]
res = prop.propagate(source_nodes=srcs)
aff = set(res.get("affected_nodes",[]))

risk_g = np.full((n_lat,n_lon),np.nan)
for nid in aff:
    a = sG.nodes.get(nid,{})
    i,j = li.get(a.get("lat")), lj.get(a.get("lon"))
    if i is not None and j is not None: risk_g[i,j]=0.6
fig1.add_trace(go.Heatmap(z=risk_g, x=lons, y=lats, zsmooth="best",
    colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(239,68,68,0.5)"]], zmin=0,zmax=1,
    showscale=False, hovertemplate="⚠ 洪水风险<extra></extra>", name="风险"))

sx = [sG.nodes[n]["lon"] for n in srcs if n in sG.nodes]
sy = [sG.nodes[n]["lat"] for n in srcs if n in sG.nodes]
fig1.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
    marker=dict(size=12, color="#dc2626", symbol="x", line=dict(width=2,color="#fff")),
    name="风险源头", showlegend=False, hovertemplate="🔴 源头<extra></extra>"))

flow = [(u,v) for u,v,d in sG.edges(data=True) if d.get("type")=="flows_to"]
step = max(1,len(flow)//50)
for u,v in flow[::step]:
    au,av = sG.nodes[u],sG.nodes[v]
    dx,dy = (av["lon"]-au["lon"])*.55, (av["lat"]-au["lat"])*.55
    if abs(dx)>.001 or abs(dy)>.001:
        fig1.add_annotation(x=au["lon"],y=au["lat"],ax=au["lon"]+dx,ay=au["lat"]+dy,
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor="#475569", text="", opacity=0.5)
for t in coast_trace(): fig1.add_trace(t)

# ============================================================
# 面板 2: 极端高温
print("生成 🔥 极端高温...")
tmax_g = arr_to_grid(tmax_arr)
fig2 = go.Figure(layout=base_layout("🔥 极端高温 — 日最高气温异常暴露", "气温 (°C)", 25, 52,
    [[0,"#fef3c7"],[.25,"#fdba74"],[.5,"#f97316"],[.75,"#ea580c"],[1,"#7c2d12"]]))
fig2.add_trace(go.Heatmap(z=tmax_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="最高温: %{z:.1f}°C<extra></extra>", name="气温"))
for t in coast_trace(): fig2.add_trace(t)

# ============================================================
# 面板 3: 沙尘强风
print("生成 🌪️ 沙尘强风...")
ws_g = arr_to_grid(ws_arr)
fig3 = go.Figure(layout=base_layout("🌪️ 沙尘强风 — 10m 风速分布", "风速 (m/s)", 0, 22,
    [[0,"#fefce8"],[.3,"#fde68a"],[.6,"#facc15"],[.8,"#eab308"],[1,"#854d0e"]]))
fig3.add_trace(go.Heatmap(z=ws_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="风速: %{z:.1f} m/s<extra></extra>", name="风速"))
for t in coast_trace(): fig3.add_trace(t)

# ============================================================
# 面板 4: 沿海风浪
print("生成 🌊 沿海风浪...")
cst_g = arr_to_grid(np.where(oro_arr <= 100, 1.0, np.nan))
fig4 = go.Figure(layout=base_layout("🌊 沿海风浪 — 低海拔沿海暴露区 (≤100m)", "暴露", 0, 1,
    [[0,"rgba(0,0,0,0)"],[1,"rgba(59,130,246,0.7)"]]))
fig4.add_trace(go.Heatmap(z=cst_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="🌊 沿海低地<extra></extra>", name="沿海"))
for t in coast_trace(): fig4.add_trace(t)

# ============================================================
# 组装 HTML（JS 标签切换）
html_parts = []
for fid, (fig, div_id) in enumerate(zip(
    [fig1, fig2, fig3, fig4],
    ["tab1","tab2","tab3","tab4"])):
    html_parts.append(f'<div id="{div_id}" class="tab-content" style="display:{"block" if fid==0 else "none"}">')
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if fid==0 else False)))
    html_parts.append('</div>')

FULL = f"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="utf-8"><title>MAZU KG 四灾害运行可视化</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",system-ui,sans-serif;background:#f0f4f8;overflow:hidden}}
.top{{background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.06);z-index:10;position:relative}}
.header{{text-align:center;padding:14px 20px 6px}}
.header h1{{font-size:19px;color:#1e293b;font-weight:700}}
.header p{{font-size:12px;color:#64748b;margin:3px 0 0}}
.tabs{{display:flex;justify-content:center;gap:6px;padding:6px 12px 10px}}
.tab-btn{{padding:7px 20px;border-radius:8px;border:2px solid #e2e8f0;background:#fff;
  font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;color:#475569}}
.tab-btn:hover{{border-color:#94a3b8;color:#1e293b}}
.tab-btn.on{{border-color:#4f46e5;background:#eef2ff;color:#4f46e5}}
.tab-content{{height:calc(100vh - 110px)}}
.tab-content .plotly-graph-div{{height:100%!important}}
.tab-content .svg-container{{height:100%!important}}
.stats{{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;padding:4px 0 6px;font-size:11px;color:#64748b}}
.stats span{{background:#fff;border-radius:6px;padding:3px 10px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
</style></head>
<body>
<div class="top">
<div class="header">
<h1>🛰️ MAZU 知识图谱 — 四灾害实际运行可视化</h1>
<p>NetworkX DiGraph · {G.number_of_nodes():,} 节点 · {G.number_of_edges():,} 边 | 2025-08-28</p>
</div>
<div class="tabs" id="tabBar">
<button class="tab-btn on" onclick="showTab(0)">⚡ 暴雨山洪</button>
<button class="tab-btn" onclick="showTab(1)">🔥 极端高温</button>
<button class="tab-btn" onclick="showTab(2)">🌪️ 沙尘强风</button>
<button class="tab-btn" onclick="showTab(3)">🌊 沿海风浪</button>
</div>
<div class="stats">
<span>⚡ 山洪: D8流向+下游传播(Asir山区)</span>
<span>🔥 高温: 日最高气温·异常暴露</span>
<span>🌪️ 沙尘: 10m风速·起沙动力</span>
<span>🌊 风浪: 海拔≤100m沿海暴露区</span>
</div>
</div>
{''.join(html_parts)}
<script>
var tabBtns = document.querySelectorAll(".tab-btn");
var tabContents = document.querySelectorAll(".tab-content");
function showTab(n) {{
  tabBtns.forEach(function(b,i) {{ b.classList.toggle("on", i===n); }});
  tabContents.forEach(function(c,i) {{ c.style.display = i===n ? "block" : "none"; }});
  // trigger plotly resize
  setTimeout(function() {{ window.dispatchEvent(new Event("resize")); }}, 100);
}}
</script>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(FULL)
print(f"✅ 已生成: {OUT}  ({len(FULL):,} chars)")

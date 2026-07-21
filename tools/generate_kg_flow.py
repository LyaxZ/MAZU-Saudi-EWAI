"""生成 KG 运行可视化 — 四灾害标签页 · 有质感"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from data.loader import load_single_day, load_to_dataframe
from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "kg_flow_map.html")

# 海岸线
RS = [(16.4,42.5),(17.0,42),(17.7,41.6),(18.5,41),(19.0,40.5),(19.5,40),(20.2,39.8),(21,39.2),(21.5,39.2),
      (22,39),(22.5,38.8),(23,38.5),(23.5,38.3),(24,37.8),(24.5,37.4),(25,37),(25.5,36.7),(26,36.2),
      (26.5,35.8),(27,35.4),(27.5,35),(28,34.8),(28.5,34.8),(29,34.7)]
GL = [(26.5,50),(27,49.7),(27.5,49.3),(27.8,48.8),(27.5,48.5),(27,48.8),(26.5,49),(26,49.3),(25.5,49.5),
      (25,49.8),(24.5,50.5),(24,50.8),(23.5,51),(23,51.2),(22.5,51.5),(22,51.8)]
SB = [(16.4,42.5),(16.4,45),(16.4,48),(16.4,51),(16.4,54),(16.4,55.5)]
NB = [(32.1,34.7),(32.1,37),(32.1,40),(32.1,43),(32.1,46),(32.1,49)]

def add_map_lines(fig):
    """添加地图线 + 图例"""
    fig.add_trace(go.Scatter(x=[p[1] for p in RS], y=[p[0] for p in RS],
        mode="lines", line=dict(color="#2563eb", width=2), name="红海海岸", legendgroup="map",
        legendgrouptitle=dict(text="地图标注"), hovertemplate="红海海岸线<extra></extra>"))
    fig.add_trace(go.Scatter(x=[p[1] for p in GL], y=[p[0] for p in GL],
        mode="lines", line=dict(color="#2563eb", width=2), name="阿拉伯湾海岸", legendgroup="map",
        hovertemplate="阿拉伯湾海岸线<extra></extra>"))
    fig.add_trace(go.Scatter(x=[p[1] for p in SB], y=[p[0] for p in SB],
        mode="lines", line=dict(color="#cbd5e1", width=1.5, dash="dash"), name="南部国界", legendgroup="map",
        hovertemplate="南部国界<extra></extra>"))
    fig.add_trace(go.Scatter(x=[p[1] for p in NB], y=[p[0] for p in NB],
        mode="lines", line=dict(color="#cbd5e1", width=1.5, dash="dash"), name="北部国界", legendgroup="map",
        hovertemplate="北部国界<extra></extra>"))

def sea_labels(fig):
    fig.add_annotation(x=37.5, y=24.5, text="红 海", showarrow=False,
        font=dict(size=14, color="#1e40af", family="SimHei"), opacity=0.4)
    fig.add_annotation(x=51.5, y=24, text="阿拉伯湾", showarrow=False,
        font=dict(size=11, color="#1e40af", family="SimHei"), opacity=0.4)

# ============================================================
print("加载数据...")
ds = load_single_day("2025-08-28")
lats = ds["latitude"].values.tolist() if "latitude" in ds.coords else ds["lat"].values.tolist()
lons = ds["longitude"].values.tolist() if "longitude" in ds.coords else ds["lon"].values.tolist()
n_lat, n_lon = len(lats), len(lons)

oro_arr = ds["orography"].values.squeeze()
tmax_arr = ds["tmax_c"].values.squeeze()
ws_arr = ds["wind10_speed"].values.squeeze()

def arr2grid(arr):
    g = np.full((n_lat, n_lon), np.nan)
    h, w = min(n_lat, arr.shape[0]), min(n_lon, arr.shape[1])
    g[:h, :w] = arr[:h, :w]
    return g

oro_g = arr2grid(oro_arr)
tmax_g = arr2grid(tmax_arr)
ws_g = arr2grid(ws_arr)

print("构建 KG...")
df = load_to_dataframe("2025-08-28", "2025-08-28", variables=["orography"], show_progress=False).fillna(0)
G = KnowledgeGraphBuilder().build(df)

# 公共布局
def layout(title, cmap_title, cmin, cmax, colors, rev=False):
    cs = colors[::-1] if rev else colors
    return go.Layout(
        title=dict(text=title, font=dict(size=20, color="#1e293b", family="SimHei"), x=0.5),
        xaxis=dict(range=[34,56], showgrid=False, zeroline=False, fixedrange=False,
            title=dict(text="经度 (°E)", font=dict(size=12, color="#94a3b8"))),
        yaxis=dict(range=[16,32], showgrid=False, zeroline=False, fixedrange=False,
            title=dict(text="纬度 (°N)", font=dict(size=12, color="#94a3b8"))),
        height=680, margin=dict(l=60, r=30, t=60, b=50),
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
        coloraxis=dict(colorscale=cs, cmin=cmin, cmax=cmax,
            colorbar=dict(title=dict(text=cmap_title, font=dict(size=12, family="SimHei")),
                thickness=18, len=0.75, x=1.01, outlinewidth=0)),
        dragmode="pan", legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,.85)",
            bordercolor="#e2e8f0", borderwidth=1, font=dict(size=11, family="SimHei")),
    )

# ============================================================
# 面板 1: 暴雨山洪
print("⚡ 暴雨山洪...")
fig1 = go.Figure(layout=layout("⚡ 暴雨山洪 — D8 地形流向 & 下游风险传播",
    "海拔 (m)", 0, 3000,
    [[0,"#f0fdf4"],[0.15,"#dcfce7"],[0.35,"#bbf7d0"],[0.55,"#86efac"],[0.75,"#ca8a04"],[0.9,"#78350f"],[1,"#451a03"]]))
fig1.add_trace(go.Heatmap(z=oro_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度: %{x:.2f}°E<br>纬度: %{y:.2f}°N<br>海拔: %{z:.0f} m<extra></extra>", name="地形"))

# Asir 子图传播
sns = [nid for nid, a in G.nodes(data=True) if 17.5<=a.get("lat",0)<=20.0 and 41.5<=a.get("lon",0)<=44.5]
sG = G.subgraph(sns).copy()
prop = RiskPropagator(sG, disaster_type="flash_flood", max_hops=10, max_distance_km=200)
high = sorted([(n,sG.nodes[n].get("orography",0)) for n in sG.nodes()], key=lambda x:-x[1])[:3]
srcs = [n for n,_ in high]
res = prop.propagate(source_nodes=srcs)
aff = set(res.get("affected_nodes",[]))

# 风险覆盖——仅 Asir 区域
risk_oro = np.full((n_lat, n_lon), np.nan)
risk_aff = np.full((n_lat, n_lon), np.nan)
for nid in sG.nodes():
    a = sG.nodes[nid]
    i = int(round((a["lat"]-lats[0])/(lats[1]-lats[0]))) if len(lats)>1 else 0
    j = int(round((a["lon"]-lons[0])/(lons[1]-lons[0]))) if len(lons)>1 else 0
    if 0 <= i < n_lat and 0 <= j < n_lon:
        risk_oro[i, j] = a.get("orography", 0)
        if nid in aff:
            risk_aff[i, j] = 0.6

fig1.add_trace(go.Heatmap(z=risk_aff, x=lons, y=lats, zsmooth="best",
    colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(239,68,68,0.45)"]], zmin=0, zmax=1,
    showscale=False, hovertemplate="⚠ 洪水风险传播区域<extra></extra>", name="风险传播"))

sx = [sG.nodes[n]["lon"] for n in srcs if n in sG.nodes]
sy = [sG.nodes[n]["lat"] for n in srcs if n in sG.nodes]
fig1.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
    marker=dict(size=14, color="#dc2626", symbol="x-thin", line=dict(width=2.5, color="#fff")),
    name="风险源头", hovertemplate="🔴 风险源头<br>海拔: %{text} m<extra></extra>",
    text=[f"{sG.nodes[n].get('orography',0):.0f}" for n in srcs if n in sG.nodes]))

# 流向箭头——只取有代表性的
flow = [(u,v) for u,v,d in sG.edges(data=True) if d.get("type")=="flows_to"]
# 去重：每个格点只画一条最陡的流向
seen = set()
uniq_flow = []
for u,v in sorted(flow, key=lambda x: abs(sG.nodes[x[0]]["orography"]-sG.nodes[x[1]]["orography"]), reverse=True):
    if u not in seen:
        seen.add(u); uniq_flow.append((u,v))
step = max(1, len(uniq_flow)//40)
for u,v in uniq_flow[::step]:
    au,av = sG.nodes[u], sG.nodes[v]
    dx,dy = (av["lon"]-au["lon"])*.55, (av["lat"]-au["lat"])*.55
    if abs(dx)>.0005 or abs(dy)>.0005:
        fig1.add_annotation(x=au["lon"], y=au["lat"], ax=au["lon"]+dx, ay=au["lat"]+dy,
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#475569",
            text="", opacity=0.55)

add_map_lines(fig1); sea_labels(fig1)

# ============================================================
# 面板 2: 极端高温
print("🔥 极端高温...")
fig2 = go.Figure(layout=layout("🔥 极端高温 — 日最高气温 & 异常暴露分析",
    "气温 (°C)", 25, 52,
    [[0,"#fef3c7"],[0.2,"#fde68a"],[0.4,"#fdba74"],[0.6,"#f97316"],[0.8,"#dc2626"],[1,"#7c2d12"]]))
fig2.add_trace(go.Heatmap(z=tmax_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度: %{x:.2f}°E<br>纬度: %{y:.2f}°N<br>最高温: %{z:.1f} °C<extra></extra>", name="气温"))

# 叠加高温区轮廓(>45°C)
hot = np.where(tmax_g > 45, 1.0, np.nan)
fig2.add_trace(go.Heatmap(z=hot, x=lons, y=lats, zsmooth="best",
    colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(220,38,38,0.35)"]], zmin=0, zmax=1,
    showscale=False, hovertemplate="⚠ 极端高温区 (>45°C)<extra></extra>", name="极端高温区"))

add_map_lines(fig2); sea_labels(fig2)

# ============================================================
# 面板 3: 沙尘强风
print("🌪️ 沙尘强风...")
fig3 = go.Figure(layout=layout("🌪️ 沙尘强风 — 10m 风速 & 强风区分布",
    "风速 (m/s)", 0, 22,
    [[0,"#fefce8"],[0.25,"#fef08a"],[0.5,"#facc15"],[0.7,"#eab308"],[0.85,"#ca8a04"],[1,"#713f12"]]))
fig3.add_trace(go.Heatmap(z=ws_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度: %{x:.2f}°E<br>纬度: %{y:.2f}°N<br>风速: %{z:.1f} m/s<extra></extra>", name="风速"))

# 强风区 (>12 m/s)
strong = np.where(ws_g > 12, 1.0, np.nan)
fig3.add_trace(go.Heatmap(z=strong, x=lons, y=lats, zsmooth="best",
    colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(202,138,4,0.4)"]], zmin=0, zmax=1,
    showscale=False, hovertemplate="💨 强风区 (>12 m/s)<extra></extra>", name="强风区"))

add_map_lines(fig3); sea_labels(fig3)

# ============================================================
# 面板 4: 沿海风浪
print("🌊 沿海风浪...")
# 用连续海拔颜色（仅沿海低地区域），而非二元
cst_g = np.where(oro_arr <= 200, oro_arr, np.nan)
fig4 = go.Figure(layout=layout("🌊 沿海风浪 — 沿海低海拔暴露分析 (≤200m)",
    "海拔 (m)", 0, 200,
    [[0,"#dbeafe"],[0.25,"#93c5fd"],[0.5,"#3b82f6"],[0.75,"#1d4ed8"],[1,"#1e3a5f"]]))
fig4.add_trace(go.Heatmap(z=cst_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度: %{x:.2f}°E<br>纬度: %{y:.2f}°N<br>海拔: %{z:.0f} m<br><i>海拔越低,风浪风险越高</i><extra></extra>",
    name="沿海低地"))

add_map_lines(fig4); sea_labels(fig4)

# ============================================================
# HTML
html_parts = []
for fid, (fig, div_id) in enumerate(zip([fig1,fig2,fig3,fig4], ["t1","t2","t3","t4"])):
    html_parts.append(f'<div id="{div_id}" class="tc" style="display:{"block" if fid==0 else "none"}">')
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if fid==0 else False)))
    html_parts.append('</div>')

FULL = f"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="utf-8"><title>MAZU KG 四灾害运行可视化</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;background:#f1f5f9}}
.top{{background:linear-gradient(135deg,#1e293b,#334155);color:#fff;z-index:10;position:relative}}
.header{{text-align:center;padding:16px 20px 8px}}
.header h1{{font-size:20px;font-weight:700;letter-spacing:1px}}
.header p{{font-size:12px;color:#94a3b8;margin:4px 0 0}}
.tabs{{display:flex;justify-content:center;gap:4px;padding:8px 12px 12px;background:rgba(0,0,0,.15)}}
.tb{{padding:8px 24px;border-radius:10px 10px 0 0;border:none;background:rgba(255,255,255,.08);
  font-size:14px;font-weight:600;cursor:pointer;transition:all .25s;color:#94a3b8;font-family:inherit}}
.tb:hover{{background:rgba(255,255,255,.15);color:#e2e8f0}}
.tb.on{{background:#f1f5f9;color:#1e293b;box-shadow:0 -2px 8px rgba(0,0,0,.08)}}
.tc{{height:calc(100vh - 126px);background:#f1f5f9}}
.tc .plotly-graph-div,.tc .svg-container{{height:100%!important}}
.desc{{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;padding:3px 0 8px;font-size:11px;color:#94a3b8}}
.desc span{{background:rgba(255,255,255,.1);border-radius:5px;padding:2px 10px}}
</style></head>
<body>
<div class="top">
<div class="header">
<h1>🛰️ MAZU 知识图谱 — 四灾害实际运行可视化</h1>
<p>NetworkX DiGraph · {G.number_of_nodes():,} 节点 · {G.number_of_edges():,} 边 | 2025-08-28 | 红海/阿拉伯湾海岸线 · 国界虚线</p>
</div>
<div class="tabs">
<button class="tb on" onclick="sw(0)">⚡ 暴雨山洪</button>
<button class="tb" onclick="sw(1)">🔥 极端高温</button>
<button class="tb" onclick="sw(2)">🌪️ 沙尘强风</button>
<button class="tb" onclick="sw(3)">🌊 沿海风浪</button>
</div>
<div class="desc">
<span>⚡ D8流向+下游传播(Asir山区)</span>
<span>🔥 日最高气温·>45℃极端高温区</span>
<span>🌪️ 10m风速·>12m/s强风区</span>
<span>🌊 海拔≤200m沿海低地·越低越危险</span>
</div>
</div>
{''.join(html_parts)}
<script>
var btns=document.querySelectorAll(".tb"),tabs=document.querySelectorAll(".tc");
function sw(n){{btns.forEach(function(b,i){{b.classList.toggle("on",i===n)}});
tabs.forEach(function(c,i){{c.style.display=i===n?"block":"none"}});
setTimeout(function(){{window.dispatchEvent(new Event("resize"))}},150)}}
</script>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(FULL)
print(f"✅ 已生成: {OUT}  ({len(FULL):,} chars)")

"""生成 KG 运行可视化 — 侧边标签 + 日期参数"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from data.loader import load_single_day, load_to_dataframe
from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "kg_flow_map.html")
DATE = "2025-08-28"

# ============================================================
def layout(title, cmap_title, cmin, cmax, colors):
    return go.Layout(
        title=dict(text=title, font=dict(size=17, color="#1e293b", family="SimHei"), x=0.5),
        xaxis=dict(range=[34,56], showgrid=False, zeroline=False,
            title=dict(text="经度 (°E)", font=dict(size=10, color="#94a3b8"))),
        yaxis=dict(range=[16,32], showgrid=False, zeroline=False,
            title=dict(text="纬度 (°N)", font=dict(size=10, color="#94a3b8"))),
        height=650, margin=dict(l=55, r=25, t=50, b=45),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        coloraxis=dict(colorscale=colors, cmin=cmin, cmax=cmax,
            colorbar=dict(title=dict(text=cmap_title, font=dict(size=11, family="SimHei")),
                thickness=16, len=0.75, x=1.01, outlinewidth=0)),
        dragmode="pan",
    )

# ============================================================
print(f"加载 {DATE} 数据...")
ds = load_single_day(DATE)
lats = ds["latitude"].values.tolist() if "latitude" in ds.coords else ds["lat"].values.tolist()
lons = ds["longitude"].values.tolist() if "longitude" in ds.coords else ds["lon"].values.tolist()
n_lat, n_lon = len(lats), len(lons)

def a2g(arr):
    g = np.full((n_lat, n_lon), np.nan)
    h, w = min(n_lat, arr.shape[0]), min(n_lon, arr.shape[1])
    g[:h, :w] = arr[:h, :w]
    return g

oro_g = a2g(ds["orography"].values.squeeze())
tmax_g = a2g(ds["tmax_c"].values.squeeze())
ws_g = a2g(ds["wind10_speed"].values.squeeze())

print("构建 KG...")
df = load_to_dataframe(DATE, DATE, variables=["orography"], show_progress=False).fillna(0)
G = KnowledgeGraphBuilder().build(df)

# ============================================================
# 面板 1: 暴雨山洪
print("⚡ 暴雨山洪...")
fig1 = go.Figure(layout=layout("⚡ 暴雨山洪 — D8 地形流向 & 下游风险传播", "海拔 (m)", 0, 3000,
    [[0,"#f0fdf4"],[.15,"#dcfce7"],[.35,"#bbf7d0"],[.55,"#86efac"],[.75,"#ca8a04"],[.9,"#78350f"],[1,"#451a03"]]))

sns = [nid for nid, a in G.nodes(data=True) if 17.5<=a.get("lat",0)<=20.0 and 41.5<=a.get("lon",0)<=44.5]
sG = G.subgraph(sns).copy()
prop = RiskPropagator(sG, disaster_type="flash_flood", max_hops=10, max_distance_km=200)
high = sorted([(n,sG.nodes[n].get("orography",0)) for n in sG.nodes()], key=lambda x:-x[1])[:3]
srcs = [n for n,_ in high]
res = prop.propagate(source_nodes=srcs)
aff = set(res.get("affected_nodes",[]))

risk_aff = np.full((n_lat, n_lon), np.nan)
for nid in sG.nodes():
    a = sG.nodes[nid]
    i = int(round((a["lat"]-lats[0])/(lats[1]-lats[0]))) if len(lats)>1 else 0
    j = int(round((a["lon"]-lons[0])/(lons[1]-lons[0]))) if len(lons)>1 else 0
    if 0 <= i < n_lat and 0 <= j < n_lon and nid in aff:
        risk_aff[i,j] = 0.5

fig1.add_trace(go.Heatmap(z=risk_aff, x=lons, y=lats, zsmooth="best",
    colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(239,68,68,0.35)"]], zmin=0,zmax=1,
    showscale=False, hoverinfo="skip"))
fig1.add_trace(go.Heatmap(z=oro_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>海拔:%{z:.0f}m<extra></extra>"))

sx = [sG.nodes[n]["lon"] for n in srcs if n in sG.nodes]
sy = [sG.nodes[n]["lat"] for n in srcs if n in sG.nodes]
fig1.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
    marker=dict(size=14, color="#dc2626", symbol="x-thin", line=dict(width=2.5,color="#fff")),
    hovertemplate="🔴 风险源头<br>海拔:%{text}m<extra></extra>",
    text=[f"{sG.nodes[n].get('orography',0):.0f}" for n in srcs if n in sG.nodes]))

flow = [(u,v) for u,v,d in sG.edges(data=True) if d.get("type")=="flows_to"]
seen = set(); uniq = []
for u,v in sorted(flow, key=lambda x: abs(sG.nodes[x[0]]["orography"]-sG.nodes[x[1]]["orography"]), reverse=True):
    if u not in seen: seen.add(u); uniq.append((u,v))
step = max(1, len(uniq)//40)
for u,v in uniq[::step]:
    au,av = sG.nodes[u],sG.nodes[v]
    dx,dy = (av["lon"]-au["lon"])*.55, (av["lat"]-au["lat"])*.55
    if abs(dx)>.0005 or abs(dy)>.0005:
        fig1.add_annotation(x=au["lon"],y=au["lat"], ax=au["lon"]+dx, ay=au["lat"]+dy,
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#475569", text="", opacity=.5)

# ============================================================
# 面板 2
print("🔥 极端高温...")
fig2 = go.Figure(layout=layout("🔥 极端高温 — 日最高气温 & 异常暴露分析", "气温 (°C)", 25, 52,
    [[0,"#fef3c7"],[.2,"#fde68a"],[.4,"#fdba74"],[.6,"#f97316"],[.8,"#dc2626"],[1,"#7c2d12"]]))
fig2.add_trace(go.Heatmap(z=np.where(tmax_g>45,1.0,np.nan), x=lons, y=lats, zsmooth="best",
    colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(220,38,38,0.3)"]], zmin=0,zmax=1,
    showscale=False, hoverinfo="skip"))
fig2.add_trace(go.Heatmap(z=tmax_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>最高温:%{z:.1f}°C<extra></extra>"))

# ============================================================
# 面板 3
print("🌪️ 沙尘强风...")
fig3 = go.Figure(layout=layout("🌪️ 沙尘强风 — 10m 风速 & 强风区分布", "风速 (m/s)", 0, 22,
    [[0,"#fefce8"],[.25,"#fef08a"],[.5,"#facc15"],[.7,"#eab308"],[.85,"#ca8a04"],[1,"#713f12"]]))
fig3.add_trace(go.Heatmap(z=np.where(ws_g>12,1.0,np.nan), x=lons, y=lats, zsmooth="best",
    colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(202,138,4,0.35)"]], zmin=0,zmax=1,
    showscale=False, hoverinfo="skip"))
fig3.add_trace(go.Heatmap(z=ws_g, x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>风速:%{z:.1f}m/s<extra></extra>"))

# ============================================================
# 面板 4
print("🌊 沿海风浪...")
fig4 = go.Figure(layout=layout("🌊 沿海风浪 — 沿海低海拔暴露分析", "海拔 (m)", 0, 200,
    [[0,"#dbeafe"],[.2,"#93c5fd"],[.5,"#3b82f6"],[.75,"#1d4ed8"],[1,"#1e3a5f"]]))
fig4.add_trace(go.Heatmap(z=np.where(oro_g<300,oro_g,np.nan), x=lons, y=lats, zsmooth="best", coloraxis="coloraxis",
    hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>海拔:%{z:.0f}m<br><i>↓越低风险越高</i><extra></extra>"))

# ============================================================
# HTML
parts = []
for fid, (fig, div_id) in enumerate(zip([fig1,fig2,fig3,fig4], ["t1","t2","t3","t4"])):
    parts.append(f'<div id="{div_id}" class="panel" style="display:{"block" if fid==0 else "none"}">')
    parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if fid==0 else False), config={"displayModeBar":True,"displaylogo":False}))
    parts.append('</div>')

FULL = f"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="utf-8"><title>MAZU KG 四灾害运行可视化</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;background:#fff;overflow:hidden}}
.top{{background:#fff;border-bottom:1px solid #f1f5f9;padding:10px 20px;
  display:flex;align-items:center;justify-content:space-between;gap:16px}}
.top h1{{font-size:17px;color:#1e293b;font-weight:700;white-space:nowrap}}
.top .info{{font-size:11px;color:#94a3b8;white-space:nowrap}}
.date-box{{display:flex;align-items:center;gap:6px}}
.date-box input{{border:2px solid #e2e8f0;border-radius:8px;padding:5px 10px;font-size:13px;
  font-family:inherit;color:#334155;width:130px;outline:none;transition:border-color .2s}}
.date-box input:focus{{border-color:#4f46e5}}
.date-box .hint{{font-size:10px;color:#94a3b8}}
.main{{display:flex;height:calc(100vh - 50px)}}
.side{{width:100px;background:#f8fafc;border-right:1px solid #f1f5f9;
  display:flex;flex-direction:column;padding:12px 8px;gap:4px;flex-shrink:0}}
.st{{padding:10px 8px;border-radius:8px;border:none;background:transparent;
  font-size:12px;font-weight:600;cursor:pointer;transition:all .2s;color:#64748b;
  font-family:inherit;text-align:center;line-height:1.4}}
.st:hover{{background:#f1f5f9;color:#334155}}
.st.on{{background:#eef2ff;color:#4f46e5;box-shadow:inset 3px 0 0 #4f46e5}}
.panel{{flex:1;overflow:hidden;background:#fff}}
.panel .plotly-graph-div,.panel .svg-container{{height:100%!important}}
</style></head>
<body>
<div class="top">
<h1>🛰️ MAZU 知识图谱 — 四灾害运行可视化</h1>
<div class="date-box">
<input id="dateInp" type="text" value="{DATE}" placeholder="YYYY-MM-DD"
  title="修改日期后运行: python tools/generate_kg_flow.py --date YYYY-MM-DD">
<span class="hint">当前展示日期</span>
</div>
<div class="info">{G.number_of_nodes():,} 节点 · {G.number_of_edges():,} 边</div>
</div>
<div class="main">
<div class="side">
<button class="st on" onclick="sw(0)">⚡<br>暴雨山洪</button>
<button class="st" onclick="sw(1)">🔥<br>极端高温</button>
<button class="st" onclick="sw(2)">🌪️<br>沙尘强风</button>
<button class="st" onclick="sw(3)">🌊<br>沿海风浪</button>
</div>
{''.join(parts)}
</div>
<script>
var btns=document.querySelectorAll(".st"),panels=document.querySelectorAll(".panel");
function sw(n){{btns.forEach(function(b,i){{b.classList.toggle("on",i===n)}});
panels.forEach(function(p,i){{p.style.display=i===n?"block":"none"}});
setTimeout(function(){{window.dispatchEvent(new Event("resize"))}},150)}}
</script>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(FULL)
print(f"✅ 已生成: {OUT}  ({len(FULL):,} chars) | 日期: {DATE}")

"""生成全部10个教师 Ground Truth 事件的可视化 HTML（下拉切换）"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, plotly.graph_objects as go
from data.loader import load_single_day, load_to_dataframe
from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "kg_events.html")

# 全部 10 个事件
# Jan/Mar 可能缺大气廓线变量(fill-0可加载但部分指标NaN), 兜底 try/except
EVENTS = [
    ("2025-01-06", "flash_flood", "麦加/吉达特大洪水+拉比格龙卷风", 21.0, 22.5, 38.5, 40.5),
    ("2025-03-06", "flash_flood", "哈伊勒/布赖代春季首场大型山洪", 26.0, 28.0, 41.0, 44.5),
    ("2025-05-04", "dust_wind",   "卡西姆/利雅得巨型哈布尘暴(Haboob)", 24.5, 28.5, 41.0, 46.0),
    ("2025-05-16", "dust_wind",   "全年最强持续性沙尘(全国4天)",         16.0, 32.0, 34.0, 56.0),
    ("2025-05-25", "extreme_heat","52.2°C 破纪录高温",                   16.0, 32.0, 34.0, 56.0),
    ("2025-06-01", "extreme_heat","朝觐季 47°C 极端高温",                20.0, 25.0, 39.0, 43.0),
    ("2025-06-30", "dust_wind",   "东部/汉志持续性沙尘+高温叠加",       16.0, 32.0, 34.0, 56.0),
    ("2025-08-14", "flash_flood", "塔伊夫冰雹洪水(巨型冰雹)",           20.5, 22.0, 40.0, 42.0),
    ("2025-08-27", "flash_flood", "阿西尔/吉赞/纳季兰(10行政区预警)",   16.5, 19.5, 42.0, 45.0),
    ("2025-12-09", "flash_flood", "吉达历史性特大洪水(179mm/6h,2人遇难)",21.0, 22.5, 38.5, 40.0),
]

TITLES = {"flash_flood": "暴雨山洪", "extreme_heat": "极端高温", "dust_wind": "沙尘强风"}
CMAPS = {
    "flash_flood": [[0,"#f0fdf4"],[.2,"#dcfce7"],[.4,"#bbf7d0"],[.6,"#86efac"],[.8,"#ca8a04"],[.95,"#78350f"],[1,"#451a03"]],
    "extreme_heat": [[0,"#fef3c7"],[.2,"#fde68a"],[.4,"#fdba74"],[.6,"#f97316"],[.8,"#dc2626"],[1,"#7c2d12"]],
    "dust_wind": [[0,"#fefce8"],[.25,"#fef08a"],[.5,"#facc15"],[.7,"#eab308"],[.85,"#ca8a04"],[1,"#713f12"]],
}
CBARS = {"flash_flood": ("海拔 (m)", 0, 3000), "extreme_heat": ("气温 (°C)", 20, 55), "dust_wind": ("风速 (m/s)", 0, 22)}

figs = []
for date, dtype, label, lat_min, lat_max, lon_min, lon_max in EVENTS:
    try:
        print(f"生成 {date} {label} ({dtype})...")
        ds = load_single_day(date)
        lats_all = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lons_all = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
        n_lat, n_lon = len(lats_all), len(lons_all)
        cbar_title, cmin, cmax = CBARS[dtype]

        if dtype == "flash_flood":
            oro = ds["orography"].values.squeeze()
            df = load_to_dataframe(date, date, variables=["orography"], show_progress=False).fillna(0)
            G = KnowledgeGraphBuilder().build(df)
            sns = [nid for nid, a in G.nodes(data=True) if lat_min <= a.get("lat", 0) <= lat_max and lon_min <= a.get("lon", 0) <= lon_max]
            sG = G.subgraph(sns).copy()
            prop = RiskPropagator(sG, disaster_type="flash_flood", max_hops=10, max_distance_km=200)
            high = sorted([(n, sG.nodes[n].get("orography", 0)) for n in sG.nodes()], key=lambda x: -x[1])[:3]
            srcs = [n for n, _ in high]
            res = prop.propagate(source_nodes=srcs)
            aff = set(res.get("affected_nodes", []))

            risk = np.full((n_lat, n_lon), np.nan)
            for nid in sG.nodes():
                a = sG.nodes[nid]
                i = int(round((a["lat"] - lats_all[0]) / (lats_all[1] - lats_all[0])))
                j = int(round((a["lon"] - lons_all[0]) / (lons_all[1] - lons_all[0])))
                if 0 <= i < n_lat and 0 <= j < n_lon and nid in aff: risk[i, j] = 0.5

            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=risk, x=lons_all, y=lats_all, zsmooth="best",
                colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(239,68,68,0.35)"]], zmin=0, zmax=1, showscale=False, hoverinfo="skip"))
            fig.add_trace(go.Heatmap(z=oro, x=lons_all, y=lats_all, zsmooth="best",
                colorscale=CMAPS[dtype], zmin=cmin, zmax=cmax,
                colorbar=dict(title=cbar_title, thickness=16, len=0.75),
                hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>海拔:%{z:.0f}m<extra></extra>"))

            sx = [sG.nodes[n]["lon"] for n in srcs if n in sG.nodes]
            sy = [sG.nodes[n]["lat"] for n in srcs if n in sG.nodes]
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                marker=dict(size=12, color="#dc2626", symbol="x-thin", line=dict(width=2.5, color="#fff")),
                name="风险源头", hovertemplate="🔴 源头<br>海拔:%{text}m<extra></extra>",
                text=[f"{sG.nodes[n].get('orography',0):.0f}" for n in srcs if n in sG.nodes]))

            flow = [(u, v) for u, v, d in sG.edges(data=True) if d.get("type") == "flows_to"]
            seen = set(); uniq = []
            for u, v in sorted(flow, key=lambda x: abs(sG.nodes[x[0]]["orography"] - sG.nodes[x[1]]["orography"]), reverse=True):
                if u not in seen: seen.add(u); uniq.append((u, v))
            step = max(1, len(uniq) // 35)
            for u, v in uniq[::step]:
                au, av = sG.nodes[u], sG.nodes[v]
                dx, dy = (av["lon"] - au["lon"]) * .55, (av["lat"] - au["lat"]) * .55
                if abs(dx) > .0003 or abs(dy) > .0003:
                    fig.add_annotation(x=au["lon"], y=au["lat"], ax=au["lon"] + dx, ay=au["lat"] + dy,
                        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="#475569", text="", opacity=.5)
            desc = f"子图 {len(sns)} 节点 · {len(srcs)} 源头 → {len(aff)} 受影响"

        elif dtype == "extreme_heat":
            tmax = ds["tmax_c"].values.squeeze()
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=np.where(tmax > 45, 1.0, np.nan), x=lons_all, y=lats_all, zsmooth="best",
                colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(220,38,38,0.3)"]], zmin=0,zmax=1, showscale=False, hoverinfo="skip"))
            fig.add_trace(go.Heatmap(z=tmax, x=lons_all, y=lats_all, zsmooth="best",
                colorscale=CMAPS[dtype], zmin=cmin, zmax=cmax,
                colorbar=dict(title=cbar_title, thickness=16, len=0.75),
                hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>最高温:%{z:.1f}°C<extra></extra>"))
            desc = f"最高温 {np.nanmax(tmax):.1f}°C · 红色覆盖 = >45°C 极端高温区"

        elif dtype == "dust_wind":
            ws = ds["wind10_speed"].values.squeeze()
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=np.where(ws > 12, 1.0, np.nan), x=lons_all, y=lats_all, zsmooth="best",
                colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(202,138,4,0.35)"]], zmin=0,zmax=1, showscale=False, hoverinfo="skip"))
            fig.add_trace(go.Heatmap(z=ws, x=lons_all, y=lats_all, zsmooth="best",
                colorscale=CMAPS[dtype], zmin=cmin, zmax=cmax,
                colorbar=dict(title=cbar_title, thickness=16, len=0.75),
                hovertemplate="经度:%{x:.2f}°E<br>纬度:%{y:.2f}°N<br>风速:%{z:.1f}m/s<extra></extra>"))
            desc = f"最高风速 {np.nanmax(ws):.1f} m/s · 棕色覆盖 = >12 m/s 强风区"

        fig.update_layout(
            title=dict(text=f"{date} {label} — {TITLES[dtype]}", font=dict(size=16, color="#1e293b"), x=.5),
            xaxis=dict(range=[lon_min, lon_max], title="经度 (°E)", showgrid=False),
            yaxis=dict(range=[lat_min, lat_max], title="纬度 (°N)", showgrid=False),
            height=650, margin=dict(l=50, r=30, t=55, b=45),
            paper_bgcolor="#fff", plot_bgcolor="#fff", dragmode="pan")
        figs.append((date, label, dtype, desc, fig))
    except Exception as e:
        print(f"  ⚠ 跳过: {e}")

# ============================================================
parts = []; opts = []
for i, (date, label, dtype, desc, fig) in enumerate(figs):
    parts.append(f'<div id="ev{i}" class="panel" style="display:{"block" if i==0 else "none"}">')
    parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False), config={"displayModeBar": True, "displaylogo": False}))
    parts.append(f'<div class="desc">{desc}</div></div>')
    opts.append(f'<option value="{i}">{date} {label}</option>')

HTML = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>MAZU KG — 真实灾害事件</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;background:#fff;overflow:hidden}}
.top{{background:#fff;border-bottom:1px solid #f1f5f9;padding:12px 20px;display:flex;align-items:center;gap:16px}}
.top h1{{font-size:17px;color:#1e293b;font-weight:700;white-space:nowrap}}
.top select{{border:2px solid #e2e8f0;border-radius:8px;padding:6px 12px;font-size:13px;font-family:inherit;color:#334155;outline:none;cursor:pointer;min-width:280px}}
.top .info{{font-size:11px;color:#94a3b8}}
.panel{{background:#fff}}
.panel .plotly-graph-div,.panel .svg-container{{height:100%!important}}
.desc{{text-align:center;padding:6px;font-size:12px;color:#64748b;background:#f8fafc}}
</style></head><body>
<div class="top">
<h1>🛰️ MAZU KG — 教师 Ground Truth 灾害事件可视化</h1>
<select id="sel" onchange="sw(this.value)">
{''.join(opts)}
</select>
<div class="info">{len(figs)} 个事件 · 下拉切换</div>
</div>
{''.join(parts)}
<script>
function sw(n){{document.querySelectorAll(".panel").forEach(function(p,i){{p.style.display=i==n?"block":"none"}});
setTimeout(function(){{window.dispatchEvent(new Event("resize"))}},150)}}
</script>
</body></html>"""

with open(OUT, "w", encoding="utf-8") as f: f.write(HTML)
print(f"✅ {OUT} ({len(HTML):,} chars) | {len(figs)}/{len(EVENTS)} 个事件成功")

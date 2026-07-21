"""生成独立知识图谱 HTML 文件 — 分层布局 + 边关系标签"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.infrastructure import CITIES, WADIS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "knowledge_graph.html")

# ============================================================
# 数据定义
# ============================================================
# Layer 0: 气象因子（列 x= -500）
factors = [
    ("CAPE",      "#fca5a5", "对流有效位能：强对流核心驱动"),
    ("相对湿度",   "#93c5fd", "近地面+850hPa 相对湿度"),
    ("降水总量",   "#60a5fa", "日降水总量+卫星 1h 强降水估算"),
    ("风速",       "#a5b4fc", "10m 风速 + 850hPa 风速"),
    ("气温距平",   "#fdba74", "最高气温偏离气候态程度"),
    ("水汽输送",   "#86efac", "IVT 积分水汽 + 850hPa 水汽通量"),
    ("地形",       "#d4d4d8", "orography + 地表气压"),
    ("VPD",        "#fde68a", "饱和水汽压差：大气干燥程度"),
    ("涡度",       "#c4b5fd", "850hPa 相对涡度 + 位势高度"),
    ("风切变",     "#fecaca", "850-300hPa 深层风切变"),
]

# Layer 1: 灾害类型（列 x=0）
disasters = [
    ("山洪", "⚡", "#ef4444", "暴雨山洪：强降水 + 对流不稳定 + 地形抬升"),
    ("高温", "🔥", "#f97316", "极端高温：气温异常偏离气候态"),
    ("沙尘", "🌪️", "#eab308", "沙尘强风：强风 + 干燥 + 低湿度 + 裸土"),
    ("风浪", "🌊", "#3b82f6", "沿海风浪：强风 + 低地形 + 水汽输送"),
]

# 因子 → 灾害 边（含标签说明）
factor_edges = [
    ("CAPE",     "山洪", "对流驱动"),
    ("CAPE",     "沙尘", "对流抬升"),
    ("相对湿度",  "山洪", "水汽条件"),
    ("相对湿度",  "沙尘", "湿度抑制"),
    ("相对湿度",  "高温", "湿热胁迫"),
    ("降水总量",  "山洪", "直接触发"),
    ("风速",      "沙尘", "起沙动力"),
    ("风速",      "风浪", "波浪驱动"),
    ("气温距平",  "高温", "核心指标"),
    ("水汽输送",  "山洪", "水汽汇聚"),
    ("水汽输送",  "风浪", "水汽通道"),
    ("地形",      "山洪", "地形抬升"),
    ("地形",      "风浪", "海岸地形"),
    ("VPD",       "高温", "干燥胁迫"),
    ("VPD",       "沙尘", "干燥起沙"),
    ("涡度",      "沙尘", "气旋驱动"),
    ("风切变",    "沙尘", "深层动力"),
]

# Layer 2: 城市 + Wadi（列 x=+500）
city_disaster = {
    "利雅得": [("高温","热浪威胁"), ("沙尘","沙尘侵袭")],
    "吉达":   [("山洪","洪涝受灾"), ("风浪","海岸侵蚀")],
    "麦加":   [("山洪","山谷洪水"), ("高温","朝觐风险")],
    "达曼":   [("高温","热浪威胁"), ("沙尘","沙尘侵袭"), ("风浪","海岸侵蚀")],
    "艾卜哈":  [("山洪","山洪受灾")],
    "塔伊夫":  [("山洪","山洪受灾")],
    "哈伊勒":  [("沙尘","沙尘侵袭")],
    "布赖代":  [("沙尘","沙尘侵袭")],
    "朱拜勒":  [("风浪","港口受损")],
    "纳季兰":  [("山洪","山洪受灾")],
    "麦地那":  [("高温","热浪威胁")],
    "胡富夫":  [("高温","热浪威胁"), ("沙尘","沙尘侵袭")],
    "延布":    [("风浪","港口受损")],
    "吉赞":    [("山洪","山洪受灾")],
    "塔布克":  [("沙尘","沙尘侵袭")],
    "阿尔阿尔": [("沙尘","沙尘侵袭")],
}

# ============================================================
# 构建 vis.js 节点 & 边 JSON
# ============================================================
nodes = []
edges = []
nid = 0

# --- 因子 ---
for i, (name, color, desc) in enumerate(factors):
    nodes.append({"id": nid, "label": name, "title": desc,
        "color": {"background": color, "border": "#94a3b8"},
        "shape": "box", "size": 22, "font": {"size": 12, "face": "Segoe UI"},
        "x": -500, "y": (i - 4.5) * 70, "fixed": True, "group": "factor"})
    nid += 1

# --- 灾害 ---
for i, (name, icon, color, desc) in enumerate(disasters):
    nodes.append({"id": nid, "label": f"{icon} {name}", "title": desc,
        "color": {"background": color, "border": "#fff"},
        "shape": "dot", "size": 45, "borderWidth": 3,
        "font": {"size": 14, "face": "Segoe UI", "color": "#1e293b", "bold": True},
        "x": 0, "y": (i - 1.5) * 120, "fixed": True, "group": "disaster"})
    nid += 1

# --- 城市 ---
added_cities = set()
ci = 0
for c in CITIES:
    cn = c["name"].split("(")[0].strip()
    if cn in city_disaster:
        added_cities.add(cn)
        nodes.append({"id": nid, "label": cn,
            "title": f"人口: {c['pop']:,} | 类型: {c['type']}",
            "color": {"background": "#475569", "border": "#334155"},
            "shape": "dot", "size": 14,
            "font": {"size": 11, "face": "Segoe UI", "color": "#e2e8f0"},
            "x": 500, "y": (ci - 7) * 42, "fixed": True, "group": "city"})
        nid += 1
        ci += 1

# --- Wadi ---
for wi, w in enumerate(WADIS):
    nodes.append({"id": nid, "label": f"🏞️ {w['name']}",
        "title": f"{w['length_km']}km | 山洪风险区",
        "color": {"background": "#06b6d4", "border": "#0891b2"},
        "shape": "diamond", "size": 11,
        "font": {"size": 9, "face": "Segoe UI", "color": "#1e293b"},
        "x": 500, "y": (ci + wi) * 42, "fixed": True, "group": "wadi"})
    nid += 1

# --- 名字 → ID 映射 ---
name2id = {}
for n in nodes:
    lbl = n["label"].replace("⚡ ","").replace("🔥 ","").replace("🌪️ ","").replace("🌊 ","").replace("🏞️ ","")
    name2id[lbl] = n["id"]

# --- 因子 → 灾害 边 ---
for sn, dn, lb in factor_edges:
    s = name2id.get(sn)
    d = name2id.get(dn)
    if s is not None and d is not None:
        edges.append({"from": s, "to": d, "label": lb,
            "color": {"color": "#cbd5e1", "highlight": "#94a3b8"},
            "width": 2, "arrows": "to",
            "font": {"size": 9, "face": "Segoe UI", "color": "#64748b",
                     "strokeWidth": 2, "strokeColor": "#f8fafc", "align": "horizontal"}})

# --- 城市 → 灾害 边 ---
for cn, dlist in city_disaster.items():
    if cn in added_cities:
        cid = name2id.get(cn)
        for dn, lb in dlist:
            did = name2id.get(dn)
            if cid is not None and did is not None:
                edges.append({"from": cid, "to": did, "label": lb,
                    "color": {"color": "#e2e8f0", "highlight": "#cbd5e1"},
                    "width": 1.2, "dashes": True, "arrows": "to",
                    "font": {"size": 8, "face": "Segoe UI", "color": "#94a3b8",
                             "strokeWidth": 2, "strokeColor": "#f8fafc", "align": "horizontal"}})

# --- Wadi → 山洪 ---
sh_id = name2id.get("山洪")
for w in WADIS:
    wid = name2id.get(w["name"])
    if wid is not None and sh_id is not None:
        edges.append({"from": wid, "to": sh_id, "label": "洪水通道",
            "color": {"color": "#a5f3fc", "highlight": "#67e8f9"},
            "width": 1.5, "arrows": "to",
            "font": {"size": 8, "face": "Segoe UI", "color": "#0891b2",
                     "strokeWidth": 2, "strokeColor": "#f8fafc", "align": "horizontal"}})

# ============================================================
# 生成 HTML
# ============================================================
HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",system-ui,-apple-system,sans-serif;background:#f8fafc;overflow:hidden}}
.header{{text-align:center;padding:20px 20px 14px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.header h1{{font-size:21px;color:#1e293b;font-weight:700}}
.header p{{font-size:12px;color:#64748b;margin:5px 0 0}}
.legend{{display:flex;justify-content:center;gap:18px;flex-wrap:wrap;padding:8px 12px;
  background:#fff;font-size:11px;color:#475569;border-top:1px solid #f1f5f9}}
.legend span{{display:flex;align-items:center;gap:4px}}
.dot,.box,.diamond{{display:inline-block;width:10px;height:10px}}
.dot{{border-radius:50%}}.box{{border-radius:2px}}
.diamond{{transform:rotate(45deg);border-radius:1px}}
#network{{height:calc(100vh - 105px)}}
</style>
</head>
<body>
<div class="header">
<h1>🛰️ MAZU 沙特多灾种预警 — 知识图谱</h1>
<p>气象因子（左） → 灾害类型（中） → 城市 / Wadi 承灾体（右） | 箭头含关系标签</p>
</div>
<div class="legend">
<span><span class="dot" style="background:#ef4444"></span>灾害类型</span>
<span><span class="box" style="background:#fca5a5"></span>气象因子</span>
<span><span class="dot" style="background:#475569"></span>城市</span>
<span><span class="diamond" style="background:#06b6d4"></span>Wadi</span>
<span style="color:#94a3b8">━━ 实线=直接驱动</span>
<span style="color:#94a3b8">┅┅ 虚线=承灾关联</span>
</div>
<div id="network"></div>
<script>
var nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
var edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});
new vis.Network(document.getElementById("network"), {{nodes:nodes,edges:edges}}, {{
  physics:{{enabled:false}},
  interaction:{{hover:true,tooltipDelay:150,navigationButtons:true,keyboard:true}},
  edges:{{smooth:{{type:"continuous",roundness:0.25}},font:{{align:"horizontal"}}}}
}});
</script>
</body>
</html>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)

n_factor = sum(1 for n in nodes if n["group"] == "factor")
n_disaster = sum(1 for n in nodes if n["group"] == "disaster")
n_city = sum(1 for n in nodes if n["group"] == "city")
n_wadi = sum(1 for n in nodes if n["group"] == "wadi")
print(f"✅ 知识图谱已生成: {OUT}  ({len(HTML):,} chars)")
print(f"   节点: {len(nodes)} ({n_factor}因子 + {n_disaster}灾害 + {n_city}城市 + {n_wadi}Wadi)")
print(f"   边:   {len(edges)} 条（全部含关系标签）")

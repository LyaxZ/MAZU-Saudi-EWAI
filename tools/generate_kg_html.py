"""生成知识图谱 HTML — 动态物理 + 层级字号 + 点击筛选灾害子图"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.infrastructure import CITIES, WADIS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "knowledge_graph.html")

# ============================================================
# 数据
# ============================================================
factors = [
    ("CAPE",       "#fca5a5", "对流有效位能：强对流核心驱动"),
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

disasters = [
    ("山洪", "⚡", "#ef4444", "暴雨山洪：强降水+对流不稳定+地形抬升"),
    ("高温", "🔥", "#f97316", "极端高温：气温异常偏离气候态"),
    ("沙尘", "🌪️", "#eab308", "沙尘强风：强风+干燥+低湿度+裸土"),
    ("风浪", "🌊", "#3b82f6", "沿海风浪：强风+低地形+水汽输送"),
]

factor_edges = [
    ("CAPE","山洪","对流驱动"), ("CAPE","沙尘","对流抬升"),
    ("相对湿度","山洪","水汽条件"), ("相对湿度","沙尘","湿度抑制"), ("相对湿度","高温","湿热胁迫"),
    ("降水总量","山洪","直接触发"), ("风速","沙尘","起沙动力"), ("风速","风浪","波浪驱动"),
    ("气温距平","高温","核心指标"), ("水汽输送","山洪","水汽汇聚"), ("水汽输送","风浪","水汽通道"),
    ("地形","山洪","地形抬升"), ("地形","风浪","海岸地形"),
    ("VPD","高温","干燥胁迫"), ("VPD","沙尘","干燥起沙"),
    ("涡度","沙尘","气旋驱动"), ("风切变","沙尘","深层动力"),
]

city_disaster = {
    "利雅得": [("高温","热浪威胁"), ("沙尘","沙尘侵袭")],
    "吉达":   [("山洪","洪涝受灾"), ("风浪","海岸侵蚀")],
    "麦加":   [("山洪","山谷洪水"), ("高温","朝觐风险")],
    "达曼":   [("高温","热浪威胁"), ("沙尘","沙尘侵袭"), ("风浪","海岸侵蚀")],
    "艾卜哈":  [("山洪","山洪受灾")], "塔伊夫": [("山洪","山洪受灾")],
    "哈伊勒":  [("沙尘","沙尘侵袭")], "布赖代": [("沙尘","沙尘侵袭")],
    "朱拜勒":  [("风浪","港口受损")], "纳季兰": [("山洪","山洪受灾")],
    "麦地那":  [("高温","热浪威胁")],
    "胡富夫":  [("高温","热浪威胁"), ("沙尘","沙尘侵袭")],
    "延布":    [("风浪","港口受损")], "吉赞":   [("山洪","山洪受灾")],
    "塔布克":  [("沙尘","沙尘侵袭")], "阿尔阿尔":[("沙尘","沙尘侵袭")],
}

# ============================================================
# 构建数据
# ============================================================
nodes, edges = [], []
nid, eid = 0, 0

# --- 因子 ---
for i, (name, color, desc) in enumerate(factors):
    nodes.append({"id": nid, "label": name, "title": desc,
        "color": {"background": color, "border": "#94a3b8"},
        "shape": "box", "size": 26, "font": {"size": 12, "face": "Segoe UI"},
        "group": "factor"})
    nid += 1

# --- 灾害 ---
for i, (name, icon, color, desc) in enumerate(disasters):
    nodes.append({"id": nid, "label": f"{icon} {name}", "title": desc,
        "color": {"background": color, "border": "#fff"},
        "shape": "dot", "size": 48, "borderWidth": 4,
        "font": {"size": 15, "face": "Segoe UI", "color": "#1e293b", "bold": True},
        "group": "disaster"})
    nid += 1

# --- 城市 ---
added_cities = set()
for c in CITIES:
    cn = c["name"].split("(")[0].strip()
    if cn in city_disaster:
        added_cities.add(cn)
        pop_str = f"{c['pop']:,}"
        nodes.append({"id": nid, "label": cn,
            "title": f"人口: {pop_str} | 类型: {c['type']}",
            "color": {"background": "#475569", "border": "#334155"},
            "shape": "dot", "size": 14, "font": {"size": 10, "face": "Segoe UI", "color": "#e2e8f0"},
            "group": "city"})
        nid += 1

# --- Wadi ---
for w in WADIS:
    nodes.append({"id": nid, "label": f"🏞️ {w['name']}",
        "title": f"{w['length_km']}km | 山洪风险区",
        "color": {"background": "#06b6d4", "border": "#0891b2"},
        "shape": "diamond", "size": 12, "font": {"size": 9, "face": "Segoe UI", "color": "#1e293b"},
        "group": "wadi"})
    nid += 1

# 名字→ID
name2id = {}
for n in nodes:
    lbl = n["label"].replace("⚡ ","").replace("🔥 ","").replace("🌪️ ","").replace("🌊 ","").replace("🏞️ ","")
    name2id[lbl] = n["id"]

# --- 因子→灾害 边 ---
for sn, dn, lb in factor_edges:
    s = name2id.get(sn); d = name2id.get(dn)
    if s is not None and d is not None:
        edges.append({"id": eid, "from": s, "to": d, "label": lb,
            "color": {"color": "#cbd5e1", "highlight": "#94a3b8"}, "width": 2, "arrows": "to",
            "font": {"size": 8, "face": "Segoe UI", "color": "#64748b", "strokeWidth": 2, "strokeColor": "#fff", "align": "horizontal"}})
        eid += 1

# --- 城市→灾害 边 ---
for cn, dlist in city_disaster.items():
    if cn in added_cities:
        cid = name2id.get(cn)
        for dn, lb in dlist:
            did = name2id.get(dn)
            if cid is not None and did is not None:
                edges.append({"id": eid, "from": cid, "to": did, "label": lb,
                    "color": {"color": "#e2e8f0", "highlight": "#cbd5e1"}, "width": 1.2, "dashes": True, "arrows": "to",
                    "font": {"size": 7, "face": "Segoe UI", "color": "#94a3b8", "strokeWidth": 2, "strokeColor": "#fff", "align": "horizontal"}})
                eid += 1

# --- Wadi→山洪 边 ---
sh_id = name2id.get("山洪")
for w in WADIS:
    wid = name2id.get(w["name"])
    if wid is not None and sh_id is not None:
        edges.append({"id": eid, "from": wid, "to": sh_id, "label": "洪水通道",
            "color": {"color": "#a5f3fc", "highlight": "#67e8f9"}, "width": 1.5, "arrows": "to",
            "font": {"size": 7, "face": "Segoe UI", "color": "#0891b2", "strokeWidth": 2, "strokeColor": "#fff", "align": "horizontal"}})
        eid += 1

# 统计
n_factor = sum(1 for n in nodes if n["group"] == "factor")
n_disaster = sum(1 for n in nodes if n["group"] == "disaster")
n_city = sum(1 for n in nodes if n["group"] == "city")
n_wadi = sum(1 for n in nodes if n["group"] == "wadi")

# ============================================================
HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:"Segoe UI",system-ui,sans-serif;background:#f0f4f8;overflow:hidden}}
.top{{background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.06);z-index:10;position:relative}}
.header{{text-align:center;padding:18px 20px 10px}}
.header h1{{font-size:20px;color:#1e293b;font-weight:700}}
.header p{{font-size:12px;color:#64748b;margin:4px 0 0}}
.filter-bar{{display:flex;justify-content:center;gap:8px;padding:8px 12px 12px;flex-wrap:wrap}}
.filter-btn{{padding:6px 16px;border-radius:8px;border:2px solid #e2e8f0;background:#fff;
  font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;color:#475569}}
.filter-btn:hover{{border-color:#94a3b8;color:#1e293b}}
.filter-btn.active{{border-color:#4f46e5;background:#eef2ff;color:#4f46e5}}
.filter-btn.all{{font-size:12px;padding:6px 12px}}
#network{{height:calc(100vh - 120px)}}
.tooltip{{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
  background:#1e293b;color:#fff;padding:8px 16px;border-radius:8px;font-size:12px;
  pointer-events:none;opacity:0;transition:opacity .3s;z-index:20}}
.tooltip.show{{opacity:1}}
</style>
</head>
<body>
<div class="top">
<div class="header">
<h1>🛰️ MAZU 沙特多灾种预警 — 知识图谱</h1>
<p>气象因子 — 灾害类型 — 城市 / Wadi 承灾体 | 点击灾害节点筛选子图 · 双击空白重置</p>
</div>
<div class="filter-bar" id="filterBar">
<button class="filter-btn all active" onclick="resetFilter()">🌐 全部</button>
<button class="filter-btn" data-disaster="⚡ 山洪" onclick="filterDisaster('⚡ 山洪')">⚡ 山洪</button>
<button class="filter-btn" data-disaster="🔥 高温" onclick="filterDisaster('🔥 高温')">🔥 高温</button>
<button class="filter-btn" data-disaster="🌪️ 沙尘" onclick="filterDisaster('🌪️ 沙尘')">🌪️ 沙尘</button>
<button class="filter-btn" data-disaster="🌊 风浪" onclick="filterDisaster('🌊 风浪')">🌊 风浪</button>
</div>
</div>
<div id="network"></div>
<div class="tooltip" id="tooltip"></div>

<script>
var nodesData = {json.dumps(nodes, ensure_ascii=False)};
var edgesData = {json.dumps(edges, ensure_ascii=False)};
var nodes = new vis.DataSet(nodesData);
var edges = new vis.DataSet(edgesData);

var options = {{
  physics: {{
    solver: "forceAtlas2Based",
    forceAtlas2Based: {{ gravitationalConstant: -60, centralGravity: 0.005, springLength: 180, springConstant: 0.03, damping: 0.4 }},
    stabilization: {{ iterations: 200, fit: true }}
  }},
  interaction: {{ hover: true, tooltipDelay: 150, navigationButtons: true, keyboard: true, dragNodes: true }},
  edges: {{ smooth: {{ type: "continuous", roundness: 0.3 }}, font: {{ align: "horizontal" }} }},
  groups: {{
    factor:   {{ shape: "box",    borderWidth: 1, color: {{ border: "#94a3b8" }} }},
    disaster: {{ shape: "dot",    borderWidth: 4, color: {{ border: "#fff" }},    font: {{ size: 15, bold: true }} }},
    city:     {{ shape: "dot",    borderWidth: 1, color: {{ border: "#334155" }}, font: {{ size: 10 }} }},
    wadi:     {{ shape: "diamond",borderWidth: 1, color: {{ border: "#0891b2" }}, font: {{ size: 9 }} }}
  }}
}};

var network = new vis.Network(document.getElementById("network"), {{ nodes: nodes, edges: edges }}, options);

// === 筛选逻辑 ===
var currentFilter = null;

// 构建邻接表
var adj = {{}};
edgesData.forEach(function(e) {{
  if (!adj[e.from]) adj[e.from] = new Set();
  if (!adj[e.to])   adj[e.to]   = new Set();
  adj[e.from].add(e.to);
  adj[e.to].add(e.from);
}});

function getSubgraph(rootId) {{
  var visited = new Set();
  var queue = [rootId];
  while (queue.length) {{
    var v = queue.shift();
    if (visited.has(v)) continue;
    visited.add(v);
    (adj[v] || new Set()).forEach(function(nb) {{ queue.push(nb); }});
  }}
  return visited;
}}

function applyFilter(rootLabel, rootId) {{
  var sub = rootId ? getSubgraph(rootId) : null;
  var allNodes = nodes.get();
  var allEdges = edges.get();

  var nodeUpdates = [];
  allNodes.forEach(function(n) {{
    var visible = !sub || sub.has(n.id);
    if (n.hidden === visible) nodeUpdates.push({{ id: n.id, hidden: !visible }});
  }});
  if (nodeUpdates.length) nodes.update(nodeUpdates);

  var edgeUpdates = [];
  allEdges.forEach(function(e) {{
    var visible = !sub || (sub.has(e.from) && sub.has(e.to));
    if (e.hidden === visible) edgeUpdates.push({{ id: e.id, hidden: !visible }});
  }});
  if (edgeUpdates.length) edges.update(edgeUpdates);

  // 更新按钮状态
  document.querySelectorAll(".filter-btn").forEach(function(b) {{ b.classList.remove("active"); }});
  if (!rootLabel) {{
    document.querySelector(".filter-btn.all").classList.add("active");
  }} else {{
    var btn = document.querySelector('.filter-btn[data-disaster="' + rootLabel + '"]');
    if (btn) btn.classList.add("active");
  }}
  currentFilter = rootLabel;

  // tooltip
  var tt = document.getElementById("tooltip");
  if (rootLabel) {{
    var cnt = sub ? sub.size : 0;
    tt.textContent = "已筛选: " + rootLabel + " — " + cnt + " 个关联节点";
    tt.classList.add("show");
    setTimeout(function() {{ tt.classList.remove("show"); }}, 2000);
  }}
}}

function filterDisaster(label) {{
  var targetId = null;
  nodesData.forEach(function(n) {{
    if (n.label === label) targetId = n.id;
  }});
  if (targetId !== null) applyFilter(label, targetId);
}}

function resetFilter() {{
  applyFilter(null, null);
}}

// 点击节点筛选
network.on("click", function(params) {{
  if (params.nodes.length === 1) {{
    var node = nodes.get(params.nodes[0]);
    if (node.group === "disaster") {{
      filterDisaster(node.label);
      return;
    }}
  }}
  if (params.nodes.length === 0 && currentFilter) {{
    resetFilter();
  }}
}});

// 双击空白重置
network.on("doubleClick", function(params) {{
  if (params.nodes.length === 0) resetFilter();
}});
</script>
</body>
</html>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"✅ 知识图谱已生成: {OUT}  ({len(HTML):,} chars)")
print(f"   节点: {len(nodes)} ({n_factor}因子 + {n_disaster}灾害 + {n_city}城市 + {n_wadi}Wadi)")
print(f"   边:   {len(edges)} 条（全部含关系标签）")
print(f"   交互: 点击灾害节点筛选子图 | 双击空白/顶部按钮重置")

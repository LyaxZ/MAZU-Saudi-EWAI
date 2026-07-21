"""生成知识图谱 HTML — 纯因子→灾害关系 + 1-hop筛选"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "knowledge_graph.html")

# ============================================================
# 节点数据
# ============================================================

# 致灾因子
factors = [
    ("对流有效位能\n(CAPE)",       "#fca5a5", "强对流核心驱动因子，数值越高越有利于雷暴发展"),
    ("近地面相对湿度",              "#93c5fd", "低层大气水汽含量，直接影响降水效率"),
    ("日降水总量",                  "#60a5fa", "24小时累积降水量，山洪直接触发条件"),
    ("10米风速",                    "#a5b4fc", "近地面风速，沙尘起沙和风浪的直接动力"),
    ("气温距平",                    "#fdba74", "当日气温偏离气候态的程度，热浪核心指标"),
    ("水汽通量散度",               "#86efac", "水汽输送的汇聚/辐散程度，决定降水落区"),
    ("地形高度",                    "#d4d4d8", "地形抬升强迫，迎风坡增强降水"),
    ("饱和水汽压差\n(VPD)",        "#fde68a", "大气干燥程度指标，VPD越高空气越干"),
    ("850hPa 涡度",                "#c4b5fd", "低层气旋式旋转，沙尘天气的动力条件"),
    ("深层风切变",                  "#fecaca", "850-300hPa风速差异，对流组织化条件"),
    ("对流抑制能量\n(CIN)",        "#e2e8f0", "抑制对流的能量，CIN越小越容易触发"),
    ("地表温度",                    "#fed7aa", "地表热力状况，影响近地面稳定度"),
]

# 形成机制（中间层）
mechanisms = [
    ("强对流发展",    "#fecaca", "CAPE高 + CIN低 → 深厚对流 → 暴雨"),
    ("水汽辐合",      "#bbf7d0", "水汽输送汇聚 + 湿度高 → 强降水"),
    ("大气干燥化",    "#fde68a", "VPD高 + 湿度低 + 高温 → 极端干旱/沙尘"),
    ("强风动力",      "#c7d2fe", "风速大 + 涡度强 → 起沙 / 波浪"),
    ("热力异常",      "#fed7aa", "气温距平大 + 地表高温 → 高温热浪"),
    ("地形强迫",      "#d4d4d8", "地形抬升 + 水汽 → 迎风坡强降水"),
]

# 灾害类型
disasters = [
    ("暴雨山洪", "⚡", "#ef4444", "强降水 + 对流不稳定 + 地形抬升 → 突发性洪水"),
    ("极端高温", "🔥", "#f97316", "气温异常偏离气候态 → 持续高温热浪"),
    ("沙尘强风", "🌪️", "#eab308", "强风 + 干燥地表 + 低湿度 → 沙尘暴"),
    ("沿海风浪", "🌊", "#3b82f6", "强风 + 低地形 + 水汽 → 风暴潮/大浪"),
]

# ============================================================
# 边定义（因子 → 机制 → 灾害）
# ============================================================

# 因子 → 机制
factor_to_mechanism = [
    ("对流有效位能\n(CAPE)",   "强对流发展",   "提供能量"),
    ("对流抑制能量\n(CIN)",     "强对流发展",   "抑制减弱"),
    ("深层风切变",              "强对流发展",   "组织维持"),
    ("水汽通量散度",            "水汽辐合",     "水汽汇聚"),
    ("近地面相对湿度",          "水汽辐合",     "湿度条件"),
    ("饱和水汽压差\n(VPD)",    "大气干燥化",   "干燥程度"),
    ("气温距平",                "大气干燥化",   "高温加剧"),
    ("地表温度",                "大气干燥化",   "地表加热"),
    ("近地面相对湿度",          "大气干燥化",   "湿度降低"),
    ("10米风速",                "强风动力",     "直接动力"),
    ("850hPa 涡度",             "强风动力",     "旋转增强"),
    ("气温距平",                "热力异常",     "偏离程度"),
    ("地表温度",                "热力异常",     "地表加热"),
    ("地形高度",                "地形强迫",     "抬升作用"),
    ("水汽通量散度",            "地形强迫",     "迎风水汽"),
]

# 机制 → 灾害
mechanism_to_disaster = [
    ("强对流发展",  "暴雨山洪", "主要成因"),
    ("水汽辐合",    "暴雨山洪", "直接触发"),
    ("地形强迫",    "暴雨山洪", "山地增强"),
    ("热力异常",    "极端高温", "直接成因"),
    ("大气干燥化",  "极端高温", "加剧作用"),
    ("大气干燥化",  "沙尘强风", "起沙条件"),
    ("强风动力",    "沙尘强风", "动力驱动"),
    ("强风动力",    "沿海风浪", "波浪驱动"),
    ("水汽辐合",    "沿海风浪", "水汽输送"),
    ("地形强迫",    "沿海风浪", "海岸效应"),
]

# ============================================================
# 构建节点和边
# ============================================================
nodes, edges = [], []
nid, eid = 0, 0

for name, color, desc in factors:
    nodes.append({"id": nid, "label": name, "title": desc,
        "color": {"background": color, "border": "#94a3b8"},
        "shape": "box", "size": 28, "font": {"size": 11, "face": "Segoe UI"},
        "group": "factor"})
    nid += 1

for name, color, desc in mechanisms:
    nodes.append({"id": nid, "label": name, "title": desc,
        "color": {"background": color, "border": "#94a3b8"},
        "shape": "ellipse", "size": 20, "font": {"size": 11, "face": "Segoe UI", "color": "#475569"},
        "group": "mechanism"})
    nid += 1

for name, icon, color, desc in disasters:
    nodes.append({"id": nid, "label": f"{icon} {name}", "title": desc,
        "color": {"background": color, "border": "#fff"},
        "shape": "dot", "size": 52, "borderWidth": 4,
        "font": {"size": 15, "face": "Segoe UI", "color": "#1e293b", "bold": True},
        "group": "disaster"})
    nid += 1

name2id = {}
for n in nodes:
    name2id[n["label"]] = n["id"]

# 边匹配
for sn, dn, lb in factor_to_mechanism:
    s = name2id.get(sn)
    d = name2id.get(dn)
    if s is not None and d is not None:
        edges.append({"id": eid, "from": s, "to": d, "label": lb,
            "color": {"color": "#e2e8f0", "highlight": "#cbd5e1"}, "width": 1.5, "arrows": "to",
            "font": {"size": 8, "face": "Segoe UI", "color": "#94a3b8", "strokeWidth": 2, "strokeColor": "#fff", "align": "horizontal"}})
        eid += 1

for sn, dn, lb in mechanism_to_disaster:
    s = name2id.get(sn)
    # 灾害名可能带 emoji 前缀，尝试多种匹配
    d = name2id.get(dn) or name2id.get(f"⚡ {dn}") or name2id.get(f"🔥 {dn}") or name2id.get(f"🌪️ {dn}") or name2id.get(f"🌊 {dn}")
    if s is not None and d is not None:
        edges.append({"id": eid, "from": s, "to": d, "label": lb,
            "color": {"color": "#cbd5e1", "highlight": "#94a3b8"}, "width": 2, "arrows": "to",
            "font": {"size": 9, "face": "Segoe UI", "color": "#64748b", "strokeWidth": 2, "strokeColor": "#fff", "align": "horizontal"}})
        eid += 1

# ============================================================
# HTML
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
.header{{text-align:center;padding:16px 20px 8px}}
.header h1{{font-size:20px;color:#1e293b;font-weight:700}}
.header p{{font-size:12px;color:#64748b;margin:4px 0 0}}
.btn-row{{display:flex;justify-content:center;gap:6px;padding:8px 12px 12px;flex-wrap:wrap}}
.fbtn{{padding:6px 16px;border-radius:8px;border:2px solid #e2e8f0;background:#fff;
  font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;color:#475569}}
.fbtn:hover{{border-color:#94a3b8}}
.fbtn.on{{border-color:#4f46e5;background:#eef2ff;color:#4f46e5}}
#network{{height:calc(100vh - 108px)}}
.legend{{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);
  display:flex;gap:14px;font-size:11px;color:#64748b;background:#fff;
  padding:6px 16px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);z-index:5}}
.legend i{{display:inline-block;width:10px;height:10px;vertical-align:middle;margin-right:3px}}
.legend .d1{{background:#fca5a5;border-radius:2px}}
.legend .d2{{background:#bbf7d0;border-radius:50%}}
.legend .d3{{background:#ef4444;border-radius:50%}}
.legend .l1{{color:#e2e8f0}} .legend .l2{{color:#cbd5e1}}
</style>
</head>
<body>
<div class="top">
<div class="header">
<h1>🛰️ MAZU 沙特多灾种预警 — 致灾因子·形成机制·灾害类型 关系图谱</h1>
<p>致灾因子 → 形成机制 → 灾害类型 | 点击灾害节点筛选 · 双击空白重置</p>
</div>
<div class="btn-row" id="btnRow">
<button class="fbtn on" onclick="resetFilter()">🌐 全部</button>
<button class="fbtn" data-d="⚡ 暴雨山洪" onclick="pick('⚡ 暴雨山洪')">⚡ 暴雨山洪</button>
<button class="fbtn" data-d="🔥 极端高温" onclick="pick('🔥 极端高温')">🔥 极端高温</button>
<button class="fbtn" data-d="🌪️ 沙尘强风" onclick="pick('🌪️ 沙尘强风')">🌪️ 沙尘强风</button>
<button class="fbtn" data-d="🌊 沿海风浪" onclick="pick('🌊 沿海风浪')">🌊 沿海风浪</button>
</div>
</div>
<div id="network"></div>
<div class="legend">
<span><i class="d1"></i>致灾因子(方框)</span>
<span><i class="d2"></i>形成机制(椭圆)</span>
<span><i class="d3"></i>灾害类型(大圆)</span>
<span class="l2">━━ 因果关系</span>
</div>

<script>
var allNodes = {json.dumps(nodes, ensure_ascii=False)};
var allEdges = {json.dumps(edges, ensure_ascii=False)};
var nodes = new vis.DataSet(allNodes);
var edges = new vis.DataSet(allEdges);

var network = new vis.Network(document.getElementById("network"), {{nodes:nodes,edges:edges}}, {{
  physics: {{
    solver: "forceAtlas2Based",
    forceAtlas2Based: {{ gravitationalConstant: -55, centralGravity: 0.004, springLength: 170, springConstant: 0.025, damping: 0.4 }},
    stabilization: {{ iterations: 200, fit: true }}
  }},
  interaction: {{ hover: true, tooltipDelay: 150, navigationButtons: true, keyboard: true, dragNodes: true }},
  edges: {{ smooth: {{ type: "continuous", roundness: 0.3 }}, font: {{ align: "horizontal" }} }}
}});

// 有向上游邻接（从 to → from：灾害→机制→因子）
var upstream = {{}};
allEdges.forEach(function(e) {{
  if (!upstream[e.to]) upstream[e.to] = [];
  upstream[e.to].push(e.from);
}});

var cur = null;

function pick(label) {{
  var root = null;
  allNodes.forEach(function(n) {{ if (n.label === label) root = n.id; }});
  if (root === null) return;

  // 有向 BFS 从灾害向上游走（灾害→机制→因子），不走向下游的其他灾害
  var vis = new Set([root]);
  var queue = [root];
  while (queue.length) {{
    var v = queue.shift();
    (upstream[v] || []).forEach(function(nb) {{
      if (!vis.has(nb)) {{ vis.add(nb); queue.push(nb); }}
    }});
  }}

  var nu = []; nodes.get().forEach(function(n) {{ nu.push({{id:n.id,hidden:!vis.has(n.id)}}); }});
  nodes.update(nu);
  var eu = []; edges.get().forEach(function(e) {{ eu.push({{id:e.id,hidden:!vis.has(e.from)||!vis.has(e.to)}}); }});
  edges.update(eu);

  document.querySelectorAll(".fbtn").forEach(function(b) {{ b.classList.remove("on"); }});
  var btn = document.querySelector('.fbtn[data-d="'+label+'"]'); if (btn) btn.classList.add("on");
  cur = label;
}}

function resetFilter() {{
  nodes.update(nodes.get().map(function(n) {{ return {{id:n.id,hidden:false}}; }}));
  edges.update(edges.get().map(function(e) {{ return {{id:e.id,hidden:false}}; }}));
  document.querySelectorAll(".fbtn").forEach(function(b) {{ b.classList.remove("on"); }});
  document.querySelector(".fbtn").classList.add("on");
  cur = null;
}}

network.on("click", function(p) {{
  if (p.nodes.length === 1) {{
    var n = nodes.get(p.nodes[0]);
    if (n.group === "disaster") {{ pick(n.label); return; }}
  }}
  if (p.nodes.length === 0 && cur) resetFilter();
}});

network.on("doubleClick", function(p) {{ if (p.nodes.length === 0) resetFilter(); }});
</script>
</body>
</html>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)

n_f = sum(1 for n in nodes if n["group"]=="factor")
n_m = sum(1 for n in nodes if n["group"]=="mechanism")
n_d = sum(1 for n in nodes if n["group"]=="disaster")
print(f"✅ 知识图谱已生成: {OUT}  ({len(HTML):,} chars)")
print(f"   节点: {len(nodes)} ({n_f}因子 + {n_m}机制 + {n_d}灾害)")
print(f"   边:   {len(edges)} 条（因子→机制→灾害，全部含中文关系标签）")
print(f"   筛选: 2-hop子图（覆盖 因子→机制→灾害 完整链路）")

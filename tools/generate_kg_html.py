"""生成独立知识图谱 HTML 文件"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyvis.network import Network
from config.infrastructure import CITIES, WADIS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "knowledge_graph.html")

net = Network(height='100%', width='100%', bgcolor='#f8fafc', font_color='#1e293b', directed=True)
net.set_options("""
{"physics":{"barnesHut":{"gravitationalConstant":-4000,"centralGravity":0.3,"springLength":140,"springConstant":0.02},
 "stabilization":{"iterations":300},"solver":"barnesHut"},
 "edges":{"smooth":{"type":"continuous"}},
 "interaction":{"hover":true,"tooltipDelay":100,"navigationButtons":true,"keyboard":true},
 "layout":{"improvedLayout":true}}
""")

# ---- Layer 0: 灾害类型 ----
disasters = [
    ("山洪", "⚡", "#ef4444", "暴雨山洪：强降水+对流不稳定+地形抬升"),
    ("高温", "🔥", "#f97316", "极端高温：气温异常偏离气候态"),
    ("沙尘", "🌪️", "#eab308", "沙尘强风：强风+干燥+低湿度+裸土"),
    ("风浪", "🌊", "#3b82f6", "沿海风浪：强风+低地形+水汽输送"),
]
for name, icon, color, desc in disasters:
    net.add_node(f"dis_{name}", label=f"{icon} {name}", title=desc,
                 color=color, size=40, shape="dot", borderWidth=3)

# ---- Layer 1: 气象因子 ----
factors = [
    ("CAPE",       "#fca5a5", "对流有效位能：强对流核心驱动"),
    ("相对湿度",    "#93c5fd", "近地面+850hPa相对湿度"),
    ("降水总量",    "#60a5fa", "日降水总量+卫星1h强降水估算"),
    ("风速",        "#a5b4fc", "10m风速+850hPa风速"),
    ("气温距平",    "#fdba74", "最高气温偏离气候态程度"),
    ("水汽输送",    "#86efac", "IVT积分水汽+850hPa水汽通量"),
    ("地形",        "#d4d4d8", "orography+地表气压"),
    ("VPD",         "#fde68a", "饱和水汽压差：大气干燥程度"),
    ("涡度",        "#c4b5fd", "850hPa相对涡度+位势高度"),
    ("风切变",      "#fecaca", "850-300hPa深层风切变"),
]
for name, color, desc in factors:
    net.add_node(f"fac_{name}", label=name, title=desc, color=color, size=26,
                 shape="box", borderWidth=1, font={"size": 12})

factor_edges = [
    ("CAPE", "山洪"), ("CAPE", "沙尘"), ("相对湿度", "山洪"), ("相对湿度", "沙尘"),
    ("相对湿度", "高温"), ("降水总量", "山洪"), ("风速", "沙尘"), ("风速", "风浪"),
    ("气温距平", "高温"), ("水汽输送", "山洪"), ("水汽输送", "风浪"),
    ("地形", "山洪"), ("地形", "风浪"), ("VPD", "高温"), ("VPD", "沙尘"),
    ("涡度", "沙尘"), ("风切变", "沙尘"),
]
for src, dst in factor_edges:
    net.add_edge(f"fac_{src}", f"dis_{dst}", color="#cbd5e1", width=2, arrows="to")

# ---- Layer 2: 城市 ----
city_disaster = {
    "利雅得": ["高温", "沙尘"], "吉达": ["山洪", "风浪"], "麦加": ["山洪", "高温"],
    "达曼": ["高温", "沙尘", "风浪"], "艾卜哈": ["山洪"], "塔伊夫": ["山洪"],
    "哈伊勒": ["沙尘"], "布赖代": ["沙尘"], "朱拜勒": ["风浪"], "纳季兰": ["山洪"],
    "麦地那": ["高温"], "胡富夫": ["高温", "沙尘"], "延布": ["风浪"],
    "吉赞": ["山洪"], "塔布克": ["沙尘"], "阿尔阿尔": ["沙尘"],
}
added = set()
for c in CITIES:
    cname = c["name"].split("(")[0].strip()
    if cname in city_disaster:
        added.add(cname)
        pop_str = f"{c['pop']:,}"
        net.add_node(f"city_{cname}", label=cname,
                     title=f"人口: {pop_str} | 类型: {c['type']}",
                     color="#475569", size=16, shape="dot", borderWidth=1)

for cname, dlist in city_disaster.items():
    if cname in added:
        for d in dlist:
            net.add_edge(f"city_{cname}", f"dis_{d}", color="#e2e8f0",
                         width=1, dashes=True, arrows="to")

# ---- Layer 3: Wadi ----
for w in WADIS:
    net.add_node(f"wadi_{w['name']}", label=f"🏞️ {w['name']}",
                 title=f"{w['length_km']}km | 山洪风险区",
                 color="#06b6d4", size=14, shape="diamond", borderWidth=1)
    net.add_edge(f"wadi_{w['name']}", "dis_山洪", color="#a5f3fc", width=1.5, arrows="to")

# ---- 生成 HTML ----
html = net.generate_html()
# 移除 pyvis 生成的本地 lib/bindings/utils.js 引用（文件不存在，改用 CDN）
html = html.replace('<script src="lib/bindings/utils.js"></script>\n            ', '')

HEADER = """</head>
<style>
body{margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:#f8fafc}
.header{text-align:center;padding:24px 20px 16px;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.header h1{font-size:22px;color:#1e293b;margin:0;font-weight:700}
.header p{font-size:13px;color:#64748b;margin:6px 0 0}
.legend{display:flex;justify-content:center;gap:20px;flex-wrap:wrap;padding:10px;font-size:12px;color:#475569}
.legend span{display:flex;align-items:center;gap:5px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.box{width:10px;height:10px;display:inline-block}
.diamond{width:8px;height:8px;transform:rotate(45deg);display:inline-block}
#mynetwork{height:calc(100vh - 110px)!important}
</style>
<body>
<div class="header">
<h1>🛰️ MAZU 沙特多灾种预警 — 知识图谱</h1>
<p>气象因子 → 灾害类型 → 城市承灾体 → Wadi水系 | 可拖拽 · 缩放 · 悬停查看详情</p>
</div>
<div class="legend">
<span><span class="dot" style="background:#ef4444"></span>灾害类型</span>
<span><span class="box" style="background:#fca5a5"></span>气象因子</span>
<span><span class="dot" style="background:#475569"></span>城市</span>
<span><span class="diamond" style="background:#06b6d4"></span>Wadi水系</span>
<span>── 直接影响</span>
<span>- - 间接关联</span>
</div>
"""

html = html.replace("</head>", HEADER)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"✅ 知识图谱已生成: {OUT}  ({len(html):,} chars)")

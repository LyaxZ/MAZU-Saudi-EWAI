"""生成知识图谱 HTML — 多语言版 (zh/en/ar)"""
import sys, os, json

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

# ============================================================
# 多语言翻译
# ============================================================
I18N = {
    "zh": {
        "title": "MAZU 沙特多灾种预警 — 致灾因子·形成机制·灾害类型 关系图谱",
        "subtitle": "致灾因子 → 形成机制 → 灾害类型 | 点击灾害节点筛选 · 双击空白重置",
        "all": "全部",
        "legend_factor": "致灾因子(方框)",
        "legend_mech": "形成机制(椭圆)",
        "legend_disaster": "灾害类型(大圆)",
        "legend_edge": "━━ 因果关系",
        "factors": [
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
        ],
        "mechanisms": [
            ("强对流发展",    "#fecaca", "CAPE高 + CIN低 → 深厚对流 → 暴雨"),
            ("水汽辐合",      "#bbf7d0", "水汽输送汇聚 + 湿度高 → 强降水"),
            ("大气干燥化",    "#fde68a", "VPD高 + 湿度低 + 高温 → 极端干旱/沙尘"),
            ("强风动力",      "#c7d2fe", "风速大 + 涡度强 → 起沙 / 波浪"),
            ("热力异常",      "#fed7aa", "气温距平大 + 地表高温 → 高温热浪"),
            ("地形强迫",      "#d4d4d8", "地形抬升 + 水汽 → 迎风坡强降水"),
        ],
        "disasters": [
            ("⚡暴雨山洪", "#ef4444", "强降水 + 对流不稳定 + 地形抬升 → 突发性洪水"),
            ("🔥极端高温", "#f97316", "气温异常偏离气候态 → 持续高温热浪"),
            ("🌪️沙尘强风", "#eab308", "强风 + 干燥地表 + 低湿度 → 沙尘暴"),
            ("🌊沿海风浪", "#3b82f6", "强风 + 低地形 + 水汽 → 风暴潮/大浪"),
        ],
        "edges_fm": [
            ("对流有效位能\n(CAPE)",   "强对流发展",   "提供能量"),
            ("对流抑制能量\n(CIN)",     "强对流发展",   "抑制减弱"),
            ("深层风切变",              "强对流发展",   "组织维持"),
            ("日降水总量",              "水汽辐合",     "降水实况"),
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
        ],
        "edges_md": [
            ("强对流发展",  "⚡暴雨山洪", "主要成因"),
            ("水汽辐合",    "⚡暴雨山洪", "直接触发"),
            ("地形强迫",    "⚡暴雨山洪", "山地增强"),
            ("热力异常",    "🔥极端高温", "直接成因"),
            ("大气干燥化",  "🔥极端高温", "加剧作用"),
            ("大气干燥化",  "🌪️沙尘强风", "起沙条件"),
            ("强风动力",    "🌪️沙尘强风", "动力驱动"),
            ("热力异常",    "🌪️沙尘强风", "热力湍流"),
            ("强风动力",    "🌊沿海风浪", "波浪驱动"),
            ("水汽辐合",    "🌊沿海风浪", "水汽输送"),
            ("地形强迫",    "🌊沿海风浪", "海岸效应"),
        ],
    },
    "en": {
        "title": "MAZU Saudi Early Warning — Hazard Factors · Mechanisms · Disaster Types",
        "subtitle": "Factors → Mechanisms → Disasters | Click disaster to filter · Double-click to reset",
        "all": "All",
        "legend_factor": "Hazard Factors (Box)",
        "legend_mech": "Mechanisms (Ellipse)",
        "legend_disaster": "Disaster Types (Circle)",
        "legend_edge": "── Causal Link",
        "factors": [
            ("CAPE\n(Convective Potential)",   "#fca5a5", "Core driver of strong convection; higher = more thunderstorms"),
            ("Near-Surface\nRel. Humidity",     "#93c5fd", "Low-level moisture; directly affects precipitation efficiency"),
            ("Daily Total\nPrecipitation",       "#60a5fa", "24h accumulated precip; direct flash flood trigger"),
            ("10m Wind\nSpeed",                  "#a5b4fc", "Near-surface wind; driver of dust & waves"),
            ("Temperature\nAnomaly",             "#fdba74", "Departure from climatology; core heatwave indicator"),
            ("Moisture Flux\nDivergence",        "#86efac", "Moisture convergence/divergence; determines rain zones"),
            ("Orography\n(Elevation)",           "#d4d4d8", "Orographic uplift; windward slopes enhance rain"),
            ("VPD\n(Vapor Pressure Deficit)",    "#fde68a", "Atmospheric dryness; higher VPD = drier air"),
            ("850hPa\nVorticity",                "#c4b5fd", "Low-level cyclonic rotation; dust storm dynamics"),
            ("Deep Layer\nWind Shear",           "#fecaca", "850-300hPa wind difference; convective organization"),
            ("CIN\n(Convective Inhibition)",     "#e2e8f0", "Energy inhibiting convection; lower CIN = easier trigger"),
            ("Surface\nTemperature",             "#fed7aa", "Surface thermal state; affects near-surface stability"),
        ],
        "mechanisms": [
            ("Strong Convection", "#fecaca", "High CAPE + Low CIN → Deep convection → Heavy rain"),
            ("Moisture Convergence", "#bbf7d0", "Moisture flux + High humidity → Heavy precipitation"),
            ("Atmospheric Drying", "#fde68a", "High VPD + Low humidity + Heat → Drought / Dust"),
            ("Strong Wind Dynamics", "#c7d2fe", "Strong wind + Vorticity → Dust lofting / Waves"),
            ("Thermal Anomaly", "#fed7aa", "Large T anomaly + Hot surface → Heatwave"),
            ("Orographic Forcing", "#d4d4d8", "Uplift + Moisture → Windward heavy rain"),
        ],
        "disasters": [
            ("⚡Flash Flood", "#ef4444", "Heavy rain + Convective instability + Orographic uplift → Flash flood"),
            ("🔥Extreme Heat", "#f97316", "Temperature anomaly → Persistent heatwave"),
            ("🌪️Dust Storm", "#eab308", "Strong wind + Dry surface + Low humidity → Dust storm"),
            ("🌊Coastal Wave", "#3b82f6", "Strong wind + Low terrain + Moisture → Storm surge"),
        ],
        "edges_fm": [
            ("对流有效位能\n(CAPE)",   "强对流发展",   "Provides energy"),
            ("对流抑制能量\n(CIN)",     "强对流发展",   "Weakens inhibition"),
            ("深层风切变",              "强对流发展",   "Organizes & sustains"),
            ("日降水总量",              "水汽辐合",     "Precipitation status"),
            ("水汽通量散度",            "水汽辐合",     "Moisture convergence"),
            ("近地面相对湿度",          "水汽辐合",     "Humidity condition"),
            ("饱和水汽压差\n(VPD)",    "大气干燥化",   "Dryness level"),
            ("气温距平",                "大气干燥化",   "Intensifies heat"),
            ("地表温度",                "大气干燥化",   "Surface heating"),
            ("近地面相对湿度",          "大气干燥化",   "Reduces humidity"),
            ("10米风速",                "强风动力",     "Direct driver"),
            ("850hPa 涡度",             "强风动力",     "Enhances rotation"),
            ("气温距平",                "热力异常",     "Deviation degree"),
            ("地表温度",                "热力异常",     "Surface heating"),
            ("地形高度",                "地形强迫",     "Uplift effect"),
            ("水汽通量散度",            "地形强迫",     "Windward moisture"),
        ],
        "edges_md": [
            ("强对流发展",  "⚡暴雨山洪", "Primary cause"),
            ("水汽辐合",    "⚡暴雨山洪", "Direct trigger"),
            ("地形强迫",    "⚡暴雨山洪", "Mountain enhancement"),
            ("热力异常",    "🔥极端高温", "Direct cause"),
            ("大气干燥化",  "🔥极端高温", "Intensifying effect"),
            ("大气干燥化",  "🌪️沙尘强风", "Dust lofting condition"),
            ("强风动力",    "🌪️沙尘强风", "Dynamic driver"),
            ("热力异常",    "🌪️沙尘强风", "Thermal turbulence"),
            ("强风动力",    "🌊沿海风浪", "Wave driver"),
            ("水汽辐合",    "🌊沿海风浪", "Moisture transport"),
            ("地形强迫",    "🌊沿海风浪", "Coastal effect"),
        ],
    },
    "ar": {
        "title": "MAZU للإنذار المبكر — عوامل الخطر · آليات التكون · أنواع الكوارث",
        "subtitle": "عوامل الخطر → آليات التكون → أنواع الكوارث | انقر على الكارثة للتصفية · انقر مرتين للإعادة",
        "all": "الكل",
        "legend_factor": "عوامل الخطر (مربع)",
        "legend_mech": "آليات التكون (بيضاوي)",
        "legend_disaster": "أنواع الكوارث (دائرة)",
        "legend_edge": "── علاقة سببية",
        "factors": [
            ("CAPE\n(طاقة الحمل)",            "#fca5a5", "المحرك الأساسي للحمل القوي"),
            ("الرطوبة النسبية\nبالقرب من السطح","#93c5fd", "رطوبة الطبقة السفلى"),
            ("إجمالي الهطول\nاليومي",           "#60a5fa", "هطول 24 ساعة متراكم"),
            ("سرعة الرياح\n10م",                "#a5b4fc", "رياح قرب السطح"),
            ("شذوذ درجة\nالحرارة",             "#fdba74", "انحراف عن المناخ"),
            ("تباعد تدفق\nالرطوبة",            "#86efac", "تقارب/تباعد الرطوبة"),
            ("الارتفاع\nالطبوغرافي",           "#d4d4d8", "رفع طبوغرافي"),
            ("VPD\n(فرق ضغط البخار)",          "#fde68a", "مؤشر جفاف الجو"),
            ("دوامية\n850hPa",                 "#c4b5fd", "دوران إعصاري منخفض"),
            ("قص الرياح\nالعميق",              "#fecaca", "فرق رياح 850-300hPa"),
            ("CIN\n(تثبيط الحمل)",             "#e2e8f0", "طاقة تثبيط الحمل"),
            ("درجة حرارة\nالسطح",              "#fed7aa", "حالة حرارية للسطح"),
        ],
        "mechanisms": [
            ("تطور الحمل القوي",  "#fecaca", "حمل عميق → أمطار غزيرة"),
            ("تقارب الرطوبة",     "#bbf7d0", "رطوبة عالية → هطول غزير"),
            ("جفاف الغلاف الجوي", "#fde68a", "جفاف/غبار"),
            ("ديناميكيات الرياح", "#c7d2fe", "تطاير غبار/أمواج"),
            ("شذوذ حراري",        "#fed7aa", "موجة حر"),
            ("إجبار طبوغرافي",    "#d4d4d8", "هطول على المنحدرات"),
        ],
        "disasters": [
            ("فيضانات مفاجئة⚡", "#ef4444", "فيضان مفاجئ"),
            ("حرارة شديدة🔥",    "#f97316", "موجة حر مستمرة"),
            ("عاصفة ترابية🌪️",   "#eab308", "عاصفة ترابية"),
            ("أمواج ساحلية🌊",    "#3b82f6", "عرام عاصفي"),
        ],
        "edges_fm": [
            ("对流有效位能\n(CAPE)",   "强对流发展",   "يوفر طاقة"),
            ("对流抑制能量\n(CIN)",     "强对流发展",   "يضعف التثبيط"),
            ("深层风切变",              "强对流发展",   "ينظم ويحافظ"),
            ("日降水总量",              "水汽辐合",     "حالة الهطول"),
            ("水汽通量散度",            "水汽辐合",     "تقارب الرطوبة"),
            ("近地面相对湿度",          "水汽辐合",     "حالة الرطوبة"),
            ("饱和水汽压差\n(VPD)",    "大气干燥化",   "مستوى الجفاف"),
            ("气温距平",                "大气干燥化",   "يفاقم الحرارة"),
            ("地表温度",                "大气干燥化",   "تسخين السطح"),
            ("近地面相对湿度",          "大气干燥化",   "يقلل الرطوبة"),
            ("10米风速",                "强风动力",     "محرك مباشر"),
            ("850hPa 涡度",             "强风动力",     "يعزز الدوران"),
            ("气温距平",                "热力异常",     "درجة الانحراف"),
            ("地表温度",                "热力异常",     "تسخين السطح"),
            ("地形高度",                "地形强迫",     "تأثير الرفع"),
            ("水汽通量散度",            "地形强迫",     "رطوبة مواجهة للرياح"),
        ],
        "edges_md": [
            ("强对流发展",  "⚡暴雨山洪", "سبب رئيسي"),
            ("水汽辐合",    "⚡暴雨山洪", "محفز مباشر"),
            ("地形强迫",    "⚡暴雨山洪", "تعزيز جبلي"),
            ("热力异常",    "🔥极端高温", "سبب مباشر"),
            ("大气干燥化",  "🔥极端高温", "تأثير مضاعف"),
            ("大气干燥化",  "🌪️沙尘强风", "شرط تطاير الغبار"),
            ("强风动力",    "🌪️沙尘强风", "محرك ديناميكي"),
            ("热力异常",    "🌪️沙尘强风", "اضطراب حراري"),
            ("强风动力",    "🌊沿海风浪", "محرك الأمواج"),
            ("水汽辐合",    "🌊沿海风浪", "نقل الرطوبة"),
            ("地形强迫",    "🌊沿海风浪", "تأثير ساحلي"),
        ],
    },
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="{lang_code}">
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
<h1>🛰️ {title}</h1>
<p>{subtitle}</p>
</div>
<div class="btn-row" id="btnRow">
<button class="fbtn on" onclick="resetFilter()">{all_prefix}{all_btn}{all_suffix}</button>
{disaster_btns}
</div>
</div>
<div id="network"></div>
<div class="legend">
<span><i class="d1"></i>{legend_factor}</span>
<span><i class="d2"></i>{legend_mech}</span>
<span><i class="d3"></i>{legend_disaster}</span>
<span class="l2">{legend_edge}</span>
</div>

<script>
var allNodes = {nodes_json};
var allEdges = {edges_json};
var nodes, edges;
var upstream = {{}};
allEdges.forEach(function(e) {{
  if (!upstream[e.to]) upstream[e.to] = [];
  upstream[e.to].push(e.from);
}});
var container = document.getElementById("network");
var network = new vis.Network(container,
  {{nodes: new vis.DataSet(allNodes), edges: new vis.DataSet(allEdges)}},
  {{ physics: {{ solver:"forceAtlas2Based", forceAtlas2Based:{{gravitationalConstant:-40,centralGravity:.005,springLength:160}} }},
     interaction: {{ hover:true, tooltipDelay:200 }},
     edges: {{ smooth:{{type:"curvedCW",roundness:.15}} }} }});
nodes = network.body.data.nodes; edges = network.body.data.edges;
var cur = null;
function pick(label) {{
  var root = null;
  allNodes.forEach(function(n) {{ if (n.label === label) root = n.id; }});
  if (root === null) return;
  var vis = new Set([root]), queue = [root];
  while (queue.length) {{
    var v = queue.shift();
    (upstream[v] || []).forEach(function(nb) {{ if (!vis.has(nb)) {{ vis.add(nb); queue.push(nb); }} }});
  }}
  nodes.update(nodes.get().map(function(n) {{ return {{id:n.id,hidden:!vis.has(n.id)}}; }}));
  edges.update(edges.get().map(function(e) {{ return {{id:e.id,hidden:!vis.has(e.from)||!vis.has(e.to)}}; }}));
  document.querySelectorAll(".fbtn").forEach(function(b) {{ b.classList.remove("on"); }});
  var btn = document.querySelector('.fbtn[data-d="' + label.replace(/"/g,'&quot;') + '"]');
  if (btn) btn.classList.add("on");
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


def generate(lang="zh"):
    t = I18N[lang]
    lang_code = {"zh": "zh-CN", "en": "en", "ar": "ar"}[lang]
    zh_t = I18N["zh"]

    nodes, edges = [], []
    nid, eid = 0, 0

    # factor nodes
    for name, color, desc in t["factors"]:
        nodes.append({"id": nid, "label": name, "title": desc,
            "color": {"background": color, "border": "#94a3b8"},
            "shape": "box", "size": 28, "font": {"size": 11, "face": "Segoe UI"},
            "group": "factor"})
        nid += 1

    # mechanism nodes
    for name, color, desc in t["mechanisms"]:
        nodes.append({"id": nid, "label": name, "title": desc,
            "color": {"background": color, "border": "#94a3b8"},
            "shape": "ellipse", "size": 20, "font": {"size": 11, "face": "Segoe UI", "color": "#475569"},
            "group": "mechanism"})
        nid += 1

    # disaster nodes
    for name, color, desc in t["disasters"]:
        nodes.append({"id": nid, "label": name, "title": desc,
            "color": {"background": color, "border": "#ef4444"},
            "shape": "dot", "size": 36, "font": {"size": 13, "face": "Segoe UI", "color": "#1e293b", "bold": True},
            "group": "disaster"})
        nid += 1

    # Build name→id map using Chinese names (ZH) for edge connectivity
    name2id = {}
    for i, (name, _, _) in enumerate(zh_t["factors"]):
        name2id[name] = i
    off = len(zh_t["factors"])
    for i, (name, _, _) in enumerate(zh_t["mechanisms"]):
        name2id[name] = off + i
    off += len(zh_t["mechanisms"])
    for i, (name, _, _) in enumerate(zh_t["disasters"]):
        name2id[name] = off + i

    # Edges: factor → mechanism
    for fn, mn, lb in t["edges_fm"]:
        s, d = name2id.get(fn), name2id.get(mn)
        if s is not None and d is not None:
            edges.append({"id": eid, "from": s, "to": d, "label": lb,
                "color": {"color": "#e2e8f0", "highlight": "#cbd5e1"}, "width": 1.5, "arrows": "to",
                "font": {"size": 8, "face": "Segoe UI", "color": "#94a3b8", "strokeWidth": 2, "strokeColor": "#fff", "align": "horizontal"}})
            eid += 1

    # Edges: mechanism → disaster
    for mn, dn, lb in t["edges_md"]:
        s, d = name2id.get(mn), name2id.get(dn)
        if s is not None and d is not None:
            edges.append({"id": eid, "from": s, "to": d, "label": lb,
                "color": {"color": "#cbd5e1", "highlight": "#94a3b8"}, "width": 2, "arrows": "to",
                "font": {"size": 9, "face": "Segoe UI", "color": "#64748b", "strokeWidth": 2, "strokeColor": "#fff", "align": "horizontal"}})
            eid += 1

    # Filter buttons
    all_prefix = "" if lang == "ar" else "🌐 "
    all_suffix = " 🌐" if lang == "ar" else ""
    disaster_btns = "\n".join(
        f'<button class="fbtn" data-d="{name}" onclick="pick(\'{name}\')">{name}</button>'
        for name, _, _ in t["disasters"]
    )

    html = HTML_TEMPLATE.format(
        lang_code=lang_code,
        title=t["title"], subtitle=t["subtitle"],
        all_prefix=all_prefix, all_btn=t["all"], all_suffix=all_suffix,
        disaster_btns=disaster_btns,
        legend_factor=t["legend_factor"], legend_mech=t["legend_mech"],
        legend_disaster=t["legend_disaster"], legend_edge=t["legend_edge"],
        nodes_json=json.dumps(nodes, ensure_ascii=False),
        edges_json=json.dumps(edges, ensure_ascii=False),
    )

    out_path = os.path.join(PROJ, "outputs", f"knowledge_graph_{lang}.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    n_f = sum(1 for n in nodes if n["group"]=="factor")
    n_m = sum(1 for n in nodes if n["group"]=="mechanism")
    n_d = sum(1 for n in nodes if n["group"]=="disaster")
    print(f"✅ knowledge_graph_{lang}.html ({len(html):,} chars) — {len(nodes)} nodes ({n_f}f+{n_m}m+{n_d}d), {len(edges)} edges")


if __name__ == "__main__":
    for lang in ("zh", "en", "ar"):
        generate(lang)
    import shutil
    shutil.copy(os.path.join(PROJ, "outputs", "knowledge_graph_zh.html"),
                os.path.join(PROJ, "outputs", "knowledge_graph.html"))
    print("✅ knowledge_graph.html (default = zh)")

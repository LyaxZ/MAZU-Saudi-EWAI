"""KG / Events HTML 多语言后处理 — 基于中文版做文本替换生成 en/ar 版本"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.i18n import OUTPUTS_DIR

# ============================================================
# KG 知识图谱翻译映射（中文 → English / Arabic）
# ============================================================
KG_TRANSLATIONS = {
    "en": {
        # 页面标题 + 图例
        "MAZU 沙特多灾种预警 —": "MAZU Saudi Multi-Hazard Early Warning —",
        "关系图谱": "Knowledge Graph",
        "点击灾害节点筛选 · 双击空白重置": "Click disaster nodes to filter · Double-click blank to reset",
        "全部": "All",
        "(方框)": "(Box)",
        "(椭圆)": "(Ellipse)",
        "(大圆)": "(Large Circle)",
        "━━ 因果关系": "── Causal Relation",
        "致灾因子": "Hazard Factors",
        "形成机制": "Formation Mechanisms",
        "灾害类型": "Disaster Types",
        # 因子节点
        "对流有效位能\n(CAPE)": "CAPE\n(Convective Available\nPotential Energy)",
        "近地面相对湿度": "Near-Surface\nRelative Humidity",
        "日降水总量": "Daily Total\nPrecipitation",
        "10米风速": "10m Wind\nSpeed",
        "气温距平": "Temperature\nAnomaly",
        "水汽通量散度": "Moisture Flux\nDivergence",
        "地形高度": "Orography\n(Elevation)",
        "饱和水汽压差\n(VPD)": "VPD\n(Vapor Pressure\nDeficit)",
        "850hPa 涡度": "850hPa\nVorticity",
        "深层风切变": "Deep Layer\nWind Shear",
        "对流抑制能量\n(CIN)": "CIN\n(Convective\nInhibition)",
        "地表温度": "Surface\nTemperature",
        # 机制节点
        "强对流发展": "Strong Convection\nDevelopment",
        "水汽辐合": "Moisture\nConvergence",
        "大气干燥化": "Atmospheric\nDrying",
        "强风动力": "Strong Wind\nDynamics",
        "热力异常": "Thermal\nAnomaly",
        "地形强迫": "Orographic\nForcing",
        # 灾害节点
        "暴雨山洪": "Flash Flood\n& Torrential Rain",
        "极端高温": "Extreme\nHeat",
        "沙尘强风": "Dust Storm\n& Strong Wind",
        "沿海风浪": "Coastal\nWave Surge",
        # 边标签
        "提供能量": "Provides energy",
        "抑制减弱": "Weakens inhibition",
        "组织维持": "Organizes & sustains",
        "降水实况": "Precipitation status",
        "水汽汇聚": "Moisture convergence",
        "湿度条件": "Humidity condition",
        "干燥程度": "Dryness level",
        "高温加剧": "Intensifies heat",
        "地表加热": "Surface heating",
        "湿度降低": "Reduces humidity",
        "直接动力": "Direct driver",
        "旋转增强": "Enhances rotation",
        "偏离程度": "Deviation degree",
        "抬升作用": "Uplift effect",
        "迎风水汽": "Windward moisture",
        "主要成因": "Primary cause",
        "直接触发": "Direct trigger",
        "山地增强": "Mountain enhancement",
        "直接成因": "Direct cause",
        "加剧作用": "Intensifying effect",
        "起沙条件": "Dust lofting condition",
        "动力驱动": "Dynamic driver",
        "热力湍流": "Thermal turbulence",
        "波浪驱动": "Wave driver",
        "水汽输送": "Moisture transport",
        "海岸效应": "Coastal effect",
        # tooltip 描述
        "强对流核心驱动因子，数值越高越有利于雷暴发展": "Core driver of strong convection; higher values favor thunderstorm development",
        "低层大气水汽含量，直接影响降水效率": "Low-level atmospheric moisture content; directly affects precipitation efficiency",
        "24小时累积降水量，山洪直接触发条件": "24-hour accumulated precipitation; direct trigger for flash floods",
        "近地面风速，沙尘起沙和风浪的直接动力": "Near-surface wind speed; direct driver for dust lofting and waves",
        "当日气温偏离气候态的程度，热浪核心指标": "Degree of temperature departure from climatology; core heatwave indicator",
        "水汽输送的汇聚/辐散程度，决定降水落区": "Degree of moisture flux convergence/divergence; determines precipitation zones",
        "地形抬升强迫，迎风坡增强降水": "Orographic uplift forcing; windward slopes enhance precipitation",
        "大气干燥程度指标，VPD越高空气越干": "Atmospheric dryness indicator; higher VPD means drier air",
        "低层气旋式旋转，沙尘天气的动力条件": "Low-level cyclonic rotation; dynamic condition for dust weather",
        "850-300hPa风速差异，对流组织化条件": "850-300hPa wind speed difference; condition for convective organization",
        "抑制对流的能量，CIN越小越容易触发": "Energy inhibiting convection; lower CIN means easier triggering",
        "地表热力状况，影响近地面稳定度": "Surface thermal condition; affects near-surface stability",
        "CAPE高 + CIN低 → 深厚对流 → 暴雨": "High CAPE + Low CIN → Deep convection → Heavy rain",
        "水汽输送汇聚 + 湿度高 → 强降水": "Moisture convergence + High humidity → Heavy precipitation",
        "VPD高 + 湿度低 + 高温 → 极端干旱/沙尘": "High VPD + Low humidity + High temperature → Extreme drought/dust",
        "风速大 + 涡度强 → 起沙 / 波浪": "Strong wind + High vorticity → Dust lofting / Waves",
        "气温距平大 + 地表高温 → 高温热浪": "Large temperature anomaly + High surface temperature → Heatwave",
        "地形抬升 + 水汽 → 迎风坡强降水": "Orographic uplift + Moisture → Windward heavy precipitation",
        "强降水 + 对流不稳定 + 地形抬升 → 突发性洪水": "Heavy rain + Convective instability + Orographic uplift → Flash flood",
        "气温异常偏离气候态 → 持续高温热浪": "Temperature anomaly departure from climatology → Persistent heatwave",
        "强风 + 干燥地表 + 低湿度 → 沙尘暴": "Strong wind + Dry surface + Low humidity → Dust storm",
        "强风 + 低地形 + 水汽 → 风暴潮/大浪": "Strong wind + Low terrain + Moisture → Storm surge/large waves",
    },
    "ar": {
        # 页面标题 + 图例
        "MAZU 沙特多灾种预警 —": "MAZU للإنذار المبكر —",
        "关系图谱": "الرسم البياني المعرفي",
        "点击灾害节点筛选 · 双击空白重置": "انقر على عقد الكوارث للتصفية · انقر مرتين للإعادة",
        "全部": "الكل",
        "(方框)": "(مربع)",
        "(椭圆)": "(بيضاوي)",
        "(大圆)": "(دائرة كبيرة)",
        "━━ 因果关系": "── علاقة سببية",
        "致灾因子 → 形成机制 → 灾害类型 知识图谱": "عوامل الخطر → آليات التكون → أنواع الكوارث",
        "点击节点查看详情 | 点击灾害筛选 | 双击空白重置": "انقر على العقد للتفاصيل | انقر على الكوارث للتصفية | انقر مرتين للإعادة",
        "致灾因子": "عوامل الخطر",
        "形成机制": "آليات التكون",
        "灾害类型": "أنواع الكوارث",
        "对流有效位能\n(CAPE)": "CAPE\n(طاقة الحمل\nالمتاحة)",
        "近地面相对湿度": "الرطوبة النسبية\nبالقرب من السطح",
        "日降水总量": "إجمالي الهطول\nاليومي",
        "10米风速": "سرعة الرياح\nعلى 10م",
        "气温距平": "شذوذ درجة\nالحرارة",
        "水汽通量散度": "تباعد تدفق\nالرطوبة",
        "地形高度": "الارتفاع\nالطبوغرافي",
        "饱和水汽压差\n(VPD)": "VPD\n(فرق ضغط\nالبخار)",
        "850hPa 涡度": "دوامية\n850hPa",
        "深层风切变": "قص الرياح\nالعميق",
        "对流抑制能量\n(CIN)": "CIN\n(طاقة تثبيط\nالحمل)",
        "地表温度": "درجة حرارة\nالسطح",
        "强对流发展": "تطور الحمل\nالقوي",
        "水汽辐合": "تقارب\nالرطوبة",
        "大气干燥化": "جفاف\nالغلاف الجوي",
        "强风动力": "ديناميكيات\nالرياح القوية",
        "热力异常": "شذوذ\nحراري",
        "地形强迫": "إجبار\nطبوغرافي",
        "暴雨山洪": "فيضانات مفاجئة\nوأمطار غزيرة",
        "极端高温": "حرارة\nشديدة",
        "沙尘强风": "عواصف ترابية\nورياح قوية",
        "沿海风浪": "أمواج\nساحلية",
        "提供能量": "يوفر طاقة",
        "抑制减弱": "يضعف التثبيط",
        "组织维持": "ينظم ويحافظ",
        "降水实况": "حالة الهطول",
        "水汽汇聚": "تقارب الرطوبة",
        "湿度条件": "حالة الرطوبة",
        "干燥程度": "مستوى الجفاف",
        "高温加剧": "يفاقم الحرارة",
        "地表加热": "تسخين السطح",
        "湿度降低": "يقلل الرطوبة",
        "直接动力": "محرك مباشر",
        "旋转增强": "يعزز الدوران",
        "偏离程度": "درجة الانحراف",
        "抬升作用": "تأثير الرفع",
        "迎风水汽": "رطوبة مواجهة للرياح",
        "主要成因": "سبب رئيسي",
        "直接触发": "محفز مباشر",
        "山地增强": "تعزيز جبلي",
        "直接成因": "سبب مباشر",
        "加剧作用": "تأثير مضاعف",
        "起沙条件": "شرط تطاير الغبار",
        "动力驱动": "محرك ديناميكي",
        "热力湍流": "اضطراب حراري",
        "波浪驱动": "محرك الأمواج",
        "水汽输送": "نقل الرطوبة",
        "海岸效应": "تأثير ساحلي",
        "强对流核心驱动因子，数值越高越有利于雷暴发展": "المحرك الأساسي للحمل القوي؛ القيم الأعلى تساعد على تطور العواصف الرعدية",
        "低层大气水汽含量，直接影响降水效率": "محتوى رطوبة الغلاف الجوي السفلي؛ يؤثر مباشرة على كفاءة الهطول",
        "24小时累积降水量，山洪直接触发条件": "هطول متراكم 24 ساعة؛ شرط مباشر للفيضانات المفاجئة",
        "近地面风速，沙尘起沙和风浪的直接动力": "سرعة الرياح قرب السطح؛ محرك مباشر لتطاير الغبار والأمواج",
        "当日气温偏离气候态的程度，热浪核心指标": "درجة انحراف الحرارة عن المناخ؛ مؤشر أساسي لموجات الحر",
        "水汽输送的汇聚/辐散程度，决定降水落区": "درجة تقارب/تباعد تدفق الرطوبة؛ يحدد مناطق الهطول",
        "地形抬升强迫，迎风坡增强降水": "إجبار الرفع الطبوغرافي؛ المنحدرات المواجهة للرياح تعزز الهطول",
        "大气干燥程度指标，VPD越高空气越干": "مؤشر جفاف الغلاف الجوي؛ كلما زاد VPD زاد جفاف الهواء",
        "低层气旋式旋转，沙尘天气的动力条件": "دوران إعصاري منخفض؛ حالة ديناميكية للطقس الترابي",
        "850-300hPa风速差异，对流组织化条件": "فرق سرعة الرياح 850-300hPa؛ شرط لتنظيم الحمل",
        "抑制对流的能量，CIN越小越容易触发": "طاقة تثبيط الحمل؛ كلما قل CIN سهل التحفيز",
        "地表热力状况，影响近地面稳定度": "الحالة الحرارية للسطح؛ تؤثر على استقرار الطبقة القريبة",
        "CAPE高 + CIN低 → 深厚对流 → 暴雨": "CAPE عالي + CIN منخفض → حمل عميق → أمطار غزيرة",
        "水汽输送汇聚 + 湿度高 → 强降水": "تقارب الرطوبة + رطوبة عالية → هطول غزير",
        "VPD高 + 湿度低 + 高温 → 极端干旱/沙尘": "VPD عالي + رطوبة منخفضة + حرارة عالية → جفاف شديد/غبار",
        "风速大 + 涡度强 → 起沙 / 波浪": "رياح قوية + دوامية عالية → تطاير غبار / أمواج",
        "气温距平大 + 地表高温 → 高温热浪": "شذوذ حراري كبير + حرارة سطح عالية → موجة حر",
        "地形抬升 + 水汽 → 迎风坡强降水": "رفع طبوغرافي + رطوبة → هطول غزير على المنحدرات",
        "强降水 + 对流不稳定 + 地形抬升 → 突发性洪水": "أمطار غزيرة + عدم استقرار حملي + رفع طبوغرافي → فيضان مفاجئ",
        "气温异常偏离气候态 → 持续高温热浪": "انحراف حراري عن المناخ → موجة حر مستمرة",
        "强风 + 干燥地表 + 低湿度 → 沙尘暴": "رياح قوية + سطح جاف + رطوبة منخفضة → عاصفة ترابية",
        "强风 + 低地形 + 水汽 → 风暴潮/大浪": "رياح قوية + تضاريس منخفضة + رطوبة → عرام عاصفي/أمواج كبيرة",
    },
}

# ============================================================
# Events 历史事件翻译映射
# ============================================================
EVENTS_TRANSLATIONS = {
    "en": {
        # 页面/UI
        "Ground Truth 灾害事件可视化": "Ground Truth Disaster Event Visualization",
        "真实灾害事件": "Ground Truth Events",
        "下拉切换": "dropdown to switch",
        "个事件": "events",
        "经度": "Longitude",
        "纬度": "Latitude",
        "海拔 (m)": "Elevation (m)",
        "气温 (°C)": "Temperature (°C)",
        "风速 (m/s)": "Wind Speed (m/s)",
        "海拔": "Elevation",
        "气温": "Temperature",
        "风速": "Wind Speed",
        "最高温": "Max Temp",
        "最高风速": "Max Wind Speed",
        "极端高温区": "Extreme Heat Zone",
        "强风区": "Strong Wind Zone",
        "棕色覆盖": "Brown coverage",
        "红色覆盖": "Red coverage",
        "受影响": "Affected",
        "风险源头": "Risk source",
        "源头": "Source",
        "节点": "Node",
        "子图": "Sub-graph",
        "全国": "Nationwide",
        "东部": "Eastern",
        "人遇难": "fatalities",
        # 灾害名
        "暴雨山洪": "Flash Flood",
        "极端高温": "Extreme Heat",
        "沙尘强风": "Dust Storm",
        # 事件名（完整匹配）
        "麦加/吉达特大洪水+拉比格龙卷风": "Mecca/Jeddah Catastrophic Flood + Rabigh Tornado",
        "哈伊勒/布赖代春季首场大型山洪": "Hail/Buraidah First Major Spring Flash Flood",
        "卡西姆/利雅得巨型哈布尘暴(Haboob)": "Qassim/Riyadh Giant Haboob Dust Storm",
        "全年最强持续性沙尘(全国4天)": "Strongest Persistent Dust Storm of the Year (4 days nationwide)",
        "52.2°C 破纪录高温": "Record-Breaking 52.2°C Extreme Heat",
        "朝觐季 47°C 极端高温": "Hajj Season 47°C Extreme Heat",
        "东部/汉志持续性沙尘+高温叠加": "Eastern/Hijaz Persistent Dust + Heatwave Compound",
        "塔伊夫冰雹洪水(巨型冰雹)": "Taif Hailstorm Flood (Giant Hail)",
        "阿西尔/吉赞/纳季兰(10行政区预警)": "Asir/Jazan/Najran (10 Administrative Regions Alert)",
        "吉达历史性特大洪水(179mm/6h,2人遇难)": "Jeddah Historic Flash Flood (179mm/6h, 2 Fatalities)",
        "10 个 Ground Truth 事件 — 下拉切换 · 拖拽缩放": "10 Ground Truth Events — Dropdown to Switch · Drag to Zoom",
        "海拔 (m)": "Elevation (m)",
        "气温 (°C)": "Temperature (°C)",
        "风速 (m/s)": "Wind Speed (m/s)",
    },
    "ar": {
        # 页面/UI
        "Ground Truth 灾害事件可视化": "تصوير أحداث الكوارث الموثقة",
        "真实灾害事件": "أحداث موثقة",
        "下拉切换": "قائمة منسدلة للتبديل",
        "个事件": "أحداث",
        "经度": "خط الطول",
        "纬度": "خط العرض",
        "海拔 (m)": "الارتفاع (م)",
        "气温 (°C)": "الحرارة (°م)",
        "风速 (m/s)": "سرعة الرياح (م/ث)",
        "海拔": "ارتفاع",
        "气温": "حرارة",
        "风速": "رياح",
        "最高温": "أقصى حرارة",
        "最高风速": "أقصى سرعة رياح",
        "极端高温区": "منطقة حرارة شديدة",
        "强风区": "منطقة رياح قوية",
        "棕色覆盖": "تغطية بنية",
        "红色覆盖": "تغطية حمراء",
        "受影响": "متأثر",
        "风险源头": "مصدر الخطر",
        "源头": "مصدر",
        "节点": "عقدة",
        "子图": "رسم فرعي",
        "全国": "جميع المناطق",
        "东部": "شرق",
        "人遇难": "وفيات",
        # 灾害名
        "暴雨山洪": "فيضانات مفاجئة",
        "极端高温": "حرارة شديدة",
        "沙尘强风": "عاصفة ترابية",
        # 事件名（完整匹配）
        "麦加/吉达特大洪水+拉比格龙卷风": "فيضانات مكة/جدة الكارثية + إعصار رابغ",
        "哈伊勒/布赖代春季首场大型山洪": "أول فيضان ربيعي كبير في حائل/بريدة",
        "卡西姆/利雅得巨型哈布尘暴(Haboob)": "عاصفة هبوب ترابية عملاقة في القصيم/الرياض",
        "全年最强持续性沙尘(全国4天)": "أقوى عاصفة ترابية مستمرة (4 أيام على مستوى الدولة)",
        "52.2°C 破纪录高温": "حرارة قياسية 52.2°م",
        "朝觐季 47°C 极端高温": "موسم الحج 47°م حرارة شديدة",
        "东部/汉志持续性沙尘+高温叠加": "غبار مستمر + حرارة مرتفعة في الشرقية/الحجاز",
        "塔伊夫冰雹洪水(巨型冰雹)": "فيضان برد الطائف (برد عملاق)",
        "阿西尔/吉赞/纳季兰(10行政区预警)": "عسير/جازان/نجران (تحذير 10 مناطق إدارية)",
        "吉达历史性特大洪水(179mm/6h,2人遇难)": "فيضان جدة التاريخي (179مم/6س، وفاتان)",
        "10 个 Ground Truth 事件 — 下拉切换 · 拖拽缩放": "10 أحداث حقيقية — قائمة منسدلة للتبديل · اسحب للتكبير",
        "海拔 (m)": "الارتفاع (م)",
        "气温 (°C)": "الحرارة (°م)",
        "风速 (m/s)": "سرعة الرياح (م/ث)",
    },
}


def _apply_translations(html: str, mapping: dict) -> str:
    """对 HTML 字符串应用翻译映射（精确替换）"""
    result = html
    for zh, translated in mapping.items():
        result = result.replace(zh, translated)
    return result


def generate_all():
    """为三种语言生成 KG 和 Events HTML"""
    zh_kg = os.path.join(OUTPUTS_DIR, "knowledge_graph.html")
    zh_ev = os.path.join(OUTPUTS_DIR, "kg_events.html")

    for lang in ("en", "ar"):
        # KG
        if os.path.isfile(zh_kg):
            with open(zh_kg, "r", encoding="utf-8") as f:
                kg_html = f.read()
            kg_out = os.path.join(OUTPUTS_DIR, f"knowledge_graph_{lang}.html")
            kg_translated = _apply_translations(kg_html, KG_TRANSLATIONS[lang])
            with open(kg_out, "w", encoding="utf-8") as f:
                f.write(kg_translated)
            print(f"✅ knowledge_graph_{lang}.html ({len(kg_translated)/1024:.0f} KB)")

        # Events
        if os.path.isfile(zh_ev):
            with open(zh_ev, "r", encoding="utf-8") as f:
                ev_html = f.read()
            ev_out = os.path.join(OUTPUTS_DIR, f"kg_events_{lang}.html")
            ev_translated = _apply_translations(ev_html, EVENTS_TRANSLATIONS[lang])
            with open(ev_out, "w", encoding="utf-8") as f:
                f.write(ev_translated)
            print(f"✅ kg_events_{lang}.html ({len(ev_translated)/1024:.0f} KB)")


if __name__ == "__main__":
    generate_all()

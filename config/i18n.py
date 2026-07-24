"""MAZU 多语言国际化配置 — 中文 / English / العربية"""
import os

# ============================================================
# 翻译字典
# ============================================================
T = {
    "zh": {
        "app_title": "MAZU 沙特多灾种预警智能体",
        "header_title": "MAZU 沙特多灾种预警智能体",
        "header_subtitle": "暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 | LightGBM · 知识图谱 · LLM Agent",
        "tab_chat": "智能对话",
        "tab_kg": "知识图谱",
        "tab_events": "历史事件",
        "tab_system": "系统",
        "sidebar_title": "对话记录",
        "new_chat_btn": "+ 新建对话",
        "new_chat_default": "新对话",
        "send": "发送",
        "placeholder": "输入问题，如：明天利雅得会有热浪吗？",
        "examples_label": "试试这些问题",
        "welcome": "你好，我是 MAZU 沙特多灾种预警智能体。\n\n可以查询四类灾害风险：暴雨山洪、极端高温、沙尘强风、沿海风浪。也可以询问处置建议和影响分析。\n\n请直接输入你的问题。",
        "kg_hint": "*点击灾害节点筛选 · 双击空白重置 · 滚轮缩放*",
        "events_hint": "*10 个真实灾害事件 — 下拉切换 · 拖拽缩放*",
        "model_status": "模型状态",
        "llm_config": "LLM 配置",
        "data_label": "数据",
        "log_label": "运行日志",
        "refresh": "刷新",
        "file_missing": "文件不存在，请先运行 python run.py tools all",
        "init_error": "初始化失败",
        "analyzing": "分析风险中...",
        "kg_querying": "影响链分析中...",
        "searching": "检索相似案例中...",
        "model_not_found": "未找到",
        "model_loaded": "已加载",
        "lang_label": "语言",
        "not_set": "未设置",
        "nc_files": "NC 文件",
        "dir": "目录",
        "language_selector": "🌐 语言 / Language / اللغة",
    },
    "en": {
        "app_title": "MAZU Saudi Multi-Hazard Early Warning Agent",
        "header_title": "MAZU Saudi Multi-Hazard Early Warning Agent",
        "header_subtitle": "Flash Flood · Extreme Heat · Dust Storm · Coastal Wave | LightGBM · Knowledge Graph · LLM Agent",
        "tab_chat": "Smart Chat",
        "tab_kg": "Knowledge Graph",
        "tab_events": "Historical Events",
        "tab_system": "System",
        "sidebar_title": "Conversations",
        "new_chat_btn": "+ New Chat",
        "new_chat_default": "New Chat",
        "send": "Send",
        "placeholder": "Ask a question, e.g.: Will there be a heatwave in Riyadh tomorrow?",
        "examples_label": "Try these questions",
        "welcome": "Hello, I'm MAZU, the Saudi Multi-Hazard Early Warning Agent.\n\nI can analyze four types of disaster risks: flash floods, extreme heat, dust storms, and coastal waves. I can also provide response recommendations and impact analysis.\n\nPlease type your question.",
        "kg_hint": "*Click disaster nodes to filter · Double-click blank to reset · Scroll to zoom*",
        "events_hint": "*10 Ground Truth Events — dropdown to switch · drag to zoom*",
        "model_status": "Model Status",
        "llm_config": "LLM Config",
        "data_label": "Data",
        "log_label": "Runtime Log",
        "refresh": "Refresh",
        "file_missing": "File not found. Run: python run.py tools all",
        "init_error": "Initialization failed",
        "analyzing": "Analyzing risks...",
        "kg_querying": "Impact chain analysis...",
        "searching": "Searching similar cases...",
        "model_not_found": "Not found",
        "model_loaded": "Loaded",
        "lang_label": "Language",
        "not_set": "Not set",
        "nc_files": "NC files",
        "dir": "Directory",
        "language_selector": "🌐 语言 / Language / اللغة",
    },
    "ar": {
        "app_title": "MAZU وكيل الإنذار المبكر للكوارث المتعددة",
        "header_title": "MAZU وكيل الإنذار المبكر للكوارث المتعددة في السعودية",
        "header_subtitle": "فيضانات مفاجئة · حرارة شديدة · عواصف ترابية · أمواج ساحلية | LightGBM · رسم بياني معرفي · LLM Agent",
        "tab_chat": "المحادثة الذكية",
        "tab_kg": "الرسم البياني المعرفي",
        "tab_events": "الأحداث التاريخية",
        "tab_system": "النظام",
        "sidebar_title": "المحادثات",
        "new_chat_btn": "+ محادثة جديدة",
        "new_chat_default": "محادثة جديدة",
        "send": "إرسال",
        "placeholder": "اطرح سؤالاً، مثال: هل سيكون هناك موجة حر في الرياض غداً؟",
        "examples_label": "جرّب هذه الأسئلة",
        "welcome": "مرحباً، أنا MAZU، وكيل الإنذار المبكر للكوارث المتعددة في السعودية.\n\nيمكنني تحليل أربعة أنواع من مخاطر الكوارث: الفيضانات المفاجئة، الحرارة الشديدة، العواصف الترابية، والأمواج الساحلية. كما يمكنني تقديم توصيات الاستجابة وتحليل التأثير.\n\nيرجى كتابة سؤالك.",
        "kg_hint": "*انقر على عقد الكوارث للتصفية · انقر مرتين للإعادة · مرر للتكبير*",
        "events_hint": "*10 أحداث حقيقية — قائمة منسدلة للتبديل · اسحب للتكبير*",
        "model_status": "حالة النموذج",
        "llm_config": "إعدادات LLM",
        "data_label": "البيانات",
        "log_label": "سجل التشغيل",
        "refresh": "تحديث",
        "file_missing": "الملف غير موجود. شغّل: python run.py tools all",
        "init_error": "فشل التهيئة",
        "analyzing": "تحليل المخاطر...",
        "kg_querying": "تحليل سلسلة التأثير...",
        "searching": "البحث عن حالات مماثلة...",
        "model_not_found": "غير موجود",
        "model_loaded": "تم التحميل",
        "lang_label": "اللغة",
        "not_set": "غير مضبوط",
        "nc_files": "ملفات NC",
        "dir": "المجلد",
        "language_selector": "🌐 中文 / English / العربية",
    },
}

# ============================================================
# 对话示例（按语言）
# ============================================================
CHAT_EXAMPLES = {
    "zh": [
        "8月28日阿西尔地区有山洪风险吗",
        "明天利雅得会有热浪吗",
        "5月中旬沙特有沙尘暴吗",
        "红海沿岸有没有风浪预警",
        "如果吉达有山洪红色预警，应该采取什么措施",
    ],
    "en": [
        "Is there a flash flood risk in Asir on August 28?",
        "Will Riyadh have a heatwave tomorrow?",
        "Any dust storms in Saudi Arabia in mid-May?",
        "Are there coastal wave warnings along the Red Sea?",
        "What measures should Jeddah take for a red flood alert?",
    ],
    "ar": [
        "هل هناك خطر فيضانات في عسير في 28 أغسطس؟",
        "هل سيكون هناك موجة حر في الرياض غداً؟",
        "هل توجد عواصف ترابية في السعودية منتصف مايو؟",
        "هل هناك تحذيرات من أمواج ساحلية على البحر الأحمر؟",
        "ما الإجراءات اللازمة لإنذار أحمر من الفيضانات في جدة؟",
    ],
}

# ============================================================
# 工具状态消息映射
# ============================================================
TOOL_STATUS = {
    "zh": {"predict_risk": "分析风险中...", "query_kg_impact": "影响链分析中...", "search_similar_cases": "检索相似案例中..."},
    "en": {"predict_risk": "Analyzing risks...", "query_kg_impact": "Impact chain analysis...", "search_similar_cases": "Searching similar cases..."},
    "ar": {"predict_risk": "تحليل المخاطر...", "query_kg_impact": "تحليل سلسلة التأثير...", "search_similar_cases": "البحث عن حالات مماثلة..."},
}

# ============================================================
# 方便函数
# ============================================================
def t(lang: str, key: str) -> str:
    """获取翻译文本，自动回退到中文"""
    return T.get(lang, T["zh"]).get(key, T["zh"].get(key, key))


def build_header(lang: str) -> str:
    """生成各语言版本头部 HTML"""
    d = T[lang]
    return (
        f'<div style="text-align:center;padding:12px 0 0">'
        f'<h1 style="font-size:1.35rem;font-weight:800;color:#0c4a6e;margin:0">{d["header_title"]}</h1>'
        f'<p style="font-size:.8rem;color:#64748b;margin:2px 0 0">{d["header_subtitle"]}</p>'
        f'</div>'
    )


# ============================================================
# KG / Events HTML 多语言加载
# ============================================================
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

KG_HTML = {}
EVENTS_HTML = {}
FILE_MISSING_HTML = {}

for lang in ("zh", "en", "ar"):
    kg_path = os.path.join(OUTPUTS_DIR, f"knowledge_graph_{lang}.html")
    ev_path = os.path.join(OUTPUTS_DIR, f"kg_events_{lang}.html")
    # 回退到默认文件
    kg_fallback = os.path.join(OUTPUTS_DIR, "knowledge_graph.html")
    ev_fallback = os.path.join(OUTPUTS_DIR, "kg_events.html")

    if os.path.isfile(kg_path):
        with open(kg_path, "r", encoding="utf-8") as f:
            KG_HTML[lang] = f.read()
    elif os.path.isfile(kg_fallback):
        with open(kg_fallback, "r", encoding="utf-8") as f:
            KG_HTML[lang] = f.read()
    else:
        KG_HTML[lang] = f"<h3 style='text-align:center;padding:40px'>{T[lang]['file_missing']}</h3>"

    if os.path.isfile(ev_path):
        with open(ev_path, "r", encoding="utf-8") as f:
            EVENTS_HTML[lang] = f.read()
    elif os.path.isfile(ev_fallback):
        with open(ev_fallback, "r", encoding="utf-8") as f:
            EVENTS_HTML[lang] = f.read()
    else:
        EVENTS_HTML[lang] = f"<h3 style='text-align:center;padding:40px'>{T[lang]['file_missing']}</h3>"

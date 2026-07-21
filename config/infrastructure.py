"""
沙特承灾体基础设施数据：城市、机场、高速、港口坐标

用于知识图谱影响分析——当模型预测某区域高风险时，
KG 可以检索受影响的承灾体并生成具体的影响链。

数据来源：OpenStreetMap / 公开地理数据
"""
from typing import Dict, List

# ═══════════════════════════════════════════════
# 主要城市（人口 > 50,000）
# ═══════════════════════════════════════════════

CITIES: List[Dict] = [
    {"name": "利雅得 (Riyadh)",        "lat": 24.71, "lon": 46.68, "pop": 7_000_000, "type": "capital"},
    {"name": "吉达 (Jeddah)",          "lat": 21.54, "lon": 39.17, "pop": 4_000_000, "type": "coastal"},
    {"name": "麦加 (Mecca)",           "lat": 21.42, "lon": 39.83, "pop": 2_000_000, "type": "holy"},
    {"name": "麦地那 (Medina)",        "lat": 24.47, "lon": 39.61, "pop": 1_300_000, "type": "holy"},
    {"name": "达曼 (Dammam)",          "lat": 26.42, "lon": 50.10, "pop": 1_200_000, "type": "coastal"},
    {"name": "塔伊夫 (Taif)",          "lat": 21.27, "lon": 40.42, "pop": 700_000,   "type": "mountain"},
    {"name": "布赖代 (Buraydah)",      "lat": 26.33, "lon": 43.97, "pop": 600_000,   "type": "inland"},
    {"name": "胡富夫 (Hofuf)",         "lat": 25.36, "lon": 49.59, "pop": 600_000,   "type": "inland"},
    {"name": "哈伊勒 (Hail)",          "lat": 27.52, "lon": 41.69, "pop": 400_000,   "type": "mountain"},
    {"name": "朱拜勒 (Jubail)",        "lat": 27.01, "lon": 49.66, "pop": 400_000,   "type": "industrial"},
    {"name": "艾卜哈 (Abha)",          "lat": 18.22, "lon": 42.51, "pop": 350_000,   "type": "mountain"},
    {"name": "纳季兰 (Najran)",        "lat": 17.49, "lon": 44.13, "pop": 300_000,   "type": "border"},
    {"name": "吉赞 (Jizan)",           "lat": 16.89, "lon": 42.55, "pop": 200_000,   "type": "coastal"},
    {"name": "延布 (Yanbu)",           "lat": 24.09, "lon": 38.06, "pop": 250_000,   "type": "port"},
    {"name": "卡西姆 (Qassim)",        "lat": 26.30, "lon": 43.97, "pop": 400_000,   "type": "inland"},
    {"name": "哈费尔巴廷 (Hafar Al-Batin)", "lat": 28.43, "lon": 45.96, "pop": 300_000, "type": "border"},
    {"name": "拉比格 (Rabigh)",        "lat": 22.80, "lon": 39.03, "pop": 100_000,   "type": "coastal"},
    {"name": "拉夫哈 (Rafha)",         "lat": 29.63, "lon": 43.51, "pop": 80_000,    "type": "border"},
    {"name": "阿尔阿尔 (Arar)",         "lat": 30.98, "lon": 41.02, "pop": 200_000,   "type": "border"},
    {"name": "塞卡凯/焦夫 (Sakaka/Al-Jouf)", "lat": 29.97, "lon": 40.20, "pop": 250_000, "type": "inland"},
    {"name": "塔布克 (Tabuk)",         "lat": 28.38, "lon": 36.57, "pop": 600_000,   "type": "border"},
]

# ═══════════════════════════════════════════════
# 省份/地区映射（省名 → 省会/代表城市坐标）
# ═══════════════════════════════════════════════

PROVINCES: Dict[str, Dict] = {
    "利雅得省":     {"capital": "利雅得",    "lat": 24.71, "lon": 46.68},
    "麦加省":       {"capital": "麦加",      "lat": 21.42, "lon": 39.83},
    "东部省":       {"capital": "达曼",      "lat": 26.42, "lon": 50.10},
    "麦地那省":     {"capital": "麦地那",    "lat": 24.47, "lon": 39.61},
    "阿西尔省":     {"capital": "艾卜哈",    "lat": 18.22, "lon": 42.51},
    "北部边境省":   {"capital": "阿尔阿尔",  "lat": 30.98, "lon": 41.02},
    "焦夫省":       {"capital": "塞卡凯",    "lat": 29.97, "lon": 40.20},
    "塔布克省":     {"capital": "塔布克",    "lat": 28.38, "lon": 36.57},
    "哈伊勒省":     {"capital": "哈伊勒",    "lat": 27.52, "lon": 41.69},
    "卡西姆省":     {"capital": "布赖代",    "lat": 26.33, "lon": 43.97},
    "吉赞省":       {"capital": "吉赞",      "lat": 16.89, "lon": 42.55},
    "纳季兰省":     {"capital": "纳季兰",    "lat": 17.49, "lon": 44.13},
}

# ═══════════════════════════════════════════════
# 主要机场
# ═══════════════════════════════════════════════

AIRPORTS: List[Dict] = [
    {"name": "哈立德国王国际机场 (RUH)",  "lat": 24.96, "lon": 46.70, "city": "利雅得"},
    {"name": "阿卜杜勒-阿齐兹国王机场 (JED)", "lat": 21.68, "lon": 39.16, "city": "吉达"},
    {"name": "法赫德国王国际机场 (DMM)",   "lat": 26.47, "lon": 49.80, "city": "达曼"},
    {"name": "穆罕默德亲王机场 (MED)",     "lat": 24.55, "lon": 39.70, "city": "麦地那"},
    {"name": "塔伊夫机场 (TIF)",          "lat": 21.48, "lon": 40.54, "city": "塔伊夫"},
    {"name": "艾卜哈机场 (AHB)",          "lat": 18.24, "lon": 42.66, "city": "艾卜哈"},
    {"name": "哈伊勒机场 (HAS)",          "lat": 27.44, "lon": 41.69, "city": "哈伊勒"},
]

# ═══════════════════════════════════════════════
# 主要港口
# ═══════════════════════════════════════════════

PORTS: List[Dict] = [
    {"name": "吉达伊斯兰港",         "lat": 21.47, "lon": 39.15, "type": "commercial"},
    {"name": "阿卜杜拉国王港",       "lat": 22.40, "lon": 39.08, "type": "commercial"},
    {"name": "朱拜勒工业港",         "lat": 27.02, "lon": 49.68, "type": "industrial"},
    {"name": "达曼阿卜杜勒-阿齐兹港", "lat": 26.50, "lon": 50.20, "type": "commercial"},
    {"name": "延布工业港",           "lat": 23.96, "lon": 38.23, "type": "industrial"},
    {"name": "吉赞港",              "lat": 16.90, "lon": 42.55, "type": "commercial"},
]

# ═══════════════════════════════════════════════
# 主要高速公路（关键路段）
# ═══════════════════════════════════════════════

HIGHWAYS: List[Dict] = [
    {"name": "40号公路 (利雅得-吉达)", "from_city": "利雅得", "to_city": "吉达",
     "path": [(24.71,46.68),(24.17,44.67),(22.96,42.75),(21.54,39.17)]},
    {"name": "85号公路 (利雅得-达曼)", "from_city": "利雅得", "to_city": "达曼",
     "path": [(24.71,46.68),(25.36,47.25),(26.42,50.10)]},
    {"name": "15号公路 (吉达-麦地那)", "from_city": "吉达", "to_city": "麦地那",
     "path": [(21.54,39.17),(22.98,39.57),(24.47,39.61)]},
    {"name": "65号公路 (利雅得-哈伊勒)", "from_city": "利雅得", "to_city": "哈伊勒",
     "path": [(24.71,46.68),(26.33,43.97),(27.52,41.69)]},
    {"name": "10号公路 (达曼-朱拜勒沿海)", "from_city": "达曼", "to_city": "朱拜勒",
     "path": [(26.42,50.10),(27.01,49.66)]},
    {"name": "5号公路 (吉达-吉赞沿海)", "from_city": "吉达", "to_city": "吉赞",
     "path": [(21.54,39.17),(18.22,42.51),(16.89,42.55)]},
]

# ═══════════════════════════════════════════════
# Wadi（干涸河道）— 山洪传播路径
# ═══════════════════════════════════════════════

WADIS: List[Dict] = [
    {"name": "Wadi Ibrahim",   "lat": 21.40, "lon": 39.85, "length_km": 15, "risk_city": "麦加"},
    {"name": "Wadi Fatima",    "lat": 21.90, "lon": 39.70, "length_km": 80, "risk_city": "吉达/麦加"},
    {"name": "Wadi Hanifa",    "lat": 24.65, "lon": 46.70, "length_km": 120, "risk_city": "利雅得"},
    {"name": "Wadi Bisha",     "lat": 20.00, "lon": 42.60, "length_km": 200, "risk_city": "阿西尔"},
    {"name": "Wadi Najran",    "lat": 17.50, "lon": 44.20, "length_km": 150, "risk_city": "纳季兰"},
    {"name": "Wadi Al-Rummah", "lat": 26.30, "lon": 44.00, "length_km": 600, "risk_city": "卡西姆/布赖代"},
]

# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def find_nearby_infrastructure(lat: float, lon: float, radius_km: float = 30) -> Dict:
    """查找指定坐标附近的基础设施。

    Args:
        lat, lon: 中心坐标
        radius_km: 搜索半径（km），约0.1°≈11km

    Returns:
        {"cities": [...], "airports": [...], "ports": [...], "highways": [...], "wadis": [...]}
    """
    deg = radius_km / 111.0  # 粗略转换

    result = {"cities": [], "airports": [], "ports": [], "highways": [], "wadis": []}

    for c in CITIES:
        if abs(c["lat"] - lat) < deg and abs(c["lon"] - lon) < deg:
            result["cities"].append(c)

    for a in AIRPORTS:
        if abs(a["lat"] - lat) < deg and abs(a["lon"] - lon) < deg:
            result["airports"].append(a)

    for p in PORTS:
        if abs(p["lat"] - lat) < deg and abs(p["lon"] - lon) < deg:
            result["ports"].append(p)

    # 高速公路：检查是否经过该区域
    for h in HIGHWAYS:
        for hp_lat, hp_lon in h["path"]:
            if abs(hp_lat - lat) < deg * 2 and abs(hp_lon - lon) < deg * 2:
                result["highways"].append(h)
                break

    for w in WADIS:
        if abs(w["lat"] - lat) < deg and abs(w["lon"] - lon) < deg:
            result["wadis"].append(w)

    return result


def format_infrastructure_impact(nearby: Dict) -> str:
    """将附近基础设施格式化为可读的影响分析文本。

    Args:
        nearby: find_nearby_infrastructure() 的返回结果

    Returns:
        格式化的 Markdown 文本
    """
    lines = []
    if nearby["cities"]:
        cities_str = "、".join(c["name"] for c in nearby["cities"])
        total_pop = sum(c["pop"] for c in nearby["cities"])
        lines.append(f"- 🏙️ 受影响城市({len(nearby['cities'])}个): {cities_str}，涉及人口约 {total_pop:,}人")
    if nearby["airports"]:
        airports_str = "、".join(a["name"] for a in nearby["airports"])
        lines.append(f"- ✈️ 受影响机场({len(nearby['airports'])}个): {airports_str}")
    if nearby["ports"]:
        ports_str = "、".join(p["name"] for p in nearby["ports"])
        lines.append(f"- 🚢 受影响港口({len(nearby['ports'])}个): {ports_str}")
    if nearby["highways"]:
        highways_str = "、".join(h["name"] for h in nearby["highways"])
        lines.append(f"- 🛣️ 受影响高速公路({len(nearby['highways'])}条): {highways_str}")
    if nearby["wadis"]:
        wadis_str = "、".join(f"{w['name']}({w['length_km']}km)" for w in nearby["wadis"])
        lines.append(f"- 🌊 受影响Wadi河道({len(nearby['wadis'])}条): {wadis_str}，山洪可沿河道向下游传播")

    return "\n".join(lines) if lines else "- 该区域附近无重大基础设施"

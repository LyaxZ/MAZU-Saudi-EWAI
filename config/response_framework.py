"""
结构化处置框架：风险等级 × 灾害类型 → 标准处置动作

替代 LLM 自由发挥，确保每次预警输出的处置建议：
1. 可操作（具体到部门、路段、时段）
2. 分层级（红/橙/黄/蓝 四级响应）
3. 全覆盖（疏散、交通、医疗、电力、通信）

用法：
    from config.response_framework import get_actions
    actions = get_actions("flash_flood", risk_level="red", area="urban")
    # → [{"category": "疏散", "action": "...", "priority": 1}, ...]
"""

from typing import Dict, List

# ═══════════════════════════════════════════════
# 风险等级定义
# ═══════════════════════════════════════════════

RISK_LEVELS = {
    "red":    {"min": 0.9, "label": "🔴 极高风险", "color": "红色预警"},
    "orange": {"min": 0.7, "label": "🟠 高风险",   "color": "橙色预警"},
    "yellow": {"min": 0.5, "label": "🟡 中等风险", "color": "黄色预警"},
    "blue":   {"min": 0.0, "label": "🟢 低风险",   "color": "蓝色预警"},
}

def get_risk_level(mean_risk: float) -> Dict:
    """根据平均风险值确定等级"""
    for key, info in RISK_LEVELS.items():
        if mean_risk >= info["min"]:
            return {"level": key, **info}
    return {"level": "blue", **RISK_LEVELS["blue"]}

# ═══════════════════════════════════════════════
# 灾害类型 × 区域类型 × 风险等级 → 标准动作
# ═══════════════════════════════════════════════

FLASH_FLOOD_ACTIONS = {
    "urban": {
        "red": [
            {"category": "疏散", "action": "立即疏散低洼街区及Wadi沿岸居民至指定避难所",
             "target": "民防部门、市政当局"},
            {"category": "交通", "action": "封闭所有Wadi跨河路段及地下通道，设置路障和警示标志",
             "target": "交通警察、市政工程"},
            {"category": "排水", "action": "启动全部排水泵站满负荷运转，部署移动泵车至历史积水点",
             "target": "市政排水部门"},
            {"category": "预警", "action": "通过短信、广播、社交媒体向受影响居民发送紧急疏散通知",
             "target": "国家气象中心(NCM)、民防部门"},
            {"category": "医疗", "action": "医院急诊科全员待命，储备急救物资，准备接收伤员",
             "target": "卫生部、各大医院"},
            {"category": "电力", "action": "切断低洼区域供电，防止漏电事故，部署应急发电车",
             "target": "电力公司"},
        ],
        "orange": [
            {"category": "监测", "action": "实时监控Wadi水位和降雨强度，每30分钟更新一次",
             "target": "NCM、水利部门"},
            {"category": "交通", "action": "在Wadi跨河路段部署交通管制人员，准备随时封闭",
             "target": "交通警察"},
            {"category": "预警", "action": "向低洼区域居民发布洪水预警，建议做好撤离准备",
             "target": "民防部门"},
            {"category": "排水", "action": "检查排水系统运行状态，清理堵塞的排水口",
             "target": "市政排水部门"},
        ],
    },
    "mountain": {
        "red": [
            {"category": "疏散", "action": "立即转移山谷及河道沿线居民至高地安全点",
             "target": "民防部门"},
            {"category": "交通", "action": "封闭山区公路，禁止任何车辆进入山谷路段",
             "target": "交通警察"},
            {"category": "监测", "action": "上游部署水位监测，下游预警洪水到达时间",
             "target": "NCM、水利部门"},
        ],
    },
}

EXTREME_HEAT_ACTIONS = {
    "urban": {
        "red": [
            {"category": "作业", "action": "11:00-16:00全面禁止户外作业，违者依法处罚",
             "target": "劳动部、市政当局"},
            {"category": "避暑", "action": "开放全部避暑中心并延长至凌晨，提供免费饮用水和空调",
             "target": "市政当局、红新月会"},
            {"category": "电力", "action": "启动电网最高级别调峰预案，确保居民空调供电",
             "target": "电力公司"},
            {"category": "医疗", "action": "医院急诊科全员待命，储备中暑急救药品和冰毯",
             "target": "卫生部"},
            {"category": "预警", "action": "通过所有渠道推送高温红色预警，建议错峰用电(空调24°C+)",
             "target": "NCM、媒体"},
        ],
        "orange": [
            {"category": "作业", "action": "建议11:00-15:00减少户外作业，提供防暑降温措施",
             "target": "劳动部"},
            {"category": "避暑", "action": "开放避暑中心，延长至晚间，提供饮水",
             "target": "市政当局"},
            {"category": "医疗", "action": "医院增加中暑应急值班，储备口服补液盐",
             "target": "卫生部"},
        ],
    },
    "hajj": {
        "red": [
            {"category": "朝觐", "action": "限制老年朝圣者11:00-15:00户外活动，喷雾降温装置全功率运行",
             "target": "朝觐部、卫生部"},
            {"category": "医疗", "action": "在阿拉法特山和米纳每500米部署移动医疗站，开通中暑急救绿色通道",
             "target": "红新月会、卫生部"},
            {"category": "物资", "action": "免费发放饮用水、遮阳伞和防晒用品",
             "target": "朝觐服务部门"},
        ],
    },
}

DUST_WIND_ACTIONS = {
    "urban": {
        "red": [
            {"category": "交通", "action": "机场暂停所有航班起降，高速公路限速降至40km/h，能见度<500m封路",
             "target": "民航局、交通警察"},
            {"category": "健康", "action": "全市学校停课，建议市民居家、紧闭门窗，免费发放N95口罩",
             "target": "教育部、卫生部"},
            {"category": "工业", "action": "暂停所有户外施工及油田露天作业",
             "target": "工业部、石油公司"},
            {"category": "预警", "action": "发布红色沙尘预警，广播电台每15分钟播报能见度更新",
             "target": "NCM、媒体"},
        ],
    },
    "industrial": {
        "red": [
            {"category": "工业", "action": "全面暂停油田和炼化厂户外作业，封闭露天设备",
             "target": "沙特阿美、工业部"},
            {"category": "健康", "action": "工人全员佩戴N95口罩，增开呼吸科门诊",
             "target": "工业安全部门、医院"},
        ],
    },
}

COASTAL_WAVE_ACTIONS = {
    "port": {
        "red": [
            {"category": "港口", "action": "全面暂停船舶进出港，已靠泊船舶加固缆绳",
             "target": "港务局、海岸警卫队"},
            {"category": "沿海", "action": "封闭沿海低洼路段，设置防浪屏障，居民远离海岸线",
             "target": "市政当局、海岸警卫队"},
            {"category": "渔船", "action": "所有小型渔船禁止出海，已出海渔船立即返航",
             "target": "渔业部门、海岸警卫队"},
        ],
    },
}

# 各灾害默认动作（未指定区域类型时使用）
DEFAULT_ACTIONS = {
    "flash_flood": FLASH_FLOOD_ACTIONS.get("urban"),
    "extreme_heat": EXTREME_HEAT_ACTIONS.get("urban"),
    "dust_wind": DUST_WIND_ACTIONS.get("urban"),
    "coastal_wave": COASTAL_WAVE_ACTIONS.get("port"),
}

DISASTER_ACTIONS = {
    "flash_flood": FLASH_FLOOD_ACTIONS,
    "extreme_heat": EXTREME_HEAT_ACTIONS,
    "dust_wind": DUST_WIND_ACTIONS,
    "coastal_wave": COASTAL_WAVE_ACTIONS,
}


def get_actions(
    disaster_type: str,
    risk_level: str = "orange",
    area_type: str = "urban",
) -> List[Dict]:
    """获取指定灾害类型、风险等级、区域类型的标准处置动作。

    Args:
        disaster_type: 灾害类型 (flash_flood/extreme_heat/dust_wind/coastal_wave)
        risk_level: 风险等级 (red/orange/yellow/blue)
        area_type: 区域类型 (urban/mountain/port/industrial/hajj)

    Returns:
        处置动作列表，每项含 category/action/target
    """
    all_actions = DISASTER_ACTIONS.get(disaster_type, {})
    area_actions = all_actions.get(area_type, {})
    if not area_actions:
        # 回退到默认区域
        area_actions = DEFAULT_ACTIONS.get(disaster_type, {})

    return area_actions.get(risk_level, [])


def format_actions_for_llm(
    disaster_type: str,
    mean_risk: float,
    area_type: str = "urban",
) -> str:
    """生成供 LLM Agent 使用的格式化处置建议文本。

    Args:
        disaster_type: 灾害类型
        mean_risk: 平均风险值
        area_type: 区域类型

    Returns:
        格式化的 Markdown 处置建议
    """
    risk = get_risk_level(mean_risk)
    actions = get_actions(disaster_type, risk["level"], area_type)

    if not actions:
        actions = get_actions(disaster_type, "orange", area_type)

    lines = [f"### 🛡️ 建议处置措施（{risk['label']} {risk['color']}）\n"]

    # 按类别分组
    by_cat: Dict[str, List[Dict]] = {}
    for a in actions:
        by_cat.setdefault(a["category"], []).append(a)

    for cat, acts in by_cat.items():
        lines.append(f"**{cat}**：")
        for i, a in enumerate(acts, 1):
            lines.append(f"  {i}. {a['action']}（{a['target']}）")
        lines.append("")

    return "\n".join(lines)

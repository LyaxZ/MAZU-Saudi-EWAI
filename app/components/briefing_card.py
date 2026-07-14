"""
预警简报卡片组件

生成结构化预警简报：
- 风险等级判定（红/橙/黄/蓝）
- 影响范围与承灾体统计
- 建议应对措施
- 历史相似案例引用
- 支持 Markdown / HTML / 纯文本输出

用法:
    from app.components.briefing_card import BriefingCardGenerator

    generator = BriefingCardGenerator()
    card = generator.generate(
        disaster_type="flash_flood",
        risk_summary={...},
        propagation_result={...},
        similar_cases=[...],
    )
    print(card.markdown)
"""

from datetime import datetime
from typing import Dict, List, Optional


class BriefingCardGenerator:
    """预警简报生成器。"""

    # 风险等级 → 颜色映射
    RISK_LEVELS = {
        4: {"name": "🔴 红色预警", "color": "#dc3545", "desc": "特别重大"},
        3: {"name": "🟠 橙色预警", "color": "#fd7e14", "desc": "重大"},
        2: {"name": "🟡 黄色预警", "color": "#ffc107", "desc": "较大"},
        1: {"name": "🔵 蓝色预警", "color": "#0d6efd", "desc": "一般"},
        0: {"name": "🟢 无预警",   "color": "#198754", "desc": "安全"},
    }

    DISASTER_NAMES = {
        "flash_flood": "暴雨山洪",
        "extreme_heat": "极端高温",
        "dust_wind": "沙尘强风",
        "coastal_wave": "沿海风浪",
    }

    def __init__(
        self,
        issuer: str = "MAZU 多灾种预警智能体",
        region: str = "沙特阿拉伯",
    ):
        """
        Args:
            issuer: 发布机构
            region: 目标区域
        """
        self.issuer = issuer
        self.region = region

    # ================================================================
    # generate: 生成简报
    # ================================================================

    def generate(
        self,
        disaster_type: str,
        risk_summary: Dict,
        propagation_result: Optional[Dict] = None,
        similar_cases: Optional[List[Dict]] = None,
        format: str = "markdown",
    ) -> "BriefingCard":
        """生成预警简报卡片。

        Args:
            disaster_type: 灾害类型
            risk_summary: {
                "n_high_risk_cells": int,
                "mean_risk": float,
                "max_risk": float,
                "affected_area_km2": float,
                "n_people_exposed": int (可选),
            }
            propagation_result: RiskPropagator 输出
            similar_cases: CaseRetrieval.search() 输出
            format: 输出格式 — "markdown" / "html" / "text"

        Returns:
            BriefingCard 对象
        """
        disaster_name = self.DISASTER_NAMES.get(disaster_type, disaster_type)

        # 1. 判定风险等级
        risk_level = self._determine_risk_level(risk_summary)
        level_info = self.RISK_LEVELS[risk_level]

        # 2. 生成各部分
        header = self._build_header(disaster_name, level_info)
        summary_section = self._build_summary(disaster_name, risk_summary, level_info)
        impact_section = self._build_impact(propagation_result)
        measures_section = self._build_measures(disaster_type, risk_level)
        cases_section = self._build_similar_cases(similar_cases)
        footer = self._build_footer()

        # 3. 组合
        if format == "html":
            content = self._assemble_html(
                header, summary_section, impact_section,
                measures_section, cases_section, footer,
            )
            markdown = self._assemble_markdown(
                header, summary_section, impact_section,
                measures_section, cases_section, footer,
            )
            text = self._assemble_text(
                header, summary_section, impact_section,
                measures_section, cases_section, footer,
            )
        else:
            markdown = self._assemble_markdown(
                header, summary_section, impact_section,
                measures_section, cases_section, footer,
            )
            content = markdown
            text = markdown

        return BriefingCard(
            disaster_type=disaster_type,
            risk_level=risk_level,
            markdown=markdown,
            html=content if format == "html" else None,
            text=text,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "issuer": self.issuer,
                "risk_summary": risk_summary,
            },
        )

    # ================================================================
    # 各模块
    # ================================================================

    def _determine_risk_level(self, risk_summary: Dict) -> int:
        """根据风险摘要判定预警等级。

        规则：
        - max_risk > 0.8 → 红色 (4)
        - max_risk > 0.6 → 橙色 (3)
        - max_risk > 0.4 → 黄色 (2)
        - max_risk > 0.2 → 蓝色 (1)
        - 其他 → 无预警 (0)
        """
        max_risk = risk_summary.get("max_risk", 0)
        n_high = risk_summary.get("n_high_risk_cells", 0)

        if max_risk > 0.8 or n_high > 1000:
            return 4
        elif max_risk > 0.6 or n_high > 500:
            return 3
        elif max_risk > 0.4 or n_high > 100:
            return 2
        elif max_risk > 0.1 or n_high > 0:
            return 1
        return 0

    def _build_header(self, disaster_name: str, level_info: Dict) -> str:
        """生成简报头部。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"## {level_info['name']}\n"
            f"**发布机构**: {self.issuer}\n"
            f"**发布时间**: {now}\n"
            f"**灾害类型**: {disaster_name}\n"
            f"**预警等级**: {level_info['desc']}\n"
        )

    def _build_summary(
        self, disaster_name: str, risk_summary: Dict, level_info: Dict
    ) -> str:
        """生成风险摘要段落。"""
        n_cells = risk_summary.get("n_high_risk_cells", 0)
        mean_risk = risk_summary.get("mean_risk", 0)
        max_risk = risk_summary.get("max_risk", 0)
        area = risk_summary.get("affected_area_km2", n_cells * 100)
        n_people = risk_summary.get("n_people_exposed")

        lines = [
            f"### 📊 风险摘要",
            f"- **高风险格点数**: {n_cells:,} 个",
            f"- **平均风险值**: {mean_risk:.3f}",
            f"- **最高风险值**: {max_risk:.3f}",
            f"- **预计影响面积**: {area:,.0f} km²",
        ]
        if n_people is not None:
            lines.append(f"- **预计暴露人口**: {n_people:,} 人")

        return "\n".join(lines)

    def _build_impact(self, propagation_result: Optional[Dict]) -> str:
        """生成影响范围段落。"""
        if propagation_result is None:
            return "### 🌊 影响范围\n（暂无传播分析数据）\n"

        n_affected = propagation_result.get("n_affected", 0)
        summary = propagation_result.get("summary", {})
        coastal = summary.get("coastal_affected", 0)

        lines = [
            f"### 🌊 影响范围",
            f"- **受影响格点数**: {n_affected:,} 个",
            f"- **其中沿海格点**: {coastal:,} 个" if coastal else "",
            f"- **最大传播距离**: {summary.get('max_distance_km', 'N/A')}",
            f"",
            f"基于知识图谱空间推理，上述高风险区域可能通过以下路径影响周边：",
        ]

        # 简要列出传播路径
        paths = propagation_result.get("paths", {})
        if paths:
            lines.append("")
            for i, (target, path) in enumerate(list(paths.items())[:3]):
                n_hops = len(path) - 1
                lines.append(f"- 路径 {i+1}: {n_hops} 跳传播 → 到达节点 {target}")

        return "\n".join(lines)

    def _build_measures(self, disaster_type: str, risk_level: int) -> str:
        """生成建议措施段落。"""
        measures = self._get_measures(disaster_type, risk_level)

        lines = ["### 🛡️ 建议措施", ""]
        for i, m in enumerate(measures, 1):
            lines.append(f"{i}. {m}")

        return "\n".join(lines)

    def _get_measures(self, disaster_type: str, risk_level: int) -> List[str]:
        """根据灾害类型和风险等级返回措施。"""
        all_measures = {
            "flash_flood": {
                4: [
                    "立即启动最高级别应急响应",
                    "强制疏散Wadi沿岸所有居民至安全高地",
                    "关闭所有跨Wadi公路和桥梁通道",
                    "部署国家级救援力量（军队+民防）",
                    "向邻国和国际组织请求援助",
                ],
                3: [
                    "启动二级应急响应",
                    "疏散Wadi沿岸高风险社区",
                    "关闭主要跨Wadi公路通道",
                    "启动排水泵站满负荷运行",
                    "向公众发布紧急避难通知",
                ],
                2: [
                    "启动三级应急响应",
                    "监测Wadi水位和降水强度",
                    "向低洼地区发布预警通知",
                    "准备救援物资和避难所",
                    "通知医疗机构做好接收伤员的准备",
                ],
                1: [
                    "加强气象和水文监测",
                    "向相关部门通报风险信息",
                    "检查排水系统和防洪设施",
                    "向公众发布安全提示",
                ],
            },
            "extreme_heat": {
                4: [
                    "发布极端高温红色预警",
                    "全市开放避暑中心",
                    "全面禁止户外作业（11:00-17:00）",
                    "启动医院高温急救绿色通道",
                    "向脆弱人群免费发放饮水和降温用品",
                ],
                3: [
                    "发布高温橙色预警",
                    "开放社区避暑中心",
                    "限制户外作业时段",
                    "增加电力供应保障空调运行",
                    "加强中暑病例监测",
                ],
                2: [
                    "发布高温黄色预警",
                    "提醒公众减少户外活动",
                    "加强老人、儿童等脆弱群体关怀",
                ],
                1: [
                    "关注气温变化趋势",
                    "向公众发布防暑提示",
                ],
            },
            "dust_wind": {
                4: [
                    "发布沙尘暴红色预警",
                    "关闭机场和主要公路",
                    "建议所有居民留在室内、关闭门窗",
                    "暂停所有户外作业和港口作业",
                    "医院调配呼吸系统急救资源",
                    "学校停课",
                ],
                3: [
                    "发布沙尘暴橙色预警",
                    "限制公路通行（降速、限行）",
                    "建议居民佩戴口罩、减少外出",
                    "暂停港口作业",
                    "增加呼吸道疾病门诊力量",
                ],
                2: [
                    "发布沙尘暴黄色预警",
                    "提醒司机注意能见度降低",
                    "建议敏感人群佩戴口罩",
                ],
                1: [
                    "监测风速和能见度变化",
                ],
            },
            "coastal_wave": {
                4: [
                    "发布沿海强风浪红色预警",
                    "强制疏散沿海低洼社区居民",
                    "全面禁止所有海上活动和航行",
                    "加固海堤和港口设施",
                    "部署海岸警卫队和海軍应急力量",
                ],
                3: [
                    "发布沿海强风浪橙色预警",
                    "建议沿海居民撤离至内陆",
                    "暂停海上作业和娱乐活动",
                    "加固港口设施和船只",
                    "加强海堤巡检",
                ],
                2: [
                    "发布沿海强风浪黄色预警",
                    "提醒渔民和海上作业人员注意安全",
                    "限制小型船只出海",
                ],
                1: [
                    "监测海面风速和浪高变化",
                ],
            },
        }

        default = {
            4: ["启动最高级别应急响应", "疏散受影响区域居民"],
            3: ["启动二级应急响应", "加强监测和预警"],
            2: ["启动三级应急响应", "发布公众预警"],
            1: ["加强监测", "通知相关部门"],
        }

        return all_measures.get(disaster_type, default).get(
            risk_level, default[1]
        )

    def _build_similar_cases(self, similar_cases: Optional[List[Dict]]) -> str:
        """生成历史相似案例段落。"""
        if not similar_cases:
            return "### 📜 历史案例\n（暂无相似历史案例数据）\n"

        lines = ["### 📜 历史相似案例", ""]
        for i, case in enumerate(similar_cases[:3], 1):
            sim = case.get("similarity", 0)
            date = case.get("date", "未知")
            loc = case.get("location", "未知")
            measures = case.get("measures", [])
            lines.append(
                f"**案例 {i}**: {date} @ {loc} "
                f"(相似度: {sim:.2%})"
            )
            if measures:
                lines.append(f"  应对措施: {'; '.join(measures[:3])}")
            lines.append("")

        return "\n".join(lines)

    def _build_footer(self) -> str:
        """生成简报尾部。"""
        return (
            "---\n"
            f"*本简报由 {self.issuer} 自动生成，仅供参考。*\n"
            f"*请结合当地实际情况做出决策。*"
        )

    # ================================================================
    # 组装
    # ================================================================

    def _assemble_markdown(self, *sections: str) -> str:
        return "\n".join(sections)

    def _assemble_html(self, *sections: str) -> str:
        """将 Markdown 各节转为 HTML。"""
        # 简单 Markdown → HTML 转换
        html_parts = ['<div class="briefing-card">']
        for section in sections:
            # 基本转换
            html = section
            html = html.replace("### ", "<h3>").replace(" ##", "</h3>\n")
            html = html.replace("- **", "<li><strong>").replace("**: ", "</strong> ")
            html = html.replace("\n- ", "</li>\n<li>")
            html = html.replace("---", "<hr>")
            html_parts.append(f"<div class='section'>{html}</div>")
        html_parts.append("</div>")
        return "\n".join(html_parts)

    def _assemble_text(self, *sections: str) -> str:
        """生成纯文本版本（去除 Markdown 标记）。"""
        import re
        text = "\n".join(sections)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'[*_~`]', '', text)
        return text


class BriefingCard:
    """预警简报卡片数据类。"""

    def __init__(
        self,
        disaster_type: str,
        risk_level: int,
        markdown: str,
        html: Optional[str],
        text: str,
        metadata: Dict,
    ):
        self.disaster_type = disaster_type
        self.risk_level = risk_level
        self.markdown = markdown
        self.html = html
        self.text = text
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"BriefingCard(disaster={self.disaster_type}, "
            f"level={self.risk_level})"
        )

    def to_dict(self) -> Dict:
        return {
            "disaster_type": self.disaster_type,
            "risk_level": self.risk_level,
            "markdown": self.markdown,
            "html": self.html,
            "text": self.text,
            "metadata": self.metadata,
        }

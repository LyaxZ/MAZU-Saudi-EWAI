"""
MAZU Agent — LLM Function Calling 主循环

用法:
    from llm_agent.agent import MazuAgent
    agent = MazuAgent()  # 自动从 .env 读取 DEEPSEEK_API_KEY
    response = agent.chat("2025年8月15日沙特有山洪风险吗？")
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Generator

from llm_agent.tools import TOOL_REGISTRY, TOOL_DEFINITIONS
from llm_agent.prompt_templates import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from llm_agent.prompt_templates import SYSTEM_PROMPT
from llm_agent.tools.predict_tool import PredictTool
from llm_agent.safety import TRUST_STATEMENTS, CSI_VALUES


def _resolve_date(text: str) -> str:
    """将中文相对日期转为 YYYY-MM-DD 格式。"""
    today = datetime.now()
    mapping = {
        "今天": today,
        "明天": today + timedelta(days=1),
        "后天": today + timedelta(days=2),
        "昨天": today - timedelta(days=1),
        "前天": today - timedelta(days=2),
    }
    for word, date in mapping.items():
        text = text.replace(word, date.strftime("%Y-%m-%d"))
    return text


def _build_system_prompt() -> str:
    """构建含当前日期的 System Prompt。"""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    return (
        SYSTEM_PROMPT
        + f"\n\n当前日期: {today}（明天 = {tomorrow}）\n"
        + "用户可能使用中文相对日期（今天/明天/后天），请自动解析为 YYYY-MM-DD 格式后调用工具。"
    )


def _load_env():
    """从项目根目录 .env 加载环境变量。"""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv 未安装，依赖已有的环境变量


_load_env()


class MazuAgent:
    """MAZU 灾害预警 Agent。

    使用 LLM API（OpenAI 兼容接口），支持 Function Calling。
    API Key 优先级: 参数 > 环境变量 > .env 文件
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        verbose: bool = False,
    ):
        """
        Args:
            api_key: API Key（默认从环境变量 DEEPSEEK_API_KEY 读取）
            model: 模型名称
            base_url: API 地址
            verbose: 是否打印调试信息
        """
        from openai import OpenAI

        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量或传入 api_key 参数")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.verbose = verbose

        # 初始化工具实例
        self.tools: Dict[str, object] = {}
        for name, tool_cls in TOOL_REGISTRY.items():
            self.tools[name] = tool_cls()

        # KG 懒加载标记
        self._kg_ready = False

        if verbose:
            print(f"[Agent] 已加载 {len(self.tools)} 个工具: {list(self.tools.keys())}")

    def _ensure_kg(self):
        """懒加载知识图谱 + 种子案例（仅首次使用时构建）。"""
        if self._kg_ready:
            return
        try:
            from data.loader import load_to_dataframe
            from kg.graph_builder import KnowledgeGraphBuilder
            print("[Agent] 构建知识图谱（约需 30 秒）...")
            df = load_to_dataframe(
                "2025-06-15", "2025-06-15",
                variables=["orography", "daily_precip_total", "wind10_speed"],
                show_progress=True,
            )
            builder = KnowledgeGraphBuilder(coastal_orography_max=100.0)
            G = builder.build(df)
            if "query_kg_impact" in self.tools:
                self.tools["query_kg_impact"].set_graph(G)

            # 种子案例（15条：基于2025年沙特真实灾害Ground Truth）
            if "search_similar_cases" in self.tools:
                cases = [
                    # ══════ 山洪 (5起真实事件) ══════
                    {"disaster_type": "flash_flood", "date": "2025-01-06",
                     "description": "麦加/吉达特大洪水：强雷暴+冰雹袭击麦加省，NCM发布红色预警；"
                                    "1月7日麦加城区内涝严重，街道完全淹没，车辆被冲毁；"
                                    "拉比格市同步遭遇龙卷风+洪水叠加灾害，为冬季最严重洪灾",
                     "severity": 0.93, "location": "麦加 (21.4°N, 39.8°E)",
                     "measures": "民防部队紧急疏散低洼居民，封锁Umm Al-Qura大学周边及Wadi跨河路段；"
                                 "启动排水泵站全力排涝，吉达/麦加高速临时管制；"
                                 "发布居民避险通告：禁止进入山谷河道，转移至高地避险"},
                    {"disaster_type": "flash_flood", "date": "2025-03-06",
                     "description": "哈伊勒/布赖代春季首场大型山洪：短时强降雨席卷中部、北部山区，"
                                    "山谷洪水爆发，多条公路被淹没封闭，居民紧急转移",
                     "severity": 0.88, "location": "哈伊勒 (27.5°N, 41.7°E)",
                     "measures": "启动民防应急响应，转移山区居民至安全安置点；"
                                 "封锁65号/70号公路跨河路段，设置绕行指示；"
                                 "NCM发布24小时山洪监测预警，通知下游城镇提前防范"},
                    {"disaster_type": "flash_flood", "date": "2025-08-14",
                     "description": "塔伊夫冰雹洪水：巨型冰雹伴随暴雨袭击西南部山区，"
                                    "塔伊夫城区严重积水，低洼村镇被淹，NCM发布全域山洪预警",
                     "severity": 0.90, "location": "塔伊夫 (21.3°N, 40.4°E)",
                     "measures": "紧急疏散低洼村镇居民，开放学校体育馆作为临时避难所；"
                                 "排水泵站全力运转，清理冰雹堵塞的排水管道；"
                                 "暂停山区旅游活动，提醒居民远离河道和山谷"},
                    {"disaster_type": "flash_flood", "date": "2025-08-27",
                     "description": "阿西尔/吉赞/纳季兰大范围山洪：8月28日夜间暴雨，"
                                    "穆海伊勒-阿西尔街道洪水冲走大量私家车；"
                                    "气象预警覆盖全国10个行政区，西南部为重灾区，同步出现冰雹、大风",
                     "severity": 0.95, "location": "艾卜哈 (18.2°N, 42.5°E)",
                     "measures": "启动最高级别应急响应，10个行政区联动救援；"
                                 "阿西尔省出动直升机搜救被困人员，转移安置5000余人；"
                                 "封锁所有山区公路及Wadi通道，暂停吉赞港作业，学校全面停课"},
                    {"disaster_type": "flash_flood", "date": "2025-12-09",
                     "description": "吉达历史性特大洪水（全年最严重）：6小时累计降雨179毫米"
                                    "（当地年均仅55.6毫米），相当于3年降雨量在6小时内倾泻；"
                                    "城市主干道全淹没、交通瘫痪，至少2人遇难；"
                                    "麦地那省多辆汽车被洪水卷走，全境学校停课、公共交通停运",
                     "severity": 0.98, "location": "吉达 (21.5°N, 39.2°E)",
                     "measures": "宣布吉达进入紧急状态，调动军队参与救援；"
                                 "全市中小学停课3天，公共交通全部停运；"
                                 "开放所有政府建筑作为临时避难所，接收受灾居民；"
                                 "启动灾后评估：清理淤泥、修复排水系统、评估基础设施损毁"},

                    # ══════ 沙尘暴 (3起真实事件) ══════
                    {"disaster_type": "dust_wind", "date": "2025-05-04",
                     "description": "卡西姆/利雅得巨型哈布尘暴（Haboob）：沙尘墙高度超2000米，"
                                    "卡西姆省能见度接近零，风速最高达100km/h；"
                                    "NCM发布5省红色警报，高速公路、机场临时管制，居民居家避险",
                     "severity": 0.94, "location": "布赖代 (26.3°N, 44.0°E)",
                     "measures": "哈立德国王国际机场取消所有航班，关闭卡西姆区域高速；"
                                 "发布红色沙尘预警5省联动，建议居民居家避险、紧闭门窗；"
                                 "医院备足呼吸系统急救物资，免费发放N95口罩；"
                                 "暂停所有户外施工及油田作业，能见度恢复后再评估复工"},
                    {"disaster_type": "dust_wind", "date": "2025-05-16",
                     "description": "全年最强持续性沙尘过程：西北风持续4天，风速超25节（46km/h），"
                                    "全国覆盖，北部拉夫哈、哈费尔巴廷、东部达曼沙尘最严重；"
                                    "全国能见度普遍低于1公里，沙尘输送至红海、阿拉伯海；"
                                    "当月累计12天沙尘天，为全年沙尘强度峰值",
                     "severity": 0.97, "location": "哈费尔巴廷 (28.4°N, 46.0°E)",
                     "measures": "全国范围启动沙尘暴应急预案，各地机场实施流量管制；"
                                 "暂停北部油田户外作业4天，东部省工业区降低生产负荷；"
                                 "高速公路限速降至40km/h，能见度低于500米时立即封路；"
                                 "学校停课2天，建议心肺疾病患者避免外出，医院增开呼吸科门诊"},
                    {"disaster_type": "dust_wind", "date": "2025-06-30",
                     "description": "东部省/汉志地区持续性沙尘：覆盖东部省、麦加东部、麦地那、利雅得；"
                                    "7月1日利雅得能见度仅3-5公里，机场限流；"
                                    "叠加同期极端高温（超50°C），健康风险双重叠加，部分中小学临时停课",
                     "severity": 0.91, "location": "利雅得 (24.7°N, 46.7°E)",
                     "measures": "利雅得机场启动低能见度运行程序，航班延误率达60%；"
                                 "中小学暂停户外活动并部分停课，医院启动高温+沙尘联合应急预案；"
                                 "环卫部门增加道路洒水降尘频次，电力部门启动空调负荷调峰；"
                                 "发布健康警告：老人儿童避免外出，户外工作者佩戴N95口罩"},

                    # ══════ 极端高温 (2起真实事件) ══════
                    {"disaster_type": "extreme_heat", "date": "2025-05-25",
                     "description": "历史性热浪：沙特录得52.2°C极端高温，打破历史纪录；"
                                    "中东/北非地区升温速度为全球平均2倍，极端高温事件频次显著增加",
                     "severity": 0.99, "location": "全国 (重点关注: 东部省、利雅得)",
                     "measures": "11:00-16:00全面禁止户外作业，违者重罚；"
                                 "全国开放避暑中心，延长开放至凌晨，提供饮水和空调；"
                                 "电力部门启动最高级别调峰预案，确保电网稳定；"
                                 "医院急诊科全员待命，储备中暑急救药品和冰毯；"
                                 "呼吁居民错峰用电，将空调设定在24°C以上节能"},
                    {"disaster_type": "extreme_heat", "date": "2025-06-01",
                     "description": "2025年朝觐季极端高温：6月初气温飙升至47°C，"
                                    "朝圣者在阿拉法特山忍受酷热；"
                                    "沙特当局发布多项防暑措施，敦促朝圣者在高峰时段待在室内",
                     "severity": 0.96, "location": "麦加 (21.4°N, 39.8°E)",
                     "measures": "在阿拉法特山和米纳设置大量喷雾降温装置和遮阳帐篷；"
                                 "免费发放饮用水和防晒用品，部署移动医疗站点每500米一处；"
                                 "限制老年朝圣者11:00-15:00户外活动，开通中暑急救绿色通道；"
                                 "利用短信和App向所有朝圣者推送高温预警和防暑指南"},

                    # ══════ 沿海风浪 (3起典型事件) ══════
                    {"disaster_type": "coastal_wave", "date": "2025-01-15",
                     "description": "红海北部强风引发大浪，延布港和吉达港船舶作业受限，"
                                    "近岸渔船无法出海，海浪高度超4米",
                     "severity": 0.80, "location": "延布 (24.1°N, 38.1°E)",
                     "measures": "暂停小型船舶出海，港口货物装卸作业推迟；"
                                 "向渔民和航运公司发布风浪红色预警；加固近岸设施防浪"},
                    {"disaster_type": "coastal_wave", "date": "2025-11-20",
                     "description": "阿拉伯湾强东北风导致达曼沿岸海浪超3米，"
                                    "沿海低洼公路部分被淹，朱拜勒工业港受影响",
                     "severity": 0.85, "location": "达曼 (26.4°N, 50.1°E)",
                     "measures": "封闭沿海低洼路段，朱拜勒工业港暂停作业；"
                                 "居民远离海岸线，海岸警卫队加强巡逻；加固海堤临时防护"},
                    {"disaster_type": "coastal_wave", "date": "2025-12-28",
                     "description": "年末强风袭击阿拉伯湾沿岸，超80km/h强风引发恶劣海况，"
                                    "沙特东部沿海多个港口发布禁航令，沙尘+风浪复合灾害",
                     "severity": 0.82, "location": "朱拜勒 (27.0°N, 49.7°E)",
                     "measures": "达曼、朱拜勒、拉斯坦努拉港口全面暂停船舶进出港；"
                                 "沿海公路限速并设置防浪屏障，海岸巡逻队24小时值守；"
                                 "同步应对沙尘+风浪复合灾害：港口工人佩戴N95口罩，暂停户外作业"},
                ]
                for c in cases:
                    self.tools["search_similar_cases"].cr.add_case(
                        disaster_type=c["disaster_type"],
                        date=c["date"],
                        description=c["description"],
                        severity=c["severity"],
                        location=c.get("location", ""),
                        measures=c.get("measures", ""),
                    )

            self._kg_ready = True
            print("[Agent] 知识图谱 + 12条案例就绪")
        except Exception as e:
            print(f"[Agent] KG/案例初始化失败（部分功能不可用）: {e}")
            self._kg_ready = True

    def chat(self, user_message: str) -> str:
        """单轮对话（含工具调用循环）。"""
        user_message = _resolve_date(user_message)
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_message},
        ]

        tool_results = {}

        for _ in range(5):
            if self.verbose:
                print(f"[Agent] → 调用 LLM (messages={len(messages)})")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.3,
                max_tokens=2048,
            )

            msg = response.choices[0].message

            if msg.content and not msg.tool_calls:
                text = self._add_source_citations(msg.content, tool_results)
                return text

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    # KG 工具懒加载
                    if tool_name in ("query_kg_impact",):
                        self._ensure_kg()
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name](**args)
                        except Exception as e:
                            import traceback
                            result = {"status": "error", "message": f"{e}", "detail": traceback.format_exc()}
                    else:
                        result = {"status": "error", "message": f"未知工具: {tool_name}"}

                    tool_results[tool_name] = result
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

        return "抱歉，处理超时。"

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """流式对话 — 最终回复逐字输出，工具调用状态实时显示。"""
        user_message = _resolve_date(user_message)
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_message},
        ]

        tool_results = {}

        for _ in range(5):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.3,
                max_tokens=2048,
                stream=True,
                stream_options={"include_usage": False},
            )

            # 收集流式响应
            msg_content = ""
            msg_tool_calls = []
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                if delta.content:
                    msg_content += delta.content
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        while len(msg_tool_calls) <= idx:
                            msg_tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                        if tc_delta.id:
                            msg_tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                msg_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                msg_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

            # 最终回复 → 直接逐字 yield
            if msg_content and not msg_tool_calls:
                text = self._add_source_citations(msg_content, tool_results)
                yield text
                return

            # 工具调用
            if msg_tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": msg_content or None,
                    "tool_calls": [{"id": tc["id"], "type": "function", "function": tc["function"]}
                                   for tc in msg_tool_calls],
                }
                messages.append(assistant_msg)

                for tc in msg_tool_calls:
                    tool_name = tc["function"]["name"]
                    # KG 工具懒加载
                    if tool_name in ("query_kg_impact",):
                        self._ensure_kg()
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    yield f"\n🔧 {tool_name}...\n"

                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name](**args)
                        except Exception as e:
                            result = {"status": "error", "message": str(e)}
                    else:
                        result = {"status": "error", "message": f"未知工具: {tool_name}"}

                    tool_results[tool_name] = result
                    preview = str(result.get("message", ""))[:120]
                    yield f"✅ {preview}\n\n"

                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })

        yield "抱歉，处理超时。"

    def _add_source_citations(self, text: str, tool_results: Dict) -> str:
        """在回复末尾追加数据来源标注（仅添加，不做校验）。"""
        if not tool_results:
            return text

        sources = []
        if "predict_risk" in tool_results:
            r = tool_results["predict_risk"]
            if isinstance(r, dict) and r.get("status") == "success":
                dtype = r.get("disaster_type", "")
                csi = CSI_VALUES.get(dtype, "N/A")
                sources.append(TRUST_STATEMENTS["predict_risk"].format(csi=csi))
        if "query_kg_impact" in tool_results:
            r = tool_results["query_kg_impact"]
            if isinstance(r, dict) and r.get("status") == "success":
                sources.append(TRUST_STATEMENTS["query_kg_impact"])
        if "search_similar_cases" in tool_results:
            r = tool_results["search_similar_cases"]
            if isinstance(r, dict) and r.get("status") == "success":
                sources.append(TRUST_STATEMENTS["search_similar_cases"])

        if sources:
            text += "\n\n---\n" + "\n".join(sources)

        return text

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
from llm_agent.safety import sanitize_llm_output, TRUST_STATEMENTS, CSI_VALUES


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

            # 种子案例（12条：四灾害各3条，覆盖不同区域和严重程度）
            if "search_similar_cases" in self.tools:
                cases = [
                    # ── 山洪 (flash_flood) ──
                    {"disaster_type": "flash_flood", "date": "2025-08-15",
                     "description": "麦加强降水引发Wadi Ibrahim山洪，水位暴涨3米",
                     "severity": 0.95, "location": "麦加 (21.4°N, 39.8°E)",
                     "measures": "疏散低洼地区居民，关闭Umm Al-Qura大学周边道路，出动民防部队救援"},
                    {"disaster_type": "flash_flood", "date": "2025-07-20",
                     "description": "阿西尔山区暴雨导致山洪，艾卜哈市多处被淹，交通中断",
                     "severity": 0.88, "location": "艾卜哈 (18.2°N, 42.5°E)",
                     "measures": "启动民防应急响应，转移山区居民至安全点，封锁Wadi跨河路段"},
                    {"disaster_type": "flash_flood", "date": "2025-10-05",
                     "description": "吉达北部突降暴雨，城市排水系统超负荷，多个街区积水",
                     "severity": 0.82, "location": "吉达 (21.5°N, 39.2°E)",
                     "measures": "关闭积水路段交通，启动排水泵站，发布居民避险通告"},

                    # ── 极端高温 (extreme_heat) ──
                    {"disaster_type": "extreme_heat", "date": "2025-07-25",
                     "description": "极端热浪袭击东部省，达曼气温达51°C，体感温度超55°C",
                     "severity": 0.90, "location": "达曼 (26.4°N, 50.1°E)",
                     "measures": "限制户外作业时间(11:00-15:00禁止)，开放避暑中心，电力部门启动调峰预案"},
                    {"disaster_type": "extreme_heat", "date": "2025-08-10",
                     "description": "利雅得连续5天高温超48°C，夜间温度不降，空调负荷创纪录",
                     "severity": 0.93, "location": "利雅得 (24.7°N, 46.7°E)",
                     "measures": "延长避暑中心开放至24:00，医院备足中暑急救物资，呼吁节约用电"},
                    {"disaster_type": "extreme_heat", "date": "2025-06-28",
                     "description": "塔布克地区异常高温达47°C，远超常年同期(40°C)",
                     "severity": 0.85, "location": "塔布克 (28.4°N, 36.6°E)",
                     "measures": "发布高温橙色预警，暂停学校户外活动，提醒朝觐者注意防暑"},

                    # ── 沙尘强风 (dust_wind) ──
                    {"disaster_type": "dust_wind", "date": "2025-03-12",
                     "description": "强西北风引发大范围沙尘暴，利雅得能见度降至200米",
                     "severity": 0.92, "location": "利雅得 (24.7°N, 46.7°E)",
                     "measures": "哈立德国王机场取消所有航班，全市学校停课，建议市民佩戴口罩减少外出"},
                    {"disaster_type": "dust_wind", "date": "2025-04-05",
                     "description": "北部哈费尔巴廷地区强沙尘暴，风速达70km/h，油田作业暂停",
                     "severity": 0.88, "location": "哈费尔巴廷 (28.4°N, 46.0°E)",
                     "measures": "暂停油田户外作业，关闭85号公路部分路段，发布红色沙尘预警"},
                    {"disaster_type": "dust_wind", "date": "2025-02-18",
                     "description": "布赖代至哈伊勒一线沙尘暴，伴随强降温，牲畜受损严重",
                     "severity": 0.78, "location": "布赖代 (26.3°N, 44.0°E)",
                     "measures": "牧民转移牲畜至棚舍，高速公路限速降至60km/h，能见度低于500米时封路"},

                    # ── 沿海风浪 (coastal_wave) ──
                    {"disaster_type": "coastal_wave", "date": "2025-01-15",
                     "description": "红海北部强风引发大浪，延布港和吉达港船舶作业受限",
                     "severity": 0.80, "location": "延布 (24.1°N, 38.1°E)",
                     "measures": "暂停小型船舶出海，港口货物装卸推迟，向渔民发布风浪预警"},
                    {"disaster_type": "coastal_wave", "date": "2025-11-20",
                     "description": "阿拉伯湾强东北风导致达曼沿岸海浪超3米，沿海公路部分被淹",
                     "severity": 0.85, "location": "达曼 (26.4°N, 50.1°E)",
                     "measures": "封闭沿海低洼路段，朱拜勒工业港暂停作业，居民远离海岸线"},
                    {"disaster_type": "coastal_wave", "date": "2025-03-08",
                     "description": "吉赞沿海受印度洋涌浪影响，渔船无法出海，近岸设施受损",
                     "severity": 0.75, "location": "吉赞 (16.9°N, 42.6°E)",
                     "measures": "渔船全部召回港口，加固近岸养殖设施，发布海上作业禁令"},
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
                text, _ = sanitize_llm_output(text, tool_results)
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

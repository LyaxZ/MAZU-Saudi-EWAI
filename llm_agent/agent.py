"""
MAZU Agent — LLM Function Calling 主循环

用法:
    from llm_agent.agent import MazuAgent
    agent = MazuAgent()  # 自动从 .env 读取 DEEPSEEK_API_KEY
    response = agent.chat("2025年8月15日沙特有山洪风险吗？")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Generator

from llm_agent.tools import TOOL_REGISTRY, TOOL_DEFINITIONS
from llm_agent.prompt_templates import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from llm_agent.safety import sanitize_llm_output, TRUST_STATEMENTS, CSI_VALUES


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

    使用 DeepSeek API（OpenAI 兼容接口），支持 Function Calling。
    API Key 优先级: 参数 > 环境变量 > .env 文件
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        verbose: bool = False,
    ):
        """
        Args:
            api_key: DeepSeek API Key（默认从环境变量 DEEPSEEK_API_KEY 读取）
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

        if verbose:
            print(f"[Agent] 已加载 {len(self.tools)} 个工具: {list(self.tools.keys())}")

    def chat(self, user_message: str) -> str:
        """单轮对话（含工具调用循环）。

        Args:
            user_message: 用户输入

        Returns:
            Agent 最终回复
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_message},
        ]

        tool_results = {}  # 记录所有工具调用结果用于安全校验

        # 最多 5 轮工具调用循环
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

            # 如果是最终回复
            if msg.content and not msg.tool_calls:
                # 添加来源标注
                text = self._add_source_citations(msg.content, tool_results)
                # 安全校验
                text, warnings = sanitize_llm_output(text, tool_results)
                if self.verbose and warnings:
                    print(f"[Agent] 安全警告: {warnings}")
                return text

            # 如果是工具调用
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if self.verbose:
                        print(f"[Agent] 🔧 调用工具: {tool_name}({args})")

                    # 执行工具
                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name](**args)
                        except Exception as e:
                            result = {"status": "error", "message": str(e)}
                    else:
                        result = {"status": "error", "message": f"未知工具: {tool_name}"}

                    tool_results[tool_name] = result

                    if self.verbose:
                        msg_preview = str(result)[:200]
                        print(f"[Agent] ✅ 工具返回: {msg_preview}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

        return "抱歉，处理超时。请简化您的问题再试。"

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """流式对话（用于 CLI 逐字输出）。

        注意：工具调用期间不流式，仅最终回复流式。
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
            )

            msg = response.choices[0].message

            if msg.content and not msg.tool_calls:
                text = self._add_source_citations(msg.content, tool_results)
                text, _ = sanitize_llm_output(text, tool_results)
                yield text
                return

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    yield f"\n🔧 调用工具: {tool_name}...\n"

                    if tool_name in self.tools:
                        try:
                            result = self.tools[tool_name](**args)
                        except Exception as e:
                            result = {"status": "error", "message": str(e)}
                    else:
                        result = {"status": "error", "message": f"未知工具: {tool_name}"}

                    tool_results[tool_name] = result
                    preview = str(result.get("message", ""))[:150]
                    yield f"✅ {preview}\n"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

        yield "抱歉，处理超时。"

    def _add_source_citations(self, text: str, tool_results: Dict) -> str:
        """在回复末尾追加数据来源标注。"""
        if not tool_results:
            return text

        sources = []
        if "predict_risk" in tool_results:
            r = tool_results["predict_risk"]
            if isinstance(r, dict):
                dtype = r.get("disaster_type", "")
                csi = CSI_VALUES.get(dtype, "N/A")
                sources.append(TRUST_STATEMENTS["predict_risk"].format(csi=csi))
        if "query_kg_impact" in tool_results:
            sources.append(TRUST_STATEMENTS["query_kg_impact"])
        if "search_similar_cases" in tool_results:
            sources.append(TRUST_STATEMENTS["search_similar_cases"])

        if sources:
            text += "\n\n---\n" + "\n".join(sources)

        return text

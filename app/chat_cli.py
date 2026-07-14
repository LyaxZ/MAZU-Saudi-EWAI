#!/usr/bin/env python
"""
MAZU 预警智能体 — 命令行对话界面

用法:
    # 设置 API Key
    set DEEPSEEK_API_KEY=sk-xxxxxxxx
    python app/chat_cli.py

    # 或直接传入
    python app/chat_cli.py --api-key sk-xxxxxxxx

支持的命令:
    /help     - 显示帮助
    /example  - 显示示例问题
    /clear    - 清屏
    /quit     - 退出
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent.agent import MazuAgent

# ═══════════════════════════════════════════════
# 界面样式
# ═══════════════════════════════════════════════

HEADER = r"""
╔══════════════════════════════════════════════╗
║        🏔 MAZU 沙特多灾种预警智能体          ║
║   暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪   ║
╚══════════════════════════════════════════════╝
"""

EXAMPLES = [
    "2025年8月15日沙特有山洪风险吗？",
    "分析明天利雅得地区的热浪风险",
    "红海沿岸有没有风浪预警？",
    "帮我看看8月20日的沙尘暴预测",
]

HELP_TEXT = """
┌─ 可用命令 ─────────────────────────────┐
│ /help      显示此帮助                   │
│ /example   显示示例问题                 │
│ /clear     清屏                        │
│ /quit      退出                        │
│                                         │
│ 直接输入自然语言问题即可查询预警信息      │
└─────────────────────────────────────────┘
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MAZU 预警智能体 CLI")
    parser.add_argument("--api-key", help="DeepSeek API Key")
    parser.add_argument("--model", default="deepseek-chat", help="模型名称")
    parser.add_argument("--verbose", action="store_true", help="显示调试信息")
    args = parser.parse_args()

    # 获取 API Key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("\n❌ 未设置 DEEPSEEK_API_KEY")
        print("请通过以下方式之一设置：")
        print('  1. 环境变量: set DEEPSEEK_API_KEY=sk-xxxxxxxx')
        print('  2. 命令行参数: python app/chat_cli.py --api-key sk-xxxxxxxx')
        return

    # 初始化 Agent
    print(HEADER)
    print("⏳ 正在加载模型和知识图谱...")
    try:
        agent = MazuAgent(
            api_key=api_key,
            model=args.model,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        return

    print("✅ 就绪！输入问题开始对话（输入 /help 查看帮助）\n")

    # 主循环
    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("👋 再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/example":
                print("\n💡 示例问题:")
                for i, e in enumerate(EXAMPLES, 1):
                    print(f"  {i}. {e}")
                print()
            elif cmd == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                print(HEADER)
            else:
                print(f"未知命令: {user_input}，输入 /help 查看帮助")
            continue

        # 调用 Agent
        print("\n🤖 MAZU: ", end="", flush=True)
        try:
            response = agent.chat(user_input)
            print(response)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
        print()


if __name__ == "__main__":
    main()

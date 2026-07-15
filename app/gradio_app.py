"""
MAZU 沙特多灾种预警智能体 — DeepSeek 风格对话界面

用法:
    python app/gradio_app.py
"""

import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from llm_agent.agent import MazuAgent


# ═══════════════════════════════════════════════
# CSS: DeepSeek 风格
# ═══════════════════════════════════════════════

DEEPSEEK_CSS = """
body, .gradio-container { background: #f7f8fa !important; }
footer { display: none !important; }
.main-header { text-align: center; padding: 20px 0 10px 0; }
.main-header h1 { font-size: 28px; font-weight: 700; color: #1a1a2e; margin: 0; }
.main-header p { color: #8e8ea0; font-size: 14px; margin: 4px 0 0 0; }
#chatbot { border-radius: 12px; max-width: 100%; }
#chatbot > div { max-width: 100% !important; }
#chatbot .message-wrap { max-width: 100% !important; }
.input-row { background: #fff; border-radius: 16px; border: 1px solid #e0e0e0; padding: 8px 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.input-row textarea { border: none !important; box-shadow: none !important; resize: none; font-size: 15px; padding: 8px; }
.send-btn { background: #4f46e5 !important; border: none !important; border-radius: 10px !important; color: white !important; font-weight: 600 !important; padding: 10px 20px !important; }
.send-btn:hover { background: #4338ca !important; }
.source-cite { font-size: 11px; color: #9ca3af; margin-top: 8px; padding-top: 6px; border-top: 1px solid #f0f0f0; }
"""


# ═══════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:

        gr.HTML("""
        <div class="main-header">
            <h1>🏔 MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 | LightGBM + 知识图谱 + DeepSeek</p>
        </div>
        """)

        chatbot = gr.Chatbot(
            label="", height=520, elem_id="chatbot", show_label=False,
        )

        with gr.Row(elem_classes=["input-row"]):
            msg_input = gr.Textbox(
                placeholder="输入问题，例如：2025年8月15日沙特有山洪风险吗？",
                scale=10, container=False, show_label=False, max_lines=4,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1, elem_classes=["send-btn"])

        _agent = MazuAgent(verbose=False)

        def respond(message, history):
            history = history or []
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "⏳ 正在分析..."},
            ]

            full = ""
            last_tool = ""
            for chunk in _agent.chat_stream(message):
                full += chunk
                tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                if tools and tools[-1] != last_tool:
                    last_tool = tools[-1]
                    names = {
                        "predict_risk": "获取风险预测数据",
                        "query_kg_impact": "分析知识图谱影响链",
                        "search_similar_cases": "检索历史相似案例",
                    }
                    yield history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": f"⏳ {names.get(last_tool, '处理中')}..."},
                    ]

            clean = re.sub(r'\n?🔧[^\n]*\n', '\n', full)
            clean = re.sub(r'\n?✅[^\n]*\n', '\n', clean)
            main_text, source_text = clean, ""
            if "\n---\n" in clean:
                parts = clean.split("\n---\n", 1)
                main_text = parts[0].strip()
                source_text = parts[1].strip() if len(parts) > 1 else ""

            if source_text:
                source_html = f'<div class="source-cite">{source_text.replace(chr(10), " · ")}</div>'
                final = main_text + "\n" + source_html
            else:
                final = main_text

            final = re.sub(r'\n{3,}', '\n\n', final)
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": final},
            ]

        send_btn.click(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot]).then(lambda: "", None, msg_input)
        msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot]).then(lambda: "", None, msg_input)

        gr.Examples(
            examples=[
                "2025年8月15日沙特有山洪风险吗？",
                "明天利雅得地区会不会有热浪？",
                "红海沿岸有没有风浪预警？",
                "帮我看看8月20日的沙尘暴预测",
            ],
            inputs=msg_input,
            label="💡 试试这些问题",
        )

    return app


def launch_app(share: bool = False, **kwargs):
    kwargs.setdefault("server_name", "0.0.0.0")
    kwargs.setdefault("server_port", 7860)
    kwargs.setdefault("css", DEEPSEEK_CSS)
    app = build_ui()
    app.launch(share=share, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAZU 预警智能体 Web")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch_app(share=args.share, server_port=args.port)

"""MAZU 沙特多灾种预警智能体 — Web 界面"""
import sys, os, re, logging

import gradio as gr
from llm_agent.agent import MazuAgent

# === 日志 ===
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("MAZU")

CSS = """
*{box-sizing:border-box}
body{background:#f1f5f9!important;font-family:'Segoe UI',system-ui,-apple-system,sans-serif}
.gradio-container{max-width:900px!important;margin:24px auto!important;padding:0 16px!important}

/* 顶部标题 */
.main-header{text-align:center;padding:32px 24px 24px;margin-bottom:16px;
  background:#fff;border-radius:16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.main-header h1{font-size:24px;margin:0;color:#1e293b;font-weight:700;letter-spacing:1px}
.main-header p{font-size:13px;color:#64748b;margin:8px 0 0}

/* 对话框 */
#chatbot{border-radius:16px!important;min-height:480px;background:#fff!important;
  box-shadow:0 1px 4px rgba(0,0,0,.06)}
#chatbot > div{padding:24px!important}
.input-box{background:#fff;border-top:1px solid #f1f5f9;padding:16px 24px;
  border-radius:0 0 16px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.input-box textarea{border:2px solid #e2e8f0!important;border-radius:12px!important;
  padding:14px 18px!important;font-size:15px!important;
  transition:border-color .2s;outline:none!important}
.input-box textarea:focus{border-color:#4f46e5!important;box-shadow:0 0 0 3px rgba(79,70,229,.1)!important}
.send-btn{background:#4f46e5!important;color:#fff!important;font-weight:600!important;
  padding:14px 32px!important;border-radius:12px!important;border:none!important;
  transition:all .2s}
.send-btn:hover{background:#4338ca!important;transform:translateY(-1px);box-shadow:0 2px 8px rgba(79,70,229,.3)}

/* 来源引用 */
.source-cite{font-size:11px;color:#94a3b8;margin-top:8px;padding-top:6px;border-top:1px solid #f1f5f9}
"""


# === UI ===
def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:
        gr.HTML("""<div class="main-header">
            <h1>MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 ｜ LightGBM · 知识图谱 · LLM Agent</p>
        </div>""")

        chatbot = gr.Chatbot(label="", height=520, elem_id="chatbot", show_label=False)
        with gr.Row(elem_classes=["input-box"]):
            msg = gr.Textbox(
                placeholder="输入问题，如：明天利雅得会有热浪吗？",
                scale=10, container=False, show_label=False, max_lines=3)
            send = gr.Button("发送", variant="primary", scale=1, elem_classes=["send-btn"])

        _agent = [None]

        def get_agent():
            if _agent[0] is None:
                _agent[0] = MazuAgent(verbose=False)
            return _agent[0]

        def respond(message, history):
            history = history or []
            yield history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "⏳"}]
            full, last = "", ""
            for chunk in get_agent().chat_stream(message):
                full += chunk
                tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                if tools and tools[-1] != last:
                    last = tools[-1]
                    n = {"predict_risk": "获取风险预测",
                         "query_kg_impact": "分析影响链",
                         "search_similar_cases": "检索案例"}
                    yield history + [{"role": "user", "content": message},
                                     {"role": "assistant",
                                      "content": f"⏳ {n.get(last, '处理中')}..."}]
            clean = re.sub(r'\n?🔧[^\n]*\n', '\n', full)
            clean = re.sub(r'\n?✅[^\n]*\n', '\n', clean)
            clean = re.sub(r'^\s*---\s*\n+', '', clean)
            main, src = clean, ""
            if "\n---\n" in clean:
                parts = clean.rsplit("\n---\n", 1)
                main, src = parts[0].strip(), parts[1].strip()
            final = main
            if src:
                final += f'\n<div class="source-cite">{src.replace(chr(10), " · ")}</div>'
            final = re.sub(r'\n{3,}', '\n\n', final)
            yield history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": final}]

        send.click(fn=respond, inputs=[msg, chatbot], outputs=[chatbot]) \
            .then(fn=lambda: "", outputs=[msg])
        msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[chatbot]) \
            .then(fn=lambda: "", outputs=[msg])
        gr.Examples(
            examples=["2025年8月15日沙特有山洪风险吗？",
                       "明天利雅得地区会不会有热浪？",
                       "红海沿岸有没有风浪预警？",
                       "看看8月20日的沙尘暴预测"],
            inputs=msg)

    return app


def launch_app(share=False, **kw):
    kw.setdefault("server_name", "0.0.0.0")
    kw.setdefault("server_port", 7860)
    log.info("MAZU Web 启动")
    app = build_ui()
    app.launch(share=share, css=CSS, **kw)


def main():
    """run.py 入口"""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    launch_app(share=args.share, server_port=args.port)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    launch_app(share=p.parse_args().share, server_port=p.parse_args().port)

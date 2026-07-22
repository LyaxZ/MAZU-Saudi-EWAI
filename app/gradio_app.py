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
body{background:linear-gradient(135deg,#e8eaf6 0%,#e3f2fd 50%,#f3e5f5 100%)!important;
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;min-height:100vh}
.gradio-container{max-width:90vw!important;margin:16px auto!important;padding:0!important;
  border:none!important;box-shadow:none!important;background:transparent!important}

/* 顶部标题 */
.main-header{text-align:center;padding:24px 28px 18px;margin-bottom:10px;
  background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.04);
  border-left:4px solid #7c3aed}
.main-header h1{font-size:22px;margin:0;color:#4c1d95;font-weight:700}
.main-header p{font-size:13px;color:#7c3aed;margin:4px 0 0;opacity:.8}

/* 对话框 */
#chatbot{border-radius:12px!important;min-height:500px;background:#fff!important;
  box-shadow:0 2px 8px rgba(0,0,0,.04)!important;border:none!important}
#chatbot > div{padding:18px!important}

/* 输入区 */
.input-box{background:#fff;border-top:1px solid #ede9fe;padding:12px 18px;
  border-radius:0 0 12px 12px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
.input-box textarea, .input-box input{
  border:1.5px solid #ddd6fe!important;border-radius:10px!important;
  padding:12px 16px!important;font-size:14px!important;
  outline:none!important;box-shadow:none!important;background:#faf5ff!important}
.input-box textarea:focus, .input-box input:focus{
  border-color:#7c3aed!important;background:#fff!important;
  box-shadow:0 0 0 3px rgba(124,58,237,.1)!important}

/* 按钮 */
.send-btn, button.primary{
  background:linear-gradient(135deg,#7c3aed,#6d28d9)!important;color:#fff!important;
  font-weight:600!important;padding:12px 28px!important;border-radius:10px!important;
  border:none!important;box-shadow:0 2px 8px rgba(124,58,237,.2)!important;
  cursor:pointer!important;transition:all .15s}
.send-btn:hover, button.primary:hover{
  background:linear-gradient(135deg,#6d28d9,#5b21b6)!important;
  box-shadow:0 4px 12px rgba(124,58,237,.3)!important;transform:translateY(-1px)}

/* 示例 */
.gr-examples label{color:#7c3aed!important;font-size:12px!important;font-weight:600!important}
.gr-examples .example{border-color:#ddd6fe!important;background:#faf5ff!important;
  color:#4c1d95!important;border-radius:8px!important}
.gr-examples .example:hover{background:#ede9fe!important;border-color:#7c3aed!important}

/* 来源引用 */
.source-cite{font-size:11px;color:#a78bfa;margin-top:8px;padding-top:6px;border-top:1px solid #ede9fe}
"""


# === UI ===
def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体", css=CSS) as app:
        gr.HTML("""<div class="main-header">
            <h1>MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 ｜ LightGBM · 知识图谱 · LLM Agent</p>
        </div>""")

        chatbot = gr.Chatbot(label="", height=520, elem_id="chatbot", show_label=False,
            value=[{"role": "assistant",
                    "content": "👋 我是 **MAZU 沙特多灾种预警智能体**。\n\n我可以帮你查询四类灾害风险：\n- ⚡ **暴雨山洪** — 如「8月28日阿西尔有山洪风险吗」\n- 🔥 **极端高温** — 如「明天利雅得会有热浪吗」\n- 🌪️ **沙尘强风** — 如「5月中旬沙特有沙尘暴吗」\n- 🌊 **沿海风浪** — 如「红海沿岸有没有风浪预警」\n\n也可以问处置建议，如「如果吉达山洪红色预警该怎么办？」"}])
        with gr.Row(elem_classes=["input-box"]):
            msg = gr.Textbox(
                placeholder="输入问题，如：明天利雅得会有热浪吗？",
                scale=10, show_label=False, max_lines=3)
            send = gr.Button("发送", variant="primary", scale=1, elem_classes=["send-btn"])

        _agent = [None]
        _first_load_error = [""]

        def get_agent():
            if _agent[0] is None:
                try:
                    _agent[0] = MazuAgent(verbose=False)
                    _first_load_error[0] = ""
                except Exception as e:
                    _first_load_error[0] = str(e)
                    log.error(f"Agent 初始化失败: {e}")
            return _agent[0], _first_load_error[0]

        def respond(message, history):
            history = history or []
            agent, init_err = get_agent()
            if init_err:
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant",
                     "content": f"❌ **系统初始化失败**\n\n{init_err}\n\n请检查 `.env` 文件中的 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 是否已正确配置。"}
                ]
                return

            yield history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "⏳"}]
            full, last = "", ""
            for chunk in agent.chat_stream(message):
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
            .then(fn=lambda: gr.update(value=""), outputs=[msg])
        msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[chatbot]) \
            .then(fn=lambda: gr.update(value=""), outputs=[msg])
        gr.Examples(
            examples=["8月28日阿西尔地区有山洪风险吗",
                       "明天利雅得会有热浪吗",
                       "如果明天吉达有山洪红色预警，应该采取什么措施",
                       "5月中旬沙特有沙尘暴吗"],
            inputs=msg)

    return app


def launch_app(share=False, **kw):
    kw.setdefault("server_name", "0.0.0.0")
    kw.setdefault("server_port", 7860)
    log.info("MAZU Web 启动")
    app = build_ui()
    app.launch(share=share, **kw)


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

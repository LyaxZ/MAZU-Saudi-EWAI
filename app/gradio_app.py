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
/* ====== 全局背景 ====== */
body, .gradio-app{
  background:linear-gradient(160deg,#eef6fb 0%,#e0f2fe 45%,#ecfeff 100%)!important;
  font-family:'Segoe UI',system-ui,sans-serif!important}

/* ====== 主容器 ====== */
.gradio-container{max-width:75vw!important;margin:18px auto!important;
  padding:0!important;background:transparent!important;border:none!important}

/* ====== 清除 Gradio 默认外框（精准） ====== */
.gradio-container > .form,
.gradio-container fieldset{border:none!important;background:transparent!important}

/* ====== 标题栏 ====== */
.main-header{text-align:center;padding:22px 30px 18px;margin-bottom:12px;
  background:#fff!important;border-radius:14px;
  box-shadow:0 2px 12px rgba(8,145,178,.08);
  border:1px solid rgba(8,145,178,.12)}
.main-header h1{font-size:22px;color:#0c4a6e;font-weight:700;margin:0}
.main-header p{font-size:13px;color:#0891b2;margin:6px 0 0}

/* ====== 聊天区 ====== */
#chatbot{border-radius:14px!important;min-height:500px;
  background:#f0f9ff!important;box-shadow:0 2px 12px rgba(8,145,178,.06)!important;
  border:1px solid rgba(8,145,178,.1)!important}
#chatbot .messages{padding:16px 20px!important}

/* 气泡本身：只作用于 .message 元素 */
#chatbot .message{border-radius:12px!important;padding:12px 16px!important;
  margin:8px 12px!important;border:none!important;font-size:14px!important;line-height:1.6!important}
/* 气泡内文字：去除 Gradio 默认内边距 */
#chatbot .message p, #chatbot .message span, #chatbot .message div,
#chatbot .message li, #chatbot .message ul{
  padding:0!important;margin:4px 0!important}
#chatbot .message.bot{background:#fff!important;box-shadow:0 1px 3px rgba(0,0,0,.04)!important}
#chatbot .message.user{background:#e0f2fe!important;color:#0c4a6e!important;
  border:1px solid #bae6fd!important}

/* 隐藏 Gradio 内部多余包装框 */
#chatbot .message > div, #chatbot .message > span,
#chatbot .message-wrap, #chatbot .message-row{
  border:none!important;background:transparent!important;padding:0!important;margin:0!important}
#chatbot [data-testid="bot"], #chatbot [data-testid="user"]{
  border:none!important;background:transparent!important;padding:0!important}

/* 复制按钮：小巧 */
#chatbot .copy-button, #chatbot button[aria-label*="copy"],
#chatbot button[title*="复制"]{
  padding:2px 6px!important;font-size:11px!important;border-radius:4px!important;
  min-width:auto!important;width:auto!important;height:auto!important;
  background:transparent!important;border:none!important;opacity:.4!important}
#chatbot .copy-button:hover{opacity:.8!important}

/* ====== 输入区 ====== */
.input-box{background:#fff!important;border-top:1px solid #cffafe!important;
  padding:12px 16px!important;border-radius:0 0 14px 14px!important;
  box-shadow:0 2px 12px rgba(8,145,178,.06)!important}
.input-box > div{border:none!important;background:transparent!important;padding:0!important}

.input-box textarea{
  border:1.5px solid #bae6fd!important;border-radius:10px!important;
  padding:12px 16px!important;font-size:14px!important;background:#f0f9ff!important;
  outline:none!important;box-shadow:none!important}
.input-box textarea:focus{
  border-color:#0891b2!important;background:#fff!important;
  box-shadow:0 0 0 3px rgba(8,145,178,.12)!important}

/* ====== 按钮 ====== */
.send-btn, button.primary{
  background:linear-gradient(135deg,#0e7490,#0891b2)!important;color:#fff!important;
  font-weight:600!important;padding:12px 28px!important;border-radius:10px!important;
  border:none!important;box-shadow:0 2px 8px rgba(8,145,178,.2)!important}
.send-btn:hover, button.primary:hover{
  background:linear-gradient(135deg,#0891b2,#06b6d4)!important;
  box-shadow:0 4px 12px rgba(8,145,178,.3)!important}

/* ====== 示例 ====== */
.gr-examples label{color:#0e7490!important;font-size:12px!important;font-weight:600!important}
.gr-examples button{background:#ecfeff!important;border:1.5px solid #a5f3fc!important;
  color:#155e75!important;border-radius:8px!important;padding:4px 10px!important;font-size:12px!important}
.gr-examples button:hover{background:#cffafe!important;border-color:#0891b2!important}

/* ====== 滚动条 ====== */
#chatbot::-webkit-scrollbar{width:6px}
#chatbot::-webkit-scrollbar-track{background:transparent}
#chatbot::-webkit-scrollbar-thumb{background:#bae6fd;border-radius:3px}

/* ====== 来源引用 ====== */
.source-cite{font-size:11px;color:#67e8f9;margin-top:8px;padding-top:6px;border-top:1px solid #cffafe}
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
                ], ""
                return

            yield history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "⏳"}], ""
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
                                      "content": f"⏳ {n.get(last, '处理中')}..."}], ""
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
                             {"role": "assistant", "content": final}], ""

        send.click(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
        msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
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

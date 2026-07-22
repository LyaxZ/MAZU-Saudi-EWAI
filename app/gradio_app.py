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
/* ====== 全局：柔和蓝青背景 + 微纹理质感 ====== */
body, .gradio-app{
  background:
    radial-gradient(circle at 15% 20%, rgba(186,230,253,.35) 0%, transparent 45%),
    radial-gradient(circle at 85% 75%, rgba(165,243,252,.3) 0%, transparent 50%),
    linear-gradient(160deg,#eef6fb 0%,#e0f2fe 45%,#ecfeff 100%)!important;
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif!important;
  min-height:100vh}

/* ====== 主容器 ====== */
.gradio-container{max-width:75vw!important;margin:18px auto!important;
  padding:0!important;background:transparent!important;border:none!important}

/* ====== 清除 Gradio 默认边框 ====== */
.gradio-container fieldset, .gradio-container .panel,
.gradio-container .form, .gradio-container .block{
  border:none!important;box-shadow:none!important;background:transparent!important}

/* ====== 标题栏：玻璃质感 ====== */
.main-header{
  text-align:center;padding:24px 32px 20px;margin-bottom:12px;
  background:rgba(255,255,255,.85)!important;backdrop-filter:blur(12px);
  border-radius:16px;box-shadow:0 4px 20px rgba(8,145,178,.08);
  border:1px solid rgba(8,145,178,.12)}
.main-header h1{
  font-size:22px;color:#0c4a6e;font-weight:700;
  background:linear-gradient(90deg,#0e7490,#0891b2);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;margin:0}
.main-header p{font-size:13px;color:#0891b2;margin:6px 0 0;letter-spacing:.5px}

/* ====== 聊天区：毛玻璃 ====== */
#chatbot{
  border-radius:16px!important;min-height:500px;
  background:rgba(240,249,255,.7)!important;backdrop-filter:blur(8px);
  box-shadow:0 4px 24px rgba(8,145,178,.06)!important;
  border:1px solid rgba(8,145,178,.1)!important}
#chatbot > div{padding:20px!important}

/* 助手气泡：柔和白 */
#chatbot [class*="bot"] [class*="message"],
#chatbot [class*="bot"] [class*="bubble"],
#chatbot .bubble:not(.user){
  background:#fff!important;border-radius:14px!important;
  box-shadow:0 1px 3px rgba(0,0,0,.05)!important;
  border:1px solid #e0f2fe!important;padding:14px 18px!important}
/* 用户气泡：青蓝 */
#chatbot [class*="user"] [class*="message"],
#chatbot [class*="user"] [class*="bubble"],
#chatbot .bubble.user{
  background:linear-gradient(135deg,#0e7490,#0891b2)!important;color:#fff!important;
  border-radius:14px!important;box-shadow:0 2px 8px rgba(8,145,178,.2)!important;
  padding:14px 18px!important}

/* ====== 输入区：贴底玻璃 ====== */
.input-box{
  background:rgba(255,255,255,.9)!important;backdrop-filter:blur(8px);
  border-top:1px solid rgba(8,145,178,.12)!important;
  padding:14px 18px!important;border-radius:0 0 16px 16px!important;
  box-shadow:0 4px 20px rgba(8,145,178,.06)!important}

/* 输入框 */
.input-box textarea, .input-box input[type="text"],
.input-box [data-testid="textbox"] textarea{
  border:1.5px solid #bae6fd!important;border-radius:12px!important;
  padding:14px 18px!important;font-size:15px!important;
  background:rgba(240,249,255,.6)!important;
  outline:none!important;box-shadow:none!important;transition:all .2s}
.input-box textarea:focus, .input-box input[type="text"]:focus{
  border-color:#0891b2!important;background:#fff!important;
  box-shadow:0 0 0 4px rgba(8,145,178,.12)!important}

/* ====== 按钮：渐变 + 阴影 ====== */
.send-btn, button.primary, button[class*="primary"]{
  background:linear-gradient(135deg,#0e7490 0%,#0891b2 100%)!important;
  color:#fff!important;font-weight:600!important;
  padding:13px 32px!important;border-radius:12px!important;border:none!important;
  box-shadow:0 4px 12px rgba(8,145,178,.25)!important;transition:all .2s!important}
.send-btn:hover, button.primary:hover, button[class*="primary"]:hover{
  background:linear-gradient(135deg,#0891b2 0%,#06b6d4 100%)!important;
  box-shadow:0 6px 18px rgba(8,145,178,.35)!important;transform:translateY(-1px)}
.send-btn:active, button.primary:active{transform:translateY(0)}

/* ====== 示例按钮 ====== */
.gr-examples label{color:#0e7490!important;font-size:12px!important;font-weight:600!important}
button[class*="example"], .gr-examples button{
  background:rgba(236,254,255,.8)!important;border:1.5px solid #a5f3fc!important;
  color:#155e75!important;border-radius:10px!important;transition:all .15s}
button[class*="example"]:hover, .gr-examples button:hover{
  background:#cffafe!important;border-color:#0891b2!important;
  box-shadow:0 2px 8px rgba(8,145,178,.15)!important}

/* ====== 滚动条 ====== */
#chatbot::-webkit-scrollbar{width:6px}
#chatbot::-webkit-scrollbar-track{background:transparent}
#chatbot::-webkit-scrollbar-thumb{background:#bae6fd;border-radius:3px}
#chatbot::-webkit-scrollbar-thumb:hover{background:#7dd3fc}

/* ====== 来源引用 ====== */
.source-cite{font-size:11px;color:#67e8f9;margin-top:10px;padding-top:8px;
  border-top:1px solid rgba(8,145,178,.1)}
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

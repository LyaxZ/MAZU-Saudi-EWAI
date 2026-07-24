"""MAZU 沙特多灾种预警智能体 — Web 界面"""
import sys, os, re, html as _html, logging, uuid
from datetime import datetime

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("MAZU")

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
KG_HTML_PATH = os.path.join(OUTPUTS_DIR, "knowledge_graph.html")
EVENTS_HTML_PATH = os.path.join(OUTPUTS_DIR, "kg_events.html")

def _load_html(path, name):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        log.info(f"已加载 {name}: {len(content)/1024:.0f} KB")
        return content
    log.warning(f"{name} 不存在: {path}")
    return f"<h3 style='text-align:center;padding:40px'>文件不存在，请先运行 python run.py tools all</h3>"

KG_HTML = _load_html(KG_HTML_PATH, "knowledge_graph.html")
EVENTS_HTML = _load_html(EVENTS_HTML_PATH, "kg_events.html")

THEME = gr.themes.Ocean(
    primary_hue="cyan",
    secondary_hue="sky",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="*neutral_100",
)

HEADER = """<div style="text-align:center;padding:12px 0 0">
<h1 style="font-size:1.35rem;font-weight:800;color:#0c4a6e;margin:0">MAZU 沙特多灾种预警智能体</h1>
<p style="font-size:.8rem;color:#64748b;margin:2px 0 0">
暴雨山洪 &middot; 极端高温 &middot; 沙尘强风 &middot; 沿海风浪 &ensp;|&ensp; LightGBM &middot; 知识图谱 &middot; LLM Agent</p>
</div>"""

WELCOME_MSG = "你好，我是 MAZU 沙特多灾种预警智能体。\n\n可以查询四类灾害风险：暴雨山洪、极端高温、沙尘强风、沿海风浪。也可以询问处置建议和影响分析。\n\n请直接输入你的问题。"

CHAT_EXAMPLES = [
    "8月28日阿西尔地区有山洪风险吗", "明天利雅得会有热浪吗",
    "5月中旬沙特有沙尘暴吗", "红海沿岸有没有风浪预警",
    "如果吉达有山洪红色预警，应该采取什么措施",
]


def build_ui():
    from llm_agent.agent import MazuAgent

    with gr.Blocks(title="MAZU 沙特多灾种预警智能体", theme=THEME) as app:
        gr.HTML(HEADER)

        with gr.Tabs(selected=0):

            # ═══════════════════════════════════════════════════
            # Tab 1: 智能对话
            # ═══════════════════════════════════════════════════
            with gr.Tab("智能对话", id=0):
                gr.HTML("""<style>
#chat-main-row { align-items: stretch !important; }
#sidebar-col { height: auto !important; }
#sidebar-col > .wrap { height: 100% !important; display: flex !important; flex-direction: column !important; }
#sidebar-col > .wrap > :first-child { flex: 1 !important; min-height: 0 !important; }
#send-btn { padding: 0 18px !important; font-size: 14px !important; }
</style>""")
                agent_ref = [None]; agent_err = [""]

                def _get_agent():
                    if agent_ref[0] is None:
                        try:
                            agent_ref[0] = MazuAgent(verbose=False)
                            log.info("MazuAgent 就绪")
                        except Exception as e:
                            agent_err[0] = str(e); log.error(f"Agent 失败: {e}")
                    return agent_ref[0], agent_err[0]

                conv_state = gr.State({
                    "convs": [{"id": "0", "title": "新对话", "time": "", "msgs": []}],
                    "active": "0",
                })

                with gr.Row(equal_height=True, elem_id="chat-main-row"):
                    # 侧边栏
                    with gr.Column(scale=1, min_width=220, elem_id="sidebar-col"):
                        conv_list = gr.HTML(
                            '<div style="border:1px solid #e2e8f0;border-radius:12px;padding:8px;'
                            'display:flex;flex-direction:column;height:100%;box-sizing:border-box">'
                            '<div style="font-weight:700;font-size:13px;color:#334155;padding:0 4px 6px;'
                            'border-bottom:1px solid #f1f5f9;margin-bottom:4px;flex-shrink:0">对话记录</div>'
                            '<div style="flex:1;overflow-y:auto;min-height:0">'
                            '<div style="padding:6px 8px;border-radius:8px;font-size:12px;text-align:left;'
                            'background:linear-gradient(135deg,rgba(14,116,144,.1),rgba(8,145,178,.08));'
                            'border:1px solid rgba(8,145,178,.2);font-weight:600;color:#0e7490;margin-bottom:2px">'
                            '新对话<div style="font-size:10px;color:#94a3b8;font-weight:400"></div></div>'
                            '</div></div>')
                        new_chat = gr.Button("+ 新建对话", variant="secondary", size="sm")

                    # 聊天区
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="", height=460, type="messages",
                            value=[{"role": "assistant", "content": WELCOME_MSG}])
                        with gr.Row(equal_height=True):
                            msg = gr.Textbox(placeholder="输入问题，如：明天利雅得会有热浪吗？",
                                scale=10, show_label=False, elem_id="msg-box")
                            send = gr.Button("发送", variant="primary", scale=0, min_width=64, elem_id="send-btn")

                gr.Examples(examples=CHAT_EXAMPLES, inputs=msg, label="试试这些问题")

                def respond(message, history, state):
                    history = history or []
                    a, e = _get_agent()
                    if e:
                        yield history + [{"role":"user","content":message},
                            {"role":"assistant","content":f"初始化失败: {e}"}], "", state
                        return
                    convs = state["convs"]
                    conv = next((c for c in convs if c["id"]==state["active"]), None)
                    if conv and not conv["msgs"]:
                        conv["title"] = message[:24] + ("..." if len(message)>24 else "")
                        conv["time"] = datetime.now().strftime("%H:%M")
                    yield history+[{"role":"user","content":message},
                        {"role":"assistant","content":"..."}], "", state
                    full, last = "", ""
                    for chunk in a.chat_stream(message):
                        full += chunk
                        tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                        if tools and tools[-1]!=last:
                            last=tools[-1]
                            nm={"predict_risk":"分析风险","query_kg_impact":"影响链分析",
                                "search_similar_cases":"检索相似案例"}
                            yield history+[{"role":"user","content":message},
                                {"role":"assistant","content":f"*{nm.get(last,last)}中...*"}], "", state
                    clean=re.sub(r'\n?🔧[^\n]*\n','\n',full)
                    clean=re.sub(r'\n?✅[^\n]*\n','\n',clean)
                    clean=re.sub(r'^\s*---\s*\n+','',clean)
                    main,src=clean,""
                    if "\n---\n" in clean:
                        parts=clean.rsplit("\n---\n",1)
                        main,src=parts[0].strip(),parts[1].strip()
                    if src:
                        main+=f'\n<span style="font-size:.75rem;opacity:.7">{" · ".join(src.split(chr(10)))}</span>'
                    main=re.sub(r'\n{3,}','\n\n',main)
                    new_hist=history+[{"role":"user","content":message},{"role":"assistant","content":main}]
                    conv["msgs"]=new_hist
                    yield new_hist,"",state

                def _render_sidebar(state):
                    cid = state["active"]
                    items = []
                    for c in reversed(state["convs"]):
                        a = "active" if c["id"] == cid else ""
                        t = c["title"] or "新对话"
                        bg = "background:linear-gradient(135deg,rgba(14,116,144,.1),rgba(8,145,178,.08));border:1px solid rgba(8,145,178,.2);font-weight:600;color:#0e7490" if a else "border:1px solid transparent"
                        items.append(
                            f'<div style="padding:6px 8px;border-radius:8px;font-size:12px;text-align:left;'
                            f'color:#475569;margin-bottom:2px;cursor:pointer;{bg}"'
                            f'onclick="document.querySelector(\'#switch-input textarea\').value=\'{c["id"]}\';'
                            f'document.querySelector(\'#switch-btn\').click()">'
                            f'{t}<div style="font-size:10px;color:#94a3b8;font-weight:400">{c.get("time","")}</div></div>')
                    return ('<div style="border:1px solid #e2e8f0;border-radius:12px;padding:8px;'
                            'display:flex;flex-direction:column;height:100%;box-sizing:border-box">'
                            '<div style="font-weight:700;font-size:13px;color:#334155;padding:0 4px 6px;'
                            'border-bottom:1px solid #f1f5f9;margin-bottom:4px;flex-shrink:0">对话记录</div>'
                            '<div style="flex:1;overflow-y:auto;min-height:0">'
                            + "".join(items) +
                            '</div></div>')

                def new_conversation(state):
                    cid = str(uuid.uuid4())[:8]
                    state["convs"].append({"id":cid,"title":"新对话","time":"","msgs":[]})
                    state["active"] = cid
                    return [{"role":"assistant","content":WELCOME_MSG}], "", state, _render_sidebar(state)

                def switch_conversation(cid, state):
                    state["active"] = cid
                    conv = next((c for c in state["convs"] if c["id"]==cid), None)
                    msgs = conv["msgs"] if conv else []
                    if not msgs:
                        msgs = [{"role":"assistant","content":WELCOME_MSG}]
                    return msgs, _render_sidebar(state), state

                switch_input = gr.Textbox(visible=False, elem_id="switch-input")
                switch_btn = gr.Button("switch", visible=False, elem_id="switch-btn")

                send.click(fn=respond, inputs=[msg, chatbot, conv_state],
                          outputs=[chatbot, msg, conv_state])
                msg.submit(fn=respond, inputs=[msg, chatbot, conv_state],
                          outputs=[chatbot, msg, conv_state])
                new_chat.click(fn=new_conversation, inputs=[conv_state],
                              outputs=[chatbot, msg, conv_state, conv_list])
                switch_btn.click(fn=switch_conversation, inputs=[switch_input, conv_state],
                                outputs=[chatbot, conv_list, conv_state])

            # ═══════════════════════════════════════════════════
            # Tab 2: 知识图谱
            # ═══════════════════════════════════════════════════
            with gr.Tab("知识图谱", id=1):
                gr.HTML(f'<iframe src="about:blank" srcdoc="{_html.escape(KG_HTML)}" '
                        f'style="width:100%;height:calc(100vh - 180px);min-height:520px;border:none;border-radius:12px"></iframe>')
                gr.Markdown("*点击灾害节点筛选 · 双击空白重置 · 滚轮缩放*")

            # ═══════════════════════════════════════════════════
            # Tab 3: 历史事件
            # ═══════════════════════════════════════════════════
            with gr.Tab("历史事件", id=2):
                gr.HTML(f'<iframe src="about:blank" srcdoc="{_html.escape(EVENTS_HTML)}" '
                        f'style="width:100%;height:calc(100vh - 160px);min-height:580px;border:none;border-radius:12px"></iframe>')
                gr.Markdown("*10 个 Ground Truth 事件 — 下拉切换 · 拖拽缩放*")

            # ═══════════════════════════════════════════════════
            # Tab 4: 系统
            # ═══════════════════════════════════════════════════
            with gr.Tab("系统", id=3):
                def _sysinfo():
                    import platform
                    md = os.path.join(OUTPUTS_DIR, "models")
                    ok = 0
                    details = []
                    for dt in ["flash_flood","extreme_heat","dust_wind","coastal_wave"]:
                        p = os.path.join(md,f"{dt}.pkl")
                        if os.path.isfile(p):
                            ok += 1
                            details.append(f"  {dt}: {os.path.getsize(p)/1024**2:.1f} MB")
                        else:
                            details.append(f"  {dt}: 未找到")
                    model_txt = f"已加载: {ok}/4\n" + "\n".join(details)

                    key = os.environ.get("LLM_API_KEY","")
                    mk = key[:6]+"..."+key[-4:] if len(key)>10 else "未设置"
                    llm_txt = f"模型: {os.environ.get('LLM_MODEL','未设置')}\n"
                    llm_txt += f"API: {os.environ.get('LLM_BASE_URL','未设置')}\n"
                    llm_txt += f"Key: {mk}"

                    ind = os.environ.get("MAZU_INDICATORS_DIR", os.path.join(os.path.dirname(OUTPUTS_DIR),"indicators"))
                    nc = len([f for f in os.listdir(ind) if f.endswith(".nc")]) if os.path.isdir(ind) else 0
                    data_txt = f"NC 文件: {nc} 个\n目录: {ind}"

                    lf = os.path.join(OUTPUTS_DIR,"app.log")
                    log_txt = ""
                    if os.path.isfile(lf):
                        try:
                            with open(lf,"r",encoding="utf-8") as f:
                                log_txt = "".join(f.readlines()[-40:])
                        except: pass
                    pyinfo = f"Python {platform.python_version()} | {platform.platform()[:60]}"
                    return model_txt, llm_txt, data_txt, log_txt, pyinfo

                # 三个信息框 + 日志
                with gr.Row():
                    model_box = gr.Textbox(label="模型状态", value="", lines=6, interactive=False, scale=1)
                    llm_box = gr.Textbox(label="LLM 配置", value="", lines=6, interactive=False, scale=1)
                    data_box = gr.Textbox(label="数据", value="", lines=6, interactive=False, scale=1)
                log_box = gr.Textbox(label="运行日志", value="", lines=10, interactive=False)
                # 刷新按钮在底部
                with gr.Row():
                    refresh = gr.Button("刷新", variant="primary")

                def refresh_all():
                    mt, lt, dt, logs, pyinfo = _sysinfo()
                    return mt, lt, dt, f"{pyinfo}\n{'='*60}\n{logs}"

                refresh.click(fn=refresh_all, outputs=[model_box, llm_box, data_box, log_box])
                # 初始加载
                mt, lt, dt, logs, pyinfo = _sysinfo()
                model_box.value = mt
                llm_box.value = lt
                data_box.value = dt
                log_box.value = f"{pyinfo}\n{'='*60}\n{logs}"

    return app


def launch_app(share=False, **kw):
    kw.setdefault("server_name", "0.0.0.0")
    kw.setdefault("server_port", 7860)
    auth_user = os.environ.get("MAZU_AUTH_USER", "")
    auth_pass = os.environ.get("MAZU_AUTH_PASS", "")
    if auth_user and auth_pass:
        kw["auth"] = (auth_user, auth_pass)
    log.info(f"MAZU Web -> http://{kw['server_name']}:{kw['server_port']}")
    app = build_ui()
    app.queue(default_concurrency_limit=int(os.environ.get("MAZU_CONCURRENCY", "5")),
              max_size=20, api_open=False)
    app.launch(share=share, **kw)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--concurrency", type=int, default=None)
    args = p.parse_args()
    if args.concurrency is not None:
        os.environ["MAZU_CONCURRENCY"] = str(args.concurrency)
    launch_app(share=args.share, server_port=args.port)


if __name__ == "__main__":
    launch_app(share=False, server_port=7866)

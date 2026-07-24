"""MAZU 沙特多灾种预警智能体 — Web 界面 (三语版: 中文/English/العربية)"""
import sys, os, re, html as _html, logging, uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from config.i18n import T, CHAT_EXAMPLES, TOOL_STATUS, KG_HTML, EVENTS_HTML, build_header

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("MAZU")

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

THEME = gr.themes.Ocean(
    primary_hue="cyan", secondary_hue="sky", neutral_hue="slate",
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_secondary_background_fill="white",
    button_secondary_background_fill_hover="*neutral_100",
)

LANG_SWITCH_JS = """
<script>
function mazuSwitchLang(lang) {
    var tabMap = {
        zh: ['智能对话','知识图谱','历史事件','系统'],
        en: ['Smart Chat','Knowledge Graph','Historical Events','System'],
        ar: ['المحادثة الذكية','الرسم البياني المعرفي','الأحداث التاريخية','النظام']
    };
    var labels = tabMap[lang];
    if (!labels) return;
    var container = document.getElementById('mazu-tabs');
    if (!container) return;
    // 更新 tab 按钮
    var buttons = container.querySelectorAll('button');
    var idx = 0;
    buttons.forEach(function(btn) {
        if (idx < 4 && btn.textContent.trim().length < 40) {
            btn.textContent = labels[idx]; idx++;
        }
    });
    // 更新 tablist 标签
    var tabs = container.querySelectorAll('[role="tab"]');
    idx = 0;
    tabs.forEach(function(tab) {
        if (idx < 4) { tab.textContent = labels[idx]; idx++; }
    });
    // 侧边栏标题
    var sh = document.getElementById('sidebar-header');
    if (sh) sh.textContent = {zh:'对话记录', en:'Conversations', ar:'المحادثات'}[lang] || '对话记录';
    // 侧边栏默认标题
    setTimeout(function() {
        var ncMap = {zh:'新对话', en:'New Chat', ar:'محادثة جديدة'};
        var ncText = ncMap[lang] || '新对话';
        var walker = document.createTreeWalker(document.getElementById('sidebar-col') || document.body, NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {
            var n = walker.currentNode;
            if (n.textContent.trim() === '新对话' || n.textContent.trim() === 'New Chat' || n.textContent.trim() === 'محادثة جديدة') {
                n.textContent = ncText;
            }
        }
    }, 200);
    // RTL
    var isRtl = (lang === 'ar');
    document.documentElement.dir = isRtl ? 'rtl' : 'ltr';
    document.body.classList.toggle('rtl-mode', isRtl);
    // 下拉三角位置
    var dd = document.getElementById('lang-dd');
    if (dd) {
        var arrow = dd.querySelector('svg, img, [class*=\"arrow\"], [class*=\"icon\"]');
        if (arrow) { arrow.style.left = isRtl ? '6px' : ''; arrow.style.right = isRtl ? '' : '6px'; }
    }
    // 浏览器标签标题
    var titles = {zh:'MAZU 沙特多灾种预警智能体', en:'MAZU Saudi Multi-Hazard Early Warning Agent', ar:'MAZU وكيل الإنذار المبكر للكوارث'};
    document.title = titles[lang] || titles['en'];
    // Gradio 底部栏 + 设置弹窗
    setTimeout(function() {
        var txtMap = {
            zh: {'通过 API 使用':'通过 API 使用','使用 Gradio 构建':'使用 Gradio 构建','设置':'设置','Settings':'设置','Built with Gradio':'使用 Gradio 构建','Use via API':'通过 API 使用'},
            en: {'通过 API 使用':'Use via API','使用 Gradio 构建':'Built with Gradio','设置':'Settings','Settings':'Settings','Built with Gradio':'Built with Gradio','Use via API':'Use via API'},
            ar: {'通过 API 使用':'استخدام عبر API','使用 Gradio 构建':'مبني بـ Gradio','设置':'إعدادات','Settings':'إعدادات','Built with Gradio':'مبني بـ Gradio','Use via API':'استخدام عبر API'}
        };
        var map = txtMap[lang] || txtMap['zh'];
        document.querySelectorAll('a, button, span, h2, h3, label').forEach(function(el) {
            var t = el.textContent.trim();
            if (map[t] && map[t] !== t) el.textContent = map[t];
        });
    }, 300);
}
</script>
"""


def _build_examples_html(lang):
    """生成可点击的示例问题 HTML"""
    items = []
    for ex in CHAT_EXAMPLES[lang]:
        escaped = _html.escape(ex)
        items.append(
            f'<span onclick="var t=document.querySelector(\'#msg-box textarea\');'
            f't.value=this.textContent;t.dispatchEvent(new Event(\'input\',{{bubbles:true}}))"'
            f'style="display:inline-block;padding:6px 14px;margin:3px 6px 3px 0;'
            f'background:#f1f5f9;border:1px solid #e2e8f0;border-radius:16px;'
            f'font-size:13px;color:#475569;cursor:pointer;transition:all .15s"'
            f'onmouseover="this.style.background=\'#e2e8f0\'"'
            f'onmouseout="this.style.background=\'#f1f5f9\'"'
            f'>{escaped}</span>'
        )
    return f'<div style="line-height:2">{"".join(items)}</div>'


def build_ui():
    from llm_agent.agent import MazuAgent

    with gr.Blocks(title="MAZU 沙特多灾种预警智能体", theme=THEME, head=LANG_SWITCH_JS) as app:
        lang_state = gr.State("zh")

        # 头部 + 语言选择器
        with gr.Row():
            header_html = gr.HTML(build_header("zh"))
            lang_dd = gr.Dropdown(
                choices=[("中文", "zh"), ("English", "en"), ("العربية", "ar")],
                value="zh", show_label=False, interactive=True, elem_id="lang-dd",
                scale=0, min_width=145,
            )

        with gr.Tabs(selected=0, elem_id="mazu-tabs") as tabs_ref:
            # ═══════════════════════════ Tab 0: 智能对话 ═══════════════════════════
            with gr.Tab(T["zh"]["tab_chat"], id=0, elem_id="tab-chat"):
                gr.HTML("""<style>
#chat-main-row { align-items: stretch !important; }
#sidebar-col { height: auto !important; }
#sidebar-col > .wrap { height: 100% !important; display: flex !important; flex-direction: column !important; }
#sidebar-col > .wrap > :first-child { flex: 1 !important; min-height: 0 !important; }
#send-btn { padding: 0 18px !important; font-size: 14px !important; }
#lang-dd input { caret-color: transparent !important; cursor: pointer !important; }
#lang-dd { margin-top: 6px !important; }
#lang-dd svg { right: 6px !important; }
.rtl-mode, .rtl-mode .block, .rtl-mode .prose, .rtl-mode .wrap,
.rtl-mode textarea, .rtl-mode input, .rtl-mode p, .rtl-mode span,
.rtl-mode h1, .rtl-mode h2, .rtl-mode h3, .rtl-mode label,
.rtl-mode button, .rtl-mode .chatbot, .rtl-mode [class*=\"message\"],
.rtl-mode .svelte-vt1mxs, .rtl-mode .md, .rtl-mode .markdown {
    direction: rtl !important; text-align: right !important;
}
.rtl-mode #lang-dd svg { right: auto !important; left: 6px !important; }
</style>""")
                agent_ref = [None]; agent_err = [""]

                def _get_agent():
                    if agent_ref[0] is None:
                        try:
                            agent_ref[0] = MazuAgent(verbose=False)
                            log.info("MazuAgent ready")
                        except Exception as e:
                            agent_err[0] = str(e); log.error(f"Agent failed: {e}")
                    return agent_ref[0], agent_err[0]

                conv_state = gr.State({
                    "convs": [{"id": "0", "title": T["zh"]["new_chat_default"], "time": "", "msgs": []}],
                    "active": "0",
                })

                with gr.Row(equal_height=True, elem_id="chat-main-row"):
                    with gr.Column(scale=1, min_width=220, elem_id="sidebar-col"):
                        conv_list = gr.HTML(
                            '<div style="border:1px solid #e2e8f0;border-radius:12px;padding:8px;'
                            'display:flex;flex-direction:column;height:100%;box-sizing:border-box">'
                            '<div id="sidebar-header" style="font-weight:700;font-size:13px;color:#334155;'
                            'padding:0 4px 6px;border-bottom:1px solid #f1f5f9;margin-bottom:4px;flex-shrink:0">'
                            f'{T["zh"]["sidebar_title"]}</div>'
                            '<div style="flex:1;overflow-y:auto;min-height:0">'
                            '<div style="padding:6px 8px;border-radius:8px;font-size:12px;text-align:left;'
                            'background:linear-gradient(135deg,rgba(14,116,144,.1),rgba(8,145,178,.08));'
                            'border:1px solid rgba(8,145,178,.2);font-weight:600;color:#0e7490;margin-bottom:2px">'
                            f'{T["zh"]["new_chat_default"]}'
                            '<div style="font-size:10px;color:#94a3b8;font-weight:400"></div></div>'
                            '</div></div>')
                        new_chat_btn = gr.Button(T["zh"]["new_chat_btn"], variant="secondary", size="sm")

                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="", height=460, type="messages",
                            value=[{"role": "assistant", "content": T["zh"]["welcome"]}])
                        with gr.Row(equal_height=True):
                            msg = gr.Textbox(placeholder=T["zh"]["placeholder"],
                                scale=10, show_label=False, elem_id="msg-box")
                            send_btn = gr.Button(T["zh"]["send"], variant="primary",
                                scale=0, min_width=64, elem_id="send-btn")

                examples_label = gr.Markdown(f"**{T['zh']['examples_label']}**", elem_id="examples-label")
                examples_html = gr.HTML(_build_examples_html("zh"), elem_id="examples-html")

                def respond(message, history, state, lang):
                    history = history or []
                    a, e = _get_agent()
                    if e:
                        yield history + [{"role":"user","content":message},
                            {"role":"assistant","content":f"{T[lang]['init_error']}: {e}"}], "", state
                        return
                    convs = state["convs"]
                    conv = next((c for c in convs if c["id"] == state["active"]), None)
                    if conv and not conv["msgs"]:
                        conv["title"] = message[:24] + ("..." if len(message) > 24 else "")
                        conv["time"] = datetime.now().strftime("%H:%M")
                    yield history + [{"role":"user","content":message},
                        {"role":"assistant","content":"..."}], "", state
                    full, last = "", ""
                    for chunk in a.chat_stream(message):
                        full += chunk
                        tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                        if tools and tools[-1] != last:
                            last = tools[-1]
                            nm = TOOL_STATUS[lang]
                            yield history + [{"role":"user","content":message},
                                {"role":"assistant","content":f"*{nm.get(last, last)}*"}], "", state
                    clean = re.sub(r'\n?🔧[^\n]*\n', '\n', full)
                    clean = re.sub(r'\n?✅[^\n]*\n', '\n', clean)
                    clean = re.sub(r'^\s*---\s*\n+', '', clean)
                    main, src = clean, ""
                    if "\n---\n" in clean:
                        parts = clean.rsplit("\n---\n", 1)
                        main, src = parts[0].strip(), parts[1].strip()
                    if src:
                        main += f'\n<span style="font-size:.75rem;opacity:.7">{" · ".join(src.split(chr(10)))}</span>'
                    main = re.sub(r'\n{3,}', '\n\n', main)
                    new_hist = history + [{"role":"user","content":message}, {"role":"assistant","content":main}]
                    conv["msgs"] = new_hist
                    yield new_hist, "", state

                def _render_sidebar(state, lang="zh"):
                    cid = state["active"]
                    items = []
                    for c in reversed(state["convs"]):
                        a_cls = "active" if c["id"] == cid else ""
                        t = c["title"] or T[lang]["new_chat_default"]
                        bg = ("background:linear-gradient(135deg,rgba(14,116,144,.1),rgba(8,145,178,.08));"
                              "border:1px solid rgba(8,145,178,.2);font-weight:600;color:#0e7490") \
                            if a_cls else "border:1px solid transparent"
                        items.append(
                            f'<div style="padding:6px 8px;border-radius:8px;font-size:12px;text-align:left;'
                            f'color:#475569;margin-bottom:2px;cursor:pointer;{bg}"'
                            f'onclick="document.querySelector(\'#switch-input textarea\').value=\'{c["id"]}\';'
                            f'document.querySelector(\'#switch-btn\').click()">'
                            f'{t}<div style="font-size:10px;color:#94a3b8;font-weight:400">{c.get("time","")}</div></div>')
                    return (
                        '<div style="border:1px solid #e2e8f0;border-radius:12px;padding:8px;'
                        'display:flex;flex-direction:column;height:100%;box-sizing:border-box">'
                        '<div id="sidebar-header" style="font-weight:700;font-size:13px;color:#334155;'
                        'padding:0 4px 6px;border-bottom:1px solid #f1f5f9;margin-bottom:4px;flex-shrink:0">'
                        f'{T["zh"]["sidebar_title"]}</div>'
                        '<div style="flex:1;overflow-y:auto;min-height:0">'
                        + "".join(items) +
                        '</div></div>')

                def new_conversation(state, lang):
                    cid = str(uuid.uuid4())[:8]
                    state["convs"].append({"id": cid, "title": T[lang]["new_chat_default"], "time": "", "msgs": []})
                    state["active"] = cid
                    return [{"role":"assistant","content":T[lang]["welcome"]}], "", state, _render_sidebar(state, lang)

                def switch_conversation(cid, state):
                    state["active"] = cid
                    conv = next((c for c in state["convs"] if c["id"] == cid), None)
                    msgs = conv["msgs"] if conv else []
                    if not msgs:
                        msgs = [{"role":"assistant","content":T["zh"]["welcome"]}]
                    return msgs, _render_sidebar(state, "zh"), state

                switch_input = gr.Textbox(visible=False, elem_id="switch-input")
                switch_btn = gr.Button("switch", visible=False, elem_id="switch-btn")

                send_btn.click(fn=respond, inputs=[msg, chatbot, conv_state, lang_state],
                              outputs=[chatbot, msg, conv_state])
                msg.submit(fn=respond, inputs=[msg, chatbot, conv_state, lang_state],
                          outputs=[chatbot, msg, conv_state])
                new_chat_btn.click(fn=new_conversation, inputs=[conv_state, lang_state],
                                  outputs=[chatbot, msg, conv_state, conv_list])
                switch_btn.click(fn=switch_conversation, inputs=[switch_input, conv_state],
                                outputs=[chatbot, conv_list, conv_state])

            # ═══════════════════════════ Tab 1: 知识图谱 ═══════════════════════════
            with gr.Tab(T["zh"]["tab_kg"], id=1, elem_id="tab-kg"):
                kg_iframe = gr.HTML(
                    f'<iframe src="about:blank" srcdoc="{_html.escape(KG_HTML["zh"])}" '
                    f'style="width:100%;height:calc(100vh - 180px);min-height:520px;border:none;border-radius:12px"></iframe>')
                kg_hint = gr.Markdown(T["zh"]["kg_hint"])

            # ═══════════════════════════ Tab 2: 历史事件 ═══════════════════════════
            with gr.Tab(T["zh"]["tab_events"], id=2, elem_id="tab-events"):
                ev_iframe = gr.HTML(
                    f'<iframe src="about:blank" srcdoc="{_html.escape(EVENTS_HTML["zh"])}" '
                    f'style="width:100%;height:calc(100vh - 160px);min-height:580px;border:none;border-radius:12px"></iframe>')
                ev_hint = gr.Markdown(T["zh"]["events_hint"])

            # ═══════════════════════════ Tab 3: 系统 ═══════════════════════════
            with gr.Tab(T["zh"]["tab_system"], id=3, elem_id="tab-system"):
                def _sysinfo(lang="zh"):
                    import platform
                    md = os.path.join(OUTPUTS_DIR, "models")
                    ok = 0; details = []
                    for dt in ["flash_flood","extreme_heat","dust_wind","coastal_wave"]:
                        p = os.path.join(md, f"{dt}.pkl")
                        if os.path.isfile(p):
                            ok += 1
                            details.append(f"  {dt}: {os.path.getsize(p)/1024**2:.1f} MB")
                        else:
                            details.append(f"  {dt}: {T[lang]['model_not_found']}")
                    model_txt = f"{T[lang]['model_loaded']}: {ok}/4\n" + "\n".join(details)

                    key = os.environ.get("LLM_API_KEY", "")
                    mk = key[:6] + "..." + key[-4:] if len(key) > 10 else T[lang]["not_set"]
                    llm_txt = (f"{T[lang]['llm_config']}: {os.environ.get('LLM_MODEL', T[lang]['not_set'])}\n"
                               f"API: {os.environ.get('LLM_BASE_URL', T[lang]['not_set'])}\n"
                               f"Key: {mk}")

                    ind = os.environ.get("MAZU_INDICATORS_DIR",
                                         os.path.join(os.path.dirname(OUTPUTS_DIR), "indicators"))
                    nc = len([f for f in os.listdir(ind) if f.endswith(".nc")]) if os.path.isdir(ind) else 0
                    data_txt = f"{T[lang]['nc_files']}: {nc}\n{T[lang]['dir']}: {ind}"

                    lf = os.path.join(OUTPUTS_DIR, "app.log")
                    log_txt = ""
                    if os.path.isfile(lf):
                        try:
                            with open(lf, "r", encoding="utf-8") as f:
                                log_txt = "".join(f.readlines()[-40:])
                        except: pass
                    pyinfo = f"Python {platform.python_version()} | {platform.platform()[:60]}"
                    return model_txt, llm_txt, data_txt, log_txt, pyinfo

                with gr.Row():
                    model_box = gr.Textbox(label=T["zh"]["model_status"], value="", lines=6, interactive=False, scale=1)
                    llm_box = gr.Textbox(label=T["zh"]["llm_config"], value="", lines=6, interactive=False, scale=1)
                    data_box = gr.Textbox(label=T["zh"]["data_label"], value="", lines=6, interactive=False, scale=1)
                log_box = gr.Textbox(label=T["zh"]["log_label"], value="", lines=10, interactive=False)
                with gr.Row():
                    refresh_btn = gr.Button(T["zh"]["refresh"], variant="primary")

                def refresh_all(lang="zh"):
                    mt, lt, dt, logs, pyinfo = _sysinfo(lang)
                    return mt, lt, dt, f"{pyinfo}\n{'='*60}\n{logs}"

                refresh_btn.click(fn=refresh_all, inputs=[lang_state],
                                 outputs=[model_box, llm_box, data_box, log_box])
                mt, lt, dt, logs, pyinfo = _sysinfo("zh")
                model_box.value = mt; llm_box.value = lt; data_box.value = dt
                log_box.value = f"{pyinfo}\n{'='*60}\n{logs}"

        # ═══════════════════════════ 语言切换 ═══════════════════════════
        def on_lang_change(lang, state):
            mt, lt, dt, logs, pyinfo = _sysinfo(lang)
            return [
                gr.update(value=build_header(lang)),                          # header_html
                gr.update(value=[{"role":"assistant","content":T[lang]["welcome"]}]),  # chatbot
                gr.update(placeholder=T[lang]["placeholder"]),                 # msg
                gr.update(value=T[lang]["send"]),                             # send_btn
                gr.update(value=T[lang]["new_chat_btn"]),                     # new_chat_btn
                gr.update(value=f"**{T[lang]['examples_label']}**"),          # examples_label
                gr.update(value=_build_examples_html(lang)),                  # examples_html
                gr.update(value=f'<iframe src="about:blank" srcdoc="{_html.escape(KG_HTML[lang])}" '
                          f'style="width:100%;height:calc(100vh - 180px);min-height:520px;border:none;border-radius:12px"></iframe>'),  # kg_iframe
                gr.update(value=T[lang]["kg_hint"]),                          # kg_hint
                gr.update(value=f'<iframe src="about:blank" srcdoc="{_html.escape(EVENTS_HTML[lang])}" '
                          f'style="width:100%;height:calc(100vh - 160px);min-height:580px;border:none;border-radius:12px"></iframe>'),  # ev_iframe
                gr.update(value=T[lang]["events_hint"]),                      # ev_hint
                gr.update(label=T[lang]["model_status"], value=mt),           # model_box
                gr.update(label=T[lang]["llm_config"], value=lt),             # llm_box
                gr.update(label=T[lang]["data_label"], value=dt),             # data_box
                gr.update(label=T[lang]["log_label"], value=f"{pyinfo}\n{'='*60}\n{logs}"),  # log_box
                gr.update(value=T[lang]["refresh"]),                          # refresh_btn
                lang,                                                          # lang_state
            ]

        lang_dd.change(
            fn=on_lang_change, inputs=[lang_dd, conv_state],
            outputs=[
                header_html, chatbot, msg, send_btn, new_chat_btn,
                examples_label, examples_html,
                kg_iframe, kg_hint, ev_iframe, ev_hint,
                model_box, llm_box, data_box, log_box, refresh_btn,
                lang_state,
            ],
            js="function(lang) { if (typeof mazuSwitchLang === 'function') mazuSwitchLang(lang); return [lang]; }"
        )

    return app


def launch_app(share=False, **kw):
    kw.setdefault("server_name", "0.0.0.0")
    kw.setdefault("server_port", 7866)
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
    p.add_argument("--port", type=int, default=7866)
    p.add_argument("--concurrency", type=int, default=None)
    args = p.parse_args()
    if args.concurrency is not None:
        os.environ["MAZU_CONCURRENCY"] = str(args.concurrency)
    launch_app(share=args.share, server_port=args.port)

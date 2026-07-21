"""MAZU 沙特多灾种预警 — Web界面（Radio切换页面）"""

import sys, os, re, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gradio as gr

from llm_agent.agent import MazuAgent
from models.inference import DisasterInference
from config.model_config import DISASTER_FEATURES
from config.infrastructure import CITIES, AIRPORTS, HIGHWAYS, WADIS

# === 日志 ===
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("MAZU")

# === CSS ===
CSS = """
body,.gradio-container{background:#f7f8fa!important}footer{display:none!important}
.main-header{text-align:center;padding:20px 0 10px 0}
.main-header h1{font-size:28px;font-weight:700;color:#1a1a2e;margin:0}
.main-header p{color:#8e8ea0;font-size:14px;margin:4px 0 0 0}
#chatbot{border-radius:12px;max-width:100%}
#chatbot>div{max-width:100%!important}
#page-nav label{padding:8px 20px!important;border-radius:8px!important;margin:0 4px!important;font-size:15px!important}
#page-nav .selected{background:#4f46e5!important;color:#fff!important}
.input-row{background:#fff;border-radius:16px;border:1px solid #e0e0e0;padding:8px 12px}
.input-row textarea{border:none!important;box-shadow:none!important}
.send-btn{background:#4f46e5!important;border:none!important;color:#fff!important;font-weight:600!important}
.send-btn:hover{background:#4338ca!important}
.source-cite{font-size:11px;color:#9ca3af;margin-top:8px;padding-top:6px;border-top:1px solid #f0f0f0}
"""

# === 数据 ===
DISASTER_META = {
    "flash_flood": ("暴雨山洪", [[0,"#fff5f0"],[.25,"#fc9272"],[.5,"#ef3b2d"],[.75,"#a50f15"]]),
    "extreme_heat": ("极端高温", [[0,"#fff7ec"],[.25,"#fdbb84"],[.5,"#ef6548"],[.75,"#7f0000"]]),
    "dust_wind": ("沙尘强风", [[0,"#ffffd4"],[.25,"#febd5a"],[.5,"#f77f00"],[.75,"#662506"]]),
    "coastal_wave": ("沿海风浪", [[0,"#f7fbff"],[.25,"#9ecae1"],[.5,"#3182bd"],[.75,"#08519c"]]),
}

_engine = None
def get_engine():
    global _engine
    if _engine is None: _engine = DisasterInference()
    return _engine

# === 地图 ===
def plot_risk_map(disaster_type, date_str):
    import plotly.graph_objects as go
    from data.loader import load_to_dataframe
    engine = get_engine()
    name, colorscale = DISASTER_META[disaster_type]
    feats = DISASTER_FEATURES[disaster_type]
    load_vars = [f for f in feats if f not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")]

    try:
        log.info(f"地图: {date_str} {name}")
        df = load_to_dataframe(date_str, date_str, variables=load_vars, show_progress=False).fillna(0)
        result = engine.predict(df, disaster_type)
    except Exception as e:
        log.error(f"地图失败: {e}")
        return go.Figure(), f"加载失败: {e}"

    proba, lat_arr, lon_arr = result["proba"], result["lat"], result["lon"]
    n_lat, n_lon = 160, 220
    grid = np.full((n_lat, n_lon), np.nan)
    for p, la, lo in zip(proba, lat_arr, lon_arr):
        grid[int(round((la - 16) / 0.1)), int(round((lo - 34) / 0.1))] = p

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=grid, x=np.linspace(34,56,n_lon), y=np.linspace(16,32,n_lat),
        colorscale=colorscale, zmin=0, zmax=1, colorbar_title="风险概率",
        hovertemplate="经度:%{x:.1f}°E<br>纬度:%{y:.1f}°N<br>风险:%{z:.3f}<extra></extra>"))

    c_lats, c_lons, c_names = [], [], []
    for c in CITIES:
        if 16 <= c["lat"] <= 32 and 34 <= c["lon"] <= 56:
            c_lats.append(c["lat"]); c_lons.append(c["lon"])
            c_names.append(c["name"].split("(")[0].strip())
    fig.add_trace(go.Scatter(x=c_lons, y=c_lats, mode="markers+text", text=c_names,
        textposition="top center", textfont=dict(size=8, color="#333"),
        marker=dict(size=5, color="black"), name="城市"))

    for h in HIGHWAYS:
        fig.add_trace(go.Scatter(x=[p[1] for p in h["path"]], y=[p[0] for p in h["path"]],
            mode="lines", line=dict(color="gray",width=.8,dash="dash"),
            opacity=.4, showlegend=False, hovertext=h["name"], hoverinfo="text"))

    w_lats, w_lons, w_names = [], [], []
    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            w_lats.append(w["lat"]); w_lons.append(w["lon"])
            w_names.append(f"{w['name']}({w['length_km']}km)")
    fig.add_trace(go.Scatter(x=w_lons, y=w_lats, mode="markers",
        marker=dict(symbol="diamond",size=8,color="#2ecc71",opacity=.6),
        name="Wadi", hovertext=w_names, hoverinfo="text"))

    n_high, pct = result["n_high"], result["high_pct"]
    fig.update_layout(
        title=dict(text=f"{date_str} {name} | 高风险:{n_high:,}({pct}%) | 均值:{result['mean_risk']:.3f}",
                   font=dict(size=14)),
        xaxis=dict(title="经度", range=[34,56]), yaxis=dict(title="纬度", range=[16,32]),
        height=600, margin=dict(l=50,r=30,t=50,b=50), hovermode="closest")
    return fig, f"{date_str} {name}: 高风险 {n_high:,} 格点 ({pct}%)"

# === 知识图谱 ===
def plot_kg():
    from pyvis.network import Network
    log.info("生成知识图谱...")
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333")
    net.repulsion(node_distance=150, spring_length=200)

    def xy(lon, lat): return int((lon-34)*50), int((32-lat)*50)

    for c in CITIES:
        x, y = xy(c["lon"], c["lat"])
        net.add_node(c["name"].split("(")[0].strip(),
            label=c["name"].split("(")[0].strip(),
            title=f"{c['name']}<br>人口:{c['pop']:,}<br>类型:{c['type']}",
            color="#e74c3c", size=np.clip(c["pop"]/50000,8,40)*1.5, x=x, y=y, physics=False)

    for a in AIRPORTS:
        x, y = xy(a["lon"], a["lat"])
        name = a["name"].split("(")[-1].replace(")","").strip()
        net.add_node(f"✈{name}", label=name,
            title=f"{a['name']}<br>城市:{a.get('city','')}",
            color="#3498db", size=15, shape="triangle", x=x, y=y, physics=False)

    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            x, y = xy(w["lon"], w["lat"])
            net.add_node(w["name"], label=w["name"],
                title=f"{w['name']}<br>长度:{w['length_km']}km<br>影响:{w.get('risk_city','')}",
                color="#2ecc71", size=w["length_km"]/8, shape="diamond", x=x, y=y, physics=False)

    for h in HIGHWAYS:
        path = h["path"]
        for i in range(len(path)-1):
            n1, n2 = f"h_{i}", f"h_{i+1}"
            x1, y1 = xy(path[i][1], path[i][0])
            x2, y2 = xy(path[i+1][1], path[i+1][0])
            net.add_node(n1, hidden=True, x=x1, y=y1, size=1, physics=False)
            net.add_node(n2, hidden=True, x=x2, y=y2, size=1, physics=False)
            net.add_edge(n1, n2, color="#bdc3c7", width=1)

    html = net.generate_html()
    log.info("图谱完成")
    return f'<iframe srcdoc="{html.replace(chr(34), "&quot;")}" width="100%" height="600" frameborder="0"></iframe>'

# === UI ===
def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:
        gr.HTML("""<div class="main-header">
            <h1>🏔 MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 | LightGBM + 知识图谱 + LLM</p>
        </div>""")

        page = gr.Radio(["💬 智能对话", "🗺️ 灾害地图", "🕸️ 知识图谱"],
                        value="💬 智能对话", label="", elem_id="page-nav")

        with gr.Row(visible=True) as chat_page:
            with gr.Column():
                chatbot = gr.Chatbot(label="", height=440, elem_id="chatbot", show_label=False)
                with gr.Row(elem_classes=["input-row"]):
                    msg = gr.Textbox(placeholder="输入问题...", scale=10, container=False, show_label=False, max_lines=4)
                    send = gr.Button("发送", variant="primary", scale=1, elem_classes=["send-btn"])

                _agent = [None]
                def get_agent():
                    if _agent[0] is None: _agent[0] = MazuAgent(verbose=False)
                    return _agent[0]

                def respond(message, history):
                    history = history or []
                    yield history+[{"role":"user","content":message},{"role":"assistant","content":"⏳ 正在分析..."}]
                    full, last = "", ""
                    for chunk in get_agent().chat_stream(message):
                        full += chunk
                        tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                        if tools and tools[-1]!=last:
                            last=tools[-1]
                            n={"predict_risk":"获取风险预测","query_kg_impact":"分析知识图谱","search_similar_cases":"检索历史案例"}
                            yield history+[{"role":"user","content":message},{"role":"assistant","content":f"⏳ {n.get(last,'处理中')}..."}]
                    clean = re.sub(r'\n?🔧[^\n]*\n','\n',full)
                    clean = re.sub(r'\n?✅[^\n]*\n','\n',clean)
                    clean = re.sub(r'^\s*---\s*\n+','',clean)
                    main,src = clean,""
                    if "\n---\n" in clean: parts=clean.rsplit("\n---\n",1); main,src=parts[0].strip(),parts[1].strip()
                    final = main
                    if src: final += f'\n<div class="source-cite">{src.replace(chr(10)," · ")}</div>'
                    final = re.sub(r'\n{3,}','\n\n',final)
                    yield history+[{"role":"user","content":message},{"role":"assistant","content":final}]

                def clear_msg(): return ""
                send.click(fn=respond, inputs=[msg,chatbot], outputs=[chatbot])
                send.click(fn=clear_msg, outputs=[msg])
                msg.submit(fn=respond, inputs=[msg,chatbot], outputs=[chatbot])
                msg.submit(fn=clear_msg, outputs=[msg])
                gr.Examples(examples=["2025年8月15日沙特有山洪风险吗？","明天利雅得地区会不会有热浪？","红海沿岸有没有风浪预警？"], inputs=msg, label="💡 试试这些问题")

        with gr.Row(visible=False) as map_page:
            with gr.Column(scale=1):
                dtype = gr.Dropdown(choices=[("暴雨山洪","flash_flood"),("极端高温","extreme_heat"),("沙尘强风","dust_wind"),("沿海风浪","coastal_wave")], value="flash_flood", label="灾害类型")
                date = gr.Textbox(value="2025-08-28", label="日期 (YYYY-MM-DD)")
                btn = gr.Button("🔍 查询风险地图", variant="primary")
                info = gr.Markdown("")
            with gr.Column(scale=3):
                img = gr.Plot(label="风险分布图", value=None)

        with gr.Row(visible=False) as kg_page:
            with gr.Column():
                kg_btn = gr.Button("🔄 加载知识图谱", variant="primary")
                kg_html = gr.HTML(value="<p style='color:#999'>点击按钮加载</p>")
                kg_info = gr.Markdown(f"🏙️ {len(CITIES)}城市 | ✈️ {len(AIRPORTS)}机场 | 🛣️ {len(HIGHWAYS)}高速 | 🌊 {len(WADIS)}Wadi | 🚢 6港口")

        def switch_page(choice):
            return (choice=="💬 智能对话", choice=="🗺️ 灾害地图", choice=="🕸️ 知识图谱")

        page.change(fn=switch_page, inputs=[page], outputs=[chat_page, map_page, kg_page])
        btn.click(fn=plot_risk_map, inputs=[dtype,date], outputs=[img,info])
        kg_btn.click(fn=plot_kg, outputs=[kg_html])

    return app

def launch_app(share=False, **kw):
    kw.setdefault("server_name","0.0.0.0"); kw.setdefault("server_port",7860); kw.setdefault("css",CSS)
    log.info("启动 MAZU Web...")
    app = build_ui(); app.launch(share=share, **kw)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true"); p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    launch_app(share=args.share, server_port=args.port)

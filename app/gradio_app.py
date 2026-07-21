"""MAZU 沙特多灾种预警 — Web界面"""

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
body,.gradio-container{background:#f0f2f5!important}
footer{display:none!important}
.main-header{text-align:center;padding:32px 0 16px 0}
.main-header h1{font-size:26px;font-weight:700;color:#1a1a2e;margin:0}
.main-header p{color:#64748b;font-size:14px;margin:8px 0 0 0}
#chatbot{border-radius:12px}
.input-row{background:#fff;border-radius:16px;border:1px solid #e2e8f0;padding:8px 16px}
.input-row textarea{border:none!important;box-shadow:none!important;font-size:15px}
.send-btn{background:#4f46e5!important;border:none!important;color:#fff!important;font-weight:600!important;padding:10px 24px!important;border-radius:10px!important}
.send-btn:hover{background:#4338ca!important}
.source-cite{font-size:11px;color:#94a3b8;margin-top:8px;padding-top:6px;border-top:1px solid #e2e8f0}
"""

# === 数据 ===
COLORS = {
    "flash_flood": [[0,"#fff5f0"],[.25,"#fc9272"],[.5,"#ef3b2d"],[.75,"#a50f15"]],
    "extreme_heat": [[0,"#fff7ec"],[.25,"#fdbb84"],[.5,"#ef6548"],[.75,"#7f0000"]],
    "dust_wind": [[0,"#ffffd4"],[.25,"#febd5a"],[.5,"#f77f00"],[.75,"#662506"]],
    "coastal_wave": [[0,"#f7fbff"],[.25,"#9ecae1"],[.5,"#3182bd"],[.75,"#08519c"]],
}
NAMES = {"flash_flood":"暴雨山洪","extreme_heat":"极端高温","dust_wind":"沙尘强风","coastal_wave":"沿海风浪"}

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
    feats = [f for f in DISASTER_FEATURES[disaster_type] if f not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")]
    try:
        df = load_to_dataframe(date_str, date_str, variables=feats, show_progress=False).fillna(0)
        result = engine.predict(df, disaster_type)
    except Exception as e:
        return go.Figure(), f"加载失败: {e}"

    proba, lat_arr, lon_arr = result["proba"], result["lat"], result["lon"]
    n_lat, n_lon = 160, 220
    grid = np.full((n_lat, n_lon), np.nan)
    for p, la, lo in zip(proba, lat_arr, lon_arr):
        grid[int(round((la-16)/0.1)), int(round((lo-34)/0.1))] = p

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=grid, x=np.linspace(34,56,n_lon), y=np.linspace(16,32,n_lat),
        colorscale=COLORS[disaster_type], zmin=0, zmax=1, colorbar_title="风险概率",
        hovertemplate="经度:%{x:.1f}<br>纬度:%{y:.1f}<br>风险:%{z:.3f}<extra></extra>"))

    cl, cn = [], []
    for c in CITIES:
        if 16<=c["lat"]<=32 and 34<=c["lon"]<=56:
            cl.append((c["lon"],c["lat"])); cn.append(c["name"].split("(")[0].strip())
    if cl:
        fig.add_trace(go.Scatter(x=[p[0] for p in cl], y=[p[1] for p in cl],
            mode="markers+text", text=cn, textposition="top center",
            textfont=dict(size=8,color="#333"), marker=dict(size=5,color="#333"), name="城市"))

    for h in HIGHWAYS:
        fig.add_trace(go.Scatter(x=[p[1] for p in h["path"]], y=[p[0] for p in h["path"]],
            mode="lines", line=dict(color="#94a3b8",width=.8,dash="dash"),
            opacity=.5, showlegend=False, hovertext=h["name"], hoverinfo="text"))

    wl, wn = [], []
    for w in WADIS:
        if 16<=w["lat"]<=32 and 34<=w["lon"]<=56:
            wl.append((w["lon"],w["lat"])); wn.append(f"{w['name']}({w['length_km']}km)")
    if wl:
        fig.add_trace(go.Scatter(x=[p[0] for p in wl], y=[p[1] for p in wl],
            mode="markers", marker=dict(symbol="diamond",size=8,color="#22c55e",opacity=.6),
            name="Wadi", hovertext=wn, hoverinfo="text"))

    fig.update_layout(
        title=dict(text=f"{date_str} {NAMES[disaster_type]}风险 | 高风险:{result['n_high']:,}({result['high_pct']}%) | 均值:{result['mean_risk']:.3f}", font=dict(size=14)),
        xaxis=dict(title="经度",range=[34,56]), yaxis=dict(title="纬度",range=[16,32]),
        height=580, margin=dict(l=50,r=30,t=50,b=40))
    return fig, f"**{date_str}** {NAMES[disaster_type]}: 高风险 {result['n_high']:,} 格点 ({result['high_pct']}%)"

# === 知识图谱 ===
def plot_kg():
    from pyvis.network import Network
    log.info("生成知识图谱...")
    net = Network(height="560px", width="100%", bgcolor="#f8fafc", font_color="#334155")
    def xy(lon,lat): return int((lon-34)*50), int((32-lat)*50)

    for c in CITIES:
        x,y=xy(c["lon"],c["lat"]); sz=np.clip(c["pop"]/40000,8,38)
        net.add_node(c["name"].split("(")[0].strip(), label=c["name"].split("(")[0].strip(),
            title=f"{c['name']}<br>人口:{c['pop']:,}<br>{c['type']}",
            color=dict(background="#ef4444",border="#b91c1c"), size=sz, x=x, y=y, physics=False)

    for a in AIRPORTS:
        x,y=xy(a["lon"],a["lat"]); nm=a["name"].split("(")[-1].replace(")","").strip()
        net.add_node(f"air_{nm}", label=nm, title=a["name"],
            color=dict(background="#3b82f6",border="#1d4ed8"), size=14, shape="triangle", x=x, y=y, physics=False)

    for w in WADIS:
        if 16<=w["lat"]<=32 and 34<=w["lon"]<=56:
            x,y=xy(w["lon"],w["lat"])
            net.add_node(w["name"], label=w["name"],
                title=f"{w['name']}<br>{w['length_km']}km<br>影响:{w.get('risk_city','')}",
                color=dict(background="#22c55e",border="#15803d"), size=max(w["length_km"]/10,6),
                shape="diamond", x=x, y=y, physics=False)

    for h in HIGHWAYS:
        for i in range(len(h["path"])-1):
            x1,y1=xy(h["path"][i][1],h["path"][i][0]); x2,y2=xy(h["path"][i+1][1],h["path"][i+1][0])
            net.add_node(f"hp_{h['name'][:4]}_{i}",hidden=True,x=x1,y=y1,size=1,physics=False)
            net.add_node(f"hp_{h['name'][:4]}_{i+1}",hidden=True,x=x2,y=y2,size=1,physics=False)
            net.add_edge(f"hp_{h['name'][:4]}_{i}",f"hp_{h['name'][:4]}_{i+1}",color="#cbd5e1",width=1)

    html = net.generate_html()
    log.info("图谱完成")
    return f'<iframe srcdoc="{html.replace(chr(34),"&quot;")}" width="100%" height="560" frameborder="0" style="border-radius:12px"></iframe>'

# === UI ===
def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:
        gr.HTML("""<div class="main-header">
            <h1>MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪  |  LightGBM + 知识图谱 + LLM Agent</p>
        </div>""")

        # ── 对话 ──
        with gr.Accordion("💬 智能对话", open=True):
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
                yield history+[{"role":"user","content":message},{"role":"assistant","content":"⏳ 分析中..."}]
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

            send.click(fn=respond,inputs=[msg,chatbot],outputs=[chatbot]).then(fn=lambda:"",outputs=[msg])
            msg.submit(fn=respond,inputs=[msg,chatbot],outputs=[chatbot]).then(fn=lambda:"",outputs=[msg])
            gr.Examples(examples=["2025年8月15日沙特有山洪风险吗？","明天利雅得地区会不会有热浪？","红海沿岸有没有风浪预警？"], inputs=msg)

        # ── 地图 ──
        with gr.Accordion("🗺️ 灾害地图", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    dtype = gr.Dropdown(choices=[("暴雨山洪","flash_flood"),("极端高温","extreme_heat"),("沙尘强风","dust_wind"),("沿海风浪","coastal_wave")], value="flash_flood", label="灾害类型")
                    date = gr.Textbox(value="2025-08-28", label="日期")
                    btn = gr.Button("🔍 查询", variant="primary")
                    info = gr.Markdown("")
                with gr.Column(scale=3):
                    img = gr.Plot(label="风险分布")

            btn.click(fn=plot_risk_map, inputs=[dtype,date], outputs=[img,info])

        # ── 知识图谱 ──
        with gr.Accordion("🕸️ 知识图谱", open=False):
            gr.Markdown(f"**承灾体网络**: {len(CITIES)}城市 · {len(AIRPORTS)}机场 · {len(HIGHWAYS)}高速 · {len(WADIS)}Wadi · 6港口")
            kg_btn = gr.Button("🔄 加载知识图谱", variant="primary")
            kg_html = gr.HTML(value="<p style='color:#94a3b8;text-align:center;padding:40px'>点击上方按钮加载</p>")
            kg_btn.click(fn=plot_kg, outputs=[kg_html])

    return app

def launch_app(share=False, **kw):
    kw.setdefault("server_name","0.0.0.0"); kw.setdefault("server_port",7860); kw.setdefault("css",CSS)
    log.info("MAZU Web 启动")
    app = build_ui(); app.launch(share=share, **kw)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true"); p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    launch_app(share=args.share, server_port=args.port)

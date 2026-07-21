"""MAZU 沙特多灾种预警智能体 — Web 界面（标签页）"""
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

CSS = """
*{box-sizing:border-box}
body{background:#f1f5f9!important;font-family:'Segoe UI',system-ui,-apple-system,sans-serif}
.gradio-container{max-width:960px!important;margin:20px auto!important;padding:0 16px!important}

/* 顶部标题 */
.main-header{text-align:center;padding:28px 20px 20px;margin-bottom:4px;
  background:#fff;border-radius:16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
.main-header h1{font-size:22px;margin:0;color:#1e293b;font-weight:700;letter-spacing:1px}
.main-header p{font-size:13px;color:#64748b;margin:6px 0 0}

/* 标签页 */
.tabs{border:none!important;background:transparent!important}
.tabs > .tab-nav{background:#fff!important;border-radius:14px 14px 0 0!important;
  padding:4px 8px 0!important;box-shadow:0 1px 4px rgba(0,0,0,.06);
  display:flex;gap:2px}
.tabs > .tab-nav button{font-size:14px!important;font-weight:600!important;
  padding:10px 24px!important;border-radius:10px 10px 0 0!important;
  border:none!important;background:transparent!important;color:#64748b!important;
  margin:0!important;transition:all .2s}
.tabs > .tab-nav button:hover{color:#334155!important;background:#f1f5f9!important}
.tabs > .tab-nav button.selected{color:#4f46e5!important;background:#eef2ff!important;
  box-shadow:inset 0 -2px 0 #4f46e5}
.tabs > .tabitem{background:#fff;border-radius:0 0 14px 14px;padding:0;
  box-shadow:0 1px 4px rgba(0,0,0,.06)}

/* 对话框 */
#chatbot{border-radius:14px!important;min-height:460px}
#chatbot > div{padding:20px!important}
.input-box{background:#fff;border-top:1px solid #f1f5f9;padding:14px 20px;border-radius:0 0 14px 14px}
.input-box textarea{border:2px solid #e2e8f0!important;border-radius:12px!important;
  padding:12px 16px!important;font-size:15px!important;
  transition:border-color .2s;outline:none!important}
.input-box textarea:focus{border-color:#4f46e5!important;box-shadow:0 0 0 3px rgba(79,70,229,.1)!important}
.send-btn{background:#4f46e5!important;color:#fff!important;font-weight:600!important;
  padding:12px 28px!important;border-radius:10px!important;border:none!important;
  transition:all .2s}
.send-btn:hover{background:#4338ca!important;transform:translateY(-1px);box-shadow:0 2px 8px rgba(79,70,229,.3)}

/* 来源引用 */
.source-cite{font-size:11px;color:#94a3b8;margin-top:8px;padding-top:6px;border-top:1px solid #f1f5f9}

/* 地图控制面板 */
.control-panel{background:#f8fafc;border-radius:14px;padding:20px;border:1px solid #e2e8f0}
.control-panel h3{font-size:15px;color:#334155;margin:0 0 12px}
.control-panel label{font-size:12px!important;font-weight:600!important;color:#64748b!important;margin-bottom:2px!important}

/* 通用圆角卡片 */
.card{background:#fff;border-radius:14px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:16px}

/* 加载占位 */
.loading-placeholder{color:#94a3b8;text-align:center;padding:60px;font-size:15px}
"""

# === 全局 ===
NAMES = {"flash_flood":"暴雨山洪","extreme_heat":"极端高温","dust_wind":"沙尘强风","coastal_wave":"沿海风浪"}
COLORS = {
    "flash_flood": [[0,"#fff5f0"],[.25,"#fc9272"],[.5,"#ef3b2d"],[.75,"#67000d"]],
    "extreme_heat": [[0,"#fff7ec"],[.25,"#fdbb84"],[.5,"#ef6548"],[.75,"#7f0000"]],
    "dust_wind": [[0,"#ffffd4"],[.25,"#febd5a"],[.5,"#f77f00"],[.75,"#662506"]],
    "coastal_wave": [[0,"#f7fbff"],[.25,"#9ecae1"],[.5,"#3182bd"],[.75,"#08519c"]],
}
_engine = None
def get_engine():
    global _engine
    if _engine is None: _engine = DisasterInference()
    return _engine

# === 地图 ===
def plot_risk_map(disaster_type, date_str, risk_min, risk_max):
    import plotly.graph_objects as go
    from data.loader import load_to_dataframe
    engine = get_engine()
    feats = [f for f in DISASTER_FEATURES[disaster_type] if f not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")]
    try:
        df = load_to_dataframe(date_str, date_str, variables=feats, show_progress=False).fillna(0)
        result = engine.predict(df, disaster_type)
    except Exception as e:
        return go.Figure(), f"❌ {e}", ""

    proba = result["proba"]
    # 应用风险筛选
    mask = (proba >= risk_min) & (proba <= risk_max)
    n_lat, n_lon = 160, 220
    grid = np.full((n_lat, n_lon), np.nan)
    for p, la, lo in zip(proba[mask], result["lat"][mask], result["lon"][mask]):
        grid[int(round((la-16)/0.1)), int(round((lo-34)/0.1))] = p

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=grid, x=np.linspace(34,56,n_lon), y=np.linspace(16,32,n_lat),
        colorscale=COLORS[disaster_type], zmin=0, zmax=1, colorbar=dict(title="风险概率", thickness=20, len=0.8),
        hovertemplate="<b>经度</b>: %{x:.1f}°E<br><b>纬度</b>: %{y:.1f}°N<br><b>风险</b>: %{z:.3f}<extra></extra>",
        name="风险"))

    # 城市标注
    c_x, c_y, c_t = [], [], []
    for c in CITIES:
        if 16<=c["lat"]<=32 and 34<=c["lon"]<=56:
            c_x.append(c["lon"]); c_y.append(c["lat"])
            c_t.append(c["name"].split("(")[0].strip())
    fig.add_trace(go.Scatter(x=c_x, y=c_y, mode="markers+text", text=c_t,
        textposition="top center", textfont=dict(size=9,color="#1e293b"),
        marker=dict(size=5,color="#1e293b"), name="城市", hoverinfo="text"))

    # 高速
    for h in HIGHWAYS:
        fig.add_trace(go.Scatter(x=[p[1] for p in h["path"]], y=[p[0] for p in h["path"]],
            mode="lines", line=dict(color="#94a3b8",width=1,dash="dot"),
            opacity=.5, showlegend=False, hovertext=h["name"], hoverinfo="text"))

    # Wadi
    w_x, w_y, w_t = [], [], []
    for w in WADIS:
        if 16<=w["lat"]<=32 and 34<=w["lon"]<=56:
            w_x.append(w["lon"]); w_y.append(w["lat"])
            w_t.append(f"{w['name']} ({w['length_km']}km)")
    fig.add_trace(go.Scatter(x=w_x, y=w_y, mode="markers",
        marker=dict(symbol="diamond",size=9,color="#06b6d4",opacity=.7,line=dict(color="#0891b2",width=1)),
        name="Wadi", hovertext=w_t, hoverinfo="text"))

    th = result["threshold"]
    fig.update_layout(
        title=dict(text=f"{date_str} {NAMES[disaster_type]}风险分布", font=dict(size=16,color="#1e293b"), x=.5),
        xaxis=dict(title="经度 (°E)", range=[34,56], showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="纬度 (°N)", range=[16,32], showgrid=True, gridcolor="#f1f5f9"),
        height=640, margin=dict(l=50,r=30,t=50,b=40),
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    stats = (f"| 指标 | 数值 |\n|---|---|\n"
             f"| 高风险格点 | **{result['n_high']:,}** ({result['high_pct']}%) |\n"
             f"| 平均风险 | {result['mean_risk']:.3f} |\n"
             f"| 最高风险 | {result['max_risk']:.3f} |\n"
             f"| 模型阈值 | {th} |")
    return fig, stats


# === 知识图谱 ===
def plot_kg():
    from pyvis.network import Network
    log.info("生成知识图谱...")

    net = Network(height="680px", width="100%", bgcolor="#ffffff", font_color="#334155", directed=True)
    net.set_options("""
    {"physics":{"barnesHut":{"gravitationalConstant":-3000,"centralGravity":0.3,"springLength":120,"springConstant":0.04},
     "stabilization":{"iterations":200}},"edges":{"smooth":{"type":"continuous"}},"interaction":{"hover":true,"tooltipDelay":200}}
    """)

    # 灾害类型节点（核心）
    disasters = [
        ("山洪", "⚡", "#ef4444", 35, "暴雨山洪：强降水+对流不稳定+地形抬升"),
        ("高温", "🔥", "#f97316", 35, "极端高温：气温异常偏离气候态"),
        ("沙尘", "🌪️", "#eab308", 35, "沙尘强风：强风+干燥+低湿度"),
        ("风浪", "🌊", "#3b82f6", 35, "沿海风浪：强风+低地形+水汽输送"),
    ]
    for name, icon, color, size, desc in disasters:
        net.add_node(f"disaster_{name}", label=f"{icon} {name}", title=desc, color=color, size=size, shape="dot", level=0)

    # 气象因子节点
    factors = [
        ("CAPE","#fca5a5", "对流有效位能：强对流核心驱动"), ("湿度","#93c5fd", "近地面相对湿度"),
        ("降水","#60a5fa", "日降水总量+卫星1h强降水"), ("风速","#a5b4fc", "10m/850hPa风速"),
        ("气温距平","#fdba74", "最高气温偏离气候态程度"), ("水汽输送","#86efac", "IVT+850hPa水汽通量"),
        ("地形","#d4d4d8", "orography+地表气压"), ("VPD","#fde68a", "饱和水汽压差：干燥指标"),
        ("涡度","#c4b5fd", "850hPa相对涡度"), ("风切变","#fecaca", "850-300hPa深层风切变"),
    ]
    factor_edges = [
        ("CAPE","山洪"),("CAPE","沙尘"),("湿度","山洪"),("湿度","沙尘"),("湿度","高温"),
        ("降水","山洪"),("风速","沙尘"),("风速","风浪"),("气温距平","高温"),
        ("水汽输送","山洪"),("水汽输送","风浪"),("地形","山洪"),("地形","风浪"),
        ("VPD","高温"),("VPD","沙尘"),("涡度","沙尘"),("风切变","沙尘"),
    ]
    for i,(name,color,desc) in enumerate(factors):
        net.add_node(f"factor_{name}", label=name, title=desc, color=color, size=22, shape="box", level=1)
    for src,dst in factor_edges:
        net.add_edge(f"factor_{src}", f"disaster_{dst}", color="#cbd5e1", width=1.5, dashes=False)

    # 基础设施节点
    added_cities = set()
    for c in CITIES[:12]:
        cname = c["name"].split("(")[0].strip()
        added_cities.add(cname)
        net.add_node(f"city_{cname}", label=cname, title=f"人口:{c['pop']:,}", color="#475569", size=18, shape="dot", level=2)

    # 城市-灾害关联（只连接已添加的城市）
    city_disaster = {"利雅得":["高温","沙尘"],"吉达":["山洪","风浪"],"麦加":["山洪","高温"],
        "达曼":["高温","沙尘","风浪"],"艾卜哈":["山洪"],"塔伊夫":["山洪"],
        "哈伊勒":["沙尘"],"布赖代":["沙尘"],"朱拜勒":["风浪"],"纳季兰":["山洪"],
        "麦地那":["高温"],"胡富夫":["高温","沙尘"],"延布":["风浪"]}
    for cname, dlist in city_disaster.items():
        if cname in added_cities:
            for d in dlist:
                net.add_edge(f"city_{cname}", f"disaster_{d}", color="#e2e8f0", width=0.8, dashes=True)

    # Wadi
    for w in WADIS:
        net.add_node(f"wadi_{w['name']}", label=w['name'], title=f"{w['length_km']}km", color="#06b6d4", size=12, shape="diamond", level=3)
        net.add_edge(f"wadi_{w['name']}", "disaster_山洪", color="#a5f3fc", width=1)

    html = net.generate_html()
    log.info("图谱完成")
    return f'<iframe srcdoc="{html.replace(chr(34),"&quot;")}" width="100%" height="680" frameborder="0" style="border:none"></iframe>'


# === UI ===
def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:
        gr.HTML("""<div class="main-header">
            <h1>MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 ｜ LightGBM · 知识图谱 · LLM Agent</p>
        </div>""")

        with gr.Tabs(elem_classes=["tabs"]):
            # ===== Tab 1: 对话 =====
            with gr.TabItem("💬 智能对话", id="chat", elem_classes=["tabitem"]):
                chatbot = gr.Chatbot(label="", height=520, elem_id="chatbot", show_label=False)
                with gr.Row(elem_classes=["input-box"]):
                    msg = gr.Textbox(placeholder="输入问题，如：明天利雅得会有热浪吗？", scale=10, container=False, show_label=False, max_lines=3)
                    send = gr.Button("发送", variant="primary", scale=1, elem_classes=["send-btn"])

                _agent = [None]
                def get_agent():
                    if _agent[0] is None: _agent[0] = MazuAgent(verbose=False)
                    return _agent[0]

                def respond(message, history):
                    history = history or []
                    yield history+[{"role":"user","content":message},{"role":"assistant","content":"⏳"}]
                    full, last = "", ""
                    for chunk in get_agent().chat_stream(message):
                        full += chunk
                        tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                        if tools and tools[-1]!=last:
                            last=tools[-1]; n={"predict_risk":"获取风险预测","query_kg_impact":"分析影响链","search_similar_cases":"检索案例"}
                            yield history+[{"role":"user","content":message},{"role":"assistant","content":f"⏳ {n.get(last,'处理中')}..."}]
                    clean = re.sub(r'\n?🔧[^\n]*\n','\n',full); clean = re.sub(r'\n?✅[^\n]*\n','\n',clean)
                    clean = re.sub(r'^\s*---\s*\n+','',clean)
                    main,src = clean,""
                    if "\n---\n" in clean: parts=clean.rsplit("\n---\n",1); main,src=parts[0].strip(),parts[1].strip()
                    final = main
                    if src: final += f'\n<div class="source-cite">{src.replace(chr(10)," · ")}</div>'
                    final = re.sub(r'\n{3,}','\n\n',final)
                    yield history+[{"role":"user","content":message},{"role":"assistant","content":final}]

                send.click(fn=respond,inputs=[msg,chatbot],outputs=[chatbot]).then(fn=lambda:"",outputs=[msg])
                msg.submit(fn=respond,inputs=[msg,chatbot],outputs=[chatbot]).then(fn=lambda:"",outputs=[msg])
                gr.Examples(examples=["2025年8月15日沙特有山洪风险吗？","明天利雅得地区会不会有热浪？","红海沿岸有没有风浪预警？","看看8月20日的沙尘暴预测"], inputs=msg)

            # ===== Tab 2: 地图 =====
            with gr.TabItem("🗺️ 灾害地图", id="map", elem_classes=["tabitem"]):
                with gr.Row():
                    with gr.Column(scale=4):
                        map_plot = gr.Plot(label="")
                    with gr.Column(scale=1, elem_classes=["control-panel"]):
                        gr.Markdown("### ⚙️ 控制面板")
                        dtype = gr.Dropdown(
                            choices=[("🌊 暴雨山洪","flash_flood"),("🔥 极端高温","extreme_heat"),
                                     ("🌪️ 沙尘强风","dust_wind"),("🌊 沿海风浪","coastal_wave")],
                            value="flash_flood", label="灾害类型")
                        date = gr.Textbox(value="2025-08-28", label="📅 日期 (YYYY-MM-DD)")
                        gr.Markdown("**风险等级筛选**")
                        risk_min = gr.Slider(0, 1, value=0, step=0.05, label="最低风险")
                        risk_max = gr.Slider(0, 1, value=1, step=0.05, label="最高风险")
                        btn = gr.Button("🔍 更新地图", variant="primary")
                        stats = gr.Markdown("")

                btn.click(fn=plot_risk_map, inputs=[dtype,date,risk_min,risk_max], outputs=[map_plot,stats])

            # ===== Tab 3: 知识图谱 =====
            with gr.TabItem("🕸️ 知识图谱", id="kg", elem_classes=["tabitem"]):
                gr.Markdown("### 多层级知识网络 — 气象因子 → 灾害类型 → 基础设施关联")
                kg_html = gr.HTML(value="<p style='color:#94a3b8;text-align:center;padding:60px;font-size:16px'>点击下方按钮加载交互式知识图谱</p>")
                with gr.Row():
                    kg_btn = gr.Button("🔄 加载知识图谱", variant="primary", scale=1)
                    gr.Markdown("🖱️ 可拖拽节点、缩放、悬停查看详情 | 实线=直接影响，虚线=间接关联")
                kg_btn.click(fn=plot_kg, outputs=[kg_html])

    return app


def launch_app(share=False, **kw):
    kw.setdefault("server_name","0.0.0.0"); kw.setdefault("server_port",7860)
    log.info("MAZU Web 启动")
    app = build_ui(); app.launch(share=share, css=CSS, **kw)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--share",action="store_true"); p.add_argument("--port",type=int,default=7860)
    launch_app(share=p.parse_args().share, server_port=p.parse_args().port)

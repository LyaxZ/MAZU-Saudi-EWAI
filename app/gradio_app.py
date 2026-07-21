"""
MAZU 沙特多灾种预警智能体 — Web 界面（三Tab：对话 | 地图 | 知识图谱）
"""

import sys, os, re, logging
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import gradio as gr

from llm_agent.agent import MazuAgent
from models.inference import DisasterInference
from config.model_config import DISASTER_FEATURES
from config.infrastructure import CITIES, AIRPORTS, HIGHWAYS, WADIS

# ═══════ 日志 ═══════
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("MAZU")

# ═══════ 字体 ═══════
for f in ["Microsoft YaHei", "SimHei", "Noto Sans SC"]:
    try:
        plt.rcParams["font.sans-serif"] = [f]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

# ═══════ CSS ═══════
CSS = """
body, .gradio-container { background: #f7f8fa !important; }
footer { display: none !important; }
.main-header { text-align: center; padding: 20px 0 10px 0; }
.main-header h1 { font-size: 28px; font-weight: 700; color: #1a1a2e; margin: 0; }
.main-header p { color: #8e8ea0; font-size: 14px; margin: 4px 0 0 0; }
#chatbot { border-radius: 12px; max-width: 100%; }
#chatbot > div { max-width: 100% !important; }
.input-row { background: #fff; border-radius: 16px; border: 1px solid #e0e0e0; padding: 8px 12px; }
.input-row textarea { border: none !important; box-shadow: none !important; }
.send-btn { background: #4f46e5 !important; border: none !important; color: white !important; font-weight: 600 !important; }
.send-btn:hover { background: #4338ca !important; }
.source-cite { font-size: 11px; color: #9ca3af; margin-top: 8px; padding-top: 6px; border-top: 1px solid #f0f0f0; }
"""

# ═══════ 工具函数 ═══════
DISASTER_META = {
    "flash_flood": ("暴雨山洪", "Reds"),
    "extreme_heat": ("极端高温", "OrRd"),
    "dust_wind": ("沙尘强风", "YlOrBr"),
    "coastal_wave": ("沿海风浪", "Blues"),
}

COLOR_SCALES = {
    "Reds": [[0,"#fff5f0"],[.25,"#fc9272"],[.5,"#ef3b2d"],[.75,"#a50f15"]],
    "OrRd": [[0,"#fff7ec"],[.25,"#fdbb84"],[.5,"#ef6548"],[.75,"#7f0000"]],
    "YlOrBr": [[0,"#ffffd4"],[.25,"#febd5a"],[.5,"#f77f00"],[.75,"#662506"]],
    "Blues": [[0,"#f7fbff"],[.25,"#9ecae1"],[.5,"#3182bd"],[.75,"#08519c"]],
}

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = DisasterInference()
    return _engine

# ═══════ 灾害地图（plotly 交互式） ═══════

def plot_risk_map(disaster_type, date_str):
    """交互式风险热力图 → plotly Figure"""
    import plotly.graph_objects as go
    from data.loader import load_to_dataframe
    engine = get_engine()
    name, cmap_name = DISASTER_META[disaster_type]
    feats = DISASTER_FEATURES[disaster_type]
    load_vars = [f for f in feats if f not in (
        "lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")]

    try:
        log.info(f"加载地图数据: {date_str} {name}")
        df = load_to_dataframe(date_str, date_str, variables=load_vars,
                               show_progress=False).fillna(0)
        result = engine.predict(df, disaster_type)
    except Exception as e:
        log.error(f"地图加载失败: {e}")
        return go.Figure(), f"加载失败: {e}"

    proba, lat_arr, lon_arr = result["proba"], result["lat"], result["lon"]
    threshold = result["threshold"]

    # 构建 160×220 网格
    n_lat, n_lon = 160, 220
    lats_u = np.linspace(16.0, 32.0, n_lat)
    lons_u = np.linspace(34.0, 56.0, n_lon)
    grid = np.full((n_lat, n_lon), np.nan)
    for p, la, lo in zip(proba, lat_arr, lon_arr):
        grid[int(round((la - 16) / 0.1)), int(round((lo - 34) / 0.1))] = p

    # 热力图
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=grid, x=lons_u, y=lats_u,
        colorscale=COLOR_SCALES.get(cmap_name, "Reds"),
        zmin=0, zmax=1,
        colorbar_title="风险概率",
        hovertemplate="经度: %{x:.1f}°E<br>纬度: %{y:.1f}°N<br>风险: %{z:.3f}<extra></extra>",
        name="风险概率",
    ))

    # 城市标注
    city_lats, city_lons, city_names = [], [], []
    for c in CITIES:
        if 16 <= c["lat"] <= 32 and 34 <= c["lon"] <= 56:
            city_lats.append(c["lat"]); city_lons.append(c["lon"])
            city_names.append(c["name"].split("(")[0].strip())
    fig.add_trace(go.Scatter(
        x=city_lons, y=city_lats, mode="markers+text",
        text=city_names, textposition="top center",
        textfont=dict(size=8, color="#333"),
        marker=dict(size=5, color="black"), name="城市",
        hovertemplate="%{text}<extra></extra>",
    ))

    # 高速路网
    for h in HIGHWAYS:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in h["path"]], y=[p[0] for p in h["path"]],
            mode="lines", line=dict(color="gray", width=0.8, dash="dash"),
            opacity=0.4, showlegend=False,
            hoverinfo="text", hovertext=h["name"],
        ))

    # Wadi
    wadi_lats, wadi_lons, wadi_names = [], [], []
    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            wadi_lats.append(w["lat"]); wadi_lons.append(w["lon"])
            wadi_names.append(f"{w['name']}({w['length_km']}km)")
    fig.add_trace(go.Scatter(
        x=wadi_lons, y=wadi_lats, mode="markers",
        marker=dict(symbol="diamond", size=8, color="#2ecc71", opacity=0.6),
        name="Wadi河道", hovertext=wadi_names, hoverinfo="text",
    ))

    n_high, pct = result["n_high"], result["high_pct"]
    fig.update_layout(
        title=dict(text=f"{date_str} {name} 风险分布 | "
                   f"高风险:{n_high:,}({pct}%) | 均值:{result['mean_risk']:.3f}",
                   font=dict(size=14)),
        xaxis=dict(title="经度 (°E)", range=[34, 56], constrain="domain"),
        yaxis=dict(title="纬度 (°N)", range=[16, 32], scaleanchor="x", scaleratio=1),
        height=600, margin=dict(l=50, r=30, t=50, b=50),
        hovermode="closest",
    )
    log.info(f"地图生成完毕")
    info = f"{date_str} {name}: 高风险 {n_high:,} 格点 ({pct}%)"
    return fig, info


# ═══════ 知识图谱（pyvis 交互式） ═══════

def plot_kg():
    """交互式网络图 → HTML 字符串"""
    from pyvis.network import Network

    log.info("生成交互式知识图谱...")
    net = Network(height="600px", width="100%", bgcolor="#ffffff",
                  font_color="#333333", directed=False)
    net.repulsion(node_distance=150, spring_length=200)

    # 经纬度→画布坐标映射
    def to_xy(lon, lat):
        return int((lon - 34) * 50), int((32 - lat) * 50)

    # 城市节点
    for c in CITIES:
        x, y = to_xy(c["lon"], c["lat"])
        size = np.clip(c["pop"] / 50000, 8, 40)
        net.add_node(c["name"].split("(")[0].strip(),
                     label=c["name"].split("(")[0].strip(),
                     title=f"{c['name']}<br>人口: {c['pop']:,}<br>类型: {c['type']}",
                     color="#e74c3c", size=size * 1.5, x=x, y=y,
                     physics=False)

    # 机场节点
    for a in AIRPORTS:
        x, y = to_xy(a["lon"], a["lat"])
        name = a["name"].split("(")[-1].replace(")", "").strip()
        net.add_node(f"✈{name}", label=name,
                     title=f"{a['name']}<br>城市: {a.get('city','')}",
                     color="#3498db", size=15, shape="triangle",
                     x=x, y=y, physics=False)

    # Wadi节点
    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            x, y = to_xy(w["lon"], w["lat"])
            net.add_node(w["name"], label=w["name"],
                         title=f"{w['name']}<br>长度: {w['length_km']}km<br>影响: {w.get('risk_city','')}",
                         color="#2ecc71", size=w["length_km"] / 8,
                         shape="diamond", x=x, y=y, physics=False)

    # 高速作为边
    for h in HIGHWAYS:
        path = h["path"]
        for i in range(len(path) - 1):
            n1, n2 = f"h_{h['name'][:6]}_{i}", f"h_{h['name'][:6]}_{i+1}"
            x1, y1 = to_xy(path[i][1], path[i][0])
            x2, y2 = to_xy(path[i+1][1], path[i+1][0])
            net.add_node(n1, hidden=True, x=x1, y=y1, size=1, physics=False)
            net.add_node(n2, hidden=True, x=x2, y=y2, size=1, physics=False)
            net.add_edge(n1, n2, color="#bdc3c7", width=1)

    html = net.generate_html()
    # 注入到 iframe 供 gr.HTML 显示
    log.info("图谱生成完毕")
    return f'<iframe srcdoc="{html.replace(chr(34), "&quot;")}" width="100%" height="600" frameborder="0"></iframe>'

# ═══════ UI ═══════

def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:
        gr.HTML("""
        <div class="main-header">
            <h1>🏔 MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 | LightGBM + 知识图谱 + LLM</p>
        </div>""")

        with gr.Tabs():
            # ── Tab 1: 对话 ──
            with gr.TabItem("💬 智能对话"):
                chatbot = gr.Chatbot(label="", height=440, elem_id="chatbot", show_label=False)
                with gr.Row(elem_classes=["input-row"]):
                    msg = gr.Textbox(placeholder="输入问题...", scale=10, container=False,
                                     show_label=False, max_lines=4)
                    send = gr.Button("发送", variant="primary", scale=1, elem_classes=["send-btn"])

                _agent = [None]  # 延迟初始化，避免阻塞 Tab 切换

                def get_agent():
                    if _agent[0] is None:
                        _agent[0] = MazuAgent(verbose=False)
                    return _agent[0]

                def respond(message, history):
                    history = history or []
                    yield history + [{"role":"user","content":message},
                                     {"role":"assistant","content":"⏳ 正在分析..."}]
                    full, last = "", ""
                    for chunk in get_agent().chat_stream(message):
                        full += chunk
                        tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                        if tools and tools[-1] != last:
                            last = tools[-1]
                            names = {"predict_risk":"获取风险预测数据",
                                     "query_kg_impact":"分析知识图谱影响链",
                                     "search_similar_cases":"检索历史相似案例"}
                            yield history + [{"role":"user","content":message},
                                             {"role":"assistant","content":f"⏳ {names.get(last,'处理中')}..."}]
                    clean = re.sub(r'\n?🔧[^\n]*\n','\n', full)
                    clean = re.sub(r'\n?✅[^\n]*\n','\n', clean)
                    clean = re.sub(r'^\s*---\s*\n+','', clean)
                    main, src = clean, ""
                    if "\n---\n" in clean:
                        parts = clean.rsplit("\n---\n",1)
                        main, src = parts[0].strip(), parts[1].strip()
                    final = main
                    if src:
                        final += f'\n<div class="source-cite">{src.replace(chr(10)," · ")}</div>'
                    final = re.sub(r'\n{3,}','\n\n', final)
                    yield history + [{"role":"user","content":message},
                                     {"role":"assistant","content":final}]

                def clear_msg():
                    return ""

                send.click(fn=respond, inputs=[msg,chatbot], outputs=[chatbot])
                send.click(fn=clear_msg, outputs=[msg])
                msg.submit(fn=respond, inputs=[msg,chatbot], outputs=[chatbot])
                msg.submit(fn=clear_msg, outputs=[msg])

                gr.Examples(
                    examples=["2025年8月15日沙特有山洪风险吗？",
                              "明天利雅得地区会不会有热浪？",
                              "红海沿岸有没有风浪预警？"],
                    inputs=msg, label="💡 试试这些问题")

            # ── Tab 2: 地图 ──
            with gr.TabItem("🗺️ 灾害地图"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dtype = gr.Dropdown(
                            choices=[("暴雨山洪","flash_flood"),("极端高温","extreme_heat"),
                                     ("沙尘强风","dust_wind"),("沿海风浪","coastal_wave")],
                            value="flash_flood", label="灾害类型")
                        date = gr.Textbox(value="2025-08-28", label="日期 (YYYY-MM-DD)")
                        btn = gr.Button("🔍 查询风险地图", variant="primary")
                        info = gr.Markdown("")
                    with gr.Column(scale=3):
                        img = gr.Plot(label="风险分布图")

                btn.click(fn=plot_risk_map, inputs=[dtype,date], outputs=[img,info])

            # ── Tab 3: 知识图谱 ──
            with gr.TabItem("🕸️ 知识图谱"):
                kg_btn = gr.Button("🔄 加载知识图谱", variant="primary")
                kg_html = gr.HTML(label="沙特承灾体网络")
                kg_info = gr.Markdown(
                    f"**知识图谱结构**\n\n"
                    f"- 🏙️ {len(CITIES)} 个城市节点 | ✈️ {len(AIRPORTS)} 个机场\n"
                    f"- 🛣️ {len(HIGHWAYS)} 条高速 | 🌊 {len(WADIS)} 条Wadi河道\n"
                    f"- 🚢 6 个港口\n\n"
                    f"> D8流向边基于地形推导，Wadi坐标为近似值\n"
                    f"> 🖱️ 可拖拽节点、缩放、悬停查看详情")
                kg_btn.click(fn=plot_kg, outputs=[kg_html])

    return app


def launch_app(share=False, **kw):
    kw.setdefault("server_name","0.0.0.0")
    kw.setdefault("server_port",7860)
    kw.setdefault("css",CSS)
    log.info("MAZU Web 界面启动中...")
    app = build_ui()
    app.launch(share=share, **kw)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MAZU")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    launch_app(share=args.share, server_port=args.port)

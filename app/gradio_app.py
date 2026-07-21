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

def fig_to_array(fig):
    """Matplotlib figure → RGBA numpy array"""
    fig.canvas.draw()
    arr = np.array(fig.canvas.buffer_rgba())
    plt.close(fig)
    return arr


DISASTER_META = {
    "flash_flood": ("暴雨山洪", "Reds"),
    "extreme_heat": ("极端高温", "OrRd"),
    "dust_wind": ("沙尘强风", "YlOrBr"),
    "coastal_wave": ("沿海风浪", "Blues"),
}

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = DisasterInference()
    return _engine

# ═══════ 灾害地图 ═══════

def plot_risk_map(disaster_type, date_str):
    """生成沙特灾害风险热力图 → numpy array"""
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
        return None, f"日期 {date_str} 数据加载失败: {e}"

    proba, lat, lon = result["proba"], result["lat"], result["lon"]
    threshold = result["threshold"]

    n_lat, n_lon = 160, 220
    lats_u = np.linspace(32.0, 16.0, n_lat)
    lons_u = np.linspace(34.0, 56.0, n_lon)
    grid = np.full((n_lat, n_lon), np.nan)
    for p, la, lo in zip(proba, lat, lon):
        i = np.argmin(np.abs(lats_u - la))
        j = np.argmin(np.abs(lons_u - lo))
        grid[i, j] = p

    fig, ax = plt.subplots(figsize=(12, 10))
    from matplotlib import colormaps
    cmap = colormaps[cmap_name].copy()
    cmap.set_bad("#f0f0f0")
    ax.pcolormesh(lons_u, lats_u, grid, cmap=cmap, vmin=0, vmax=1, shading="auto")

    for c in CITIES:
        if 16 <= c["lat"] <= 32 and 34 <= c["lon"] <= 56:
            ax.plot(c["lon"], c["lat"], "ko", markersize=3)
            ax.text(c["lon"]+.15, c["lat"]+.15, c["name"].split("(")[0].strip(),
                    fontsize=6, color="#333", alpha=.8)
    for h in HIGHWAYS:
        ax.plot([p[1] for p in h["path"]], [p[0] for p in h["path"]],
                "gray", lw=.8, alpha=.4, ls="--")
    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            ax.plot(w["lon"], w["lat"], "D", color="blue", ms=4, alpha=.5)

    n_high, pct = result["n_high"], result["high_pct"]
    ax.set_title(f"{date_str} {name} 风险分布 | 高风险:{n_high:,}({pct}%) | "
                 f"均值:{result['mean_risk']:.3f} | 阈值:{threshold}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    ax.set_xlim(34, 56); ax.set_ylim(16, 32)
    fig.colorbar(ax.collections[0], ax=ax, shrink=.75).set_label("风险概率", fontsize=9)
    plt.tight_layout()

    info = f"{date_str} {name}: 高风险 {n_high:,} 格点 ({pct}%)"
    log.info(f"地图生成完毕: {info}")
    return fig_to_array(fig), info

# ═══════ 知识图谱 ═══════

def plot_kg():
    """知识图谱 → numpy array"""
    log.info("生成知识图谱...")
    fig, ax = plt.subplots(figsize=(12, 10))

    for c in CITIES:
        ax.scatter(c["lon"], c["lat"], c="#e74c3c", s=c["pop"]/50000,
                   alpha=.8, edgecolors="white", lw=.5, zorder=3)
        ax.text(c["lon"]+.15, c["lat"]+.15, c["name"].split("(")[0].strip(),
                fontsize=5, alpha=.7, zorder=3)

    for a in AIRPORTS:
        ax.scatter(a["lon"], a["lat"], marker="^", c="#3498db", s=80,
                   alpha=.9, edgecolors="white", zorder=3)
        ax.text(a["lon"]+.2, a["lat"]+.1,
                a["name"].split("(")[-1].replace(")","").strip(),
                fontsize=5, color="#3498db", alpha=.8)

    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            ax.plot(w["lon"], w["lat"], "D", color="#2ecc71",
                    ms=w["length_km"]/15, alpha=.6, zorder=2)

    for h in HIGHWAYS:
        ax.plot([p[1] for p in h["path"]], [p[0] for p in h["path"]],
                "#bdc3c7", lw=.6, alpha=.5, zorder=1)

    ax.set_xlim(34, 56); ax.set_ylim(16, 32)
    ax.set_xlabel("经度"); ax.set_ylabel("纬度")
    ax.set_title(f"MAZU 知识图谱 — 承灾体网络 | 城市{len(CITIES)} | "
                 f"机场{len(AIRPORTS)} | 高速{len(HIGHWAYS)} | Wadi{len(WADIS)}",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor="#e74c3c",ms=8,label="城市"),
        Line2D([0],[0],marker="^",color="w",markerfacecolor="#3498db",ms=8,label="机场"),
        Line2D([0],[0],marker="D",color="w",markerfacecolor="#2ecc71",ms=8,label="Wadi"),
        Line2D([0],[0],color="#bdc3c7",lw=1,label="高速"),
    ], loc="lower right", fontsize=8)
    plt.tight_layout()

    log.info("图谱生成完毕")
    return fig_to_array(fig)

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
                chatbot = gr.Chatbot(label="", height=460, elem_id="chatbot", show_label=False)
                with gr.Row(elem_classes=["input-row"]):
                    msg = gr.Textbox(placeholder="输入问题...", scale=10, container=False,
                                     show_label=False, max_lines=4)
                    send = gr.Button("发送", variant="primary", scale=1, elem_classes=["send-btn"])

                agent = MazuAgent(verbose=False)

                def respond(message, history):
                    history = history or []
                    yield history + [{"role":"user","content":message},
                                     {"role":"assistant","content":"⏳ 正在分析..."}]
                    full, last = "", ""
                    for chunk in agent.chat_stream(message):
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

                send.click(fn=respond, inputs=[msg,chatbot], outputs=[chatbot]).then(lambda:"",None,msg)
                msg.submit(fn=respond, inputs=[msg,chatbot], outputs=[chatbot]).then(lambda:"",None,msg)
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
                        img = gr.Image(label="风险分布图", type="numpy")

                btn.click(fn=plot_risk_map, inputs=[dtype,date], outputs=[img,info])

            # ── Tab 3: 知识图谱 ──
            with gr.TabItem("🕸️ 知识图谱"):
                kg_btn = gr.Button("🔄 加载知识图谱", variant="primary")
                kg_img = gr.Image(label="沙特承灾体网络", type="numpy")
                kg_info = gr.Markdown(
                    f"**知识图谱结构**\n\n"
                    f"- 🏙️ {len(CITIES)} 个城市节点 | ✈️ {len(AIRPORTS)} 个机场\n"
                    f"- 🛣️ {len(HIGHWAYS)} 条高速 | 🌊 {len(WADIS)} 条Wadi河道\n"
                    f"- 🚢 6 个港口\n\n"
                    f"> D8流向边基于地形推导，Wadi坐标为近似值")
                kg_btn.click(fn=plot_kg, outputs=[kg_img])

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

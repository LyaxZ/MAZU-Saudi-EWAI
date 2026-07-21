"""
MAZU 沙特多灾种预警智能体 — Web 界面（三Tab：对话 | 地图 | 知识图谱）

用法:
    python app/gradio_app.py
"""

import sys, os, re, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gradio as gr

from llm_agent.agent import MazuAgent
from models.inference import DisasterInference
from config.model_config import DISASTER_FEATURES
from config.infrastructure import CITIES, AIRPORTS, HIGHWAYS, WADIS

# ═══════════════════════════════════════════════
# 中文字体
# ═══════════════════════════════════════════════

for font in ["Microsoft YaHei", "SimHei", "Noto Sans SC"]:
    try:
        plt.rcParams["font.sans-serif"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

# ═══════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════

CSS = """
body, .gradio-container { background: #f7f8fa !important; }
footer { display: none !important; }
.main-header { text-align: center; padding: 20px 0 10px 0; }
.main-header h1 { font-size: 28px; font-weight: 700; color: #1a1a2e; margin: 0; }
.main-header p { color: #8e8ea0; font-size: 14px; margin: 4px 0 0 0; }
#chatbot { border-radius: 12px; max-width: 100%; }
#chatbot > div { max-width: 100% !important; }
#chatbot .message-wrap { max-width: 100% !important; }
.input-row { background: #fff; border-radius: 16px; border: 1px solid #e0e0e0; padding: 8px 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.input-row textarea { border: none !important; box-shadow: none !important; resize: none; font-size: 15px; padding: 8px; }
.send-btn { background: #4f46e5 !important; border: none !important; border-radius: 10px !important; color: white !important; font-weight: 600 !important; padding: 10px 20px !important; }
.send-btn:hover { background: #4338ca !important; }
.source-cite { font-size: 11px; color: #9ca3af; margin-top: 8px; padding-top: 6px; border-top: 1px solid #f0f0f0; }
.tab-nav button { font-size: 15px !important; font-weight: 600 !important; }
"""

# ═══════════════════════════════════════════════
# 灾害地图
# ═══════════════════════════════════════════════

DISASTER_NAMES = {
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


def plot_risk_map(disaster_type, date_str):
    engine = get_engine()
    from data.loader import load_to_dataframe

    feats = DISASTER_FEATURES[disaster_type]
    load_vars = [f for f in feats if f not in (
        "lat_sin", "lat_cos", "lon_sin", "lon_cos", "sst_celsius")]

    try:
        df = load_to_dataframe(date_str, date_str, variables=load_vars,
                               show_progress=False).fillna(0)
    except Exception:
        return None, f"日期 {date_str} 数据加载失败"

    result = engine.predict(df, disaster_type)
    proba = result["proba"]
    lat = result["lat"]
    lon = result["lon"]
    threshold = result["threshold"]
    name, cmap_name = DISASTER_NAMES[disaster_type]

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
    im = ax.pcolormesh(lons_u, lats_u, grid, cmap=cmap,
                        vmin=0, vmax=1, shading="auto")

    for c in CITIES:
        if 16 <= c["lat"] <= 32 and 34 <= c["lon"] <= 56:
            ax.plot(c["lon"], c["lat"], "ko", markersize=3)
            ax.text(c["lon"] + 0.15, c["lat"] + 0.15,
                    c["name"].split("(")[0].strip(),
                    fontsize=6, color="#333", alpha=0.8)

    for h in HIGHWAYS:
        lons_p = [p[1] for p in h["path"]]
        lats_p = [p[0] for p in h["path"]]
        ax.plot(lons_p, lats_p, "gray", linewidth=0.8, alpha=0.4, linestyle="--")

    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            ax.plot(w["lon"], w["lat"], "D", color="blue", markersize=4, alpha=0.5)

    n_high = result["n_high"]
    pct = result["high_pct"]
    ax.set_title(f"{date_str} {name} 风险分布 | "
                 f"高风险: {n_high:,} ({pct}%) | 均值: {result['mean_risk']:.3f} | 阈值: {threshold}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_xlim(34, 56)
    ax.set_ylim(16, 32)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("风险概率", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf, f"{date_str} {name}: 高风险 {n_high:,} 格点 ({pct}%)"


# ═══════════════════════════════════════════════
# 知识图谱
# ═══════════════════════════════════════════════

def plot_kg():
    fig, ax = plt.subplots(figsize=(12, 10))

    for c in CITIES:
        ax.scatter(c["lon"], c["lat"], c="#e74c3c", s=c["pop"] / 50000,
                   alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)
        ax.text(c["lon"] + 0.15, c["lat"] + 0.15,
                c["name"].split("(")[0].strip(),
                fontsize=5, alpha=0.7, zorder=3)

    for a in AIRPORTS:
        ax.scatter(a["lat"] + 0.3, a["lon"] + 0.1, marker="^", c="#3498db",
                   s=80, alpha=0.9, edgecolors="white", zorder=3)
        ax.text(a["lat"] + 0.5, a["lon"] + 0.3,
                a["name"].split("(")[-1].replace(")", "").strip(),
                fontsize=5, color="#3498db", alpha=0.8)

    for w in WADIS:
        if 16 <= w["lat"] <= 32 and 34 <= w["lon"] <= 56:
            ax.plot(w["lon"], w["lat"], "D", color="#2ecc71",
                    markersize=w["length_km"] / 15, alpha=0.6, zorder=2)

    for h in HIGHWAYS:
        lons_p = [p[1] for p in h["path"]]
        lats_p = [p[0] for p in h["path"]]
        ax.plot(lons_p, lats_p, "#bdc3c7", linewidth=0.6, alpha=0.5, zorder=1)

    ax.set_xlim(34, 56)
    ax.set_ylim(16, 32)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title(f"MAZU 知识图谱 — 沙特承灾体网络 | "
                 f"城市{len(CITIES)} | 机场{len(AIRPORTS)} | 高速{len(HIGHWAYS)} | Wadi{len(WADIS)}",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="城市"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#3498db",
               markersize=8, label="机场"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ecc71",
               markersize=8, label="Wadi河道"),
        Line2D([0], [0], color="#bdc3c7", linewidth=1, label="高速路网"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:

        gr.HTML("""
        <div class="main-header">
            <h1>🏔 MAZU 沙特多灾种预警智能体</h1>
            <p>暴雨山洪 · 极端高温 · 沙尘强风 · 沿海风浪 | LightGBM + 知识图谱 + LLM</p>
        </div>
        """)

        with gr.Tabs(elem_classes=["tab-nav"]):
            # ── Tab 1: 智能对话 ──
            with gr.TabItem("💬 智能对话"):
                chatbot = gr.Chatbot(label="", height=460, elem_id="chatbot",
                                     show_label=False)
                with gr.Row(elem_classes=["input-row"]):
                    msg_input = gr.Textbox(
                        placeholder="输入问题，例如：2025年8月15日沙特有山洪风险吗？",
                        scale=10, container=False, show_label=False, max_lines=4)
                    send_btn = gr.Button("发送", variant="primary", scale=1,
                                         elem_classes=["send-btn"])

                _agent = MazuAgent(verbose=False)

                def respond(message, history):
                    history = history or []
                    yield history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": "⏳ 正在分析..."}]
                    full, last_tool = "", ""
                    for chunk in _agent.chat_stream(message):
                        full += chunk
                        tools = re.findall(r'🔧\s*(\S+)\.\.\.', full)
                        if tools and tools[-1] != last_tool:
                            last_tool = tools[-1]
                            names = {"predict_risk": "获取风险预测数据",
                                     "query_kg_impact": "分析知识图谱影响链",
                                     "search_similar_cases": "检索历史相似案例"}
                            yield history + [
                                {"role": "user", "content": message},
                                {"role": "assistant",
                                 "content": f"⏳ {names.get(last_tool, '处理中')}..."}]
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
                    yield history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": final}]

                send_btn.click(fn=respond, inputs=[msg_input, chatbot],
                               outputs=[chatbot]).then(lambda: "", None, msg_input)
                msg_input.submit(fn=respond, inputs=[msg_input, chatbot],
                                 outputs=[chatbot]).then(lambda: "", None, msg_input)
                gr.Examples(
                    examples=["2025年8月15日沙特有山洪风险吗？",
                              "明天利雅得地区会不会有热浪？",
                              "红海沿岸有没有风浪预警？",
                              "帮我看看8月20日的沙尘暴预测"],
                    inputs=msg_input, label="💡 试试这些问题")

            # ── Tab 2: 灾害地图 ──
            with gr.TabItem("🗺️ 灾害地图"):
                with gr.Row():
                    with gr.Column(scale=1):
                        disaster_sel = gr.Dropdown(
                            choices=[("暴雨山洪", "flash_flood"),
                                     ("极端高温", "extreme_heat"),
                                     ("沙尘强风", "dust_wind"),
                                     ("沿海风浪", "coastal_wave")],
                            value="flash_flood", label="灾害类型")
                        date_input = gr.Textbox(value="2025-08-28",
                                                label="日期 (YYYY-MM-DD)")
                        map_btn = gr.Button("🔍 查询风险地图", variant="primary")
                        map_info = gr.Markdown("")
                    with gr.Column(scale=3):
                        map_plot = gr.Image(label="风险分布图", type="pil")

                map_btn.click(fn=plot_risk_map,
                              inputs=[disaster_sel, date_input],
                              outputs=[map_plot, map_info])

            # ── Tab 3: 知识图谱 ──
            with gr.TabItem("🕸️ 知识图谱"):
                kg_btn = gr.Button("🔄 加载知识图谱", variant="primary")
                kg_plot = gr.Image(label="沙特承灾体网络", type="pil")
                kg_info = gr.Markdown(
                    f"**知识图谱结构**\n\n"
                    f"- 🏙️ {len(CITIES)} 个城市节点 | ✈️ {len(AIRPORTS)} 个机场\n"
                    f"- 🛣️ {len(HIGHWAYS)} 条高速 | 🌊 {len(WADIS)} 条Wadi河道\n"
                    f"- 🚢 6 个港口：吉达/朱拜勒/达曼/延布/吉赞/阿卜杜拉国王港\n\n"
                    f"> D8流向边基于地形推导，Wadi坐标为近似值")
                kg_btn.click(fn=plot_kg, outputs=[kg_plot])

    return app


def launch_app(share=False, **kwargs):
    kwargs.setdefault("server_name", "0.0.0.0")
    kwargs.setdefault("server_port", 7860)
    kwargs.setdefault("css", CSS)
    app = build_ui()
    app.launch(share=share, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAZU 预警智能体 Web")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch_app(share=args.share, server_port=args.port)

"""
MAZU 沙特多灾种预警智能体 — Gradio Web 演示

四个 Tab：
1. 🔍 风险查询: 选择日期 + 灾害 → 风险热力图 + 统计信息
2. 🌊 影响分析: 点击高风险区 → KG 影响链图 + 承灾体清单
3. 📋 预警简报: 一键生成结构化预警简报
4. 💬 智能对话: 自然语言交互，Agent 自动调用工具

用法:
    cd D:/Mazu/MAZU-Saudi-EWAI
    python app/gradio_app.py

    或

    from app.gradio_app import launch_app
    launch_app()
"""

import sys
import os
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr

# ── 路径 ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import INDICATORS_DIR, DISASTER_TYPES
from config.disaster_config import get_label_config

from data.loader import load_to_dataframe, get_variable_list
from data.label_builder import DisasterLabelBuilder
from data.preprocessor import DataPreprocessor, quick_preprocess

from features.temporal_features import TemporalFeatureBuilder
from features.spatial_features import SpatialFeatureBuilder
from features.feature_registry import FeatureRegistry

from kg.graph_builder import KnowledgeGraphBuilder
from kg.risk_propagation import RiskPropagator
from kg.case_retrieval import CaseRetrieval

from app.components.risk_heatmap import RiskHeatmap
from app.components.impact_graph import ImpactGraphVisualizer
from app.components.briefing_card import BriefingCardGenerator

# 模型推理引擎（懒加载单例）
from models.inference import DisasterInference, add_latlon_features
_inference_engine: Optional[DisasterInference] = None

def _get_engine() -> DisasterInference:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = DisasterInference()
    return _inference_engine


# ═══════════════════════════════════════════════════════════
# 全局状态（懒加载）
# ═══════════════════════════════════════════════════════════

class AppState:
    """全局应用状态，缓存图、模型等大对象。"""

    def __init__(self):
        self.G = None                  # KG 图
        self.graph_builder = None
        self.case_retrieval = None
        self.registry = None
        self.heatmap = RiskHeatmap()
        self.impact_viz = ImpactGraphVisualizer()
        self.briefing_gen = BriefingCardGenerator()
        self._initialized = False

    def ensure_initialized(self):
        """懒加载初始化。"""
        if self._initialized:
            return

        print("[AppState] 初始化知识图谱（可能需要1-2分钟）...")

        # 加载1天数据构建图
        try:
            df = load_to_dataframe(
                "2025-06-15", "2025-06-15",
                variables=["orography", "daily_precip_total", "wind10_speed"],
                show_progress=True,
            )
            self.graph_builder = KnowledgeGraphBuilder(coastal_orography_max=100.0)
            self.G = self.graph_builder.build(df)
            print("[AppState] 知识图谱构建完成")
        except Exception as e:
            print(f"[AppState] 知识图谱构建失败: {e}")

        # 预加载案例库（空库，运行时动态添加）
        self.case_retrieval = CaseRetrieval()

        # 特征注册表
        self.registry = FeatureRegistry()

        self._initialized = True
        print("[AppState] 初始化完成")


_state = AppState()


# ═══════════════════════════════════════════════════════════
# Tab 1: 风险查询
# ═══════════════════════════════════════════════════════════

def _get_available_dates() -> List[str]:
    """从 indicators 目录读取可用的日期列表。"""
    import glob
    files = glob.glob(os.path.join(INDICATORS_DIR, "saudi_indicators_*.nc"))
    dates = []
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace("saudi_indicators_", "").replace(".nc", "")
        if len(date_str) == 8:
            dates.append(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    return sorted(dates)


def tab1_query(date: str, disaster_type: str, use_features: bool = True):
    """Tab1 核心逻辑：加载数据 → 预处理 → 推理 → 热力图。

    Args:
        date: 日期 (YYYY-MM-DD)
        disaster_type: 灾害类型
        use_features: 是否构建衍生特征
    """
    if not date:
        return None, "⚠ 请先选择日期"

    try:
        _state.ensure_initialized()

        # ── 1. 加载数据 ──
        status = f"📡 加载 {date} 数据中..."
        yield None, status

        base_vars = [
            "daily_precip_total", "cape", "cin",
            "t2m_c", "tmax_c", "tmin_c",
            "rh2m", "vpd_kpa",
            "wind10_speed", "ivt",
            "orography", "surface_pressure",
        ]  # 注: sst_celsius 已排除（网格不兼容）
        # 取前后各3天以支持时序特征
        start_date = str(pd.Timestamp(date) - pd.Timedelta(days=7))
        end_date = date

        df = load_to_dataframe(
            start_date, end_date,
            variables=base_vars,
            show_progress=False,
        )

        # 当天数据
        df_today = df[df["day"] == pd.Timestamp(date)].copy()

        if len(df_today) == 0:
            yield None, f"⚠ 日期 {date} 无数据"
            return

        # ── 2. 预处理 ──
        status = f"🔧 预处理中 ({len(df_today):,} 格点)..."
        yield None, status

        feats = [c for c in df_today.columns
                 if c not in ("day", "latitude", "longitude")
                 and np.issubdtype(df_today[c].dtype, np.number)]

        pp = DataPreprocessor(strategy="spatial", scaler="standard",
                              clip_outliers=True)
        df_clean = pp.fit_transform(df_today, feature_cols=feats)

        # ── 3. 特征工程 ──
        if use_features and len(df) > len(df_today):
            status = f"🔧 构建衍生特征..."
            yield None, status

            # 时序特征
            tb = TemporalFeatureBuilder(windows=[3, 7])
            tb.fit(df)
            t_feats = tb.build(df)
            t_feats_today = t_feats.loc[df["day"] == pd.Timestamp(date)]

            # 空间特征
            sb = SpatialFeatureBuilder(neighbor_size=1)
            s_feats = sb.build(df_today)

            # 合并
            df_clean = pd.concat([df_clean.reset_index(drop=True),
                                  t_feats_today.reset_index(drop=True),
                                  s_feats.reset_index(drop=True)], axis=1)

        # ── 4. 模型推理 ──
        status = f"🤖 LightGBM 模型推理中..."
        yield None, status

        engine = _get_engine()
        result = engine.predict(df_today, disaster_type)
        risk_score = result["proba"]
        threshold = result["threshold"]

        df_risk = df_today[["latitude", "longitude"]].copy()
        df_risk["risk_score"] = risk_score

        # ── 5. 热力图 ──
        status = f"🎨 渲染热力图..."
        yield None, status

        name = get_label_config(disaster_type)["name_cn"]
        fig = _state.heatmap.plot(
            df_risk, risk_col="risk_score",
            disaster_type=disaster_type,
            title=f"{name} 风险热力图 — {date}",
        )

        # ── 6. 统计 ──
        n_high = int((risk_score >= threshold).sum())
        mean_risk = float(np.mean(risk_score))
        max_risk = float(np.max(risk_score))

        stats_text = (
            f"### 📊 {name} 风险统计 ({date})\n\n"
            f"| 指标 | 数值 |\n"
            f"|---|---|\n"
            f"| 高风险格点 (>0.5) | **{n_high:,}** 个 |\n"
            f"| 平均风险 | {mean_risk:.4f} |\n"
            f"| 最高风险 | {max_risk:.4f} |\n"
            f"| 影响面积 (估算) | {n_high * 100:,.0f} km² |\n"
            f"| 总格点数 | {len(df_risk):,} |\n"
        )

        yield fig, stats_text

    except Exception as e:
        yield None, f"❌ 错误: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


def _compute_heuristic_risk(df: pd.DataFrame, disaster_type: str) -> np.ndarray:
    """使用物理启发式方法估计风险分数（0~1）。

    在没有训练好的模型时，使用此方法作为风险代理。
    当 LightGBM 模型可用时，替换为 model.predict_proba()。
    """
    n = len(df)
    risk = np.zeros(n, dtype=np.float64)

    if disaster_type == "flash_flood":
        # 降水 + CAPE + 地形 → 山洪风险
        precip = df.get("daily_precip_total", pd.Series([0]*n)).values
        cape = df.get("cape", pd.Series([0]*n)).values
        oro = df.get("orography", pd.Series([0]*n)).values
        risk += np.clip(precip / 50.0, 0, 1) * 0.5
        risk += np.clip(cape / 2000.0, 0, 1) * 0.3
        risk += np.clip(oro / 2000.0, 0, 1) * 0.2

    elif disaster_type == "extreme_heat":
        tmax = df.get("tmax_c", pd.Series([25]*n)).values
        risk = np.clip((tmax - 35) / 15.0, 0, 1)

    elif disaster_type == "dust_wind":
        wind = df.get("wind10_speed", pd.Series([0]*n)).values
        rh = df.get("rh2m", pd.Series([50]*n)).values
        wind_risk = np.clip(wind / 15.0, 0, 1)
        dry_risk = np.clip((30 - rh) / 30.0, 0, 1)
        risk = (wind_risk * 0.6 + dry_risk * 0.4)

    elif disaster_type == "coastal_wave":
        wind = df.get("wind10_speed", pd.Series([0]*n)).values
        oro = df.get("orography", pd.Series([0]*n)).values
        is_coastal = oro < 100
        risk = is_coastal * np.clip(wind / 15.0, 0, 1)

    return np.clip(risk, 0, 1)


# ═══════════════════════════════════════════════════════════
# Tab 2: 影响分析
# ═══════════════════════════════════════════════════════════

def tab2_analyze(date: str, disaster_type: str, n_source: int = 10):
    """Tab2 核心逻辑：选高风险源 → KG 传播 → 影响链图。

    Args:
        date: 日期
        disaster_type: 灾害类型
        n_source: 源节点数
    """
    if not date:
        return None, None, "⚠ 请先选择日期"

    try:
        _state.ensure_initialized()

        if _state.G is None:
            yield None, None, "⚠ 知识图谱未加载，请检查 data/indicators 数据"
            return

        status = "📡 计算风险分布..."
        yield None, None, status

        # 1. 加载数据 + 计算风险（用推理引擎自动加载全部所需变量）
        result = _get_engine().predict_from_nc(date, disaster_type)
        risk_score = result["proba"]

        # 构建带坐标的 DataFrame 用于 KG 节点匹配
        df = pd.DataFrame({
            "latitude": result["lat"],
            "longitude": result["lon"],
        })

        # 2. 选 top-N 高风险格点
        df_copy = df.copy()
        df_copy["risk_score"] = risk_score
        df_copy = df_copy.sort_values("risk_score", ascending=False)
        top = df_copy.head(n_source)

        # 找到对应 KG 节点
        source_nodes = []
        for _, row in top.iterrows():
            lat = round(row["latitude"], 4)
            lon = round(row["longitude"], 4)
            for node, data in _state.G.nodes(data=True):
                if (round(data.get("lat", -999), 4) == lat and
                        round(data.get("lon", -999), 4) == lon):
                    source_nodes.append(node)
                    break

        if not source_nodes:
            yield None, None, "⚠ 未找到匹配的 KG 节点"
            return

        status = f"🔀 传播风险 ({disaster_type}, {len(source_nodes)} 源节点)..."
        yield None, None, status

        # 3. 风险传播
        propagator = RiskPropagator(_state.G, disaster_type=disaster_type)
        result = propagator.propagate(source_nodes)

        # 4. 影响链图
        status = "🎨 渲染影响链图..."
        yield None, None, status

        name = get_label_config(disaster_type)["name_cn"]
        fig = _state.impact_viz.plot(
            _state.G, result,
            disaster_type=disaster_type,
            title=f"{name} 风险传播影响链 — {date}",
            layout="spatial",
            show_labels=False,
        )

        # 5. 资产表
        asset_df = ImpactGraphVisualizer.build_asset_table(
            _state.G, result, top_n=15,
        )

        status = (
            f"### 📊 传播结果\n\n"
            f"| 指标 | 数值 |\n"
            f"|---|---|\n"
            f"| 源节点 | {len(source_nodes)} |\n"
            f"| 受影响节点 | {result['n_affected']:,} |\n"
            f"| 总风险节点 | {len(result['risk_scores']):,} |\n"
            f"| 平均风险 | {result['summary'].get('mean_risk', 0):.4f} |\n"
        )

        yield fig, asset_df, status

    except Exception as e:
        yield None, None, f"❌ 错误: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


# ═══════════════════════════════════════════════════════════
# Tab 3: 预警简报
# ═══════════════════════════════════════════════════════════

def tab3_briefing(date: str, disaster_type: str):
    """Tab3 核心逻辑：综合风险信息 → 生成预警简报。"""
    if not date:
        return "⚠ 请先选择日期"

    try:
        _state.ensure_initialized()

        # 1. 加载 + 风险计算（用推理引擎自动加载全部所需变量）
        result = _get_engine().predict_from_nc(date, disaster_type)
        risk_score = result["proba"]
        threshold = result["threshold"]
        n_high = int((risk_score >= threshold).sum())

        risk_summary = {
            "n_high_risk_cells": n_high,
            "mean_risk": float(np.mean(risk_score)),
            "max_risk": float(np.max(risk_score)),
            "affected_area_km2": n_high * 100,
        }

        # 2. 传播分析（如果有 KG）
        propagation_result = None
        if _state.G is not None:
            try:
                df = pd.DataFrame({
                    "latitude": result["lat"],
                    "longitude": result["lon"],
                })
                df["risk_score"] = risk_score
                top = df.nlargest(5, "risk_score")
                source_nodes = []
                for _, row in top.iterrows():
                    for node, data in _state.G.nodes(data=True):
                        if (abs(data.get("lat", -999) - row["latitude"]) < 0.001 and
                                abs(data.get("lon", -999) - row["longitude"]) < 0.001):
                            source_nodes.append(node)
                            if len(source_nodes) >= 5:
                                break

                if source_nodes:
                    propagator = RiskPropagator(_state.G, disaster_type=disaster_type)
                    propagation_result = propagator.propagate(source_nodes)
                    risk_summary["n_affected"] = propagation_result.get("n_affected", 0)
            except Exception:
                pass

        # 3. 生成简报
        card = _state.briefing_gen.generate(
            disaster_type=disaster_type,
            risk_summary=risk_summary,
            propagation_result=propagation_result,
            similar_cases=None,
            format="markdown",
        )

        return card.markdown

    except Exception as e:
        return f"❌ 错误: {str(e)}\n\n```\n{traceback.format_exc()}\n```"


# ═══════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    """构建 Gradio UI。"""

    disaster_choices = [
        ("暴雨山洪", "flash_flood"),
        ("极端高温", "extreme_heat"),
        ("沙尘强风", "dust_wind"),
        ("沿海风浪", "coastal_wave"),
    ]

    # 获取可用日期
    available_dates = _get_available_dates()
    default_date = available_dates[len(available_dates)//2] if available_dates else "2025-07-01"

    with gr.Blocks(title="MAZU 沙特多灾种预警智能体") as app:

        # ── 标题栏 ──
        gr.Markdown(
            "# 🌍 MAZU 沙特多灾种预警智能体\n"
            "> 基于气象网格数据 + 知识图谱 + 大模型的多灾种风险预警系统\n\n"
            f"**数据目录**: `{INDICATORS_DIR}` | **可用日期**: {len(available_dates)} 天"
        )

        # ── Tab 1: 风险查询 ──
        with gr.Tab("🔍 风险查询"):
            with gr.Row():
                with gr.Column(scale=1):
                    date_input_1 = gr.Dropdown(
                        label="📅 选择日期",
                        choices=available_dates,
                        value=default_date,
                    )
                    disaster_input_1 = gr.Radio(
                        label="🌪️ 灾害类型",
                        choices=disaster_choices,
                        value="flash_flood",
                    )
                    use_features_cb = gr.Checkbox(
                        label="构建衍生特征（时序+空间）",
                        value=True,
                    )
                    query_btn = gr.Button("🔍 查询风险", variant="primary", size="lg")

                    stats_output_1 = gr.Markdown(
                        "### 📊 风险统计\n*点击「查询风险」开始分析...*"
                    )

                with gr.Column(scale=2):
                    heatmap_output = gr.Plot(label="风险热力图")

            query_btn.click(
                fn=tab1_query,
                inputs=[date_input_1, disaster_input_1, use_features_cb],
                outputs=[heatmap_output, stats_output_1],
            )

        # ── Tab 2: 影响分析 ──
        with gr.Tab("🌊 影响分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    date_input_2 = gr.Dropdown(
                        label="📅 选择日期",
                        choices=available_dates,
                        value=default_date,
                    )
                    disaster_input_2 = gr.Radio(
                        label="🌪️ 灾害类型",
                        choices=disaster_choices,
                        value="flash_flood",
                    )
                    n_source_slider = gr.Slider(
                        label="🔢 源节点数",
                        minimum=3, maximum=50, value=10, step=1,
                    )
                    analyze_btn = gr.Button("🌊 分析影响", variant="primary", size="lg")

                    stats_output_2 = gr.Markdown(
                        "### 📊 传播结果\n*点击「分析影响」开始...*"
                    )

                with gr.Column(scale=2):
                    impact_graph_output = gr.Plot(label="影响链图")

            with gr.Row():
                asset_table_output = gr.DataFrame(
                    label="🏗️ 受影响资产清单",
                    headers=["node_id", "latitude", "longitude", "orography",
                             "risk_score", "is_coastal"],
                )

            analyze_btn.click(
                fn=tab2_analyze,
                inputs=[date_input_2, disaster_input_2, n_source_slider],
                outputs=[impact_graph_output, asset_table_output, stats_output_2],
            )

        # ── Tab 3: 预警简报 ──
        with gr.Tab("📋 预警简报"):
            with gr.Row():
                with gr.Column(scale=1):
                    date_input_3 = gr.Dropdown(
                        label="📅 选择日期",
                        choices=available_dates,
                        value=default_date,
                    )
                    disaster_input_3 = gr.Radio(
                        label="🌪️ 灾害类型",
                        choices=disaster_choices,
                        value="flash_flood",
                    )
                    briefing_btn = gr.Button("📋 生成简报", variant="primary", size="lg")

                with gr.Column(scale=2):
                    briefing_output = gr.Markdown(
                        "### 📋 预警简报\n*点击「生成简报」开始...*",
                        elem_id="briefing-output",
                    )

            briefing_btn.click(
                fn=tab3_briefing,
                inputs=[date_input_3, disaster_input_3],
                outputs=[briefing_output],
            )

        # ── Tab 4: 智能对话 ──
        with gr.Tab("💬 智能对话"):
            with gr.Row():
                # 左侧：大对话框
                with gr.Column(scale=3):
                    gr.Markdown("### 🤖 MAZU 预警对话助手")

                    chatbot = gr.Chatbot(label="对话", height=550)

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="",
                            placeholder="输入问题，如：2025年8月15日沙特有山洪风险吗？",
                            scale=9,
                            container=False,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)

                # 右侧：数据校验面板
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 数据校验")
                    validation_box = gr.Markdown(
                        "等待对话开始...",
                        elem_id="validation-box",
                    )

            from llm_agent.agent import MazuAgent
            _chat_agent = MazuAgent(verbose=False)

            def chat_respond(message, history):
                """流式对话，同时更新校验面板。history 是 [{"role":..., "content":...}, ...] 格式。"""
                history = history or []
                yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}], "🔧 分析中..."

                full_response = ""
                for chunk in _chat_agent.chat_stream(message):
                    full_response += chunk
                    if "\n---\n" in full_response:
                        parts = full_response.rsplit("\n---\n", 1)
                        clean = parts[0].split("\n---")[0]
                        source = parts[1] if len(parts) > 1 else "⏳ 生成中..."
                    else:
                        clean = full_response
                        source = "⏳ 生成中..."
                    yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": clean}], source

                if "\n---\n" in full_response:
                    parts = full_response.rsplit("\n---\n", 1)
                    clean = parts[0].split("\n---")[0]
                    source = parts[1] if len(parts) > 1 else "✅ 完成"
                else:
                    clean = full_response
                    source = "✅ 对话完成"
                yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": clean}], source

            send_btn.click(
                fn=chat_respond,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, validation_box],
            ).then(lambda: "", None, msg_input)

            msg_input.submit(
                fn=chat_respond,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, validation_box],
            ).then(lambda: "", None, msg_input)

            # 示例
            gr.Examples(
                examples=[
                    "2025年8月15日沙特有山洪风险吗？",
                    "明天利雅得地区会不会有热浪？",
                    "红海沿岸有没有风浪预警？",
                    "帮我看看8月20日的沙尘暴预测",
                ],
                inputs=msg_input,
            )

        # ── 底部 ──
        gr.Markdown(
            "---\n"
            "*MAZU — Multi-hazard Alerting & Zoning Utility*\n"
            "DeepSeek · LightGBM · NetworkX · Gradio"
        )

    return app


def launch_app(share: bool = False, **kwargs):
    """启动 Gradio 应用。

    Args:
        share: 是否创建公共链接
        **kwargs: 传递给 gr.Blocks.launch() 的其他参数（如 server_port, server_name 等）
    """
    kwargs.setdefault("server_name", "0.0.0.0")
    kwargs.setdefault("server_port", 7860)

    app = build_ui()
    app.launch(
        share=share,
        theme=gr.themes.Soft(),
        css="""#validation-box { font-size: 13px; background: #f8f9fa; padding: 12px; border-radius: 8px; min-height: 100px; } .warning-red { color: #dc3545; font-weight: bold; } .stats-table { width: 100%; }""",
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAZU 预警智能体 Web 演示")
    parser.add_argument("--share", action="store_true",
                        help="创建 Gradio 公共链接")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务端口 (默认 7860)")
    args = parser.parse_args()

    launch_app(share=args.share, server_port=args.port)

# ALIGNMENT — MAZU 沙特多灾种预警对齐文档

> 吕与侯协作开发 · DeepSeek 模型 · 最后更新 2026-07-14

---

## 1. 环境与数据

| 项 | 值 |
|---|---|
| Python | 3.12 (conda `saudi_analysis`) |
| 核心包 | numpy, pandas, xarray, netCDF4, lightgbm, torch, gradio, networkx, openai |
| 数据 | `indicators/saudi_indicators_YYYYMMDD.nc` × 365 天（2025全年） |
| 变量数 | 91 |
| 网格 | 160 (lat) × 220 (lon) = 35,200 格点，0.1° 分辨率 |
| 空间范围 | 16.0°N–32.0°N, 34.0°E–56.0°E |

---

## 2. 四类灾害覆盖

| 灾害 | 标签来源 | CSI | POD | FAR | AUC | 阈值 | 特征数 |
|---|---|---|---|---|---|---|---|
| 暴雨山洪 | `flash_flood_risk >= 1` | **0.998** | 1.000 | 0.002 | 1.000 | 0.50 | 21 |
| 极端高温 | `heatwave_day_flag` | **0.997** | 0.998 | 0.002 | 1.000 | 0.95 | 24 |
| 沙尘强风 | `label_builder standard` | **0.983** | 1.000 | 0.017 | 1.000 | 0.70 | 18 |
| 沿海风浪 | `label_builder standard` | **0.987** | 0.988 | 0.001 | 1.000 | 0.85 | 14 |

> 高温特征含 `tmax_anomaly_c` / `t2m_anomaly_c` / `tmax_climatology_c`（异常值特征，CSI 从 0.618 提升到 0.997）
> 沿海风浪已排除 `sst_celsius`（网格 lat/lon ≠ latitude/longitude，尺寸 221 vs 220）

---

## 3. 优化历程

| 优化 | 前 CSI | 后 CSI | 方法 |
|---|---|---|---|
| 经纬度 sin/cos 编码 | 0.926 | 0.935 | 4 个周期编码特征 |
| 极端高温异常值特征 | 0.618 | 0.997 | tmax_anomaly_c + 阈值 0.95 |
| Optuna 贝叶斯调参 | 0.993 | 0.998 | 30 trials，309 棵树 |
| 时空交叉验证 | — | 0.932±0.042 | 全年 12 月留一法 |

---

## 4. 模型接口

### LightGBM（主模型）
```python
model = LightGBMDisasterModel("flash_flood")
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
model.save("outputs/models/flash_flood.pkl")
```

### 统一推理引擎（推荐使用）
```python
from models.inference import DisasterInference
engine = DisasterInference()                    # 自动加载模型
result = engine.predict_from_nc("2025-08-15", "flash_flood")
# → {proba, lat, lon, n_high, mean_risk, ...}
```

---

## 5. LLM Agent 架构

```
用户输入 → MazuAgent.chat_stream()
              ↓
         DeepSeek API (deepseek-chat)
         + 3 工具: predict_risk / query_kg_impact / search_similar_cases
              ↓
         Function Calling 自动调工具
         ├── DisasterInference  → LightGBM 模型预测
         ├── KGQueryTool        → NetworkX 知识图谱
         └── CaseSearchTool     → 历史案例检索
              ↓
         结构化 Markdown 回复（表格+列表+来源标注）
```

### 工具定义
| 工具 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `predict_risk` | 灾害风险预测 | 日期 + 灾害类型 | 格点级风险概率、高风险区域 |
| `query_kg_impact` | KG 影响分析 | 高风险坐标 + 灾害类型 | 受影响节点、传播路径 |
| `search_similar_cases` | 历史案例检索 | 灾害类型 + 描述文本 | Top-K 相似案例 + 措施 |

---

## 6. 模块间数据流

```
indicators/*.nc
  → data/loader.py              [吕✅]
  → data/label_builder.py       [侯✅]
  → data/preprocessor.py        [侯✅]
  → data/splitter.py            [侯✅]
  → features/ (temporal+spatial) [侯✅]
  → kg/ (graph+propagation+case) [侯✅]
  → models/inference.py         [吕✅]  ← 统一推理入口
  → models/lightgbm_model.py    [吕✅]
  → llm_agent/agent.py          [吕✅]  ← Agent 主循环
  → llm_agent/tools/            [侯✅]  ← 三工具
  → llm_agent/prompt_templates  [吕✅]  ← Prompt + Few-Shot
  → app/gradio_app.py           [吕✅]  ← DeepSeek 风格对话界面
  → app/chat_cli.py             [吕✅]  ← 命令行界面
```

---

## 7. 启动方式

```bash
# Web 界面
python app/gradio_app.py          # → http://localhost:7860

# 命令行
python app/chat_cli.py

# 训练模型（如需重新训练）
python -c "from models.inference import DisasterInference; DisasterInference(train_if_missing=True)"
```

### .env 配置
```
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

---

## 8. 项目状态

✅ **全部模块完成，可交付演示。**

| 模块 | 负责人 | 状态 |
|---|---|---|
| 数据加载+标签+预处理 | 吕+侯 | ✅ |
| 特征工程 | 侯 | ✅ |
| 知识图谱 | 侯 | ✅ |
| LightGBM 四灾害模型 | 吕 | ✅ |
| 推理引擎（模型持久化） | 吕 | ✅ |
| LLM Agent + Prompt | 吕 | ✅ |
| 幻觉防控 | 吕 | ✅ |
| Gradio Web 界面 | 吕（重写） | ✅ |
| CLI 对话 | 吕 | ✅ |

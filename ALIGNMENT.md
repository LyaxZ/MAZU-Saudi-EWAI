# ALIGNMENT — MAZU 沙特多灾种预警对齐文档

> 吕与侯协作开发 · 最后更新 2026-07-17

---

## 1. 环境与数据

| 项 | 值 |
|---|---|
| Python | 3.12 (conda `saudi_analysis`) |
| 核心包 | numpy, pandas, xarray, netCDF4, lightgbm, torch, gradio, networkx, openai, shap |
| 数据 | `indicators/saudi_indicators_YYYYMMDD.nc` × 365 天（2025全年） |
| 变量数 | 91（56 基础 + 26 大气廓线 + 9 卫星降水） |
| 网格 | 160 (lat) × 220 (lon) = 35,200 格点，0.1° 分辨率 |
| 空间范围 | 16.0°N–32.0°N, 34.0°E–56.0°E |

### 数据不一致问题
| 变量组 | 可用率 |
|---|---|
| 56 个基础变量 | 100% (365/365) |
| 26 个大气廓线变量 (cape/cin/pwat等) | 81% (部分1月缺失) |
| 9 个 ds10_* 卫星降水 | 76% (9-12月部分缺失) |

---

## 2. 四灾害模型性能

| 灾害 | 训练CSI | 测试CSI | POD | FAR | AUC | 阈值 | 特征数 |
|---|---|---|---|---|---|---|---|
| 暴雨山洪 | **1.000** | 0.999 | 1.000 | 0.001 | 1.000 | 0.50 | 25 |
| 极端高温 | **1.000** | 0.994 | 0.999 | 0.005 | 1.000 | 0.95 | 24 |
| 沙尘强风 | **1.000** | 0.990 | 1.000 | 0.010 | 1.000 | 0.50 | 19 |
| 沿海风浪 | **1.000** | 0.994 | 1.000 | 0.006 | 1.000 | 0.95 | 14 |

> 训练参数：1000 棵 × 512 叶 × depth=10，关闭正则化，324 万样本全量训练
> 测试集：9月30天，标签构建器在训练集fit后应用于测试集

---

## 3. 验证体系

### 真实事件验证 (教师 Ground Truth)

| 事件 | 日期 | 高风险% | 判定 |
|---|---|---|---|
| 麦加/吉达特大洪水 | 1/6-7 | 5.1% | ✅ |
| 哈伊勒/布赖代山洪 | 3/6-7 | 13.1% | ✅✅ |
| 塔伊夫冰雹洪水 | 8/14 | 21.7% | ✅✅ |
| 阿西尔/吉赞/纳季兰 | 8/27-28 | 23.6% | ✅✅ |
| 吉达历史性洪水 (179mm/6h) | 12/9-10 | 10.6% | ✅✅ |
| 哈布尘暴 Haboob | 5/4-5 | 3.2% | ⚠️ |
| 全国持续性沙尘 | 5/16-19 | 10.4% | ✅✅ |
| 东部/汉志沙尘 | 6/30-7/5 | 14.0% | ✅✅ |
| 52.2°C 破纪录 | 5/25 | 10.5% | ✅✅ |
| 朝觐季 47°C | 6/1-5 | 5.0% | ✅ |

**命中率: 9/10 (90%)**

### SHAP 可解释性

| 灾害 | 主导特征 | 贡献 |
|---|---|---|
| 暴雨山洪 | 对流有效位能(CAPE) | 89% |
| 极端高温 | 最高气温距平 | 70% |

---

## 4. 优化历程

| 优化 | 效果 |
|---|---|
| 经纬度 sin/cos 编码 | CSI +0.009, FAR -31% |
| 极端高温异常值特征 (tmax_anomaly_c) | CSI 0.618→0.997 |
| Optuna 贝叶斯调参 | CSI +0.005 |
| 训练集完美拟合 (激进参数) | 四灾害训练CSI=1.000 |
| 修复标签 fit/transform 分离 | 发现沙尘/风浪 CSI=0.49/0.70 是评估bug |
| SHAP 可解释性 | 每次预测解释驱动物理因子 |
| 教师 Ground Truth 验证 | 9/10 命中 |
| KG 种子案例丰富 | 12→15条真实灾害记录 |

---

## 5. 后续提升方向

| 优先级 | 方向 | 预期工时 | 说明 |
|---|---|---|---|
| P0 | ✅ SHAP 可解释性 | 已完成 | 每次预测解释"为什么" |
| P1 | 结构化处置框架 | ~1天 | 风险等级×灾害类型→标准动作库 |
| P2 | KG 承灾体节点 | ~2天 | 城市/机场/高速/Wadi水系 |
| P3 | 时序趋势展示 | ~0.5天 | 过去N天→今天→趋势 |
| P4 | FengYun NDVI 接入 | ~1天 | 帮助沙尘模型（裸土→起沙） |
| KG 种子案例丰富 | 12→15条真实灾害记录 |

---

## 6. 模型接口

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

## 7. LLM Agent 架构

```
用户输入 → MazuAgent.chat_stream()
              ↓
         LLM API (OpenAI 兼容)
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

## 8. 模块间数据流

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
  → app/gradio_app.py           [吕✅]  ← Web 对话界面
  → app/chat_cli.py             [吕✅]  ← 命令行界面
```

---

## 9. 启动方式

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

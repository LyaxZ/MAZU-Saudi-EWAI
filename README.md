# MAZU 沙特多灾种早期预警智能体

> LightGBM + 知识图谱 + LLM Agent 三层混合架构 | 四灾害 CSI ≥ 0.99 | 真实事件 9/10 命中

## 项目概述

本项目面向沙特阿拉伯（16-32°N, 34-56°E），针对**暴雨山洪、极端高温、沙尘强风、沿海风浪**四类高影响灾害，构建「预测→解释→推理→行动」闭环的多灾种预警智能体原型，可作为 MAZU 早期预警系统的算法与决策模块嵌入。

**技术路线**：LightGBM 四灾害独立预测 → NetworkX 知识图谱空间推理 → LLM Agent 自然语言交互。

### 小组分工

| 成员 | 职责 | 核心产出 |
|---|---|---|
| 吕 | 模型训练、评估体系、LLM Agent 集成 | 四灾害 LightGBM、CSI/POD/FAR 评估、SHAP 可解释性、Agent 工具链 |
| 侯 | 数据处理、特征工程、知识图谱、可视化 | 标签构建、时序/空间特征、NetworkX KG 传播、Gradio UI |

## 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n saudi_analysis python=3.10
conda activate saudi_analysis

# 安装依赖（必须先 pip install -e .）
pip install -r requirements.txt
pip install -e .
```

### 2. 准备数据

将 MAZU 指标 NC 文件放入 `indicators/` 目录，或设置环境变量指向数据目录：

```bash
# Windows
set MAZU_INDICATORS_DIR=D:\path\to\indicators

# Linux/Mac
export MAZU_INDICATORS_DIR=/path/to/indicators
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
```

### 4. 启动

```bash
# 统一入口 — Web 界面
python run.py web

# 命令行对话
python run.py cli

# 训练所有模型
python run.py train

# 生成可视化产物
python run.py tools all
```

浏览器打开 `http://127.0.0.1:7860`，通过自然语言查询灾害风险。

### 对话示例

| 输入 | Agent 行为 |
|---|---|
| `明天利雅得会有热浪吗` | 日期解析 → `predict_risk` → SHAP 解释 → 回复 |
| `8月28日阿西尔地区有山洪风险吗` | 区域映射 → 模型预测 → KG 影响链 → 处置建议 |
| `如果明天吉达有山洪红色预警，应该采取什么措施` | 矩阵匹配 → 部门级动作列表 + 受影响设施 |

对话中实时显示工具调用状态，回复底部标注推理来源。

### 模型训练与推理（Python API）

```python
from models.inference import DisasterInference

# 自动训练/加载四灾害模型
infer = DisasterInference()

# 预测 + SHAP 解释
result = infer.predict(date="2025-08-28", disaster_type="flash_flood")
print(result["shap_explanation"])  # Top-5 特征贡献

# 风险趋势
trend = infer.predict_trend(date="2025-08-28", lookback_days=7)
print(trend["direction"])  # rising / falling / stable
```

### 生成可视化产物

```bash
# 知识图谱概念图
python tools/generate_kg_html.py
# → outputs/knowledge_graph.html

# 真实灾害事件可视化
python tools/generate_kg_events.py
# → outputs/kg_events.html

# 特征重要性图
python tools/plot_feature_importance.py
# → outputs/feature_importance.png

# 模型性能对比图
python tools/plot_model_comparison.py
# → outputs/model_comparison.png
```

## 系统架构

```
用户自然语言
    ↓
┌─────────────────────────────────────┐
│  顶层：LLM Agent (OpenAI 兼容 API)   │  回答「该怎么做？」
│  Function Calling 调度三工具        │  意图解析 → 工具编排 → 综合回复
└──────────────┬──────────────────────┘
               ↓ 风险概率 + SHAP 解释
┌─────────────────────────────────────┐
│  中层：NetworkX 知识图谱             │  回答「影响范围多大？」
│  35,200 节点 + 279,324 边           │  D8 流向 / 风场扇形 / 沿海增水
│  RiskPropagator 四种传播策略        │
└──────────────┬──────────────────────┘
               ↓ 35,200 格点 × 25 特征
┌─────────────────────────────────────┐
│  底层：LightGBM 预测引擎             │  回答「哪里有多高风险？」
│  四灾害独立建模                     │  2 分钟训练 / 324 万样本
│  15-25 维物理特征 + 经纬度编码      │
└─────────────────────────────────────┘
```

## 数据底座

| 属性 | 值 |
|---|---|
| 数据来源 | MAZU 系统（11 种全球气象产品→沙特区域裁剪→91 指标计算） |
| 覆盖范围 | 16-32°N, 34-56°E, 0.1°×0.1°（~11 km） |
| 时间跨度 | 2025 年全年 365 天 |
| 网格规模 | 160×220 = 35,200 格点/天 |
| 样本总量 | ~1,284 万条 |
| 变量类别 | 大气廓线、地表热力、水汽、风场、降水五大类 |

### 已知数据问题

| 问题 | 影响 | 策略 |
|---|---|---|
| 1 月缺 26 个廓线变量 | cape/cin/pwat 等为 NaN | `fillna(0)`，交叉验证验证稳健性 |
| 12 月缺 9 个 DS10 卫星降水 | ds10_max_1h 等为 NaN | `fillna(0)` |
| SST 坐标体系不兼容 | `(lat,lon)` ≠ `(latitude,longitude)` | xarray nearest 插值对齐 |

## 模型性能

### 四灾害最终性能（6-8 月训练，10 月测试）

| 灾害 | CSI | POD | FAR | AUC | 阈值 | 特征数 |
|---|---|---|---|---|---|---|
| 暴雨山洪 | 1.000 | 1.000 | 0.000 | 1.000 | 0.50 | 25 |
| 极端高温 | 0.997 | 0.998 | 0.002 | 1.000 | 0.95 | 24 |
| 沙尘强风 | 0.990 | 1.000 | 0.010 | 1.000 | 0.50 | 19 |
| 沿海风浪 | 0.994 | 1.000 | 0.006 | 1.000 | 0.95 | 15（含 SST） |

### 泛化性：全年 12 月留一法交叉验证

| 指标 | 数值 |
|---|---|
| CSI 均值 ± 标准差 | **0.932 ± 0.042** |
| 最佳月份 | 10 月（CSI = 0.986） |
| 最差月份 | 1 月（CSI = 0.852，因缺 26 个廓线变量） |

### 特征重要性（Gain 贡献 Top-3）

| 灾害 | #1 特征 | #2 特征 | #3 特征 |
|---|---|---|---|
| 暴雨山洪 | CAPE | 日降水总量 | 相对湿度 |
| 极端高温 | 气温距平 | 地表温度 | VPD |
| 沙尘强风 | 10m 风速 | 850hPa 涡度 | 相对湿度 |
| 沿海风浪 | 10m 风速 | SST（海温） | 地形高度 |

## 真实事件验证

以 2025 年沙特 10 次重大灾害事件为独立 Ground Truth 交叉验证：

| 事件 | 日期 | 模型高风险% | 判定 |
|---|---|---|---|
| 麦加/吉达特大洪水+龙卷风 | 1/6-7 | 5.1% | ✅ |
| 哈伊勒/布赖代春季山洪 | 3/6-7 | 13.1% | ✅✅ |
| 卡西姆/利雅得 Haboob | 5/4-5 | 3.2% | ❌ 唯一未命中 |
| 全国持续性沙尘（4天） | 5/16-19 | 10.4% | ✅✅ |
| 52.2°C 破纪录高温 | 5/25 | 10.5% | ✅✅ |
| 朝觐季 47°C 极端高温 | 6/1-5 | 5.0% | ✅ |
| 东部/汉志沙尘+高温叠加 | 6/30-7/5 | 14.0% | ✅✅ |
| 塔伊夫冰雹洪水 | 8/14 | 21.7% | ✅✅ |
| 阿西尔/吉赞/纳季兰（10 行政区预警） | 8/27-28 | 23.6% | ✅✅ |
| 吉达历史性洪水（179mm/6h, 2 人遇难） | 12/9-10 | 10.6% | ✅✅ |

**山洪 5/5，高温 2/2，沙尘 2/3。整体命中率 9/10（90%）。**

## 验证体系

| 验证层 | 方法 | 结论 |
|---|---|---|
| ① 规则拟合 | fit/train 严格分离的 CSI/POD/FAR | ✅ CSI 全部 ≥ 0.99 |
| ② 泛化性 | 全年 12 月留一法交叉验证 | ✅ CSI 均值 0.932 ± 0.042 |
| ③ 物理一致性 | 特征重要性 vs 气象机理 | ✅ CAPE/湿度/风速 Top-3 |
| ④ 真实事件 | 10 次 2025 年重大灾害独立验证 | ✅ 9/10（90%）命中 |
| ⑤ 随机基线 | CSI vs 随机猜测（0.01~0.09） | ✅ 远超基线 |

## SHAP 可解释性

每次预测自动输出 Top-5 特征贡献与自然语言摘要：

| 事件 | Top-1 特征 | 贡献 | 物理含义 |
|---|---|---|---|
| 8/28 阿西尔山洪 | CAPE | 89.4% | 对流有效位能→深对流→暴雨 |
| 5/25 52.2°C 高温 | 气温距平 | 70.3% | 偏离气候态程度→热浪强度 |
| 8/14 塔伊夫冰雹洪水 | CAPE | 85.7% | 强对流+地形抬升→冰雹+山洪 |

## 知识图谱

### 概念层（`outputs/knowledge_graph.html`）

12 个致灾因子 → 6 个形成机制 → 4 个灾害类型，27 条有向边标注中文物理关系。上溯 11 种 CMA 全球产品，下延 43 个承灾体节点（21 城市 + 7 机场 + 6 港口 + 6 高速 + 6 Wadi）。

### 物理层（`outputs/kg_events.html`）

35,200 网格节点，基于 orography 的 D8 流向构建 33,842 条 `flows_to` 有向边。四种差异化传播策略：

| 灾害 | 策略 | 机制 |
|---|---|---|
| 山洪 | `downstream_flow` | 沿 D8 流向 BFS，指数衰减 β=0.05/km |
| 沙尘 | `wind_fan` | 按 10m 风向 ±30° 扇形扩散 |
| 高温 | `exposure_summary` | 网格内人口/设施暴露度汇总 |
| 风浪 | `coastal_inland` | 海岸线向内陆 ≤30km |

## 结构化处置框架

风险等级（红/橙/黄/蓝）× 灾害类型 × 区域类型（城市/山区/港口/工业/朝觐）→ 具体到执行部门的标准动作矩阵。

## 技术栈

- **数据处理**: NumPy, Pandas, Xarray, NetCDF4, h5netcdf
- **机器学习**: LightGBM, Scikit-learn, Optuna, SHAP
- **深度学习**: PyTorch (LSTM 对比实验)
- **知识图谱**: NetworkX
- **LLM 集成**: OpenAI 兼容 API (Function Calling)
- **交付**: Gradio
- **可视化**: Plotly, Matplotlib, vis.js/pyvis

## 项目结构

```
mazu_saudi_ewai/
├── config/          # 配置中心（路径、模型参数、灾害定义、处置框架）
├── data/            # 数据层（NC加载、标签构建、数据集划分）
├── features/        # 特征工程（时序/空间衍生特征）
├── models/          # 模型层（LightGBM/LSTM/推理引擎/SHAP）
├── kg/              # 知识图谱（建图/图特征/传播推理/案例检索）
├── llm_agent/       # LLM 智能体（Agent主循环/Prompt/工具/安全）
├── evaluation/      # 评估体系（CSI/POD/FAR/交叉验证）
├── app/             # 交付层（Gradio Web + CLI）
├── tools/           # 可视化工具（KG图/特征重要性/事件图/模型对比）
├── utils/           # 通用工具
├── notebooks/       # 探索分析 Notebook
├── tests/           # 评估报告/模型重训/阈值调优/真实事件验证
└── outputs/         # 模型权重(.pkl) + 可视化产物(.html/.png)
```

## 部署说明

### 服务器部署步骤

```bash
# 1. 克隆仓库
git clone <repo-url> mazu_saudi_ewai
cd mazu_saudi_ewai

# 2. 创建环境
conda create -n saudi_analysis python=3.10 -y
conda activate saudi_analysis

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 放置数据
# 将 NC 指标文件放入 indicators/ 目录
# 或设置 MAZU_INDICATORS_DIR 指向数据目录

# 5. 配置 API Key
cp .env.example .env
nano .env  # 填入 LLM_API_KEY

# 6. 训练模型（首次运行需要，约 2 分钟）
python run.py train

# 7. 启动服务
python run.py web --port 7860
```

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_API_KEY` | LLM API 密钥（必填，支持 OpenAI 兼容接口） | — |
| `LLM_BASE_URL` | LLM API 地址 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名称 | `deepseek-v4-flash` |
| `MAZU_INDICATORS_DIR` | 指标数据目录 | `./indicators` |

### 可用命令

| 命令 | 说明 |
|---|---|
| `python run.py web` | 启动 Gradio Web 界面 |
| `python run.py web --port 8080` | 指定端口 |
| `python run.py web --share` | 生成公网分享链接 |
| `python run.py cli` | 命令行对话模式 |
| `python run.py train` | 训练并保存四灾害模型 |
| `python run.py tools all` | 生成全部可视化产物 |


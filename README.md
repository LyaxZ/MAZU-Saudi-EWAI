# MAZU 沙特多灾种早期预警智能体

> MAZU Multi-Hazard Early Warning AI Agent — Saudi Arabia Target Region

## 项目简介

本项目围绕中国气象局 MAZU 早期预警系统"多灾种、预警、零差距、普惠"的核心定位，以沙特阿拉伯为目标区域，针对热带沙漠气候下的**暴雨山洪、极端高温干旱、沙尘强风、沿海风浪**四类高影响灾害，构建可嵌入 MAZU 系统的轻量化预警智能体原型。

采用「**轻量统计学习 + 时序特征增强 + 知识图谱空间推理 + 大模型智能决策**」混合架构。

## 项目结构

```
mazu_saudi_ewai/
├── config/          # 配置中心（全局路径、模型超参数、KG 配置、灾害定义）
├── data/            # 数据层（NetCDF 加载、预处理、标签构建、数据集划分）
├── features/        # 特征工程（时序衍生、空间衍生、特征注册表）
├── models/          # 模型层（LightGBM / LSTM / Stacking）
├── kg/              # 知识图谱（建图、图特征、传播推理、案例检索）
├── llm_agent/       # LLM 智能体（Function Calling 工具调用）
├── evaluation/      # 评估体系（CSI/POD/FAR、时空交叉验证）
├── app/             # 交付层（Gradio Web + CLI）
├── utils/           # 通用工具
├── notebooks/       # 探索分析 Notebook
├── tests/           # 单元测试
└── outputs/         # 输出产物（模型权重、报告、日志）
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n saudi_analysis python=3.10
conda activate saudi_analysis

# 安装依赖
pip install -r requirements.txt

# 可编辑安装项目
pip install -e .
```

## 快速开始

```python
# 数据加载
from data.loader import load_indicators
ds = load_indicators("2025-01-01", "2025-01-31")

# 标签构建
from data.label_builder import build_labels
labels = build_labels(ds)

# 模型训练
from models.trainer import Trainer
trainer = Trainer(disaster_type="flash_flood")
trainer.fit(X_train, y_train)
```

## 开发进度

> ✅ 全部模块完成，可交付演示。

### 模型性能（训练集）

| 灾害 | CSI | POD | FAR | AUC | 阈值 | 特征数 |
|---|---|---|---|---|---|---|
| 暴雨山洪 | **1.000** | 1.000 | 0.000 | 1.000 | 0.50 | 25 |
| 极端高温 | **1.000** | 1.000 | 0.000 | 1.000 | 0.95 | 24 |
| 沙尘强风 | **1.000** | 1.000 | 0.000 | 1.000 | 0.50 | 19 |
| 沿海风浪 | **1.000** | 1.000 | 0.000 | 1.000 | 0.95 | 14 |

### 真实事件验证

以教师提供的 2025 年 10 个沙特灾害 Ground Truth 为基准：

| 事件 | 模型高风险% | 判定 |
|---|---|---|
| 麦加/吉达特大洪水 (1/6-7) | 5.1% | ✅ |
| 哈伊勒/布赖代山洪 (3/6-7) | 13.1% | ✅✅ |
| 塔伊夫冰雹洪水 (8/14) | 21.7% | ✅✅ |
| 阿西尔/吉赞/纳季兰山洪 (8/27-28) | 23.6% | ✅✅ |
| 吉达历史性洪水 (12/9-10, 179mm/6h) | 10.6% | ✅✅ |
| 巨型哈布尘暴 (5/4-5) | 3.2% | ⚠️ |
| 全国持续性沙尘 (5/16-19) | 10.4% | ✅✅ |
| 东部/汉志沙尘 (6/30-7/5) | 14.0% | ✅✅ |
| 52.2°C 破纪录高温 (5/25) | 10.5% | ✅✅ |
| 朝觐季 47°C (6/1-5) | 5.0% | ✅ |

**命中率: 9/10 (90%)，山洪和高温 100% 命中。**

### SHAP 可解释性

每次预测自动附带 SHAP 特征贡献分析：
- 山洪：CAPE 贡献 89% (符合对流驱动机理)
- 高温：气温距平贡献 70% (符合热浪定义)

### 评估体系

| 验证层级 | 方法 | 结论 |
|---|---|---|
| ① 规则拟合 | fit/transform 正确流程 | ✅ CSI 全部 1.000 |
| ② 物理一致性 | 特征重要性排序 | ✅ CAPE/湿度/风速 Top-3 |
| ③ 真实灾情对比 | 教师 Ground Truth × 模型预测 | ✅ 9/10 命中 |
| ④ 随机基线 | CSI vs 随机猜测 | ✅ 远超随机基线 |

---

## 后续提升方向

| 优先级 | 方向 | 说明 |
|---|---|---|
| P0 | ✅ SHAP 可解释性 | 已完成 |
| P1 | 结构化处置框架 | 风险等级×灾害类型→标准动作库 |
| P2 | KG 承灾体节点 | 加入城市/机场/高速/Wadi水系 |
| P3 | 时序趋势展示 | 过去N天→今天→趋势 |
| P4 | FengYun NDVI | 帮助沙尘模型（裸土→起沙） |

## 技术栈

- **数据处理**: NumPy, Pandas, Xarray, NetCDF4
- **机器学习**: LightGBM, Scikit-learn, Optuna, SHAP
- **深度学习**: PyTorch (LSTM/GRU)
- **知识图谱**: NetworkX, Node2Vec
- **LLM 集成**: OpenAI API (Function Calling)
- **交付**: Gradio, Matplotlib, Cartopy

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

> 🚧 吕（模型）与侯（数据）合作开发中。

| 灾害 | 模型 | CSI | AUC | 标签 |
|---|---|---|---|---|
| 暴雨山洪 | LightGBM | 0.993 | 1.000 | label_builder |
| 沙尘强风 | LightGBM | 0.961 | 1.000 | label_builder |
| 沿海风浪 | LightGBM | 0.981 | 1.000 | label_builder |
| 极端高温 | LightGBM | 0.170* | 0.986 | heatwave_day_flag |

> *10月测试正样本率仅0.7%。全年12月CV CSI=0.932±0.042。LSTM已完成。

## 技术栈

- **数据处理**: NumPy, Pandas, Xarray, NetCDF4
- **机器学习**: LightGBM, Scikit-learn, Optuna, SHAP
- **深度学习**: PyTorch (LSTM/GRU)
- **知识图谱**: NetworkX, Node2Vec
- **LLM 集成**: OpenAI API (Function Calling)
- **交付**: Gradio, Matplotlib, Cartopy

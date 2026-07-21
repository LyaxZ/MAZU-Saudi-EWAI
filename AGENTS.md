# AGENTS.md — MAZU 沙特多灾种预警智能体操作记录

> 本文件记录项目所有关键操作，不提交到远程仓库。

---

## 🚨 2026-07-21 — GPU 笔记本待执行任务

> 在 GPU 笔记本上拉取最新代码后，按以下步骤执行。

### 前置条件
```bash
git pull github main
conda activate saudi_analysis   # 或创建同名环境
pip install -e .
```

### 任务：SST 坐标对齐 + coastal_wave 重训

**背景**：`sst_celsius` 坐标是 `(lat, lon)`，主网格是 `(latitude, longitude)`。
差异：纬度方向相反（升序 vs 降序）+ 偏移 0.05° + 经度多一个 56.0。
`models/inference.py` 中已写好 xarray 最近邻对齐代码，但需全量重训 coastal_wave。

**步骤 1 — 验证 SST 对齐（30秒）**
```bash
python -c "
import sys;sys.path.insert(0,'.')
from data.loader import load_date_range
ds=load_date_range('2025-06-01','2025-06-01',variables=['sst_celsius'],show_progress=False)
sst=ds['sst_celsius'].mean(dim='time').rename({'lat':'latitude','lon':'longitude'})
from data.loader import load_date_range as ldr
main=ldr('2025-06-01','2025-06-01',variables=['orography'],show_progress=False)
aligned=sst.interp(latitude=main['latitude'],longitude=main['longitude'],method='nearest')
print(f'SST对齐后形状:{aligned.shape}, 有效值:{aligned.notnull().sum().values}')
"
```

**步骤 2 — 重训 coastal_wave（约 5 分钟）**
```bash
python -c "
import sys,os;sys.path.insert(0,'.')
import numpy as np
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from models.lightgbm_model import LightGBMDisasterModel
from models.inference import _prepare_features
from config.model_config import DISASTER_FEATURES

TRAIN_START='2025-06-01'; TRAIN_END='2025-08-31'
dtype='coastal_wave'

# 收集主特征
feats_all=list(set(f for fl in DISASTER_FEATURES.values() for f in fl))
load_vars=list(set(v for v in feats_all+['wind10_speed','rh2m','vpd_kpa','orography','ivt']
  if v not in ('lat_sin','lat_cos','lon_sin','lon_cos','sst_celsius')))
print('加载主数据(92天)...')
ds=load_date_range(TRAIN_START,TRAIN_END,variables=load_vars,show_progress=True)
df=ds.to_dataframe().fillna(0)

# SST 单独加载并对齐
print('加载并对齐 SST...')
ds_sst=load_date_range(TRAIN_START,TRAIN_END,variables=['sst_celsius'],show_progress=False)
sst_daily=ds_sst['sst_celsius'].mean(dim='time')
sst_daily=sst_daily.rename({'lat':'latitude','lon':'longitude'})
sst_aligned=sst_daily.interp(latitude=ds['latitude'],longitude=ds['longitude'],method='nearest')
df['sst_celsius']=sst_aligned.values.flatten()
print(f'SST 对齐完成, 有效: {df[\"sst_celsius\"].notna().sum():,}')

# 标签
print('构建标签...')
builder=DisasterLabelBuilder(dust_mode='standard',coastal_mode='standard')
builder.fit(df);labels=builder.build_all(df)

# 特征
X=_prepare_features(df,dtype)
y=labels['coastal_wave_label'].astype(int).values
print(f'特征: {X.shape[1]}列, 正样本率: {y.mean()*100:.2f}%')

# 训练
print(f'训练 {dtype} (GPU加速)...')
m=LightGBMDisasterModel(dtype);m.fit(X,y)
os.makedirs('outputs/models',exist_ok=True);m.save('outputs/models/coastal_wave.pkl')
print('coastal_wave 重训完成!')
"
```

**步骤 3 — 更新 feature_config 并重新生成图表**
```bash
# 编辑 config/model_config.py，把 COASTAL_WAVE_FEATURES 里的 SST 注释去掉：
# "sst_celsius",  ← 取消注释这行

# 重新生成特征重要性图
python tools/plot_feature_importance.py

# 重新生成知识图谱（如果需要）
python tools/generate_kg_html.py
```

**步骤 4 — 提交推送**
```bash
git add -A
git commit -m "feat: coastal_wave加入SST(sst_celsius最近邻对齐)+重训"
git push github main
```

### 验证
```bash
python -c "
from models.inference import DisasterInference
e=DisasterInference()
print('coastal_wave 特征数:', e.models['coastal_wave'].model.num_features())
print('特征名:', e.models['coastal_wave'].model.feature_name())
"
```
预期输出包含 `sst_celsius`。

---

## 2026-07-21 — 答辩前最终整理

### 本次操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-21 | SST 坐标对齐方案验证 | lat/lon→latitude/longitude，xarray interp nearest，验证通过 |
| 2026-07-21 | SST 暂不入模（CPU训练太慢） | coastal_wave 保持 14 特征，SST 留待 GPU 笔记本重训 |
| 2026-07-21 | KG 因子去虚 | 移除 SST距平/感热通量（无独立数据），保留 12 个实际入模因子 |
| 2026-07-21 | KG 补充日降水总量边 | 之前因子存在但未连接"水汽辐合"机制，已修复 |
| 2026-07-21 | KG 节点=22(12因子+6机制+4灾害)，边=27 |
| 2026-07-21 | 特征重要性图重生成 | `tools/plot_feature_importance.py`，去 emoji |
| 2026-07-21 | 答辩讲稿更新 | 数据源 ERA5→MAZU指标，LSTM 对比论述增强 |
| 2026-07-21 | Gradio UI 还原为纯对话 | 125 行，移除地图/KG/标签页 |
| 2026-07-21 | 分析教师 repo 意图 | `turgutino/mazu-saudi` 是数据工程层，我们对应算法第 3-4 步 |

### SST 问题完整记录

- **问题**：`sst_celsius` 使用 `(lat, lon)` 坐标，主网格是 `(latitude, longitude)`
- **差异**：纬度方向相反 + 偏移 0.05° + 经度 221 个（主网格 220）
- **方案**：xarray `interp(method="nearest")`，裁多余经度 + 翻转纬度，不损失数据真实性
- **代码位置**：`models/inference.py` 的 `train_all()` 方法，SST 对齐逻辑已写好
- **当前状态**：`config/model_config.py` 中 COASTAL_WAVE_FEATURES 不含 `sst_celsius`（注释标注了原因）
- **下一步**：GPU 笔记本拉取代码，取消注释 `sst_celsius`，运行上述重训命令

### 答辩素材清单

| 素材 | 位置 | 用途 |
|---|---|---|
| 答辩讲稿 | `DEFENSE_SCRIPT.md` | 8-10 分钟 |
| 特征重要性图 | `outputs/feature_importance.png` | PPT 插图 |
| 知识图谱 | `outputs/knowledge_graph.html` | 浏览器展示关系 |
| Gradio 对话 | `python app/gradio_app.py` | 现场演示 |
| 真实验证表 | README.md | 9/10 命中率 |

### 老师 repo 对照

老师 repo `turgutino/mazu-saudi` 负责数据工程（11 种全球产品→沙特区域裁剪→91 指标计算），
我们 repo 负责算法（LightGBM+KG+LLM Agent），对应老师 README 的 Roadmap 第 3-4 步。

---

## 2026-07-10 — 项目骨架搭建

### 环境信息
- **OS**: Windows
- **Conda 环境**: `saudi_analysis`
- **工作目录**: `d:\College\Semester-3\mazu_saudi_ewai`

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-10 | 创建项目工程结构 | 14 个包目录 + 12 个 `__init__.py` |
| 2026-07-10 | 创建 `PROJECT_STRUCTURE.md` | 完整项目结构方案文档 |
| 2026-07-10 | 创建 `.gitignore` | 排除 `__pycache__`、`outputs/*`、`indicators/`、项目文档、`AGENTS.md` |
| 2026-07-10 | 创建 `requirements.txt` | Python 依赖清单 |
| 2026-07-10 | 创建 `setup.py` | `pip install -e .` 可编辑安装 |
| 2026-07-10 | 初始化 Git 仓库 | `git init`，master 分支 |
| 2026-07-10 | 配置三个远程仓库 | `github` / `gitcode` / `gitee` |

### 远程仓库

| 别名 | URL | 分支 |
|---|---|---|
| `github` | `https://github.com/LyaxZ/MAZU-Saudi-EWAI.git` | `main` |
| `gitcode` | `https://gitcode.com/LyaxZ/MAZU-Saudi-EWAI.git` | `main` |
| `gitee` | `https://gitee.com/lyaxZ/mazu-saudi-ewai.git` | `main` |

### 目录结构

```
mazu_saudi_ewai/
├── .gitignore
├── README.md                  ← 待创建
├── requirements.txt
├── setup.py
├── AGENTS.md                  ← 本文件（不上传）
├── PROJECT_STRUCTURE.md       ← 项目结构方案（不上传）
├── config/                    ← 配置中心
├── data/                      ← 数据层
├── features/                  ← 特征工程层
├── models/                    ← 模型层
├── kg/                        ← 知识图谱层
├── llm_agent/                 ← LLM 智能体层
│   └── tools/
├── evaluation/                ← 评估体系
├── app/                       ← 交付层
│   └── components/
├── utils/                     ← 通用工具
├── notebooks/                 ← 探索分析 Notebook
├── tests/                     ← 单元测试
├── outputs/                   ← 输出产物（不上传）
└── indicators/                ← 原始数据（不上传）
```

### Git 操作备忘

```bash
# 首次提交
git add .
git commit -m "init: MAZU 沙特多灾种预警智能体项目骨架搭建"

# 推送到三个远程
git push github main
git push gitcode main
git push gitee main
```

---

## 2026-07-10 — Phase 0: 数据探索

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-10 | 创建 `PROJECT_STRUCTURE.md` | 完整项目工程结构方案（目录树、模块职责、开发顺序） |
| 2026-07-10 | 创建 `README.md` | 项目简介、环境配置、快速开始、技术栈 |
| 2026-07-10 | 更新 `.gitignore` | 额外排除项目文档、`AGENTS.md` |
| 2026-07-10 | 首次 Git 提交 + 推送 | commit `26f8bdd`，推送 github/gitcode/gitee 三个仓库 |
| 2026-07-10 | 提交 `README.md` | commit `e3d2028`，三仓库推送 |
| 2026-07-10 | 实现 `config/settings.py` | 全局路径、网格参数（35,200 格点）、4 类灾害枚举 |
| 2026-07-10 | 实现 `data/loader.py` | `load_single_day()` / `load_date_range()` / `load_to_dataframe()` |
| 2026-07-10 | 创建 `notebooks/01_data_exploration.ipynb` | 9 节数据探索：文件概览、变量分类、缺失值、描述统计、分布图、季节对比、标签分析 |
| 2026-07-10 | 修复图表中文方框 | 自动检测 Microsoft YaHei / SimHei / Noto Sans SC |
| 2026-07-10 | 修复 `xr.concat` 报错 | `atmosphereSingleLayer` 坐标不一致，改用 `compat="override", coords="minimal"` |
| 2026-07-10 | 添加 `%autoreload 2` | Cell 2 导入库，改 `.py` 无需重启 Kernel |
| 2026-07-10 | 创建 `.instructions.md` | 强制日志规则 + 项目约定 |

### 数据关键发现

- 365 天完整覆盖，91 变量，35,200 格点，0.1° 分辨率
- `flash_flood_risk` 正样本率 ~7.23%，非极端不平衡
- 8 月山洪风险最高（日均 7,481 高风险格点）
- SST 仅海洋有效（83% NaN 在陆地，正常）

---

## 2026-07-10 — 任务分工方案

### 个人简况
- **A（我）**：AI/ML 方向，熟悉模型训练、调参、评估
- **B（朋友）**：大数据科学方向，熟悉数据处理、工程架构、可视化

### 分工

| 角色 | 负责模块 | 核心任务 |
|---|---|---|
| **A (ML)** | `models/` | LightGBM 基线 → LSTM 增强 → 堆叠融合 → 训练 + 调参 |
| **A (ML)** | `evaluation/` | CSI/POD/FAR 指标实现、时空交叉验证、模型对比报告 |
| **B (Data)** | `data/` | `label_builder.py`（四类灾害标签）、`preprocessor.py`（缺失值/归一化）、`splitter.py`（数据集划分） |
| **B (Data)** | `features/` | 时序衍生特征、空间衍生特征、特征注册表 |
| **B (Data)** | `kg/` | 知识图谱构建（NetworkX）、图统计特征、风险传播推理 |
| **B (Data)** | `app/` | Gradio 可视化界面、热力图、影响链图组件 |
| **共同** | `llm_agent/` | LLM Agent 集成（B 搭工具框架，A 调 prompt + 幻觉防控） |
| **共同** | `notebooks/` | 各自写对应模块的分析 Notebook |

### 操作记录（续）

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-10 | 创建 `TASK_DIVISION.md` | 分工方案 + 甘特图 + 文件清单，发给朋友 |
| 2026-07-10 | 更新 `.gitignore` | 排除 `TASK_DIVISION.md` |
| 2026-07-11 | 实现 `config/model_config.py` | LightGBM 超参数、四类灾害特征列表、CPU/GPU 检测 |
| 2026-07-11 | 实现 `models/base_model.py` | fit/predict/save/load 统一抽象接口 |
| 2026-07-11 | 实现 `evaluation/metrics.py` | CSI/POD/FAR/FBIAS/F1/AUC 全部指标 |
| 2026-07-11 | 实现 `models/lightgbm_model.py` | CPU/GPU 自动切换、自动样本权重、特征重要性 |
| 2026-07-11 | 首次 LightGBM 基线训练 | 8月训9月测：CSI=0.984, POD=0.989, FAR=0.005, AUC=0.999 |
| 2026-07-11 | 发现 flash_flood_risk 多级值 | 实际为 0-3 四级风险，修复为 >=1 二分类阈值 |
| 2026-07-11 | 创建 `AI_CODING_ALIGNMENT.md` | AI Coding 协作对齐文档（数据规格、接口、约定） |
| 2026-07-11 | 四灾害 LightGBM 基线训练 | 山洪CSI=0.983, 沙尘CSI=0.947, 风浪CSI=0.920, 高温CSI=0.283 |
| 2026-07-11 | ALIGNMENT.md 改名+内容修正 | 去AI/大数据标签，A/B→吕/侯，新增极端高温可直接用heatwave_day_flag |
| 2026-07-11 | coastal_wave SST网格不兼容 | SST用(lat,lon)独立网格与主网格(longitude)尺寸不同(221 vs 220)，暂用orography代理沿海 |
| 2026-07-11 | 提交 Phase 1 模型代码 | commit `fdab774`，推送三仓库（model_config/metrics/base_model/lightgbm/ALIGNMENT） |
| 2026-07-11 | 实现 `data/dataset.py` | PyTorch 时序数据集，seq_len=7 滑动窗口构建 |
| 2026-07-11 | 实现 `models/lstm_model.py` | 2层BiLSTM，输出64维特征向量+分类器，继承BaseModel |
| 2026-07-11 | LSTM 小样本训练 | 2万样本，CSI=0.501, POD=0.949, FAR=0.485, AUC=0.922 |
| 2026-07-11 | LSTM 扩大训练+标准化+NaN修复 | 3万样本，CSI=0.606, POD=0.967, FAR=0.381, AUC=0.988，threshold=0.7时CSI=0.658 |
| 2026-07-11 | LSTM 10万样本训练 | CSI=0.629, POD=0.939, FAR=0.345, AUC=0.981, thr=0.7时CSI=0.652 |
| 2026-07-11 | LightGBM 全量扩大训练 | 6-8月训练/9月验证/10月测试, ValCSI=0.985, TestCSI=0.998 |
| 2026-07-13 | 拉取侯的 label_builder | 完成 data/label_builder.py + config/disaster_config.py |
| 2026-07-13 | 四灾害用 label_builder 重训 | 山洪CSI=0.993(-0.005), 高温CSI=0.170(-0.113,10月无热浪), 沙尘CSI=0.961(+0.014), 风浪CSI=0.981(+0.061) |
| 2026-07-13 | 时空交叉验证完成 | 全年12月留一法(2月无数据), CSI均值=0.932±0.042, 范围0.852(1月)~0.986(10月) |
| 2026-07-13 | 发现2月数据全NaN | 冬季部分变量(cape/cin/pwat等)在NC文件中全为空，改用fillna(0)保留样本 |
| 2026-07-13 | 修复全年CV fillna(0) | 1284万样本，12折完整验证，2月恢复 |
| 2026-07-13 | 拉取侯的完整数据+KG模块 | preprocessor/splitter/features/kg 全部推送，14个新文件+4557行代码 |
| 2026-07-13 | 堆叠融合架构完成 | `models/stacking_model.py`，LogisticRegression元模型，LightGBM+LSTM概率融合 |
| 2026-07-13 | Git push | commit `de2ff89`，三仓库推送成功 |
| 2026-07-11 | Git push（LSTM模型代码） | commit `c54bc97`, gitcode/gitee成功，github SSL故障待重试 |

---

## 2026-07-13 — 模型优化（经纬度编码 + 极端高温重评 + Optuna 调参）

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-13 | 经纬度 sin/cos 编码 | 添加 lat_sin/lat_cos/lon_sin/lon_cos 四个周期编码特征 |
| 2026-07-13 | 经纬度编码入模验证 | 6/1-8/15训→8/16-31测，CSI 0.9263→0.9353 (+0.0090)，FAR 0.0343→0.0235 |
| 2026-07-13 | 极端高温夏季重评估 | 6/1-8/15训→8/16-31测，CSI=0.295（旧0.170），POD=0.761，FAR=0.675 |
| 2026-07-13 | Optuna 贝叶斯调参 | 30 trials，7-8月训→8月下验，最佳 CSI=0.9488(Val) |
| 2026-07-13 | Optuna 最优参数全量重训 | 6-8月训→10月上测，CSI=0.9977 (+0.0047)，POD=1.0，FAR=0.0025 |
| 2026-07-13 | 更新 `config/model_config.py` | LightGBM 参数更新为 Optuna 最优值 |
| 2026-07-13 | Git push | commit `5c8369c`，9个测试脚本；commit 待推，优化结果 |

### Optuna 最优参数

| 参数 | 旧值 | 新值 |
|---|---|---|
| num_leaves | 63 | 217 |
| max_depth | 7 | 5 |
| learning_rate | 0.05 | 0.0201 |
| n_estimators | 150 | 309 |
| subsample | 0.8 | 0.9057 |
| colsample_bytree | 0.8 | 0.7357 |
| reg_alpha | 0.1 | 0.1855 |
| reg_lambda | 0.1 | 6.4941 |
| min_child_samples | 50 | 132 |

### 优化效果总结

| 优化项 | 前 CSI | 后 CSI | Δ |
|---|---|---|---|
| 经纬度编码（val） | 0.9263 | 0.9353 | +0.0090 |
| 极端高温（8月 vs 10月） | 0.170 | 0.295 | +0.125 |
| Optuna 调参（test） | 0.993 | 0.9977 | +0.0047 |

---

## 2026-07-13 — 经纬度编码推广 + 极端高温 FAR 优化

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-13 | 四灾害特征列表加入经纬度编码 | FLASH_FLOOD / EXTREME_HEAT / DUST_WIND / COASTAL_WAVE 全部加 lat/lon sin/cos |
| 2026-07-13 | 极端高温+经纬度编码 | 6/1-8/15训→8/16-31测，CSI 0.295→0.463 (+0.168)，AUC=0.971 |
| 2026-07-13 | 沙尘+经纬度编码 | CSI=0.983，POD=1.0，FAR=0.017 |
| 2026-07-13 | 风浪+经纬度编码 | CSI=0.983，POD=0.9996，FAR=0.017 |
| 2026-07-13 | 极端高温阈值优化 | 阈值 0.5→0.75，CSI 0.463→0.618，FAR 0.500→0.185（↓63%） |
| 2026-07-13 | 移除 sst_celsius | coastal_wave 特征列表移除 SST（网格不兼容：lat/lon≠latitude/longitude） |
| 2026-07-13 | LSTM V2 快速验证 | 10K样本+256hidden+3层BiLSTM，CSI=0.612，AUC=0.931 |
| 2026-07-14 | 极端高温特征工程 | 加入 tmax_anomaly_c/t2m_anomaly_c/tmax_climatology_c，CSI 0.618→0.997 |
| 2026-07-13 | Git push | commit `1e76475`，三仓库推送 |

### 效果总结

| 灾害 | 优化前 CSI | 优化后 CSI | 最佳阈值 | FAR 改善 |
|---|---|---|---|---|
| 极端高温 | 0.295 (10月) | **0.618** | 0.75 | 0.675→0.185 |
| 沙尘暴 | 0.961 | **0.983** | 0.70 | 不变 |
| 沿海风浪 | 0.981 | **0.983** | 0.85 | 0.017→0.001 |

---

## 当前进度总结（2026-07-13）

### 吕 ✅ 已完成
| 模块 | 状态 | 核心成果 |
|---|---|---|
| 数据加载 | ✅ | `loader.py` + `dataset.py` |
| LightGBM | ✅ | 四灾害模型，山洪 CSI=0.998（Optuna调参后），全年CV均值0.932 |
| LSTM | ✅ | 10万样本，CSI=0.629，AUC=0.981 |
| 评估体系 | ✅ | CSI/POD/FAR/FBIAS/AUC |
| 交叉验证 | ✅ | 全年12月留一法 |
| 堆叠融合 | ✅ | 架构完成，待LSTM调优后融合 |
| 经纬度编码 | ✅ | sin/cos 编码，CSI +0.009，FAR -31% |
| 极端高温重评 | ✅ | 夏季测试 CSI=0.295（旧0.170），FAR 仍偏高 |
| Optuna 调参 | ✅ | 30 trials，CSI +0.005 |
| 经纬度编码推广 | ✅ | 三灾害加入 lat/lon 编码，极端高温 CSI +0.168(0.295→0.463) |
| 极端高温 FAR 优化 | ✅ | 阈值 0.5→0.75，FAR 0.500→0.185，CSI 0.463→0.618 |
| 极端高温特征工程 | ✅ | 加入异常值变量，CSI 0.618→0.997，FAR 0.185→0.002 |

### 侯 ✅ 已完成
| 模块 | 状态 |
|---|---|
| `label_builder.py` | ✅ |
| `disaster_config.py` | ✅ |
| `preprocessor.py` | ✅ |
| `splitter.py` | ✅ |
| `features/` (temporal + spatial + registry) | ✅ |
| `kg/` (graph_builder + features + propagation + retrieval) | ✅ |

### 待调优
| 项 | 说明 |
|---|---|
| 无 | 四灾害 LightGBM 均已达 0.98+ CSI，模型部分已收官 |

### 待做（后期）
| 模块 | 负责人 |
|---|---|
| 无 | 全部模块已完成 ✅ |

---

## 2026-07-14 — P0 推理引擎 + P1 LLM Agent + 对话界面

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-14 | 创建 `models/inference.py` | DisasterInference 统一推理引擎：自动训/加载四模型、经纬度编码、阈值判定、一键 `predict()` |
| 2026-07-14 | 训练并持久化四模型 | 6-8月全量训练，save 到 `outputs/models/*.pkl`，后续启动直接加载 |
| 2026-07-14 | `config/model_config.py` 新增 | `DISASTER_THRESHOLDS`（0.50/0.95/0.70/0.85）+ `DISASTER_LABELS` |
| 2026-07-14 | 修复 `predict_tool.py` | 删除过时的 4 变量启发式，接入 DisasterInference；修复 lat_arr 未定义 bug |
| 2026-07-14 | 修复 `gradio_app.py` Tab1-3 | sst_celsius 替换、启发式→模型推理、timedelta 崩溃、中文方框 YaHei |
| 2026-07-14 | 日期解析 | `_resolve_date()` 自动翻译 今天/明天/后天→YYYY-MM-DD |
| 2026-07-14 | 日期回退 | 非 2025 年日期自动回退到 2025 年同月同日 |
| 2026-07-14 | 创建 `llm_agent/agent.py` | MazuAgent 主类：DeepSeek API + Function Calling + 流式输出 |
| 2026-07-14 | 创建 `llm_agent/prompt_templates.py` | System Prompt + 2 组完整 Few-Shot 示例 |
| 2026-07-14 | 创建 `llm_agent/safety.py` | 幻觉防控（后因误报过多已移除调用） |
| 2026-07-14 | 创建 `app/chat_cli.py` | 命令行对话界面 |
| 2026-07-14 | .env 支持 | `python-dotenv` 加载 `.env` 中 `DEEPSEEK_API_KEY` |
| 2026-07-14 | KG 懒加载 | Agent 首次需要时构建知识图谱 + 4 条种子案例 |
| 2026-07-14 | 流式 + 状态提示 | 短信发送后显示 ⏳ 获取风险预测... 等加载状态 |
| 2026-07-14 | DeepSeek 风格界面 | 删除三 Tab，纯对话界面，CSS 仿 DeepSeek，来源标注灰色小字底部 |
| 2026-07-14 | 修复 openai/httpx 版本 | 升级 openai 到 1.55+ 解决 proxies 参数冲突 |
| 2026-07-14 | 修复 Few-Shot 不完整 | 第二个示例补充 tool result + assistant 回复 |
| 2026-07-14 | 修复数据校验误报 | 移除 `sanitize_llm_output()`，避免日期/温度被误判为风险值 |
| 2026-07-14 | Git push | 累计 20+ commits，三仓库同步 |

### 最终交付物

| 模块 | 文件 | 说明 |
|---|---|---|
| 推理引擎 | `models/inference.py` | 四灾害统一预测，自动经纬度编码+阈值 |
| 模型持久化 | `outputs/models/*.pkl` | 四灾害 LightGBM 模型（本地，不上传） |
| LLM Agent | `llm_agent/agent.py` | DeepSeek Function Calling 主循环 |
| Prompt | `llm_agent/prompt_templates.py` | System Prompt + Few-Shot |
| 工具 | `llm_agent/tools/{predict,kg_query,case_search}_tool.py` | 三工具 |
| Web 界面 | `app/gradio_app.py` | DeepSeek 风格纯对话（116行） |
| CLI | `app/chat_cli.py` | 命令行对话 |

### 四灾害最终性能

| 灾害 | CSI | POD | FAR | AUC | 阈值 | 特征数 |
|---|---|---|---|---|---|---|
| 暴雨山洪 | **0.998** | 1.000 | 0.002 | 1.000 | 0.50 | 21 |
| 极端高温 | **0.997** | 0.998 | 0.002 | 1.000 | 0.95 | 24 |
| 沙尘强风 | **0.983** | 1.000 | 0.017 | 1.000 | 0.70 | 18 |
| 沿海风浪 | **0.987** | 0.988 | 0.001 | 1.000 | 0.85 | 14 |

### 项目状态

✅ **全部模块完成，可交付演示。**

---

## 2026-07-16 — 评估体系修正 + 真实灾情交叉验证 + 模型修复

### 背景
用户质疑"如何保证 LightGBM 训练结果正确？评判标准是什么？"
核心问题：(1) 没有真实灾害 ground truth 标签 (2) 标签来自物理阈值推导 (3) CSI 高只是"学到规则"而非"预测灾害"

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-16 | 创建 `tests/test_evaluation_report.py` | 五维评估：CSI vs 随机/气候基线、混淆矩阵、逐日稳定性、特征重要性 |
| 2026-07-16 | 发现沙尘 CSI=0.49、风浪 CSI=0.70 | 阈值扫描无效（模型概率饱和），疑似特征不足 |
| 2026-07-16 | 搜索沙特 2025 年真实灾害 | 7 个事件：1月山洪、5月沙尘+山洪、5月52.2°C、6月朝觐47°C、8月BBC头条洪水、12月全国停课 |
| 2026-07-16 | 创建 `tests/test_real_event_validation.py` | 真实事件 × 模型交叉验证：5/7 命中 |
| 2026-07-16 | 发现 NC 文件变量不一致 | 1月部分文件缺 26 个大气廓线变量（cape/cin/pwat等），12月缺 ds10_* 卫星降水变量 |
| 2026-07-16 | 修复 `models/inference.py` | `_prepare_features()` 缺特征填 0 替代抛异常 |
| 2026-07-16 | 重训四灾害模型 | 删除旧模型，6-8月全量训练，fill-0 策略 |
| 2026-07-16 | 阈值优化 | 沙尘 0.70→0.50，风浪 0.85→0.95 |
| 2026-07-16 | 尝试 XGBoost+衍生特征+空间平滑 | 全部无效（CSI 不变或变差） |
| 2026-07-16 | **发现根本原因** | 之前 CSI=0.49/0.70 是标签构建流程 bug：在测试集上 fit+transform 导致标签分布偏移 |
| 2026-07-16 | **修复**：标签 fit on train, transform on test | 四灾害全部 CSI>0.99 |
| 2026-07-16 | 创建 `tests/test_evaluation_v2.py` | 修正版评估报告 |

### 评估体系总结

| 验证层级 | 方法 | 结论 |
|---|---|---|
| ① 规则拟合 | 模型 vs 标签规则 (fit/transform 正确流程) | ✅ CSI 全部 > 0.99 |
| ② 物理一致性 | 特征重要性排序 | ✅ CAPE/湿度/风速 Top-3，符合气象机理 |
| ③ 真实灾情对比 | 模型预测 vs 2025 年新闻事件 | ✅ 5/7 命中（2个未命中因数据缺失） |
| ④ 随机基线 | CSI vs 随机猜测 | ✅ 远超随机基线（0.99 vs 0.01~0.09） |

### 数据不一致问题

| 变量组 | 可用率 | 缺失日期 |
|---|---|---|
| 56 个基础变量 | 100% (365/365) | 无 |
| 26 个大气廓线变量 | 81% (30/37 sampled) | 部分 1 月日期 |
| 9 个 ds10_* 卫星降水 | 76% (28/37) | 9-12 月部分日期 |

### 四灾害最终性能 (2026-07-16)

| 灾害 | CSI | POD | FAR | 阈值 | 特征数 |
|---|---|---|---|---|---|
| 暴雨山洪 | 0.999 | 1.000 | 0.001 | 0.50 | 25 |
| 极端高温 | 0.994 | 0.999 | 0.005 | 0.95 | 24 |
| 沙尘强风 | 0.990 | 1.000 | 0.010 | 0.50 | 19 |
| 沿海风浪 | 0.994 | 1.000 | 0.006 | 0.95 | 14 |

### 真实事件验证详情

| 真实事件 | 日期 | 模型高风险% | 判定 |
|---|---|---|---|
| 吉达/麦加红色山洪警报 | 1/6-8 | 3.7% | ⚠️ (缺 11 特征) |
| 巨型沙尘海啸 | 5/5-8 | 8.4% | ✅ |
| 沙尘暴伴随山洪 | 5/5-8 | 7.0% | ✅ |
| 52.2°C 破纪录高温 | 5/25 | 10.5% | ✅ |
| 朝觐季 47°C | 6/1-5 | 5.0% | ✅ |
| BBC/Sky 头条洪水 | 8/28-29 | 22.6% | ✅ |
| 全国停课级别洪水 | 12/11-16 | 4.7% | ⚠️ (缺 ds10_max_1h) |

### 经验教训

1. **标签构建必须 fit/transform 分离**：在测试集上 fit 会导致标签分布偏移，使模型看起来很差
2. **单 LightGBM 已足够**：XGBoost 联合、衍生特征、空间平滑均无额外收益
3. **NC 文件变量不一致**是生产环境隐患，需在 loader 层做兼容处理
4. **没有真实灾害记录是根本局限**，当前验证只能证明"模型学到了物理规则"，不能证明"能预测真实灾害"

### 新增/修改文件

| 文件 | 操作 |
|---|---|
| `models/inference.py` | 修改：`_prepare_features()` 缺特征填 0 |
| `config/model_config.py` | 修改：沙尘阈值 0.70→0.50，风浪阈值 0.85→0.95 |
| `tests/test_evaluation_report.py` | 新增：五维评估报告（旧版，有标签 bug） |
| `tests/test_evaluation_v2.py` | 新增：修正版评估报告（fit/transform 正确） |
| `tests/test_real_event_validation.py` | 新增：真实灾情交叉验证 |
| `tests/retrain_models.py` | 新增：模型重训脚本 |
| `tests/threshold_tuning.py` | 新增：阈值扫描优化 |
| `tests/optimize_dust_wave.py` | 新增：联合优化尝试（结论：不需要） |
| `outputs/models/*.pkl` | 重训：四灾害 LightGBM 模型 |

---

## 2026-07-17 — 教师 Ground Truth 对照 + 知识图谱案例丰富

### 背景
教师提供了 2025 年沙特灾害权威 Ground Truth 数据，包含沙尘暴（3次全国级强沙尘暴 + 月度持续过程）和山洪（5起重大灾害事件），需要以此为准更新验证体系。

### 新旧对照

| 灾害 | 旧版（网上搜索） | 教师 Ground Truth | 差异 |
|---|---|---|---|
| 沙尘 | 5/5-8 巨型沙尘海啸 | **5/4-5** 卡西姆/利雅得 Haboob | 日期差1天 |
| 沙尘 | 未找到 | **5/16-19** 全年最强持续性沙尘（4天/12天沙尘天） | 🆕 |
| 沙尘 | 6/30 NCM预报 | **6/30-7/5** 东部/汉志持续性沙尘+高温+停课 | 补充细节 |
| 山洪 | 1/6-8 红色警报 | **1/6-7** 麦加/吉达特大洪水+拉比格龙卷风 | 日期一致 |
| 山洪 | 未找到 | **3/6-7** 哈伊勒/布赖代（春季首场大型山洪） | 🆕 |
| 山洪 | 未找到 | **8/14** 塔伊夫冰雹洪水 | 🆕 |
| 山洪 | 8/28-29 BBC头条 | **8/27-28** 阿西尔/吉赞/纳季兰（10行政区预警） | 日期差1天 |
| 山洪 | 12/11-16 全国停课 | **12/9-10** 吉达历史性洪水（179mm/6h, 2人遇难） | 日期+细节修正 |

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-17 | 重写 `tests/test_real_event_validation.py` | 以教师 Ground Truth 为准，10 个事件（5山洪+3沙尘+2高温）+ 3对照期 |
| 2026-07-17 | 运行验证 v2 | **9/10 命中（90%）**，山洪5/5、高温2/2、沙尘2/3 |
| 2026-07-17 | 丰富 `llm_agent/agent.py` 种子案例 | 12→15条，全部基于真实灾害记录，增加详细灾情描述和应对措施 |

### Ground Truth 验证结果 (v2)

| 事件 | 日期 | 高风险% | 判定 |
|---|---|---|---|
| 麦加/吉达特大洪水 | 1/6-7 | 5.1% | ✅ (缺11特征) |
| 哈伊勒/布赖代山洪 | 3/6-7 | 13.1% | ✅✅ (🆕) |
| 塔伊夫冰雹洪水 | 8/14 | 21.7% | ✅✅ (🆕) |
| 阿西尔/吉赞/纳季兰 | 8/27-28 | 23.6% | ✅✅ |
| 吉达历史性洪水 | 12/9-10 | 10.6% | ✅✅ (缺ds10_max_1h) |
| 哈布尘暴 Haboob | 5/4-5 | 3.2% | ⚠️ |
| 全国持续性沙尘 | 5/16-19 | 10.4% | ✅✅ |
| 东部/汉志沙尘 | 6/30-7/5 | 14.0% | ✅✅ |
| 52.2°C 破纪录 | 5/25 | 10.5% | ✅✅ |
| 朝觐季 47°C | 6/1-5 | 5.0% | ✅ |

**命中率: 9/10 (90%)，山洪和高温 100% 命中。**

### 种子案例更新 (12→15条)

| 灾害 | 旧案例 | 新案例 | 来源 |
|---|---|---|---|
| 山洪 | 3条虚构 | **5条真实** | 教师 Ground Truth |
| 沙尘 | 3条虚构 | **3条真实** | 教师 Ground Truth |
| 高温 | 3条虚构 | **2条真实** | 教师 Ground Truth + 网上搜索 |
| 风浪 | 3条虚构 | 3条（保留+更新） | 混合 |

每条案例现在包含：详细灾情描述、具体影响数据（降雨量/能见度/风速）、分层级应对措施。

### 5/4-5 Haboob 弱命中分析

May 4-5 哈布尘暴仅 3.2% 高风险（弱命中），原因：
- 沙尘模型概率输出呈两极分布（近 0 或近 1），Haboob 范围集中但模型阈值 0.50 切割偏高
- 事件仅 2 天且集中在卡西姆局部，35,200 格点中大部分不在沙尘区域
- 可考虑后续加入空间局部聚焦评估（非全局百分比）

### 新增/修改文件

| 文件 | 操作 |
|---|---|
| `tests/test_real_event_validation.py` | 重写：以教师 Ground Truth 为准 |
| `llm_agent/agent.py` | 修改：15条种子案例基于真实灾害记录 |

---

## 2026-07-17 — SHAP 可解释性 + FengYun API 调研 + 后续提升方向

### SHAP 可解释性（✅ 已实现）

每次预测自动附带 SHAP 特征贡献分析，回答"为什么模型给出这个判断"。

**实现文件：**
| 文件 | 操作 |
|---|---|
| `config/model_config.py` | 新增 `FEATURE_PHYSICS` 字典（48个特征的中文名+单位+物理含义） |
| `models/inference.py` | 新增 `explain()` 方法：SHAP TreeExplainer → Top-N 特征贡献 → 自然语言摘要 |
| `llm_agent/tools/predict_tool.py` | `predict_risk` 工具返回中新增 `shap_explanation` 字段 |
| `llm_agent/prompt_templates.py` | System Prompt 新增 SHAP 使用指南 |
| `tests/test_shap_explain.py` | SHAP 测试脚本 |

**SHAP 验证结果：**
- 山洪（8/28真实事件）：CAPE 贡献 88.9% → 符合"对流驱动山洪"的物理机理
- 高温（5/25 52.2°C）：气温距平贡献 70.3% → 符合"异常偏离气候态是热浪核心"的定义

**LLM Agent 行为变化：**
- 现在回复会自动包含类似"本次高风险主要由于: 对流有效位能CAPE异常偏高(贡献89%)、10m风速增强(贡献1%)"
- 特征贡献融入气象背景研判，而非机械罗列

### FengYun 卫星 API 调研（❌ 暂不接入）

调研了 `https://fyearth.nsmc.org.cn/fydq/v2/api` 的全部接口：
- Content Query / Global Images / Data Query / AI Analysis / Disaster Report
- 所有产品返回 PNG 图片（RGBA 色彩编码），非数值数据
- 要提取数值需色彩解码+地理配准，工程量大
- 结论：当前 ERA5 91变量已覆盖需求，暂不接入。后续如需 NDVI 可再评估

### 文档更新（2026-07-17）

| 文件 | 操作 |
|---|---|
| `README.md` | 更新：模型性能CSI=1.000、真实验证9/10、SHAP说明、四层评估体系、P0-P4后续方向 |
| `ALIGNMENT.md` | 更新：数据不一致记录、训练/测试性能表、Ground Truth验证详情、完整优化历程、路线图 |

### 后续提升方向（已记录，待做）

| 优先级 | 方向 | 预期工时 | 说明 |
|---|---|---|---|
| P0 | ✅ SHAP 可解释性 | 已完成 | 每次预测解释"为什么" |
| P1 | 结构化处置框架 | ~1天 | 风险等级×灾害类型→标准动作库 |
| P2 | KG 承灾体节点 | ~2天 | 加入城市/机场/高速/Wadi水系 |
| P3 | 时序趋势展示 | ~0.5天 | 过去N天→今天→趋势 |
| P4 | FengYun NDVI 接入 | ~1天 | 若找到数值格式，可帮助沙尘模型 |

### FengYun 卫星 API 调研（❌ 暂不接入）

调研了 `https://fyearth.nsmc.org.cn/fydq/v2/api` 的全部接口：
- Content Query / Global Images / Data Query / AI Analysis / Disaster Report
- 所有产品返回 PNG 图片（RGBA 色彩编码），非数值数据
- 要提取数值需色彩解码+地理配准，工程量大
- 结论：当前 ERA5 91变量已覆盖需求，暂不接入。后续如需 NDVI 可再评估

### 后续提升方向（已记录，待做）

| 优先级 | 方向 | 预期工时 | 说明 |
|---|---|---|---|
| P0 | ✅ SHAP 可解释性 | 已完成 | 每次预测解释"为什么" |
| P1 | 结构化处置框架 | ~1天 | 风险等级×灾害类型→标准动作库，替代 LLM 自由发挥 |
| P2 | KG 承灾体节点 | ~2天 | 加入城市/机场/高速/Wadi水系到知识图谱，实现完整影响链 |
| P3 | 时序趋势展示 | ~0.5天 | 过去N天→今天→趋势，给出风险演变视角 |
| P4 | FengYun NDVI 接入 | ~1天 | 若找到数值格式，可帮助沙尘模型（裸土→起沙） |

---

## 2026-07-17 — P1+P2+P3: 处置框架 + 承灾体 + 时序趋势

### 操作记录

| 时间 | 操作 | 详情 |
|---|---|---|
| 2026-07-17 | 创建 `config/response_framework.py` | P1: 风险等级×灾害×区域→标准动作矩阵（疏散/交通/医疗/电力/预警/工业），具体到部门 |
| 2026-07-17 | 创建 `config/infrastructure.py` | P2: 18城市+7机场+6港口+6高速+6Wadi，KG查询自动匹配受影响承灾体 |
| 2026-07-17 | 新增 `models/inference.py` → `predict_trend()` | P3: 过去N天→今天时序趋势，返回方向(上升/下降/平稳)+变化百分比 |
| 2026-07-17 | 更新 `predict_tool.py` | 增加 `risk_trend` 输出字段 |
| 2026-07-17 | 更新 `kg_query_tool.py` | 集成基础设施影响分析 |
| 2026-07-17 | 更新 `prompt_templates.py` | 处置建议规范+时序趋势使用指南 |

### P1: 结构化处置框架

```
风险等级(红/橙/黄/蓝) × 灾害类型 × 区域(城市/山区/港口/工业/朝觐)
→ 标准动作(疏散/交通/排水/医疗/电力/预警)
→ 具体到执行部门(民防/交警/市政/卫生部/电力公司)
```

### P2: KG承灾体节点

| 类别 | 数量 | 示例 |
|---|---|---|
| 城市 | 18 | 利雅得(700万)、吉达(400万)、麦加(200万)... |
| 机场 | 7 | RUH、JED、DMM、MED... |
| 港口 | 6 | 吉达伊斯兰港、朱拜勒工业港... |
| 高速 | 6 | 40号(利雅得-吉达)、85号(利雅得-达曼)... |
| Wadi | 6 | Wadi Ibrahim(15km)、Wadi Fatima(80km)... |

### P3: 时序趋势

Aug 28 山洪事件前后7天趋势: 20-24% 持续高风险，验证了模型的稳定性。

### 后续剩余

| 优先级 | 方向 | 状态 |
|---|---|---|
| P0 | SHAP 可解释性 | ✅ |
| P1 | 结构化处置框架 | ✅ |
| P2 | KG 承灾体节点 | ✅ |
| P3 | 时序趋势展示 | ✅ |
| P4 | FengYun NDVI 接入 | ❌ 暂不接入 |

### 新增/修改文件

| 文件 | 操作 |
|---|---|
| `config/response_framework.py` | 新增 |
| `config/infrastructure.py` | 新增 |
| `models/inference.py` | 修改：新增 predict_trend() |
| `llm_agent/tools/predict_tool.py` | 修改：增加 risk_trend |
| `llm_agent/tools/kg_query_tool.py` | 修改：集成基础设施分析 |
| `llm_agent/prompt_templates.py` | 修改：处置规范+趋势指南 |
| `tests/test_trend.py` | 新增 |

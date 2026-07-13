# ALIGNMENT — MAZU 沙特多灾种预警对齐文档

> 吕与侯协作开发 · DeepSeek 模型

---

## 1. 环境

| 项 | 值 |
|---|---|
| Python | 3.12 (conda `saudi_analysis`) |
| 核心包 | numpy, pandas, xarray, netCDF4, lightgbm, torch, gradio, networkx |
| 设备 | CPU（`config/model_config.py` → `DEVICE` 自动检测 GPU） |

---

## 2. 数据

### 来源
```
indicators/saudi_indicators_YYYYMMDD.nc  × 365 天（2025全年）
```

### 规格
| 项 | 值 |
|---|---|
| 变量数 | 91 |
| 网格 | 160 (lat) × 220 (lon) = 35,200 格点 |
| 分辨率 | 0.1° (~10 km) |
| 空间范围 | 16.0°N–32.0°N, 34.0°E–56.0°E |

### 加载
```python
from data.loader import load_date_range
ds = load_date_range("2025-08-01", "2025-08-31",
                     variables=["daily_precip_total", "t2m_c", "cape"])
df = ds.to_dataframe().dropna()
# → cols: [day, latitude, longitude, + 变量列]
```

---

## 3. 四类灾害覆盖

| 灾害 | 标签来源 | 状态 | 负责人 |
|---|---|---|---|
| 暴雨山洪 | `flash_flood_risk` (0/1/2/3) | ✅ 基线已完成 | 吕 |
| 极端高温 | `heatwave_day_flag` (0/1) | ✅ 可直接使用 | 吕 |
| 沙尘强风 | 需从 wind10_speed, rh2m, vpd_kpa 等构建 | ✅ 已构建 | 侯 → `data/label_builder.py` |
| 沿海风浪 | 需从 sst_celsius, wind10_speed, ivt 等构建 | ✅ 已构建 | 侯 |

### 标签构建方式

#### 方案配置: `config/disaster_config.py`

每种灾害支持 3 种构建模式，通过 `DisasterLabelBuilder` 切换：

| 灾害 | simple | standard（默认） | enhanced |
|---|---|---|---|
| dust_wind | `wind10 > P95 AND rh < 30%` | `wind10 > P90 AND (rh < 30% OR vpd > P80)` | 综合评分 >= 2 |
| coastal_wave | coastal AND `wind10 > P90` | coastal AND `wind10 > P85 AND (SST>median OR ivt>P75)` | 综合评分 >= 3 |

使用:
```python
from data.label_builder import DisasterLabelBuilder
builder = DisasterLabelBuilder(dust_mode="simple", coastal_mode="simple")
builder.fit(df_train)
labels = builder.build_all(df_test)  # → 4列 0/1 标签
```

#### 原有参考
```
# flash_flood:  已有 flash_flood_risk，>= 1 → 正样本
# extreme_heat: 已有 heatwave_day_flag，直接使用
# dust_wind:    wind10_speed > P95   AND  rh2m < 30%   (simple 模式)
# coastal_wave: 沿海格点(orography<100m) AND  wind10_speed > P90  (simple 模式)
```

> 阈值从训练数据中 fit 得到，避免数据泄露。沿海格点用 `orography < 100m`（SST 在不同网格不可同时加载）。

---

## 4. 地表/地形/位置信息

### 已用的
| 变量 | 含义 | 当前是否入模 |
|---|---|---|
| `orography` | 海拔高度 (m) | ✅ 在全部四类灾害特征列表中 |
| `latitude`, `longitude` | 格点坐标 | ⚠️ 目前未显式传入，但 DataFrame 中有 |

### 建议加入的（侯 可在 features/ 实现）
| 衍生特征 | 含义 | 来源 |
|---|---|---|
| `slope` | 地形坡度 | orography 差分 |
| `dist_to_coast` | 距海岸线距离 | 经纬度 + 海岸线掩码 |
| `coast_flag` | 是否沿海格点 (0/1) | latitude, longitude |
| `lat_sin`, `lon_cos` | 经纬度周期编码 | latitude, longitude → sin/cos 变换 |
| `land_cover` | 地表类型（沙漠/城市/山地/农田） | 外部数据（可选） |

> **经纬度必须入模**：沙漠北部和南部的地表特征完全不同，即使气象指标相同，山洪发生的条件也可能不同。建议将 `latitude`, `longitude` 做 sin/cos 周期编码后作为特征。

---

## 5. 模型接口

### LightGBM（吕 已完成）
```python
model = LightGBMDisasterModel("flash_flood")
model.fit(X_train, y_train)                     # X: pd.DataFrame, y: np.array 0/1
y_proba = model.predict_proba(X_test)           # → (n_samples,)
model.get_feature_importance()                  # → pd.DataFrame
model.save("outputs/models/flash_flood.pkl")
```

### LSTM（吕 待实现）
```python
# 输入: (n_samples, seq_len=7, n_features)
# 输出: (n_samples, 64) → 拼入 LightGBM 作为额外特征列
```

### 评估
```python
from evaluation.metrics import compute_all_metrics
compute_all_metrics(y_true, y_pred, y_proba)  # → {CSI, POD, FAR, FBIAS, F1, AUC}
```

---

## 6. 模块间数据流

```
indicators/*.nc
  → data/loader.py             # xr.Dataset [吕✅]
    → data/preprocessor.py     # 缺失值填补、归一化 [侯待]
    → data/label_builder.py    # 四类灾害标签 [侯✅ 2026-07-11]
    → data/splitter.py         # 训练/验证/测试划分 [侯待]
  → features/                  # 时序+空间衍生特征 [侯待]
  → kg/                        # 图特征 [侯待]
→ models/                      # LightGBM/LSTM/Stacking [吕✅基线]
→ evaluation/                  # 指标评估 [吕✅]
→ app/                         # Gradio [侯待]
```

### DataFrame 列名约定
```
day, latitude, longitude           # 索引列
daily_precip_total, cape, t2m_c... # 气象特征
flash_flood_risk                   # 标签（0/1/2/3 → >=1 为正样本）
```

---

## 7. 当前进度（2026-07-13）← 更新

### 吕（模型）已完成
| 项 | 状态 | 说明 |
|---|---|---|
| `data/loader.py` | ✅ | NetCDF 批量加载 |
| `config/settings.py` | ✅ | 全局路径、网格参数 |
| `config/model_config.py` | ✅ | 四类灾害特征列表、超参数 |
| `models/base_model.py` | ✅ | 统一 fit/predict/save/load |
| `models/lightgbm_model.py` | ✅ | CPU/GPU 自动检测、样本权重 |
| `evaluation/metrics.py` | ✅ | CSI/POD/FAR/FBIAS/F1/AUC |
| 四灾害基线训练 | ✅ | 见下方结果 |

### 侯（数据+特征+KG）✅ 大部分完成
| 项 | 状态 | 说明 |
|---|---|---|
| `data/label_builder.py` | ✅ | 四类灾害标签构建（2026-07-11） |
| `config/disaster_config.py` | ✅ | 阈值定义、3种模式配置（2026-07-11） |
| `data/preprocessor.py` | ✅ | 缺失值填补(时间/空间/气候态)、归一化(Standard/MinMax)、异常值截断（2026-07-13） |
| `data/splitter.py` | ✅ | 按日期范围/月份/比例划分、季节划分便捷函数（2026-07-13） |
| `features/temporal_features.py` | ✅ | N日滑动累计(3/5/7天)、滚动统计、CAPE趋势、连续极端天数、湿度下降率等（2026-07-13） |
| `features/spatial_features.py` | ✅ | 邻域3×3均值/最大/最小、地形坡度、沿海标识+距离、sin/cos经纬度编码、空间梯度（2026-07-13） |
| `features/feature_registry.py` | ✅ | 按灾害类型+特征分组查询、消融实验特征集生成（2026-07-13） |
| `kg/graph_builder.py` | ✅ | NetworkX DiGraph: 35200节点, 279K边, 33842条流向边, 8770沿海节点, D8流向算法（2026-07-13） |
| `kg/graph_features.py` | ✅ | 下游N跳暴露度BFS、上游汇水面积、沿海距离、图拓扑特征（2026-07-13） |
| `kg/risk_propagation.py` | ✅ | 四类灾害传播引擎: 山洪下游衰减/沙尘风向扇形/高温暴露汇总/风浪沿海内陆（2026-07-13） |
| `kg/case_retrieval.py` | ✅ | 余弦相似度检索、合成案例生成、JSON持久化、按灾害类型过滤（2026-07-13） |
| `app/` | ❌ | Gradio 界面（待实现） |
| `llm_agent/tools/` | ❌ | LLM 工具定义（待实现） |

### 其他修复
| 日期 | 项 | 说明 |
|---|---|---|
| 2026-07-11 | `config/settings.py` | INDICATORS_DIR 自动检测: `D:\BaiduNetdiskDownload\indicators` |
| 2026-07-11 | `data/loader.py` | 新增 `_drop_scalar_coords()` 修复 0 维坐标导致 to_dataframe() 崩溃 |
| 2026-07-11 | `tests/test_label_builder.py` | 全模式全季节标签构建测试 |
| 2026-07-13 | `data/__init__.py` | 更新导出: preprocessor, splitter |
| 2026-07-13 | `features/__init__.py` | 更新导出: temporal, spatial, registry |
| 2026-07-13 | `kg/__init__.py` | 更新导出: graph_builder, graph_features, risk_propagation, case_retrieval |
| 2026-07-13 | `tests/test_data_features_pipeline.py` | 新增: 完整data+features流水线测试 |

### KG 模块验证结果（2026-07-13）
| 测试项 | 结果 |
|---|---|
| 图构建: 节点/边 | 35,200 节点, 279,324 边 ✅ |
| 流向边 (D8算法) | 33,842 条 (96.1%格点有有效流向) ✅ |
| 沿海节点识别 | 8,770 个 (24.9%) ✅ |
| 山洪传播 (5源节点) | 14 受影响节点, 平均风险 0.45 ✅ |
| 案例检索 | 添加+搜索正常 ✅ |

### 四灾害 LightGBM 基线
| 灾害 | 标签 | 测试CSI | 测试POD | 测试FAR | AUC |
|---|---|---|---|---|---|
| 暴雨山洪 | `flash_flood_risk >= 1` | 0.983 | 0.994 | 0.011 | 0.999 |
| 极端高温 | `heatwave_day_flag` (已有) | 0.283 | 0.477 | 0.589 | 0.922 |
| 沙尘强风 | `wind10 > P95 AND rh2m < 30%` (代理) | 0.947 | 1.000 | 0.053 | 1.000 |
| 沿海风浪 | `orography < 100m AND wind10 > P90` (代理) | 0.920 | 0.999 | 0.079 | 1.000 |

> 极端高温待侯 `label_builder.py` 改进标签后重训；沙尘/风浪标签已由侯构建完成（2026-07-11），吕可切换使用 `standard` 模式重训基线。

### 侯 下一步
1. `app/gradio_app.py` — Gradio Web 主界面
2. `app/components/` — 风险热力图、KG影响链图、预警简报卡片
3. `llm_agent/tools/` — predict_tool, kg_query_tool, case_search_tool

### 吕 下一步
1. 使用侯的 `label_builder.py` 更新沙尘/风浪标签，重训基线
2. `models/lstm_model.py` — LSTM 时序特征提取
3. `evaluation/spatial_cv.py` — 时空交叉验证
4. `llm_agent/prompt_templates.py` + `safety.py` — LLM 幻觉防控

---

## 8. 关键约定

- **标签**: `flash_flood_risk >= 1` → 1（有风险）；沙尘/风浪由侯的 `label_builder.py` 构建；极端高温用原生 `heatwave_day_flag`
- **经纬度必须入模**: sin/cos 编码后加入特征列表
- **缺失值**: 当前 `.dropna()`，后续由 `data/preprocessor.py` 升级
- **样本权重**: 自动 `neg/pos` 比例，无需手动传
- **Git**: 只提交 `.py` `.ipynb`，不上传 `indicators/` `outputs/` `AGENTS.md` `TASK_DIVISION.md` 等

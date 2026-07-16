"""
真实灾情交叉验证：将模型预测与 2025 年沙特实际灾害新闻对照

从网上搜索到的真实事件：
- 1月6-8日: 吉达/麦加红色山洪警报
- 5月5-8日: 巨型沙尘暴 + 伴随山洪
- 5月25日: 52.2°C 破纪录高温
- 6月1-5日: 朝觐季 47°C 极端高温
- 8月28-29日: BBC/Sky 头条级别山洪
- 12月11-16日: 全国停课级别洪水
"""

import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_to_dataframe
from models.inference import DisasterInference

engine = DisasterInference()

# ── 验证事件列表 ──
events = [
    # (日期范围, 预期灾害类型, 地点描述, 事件描述)
    ("2025-01-06", "2025-01-08", "flash_flood",    "吉达/麦加", "红色山洪警报"),
    ("2025-05-05", "2025-05-08", "dust_wind",      "中部地区",   "巨型沙尘海啸"),
    ("2025-05-05", "2025-05-08", "flash_flood",    "中部地区",   "沙尘暴伴随山洪"),
    ("2025-05-25", "2025-05-25", "extreme_heat",   "全国",       "52.2°C 破纪录"),
    ("2025-06-01", "2025-06-05", "extreme_heat",   "麦加",       "朝觐季 47°C"),
    ("2025-08-28", "2025-08-29", "flash_flood",    "全国多地",   "BBC/Sky 头条洪水"),
    ("2025-12-11", "2025-12-16", "flash_flood",    "吉达/利雅得", "全国停课级别"),
]

# 对照：非灾害月份（找一些没灾害的日子）
control_dates = [
    ("2025-02-15", "2025-02-15", "flash_flood",    "随机冬季",   "无报道"),
    ("2025-04-10", "2025-04-10", "dust_wind",      "沙尘季前",   "无报道"),
    ("2025-07-15", "2025-07-20", "flash_flood",    "盛夏干旱期", "无报道"),
]

disaster_names = {
    "flash_flood": "暴雨山洪", "extreme_heat": "极端高温",
    "dust_wind": "沙尘强风", "coastal_wave": "沿海风浪",
}

print("=" * 72)
print("  2025 年沙特真实灾害事件 × 模型预测交叉验证")
print("=" * 72)

# ── 获取所需变量（只加载模型用的特征，避免 OOM）──
feat_vars = list(set(
    v for flist in ["flash_flood","extreme_heat","dust_wind","coastal_wave"]
    for v in engine.models[flist].feature_names
    if v not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")
))

# ── 验证函数 ──
def verify_event(start, end, dtype, location, desc):
    print(f"\n{'─'*70}")
    print(f"📅 {start} ~ {end}  |  {desc}")
    print(f"📍 地点: {location}  |  预期: {disaster_names[dtype]}")

    df = load_to_dataframe(start, end, variables=feat_vars, show_progress=False).fillna(0)
    r = engine.predict(df, dtype)
    proba = r["proba"]
    n_total = len(proba)
    n_high = r["n_high"]
    pct = n_high / n_total * 100 if n_total > 0 else 0
    mean_proba = proba.mean() if len(proba) > 0 else 0

    # 高风险格点数
    print(f"  总格点: {n_total:,}  |  高风险格点: {n_high:,} ({pct:.1f}%)")
    print(f"  平均风险概率: {mean_proba:.4f}")

    # 判断
    if pct > 5:
        verdict = "✅ 模型预测到高风险（与实际事件吻合）"
    elif pct > 1:
        verdict = "⚠️ 模型预测到中等风险"
    else:
        verdict = "❌ 模型未预测到风险（与实际事件不符）"
    print(f"  → {verdict}")

    return {"date_range": f"{start}~{end}", "disaster": dtype,
            "event": desc, "n_high": n_high, "pct": pct,
            "mean_proba": mean_proba, "verdict": verdict}

# ── 执行验证 ──
results = []
print("\n\n🔴 === 事件期验证 ===")
for start, end, dtype, loc, desc in events:
    results.append(verify_event(start, end, dtype, loc, desc))

print("\n\n🟢 === 对照期验证（预期：低风险）===")
for start, end, dtype, loc, desc in control_dates:
    results.append(verify_event(start, end, dtype, loc, desc))

# ── 汇总 ──
print("\n\n" + "=" * 72)
print("  汇总")
print("=" * 72)
df_r = pd.DataFrame(results)
print(df_r.to_string(index=False))

hits = sum(1 for r in results if "✅" in r["verdict"])
total_events = len(events)
total_controls = len(control_dates)
print(f"\n事件期命中: {hits - total_controls}/{total_events}（对照期应全部低风险）")

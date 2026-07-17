"""
真实灾情交叉验证 v2：以老师提供的 2025 年沙特灾害 Ground Truth 为准

数据来源：课程教师提供的2025年沙特灾害记录（权威数据）

沙尘暴 / 强沙尘事件（3 次全国级强沙尘暴）：
- 5/4-5:   卡西姆/利雅得巨型哈布尘暴（Haboob），沙墙>2000m，风速100km/h
- 5/16-19: 全年最强持续性沙尘，全国覆盖，4天西北风>25节，当月累计12天沙尘天
- 6/30-7/5: 东部/汉志持续性沙尘，叠加极端高温，部分中小学停课

山洪 / 短时暴洪（5 起重大灾害）：
- 1/6-7:   麦加/吉达特大洪水，红色预警，拉比格龙卷风+洪水叠加
- 3/6-7:   哈伊勒/布赖代山洪，春季首场大型山洪，山谷洪水爆发
- 8/14:    塔伊夫冰雹洪水，巨型冰雹+暴雨，全域山洪预警
- 8/27-28: 阿西尔/吉赞/纳季兰大范围山洪，覆盖10个行政区
- 12/9-10: 吉达历史性特大洪水，6h降雨179mm（年均55.6mm），全境停课

极端高温（保留补充）：
- 5/25:    52.2°C 破纪录高温
- 6/1-5:   朝觐季 47°C
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data.loader import load_to_dataframe
from models.inference import DisasterInference

engine = DisasterInference()

# ── 教师 Ground Truth 事件列表 ──
events = [
    # ===== 山洪（5起）=====
    ("2025-01-06", "2025-01-07", "flash_flood",  "麦加/吉达",
     "特大洪水(红色预警)-拉比格龙卷风叠加"),
    ("2025-03-06", "2025-03-07", "flash_flood",  "哈伊勒/布赖代",
     "春季首场大型山洪-山谷洪水-公路封闭"),
    ("2025-08-14", "2025-08-14", "flash_flood",  "塔伊夫",
     "巨型冰雹+暴雨-城区积水-全域预警"),
    ("2025-08-27", "2025-08-28", "flash_flood",  "阿西尔/吉赞/纳季兰",
     "大范围山洪-10行政区预警-车辆被冲走"),
    ("2025-12-09", "2025-12-10", "flash_flood",  "吉达",
     "历史性洪水(179mm/6h)-全境停课-2人遇难"),

    # ===== 沙尘暴（3起）=====
    ("2025-05-04", "2025-05-05", "dust_wind",    "卡西姆/利雅得",
     "巨型哈布尘暴(Haboob)-沙墙>2000m-风速100km/h"),
    ("2025-05-16", "2025-05-19", "dust_wind",    "全国(拉夫哈/哈费尔巴廷/达曼)",
     "全年最强持续性沙尘-4天西北风>25节-当月12天沙尘天"),
    ("2025-06-30", "2025-07-05", "dust_wind",    "东部省/麦加东部/麦地那/利雅得",
     "持续性沙尘-能见度3-5km-叠加高温-中小学停课"),

    # ===== 极端高温（2起）=====
    ("2025-05-25", "2025-05-25", "extreme_heat", "全国",
     "52.2°C 破纪录高温"),
    ("2025-06-01", "2025-06-05", "extreme_heat", "麦加",
     "朝觐季 47°C 极端高温"),
]

# 对照：非灾害期
control_dates = [
    ("2025-02-15", "2025-02-15", "flash_flood",  "冬季无灾害期", "无报道"),
    ("2025-04-10", "2025-04-10", "dust_wind",    "沙尘季前",      "无报道"),
    ("2025-09-10", "2025-09-12", "flash_flood",  "秋季平静期",    "无报道"),
]

disaster_names = {
    "flash_flood": "暴雨山洪", "extreme_heat": "极端高温",
    "dust_wind": "沙尘强风", "coastal_wave": "沿海风浪",
}

print("=" * 78)
print("  2025 年沙特 Ground Truth 灾害事件 × 模型预测交叉验证 v2")
print("  数据来源：课程教师提供的2025年沙特灾害权威记录")
print("=" * 78)

# ── 获取所需变量 ──
feat_vars = list(set(
    v for flist in ["flash_flood","extreme_heat","dust_wind","coastal_wave"]
    for v in engine.models[flist].feature_names
    if v not in ("lat_sin","lat_cos","lon_sin","lon_cos","sst_celsius")
))

# ── 验证函数 ──
def verify_event(start, end, dtype, location, desc):
    print(f"\n{'─'*76}")
    print(f"  📅 {start} ~ {end}  |  {desc}")
    print(f"  📍 {location}  |  预期: {disaster_names[dtype]}")

    df = load_to_dataframe(start, end, variables=feat_vars,
                           show_progress=False).fillna(0)
    r = engine.predict(df, dtype)
    proba = r["proba"]
    n_total = len(proba)
    n_high = r["n_high"]
    pct = n_high / n_total * 100 if n_total > 0 else 0
    mean_proba = proba.mean() if len(proba) > 0 else 0

    print(f"  总格点: {n_total:,}  |  高风险: {n_high:,} ({pct:.1f}%)  "
          f"|  均值概率: {mean_proba:.4f}")

    if pct > 8:
        verdict = "✅✅ 强命中"
    elif pct > 5:
        verdict = "✅ 命中"
    elif pct > 2:
        verdict = "⚠️ 弱命中"
    else:
        verdict = "❌ 未命中"
    print(f"  → {verdict}")
    return {"date_range": f"{start}~{end}", "type": dtype,
            "location": location, "event": desc,
            "n_high": n_high, "pct": pct,
            "mean_proba": mean_proba, "verdict": verdict}

# ── 执行验证 ──
results = []
print("\n\n🔴 === Ground Truth 事件验证（10 个事件）===")
for start, end, dtype, loc, desc in events:
    results.append(verify_event(start, end, dtype, loc, desc))

print("\n\n🟢 === 对照期验证（预期：低风险）===")
for start, end, dtype, loc, desc in control_dates:
    results.append(verify_event(start, end, dtype, loc, desc))

# ── 汇总 ──
print("\n\n" + "=" * 78)
print("  验证汇总")
print("=" * 78)

event_results = results[:len(events)]
control_results = results[len(events):]

hits = sum(1 for r in event_results if "✅" in r["verdict"])
weak = sum(1 for r in event_results if "⚠️" in r["verdict"])
miss = sum(1 for r in event_results if "❌" in r["verdict"])

print(f"\n  事件期: {len(events)} 个事件")
print(f"    ✅ 命中: {hits}  |  ⚠️ 弱命中: {weak}  |  ❌ 未命中: {miss}")
print(f"    命中率: {hits}/{len(events)} ({hits/len(events)*100:.0f}%)")

print(f"\n  按灾害类型:")
for dtype in ["flash_flood", "dust_wind", "extreme_heat"]:
    sub = [r for r in event_results if r["type"] == dtype]
    sub_hits = sum(1 for r in sub if "✅" in r["verdict"])
    print(f"    {disaster_names[dtype]:<10s}: {sub_hits}/{len(sub)} 命中")

ctrl_hits = sum(1 for r in control_results if "✅" in r["verdict"])
print(f"\n  对照期: {len(control_results)} 个时段")
print(f"    误报(对照期命中): {ctrl_hits}  (越少越好)")

print(f"\n  {'日期':<22s} {'类型':<10s} {'高风险%':>8s} {'判定':<8s}  事件描述")
print(f"  {'─'*76}")
for r in results:
    flag = r['verdict'].split()[0] if r['verdict'] else '?'
    print(f"  {r['date_range']:<22s} {disaster_names.get(r['type'],r['type']):<10s} "
          f"{r['pct']:>7.1f}% {flag:<8s}  {r['event'][:42]}")

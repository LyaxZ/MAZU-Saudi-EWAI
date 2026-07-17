"""测试 SHAP 可解释性"""
import sys; sys.path.insert(0, ".")
from data.loader import load_to_dataframe
from models.inference import DisasterInference

engine = DisasterInference()

# 加载8月28日山洪高风险日期
feat_vars = [
    "daily_precip_total","daily_convective_precip","daily_large_scale_precip",
    "monthly_precip_total","ds10_max_1h","cape","cin","pwat","ivt","ivt_convergence",
    "rh2m","vpd_kpa","moisture_transport850","omega500","wind10_speed","wind925_speed",
    "wind850_speed","total_cloud_cover","net_radiation","orography","surface_pressure",
]
df = load_to_dataframe("2025-08-28", "2025-08-28", variables=feat_vars,
                       show_progress=False).fillna(0)

print("测试山洪 SHAP 解释 (2025-08-28, 真实山洪事件日)...")
result = engine.explain(df, "flash_flood")
print(f"  高风险格点: {result['n_high_total']:,}")
print(f"  SHAP采样: {result['n_samples_used']}")
print(f"  摘要: {result['summary']}")
print(f"  Top-5 特征贡献:")
for f in result["top_features"]:
    bar = "█" * int(f["contribution"] / 2)
    print(f"    {f['name']:<22s} {f['contribution']:>5.1f}% {bar}")

# 测试高温
print("\n测试极端高温 SHAP 解释 (2025-05-25, 52.2C破纪录)...")
heat_feats = [
    "tmax_anomaly_c","t2m_anomaly_c","tmax_climatology_c","t2m_c","tmax_c","tmin_c",
    "diurnal_temp_range_c","heat_index_c","apparent_temp_c","heat_stress_index",
    "vpd_kpa","rh2m","d2m_c","sw_net","lw_net","net_radiation","bowen_ratio",
    "total_cloud_cover","orography","surface_pressure",
]
df2 = load_to_dataframe("2025-05-25", "2025-05-25", variables=heat_feats,
                        show_progress=False).fillna(0)
result2 = engine.explain(df2, "extreme_heat")
print(f"  高风险格点: {result2['n_high_total']:,}")
print(f"  摘要: {result2['summary']}")
for f in result2["top_features"]:
    bar = "█" * int(f["contribution"] / 2)
    print(f"    {f['name']:<22s} {f['contribution']:>5.1f}% {bar}")

print("\n✅ SHAP 可解释性测试通过!")

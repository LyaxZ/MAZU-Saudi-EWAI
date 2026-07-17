"""测试时序趋势"""
import sys; sys.path.insert(0, ".")
from models.inference import DisasterInference
engine = DisasterInference()

trend = engine.predict_trend("2025-08-28", "flash_flood", lookback_days=7)
print(f"方向: {trend['direction']}  变化: {trend['change_pct']}%")
for d in trend["trend"]:
    bar = "#" * int(d["high_pct"])
    print(f"  {d['date']} mean_risk={d['mean_risk']:.4f} high={d['n_high']:,} ({d['high_pct']:.1f}%) {bar}")

"""验证推理引擎"""
import sys; sys.path.insert(0, ".")
from models.inference import DisasterInference
engine = DisasterInference()
for d in ["flash_flood", "extreme_heat", "dust_wind", "coastal_wave"]:
    r = engine.predict_from_nc("2025-08-15", d)
    print(f"{d:>14s}: n_high={r['n_high']:>7,} ({r['high_pct']:>5.1f}%) "
          f"mean={r['mean_risk']:.4f} max={r['max_risk']:.4f}")

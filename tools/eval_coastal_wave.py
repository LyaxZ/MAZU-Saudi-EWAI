import sys, numpy as np
sys.path.insert(0, '.')
from models.inference import DisasterInference
e = DisasterInference()

print('=== coastal_wave (15特征,含SST) ===')
for date in ['2025-08-28', '2025-09-15']:
    r = e.predict_from_nc(date, 'coastal_wave')
    n = r['n_high']
    m = r['mean_risk']
    print(date, ': 高风险=', n, ', 均值=', round(m, 4))

print()
print('=== 特征重要性 ===')
model = e.models['coastal_wave'].model
imp = model.feature_importance()
names = model.feature_name()
pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)
max_imp = max(imp)
for name, val in pairs:
    bar = '#' * int(val / max_imp * 40)
    print('  ', name.ljust(20), str(int(val)).rjust(8), bar)

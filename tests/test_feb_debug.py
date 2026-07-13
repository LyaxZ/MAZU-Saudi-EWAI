import sys; sys.path.insert(0,'.')
import numpy as np
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from config.model_config import FLASH_FLOOD_FEATURES

LV = ["flash_flood_risk","heatwave_day_flag","tmax_c","wind10_speed","rh2m","vpd_kpa","orography","ivt"]
AV = list(set(FLASH_FLOOD_FEATURES + LV))

ds = load_date_range("2025-01-01","2025-12-31",variables=AV,show_progress=True)
df = ds.to_dataframe().dropna()
df["month"] = df.index.get_level_values("day").month

print("逐月样本数:")
for m in range(1,13):
    count = len(df[df["month"]==m])
    print(f"  {m:2d}月: {count:>10,} 样本 ({count/len(df)*100:.1f}%)")
print(f"  全年: {len(df):,}")

# 专门查2月
feb = df[df["month"]==2]
print(f"\n2月: {len(feb):,} rows")

if len(feb) > 0:
    X = feb[FLASH_FLOOD_FEATURES]
    print(f"  特征shape: {X.shape}")
    b = DisasterLabelBuilder(dust_mode="standard",coastal_mode="standard")
    other = df[df["month"]!=2]
    b.fit(other)
    lb = b.build_all(feb)
    pos = lb["flash_flood_label"].mean()*100
    print(f"  pos率: {pos:.2f}%")
else:
    print("  2月数据为空! 排查原因...")
    # 不dropna看原始
    df_raw = ds.to_dataframe()
    feb_raw = df_raw[df_raw.index.get_level_values("day").month == 2]
    print(f"  原始(含NaN): {len(feb_raw):,}")
    for v in AV:
        n_nan = feb_raw[v].isna().sum()
        if n_nan > 0:
            print(f"    {v}: {n_nan} NaN")

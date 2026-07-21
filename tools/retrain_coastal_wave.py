"""GPU 笔记本 - coastal_wave SST重训脚本 (带缓存优化)"""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from data.loader import load_date_range
from data.label_builder import DisasterLabelBuilder
from models.lightgbm_model import LightGBMDisasterModel
from models.inference import _prepare_features
from config.model_config import DISASTER_FEATURES

TRAIN_START = '2025-06-01'
TRAIN_END = '2025-08-31'
dtype = 'coastal_wave'
CACHE_PATH = 'outputs/cache_coastal_train.parquet'

if os.path.exists(CACHE_PATH):
    print(f'从缓存加载: {CACHE_PATH}')
    df = pd.read_parquet(CACHE_PATH)
else:
    feats_all = list(set(f for fl in DISASTER_FEATURES.values() for f in fl))
    load_vars = list(set(v for v in feats_all + ['wind10_speed','rh2m','vpd_kpa','orography','ivt']
        if v not in ('lat_sin','lat_cos','lon_sin','lon_cos','sst_celsius')))
    print(f'加载主数据(92天, {len(load_vars)}变量)...')
    ds = load_date_range(TRAIN_START, TRAIN_END, variables=load_vars, show_progress=True)
    print('转为DataFrame...')
    df = ds.to_dataframe().fillna(0)
    print('加载并对齐 SST...')
    ds_sst = load_date_range(TRAIN_START, TRAIN_END, variables=['sst_celsius'], show_progress=False)
    sst_daily = ds_sst['sst_celsius'].mean(dim='time')
    sst_daily = sst_daily.rename({'lat':'latitude','lon':'longitude'})
    sst_aligned = sst_daily.interp(latitude=ds['latitude'], longitude=ds['longitude'], method='nearest')
    df['sst_celsius'] = sst_aligned.values.flatten()
    os.makedirs('outputs', exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f'数据已缓存至 {CACHE_PATH}')

sst_valid = df['sst_celsius'].notna().sum()
print(f'SST 有效值: {sst_valid:,}')

print('构建 coastal_wave 标签...')
builder = DisasterLabelBuilder(dust_mode='standard', coastal_mode='standard')
builder.fit(df)
labels = builder.build_coastal_wave(df)

X = _prepare_features(df, dtype)
y = labels.astype(int).values
print(f'特征: {X.shape[1]} 列, 正样本率: {y.mean()*100:.2f}%')

print(f'训练 {dtype} (CUDA GPU)...')
m = LightGBMDisasterModel(dtype, use_gpu=False)
m.fit(X, y)
os.makedirs('outputs/models', exist_ok=True)
m.save('outputs/models/coastal_wave.pkl')
print('coastal_wave 重训完成!')
print('特征数:', m.model.num_features())
print('特征名:', m.model.feature_name())

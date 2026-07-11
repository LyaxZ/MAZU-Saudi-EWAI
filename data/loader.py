"""
数据加载器：批量加载 NetCDF 指标文件，合并为统一 xarray.Dataset。

支持：
- 按日期范围加载
- 按变量名子集加载（节省内存）
- 按空间范围裁剪
- 进度条显示
"""

import os
import glob
from datetime import datetime, timedelta
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from config.settings import INDICATORS_DIR, INDICATOR_FILENAME_TEMPLATE


def _get_file_list(
    start_date: str, end_date: str
) -> List[str]:
    """根据日期范围获取已存在的 NetCDF 文件路径列表。

    Args:
        start_date: 起始日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'

    Returns:
        有序的文件路径列表
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    files = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        filename = INDICATOR_FILENAME_TEMPLATE.format(date=date_str)
        filepath = os.path.join(INDICATORS_DIR, filename)
        if os.path.exists(filepath):
            files.append(filepath)
        current += timedelta(days=1)

    if not files:
        raise FileNotFoundError(
            f"在 {INDICATORS_DIR} 中未找到 {start_date} 至 {end_date} 的指标文件"
        )
    return files


def _drop_scalar_coords(ds: xr.Dataset) -> None:
    """原地删除所有 0 维标量坐标，防止 to_dataframe() 报错。

    NetCDF 文件中的 step, surface, atmosphere 等标量坐标对分析无用，
    且会阻塞 xarray 的 MultiIndex 构建。
    """
    to_drop = []
    for coord_name in ds.coords:
        if ds.coords[coord_name].ndim == 0:
            to_drop.append(coord_name)
    if to_drop:
        ds.drop_vars(to_drop, errors="ignore")


def load_single_day(
    date: str, variables: Optional[List[str]] = None
) -> xr.Dataset:
    """加载单日的指标数据。

    Args:
        date: 日期，格式 'YYYY-MM-DD' 或 'YYYYMMDD'
        variables: 要加载的变量名列表，None 表示加载全部

    Returns:
        xarray.Dataset
    """
    date_str = pd.Timestamp(date).strftime("%Y%m%d")
    filename = INDICATOR_FILENAME_TEMPLATE.format(date=date_str)
    filepath = os.path.join(INDICATORS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    ds = xr.open_dataset(filepath)

    if variables is not None:
        # 只保留存在的变量
        available = [v for v in variables if v in ds.data_vars]
        missing = set(variables) - set(available)
        if missing:
            print(f"警告：以下变量不存在，已跳过: {missing}")
        ds = ds[available]

    # 添加日期坐标
    ds = ds.assign_coords(day=pd.Timestamp(date_str))

    return ds


def load_date_range(
    start_date: str,
    end_date: str,
    variables: Optional[List[str]] = None,
    show_progress: bool = True,
) -> xr.Dataset:
    """加载指定日期范围内的所有指标数据，沿时间轴合并。

    合并策略：
    - 大部分变量为 (latitude, longitude)，沿新增的 `day` 维度堆叠
    - SST 等含 `time` 维度的变量沿 `day` 维度堆叠

    Args:
        start_date: 起始日期
        end_date: 结束日期
        variables: 要加载的变量名列表，None 表示加载全部
        show_progress: 是否显示进度条

    Returns:
        xarray.Dataset，新增 `day` 维度
    """
    files = _get_file_list(start_date, end_date)
    datasets = []

    iterator = tqdm(files, desc="加载指标文件") if show_progress else files
    for fp in iterator:
        try:
            ds = xr.open_dataset(fp)
            if variables is not None:
                available = [v for v in variables if v in ds.data_vars]
                ds = ds[available]
            # 提取日期
            date_str = os.path.basename(fp).replace("saudi_indicators_", "").replace(".nc", "")
            ds = ds.assign_coords(day=pd.Timestamp(date_str))
            datasets.append(ds)
        except Exception as e:
            print(f"警告：加载 {fp} 失败: {e}")

    if not datasets:
        raise RuntimeError("未能加载任何文件")

    # 沿 day 维度合并
    # 使用 compat='override' + coords='minimal' 处理不同日期文件间
    # 坐标变量不一致的问题（如 atmosphereSingleLayer 等标量坐标）
    merged = xr.concat(datasets, dim="day", compat="override", coords="minimal")

    # 清理 0 维标量坐标（step, surface, atmosphere 等），否则 to_dataframe() 会报错
    _drop_scalar_coords(merged)

    return merged


def load_to_dataframe(
    start_date: str,
    end_date: str,
    variables: Optional[List[str]] = None,
    lat_slice: Optional[tuple] = None,
    lon_slice: Optional[tuple] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """加载指标数据并转换为 pandas DataFrame（适合输入 LightGBM）。

    DataFrame 列：
    - day: 日期
    - lat_idx, lon_idx: 网格索引
    - latitude, longitude: 实际经纬度
    - 各类指标变量列

    Args:
        start_date: 起始日期
        end_date: 结束日期
        variables: 变量名列表
        lat_slice: 纬度范围 (min, max)，None 表示全部
        lon_slice: 经度范围 (min, max)，None 表示全部
        show_progress: 是否显示进度条

    Returns:
        pd.DataFrame，行为 (day, lat, lon) 组合
    """
    ds = load_date_range(start_date, end_date, variables, show_progress)

    # 空间裁剪
    if lat_slice is not None:
        ds = ds.sel(latitude=slice(lat_slice[1], lat_slice[0]))  # latitude 降序
    if lon_slice is not None:
        ds = ds.sel(longitude=slice(lon_slice[0], lon_slice[1]))

    # 转换为 DataFrame
    df = ds.to_dataframe().reset_index()
    return df


def get_variable_list(date: str = "2025-01-01") -> List[str]:
    """获取某个日期的完整变量名列表。

    Args:
        date: 参考日期

    Returns:
        变量名列表
    """
    ds = load_single_day(date)
    return list(ds.data_vars.keys())


def get_variable_info(date: str = "2025-01-01") -> pd.DataFrame:
    """获取变量信息表（名称、维度、形状）。

    Args:
        date: 参考日期

    Returns:
        DataFrame，列：variable, dims, shape
    """
    ds = load_single_day(date)
    records = []
    for v in ds.data_vars:
        records.append({
            "variable": v,
            "dims": str(list(ds[v].dims)),
            "shape": ds[v].shape,
        })
    return pd.DataFrame(records)

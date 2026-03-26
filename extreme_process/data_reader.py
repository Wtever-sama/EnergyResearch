"""
data_reader.py

快速读取 NetCDF (.nc) / NumPy (.npy) 文件并在控制台打印摘要（方便快速检查文件头）。

用法示例 (PowerShell):
python f:\code_program\EnergyResearch\extreme_process\data_reader.py --pattern "G:\extreme_analysis\data\CMIP6_Research_Data\*.nc"
python f:\code_program\EnergyResearch\extreme_process\data_reader.py --pattern "G:\extreme_analysis\data\CMIP6_Research_Data\*.npy"

如果不提供 --pattern，脚本将在当前目录查找 "*.nc" 文件。
"""

import xarray as xr
import glob
import os
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_nc_head(file_path: str, n_lines: int = 20):
    """打开一个 NetCDF 文件，打印 xarray 对象的前 n_lines 行描述。"""
    try:
        # 使用上下文管理器确保打开后在退出时关闭，且在关闭前完成所有读取操作
        with xr.open_dataset(file_path) as ds:
            rep = ds.__repr__()
            lines = rep.splitlines()
            header = lines[:n_lines]

            print(f"\n--- {os.path.basename(file_path)} ---")
            for ln in header:
                print(ln)

            # 追加：针对每个 data variable 打印描述性统计（安全计算）
            def _safe_reduce(da, func_name):
                try:
                    # Prefer xarray reduction over numpy to allow dask-backed arrays to reduce chunkwise
                    if hasattr(da, func_name):
                        # reduce across all dims
                        try:
                            res = getattr(da, func_name)(dim=None, skipna=True)
                        except TypeError:
                            # some reductions (older xarray) may not accept dim=None
                            res = getattr(da, func_name)()
                    else:
                        # fallback to numpy on in-memory values
                        res = getattr(np, func_name)(da.values)

                    # res may be xarray.DataArray, numpy scalar, or dask-backed scalar
                    try:
                        # For xarray/dask results, compute scalar safely
                        if hasattr(res, 'compute'):
                            scalar = res.compute()
                        else:
                            scalar = res
                        if hasattr(scalar, 'values'):
                            scalar = scalar.values
                        if hasattr(scalar, 'item'):
                            return scalar.item()
                        return float(scalar)
                    except Exception:
                        # as a last resort, try to access .values then item
                        try:
                            val = res.values
                            if hasattr(val, 'item'):
                                return val.item()
                            return float(val)
                        except Exception:
                            pass
                except Exception:
                    pass

                # fallback to sampling small chunk to approximate
                try:
                    dims = da.dims
                    if len(dims) > 0:
                        dim0 = dims[0]
                        size0 = da.sizes.get(dim0, 0)
                        sample_n = min(1000, max(1, size0))
                        sample = da.isel({dim0: slice(0, sample_n)}).values.ravel()
                    else:
                        sample = da.values.ravel()
                    if sample.size == 0:
                        return None
                    sample = sample[:min(100000, sample.size)]
                    if func_name == 'mean':
                        return float(np.nanmean(sample))
                    if func_name == 'min':
                        return float(np.nanmin(sample))
                    if func_name == 'max':
                        return float(np.nanmax(sample))
                    if func_name == 'std':
                        return float(np.nanstd(sample))
                except Exception:
                    return None

            def _safe_missing_prop(da):
                # 返回缺失值占比（0..1），在无法完全扫描时使用采样估计
                try:
                    # 尝试使用 xarray 的按维度归约，这对 dask-backed DataArray 是安全的
                    total = int(da.size)
                    try:
                        miss = da.isnull().sum(dim=None, skipna=True)
                    except TypeError:
                        miss = da.isnull().sum()
                    # 如果是 dask/xarray 对象，安全 compute
                    if hasattr(miss, 'compute'):
                        miss = miss.compute()
                    # 取出标量值
                    if hasattr(miss, 'values'):
                        missing = int(miss.values)
                    else:
                        missing = int(miss)
                    return missing / total if total > 0 else None
                except Exception:
                    # fallback to sampling small chunk to approximate
                    try:
                        dims = da.dims
                        if len(dims) > 0:
                            dim0 = dims[0]
                            size0 = da.sizes.get(dim0, 0)
                            sample_n = min(1000, max(1, size0))
                            sample = da.isel({dim0: slice(0, sample_n)}).values.ravel()
                        else:
                            sample = da.values.ravel()
                        if sample.size == 0:
                            return None
                        sample = sample[:min(100000, sample.size)]
                        n_missing = int(np.count_nonzero(np.isnan(sample)))
                        return float(n_missing) / float(sample.size)
                    except Exception:
                        return None

            print('\nData variables statistics:')
            for var in ds.data_vars:
                da = ds[var]
                print(f"\n> {var}: shape={da.shape}, dims={da.dims}, dtype={getattr(da, 'dtype', 'unknown')}")
                v_mean = _safe_reduce(da, 'mean')
                v_min = _safe_reduce(da, 'min')
                v_max = _safe_reduce(da, 'max')
                v_std = _safe_reduce(da, 'std')
                # median via sampling to avoid heavy compute
                try:
                    # prefer xarray median/quantile to support dask-backed arrays
                    if hasattr(da, 'median'):
                        try:
                            med = da.median(dim=None, skipna=True)
                        except TypeError:
                            med = da.median()
                        if hasattr(med, 'compute'):
                            med = med.compute()
                        v_median = float(getattr(med, 'values', med))
                    else:
                        # fallback to numpy/sample
                        flat = da.values.ravel()
                        if flat.size > 0:
                            v_median = float(np.nanmedian(flat[:min(100000, flat.size)]))
                        else:
                            v_median = None
                except Exception:
                    v_median = None

                # missing proportion (0..1)
                v_missing_prop = _safe_missing_prop(da)
                # missing count where possible
                try:
                    if v_missing_prop is not None and int(da.size) > 0:
                        n_missing = int(round(v_missing_prop * int(da.size)))
                    else:
                        n_missing = None
                except Exception:
                    n_missing = None

                if v_mean is not None:
                    print(f"   - mean: {v_mean}")
                if v_min is not None and v_max is not None:
                    print(f"   - range: {v_min} to {v_max}")
                if v_median is not None:
                    print(f"   - median(sample): {v_median}")
                if v_std is not None:
                    print(f"   - std: {v_std}")
                if n_missing is not None:
                    print(f"   - missing (NaN) count (est.): {n_missing}")
                if v_missing_prop is not None:
                    print(f"   - missing proportion (est.): {v_missing_prop:.4f} ({v_missing_prop*100:.2f}%)")

                # 前后样本（沿第一个维度），但只展示最多 10 个标量值以避免输出大量数据
                try:
                    dims = da.dims
                    if len(dims) > 0:
                        dim0 = dims[0]
                        size0 = da.sizes.get(dim0, 0)
                        head = da.isel({dim0: slice(0, min(5, size0))}).values
                        tail = da.isel({dim0: slice(max(size0-5,0), size0)}).values
                        # 扁平化并截断显示
                        head_flat = np.asarray(head).ravel()
                        tail_flat = np.asarray(tail).ravel()
                        def _lim_list(arr):
                            if arr.size == 0:
                                return [], 0
                            max_show = 10
                            shown = arr[:max_show].tolist()
                            more = arr.size - len(shown)
                            return shown, more

                        shown_h, more_h = _lim_list(head_flat)
                        shown_t, more_t = _lim_list(tail_flat)
                        more_h_str = f" ...(+{more_h} more)" if more_h>0 else ""
                        more_t_str = f" ...(+{more_t} more)" if more_t>0 else ""
                        print(f"   - head sample (along {dim0}, up to 10 vals): {shown_h}{more_h_str}")
                        print(f"   - tail sample (along {dim0}, up to 10 vals): {shown_t}{more_t_str}")
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"无法读取文件 {file_path}: {e}")


def print_npy_head(file_path: str, n_lines: int = 20):
    """读取一个 NPY 文件并打印数组摘要与样本值。"""
    try:
        arr = np.load(file_path, allow_pickle=True)

        print(f"\n--- {os.path.basename(file_path)} ---")
        print(f"type: {type(arr)}")

        # 常规 ndarray
        if isinstance(arr, np.ndarray):
            print(f"shape: {arr.shape}")
            print(f"dtype: {arr.dtype}")
            print(f"ndim: {arr.ndim}")
            print(f"size: {arr.size}")

            # object 数组可能包含字典等复杂对象
            if arr.dtype == object:
                flat_obj = arr.ravel()
                show_n = min(n_lines, flat_obj.size)
                print(f"object array sample (first {show_n}):")
                for i in range(show_n):
                    print(f"  [{i}] {repr(flat_obj[i])}")
                return

            # 数值数组统计
            if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
                finite_arr = arr.astype(np.float64, copy=False)
                print(f"mean: {float(np.nanmean(finite_arr))}")
                print(f"min: {float(np.nanmin(finite_arr))}")
                print(f"max: {float(np.nanmax(finite_arr))}")
                print(f"std: {float(np.nanstd(finite_arr))}")

                if arr.size > 0:
                    missing = int(np.count_nonzero(np.isnan(finite_arr)))
                    miss_prop = missing / arr.size
                    print(f"missing (NaN) count: {missing}")
                    print(f"missing proportion: {miss_prop:.4f} ({miss_prop*100:.2f}%)")

            # 打印首尾样本（最多 n_lines 个）
            flat = arr.ravel()
            show_n = min(n_lines, flat.size)
            head_vals = flat[:show_n].tolist()
            tail_vals = flat[-show_n:].tolist() if flat.size > 0 else []
            print(f"head sample (first {show_n}): {head_vals}")
            print(f"tail sample (last {show_n}): {tail_vals}")
            return

        # 其他类型兜底输出
        print(repr(arr))
    except Exception as e:
        logger.error(f"无法读取文件 {file_path}: {e}")


def check_sticc_txt(file_path: str):
    import pandas as pd

    # 读取生成的 txt 文件（因为没有表头，所以 header=None）
    df = pd.read_csv(file_path, header=None)

    print(f"总列数: {df.shape[1]}")

    # 1. 检查第0列是否为 ID (应该是从 0 开始的整数)
    print("前5行的 ID (第0列):", df.iloc[:5, 0].values)

    # 2. 检查属性列 (例如第100列，应该是一个浮点数，即 SolarCF 值)
    print("前5行的属性值 (第100列):", df.iloc[:5, 100].values)

    # 3. 检查邻居列（比如第1001列，这应该是一个代表邻居网格点的整数 ID）
    # 注意：如果这里输出的是浮点数，说明设置的 start 偏小了，读到了属性列里
    print("前5行的邻居索引 (第1000列):", df.iloc[:5, 1000].values)
    print("前5行的邻居索引 (第1001列):", df.iloc[:5, 1001].values)


def main():
    parser = argparse.ArgumentParser(description='Print summary for NetCDF(.nc) and NumPy(.npy) files.')
    parser.add_argument('--pattern', '-p', default='*.nc', help='文件匹配模式或单个文件路径 (示例: *.nc, *.npy)')
    parser.add_argument('--lines', '-n', type=int, default=20, help='要打印的行数 (默认: 20)')

    args = parser.parse_args()
    pattern = args.pattern

    # 如果是单文件路径且存在，直接使用；否则当作 glob 模式
    files = []
    if os.path.exists(pattern) and os.path.isfile(pattern):
        files = [pattern]
    else:
        files = glob.glob(pattern)

    if not files:
        logger.warning(f"未找到匹配文件: {pattern}")
        return

    supported_ext = {'.nc', '.npy', '.txt'}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in supported_ext:
            logger.warning(f"跳过不支持的文件类型: {f}")
            continue

        if ext == '.nc':
            print_nc_head(f, n_lines=args.lines)
        elif ext == '.npy':
            print_npy_head(f, n_lines=args.lines)
        elif ext == '.txt':
            check_sticc_txt(f)  # "G:\extreme_analysis\sticc_dataset\ssp126\SolarCF\sticc_final_input.txt"


if __name__ == '__main__':
    main()

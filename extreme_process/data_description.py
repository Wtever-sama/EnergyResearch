'''
读取NC文件的基本信息, 输出到logs/文件名_data_description.txt
'''
import xarray as xr
import os
import glob
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


def _format_bytes(num):
    """Human readable file size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def _safe_reduce(arr_like, func_name):
    """Try to compute a reduction safely; return None on failure."""
    try:
        if hasattr(arr_like, func_name):
            val = getattr(arr_like, func_name)().values
        else:
            # fallback to numpy
            val = getattr(np, func_name)(arr_like)
        # convert numpy types to python scalars
        if hasattr(val, 'item'):
            return val.item()
        return val
    except Exception:
        return None

def inspect_nc_file(file_path, log_dir="logs"):
    """
    读取NC文件基本信息并输出到描述性文本文件中
    具有良好的兼容性，适用于各类能源气象数据
    """
    # 1. 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_name = os.path.basename(file_path)
    # 去除后缀
    file_name_no_ext = os.path.splitext(file_name)[0]
    log_file = os.path.join(log_dir, f"{file_name_no_ext}_data_description.txt")

    try:
        # 2. 使用 xarray 打开数据集 (使用 netcdf4 引擎)
        # decode_times=True 会自动解析时间轴
        ds = xr.open_dataset(file_path, engine="netcdf4")

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"--- NC 文件描述报告 ---\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"文件名: {file_name}\n")
            f.write(f"文件路径: {file_path}\n")
            f.write(f"文件大小: {_format_bytes(file_size)} ({file_size} bytes)\n")
            f.write("-" * 30 + "\n\n")

            # 3. 基础维度信息 (Dimensions)
            f.write("[1. 维度信息 (Dimensions)]\n")
            for dim, size in ds.dims.items():
                f.write(f" - {dim}: {size}\n")
            f.write("\n")

            # 4. 坐标信息 (Coordinates) - 包含经纬度范围和时间范围
            f.write("[2. 坐标范围 (Coordinates Info)]\n")
            for coord in ds.coords:
                try:
                    c = ds[coord]
                    c_min = _safe_reduce(c, 'min')
                    c_max = _safe_reduce(c, 'max')
                    dtype = str(c.dtype) if hasattr(c, 'dtype') else '未知'
                    f.write(f" - {coord}: dtype={dtype}; 从 {c_min} 到 {c_max}\n")

                    # 时间轴的额外信息
                    if 'time' in coord.lower() or coord == 'time':
                        try:
                            times = pd.to_datetime(c.values)
                            if len(times) > 1:
                                freq = pd.infer_freq(times)
                            else:
                                freq = None
                            f.write(f"   - time count={len(times)}; sample start={times[0]}; end={times[-1]}; inferred_freq={freq}\n")
                        except Exception:
                            pass
                except Exception:
                    f.write(f" - {coord}: 无法读取\n")
            f.write("\n")

            # 5. 变量信息 (Data Variables)
            f.write("[3. 数据变量 (Data Variables)]\n")
            for var in ds.data_vars:
                attrs = ds[var].attrs
                f.write(f" > 变量名: {var}\n")
                da = ds[var]
                f.write(f"   - 数据形状: {da.shape}\n")
                f.write(f"   - 维度组合: {da.dims}\n")
                f.write(f"   - dtype: {getattr(da, 'dtype', '未知')}\n")
                f.write(f"   - 单位 (units): {attrs.get('units', '未知')}\n")
                f.write(f"   - 含义 (long_name): {attrs.get('long_name', '未定义')}\n")
                
                # 统计摘要 (仅针对数值型变量)
                try:
                    v_mean = _safe_reduce(da, 'mean')
                    v_min = _safe_reduce(da, 'min')
                    v_max = _safe_reduce(da, 'max')
                    v_std = _safe_reduce(da, 'std')
                    # median may not be available; try via numpy on small sample
                    try:
                        v_median = np.median(da.values.ravel())
                    except Exception:
                        v_median = None

                    if v_mean is not None:
                        f.write(f"   - 全局平均值: {v_mean}\n")
                    if v_min is not None and v_max is not None:
                        f.write(f"   - 值域: {v_min} 到 {v_max}\n")
                    if v_median is not None:
                        f.write(f"   - 中位数: {v_median}\n")
                    if v_std is not None:
                        f.write(f"   - 标准差: {v_std}\n")
                except Exception:
                    pass

                # 缺失值统计
                try:
                    total_count = da.size
                    # use isnull then sum (safe)
                    n_missing = int(da.isnull().sum().values) if hasattr(da, 'isnull') else 0
                    f.write(f"   - 总元素数量: {int(total_count)}; 缺失值数量: {n_missing}\n")
                except Exception:
                    pass

                # encoding/_FillValue/scale/offset/compression info
                try:
                    enc = da.encoding if hasattr(da, 'encoding') else {}
                    if enc:
                        f.write(f"   - encoding: {enc}\n")
                    for key in ['_FillValue', 'scale_factor', 'add_offset']:
                        if key in attrs:
                            f.write(f"   - {key}: {attrs.get(key)}\n")
                except Exception:
                    pass

                # 采样：写入前10个值和尾10个值（尽量避免一次性读取超大数组）
                try:
                    # 优先尝试扁平化读取小样本
                    flat = ds[var].values.ravel()
                    n = flat.shape[0]
                    first_n = flat[:10].tolist()
                    last_n = flat[-10:].tolist() if n > 0 else []
                    f.write(f"   - 前{min(10, n)}值样本: {first_n}\n")
                    f.write(f"   - 尾{min(10, n)}值样本: {last_n}\n")
                except Exception:
                    # 回退：沿第一个维度取样（更节省内存的方式）
                    try:
                        dims = ds[var].dims
                        if len(dims) > 0:
                            dim0 = dims[0]
                            size0 = ds.dims[dim0]
                            head_count = min(10, size0)
                            tail_start = max(size0 - 10, 0)
                            head_sample = ds[var].isel({dim0: slice(0, head_count)}).values
                            tail_sample = ds[var].isel({dim0: slice(tail_start, size0)}).values
                            f.write(f"   - 前{head_count}沿'{dim0}'维度的样本: {np.array(head_sample).tolist()}\n")
                            f.write(f"   - 尾{min(10, size0)}沿'{dim0}'维度的样本: {np.array(tail_sample).tolist()}\n")
                        else:
                            f.write("   - 无法读取样本值（无维度信息）\n")
                    except Exception:
                        f.write("   - 无法读取样本值（读取过程中出错）\n")

                f.write("\n")

            # 6. 全局属性 (Global Attributes)
            f.write("[4. 全局元数据 (Global Attributes)]\n")
            for attr_name, attr_val in ds.attrs.items():
                f.write(f" - {attr_name}: {attr_val}\n")

        print(f"成功完成! 描述文件已保存至: {log_file}")
        try:
            ds.close()
        except Exception:
            pass

    except Exception as e:
        print(f"读取文件时出错: {e}")


def inspect_npy_file(file_path, log_dir="logs"):
    '''
    读取NPY文件并做数据描述性统计
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    log_file = os.path.join(log_dir, f"{file_name_no_ext}_data_description.txt")

    try:
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"--- NPY/NPZ 文件描述报告 ---\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"文件名: {file_name}\n")
            f.write(f"文件路径: {file_path}\n")
            f.write(f"文件大小: {_format_bytes(file_size)} ({file_size} bytes)\n")
            f.write('-' * 30 + "\n\n")

            # 区分 npy 与 npz
            if file_path.lower().endswith('.npz'):
                f.write('[Archive: .npz]\n')
                try:
                    archive = np.load(file_path, mmap_mode='r')
                    keys = archive.files
                    f.write(f" 包含数组键: {keys}\n\n")
                except Exception as e:
                    f.write(f" 无法读取 npz 存档: {e}\n")
                    print(f"读取文件时出错: {e}")
                    return

                for key in keys:
                    arr = archive[key]
                    f.write(f" > 数组: {key}\n")
                    f.write(f"   - shape: {getattr(arr, 'shape', '未知')}\n")
                    f.write(f"   - dtype: {getattr(arr, 'dtype', '未知')}\n")

                    # 统计摘要
                    v_mean = _safe_reduce(arr, 'mean')
                    v_min = _safe_reduce(arr, 'min')
                    v_max = _safe_reduce(arr, 'max')
                    v_std = _safe_reduce(arr, 'std')
                    try:
                        flat = arr.ravel()
                        # 尽量用小样本计算中位数，避免全量排序
                        sample_for_median = flat[:1000]
                        v_median = np.median(sample_for_median) if sample_for_median.size > 0 else None
                    except Exception:
                        v_median = None

                    if v_mean is not None:
                        f.write(f"   - mean: {v_mean}\n")
                    if v_min is not None and v_max is not None:
                        f.write(f"   - range: {v_min} to {v_max}\n")
                    if v_median is not None:
                        f.write(f"   - median(sample): {v_median}\n")
                    if v_std is not None:
                        f.write(f"   - std: {v_std}\n")

                    # 缺失值统计（仅浮点数有效）
                    try:
                        if np.issubdtype(arr.dtype, np.floating):
                            n_missing = int(np.count_nonzero(np.isnan(arr)))
                        else:
                            n_missing = 0
                        f.write(f"   - total elements: {arr.size}; missing (NaN): {n_missing}\n")
                    except Exception:
                        pass

                    # 前/尾样本
                    try:
                        flat = arr.ravel()
                        n = flat.size
                        first_n = flat[:10].tolist() if n > 0 else []
                        last_n = flat[-10:].tolist() if n > 0 else []
                        f.write(f"   - first{min(10,n)} sample: {first_n}\n")
                        f.write(f"   - last{min(10,n)} sample: {last_n}\n")
                    except Exception:
                        f.write("   - 无法读取样本值\n")

                    f.write('\n')

                try:
                    archive.close()
                except Exception:
                    pass

            else:
                # 单个 .npy 数组
                try:
                    arr = np.load(file_path, mmap_mode='r')
                except Exception as e:
                    f.write(f"无法读取 NPY 文件: {e}\n")
                    print(f"读取文件时出错: {e}")
                    return

                f.write('[Array: .npy]\n')
                f.write(f" - shape: {getattr(arr, 'shape', '未知')}\n")
                f.write(f" - dtype: {getattr(arr, 'dtype', '未知')}\n")

                v_mean = _safe_reduce(arr, 'mean')
                v_min = _safe_reduce(arr, 'min')
                v_max = _safe_reduce(arr, 'max')
                v_std = _safe_reduce(arr, 'std')
                try:
                    flat = arr.ravel()
                    sample_for_median = flat[:1000]
                    v_median = np.median(sample_for_median) if sample_for_median.size > 0 else None
                except Exception:
                    v_median = None

                if v_mean is not None:
                    f.write(f" - mean: {v_mean}\n")
                if v_min is not None and v_max is not None:
                    f.write(f" - range: {v_min} to {v_max}\n")
                if v_median is not None:
                    f.write(f" - median(sample): {v_median}\n")
                if v_std is not None:
                    f.write(f" - std: {v_std}\n")

                try:
                    if np.issubdtype(arr.dtype, np.floating):
                        n_missing = int(np.count_nonzero(np.isnan(arr)))
                    else:
                        n_missing = 0
                    f.write(f" - total elements: {arr.size}; missing (NaN): {n_missing}\n")
                except Exception:
                    pass

                try:
                    n = arr.ravel().size
                    first_n = arr.ravel()[:10].tolist() if n > 0 else []
                    last_n = arr.ravel()[-10:].tolist() if n > 0 else []
                    f.write(f" - first{min(10,n)} sample: {first_n}\n")
                    f.write(f" - last{min(10,n)} sample: {last_n}\n")
                except Exception:
                    f.write(" - 无法读取样本值\n")

        print(f"成功完成! 描述文件已保存至: {log_file}")

    except Exception as e:
        print(f"读取文件时出错: {e}")


if __name__ == "__main__":
    # data_name = r"G:\extreme_analysis\data\CMIP6_QDM_MME\*.nc"
    data_name = r"G:\extreme_analysis\results\Wind\*.nc"
    files = glob.glob(pathname=data_name)
    for f in files:
        inspect_nc_file(f)
        # inspect_npy_file(f)
''' 从极端情况的持续时间判断是否属于极端事件: > 12h 属于极端低可靠性事件 > 100h 属于极端长期事件 '''

import xarray as xr
import numpy as np
import os
import logging
import time
from tqdm import tqdm

# basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def identify_continuous_events(da_v1, min_steps):
    """
    识别连续极端时段并按持续时间过滤
    da_v1: 0-1 标识的 DataArray (包含 NaN 掩码)
    min_steps: 最小持续时间步数 (12h=4步, 100h=34步)
    """
    # 对每个空间格点沿 time 轴应用一个 numpy 1D 函数，计算属于持续 >= min_steps 的位置
    logger.debug(f"开始 identify_continuous_events (apply_ufunc 方式): min_steps={min_steps}; data shape={getattr(da_v1, 'shape', 'unknown')}")
    start_ts = time.time()

    def _mark_long_runs_1d(arr, min_steps=1):
        """输入 1D numpy 数组（含 NaN），返回 0/1 数组，位置属于长度>=min_steps 的连续1段标记为1。"""
        # 将 NaN 当作 0 处理
        try:
            # 若为 masked or non-numeric, coerce
            a = np.asarray(arr)
        except Exception:
            return np.zeros_like(arr, dtype=np.int8)

        # 空或长度为0
        if a.size == 0:
            return np.zeros_like(a, dtype=np.int8)

        is_one = np.ones_like(a, dtype=np.int8)
        # treat values equal to 1 as event, others (including nan) as 0
        # use numpy comparison that tolerates nan
        with np.errstate(invalid='ignore'):
            is_one = np.where(a == 1, 1, 0).astype(np.int8)

        # pad and find run starts/ends
        padded = np.concatenate(([0], is_one, [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        out = np.zeros_like(is_one, dtype=np.int8)
        for s, e in zip(starts, ends):
            length = e - s
            if length >= min_steps:
                out[s:e] = 1
        return out

    # 使用 xarray.apply_ufunc 在除 time 轴外的位置上并行映射该函数
    try:
        filtered_v1 = xr.apply_ufunc(
            _mark_long_runs_1d,
            da_v1,
            kwargs={'min_steps': int(min_steps)},
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.int8],
        )
        # restore mask: 保留原始 NaN
        filtered_v1 = filtered_v1.where(da_v1.notnull())
    except Exception as e:
        logger.error(f"apply_ufunc 处理失败，回退到内存密集方法: {e}")
        # 回退：尽量用原有逻辑但谨慎地尝试（可能仍然内存密集）
        mask = da_v1.fillna(0).astype(int)
        is_diff = mask.diff(dim='time', label='upper') != 0
        block_id = is_diff.cumsum(dim='time')
        event_blocks = block_id.where(mask == 1)
        counts = event_blocks.groupby(event_blocks).transform('count')
        filtered_v1 = xr.where(counts >= min_steps, 1, 0).where(da_v1.notnull())

    # 调试信息：统计满足阈值的时间步总数与近似事件计数（上升沿计数）
    try:
        total_qualifying_steps = int(filtered_v1.sum(dim='time').values.sum())
        # 近似事件数：统计从0到1的上升沿次数
        rise = filtered_v1.fillna(0).diff(dim='time', label='upper') == 1
        approx_events = int(rise.sum(dim='time').values.sum())
        logger.info(f"识别到满足 min_steps={min_steps} 的总时间步 (全域汇总): {total_qualifying_steps}; 近似事件数: {approx_events}")
    except Exception:
        logger.debug("无法计算事件统计信息（可能是大数据或懒加载），将跳过统计输出")

    elapsed = time.time() - start_ts
    logger.info(f"identify_continuous_events 完成 (min_steps={min_steps})，耗时 {elapsed:.1f}s")

    return filtered_v1.where(da_v1.notnull()).rename("event_flag")

def main():
    # 路径配置
    results_dir = r"G:\extreme_analysis\results"
    scenarios = ["ssp126", "ssp245", "ssp585"]
    time_res_hours = 3 # 3小时分辨率
    
    # 阈值定义 (小时 -> 步数)
    reliability_threshold = int(12 / time_res_hours)    # 4步
    long_duration_threshold = int(100 / time_res_hours) # 34步

    for scn in scenarios:
        for energy_type in tqdm(["solar", "wind"], desc=f"scn={scn}"):
            # 动态匹配文件名
            results_dir_with_energy_type = os.path.join(results_dir, energy_type)
            prefix = "solar" if energy_type == "solar" else "wind"
            file_name = f"{prefix}_{scn}_V1_extreme_flag_2040-2060.nc"
            input_path = os.path.join(results_dir_with_energy_type, file_name)

            if not os.path.exists(input_path):
                logger.warning(f"跳过不存在的文件: {input_path}")
                continue

            logger.info(f"正在处理 {scn} {energy_type} -> {input_path}")
            file_size = os.path.getsize(input_path)
            logger.info(f"输入文件大小: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
            t0 = time.time()
            try:
                ds = xr.open_dataset(input_path)
                if 'is_extreme' not in ds.data_vars:
                    logger.error(f"文件中未找到 'is_extreme' 变量: {input_path}; 可用变量: {list(ds.data_vars)}")
                    ds.close()
                    continue
                da_v1 = ds['is_extreme']
                logger.info(f"打开数据集完成: shape={getattr(da_v1,'shape', 'unknown')}; chunks={getattr(da_v1.data, 'chunks', 'unknown')}")
            except Exception as e:
                logger.error(f"打开文件失败: {input_path}: {e}")
                continue

            # 1. 识别 12h+ 极端低可靠性事件
            v1_12h = identify_continuous_events(da_v1, reliability_threshold)
            out_12h = os.path.join(results_dir, energy_type, f"{energy_type}_{scn}_V1_Reliability_12h.nc")
            t1 = time.time()
            try:
                v1_12h.to_netcdf(out_12h)
                logger.info(f"保存完成: {out_12h} (耗时 {time.time()-t1:.1f}s)")
            except Exception as e:
                logger.error(f"保存失败: {out_12h}: {e}")

            # 2. 识别 100h+ 极端长期事件
            v1_100h = identify_continuous_events(da_v1, long_duration_threshold)
            out_100h = os.path.join(results_dir, energy_type, f"{energy_type}_{scn}_V1_LongDuration_100h.nc")
            t2 = time.time()
            try:
                v1_100h.to_netcdf(out_100h)
                logger.info(f"保存完成: {out_100h} (耗时 {time.time()-t2:.1f}s)")
            except Exception as e:
                logger.error(f"保存失败: {out_100h}: {e}")

            # 关闭 dataset
            try:
                ds.close()
            except Exception:
                pass
            logger.info(f"处理 {scn} {energy_type} 完成，整个文件处理耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

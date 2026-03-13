import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import dask

def plot_daily_climatological_profiles(
    delta_x_path,
    cluster_map_path,
    p10_path,
    start_year='2040',
    end_year='2060',
    year_block=5,
):
    print("正在加载数据并执行日平均预处理...")
    
    # 1. 加载数据 (启用 Dask 延迟加载)
    ds_delta = xr.open_dataset(delta_x_path, chunks={'time': 10000})
    ds_clusters = xr.open_dataset(cluster_map_path)
    ds_p10 = xr.open_dataset(p10_path)

    # 仅保留分析所需时段，减少I/O和内存压力
    ds_delta = ds_delta.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    
    # 2. 还原容量因子 CF = delta_x + p10
    da_cf = ds_delta.delta_x + ds_p10.p10
    
    # 3. 按 5 年时间块进行日平均并累加，避免一次性读取全时段
    print(f"  -> 步骤1: 按 {year_block} 年分块计算日平均并聚合多年气候态...")
    all_doys = np.arange(1, 367, dtype=np.int16)
    total_sum = None
    total_count = None

    start_year_int = int(start_year)
    end_year_int = int(end_year)

    for chunk_start in range(start_year_int, end_year_int + 1, year_block):
        chunk_end = min(chunk_start + year_block - 1, end_year_int)
        print(f"     - 正在处理时间块: {chunk_start}-{chunk_end}")
        da_cf_chunk = da_cf.sel(time=slice(f"{chunk_start}-01-01", f"{chunk_end}-12-31"))

        if int(da_cf_chunk.sizes.get('time', 0)) == 0:
            print(f"       时间块 {chunk_start}-{chunk_end} 无数据，跳过")
            continue

        try:
            with dask.config.set(scheduler='single-threaded'):
                daily_chunk = da_cf_chunk.resample(time='1D').mean('time')
                chunk_sum = daily_chunk.groupby('time.dayofyear').sum('time').reindex(dayofyear=all_doys, fill_value=0).compute()
                chunk_count = daily_chunk.groupby('time.dayofyear').count('time').reindex(dayofyear=all_doys, fill_value=0).compute()
        except RuntimeError as exc:
            if "NetCDF: HDF error" in str(exc):
                raise RuntimeError(
                    f"读取 NetCDF 失败（HDF error），发生在时间块 {chunk_start}-{chunk_end}。\n"
                    "常见原因：\n"
                    "1) 输入 nc 文件存在损坏/不完整压缩块；\n"
                    "2) netCDF4+hdf5 读取压缩数据时兼容性问题。\n"
                    "建议先对该时间块做小切片读取自检，确认源文件完整性。"
                ) from exc
            raise

        if total_sum is None:
            total_sum = chunk_sum
            total_count = chunk_count
        else:
            total_sum = total_sum + chunk_sum
            total_count = total_count + chunk_count

    if total_sum is None or total_count is None:
        raise ValueError("未在目标时段内读取到可用数据，请检查时间范围与输入文件。")

    # 4. 多年日平均气候态 = 各天总和 / 各天样本数
    print("  -> 步骤2: 正在生成多年日平均气候态 (Climatological Day of Year)...")
    da_cf_annual_cycle = xr.where(total_count > 0, total_sum / total_count, np.nan)
    
    # 5. 计算年均数据中的全局最小值和最大值 (统一纵轴)
    v_min = float(da_cf_annual_cycle.min())
    v_max = float(da_cf_annual_cycle.max())
    print(f"日平均年均数据的全局范围: vmin = {v_min:.4f}, vmax = {v_max:.4f}")
    
    cluster_zones = ds_clusters.cluster_zone
    n_clusters = 8 
    
    # 6. 设置绘图布局 (8行1列)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 22), sharex=True, dpi=120)
    plt.subplots_adjust(hspace=0.45)
    
    colors = plt.get_cmap('tab10', n_clusters)
    day_axis = da_cf_annual_cycle.dayofyear.values
    
    for i in range(n_clusters):
        ax = axes[i]
        print(f"  -> 正在绘制 Zone {i}...")
        
        # 7. 提取区域掩膜并计算空间平均
        zone_mask = (cluster_zones == i)
        zone_cf_cycle = da_cf_annual_cycle.where(zone_mask).mean(dim=['lat', 'lon'])
        
        # 计算区域 P10 基准线 (注意: P10 是固定的空间分布，直接求该区域均值)
        zone_p10_base = float(ds_p10.p10.where(zone_mask).mean().values)
        cf_values = zone_cf_cycle.values
        
        # 8. 绘图
        ax.plot(day_axis, cf_values, color=colors(i), linewidth=1.5, label=f'Zone {i} Daily Mean CF')
        ax.axhline(y=zone_p10_base, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
        
        # 9. 填充阴影：当日平均出力低于区域 P10 阈值时
        ax.fill_between(day_axis, zone_p10_base, cf_values, 
                        where=(cf_values < zone_p10_base),
                        color=colors(i), alpha=0.4, interpolate=True)
        
        # 10. 统一纵轴刻度
        ax.set_ylim(v_min, v_max)
        
        # 装饰
        ax.set_ylabel(f'Zone {i}', fontsize=10, fontweight='bold', rotation=0, labelpad=20)
        ax.grid(axis='both', linestyle=':', alpha=0.3)
        
        # 在轴上标注 P10 具体数值
        ax.text(day_axis[-1]+2, zone_p10_base, f'$P_{{10}}$={zone_p10_base:.2f}', 
                color='red', va='center', fontsize=9)

    # 设置底部 X 轴标签
    axes[-1].set_xlabel('Day of Year (Jan 1st - Dec 31st)', fontsize=12)
    plt.suptitle(f'Daily Mean Solar CF & Drought Profiles (Climatological 2040-2060)\nUniform Scale: [{v_min:.2f}, {v_max:.2f}]', 
                 fontsize=16, y=0.93)
    
    # 11. 保存
    output_path = "G:/extreme_analysis/results/Analysis/Regional_Daily_Mean_Climatology.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()
    plt.close()
    print(f"日平均分析图已保存: {output_path}")

if __name__ == "__main__":
    # 配置路径
    delta_x_file = "G:/extreme_analysis/results/Solar/Solar_ssp585_V2_delta_x.nc"
    cluster_file = "G:/extreme_analysis/results/Clustering/kmeans_Spatial_Cluster_Zones.nc"
    p10_file = "G:/extreme_analysis/results/Solar/Solar_ssp585_p10_thresholds.nc"
    
    plot_daily_climatological_profiles(delta_x_file, cluster_file, p10_file, start_year='2040', end_year='2060')
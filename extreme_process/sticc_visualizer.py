import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def post_process_sticc(nc_path, label_path, var_name="SolarCF", time_slice=None):
    """
    恢复空间坐标并将聚类标签映射回地图。
    """
    # 1. 加载原始数据以获取坐标和 Mask
    print("正在加载 NC 文件以提取坐标...")
    with xr.open_dataset(nc_path) as ds:
        data_var = ds[var_name]
        # 与 STICC_data_maker.py 完全一致：先按 time_slice(可选)截取，再沿 (lat, lon) 展平。
        # 注意不要只取 time=0，否则有效格点集合与 STICC 训练时可能不一致。
        if time_slice is not None:
            data_var = data_var.isel(time=time_slice)

        stacked = data_var.stack(pos=('lat', 'lon'))
        df_flattened = stacked.dropna('pos', how='all').to_pandas()
        valid_positions = df_flattened.columns.to_frame(index=False)
        valid_positions.columns = ['lat', 'lon']
        valid_positions = valid_positions.reset_index(drop=True)

    # 2. 读取聚类标签
    print("正在读取聚类结果...")
    labels = np.atleast_1d(np.loadtxt(label_path, dtype=int))
    
    # 校验长度
    if len(labels) != len(valid_positions):
        raise ValueError(f"长度不匹配！标签数量: {len(labels)}, 有效空间点数量: {len(valid_positions)}。请检查是否预处理逻辑一致。")

    # 3. 合并数据
    result_df = valid_positions.copy()
    result_df['grid_id'] = np.arange(len(result_df), dtype=int)
    result_df['cluster'] = labels
    
    return result_df

def plot_clusters(df):
    """
    绘制聚类分布图
    """
    plt.figure(figsize=(12, 6))
    # 使用散点图映射经纬度
    scatter = plt.scatter(df['lon'], df['lat'], c=df['cluster'], cmap='tab10', s=15, marker='s')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("STICC Spatial Clustering Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    fig_path = r"G:\extreme_analysis\results\solar_ver0\clustered_map.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"已保存结果至 {fig_path}")

if __name__ == "__main__":
    nc_file = r"G:\extreme_analysis\data\CMIP6_QDM_MME\QDM_cmip6_MME_SolarCF_ssp126_1deg_china_2025-2060.nc"
    label_file = r"F:\code_program\EnergyResearch\extreme_process\STICC\STICC-master\ssp126_SolarCF_results.txt"
    # 若 STICC_data_maker.py 训练时用了 time_slice（如 slice(0, 1000)），这里必须保持一致。
    sticc_time_slice = None

    try:
        df_result = post_process_sticc(nc_file, label_file, time_slice=sticc_time_slice)
        print("映射完成，正在绘制地图...")
        plot_clusters(df_result)
        
        # 可选：保存为 CSV 方便在 ArcGIS/Excel 中查看
        csv_path = r"G:\extreme_analysis\results\solar_ver0\clustered_map_data.csv"
        df_result.to_csv(csv_path, index=False)
        print(f"已保存结果至 {csv_path}")
    except Exception as e:
        print(f"处理失败: {e}")
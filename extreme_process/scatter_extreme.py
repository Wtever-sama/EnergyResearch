import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.ndimage import label


def analyze_cluster_extremes(delta_x_path, cluster_map_path, output_csv, output_fig):
    """
    分析各聚类区域的强度-持续时间模式
    """
    print("正在加载数据...")
    # 加载 delta_x 数据 (CF - Threshold)
    ds_delta = xr.open_dataset(delta_x_path, chunks={'time': 5000})
    # 加载聚类掩膜
    ds_clusters = xr.open_dataset(cluster_map_path)
    
    # 1. 计算瞬时亏空量 D = max(0, -delta_x)
    # 原始 delta_x = CF - Threshold, 所以负值代表亏空
    deficit = xr.where(ds_delta.delta_x < 0, -ds_delta.delta_x, 0)
    
    cluster_zones = ds_clusters.cluster_zone
    n_clusters = int(cluster_zones.max().values) + 1
    
    all_events = []

    print(f"开始分析各区域 (共 {n_clusters} 个聚类)...")
    for i in range(n_clusters):
        print(f"  正在处理 Zone {i}...")
        
        # 2. 空间聚合：计算该区域在每个时刻的平均亏空强度
        zone_mask = (cluster_zones == i)
        # 仅在有效掩膜区计算空间均值
        zone_ts = deficit.where(zone_mask).mean(dim=['lat', 'lon']).compute()
        
        # 3. 识别连续极端事件 (逻辑参考 duration_judge.py)
        # 定义：平均亏空量 > 0 视为发生极端事件
        event_binary = (zone_ts > 0).astype(int).values
        labeled_array, num_features = label(event_binary)
        
        # 4. 提取事件特征
        for event_id in range(1, num_features + 1):
            event_indices = np.where(labeled_array == event_id)[0]
            event_data = zone_ts.values[event_indices]
            
            duration = len(event_indices)
            severity = np.sum(event_data)
            peak_intensity = np.max(event_data)
            
            all_events.append({
                'Zone_ID': i,
                'Severity': severity,
                'Duration_D': duration,
                'Peak_Intensity': peak_intensity
            })

    # 5. 保存结果为 CSV
    df_events = pd.DataFrame(all_events)
    df_events.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"事件统计结果已保存至: {output_csv}")

    # 6. 绘图：强度-持续时间散点图
    plot_sdf_scatter(df_events, output_fig)


def plot_sdf_scatter(df, output_fig):
    """
    绘制强度-持续时间二维图
    """
    plt.figure(figsize=(10, 7), dpi=150)
    sns.set_style("whitegrid")
    
    # 使用不同的颜色标记不同的 ClusterID
    scatter = sns.scatterplot(
        data=df,
        x='Duration_D',
        y='Severity',
        hue='Zone_ID',
        palette='viridis',
        alpha=0.6,
        edgecolor=None,
        s=40
    )
    
    plt.title("Extreme Events: Severity vs. Duration by Cluster Zone", fontsize=14)
    plt.xlabel("Duration (Hours)", fontsize=12)
    plt.ylabel("Cumulative Severity (Capacity Factor Hours)", fontsize=12)
    plt.yscale('log') # 严重程度差异可能很大，建议使用对数坐标
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_fig)
    plt.close()
    print(f"SDF 散点图已保存至: {output_fig}")


if __name__ == "__main__":
    # 配置路径
    delta_x_file = "G:/extreme_analysis/results/Solar/Solar_ssp585_V2_delta_x.nc"
    cluster_file = "G:/extreme_analysis/results/Clustering/kmeans_Spatial_Cluster_Zones.nc"
    
    results_dir = "G:/extreme_analysis/results/Solar/"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_out = os.path.join(results_dir, "Cluster_Extreme_Events_Stats.csv")
    fig_out = os.path.join(results_dir, "Solar_ssp585_extreme_scatter.png")
    
    analyze_cluster_extremes(delta_x_file, cluster_file, csv_out, fig_out)
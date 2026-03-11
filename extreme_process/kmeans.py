import xarray as xr
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import os


def process_spatial_clustering(solar_path, wind_path, n_clusters=7, spatial_chunk_size=128, n_components=50):
    print("正在加载数据并识别有效掩膜...")
    # 使用 chunks 开启 Dask 支持
    ds_s = xr.open_dataset(solar_path, chunks={'time': 5000})
    ds_w = xr.open_dataset(wind_path, chunks={'time': 5000})

    lat_coords = ds_s['lat']
    lon_coords = ds_s['lon']

    # 1. 更加鲁棒的掩膜提取
    mask = ds_s.is_extreme.notnull().any(dim='time').compute()
    valid_indices = np.flatnonzero(mask.values.flatten())
    
    # 2. 预处理：在 stack 之前先 fillna(0)，这样可以避免后续计算出现 NaN
    # 但要注意：只对有效格点填 0，不要把海洋也填成 0
    s_clean = ds_s.is_extreme.where(mask, 0).fillna(0).astype(np.float32)
    w_clean = ds_w.is_extreme.where(mask, 0).fillna(0).astype(np.float32)
    
    s_flat = s_clean.stack(pos=('lat', 'lon')).transpose('pos', 'time')
    w_flat = w_clean.stack(pos=('lat', 'lon')).transpose('pos', 'time')

    valid_indices = np.flatnonzero(mask.values.flatten())
    n_valid = valid_indices.size

    if n_valid == 0:
        raise ValueError("未找到有效格点，请检查输入数据或掩膜构建逻辑。")

    def iter_valid_chunks(indices, chunk_size):
        for start in range(0, indices.size, chunk_size):
            end = min(start + chunk_size, indices.size)
            yield start, end, indices[start:end]

    # 先读取一个分块，确定特征维度
    _, _, first_idx = next(iter_valid_chunks(valid_indices, spatial_chunk_size))
    first_s = s_flat.isel(pos=first_idx).compute().values.astype(np.float32, copy=False)
    first_w = w_flat.isel(pos=first_idx).compute().values.astype(np.float32, copy=False)
    first_combined = np.concatenate([first_s, first_w], axis=1).astype(np.float32, copy=False)
    n_features = first_combined.shape[1]

    # IncrementalPCA 要求 n_components 不能超过样本数和特征数
    n_components = min(n_components, n_valid, n_features)
    if n_components < 1:
        raise ValueError("有效样本或特征不足，无法执行 PCA。")

    print(f"有效格点数: {n_valid}, 原始特征维度: {n_features}")
    print(f"使用 float32 + 分块增量计算，分块大小: {spatial_chunk_size}, PCA 维度: {n_components}")

    # 3. 第一遍：分块拟合标准化器（避免一次性构造完整矩阵）
    print("第一遍：拟合 StandardScaler...")
    scaler = StandardScaler(copy=False)
    scaler.partial_fit(first_combined)

    for start, end, chunk_idx in iter_valid_chunks(valid_indices[first_idx.size:], spatial_chunk_size):
        s_chunk = s_flat.isel(pos=chunk_idx).compute().values.astype(np.float32, copy=False)
        w_chunk = w_flat.isel(pos=chunk_idx).compute().values.astype(np.float32, copy=False)
        combined_chunk = np.concatenate([s_chunk, w_chunk], axis=1).astype(np.float32, copy=False)
        scaler.partial_fit(combined_chunk)
        print(f"  StandardScaler 进度: {first_idx.size + end}/{n_valid}")

    # 4. 第二遍：分块拟合 IncrementalPCA
    print("第二遍：拟合 IncrementalPCA...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=max(spatial_chunk_size, n_components))

    first_scaled = scaler.transform(first_combined).astype(np.float32, copy=False)
    ipca.partial_fit(first_scaled)

    for start, end, chunk_idx in iter_valid_chunks(valid_indices[first_idx.size:], spatial_chunk_size):
        s_chunk = s_flat.isel(pos=chunk_idx).compute().values.astype(np.float32, copy=False)
        w_chunk = w_flat.isel(pos=chunk_idx).compute().values.astype(np.float32, copy=False)
        combined_chunk = np.concatenate([s_chunk, w_chunk], axis=1).astype(np.float32, copy=False)
        scaled_chunk = scaler.transform(combined_chunk).astype(np.float32, copy=False)
        ipca.partial_fit(scaled_chunk)
        print(f"  IncrementalPCA 进度: {first_idx.size + end}/{n_valid}")

    # 5. 第三遍：生成降维后的空间特征
    print("第三遍：生成降维后的特征矩阵...")
    reduced_features = np.empty((n_valid, n_components), dtype=np.float32)

    first_reduced = ipca.transform(first_scaled).astype(np.float32, copy=False)
    reduced_features[:first_idx.size] = first_reduced

    write_pos = first_idx.size
    for _, _, chunk_idx in iter_valid_chunks(valid_indices[first_idx.size:], spatial_chunk_size):
        s_chunk = s_flat.isel(pos=chunk_idx).compute().values.astype(np.float32, copy=False)
        w_chunk = w_flat.isel(pos=chunk_idx).compute().values.astype(np.float32, copy=False)
        combined_chunk = np.concatenate([s_chunk, w_chunk], axis=1).astype(np.float32, copy=False)
        scaled_chunk = scaler.transform(combined_chunk).astype(np.float32, copy=False)
        chunk_reduced = ipca.transform(scaled_chunk).astype(np.float32, copy=False)
        chunk_n = chunk_reduced.shape[0]
        reduced_features[write_pos:write_pos + chunk_n] = chunk_reduced
        write_pos += chunk_n
        print(f"  特征写入进度: {write_pos}/{n_valid}")
    
    # 6. 执行 K-means 聚类
    print(f"正在划分为 {n_clusters} 个风光一致区...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_features)
    
    # --- 修正后的第 7 步：将结果正确映射回空间地图 ---
    print("正在将聚类标签映射回空间维度...")
    # 先创建一个全 NaN 的一维数组
    full_labels_1d = np.full(mask.size, np.nan, dtype=np.float32)
    # 使用布尔索引直接在一维数组上赋值（这会直接修改数组内容）
    full_labels_1d[mask.values.flatten()] = cluster_labels
    # 变形回 (lat, lon) 形状
    full_labels_2d = full_labels_1d.reshape(mask.shape)
    
    da_clusters = xr.DataArray(
        full_labels_2d, 
        coords={'lat': lat_coords, 'lon': lon_coords}, # 显式指定坐标映射
        dims=['lat', 'lon'], 
        name='cluster_zone'
    )

    # --- 修正后的第 8 步：鲁棒的区域聚合 ---
    print("正在计算各区域受灾面积比例序列...")
    # 预处理：将原始 flag 中的 NaN 填充为 0（代表非极端），确保 mean 计算不为 NaN
    s_flag_filled = ds_s.is_extreme.fillna(0).astype(np.float32)
    w_flag_filled = ds_w.is_extreme.fillna(0).astype(np.float32)
    
    regional_series = {}

    for i in range(n_clusters):
        # 找到属于该簇的格点
        zone_mask = (da_clusters == i)
        
        # 检查该分区是否包含有效格点
        if not zone_mask.any():
            print(f"警告: Zone {i} 没有匹配到任何格点，跳过。")
            continue

        # 使用 where 过滤出该分区的数据，注意 xarray 会自动处理坐标对齐
        s_zone = s_flag_filled.where(zone_mask)
        w_zone = w_flag_filled.where(zone_mask)
        
        # 计算该区域的受灾比例（即 1 的占比）
        # skipna=True 是默认行为，它会忽略掉 zone_mask 之外的 NaN
        s_area_frac = s_zone.mean(dim=['lat', 'lon']).compute()
        w_area_frac = w_zone.mean(dim=['lat', 'lon']).compute()
        
        regional_series[f'Zone_{i}_Solar'] = s_area_frac.to_series()
        regional_series[f'Zone_{i}_Wind'] = w_area_frac.to_series()

    df_vine_input = pd.DataFrame(regional_series)

    print(f"DEBUG: Mask valid points = {mask.sum().values}")
    print(f"DEBUG: Cluster labels unique = {np.unique(cluster_labels)}")
    print(f"DEBUG: da_clusters non-nan count = {da_clusters.notnull().sum().values}")
    
    return da_clusters, df_vine_input


if __name__ == "__main__":
    # --- 运行示例 ---
    solar_file = "G:/extreme_analysis/results/Solar/Solar_ssp585_V1_flag.nc"
    wind_file = "G:/extreme_analysis/results/Wind/Wind_ssp585_V1_flag.nc"

    clusters_map, vine_input_df = process_spatial_clustering(solar_file, wind_file, n_clusters=8)

    # 保存结果供 Vine Copula 使用
    output_dir = "G:/extreme_analysis/results/Clustering/"
    os.makedirs(output_dir, exist_ok=True)
    csv_output_path = os.path.join(output_dir, "kmeans_Vine_Copula_Input_Data.csv")
    nc_output_path = os.path.join(output_dir, "kmeans_Spatial_Cluster_Zones.nc")
    vine_input_df.to_csv(csv_output_path, index=False)
    clusters_map.to_netcdf(nc_output_path)

    print(f"CSV 保存到 {csv_output_path}")
    print(f"NetCDF 保存到 {nc_output_path}")

    print("分析完成！")
    print(f"Vine Copula 输入矩阵形状: {vine_input_df.shape}")
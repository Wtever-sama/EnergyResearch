import xarray as xr
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional
import os

class STICCDataProcessor:
    """
    STICC 数据预处理工具类。
    """

    def __init__(self, file_path: str, var_name: str):
        """
        初始化处理器。
        """
        self.file_path = file_path
        self.var_name = var_name
        self.df_flattened: Optional[pd.DataFrame] = None
        self.valid_positions: Optional[pd.DataFrame] = None

    def load_and_preprocess(self, time_slice: Optional[slice] = None) -> pd.DataFrame:
        """
        加载数据并进行空间维度展平。
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"找不到文件: {self.file_path}")

        print(f"正在加载文件: {self.file_path} ...")
        with xr.open_dataset(self.file_path) as ds:
            data_var = ds[self.var_name]
            if time_slice:
                data_var = data_var.isel(time=time_slice)
            
            print(f"正在转换数据格式并剔除无效网格...")
            stacked = data_var.stack(pos=('lat', 'lon'))
            self.df_flattened = stacked.dropna('pos', how='all').to_pandas()
            self.valid_positions = self.df_flattened.columns.to_frame(index=False)
            
        print(f"预处理完成。有效空间点数量: {len(self.valid_positions)}, 时间步数: {len(self.df_flattened)}")
        return self.df_flattened

    def build_adjacency(self, n_neighbors: int = 4) -> np.ndarray:
        """
        使用 KNN 算法构建地理邻接表。
        """
        if self.valid_positions is None:
            raise ValueError("错误：必须先调用 load_and_preprocess()。")

        print(f"正在构建空间邻居关系 (n_neighbors={n_neighbors})...")
        coords = self.valid_positions[['lat', 'lon']].values
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        adj_list = []
        for i, neighbors in enumerate(indices):
            for neighbor_idx in neighbors[1:]:  # 排除自身
                adj_list.append((i, neighbor_idx))
                
        return np.array(adj_list)

    def format_sticc_input(self, adj_edges: np.ndarray, output_path: str):
        """
        将属性与邻居关系合并，生成 STICC 要求的 .txt 格式。
        
        结构：[ID, Attribute_1, ..., Attribute_N, Neighbor_1, ..., Neighbor_K]
        """
        if self.df_flattened is None:
            raise ValueError("错误：必须先调用 load_and_preprocess()。")

        print(f"正在合并数据并生成最终输入文件: {output_path} ...")
        
        # 1. 转置属性数据：从 (Time, Location) 转为 (Location, Time)
        # 这样每一行代表一个地理网格点
        df_attr = self.df_flattened.T.reset_index(drop=True)
        
        # 2. 将边表转换为宽表格式（每个节点的邻居列表）
        adj_df = pd.DataFrame(adj_edges, columns=['node1', 'node2'])
        neighbors_list = adj_df.groupby('node1')['node2'].apply(list).reset_index()
        df_neighbors = pd.DataFrame(neighbors_list['node2'].tolist(), index=neighbors_list['node1'])
        
        # 3. 合并：ID + 属性 + 邻居
        # 重置索引确保拼接对齐
        final_df = pd.concat([
            pd.Series(df_attr.index, name='id'), 
            df_attr, 
            df_neighbors
        ], axis=1)
        
        # 4. 保存为 .txt (采用 CSV 结构，无表头)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, header=False, index=False)
        
        print(f"合并完成！")
        print(f" - 属性列索引范围: 1 到 {df_attr.shape[1]}")
        print(f" - 邻居列索引范围: {df_attr.shape[1] + 1} 到 {final_df.shape[1] - 1}")

    def export_for_sticc(self, data_csv: str, adj_csv: str, adj_edges: np.ndarray):
        """
        导出中间 CSV 文件（保留原功能）。
        """
        os.makedirs(os.path.dirname(data_csv), exist_ok=True)
        os.makedirs(os.path.dirname(adj_csv), exist_ok=True)
        self.df_flattened.to_csv(data_csv, index=False)
        pd.DataFrame(adj_edges, columns=['node1', 'node2']).to_csv(adj_csv, index=False)
        print(f"中间 CSV 已导出。")


if __name__ == "__main__":
    # 1. 设置参数
    file_dir = "G:/extreme_analysis"
    variable = "SolarCF"
    scenario = "ssp126"
    
    FILE_CONFIG = {
        "input_nc": os.path.join(file_dir, f"data/CMIP6_QDM_MME/QDM_cmip6_MME_{variable}_{scenario}_1deg_china_2025-2060.nc"),
        "variable": variable,
        # 最终 STICC 要求的合并文件路径
        "final_txt": os.path.join(file_dir, f"sticc_dataset/{scenario}/{variable}/sticc_final_input.txt"),
        "neighbors": 8 
    }

    # 2. 实例化
    processor = STICCDataProcessor(FILE_CONFIG["input_nc"], FILE_CONFIG["variable"])

    try:
        # 3. 处理数据 (此处取 slice 仅用于快速测试，全量运行请设为 None)
        data_matrix = processor.load_and_preprocess(time_slice=slice(0, 1000))

        # 4. 生成邻接关系
        edges = processor.build_adjacency(n_neighbors=FILE_CONFIG["neighbors"])

        # 5. 合并并生成最终 STICC 输入文件
        processor.format_sticc_input(edges, FILE_CONFIG["final_txt"])

    except Exception as e:
        print(f"程序运行期间发生错误: {e}")
import xarray as xr
import os


def save(data: xr.DataArray, output_path: str, format: str='nc') -> None:
    """
    Utils 方法：保存数据，优化 CSV 输出的可读性
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'nc':
        data.to_netcdf(output_path)
    elif format == 'csv':
        # 1. 转换为 DataFrame
        df = data.to_dataframe()
        # 2. 重置索引：将 time, lat, lon 从索引变为普通的列，方便在 Excel 中查看 
        df = df.reset_index()
        # 3. 写入 CSV：使用 utf-8-sig 编码修复 Windows Excel 打开乱码的问题
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
    print(f"文件已保存至: {output_path}")
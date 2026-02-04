import xarray as xr
import glob
import os
#查看db文件的基本信息
#用于查看db文件的变量名
def explore_nc_files(folder_path, sample_count=3):
    """
    探索nc文件的结构和变量信息

    Parameters:
    folder_path: 文件夹路径
    sample_count: 要详细查看的样本文件数量
    """
    # 获取所有nc文件
    nc_files = glob.glob(os.path.join(folder_path, "era5_2m_temperature_*.nc"))#nc文件对应的文件名
    #CMIP6数据的文件名tas_3hr_BCC-CSM2-MR_historical_r1i1p1f1_gn_199801010000-200012312100
    #era5数据的文件匹配era5_2m_temperature_celsius_utc+8_*.nc
    if not nc_files:
        print("未找到匹配的nc文件！")
        return

    print(f"找到 {len(nc_files)} 个nc文件")
    print("文件列表:")
    for i, file in enumerate(sorted(nc_files)[:10]):  # 只显示前10个文件
        print(f"  {i + 1}. {os.path.basename(file)}")
    if len(nc_files) > 10:
        print(f"  ... 还有 {len(nc_files) - 10} 个文件")

    print("\n" + "=" * 50)
    print("文件结构探索:")
    print("=" * 50)

    # 详细查看前几个文件
    for i, file_path in enumerate(sorted(nc_files)[:sample_count]):
        print(f"\n--- 文件 {i + 1}: {os.path.basename(file_path)} ---")

        try:
            # 打开数据集
            ds = xr.open_dataset(file_path)

            # 1. 基本数据集信息
            print("\n1. 数据集信息:")
            print(ds)

            # 2. 变量信息
            print("\n2. 变量列表:")
            for var_name in ds.data_vars:
                var = ds[var_name]
                print(f"   - {var_name}: {var.attrs.get('long_name', '无描述')}")
                print(f"     维度: {var.dims}, 形状: {var.shape}")
                if 'units' in var.attrs:
                    print(f"     单位: {var.attrs['units']}")

            # 3. 坐标信息
            print("\n3. 坐标信息:")
            for coord_name in ds.coords:
                coord = ds[coord_name]
                print(f"   - {coord_name}: {len(coord)} 个值")
                if len(coord) > 0:
                    print(f"     范围: {coord.values.min()} 到 {coord.values.max()}")

            # 4. 全局属性
            print("\n4. 全局属性:")
            for attr_name, attr_value in ds.attrs.items():
                print(f"   - {attr_name}: {attr_value}")

            # 5. 时间信息（如果有时间维度）
            if 'time' in ds.dims:
                print(f"\n5. 时间信息:")
                print(f"   时间范围: {ds.time.min().values} 到 {ds.time.max().values}")
                print(f"   时间步长: {len(ds.time)} 个时间点")

            ds.close()

        except Exception as e:
            print(f"读取文件时出错: {e}")


# 使用脚本
folder_path = r"G:\CMIP6_tas_2000-2014\2m_temperature"
  # 替换为实际路径

explore_nc_files(folder_path)
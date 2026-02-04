#网格重置代码
#第一步网格重置化
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
import glob
import re


def regrid_to_1x1_degree(ds, data_source, china_bounds=None):
    """
    将数据重采样到1.0×1.0度网格

    参数:
    ds: xarray Dataset 或 DataArray
    data_source: 数据源类型 ('era5' 或 'cmip6')
    china_bounds: 中国区域边界 [lon_min, lon_max, lat_min, lat_max]，默认为None(使用中国常用范围)

    返回:
    重采样后的Dataset
    """
    if china_bounds is None:
        # 中国常用范围 (稍微扩大一点确保覆盖完整)
        china_bounds = [70, 140, 15, 55]

    lon_min, lon_max, lat_min, lat_max = china_bounds

    # 创建1.0×1.0度的目标网格
    target_lon = np.arange(lon_min, lon_max + 1, 1.0)
    target_lat = np.arange(lat_min, lat_max + 1, 1.0)

    if data_source == 'era5':
        # ERA5数据的坐标名
        lon_var = 'longitude'
        lat_var = 'latitude'
        # 时间坐标名
        time_var = 'valid_time'
        #变量名可以到自己查看后更改
        # 重采样到目标网格
        print(f"重采样ERA5到1.0×1.0度网格...")
        ds_regridded = ds.interp(
            {lon_var: target_lon, lat_var: target_lat},
            method='linear'
        )

    elif data_source == 'cmip6':
        # 对于CMIP6数据，我们需要提取主要的数据变量和坐标
        # 找到主要的数据变量（通常是'tas'）
        data_vars = list(ds.data_vars.keys())
        main_var = None
        for var in ['tas', 'temp', 'temperature']:
            if var in data_vars:
                main_var = var
                break
        if main_var is None and data_vars:
            main_var = data_vars[0]  # 使用第一个数据变量

        if main_var is None:
            raise ValueError("找不到主要的数据变量")

        print(f"使用主要数据变量: {main_var}")

        # 提取主要的数据变量，这会自动包含坐标信息
        data_array = ds[main_var]

        # 确定坐标名称
        if 'lon' in data_array.dims and 'lat' in data_array.dims:
            lon_var = 'lon'
            lat_var = 'lat'
        elif 'longitude' in data_array.dims and 'latitude' in data_array.dims:
            lon_var = 'longitude'
            lat_var = 'latitude'
        else:
            available_dims = list(data_array.dims)
            raise ValueError(f"无法确定空间坐标，可用的维度: {available_dims}")

        # 时间坐标名
        time_var = 'time'

        # 检查经度范围，如果是0-360，转换为-180-180
        if data_array[lon_var].max() > 180:
            print("检测到经度范围为0-360，正在转换为-180-180...")
            data_array = data_array.assign_coords({lon_var: (((data_array[lon_var] + 180) % 360) - 180)})
            data_array = data_array.sortby(lon_var)

        # 重采样到目标网格
        print(f"重采样CMIP6到1.0×1.0度网格...")
        data_regridded = data_array.interp(
            {lon_var: target_lon, lat_var: target_lat},
            method='linear'
        )

        # 统一坐标名称并转换回Dataset
        if lon_var != 'longitude' or lat_var != 'latitude':
            print(f"统一坐标名称: {lon_var}->longitude, {lat_var}->latitude")
            data_regridded = data_regridded.rename({lon_var: 'longitude', lat_var: 'latitude'})

        # 转换回Dataset
        ds_regridded = data_regridded.to_dataset(name=main_var)

    else:
        raise ValueError("data_source 必须是 'era5' 或 'cmip6'")

    return ds_regridded


def process_era5_data(era5_folder, output_folder, start_year=2000, end_year=2014):
    """
    处理ERA5数据：重采样到1.0×1.0度网格并裁剪到中国区域
    """
    print("处理ERA5数据...")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取ERA5文件
    era5_files = []
    for year in range(start_year, end_year + 1):
        pattern = os.path.join(era5_folder, f"era5_2m_temperature_{year}.nc")
        matches = glob.glob(pattern)
        era5_files.extend(matches)

    era5_files.sort()

    if not era5_files:
        print("未找到ERA5文件!")
        return None

    print(f"找到 {len(era5_files)} 个ERA5文件")

    try:
        # 读取所有ERA5文件
        era5_ds = xr.open_mfdataset(era5_files, combine='by_coords')
        print(f"ERA5原始数据维度: {dict(era5_ds.dims)}")
        print(f"ERA5数据变量: {list(era5_ds.data_vars.keys())}")

        # 重采样到1.0×1.0度
        era5_regridded = regrid_to_1x1_degree(era5_ds, 'era5')
        print(f"ERA5重采样后维度: {dict(era5_regridded.dims)}")

        # 保存处理后的数据
        output_file = os.path.join(output_folder, f"era5_t2m_1deg_china_{start_year}-{end_year}.nc")
        era5_regridded.to_netcdf(output_file)
        print(f"ERA5处理完成，保存至: {output_file}")

        return output_file

    except Exception as e:
        print(f"处理ERA5数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 确保关闭数据集
        if 'era5_ds' in locals():
            era5_ds.close()
        if 'era5_regridded' in locals():
            era5_regridded.close()


def process_cmip6_data(cmip6_folder, output_folder, models=None, start_year=2000, end_year=2014):
    """
    处理CMIP6数据：重采样到1.0×1.0度网格并裁剪到中国区域

    返回:
    dict: 处理后的文件路径字典，键为模型名
    """
    print("处理CMIP6数据...")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取CMIP6文件
    cmip6_files = glob.glob(os.path.join(cmip6_folder, "tas_3hr_*_historical_*_gn_*.nc"))

    if not cmip6_files:
        print("未找到CMIP6文件!")
        return {}

    # 按模型分组
    model_files = {}
    for file_path in cmip6_files:
        filename = os.path.basename(file_path)
        pattern = r'tas_3hr_([^_]+)_historical_([^_]+)_gn_(\d{12})-(\d{12})\.nc'
        match = re.match(pattern, filename)

        if match:
            model = match.group(1)
            start_year_file = int(match.group(3)[:4])
            end_year_file = int(match.group(4)[:4])

            # 检查时间范围是否与需求重叠
            if start_year_file <= end_year and end_year_file >= start_year:
                if model not in model_files:
                    model_files[model] = []
                model_files[model].append(file_path)

    # 如果指定了模型，只处理指定的模型
    if models is not None:
        model_files = {model: files for model, files in model_files.items() if model in models}

    if not model_files:
        print("没有找到符合条件的CMIP6模型!")
        return {}

    processed_files = {}

    for model, files in model_files.items():
        print(f"\n处理模型: {model}")
        files.sort()

        try:
            # 读取该模型的所有文件
            cmip6_ds = xr.open_mfdataset(files, combine='by_coords')
            print(f"{model} 原始数据维度: {dict(cmip6_ds.dims)}")
            print(f"{model} 数据变量: {list(cmip6_ds.data_vars.keys())}")
            print(f"{model} 坐标变量: {list(cmip6_ds.coords.keys())}")

            # 筛选时间范围
            time_slice = slice(f"{start_year}-01-01", f"{end_year}-12-31")
            if 'time' in cmip6_ds.coords:
                cmip6_ds = cmip6_ds.sel(time=time_slice)
                print(f"{model} 时间筛选后维度: {dict(cmip6_ds.dims)}")

            # 重采样到1.0×1.0度
            cmip6_regridded = regrid_to_1x1_degree(cmip6_ds, 'cmip6')
            print(f"{model} 重采样后维度: {dict(cmip6_regridded.dims)}")
            print(f"{model} 重采样后坐标: {list(cmip6_regridded.coords)}")

            # 保存处理后的数据
            output_file = os.path.join(output_folder, f"cmip6_{model}_tas_1deg_china_{start_year}-{end_year}.nc")
            cmip6_regridded.to_netcdf(output_file)

            processed_files[model] = output_file
            print(f"{model} 处理完成，保存至: {output_file}")

        except Exception as e:
            print(f"处理模型 {model} 时出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保关闭数据集
            if 'cmip6_ds' in locals():
                cmip6_ds.close()
            if 'cmip6_regridded' in locals():
                cmip6_regridded.close()

    return processed_files


def check_data_units(era5_file, cmip6_files):
    """
    检查数据的单位并确保一致性
    """
    print("\n检查数据单位...")

    # 检查ERA5数据
    era5_ds = xr.open_dataset(era5_file)
    era5_var = list(era5_ds.data_vars.keys())[0]
    print(f"ERA5 变量: {era5_var}, 单位: {era5_ds[era5_var].attrs.get('units', '未知')}")
    era5_ds.close()

    # 检查CMIP6数据
    for model, file_path in cmip6_files.items():
        cmip6_ds = xr.open_dataset(file_path)
        cmip6_var = list(cmip6_ds.data_vars.keys())[0]
        print(f"{model} 变量: {cmip6_var}, 单位: {cmip6_ds[cmip6_var].attrs.get('units', '未知')}")
        cmip6_ds.close()


def visualize_processed_data(era5_file, cmip6_files, output_folder):
    """
    可视化处理后的数据，确保重采样和裁剪成功
    """
    print("生成处理后的数据可视化...")

    # 读取ERA5数据
    era5_ds = xr.open_dataset(era5_file)

    # 确定变量名
    era5_var = list(era5_ds.data_vars.keys())[0]

    # 创建图形
    n_models = len(cmip6_files)
    n_cols = min(3, n_models + 1)
    n_rows = (n_models + 1) // n_cols + 1

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # ERA5数据
    ax1 = fig.add_subplot(n_rows, n_cols, 1, projection=ccrs.PlateCarree())
    era5_ds[era5_var].isel(valid_time=0).plot(ax=ax1, transform=ccrs.PlateCarree(),
                                              cbar_kwargs={'shrink': 0.8})
    ax1.coastlines()
    ax1.set_title('ERA5 处理后的数据')

    # CMIP6数据
    for i, (model, file_path) in enumerate(cmip6_files.items()):
        cmip6_ds = xr.open_dataset(file_path)

        # 确定变量名
        cmip6_var = list(cmip6_ds.data_vars.keys())[0]

        ax = fig.add_subplot(n_rows, n_cols, i + 2, projection=ccrs.PlateCarree())
        if 'time' in cmip6_ds.dims:
            cmip6_ds[cmip6_var].isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree(),
                                                  cbar_kwargs={'shrink': 0.8})
        elif 'valid_time' in cmip6_ds.dims:
            cmip6_ds[cmip6_var].isel(valid_time=0).plot(ax=ax, transform=ccrs.PlateCarree(),
                                                        cbar_kwargs={'shrink': 0.8})
        ax.coastlines()
        ax.set_title(f'{model} 处理后的数据')

        cmip6_ds.close()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'processed_data_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    era5_ds.close()


def compare_grids_after_processing(era5_file, cmip6_files):
    """
    比较处理后的数据网格
    """
    print("\n处理后数据网格比较:")
    print("=" * 50)

    era5_ds = xr.open_dataset(era5_file)

    # ERA5网格信息
    era5_lon = era5_ds.longitude
    era5_lat = era5_ds.latitude

    print(f"ERA5处理后网格:")
    print(f"  经度范围: {float(era5_lon.min()):.2f} 到 {float(era5_lon.max()):.2f}")
    print(f"  纬度范围: {float(era5_lat.min()):.2f} 到 {float(era5_lat.max()):.2f}")
    print(f"  经度分辨率: {abs(float(era5_lon[1] - era5_lon[0])):.2f}")
    print(f"  纬度分辨率: {abs(float(era5_lat[1] - era5_lat[0])):.2f}")
    print(f"  网格点数: {len(era5_lon)} x {len(era5_lat)}")

    # CMIP6网格信息
    for model, file_path in cmip6_files.items():
        cmip6_ds = xr.open_dataset(file_path)

        # 检查坐标名称
        if 'longitude' in cmip6_ds.dims:
            cmip6_lon = cmip6_ds.longitude
            cmip6_lat = cmip6_ds.latitude
        elif 'lon' in cmip6_ds.dims:
            cmip6_lon = cmip6_ds.lon
            cmip6_lat = cmip6_ds.lat
        else:
            print(f"\n{model} 无法确定坐标名称，可用的维度: {list(cmip6_ds.dims)}")
            continue

        print(f"\n{model}处理后网格:")
        print(f"  经度范围: {float(cmip6_lon.min()):.2f} 到 {float(cmip6_lon.max()):.2f}")
        print(f"  纬度范围: {float(cmip6_lat.min()):.2f} 到 {float(cmip6_lat.max()):.2f}")
        print(f"  经度分辨率: {abs(float(cmip6_lon[1] - cmip6_lon[0])):.2f}")
        print(f"  纬度分辨率: {abs(float(cmip6_lat[1] - cmip6_lat[0])):.2f}")
        print(f"  网格点数: {len(cmip6_lon)} x {len(cmip6_lat)}")

        cmip6_ds.close()

    era5_ds.close()


# 主函数
def main():
    # 设置路径
    era5_folder = r"G:\CMIP6_tas_2000-2014\2m_temperature"
    cmip6_folder = r"G:\CMIP6_tas_2000-2014\CMIP6"
    output_folder = r"G:\CMIP6_tas_2000-2014\processed_1deg_china"

    # 年份范围
    start_year = 2000
    end_year = 2014

    # 要处理的CMIP6模型（如果为None则处理所有找到的模型）
    target_models = ['BCC-CSM2-MR', 'NESM3', 'MRI-ESM2-0', 'MIROC-ES2L', 'MIROC6', 'CanESM5']

    print("开始处理数据...")
    print("=" * 60)

    # 1. 处理ERA5数据
    era5_processed = process_era5_data(
        era5_folder, output_folder,
        start_year=start_year, end_year=end_year
    )

    # 2. 处理CMIP6数据
    cmip6_processed = process_cmip6_data(
        cmip6_folder, output_folder,
        models=target_models,
        start_year=start_year, end_year=end_year
    )

    if era5_processed and cmip6_processed:
        # 3. 检查数据单位
        check_data_units(era5_processed, cmip6_processed)

        # 4. 可视化处理后的数据
        visualize_processed_data(era5_processed, cmip6_processed, output_folder)

        # 5. 比较处理后的网格
        compare_grids_after_processing(era5_processed, cmip6_processed)

        print("\n" + "=" * 60)
        print("数据处理完成!")
        print(f"处理后的数据保存在: {output_folder}")
        print(f"处理了 {len(cmip6_processed)} 个CMIP6模型")
    else:
        print("数据处理失败!")


if __name__ == "__main__":
    main()
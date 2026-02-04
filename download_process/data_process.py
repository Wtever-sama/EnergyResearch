import xarray as xr
import os
import numpy as np
from datetime import timedelta
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.colors import LinearSegmentedColormap

# 打开NetCDF文件
#file_path = r"E:\DataDownload\CMIP6\uas\uas_3hr_ACCESS-CM2_ssp245_r1i1p1f1_gn_209501010300-210101010000.nc"
'''ds = xr.open_dataset(file_path)

# 查看数据集信息
print(ds)
print("实际经度范围：", ds.lon.min().values, ds.lon.max().values)
# 实际经度范围： 0.0 358.125
print("实际纬度范围：", ds.lat.min().values, ds.lat.max().values)
# 实际纬度范围： -89.375 89.375'''

# 定义裁剪范围
# (75,18.125)lon 40;lat 86;uas data
min_lon, max_lon = 73.29, 135.50  # 经度范围
min_lat, max_lat = 3.50, 53.60  # 纬度范围
lon_delta, lat_delta = 1.25, 0.94  # 100km的空间分辨率
#(75,18) lon 2;lat 0;uas data

# 定义MERRA2目标均匀网格，用于空间插值
target_lon = np.arange(73.75, 135 + 0.625, 0.625)  # 0.625°分辨率
target_lat = np.arange(18, 53.5 + 0.5, 0.5)  # 0.5°分辨率

# 检查经度范围
'''if ds.lon.min() >= 0 and ds.lon.max() <= 360:
    # 如果经度范围是0-360，且需要裁剪的范围跨越180°
    if max_lon < min_lon:
        # 将经度范围调整为-180到180
        ds = ds.assign_coords(lon=(ds.lon - 180) % 360 - 180)
        ds = ds.sortby(ds.lon)  # 按经度排序'''

# 裁剪空间范围
'''cropped_data = ds.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))

# 查看裁剪后的数据
print(cropped_data)'''

# 提取变量
'''Var = cropped_data["uas"].isel(time=0).values  # 提取第一个时间步的数据
lon = cropped_data["lon"].values
lat = cropped_data["lat"].values

# 定义颜色映射
blue_color_list = ["#D0D5F2", "#A2A6F2", "#555BD9", "#040FD9"]  # blue
blue_cmap = LinearSegmentedColormap.from_list('custom_cmap', blue_color_list)

# 创建地图
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())

# 绘制数据
show1 = ax1.imshow(Var, extent=[min(lon), max(lon), min(lat), max(lat)], cmap=blue_cmap, transform=ccrs.PlateCarree())

# 设置地图范围
extents = [min(lon), max(lon), min(lat), max(lat)]
ax1.set_extent(extents, ccrs.PlateCarree())

# 添加经纬度网格
ax1.set_xticks(lon[::15], crs=ccrs.PlateCarree())
ax1.set_yticks(lat[::10], crs=ccrs.PlateCarree())
ax1.set_xticklabels(lon[::15], rotation=45)

# 添加中国省界
shp_filelist = ['中国_省_shp_edit/中国_省1.shp', '中国_省_shp_edit/中国_省2.shp']
for shpfile in shp_filelist:
    reader = shpreader.Reader(shpfile)
    for geometry in reader.geometries():
        ax1.add_geometries([geometry], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1)

# 添加颜色条
cbar = plt.colorbar(show1, ax=ax1, location='right', pad=0.03, shrink=0.5, aspect=15)
cbar.set_label("UAS (m/s)")

# 添加标题
ax1.set_title("MERRA2_2015-01-01T03:00:00_uas")

# 显示地图
plt.show()'''



def data_cropped(ds):
    # 裁剪空间范围
    try:
        cropped_data = ds.sel(lon=slice(min_lon - lon_delta, max_lon + lon_delta),
                              lat=slice(min_lat - lat_delta, max_lat + lat_delta))
        return cropped_data
    except Exception as e:
        print(f"裁剪文件 {file_name} 时出错：{e}")
        ds.close()


def data_interp(data):
    # 插值到目标网格
    interpolated_data = data.interp(
        lon=target_lon,
        lat=target_lat,
        method="linear"  # 插值方法，可选 "linear"、"nearest" 等
    )

    # print(ds['vas'].isel(time=1, lat=86, lon=40).values)
    # print(interpolated_data['vas'].isel(time=1, lat=0, lon=2).values)
    return interpolated_data


def data_concat():
    pass


# 定义输入和输出文件夹路径
input_folder = r"E:\DataDownload\CMIP6\vas-temp"
output_folder = r"E:\DataDownload\CMIP6\vas"
# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".nc") and "E3hr" in file_name:  # 确保处理的是NetCDF文件且名称包含指定部分
        file_path = os.path.join(input_folder, file_name)
        print(f"正在处理文件：{file_path}")

        try:
            ds = xr.open_dataset(file_path, engine='netcdf4')  # 尝试使用 NetCDF4 引擎
            print(f"文件 {file_path} 是 NetCDF4 格式")
        except Exception as e:
            ds = xr.open_dataset(file_path, engine='scipy')  # 如果失败，尝试 NetCDF3
            print(f"文件 {file_path} 是 NetCDF3 格式")
        print(ds)
        # 将时间维度从 UTC 转换为 UTC+8
        ds = ds.assign_coords(time=ds.time + timedelta(hours=8))
        # 查看转换后的时间范围
        print(ds.time)

        cropped_data = data_cropped(ds)  # 数据裁剪
        # interpolated_data = data_interp(cropped_data)  # 空间插值

        # 修改文件名，添加“-china”后缀
        base_name, extension = os.path.splitext(file_name)
        output_file_name = f"{base_name}-china{extension}"
        output_file_path = os.path.join(output_folder, output_file_name)

        # 保存处理后的数据为新的NetCDF文件
        cropped_data.to_netcdf(output_file_path)
        print(f"处理后的数据已保存到 {output_file_path}")

        # 关闭原始数据集
        ds.close()

        # 删除原始文件 [建议检查无误后再删除]
        #os.remove(file_path)
        #print(f"原始文件 {file_path} 已删除")

print("所有文件处理完成。")

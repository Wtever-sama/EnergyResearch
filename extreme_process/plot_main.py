import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
from cartopy.io.shapereader import Reader
import geopandas as gpd
import matplotlib.patches as mpatches
from cartopy.mpl.patch import geos_to_path
import matplotlib.path as mpath

plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

def add_north(ax, labelsize=14, loc_x=0.92, loc_y=0.88, width=0.03, height=0.08, pad=0.14):
    """
    画一个比例尺带'N'文字注释
    主要参数如下
    :param ax: 要画的坐标区域 Axes实例 plt.gca()获取即可
    :param labelsize: 显示'N'文字的大小
    :param loc_x: 以文字下部为中心的占整个ax横向比例
    :param loc_y: 以文字下部为中心的占整个ax纵向比例
    :param width: 指南针占ax比例宽度
    :param height: 指南针占ax比例高度
    :param pad: 文字符号占ax比例间隙
    :return: None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)


def draw_china_grid_imshow(data, var_name, lats, lons, c_map=None,v_min=0, v_max=1, start_year=None, end_year=None, units=None, save_filename=None, title_name=None, scene="ssp126"):
    projn = ccrs.LambertConformal(central_longitude=105,
                                  central_latitude=40,
                                  standard_parallels=(25.0, 47.0))
    shp_filelist = ['./Data/中国_省_shp_edit/中国_省1.shp', './Data/中国_省_shp_edit/中国_省2.shp']
    china_shape = gpd.read_file("./Data/中国_省.geojson")
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection=projn)
    ax1.set_global()  # '#FFFFFF'如果最低值为空白
    rainbow_color_list = ['#4E63AC', '#3990BA', '#62BDA7', '#94D4A4', '#C6E89F', '#ECF7A1',
                          '#FFFEBE', '#FEE797', '#FDC574', '#F26944', '#D9444D', '#B11747']
    cmap_select = LinearSegmentedColormap.from_list('custom_cmap', rainbow_color_list)
    lat_ref, lon_ref = lats[1]-lats[0], lons[1]-lons[0]
    extents = [min(lons) - lon_ref / 2, max(lons) + lon_ref / 2, min(lats) - lat_ref / 2, max(lats) + lat_ref / 2]
    print(extents)  # [np.float64(69.5), np.float64(140.5), np.float64(14.5), np.float64(55.5)]
    ax1.set_extent(extents, crs=ccrs.PlateCarree())
    ax1.add_geometries(china_shape["geometry"], crs=ccrs.PlateCarree(),
                       facecolor='none', edgecolor='k', linewidth=.6)
    if c_map is None:
        show1 = ax1.imshow(data[::-1, :], cmap=cmap_select, transform=ccrs.PlateCarree(), extent=extents, vmin=v_min, vmax=v_max)
    else:
        show1 = ax1.imshow(data[::-1, :], cmap=c_map, transform=ccrs.PlateCarree(), extent=extents, vmin=v_min, vmax=v_max)
    if title_name is None:
        ax1.set_title(f'AVG {var_name} in the {scene} period of {start_year}-{end_year}')
    else:
        ax1.set_title(title_name, fontsize=16)
    for shpfile in shp_filelist:
        ax1.add_geometries(Reader(shpfile).geometries(), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1)
    cbar = plt.colorbar(show1, ax=ax1, location='right', pad=0.03, shrink=0.5, aspect=15)
    if units is None:
        cbar.set_label(f"{var_name}", fontsize=14)
    else:
        cbar.set_label(f"{var_name}({units})", fontsize=14)
    ax3 = plt.gca()
    add_north(ax3)
    # 添加海岸线，地图经纬网等
    gls = ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                        color='None', linestyle='dashed', linewidth=0.3,
                        y_inline=False, x_inline=False,
                        rotate_labels=0, xpadding=5,
                        xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 10),
                        xlabel_style={"size": 12, "weight": "bold"},
                        ylabel_style={"size": 12, "weight": "bold"}
                        )
    gls.top_labels = False  # 取消顶部和右部的标签，并使图片延展
    gls.right_labels = False
    plt.tight_layout()
    # 添加南海小地图
    ax2 = fig.add_axes([0.68, 0.17, 0.12, 0.25], projection=projn)
    ax2.set_extent([104.5, 125, 0, 26])
    # 设置网格点
    lb = ax2.gridlines(draw_labels=False, x_inline=False, y_inline=False,
                       linewidth=0.1, color='none', alpha=0.8,
                       linestyle='--')
    ax2.add_geometries(china_shape["geometry"], crs=ccrs.PlateCarree(),
                       facecolor='none', edgecolor='k', linewidth=.3)
    if c_map is None:
        show2 = ax2.imshow(data[::-1, :], cmap=cmap_select, transform=ccrs.PlateCarree(), extent=extents, vmin=v_min, vmax=v_max)
    else:
        show2 = ax2.imshow(data[::-1, :], cmap=c_map, transform=ccrs.PlateCarree(), extent=extents, vmin=v_min,
                           vmax=v_max)
    # 保存图片
    if save_filename is None:
        save_filename = f"./Pictures/the average of {var_name} in the {scene} period of {start_year}-{end_year}.png"
    # plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    # print(f"{save_filename} has been saved.")
    return ax1



def draw_china_grid_v2(data, var_name, lats, lons, c_map=None,v_min=0, v_max=1, start_year=None, end_year=None, units=None, save_filename=None, title_name=None, scene="ssp126"):
    projn = ccrs.LambertConformal(central_longitude=105, central_latitude=40, standard_parallels=(25.0, 47.0))
    shp_filelist = ['./Data/中国_省_shp_edit/中国_省1.shp', './Data/中国_省_shp_edit/中国_省2.shp']
    china_shape = gpd.read_file("./Data/中国_省.geojson")
    from shapely.ops import unary_union
    china_geom = unary_union(china_shape.geometry)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111, projection=projn)
    ax1.set_global()  # '#FFFFFF'如果最低值为空白
    rainbow_color_list = ['#4E63AC', '#3990BA', '#62BDA7', '#94D4A4', '#C6E89F', '#ECF7A1',
                          '#FFFEBE', '#FEE797', '#FDC574', '#F26944', '#D9444D', '#B11747']
    cmap_define = LinearSegmentedColormap.from_list('custom_cmap', rainbow_color_list)
    final_cmap = c_map if c_map is not None else cmap_define
    lat_ref, lon_ref = lats[1]-lats[0], lons[1]-lons[0]
    extents = [min(lons) - lon_ref / 2, max(lons) + lon_ref / 2, min(lats) - lat_ref / 2, max(lats) + lat_ref / 2]
    # print(extents)  # [np.float64(69.5), np.float64(140.5), np.float64(14.5), np.float64(55.5)]
    ax1.set_extent([73,136,16,54], crs=ccrs.PlateCarree())
    ax1.add_geometries(china_shape["geometry"], crs=ccrs.PlateCarree(),
                       facecolor='none', edgecolor='k', linewidth=.6)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    show1 = ax1.pcolormesh(lon_grid, lat_grid, data, cmap=final_cmap,transform=ccrs.PlateCarree(), vmin=v_min, vmax=v_max, shading='auto')
    # 构造裁切路径
    paths = geos_to_path(china_geom)  # 将 Shapely 几何体转换为 Matplotlib 的 Path 列表
    combined_path = mpath.Path.make_compound_path(*paths)  # 合并所有路径（包括岛屿等多个闭合环）
    patch1 = mpatches.PathPatch(combined_path, transform=ccrs.PlateCarree(), facecolor='none')  # 创建 Patch 补丁
    ax1.add_patch(patch1)
    show1.set_clip_path(patch1)
    ax1.set_title(title_name, fontsize=16)
    for shpfile in shp_filelist:
        ax1.add_geometries(Reader(shpfile).geometries(), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1)
    cbar = plt.colorbar(show1, ax=ax1, location='right', pad=0.03, shrink=0.5, aspect=15)
    cbar.set_label(f"{var_name}({units})" if units is not None else f"{var_name}", fontsize=14)
    ax3 = plt.gca()
    add_north(ax3)
    # 添加海岸线，地图经纬网等
    gls = ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                        color='gray', linestyle='dashed', linewidth=0.3, alpha=0.5,
                        y_inline=False, x_inline=False,rotate_labels=0, xpadding=5,
                        xlocs=np.arange(70, 140, 10), ylocs=np.arange(10, 60, 10),
                        xlabel_style={"size": 12, "weight": "bold"},ylabel_style={"size": 12, "weight": "bold"})
    gls.top_labels, gls.right_labels = False, False  # 取消顶部和右部的标签，并使图片延展
    # 添加南海小地图
    ax2 = fig.add_axes([0.68, 0.16, 0.12, 0.25], projection=projn)  # [left, bottom, width, height]
    ax2.set_extent([104.5, 125, 0, 26])
    # 设置网格点
    ax2.gridlines(draw_labels=False, x_inline=False, y_inline=False,linewidth=0.1, color='none', alpha=0.8,linestyle='--')
    ax2.add_geometries(china_shape["geometry"], crs=ccrs.PlateCarree(),facecolor='none', edgecolor='k', linewidth=.3)
    show2 = ax2.imshow(data[::-1, :], cmap=final_cmap, transform=ccrs.PlateCarree(), extent=extents, vmin=v_min, vmax=v_max)
    # 构造裁切路径
    patch2 = mpatches.PathPatch(combined_path, transform=ccrs.PlateCarree(), facecolor='none')
    ax2.add_patch(patch2)
    show2.set_clip_path(patch2)
    plt.subplots_adjust(bottom=0.05, top=0.99, left=0.10, right=0.99)
    if save_filename is not None:  # 保存图片
        plt.savefig(save_filename, dpi=300)
        print(f"{save_filename} has been saved.")
    plt.show()
    return ax1


if __name__=="__main__":
    # 以某个nc文件为例
    ds = xr.open_dataset(rf"D:\PythonCodes\2026NengjingCompetetion\NinjiaDemand-CMIP6\Masks\pop2018_cmip6.nc")
    var_da = ds['pop'].values / 1000000  # 百万
    lat_da = ds['lat'].values
    lon_da = ds['lon'].values
    draw_china_grid_v2(var_da, 'pop', lat_da, lon_da, v_max=10, save_filename='./Plots/population_1deg.png')


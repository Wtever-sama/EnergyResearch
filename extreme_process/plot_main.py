import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
from cartopy.io.shapereader import Reader
import geopandas as gpd
import pandas as pd
import matplotlib.patches as mpatches

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


def draw_china_grid(data, var_name, lats, lons, c_map=None,v_min=0, v_max=1, start_year=None, end_year=None, units=None, save_filename=None, title_name=None, scene="ssp126"):
    projn = ccrs.LambertConformal(central_longitude=105,
                                  central_latitude=40,
                                  standard_parallels=(25.0, 47.0))
    shp_filelist = ['../pythonProject/中国_省_shp_edit/中国_省1.shp', '../pythonProject/中国_省_shp_edit/中国_省2.shp']
    china_shape = gpd.read_file("../pythonProject/中国_省.geojson")
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection=projn)
    ax1.set_global()  # '#FFFFFF'如果最低值为空白
    rainbow_color_list = ['#4E63AC', '#3990BA', '#62BDA7', '#94D4A4', '#C6E89F', '#ECF7A1',
                          '#FFFEBE', '#FEE797', '#FDC574', '#F26944', '#D9444D', '#B11747']
    cmap_select = LinearSegmentedColormap.from_list('custom_cmap', rainbow_color_list)
    lat_ref, lon_ref = lats[0]-lats[1], lons[0]-lons[1]
    extents = [min(lons) - lon_ref / 2, max(lons) + lon_ref / 2, min(lats) - lat_ref / 2, max(lats) + lat_ref / 2]
    # print(extents)  # [73.4375, 135.3125, 17.75, 53.75]
    ax1.set_extent(extents, crs=ccrs.PlateCarree())
    ax1.add_geometries(china_shape["geometry"], crs=ccrs.PlateCarree(),
                       facecolor='none', edgecolor='k', linewidth=.6)
    if c_map is None:
        show1 = ax1.imshow(data[::-1, :], cmap=cmap_select, transform=ccrs.PlateCarree(), extent=extents, vmin=v_min, vmax=v_max)
    else:
        show1 = ax1.imshow(data[::-1, :], cmap=c_map, transform=ccrs.PlateCarree(), extent=extents, vmin=v_min,
                           vmax=v_max)
    if title_name is None:
        ax1.set_title(f'AVG {var_name} in the {scene} period of {start_year}-{end_year}')
    else:
        ax1.set_title(title_name, fontsize=16)
    for shpfile in shp_filelist:
        ax1.add_geometries(Reader(shpfile).geometries(),
                           crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1)
    # cbar = plt.colorbar(show1, ax=ax1, orientation='horizontal', pad=0.06, shrink=0.8, aspect=30)
    cbar = plt.colorbar(show1, ax=ax1, location='right', pad=0.03,
                        shrink=0.5, aspect=15)
    # cbar.set_ticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
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
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"{save_filename} has been saved.")
    return ax1



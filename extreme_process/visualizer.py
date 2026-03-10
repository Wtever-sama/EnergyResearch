''' 绘制各种变量的2040-2060年平均极端事件平均地理图谱, 支持南海诸岛子图与1:1比例 '''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import regionmask
import geopandas as gpd
from pathlib import Path
import os
import json
from shapely.geometry import shape
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def add_north(ax, labelsize=14, loc_x=0.92, loc_y=0.88, width=0.03, height=0.08, pad=0.14):
    """在地图轴上添加与 plot_main 一致风格的指北针。"""
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * 0.5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * 0.5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * 0.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(
        s='N',
        x=minx + xlen * loc_x,
        y=miny + ylen * (loc_y - pad + height),
        fontsize=labelsize,
        horizontalalignment='center',
        verticalalignment='bottom'
    )
    ax.add_patch(triangle)


def get_spatial_contribution_percentage(file_path: str | Path, variable_name: str, year_range: list[int, int]) -> xr.DataArray:
    """
    计算各格点事件数量占【所有发生过事件的格点】年均总事件数量的百分比
    """
    start_year, end_year = year_range
    n_years = end_year - start_year + 1
    
    ds = xr.open_dataset(file_path)
    da = ds[variable_name].sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # 1. 识别上升沿并计算每个格点的年均发生次数
    mask = da.fillna(0)
    is_start = (mask == 1) & (mask.shift(time=1) != 1)
    grid_annual_freq = is_start.sum(dim='time') / float(n_years) #
    
    # 2. 关键修改：分母只对发生过事件的格点求和，剔除零频区域
    # 这样可以让高频区域的占比显著提升
    active_grids = grid_annual_freq.where(grid_annual_freq > 0)
    total_annual_avg = float(active_grids.sum().values) #
    
    if total_annual_avg == 0:
        return grid_annual_freq * 0

    # 3. 计算占比
    spatial_contribution = (grid_annual_freq / total_annual_avg) * 100
    
    return spatial_contribution.astype(np.float32)


def plot_frequency(freq_data: xr.DataArray, 
                   output_dir: str, 
                   energy_label: str, 
                   scenario: str, 
                   vmin: float, 
                   vmax: float, 
                   geo_path: str | Path,
                   cmap: str = 'YlOrRd') -> str | Path:
    # 与 plot_main.py 保持一致：LambertConformal 投影 + 自定义彩虹色带 + imshow
    use_geopandas = True
    merged = None
    try:
        gdf = gpd.read_file(geo_path)
        try:
            gdf.geometry = gdf.geometry.make_valid()
        except Exception:
            pass
    except Exception:
        use_geopandas = False
        with open(geo_path, 'r', encoding='utf-8') as f:
            js_data = json.load(f)
        features = js_data['features'] if 'features' in js_data else js_data
        shapes = [shape(feat['geometry']) for feat in features]
        merged = unary_union(shapes)

    if use_geopandas:
        mask = regionmask.mask_3D_geopandas(gdf, freq_data.lon, freq_data.lat)
        freq_masked = freq_data.where(mask.any('region'))
        boundary_geoms = gdf.geometry
    else:
        lon_vals = freq_data.lon.values
        lat_vals = freq_data.lat.values
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        pts = np.vstack((lon2d.ravel(), lat2d.ravel())).T
        masks = np.zeros(len(pts), dtype=bool)
        polys = list(merged.geoms) if hasattr(merged, 'geoms') else [merged]
        for p in polys:
            try:
                path = mpath.Path(np.array(p.exterior.coords))
            except Exception:
                continue
            masks |= path.contains_points(pts)
        mask2d = masks.reshape(lon2d.shape)
        mask_da = xr.DataArray(mask2d, dims=('lat', 'lon'), coords={'lat': lat_vals, 'lon': lon_vals})
        freq_masked = freq_data.where(mask_da)
        boundary_geoms = [merged]

    projn = ccrs.LambertConformal(
        central_longitude=105,
        central_latitude=40,
        standard_parallels=(25.0, 47.0)
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projn)
    ax.set_global()

    lats = freq_masked['lat'].values
    lons = freq_masked['lon'].values
    lat_ref = lats[0] - lats[1] if lats.size > 1 else 1.0
    lon_ref = lons[0] - lons[1] if lons.size > 1 else 1.0
    extents = [
        float(np.nanmin(lons) - lon_ref / 2),
        float(np.nanmax(lons) + lon_ref / 2),
        float(np.nanmin(lats) - lat_ref / 2),
        float(np.nanmax(lats) + lat_ref / 2),
    ]
    ax.set_extent(extents, crs=ccrs.PlateCarree())
    ax.add_geometries(boundary_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.6)

    rainbow_color_list = [
        '#4E63AC', '#3990BA', '#62BDA7', '#94D4A4', '#C6E89F', '#ECF7A1',
        '#FFFEBE', '#FEE797', '#FDC574', '#F26944', '#D9444D', '#B11747'
    ]
    cmap_select = LinearSegmentedColormap.from_list('custom_cmap', rainbow_color_list)
    selected_cmap = cmap_select if cmap == 'YlOrRd' else cmap

    show_main = ax.imshow(
        freq_masked.values[::-1, :],
        cmap=selected_cmap,
        transform=ccrs.PlateCarree(),
        extent=extents,
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(show_main, ax=ax, location='right', pad=0.03, shrink=0.5, aspect=15)
    cbar.set_label('空间贡献占比 (%)', fontsize=14)

    add_north(ax)

    gls = ax.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        color='None',
        linestyle='dashed',
        linewidth=0.3,
        y_inline=False,
        x_inline=False,
        rotate_labels=0,
        xpadding=5,
        xlocs=range(-180, 180, 10),
        ylocs=range(-90, 90, 10),
        xlabel_style={"size": 12, "weight": "bold"},
        ylabel_style={"size": 12, "weight": "bold"}
    )
    gls.top_labels = False
    gls.right_labels = False

    plt.tight_layout()

    sub_ax = fig.add_axes([0.68, 0.17, 0.12, 0.25], projection=projn)
    sub_ax.set_extent([104.5, 125, 0, 26], crs=ccrs.PlateCarree())
    sub_ax.gridlines(
        draw_labels=False,
        x_inline=False,
        y_inline=False,
        linewidth=0.1,
        color='none',
        alpha=0.8,
        linestyle='--'
    )
    sub_ax.add_geometries(boundary_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.3)
    sub_ax.imshow(
        freq_masked.values[::-1, :],
        cmap=selected_cmap,
        transform=ccrs.PlateCarree(),
        extent=extents,
        vmin=vmin,
        vmax=vmax
    )

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{energy_label}_{scenario}_Frequency_Map.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"频率图已保存至: {save_path}")
    # plt.show()
    plt.close()
    return save_path

if __name__ == "__main__":
    geo_boundary = "G:/extreme_analysis/geoshape_file/CHN_ALI/CHN_ALIYUN.geojson"
    # 输入文件：12h 持续时间判断后的结果
    fpath = "G:/extreme_analysis/results/Wind/Wind_ssp126_V1_Reliability_12h.nc"
    year_range = [2040, 2060]
    variable_name = "event_flag" # duration_judge.py 生成的变量名

    # 1. 获取空间贡献比例数据
    spatial_perc = get_spatial_contribution_percentage(fpath, variable_name, year_range)

    # 2. 设定输出目录
    fig_output_dir = "G:/extreme_analysis/results/Wind/spatial_contribution_figures"
    os.makedirs(fig_output_dir, exist_ok=True)
    
    # 3. 绘图
    # 注意：由于单个格点的占比通常很小，vmin/vmax 可能需要根据结果动态调整
    max_val = float(spatial_perc.max().values)
    
    plot_frequency(freq_data=spatial_perc,
                   output_dir=fig_output_dir,
                   energy_label="Wind_12h_Spatial_Contribution",
                   scenario="ssp126",
                   vmin=0,
                   vmax=max_val if max_val > 0 else 1, # 动态设置最大值以增强对比度
                   geo_path=geo_boundary)
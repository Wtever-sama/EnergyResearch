''' 使用期望偏差的 (-1) 绘制平均强度分布图 '''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import cartopy.crs as ccrs
from cartopy.mpl.patch import geos_to_path
from cartopy.io.shapereader import Reader
import geopandas as gpd
from pathlib import Path
import os
import json
from shapely.geometry import shape
from shapely.geometry import LineString
from shapely.ops import unary_union
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

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


def plot_delta_x_spatial_mean(fpath, geo_boundary, province_boundary, output_dir, save_name):
    print(f"正在处理文件: {os.path.basename(fpath)}")
    
    # 1. 加载并计算 2040-2060 时间均值
    ds = xr.open_dataset(fpath, chunks={'time': 10000})
    # 选取时间段并计算均值
    spatial_mean = ds.delta_x.sel(time=slice('2040', '2060')).mean(dim='time').compute()
    
    # 获取坐标与值（lat/lon 顺序与 frequency_visualizer.py 保持一致）
    intensity_plot = spatial_mean.transpose('lat', 'lon')
    lats = np.asarray(intensity_plot['lat'].values, dtype=float)
    lons = np.asarray(intensity_plot['lon'].values, dtype=float)
    data = np.asarray(intensity_plot.values, dtype=float)
    # 将绘图数据正负值符号颠倒
    data = -data

    # 2. 读取国界几何
    use_geopandas = True
    merged = None
    try:
        gdf = gpd.read_file(geo_boundary)
        try:
            gdf.geometry = gdf.geometry.make_valid()
        except Exception:
            pass
    except Exception:
        use_geopandas = False
        with open(geo_boundary, 'r', encoding='utf-8') as f:
            js_data = json.load(f)
        features = js_data['features'] if 'features' in js_data else js_data
        shapes = [shape(feat['geometry']) for feat in features]
        merged = unary_union(shapes)

    if use_geopandas:
        boundary_geoms = gdf.geometry
        china_geom = unary_union(gdf.geometry)
    else:
        boundary_geoms = [merged]
        china_geom = merged

    province_geoms = list(Reader(str(Path(province_boundary))).geometries())

    # 3. 设置绘图投影与画布比例
    projn = ccrs.LambertConformal(
        central_longitude=105,
        central_latitude=40,
        standard_parallels=(25.0, 47.0)
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projn)
    ax.set_global()

    def _coord_to_edges(coord: np.ndarray) -> np.ndarray:
        coord = np.asarray(coord, dtype=float)
        if coord.size == 1:
            return np.array([coord[0] - 0.5, coord[0] + 0.5], dtype=float)
        mids = (coord[:-1] + coord[1:]) / 2.0
        first = coord[0] - (coord[1] - coord[0]) / 2.0
        last = coord[-1] + (coord[-1] - coord[-2]) / 2.0
        return np.concatenate(([first], mids, [last]))

    lon_edges = _coord_to_edges(lons)
    lat_edges = _coord_to_edges(lats)
    extents = [
        float(np.nanmin(lon_edges)),
        float(np.nanmax(lon_edges)),
        float(np.nanmin(lat_edges)),
        float(np.nanmax(lat_edges)),
    ]
    ax.set_extent(extents, crs=ccrs.PlateCarree())

    # 4. 颜色映射设置
    # delta_x 均值通常在 0 附近波动，使用分层色彩映射 (Diverging Colormap)
    # 使用 TwoSlopeNorm 确保 0 值对应白色
    vmax = np.nanmax(np.abs(data)) * 0.8  # 适当收紧范围增加对比度
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r  # 蓝色表示正裕度，红色表示负偏移（接近阈值）

    # 5. 绘制主图
    im = ax.pcolormesh(
        lon_edges,
        lat_edges,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading='flat',
        zorder=2,
        antialiased=False
    )

    # 6. 添加边界
    ax.add_geometries(boundary_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.6, zorder=20)
    ax.add_geometries(province_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.35, zorder=21)
    if hasattr(ax, 'outline_patch'):
        ax.outline_patch.set_linewidth(0.8)
        ax.outline_patch.set_edgecolor('k')
    
    # 7. 经纬线：曲线化 + 国境内不显示
    gls = ax.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        color='gray',
        linestyle='dashed',
        linewidth=0.3,
        alpha=0.5,
        zorder=0,
        y_inline=False,
        x_inline=False,
        rotate_labels=0,
        xpadding=5,
        xlocs=np.arange(70, 150, 10),
        ylocs=np.arange(10, 70, 10),
        xlabel_style={"size": 12, "weight": "bold"},
        ylabel_style={"size": 12, "weight": "bold"}
    )
    if hasattr(gls, 'xlines'):
        gls.xlines = False
    if hasattr(gls, 'ylines'):
        gls.ylines = False
    gls.top_labels = False
    gls.right_labels = False

    lon_min, lon_max, lat_min, lat_max = extents
    x_ticks = np.arange(70, 150, 10)
    y_ticks = np.arange(10, 70, 10)

    def _plot_outside_parts(geom_obj):
        if geom_obj is None or geom_obj.is_empty:
            return
        if geom_obj.geom_type == 'LineString':
            xy = np.asarray(geom_obj.coords)
            if xy.shape[0] >= 2:
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    transform=ccrs.PlateCarree(),
                    color='gray',
                    linestyle='dashed',
                    linewidth=0.3,
                    alpha=0.5,
                    zorder=0,
                )
            return
        for sub_geom in getattr(geom_obj, 'geoms', []):
            _plot_outside_parts(sub_geom)

    sample_n = 721
    lat_samples = np.linspace(float(lat_min), float(lat_max), sample_n)
    lon_samples = np.linspace(float(lon_min), float(lon_max), sample_n)

    for x_tick in x_ticks:
        meridian_xy = np.column_stack([np.full(sample_n, float(x_tick)), lat_samples])
        full_meridian = LineString(meridian_xy)
        outside_meridian = full_meridian.difference(china_geom)
        _plot_outside_parts(outside_meridian)

    for y_tick in y_ticks:
        parallel_xy = np.column_stack([lon_samples, np.full(sample_n, float(y_tick))])
        full_parallel = LineString(parallel_xy)
        outside_parallel = full_parallel.difference(china_geom)
        _plot_outside_parts(outside_parallel)

    # 8. 矢量裁剪：国境外色块不显示
    clip_paths = geos_to_path(china_geom)
    if clip_paths:
        combined_path = mpath.Path.make_compound_path(*clip_paths)
        main_clip_patch = mpatches.PathPatch(
            combined_path,
            transform=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='none',
            linewidth=0,
            antialiased=False
        )
        ax.add_patch(main_clip_patch)
        im.set_clip_path(main_clip_patch)
    im.set_clip_box(ax.bbox)

    # 9. 添加南海诸岛子图
    sub_ax = fig.add_axes([0.66, 0.19, 0.12, 0.25], projection=projn)
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
    sub_ax.add_geometries(province_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.2)
    show_sub = sub_ax.pcolormesh(
        lon_edges,
        lat_edges,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading='flat',
        antialiased=False
    )
    if hasattr(sub_ax, 'outline_patch'):
        sub_ax.outline_patch.set_linewidth(0.8)
        sub_ax.outline_patch.set_edgecolor('k')
    if clip_paths:
        sub_clip_patch = mpatches.PathPatch(
            combined_path,
            transform=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='none',
            linewidth=0,
            antialiased=False
        )
        sub_ax.add_patch(sub_clip_patch)
        show_sub.set_clip_path(sub_clip_patch)
    show_sub.set_clip_box(sub_ax.bbox)
    
    # 10. 颜色条
    cbar = plt.colorbar(im, ax=ax, location='right', pad=0.03, shrink=0.5, aspect=15)
    cbar.set_label('极端事件平均强度', fontsize=14)

    add_north(ax)

    plt.tight_layout()

    # 11. 保存
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {save_path}")
    plt.close()


if __name__ == "__main__":
    # energys = ["Solar", "Wind"]
    # scenarios = ["ssp126", "ssp245", "ssp585"]

    # for energy in energys:
    #     for scenario in scenarios:
    #         print("绘制 {} - {} 情景的空间强度图...".format(energy, scenario))
    #         # 路径配置
    #         input_nc = f"G:/extreme_analysis/results/{energy}/{energy}_{scenario}_V2_delta_x.nc"
            
    #         # 边界文件
    #         geo_json = "G:/extreme_analysis/geoshape_file/CHN_ALI/CHN_ALIYUN.geojson"
    #         province_shp = "G:/extreme_analysis/geoshape_file/gadm41_CHN_shp/gadm41_CHN_1.shp"
            
    #         out_dir = f"G:/extreme_analysis/results/{energy}/spatial_intensity"
    #         os.makedirs(out_dir, exist_ok=True)

    #         save_name = f"{energy}_{scenario}_Intensity_Map.png"
            
    #         plot_delta_x_spatial_mean(input_nc, geo_json, province_shp, out_dir, save_name)
    # print("所有图像绘制完成！")
    input_nc = "G:/extreme_analysis/results/Solar/Solar_ssp126_V2_delta_x.nc"
    geo_json = "G:/extreme_analysis/geoshape_file/CHN_ALI/CHN_ALIYUN.geojson"
    province_shp = "G:/extreme_analysis/geoshape_file/gadm41_CHN_shp/gadm41_CHN_1.shp"
    outdir = "G:/extreme_analysis/results/Solar/spatial_intensity"
    save_name = "Solar_ssp126_Intensity_Map_tmp.png"
    plot_delta_x_spatial_mean(input_nc, geo_json, province_shp, outdir, save_name)
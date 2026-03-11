''' 绘制各种变量的2040-2060年平均极端事件平均地理图谱, 支持南海诸岛子图与1:1比例 '''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.patch import geos_to_path
from cartopy.io.shapereader import Reader
import regionmask
import geopandas as gpd
from pathlib import Path
import os
import json
from shapely.geometry import shape
from shapely.geometry import LineString
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


def get_spatial_frequency_percentage(file_path: str | Path, variable_name: str, year_range: list[int, int]) -> xr.DataArray:
    """计算各格点年均事件频率占所有格点总年均频率的百分比。"""
    start_year, end_year = year_range
    n_years = end_year - start_year + 1
    
    ds = xr.open_dataset(file_path)
    da = ds[variable_name].sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # 1. 识别上升沿并计算每个格点的年均发生次数
    mask = da.fillna(0)
    is_start = (mask == 1) & (mask.shift(time=1) != 1)
    grid_annual_freq = is_start.sum(dim='time') / float(n_years) 

    # 2. 计算所有格点的年均事件总频率（步骤1之后，time维已被聚合）
    annual_frequency = float(grid_annual_freq.sum(skipna=True).values)
    # print("type(annual_frequency):", type(annual_frequency))
    # print("annual_frequency:", annual_frequency)

    if annual_frequency == 0:
        return (grid_annual_freq * 0).astype(np.float32)

    # 3. 计算每个格点年均发生次数占总频率比例（百分比）
    grid_annual_freq = grid_annual_freq / annual_frequency * 100

    # 4. 反向验证所有格点的频率比例之和是否为 100
    assert np.isclose(grid_annual_freq.sum(skipna=True).values, 100.0, atol=1e-6), "频率比例之和不为100，可能存在计算错误！"
    
    return grid_annual_freq.astype(np.float32)


def plot_frequency(freq_data: xr.DataArray, 
                   output_dir: str, 
                   energy_label: str, 
                   scenario: str, 
                   vmin: float, 
                   vmax: float, 
                   geo_path: str | Path,
                   province_shp_path: str | Path | None = None,
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

    # 可选：读取省级边界（类比 plot_main.py 的 shp_filelist 叠加边界）
    province_geoms = None
    if province_shp_path is not None:
        province_path = Path(province_shp_path)
        # 若传入的是国家级 gadm41_CHN_0.shp，优先尝试同目录下的省级 gadm41_CHN_1.shp
        if province_path.name.endswith('_0.shp'):
            level1_candidate = province_path.with_name(province_path.name.replace('_0.shp', '_1.shp'))
            if level1_candidate.exists():
                print(f"检测到省级边界文件，优先使用: {level1_candidate}")
                province_path = level1_candidate
            else:
                print("提示: 当前传入为国家级 _0.shp；若需省级边界，请提供 _1.shp。")

        if province_path.exists():
            try:
                # 与 plot_main.py 一致，直接用 Reader 读取 shp 几何，规避 geopandas/shapely 兼容问题
                province_geoms = list(Reader(str(province_path)).geometries())
            except Exception as e:
                print(f"省级边界读取失败，将跳过省界绘制: {province_path}; error={e}")
        else:
            print(f"省级边界文件不存在，将跳过省界绘制: {province_path}")

    if use_geopandas:
        mask = regionmask.mask_3D_geopandas(gdf, freq_data.lon, freq_data.lat)
        freq_masked = freq_data.where(mask.any('region'))
        boundary_geoms = gdf.geometry
        china_geom = unary_union(gdf.geometry)
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
        china_geom = merged

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

    # 用坐标差分的绝对值估计网格分辨率，避免坐标升序/降序引起的负步长错位
    lat_ref = float(np.nanmedian(np.abs(np.diff(lats)))) if lats.size > 1 else 1.0
    lon_ref = float(np.nanmedian(np.abs(np.diff(lons)))) if lons.size > 1 else 1.0
    print(f"检测到网格分辨率: lat_ref={lat_ref:.4f}°, lon_ref={lon_ref:.4f}°")
    if not np.isclose(lat_ref, 1.0, atol=1e-6) or not np.isclose(lon_ref, 1.0, atol=1e-6):
        print("提示: 当前网格分辨率不是严格1°，将按检测到的分辨率自动设置 extent。")

    extents = [
        float(np.nanmin(lons) - lon_ref / 2),
        float(np.nanmax(lons) + lon_ref / 2),
        float(np.nanmin(lats) - lat_ref / 2),
        float(np.nanmax(lats) + lat_ref / 2),
    ]
    ax.set_extent(extents, crs=ccrs.PlateCarree())
    ax.add_geometries(boundary_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.6, zorder=20)
    if province_geoms is not None and len(province_geoms) > 0:
        ax.add_geometries(province_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.35, zorder=21)

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
        vmax=vmax,
        zorder=2
    )

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
        xlocs=np.arange(70, 140, 10),
        ylocs=np.arange(10, 60, 10),
        xlabel_style={"size": 12, "weight": "bold"},
        ylabel_style={"size": 12, "weight": "bold"}
    )
    if hasattr(gls, 'xlines'):
        gls.xlines = False
    if hasattr(gls, 'ylines'):
        gls.ylines = False
    gls.top_labels = False
    gls.right_labels = False

    # 仅在国界外绘制经纬线，避免覆盖中国国界内区域
    lon_min, lon_max, lat_min, lat_max = extents
    x_ticks = np.arange(70, 140, 10)
    y_ticks = np.arange(10, 60, 10)

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

    # 使用高密度采样构造经纬线，避免投影后出现明显折线/断裂感
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

    # 与 draw_china_grid_v2 对齐：构造几何裁剪路径，确保图像严格落在中国边界内
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
        show_main.set_clip_path(main_clip_patch)

    cbar = plt.colorbar(show_main, ax=ax, location='right', pad=0.03, shrink=0.5, aspect=15)
    cbar.set_label('空间贡献率 (%)', fontsize=14)

    add_north(ax)

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
    if province_geoms is not None and len(province_geoms) > 0:
        sub_ax.add_geometries(province_geoms, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.2)
    show_sub = sub_ax.imshow(
        freq_masked.values[::-1, :],
        cmap=selected_cmap,
        transform=ccrs.PlateCarree(),
        extent=extents,
        vmin=vmin,
        vmax=vmax
    )

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

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{energy_label}_{scenario}_Frequency_Map.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"频率图已保存至: {save_path}")
    # plt.show()
    plt.close()
    return save_path

if __name__ == "__main__":
    geo_boundary = "G:/extreme_analysis/geoshape_file/CHN_ALI/CHN_ALIYUN.geojson"
    province_boundary = "G:/extreme_analysis/geoshape_file/gadm41_CHN_shp/gadm41_CHN_0.shp"
    # 输入文件：12h 持续时间判断后的结果
    fpath = "G:/extreme_analysis/results/Wind/Wind_ssp126_V1_1h.nc"
    year_range = [2040, 2060]
    variable_name = "event_flag" # duration_judge.py 生成的变量名

    # 1. 获取年均事件频率数据
    spatial_freq = get_spatial_frequency_percentage(fpath, variable_name, year_range)

    # 2. 设定输出目录
    fig_output_dir = "G:/extreme_analysis/results/Wind/spatial_contribution_figures"
    os.makedirs(fig_output_dir, exist_ok=True)
    
    # 3. 绘图
    max_val = float(spatial_freq.max().values)
    
    plot_frequency(freq_data=spatial_freq,
                   output_dir=fig_output_dir,
                   energy_label="Wind_Spatial_Contribution",
                   scenario="ssp126",
                   vmin=0,
                   vmax=max_val if max_val > 0 else 1,
                   geo_path=geo_boundary,
                   province_shp_path=province_boundary)
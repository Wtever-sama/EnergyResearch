''' 绘制各种变量的2040-2060年平均极端事件平均地理图谱, 支持南海诸岛子图与1:1比例 '''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

def get_annual_frequency(file_path: str | Path) -> xr.DataArray | None:
    if not os.path.exists(file_path): 
        return None
    ds = xr.open_dataset(file_path)
    da = ds['event_flag']
    mask = da.fillna(0)
    is_start = (mask == 1) & (mask.shift(time=1) != 1)
    freq = is_start.sum(dim='time') / 21.0
    return freq

def plot_frequency(freq_data: xr.DataArray, 
                   output_dir: str, 
                   energy_label: str, 
                   scenario: str, 
                   vmin: float, 
                   vmax: float, 
                   geo_path: str | Path,
                   cmap: str = 'YlOrRd') -> str | Path:
    
    # --- 修复逻辑：鲁棒性读取非标准 GeoJSON，解决 NotImplementedError ---
    use_geopandas = True
    merged = None
    try:
        gdf = gpd.read_file(geo_path)
        # ensure valid geometry column exists
        try:
            gdf.geometry = gdf.geometry.make_valid()
        except Exception:
            pass
    except Exception:
        # 回退：手动解析 GeoJSON（避免 geopandas/fiona 的 JSON decode 问题）
        use_geopandas = False
        with open(geo_path, 'r', encoding='utf-8') as f:
            js_data = json.load(f)
        features = js_data['features'] if 'features' in js_data else js_data
        # 直接构造 shapely 几何列表，然后合并为一个
        shapes = [shape(feat['geometry']) for feat in features]
        merged = unary_union(shapes)

    # 提高插值分辨率
    freq_data = freq_data.interp(
        lon=np.arange(freq_data.lon.min(), freq_data.lon.max(), 0.1), 
        lat=np.arange(freq_data.lat.min(), freq_data.lat.max(), 0.1), 
        method='linear'
    )
    
    if use_geopandas:
        mask = regionmask.mask_3D_geopandas(gdf, freq_data.lon, freq_data.lat)
        freq_masked = freq_data.where(mask.any('region'))
    else:
        # 使用 shapely geometry + matplotlib.path 在网格上做点内判断，构造掩膜
        lon_vals = freq_data.lon.values
        lat_vals = freq_data.lat.values
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
        pts = np.vstack((lon2d.ravel(), lat2d.ravel())).T
        masks = np.zeros(len(pts), dtype=bool)
        # merged 可能是多多边形或单多边形
        if hasattr(merged, 'geoms'):
            polys = list(merged.geoms)
        else:
            polys = [merged]
        for p in polys:
            try:
                path = mpath.Path(np.array(p.exterior.coords))
            except Exception:
                # 跳过无法处理的子几何
                continue
            masks |= path.contains_points(pts)
        mask2d = masks.reshape(lon2d.shape)
        # mask2d 的维度是 (lat, lon)
        mask_da = xr.DataArray(mask2d, dims=('lat', 'lon'), coords={'lat': lat_vals, 'lon': lon_vals})
        freq_masked = freq_data.where(mask_da)

    # --- 关键修复：设置正确的地图比例 ---
    # 计算中国地图范围的比例，确保地图不被拉伸
    lon_min, lon_max = 73, 135
    lat_min, lat_max = 15, 54
    
    # 计算经纬度范围的比例
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    # 根据经纬度范围设置合适的图形宽高比
    # 注意：PlateCarree投影中，经度纬度的单位长度在赤道处相等，但随纬度变化
    # 这里使用近似的赤道纬度(约23.5度)的cos值进行修正
    cos_mid_lat = np.cos(np.radians(23.5))  # 中国大约的中纬度
    aspect_ratio = (lon_range * cos_mid_lat) / lat_range
    
    # 设置图形尺寸，保持正确的比例
    fig_width = 10
    fig_height = fig_width / aspect_ratio
    
    # 如果高度过大或过小，进行调整
    if fig_height > 14:
        fig_height = 14
        fig_width = fig_height * aspect_ratio
    elif fig_height < 8:
        fig_height = 8
        fig_width = fig_height * aspect_ratio
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    
    # 设置坐标轴的长宽比为数据的长宽比，确保1:1比例
    ax.set_aspect('auto', adjustable='box')
    
    levels = np.linspace(vmin, vmax, 21)
    cf = ax.contourf(
        freq_masked.lon, freq_masked.lat, freq_masked,
        levels=levels, transform=proj, cmap=cmap, extend='max'
    )

    # 制作物理裁剪 Patch
    # 获取用于裁剪与绘制的多边形（可能来自 geopandas 或回退的 merged）
    if use_geopandas:
        poly = unary_union(gdf.geometry)
    else:
        poly = merged

    if hasattr(poly, 'geoms'):
        subpaths = []
        for p in poly.geoms:
            try:
                subpaths.append(mpath.Path(np.array(p.exterior.coords)))
            except Exception:
                continue
    else:
        subpaths = [mpath.Path(np.array(poly.exterior.coords))]
    compound = mpath.Path.make_compound_path(*subpaths)
    patch = mpatches.PathPatch(compound, transform=ax.transData, facecolor='none')

    # 裁剪主图（防守式：cf 可能没有 collections 属性）
    coll_iter = getattr(cf, 'collections', [])
    for collection in coll_iter:
        collection.set_clip_path(patch)

    # 地理底图
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    if use_geopandas:
        # 使用 geopandas 绘制边界
        gdf.boundary.plot(ax=ax, linewidth=0.8, color='black', zorder=4, transform=proj)
    else:
        from cartopy.feature import ShapelyFeature
        ax.add_feature(ShapelyFeature([poly], proj, edgecolor='black', facecolor='none', linewidth=0.8), zorder=4)

    # --- 添加南海诸岛缩略图 ---
    # 调整小图位置和大小，使用 inset_scale 快速放大/缩小（保持简洁，不使用 try/except）
    inset_scale = 1  # >1 放大, <1 缩小
    base_left, base_bottom, base_w, base_h = 0.65, 0.129, 0.1, 0.185
    sub_ax = fig.add_axes([base_left, base_bottom, base_w * inset_scale, base_h * inset_scale], projection=proj)
    sub_ax.set_extent([107, 122.5, 0, 23.5], crs=proj)
    # 保持原有的 auto aspect
    sub_ax.set_aspect('auto', adjustable='box')
    
    sub_cf = sub_ax.contourf(
        freq_masked.lon, freq_masked.lat, freq_masked,
        levels=levels, transform=proj, cmap=cmap, extend='max'
    )
    
    sub_patch = mpatches.PathPatch(compound, transform=sub_ax.transData, facecolor='none')
    sub_coll_iter = getattr(sub_cf, 'collections', [])
    for collection in sub_coll_iter:
        collection.set_clip_path(sub_patch)
        
    sub_ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    if use_geopandas:
        gdf.boundary.plot(ax=sub_ax, linewidth=0.6, color='black', zorder=4, transform=proj)
    else:
        from cartopy.feature import ShapelyFeature
        sub_ax.add_feature(ShapelyFeature([poly], proj, edgecolor='black', facecolor='none', linewidth=0.6), zorder=4)

    # 装饰 (保持中文标题)
    plt.sca(ax)
    plt.title(f'年均极端{energy_label}事件发生频率\n({scenario}, 12h阈值, 2040-2060)', fontsize=14)
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.03, aspect=30, shrink=0.7)
    cbar.set_label('年均发生频率 (次/年)')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{energy_label}_{scenario}_Frequency_Map.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

if __name__ == "__main__":
    # 单数据文件绘图模式
    input_file = r"G:\extreme_analysis\results\Wind\Wind_ssp126_V1_Reliability_12h.nc"
    output_dir = r"G:\extreme_analysis\results\Wind\frequency_figures"
    geo_boundary = r"G:\extreme_analysis\geoshape_file\CHN_ALI\CHN_ALIYUN.geojson"

    # 用于图名与色带配置
    energy_label = "ws100m"
    scenario = "ssp126"
    cmap = "YlGnBu"

    print(f"正在处理单文件: {input_file}")
    freq = get_annual_frequency(input_file)

    if freq is None:
        print(f"文件不存在，跳过绘图: {input_file}")
    else:
        vmax = float(freq.max().values)
        save_path = plot_frequency(
            freq_data=freq,
            output_dir=output_dir,
            energy_label=energy_label,
            scenario=scenario,
            vmin=0,
            vmax=vmax,
            geo_path=geo_boundary,
            cmap=cmap,
        )
        print(f"完成, 图片保存到 {save_path}")
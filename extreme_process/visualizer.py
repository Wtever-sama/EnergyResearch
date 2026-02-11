''' 绘制各种变量的2040-2060年平均极端事件平均地理图谱, 目前支持solar wind '''

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
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_annual_frequency(file_path: str | Path) -> xr.DataArray | None:
    """
    计算频率数据（保留原逻辑）
    :param file_path: 输入的极端事件NC文件路径
    :return: 年频率数据
    """
    if not os.path.exists(file_path): 
        return None
    ds = xr.open_dataset(file_path)
    da = ds['event_flag']
    mask = da.fillna(0)
    is_start = (mask == 1) & (mask.shift(time=1) != 1)
    # 计算21年均值 (2040-2060)
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
    """
    使用 contourf 实现平滑过渡，并支持 .shp 或 .geojson 格式进行严格边界裁剪。
    :param freq_data: 年频率数据
    :param output_dir: 输出目录
    :param energy_label: 能源类型标签, 如"rsds"
    :param scenario: 情景名称
    :param vmin: 颜色映射最小值
    :param vmax: 颜色映射最大值
    :param geo_path: 地理边界文件路径
    :param cmap: 颜色映射
    :return: 保存的图像路径
    :rtype: str | Path
    """
    # 自动识别并加载矢量边界 (shp 或 geojson)
    gdf = gpd.read_file(geo_path)
    gdf.geometry = gdf.geometry.make_valid()
    # 将数据从 1度 插值到 0.1度
    freq_data = freq_data.interp(
        lon=np.arange(73, 135.1, 0.1), 
        lat=np.arange(15, 55.1, 0.1), 
        method='linear'
    )
    
    # 严格地理边界掩码处理
    mask = regionmask.mask_3D_geopandas(gdf, freq_data.lon, freq_data.lat)
    freq_masked = freq_data.where(mask.any('region'))

    # 初始化地图投影
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([73, 135, 15, 55], crs=ccrs.PlateCarree())
    # ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    
    # 绘制平滑等值线填充
    levels = np.linspace(vmin, vmax, 21)
    cf = ax.contourf(
        freq_masked.lon, freq_masked.lat, freq_masked,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend='max'
    )

    # 从 GeoDataFrame 获取多边形路径, gdf 是读入的中国边界
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        invalid_idx = list(gdf.index[invalid_mask])
        raise ValueError(
            f"输入边界文件包含无效几何对象，索引: {invalid_idx}，请修复源文件后重试（例如在 geopandas 中使用 `gdf.geometry = gdf.geometry.buffer(0)` 或 shapely.make_valid）。"
        )
    
    poly = unary_union(gdf.geometry)
    # 为每个多边形/环创建 matplotlib.path.Path
    subpaths = [mpath.Path(np.array(p.exterior.coords)) for p in (poly.geoms if hasattr(poly, 'geoms') else [poly])]
    compound = mpath.Path.make_compound_path(*subpaths)
    patch = mpatches.PathPatch(compound, transform=ax.transData, facecolor='none')

    # 强制裁剪图层
    collections = getattr(cf, 'collections', None)
    if collections is None and hasattr(cf, 'ax') and hasattr(cf.ax, 'collections'):
        collections = cf.ax.collections
    if collections:
        for collection in collections:
            collection.set_clip_path(patch)

    # 叠加地理底图与掩码要素
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    
    # 绘制行政边界线
    gdf.boundary.plot(ax=ax, linewidth=0.8, color='black', zorder=4, transform=ccrs.PlateCarree())

    # 设置标题与图例 (保留原始格式)
    # 【使用中文绘图】
    plt.title(f'极端 {energy_label} 事件的年频率\n({scenario}, 12小时阈值, 2040-2060)')
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=25)
    cbar.set_label('年频率 (事件/年)')

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{energy_label}_{scenario}_Frequency_Map.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


if __name__ == "__main__":
    # 路径配置
    results_root = r"G:\extreme_analysis\results"
    figure_path_base = "frequency_figures"
    # 支持 .shp 或 .geojson
    geo_boundary = r"G:\extreme_analysis\geoshape_file\CHN_ALI\CHN_ALIYUN.geojson"
    
    scenarios = ['ssp126', 'ssp245', 'ssp585']
    energy_types = ['solar', 'wind']
    energy_types_mapping = {
        "solar": "rsds",
        "wind": "ws100m"
    }

    # colormap mapping by energy type (可根据需要调整)
    cmap_mapping = {
        'solar': 'YlOrRd',  # 暖色系
        'wind': 'Blues'     # 冷色系
    }

    for energy_type in energy_types:
        print(f"正在处理 {energy_type} 跨情景频率并集...")
        scenario_data = {}
        all_max = []

        # 阶段1：收集所有情景数据并确定全局 Vmax
        for scenario in scenarios:
            if energy_type == 'solar':
                fpath = os.path.join(results_root, "solar", f"solar_{scenario}_V1_Reliability_12h.nc")
            else:
                fpath = os.path.join(results_root, "wind", f"wind_{scenario}_V1_Reliability_12h.nc")
            
            freq = get_annual_frequency(fpath)
            if freq is not None:
                scenario_data[scenario] = freq
                all_max.append(float(freq.max().values))
        
        if not all_max: 
            continue
        
        global_vmax = max(all_max)
        global_vmin = 0
        
        # 阶段2：统一值域绘图
        # 保存路径: 【结果根路径/变量类型/图片文件夹名称】
        plot_out_dir = os.path.join(results_root, energy_type, figure_path_base)
        for scenario, da in scenario_data.items():
            save_path = plot_frequency(
                freq_data=da,
                output_dir=plot_out_dir,
                energy_label=energy_types_mapping[energy_type],
                scenario=scenario,
                vmin=global_vmin,
                vmax=global_vmax,
                geo_path=geo_boundary,
                cmap=cmap_mapping.get(energy_type, 'viridis')
            )
            print(f"  {energy_type.upper()} {scenario} 绘图完成, 图片保存到 {save_path}")
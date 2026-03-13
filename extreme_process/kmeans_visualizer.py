import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import geopandas as gpd
import json
from matplotlib.colors import ListedColormap
from shapely.geometry import shape
from shapely.ops import unary_union
import os


def add_north(ax, labelsize=14, loc_x=0.92, loc_y=0.88, width=0.03, height=0.08, pad=0.14):
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


def _load_country_boundary(geojson_file):
    use_geopandas = True
    merged = None
    gdf = None

    try:
        gdf = gpd.read_file(geojson_file)
        try:
            gdf.geometry = gdf.geometry.make_valid()
        except Exception:
            pass
    except Exception:
        use_geopandas = False
        with open(geojson_file, 'r', encoding='utf-8') as f:
            js_data = json.load(f)
        features = js_data['features'] if 'features' in js_data else js_data
        shapes = [shape(feat['geometry']) for feat in features]
        merged = unary_union(shapes)

    country_geom = unary_union(gdf.geometry) if use_geopandas else merged
    return use_geopandas, gdf, country_geom


def _build_clip_path(geom):
    if hasattr(geom, 'geoms'):
        subpaths = []
        for poly in geom.geoms:
            try:
                subpaths.append(mpath.Path(np.array(poly.exterior.coords)))
            except Exception:
                continue
    else:
        subpaths = [mpath.Path(np.array(geom.exterior.coords))]

    if not subpaths:
        raise ValueError("无法从国家边界几何构建裁剪路径")
    return mpath.Path.make_compound_path(*subpaths)


def _get_distinct_cmap(n_clusters: int) -> ListedColormap:
    """返回更美观且区分度更高的离散调色板"""
    base_colors = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
        "#EDC948", "#D786BE", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#AE76E3",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ]

    if n_clusters <= len(base_colors):
        colors = base_colors[:n_clusters]
    else:
        extra = plt.get_cmap('turbo')(np.linspace(0.08, 0.92, n_clusters - len(base_colors)))
        extra_hex = [
            '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b, _ in extra
        ]
        colors = base_colors + extra_hex

    return ListedColormap(colors, name='distinct_cluster_palette')


def _plot_province_boundaries(ax, shp_file):
    """优先使用 geopandas 绘制省界，失败时回退 cartopy Reader。"""
    try:
        province_gdf = gpd.read_file(shp_file)
        try:
            province_gdf.geometry = province_gdf.geometry.make_valid()
        except Exception:
            pass
        province_gdf.boundary.plot(
            ax=ax,
            linewidth=0.5,
            color='dimgray',
            zorder=5,
            transform=ccrs.PlateCarree(),
        )
    except Exception:
        province_geoms = list(Reader(shp_file).geometries())
        if province_geoms:
            ax.add_feature(
                ShapelyFeature(
                    province_geoms,
                    ccrs.PlateCarree(),
                    edgecolor='dimgray',
                    facecolor='none',
                    linewidth=0.5,
                ),
                zorder=5,
            )

def visualize_clusters(file_path, shp_file=None, geojson_file=None):
    '''
    仅使用传入的 geojson_file 绘制国家边界，shp_file 绘制省份边界
    '''
    if geojson_file is None:
        raise ValueError("请提供 geojson_file 用于国家边界裁剪")

    # 1. 加载数据
    ds = xr.open_dataset(file_path)
    clusters = ds.cluster_zone

    # 1.1 读取国家边界并准备裁剪路径
    use_geopandas, country_gdf, country_geom = _load_country_boundary(geojson_file)
    
    # 2. 准备底图和投影（与 visualizer.py 对齐）
    projn = ccrs.LambertConformal(
        central_longitude=105,
        central_latitude=40,
        standard_parallels=(25.0, 47.0)
    )
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection=projn)
    ax.set_global()
    ax.set_extent([75, 135, 15, 55], crs=ccrs.PlateCarree())
    
    # 3. 设置离散颜色映射
    n_clusters = int(clusters.max().values) + 1
    cmap = _get_distinct_cmap(n_clusters)
    
    # 4. 绘图
    mesh = clusters.plot.pcolormesh(
        ax=ax, 
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False,
        zorder=2  # <--- 提高层级，使其覆盖在 gridlines 之上
    )

    # 4.1 国家边界裁剪（类比 visualizer_0.py）
    compound = _build_clip_path(country_geom)
    patch = mpatches.PathPatch(
        compound,
        transform=ccrs.PlateCarree()._as_mpl_transform(ax),
        facecolor='none'
    )
    mesh.set_clip_path(patch)
    
    # 5. 绘制国家边界
    if use_geopandas:
        country_gdf.boundary.plot(ax=ax, linewidth=0.6, color='black', zorder=20, transform=ccrs.PlateCarree())
    else:
        ax.add_feature(
            ShapelyFeature([country_geom], ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.6),
            zorder=20,
        )

    # shp_file 仅绘制省级边界
    if shp_file is not None:
        _plot_province_boundaries(ax, shp_file)
    
    # 添加经纬度网格线（与 visualizer.py 风格对齐）
    gl = ax.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        color='gray',
        linestyle='dashed',
        linewidth=0.5, # 稍微加粗一点点，因为在底层可能不明显
        alpha=0.6,
        zorder=0,      # <--- 降低层级，确保在最底层
        y_inline=False,
        x_inline=False,
        rotate_labels=0,
        xpadding=5,
        xlocs=np.arange(70, 150, 10),
        ylocs=np.arange(10, 70, 10),
        xlabel_style={"size": 12, "weight": "bold"},
        ylabel_style={"size": 12, "weight": "bold"}
    )

    gl.xlines = True 
    gl.ylines = True
    gl.top_labels = False
    gl.right_labels = False

    # 6. 添加图例/Colorbar
    cbar = plt.colorbar(mesh, ax=ax, location='right', pad=0.03, shrink=0.5, aspect=15)
    cbar.set_label('Zone Index')
    cbar.set_ticks(np.arange(n_clusters))

    add_north(ax)
    plt.tight_layout()

    # 7. 南海小地图（尺寸与位置与 visualizer.py 完全一致）
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

    if use_geopandas:
        country_gdf.boundary.plot(ax=sub_ax, linewidth=0.8, color='black', zorder=20, transform=ccrs.PlateCarree())
    else:
        sub_ax.add_feature(
            ShapelyFeature([country_geom], ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.8),
            zorder=20,
        )

    if shp_file is not None:
        try:
            province_gdf = gpd.read_file(shp_file)
            try:
                province_gdf.geometry = province_gdf.geometry.make_valid()
            except Exception:
                pass
            province_gdf.boundary.plot(
                ax=sub_ax,
                linewidth=0.2,
                color='dimgray',
                zorder=21,
                transform=ccrs.PlateCarree(),
            )
        except Exception:
            province_geoms = list(Reader(shp_file).geometries())
            if province_geoms:
                sub_ax.add_feature(
                    ShapelyFeature(
                        province_geoms,
                        ccrs.PlateCarree(),
                        edgecolor='dimgray',
                        facecolor='none',
                        linewidth=0.2,
                    ),
                    zorder=21,
                )

    mesh_sub = clusters.plot.pcolormesh(
        ax=sub_ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=False,
        zorder=1
    )
    sub_patch = mpatches.PathPatch(
        compound,
        transform=ccrs.PlateCarree()._as_mpl_transform(sub_ax),
        facecolor='none'
    )
    mesh_sub.set_clip_path(sub_patch)
    mesh_sub.set_clip_path(sub_ax.patch)
    
    pic_name = f'Spatial_Clustering_of_Wind-Solar_Consistent_Zones_(K={n_clusters}).png'
    pic_dir = os.path.dirname(file_path)
    
    # 8. 保存并展示
    output_img = os.path.join(pic_dir, pic_name)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"可视化结果已保存至: {output_img}")


if __name__ == "__main__":
    cluster_nc = "G:/extreme_analysis/results/Clustering/kmeans_Spatial_Cluster_Zones.nc"
    shp_file = "G:/extreme_analysis/geoshape_file/gadm41_CHN_shp/gadm41_CHN_1.shp"
    geojson_file = "G:/extreme_analysis/geoshape_file/CHN_ALI/CHN_ALIYUN.geojson"
    visualize_clusters(cluster_nc, shp_file=shp_file, geojson_file=geojson_file)
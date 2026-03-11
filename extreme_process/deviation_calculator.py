import xarray as xr
import os
import glob
import logging
import warnings
import gc
import pandas as pd
import numpy as np
import re
import tempfile
from netCDF4 import Dataset, date2num

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# --- 核心处理类 (精简版结构) ---
class DeviationProcessor:
    def __init__(self, file_paths, variant, is_era5=False, time_range=None):
        '''
        :param file_paths: 文件路径列表
        :type file_paths: list[str]
        :param variant: 变体名称, 'Solar' 'Wind'
        :type variant: str
        :param is_era5: 是否为 ERA5 数据
        :type is_era5: bool
        :param time_range: 时间范围 (可选)
        :type time_range: tuple[str, str] | None
        '''
        self.variant = variant
        self.var_name = f"{variant}CF"
        logger.info(f"初始化 DeviationProcessor: variant={variant}, is_era5={is_era5}, files={len(file_paths)}")
        try:
            # 显式指定 netCDF 引擎，避免 xarray 无法自动识别的错误
            logger.debug(f"打开文件集合（open_mfdataset）: {file_paths}")
            self.ds = xr.open_mfdataset(file_paths, combine='by_coords', parallel=True, engine='netcdf4').chunk({'time': 2000})
            logger.info(f"已加载数据集: dims={self.ds.dims}, sizes={{k: v for k,v in self.ds.sizes.items()}}")
        except ValueError:
            # 引擎探测失败，给出更明确的提示
            raise ValueError("无法使用已安装的 xarray IO 后端打开文件；请确保已安装 netCDF4 或指定正确的 engine。")
        
        # 统一维度名
        dim_map = {'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'}
        rename_map = {k: v for k, v in dim_map.items() if k in self.ds.dims or k in self.ds.coords}
        if rename_map:
            logger.debug(f"重命名维度/坐标: {rename_map}")
            self.ds = self.ds.rename(rename_map)
        
        if time_range:
            logger.info(f"按时间范围切片: {time_range}")
            self.ds = self.ds.sel(time=slice(time_range[0], time_range[1]))
            
        try:
            self.cf = self.ds[self.var_name]
            logger.info(f"数据变量 '{self.var_name}' 已加载, shape={self.cf.shape}, dims={self.cf.dims}")
        except Exception:
            logger.error(f"未找到变量 {self.var_name} 于数据集中; 可用变量: {list(self.ds.data_vars)}")
            raise

    def get_expectations(self):
        logger.info(f"正在计算 {self.variant} 历史期望...")
        # 标准化坐标名，确保 season/hour 处理正确
        exp = self.cf.groupby(self.ds.time.dt.season).apply(lambda x: x.groupby(x.time.dt.hour).mean())
        exp = exp.rename({'season': 'season', 'hour': 'hour'})
        # 某些 groupby/apply 流程会丢失真实经纬度坐标，导致后续与目标网格对齐异常
        if {'lat', 'lon'}.issubset(exp.dims) and {'lat', 'lon'}.issubset(self.cf.coords):
            exp = exp.assign_coords(lat=self.cf['lat'], lon=self.cf['lon'])
        try:
            logger.info(f"期望矩阵计算完成: dims={exp.dims}, sizes={exp.sizes}")
        except Exception:
            logger.debug("期望矩阵已计算完成 (无法读取 sizes)")
        return exp

    def get_deviations(self, exp_matrix):
        logger.info("正在计算偏差...")
        # 兼容期望矩阵维度命名：latitude/longitude -> lat/lon, valid_time -> time
        exp_dim_map = {'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'}
        exp_rename_map = {k: v for k, v in exp_dim_map.items() if k in exp_matrix.dims or k in exp_matrix.coords}
        if exp_rename_map:
            exp_matrix = exp_matrix.rename(exp_rename_map)

        # 避免使用 exp_matrix.sel(season=<time-array>, hour=<time-array>) 的高级索引
        # 该写法会触发大规模 int64 广播索引（可达数 GiB），在长时间序列下容易 OOM。
        delta_parts = []
        seasons = [s for s in exp_matrix['season'].values]
        hours = [int(h) for h in exp_matrix['hour'].values]

        for season in seasons:
            season_mask = self.ds.time.dt.season == season
            season_cf = self.cf.where(season_mask, drop=True)
            if season_cf.sizes.get('time', 0) == 0:
                continue

            for hour in hours:
                hour_mask = season_cf.time.dt.hour == hour
                sub_cf = season_cf.where(hour_mask, drop=True)
                if sub_cf.sizes.get('time', 0) == 0:
                    continue

                exp_slice = exp_matrix.sel(season=season, hour=hour).squeeze(drop=True)

                # 统一到 (lat, lon) 并避免 xarray 自动对齐触发额外广播维度
                if {'lat', 'lon'}.issubset(exp_slice.dims):
                    exp_slice = exp_slice.transpose('lat', 'lon')
                else:
                    raise ValueError(f"期望切片缺少 lat/lon 维度: dims={exp_slice.dims}")

                # 显式按数组广播： (time, lat, lon) - (lat, lon) -> (time, lat, lon)
                delta_data = sub_cf.data - exp_slice.data[None, :, :]
                delta_sub = xr.DataArray(
                    delta_data,
                    coords=sub_cf.coords,
                    dims=sub_cf.dims,
                    name='delta_x',
                )
                delta_parts.append(delta_sub)

        if not delta_parts:
            raise ValueError("未生成任何 delta 分块，请检查 time/season/hour 是否匹配。")

        delta = xr.concat(delta_parts, dim='time').rename("delta_x")
        logger.info(f"偏差计算完成: delta shape (lazy)={getattr(delta, 'shape', 'unknown')}")
        return delta


def get_expectations(era5_paths, variant, output_path) -> None:
    '''
    计算 (历史) 期望值并保存到指定路径

    :param era5_paths: ERA5 数据路径列表
    :type era5_paths: list[str]
    :param variant: 变体名称, 'Solar' 'Wind'
    :type variant: str
    :param output_path: 输出文件路径
    :type output_path: str
    '''
    logger.info(f"开始计算并保存期望: variant={variant}, output={output_path}")
    proc = DeviationProcessor(era5_paths, variant, is_era5=True)
    exp = proc.get_expectations()
    exp.to_netcdf(output_path)
    logger.info(f"期望已保存: {output_path}")


def get_delta(ssp_paths, variant, exp_path, output_path) -> None:
    '''
    计算期望偏差并保存到指定路径

    :param ssp_paths: SSP 数据路径列表
    :type ssp_paths: list[str]
    :param variant: 变体名称, 'Solar' 'Wind'
    :type variant: str
    :param exp_path: 期望矩阵路径
    :type exp_path: str
    :param output_path: 输出文件路径
    :type output_path: str
    '''
    logger.info(f"开始计算 delta（5年分块）: variant={variant}, output={output_path}")
    proc = DeviationProcessor(ssp_paths, variant, is_era5=False)
    try:
        logger.debug(f"打开期望文件: {exp_path}")
        exp = xr.open_dataarray(exp_path, engine='netcdf4', chunks={})
    except ValueError as e:
        logger.error(f"打开期望矩阵文件失败: {exp_path}: {e}")
        raise

    # 统一期望矩阵维度命名，兼容 (latitude, longitude) / (valid_time, latitude, longitude)
    exp_dim_map = {'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'}
    exp_rename_map = {k: v for k, v in exp_dim_map.items() if k in exp.dims or k in exp.coords}
    if exp_rename_map:
        logger.info(f"期望矩阵维度/坐标重命名: {exp_rename_map}")
        exp = exp.rename(exp_rename_map)

    # 对齐期望矩阵网格，避免坐标不一致导致空间维度被意外裁剪
    if {'lat', 'lon'}.issubset(exp.dims) and {'lat', 'lon'}.issubset(proc.cf.dims):
        target_lat = proc.cf['lat']
        target_lon = proc.cf['lon']
        same_shape = (exp.sizes['lat'] == target_lat.size) and (exp.sizes['lon'] == target_lon.size)

        if same_shape:
            if (not np.array_equal(exp['lat'].values, target_lat.values)) or (not np.array_equal(exp['lon'].values, target_lon.values)):
                logger.info("期望矩阵与目标网格尺寸一致但坐标值不同，按目标网格重建坐标")
                exp = exp.assign_coords(lat=target_lat, lon=target_lon)
        else:
            logger.info("期望矩阵与目标网格尺寸不同，插值到目标网格")
            exp = exp.interp(lat=target_lat, lon=target_lon, method='linear')

    # 自动识别时间范围，并按 5 年分块处理，降低峰值内存占用
    time_years = proc.ds['time'].dt.year
    start_year = int(time_years.min().compute())
    end_year = int(time_years.max().compute())
    logger.info(f"检测到 SSP 时间范围: {start_year}-{end_year}")

    # 以 netCDF4 逐块写出，避免一次性拼接全时段结果
    nt_lat = int(proc.cf.sizes['lat'])
    nt_lon = int(proc.cf.sizes['lon'])
    time_units = 'seconds since 1970-01-01 00:00:00'
    total_written = 0

    logger.info(f"创建输出文件并准备分块写出: lat={nt_lat}, lon={nt_lon}")
    with Dataset(output_path, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', None)
        nc.createDimension('lat', nt_lat)
        nc.createDimension('lon', nt_lon)

        time_var = nc.createVariable('time', 'f8', ('time',))
        time_var.units = time_units
        lat_var = nc.createVariable('lat', 'f4', ('lat',))
        lon_var = nc.createVariable('lon', 'f4', ('lon',))
        delta_var = nc.createVariable('delta_x', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4)

        lat_var[:] = proc.cf['lat'].values
        lon_var[:] = proc.cf['lon'].values

        # 不再使用全时段 proc，后续按块重新打开并处理
        del proc

        for chunk_start in range(start_year, end_year + 1, 5):
            chunk_end = min(chunk_start + 4, end_year)
            chunk_time_range = (f"{chunk_start}-01-01", f"{chunk_end}-12-31")
            logger.info(f"开始处理时间块: {chunk_start}-{chunk_end}")

            chunk_proc = DeviationProcessor(
                ssp_paths,
                variant,
                is_era5=False,
                time_range=chunk_time_range,
            )

            chunk_nt = int(chunk_proc.cf.sizes.get('time', 0))
            if chunk_nt == 0:
                logger.info(f"时间块 {chunk_start}-{chunk_end} 无数据，跳过")
                continue

            logger.info(f"时间块 {chunk_start}-{chunk_end} 数据量: time={chunk_nt}")
            chunk_delta = chunk_proc.get_deviations(exp)

            # 仅将当前 5 年块加载到内存并写入
            chunk_delta_np = chunk_delta.astype(np.float32).load().values
            chunk_time_vals = pd.to_datetime(chunk_delta['time'].values)

            # 避免 dask sortby 的 shuffle 开销，改为本地 numpy 排序后写出
            if chunk_time_vals.size > 1:
                order = np.argsort(chunk_time_vals.values)
                if not np.all(order == np.arange(order.size)):
                    chunk_delta_np = chunk_delta_np[order, :, :]
                    chunk_time_vals = chunk_time_vals[order]

            chunk_times = chunk_time_vals.to_pydatetime()
            write_end = total_written + chunk_delta_np.shape[0]

            time_var[total_written:write_end] = date2num(chunk_times, time_units)
            delta_var[total_written:write_end, :, :] = chunk_delta_np

            total_written = write_end
            logger.info(
                f"时间块 {chunk_start}-{chunk_end} 写入完成: 当前累计 time={total_written}"
            )

    if total_written == 0:
        raise ValueError("未写入任何 delta 数据，请检查输入文件和时间范围。")

    logger.info(f"delta 已按 5 年分块写入完成: {output_path}, total_time={total_written}")


def get_p10(delta_path, p10_output_path) -> None:
    '''
    使用 numpy partition 加速计算 delta_X 的 P10 阈值

    :param delta_path: 输入 delta 文件路径
    :type delta_path: str
    :param p10_output_path: 输出 P10 文件路径
    :type p10_output_path: str
    '''
    logger.info(f"开始加速计算 P10（5年分块）: delta={delta_path}")

    # 1. 以分块方式打开数据
    delta_ds = xr.open_dataarray(delta_path, chunks={'time': 2000, 'lat': 5, 'lon': 71})
    ntime_total = int(delta_ds.sizes['time'])
    nlat, nlon = int(delta_ds.sizes['lat']), int(delta_ds.sizes['lon'])

    years = delta_ds['time'].dt.year
    start_year = int(years.min().compute())
    end_year = int(years.max().compute())
    logger.info(f"delta 时间范围: {start_year}-{end_year}, 总 time={ntime_total}, lat={nlat}, lon={nlon}")

    # 2. 准备存储结果的空阵
    p10_result = np.full((nlat, nlon), np.nan, dtype=np.float32)

    # 3. 按纬度块 + 5 年时间块处理，避免一次性读取全时段
    lat_step = 5
    for i in range(0, nlat, lat_step):
        lat_slice = slice(i, min(i + lat_step, nlat))
        lat_size = int(lat_slice.stop - lat_slice.start)
        logger.info(f"开始处理 lat 块: {i}:{lat_slice.stop} (lat_size={lat_size})")

        # 使用磁盘 memmap 暂存当前 lat 块的全部时间序列，避免占用过多内存
        with tempfile.NamedTemporaryFile(prefix='p10_block_', suffix='.dat', delete=False) as tmpf:
            tmp_path = tmpf.name

        block_mm = None
        try:
            block_mm = np.memmap(tmp_path, dtype=np.float32, mode='w+', shape=(ntime_total, lat_size, nlon))
            write_pos = 0

            for chunk_start in range(start_year, end_year + 1, 5):
                chunk_end = min(chunk_start + 4, end_year)
                time_slice = slice(f"{chunk_start}-01-01", f"{chunk_end}-12-31")

                block_da = delta_ds.sel(time=time_slice).isel(lat=lat_slice)
                chunk_nt = int(block_da.sizes.get('time', 0))
                if chunk_nt == 0:
                    logger.info(f"  跳过空时间块: {chunk_start}-{chunk_end}")
                    continue

                logger.info(
                    f"  读取时间块 {chunk_start}-{chunk_end}: time={chunk_nt}, 写入位置={write_pos}:{write_pos + chunk_nt}"
                )
                block_np = block_da.load().values.astype(np.float32, copy=False)
                block_mm[write_pos:write_pos + chunk_nt, :, :] = block_np
                write_pos += chunk_nt

            if write_pos == 0:
                logger.warning(f"lat 块 {i}:{lat_slice.stop} 未读取到任何有效时间数据，结果保持 NaN")
                continue

            k = int(write_pos * 0.1)
            logger.info(f"lat 块 {i}:{lat_slice.stop} 开始计算 P10: valid_time={write_pos}, p=10%, k={k}")

            # 某些 numpy 版本无 nanpartition，这里使用 nanpercentile 保持兼容
            try:
                p10_block = np.nanpercentile(block_mm[:write_pos, :, :], 10, axis=0, method='lower')
            except TypeError:
                # 兼容旧版本 numpy（method 参数名称为 interpolation）
                p10_block = np.nanpercentile(block_mm[:write_pos, :, :], 10, axis=0, interpolation='lower')

            p10_result[lat_slice, :] = p10_block.astype(np.float32, copy=False)
            logger.info(f"lat 块 {i}:{lat_slice.stop} P10 计算完成")
        finally:
            # Windows 下需显式释放 memmap 句柄，否则删除临时文件会报 WinError 32
            if block_mm is not None:
                try:
                    block_mm.flush()
                    if hasattr(block_mm, '_mmap') and block_mm._mmap is not None:
                        block_mm._mmap.close()
                except Exception:
                    logger.warning(f"lat 块 {i}:{lat_slice.stop} memmap 关闭时出现异常")
                del block_mm
                gc.collect()

            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except PermissionError:
                    logger.warning(f"临时文件删除失败（可能仍被占用）: {tmp_path}")

    # 5. 将结果包装回 DataArray 并保存
    p10_da = xr.DataArray(
        p10_result, 
        coords=[delta_ds.lat, delta_ds.lon], 
        dims=['lat', 'lon'], 
        name='p10'
    )
    p10_da.to_netcdf(p10_output_path)
    logger.info(f"P10 5年分块计算完成并保存: {p10_output_path}")


def get_v1(delta_path, p10_path, output_path) -> None:
    '''
    计算 V1 标志 (是否属于极端事件, 0-1) 并保存到指定路径

    :param delta_path: 输入 delta 文件路径
    :type delta_path: str
    :param p10_path: 输入 P10 文件路径
    :type p10_path: str
    :param output_path: 输出 V1 文件路径
    :type output_path: str
    '''
    logger.info(f"开始计算 V1 标志（5年分块写出）: delta={delta_path}, p10={p10_path}, output={output_path}")
    # 打开 delta 为 dask-backed DataArray（按 time 分块）
    try:
        delta = xr.open_dataarray(delta_path, engine='netcdf4', chunks={'time': 2000})
    except ValueError as e:
        logger.error(f"无法打开 delta 文件: {delta_path}: {e}")
        raise

    # p10 应该是小数组 (lat, lon)，尽量加载到内存
    try:
        p10_da = xr.open_dataarray(p10_path, engine='netcdf4')
        p10 = p10_da.compute()
    except ValueError as e:
        logger.error(f"无法打开 p10 文件: {p10_path}: {e}")
        raise

    # 若 p10 无法直接广播到 delta 网格，则插值到 delta 的经纬度
    if set(p10.dims) >= {'lat', 'lon'} and set(delta.dims) >= {'time', 'lat', 'lon'}:
        if not np.array_equal(p10['lat'].values, delta['lat'].values) or not np.array_equal(p10['lon'].values, delta['lon'].values):
            logger.info("p10 网格与 delta 网格不一致，插值 p10 到 delta 网格")
            p10 = p10.interp(lat=delta.lat, lon=delta.lon, method='linear')

    # 确定尺寸
    nt = int(delta.sizes['time'])
    nlat = int(delta.sizes['lat'])
    nlon = int(delta.sizes['lon'])

    # 自动识别时间范围，并按 5 年分块处理
    years = delta['time'].dt.year
    start_year = int(years.min().compute())
    end_year = int(years.max().compute())
    logger.info(f"V1 处理时间范围: {start_year}-{end_year}, 总 time={nt}, nlat={nlat}, nlon={nlon}")

    # 选择分块策略
    time_chunk = 2000
    lat_chunk = 5

    logger.info(
        f"准备写出 NetCDF: nt={nt}, nlat={nlat}, nlon={nlon}, time_chunk={time_chunk}, lat_chunk={lat_chunk}, year_step=5"
    )

    # 创建输出文件，用 netCDF4 逐块写入
    with Dataset(output_path, 'w', format='NETCDF4') as nc:
        # 创建维度
        nc.createDimension('time', None)
        nc.createDimension('lat', nlat)
        nc.createDimension('lon', nlon)

        # 创建坐标变量并写入
        time_var = nc.createVariable('time', 'f8', ('time',))
        lat_var = nc.createVariable('lat', 'f4', ('lat',))
        lon_var = nc.createVariable('lon', 'f4', ('lon',))

        # 写入 coordinate data
        units = 'seconds since 1970-01-01 00:00:00'
        time_var.units = units
        lat_var[:] = delta['lat'].values
        lon_var[:] = delta['lon'].values

        # 创建数据变量，int8，填充值 -1
        var = nc.createVariable('is_extreme', 'i1', ('time', 'lat', 'lon'), fill_value=-1, zlib=True, complevel=4)
        var.long_name = 'V1 flag: 1 extreme, 0 non-extreme, -1 missing'

        total_written = 0

        # 按 5 年时间块分步处理并写出
        for year_start in range(start_year, end_year + 1, 5):
            year_end = min(year_start + 4, end_year)
            year_time_slice = slice(f"{year_start}-01-01", f"{year_end}-12-31")
            delta_year = delta.sel(time=year_time_slice)
            nt_year = int(delta_year.sizes.get('time', 0))

            if nt_year == 0:
                logger.info(f"时间块 {year_start}-{year_end} 无数据，跳过")
                continue

            logger.info(f"开始处理时间块 {year_start}-{year_end}: time={nt_year}")

            # 写入当前 5 年块的 time 坐标
            chunk_times = pd.to_datetime(delta_year['time'].values).to_pydatetime()
            write_year_end = total_written + nt_year
            time_var[total_written:write_year_end] = date2num(chunk_times, units)

            # 在该 5 年块内继续按 lat 和 time 小块处理
            for i_lat in range(0, nlat, lat_chunk):
                lat_slice = slice(i_lat, min(i_lat + lat_chunk, nlat))
                p10_block = p10.isel(lat=lat_slice).values
                logger.info(
                    f"  时间块 {year_start}-{year_end}, lat 块 {i_lat}:{min(i_lat + lat_chunk, nlat)} (size {p10_block.shape})"
                )

                for i_time in range(0, nt_year, time_chunk):
                    local_time_slice = slice(i_time, min(i_time + time_chunk, nt_year))
                    logger.debug(
                        f"    处理子 time 块 {i_time}:{min(i_time + time_chunk, nt_year)}"
                    )

                    # 读取 delta 的这一小块到内存
                    block = delta_year.isel(time=local_time_slice, lat=lat_slice).load()
                    block_np = block.values  # shape (t_chunk, lat_block, lon)

                    # 生成 mask，默认填充值 -1
                    mask = np.full(block_np.shape, -1, dtype=np.int8)
                    valid = ~np.isnan(block_np)
                    if np.any(valid):
                        # 比较：对于有效值，1 if delta < p10 else 0
                        comp = (block_np < p10_block[None, :, :])
                        mask[valid] = comp[valid].astype(np.int8)

                    # 将当前 5 年块的局部 time 索引映射到全局写入索引
                    global_time_start = total_written + i_time
                    global_time_end = global_time_start + mask.shape[0]
                    var[global_time_start:global_time_end, i_lat:i_lat + mask.shape[1], :] = mask

            total_written = write_year_end
            logger.info(
                f"时间块 {year_start}-{year_end} 写入完成: 当前累计写入 time={total_written}"
            )

    if total_written == 0:
        raise ValueError("V1 未写入任何数据，请检查 delta/p10 输入。")

    logger.info(f"V1 已按 5 年分块写入并保存: {output_path}, total_time={total_written}")


if __name__ == "__main__":
    era5_paths = ["G:/extreme_analysis/data/CMIP6_Research_Data/SolarCF/era5_SolarCF_1deg_china_2000-2009.nc",
                 "G:/extreme_analysis/data/CMIP6_Research_Data/SolarCF/era5_SolarCF_1deg_china_2010-2019.nc",
                 "G:/extreme_analysis/data/CMIP6_Research_Data/SolarCF/era5_SolarCF_1deg_china_2020-2025.nc"
    ]
    variant = "Solar"
    exp_output_path = "G:/extreme_analysis/results/Solar/Solar_ERA5_historical_exp_4x24.nc"

    ssp_paths = [
        "G:/extreme_analysis/data/CMIP6_Research_Data/SolarCF/CMIP6_QDM_MME_SolarCF_ssp585_1deg_utc+8_1h_interpolated_2040-2060.nc"
    ]
    delta_out_path = "G:/extreme_analysis/results/Solar/Solar_ssp585_V2_delta_x.nc"
    # get_delta(ssp_paths, variant, exp_output_path, delta_out_path)

    p10_output_path = "G:/extreme_analysis/results/Solar/Solar_ssp585_p10_thresholds.nc"
    # get_p10(delta_path=delta_out_path, p10_output_path=p10_output_path)

    v1_output_path = "G:/extreme_analysis/results/Solar/Solar_ssp585_V1_flag.nc"
    get_v1(delta_path=delta_out_path, p10_path=p10_output_path, output_path=v1_output_path)
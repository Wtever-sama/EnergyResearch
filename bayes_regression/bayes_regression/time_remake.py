#第二步进行时间统一计算月尺度数据
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import cftime
from tqdm import tqdm
import warnings
import traceback

warnings.filterwarnings("ignore")


def inspect_cmip6_time_resolution(cmip6_folder, sample_models=None):
    """
    检查CMIP6数据的时间分辨率
    """
    print("检查CMIP6数据的时间分辨率...")

    # 获取CMIP6文件
    cmip6_files = glob.glob(os.path.join(cmip6_folder, "*.nc"))

    if not cmip6_files:
        print("未找到CMIP6文件!")
        return {}

    time_info = {}

    for file_path in cmip6_files:
        try:
            # 提取模型名
            filename = os.path.basename(file_path)
            if filename.startswith('cmip6_'):
                model = filename.split('_')[1]
            else:
                continue

            # 如果指定了模型列表，只检查指定的模型
            if sample_models is not None and model not in sample_models:
                continue

            # 打开文件检查时间信息
            ds = xr.open_dataset(file_path)

            # 确定时间坐标名称
            time_coord = None
            for coord in ['time', 'valid_time']:
                if coord in ds.coords:
                    time_coord = coord
                    break

            if time_coord is None:
                print(f"警告: {model} 找不到时间坐标")
                ds.close()
                continue

            times = ds[time_coord]

            # 计算时间分辨率
            if len(times) > 1:
                # 对于cftime对象，我们需要特殊处理
                if isinstance(times[0].values, cftime.DatetimeNoLeap):
                    # 转换为pandas时间戳以便计算差异
                    time_diffs = []
                    for i in range(1, len(times)):
                        t1 = pd.Timestamp(times[i - 1].values)
                        t2 = pd.Timestamp(times[i].values)
                        time_diffs.append((t2 - t1).total_seconds() / 3600)  # 转换为小时
                    avg_resolution = np.mean(time_diffs)
                else:
                    time_diffs = np.diff(times.values)
                    avg_resolution = np.mean([td / np.timedelta64(1, 'h') for td in time_diffs])

                time_info[model] = {
                    'file': file_path,
                    'time_coord': time_coord,
                    'n_times': len(times),
                    'time_range': [times[0].values, times[-1].values],
                    'avg_resolution_hours': avg_resolution,
                    'time_values': times.values,
                    'time_type': type(times[0].values)
                }

                print(
                    f"{model}: {avg_resolution:.1f} 小时, {len(times)} 个时间点, 时间类型: {time_info[model]['time_type']}")

            ds.close()

        except Exception as e:
            print(f"检查 {file_path} 时出错: {e}")
            continue

    return time_info


def resample_era5_time(era5_file, target_resolution_hours=3, output_suffix="_3h"):
    """
    对已经网格变尺度后的ERA5数据进行时间降尺度
    """
    print(f"对ERA5数据进行时间降尺度: {target_resolution_hours}小时分辨率")

    try:
        # 读取ERA5数据
        era5_ds = xr.open_dataset(era5_file)
        print(f"原始数据时间维度: {era5_ds.sizes}")

        # 确定时间坐标名称
        time_coord = None
        for coord in ['valid_time', 'time']:
            if coord in era5_ds.coords:
                time_coord = coord
                break

        if time_coord is None:
            raise ValueError("找不到时间坐标")

        print(f"使用时间坐标: {time_coord}")
        print(f"原始时间点数量: {len(era5_ds[time_coord])}")

        # 计算原始时间分辨率
        if len(era5_ds[time_coord]) > 1:
            time_diffs = np.diff(era5_ds[time_coord].values)
            avg_resolution = np.mean([td / np.timedelta64(1, 'h') for td in time_diffs])
            print(f"原始时间分辨率: {avg_resolution:.2f} 小时")

        # 进行时间重采样
        era5_resampled = era5_ds.resample({time_coord: f'{target_resolution_hours}H'}).nearest()

        print(f"重采样后时间点数量: {len(era5_resampled[time_coord])}")

        # 生成输出文件路径
        output_file = era5_file.replace('.nc', f'{output_suffix}.nc')

        # 保存处理后的数据
        era5_resampled.to_netcdf(output_file)
        print(f"时间降尺度完成，保存至: {output_file}")

        era5_ds.close()

        return output_file

    except Exception as e:
        print(f"时间降尺度处理出错: {e}")
        traceback.print_exc()
        return None


def convert_to_comparable_times(times):
    """
    将时间转换为可比较的格式
    """
    if len(times) == 0:
        return pd.DatetimeIndex([])

    # 检查时间类型
    first_time = times[0]

    # 如果是cftime对象，转换为pandas时间戳
    if isinstance(first_time, cftime.DatetimeNoLeap) or hasattr(first_time, 'year'):
        # 处理cftime对象
        comparable_times = []
        for t in times:
            # 将cftime转换为pandas时间戳
            if hasattr(t, 'strftime'):
                # 使用字符串格式作为中介
                time_str = t.strftime('%Y-%m-%d %H:%M:%S')
                comparable_times.append(pd.Timestamp(time_str))
            else:
                comparable_times.append(pd.Timestamp(t))
        return pd.DatetimeIndex(comparable_times)
    else:
        # 已经是标准时间格式
        return pd.DatetimeIndex(times)


def compare_time_alignment(era5_file, cmip6_time_info):
    """
    比较ERA5和CMIP6数据的时间对齐情况
    """
    print("\n比较时间对齐情况:")
    print("=" * 50)

    # 读取ERA5数据
    era5_ds = xr.open_dataset(era5_file)

    # 确定ERA5时间坐标
    era5_time_coord = None
    for coord in ['valid_time', 'time']:
        if coord in era5_ds.coords:
            era5_time_coord = coord
            break

    if era5_time_coord is None:
        print("ERA5数据找不到时间坐标")
        era5_ds.close()
        return

    era5_times = era5_ds[era5_time_coord].values
    print(f"ERA5时间点数量: {len(era5_times)}")
    print(f"ERA5时间范围: {era5_times[0]} 到 {era5_times[-1]}")

    # 计算ERA5时间分辨率
    if len(era5_times) > 1:
        time_diffs = np.diff(era5_times)
        avg_resolution = np.mean([td / np.timedelta64(1, 'h') for td in time_diffs])
        print(f"ERA5时间分辨率: {avg_resolution:.2f} 小时")

    # 转换为可比较的时间格式
    era5_comparable = convert_to_comparable_times(era5_times)

    # 比较与每个CMIP6模型的时间对齐情况
    for model, info in cmip6_time_info.items():
        print(f"\n{model} 时间对齐情况:")
        cmip6_times = info['time_values']

        # 转换为可比较的时间格式
        cmip6_comparable = convert_to_comparable_times(cmip6_times)

        # 查找共同的时间点
        common_times = np.intersect1d(era5_comparable, cmip6_comparable)
        print(f"  共同时间点数量: {len(common_times)}")

        if len(common_times) > 0:
            coverage_era5 = len(common_times) / len(era5_comparable) * 100
            coverage_cmip6 = len(common_times) / len(cmip6_comparable) * 100
            print(f"  相对于ERA5的覆盖度: {coverage_era5:.1f}%")
            print(f"  相对于CMIP6的覆盖度: {coverage_cmip6:.1f}%")
        else:
            print(f"  警告: 没有共同时间点!")

    era5_ds.close()


def convert_cftime_to_datetime64(time_array):
    """
    将cftime时间数组转换为numpy datetime64数组

    参数:
    time_array: cftime时间数组

    返回:
    numpy.datetime64数组
    """
    if len(time_array) == 0:
        return np.array([], dtype='datetime64[ns]')

    # 检查是否为cftime类型
    first_time = time_array[0]
    if isinstance(first_time, cftime.DatetimeNoLeap) or hasattr(first_time, 'year'):
        # 转换为字符串，然后转换为datetime64
        datetime_list = []
        for t in time_array:
            # 使用cftime对象的strftime方法
            time_str = t.strftime('%Y-%m-%d %H:%M:%S')
            # 转换为datetime64
            dt64 = np.datetime64(time_str)
            datetime_list.append(dt64)
        return np.array(datetime_list)
    else:
        # 已经是datetime64或其他格式，直接返回
        return time_array


def calculate_monthly_mean(input_file, output_dir, variable_name=None):
    """
    计算月平均并保存 - 修复版，支持cftime时间坐标

    参数:
    input_file: 输入文件路径
    output_dir: 输出目录
    variable_name: 变量名，如果为None则使用第一个变量
    """
    print(f"\n处理: {os.path.basename(input_file)}")

    try:
        # 读取数据
        ds = xr.open_dataset(input_file)
        print(f"  原始数据形状: {ds.dims}")

        # 确定时间坐标名称
        time_coord = None
        for coord in ['time', 'valid_time']:
            if coord in ds.dims:
                time_coord = coord
                break

        if time_coord is None:
            print(f"  警告: {os.path.basename(input_file)} 找不到时间坐标")
            ds.close()
            return None

        print(f"  时间坐标: {time_coord}")
        print(f"  原始时间点数量: {len(ds[time_coord])}")

        # 检查时间坐标类型
        time_values = ds[time_coord].values
        first_time = time_values[0] if len(time_values) > 0 else None

        # 如果是cftime类型，需要转换
        if first_time is not None and (isinstance(first_time, cftime.DatetimeNoLeap) or hasattr(first_time, 'year')):
            print(f"  检测到cftime时间坐标，进行转换...")

            # 方法1: 使用xarray的decode_cf方法
            try:
                ds_decoded = xr.decode_cf(ds, decode_times=True)
                if time_coord in ds_decoded.coords:
                    ds = ds_decoded
                    print(f"  decode_cf转换成功")
            except:
                print(f"  decode_cf转换失败，使用手动转换")
                # 方法2: 手动转换
                new_times = convert_cftime_to_datetime64(time_values)
                ds = ds.assign_coords({time_coord: new_times})
        else:
            print(f"  时间坐标已经是datetime类型")

        # 确定变量名
        if variable_name is None:
            # 获取第一个变量名
            data_vars = list(ds.data_vars.keys())
            if data_vars:
                variable_name = data_vars[0]
                print(f"  使用变量: {variable_name}")
            else:
                print(f"  警告: {os.path.basename(input_file)} 没有数据变量")
                ds.close()
                return None

        # 检查时间坐标是否可以用于重采样
        try:
            # 尝试进行月平均计算
            print(f"  开始计算月平均...")
            monthly_ds = ds.resample({time_coord: '1MS'}).mean()

            print(f"  月平均后时间点数量: {len(monthly_ds[time_coord])}")
            print(f"  月平均后数据形状: {monthly_ds.dims}")

            # 保存结果
            output_filename = f"monthly_{os.path.basename(input_file)}"
            output_file = os.path.join(output_dir, output_filename)
            monthly_ds.to_netcdf(output_file)
            print(f"  保存到: {output_file}")

            ds.close()
            return output_file

        except Exception as e:
            print(f"  重采样失败: {e}")
            print(f"  尝试使用替代方法...")

            # 替代方法：使用groupby计算月平均
            try:
                # 添加月份和年份作为坐标
                ds_with_month = ds.assign_coords({
                    'year': ds[time_coord].dt.year,
                    'month': ds[time_coord].dt.month
                })

                # 按年月分组计算平均
                monthly_ds = ds_with_month.groupby('year').apply(
                    lambda x: x.groupby('month').mean()
                )

                # 重建时间坐标
                years = monthly_ds['year'].values
                months = monthly_ds['month'].values

                # 创建新的时间坐标
                new_times = pd.to_datetime([f'{int(y)}-{int(m):02d}-01' for y, m in zip(years, months)])
                monthly_ds = monthly_ds.assign_coords({time_coord: new_times})
                monthly_ds = monthly_ds.drop_vars(['year', 'month'], errors='ignore')

                print(f"  替代方法月平均后时间点数量: {len(monthly_ds[time_coord])}")

                # 保存结果
                output_filename = f"monthly_{os.path.basename(input_file)}"
                output_file = os.path.join(output_dir, output_filename)
                monthly_ds.to_netcdf(output_file)
                print(f"  保存到: {output_file}")

                ds.close()
                return output_file

            except Exception as e2:
                print(f"  替代方法也失败: {e2}")
                ds.close()
                return None

    except Exception as e:
        print(f"处理 {input_file} 时出错: {e}")
        traceback.print_exc()
        return None


def process_all_monthly_means(era5_file, cmip6_folder, output_dir):
    """
    处理所有数据的月平均

    参数:
    era5_file: ERA5数据文件路径
    cmip6_folder: CMIP6数据文件夹路径
    output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("开始计算月平均...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理ERA5数据
    print("\n处理ERA5数据月平均...")
    era5_result = calculate_monthly_mean(era5_file, output_dir, 't2m')

    # 处理所有CMIP6模型
    cmip6_files = [f for f in os.listdir(cmip6_folder) if f.startswith('cmip6_') and f.endswith('.nc')]

    if not cmip6_files:
        print("未找到CMIP6文件!")
        return

    print(f"\n处理 {len(cmip6_files)} 个CMIP6模型的月平均...")

    success_count = 0
    fail_count = 0

    for file in tqdm(cmip6_files, desc="处理CMIP6模型"):
        input_path = os.path.join(cmip6_folder, file)
        result = calculate_monthly_mean(input_path, output_dir, 'tas')

        if result is not None:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n月平均计算完成!")
    print(f"成功: {success_count}, 失败: {fail_count}")

    # 列出生成的文件
    print(f"\n生成的文件列表:")
    monthly_files = [f for f in os.listdir(output_dir) if f.startswith('monthly_') and f.endswith('.nc')]
    for f in monthly_files:
        file_path = os.path.join(output_dir, f)
        try:
            ds = xr.open_dataset(file_path)
            time_coord = 'time' if 'time' in ds.dims else 'valid_time'
            print(f"  {f}: {len(ds[time_coord])} 个时间点, {list(ds.data_vars.keys())}")
            ds.close()
        except:
            print(f"  {f}: 无法读取")


def main():
    """
    主函数：整合时间降尺度和月平均计算
    """
    # 设置路径
    era5_file = r"G:\CMIP6_tas_2000-2014\processed_1deg_china\era5_t2m_1deg_china_2000-2014.nc"
    cmip6_folder = r"G:\CMIP6_tas_2000-2014\processed_1deg_china"
    monthly_output_dir = r"G:\CMIP6_tas_2000-2014\monthly_1deg_china"

    # 要检查的CMIP6模型
    sample_models = ['BCC-CSM2-MR', 'NESM3', 'MRI-ESM2-0', 'MIROC-ES2L', 'MIROC6', 'CanESM5']

    print("开始整合处理：时间降尺度 + 月平均计算")
    print("=" * 80)

    # 1. 检查CMIP6数据的时间分辨率
    cmip6_time_info = inspect_cmip6_time_resolution(cmip6_folder, sample_models)

    if not cmip6_time_info:
        print("未找到有效的CMIP6时间信息")
        return

    # 2. 计算CMIP6的平均时间分辨率
    resolutions = [info['avg_resolution_hours'] for info in cmip6_time_info.values()]
    avg_cmip6_resolution = np.mean(resolutions)
    print(f"\nCMIP6平均时间分辨率: {avg_cmip6_resolution:.2f} 小时")

    # 确定目标分辨率（取最接近的整数小时）
    target_resolution = round(avg_cmip6_resolution)
    print(f"目标时间分辨率: {target_resolution} 小时")

    # 3. 比较时间对齐情况（处理前）
    print("\n处理前时间对齐情况:")
    compare_time_alignment(era5_file, cmip6_time_info)

    # 4. 对ERA5进行时间降尺度
    print(f"\n" + "=" * 80)
    print(f"步骤1：对ERA5进行时间降尺度到 {target_resolution} 小时分辨率")

    era5_resampled = resample_era5_time(era5_file, target_resolution_hours=target_resolution)

    if era5_resampled:
        # 5. 比较时间对齐情况（处理后）
        print("\n处理后时间对齐情况:")
        compare_time_alignment(era5_resampled, cmip6_time_info)

        # 6. 对所有数据进行月平均计算
        print(f"\n" + "=" * 80)
        print(f"步骤2：计算所有数据的月平均")
        print(f"ERA5文件: {era5_resampled}")
        print(f"CMIP6文件夹: {cmip6_folder}")
        print(f"输出目录: {monthly_output_dir}")

        process_all_monthly_means(era5_resampled, cmip6_folder, monthly_output_dir)

        print(f"\n" + "=" * 80)
        print("完整处理流程完成!")
        print(f"1. ERA5已降尺度到 {target_resolution} 小时分辨率")
        print(f"2. 所有数据已计算月平均")
        print(f"3. 月平均数据保存在: {monthly_output_dir}")
    else:
        print("ERA5时间降尺度失败，跳过月平均计算")

    print("\n" + "=" * 80)
    print("整合处理完成!")


def alternative_main():
    """
    备选主函数：如果已经有降尺度后的数据，直接计算月平均
    """
    # 设置路径
    era5_file = r"G:\CMIP6_tas_2000-2014\processed_1deg_china\era5_t2m_1deg_china_2000-2014_3h.nc"
    cmip6_folder = r"G:\CMIP6_tas_2000-2014\processed_1deg_china"
    monthly_output_dir = r"G:\CMIP6_tas_2000-2014\monthly_1deg_china"

    print("直接计算月平均（使用已有降尺度数据）")
    print("=" * 80)

    if not os.path.exists(era5_file):
        print(f"ERA5文件不存在: {era5_file}")
        print("请先运行完整流程进行时间降尺度")
        return

    process_all_monthly_means(era5_file, cmip6_folder, monthly_output_dir)


if __name__ == "__main__":
    # 选择运行模式
    # print("选择运行模式:")
    # print("1. 完整流程（时间降尺度 + 月平均）")
    # print("2. 仅月平均计算（已有降尺度数据）")
    #
    # choice = input("请输入选择 (1 或 2): ").strip()
    #
    # if choice == "1":
    #     main()
    # elif choice == "2":
    #     alternative_main()
    # else:
    #     print("无效选择，默认运行完整流程")
    #     main()
    main()
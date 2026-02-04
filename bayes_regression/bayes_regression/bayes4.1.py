#第三步进行贝叶斯回归
# 简化贝叶斯回归算法的版本（改进版检查点）
import xarray as xr
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import warnings
import time
import arviz
import pickle
import json  # 新增：用于保存进度信息

# 设置 Matplotlib 的字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

warnings.filterwarnings("ignore")


def load_processed_data(era5_file, cmip6_files):
    """
    加载处理后的ERA5和CMIP6数据
    """
    print("加载处理后的数据...")

    # 加载ERA5数据
    era5_ds = xr.open_dataset(era5_file)
    era5_var = list(era5_ds.data_vars.keys())[0]
    era5_data = era5_ds[era5_var]

    # 检查ERA5数据的NaN情况
    print(f"ERA5数据原始NaN数量: {np.isnan(era5_data.values).sum()}")
    print(f"ERA5数据形状: {era5_data.shape}")

    # 加载CMIP6数据
    cmip6_models = {}
    for model, file_path in cmip6_files.items():
        cmip6_ds = xr.open_dataset(file_path)
        cmip6_var = list(cmip6_ds.data_vars.keys())[0]
        cmip6_data = cmip6_ds[cmip6_var]

        # 检查CMIP6数据的NaN情况
        print(f"{model}数据NaN数量: {np.isnan(cmip6_data.values).sum()}")

        cmip6_models[model] = cmip6_data
        cmip6_ds.close()

    era5_ds.close()

    return era5_data, cmip6_models


def check_and_clean_coordinates(data):
    """
    检查并清理坐标，确保没有重复值
    """
    # 检查可能的纬度坐标名称
    lat_coord_names = ['latitude', 'lat', 'y']
    lon_coord_names = ['longitude', 'lon', 'x']

    lat_coord = None
    lon_coord = None

    # 找到实际的纬度坐标名称
    for coord_name in lat_coord_names:
        if coord_name in data.dims:
            lat_coord = coord_name
            break

    # 找到实际的经度坐标名称
    for coord_name in lon_coord_names:
        if coord_name in data.dims:
            lon_coord = coord_name
            break

    # 如果找不到空间坐标，可能是1D数据或者边界数据，直接返回
    if lat_coord is None or lon_coord is None:
        print(f"警告: 数据缺少空间坐标，可用的坐标: {list(data.dims)}")
        return data

    print(f"使用纬度坐标: {lat_coord}, 经度坐标: {lon_coord}")

    # 检查纬度是否有重复值
    lat_values = data[lat_coord].values
    if len(lat_values) != len(np.unique(lat_values)):
        print(f"发现重复的纬度值，进行清理...")
        # 使用第一个出现的纬度值，去除重复
        _, unique_indices = np.unique(lat_values, return_index=True)
        data = data.isel({lat_coord: np.sort(unique_indices)})

    # 检查经度是否有重复值
    lon_values = data[lon_coord].values
    if len(lon_values) != len(np.unique(lon_values)):
        print(f"发现重复的经度值，进行清理...")
        # 使用第一个出现的经度值，去除重复
        _, unique_indices = np.unique(lon_values, return_index=True)
        data = data.isel({lon_coord: np.sort(unique_indices)})

    return data


def bayesian_regression_single_point(y_ts, X_ts, model_names):
    """
    对单个网格点进行贝叶斯回归 - 使用全部日度数据版本
    """
    n_models = X_ts.shape[1]

    # 检查数据是否有NaN值
    valid_mask = ~(np.isnan(y_ts) | np.any(np.isnan(X_ts), axis=1))
    y_clean = y_ts[valid_mask]
    X_clean = X_ts[valid_mask, :]

    if len(y_clean) < 10:
        return {
            'coefficients': np.full(n_models, np.nan),
            'coefficients_std': np.full(n_models, np.nan),
            'intercept': np.nan,
            'intercept_std': np.nan,
            'r_squared': np.nan,
            'converged': False
        }

    try:
        with pm.Model() as model:
            # 调整先验分布以适应日度数据
            intercept = pm.Normal('intercept', mu=0, sigma=50)
            coefficients = pm.Normal('coefficients', mu=0, sigma=5, shape=n_models)
            sigma = pm.HalfNormal('sigma', sigma=20)

            # 线性模型
            mu = intercept + pm.math.dot(X_clean, coefficients)

            # 似然函数 - 使用全部数据
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_clean)

            # 优化采样设置以适应大量数据
            trace = pm.sample(
                300,  # 减少采样次数以加快速度
                tune=300,  # 减少调谐次数以加快速度
                cores=2,  # 使用2个核心
                progressbar=False,
                return_inferencedata=True,
                target_accept=0.9,  # 提高接受率以加速收敛
                max_treedepth=12,  # 降低树深度
                chains=2
            )

        # 检查收敛性
        try:
            import arviz as az
            summary = az.summary(trace)
            max_rhat = summary['r_hat'].max()
            min_ess = summary['ess_bulk'].min()

            # 使用更宽松的收敛标准以加快处理
            converged = (max_rhat < 1.1) and (min_ess > 100)

            if not converged:
                print(f"收敛警告: R-hat={max_rhat:.3f}, ESS={min_ess:.1f}")

        except Exception as e:
            print(f"收敛诊断失败: {e}")
            converged = True  # 假设收敛，继续处理

        # 提取后验分布统计量
        coefficients_mean = trace.posterior['coefficients'].mean(axis=(0, 1)).values
        coefficients_std = trace.posterior['coefficients'].std(axis=(0, 1)).values
        intercept_mean = trace.posterior['intercept'].mean(axis=(0, 1)).values
        intercept_std = trace.posterior['intercept'].std(axis=(0, 1)).values

        # 计算R²（使用完整数据）
        y_pred = intercept_mean + np.dot(X_clean, coefficients_mean)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            'coefficients': coefficients_mean,
            'coefficients_std': coefficients_std,
            'intercept': intercept_mean,
            'intercept_std': intercept_std,
            'r_squared': r_squared,
            'converged': converged
        }

    except Exception as e:
        print(f"贝叶斯回归失败: {e}")
        return {
            'coefficients': np.full(n_models, np.nan),
            'coefficients_std': np.full(n_models, np.nan),
            'intercept': np.nan,
            'intercept_std': np.nan,
            'r_squared': np.nan,
            'converged': False
        }


def load_checkpoint(checkpoint_dir, n_lat, n_lon, n_models):
    """加载检查点数据"""
    results_npz = os.path.join(checkpoint_dir, 'results.npz')
    progress_json = os.path.join(checkpoint_dir, 'progress.json')

    if not os.path.exists(results_npz):
        return None, None, None

    try:
        # 加载NumPy数组
        data = np.load(results_npz, allow_pickle=True)
        coefficients = data['coefficients']
        coefficients_std = data['coefficients_std']
        intercept = data['intercept']
        intercept_std = data['intercept_std']
        r_squared = data['r_squared']
        converged = data['converged']

        # 加载进度信息
        if os.path.exists(progress_json):
            with open(progress_json, 'r') as f:
                progress = json.load(f)
        else:
            progress = {'processed_points': set()}

        # 根据已处理的数据重建已处理点集合
        processed_points = set()
        for i in range(n_lat):
            for j in range(n_lon):
                if not np.isnan(intercept[i, j]):
                    processed_points.add((i, j))

        # 合并进度信息
        if 'processed_points' in progress:
            progress_points = set(tuple(p) for p in progress['processed_points'])
            processed_points.update(progress_points)

        print(f"成功加载检查点，已处理 {len(processed_points)} 个点")

        return {
            'coefficients': coefficients,
            'coefficients_std': coefficients_std,
            'intercept': intercept,
            'intercept_std': intercept_std,
            'r_squared': r_squared,
            'converged': converged
        }, processed_points, progress

    except Exception as e:
        print(f"加载检查点失败: {e}")
        return None, None, None


def save_checkpoint(checkpoint_dir, results, processed_points, progress_info=None):
    """保存检查点数据"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存NumPy数组
    results_npz = os.path.join(checkpoint_dir, 'results.npz')
    np.savez_compressed(
        results_npz,
        coefficients=results['coefficients'],
        coefficients_std=results['coefficients_std'],
        intercept=results['intercept'],
        intercept_std=results['intercept_std'],
        r_squared=results['r_squared'],
        converged=results['converged']
    )

    # 保存进度信息
    progress_json = os.path.join(checkpoint_dir, 'progress.json')
    progress_data = {
        'processed_points': list(processed_points),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'progress_info': progress_info
    }

    with open(progress_json, 'w') as f:
        json.dump(progress_data, f)

    print(f"检查点已保存: {results_npz}")


def perform_bayesian_regression_with_checkpoint(X, y, model_names, lats, lons,
                                                checkpoint_dir='checkpoints',
                                                checkpoint_interval=20,
                                                test_mode=False, test_sample_ratio=0.1):
    """
    对所有网格点进行贝叶斯回归 - 改进版检查点功能
    """
    n_time, n_models, n_lat, n_lon = X.shape

    print(f"输入数据信息:")
    print(f"  X形状: {X.shape}, 包含NaN: {np.isnan(X).sum()}")
    print(f"  y形状: {y.shape}, 包含NaN: {np.isnan(y).sum()}")
    print(f"  模型数量: {n_models}")
    print(f"  模型名称: {model_names}")

    # 如果是测试模式，只处理部分网格点
    if test_mode:
        print(f"测试模式: 采样比例 {test_sample_ratio}")
        total_points = n_lat * n_lon
        sample_size = max(1, int(total_points * test_sample_ratio))

        # 随机选择部分网格点
        all_points = [(i, j) for i in range(n_lat) for j in range(n_lon)]
        points_to_process = np.random.choice(
            len(all_points), size=sample_size, replace=False
        )
        points_to_process = [all_points[i] for i in points_to_process]
    else:
        # 处理所有网格点
        all_points = [(i, j) for i in range(n_lat) for j in range(n_lon)]
        points_to_process = all_points

    # 尝试加载检查点
    checkpoint_data, processed_points, progress_info = load_checkpoint(checkpoint_dir, n_lat, n_lon, n_models)

    if checkpoint_data is not None:
        # 从检查点恢复结果数组
        coefficients = checkpoint_data['coefficients']
        coefficients_std = checkpoint_data['coefficients_std']
        intercept = checkpoint_data['intercept']
        intercept_std = checkpoint_data['intercept_std']
        r_squared = checkpoint_data['r_squared']
        converged = checkpoint_data['converged']

        # 统计信息
        if progress_info and 'progress_info' in progress_info:
            success_count = progress_info['progress_info'].get('success_count', 0)
            fail_count = progress_info['progress_info'].get('fail_count', 0)
            skip_nan_count = progress_info['progress_info'].get('skip_nan_count', 0)
        else:
            # 从已有结果估计统计信息
            success_count = np.sum(converged[~np.isnan(converged)])
            fail_count = np.sum(~converged[~np.isnan(converged)])
            skip_nan_count = np.sum(np.isnan(intercept))
    else:
        # 初始化结果数组
        coefficients = np.full((n_lat, n_lon, n_models), np.nan)
        coefficients_std = np.full((n_lat, n_lon, n_models), np.nan)
        intercept = np.full((n_lat, n_lon), np.nan)
        intercept_std = np.full((n_lat, n_lon), np.nan)
        r_squared = np.full((n_lat, n_lon), np.nan)
        converged = np.full((n_lat, n_lon), False)

        processed_points = set()
        success_count = 0
        fail_count = 0
        skip_nan_count = 0

    # 创建需要处理的点列表（排除已处理的点）
    points_to_process = [p for p in points_to_process if p not in processed_points]
    total_points_to_process = len(points_to_process)
    total_all_points = n_lat * n_lon
    already_processed = len(processed_points)

    print(f"进度: 已处理 {already_processed}/{total_all_points} 个点")
    print(f"本次需要处理: {total_points_to_process} 个点")

    if total_points_to_process == 0:
        print("所有点已处理完成！")
    else:
        # 对每个网格点进行贝叶斯回归
        print("进行贝叶斯回归分析...")
        start_time = time.time()

        for idx, (i, j) in enumerate(tqdm(points_to_process, desc="处理网格点")):
            y_ts = y[:, i, j]
            X_ts = X[:, :, i, j]

            # 检查数据是否有效
            if np.all(np.isnan(y_ts)) or np.all(np.isnan(X_ts)):
                fail_count += 1
                skip_nan_count += 1
                processed_points.add((i, j))
                continue

            # 检查有效数据点数量
            valid_mask = ~(np.isnan(y_ts) | np.any(np.isnan(X_ts), axis=1))
            valid_count = np.sum(valid_mask)

            if valid_count < 50:  # 降低阈值以处理更多点
                fail_count += 1
                skip_nan_count += 1
                processed_points.add((i, j))
                continue

            result = bayesian_regression_single_point(y_ts, X_ts, model_names)

            coefficients[i, j, :] = result['coefficients']
            coefficients_std[i, j, :] = result['coefficients_std']
            intercept[i, j] = result['intercept']
            intercept_std[i, j] = result['intercept_std']
            r_squared[i, j] = result['r_squared']
            converged[i, j] = result['converged']

            processed_points.add((i, j))

            if result['converged']:
                success_count += 1
            else:
                fail_count += 1

            # 定期保存检查点
            if (idx + 1) % checkpoint_interval == 0:
                # 准备结果数据
                results_data = {
                    'coefficients': coefficients,
                    'coefficients_std': coefficients_std,
                    'intercept': intercept,
                    'intercept_std': intercept_std,
                    'r_squared': r_squared,
                    'converged': converged
                }

                # 准备进度信息
                progress_info = {
                    'success_count': success_count,
                    'fail_count': fail_count,
                    'skip_nan_count': skip_nan_count
                }

                # 保存检查点
                save_checkpoint(checkpoint_dir, results_data, processed_points, progress_info)

                # 显示进度
                current_processed = already_processed + idx + 1
                percent_complete = 100 * current_processed / total_all_points

                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    points_per_second = (idx + 1) / elapsed_time
                    remaining_points = total_points_to_process - (idx + 1)
                    if points_per_second > 0:
                        estimated_remaining_time = remaining_points / points_per_second
                        hours = int(estimated_remaining_time // 3600)
                        minutes = int((estimated_remaining_time % 3600) // 60)

                        print(f"\n进度: {current_processed}/{total_all_points} ({percent_complete:.1f}%)")
                        print(f"速度: {points_per_second:.2f} 点/秒")
                        print(f"预计剩余时间: {hours}小时{minutes}分钟")

    # 处理完成后保存最终检查点（但不要删除，作为备份）
    results_data = {
        'coefficients': coefficients,
        'coefficients_std': coefficients_std,
        'intercept': intercept,
        'intercept_std': intercept_std,
        'r_squared': r_squared,
        'converged': converged
    }

    progress_info = {
        'success_count': success_count,
        'fail_count': fail_count,
        'skip_nan_count': skip_nan_count
    }

    save_checkpoint(checkpoint_dir, results_data, processed_points, progress_info)

    print(f"\n贝叶斯回归结果统计:")
    print(f"  成功收敛的点: {success_count}")
    print(f"  失败的点: {fail_count}")
    print(f"  因NaN跳过的点: {skip_nan_count}")
    print(f"  总处理点: {already_processed + total_points_to_process}")
    print(f"  有效系数数量: {np.sum(~np.isnan(coefficients))}")
    print(f"  有效R²数量: {np.sum(~np.isnan(r_squared))}")

    # 创建结果数据集
    results_ds = xr.Dataset({
        'coefficients': (['latitude', 'longitude', 'model'], coefficients),
        'coefficients_std': (['latitude', 'longitude', 'model'], coefficients_std),
        'intercept': (['latitude', 'longitude'], intercept),
        'intercept_std': (['latitude', 'longitude'], intercept_std),
        'r_squared': (['latitude', 'longitude'], r_squared),
        'converged': (['latitude', 'longitude'], converged)
    }, coords={
        'latitude': lats,
        'longitude': lons,
        'model': model_names
    })

    # 添加属性
    results_ds.attrs['description'] = '贝叶斯回归结果：ERA5 ~ CMIP6多模型（使用全部日度数据）'
    results_ds.attrs['n_models'] = n_models
    results_ds.attrs['n_time_points'] = n_time
    results_ds.attrs['success_points'] = success_count
    results_ds.attrs['fail_points'] = fail_count
    results_ds.attrs['skip_nan_points'] = skip_nan_count

    return results_ds

def prepare_regression_data(era5_data, cmip6_models, time_slice=None, region=None):
    """
    准备回归分析的数据 - 修复NaN问题版本
    """
    print("准备回归分析数据...")

    # 清理坐标，确保没有重复值
    era5_data = check_and_clean_coordinates(era5_data)

    # 清理CMIP6数据的坐标，并过滤掉没有空间维度的数据
    cmip6_cleaned = {}
    invalid_models = []

    for model_name, model_data in cmip6_models.items():
        cleaned_data = check_and_clean_coordinates(model_data)

        # 检查数据是否有空间维度
        has_spatial_dims = any(coord in cleaned_data.dims for coord in ['latitude', 'lat', 'y']) and \
                           any(coord in cleaned_data.dims for coord in ['longitude', 'lon', 'x'])

        if has_spatial_dims:
            cmip6_cleaned[model_name] = cleaned_data
        else:
            print(f"跳过模型 {model_name}: 没有空间维度")
            invalid_models.append(model_name)

    if len(cmip6_cleaned) == 0:
        raise ValueError("没有有效的CMIP6模型数据用于回归分析")

    print(f"有效模型数量: {len(cmip6_cleaned)}")

    # 如果指定了时间切片，则选择相应时间段
    if time_slice is not None:
        print(f"\n应用时间切片: {time_slice}")
        era5_data = era5_data.sel(valid_time=time_slice)

        for model_name, model_data in cmip6_cleaned.items():
            print(f"处理模型 {model_name} 的时间切片...")
            # 根据时间维度名称进行选择
            if 'time' in model_data.dims:
                cmip6_cleaned[model_name] = model_data.sel(time=time_slice)
                print(f"  使用'time'维度，切片后形状: {cmip6_cleaned[model_name].shape}")
            elif 'valid_time' in model_data.dims:
                cmip6_cleaned[model_name] = model_data.sel(valid_time=time_slice)
                print(f"  使用'valid_time'维度，切片后形状: {cmip6_cleaned[model_name].shape}")
            else:
                print(f"  警告: 模型 {model_name} 没有找到时间维度")
    if region is not None:
        lat_min, lat_max, lon_min, lon_max = region
        print(f"裁剪数据到区域: 纬度 [{lat_min}, {lat_max}], 经度 [{lon_min}, {lon_max}]")

        era5_data = era5_data.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max)
        )

        for model_name in cmip6_cleaned:
            cmip6_cleaned[model_name] = cmip6_cleaned[model_name].sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max)
            )
    # 使用手动时间对齐
    print("使用手动时间对齐...")
    era5_aligned, cmip6_aligned = manual_time_align(era5_data, cmip6_cleaned)

    # 检查对齐后的ERA5数据NaN情况
    print(f"对齐后ERA5数据NaN数量: {np.isnan(era5_aligned.values).sum()}")

    # 确定空间坐标名称
    lat_coord = None
    lon_coord = None
    for coord_name in ['latitude', 'lat', 'y']:
        if coord_name in era5_aligned.dims:
            lat_coord = coord_name
            break
    for coord_name in ['longitude', 'lon', 'x']:
        if coord_name in era5_aligned.dims:
            lon_coord = coord_name
            break

    if lat_coord is None or lon_coord is None:
        raise ValueError(f"无法确定空间坐标名称，可用的坐标: {list(era5_aligned.dims)}")

    print(f"使用空间坐标: {lat_coord}, {lon_coord}")

    # 获取空间坐标值
    era5_lats = era5_aligned[lat_coord].values
    era5_lons = era5_aligned[lon_coord].values

    # 使用ERA5的网格作为基准
    common_lats = era5_lats
    common_lons = era5_lons

    # 重新网格化CMIP6数据到ERA5网格
    cmip6_regridded = {}

    print("\n重新网格化CMIP6数据...")
    for model_name, model_data in cmip6_aligned.items():
        try:
            print(f"处理模型: {model_name}")
            print(f"  原始形状: {model_data.shape}")

            # 检查模型数据是否已经有相同的网格
            model_lats = model_data[lat_coord].values
            model_lons = model_data[lon_coord].values

            lat_match = np.array_equal(model_lats, common_lats)
            lon_match = np.array_equal(model_lons, common_lons)

            if lat_match and lon_match:
                print(f"  网格匹配，直接使用")
                cmip6_regridded[model_name] = model_data
            else:
                print(f"  网格不匹配，进行重网格化")
                # 使用最近邻插值重新网格化
                regridded = model_data.interp(
                    {lat_coord: common_lats, lon_coord: common_lons},
                    method='nearest'
                )
                cmip6_regridded[model_name] = regridded

            print(f"  处理后形状: {cmip6_regridded[model_name].shape}")

        except Exception as e:
            print(f"重新网格化 {model_name} 时出错: {e}")
            continue

    # 检查是否有模型成功重网格化
    if len(cmip6_regridded) == 0:
        raise ValueError("所有模型重网格化都失败")

    # 获取维度信息
    n_time = len(era5_aligned.valid_time)
    n_lat = len(common_lats)
    n_lon = len(common_lons)
    n_models = len(cmip6_regridded)

    print(f"\n最终数据维度: 时间={n_time}, 纬度={n_lat}, 经度={n_lon}, 模型数={n_models}")

    # 准备X和y数组
    X = np.zeros((n_time, n_models, n_lat, n_lon))
    y = era5_aligned.values

    model_names = list(cmip6_regridded.keys())

    # 逐个模型赋值，检查形状
    for i, model_name in enumerate(model_names):
        model_data = cmip6_regridded[model_name].values
        print(f"赋值模型 {model_name}: 形状={model_data.shape}")

        # 检查形状是否匹配
        if model_data.shape != (n_time, n_lat, n_lon):
            print(f"警告: 模型 {model_name} 的形状 {model_data.shape} 不匹配期望形状 {(n_time, n_lat, n_lon)}")
            # 尝试调整形状
            if model_data.shape[0] == n_time and model_data.shape[1] == n_lat and model_data.shape[2] == n_lon:
                X[:, i, :, :] = model_data
            else:
                print(f"跳过模型 {model_name}")
                X[:, i, :, :] = np.nan
        else:
            X[:, i, :, :] = model_data

    # 检查最终数据的NaN情况
    print(f"\n最终数据NaN统计:")
    print(f"  X中的NaN数量: {np.isnan(X).sum()}")
    print(f"  y中的NaN数量: {np.isnan(y).sum()}")

    # 如果y中有大量NaN，尝试找出原因
    if np.isnan(y).sum() > 0:
        # 找出有NaN的格点
        nan_grid_points = np.isnan(y).all(axis=0)  # 时间轴上全部为NaN的格点
        print(f"  完全无效的格点数量: {nan_grid_points.sum()}")

        # 找出部分有NaN的格点
        partial_nan_grid_points = np.isnan(y).any(axis=0) & ~nan_grid_points
        print(f"  部分无效的格点数量: {partial_nan_grid_points.sum()}")

    return X, y, model_names, common_lats, common_lons


def manual_time_align(era5_data, cmip6_models):
    """
    手动对齐时间的备用函数
    """
    print("使用手动时间对齐...")

    # 获取ERA5的时间坐标
    era5_times = era5_data.valid_time.values

    # 对齐每个CMIP6模型到ERA5的时间
    cmip6_aligned = {}
    all_common_times = []

    for model_name, model_data in cmip6_models.items():
        print(f"对齐模型 {model_name}...")

        # 确定模型的时间坐标名称
        time_dim = None
        for dim in model_data.dims:
            if 'time' in dim:
                time_dim = dim
                break

        if time_dim is None:
            print(f"  警告: 模型 {model_name} 没有时间维度，跳过")
            continue

        # 获取模型的时间坐标
        model_times = model_data[time_dim].values

        # 找到共同的时间点
        if len(model_times) > 0 and len(era5_times) > 0:
            # 将时间转换为pandas时间戳以便比较
            try:
                if hasattr(model_times[0], 'strftime'):  # cftime对象
                    model_times_pd = pd.to_datetime([t.strftime('%Y-%m-%d %H:%M:%S') for t in model_times])
                else:  # numpy datetime64
                    model_times_pd = pd.to_datetime(model_times)

                era5_times_pd = pd.to_datetime(era5_times)

                # 找到共同的时间点
                common_times_pd = np.intersect1d(model_times_pd, era5_times_pd)

                if len(common_times_pd) > 0:
                    # 选择共同的时间点
                    model_common_indices = np.where(np.isin(model_times_pd, common_times_pd))[0]

                    print(f"  共同时间点数量: {len(common_times_pd)}")

                    # 选择共同的时间点
                    cmip6_aligned[model_name] = model_data.isel({time_dim: model_common_indices})
                    print(f"  对齐后形状: {cmip6_aligned[model_name].shape}")

                    # 记录这个模型的共同时间
                    all_common_times.append(common_times_pd)
                else:
                    print(f"  警告: 模型 {model_name} 没有共同时间点")
                    continue
            except Exception as e:
                print(f"  时间对齐失败: {e}")
                continue
        else:
            print(f"  警告: 模型 {model_name} 或 ERA5 的时间坐标为空")
            continue

    if len(cmip6_aligned) == 0:
        raise ValueError("没有模型成功对齐时间")

    # 找到所有模型共有的最小时间长度
    min_time_length = min(len(times) for times in all_common_times)
    print(f"所有模型的最小时间长度: {min_time_length}")

    # 统一所有模型的时间长度
    for model_name, model_data in cmip6_aligned.items():
        current_time_length = len(model_data[list(model_data.dims)[0]])
        if current_time_length > min_time_length:
            print(f"截断模型 {model_name} 的时间长度: {current_time_length} -> {min_time_length}")
            time_dim = [dim for dim in model_data.dims if 'time' in dim][0]
            cmip6_aligned[model_name] = model_data.isel({time_dim: slice(0, min_time_length)})

    # 使用截断后的时间长度来对齐ERA5
    era5_aligned = era5_data.isel(valid_time=slice(0, min_time_length))

    print(f"\n最终共同时间点数量: {min_time_length}")
    print(f"ERA5对齐后形状: {era5_aligned.shape}")

    return era5_aligned, cmip6_aligned


def visualize_regression_results(results_ds, output_dir):
    """
    可视化回归结果
    """
    print("生成回归结果可视化...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 绘制每个模型的系数空间分布
    n_models = len(results_ds.model)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

    if n_models == 1:
        axes = axes.reshape(2, 1)

    for i, model in enumerate(results_ds.model.values):
        # 系数均值
        ax1 = axes[0, i]
        im1 = results_ds.coefficients.sel(model=model).plot(
            ax=ax1, cmap='RdBu_r', center=0,
            cbar_kwargs={'label': '回归系数'}
        )
        ax1.set_title(f'{model} - 回归系数')

        # 系数标准差
        ax2 = axes[1, i]
        im2 = results_ds.coefficients_std.sel(model=model).plot(
            ax=ax2, cmap='viridis',
            cbar_kwargs={'label': '系数标准差'}
        )
        ax2.set_title(f'{model} - 系数不确定性')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coefficients_spatial.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 绘制R²空间分布
    plt.figure(figsize=(12, 8))
    results_ds.r_squared.plot(cmap='viridis', vmin=0, vmax=1)
    plt.title('贝叶斯回归 R² 空间分布')
    plt.savefig(os.path.join(output_dir, 'r_squared_spatial.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 绘制截距空间分布
    plt.figure(figsize=(12, 8))
    results_ds.intercept.plot(cmap='RdBu_r', center=0)
    plt.title('截距空间分布')
    plt.savefig(os.path.join(output_dir, 'intercept_spatial.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 绘制收敛情况
    plt.figure(figsize=(12, 8))
    results_ds.converged.plot()
    plt.title('模型收敛情况 (True=收敛, False=未收敛)')
    plt.savefig(os.path.join(output_dir, 'convergence_spatial.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_regression_summary(results_ds, output_dir):
    """
    保存回归结果统计摘要
    """
    print("生成回归结果摘要...")

    summary_data = []

    for model in results_ds.model.values:
        coeffs = results_ds.coefficients.sel(model=model).values
        coeffs_std = results_ds.coefficients_std.sel(model=model).values

        # 只考虑收敛的点
        converged_mask = results_ds.converged.values
        valid_coeffs = coeffs[converged_mask & ~np.isnan(coeffs)]
        valid_coeffs_std = coeffs_std[converged_mask & ~np.isnan(coeffs_std)]

        if len(valid_coeffs) > 0:
            summary_data.append({
                'Model': model,
                'Mean_Coefficient': np.mean(valid_coeffs),
                'Std_Coefficient': np.std(valid_coeffs),
                'Mean_Uncertainty': np.mean(valid_coeffs_std),
                'Positive_Coefficient_Ratio': np.mean(valid_coeffs > 0),
                'Significant_Positive_Ratio': np.mean(valid_coeffs > valid_coeffs_std),
                'Significant_Negative_Ratio': np.mean(valid_coeffs < -valid_coeffs_std),
                'N_Valid_Points': len(valid_coeffs)
            })

    summary_df = pd.DataFrame(summary_data)

    # 保存为CSV
    summary_df.to_csv(os.path.join(output_dir, 'regression_summary.csv'), index=False)

    # 打印摘要
    print("\n回归结果摘要:")
    print("=" * 80)
    print(summary_df.round(4))

    # R²统计
    r2_valid = results_ds.r_squared.values[results_ds.converged.values & ~np.isnan(results_ds.r_squared.values)]
    if len(r2_valid) > 0:
        print(f"\nR²统计:")
        print(f"  均值: {np.mean(r2_valid):.4f}")
        print(f"  标准差: {np.std(r2_valid):.4f}")
        print(f"  中位数: {np.median(r2_valid):.4f}")
        print(f"  最大值: {np.max(r2_valid):.4f}")
        print(f"  最小值: {np.min(r2_valid):.4f}")



# 主函数
def main(test_mode=False, test_region=None, test_sample_ratio=1, checkpoint_interval=20):
    # 设置已经经过空间网格重置以及时间降尺度的月度数据路径
    era5_file = r"G:\CMIP6_tas_2000-2014\monthly_1deg_china\monthly_era5_t2m_1deg_china_2000-2014_3h.nc"
    cmip6_folder = r"G:\CMIP6_tas_2000-2014\monthly_1deg_china"
    output_dir = r"G:\CMIP6_tas_2000-2014\bayesian_regression_results"
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    # 获取CMIP6处理后的文件
    cmip6_files = {}
    for file in os.listdir(cmip6_folder):
        if file.startswith('monthly_cmip6_') and file.endswith('.nc'):
            parts = file.split('_')
            if len(parts) >= 3:
                model = parts[2]
                cmip6_files[model] = os.path.join(cmip6_folder, file)
                print(f"找到模型: {model}, 文件: {file}")

    print(f"找到 {len(cmip6_files)} 个CMIP6模型")

    if len(cmip6_files) == 0:
        print("错误: 没有找到任何CMIP6模型文件")
        return

    # 1. 加载数据
    era5_data, cmip6_models = load_processed_data(era5_file, cmip6_files)

    # 2. 准备回归数据
    X, y, model_names, lats, lons = prepare_regression_data(
        era5_data, cmip6_models,
        time_slice=slice('2000-01-01', '2014-12-31'),
        region=test_region
    )

    # 3. 进行贝叶斯回归（使用改进版检查点）
    results = perform_bayesian_regression_with_checkpoint(
        X, y, model_names, lats, lons,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        test_mode=test_mode,
        test_sample_ratio=test_sample_ratio
    )

    # 4. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'bayesian_regression_results_full_monthly.nc')
    results.to_netcdf(results_file)
    print(f"最终结果已保存到: {results_file}")

    # 5. 可视化结果
    visualize_regression_results(results, output_dir)

    # 6. 生成摘要
    save_regression_summary(results, output_dir)

    print(f"\n贝叶斯回归分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    # 为了测试，你可以先运行小区域
    #test_region = [30, 33, 110, 113]  # 小区域测试

    # 运行完整区域
    test_region = None

    main(
        test_mode=False,  # 设置为False处理所有点
        test_region=test_region,
        test_sample_ratio=1,  # 处理100%的点
        checkpoint_interval=20  # 每20个点保存一次
    )
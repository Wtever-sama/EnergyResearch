import xarray as xr
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_composite_v1(solar_v1_path, wind_v1_path, output_path):
    '''
    计算风光复合极端事件标志 (1: 共同极端, 0: 非共同极端, -1: 缺失)
    
    :param solar_v1_path: 太阳能 V1 标志文件路径
    :param wind_v1_path: 风能 V1 标志文件路径
    :param output_path: 输出复合标志文件路径
    '''
    logger.info("正在加载输入文件...")
    
    # 使用 chunks 开启延迟加载，避免内存溢出
    ds_solar = xr.open_dataset(solar_v1_path, chunks={'time': 5000})
    ds_wind = xr.open_dataset(wind_v1_path, chunks={'time': 5000})
    
    # 提取变量 (假设变量名均为 'is_extreme')
    v1_s = ds_solar['is_extreme']
    v1_w = ds_wind['is_extreme']
    
    logger.info("检查坐标对齐情况...")
    # 确保时间、纬度、经度完全对齐
    # 如果网格不一致，这里会触发 xarray 的自动对齐或报错
    xr.align(v1_s, v1_w, join='exact')

    logger.info("开始计算复合极端标志 (AND 逻辑)...")
    
    # 核心逻辑：
    # 1. 只有当两者都为 1 时，结果为 1
    # 2. 如果其中任一为 NaN 或 -1 (缺失数据)，结果设为 -1
    # 3. 其他情况（至少一个为 0）结果为 0
    
    # 首先识别有效数据区域 (两者都不为 NaN 且都不为 -1)
    mask_valid = (v1_s >= 0) & (v1_w >= 0)
    
    # 计算复合极端：两者同时为 1
    composite_condition = (v1_s == 1) & (v1_w == 1)
    
    # 构建结果数组
    # 初始化为 -1 (int8 节省空间)
    res_array = xr.full_like(v1_s, -1, dtype='i1')
    
    # 填充 0 和 1
    # where(condition, x, y) -> 如果 condition 为真则选 x，否则选 y
    res_array = xr.where(mask_valid, xr.where(composite_condition, 1, 0), -1)
    
    # 设置元数据
    res_array.name = "is_extreme"
    res_array.attrs = {
        'long_name': 'Compound Extreme Flag (Solar & Wind)',
        'description': '1: Both solar and wind are extreme; 0: Non-compound extreme; -1: Missing data',
        'units': '1'
    }

    logger.info(f"正在保存结果至: {output_path}")
    # 保存时使用压缩
    encoding = {res_array.name: {'zlib': True, 'complevel': 4, 'dtype': 'i1', '_FillValue': -1}}
    res_array.to_netcdf(output_path, encoding=encoding)
    
    logger.info("复合极端事件计算完成！")


if __name__ == "__main__":
    # 填入你的文件路径
    solar_path = "G:/extreme_analysis/results/Solar/Solar_ssp585_V1_flag.nc"
    wind_path = "G:/extreme_analysis/results/Wind/Wind_ssp585_V1_flag.nc"
    comp_out_path = "G:/extreme_analysis/results/Composite/Solar_Wind_ssp585_V1_flag.nc"
    
    import os
    os.makedirs(os.path.dirname(comp_out_path), exist_ok=True)
    
    get_composite_v1(solar_path, wind_path, comp_out_path)
''' 期望偏差法: 计算风速4*8*N CMIP6 的历史偏差 -> delta X 从小到大排序, 使用 10 分位数计算异常值'''

import xarray as xr
import os
import logging
import warnings
warnings.filterwarnings("ignore")

from utils import save

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class WindProcessor:
    def __init__(self, file_path, var_name, time_range=None):
        self.file_path = file_path
        logging.info(f"打开数据集: {os.path.basename(file_path)}")
        try:
            self.ds = xr.open_dataset(file_path)
        except Exception as e:
            logging.error(f"无法打开文件 {file_path}: {e}")
            raise
        if time_range:
            self.ds = self.ds.sel(time=slice(time_range[0], time_range[1]))
            logging.info(f"已选择时间范围: {time_range[0]} 到 {time_range[1]}")
        
        # 灵活匹配变量名: 历史数据为 'ws100m', SSP数据为 'WindCF'
        if var_name in self.ds.data_vars:
            self.raw_data = self.ds[var_name]
            logging.info(f"已载入变量 '{var_name}'，形状={self.raw_data.shape}")
        else:
            logging.error(f"变量 '{var_name}' 未在文件中找到。可用变量: {list(self.ds.data_vars)}")
            raise KeyError(f"变量 {var_name} 不存在于数据集中")
        self.cf = None

    def calculate_wind_cf(self):
        """将风速转化为容量因子 CF (仅针对历史风速数据)"""
        if "WindCF" in self.ds.data_vars:
            logging.info(f"检测到直接 CF 变量: {os.path.basename(self.file_path)}")
            self.cf = self.raw_data.rename("wind_cf")
        else:
            logging.info(f"正在转换风速为 CF: {os.path.basename(self.file_path)}")
            # 简化版风机功率曲线 (可根据实际需求更替更复杂的非线性函数)
            # 假设 v < 3m/s 为 0, v > 25m/s 切出为 0, 中间线性或额定
            v = self.raw_data
            cf = xr.where((v > 3) & (v < 12), (v - 3) / (12 - 3), 0)
            cf = xr.where((v >= 12) & (v <= 25), 1, cf)
            self.cf = cf.clip(0, 1).rename("wind_cf")
        try:
            v_mean = float(v.mean().values)
            v_min = float(v.min().values)
            v_max = float(v.max().values)
            logging.info(f"原始风速样本统计: mean={v_mean:.3f}, min={v_min:.3f}, max={v_max:.3f}")
            cf_mean = float(self.cf.mean().values)
            logging.info(f"生成的 CF 统计: mean={cf_mean:.3f}, shape={self.cf.shape}")
        except Exception:
            logging.debug("无法计算风速/CF 的聚合统计，可能是懒加载或数据量太大")

        return self.cf

    def get_expectations(self):
        """计算历史期望矩阵 (4季节 * 8时段)"""
        if self.cf is None: self.calculate_wind_cf()
        # 根据定义, 96 或 32 个典型时段 [cite: 171]
        expectations = (self.cf.groupby("time.season")
                .apply(lambda x: x.groupby("time.hour").mean(dim="time")))
        logging.info(f"计算历史期望矩阵完成: seasons={list(expectations['season'].values) if 'season' in expectations.coords else 'unknown'}; shape={expectations.shape}")
        return expectations

    def get_deviations(self, exp_matrix):
        """偏差计算"""
        if self.cf is None: self.calculate_wind_cf()
        logging.info("开始计算偏差序列...")
        
        # 1. 确保期望矩阵的维度顺序正确 (season, hour, lat, lon)
        # 原日志显示形状为 (4, 41, 71, 8)，需要将 hour 调到前面
        if 'hour' in exp_matrix.dims:
            exp_matrix = exp_matrix.transpose('season', 'hour', 'lat', 'lon')

        # 2. 预先提取季节和小时索引，避免在 apply 中反复计算
        # 这比在 apply 内部调用 dt.season 快得多
        seasons = self.cf.time.dt.season
        hours = self.cf.time.dt.hour

        # 3. 手动广播减法 (比 groupby.apply 更快且更节省内存)
        # 获取对应的期望值
        # 注意：这里利用了 xarray 的高级索引功能
        exp_broadcasted = exp_matrix.sel(season=seasons, hour=hours)
        
        # 执行减法
        dev = (self.cf - exp_broadcasted).rename("delta_x")
        
        # 移除辅助坐标
        if 'season' in dev.coords: dev = dev.drop_vars('season')
        if 'hour' in dev.coords: dev = dev.drop_vars('hour')

        dev = dev.sortby("time")
        logging.info("计算偏差序列完成")
        return dev
    

def main():
    hist_path = r"G:\extreme_analysis\data\CMIP6_QDM_MME\QDM_cmip6_MME_ws100m_historical_1deg_china_2000-2014.nc"
    output_dir = r"G:\extreme_analysis\results\wind"
    
    logging.info("--- 第一步：建立历史基准 ---")
    # 注意变量名传 'ws100m'
    hist_proc = WindProcessor(hist_path, var_name='ws100m')
    exp_matrix = hist_proc.get_expectations()
    out_hist = os.path.join(output_dir, "windCF_historical_exp_32_slots.nc")
    logging.info(f"保存历史期望到: {out_hist}")
    try:
        save(exp_matrix, out_hist)
    except Exception as e:
        logging.error(f"保存历史期望时出错: {e}")
    
    # --- 第二步：处理 SSP 2040-2060 ---
    scenarios = ["ssp126", "ssp245", "ssp585"]
    time_range = ('2040-01-01', '2060-12-31')
    
    for scn in scenarios:
        ssp_path = rf"G:\extreme_analysis\data\CMIP6_QDM_MME\QDM_cmip6_MME_WindCF_{scn}_1deg_china_2025-2060.nc"
        
        # 注意 SSP 文件中变量名是 'WindCF'
        logging.info(f"处理情景: {scn}; 文件: {ssp_path}")
        ssp_proc = WindProcessor(ssp_path, var_name='WindCF', time_range=time_range)

        # 偏差 (V2)
        delta_x = ssp_proc.get_deviations(exp_matrix)
        logging.info(f"delta_x 统计: mean={float(delta_x.mean().values):.4f}")
        
        # 阈值 (P10)
        p10 = delta_x.quantile(0.1, dim="time") 
        logging.info(f"P10 阈值计算完成: shape={p10.shape}")
        
        # --- 极端标识 (V1) 并保持 NaN 掩码 ---
        # 1. 首先生成 0-1 标识
        is_extreme_raw = xr.where(delta_x < p10, 1, 0)
        
        # 2. 利用 delta_x 的有效值区域对 0-1 标识进行遮罩
        # 只有当 delta_x 不是 NaN 时才保留 is_extreme_raw 的值，否则设为 NaN
        is_extreme = is_extreme_raw.where(delta_x.notnull()).rename("is_extreme") 
        
        logging.info(f"is_extreme 生成 (已保留 NaN 掩码): shape={is_extreme.shape}")
        
        # 保存 V1, V2
        out_v2 = os.path.join(output_dir, f"wind_{scn}_V2_delta_x_2040-2060.nc")
        out_v1 = os.path.join(output_dir, f"wind_{scn}_V1_extreme_flag_2040-2060.nc")
        
        logging.info(f"保存: V2 -> {out_v2}")
        try:
            save(delta_x, out_v2)
        except Exception as e:
            logging.error(f"保存 V2 时出错: {e}")
        logging.info(f"保存: V1 -> {out_v1}")
        try:
            save(is_extreme, out_v1)
        except Exception as e:
            logging.error(f"保存 V1 时出错: {e}")


if __name__ == "__main__":
    main()
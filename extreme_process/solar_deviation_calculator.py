''' 期望偏差法: 计算辐射4*8*N CMIP6 的历史偏差 -> delta X 从小到大排序, 使用 10 分位数计算异常值'''

import xarray as xr
import os
from tqdm import tqdm
import warnings 
warnings.filterwarnings("ignore")

from utils import save


class SolarProcessor:
    def __init__(self, file_path, time_range=None):
        self.file_path = file_path
        self.ds = xr.open_dataset(file_path)
        
        # --- 仅增加年份截断逻辑 ---
        if time_range:
            self.ds = self.ds.sel(time=slice(time_range[0], time_range[1]))
            
        # 兼容性处理：如果变量名不是 rsds 请自行修改
        self.raw_data = self.ds['rsds'] 
        self.cf = None

    def calculate_solar_cf(self, r_std: float | int=1000.0) -> xr.DataArray:
        """将辐射量转化为容量因子 CF"""
        print(f"正在计算 CF: {os.path.basename(self.file_path)}")
        cf = self.raw_data / r_std
        self.cf = cf.clip(0, 1).rename("solar_cf")
        return self.cf

    def get_expectations(self) -> xr.DataArray:
        """计算历史期望矩阵 (4季节 * 8时段)"""
        if self.cf is None: self.calculate_solar_cf()
        print("计算历史期望矩阵...")
        # 分组聚合：按季节和小时计算均值 [cite: 171, 175]
        expectations = (self.cf.groupby("time.season")
                        .apply(lambda x: x.groupby("time.hour").mean(dim="time")))
        return expectations

    def get_deviations(self, exp_matrix: xr.DataArray) -> xr.DataArray:
        """
        利用计算出的 expectation 计算偏差 delta_x [cite: 176, 177]
        """
        if self.cf is None: 
            self.calculate_solar_cf()
        print("正在计算偏差序列 (Delta X)...")
        
        def _subtract_exp(group):
            # --- 从 time 坐标提取季节名称 ---
            # group.time.dt.season 会返回该数据组对应的季节标识（如 'DJF'）
            season_name = str(group.time.dt.season[0].values) 
            
            exp_season = exp_matrix.sel(season=season_name)
            # 在该季节内按小时相减
            return group.groupby("time.hour") - exp_season

        # 执行广播减法
        deviations = self.cf.groupby("time.season").apply(_subtract_exp)
        
        if 'season' in deviations.coords:
            deviations = deviations.drop_vars('season')
        
        return deviations.sortby("time").rename("delta_x")


def main(hist_path: str=\
         r"G:\extreme_analysis\data\CMIP6_QDM_MME\QDM_cmip6_MME_rsds_historical_1deg_china_2000-2014.nc",
         output_dir: str=\
         r"G:\extreme_analysis\results\solar",
         save_format: str="nc") -> None:
    # --- 第一步：从 Historical 获取期望基准 ---
    hist_proc = SolarProcessor(hist_path) # 历史基准保持原样
    exp_matrix = hist_proc.get_expectations()
    save(exp_matrix, os.path.join(output_dir, "solarCF_historical_exp_32_slots.{}".format(save_format)))

    # --- 第二步：从 SSP 计算偏差 (加入 2040-2060 截断) ---
    scenarios = ["ssp126", "ssp245", "ssp585"]
    time_range = ('2040-01-01', '2060-12-31') # 定义目标截断区间 
    
    for scenario in tqdm(scenarios, desc="Processing Scenarios"):
        ssp_path = r"G:\extreme_analysis\data\CMIP6_QDM_MME\QDM_cmip6_MME_rsds_{}_1deg_china_2025-2060.nc".format(scenario)
        
        # 1. 初始化并截断时间
        ssp_proc = SolarProcessor(ssp_path, time_range=time_range)
        
        # 2. 计算 V2: 具体差值 (Delta X)
        delta_x = ssp_proc.get_deviations(exp_matrix)
        save(delta_x, os.path.join(output_dir, "solar_{}_V2_delta_x_2040-2060.{}".format(scenario, save_format)))

        # 3. 计算 P10 阈值
        p10_threshold = delta_x.quantile(0.1, dim="time") 
        save(p10_threshold, os.path.join(output_dir, "solar_{}_p10_thresholds_2040-2060.{}".format(scenario, save_format)))

        # --- 计算 V1 (0-1 标识) 并保持 NaN 掩码 --- 
        # 1. 首先生成布尔逻辑下的 0-1 标识
        is_extreme_raw = xr.where(delta_x < p10_threshold, 1, 0)
        
        # 2. 利用 delta_x 的有效值区域对 0-1 标识进行遮罩
        # 只有当 delta_x 不是 NaN 时（即陆地区域）才保留值，否则设为 NaN
        is_extreme = is_extreme_raw.where(delta_x.notnull()).rename("is_extreme")
        
        # 确保 V1 文件的维度与 V2 一致
        save(is_extreme, os.path.join(output_dir, "solar_{}_V1_extreme_flag_2040-2060.{}".format(scenario, save_format)))


if __name__ == "__main__":
    main()
import cdsapi
import os
from tqdm import tqdm

# 初始化CDS客户端
c = cdsapi.Client()

# 定义要下载的变量列表（按需修改）

VARIABLES = [
    '2m dewpoint temperature',  # 2米露点温度
    'Surface pressure',  # 地表气压
]

# 定义存储目录（按变量分类）
BASE_DIR = r"D:\ERA5DataDownload"

MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
DAYS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
        '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']


def download_era5_variable_year(variable, year):
    """下载单个变量单年的数据"""
    # 按变量创建子目录（如 E:\DataDownload\ERA5\2m_temperature）
    output_dir = os.path.join(BASE_DIR, variable)
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名（如 E:\DataDownload\ERA5\mean_runoff_rate\era5_runoff_1980_01.nc）
    output_file = os.path.join(output_dir, f"era5_{variable}_{year}.nc")

    # 如果文件已存在，则跳过
    if os.path.exists(output_file):
        print(f"File exists: {output_file}")
        return

    # 提交请求
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variable,
                'year': str(year),
                'month': MONTHS,
                'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                        '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
                'time': ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                        "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [53.6, 73.29, 16, 135.5]
            },
            output_file
        )
        print(f"✅ Downloaded: {variable}_{year}")
    except Exception as e:
        print(f"❌ Failed: {variable}_{year}. Error: {e}")


# 循环下载所有变量和年份
for variable in tqdm(VARIABLES, desc="Variables"):
    for year in tqdm(range(2000, 2021), desc="Years", leave=False):  # （1980-2024）优先2000-2020
            download_era5_variable_year(variable, year)

print("All downloads completed!")
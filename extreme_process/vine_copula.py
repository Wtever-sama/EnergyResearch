'''
Kendall's tau 矩阵量化空间相关性, 反映了【极端】风险同步性
区域间 tau>0.4, 强正相关区域 (风险集群)
区域间 -0.1<tau<0.1, 风险独立
变量间 tau<0, 风光互补, 否则容易面临双重打击
'''

import pandas as pd
import numpy as np
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from scipy.stats import gaussian_kde


logger = logging.getLogger(__name__)


def load_input_data(file_path: str) -> pd.DataFrame:
    logger.info("开始加载输入数据: %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("数据加载完成: rows=%d, cols=%d", df.shape[0], df.shape[1])
    logger.debug("列名: %s", list(df.columns))
    return df


def pit_transform(df: pd.DataFrame) -> pd.DataFrame:
    '''边缘分布转换: 使用 KDE 拟合边缘分布并执行 PIT 转换'''
    logger.info("开始执行 PIT 转换")

    # 原本算法-秩变换
    # df_jittered = df + np.random.normal(0, 1e-8, size=df.shape)
    # u_data = df + np.random.normal(0, 1e-8, size=df.shape)
    # u_data = df.rank(method="average") / (len(df) + 1)
    
    # --- KDE 核密度估计拟合 (零很多，可能导致分布不连续) ---
    rng = np.random.default_rng(42)
    u_data = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        finite_mask = np.isfinite(x)
        u_col = np.full_like(x, np.nan, dtype=float)

        if finite_mask.sum() == 0:
            logger.warning("列 %s 全为缺失或非数值，已跳过", col)
            u_data[col] = u_col
            continue

        x_valid = x[finite_mask]

        # 大量重复值场景下加入极小扰动，避免 KDE 拟合奇异
        spread = float(np.std(x_valid))
        jitter_scale = 1e-8 if spread == 0.0 else spread * 1e-8
        x_fit = x_valid + rng.uniform(-jitter_scale, jitter_scale, size=x_valid.shape)

        # 常数列无法估计有效核密度，映射到中位概率
        if np.allclose(x_fit, x_fit[0]):
            u_valid = np.full_like(x_valid, 0.5, dtype=float)
            u_col[finite_mask] = u_valid
            u_data[col] = u_col
            logger.warning("列 %s 近似常数，PIT 映射为 0.5", col)
            continue

        kde = gaussian_kde(x_fit)
        x_min = float(np.min(x_fit))
        x_max = float(np.max(x_fit))
        pad = (x_max - x_min) * 0.1
        if pad == 0.0:
            pad = 1.0

        grid = np.linspace(x_min - pad, x_max + pad, 2048)
        pdf = kde(grid)
        dx = np.diff(grid)
        cdf = np.concatenate(([0.0], np.cumsum((pdf[:-1] + pdf[1:]) * dx * 0.5)))

        if cdf[-1] <= 0:
            u_valid = (pd.Series(x_valid).rank(method="average").to_numpy()) / (len(x_valid) + 1)
        else:
            cdf = cdf / cdf[-1]
            u_valid = np.interp(x_valid, grid, cdf)

        u_valid = np.clip(u_valid, 1e-6, 1 - 1e-6)
        u_col[finite_mask] = u_valid
        u_data[col] = u_col
    # --- KDE 拟合结束 ---

    logger.info("PIT 转换完成")
    logger.debug(
        "PIT 数据范围: min=%.6f, max=%.6f",
        float(u_data.min().min()),
        float(u_data.max().max()),
    )
    return u_data


def fit_vine_model(u_data: pd.DataFrame) -> pv.Vinecop:
    logger.info("开始拟合 Vine Copula 模型")
    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.gumbel,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gaussian,
        ]
    )
    vine = pv.Vinecop.from_data(u_data.to_numpy(), controls=controls)
    logger.info("Vine Copula 拟合完成, dim=%d", vine.dim)
    logger.debug("Vine 结构: %s", str(vine))
    return vine


def build_tau_matrix(vine: pv.Vinecop, columns: pd.Index) -> pd.DataFrame:
    logger.info("开始计算 Kendall's tau 矩阵")
    # 当前 pyvinecopulib 版本没有 matrix_tau()，改为从拟合 Vine 模型采样后估计 Kendall 相关
    simulated_u = vine.simulate(10000)
    tau_df = pd.DataFrame(simulated_u, columns=columns).corr(method="kendall")
    logger.info("Kendall's tau 矩阵计算完成: shape=%s", tau_df.shape)
    return tau_df


def reorder_matrix_odd_even(tau_df: pd.DataFrame) -> pd.DataFrame:
    '''将矩阵按原顺序的奇数位行列排在前，偶数位行列排在后（1-based）'''
    logger.info("开始重排行列: 奇数位在前，偶数位在后")

    total = len(tau_df.columns)
    odd_positions = list(range(0, total, 2))
    even_positions = list(range(1, total, 2))
    reordered_cols = tau_df.columns[odd_positions].tolist() + tau_df.columns[even_positions].tolist()

    tau_reordered = tau_df.loc[reordered_cols, reordered_cols]
    logger.info("矩阵重排完成")
    logger.debug("重排后的列顺序: %s", reordered_cols)
    return tau_reordered


def save_tau_heatmap(tau_df: pd.DataFrame, fig_path: str) -> None:
    logger.info("开始绘制热力图")
    plt.figure(figsize=(12, 10))
    sns.heatmap(tau_df, annot=False, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    # plt.title("Kendall's Tau Matrix Between Zones (Extreme Event Synchrony)")

    fig_parent = Path(fig_path).parent
    fig_parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    logger.info("热力图已保存: %s", fig_path)


def save_tau_matrix(tau_df: pd.DataFrame, output_csv_path: str) -> None:
    logger.info("开始保存 Kendall's tau 矩阵")
    output_parent = Path(output_csv_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    tau_df.to_csv(output_csv_path, index=True, encoding="utf-8-sig", float_format="%.3f")
    logger.info("Kendall's tau 矩阵已保存: %s", output_csv_path)


def run_pipeline(input_csv_path: str, output_fig_path: str, output_tau_csv_path: str) -> None:
    df = load_input_data(input_csv_path)
    u_data = pit_transform(df)
    vine = fit_vine_model(u_data)

    print("--- Vine Copula 建模完成 ---")
    print(f"拟合的藤结构层数: {vine.dim}")
    print(str(vine))

    tau_df = build_tau_matrix(vine, df.columns)
    tau_df = reorder_matrix_odd_even(tau_df)
    save_tau_matrix(tau_df, output_tau_csv_path)
    save_tau_heatmap(tau_df, output_fig_path)

    print("部分相关性数据已计算。")
    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    input_csv_path = "G:/extreme_analysis/results/Clustering/kmeans_Vine_Copula_Input_Data.csv"
    output_fig_path = "G:/extreme_analysis/results/Clustering/Vine_Copula_Correlation_Matrix.png"
    output_tau_csv_path = "G:/extreme_analysis/results/Clustering/Vine_Copula_Kendall_Tau_Matrix.csv"

    logger.info("脚本启动")
    logger.info("输入文件路径: %s", input_csv_path)
    logger.info("输出图片路径: %s", output_fig_path)
    logger.info("输出矩阵路径: %s", output_tau_csv_path)
    run_pipeline(
        input_csv_path=input_csv_path,
        output_fig_path=output_fig_path,
        output_tau_csv_path=output_tau_csv_path,
    )
    logger.info("脚本结束")
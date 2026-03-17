#!/usr/bin/env python3
"""
合成数据生成器

用于基准测试的合成数据生成，避免依赖外部数据集。

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
from typing import Tuple, Optional


def generate_precipitation_field(
    shape: Tuple[int, int],
    correlation_length: float = 10.0,
    intensity: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成合成的降水场

    使用高斯随机场生成空间相关的降水数据。

    Parameters
    ----------
    shape : tuple
        场的形状 (height, width)
    correlation_length : float
        相关长度（像素）
    intensity : float
        平均强度
    seed : int, optional
        随机种子

    Returns
    -------
    field : ndarray
        生成的降水场
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = shape

    # 生成高斯随机场
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    # 简化的空间相关模型
    field = np.random.randn(height, width)

    # 应用平滑
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(field, sigma=correlation_length / 2.0)

    # 调整强度
    field = field * intensity

    # 确保非负
    field = np.maximum(field, 0)

    return field


def generate_ensemble_forecast(
    n_members: int,
    shape: Tuple[int, int],
    spread: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成集合预报数据

    Parameters
    ----------
    n_members : int
        集合成员数
    shape : tuple
        每个成员的形状
    spread : float
        集合离散度
    seed : int, optional
        随机种子

    Returns
    -------
    ensemble : ndarray
        集合预报数据 (n_members, height, width)
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成均值场
    mean_field = generate_precipitation_field(shape, seed=seed)

    # 生成集合扰动
    ensemble = []
    for i in range(n_members):
        perturbation = np.random.randn(*shape) * spread
        member = mean_field + perturbation
        member = np.maximum(member, 0)  # 非负
        ensemble.append(member)

    return np.array(ensemble)


def generate_time_series(
    length: int,
    ar_order: int = 2,
    noise_std: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成AR时间序列

    Parameters
    ----------
    length : int
        时间序列长度
    ar_order : int
        AR阶数
    noise_std : float
        噪声标准差
    seed : int, optional
        随机种子

    Returns
    -------
    series : ndarray
        生成的时间序列
    """
    if seed is not None:
        np.random.seed(seed)

    # AR系数
    phi = np.random.randn(ar_order) * 0.5

    # 生成时间序列
    series = np.zeros(length)
    for t in range(ar_order, length):
        series[t] = np.sum(phi * series[t-ar_order:t][::-1])
        series[t] += np.random.randn() * noise_std

    return series


def generate_radar_composite(
    shape: Tuple[int, int] = (512, 512),
    n_levels: int = 8,
    min_dbz: float = 5.0,
    max_dbz: float = 55.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成合成雷达复合图

    Parameters
    ----------
    shape : tuple
        图像形状
    n_levels : int
        离散级别数
    min_dbz : float
        最小反射率 (dBZ)
    max_dbz : float
        最大反射率 (dBZ)
    seed : int, optional
        随机种子

    Returns
    -------
    composite : ndarray
        雷达复合图 (dBZ)
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成连续场
    field = generate_precipitation_field(
        shape,
        correlation_length=20.0,
        intensity=1.0,
        seed=seed
    )

    # 转换为 dBZ
    field_dbz = min_dbz + (max_dbz - min_dbz) * field

    # 离散化
    levels = np.linspace(min_dbz, max_dbz, n_levels)
    digitized = np.digitize(field_dbz, levels)

    return field_dbz


# 预定义的测试数据集
TEST_DATASETS = {
    'small': {
        'ensemble': (20, 100, 100),
        'radar': (256, 256),
        'timeseries': 100,
    },
    'medium': {
        'ensemble': (50, 256, 256),
        'radar': (512, 512),
        'timeseries': 500,
    },
    'large': {
        'ensemble': (100, 512, 512),
        'radar': (1024, 1024),
        'timeseries': 1000,
    },
}


def get_test_data(size: str = 'medium', seed: int = 42):
    """
    获取测试数据集

    Parameters
    ----------
    size : str
        数据集大小 ('small', 'medium', 'large')
    seed : int
        随机种子

    Returns
    -------
    data : dict
        包含各种测试数据的字典
    """
    if size not in TEST_DATASETS:
        raise ValueError(f"未知数据集大小: {size}")

    config = TEST_DATASETS[size]

    data = {}

    # 集合预报数据
    n_members, height, width = config['ensemble']
    data['ensemble_forecast'] = generate_ensemble_forecast(
        n_members,
        (height, width),
        seed=seed
    )

    # 雷达数据
    radar_shape = config['radar']
    data['radar_composite'] = generate_radar_composite(
        radar_shape,
        seed=seed + 1
    )

    # 时间序列
    ts_length = config['timeseries']
    data['timeseries'] = generate_time_series(
        ts_length,
        seed=seed + 2
    )

    return data


if __name__ == '__main__':
    # 测试数据生成
    print("生成测试数据集...")

    for size in ['small', 'medium', 'large']:
        print(f"\n{size.upper()} 数据集:")
        data = get_test_data(size)

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}, {value.nbytes / 1024:.1f} KB")

    print("\n✓ 数据生成完成")

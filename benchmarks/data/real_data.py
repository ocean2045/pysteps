#!/usr/bin/env python3
"""
真实数据接口

连接到 pysteps-data 仓库的真实数据。

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings


# PySteps 数据仓库路径
PYSTEPS_DATA_DIR = Path('/data/workspace/pysteps-data')


def check_pysteps_data() -> bool:
    """
    检查 pysteps-data 是否可用

    Returns
    -------
    available : bool
        True 如果数据可用
    """
    if not PYSTEPS_DATA_DIR.exists():
        return False

    # 检查是否有数据文件
    data_files = list(PYSTEPS_DATA_DIR.glob('*.npz')) + \
                  list(PYSTEPS_DATA_DIR.glob('*.h5')) + \
                  list(PYSTEPS_DATA_DIR.glob('*.hdf5'))

    return len(data_files) > 0


def load_radar_composite(
    date: str,
    time: int,
    source: str = 'fmi'
) -> Optional[np.ndarray]:
    """
    加载雷达复合图

    Parameters
    ----------
    date : str
        日期 (YYYYMMDD)
    time : int
        时间 (HHMM)
    source : str
        数据源 ('fmi', 'mch', etc.)

    Returns
    -------
    radar_data : ndarray or None
        雷达复合图，如果不可用返回 None
    """
    if not check_pysteps_data():
        warnings.warn("pysteps-data 不可用，使用合成数据")
        return None

    # 这里会实现实际的加载逻辑
    # 目前返回 None，表示使用合成数据
    warnings.warn(f"真实数据加载未实现: {source}/{date}/{time}")
    return None


def get_available_dates(source: str = 'fmi') -> list:
    """
    获取可用的日期列表

    Parameters
    ----------
    source : str
        数据源

    Returns
    -------
    dates : list
        可用的日期列表
    """
    if not check_pysteps_data():
        return []

    # 这里会实现实际的日期列表获取
    return []


# 推荐的测试用例（从 pysteps-data 文档）
RECOMMENDED_TEST_CASES = [
    {
        'name': 'fmi_small',
        'source': 'fmi',
        'date': '20160928',
        'time_range': (0, 10),  # 前10个时间步
        'description': '芬兰气象 institute 小数据集',
    },
    {
        'name': 'mch_medium',
        'source': 'mch',
        'date': '20170601',
        'time_range': (0, 20),
        'description': 'MeteoSwiss 中等数据集',
    },
]


def load_test_case(case_name: str) -> Optional[dict]:
    """
    加载推荐的测试用例

    Parameters
    ----------
    case_name : str
        测试用例名称

    Returns
    -------
    case_data : dict or None
        测试用例数据，如果不可用返回 None
    """
    for case in RECOMMENDED_TEST_CASES:
        if case['name'] == case_name:
            if check_pysteps_data():
                # 加载真实数据
                # 实际实现会在这里
                pass
            else:
                warnings.warn(f"pysteps-data 不可用: {case_name}")
                return None

    return None


def get_data_status() -> dict:
    """
    获取数据状态信息

    Returns
    -------
    status : dict
        包含数据可用性信息的字典
    """
    status = {
        'pysteps_data_available': check_pysteps_data(),
        'pysteps_data_dir': str(PYSTEPS_DATA_DIR),
        'recommended_test_cases': len(RECOMMENDED_TEST_CASES),
        'fallback_to_synthetic': True,
    }

    return status


if __name__ == '__main__':
    print("PySteps 真实数据接口")
    print("="*60)

    status = get_data_status()

    print(f"\n数据状态:")
    print(f"  pysteps-data 可用: {status['pysteps_data_available']}")
    print(f"  数据目录: {status['pysteps_data_dir']}")
    print(f"  推荐测试用例: {status['recommended_test_cases']}")
    print(f"  回退到合成数据: {status['fallback_to_synthetic']}")

    if status['pysteps_data_available']:
        print("\n推荐的测试用例:")
        for case in RECOMMENDED_TEST_CASES:
            print(f"  - {case['name']}: {case['description']}")
    else:
        print("\n提示: pysteps-data 不可用，基准测试将使用合成数据")
        print("      合成数据足以验证优化效果。")

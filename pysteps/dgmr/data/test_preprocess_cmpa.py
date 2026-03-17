"""
快速测试脚本 - 验证CMPA数据预处理流程
只处理少量数据用于验证
"""

import os
import sys
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 测试配置
TEST_CONFIG = {
    'data_root': '/data/data/CMPAS_P05_10MIN',
    'output_root': '/data/workspace/PyStepsDashu/data/dgmr_training/test',

    # 测试：只处理2天的数据
    'test_days': 2,

    # 模型参数
    'input_frames': 12,      # 2小时历史
    'output_frames': 18,     # 3小时预报
    'total_frames': 30,      # 总帧数

    # 数据处理参数
    'precip_threshold': 0.1,
    'max_precip': 100.0,
    'min_precip_fraction': 0.03,
    'precip_event_threshold': 0.5,
}


def test_read_grb2(file_path):
    """测试读取GRB2文件"""
    try:
        ds = xr.open_dataset(file_path, engine='cfgrib')

        if 'unknown' in ds:
            data = ds['unknown'].values
        else:
            return None

        ds.close()

        # 处理无效值
        data = np.where(data >= 9998, 0, data)

        # 确保二维
        if data.ndim == 3:
            data = data[0]

        # 裁剪异常值
        data = np.clip(data, 0, 100)

        return data

    except Exception as e:
        print(f"读取失败: {e}")
        return None


def test_preprocess():
    """测试预处理流程"""
    print("="*70)
    print("CMPA数据预处理测试")
    print("="*70)

    # 1. 获取测试数据目录
    print("\n1. 获取测试数据...")
    data_root = Path(TEST_CONFIG['data_root'])

    # 获取2023年5月的前几天
    may_dirs = sorted([d for d in data_root.glob("202305*") if d.is_dir()])[:TEST_CONFIG['test_days']]

    print(f"选择 {len(may_dirs)} 个测试目录:")
    for d in may_dirs:
        print(f"  - {d.name}")

    # 2. 处理每一天
    all_samples = []
    total_frames = TEST_CONFIG['total_frames']

    for date_dir in tqdm(may_dirs, desc="处理日期"):
        # 获取所有GRB2文件并排序
        grb2_files = sorted(list(date_dir.glob("*.GRB2")))

        print(f"\n  {date_dir.name}: {len(grb2_files)} 个文件")

        if len(grb2_files) < total_frames:
            print(f"  跳过 (文件不足 {total_frames} 个)")
            continue

        # 读取数据
        daily_data = []
        for file_path in tqdm(grb2_files, desc="  读取文件", leave=False):
            data = test_read_grb2(str(file_path))
            if data is not None:
                daily_data.append(data)

        print(f"  成功读取: {len(daily_data)} 个文件")

        # 检查是否有足够的帧
        if len(daily_data) < total_frames:
            print(f"  跳过 (有效数据不足 {total_frames} 帧)")
            continue

        # 创建样本序列
        print(f"  创建样本序列...")
        for i in range(len(daily_data) - total_frames + 1):
            # 提取序列
            sequence = daily_data[i:i + total_frames]
            sequence = np.stack(sequence, axis=0)

            # 检查输出时段是否有降水
            output_data = sequence[TEST_CONFIG['input_frames']:]
            has_output_precip = (output_data > TEST_CONFIG['precip_threshold']).any()

            if has_output_precip:
                all_samples.append(sequence)

        print(f"  创建样本数: {len(all_samples)}")

    # 3. 统计信息
    print("\n" + "="*70)
    print("测试统计:")
    print("="*70)
    print(f"总样本数: {len(all_samples)}")

    if all_samples:
        # 数据统计
        all_data = np.concatenate([s.flatten() for s in all_samples])

        print(f"\n数据统计:")
        print(f"  总像素: {all_data.size:,}")
        print(f"  有效降水像素: {np.sum(all_data > TEST_CONFIG['precip_threshold']):,}")
        print(f"  降水覆盖比例: {np.sum(all_data > TEST_CONFIG['precip_threshold']) / all_data.size * 100:.2f}%")
        print(f"  值范围: [{all_data.min():.2f}, {all_data.max():.2f}]")
        print(f"  平均值: {all_data[all_data > TEST_CONFIG['precip_threshold']].mean():.2f}")
        print(f"  中位数: {np.median(all_data[all_data > TEST_CONFIG['precip_threshold']]):.2f}")

        # 样本形状
        print(f"\n样本形状:")
        print(f"  单个样本: {all_samples[0].shape}")
        print(f"  解释: [总帧数, 纬度, 经度]")
        print(f"  输入帧: 0-{TEST_CONFIG['input_frames']} (2小时历史)")
        print(f"  输出帧: {TEST_CONFIG['input_frames']}-{total_frames} (3小时预报)")

        print(f"\n数据质量:")
        has_nan = any(np.isnan(s).any() for s in all_samples)
        has_inf = any(np.isinf(s).any() for s in all_samples)
        has_negative = any((s < 0).any() for s in all_samples)

        print(f"  NaN: {'是' if has_nan else '否'}")
        print(f"  Inf: {'是' if has_inf else '否'}")
        print(f"  负值: {'是' if has_negative else '否'}")

        if not (has_nan or has_inf or has_negative):
            print(f"  ✓ 数据质量良好")

    # 4. 测试保存
    print(f"\n测试保存功能...")
    output_dir = Path(TEST_CONFIG['output_root'])
    output_dir.mkdir(parents=True, exist_ok=True)

    import h5py

    test_file = output_dir / "test_sample.h5"

    # 保存前10个样本
    samples_to_save = all_samples[:10]

    with h5py.File(test_file, 'w') as f:
        data_shape = (len(samples_to_save),) + samples_to_save[0].shape

        dset = f.create_dataset(
            'precipitation',
            shape=data_shape,
            dtype='float32',
            compression='gzip'
        )

        for i, sample in enumerate(samples_to_save):
            dset[i] = sample.astype('float32')

        # 保存元数据
        for key, value in TEST_CONFIG.items():
            if not isinstance(value, (str, Path)):
                f.attrs[key] = value

    file_size_mb = test_file.stat().st_size / (1024 * 1024)
    print(f"  保存: {test_file}")
    print(f"  大小: {file_size_mb:.2f} MB")
    print(f"  样本数: {len(samples_to_save)}")

    # 验证读取
    print(f"\n验证读取...")
    with h5py.File(test_file, 'r') as f:
        loaded_data = f['precipitation'][:]
        print(f"  读取数据形状: {loaded_data.shape}")
        print(f"  数据类型: {loaded_data.dtype}")
        print(f"  ✓ 保存/读取成功")

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

    # 估算完整数据集大小
    if len(all_samples) > 0:
        # 假设5-9月，每月30天，共153天
        estimated_total_days = 153
        # 假设70%的日子有降水
        estimated_precip_days = int(estimated_total_days * 0.7)
        # 假设每天可以创建40个样本（滑动窗口）
        samples_per_day = 40
        estimated_total_samples = estimated_precip_days * samples_per_day

        sample_size_mb = file_size_mb / len(samples_to_save)
        estimated_total_size_gb = (estimated_total_samples * sample_size_mb) / 1024

        print(f"\n完整数据集估算:")
        print(f"  预计总样本数: {estimated_total_samples:,}")
        print(f"  单个样本大小: {sample_size_mb:.2f} MB")
        print(f"  预计总大小: {estimated_total_size_gb:.2f} GB")


if __name__ == "__main__":
    test_preprocess()

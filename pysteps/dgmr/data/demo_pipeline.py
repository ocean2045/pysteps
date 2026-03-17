"""
快速开始示例 - 完整流程演示

演示如何使用灵活数据流水线：
1. 保存年度数据到HDF
2. 使用配置文件构建样本
3. 加载样本用于训练
"""

import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_save_to_hdf():
    """演示：保存年度数据到HDF"""
    print("="*70)
    print("演示1: 保存年度数据到HDF")
    print("="*70)

    print("\n运行命令:")
    print("  python pysteps/dgmr/data/save_to_hdf.py")

    print("\n预期输出:")
    print("  data/cmpa_h5/")
    print("    ├── 2023.h5")
    print("    └── 2024.h5")

    print("\n文件内容:")
    print("  - precipitation: [time, lat, lon]")
    print("  - time: [time]")
    print("  - latitude: [lat]")
    print("  - longitude: [lon]")


def demo_create_config():
    """演示：创建配置文件"""
    print("\n" + "="*70)
    print("演示2: 创建JSON配置文件")
    print("="*70)

    # 推荐配置
    config = {
        "input_frames": 12,
        "output_frames": 18,
        "description": "12帧输入（2小时历史）+ 18帧输出（3小时预报）",
        "oversample_heavy": True,
        "heavy_precipitation_threshold": 5.0,
        "oversample_ratio": 3.0
    }

    config_file = "my_config.json"

    print(f"\n创建配置文件: {config_file}")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n配置内容:")
    print(json.dumps(config, indent=2))

    print(f"\n✓ 配置已保存: {config_file}")

    return config_file


def demo_build_samples(config_file):
    """演示：构建样本"""
    print("\n" + "="*70)
    print("演示3: 使用配置文件构建样本")
    print("="*70)

    hdf_file = "data/cmpa_h5/2023.h5"  # 假设已存在
    output_dir = "samples_output"

    print(f"\n运行命令:")
    print(f"  python pysteps/dgmr/data/build_samples.py \\")
    print(f"    {config_file} \\")
    print(f"    {hdf_file} \\")
    print(f"    {output_dir}")

    print(f"\n预期输出:")
    print(f"  {output_dir}/")
    print(f"    ├── samples.h5          # 样本数据")
    print(f"    ├── sample_stats.json   # 统计信息")
    print(f"    └── statistics.txt      # 统计摘要")


def demo_load_samples():
    """演示：加载和使用样本"""
    print("\n" + "="*70)
    print("演示4: 加载样本用于训练")
    print("="*70)

    sample_file = "samples_output/samples.h5"
    stats_file = "samples_output/sample_stats.json"

    print(f"\n加载样本...")
    with h5py.File(sample_file, 'r') as f:
        samples = f['samples'][:]

    print(f"  样本数: {samples.shape[0]}")
    print(f"  样本形状: {samples.shape}")
    print(f"  数据类型: {samples.dtype}")

    print(f"\n加载统计...")
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # 分析强度分布
    intensity_counts = {}
    for s in stats['samples']:
        intensity = s['intensity_class']
        intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1

    print(f"\n强度等级分布:")
    for intensity, count in intensity_counts.items():
        pct = count / len(stats['samples']) * 100
        print(f"  {intensity.capitalize():10s}: {count:6d} ({pct:5.2f}%)")

    print(f"\n数据形状:")
    input_frames = 12
    output_frames = 18

    print(f"  输入数据: samples[:, :{input_frames}, :, :]")
    print(f"  输出数据: samples[:, {input_frames}:{input_frames+output_frames}, :, :]")

    return samples, stats


def demo_training(samples):
    """演示：训练流程"""
    print("\n" + "="*70)
    print("演示5: 训练Improved DGMR")
    print("="*70)

    print(f"\n准备训练数据...")
    print(f"  总样本数: {samples.shape[0]}")
    print(f"  训练集: {int(samples.shape[0] * 0.7)} 样本")
    print(f"  验证集: {int(samples.shape[0] * 0.15)} 样本")
    print(f"  测试集: {int(samples.shape[0] * 0.15)} 样本")

    print(f"\n训练命令:")
    print(f"  python pysteps/dgmr/training/trainer.py \\")
    print(f"    --config config_12_18.json \\")
    print(f"    --data_path samples_output \\")
    print(f"    --batch_size 4 \\")
    print(f"    --max_epochs 100")


def main():
    """运行完整演示"""
    print("\n" + "="*70)
    print("CMPA灵活数据流水线 - 完整演示")
    print("="*70)
    print("\n本演示将展示:")
    print("  1. 保存年度数据到HDF")
    print("  2. 创建JSON配置文件")
    print("  3. 构建训练样本")
    print("  4. 加载样本数据")
    print("  5. 训练模型")

    input("\n按Enter继续...")

    # 演示1
    demo_save_to_hdf()
    input("\n按Enter继续...")

    # 演示2
    config_file = demo_create_config()
    input("\n按Enter继续...")

    # 演示3
    demo_build_samples(config_file)
    input("\n按Enter继续...")

    # 演示4
    # samples, stats = demo_load_samples()
    input("\n按Enter继续...")

    # 演示5
    # demo_training(samples)
    input("\n按Enter继续...")

    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)

    print("\n实际使用时:")
    print("  1. 运行: python pysteps/dgmr/data/save_to_hdf.py")
    print("  2. 创建: 编辑 config_*.json 文件")
    print("  3. 构建: python pysteps/dgmr/data/build_samples.py <config> <hdf> <output>")
    print("  4. 训练: python pysteps/dgmr/training/trainer.py ...")


if __name__ == "__main__":
    main()

"""
快速验证脚本 - 最小化测试
只读取少量文件来验证数据格式
"""

import numpy as np
import xarray as xr
from pathlib import Path

print("="*60)
print("CMPA数据快速验证")
print("="*60)

# 测试读取单个文件
file_path = "/data/data/CMPAS_P05_10MIN/20230501/Z_SURF_C_BABJ_20230501002754_P_CMPA_FAST_CHN_0P05_10MIN-PRE-202305010000.GRB2"

print(f"\n读取文件: {file_path}")

ds = xr.open_dataset(file_path, engine='cfgrib')
data = ds['unknown'].values
ds.close()

print(f"原始数据形状: {data.shape}")
print(f"原始数据类型: {data.dtype}")
print(f"原始值范围: [{data.min():.2f}, {data.max():.2f}]")
print(f"原始值统计:")
print(f"  = 9999: {(data == 9999).sum()}")
print(f"  > 100: {((data > 100) & (data < 9999)).sum()}")
print(f"  > 0: {(data > 0).sum()}")

# 处理数据
data_cleaned = np.where(data >= 9998, 0, data)
data_cleaned = np.clip(data_cleaned, 0, 100)

print(f"\n清洗后:")
print(f"  值范围: [{data_cleaned.min():.2f}, {data_cleaned.max():.2f}]")
print(f"  有效降水: {(data_cleaned > 0.1).sum()} ({(data_cleaned > 0.1).sum() / data_cleaned.size * 100:.2f}%)")

# 测试序列构建
print(f"\n测试序列构建:")
print(f"  输入帧数: 12 (2小时)")
print(f"  输出帧数: 18 (3小时)")
print(f"  总帧数: 30")

# 获取该目录下的所有文件
data_dir = Path("/data/data/CMPAS_P05_10MIN/20230501")
grb2_files = sorted(list(data_dir.glob("*.GRB2")))

print(f"\n该目录文件数: {len(grb2_files)}")
print(f"需要文件数: 30")
print(f"是否足够: {'是' if len(grb2_files) >= 30 else '否'}")

if len(grb2_files) >= 30:
    print(f"\n✓ 数据充足，可以创建训练样本")

    # 估算完整数据集
    total_days = 306  # 2023-2024年5-9月，约306天
    files_per_day = 144
    precip_days_ratio = 0.7  # 70%的日子有降水

    total_files = total_days * files_per_day
    precip_days = int(total_days * precip_days_ratio)
    samples_per_day = 100  # 保守估计

    estimated_samples = precip_days * samples_per_day

    single_sample_size_mb = (30 * 1201 * 1401 * 4) / (1024 * 1024)  # float32

    print(f"\n完整数据集估算:")
    print(f"  总天数: {total_days}")
    print(f"  有降水天数(估计): {precip_days}")
    print(f"  预计样本数: {estimated_samples:,}")
    print(f"  单样本大小: {single_sample_size_mb:.2f} MB")
    print(f"  预计总大小: {estimated_samples * single_sample_size_mb / 1024:.2f} GB")

    print(f"\n数据集划分 (70%/15%/15%):")
    print(f"  训练集: {int(estimated_samples * 0.7):,} 样本")
    print(f"  验证集: {int(estimated_samples * 0.15):,} 样本")
    print(f"  测试集: {int(estimated_samples * 0.15):,} 样本")

print("\n" + "="*60)
print("验证完成！")
print("="*60)

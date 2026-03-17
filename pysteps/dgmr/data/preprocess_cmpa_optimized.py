"""
优化版CMPA降水数据预处理

优化策略：
1. 裁剪到中国核心区域（减少无效边界）
2. 降采样空间分辨率（0.05° → 0.1°）
3. 只选取有明显降水过程的日子
4. 使用float16减少内存占用
5. 高效的HDF5压缩

预期数据集大小: 约200-300GB
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
import h5py
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# 优化配置
CONFIG = {
    # 数据路径
    'data_root': '/data/data/CMPAS_P05_10MIN',
    'output_root': '/data/workspace/PyStepsDashu/data/dgmr_training',

    # 时间范围
    'years': [2023, 2024],
    'months': [5, 6, 7, 8, 9],

    # 模型参数
    'input_frames': 12,
    'output_frames': 18,
    'total_frames': 30,

    # 空间优化
    'spatial_stride': 2,      # 空间降采样因子 (0.05° → 0.1°)
    'crop_lat': (100, 500),   # 裁剪纬度范围 (中国核心区域)
    'crop_lon': (700, 1300),  # 裁剪经度范围

    # 数据处理
    'precip_threshold': 0.1,
    'max_precip': 100.0,
    'min_precip_fraction': 0.05,
    'min_precip_pixels': 500,  # 最小降水像素数

    # 输出优化
    'dtype': 'float16',            # 使用float16节省空间
    'samples_per_file': 200,
    'compression': 'lzf',          # 快速压缩

    # 降水事件筛选（只选取有明显降水的日子）
    'min_daily_precip_fraction': 0.01,  # 至少1%区域有降水
    'min_max_precip': 1.0,               # 至少有1mm/h的降水
}


def print_stats(data, label="Data"):
    """打印数据统计信息"""
    print(f"\n{label}:")
    print(f"  形状: {data.shape}")
    print(f"  类型: {data.dtype}")
    print(f"  大小: {data.nbytes / (1024**2):.2f} MB")
    if data.size > 0:
        print(f"  范围: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  有效降水: {(data > CONFIG['precip_threshold']).sum()} ({(data > CONFIG['precip_threshold']).sum() / data.size * 100:.2f}%)")


class OptimizedCMPAPreprocessor:
    """优化的CMPA数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.data_root = Path(config['data_root'])
        self.output_root = Path(config['output_root'])

        # 创建输出目录
        for split in ['train', 'val', 'test']:
            (self.output_root / split).mkdir(parents=True, exist_ok=True)
        (self.output_root / 'info').mkdir(parents=True, exist_ok=True)

        # 统计
        self.stats = {
            'total_days': 0,
            'processed_days': 0,
            'total_samples': 0,
            'removed_outliers': 0,
        }

    def read_and_process_file(self, file_path: str) -> Optional[np.ndarray]:
        """读取并处理单个GRB2文件"""
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib')

            if 'unknown' not in ds:
                ds.close()
                return None

            data = ds['unknown'].values
            ds.close()

            # 处理无效值
            data = np.where(data >= 9998, 0, data)

            # 裁剪到感兴趣区域
            if 'crop_lat' in self.config and 'crop_lon' in self.config:
                lat_slice = slice(*self.config['crop_lat'])
                lon_slice = slice(*self.config['crop_lon'])
                data = data[lat_slice, lon_slice]

            # 降采样
            if self.config['spatial_stride'] > 1:
                data = data[::self.config['spatial_stride'], ::self.config['spatial_stride']]

            # 裁剪异常值
            data = np.clip(data, 0, self.config['max_precip'])

            return data

        except Exception as e:
            warnings.warn(f"读取失败 {file_path}: {e}")
            return None

    def has_significant_precipitation(self, daily_data: List[np.ndarray]) -> bool:
        """检查是否有显著降水"""
        if not daily_data:
            return False

        # 合并一天的数据
        daily_stack = np.stack(daily_data, axis=0)

        # 检查1: 最大降水
        max_precip = daily_stack.max()
        if max_precip < self.config['min_max_precip']:
            return False

        # 检查2: 降水覆盖面积
        precip_pixels = (daily_stack > self.config['precip_threshold']).sum()
        precip_fraction = precip_pixels / daily_stack.size

        if precip_fraction < self.config['min_daily_precip_fraction']:
            return False

        return True

    def create_sequences(self, daily_data: List[np.ndarray]) -> List[np.ndarray]:
        """创建训练序列"""
        sequences = []
        total_frames = self.config['total_frames']

        if len(daily_data) < total_frames:
            return sequences

        # 滑动窗口
        for i in range(len(daily_data) - total_frames + 1):
            seq = daily_data[i:i + total_frames]
            seq = np.stack(seq, axis=0)

            # 检查输出时段是否有降水
            output_data = seq[self.config['input_frames']:]
            if (output_data > self.config['precip_threshold']).any():
                sequences.append(seq)

        return sequences

    def save_batch(self, samples: List[np.ndarray], split: str, batch_idx: int):
        """保存一批样本"""
        if not samples:
            return

        output_file = self.output_root / split / f"precip_{split}_{batch_idx:04d}.h5"

        # 转换为目标类型
        dtype = np.float16 if self.config['dtype'] == 'float16' else np.float32

        with h5py.File(output_file, 'w') as f:
            shape = (len(samples),) + samples[0].shape

            # 创建压缩数据集
            dset = f.create_dataset(
                'precipitation',
                shape=shape,
                dtype=dtype,
                compression=self.config['compression'],
                chunks=(1, shape[1], shape[2])
            )

            # 写入数据
            for i, sample in enumerate(samples):
                dset[i] = sample.astype(dtype)

            # 元数据
            f.attrs['input_frames'] = self.config['input_frames']
            f.attrs['output_frames'] = self.config['output_frames']
            f.attrs['total_frames'] = self.config['total_frames']
            f.attrs['num_samples'] = len(samples)
            f.attrs['spatial_stride'] = self.config['spatial_stride']
            f.attrs['dtype'] = self.config['dtype']

        return len(samples)

    def process_year(self, year: int):
        """处理一年的数据"""
        print(f"\n{'='*60}")
        print(f"处理 {year} 年数据")
        print(f"{'='*60}")

        all_samples = []
        all_dates = []

        for month in self.config['months']:
            month_str = f"{year}{month:02d}"
            month_dir = self.data_root / month_str

            if not month_dir.exists():
                continue

            # 获取该月的日期目录
            date_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()])

            print(f"\n{year}-{month:02d}: {len(date_dirs)} 天")

            for date_dir in tqdm(date_dirs, desc=f"  处理", leave=False):
                # 获取文件
                grb2_files = sorted(list(date_dir.glob("*.GRB2")))

                if len(grb2_files) < self.config['total_frames']:
                    continue

                # 读取数据
                daily_data = []
                for file_path in grb2_files:
                    data = self.read_and_process_file(str(file_path))
                    if data is not None:
                        daily_data.append(data)

                # 检查数据完整性
                if len(daily_data) < self.config['total_frames']:
                    continue

                # 检查是否有显著降水
                if not self.has_significant_precipitation(daily_data):
                    continue

                # 创建序列
                sequences = self.create_sequences(daily_data)

                if sequences:
                    all_samples.extend(sequences)
                    date_str = date_dir.name
                    all_dates.extend([datetime.strptime(date_str, "%Y%m%d")] * len(sequences))

                    self.stats['processed_days'] += 1

            # 释放内存
            gc.collect()

        return all_samples, all_dates

    def split_and_save(self, all_samples: List[np.ndarray], all_dates: List[datetime]):
        """划分数据集并保存"""
        if not all_samples:
            print("没有生成任何样本")
            return

        print(f"\n{'='*60}")
        print(f"数据集划分和保存")
        print(f"{'='*60}")

        # 按日期排序
        sorted_idx = np.argsort(all_dates)
        sorted_samples = [all_samples[i] for i in sorted_idx]

        # 划分
        n_total = len(sorted_samples)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        train = sorted_samples[:n_train]
        val = sorted_samples[n_train:n_train + n_val]
        test = sorted_samples[n_train + n_val:]

        print(f"训练集: {len(train)} 样本")
        print(f"验证集: {len(val)} 样本")
        print(f"测试集: {len(test)} 样本")

        # 保存
        self.stats['total_samples'] = n_total

        for split, samples in [('train', train), ('val', val), ('test', test)]:
            print(f"\n保存 {split} 集...")
            batch_size = self.config['samples_per_file']

            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                batch_idx = i // batch_size
                count = self.save_batch(batch, split, batch_idx)

                if (batch_idx + 1) % 10 == 0:
                    print(f"  已保存 {(batch_idx + 1) * batch_size} 样本")

        # 保存统计信息
        self.save_info()

    def save_info(self):
        """保存数据集信息"""
        info_path = self.output_root / 'info' / 'dataset_info.txt'

        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CMPA降水数据集 - Improved DGMR训练\n")
            f.write("="*60 + "\n\n")

            f.write("数据范围:\n")
            f.write(f"  年份: {self.config['years']}\n")
            f.write(f"  月份: {self.config['months']}\n")
            f.write(f"  处理天数: {self.stats['processed_days']}\n\n")

            f.write("模型参数:\n")
            f.write(f"  输入帧数: {self.config['input_frames']} (2小时)\n")
            f.write(f"  输出帧数: {self.config['output_frames']} (3小时)\n")
            f.write(f"  总帧数: {self.config['total_frames']}\n\n")

            f.write("空间处理:\n")
            f.write(f"  空间降采样: {self.config['spatial_stride']}x (0.05° → 0.1°)\n")
            f.write(f"  数据类型: {self.config['dtype']}\n\n")

            f.write("数据集统计:\n")
            f.write(f"  总样本数: {self.stats['total_samples']:,}\n\n")

            # 估算文件大小
            sample_shape = self.get_sample_shape()
            single_size_mb = (self.config['total_frames'] * sample_shape[0] *
                             sample_shape[1] * 2) / (1024**2)  # float16
            total_size_gb = (self.stats['total_samples'] * single_size_mb) / 1024

            f.write(f"存储信息:\n")
            f.write(f"  单样本大小: {single_size_mb:.2f} MB\n")
            f.write(f"  预计总大小: {total_size_gb:.2f} GB\n")
            f.write(f"  输出目录: {self.output_root}\n")

        print(f"\n数据集信息已保存到: {info_path}")

    def get_sample_shape(self):
        """获取样本形状（估算）"""
        # 读取一个样本获取形状
        test_file = self.data_root / "20230501" / "*.GRB2"
        files = list(self.data_root.glob("20230501/*.GRB2"))

        if files:
            data = self.read_and_process_file(str(files[0]))
            if data is not None:
                return data.shape

        # 默认估算
        orig_h = self.config['crop_lat'][1] - self.config['crop_lat'][0]
        orig_w = self.config['crop_lon'][1] - self.config['crop_lon'][0]
        h = orig_h // self.config['spatial_stride']
        w = orig_w // self.config['spatial_stride']
        return (h, w)

    def process(self):
        """执行完整的预处理流程"""
        print("="*70)
        print("CMPA降水数据预处理 - 优化版")
        print("="*70)

        print("\n优化策略:")
        print(f"  1. 空间降采样: {self.config['spatial_stride']}x")
        print(f"  2. 数据类型: {self.config['dtype']}")
        print(f"  3. 降水事件筛选")
        print(f"  4. 高效压缩: {self.config['compression']}")

        all_samples = []
        all_dates = []

        # 处理每年数据
        for year in self.config['years']:
            year_samples, year_dates = self.process_year(year)
            all_samples.extend(year_samples)
            all_dates.extend(year_dates)

        # 划分和保存
        if all_samples:
            self.split_and_save(all_samples, all_dates)

        print("\n" + "="*70)
        print("预处理完成!")
        print("="*70)


def main():
    """主函数"""
    preprocessor = OptimizedCMPAPreprocessor(CONFIG)
    preprocessor.process()


if __name__ == "__main__":
    main()

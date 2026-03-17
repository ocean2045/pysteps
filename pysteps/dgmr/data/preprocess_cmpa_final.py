"""
基于测试结果的CMPA数据预处理 - 最终优化版本

测试结果分析:
- 2天数据生成227个样本
- 单样本大小0.51 MB (原始分辨率)
- 预计完整数据集仅3-5 GB

结论: 保持原始分辨率，只需轻度优化
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
import h5py

warnings.filterwarnings('ignore')

# 基于测试结果的优化配置
CONFIG = {
    # 数据路径
    'data_root': '/data/data/CMPAS_P05_10MIN',
    'output_root': '/data/workspace/PyStepsDashu/data/dgmr_training',

    # 时间范围
    'years': [2023, 2024],
    'months': [5, 6, 7, 8, 9],

    # 模型参数
    'input_frames': 12,      # 2小时历史
    'output_frames': 18,     # 3小时预报
    'total_frames': 30,      # 总帧数

    # 空间处理（轻度裁剪，保持分辨率）
    'crop_lat': (100, 500),   # 裁剪无效边界
    'crop_lon': (700, 1300),

    # 数据处理
    'precip_threshold': 0.1,
    'max_precip': 100.0,
    'min_output_precip': True,  # 输出时段必须有降水

    # 输出设置
    'dtype': 'float32',         # 保持精度
    'samples_per_file': 500,    # 每个文件500个样本
    'compression': 'gzip',      # 良好压缩
    'compression_opts': 4,      # 压缩级别
}


class FinalCMPAPreprocessor:
    """最终版CMPA预处理器 - 基于测试结果优化"""

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
        }

    def read_and_process_file(self, file_path: str):
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

            # 轻度裁剪（如果配置了）
            if self.config.get('crop_lat') and self.config.get('crop_lon'):
                lat_slice = slice(*self.config['crop_lat'])
                lon_slice = slice(*self.config['crop_lon'])
                data = data[lat_slice, lon_slice]

            # 裁剪异常值
            data = np.clip(data, 0, self.config['max_precip'])

            return data

        except Exception as e:
            warnings.warn(f"读取失败 {file_path}: {e}")
            return None

    def create_sequences(self, daily_data):
        """创建训练序列"""
        sequences = []
        total_frames = self.config['total_frames']

        if len(daily_data) < total_frames:
            return sequences

        # 滑动窗口
        for i in range(len(daily_data) - total_frames + 1):
            seq = daily_data[i:i + total_frames]
            seq = np.stack(seq, axis=0)

            # 检查输出时段
            if self.config['min_output_precip']:
                output_data = seq[self.config['input_frames']:]
                if not (output_data > self.config['precip_threshold']).any():
                    continue

            sequences.append(seq)

        return sequences

    def save_batch(self, samples, split, batch_idx):
        """保存一批样本"""
        if not samples:
            return 0

        output_file = self.output_root / split / f"precip_{split}_{batch_idx:04d}.h5"

        with h5py.File(output_file, 'w') as f:
            shape = (len(samples),) + samples[0].shape

            # 创建压缩数据集
            dset = f.create_dataset(
                'precipitation',
                shape=shape,
                dtype=self.config['dtype'],
                compression=self.config['compression'],
                compression_opts=self.config.get('compression_opts', 4),
                chunks=(1, shape[1], shape[2])
            )

            # 写入数据
            for i, sample in enumerate(samples):
                dset[i] = sample.astype(self.config['dtype'])

            # 元数据
            f.attrs['input_frames'] = self.config['input_frames']
            f.attrs['output_frames'] = self.config['output_frames']
            f.attrs['total_frames'] = self.config['total_frames']
            f.attrs['num_samples'] = len(samples)
            f.attrs['dtype'] = str(self.config['dtype'])

        return len(samples)

    def process_all(self):
        """处理所有数据"""
        print("="*70)
        print("CMPA降水数据预处理 - 最终优化版")
        print("="*70)

        print("\n配置参数:")
        print(f"  分辨率: 原始 (0.05°)")
        print(f"  数据类型: {self.config['dtype']}")
        print(f"  压缩: {self.config['compression']}")
        print(f"  裁剪: {self.config.get('crop_lat', '无')}")
        print(f"  样本/文件: {self.config['samples_per_file']}")

        all_samples = []
        all_dates = []

        for year in self.config['years']:
            print(f"\n{'='*60}")
            print(f"处理 {year} 年")
            print(f"{'='*60}")

            for month in tqdm(self.config['months'], desc=f"  月份"):
                month_str = f"{year}{month:02d}"
                month_dir = self.data_root / month_str

                if not month_dir.exists():
                    continue

                date_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()])

                for date_dir in date_dirs:
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

                    if len(daily_data) < self.config['total_frames']:
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

        # 划分和保存
        if all_samples:
            self.split_and_save(all_samples, all_dates)

    def split_and_save(self, all_samples, all_dates):
        """划分数据集并保存"""
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

        print(f"总样本数: {n_total:,}")
        print(f"训练集: {len(train):,} (70%)")
        print(f"验证集: {len(val):,} (15%)")
        print(f"测试集: {len(test):,} (15%)")

        self.stats['total_samples'] = n_total

        # 保存
        for split, samples in [('train', train), ('val', val), ('test', test)]:
            print(f"\n保存 {split} 集...")
            batch_size = self.config['samples_per_file']

            for i in tqdm(range(0, len(samples), batch_size), desc=f"  {split}"):
                batch = samples[i:i + batch_size]
                batch_idx = i // batch_size
                self.save_batch(batch, split, batch_idx)

        # 保存信息
        self.save_info()

    def save_info(self):
        """保存数据集信息"""
        info_path = self.output_root / 'info' / 'dataset_info.txt'

        # 计算实际大小
        actual_size_gb = 0
        for split in ['train', 'val', 'test']:
            split_dir = self.output_root / split
            for file in split_dir.glob("*.h5"):
                actual_size_gb += file.stat().st_size
        actual_size_gb /= (1024**3)

        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CMPA降水数据集 - Improved DGMR训练\n")
            f.write("="*60 + "\n\n")

            f.write("数据范围:\n")
            f.write(f"  年份: {self.config['years']}\n")
            f.write(f"  月份: {self.config['months']}\n")
            f.write(f"  处理天数: {self.stats['processed_days']}\n\n")

            f.write("数据参数:\n")
            f.write(f"  输入帧数: {self.config['input_frames']} (2小时)\n")
            f.write(f"  输出帧数: {self.config['output_frames']} (3小时)\n")
            f.write(f"  总帧数: {self.config['total_frames']}\n\n")

            f.write("数据集统计:\n")
            f.write(f"  总样本数: {self.stats['total_samples']:,}\n\n")

            f.write("存储信息:\n")
            f.write(f"  实际大小: {actual_size_gb:.2f} GB\n")
            f.write(f"  输出目录: {self.output_root}\n")

        print(f"\n数据集信息已保存到: {info_path}")


def main():
    """主函数"""
    preprocessor = FinalCMPAPreprocessor(CONFIG)
    preprocessor.process_all()


if __name__ == "__main__":
    main()

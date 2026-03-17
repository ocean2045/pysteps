"""
CMPA中国区域降水数据预处理脚本

功能：
1. 读取2023-2024年5-9月GRB2格式数据
2. 筛选有降水过程的日子
3. 数据清洗和异常值处理
4. 构建训练集/验证集/测试集
5. 支持未来3小时逐10分钟预报（18帧输出）

数据规格：
- 时间步长: 10分钟
- 空间分辨率: 0.05°
- 输入帧数: 12帧 (2小时历史)
- 输出帧数: 18帧 (3小时预报)
- 总帧数: 30帧
"""

import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
import h5py
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 配置参数
CONFIG = {
    # 数据路径
    'data_root': '/data/data/CMPAS_P05_10MIN',
    'output_root': '/data/workspace/PyStepsDashu/data/dgmr_training',

    # 时间范围
    'years': [2023, 2024],
    'months': [5, 6, 7, 8, 9],  # 5-9月汛期

    # 模型参数
    'input_frames': 12,      # 2小时历史 (12 × 10min)
    'output_frames': 18,     # 3小时预报 (18 × 10min)
    'total_frames': 30,      # 总帧数

    # 数据处理参数
    'precip_threshold': 0.1,     # 最小降水阈值 (mm)
    'max_precip': 100.0,         # 最大降水值 (mm)
    'min_precip_fraction': 0.05, # 最小降水覆盖比例
    'precip_event_threshold': 0.2, # 降水事件阈值

    # 数据集划分
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    # 文件参数
    'samples_per_file': 500,   # 每个文件保存的样本数
    'compression': 'gzip',     # 压缩方式
}


class CMPADataPreprocessor:
    """
    CMPA降水数据预处理器

    功能：
    1. 扫描和筛选数据文件
    2. 读取和清洗GRB2数据
    3. 构建训练样本序列
    4. 保存为HDF5格式
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_root = Path(config['data_root'])
        self.output_root = Path(config['output_root'])

        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / 'train').mkdir(exist_ok=True)
        (self.output_root / 'val').mkdir(exist_ok=True)
        (self.output_root / 'test').mkdir(exist_ok=True)
        (self.output_root / 'info').mkdir(exist_ok=True)

        # 统计信息
        self.stats = {
            'total_days': 0,
            'precipitation_days': 0,
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'removed_outliers': 0
        }

    def scan_data_files(self) -> List[Tuple[str, datetime]]:
        """
        扫描数据文件

        Returns
        -------
        files : list of tuple
            (文件路径, 日期) 列表
        """
        print("\n" + "="*60)
        print("扫描数据文件...")
        print("="*60)

        files = []

        for year in self.config['years']:
            for month in self.config['months']:
                # 构建月份目录
                month_str = f"{year}{month:02d}"
                month_dir = self.data_root / month_str

                if not month_dir.exists():
                    print(f"警告: 目录不存在 - {month_dir}")
                    continue

                # 获取该月所有日期目录
                date_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()])

                for date_dir in date_dirs:
                    # 检查日期名称格式
                    try:
                        date = datetime.strptime(date_dir.name, "%Y%m%d")
                    except ValueError:
                        continue

                    # 检查是否在目标月份
                    if date.month != month or date.year != year:
                        continue

                    # 获取该目录下的所有GRB2文件
                    grb2_files = list(date_dir.glob("*.GRB2"))

                    if len(grb2_files) >= self.config['total_frames']:
                        files.append((date_dir, date))

        self.stats['total_days'] = len(files)
        print(f"找到 {len(files)} 天的数据")

        return files

    def read_grb2_data(self, file_path: str) -> Optional[np.ndarray]:
        """
        读取GRB2格式降水数据

        Parameters
        ----------
        file_path : str
            GRB2文件路径

        Returns
        -------
        data : np.ndarray or None
            降水数据 [H, W]，如果读取失败返回None
        """
        try:
            # 使用xarray读取GRIB2文件
            ds = xr.open_dataset(file_path, engine='cfgrib')

            # CMPA数据通常变量名为'unknown'
            if 'unknown' in ds:
                data = ds['unknown'].values
            else:
                # 尝试其他变量名
                possible_names = ['precipitation', 'tp', 'precip', 'PRE', 'precipitation_amount']
                data = None
                for name in possible_names:
                    if name in ds:
                        data = ds[name].values
                        break

                # 如果没找到，尝试第一个数据变量
                if data is None:
                    data_vars = [v for v in ds.data_vars if v not in ['time', 'latitude', 'longitude', 'lat', 'lon', 'step', 'surface', 'valid_time']]
                    if data_vars:
                        data = ds[data_vars[0]].values

            ds.close()

            if data is not None:
                # 确保是二维数组
                if data.ndim == 3:
                    data = data[0]  # 去掉时间维度
                elif data.ndim == 1:
                    # 可能是展平的数据，需要重塑
                    pass

                # 处理CMPA数据特性
                # 9999是无效值，需要替换为0
                data = np.where(data >= 9998, 0, data)

                # 数据已经是mm单位，保持原样
                return data
            else:
                return None

        except Exception as e:
            warnings.warn(f"读取文件失败 {file_path}: {e}")
            return None

    def has_precipitation_event(self, data: np.ndarray) -> bool:
        """
        检查是否有降水事件

        判断标准：
        1. 降水面积占比 > 阈值
        2. 存在强降水（>5mm/h）

        Parameters
        ----------
        data : np.ndarray
            降水数据

        Returns
        -------
        has_event : bool
            是否有降水事件
        """
        # 过滤无效值
        valid_data = data[~np.isnan(data)]
        valid_data = valid_data[~np.isinf(valid_data)]

        if len(valid_data) == 0:
            return False

        # 检查降水覆盖比例
        precip_pixels = (valid_data > self.config['precip_threshold']).sum()
        precip_fraction = precip_pixels / len(valid_data)

        # 检查强降水
        has_heavy = (valid_data > self.config['precip_event_threshold']).any()

        return precip_fraction > self.config['min_precip_fraction'] or has_heavy

    def clean_data(self, data: np.ndarray) -> np.ndarray:
        """
        清洗数据：处理异常值

        Parameters
        ----------
        data : np.ndarray
            原始数据

        Returns
        -------
        cleaned_data : np.ndarray
            清洗后的数据
        """
        # 替换NaN和Inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # 裁剪到合理范围
        data = np.clip(data, 0, self.config['max_precip'])

        return data

    def create_sample_sequences(
        self,
        daily_data: List[np.ndarray],
        date: datetime
    ) -> List[np.ndarray]:
        """
        创建训练样本序列

        Parameters
        ----------
        daily_data : list of np.ndarray
            一天的数据列表
        date : datetime
            日期

        Returns
        -------
        samples : list of np.ndarray
            样本序列，每个样本形状为 [total_frames, H, W]
        """
        samples = []
        total_frames = self.config['total_frames']

        # 检查数据量是否足够
        if len(daily_data) < total_frames:
            return samples

        # 滑动窗口创建样本
        for i in range(len(daily_data) - total_frames + 1):
            # 提取序列
            sequence = daily_data[i:i + total_frames]

            # 转换为numpy数组
            sequence = np.stack(sequence, axis=0)

            # 检查序列质量
            if self._is_valid_sequence(sequence):
                samples.append(sequence)

        return samples

    def _is_valid_sequence(self, sequence: np.ndarray) -> bool:
        """
        检查序列是否有效

        Parameters
        ----------
        sequence : np.ndarray
            数据序列 [total_frames, H, W]

        Returns
        -------
        is_valid : bool
            是否有效
        """
        # 检查NaN
        if np.isnan(sequence).any():
            return False

        # 检查Inf
        if np.isinf(sequence).any():
            return False

        # 检查负值
        if (sequence < 0).any():
            return False

        # 检查输出时段是否有降水（至少有一帧）
        output_data = sequence[self.config['input_frames']:]
        has_output_precip = (output_data > self.config['precip_threshold']).any()

        return has_output_precip

    def split_dataset(self, all_samples: List[np.ndarray], dates: List[datetime]) -> dict:
        """
        划分数据集

        Parameters
        ----------
        all_samples : list of np.ndarray
            所有样本
        dates : list of datetime
            对应的日期

        Returns
        -------
        splits : dict
            {'train': [...], 'val': [...], 'test': [...]}
        """
        # 按日期排序
        sorted_indices = np.argsort(dates)
        sorted_samples = [all_samples[i] for i in sorted_indices]
        sorted_dates = [dates[i] for i in sorted_indices]

        # 计算划分点
        n_total = len(sorted_samples)
        n_train = int(n_total * self.config['train_ratio'])
        n_val = int(n_total * self.config['val_ratio'])

        # 划分
        train_samples = sorted_samples[:n_train]
        val_samples = sorted_samples[n_train:n_train + n_val]
        test_samples = sorted_samples[n_train + n_val:]

        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }

        # 更新统计
        self.stats['train_samples'] = len(train_samples)
        self.stats['val_samples'] = len(val_samples)
        self.stats['test_samples'] = len(test_samples)
        self.stats['total_samples'] = n_total

        return splits

    def save_samples(self, samples: List[np.ndarray], split: str):
        """
        保存样本到HDF5文件

        Parameters
        ----------
        samples : list of np.ndarray
            样本列表
        split : str
            数据集划分 ('train', 'val', 'test')
        """
        if not samples:
            return

        samples_per_file = self.config['samples_per_file']
        output_dir = self.output_root / split

        # 分批保存
        for i in range(0, len(samples), samples_per_file):
            batch = samples[i:i + samples_per_file]
            file_idx = i // samples_per_file

            # 文件路径
            file_path = output_dir / f"precipitation_{split}_{file_idx:04d}.h5"

            # 保存为HDF5
            with h5py.File(file_path, 'w') as f:
                # 创建数据集
                data_shape = (len(batch),) + batch[0].shape

                # 创建可压缩的数据集
                dset = f.create_dataset(
                    'precipitation',
                    shape=data_shape,
                    dtype='float32',
                    compression=self.config['compression'],
                    chunks=True
                )

                # 写入数据
                for j, sample in enumerate(batch):
                    dset[j] = sample.astype('float32')

                # 保存元数据
                f.attrs['input_frames'] = self.config['input_frames']
                f.attrs['output_frames'] = self.config['output_frames']
                f.attrs['total_frames'] = self.config['total_frames']
                f.attrs['num_samples'] = len(batch)
                f.attrs['split'] = split
                f.attrs['created'] = datetime.now().isoformat()

            print(f"保存: {file_path} ({len(batch)} 样本)")

    def save_statistics(self):
        """保存统计信息"""
        stats_path = self.output_root / 'info' / 'preprocessing_stats.txt'

        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CMPA降水数据预处理统计\n")
            f.write("="*60 + "\n\n")

            f.write("数据范围:\n")
            f.write(f"  年份: {self.config['years']}\n")
            f.write(f"  月份: {self.config['months']}\n")
            f.write(f"  总天数: {self.stats['total_days']}\n")
            f.write(f"  有降水天数: {self.stats['precipitation_days']}\n\n")

            f.write("模型参数:\n")
            f.write(f"  输入帧数: {self.config['input_frames']} (2小时)\n")
            f.write(f"  输出帧数: {self.config['output_frames']} (3小时)\n")
            f.write(f"  总帧数: {self.config['total_frames']}\n\n")

            f.write("数据集划分:\n")
            f.write(f"  总样本数: {self.stats['total_samples']}\n")
            f.write(f"  训练集: {self.stats['train_samples']} ({self.config['train_ratio']*100:.0f}%)\n")
            f.write(f"  验证集: {self.stats['val_samples']} ({self.config['val_ratio']*100:.0f}%)\n")
            f.write(f"  测试集: {self.stats['test_samples']} ({self.config['test_ratio']*100:.0f}%)\n\n")

            f.write("数据处理:\n")
            f.write(f"  最小降水阈值: {self.config['precip_threshold']} mm\n")
            f.write(f"  最大降水值: {self.config['max_precip']} mm\n")
            f.write(f"  移除异常值: {self.stats['removed_outliers']}\n\n")

            f.write("输出目录:\n")
            f.write(f"  {self.output_root}\n")

        print(f"\n统计信息已保存到: {stats_path}")

    def process(self):
        """
        执行完整的数据预处理流程
        """
        print("\n" + "="*70)
        print("CMPA降水数据预处理 - Improved DGMR训练数据准备")
        print("="*70)

        print("\n配置参数:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        # 1. 扫描数据文件
        data_dirs = self.scan_data_files()

        if not data_dirs:
            print("错误: 没有找到数据文件")
            return

        # 2. 处理每一天的数据
        print("\n" + "="*60)
        print("处理数据...")
        print("="*60)

        all_samples = []
        all_dates = []
        precip_days = 0

        for date_dir, date in tqdm(data_dirs, desc="处理日期"):
            # 获取该目录下的所有GRB2文件并排序
            grb2_files = sorted(list(date_dir.glob("*.GRB2")))

            if len(grb2_files) < self.config['total_frames']:
                continue

            # 读取所有文件
            daily_data = []
            for file_path in grb2_files:
                data = self.read_grb2_data(str(file_path))

                if data is not None:
                    # 清洗数据
                    data = self.clean_data(data)
                    daily_data.append(data)

            # 检查是否至少有完整的序列
            if len(daily_data) < self.config['total_frames']:
                continue

            # 检查是否有降水事件
            has_precip = False
            for data in daily_data:
                if self.has_precipitation_event(data):
                    has_precip = True
                    break

            if not has_precip:
                continue

            precip_days += 1

            # 创建样本序列
            samples = self.create_sample_sequences(daily_data, date)

            if samples:
                all_samples.extend(samples)
                all_dates.extend([date] * len(samples))

        self.stats['precipitation_days'] = precip_days
        print(f"\n找到 {len(all_samples)} 个有效样本，来自 {precip_days} 个有降水日")

        if not all_samples:
            print("错误: 没有创建任何样本")
            return

        # 3. 划分数据集
        print("\n" + "="*60)
        print("划分数据集...")
        print("="*60)

        splits = self.split_dataset(all_samples, all_dates)

        print(f"训练集: {len(splits['train'])} 样本")
        print(f"验证集: {len(splits['val'])} 样本")
        print(f"测试集: {len(splits['test'])} 样本")

        # 4. 保存数据
        print("\n" + "="*60)
        print("保存数据...")
        print("="*60)

        for split in ['train', 'val', 'test']:
            print(f"\n保存 {split} 集...")
            self.save_samples(splits[split], split)

        # 5. 保存统计信息
        self.save_statistics()

        print("\n" + "="*70)
        print("数据预处理完成!")
        print("="*70)
        print(f"\n输出目录: {self.output_root}")
        print(f"总样本数: {len(all_samples)}")


def main():
    """主函数"""
    # 创建预处理器
    preprocessor = CMPADataPreprocessor(CONFIG)

    # 执行预处理
    preprocessor.process()


if __name__ == "__main__":
    main()

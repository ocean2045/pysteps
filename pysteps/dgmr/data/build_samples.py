"""
灵活的样本构建器 - 基于JSON配置

从年度HDF文件中构建训练样本，通过JSON配置文件
灵活指定输入帧数和输出帧数

功能：
1. 根据配置滑动构建样本
2. 降雨量级统计分析
3. 强降水过采样
"""

import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class FlexibleSampleBuilder:
    """灵活样本构建器"""

    def __init__(self, config_path):
        """
        初始化样本构建器

        Parameters
        ----------
        config_path : str
            JSON配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        """加载JSON配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_samples(self, hdf_file, output_path):
        """
        从HDF文件构建样本

        Parameters
        ----------
        hdf_file : str
            HDF文件路径
        output_path : str
            输出目录
        """
        print("="*70)
        print("构建训练样本")
        print("="*70)
        print(f"\n配置文件: {self.config_path}")
        print(f"HDF文件: {hdf_file}")
        print(f"输出目录: {output_path}")

        # 打印配置
        print(f"\n配置参数:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        # 读取HDF数据
        print(f"\n读取HDF文件...")
        with h5py.File(hdf_file, 'r') as f:
            precip = f['precipitation'][:]  # [time, lat, lon]
            times = f['time'][:]
            lats = f['latitude'][:]
            lons = f['longitude'][:]

        print(f"数据形状: {precip.shape}")
        print(f"时间步数: {len(times)}")

        # 构建样本
        samples = self._build_samples(precip)

        # 降雨量级统计
        print(f"\n分析降雨量级...")
        stats = self._analyze_precipitation_stats(samples)

        # 强降水过采样
        if self.config.get('oversample_heavy', False):
            print(f"\n强降水过采样...")
            samples = self._oversample_heavy_precipitation(samples, stats)

        # 保存样本
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self._save_samples(samples, stats, output_path)

        # 保存统计信息
        self._save_statistics(stats, output_path)

    def _build_samples(self, precip):
        """滑动构建样本"""
        input_frames = self.config['input_frames']
        output_frames = self.config['output_frames']
        total_frames = input_frames + output_frames

        samples = []

        # 滑动窗口
        for i in range(len(precip) - total_frames + 1):
            sample = precip[i:i + total_frames]
            samples.append(sample)

        print(f"构建样本数: {len(samples)}")
        return samples

    def _analyze_precipitation_stats(self, samples):
        """分析降雨量级统计"""

        # 降雨量级定义 (mm/h)
        intensity_levels = {
            'none': 0.0,
            'light': 0.1,
            'moderate': 2.0,
            'heavy': 5.0,
            'extreme': 10.0,
            'violent': 20.0,
        }

        stats = {
            'intensity_levels': intensity_levels,
            'samples': [],
        }

        for sample in tqdm(samples, desc="  统计样本"):
            # 输入时段统计
            input_data = sample[:self.config['input_frames']]

            # 输出时段统计
            output_data = sample[self.config['input_frames']:]

            # 计算量级分布
            input_stats = self._compute_intensity_stats(input_data)
            output_stats = self._compute_intensity_stats(output_data)

            # 最大降水
            max_precip = sample.max()

            stats['samples'].append({
                'input_stats': input_stats,
                'output_stats': output_stats,
                'max_precip': float(max_precip),
                'mean_precip': float(sample.mean()),
                'intensity_class': self._classify_intensity(max_precip),
            })

        return stats

    def _compute_intensity_stats(self, data):
        """计算时段内的降雨强度统计"""
        return {
            'mean': float(data.mean()),
            'max': float(data.max()),
            'std': float(data.std()),
            'precip_pixels': int((data > 0.1).sum()),
            'precip_fraction': float((data > 0.1).sum() / data.size),
            'heavy_pixels': int((data > 5.0).sum()),
            'extreme_pixels': int((data > 10.0).sum()),
        }

    def _classify_intensity(self, max_precip):
        """根据最大降水分类强度"""
        if max_precip < 0.1:
            return 'none'
        elif max_precip < 2.0:
            return 'light'
        elif max_precip < 5.0:
            return 'moderate'
        elif max_precip < 10.0:
            return 'heavy'
        elif max_precip < 20.0:
            return 'extreme'
        else:
            return 'violent'

    def _oversample_heavy_precipitation(self, samples, stats, target_ratio=3.0):
        """对强降水样本进行过采样"""

        # 找出强降水样本
        heavy_indices = []
        light_indices = []

        for i, sample_stats in enumerate(stats['samples']):
            intensity = sample_stats['intensity_class']
            if intensity in ['heavy', 'extreme', 'violent']:
                heavy_indices.append(i)
            else:
                light_indices.append(i)

        print(f"  强降水样本: {len(heavy_indices)}")
        print(f"  弱降水样本: {len(light_indices)}")

        if len(heavy_indices) == 0:
            print("  警告: 没有强降水样本，跳过过采样")
            return samples

        # 计算需要复制的次数
        target_heavy = int(len(light_indices) * target_ratio / len(heavy_indices))
        copies_per_heavy = max(1, target_heavy)

        print(f"  过采样比例: {copies_per_heavy}x")

        # 复制强降水样本
        oversampled_samples = list(samples)

        for idx in tqdm(heavy_indices, desc="  过采样"):
            for _ in range(copies_per_heavy):
                oversampled_samples.append(samples[idx])

        print(f"  过采样后样本数: {len(oversampled_samples)}")

        return oversampled_samples

    def _save_samples(self, samples, stats, output_path):
        """保存样本到HDF文件"""
        output_file = output_path / "samples.h5"

        with h5py.File(output_file, 'w') as f:
            # 保存样本数据
            samples_array = np.stack(samples, axis=0)

            f.create_dataset(
                'samples',
                data=samples_array.astype('float32'),
                compression='gzip',
                chunks=(1, samples_array.shape[1], samples_array.shape[2])
            )

            # 保存统计信息
            stats_file = output_path / "sample_stats.json"
            with open(stats_file, 'w') as f_json:
                json.dump(stats, f_json, indent=2)

        print(f"\n✓ 样本已保存: {output_file}")
        self._print_sample_info(output_file)

    def _print_sample_info(self, file_path):
        """打印样本信息"""
        with h5py.File(file_path, 'r') as f:
            samples = f['samples'][:]

        size_mb = file_path.stat().st_size / (1024 * 1024)

        print(f"\n样本文件信息:")
        print(f"  路径: {file_path}")
        print(f"  大小: {size_mb:.2f} MB")
        print(f"  样本数: {samples.shape[0]}")
        print(f"  样本形状: {samples.shape[1:]} (帧, 纬度, 经度)")
        print(f"  总数据量: {samples.nbytes / (1024**3):.2f} GB")

    def _save_statistics(self, stats, output_path):
        """保存统计信息到文本文件"""
        stats_file = output_path / "statistics.txt"

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("样本统计信息\n")
            f.write("="*60 + "\n\n")

            f.write("配置信息:\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write(f"总样本数: {len(stats['samples'])}\n\n")

            # 量级分布
            intensity_counts = {}
            for sample_stats in stats['samples']:
                intensity = sample_stats['intensity_class']
                intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1

            f.write("强度等级分布:\n")
            for intensity in ['none', 'light', 'moderate', 'heavy', 'extreme', 'violent']:
                count = intensity_counts.get(intensity, 0)
                pct = count / len(stats['samples']) * 100 if stats['samples'] else 0
                f.write(f"  {intensity.capitalize():10s}: {count:6d} ({pct:5.2f}%)\n")

            f.write("\n")

            # 降水统计
            max_precips = [s['max_precip'] for s in stats['samples']]
            mean_precips = [s['mean_precip'] for s in stats['samples']]

            f.write("降水统计:\n")
            f.write(f"  最大降水范围: [{min(max_precips):.2f}, {max(max_precips):.2f}] mm/h\n")
            f.write(f"  平均降水范围: [{min(mean_precips):.4f}, {max(mean_precips):.4f}] mm/h\n")
            f.write(f"  最大降水中位数: {np.median(max_precips):.2f} mm/h\n")
            f.write(f"  平均降水中位数: {np.median(mean_precips):.4f} mm/h\n")

        print(f"✓ 统计信息已保存: {stats_file}")


def main():
    """主函数"""
    import sys

    if len(sys.argv) < 3:
        print("用法: python build_samples.py <config.json> <hdf_file> [output_dir]")
        print("\n示例:")
        print("  python build_samples.py config_6_18.json 2023.h5")
        print("  python build_samples.py config_12_24.json 2024.h5 output_2024")
        sys.exit(1)

    config_file = sys.argv[1]
    hdf_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 2 else f"samples_{Path(hdf_file).stem}"

    builder = FlexibleSampleBuilder(config_file)
    builder.build_samples(hdf_file, output_dir)


if __name__ == "__main__":
    main()

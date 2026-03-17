#!/usr/bin/env python3
"""
PySteps 多数据集短临预报评估系统

支持多个数据源:
- DWD (德国气象局)
- KNMI (荷兰皇家气象研究所)
- MCH (瑞士气象局)
- FMI (芬兰气象研究所)
- BOM (澳大利亚气象局)
- RMI (比利时皇家气象研究所)

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import sys
import numpy as np
import h5py
import time
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict

sys.path.insert(0, '/data/workspace/PyStepsDashu')


class MultiDatasetEvaluator:
    """多数据集评估器"""

    # 评估阈值
    THRESHOLDS = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]

    # 预报时效
    LEAD_TIMES = [5, 10, 15, 30, 45, 60]

    def __init__(self, base_dir='/data/workspace/pysteps-data/radar',
                 n_ensemble_members=20):
        """
        初始化评估器

        Parameters
        ----------
        base_dir : str
            雷达数据基础目录
        n_ensemble_members : int
            集合成员数
        """
        self.base_dir = Path(base_dir)
        self.n_ensemble_members = n_ensemble_members
        self.thresholds = self.THRESHOLDS
        self.lead_times = self.LEAD_TIMES

        # 所有数据集的评估结果
        self.all_results = {}

    def discover_datasets(self):
        """发现可用的数据集"""
        print("\n" + "="*70)
        print("发现可用数据集")
        print("="*70)

        datasets = {}

        # 定义数据集信息
        dataset_configs = {
            'dwd': {
                'name': 'DWD',
                'full_name': 'German Weather Service (德国气象局)',
                'pattern': 'dwd/**/*.h5',
                'data_format': 'odimh5_dwd',
            },
            'knmi': {
                'name': 'KNMI',
                'full_name': 'Royal Netherlands Meteorological Institute (荷兰皇家气象研究所)',
                'pattern': 'KNMI/**/*.h5',
                'data_format': 'odimh5_knmi',
            },
            'fmi': {
                'name': 'FMI',
                'full_name': 'Finnish Meteorological Institute (芬兰气象研究所)',
                'pattern': 'fmi/**/*.h5',
                'data_format': 'odimh5_fmi',
            },
            'bom': {
                'name': 'BOM',
                'full_name': 'Bureau of Meteorology (澳大利亚气象局)',
                'pattern': 'bom/**/*.h5',
                'data_format': 'odimh5_bom',
            },
            'rmi': {
                'name': 'RMI',
                'full_name': 'Royal Meteorological Institute (比利时皇家气象研究所)',
                'pattern': 'rmi/**/*.h5',
                'data_format': 'odimh5_rmi',
            },
        }

        for key, config in dataset_configs.items():
            files = list(self.base_dir.glob(config['pattern']))
            if files:
                datasets[key] = {
                    **config,
                    'n_files': len(files),
                    'sample_file': str(files[0]),
                }
                print(f"\n✓ {config['name']}: {len(files)} 个文件")
                print(f"  {config['full_name']}")

        return datasets

    def load_knmi_data(self, file_path):
        """加载 KNMI 数据"""
        try:
            with h5py.File(file_path, 'r') as f:
                # KNMI 使用 image1/image_data 路径
                if 'image1/image_data' in f:
                    data = f['image1/image_data'][()]
                else:
                    # 尝试其他路径
                    for key in f:
                        if isinstance(f[key], h5py.Group):
                            for subkey in f[key]:
                                if isinstance(f[key][subkey], h5py.Dataset):
                                    data = f[key][subkey][()]
                                    break
                            else:
                                continue
                            break
                    else:
                        raise ValueError("无法找到数据集")

                # 获取元数据
                metadata = {
                    'format': 'odimh5_knmi',
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                }

                # KNMI 数据转换 (通常是 dBZ)
                # 简化处理：假设已经是合适的格式
                precip = data.astype(np.float32)
                precip[precip == 65535] = np.nan  # nodata

                # 尝试获取增益和偏移
                gain = 1.0
                offset = 0.0

                # 查找 what 属性
                def find_gain_offset(name, obj):
                    nonlocal gain, offset
                    if hasattr(obj, 'attrs'):
                        if 'gain' in obj.attrs:
                            gain = float(obj.attrs['gain'])
                        if 'offset' in obj.attrs:
                            offset = float(obj.attrs['offset'])

                # 转换为物理值
                precip = precip * gain + offset
                precip = np.nan_to_num(precip, nan=0.0)

                # 限制范围
                precip = np.clip(precip, 0, 500)

                metadata['gain'] = gain
                metadata['offset'] = offset

                return precip, metadata

        except Exception as e:
            print(f"✗ 加载 KNMI 数据失败: {e}")
            return None, None

    def load_dwd_data(self, file_path):
        """加载 DWD 数据"""
        try:
            with h5py.File(file_path, 'r') as f:
                # DWD 使用 dataset1/data1/data 路径
                if 'dataset1/data1/data' in f:
                    data = f['dataset1/data1/data'][()]
                    what_attrs = dict(f['dataset1/data1/what'].attrs)
                else:
                    raise ValueError("无法找到数据集")

                # 获取元数据
                gain = float(what_attrs.get('gain', 0.01))
                offset = float(what_attrs.get('offset', 0.0))
                nodata = float(what_attrs.get('nodata', 65535.0))

                metadata = {
                    'format': 'odimh5_dwd',
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'gain': gain,
                    'offset': offset,
                }

                # 转换为物理值
                precip = data.astype(np.float32)
                precip[precip == nodata] = np.nan
                precip = precip * gain + offset
                precip = np.nan_to_num(precip, nan=0.0)
                precip = np.clip(precip, 0, 500)

                return precip, metadata

        except Exception as e:
            print(f"✗ 加载 DWD 数据失败: {e}")
            return None, None

    def load_dataset_series(self, dataset_key, n_frames=6):
        """加载一个数据集的序列"""
        print(f"\n{'='*70}")
        print(f"加载 {dataset_key.upper()} 数据集")
        print(f"{'='*70}")

        # 查找文件
        if dataset_key == 'dwd':
            pattern = 'dwd/**/*.h5'
            loader = self.load_dwd_data
        elif dataset_key == 'knmi':
            pattern = 'KNMI/**/*.h5'
            loader = self.load_knmi_data
        else:
            print(f"✗ 不支持的数据集: {dataset_key}")
            return None, None

        files = sorted(self.base_dir.glob(pattern))
        print(f"找到 {len(files)} 个文件")

        if len(files) < n_frames:
            n_frames = min(n_frames, len(files))
            print(f"⚠ 只有 {len(files)} 个文件，将使用 {n_frames} 帧")

        # 加载序列
        precip_series = []
        metadata = None

        for i, file_path in enumerate(files[:n_frames]):
            precip, meta = loader(file_path)
            if precip is not None:
                precip_series.append(precip)
                if metadata is None:
                    metadata = meta
                print(f"  帧 {i+1}/{n_frames}: {precip.shape}")
            else:
                print(f"  ✗ 帧 {i+1}: 加载失败")

        if not precip_series:
            return None, None

        precip_series = np.array(precip_series)
        print(f"✓ 加载完成: {precip_series.shape}")

        return precip_series, metadata

    def evaluate_dataset(self, dataset_key, precip_series, metadata):
        """评估单个数据集"""
        print(f"\n{'='*70}")
        print(f"评估 {dataset_key.upper()} 数据集")
        print(f"{'='*70}")

        results = {
            'dataset_key': dataset_key,
            'metadata': metadata,
            'performance_by_threshold': {},
            'summary': {},
        }

        observation = precip_series[-1]

        # 模拟预报 (简化版本)
        ensemble_mean = self._simulate_forecast(precip_series)

        # 评估每个阈值
        for threshold in self.thresholds:
            print(f"\n  阈值 {threshold:.1f} mm/h:")

            threshold_results = {
                'threshold_mm_h': threshold,
                'lead_times': {},
                'observation_stats': {},
            }

            # 观测统计
            obs_binary = observation > threshold
            n_pixels_obs = np.sum(obs_binary)
            coverage_obs = 100.0 * n_pixels_obs / observation.size

            threshold_results['observation_stats'] = {
                'precipitation_pixels': int(n_pixels_obs),
                'coverage_percent': float(coverage_obs),
                'max_intensity': float(np.max(observation)),
                'mean_intensity': float(np.mean(observation[obs_binary])) if n_pixels_obs > 0 else 0.0,
            }

            print(f"    观测: {n_pixels_obs} 像素 ({coverage_obs:.2f}%)")

            # 评估各时效
            for lead_time in self.lead_times:
                timestep = lead_time // 5
                if timestep >= ensemble_mean.shape[0]:
                    continue

                forecast = ensemble_mean[timestep]
                fcst_binary = forecast > threshold

                # 计算指标
                hits = np.sum((obs_binary == True) & (fcst_binary == True))
                misses = np.sum((obs_binary == True) & (fcst_binary == False))
                false_alarms = np.sum((obs_binary == False) & (fcst_binary == True))

                csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
                pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
                far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
                bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else 0.0

                threshold_results['lead_times'][f'{lead_time}min'] = {
                    'csi': float(csi),
                    'pod': float(pod),
                    'far': float(far),
                    'bias': float(bias),
                }

                print(f"    {lead_time:3d}min: CSI={csi:.3f}, POD={pod:.3f}, FAR={far:.3f}")

            results['performance_by_threshold'][f'{threshold:.1f}mm'] = threshold_results

        return results

    def _simulate_forecast(self, precip_series):
        """模拟预报 (简化版本)"""
        n_timesteps = max(self.lead_times) // 5 + 1
        shape = precip_series.shape[1:]

        ensemble_mean = np.zeros((n_timesteps, *shape), dtype=np.float32)
        ensemble_mean[0] = precip_series[-1]

        for t in range(1, n_timesteps):
            # 简化的衰减模型
            decay = 0.98 ** t
            noise = np.random.randn(*shape) * 0.05
            ensemble_mean[t] = ensemble_mean[t-1] * decay + noise
            ensemble_mean[t] = np.maximum(ensemble_mean[t], 0)

        return ensemble_mean

    def generate_comparison_report(self, all_results):
        """生成数据集对比报告"""
        print("\n" + "="*70)
        print("多数据集性能对比报告")
        print("="*70)

        if not all_results:
            print("无评估结果")
            return

        # 数据集列表
        datasets = list(all_results.keys())

        # 对比表: 15分钟时效的 CSI
        print(f"\n📊 CSI 对比 (15分钟时效):")
        print(f"{'数据集':<10} " + " ".join([f"{t:>7}" for t in self.thresholds]))
        print("  " + "-" * (10 + 8 * len(self.thresholds)))

        for ds_key in datasets:
            results = all_results[ds_key]
            row = [ds_key.upper()]

            for threshold in self.thresholds:
                key = f'{threshold:.1f}mm'
                data = results.get('performance_by_threshold', {}).get(key, {})
                lt_data = data.get('lead_times', {}).get('15min', {})
                csi = lt_data.get('csi', 0.0)
                row.append(f"{csi:7.3f}")

            print("  " + " ".join(row))

        # 平均性能
        print(f"\n📈 平均 CSI (所有时效):")
        print(f"{'数据集':<10} {'平均CSI':<10} {'数据尺寸':<15} {'覆盖率':<10}")
        print("  " + "-" * 50)

        for ds_key in datasets:
            results = all_results[ds_key]
            metadata = results.get('metadata', {})

            # 计算平均 CSI
            all_csi = []
            for threshold in self.thresholds:
                key = f'{threshold:.1f}mm'
                data = results.get('performance_by_threshold', {}).get(key, {})
                obs_stats = data.get('observation_stats', {})

                # 只统计有降水区域的阈值
                if obs_stats.get('coverage_percent', 0) > 0.1:
                    lt_data = data.get('lead_times', {})
                    csi_values = [lt.get('csi', 0) for lt in lt_data.values()]
                    all_csi.extend(csi_values)

            avg_csi = np.mean(all_csi) if all_csi else 0.0
            shape = metadata.get('shape', [0, 0])
            shape_str = f"{shape[0]}x{shape[1]}"

            # 计算平均覆盖率
            coverages = []
            for threshold in self.thresholds:
                key = f'{threshold:.1f}mm'
                data = results.get('performance_by_threshold', {}).get(key, {})
                cov = data.get('observation_stats', {}).get('coverage_percent', 0)
                if cov > 0:
                    coverages.append(cov)

            avg_cov = np.mean(coverages) if coverages else 0.0

            print(f"  {ds_key.upper():<10} {avg_csi:<10.3f} {shape_str:<15} {avg_cov:<10.2f}%")

    def save_results(self, output_file=None):
        """保存结果"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'/data/workspace/PyStepsDashu/benchmarks/results/multi_dataset_eval_{timestamp}.json'

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)

        print(f"\n✓ 结果已保存: {output_file}")
        return output_file


def main():
    """主函数"""
    print("="*70)
    print("PySteps 多数据集短临预报评估系统")
    print("="*70)

    evaluator = MultiDatasetEvaluator(
        base_dir='/data/workspace/pysteps-data/radar',
        n_ensemble_members=20
    )

    # 发现数据集
    datasets = evaluator.discover_datasets()

    # 评估每个数据集
    for dataset_key in ['dwd', 'knmi']:  # 先评估这两个
        if dataset_key in datasets:
            precip_series, metadata = evaluator.load_dataset_series(
                dataset_key,
                n_frames=6
            )

            if precip_series is not None:
                results = evaluator.evaluate_dataset(
                    dataset_key,
                    precip_series,
                    metadata
                )
                evaluator.all_results[dataset_key] = results

    # 生成对比报告
    evaluator.generate_comparison_report(evaluator.all_results)

    # 保存结果
    evaluator.save_results()

    print("\n" + "="*70)
    print("✓ 多数据集评估完成")
    print("="*70)

    return evaluator.all_results


if __name__ == '__main__':
    results = main()

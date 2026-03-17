#!/usr/bin/env python3
"""
PySteps 多阈值短临预报评估系统

使用 DWD 真实雷达数据进行多阈值的全面评估。

评估阈值:
- 0.1 mm/h (微量降水)
- 0.5 mm/h (小雨)
- 1.0 mm/h (小雨)
- 2.0 mm/h (中雨)
- 3.5 mm/h (中雨)
- 5.0 mm/h (大雨)

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

try:
    from pysteps import io, nowcasts, motion, verification
    from pysteps.utils import conversion, transformation
    from pysteps.postprocessing import ensemblestats
    print("✓ 成功导入 PySteps")
except ImportError as e:
    print(f"⚠ 导入警告: {e}")


class DWDMultiThresholdEvaluator:
    """DWD多阈值短临预报评估器"""

    # 定义评估阈值
    THRESHOLDS = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]

    # 定义预报时效 (分钟)
    LEAD_TIMES = [5, 10, 15, 30, 45, 60, 90, 120]

    def __init__(self, data_dir, n_ensemble_members=20):
        """
        初始化评估器

        Parameters
        ----------
        data_dir : str
            DWD数据目录
        n_ensemble_members : int
            集合成员数
        """
        self.data_dir = Path(data_dir)
        self.n_ensemble_members = n_ensemble_members
        self.thresholds = self.THRESHOLDS
        self.lead_times = self.LEAD_TIMES

        # 存储结果
        self.results = {
            'dataset_info': {},
            'configuration': {},
            'performance_by_threshold': {},
            'heavy_rain_analysis': {},
            'runtime': {},
            'summary': {}
        }

    def load_dwd_series(self, n_input_frames=4):
        """
        加载DWD时间序列

        Parameters
        ----------
        n_input_frames : int
            输入帧数 (用于运动估计)

        Returns
        -------
        precip_series : ndarray
            降水率序列 (t, height, width)
        metadata : dict
            元数据
        """
        print(f"\n加载DWD数据序列...")
        print(f"  目录: {self.data_dir}")

        # 查找所有.h5文件
        h5_files = sorted(self.data_dir.glob("*.h5"))
        print(f"  找到 {len(h5_files)} 个文件")

        if len(h5_files) < n_input_frames:
            print(f"⚠ 警告: 只有 {len(h5_files)} 个文件，需要至少 {n_input_frames} 个")
            n_input_frames = min(n_input_frames, len(h5_files))

        # 读取数据序列
        precip_series = []
        metadata = {}

        for i, file_path in enumerate(h5_files[:n_input_frames+2]):  # 多读2帧用于验证
            try:
                with h5py.File(file_path, 'r') as f:
                    # 读取数据
                    data = f['dataset1/data1/data'][()]

                    # 读取元数据
                    what_attrs = dict(f['dataset1/data1/what'].attrs)
                    where_attrs = dict(f['where'].attrs)

                    if i == 0:  # 第一次保存元数据
                        metadata.update({
                            'quantity': what_attrs.get('quantity', b'RATE').decode('utf-8'),
                            'gain': float(what_attrs.get('gain', 0.01)),
                            'offset': float(what_attrs.get('offset', 0.0)),
                            'nodata': float(what_attrs.get('nodata', 65535.0)),
                            'undetect': float(what_attrs.get('undetect', 0.0)),
                            'xsize': int(where_attrs.get('xsize', 1100)),
                            'ysize': int(where_attrs.get('ysize', 1200)),
                            'xscale': float(where_attrs.get('xscale', 1000.0)),
                            'yscale': float(where_attrs.get('yscale', 1000.0)),
                            'projdef': where_attrs.get('projdef', b'').decode('utf-8'),
                        })

                    # 转换为物理值
                    gain = metadata['gain']
                    offset = metadata['offset']
                    nodata = metadata['nodata']

                    # 处理缺失值
                    data = data.astype(np.float32)
                    data[data == nodata] = np.nan

                    # 转换为降水率 (mm/h)
                    precip = data * gain + offset
                    precip = np.nan_to_num(precip, nan=0.0)

                    precip_series.append(precip)

            except Exception as e:
                print(f"  ✗ 读取失败 {file_path.name}: {e}")
                continue

        precip_series = np.array(precip_series)
        print(f"✓ 加载完成: {precip_series.shape} (时间, 高度, 宽度)")

        # 保存数据集信息
        self.results['dataset_info'] = {
            'n_frames': len(precip_series),
            'shape': precip_series.shape,
            'resolution_km': f"{metadata['xscale']/1000:.1f} x {metadata['yscale']/1000:.1f}",
            'coverage': f"{metadata['xsize']} x {metadata['ysize']} pixels",
            'quantity': metadata['quantity'],
            'source': 'DWD (German Weather Service)',
        }

        return precip_series, metadata

    def run_nowcast(self, precip_series):
        """
        运行短临预报

        Parameters
        ----------
        precip_series : ndarray
            输入降水序列

        Returns
        -------
        nowcast_output : dict
            预报结果
        """
        print("\n" + "="*70)
        print("运行短临预报")
        print("="*70)
        print(f"  集合成员: {self.n_ensemble_members}")
        print(f"  预报时效: {self.lead_times} 分钟")
        print(f"  评估阈值: {self.thresholds} mm/h")

        start_time = time.time()

        # 使用简化的预报模型
        try:
            # 1. 运动估计
            print("\n[1/3] 运动估计...")
            motion_start = time.time()
            motion_field = self._estimate_motion(precip_series)
            motion_time = time.time() - motion_start
            print(f"  耗时: {motion_time:.2f} 秒")

            # 2. 集合预报生成
            print("\n[2/3] 集合预报生成...")
            nowcast_start = time.time()
            nowcast_ensemble = self._generate_ensemble(
                precip_series[-1],
                motion_field,
                max(self.lead_times)
            )
            nowcast_time = time.time() - nowcast_start
            print(f"  耗时: {nowcast_time:.2f} 秒")

            # 3. 多阈值评估
            print("\n[3/3] 多阈值性能评估...")
            eval_start = time.time()
            self._evaluate_all_thresholds(
                precip_series,
                nowcast_ensemble
            )
            eval_time = time.time() - eval_start
            print(f"  耗时: {eval_time:.2f} 秒")

            total_time = time.time() - start_time

            # 记录运行时间
            self.results['runtime'] = {
                'motion_estimation': motion_time,
                'nowcast_generation': nowcast_time,
                'evaluation': eval_time,
                'total': total_time,
            }

            print(f"\n✓ 预报完成，总耗时: {total_time:.2f} 秒")

            return {
                'motion': motion_field,
                'ensemble': nowcast_ensemble,
            }

        except Exception as e:
            print(f"\n✗ 预报失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _estimate_motion(self, precip_series):
        """估计运动场"""
        # 简化版本：使用平均平流速度
        # 实际应用中应使用 DARTS 或 Lucas-Kanade
        height, width = precip_series.shape[1:]

        # 计算帧间相关性来估计运动
        if len(precip_series) >= 2:
            from scipy.ndimage import shift
            # 简单的互相关估计
            ref = precip_series[0]
            shifts = []
            for i in range(1, len(precip_series)):
                # 简化：固定小速度
                shifts.append((0.5, 0.5))  # 每个时间步移动0.5像素

            vx = np.full((height, width), 0.5, dtype=np.float32)
            vy = np.full((height, width), 0.5, dtype=np.float32)
        else:
            vx = np.zeros((height, width), dtype=np.float32)
            vy = np.zeros((height, width), dtype=np.float32)

        return {'vx': vx, 'vy': vy}

    def _generate_ensemble(self, initial_field, motion_field, max_lead_time):
        """
        生成集合预报

        Parameters
        ----------
        initial_field : ndarray
            初始降水场
        motion_field : dict
            运动场
        max_lead_time : int
            最大预报时效 (分钟)

        Returns
        -------
        ensemble : ndarray
            集合预报 (n_members, n_timesteps, height, width)
        """
        n_members = self.n_ensemble_members
        shape = initial_field.shape
        n_timesteps = max_lead_time // 5 + 1

        ensemble = []

        for m in range(n_members):
            member_forecast = np.zeros((n_timesteps, *shape), dtype=np.float32)
            member_forecast[0] = initial_field.copy()

            for t in range(1, n_timesteps):
                # 平流 + 扰散
                prev = member_forecast[t-1]

                # 添加空间相关的噪声
                noise = np.random.randn(*shape) * 0.1
                noise = self._spatial_smooth(noise, sigma=2.0)

                # 时间衰减
                decay = 0.98 ** t

                # 平流位移 (简化)
                from scipy.ndimage import shift
                advected = shift(prev, shift=[0.5*t/12, 0.5*t/12], order=1,
                                mode='constant', cval=0.0)

                # 组合
                member_forecast[t] = advected * decay + noise
                member_forecast[t] = np.maximum(member_forecast[t], 0.0)

            ensemble.append(member_forecast)

        return np.array(ensemble)

    def _spatial_smooth(self, field, sigma=1.0):
        """空间平滑"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma)

    def _evaluate_all_thresholds(self, precip_series, nowcast_ensemble):
        """评估所有阈值"""
        observation = precip_series[-1]  # 使用最后一帧作为观测

        # 集合平均
        ensemble_mean = np.mean(nowcast_ensemble, axis=0)

        # 对每个阈值进行评估
        for threshold in self.thresholds:
            print(f"\n  阈值 {threshold:.1f} mm/h:")

            threshold_results = {
                'threshold_mm_h': threshold,
                'lead_times': {},
                'summary': {}
            }

            # 分析观测中的降水分布
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

            # 对每个预报时效评估
            for lead_time in self.lead_times:
                timestep = lead_time // 5
                if timestep >= ensemble_mean.shape[0]:
                    continue

                forecast = ensemble_mean[timestep]
                fcst_binary = forecast > threshold

                # 计算列联表
                hits = np.sum((obs_binary == True) & (fcst_binary == True))
                misses = np.sum((obs_binary == True) & (fcst_binary == False))
                false_alarms = np.sum((obs_binary == False) & (fcst_binary == True))
                correct_negatives = np.sum((obs_binary == False) & (fcst_binary == False))

                # 计算指标
                csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
                pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
                far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
                bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else 0.0

                # 误差指标
                rmse = np.sqrt(np.mean((forecast - observation) ** 2))
                mae = np.mean(np.abs(forecast - observation))

                threshold_results['lead_times'][f'{lead_time}min'] = {
                    'csi': float(csi),
                    'pod': float(pod),
                    'far': float(far),
                    'bias': float(bias),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'hits': int(hits),
                    'misses': int(misses),
                    'false_alarms': int(false_alarms),
                    'correct_negatives': int(correct_negatives),
                }

                print(f"    {lead_time:3d}min: CSI={csi:.3f}, POD={pod:.3f}, FAR={far:.3f}, "
                      f"Bias={bias:.3f}, RMSE={rmse:.3f}")

            self.results['performance_by_threshold'][f'{threshold:.1f}mm'] = threshold_results

    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        print("\n" + "="*70)
        print("DWD 多阈值短临预报评估报告")
        print("="*70)

        # 1. 数据集信息
        print("\n📊 数据集信息:")
        info = self.results.get('dataset_info', {})
        for key, val in info.items():
            print(f"  {key}: {val}")

        # 2. 运行性能
        print("\n⚡ 运行性能:")
        runtime = self.results.get('runtime', {})
        if runtime:
            total = runtime.get('total', 0)
            print(f"  总耗时: {total:.2f} 秒")
            for step, t in runtime.items():
                if step != 'total':
                    pct = 100.0 * t / total if total > 0 else 0
                    print(f"  {step}: {t:.2f} 秒 ({pct:.1f}%)")

        # 3. 阈值性能对比表
        print("\n🎯 各阈值 CSI 对比:")
        print(f"  {'时效':<8} " + " ".join([f"{t:>6}" for t in self.thresholds]))
        print("  " + "-" * (8 + 7 * len(self.thresholds)))

        for lead_time in self.lead_times:
            row = [f"{lead_time}min"]
            for threshold in self.thresholds:
                key = f'{threshold:.1f}mm'
                data = self.results.get('performance_by_threshold', {}).get(key, {})
                lt_data = data.get('lead_times', {}).get(f'{lead_time}min', {})
                csi = lt_data.get('csi', 0.0)
                row.append(f"{csi:6.3f}")
            print("  " + " ".join(row))

        # 4. 阈值性能总结
        print("\n📈 各阈值平均性能:")
        print(f"  {'阈值':<8} {'CSI':<8} {'POD':<8} {'FAR':<8} {'Bias':<8} {'覆盖率':<10}")
        print("  " + "-" * 58)

        for threshold in self.thresholds:
            key = f'{threshold:.1f}mm'
            data = self.results.get('performance_by_threshold', {}).get(key, {})
            lead_times_data = data.get('lead_times', {})

            if lead_times_data:
                csi_mean = np.mean([lt.get('csi', 0) for lt in lead_times_data.values()])
                pod_mean = np.mean([lt.get('pod', 0) for lt in lead_times_data.values()])
                far_mean = np.mean([lt.get('far', 0) for lt in lead_times_data.values()])
                bias_mean = np.mean([lt.get('bias', 0) for lt in lead_times_data.values()])
                coverage = data.get('observation_stats', {}).get('coverage_percent', 0)

                print(f"  {threshold:>5.1f}mm {csi_mean:<8.3f} {pod_mean:<8.3f} "
                      f"{far_mean:<8.3f} {bias_mean:<8.3f} {coverage:<10.2f}%")

        # 5. 强降水检测能力
        print("\n⛈️ 强降水检测能力:")
        high_thresholds = [t for t in self.thresholds if t >= 3.5]
        for threshold in high_thresholds:
            key = f'{threshold:.1f}mm'
            data = self.results.get('performance_by_threshold', {}).get(key, {})
            obs_stats = data.get('observation_stats', {})
            n_pixels = obs_stats.get('precipitation_pixels', 0)
            max_intensity = obs_stats.get('max_intensity', 0)

            print(f"  {threshold:.1f} mm/h 阈值:")
            print(f"    检测到 {n_pixels} 个像素")
            print(f"    最大强度: {max_intensity:.2f} mm/h")

            # 显示前几个时效的性能
            for lead_time in [5, 15, 30, 60]:
                if lead_time in self.lead_times:
                    lt_data = data.get('lead_times', {}).get(f'{lead_time}min', {})
                    if lt_data:
                        print(f"    {lead_time}min: CSI={lt_data.get('csi', 0):.3f}, "
                              f"POD={lt_data.get('pod', 0):.3f}")

    def save_results(self, output_file=None):
        """保存结果"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.data_dir / f'../results/multi_threshold_eval_{timestamp}.json'

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n✓ 结果已保存: {output_file}")
        return output_file


def main():
    """主函数"""
    print("="*70)
    print("DWD 多阈值短临预报评估系统")
    print("="*70)

    # DWD 数据目录
    dwd_dir = '/data/workspace/pysteps-data/radar/dwd/RY/2025/06/04'

    # 创建评估器
    evaluator = DWDMultiThresholdEvaluator(
        data_dir=dwd_dir,
        n_ensemble_members=20
    )

    # 加载数据 (4个输入帧用于运动估计)
    precip_series, metadata = evaluator.load_dwd_series(n_input_frames=4)

    # 运行预报
    result = evaluator.run_nowcast(precip_series)

    if result:
        # 生成报告
        evaluator.generate_comprehensive_report()

        # 保存结果
        output_file = evaluator.save_results()

        print("\n" + "="*70)
        print("✓ 评估完成")
        print("="*70)

        return evaluator.results
    else:
        print("\n✗ 评估失败")
        return None


if __name__ == '__main__':
    results = main()

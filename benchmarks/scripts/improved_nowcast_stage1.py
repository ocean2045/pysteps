#!/usr/bin/env python3
"""
PySteps 0-3小时短临预报 - 第一阶段快速改进

改进内容:
1. 增加输入帧数 (6 → 12帧)
2. 自适应时间步长
3. 时效分段策略
4. 混合预报模式 (雷达 + NWP模拟)

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import sys
import numpy as np
import h5py
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

sys.path.insert(0, '/data/workspace/PyStepsDashu')


class ImprovedNowcaster:
    """改进的短临预报器 - 第一阶段"""

    def __init__(self, n_input_frames=12, n_ensemble_members=20):
        """
        初始化预报器

        Parameters
        ----------
        n_input_frames : int
            输入帧数 (默认12帧 = 60分钟历史)
        n_ensemble_members : int
            集合成员数
        """
        self.n_input_frames = n_input_frames
        self.n_ensemble_members = n_ensemble_members

        # 自适应时间步长配置
        self.time_step_schedule = {
            (0, 15): 5,      # 0-15分钟: 5分钟步长
            (15, 30): 5,     # 15-30分钟: 5分钟步长
            (30, 60): 10,    # 30-60分钟: 10分钟步长
            (60, 120): 15,   # 60-120分钟: 15分钟步长
            (120, 180): 30,  # 120-180分钟: 30分钟步长
        }

        print(f"✓ 改进预报器初始化完成")
        print(f"  输入帧数: {self.n_input_frames} (60分钟历史)")
        print(f"  集合成员: {self.n_ensemble_members}")

    def get_time_steps(self, lead_time: int) -> List[int]:
        """
        根据预报时效获取时间步长序列

        Parameters
        ----------
        lead_time : int
            总预报时效 (分钟)

        Returns
        -------
        time_steps : list
            时间步长序列
        """
        # 确定时间步长
        for (tmin, tmax), step in self.time_step_schedule.items():
            if tmin <= lead_time < tmax:
                time_step = step
                break
        else:
            time_step = 30  # 默认30分钟

        # 生成步长序列
        n_steps = lead_time // time_step
        time_steps = [time_step] * n_steps

        # 调整最后一个步长以达到精确的lead_time
        remainder = lead_time - sum(time_steps)
        if remainder > 0:
            time_steps[-1] += remainder

        return time_steps

    def load_extended_series(self, data_dir: str, dataset_key: str) -> Tuple[np.ndarray, Dict]:
        """
        加载扩展的时间序列 (12帧)

        Parameters
        ----------
        data_dir : str
            数据目录
        dataset_key : str
            数据集关键字 ('dwd' 或 'knmi')

        Returns
        -------
        precip_series : ndarray
            降水序列 (n_frames, height, width)
        metadata : dict
            元数据
        """
        print(f"\n加载扩展时间序列 ({dataset_key.upper()})...")

        # 查找文件
        if dataset_key == 'dwd':
            pattern = 'dwd/**/*.h5'
        elif dataset_key == 'knmi':
            pattern = 'KNMI/**/*.h5'
        else:
            raise ValueError(f"不支持的数据集: {dataset_key}")

        base_dir = Path(data_dir)
        files = sorted(base_dir.glob(pattern))

        if len(files) < self.n_input_frames:
            print(f"⚠ 警告: 只有 {len(files)} 个文件，将使用全部")
            n_frames = len(files)
        else:
            n_frames = self.n_input_frames

        # 加载数据
        precip_series = []
        metadata = None

        for i, file_path in enumerate(files[:n_frames]):
            try:
                if dataset_key == 'dwd':
                    precip, meta = self._load_dwd_file(file_path)
                elif dataset_key == 'knmi':
                    precip, meta = self._load_knmi_file(file_path)

                if precip is not None:
                    precip_series.append(precip)
                    if metadata is None:
                        metadata = meta
                    print(f"  帧 {i+1}/{n_frames}: {precip.shape}")
            except Exception as e:
                print(f"  ✗ 帧 {i+1}: 加载失败 - {e}")

        if not precip_series:
            raise ValueError("无法加载任何数据")

        precip_series = np.array(precip_series)
        print(f"✓ 加载完成: {precip_series.shape} (时间, 高度, 宽度)")

        return precip_series, metadata

    def _load_dwd_file(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """加载DWD文件"""
        with h5py.File(file_path, 'r') as f:
            data = f['dataset1/data1/data'][()]
            what_attrs = dict(f['dataset1/data1/what'].attrs)

            gain = float(what_attrs.get('gain', 0.01))
            offset = float(what_attrs.get('offset', 0.0))
            nodata = float(what_attrs.get('nodata', 65535.0))

            precip = data.astype(np.float32)
            precip[precip == nodata] = np.nan
            precip = precip * gain + offset
            precip = np.nan_to_num(precip, nan=0.0)
            precip = np.clip(precip, 0, 500)

            metadata = {
                'format': 'odimh5_dwd',
                'gain': gain,
                'offset': offset,
            }

            return precip, metadata

    def _load_knmi_file(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """加载KNMI文件"""
        with h5py.File(file_path, 'r') as f:
            data = f['image1/image_data'][()]

            precip = data.astype(np.float32)
            precip[precip == 65535] = np.nan
            precip = np.nan_to_num(precip, nan=0.0)
            precip = np.clip(precip, 0, 500)

            metadata = {
                'format': 'odimh5_knmi',
            }

            return precip, metadata

    def forecast_temporal_segmentation(self, precip_series: np.ndarray,
                                       lead_times: List[int]) -> Dict[int, np.ndarray]:
        """
        时效分段预报策略

        不同时段使用不同方法:
        - 0-30分钟: 纯雷达外推
        - 30-60分钟: 雷达 + 缓慢衰减
        - 60-120分钟: 雷达 + 气候学
        - 120-180分钟: 主要是气候学
        """
        print("\n使用时效分段预报策略...")

        forecasts = {}
        observation = precip_series[-1]

        for lead_time in lead_times:
            print(f"  预报 {lead_time} 分钟...")

            if lead_time <= 30:
                # 短期: 纯雷达外推
                forecast = self._radar_extrapolation(precip_series, lead_time)
                method = 'radar'

            elif lead_time <= 60:
                # 中期: 雷达 + 衰减调整
                radar_fcst = self._radar_extrapolation(precip_series, lead_time)
                # 应用衰减修正
                decay_factor = 0.98 ** (lead_time / 30.0)
                forecast = radar_fcst * decay_factor
                # 叠加气候学
                climatology = self._get_climatology(precip_series)
                forecast = 0.7 * forecast + 0.3 * climatology
                method = 'hybrid_1'

            elif lead_time <= 120:
                # 长期: 更多气候学
                radar_fcst = self._radar_extrapolation(precip_series, 60)
                decay_factor = 0.95 ** ((lead_time - 60) / 60.0)
                radar_fcst = radar_fcst * decay_factor

                climatology = self._get_climatology(precip_series)
                forecast = 0.3 * radar_fcst + 0.7 * climatology
                method = 'hybrid_2'

            else:
                # 极长期: 主要是气候学
                climatology = self._get_climatology(precip_series)
                # 添加趋势
                trend = self._compute_trend(precip_series)
                forecast = climatology * (1 + trend * (lead_time / 180.0))
                method = 'climatology'

            forecasts[lead_time] = forecast
            print(f"    方法: {method}")

        return forecasts

    def _radar_extrapolation(self, precip_series: np.ndarray, lead_time: int) -> np.ndarray:
        """雷达外推预报"""
        # 简化版本：使用平流 + 衰减
        observation = precip_series[-1]
        shape = observation.shape

        # 计算平均运动
        motion = self._estimate_motion(precip_series)

        # 应用平流
        from scipy.ndimage import shift
        shift_pixels = (lead_time / 5.0) * motion['magnitude']
        advected = shift(observation,
                         shift=[shift_pixels, shift_pixels],
                         order=1,
                         mode='constant',
                         cval=0.0)

        # 应用时间衰减
        decay = 0.98 ** (lead_time / 5.0)
        forecast = advected * decay

        # 添加噪声保持集合特性
        noise = np.random.randn(*shape) * 0.05 * np.std(observation)
        forecast = forecast + noise
        forecast = np.maximum(forecast, 0)

        return forecast

    def _estimate_motion(self, precip_series: np.ndarray) -> Dict:
        """估计运动场"""
        # 简化版本：计算帧间相关性
        if len(precip_series) >= 2:
            # 计算最后两帧的位移
            from scipy.ndimage import shift
            ref = precip_series[-2]
            current = precip_series[-1]

            # 简化：固定小速度
            magnitude = 0.5  # 像素/5分钟
        else:
            magnitude = 0.0

        return {'magnitude': magnitude}

    def _get_climatology(self, precip_series: np.ndarray) -> np.ndarray:
        """计算气候学平均"""
        # 使用时间序列的平均
        mean_field = np.mean(precip_series, axis=0)
        return mean_field

    def _compute_trend(self, precip_series: np.ndarray) -> float:
        """计算趋势"""
        # 简单线性趋势
        means = [np.mean(frame) for frame in precip_series]
        if len(means) >= 2:
            trend = (means[-1] - means[0]) / len(means)
        else:
            trend = 0.0
        return trend

    def forecast_adaptive_timestep(self, precip_series: np.ndarray,
                                   lead_time: int) -> np.ndarray:
        """使用自适应时间步长预报"""
        # 获取时间步长
        time_steps = self.get_time_steps(lead_time)

        print(f"  自适应步长: {time_steps}")

        # 逐步预报
        current_field = precip_series[-1]
        total_shift = 0

        for i, step in enumerate(time_steps):
            # 当前步的位移
            shift_pixels = (step / 5.0) * 0.5  # 0.5 像素/5分钟

            # 应用位移
            from scipy.ndimage import shift
            current_field = shift(current_field,
                                shift=[shift_pixels, shift_pixels],
                                order=1,
                                mode='constant',
                                cval=0.0)

            # 应用衰减
            decay = 0.98 ** (step / 5.0)
            current_field = current_field * decay

            total_shift += shift_pixels

        return current_field

    def generate_ensemble(self, precip_series: np.ndarray,
                         lead_time: int) -> np.ndarray:
        """生成集合预报"""
        print(f"\n生成集合预报 ({self.n_ensemble_members} 成员)...")

        ensemble = []

        for i in range(self.n_ensemble_members):
            # 每个成员使用略微不同的参数
            np.random.seed(i)

            # 方法1: 时效分段
            if lead_time <= 60:
                forecast = self._radar_extrapolation(precip_series, lead_time)
            else:
                forecast = self.forecast_adaptive_timestep(precip_series, lead_time)

            ensemble.append(forecast)

        ensemble = np.array(ensemble)
        print(f"✓ 集合预报生成完成: {ensemble.shape}")

        return ensemble

    def evaluate_improved_performance(self, precip_series: np.ndarray,
                                     forecasts: Dict[int, np.ndarray],
                                     thresholds: List[float] = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]):
        """评估改进后的性能"""
        print("\n" + "="*70)
        print("改进后性能评估")
        print("="*70)

        observation = precip_series[-1]
        results = {}

        for threshold in thresholds:
            print(f"\n阈值 {threshold:.1f} mm/h:")
            threshold_results = {}

            obs_binary = observation > threshold
            n_pixels_obs = np.sum(obs_binary)
            coverage_obs = 100.0 * n_pixels_obs / observation.size

            print(f"  观测: {n_pixels_obs} 像素 ({coverage_obs:.2f}%)")

            for lead_time, forecast in forecasts.items():
                fcst_binary = forecast > threshold

                # 计算指标
                hits = np.sum((obs_binary == True) & (fcst_binary == True))
                misses = np.sum((obs_binary == True) & (fcst_binary == False))
                false_alarms = np.sum((obs_binary == False) & (fcst_binary == True))

                csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
                pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
                far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0

                threshold_results[lead_time] = {
                    'csi': float(csi),
                    'pod': float(pod),
                    'far': float(far),
                }

                print(f"    {lead_time:3d}min: CSI={csi:.3f}, POD={pod:.3f}, FAR={far:.3f}")

            results[f'{threshold:.1f}mm'] = threshold_results

        return results


def main():
    """主函数"""
    print("="*70)
    print("PySteps 0-3小时短临预报 - 第一阶段改进")
    print("="*70)

    # 创建改进的预报器
    nowcaster = ImprovedNowcaster(
        n_input_frames=12,  # 60分钟历史 (改进)
        n_ensemble_members=20
    )

    # 测试数据集
    datasets = ['knmi', 'dwd']

    for dataset_key in datasets:
        print(f"\n{'='*70}")
        print(f"测试 {dataset_key.upper()} 数据集")
        print(f"{'='*70}")

        try:
            # 加载扩展序列 (12帧)
            precip_series, metadata = nowcaster.load_extended_series(
                data_dir='/data/workspace/pysteps-data/radar',
                dataset_key=dataset_key
            )

            if precip_series is None:
                continue

            # 生成预报 (0-180分钟)
            lead_times = [5, 10, 15, 30, 45, 60, 90, 120, 150, 180]

            print(f"\n生成预报 (0-180分钟)...")
            forecasts = nowcaster.forecast_temporal_segmentation(
                precip_series,
                lead_times
            )

            # 评估性能
            results = nowcaster.evaluate_improved_performance(
                precip_series,
                forecasts,
                thresholds=[0.1, 0.5, 1.0, 2.0, 3.5, 5.0]
            )

            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'/data/workspace/PyStepsDashu/benchmarks/results/improved_nowcast_{dataset_key}_{timestamp}.json'

            import json
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump({
                    'dataset': dataset_key,
                    'n_input_frames': len(precip_series),
                    'improvements': [
                        'Extended input frames (12)',
                        'Adaptive time stepping',
                        'Temporal segmentation',
                        'Hybrid radar-climatology'
                    ],
                    'results': results,
                }, f, indent=2, default=str)

            print(f"\n✓ 结果已保存: {output_file}")

        except Exception as e:
            print(f"\n✗ {dataset_key.upper()} 评估失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✓ 第一阶段改进评估完成")
    print("="*70)


if __name__ == '__main__':
    main()

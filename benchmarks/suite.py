#!/usr/bin/env python3
"""
PySteps 性能基准测试套件

提供一组标准化的基准测试，用于验证优化效果和检测性能回归。

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import sys
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Callable
from datetime import datetime


class Benchmark:
    """基准测试基类"""

    def __init__(self, name: str, repeats: int = 10, warmup: int = 2):
        self.name = name
        self.repeats = repeats
        self.warmup = warmup
        self.results = []

    def setup(self):
        """设置测试环境（子类实现）"""
        raise NotImplementedError

    def run(self):
        """运行测试（子类实现）"""
        raise NotImplementedError

    def teardown(self):
        """清理测试环境（子类可选）"""
        pass

    def execute(self, **config) -> Dict[str, Any]:
        """执行基准测试"""
        print(f"\n{'='*60}")
        print(f"基准测试: {self.name}")
        print(f"配置: {config}")
        print(f"{'='*60}")

        # 设置
        self.setup()

        # 预热
        print(f"预热 ({self.warmup} 次)...")
        for _ in range(self.warmup):
            self.run()

        # 正式测试
        print(f"测试 ({self.repeats} 次)...")
        times = []
        memories = []

        for i in range(self.repeats):
            start = time.time()
            result = self.run()
            elapsed = time.time() - start

            times.append(elapsed)
            print(f"  运行 {i+1}/{self.repeats}: {elapsed*1000:.2f} ms")

        # 清理
        self.teardown()

        # 统计
        times_ms = [t * 1000 for t in times]
        result_stats = {
            'test_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results': {
                'time_median_ms': float(np.median(times_ms)),
                'time_mean_ms': float(np.mean(times_ms)),
                'time_std_ms': float(np.std(times_ms)),
                'time_min_ms': float(np.min(times_ms)),
                'time_max_ms': float(np.max(times_ms)),
                'iterations': self.repeats,
            }
        }

        print(f"\n结果统计:")
        print(f"  中位数: {result_stats['results']['time_median_ms']:.2f} ms")
        print(f"  平均值: {result_stats['results']['time_mean_ms']:.2f} ms")
        print(f"  标准差: {result_stats['results']['time_std_ms']:.2f} ms")
        print(f"  最小值: {result_stats['results']['time_min_ms']:.2f} ms")
        print(f"  最大值: {result_stats['results']['time_max_ms']:.2f} ms")

        return result_stats


class BenchmarkEnsembleSpread(Benchmark):
    """集合扩展计算基准测试"""

    def __init__(self):
        super().__init__("ensemble_spread")

    def setup(self):
        """生成测试数据"""
        sys.path.insert(0, '/data/workspace/PyStepsDashu')
        from pysteps.verification.enscores import ensemble_spread

        self.ensemble_spread = ensemble_spread

    def run(self):
        """运行测试"""
        # 生成100成员的集合预报
        np.random.seed(42)
        n_members = 100
        shape = (100, 100)

        X_f = np.random.rand(n_members, *shape)

        # 计算集合扩展
        spread = self.ensemble_spread(X_f, metric='MSE')

        return spread


class BenchmarkSpectralFit(Benchmark):
    """光谱斜率拟合基准测试"""

    def __init__(self, mode='repeat'):
        super().__init__(f"spectral_fit_{mode}")
        self.mode = mode

    def setup(self):
        """生成测试数据"""
        sys.path.insert(0, '/data/workspace/PyStepsDashu')
        from pysteps.noise.fftgenerators import (
            _fit_spectral_slope_cached,
            clear_spectral_fit_cache
        )

        self.fit_cached = _fit_spectral_slope_cached
        self.clear_cache = clear_spectral_fit_cache

        # 生成固定的测试数据
        np.random.seed(42)
        L = 100
        self.wn = np.arange(0, int(L / 2))
        psd = np.exp(-2.0 * np.log(self.wn[1:])) * (1 + 0.1 * np.random.randn(len(self.wn)-1))
        self.psd = np.concatenate([[1.0], psd])

        if self.mode == 'repeat':
            # 预热缓存
            _ = self.fit_cached(tuple(self.wn), tuple(self.psd), False)

    def run(self):
        """运行测试"""
        if self.mode == 'first':
            self.clear_cache()

        result = self.fit_cached(tuple(self.wn), tuple(self.psd), False)
        return result


class BenchmarkARParams(Benchmark):
    """AR模型参数计算基准测试"""

    def __init__(self, n_pixels=4096):
        super().__init__(f"ar_params_{n_pixels}")
        self.n_pixels = n_pixels

    def setup(self):
        """生成测试数据"""
        sys.path.insert(0, '/data/workspace/PyStepsDashu')
        from pysteps.timeseries.autoregression import estimate_ar_params_yw_localized

        self.estimate_ar = estimate_ar_params_yw_localized

        # 生成测试数据
        np.random.seed(42)
        p = 3
        shape = (int(np.sqrt(self.n_pixels)), int(np.sqrt(self.n_pixels)))

        self.gamma = [np.random.rand(*shape) * 0.5 for _ in range(p)]

    def run(self):
        """运行测试"""
        result = self.estimate_ar(self.gamma, d=0)
        return result


class BenchmarkDARTS(Benchmark):
    """DARTS运动估计基准测试"""

    def __init__(self, size='medium'):
        super().__init__(f"darts_{size}")
        self.size = size

        # 配置
        self.configs = {
            'small': (2, 2, 0, 2, 2),
            'medium': (3, 3, 0, 3, 3),
            'large': (4, 4, 1, 4, 4),
        }
        self.N_x, self.N_y, self.N_t, self.M_x, self.M_y = self.configs[size]

    def setup(self):
        """生成测试数据"""
        import importlib.util

        # 导入优化后的 DARTS 函数
        spec = importlib.util.spec_from_file_location(
            "test_darts",
            "/data/workspace/PyStepsDashu/test_darts_optimization.py"
        )
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)

        self.construct_h = test_module.construct_h_matrix_fully_vectorized

        # 生成测试数据
        np.random.seed(42)
        shape = (2 * self.N_y + 1, 2 * self.N_x + 1, 2 * self.N_t + 1)
        self.input_images = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    def run(self):
        """运行测试"""
        result = self.construct_h(
            self.input_images,
            self.N_x, self.N_y, self.N_t,
            self.M_x, self.M_y,
            1.0, 1.0, 1.0
        )
        return result


# 基准测试套件
BENCHMARK_SUITE = {
    'ensemble_spread_100': BenchmarkEnsembleSpread(),
    'spectral_fit_first': BenchmarkSpectralFit(mode='first'),
    'spectral_fit_repeat': BenchmarkSpectralFit(mode='repeat'),
    'ar_params_4k': BenchmarkARParams(n_pixels=4096),
    'ar_params_16k': BenchmarkARParams(n_pixels=16384),
    'darts_small': BenchmarkDARTS(size='small'),
    'darts_medium': BenchmarkDARTS(size='medium'),
    'darts_large': BenchmarkDARTS(size='large'),
}


def run_benchmark(name: str, **config) -> Dict[str, Any]:
    """运行单个基准测试"""
    if name not in BENCHMARK_SUITE:
        raise ValueError(f"未知基准测试: {name}")

    benchmark = BENCHMARK_SUITE[name]
    return benchmark.execute(**config)


def run_all_benchmarks(tests: List[str] = None) -> List[Dict[str, Any]]:
    """运行所有基准测试"""
    if tests is None:
        tests = list(BENCHMARK_SUITE.keys())

    results = []
    for test_name in tests:
        try:
            result = run_benchmark(test_name)
            results.append(result)
        except Exception as e:
            print(f"\n✗ 测试失败: {test_name}")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    return results


def save_results(results: List[Dict[str, Any]], output_dir: str = None):
    """保存结果到文件"""
    if output_dir is None:
        output_dir = '/data/workspace/PyStepsDashu/benchmarks/results/current'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存 JSON
    json_file = Path(output_dir) / 'benchmark_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {json_file}")

    # 保存 CSV
    import csv
    csv_file = Path(output_dir) / 'benchmark_results.csv'

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'test_name', 'time_median_ms', 'time_mean_ms', 'time_std_ms',
            'time_min_ms', 'time_max_ms', 'iterations'
        ])

        for result in results:
            stats = result['results']
            writer.writerow([
                result['test_name'],
                stats['time_median_ms'],
                stats['time_mean_ms'],
                stats['time_std_ms'],
                stats['time_min_ms'],
                stats['time_max_ms'],
                stats['iterations'],
            ])

    print(f"CSV报告: {csv_file}")


if __name__ == '__main__':
    import sys

    # 运行所有基准测试
    print("="*70)
    print("PySteps 性能基准测试套件")
    print("="*70)

    results = run_all_benchmarks()

    # 保存结果
    save_results(results)

    print("\n" + "="*70)
    print(f"✓ 完成 {len(results)} 个基准测试")
    print("="*70)

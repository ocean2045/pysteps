#!/usr/bin/env python3
"""
FFT批量计算优化 - 改进版本

问题分析:
1. 原始axes方法在某些情况下反而更慢（内存分配开销）
2. 单核环境下axes参数无法并行化
3. 需要更智能的优化策略

改进方案:
1. 对于少量帧(<10): 使用原始循环（减少内存开销）
2. 对于大量帧(>=10): 使用向量化方法
3. 使用scipy.fft的workers参数进行真正的并行

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
import time
from scipy import fft


def fft_batch_original(field, tapering):
    """原始实现"""
    nr_fields, M, N = field.shape
    F = np.zeros((M, N), dtype=complex)

    for i in range(nr_fields):
        F += fft.fftshift(fft.fft2(field[i] * tapering))

    F /= nr_fields
    F = abs(F) ** 2 / F.size

    return F


def fft_batch_vectorized(field, tapering):
    """向量化实现"""
    field_tapered = field * tapering[np.newaxis, :, :]
    F_all = fft.fft2(field_tapered, axes=(1, 2))
    F_shifted = fft.fftshift(F_all, axes=(1, 2))
    F_avg = F_shifted.mean(axis=0)
    F_psd = abs(F_avg) ** 2 / F_avg.size
    return F_psd


def fft_batch_adaptive(field, tapering):
    """
    自适应实现 - 根据数据大小选择最优方法

    对于小数据集: 使用原始循环（避免内存分配开销）
    对于大数据集: 使用向量化方法（利用并行化）
    """
    nr_fields = field.shape[0]

    # 阈值: 10帧或总像素>20000时使用向量化
    total_pixels = field[0].size
    if nr_fields >= 10 or total_pixels > 20000:
        # 使用向量化方法
        field_tapered = field * tapering[np.newaxis, :, :]
        F_all = fft.fft2(field_tapered, axes=(1, 2))
        F_shifted = fft.fftshift(F_all, axes=(1, 2))
        F_avg = F_shifted.mean(axis=0)
        F_psd = abs(F_avg) ** 2 / F_avg.size
        return F_psd
    else:
        # 使用原始循环方法
        F = np.zeros(field.shape[1:], dtype=complex)
        for i in range(nr_fields):
            F += fft.fftshift(fft.fft2(field[i] * tapering))
        F /= nr_fields
        F = abs(F) ** 2 / F.size
        return F


def fft_batch_parallel(field, tapering):
    """
    并行化版本 - 使用scipy.fft的workers参数

    注意: 这需要多核CPU才能看到性能提升
    """
    nr_fields = field.shape[0]

    field_tapered = field * tapering[np.newaxis, :, :]

    # 使用workers参数进行并行化
    try:
        F_all = fft.fft2(field_tapered, axes=(1, 2), workers=-1)
    except TypeError:
        # 旧版本scipy不支持workers参数
        F_all = fft.fft2(field_tapered, axes=(1, 2))

    F_shifted = fft.fftshift(F_all, axes=(1, 2))
    F_avg = F_shifted.mean(axis=0)
    F_psd = abs(F_avg) ** 2 / F_avg.size

    return F_psd


def benchmark_all_methods():
    """对比所有方法的性能"""
    print("="*70)
    print("FFT批量计算优化 - 全面对比测试")
    print("="*70)

    test_cases = [
        # (nr_fields, M, N, label)
        (3, 64, 64, "极小数据集"),
        (5, 64, 64, "小数据集"),
        (10, 64, 64, "中等数据集(小)"),
        (10, 128, 128, "中等数据集(大)"),
        (20, 128, 128, "大数据集"),
        (50, 128, 128, "超大数据集"),
    ]

    n_repeats = 10

    print(f"\n{'配置':<20} {'原始':<12} {'向量化':<12} {'自适应':<12} {'并行':<12}")
    print("-"*70)

    results = []

    for nr_fields, M, N, label in test_cases:
        np.random.seed(42)
        field = np.random.rand(nr_fields, M, N)
        tapering = np.ones((M, N))

        # 预热
        _ = fft_batch_original(field.copy(), tapering)
        _ = fft_batch_vectorized(field.copy(), tapering)
        _ = fft_batch_adaptive(field.copy(), tapering)
        _ = fft_batch_parallel(field.copy(), tapering)

        # 原始方法
        times_orig = []
        for _ in range(n_repeats):
            start = time.time()
            _ = fft_batch_original(field.copy(), tapering)
            times_orig.append(time.time() - start)
        time_orig = np.median(times_orig)

        # 向量化方法
        times_vec = []
        for _ in range(n_repeats):
            start = time.time()
            _ = fft_batch_vectorized(field.copy(), tapering)
            times_vec.append(time.time() - start)
        time_vec = np.median(times_vec)

        # 自适应方法
        times_adapt = []
        for _ in range(n_repeats):
            start = time.time()
            _ = fft_batch_adaptive(field.copy(), tapering)
            times_adapt.append(time.time() - start)
        time_adapt = np.median(times_adapt)

        # 并行方法
        times_para = []
        for _ in range(n_repeats):
            start = time.time()
            _ = fft_batch_parallel(field.copy(), tapering)
            times_para.append(time.time() - start)
        time_para = np.median(times_para)

        speedup_vec = time_orig / time_vec
        speedup_adapt = time_orig / time_adapt
        speedup_para = time_orig / time_para

        print(f"{label:<20} {time_orig*1000:<12.2f} {time_vec*1000:<12.2f} "
              f"{time_adapt*1000:<12.2f} {time_para*1000:<12.2f}")
        print(f"{'加速比':<20} {1.0:<12.1f} {speedup_vec:<12.1f} "
              f"{speedup_adapt:<12.1f} {speedup_para:<12.1f}")
        print()

        results.append({
            'label': label,
            'time_orig': time_orig,
            'time_vec': time_vec,
            'time_adapt': time_adapt,
            'time_para': time_para,
            'speedup_adapt': speedup_adapt,
        })

    # 推荐最佳方法
    print("="*70)
    print("推荐策略:")
    print("="*70)

    adaptive_wins = sum(1 for r in results if r['time_adapt'] <= r['time_orig'] and
                        r['time_adapt'] <= r['time_vec'])
    total = len(results)

    print(f"\n自适应方法在 {total} 个测试中的 {adaptive_wins} 个表现最佳")

    if adaptive_wins >= total * 0.7:
        print("\n✓ 推荐: 使用自适应策略")
        print("  - 小数据集(<10帧): 使用原始循环")
        print("  - 大数据集(>=10帧): 使用向量化方法")
    else:
        # 检查并行方法
        parallel_wins = sum(1 for r in results if r['time_para'] <= r['time_orig'])
        if parallel_wins >= total * 0.7:
            print("\n✓ 推荐: 使用并行化方法 (需要多核CPU)")
        else:
            print("\n✓ 推荐: 保持原始实现")
            print("  - 向量化在此环境下没有明显优势")
            print("  - 可能需要GPU加速或其他优化")

    return results


if __name__ == "__main__":
    benchmark_all_methods()

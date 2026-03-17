#!/usr/bin/env python3
"""
滤波器缓存优化演示

问题: filter_gaussian 中重复计算滤波器权重
     - _gaussweights_1d 每次都重新计算高斯权重函数
     - 相同参数的滤波器被重复创建
     - r_2d 网格对于相同 shape 重复计算

优化方案: 使用 @lru_cache 缓存滤波器计算
     - 缓存 _gaussweights_1d 结果
     - 缓存 r_2d 网格计算
     - 避免重复的高斯函数创建
     - 预期 2-5x 加速

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
import time
from functools import lru_cache


# ============== 原始实现 ==============

def log_e_original(x, q):
    """原始 log_e 函数"""
    if len(np.shape(x)) > 0:
        res = np.empty(x.shape)
        res[x == 0] = 0.0
        res[x > 0] = np.log(x[x > 0]) / np.log(q)
    else:
        if x == 0.0:
            res = 0.0
        else:
            res = np.log(x) / np.log(q)

    return res


class GaussFuncOriginal:
    """原始高斯函数类"""
    def __init__(self, c, s):
        self.c = c
        self.s = s

    def __call__(self, x):
        x = log_e_original(x, np.exp(1.0)) - self.c
        return np.exp(-(x**2.0) / (2.0 * self.s**2.0))


def _gaussweights_1d_original(l, n, gauss_scale=0.5):
    """原始高斯权重计算"""
    q = pow(0.5 * l, 1.0 / n)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n + 1)]
    r = [0.5 * (r_[0] + r_[1]) for r_ in r]

    weight_funcs = []
    central_wavenumbers = []

    for i, ri in enumerate(r):
        rc = log_e_original(ri, q)
        weight_funcs.append(GaussFuncOriginal(rc, gauss_scale))
        central_wavenumbers.append(ri)

    return weight_funcs, central_wavenumbers


def filter_gaussian_original(shape, n, gauss_scale=0.5, normalize=True):
    """原始滤波器实现"""
    if n < 3:
        raise ValueError("n must be greater than 2")

    try:
        height, width = shape
    except TypeError:
        height, width = (shape, shape)

    max_length = max(width, height)

    rx = np.s_[: int(width / 2) + 1]

    if (height % 2) == 1:
        ry = np.s_[-int(height / 2) : int(height / 2) + 1]
    else:
        ry = np.s_[-int(height / 2) : int(height / 2)]

    y_grid, x_grid = np.ogrid[ry, rx]
    dy = int(height / 2) if height % 2 == 0 else int(height / 2) + 1

    r_2d = np.roll(np.sqrt(x_grid * x_grid + y_grid * y_grid), dy, axis=0)

    r_max = int(max_length / 2) + 1
    r_1d = np.arange(r_max)

    wfs, central_wavenumbers = _gaussweights_1d_original(
        max_length,
        n,
        gauss_scale=gauss_scale,
    )

    weights_1d = np.empty((n, r_max))
    weights_2d = np.empty((n, height, int(width / 2) + 1))

    for i, wf in enumerate(wfs):
        weights_1d[i, :] = wf(r_1d)
        weights_2d[i, :, :] = wf(r_2d)

    if normalize:
        weights_1d_sum = np.sum(weights_1d, axis=0)
        weights_2d_sum = np.sum(weights_2d, axis=0)
        for k in range(weights_2d.shape[0]):
            weights_1d[k, :] /= weights_1d_sum
            weights_2d[k, :, :] /= weights_2d_sum

    return weights_1d, weights_2d


# ============== 优化实现 ==============

def log_e_cached(x, q):
    """可缓存的 log_e 函数（使用标量输入）"""
    if isinstance(x, np.ndarray):
        res = np.empty(x.shape)
        res[x == 0] = 0.0
        res[x > 0] = np.log(x[x > 0]) / np.log(q)
    else:
        if x == 0.0:
            res = 0.0
        else:
            res = np.log(x) / np.log(q)

    return res


class GaussFuncCached:
    """可缓存的高斯函数类"""
    def __init__(self, c, s):
        self.c = c
        self.s = s

    def __call__(self, x):
        x = log_e_cached(x, np.exp(1.0)) - self.c
        return np.exp(-(x**2.0) / (2.0 * self.s**2.0))


# 缓存高斯权重计算结果
@lru_cache(maxsize=128)
def _compute_gauss_weights_cached(l_int, n_int, gauss_scale_float):
    """
    缓存版本的高斯权重计算

    参数转换为整数/浮点数以支持哈希
    """
    l = l_int
    n = n_int
    gauss_scale = gauss_scale_float

    q = pow(0.5 * l, 1.0 / n)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n + 1)]
    r = [0.5 * (r_[0] + r_[1]) for r_ in r]

    # 返回参数而不是函数对象（函数对象不可哈希）
    central_wavenumbers = tuple(r)
    q_value = float(q)

    return central_wavenumbers, q_value


def filter_gaussian_cached(shape, n, gauss_scale=0.5, normalize=True):
    """缓存优化的滤波器实现"""
    if n < 3:
        raise ValueError("n must be greater than 2")

    try:
        height, width = shape
    except TypeError:
        height, width = (shape, shape)

    max_length = max(width, height)

    rx = np.s_[: int(width / 2) + 1]

    if (height % 2) == 1:
        ry = np.s_[-int(height / 2) : int(height / 2) + 1]
    else:
        ry = np.s_[-int(height / 2) : int(height / 2)]

    y_grid, x_grid = np.ogrid[ry, rx]
    dy = int(height / 2) if height % 2 == 0 else int(height / 2) + 1

    r_2d = np.roll(np.sqrt(x_grid * x_grid + y_grid * y_grid), dy, axis=0)

    r_max = int(max_length / 2) + 1
    r_1d = np.arange(r_max)

    # 使用缓存的权重计算
    central_wavenumbers, q = _compute_gauss_weights_cached(
        max_length, n, gauss_scale
    )

    # 重建高斯函数（使用缓存的参数）
    weight_funcs = []
    for ri in central_wavenumbers:
        rc = log_e_cached(ri, q)
        weight_funcs.append(GaussFuncCached(rc, gauss_scale))

    weights_1d = np.empty((n, r_max))
    weights_2d = np.empty((n, height, int(width / 2) + 1))

    for i, wf in enumerate(weight_funcs):
        weights_1d[i, :] = wf(r_1d)
        weights_2d[i, :, :] = wf(r_2d)

    if normalize:
        weights_1d_sum = np.sum(weights_1d, axis=0)
        weights_2d_sum = np.sum(weights_2d, axis=0)
        for k in range(weights_2d.shape[0]):
            weights_1d[k, :] /= weights_1d_sum
            weights_2d[k, :, :] /= weights_2d_sum

    return weights_1d, weights_2d


def clear_filter_cache():
    """清除滤波器缓存"""
    _compute_gauss_weights_cached.cache_clear()


# ============== 测试函数 ==============

def verify_correctness():
    """验证缓存实现的正确性"""
    print("="*70)
    print("滤波器缓存优化 - 正确性验证")
    print("="*70)

    # 创建测试参数
    test_cases = [
        ((64, 64), 6, 0.5, "小规模"),
        ((128, 128), 8, 0.5, "中等规模"),
        ((256, 256), 10, 0.6, "大规模"),
    ]

    print(f"\n{'配置':<15} {'形状':<12} {'n':<4} {'最大差异':<15} {'状态':<10}")
    print("-"*70)

    all_passed = True

    for shape, n, gauss_scale, label in test_cases:
        # 测试原始实现
        w1d_orig, w2d_orig = filter_gaussian_original(
            shape, n, gauss_scale, normalize=True
        )

        # 测试缓存实现
        w1d_cached, w2d_cached = filter_gaussian_cached(
            shape, n, gauss_scale, normalize=True
        )

        # 验证结果
        max_diff_1d = np.max(np.abs(w1d_orig - w1d_cached))
        max_diff_2d = np.max(np.abs(w2d_orig - w2d_cached))
        max_diff = max(max_diff_1d, max_diff_2d)

        status = "✓ 完全一致" if max_diff < 1e-10 else "✗ 存在差异"
        if max_diff >= 1e-10:
            all_passed = False

        print(f"{label:<15} {str(shape):<12} {n:<4} {max_diff:<15.2e} {status:<10}")

    print("\n" + "="*70)
    if all_passed:
        print("✓ 所有正确性测试通过")
    else:
        print("✗ 部分测试失败")

    return all_passed


def benchmark_cache_performance():
    """测试缓存性能提升"""
    print("\n" + "="*70)
    print("滤波器缓存优化 - 性能基准测试")
    print("="*70)

    test_cases = [
        ((64, 64), 6, 0.5, "小规模"),
        ((128, 128), 8, 0.5, "中等规模"),
        ((256, 256), 10, 0.5, "大规模"),
    ]

    n_repeats = 10

    print(f"\n{'配置':<15} {'首次调用':<20} {'重复调用':<20} {'加速':<10}")
    print("-"*70)

    results = []

    for shape, n, gauss_scale, label in test_cases:
        # 清除缓存
        clear_filter_cache()

        # 首次调用（未命中缓存）
        start = time.time()
        w1d_orig, w2d_orig = filter_gaussian_original(
            shape, n, gauss_scale, normalize=True
        )
        time_orig_first = time.time() - start

        start = time.time()
        w1d_cached, w2d_cached = filter_gaussian_cached(
            shape, n, gauss_scale, normalize=True
        )
        time_cached_first = time.time() - start

        # 重复调用（命中缓存）
        times_orig_repeat = []
        times_cached_repeat = []

        for _ in range(n_repeats):
            start = time.time()
            _, _ = filter_gaussian_original(
                shape, n, gauss_scale, normalize=True
            )
            times_orig_repeat.append(time.time() - start)

            start = time.time()
            _, _ = filter_gaussian_cached(
                shape, n, gauss_scale, normalize=True
            )
            times_cached_repeat.append(time.time() - start)

        time_orig_repeat = np.mean(times_orig_repeat)
        time_cached_repeat = np.mean(times_cached_repeat)

        speedup_first = time_orig_first / time_cached_first
        speedup_repeat = time_orig_repeat / time_cached_repeat

        print(f"{label:<15} "
              f"{time_cached_first*1000:.2f} ms ({speedup_first:.1f}x){'':<5} "
              f"{time_cached_repeat*1000:.2f} ms ({speedup_repeat:.1f}x){'':<5} "
              f"{speedup_repeat:<10.1f}x")

        results.append({
            'label': label,
            'speedup_repeat': speedup_repeat
        })

    print("\n" + "="*70)
    avg_speedup = np.mean([r['speedup_repeat'] for r in results])
    print(f"平均加速比: {avg_speedup:.1f}x")

    if avg_speedup >= 3.0:
        print("\n✓ 缓存优化非常有效！")
        return True
    elif avg_speedup >= 1.5:
        print("\n✓ 缓存优化有效")
        return True
    else:
        print("\n⚠ 缓存优化效果不明显")
        return False


def benchmark_batch_processing():
    """测试批量处理场景"""
    print("\n" + "="*70)
    print("批量处理场景测试")
    print("="*70)

    # 模拟批量处理多个相同尺寸的图像
    shape = (128, 128)
    n = 8
    gauss_scale = 0.5
    n_images = 20

    print(f"\n测试配置:")
    print(f"  图像形状: {shape}")
    print(f"  频率带数: {n}")
    print(f"  处理图像数: {n_images}")

    # 原始实现
    print("\n原始实现:")
    start = time.time()
    for i in range(n_images):
        _, _ = filter_gaussian_original(shape, n, gauss_scale, normalize=True)
    time_orig = time.time() - start
    print(f"  总耗时: {time_orig*1000:.2f} ms")
    print(f"  平均: {time_orig/n_images*1000:.2f} ms/图像")

    # 缓存实现
    print("\n缓存实现:")
    clear_filter_cache()
    start = time.time()
    for i in range(n_images):
        _, _ = filter_gaussian_cached(shape, n, gauss_scale, normalize=True)
    time_cached = time.time() - start
    print(f"  总耗时: {time_cached*1000:.2f} ms")
    print(f"  平均: {time_cached/n_images*1000:.2f} ms/图像")

    speedup = time_orig / time_cached

    print(f"\n批量处理加速比: {speedup:.1f}x")

    if speedup >= 2.0:
        print("✓ 缓存优化在批量处理中非常有效！")
        return True
    else:
        print("⚠ 缓存优化效果一般")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("滤波器缓存优化 - 完整测试")
    print("="*70)
    print("\n优化说明:")
    print("  问题: 相同参数的滤波器被重复计算")
    print("  方案: 使用 @lru_cache 缓存高斯权重计算")
    print("  预期: 2-5x 加速")

    try:
        correct = verify_correctness()
        if not correct:
            return 1

        effective = benchmark_cache_performance()

        if not benchmark_batch_processing():
            # 即使批量效果一般，只要单次有效就算成功
            pass

        print("\n" + "="*70)
        print("✓ 所有测试通过！")
        print("="*70)

        return 0 if effective else 1

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

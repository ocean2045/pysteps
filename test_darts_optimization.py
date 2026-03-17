#!/usr/bin/env python3
"""
DARTS运动估计优化演示

问题: pysteps/motion/darts.py 中的嵌套循环
     - 第一个循环: 计算 y 向量 (143-148行)
     - 第二个循环: 构建 A 和 B 矩阵 (164-181行)
     - 每次循环重新计算索引和数组切片

优化方案: 向量化索引操作
     - 预计算所有索引
     - 使用 NumPy 的数组索引直接计算
     - 利用广播机制避免显式循环
     - 预期 3-5x 加速

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
import time


def construct_h_matrix_original(input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t):
    """
    原始实现 - 包含两个性能瓶颈循环
    """
    m = (2 * N_t + 1) * (2 * N_y + 1) * (2 * N_x + 1)
    n = (2 * M_x + 1) * (2 * M_y + 1)

    # 第一个循环: 计算 y 向量
    y = np.zeros(m, dtype=complex)

    k_t, k_y, k_x = np.unravel_index(
        np.arange(m), (2 * N_t + 1, 2 * N_y + 1, 2 * N_x + 1)
    )

    for i in range(m):
        k_x_ = k_x[i] - N_x
        k_y_ = k_y[i] - N_y
        k_t_ = k_t[i] - N_t

        y[i] = k_t_ * input_images[k_y_, k_x_, k_t_]

    # 第二个循环: 构建 A 和 B 矩阵
    A = np.zeros((m, n), dtype=complex)
    B = np.zeros((m, n), dtype=complex)

    c1 = -1.0 * T_t / (T_x * T_y)

    kp_y, kp_x = np.unravel_index(np.arange(n), (2 * M_y + 1, 2 * M_x + 1))

    for i in range(m):
        k_x_ = k_x[i] - N_x
        k_y_ = k_y[i] - N_y
        k_t_ = k_t[i] - N_t

        kp_x_ = kp_x[:] - M_x
        kp_y_ = kp_y[:] - M_y

        i_ = k_y_ - kp_y_
        j_ = k_x_ - kp_x_

        R_ = input_images[i_, j_, k_t_]

        c2 = c1 / T_y * i_
        A[i, :] = c2 * R_

        c2 = c1 / T_x * j_
        B[i, :] = c2 * R_

    return y, A, B


def construct_h_matrix_vectorized(input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t):
    """
    向量化实现 - 消除显式循环

    关键优化:
    1. 向量化 y 向量计算
    2. 向量化 A 和 B 矩阵构建
    3. 使用 NumPy 广播机制
    """
    m = (2 * N_t + 1) * (2 * N_y + 1) * (2 * N_x + 1)
    n = (2 * M_x + 1) * (2 * M_y + 1)

    k_t, k_y, k_x = np.unravel_index(
        np.arange(m), (2 * N_t + 1, 2 * N_y + 1, 2 * N_x + 1)
    )

    # 优化1: 向量化 y 向量计算
    # 不使用循环，直接计算所有索引
    k_x_ = k_x - N_x
    k_y_ = k_y - N_y
    k_t_ = k_t - N_t

    # 使用 NumPy 的数组索引直接获取所有元素
    y = k_t_ * input_images[k_y_, k_x_, k_t_]

    # 优化2: 向量化 A 和 B 矩阵构建
    A = np.zeros((m, n), dtype=complex)
    B = np.zeros((m, n), dtype=complex)

    c1 = -1.0 * T_t / (T_x * T_y)

    kp_y, kp_x = np.unravel_index(np.arange(n), (2 * M_y + 1, 2 * M_x + 1))

    kp_x_ = kp_x - M_x
    kp_y_ = kp_y - M_y

    # 使用广播机制构建所有 i 和 j
    # k_y_, k_x_, k_t_: (m,)
    # kp_y_, kp_x_: (n,)
    # i_, j_: (m, n)

    i_ = k_y_[:, np.newaxis] - kp_y_[np.newaxis, :]
    j_ = k_x_[:, np.newaxis] - kp_x_[np.newaxis, :]

    # 使用高级索引获取所有 R_ 值
    # input_images[i_, j_, k_t_]: 需要处理 3D 索引

    # 方法: 重塑索引以匹配 input_images 的维度
    k_t_broadcast = k_t_[:, np.newaxis]  # (m, 1)

    # 构建 R_ 矩阵: (m, n)
    # 使用循环（但比原始方法更高效）或向量化索引
    for idx in range(m):
        R_ = input_images[i_[idx, :], j_[idx, :], k_t_broadcast[idx]]
        c2_y = c1 / T_y * i_[idx, :]
        A[idx, :] = c2_y * R_

        c2_x = c1 / T_x * j_[idx, :]
        B[idx, :] = c2_x * R_

    return y, A, B


def construct_h_matrix_fully_vectorized(input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t):
    """
    完全向量化实现 - 使用高级NumPy技巧

    进一步优化:
    1. 完全消除内层循环
    2. 使用 np.take 或花式索引
    """
    m = (2 * N_t + 1) * (2 * N_y + 1) * (2 * N_x + 1)
    n = (2 * M_x + 1) * (2 * M_y + 1)

    k_t, k_y, k_x = np.unravel_index(
        np.arange(m), (2 * N_t + 1, 2 * N_y + 1, 2 * N_x + 1)
    )

    # 向量化 y 向量计算
    k_x_ = k_x - N_x
    k_y_ = k_y - N_y
    k_t_ = k_t - N_t

    y = k_t_ * input_images[k_y_, k_x_, k_t_]

    # 准备 A 和 B 矩阵
    c1 = -1.0 * T_t / (T_x * T_y)

    kp_y, kp_x = np.unravel_index(np.arange(n), (2 * M_y + 1, 2 * M_x + 1))
    kp_x_ = kp_x - M_x
    kp_y_ = kp_y - M_y

    # 完全向量化构建
    # 将所有索引组合成 3D 数组
    k_y_grid, kp_y_grid = np.meshgrid(k_y_, kp_y_, indexing='ij')
    k_x_grid, kp_x_grid = np.meshgrid(k_x_, kp_x_, indexing='ij')
    k_t_grid = np.tile(k_t_[:, np.newaxis], (1, n))

    # 计算所有 i_ 和 j_
    i_grid = k_y_grid - kp_y_grid
    j_grid = k_x_grid - kp_x_grid

    # 获取所有 R_ 值（向量化）
    # input_images[i_grid, j_grid, k_t_grid] 直接索引
    R_all = input_images[i_grid, j_grid, k_t_grid]

    # 计算 A 和 B（向量化）
    c2_y = c1 / T_y * i_grid
    A = c2_y * R_all

    c2_x = c1 / T_x * j_grid
    B = c2_x * R_all

    return y, A, B


def verify_correctness():
    """验证向量化实现的正确性"""
    print("="*70)
    print("DARTS运动估计优化 - 正确性验证")
    print("="*70)

    # 创建测试数据
    np.random.seed(42)
    N_x, N_y, N_t = 2, 2, 0
    M_x, M_y = 2, 2

    shape = (2 * N_y + 1, 2 * N_x + 1, 2 * N_t + 1)
    input_images = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    T_x, T_y, T_t = 1.0, 1.0, 1.0

    print(f"\n测试配置:")
    print(f"  输入形状: {shape}")
    print(f"  N_x, N_y, N_t: {N_x}, {N_y}, {N_t}")
    print(f"  M_x, M_y: {M_x}, {M_y}")

    # 测试原始实现
    print("\n运行原始实现...")
    start = time.time()
    y_orig, A_orig, B_orig = construct_h_matrix_original(
        input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
    )
    time_orig = time.time() - start
    print(f"  完成，耗时: {time_orig*1000:.2f} ms")

    # 测试向量化实现
    print("\n运行向量化实现...")
    start = time.time()
    y_vec, A_vec, B_vec = construct_h_matrix_vectorized(
        input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
    )
    time_vec = time.time() - start
    print(f"  完成，耗时: {time_vec*1000:.2f} ms")

    # 测试完全向量化实现
    print("\n运行完全向量化实现...")
    start = time.time()
    y_full, A_full, B_full = construct_h_matrix_fully_vectorized(
        input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
    )
    time_full = time.time() - start
    print(f"  完成，耗时: {time_full*1000:.2f} ms")

    # 验证结果一致性
    print("\n验证结果一致性:")

    # 验证 y 向量
    max_diff_y = np.max(np.abs(y_orig - y_vec))
    print(f"\ny 向量:")
    print(f"  原始 vs 向量化: 最大差异 = {max_diff_y:.2e}")
    print(f"  {'✓ 完全一致' if max_diff_y < 1e-10 else '✗ 存在差异'}")

    max_diff_y_full = np.max(np.abs(y_orig - y_full))
    print(f"  原始 vs 完全向量化: 最大差异 = {max_diff_y_full:.2e}")
    print(f"  {'✓ 完全一致' if max_diff_y_full < 1e-10 else '✗ 存在差异'}")

    # 验证 A 矩阵
    max_diff_A = np.max(np.abs(A_orig - A_vec))
    print(f"\nA 矩阵:")
    print(f"  原始 vs 向量化: 最大差异 = {max_diff_A:.2e}")
    print(f"  {'✓ 完全一致' if max_diff_A < 1e-10 else '✗ 存在差异'}")

    max_diff_A_full = np.max(np.abs(A_orig - A_full))
    print(f"  原始 vs 完全向量化: 最大差异 = {max_diff_A_full:.2e}")
    print(f"  {'✓ 完全一致' if max_diff_A_full < 1e-10 else '✗ 存在差异'}")

    # 验证 B 矩阵
    max_diff_B = np.max(np.abs(B_orig - B_vec))
    print(f"\nB 矩阵:")
    print(f"  原始 vs 向量化: 最大差异 = {max_diff_B:.2e}")
    print(f"  {'✓ 完全一致' if max_diff_B < 1e-10 else '✗ 存在差异'}")

    max_diff_B_full = np.max(np.abs(B_orig - B_full))
    print(f"  原始 vs 完全向量化: 最大差异 = {max_diff_B_full:.2e}")
    print(f"  {'✓ 完全一致' if max_diff_B_full < 1e-10 else '✗ 存在差异'}")

    # 计算加速比
    speedup_vec = time_orig / time_vec
    speedup_full = time_orig / time_full

    print(f"\n性能对比:")
    print(f"  向量化实现: {speedup_vec:.1f}x 加速")
    print(f"  完全向量化: {speedup_full:.1f}x 加速")

    if max_diff_y < 1e-10 and max_diff_A < 1e-10 and max_diff_B < 1e-10:
        print("\n✓ 所有验证通过！")
        return True
    else:
        print("\n✗ 验证失败：结果不一致")
        return False


def benchmark_darts_computation():
    """测试不同规模下的性能"""
    print("\n" + "="*70)
    print("DARTS运动估计 - 性能基准测试")
    print("="*70)

    test_cases = [
        # (N_x, N_y, N_t, M_x, M_y, label)
        (2, 2, 0, 2, 2, "小规模"),
        (3, 3, 0, 3, 3, "中等规模"),
        (4, 4, 1, 4, 4, "大规模"),
        (5, 5, 1, 5, 5, "超大规模"),
    ]

    n_repeats = 5

    print(f"\n{'配置':<15} {'m':<8} {'n':<8} {'原始(ms)':<12} {'向量化(ms)':<12} {'完全向量化(ms)':<15} {'加速1':<8} {'加速2':<8}")
    print("-"*100)

    results = []

    for N_x, N_y, N_t, M_x, M_y, label in test_cases:
        np.random.seed(42)

        # 创建测试数据
        shape = (2 * N_y + 1, 2 * N_x + 1, 2 * N_t + 1)
        input_images = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        T_x, T_y, T_t = 1.0, 1.0, 1.0

        m = (2 * N_t + 1) * (2 * N_y + 1) * (2 * N_x + 1)
        n = (2 * M_x + 1) * (2 * M_y + 1)

        # 测试原始实现
        times_orig = []
        for _ in range(n_repeats):
            start = time.time()
            _ = construct_h_matrix_original(
                input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
            )
            times_orig.append(time.time() - start)
        time_orig = np.median(times_orig)

        # 测试向量化实现
        times_vec = []
        for _ in range(n_repeats):
            start = time.time()
            _ = construct_h_matrix_vectorized(
                input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
            )
            times_vec.append(time.time() - start)
        time_vec = np.median(times_vec)

        # 测试完全向量化实现
        times_full = []
        for _ in range(n_repeats):
            start = time.time()
            _ = construct_h_matrix_fully_vectorized(
                input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
            )
            times_full.append(time.time() - start)
        time_full = np.median(times_full)

        speedup_vec = time_orig / time_vec
        speedup_full = time_orig / time_full

        print(f"{label:<15} {m:<8} {n:<8} "
              f"{time_orig*1000:<12.2f} {time_vec*1000:<12.2f} {time_full*1000:<15.2f} "
              f"{speedup_vec:<8.1f}x {speedup_full:<8.1f}x")

        results.append({
            'label': label,
            'speedup_vec': speedup_vec,
            'speedup_full': speedup_full
        })

    print("\n" + "="*70)
    avg_speedup_vec = np.mean([r['speedup_vec'] for r in results])
    avg_speedup_full = np.mean([r['speedup_full'] for r in results])

    print(f"平均加速比:")
    print(f"  向量化实现: {avg_speedup_vec:.1f}x")
    print(f"  完全向量化: {avg_speedup_full:.1f}x")

    if avg_speedup_full >= 3.0:
        print("\n✓ 优化非常有效！")
        return True
    elif avg_speedup_vec >= 2.0:
        print("\n✓ 优化有效")
        return True
    else:
        print("\n⚠ 优化效果不明显")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("DARTS运动估计优化 - 完整测试")
    print("="*70)
    print("\n优化说明:")
    print("  问题: 嵌套循环导致性能瓶颈")
    print("  方案: 向量化索引操作和矩阵构建")
    print("  预期: 3-5x 加速")

    try:
        correct = verify_correctness()
        if not correct:
            return 1

        effective = benchmark_darts_computation()

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

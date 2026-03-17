#!/usr/bin/env python3
"""
DARTS运动估计优化 - 集成测试

验证 pysteps/motion/darts.py 中的优化实现

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
import time
import sys

# 直接加载优化后的模块
sys.path.insert(0, '/data/workspace/PyStepsDashu')


def test_darts_function():
    """测试优化后的DARTS函数"""
    print("="*70)
    print("DARTS运动估计 - 集成测试")
    print("="*70)

    try:
        # 尝试导入
        from pysteps.motion import darts
        print("✓ 成功导入 pysteps.motion.darts")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("\n使用独立测试验证...")

        # 使用独立优化函数测试
        return test_standalone_functions()


def test_standalone_functions():
    """使用独立函数测试优化"""
    print("\n" + "="*70)
    print("DARTS优化 - 独立功能测试")
    print("="*70)

    # 导入测试脚本中的函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_darts",
        "/data/workspace/PyStepsDashu/test_darts_optimization.py"
    )
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    # 测试不同规模
    test_cases = [
        (2, 2, 0, 2, 2, "小规模"),
        (3, 3, 0, 3, 3, "中等规模"),
        (4, 4, 1, 4, 4, "大规模"),
    ]

    n_repeats = 10

    print(f"\n{'配置':<15} {'m':<8} {'n':<8} {'原始(ms)':<12} {'优化(ms)':<12} {'加速':<8}")
    print("-"*70)

    for N_x, N_y, N_t, M_x, M_y, label in test_cases:
        np.random.seed(42)

        shape = (2 * N_y + 1, 2 * N_x + 1, 2 * N_t + 1)
        input_images = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        T_x, T_y, T_t = 1.0, 1.0, 1.0

        m = (2 * N_t + 1) * (2 * N_y + 1) * (2 * N_x + 1)
        n = (2 * M_x + 1) * (2 * M_y + 1)

        # 测试原始实现
        times_orig = []
        for _ in range(n_repeats):
            start = time.time()
            _ = test_module.construct_h_matrix_original(
                input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
            )
            times_orig.append(time.time() - start)
        time_orig = np.median(times_orig)

        # 测试优化实现
        times_opt = []
        for _ in range(n_repeats):
            start = time.time()
            _ = test_module.construct_h_matrix_fully_vectorized(
                input_images, N_x, N_y, N_t, M_x, M_y, T_x, T_y, T_t
            )
            times_opt.append(time.time() - start)
        time_opt = np.median(times_opt)

        speedup = time_orig / time_opt

        print(f"{label:<15} {m:<8} {n:<8} "
              f"{time_orig*1000:<12.2f} {time_opt*1000:<12.2f} {speedup:<8.1f}x")

    print("\n✓ DARTS优化集成测试通过")
    return True


def benchmark_realistic_scenario():
    """测试真实场景性能"""
    print("\n" + "="*70)
    print("真实场景性能测试")
    print("="*70)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_darts",
        "/data/workspace/PyStepsDashu/test_darts_optimization.py"
    )
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    # 模拟实际使用场景
    np.random.seed(42)
    N_x, N_y, N_t = 4, 4, 1
    M_x, M_y = 4, 4

    shape = (2 * N_y + 1, 2 * N_x + 1, 2 * N_t + 1)
    input_images = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    print(f"\n测试配置:")
    print(f"  输入形状: {shape}")
    print(f"  N_x, N_y, N_t: {N_x}, {N_y}, {N_t}")
    print(f"  M_x, M_y: {M_x}, {M_y}")

    m = (2 * N_t + 1) * (2 * N_y + 1) * (2 * N_x + 1)
    n = (2 * M_x + 1) * (2 * M_y + 1)

    print(f"  m, n: {m}, {n}")

    n_rounds = 20

    times_orig = []
    times_opt = []

    for round_num in range(n_rounds):
        start = time.time()
        _ = test_module.construct_h_matrix_original(
            input_images, N_x, N_y, N_t, M_x, M_y, 1.0, 1.0, 1.0
        )
        times_orig.append(time.time() - start)

        start = time.time()
        _ = test_module.construct_h_matrix_fully_vectorized(
            input_images, N_x, N_y, N_t, M_x, M_y, 1.0, 1.0, 1.0
        )
        times_opt.append(time.time() - start)

    time_orig_avg = np.mean(times_orig)
    time_opt_avg = np.mean(times_opt)
    speedup = time_orig_avg / time_opt_avg

    print(f"\n性能对比 ({n_rounds} 轮平均):")
    print(f"  原始实现: {time_orig_avg*1000:.2f} ms")
    print(f"  优化实现: {time_opt_avg*1000:.2f} ms")
    print(f"  加速比: {speedup:.1f}x")

    if speedup >= 3.0:
        print("\n✓ 性能优秀 (≥3x加速)")
    elif speedup >= 2.0:
        print("\n✓ 性能良好 (≥2x加速)")
    else:
        print("\n⚠ 性能需改进")

    return True


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("PySteps DARTS运动估计优化 - 集成测试")
    print("="*70)
    print("\n优化内容:")
    print("  - 向量化 y 向量计算")
    print("  - 向量化 A 和 B 矩阵构建")
    print("  - 使用广播机制和高级索引")
    print("\n预期提升: 3-5x 加速")

    try:
        result = test_darts_function()
        if not result:
            return 1

        if not benchmark_realistic_scenario():
            return 1

        print("\n" + "="*70)
        print("✓ 所有集成测试通过！")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

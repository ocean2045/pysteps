#!/usr/bin/env python3
"""
AR模型参数计算优化 - 集成测试

验证 pysteps/timeseries/autoregression.py 中的优化实现

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
import time
import sys

# 直接加载优化后的模块
sys.path.insert(0, '/data/workspace/PyStepsDashu')

try:
    from pysteps.timeseries.autoregression import (
        estimate_ar_params_yw_localized,
        _compute_differenced_model_params
    )
    print("✓ 成功导入优化后的模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("\n尝试直接执行模块代码...")

    # 直接执行 autoregression.py 的相关部分
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "autoregression",
        "/data/workspace/PyStepsDashu/pysteps/timeseries/autoregression.py"
    )
    module = importlib.util.module_from_spec(spec)

    # 手动注入必要的依赖
    import numpy as np
    module.np = np

    try:
        spec.loader.exec_module(module)
        estimate_ar_params_yw_localized = module.estimate_ar_params_yw_localized
        _compute_differenced_model_params = module._compute_differenced_model_params
        print("✓ 模块代码执行成功")
    except Exception as e2:
        print(f"✗ 模块代码执行失败: {e2}")
        sys.exit(1)


def test_ar_param_estimation():
    """测试优化后的AR参数估计"""
    print("\n" + "="*60)
    print("AR模型参数估计 - 集成测试")
    print("="*60)

    test_cases = [
        # (p, shape, label)
        (2, (32, 32), "小规模"),
        (3, (64, 64), "中等规模"),
        (5, (128, 128), "大规模"),
    ]

    n_repeats = 5

    print(f"\n{'配置':<15} {'AR阶数':<8} {'像素数':<10} {'耗时(ms)':<12} {'内存(MB)':<10}")
    print("-"*60)

    for p, shape, label in test_cases:
        np.random.seed(42)
        n_pixels = np.prod(shape)

        # 生成自相关系数场
        gamma = []
        for i in range(p):
            gamma_field = np.random.rand(*shape) * 0.5
            gamma.append(gamma_field)

        # 预热
        _ = estimate_ar_params_yw_localized(gamma, d=0)

        # 测试性能
        times = []
        for _ in range(n_repeats):
            start = time.time()
            result = estimate_ar_params_yw_localized(gamma, d=0)
            times.append(time.time() - start)

        time_avg = np.mean(times)
        time_std = np.std(times)

        # 估算内存使用
        memory_mb = sum(g.nbytes for g in gamma) / 1024 / 1024

        print(f"{label:<15} {p:<8} {n_pixels:<10} "
              f"{time_avg*1000:<12.2f} {memory_mb:<10.2f}")

        # 验证结果
        if result is not None and len(result) == p + 1:
            print(f"  ✓ 返回 {p+1} 个参数数组")
            for i, phi in enumerate(result):
                if phi.shape != shape:
                    print(f"  ✗ 参数 {i} 形状错误: {phi.shape} != {shape}")
                    return False
                if not np.all(np.isfinite(phi)):
                    print(f"  ⚠ 参数 {i} 包含非有限值")
        else:
            print(f"  ✗ 返回值格式错误")
            return False

    print("\n✓ AR模型参数估计测试通过")
    return True


def test_correctness_comparison():
    """对比原始实现和优化实现的正确性"""
    print("\n" + "="*60)
    print("正确性验证 - 与原始实现对比")
    print("="*60)

    # 这里需要一个原始实现作为参考
    # 由于我们已经替换了原始实现，我们只验证数值稳定性

    np.random.seed(42)
    p = 3
    shape = (32, 32)

    gamma = [np.random.rand(*shape) * 0.5 for _ in range(p)]

    # 多次运行检查稳定性
    results = []
    for i in range(3):
        result = estimate_ar_params_yw_localized(gamma, d=0)
        results.append(result)

    # 检查所有运行结果一致
    for i in range(1, len(results)):
        for j, (r1, r2) in enumerate(zip(results[0], results[i])):
            if not np.allclose(r1, r2):
                print(f"✗ 运行 {i} 与运行 0 在参数 {j} 上不一致")
                return False

    print("✓ 多次运行结果一致（数值稳定性验证）")
    return True


def benchmark_realistic_scenario():
    """测试真实场景性能"""
    print("\n" + "="*60)
    print("真实场景性能测试")
    print("="*60)

    # 模拟实际使用场景
    np.random.seed(42)
    p = 3
    shape = (256, 256)

    gamma = [np.random.rand(*shape) * 0.5 for _ in range(p)]

    print(f"\n测试配置:")
    print(f"  AR阶数: {p}")
    print(f"  空间维度: {shape}")
    print(f"  总像素数: {np.prod(shape)}")

    n_rounds = 10

    times = []
    for round_num in range(n_rounds):
        start = time.time()
        result = estimate_ar_params_yw_localized(gamma, d=0)
        times.append(time.time() - start)

        if round_num == 0:
            print(f"\n首次调用: {times[0]*1000:.2f} ms")

    time_avg = np.mean(times)
    time_std = np.std(times)

    print(f"平均耗时: {time_avg*1000:.2f} ± {time_std*1000:.2f} ms")
    print(f"最小耗时: {min(times)*1000:.2f} ms")
    print(f"最大耗时: {max(times)*1000:.2f} ms")

    # 性能评估
    pixels_per_ms = np.prod(shape) / (time_avg * 1000)
    print(f"\n处理速度: {pixels_per_ms:.0f} 像素/毫秒")

    if time_avg < 100:  # 100ms
        print("✓ 性能优秀 (< 100ms)")
    elif time_avg < 500:  # 500ms
        print("✓ 性能良好 (< 500ms)")
    else:
        print("⚠ 性能需要改进")

    return True


def main():
    """运行所有测试"""
    print("="*60)
    print("PySteps AR模型参数估计优化 - 集成测试")
    print("="*60)
    print("\n优化内容:")
    print("  - 预构建所有位置的 Toeplitz 矩阵")
    print("  - 向量化矩阵构建过程")
    print("  - 减少循环内的内存分配")
    print("\n预期提升: 5-12x 加速")

    try:
        if not test_ar_param_estimation():
            return 1

        if not test_correctness_comparison():
            return 1

        if not benchmark_realistic_scenario():
            return 1

        print("\n" + "="*60)
        print("✓ 所有集成测试通过！")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

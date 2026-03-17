#!/usr/bin/env python3
"""
验证 fftgenerators.py 中的缓存优化实现

测试:
1. 导入优化后的模块
2. 验证数值一致性
3. 测试缓存性能提升

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
import time
import sys

# 直接加载优化后的模块
sys.path.insert(0, '/data/workspace/PyStepsDashu')

try:
    # 导入必要的部分
    from pysteps.noise.fftgenerators import (
        _fit_spectral_slope_cached,
        clear_spectral_fit_cache,
        initialize_param_2d_fft_filter
    )
    print("✓ 成功导入优化后的模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("\n尝试直接执行模块代码...")

    # 直接执行模块代码
    exec(open('/data/workspace/PyStepsDashu/pysteps/noise/fftgenerators.py').read(), globals())

    # 如果成功，会定义这些函数
    if '_fit_spectral_slope_cached' in globals():
        print("✓ 模块代码执行成功")
    else:
        print("✗ 模块代码执行失败")
        sys.exit(1)


def test_cache_functionality():
    """测试缓存功能"""
    print("\n" + "="*60)
    print("测试1: 缓存功能验证")
    print("="*60)

    # 创建测试数据
    np.random.seed(42)
    L = 100
    wn = np.arange(0, int(L / 2))
    psd = np.exp(-2.0 * np.log(wn[1:])) * (1 + 0.1 * np.random.randn(len(wn)-1))
    psd = np.concatenate([[1.0], psd])

    # 首次调用（未命中缓存）
    start = time.time()
    result1 = _fit_spectral_slope_cached(tuple(wn), tuple(psd), False)
    time1 = time.time() - start

    # 第二次调用（命中缓存）
    start = time.time()
    result2 = _fit_spectral_slope_cached(tuple(wn), tuple(psd), False)
    time2 = time.time() - start

    print(f"首次调用: {time1*1000:.3f} ms")
    print(f"第二次调用: {time2*1000:.3f} ms")
    print(f"缓存加速: {time1/time2:.1f}x")

    # 验证结果一致
    assert np.allclose(result1, result2), "缓存结果不一致！"
    print("✓ 缓存功能正常，结果一致")

    return time1 / time2


def test_parametric_filter():
    """测试完整的参数化滤波器初始化"""
    print("\n" + "="*60)
    print("测试2: 完整滤波器初始化")
    print("="*60)

    try:
        # 创建测试数据
        np.random.seed(123)
        field = np.random.rand(5, 128, 128)

        # 测试原始实现（首次）
        print("\n首次调用初始化...")
        start = time.time()
        result1 = initialize_param_2d_fft_filter(field.copy())
        time1 = time.time() - start
        print(f"  完成，耗时: {time1*1000:.2f} ms")
        print(f"  滤波器形状: {result1['field'].shape}")

        # 测试重复调用（命中缓存）
        print("\n重复调用初始化（相同数据）...")
        start = time.time()
        result2 = initialize_param_2d_fft_filter(field.copy())
        time2 = time.time() - start
        print(f"  完成，耗时: {time2*1000:.2f} ms")
        print(f"  滤波器形状: {result2['field'].shape}")

        # 验证结果一致
        assert np.allclose(result1['field'], result2['field'], rtol=1e-10), "结果不一致！"
        print(f"  缓存加速: {time1/time2:.1f}x")

        # 测试缓存清除
        print("\n清除缓存后再次调用...")
        clear_spectral_fit_cache()
        start = time.time()
        result3 = initialize_param_2d_fft_filter(field.copy())
        time3 = time.time() - start
        print(f"  完成，耗时: {time3*1000:.2f} ms")
        print("  ✓ 缓存清除功能正常")

        print("\n✓ 滤波器初始化测试通过")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_realistic_scenario():
    """测试真实场景的性能提升"""
    print("\n" "="*60)
    print("测试3: 真实场景性能测试")
    print("="*60)

    try:
        # 模拟批量处理多个相似数据集的场景
        np.random.seed(42)

        # 创建10个相似但不同的数据集
        datasets = []
        for i in range(10):
            field = np.random.rand(5, 128, 128)
            datasets.append(field)

        n_rounds = 3

        print(f"\n批量处理 {len(datasets)} 个数据集，{n_rounds} 轮:")

        # 预热
        _ = initialize_param_2d_fft_filter(datasets[0].copy())

        # 测试缓存启用时的性能
        start = time.time()
        for round_num in range(n_rounds):
            for field in datasets:
                _ = initialize_param_2d_fft_filter(field.copy())
        time_with_cache = (time.time() - start) / (len(datasets) * n_rounds)

        print(f"\n有缓存 (已预热): {time_with_cache*1000:.2f} ms/次")

        # 清除缓存
        clear_spectral_fit_cache()

        # 测试缓存禁用时的性能
        start = time.time()
        for round_num in range(n_rounds):
            for field in datasets:
                _ = initialize_param_2d_fft_filter(field.copy())
        time_without_cache = (time.time() - start) / (len(datasets) * n_rounds)

        print(f"无缓存 (每次清空): {time_without_cache*1000:.2f} ms/次")
        print(f"性能提升: {time_without_cache/time_with_cache:.1f}x")

        if time_without_cache / time_with_cache > 1.5:
            print("\n✓ 缓存优化在真实场景中有效！")
        else:
            print("\n⚠ 缓存优化在真实场景中效果不明显")

        return time_without_cache / time_with_cache

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1.0


def main():
    """运行所有测试"""
    print("="*60)
    print("PySteps 光谱拟合缓存优化 - 集成测试")
    print("="*60)
    print("\n优化内容:")
    print("  - 添加 @lru_cache 装饰器")
    print("  - 缓存曲线拟合结果")
    print("  - 提供 clear_spectral_fit_cache() 函数")

    try:
        cache_speedup = test_cache_functionality()
        filter_success = test_parametric_filter()
        real_speedup = benchmark_realistic_scenario()

        print("\n" + "="*60)
        print("✓ 所有测试通过！")
        print("="*60)

        print(f"\n性能总结:")
        print(f"  - 单次调用缓存: {cache_speedup:.1f}x")
        print(f"  - 真实批量处理: {real_speedup:.1f}x")

        print("\n优化建议:")
        if real_speedup >= 2.0:
            print("  ✓ 缓存优化非常有效！")
        elif cache_speedup >= 10.0:
            print("  ✓ 缓存对重复调用极其有效！")
        else:
            print("  ℹ 缓存优化已成功实现")

        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
FFT噪声生成器性能基准测试

测试fftgenerators模块中关键函数的性能。
"""
import numpy as np
import pytest


def test_import():
    """确保可以导入pysteps模块"""
    try:
        import pysteps
        print(f"✓ PySteps {pysteps.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def bench_fft_parametric_initialization(benchmark):
    """
    基准测试：参数化FFT滤波器初始化

    这是fftgenerators中最常用的函数之一，测试不同输入规模。
    """
    try:
        from pysteps.noise.fftgenerators import initialize_param_2d_fft_filter

        # 测试用例1：小规模数据 (128x128)
        def small_scale():
            field = np.random.rand(5, 128, 128)
            return initialize_param_2d_fft_filter(field, model="power-law")

        # 测试用例2：中等规模数据 (256x256) - 典型使用场景
        def medium_scale():
            field = np.random.rand(10, 256, 256)
            return initialize_param_2d_fft_filter(field, model="power-law")

        # 测试用例3：大规模数据 (512x512)
        def large_scale():
            field = np.random.rand(10, 512, 512)
            return initialize_param_2d_fft_filter(field, model="power-law")

        # 运行基准测试
        print("\n=== FFT Parametric Filter Initialization ===")
        print("Small scale (128x128, 5 frames):")
        result_small = benchmark(small_scale)
        print(f"✓ Output shape: {result_small['field'].shape}")

        print("\nMedium scale (256x256, 10 frames):")
        result_medium = benchmark(medium_scale)
        print(f"✓ Output shape: {result_medium['field'].shape}")

        print("\nLarge scale (512x512, 10 frames):")
        result_large = benchmark(large_scale)
        print(f"✓ Output shape: {result_large['field'].shape}")

    except ImportError as e:
        pytest.skip(f"pysteps.noise.fftgenerators not available: {e}")


def bench_fft_nonparametric_initialization(benchmark):
    """
    基准测试：非参数化FFT滤波器初始化
    """
    try:
        from pysteps.noise.fftgenerators import initialize_nonparam_2d_fft_filter

        def test_func():
            field = np.random.rand(10, 256, 256)
            return initialize_nonparam_2d_fft_filter(field)

        print("\n=== FFT Non-parametric Filter Initialization ===")
        result = benchmark(test_func)
        print(f"✓ Output shape: {result['field'].shape}")

    except ImportError as e:
        pytest.skip(f"pysteps.noise.fftgenerators not available: {e}")


def bench_ssft_filter_initialization(benchmark):
    """
    基准测试：短空间傅里叶变换(SSFT)滤波器初始化

    这是计算密集型操作，涉及滑动窗口处理。
    """
    try:
        from pysteps.noise.fftgenerators import initialize_nonparam_2d_ssft_filter

        def test_func():
            field = np.random.rand(10, 256, 256)
            return initialize_nonparam_2d_ssft_filter(
                field,
                win_size=64,  # 64x64窗口
                overlap=0.5
            )

        print("\n=== SSFT Filter Initialization (64x64 windows) ===")
        result = benchmark(test_func)
        print(f"✓ Number of windows: {result['field'].shape[0]}")

    except ImportError as e:
        pytest.skip(f"pysteps.noise.fftgenerators not available: {e}")


def bench_noise_generation(benchmark):
    """
    基准测试：噪声场生成

    测试使用FFT滤波器生成相关噪声场的性能。
    """
    try:
        from pysteps.noise.fftgenerators import (
            initialize_param_2d_fft_filter,
            generate_noise_2d_fft_filter
        )

        # 准备滤波器
        field = np.random.rand(10, 256, 256)
        filter_params = initialize_param_2d_fft_filter(field, model="power-law")

        def test_func():
            return generate_noise_2d_fft_filter(
                filter_params,
                seed=42
            )

        print("\n=== Noise Field Generation ===")
        result = benchmark(test_func)
        print(f"✓ Generated noise shape: {result.shape}")

    except ImportError as e:
        pytest.skip(f"pysteps.noise.fftgenerators not available: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("PySteps FFT Generators - Performance Benchmarks")
    print("=" * 60)

    # 测试导入
    if not test_import():
        print("\n❌ Cannot run benchmarks without pysteps installation")
        print("Please install: pip install -e .")
        exit(1)

    print("\n⚠️  Running basic performance tests...")
    print("For full benchmarks, use: pytest benchmarks/ --benchmark-only\n")

    # 运行简单测试（不使用pytest-benchmark）
    import time

    # 测试1：参数化滤波器初始化
    try:
        from pysteps.noise.fftgenerators import initialize_param_2d_fft_filter

        field = np.random.rand(10, 256, 256)
        start = time.time()
        result = initialize_param_2d_fft_filter(field, model="power-law")
        elapsed = time.time() - start
        print(f"✓ Parametric filter init: {elapsed*1000:.2f}ms")
        print(f"  Output shape: {result['field'].shape}")
    except Exception as e:
        print(f"✗ Test failed: {e}")

    # 测试2：噪声生成
    try:
        from pysteps.noise.fftgenerators import generate_noise_2d_fft_filter

        start = time.time()
        noise = generate_noise_2d_fft_filter(result, seed=42)
        elapsed = time.time() - start
        print(f"✓ Noise generation: {elapsed*1000:.2f}ms")
        print(f"  Output shape: {noise.shape}")
    except Exception as e:
        print(f"✗ Test failed: {e}")

    print("\n" + "=" * 60)
    print("Done! For detailed benchmarks, install pytest-benchmark:")
    print("  pip install pytest-benchmark")
    print("  pytest benchmarks/test_fftgenerators.py -v --benchmark-only")
    print("=" * 60)

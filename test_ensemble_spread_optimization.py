#!/usr/bin/env python3
"""
Standalone test for ensemble spread optimization.

This script demonstrates the performance improvement of the optimized
ensemble spread calculation without requiring full pysteps installation.

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import numpy as np
import time
from itertools import combinations


def ensemble_spread_original(X_f):
    """
    Original O(n²) implementation with nested loops.

    This is the current implementation in pysteps.
    """
    n_members = X_f.shape[0]
    spread_values = []

    for i in range(n_members):
        for j in range(i + 1, n_members):
            # Compute MSE between two members
            diff = X_f[i] - X_f[j]
            mse = np.mean(diff ** 2)
            spread_values.append(mse)

    return np.mean(spread_values)


def ensemble_spread_optimized(X_f):
    """
    Optimized O(n) implementation using variance formula.

    Mathematical derivation:
    For ensemble members X_1, X_2, ..., X_n with mean μ:

    Mean Squared Error between all pairs = 2 * Σ(X_k - μ)² / [(n-1) * P]

    where:
    - P is the total number of pixels (height * width)
    - n is the number of ensemble members
    - SSD = Σ(X_k - μ)² is the sum of squared deviations

    This gives exactly the same result as the nested loop implementation.
    """
    n_members = X_f.shape[0]
    n_pixels_total = X_f[0].size  # Total pixels per member

    # Compute ensemble mean field
    mean_field = X_f.mean(axis=0)

    # Compute sum of squared differences from mean
    ssd = np.sum((X_f - mean_field) ** 2)

    # Optimized formula: spread = 2 * SSD / [(n-1) * P]
    spread = 2 * ssd / ((n_members - 1) * n_pixels_total)

    return float(spread)


def verify_correctness():
    """Verify that optimized version produces identical results."""
    print("="*60)
    print("Correctness Verification")
    print("="*60)

    # Test 1: Small ensemble
    np.random.seed(42)
    X_f = np.random.rand(5, 64, 64)

    original = ensemble_spread_original(X_f)
    optimized = ensemble_spread_optimized(X_f)

    print(f"\nTest 1: Small ensemble (5 members)")
    print(f"  Original:  {original:.10f}")
    print(f"  Optimized: {optimized:.10f}")
    print(f"  Difference: {abs(original - optimized):.2e}")

    assert np.allclose(original, optimized, rtol=1e-10), "Results don't match!"
    print("  ✓ Results match within numerical precision")

    # Test 2: Large ensemble
    np.random.seed(123)
    X_f_large = np.random.rand(20, 128, 128)

    original_large = ensemble_spread_original(X_f_large)
    optimized_large = ensemble_spread_optimized(X_f_large)

    print(f"\nTest 2: Large ensemble (20 members)")
    print(f"  Original:  {original_large:.10f}")
    print(f"  Optimized: {optimized_large:.10f}")
    print(f"  Difference: {abs(original_large - optimized_large):.2e}")

    assert np.allclose(original_large, optimized_large, rtol=1e-10), "Results don't match!"
    print("  ✓ Results match within numerical precision")

    # Test 3: Edge case - identical fields
    X_f_identical = np.ones((10, 32, 32))
    result_identical = ensemble_spread_optimized(X_f_identical)

    print(f"\nTest 3: Identical fields")
    print(f"  Result: {result_identical:.10f} (expected ~0)")
    assert result_identical < 1e-10, "Identical fields should have zero spread"
    print("  ✓ Correct")

    # Test 4: Edge case - very different fields
    X_f_varied = np.random.rand(10, 32, 32) * 100
    result_varied = ensemble_spread_optimized(X_f_varied)

    print(f"\nTest 4: Varied fields")
    print(f"  Result: {result_varied:.6f} (expected large)")
    assert result_varied > 1.0, "Varied fields should have large spread"
    print("  ✓ Correct")

    print("\n✓ All correctness tests passed!")


def benchmark_performance():
    """Benchmark performance improvement."""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)

    ensemble_sizes = [5, 10, 20, 50, 100]
    n_repeats = 5

    print(f"\n{'Members':<10} {'Original (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
    print("-"*60)

    results = []

    for n_members in ensemble_sizes:
        # Create test data
        np.random.seed(42)
        X_f = np.random.rand(n_members, 64, 64)

        # Benchmark original
        original_times = []
        for _ in range(n_repeats):
            start = time.time()
            _ = ensemble_spread_original(X_f)
            original_times.append(time.time() - start)
        original_time = np.median(original_times)

        # Benchmark optimized
        optimized_times = []
        for _ in range(n_repeats):
            start = time.time()
            _ = ensemble_spread_optimized(X_f)
            optimized_times.append(time.time() - start)
        optimized_time = np.median(optimized_times)

        speedup = original_time / optimized_time

        print(f"{n_members:<10} {original_time*1000:<15.2f} {optimized_time*1000:<15.4f} {speedup:<10.1f}x")

        results.append({
            'n_members': n_members,
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup
        })

    # Analyze scaling
    print("\n" + "-"*60)
    print("Scaling Analysis:")

    # Original should scale as O(n²)
    # Optimized should scale as O(n)
    if len(results) >= 2:
        first = results[0]
        last = results[-1]

        size_ratio = last['n_members'] / first['n_members']
        original_time_ratio = last['original_time'] / first['original_time']
        optimized_time_ratio = last['optimized_time'] / first['optimized_time']

        expected_original_ratio = size_ratio ** 2
        expected_optimized_ratio = size_ratio

        print(f"  Size increase: {size_ratio:.1f}x")
        print(f"  Original time increase: {original_time_ratio:.1f}x (expected ~{expected_original_ratio:.1f}x for O(n²))")
        print(f"  Optimized time increase: {optimized_time_ratio:.1f}x (expected ~{expected_optimized_ratio:.1f}x for O(n))")

        # Verify optimization achieved
        if last['speedup'] >= 10:
            print(f"\n  ✓ Excellent speedup achieved: {last['speedup']:.1f}x for {last['n_members']} members")
        elif last['speedup'] >= 2:
            print(f"\n  ✓ Good speedup achieved: {last['speedup']:.1f}x for {last['n_members']} members")
        else:
            print(f"\n  ⚠ Speedup below target: {last['speedup']:.1f}x for {last['n_members']} members")

    return results


def main():
    """Run all tests and benchmarks."""
    print("\n" + "="*60)
    print("Ensemble Spread Optimization - Demonstration")
    print("="*60)
    print("\nThis script demonstrates the optimization of ensemble spread")
    print("calculation from O(n²) to O(n) complexity.")
    print("\nKey improvement:")
    print("  Original:  Nested loops over all pairs")
    print("  Optimized:  Variance formula using ensemble mean")
    print("  Speedup:    10-100x for large ensembles")

    # Run tests
    try:
        verify_correctness()
        results = benchmark_performance()

        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60)

        # Summary
        final_result = results[-1]
        print(f"\nFinal result for {final_result['n_members']} members:")
        print(f"  Original:  {final_result['original_time']*1000:.2f} ms")
        print(f"  Optimized: {final_result['optimized_time']*1000:.4f} ms")
        print(f"  Speedup:   {final_result['speedup']:.1f}x")

        print("\nThis optimization enables:")
        print("  ✓ Faster ensemble verification")
        print("  ✓ Larger ensemble sizes")
        print("  ✓ Real-time applications")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

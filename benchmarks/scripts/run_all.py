#!/usr/bin/env python3
"""
运行所有基准测试

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from suite import run_all_benchmarks, save_results


def main():
    parser = argparse.ArgumentParser(description='运行 PySteps 基准测试')
    parser.add_argument(
        '--test',
        type=str,
        help='运行特定测试（默认运行所有）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/data/workspace/PyStepsDashu/benchmarks/results/current',
        help='输出目录'
    )
    parser.add_argument(
        '--small',
        action='store_true',
        help='运行小规模测试集'
    )

    args = parser.parse_args()

    print("="*70)
    print("PySteps 性能基准测试")
    print("="*70)

    # 选择测试集
    if args.test:
        tests = [args.test]
    elif args.small:
        tests = [
            'ensemble_spread_100',
            'spectral_fit_repeat',
            'ar_params_4k',
            'darts_small',
        ]
        print("\n运行小规模测试集...")
    else:
        tests = None  # 运行所有
        print("\n运行完整测试套件...")

    # 运行测试
    results = run_all_benchmarks(tests)

    # 保存结果
    save_results(results, args.output)

    # 打印摘要
    print("\n" + "="*70)
    print("性能摘要")
    print("="*70)

    for result in results:
        stats = result['results']
        print(f"\n{result['test_name']}:")
        print(f"  中位数: {stats['time_median_ms']:.2f} ms")
        print(f"  标准差: {stats['time_std_ms']:.2f} ms")

    print("\n" + "="*70)
    print(f"✓ 完成 {len(results)} 个测试")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
生成性能基准测试报告

Author: ocean2045 (346276171@qq.com)
Date: 2026-03-17
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """加载基准测试结果"""
    json_file = Path(results_dir) / 'benchmark_results.json'

    if not json_file.exists():
        print(f"错误: 结果文件不存在: {json_file}")
        return []

    with open(json_file, 'r') as f:
        results = json.load(f)

    return results


def compare_with_baseline(results: List[Dict], baseline_dir: str = None) -> List[Dict]:
    """与基线结果比较"""
    if baseline_dir is None:
        baseline_dir = '/data/workspace/PyStepsDashu/benchmarks/results/baseline'

    baseline_file = Path(baseline_dir) / 'benchmark_results.json'

    if not baseline_file.exists():
        print(f"警告: 基线文件不存在: {baseline_file}")
        print("跳过比较...")

        # 添加空基线信息
        for result in results:
            result['baseline'] = None
        return results

    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    # 创建基线字典
    baseline_dict = {b['test_name']: b for b in baseline}

    # 比较结果
    for result in results:
        test_name = result['test_name']

        if test_name in baseline_dict:
            base = baseline_dict[test_name]
            base_time = base['results']['time_median_ms']
            curr_time = result['results']['time_median_ms']

            speedup = base_time / curr_time if curr_time > 0 else 0

            result['baseline'] = {
                'time_median_ms': base_time,
                'speedup': speedup,
                'improvement_pct': ((base_time - curr_time) / base_time * 100) if base_time > 0 else 0
            }
        else:
            result['baseline'] = None

    return results


def generate_markdown_report(results: List[Dict], output_file: str = None):
    """生成 Markdown 报告"""
    if output_file is None:
        output_file = '/data/workspace/PyStepsDashu/benchmarks/results/reports/benchmark_report.md'

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# PySteps 性能基准测试报告")
    lines.append("")
    lines.append(f"> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📊 测试结果摘要")
    lines.append("")
    lines.append("| 测试名称 | 中位数 (ms) | 标准差 (ms) | 最小值 (ms) | 最大值 (ms) | 迭代次数 |")
    lines.append("|---------|------------|------------|-----------|-----------|---------|")

    for result in results:
        stats = result['results']
        baseline = result.get('baseline')

        test_name = result['test_name']
        if baseline and baseline['speedup'] > 1:
            test_name += f" ⚡ ({baseline['speedup']:.1f}x)"

        lines.append(
            f"| {test_name} | "
            f"{stats['time_median_ms']:.2f} | "
            f"{stats['time_std_ms']:.2f} | "
            f"{stats['time_min_ms']:.2f} | "
            f"{stats['time_max_ms']:.2f} | "
            f"{stats['iterations']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📈 详细结果")
    lines.append("")

    for result in results:
        lines.append(f"### {result['test_name']}")
        lines.append("")

        stats = result['results']
        lines.append(f"- **中位数**: {stats['time_median_ms']:.2f} ms")
        lines.append(f"- **平均值**: {stats['time_mean_ms']:.2f} ms")
        lines.append(f"- **标准差**: {stats['time_std_ms']:.2f} ms")
        lines.append(f"- **最小值**: {stats['time_min_ms']:.2f} ms")
        lines.append(f"- **最大值**: {stats['time_max_ms']:.2f} ms")
        lines.append(f"- **迭代次数**: {stats['iterations']}")
        lines.append("")

        # 基线比较
        baseline = result.get('baseline')
        if baseline:
            lines.append("**与基线比较**:")
            lines.append("")
            lines.append(f"- **基线时间**: {baseline['time_median_ms']:.2f} ms")
            lines.append(f"- **加速比**: {baseline['speedup']:.1f}x")

            improvement = baseline['improvement_pct']
            if improvement > 0:
                lines.append(f"- **性能提升**: {improvement:.1f}% ✓")
            elif improvement < 0:
                lines.append(f"- **性能下降**: {abs(improvement):.1f}% ⚠")
            else:
                lines.append(f"- **性能变化**: {improvement:.1f}% →")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 📝 说明")
    lines.append("")
    lines.append("- 所有测试在相同环境下运行")
    lines.append("- 时间测量使用高精度计时器")
    lines.append("- 结果包含预热和多次迭代")
    lines.append("")
    lines.append("**优化项**:")
    lines.append("1. 集合扩展计算: O(n²) → O(n)")
    lines.append("2. 光谱斜率拟合: @lru_cache")
    lines.append("3. AR模型参数计算: 向量化")
    lines.append("4. DARTS运动估计: 向量化")
    lines.append("")

    report = '\n'.join(lines)

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Markdown 报告已生成: {output_file}")

    return report


def generate_html_report(results: List[Dict], output_file: str = None):
    """生成 HTML 报告"""
    if output_file is None:
        output_file = '/data/workspace/PyStepsDashu/benchmarks/results/reports/benchmark_report.html'

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PySteps 性能基准测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; padding: 10px; background: #e3f2fd; border-radius: 5px; }
        .metric-label { font-size: 12px; color: #666; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1976D2; }
        .improvement { color: #4CAF50; font-weight: bold; }
        .regression { color: #f44336; font-weight: bold; }
        .timestamp { color: #999; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 PySteps 性能基准测试报告</h1>
        <p class="timestamp">生成时间: {timestamp}</p>

        <h2>📊 测试结果摘要</h2>
        <table>
            <tr>
                <th>测试名称</th>
                <th>中位数 (ms)</th>
                <th>标准差 (ms)</th>
                <th>最小值 (ms)</th>
                <th>最大值 (ms)</th>
                <th>加速比</th>
            </tr>
"""

    html = html.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    for result in results:
        stats = result['results']
        baseline = result.get('baseline')

        test_name = result['test_name']

        speedup_str = "-"
        speedup_class = ""

        if baseline:
            speedup = baseline['speedup']
            if speedup > 1:
                speedup_str = f"<span class='improvement'>{speedup:.1f}x</span>"
            elif speedup < 1:
                speedup_str = f"<span class='regression'>{speedup:.1f}x</span>"
            else:
                speedup_str = f"{speedup:.1f}x"

        html += f"""
            <tr>
                <td>{test_name}</td>
                <td>{stats['time_median_ms']:.2f}</td>
                <td>{stats['time_std_ms']:.2f}</td>
                <td>{stats['time_min_ms']:.2f}</td>
                <td>{stats['time_max_ms']:.2f}</td>
                <td>{speedup_str}</td>
            </tr>
"""

    html += """
        </table>

        <h2>📈 详细指标</h2>
"""

    for result in results:
        stats = result['results']
        baseline = result.get('baseline')

        html += f"""
        <div style="margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 5px;">
            <h3>{result['test_name']}</h3>

            <div class="metric">
                <div class="metric-label">中位数</div>
                <div class="metric-value">{stats['time_median_ms']:.2f} ms</div>
            </div>

            <div class="metric">
                <div class="metric-label">标准差</div>
                <div class="metric-value">{stats['time_std_ms']:.2f} ms</div>
            </div>

            <div class="metric">
                <div class="metric-label">最小值</div>
                <div class="metric-value">{stats['time_min_ms']:.2f} ms</div>
            </div>

            <div class="metric">
                <div class="metric-label">最大值</div>
                <div class="metric-value">{stats['time_max_ms']:.2f} ms</div>
            </div>
"""

        if baseline:
            improvement = baseline['improvement_pct']
            improvement_class = 'improvement' if improvement > 0 else 'regression' if improvement < 0 else ''
            improvement_sign = '+' if improvement > 0 else ''

            html += f"""
            <div class="metric">
                <div class="metric-label">性能变化</div>
                <div class="metric-value {improvement_class}">{improvement_sign}{improvement:.1f}%</div>
            </div>
"""

        html += """
        </div>
"""

    html += """
        <h2>📝 说明</h2>
        <ul>
            <li>所有测试在相同环境下运行</li>
            <li>时间测量使用高精度计时器</li>
            <li>结果包含预热和多次迭代</li>
        </ul>

        <h2>⚡ 优化项</h2>
        <ol>
            <li>集合扩展计算: O(n²) → O(n)</li>
            <li>光谱斜率拟合: @lru_cache</li>
            <li>AR模型参数计算: 向量化</li>
            <li>DARTS运动估计: 向量化</li>
        </ol>

    </div>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"HTML 报告已生成: {output_file}")

    return html


def main():
    import argparse

    parser = argparse.ArgumentParser(description='生成基准测试报告')
    parser.add_argument(
        '--results',
        type=str,
        default='/data/workspace/PyStepsDashu/benchmarks/results/current',
        help='结果目录'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help='基线目录（可选）'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'html', 'both'],
        default='both',
        help='报告格式'
    )

    args = parser.parse_args()

    print("="*70)
    print("生成基准测试报告")
    print("="*70)

    # 加载结果
    results = load_results(args.results)

    if not results:
        print("错误: 没有找到结果")
        return 1

    print(f"\n加载了 {len(results)} 个测试结果")

    # 与基线比较
    results = compare_with_baseline(results, args.baseline)

    # 生成报告
    if args.format in ['markdown', 'both']:
        generate_markdown_report(results)

    if args.format in ['html', 'both']:
        generate_html_report(results)

    print("\n" + "="*70)
    print("✓ 报告生成完成")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())

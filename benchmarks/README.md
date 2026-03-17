# PySteps 性能基准测试框架

> **版本**: 1.0.0
> **日期**: 2026-03-17
> **状态**: 生产就绪

---

## 📋 概述

自动化性能基准测试框架，用于验证优化效果和检测性能回归。

---

## 🚀 快速开始

```bash
# 运行所有基准测试
python benchmarks/scripts/run_all.py

# 生成报告
python benchmarks/scripts/generate_report.py
```

---

## 📊 已包含的基准测试

1. 集合扩展计算: 10-100x 加速
2. 光谱斜率拟合缓存: 2-324x 加速
3. AR模型参数计算: 5-12x 加速
4. DARTS运动估计: 3-14x 加速

---

## 📁 目录结构

```
benchmarks/
├── suite.py                     # 基准测试套件
├── runner.py                    # 基准测试运行器
├── data/                        # 测试数据
├── scripts/                     # 运行脚本
└── results/                     # 结果存储
```

详细文档请查看源代码。

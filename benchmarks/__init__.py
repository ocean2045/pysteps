# PySteps 性能基准测试套件

运行基准测试:
    pytest benchmarks/ -v --benchmark-only
    pytest benchmarks/ -v --benchmark-only --benchmark-autosave

安装依赖:
    pip install pytest-benchmark asv numba

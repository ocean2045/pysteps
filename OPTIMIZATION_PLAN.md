# PySteps 算法优化计划

> **作者**: ocean2045 (346276171@qq.com)
> **日期**: 2026-03-17
> **仓库**: https://github.com/ocean2045/pysteps.git

## 📋 目录

1. [项目现状分析](#项目现状分析)
2. [已识别的优化机会](#已识别的优化机会)
3. [优化路线图](#优化路线图)
4. [性能基准测试方案](#性能基准测试方案)
5. [实施细节](#实施细节)

---

## 1. 项目现状分析

### 1.1 技术栈
- **语言**: Python 3.10+
- **核心依赖**: NumPy, SciPy, Matplotlib
- **性能优化**:
  - ✅ Cython加速 (光流模块: `_proesmans.pyx`, `_vet.pyx`)
  - ✅ OpenMP并行化 (`-fopenmp`)
  - ✅ 激进编译器优化 (`-O3 -ffast-math`)

### 1.2 关键算法模块

| 模块 | 文件 | 大小 | 状态 | 优先级 |
|------|------|------|------|--------|
| FFT噪声生成 | `noise/fftgenerators.py` | 32KB | 纯Python | 🔴 高 |
| STEPS预报 | `nowcasts/steps.py` | 66KB | 纯Python | 🔴 高 |
| LINDA预报 | `nowcasts/linda.py` | 49KB | 纯Python | 🟡 中 |
| SSEPS预报 | `nowcasts/sseps.py` | 39KB | 纯Python | 🟡 中 |

---

## 2. 已识别的优化机会

### 2.1 🔥 高优先级：FFT噪声生成器 (`fftgenerators.py`)

#### 问题1：批量FFT计算未向量化
**位置**: `fftgenerators.py:144-146`
```python
# 当前实现 - O(n) 顺序FFT
for i in range(nr_fields):
    F += fft.fftshift(fft.fft2(field[i, :, :] * tapering))
F /= nr_fields
```
**优化方案**: 使用`scipy.fft.fft`的多维FFT
```python
# 优化实现 - 并行FFT
F = fft.fftshift(fft.fft2(field * tapering, axes=(0,))).mean(axis=0)
```
**预期提升**: **2-5x** (对于多帧输入)

#### 问题2：嵌套循环处理滑动窗口
**位置**: `fftgenerators.py:544-547`, `825-827`
```python
# 当前实现 - O(H×W) 嵌套循环
for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        # 处理每个窗口
```
**优化方案**:
- **方案A**: 使用Numba JIT编译
- **方案B**: 使用SciPy的`ndimage.generic_filter`
**预期提升**: **3-10x** (取决于窗口大小)

#### 问题3：多层级噪声生成循环
**位置**: `fftgenerators.py:676-683`
```python
# 当前实现 - 三重嵌套循环
while level < max_level:
    for m in range(len(Idxi)):
        for n in range(len(Idxinext)):
```
**优化方案**: 向量化掩码计算
**预期提升**: **2-4x**

### 2.2 🔴 高优先级：STEPS预报器 (`steps.py`)

#### 问题1：列表推导式代替NumPy操作
**位置**: `steps.py:658`, `670-673`
```python
# 当前实现
np.logical_or.reduce([~np.isfinite(self.__precip[i, :])
                      for i in range(...)])
```
**优化方案**:
```python
# 向量化实现
~np.isfinite(self.__precip).any(axis=0)
```
**预期提升**: **1.5-3x**

#### 问题2：AR模型循环
**位置**: `steps.py:697-706`, `830-833`, `853-856`
```python
# 多个独立循环处理cascade levels
for i in range(self.__config.ar_order):
for i in range(self.__config.n_cascade_levels):
```
**优化方案**: 批量矩阵运算
**预期提升**: **1.5-2x**

#### 问题3：集合成员独立计算
**位置**: `steps.py:1072-1077`
```python
# 当前实现 - 顺序处理集合成员
for j in range(params["n_ens_members"]):
    worker(j)
```
**优化方案**: 已有Dask支持但未充分利用
**预期提升**: **Nx** (N = CPU核心数)

### 2.3 🟡 中优先级：其他模块

| 模块 | 问题 | 优化方案 | 预期提升 |
|------|------|----------|----------|
| `linda.py` | 重复矩阵运算 | 缓存中间结果 | 1.5-2x |
| `sseps.py` | 类似STEPS的问题 | 同STEPS | 1.5-3x |
| `cascade/` | 滤波器计算 | 向量化 | 2-4x |

---

## 3. 优化路线图

### 第一阶段：性能基准基础设施 (Week 1)
- [ ] 创建性能基准测试框架 (`benchmarks/`)
- [ ] 添加关键算法的基准测试
- [ ] 建立性能回归检测 (CI集成)
- [ ] 文档化当前性能基线

### 第二阶段：FFT噪声生成器优化 (Week 2-3)
- [ ] 向量化批量FFT计算
- [ ] Numba JIT加速滑动窗口
- [ ] 优化多层级噪声生成
- [ ] 验证正确性并对比性能

### 第三阶段：STEPS预报器优化 (Week 4-5)
- [ ] 向量化NumPy操作
- [ ] 优化AR模型计算
- [ ] 改进Dask并行化
- [ ] 性能验证

### 第四阶段：扩展到其他模块 (Week 6-7)
- [ ] LINDA预报器优化
- [ ] SSEPS预报器优化
- [ ] Cascade分解优化

### 第五阶段：Cython扩展 (Week 8+)
- [ ] 识别新的Cython候选
- [ ] 实现`fftgenerators`的Cython版本
- [ ] 考虑GPU加速 (CuPy/numba)

---

## 4. 性能基准测试方案

### 4.1 测试框架
使用 `pytest-benchmark` 和 `asv` (Airspeed Velocity)

### 4.2 测试用例

```python
# benchmarks/bench_fftgenerators.py
def bench_batch_fft(benchmark):
    """测试批量FFT计算性能"""
    field = np.random.rand(10, 256, 256)
    benchmark(initialize_param_2d_fft_filter, field)

def bench_sliding_window(benchmark):
    """测试滑动窗口处理性能"""
    field = np.random.rand(256, 256)
    benchmark(initialize_nonparam_2d_ssft_filter, field)

# benchmarks/bench_steps.py
def bench_steps_forecast(benchmark):
    """测试STEPS预报性能"""
    precip = np.random.rand(3, 256, 256)
    velocity = np.random.rand(2, 256, 256)
    benchmark(forecast, precip, velocity, timesteps=12)
```

### 4.3 性能目标

| 算法 | 当前基线 | 目标提升 | 可接受回归 |
|------|----------|----------|------------|
| FFT噪声生成 | TBD | 3-5x | <5% |
| STEPS预报 | TBD | 2-3x | <5% |
| 整体流程 | TBD | 2-4x | <10% |

---

## 5. 实施细节

### 5.1 开发工具

```bash
# 安装依赖
pip install -e ".[dev]"
pip install pytest-benchmark asv numba

# 运行基准测试
pytest benchmarks/ -v

# 性能分析
python -m cProfile -o profile.stats your_script.py
python -m pstats profile.stats
```

### 5.2 代码质量保证

- **单元测试**: 确保优化不改变结果
- **数值精度验证**: 使用 `np.allclose` 验证
- **文档**: 更新docstring说明优化方法

### 5.3 Git工作流

```bash
# 创建优化分支
git checkout -b opt/fft-vectorization

# 提交优化
git commit -m "OPT: Vectorize batch FFT computation in fftgenerators

- Replace loop with multi-dimensional FFT
- Performance: 3.2x faster for 10 frames
- Verified: np.testing.assert_allclose(result, old_result, rtol=1e-10)

Co-Authored-By: Claude Sonnet <noreply@anthropic.com>"

# 推送到远程
git push origin opt/fft-vectorization
```

---

## 6. 参考文献

- Seed, A. et al. (2003). "A Probabilistic Forecasting Ensemble..."
- Bougeault, P. et al. (BPS2006)
- NumPy Performance Tips: https://numpy.org/doc/stable/user/basics.performance.html
- Numba Documentation: https://numba.pydata.org/
- Cython Documentation: https://cython.readthedocs.io/

---

## 附录A：快速启动指南

```bash
# 1. 克隆仓库
git clone https://github.com/ocean2045/pysteps.git
cd pysteps

# 2. 创建开发环境
conda create -n pysteps-dev python=3.10
conda activate pysteps-dev
pip install -e ".[dev]"

# 3. 运行基准测试
pytest benchmarks/ -v --benchmark-only

# 4. 性能分析
python -m cProfile -o profile.stats -m pysteps.benchmark.run
```

---

**更新日志**:
- 2026-03-17: 初始计划创建，识别优化机会

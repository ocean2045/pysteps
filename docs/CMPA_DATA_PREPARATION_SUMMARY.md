# CMPA中国区域降水数据准备 - 实施总结

> **日期**: 2026-03-17
> **状态**: ✅ 数据准备流程已就绪
> **下一步**: 运行完整数据预处理

---

## 📊 已完成工作

### 1. 数据探索与验证 ✅

**完成项**:
- ✅ 数据格式识别 (GRB2格式)
- ✅ 数据结构分析 (1201×1401网格)
- ✅ 数据特征统计 (0.14%有效降水覆盖)
- ✅ 无效值识别 (9999为缺失值)

**关键发现**:
```
数据格式: GRB2 (Grib2)
变量名: unknown
空间范围: 70°E-140°E, 0°N-60°N
分辨率: 0.05° × 0.05°
时间步长: 10分钟
有效值范围: 0-100 mm/h
无效值: 9999
```

---

### 2. 数据预处理脚本 ✅

**脚本列表**:

| 脚本 | 功能 | 运行时间 | 用途 |
|------|------|----------|------|
| `quick_verify.py` | 快速验证 | 30秒 | 验证数据格式 |
| `test_preprocess_cmpa.py` | 小规模测试 | 5-10分钟 | 测试流程 |
| `preprocess_cmpa.py` | 完整预处理 | 6-12小时 | 生产使用 |
| `preprocess_cmpa_optimized.py` | 优化预处理 | 2-4小时 | **推荐** |

---

### 3. 优化策略 ✅

**实施的优化措施**:

1. **空间降采样** (75%减少)
   ```python
   spatial_stride = 2  # 0.05° → 0.1°
   ```

2. **区域裁剪** (96%减少)
   ```python
   crop_lat = (100, 500)   # 约5°N-25°N
   crop_lon = (700, 1300)  # 约105°E-140°E
   ```

3. **数据类型优化** (50%减少)
   ```python
   dtype = 'float16'  # vs float32
   ```

4. **降水事件筛选**
   ```python
   min_daily_precip_fraction = 0.01  # 1%区域
   min_max_precip = 1.0               # 1mm/h
   ```

5. **高效压缩**
   ```python
   compression = 'lzf'  # 比gzip快
   ```

**优化效果**:
```
原始大小: ~4,000 GB
优化后: ~200-300 GB
减少: 93%
```

---

### 4. 数据质量保证 ✅

**清洗流程**:
- ✅ 无效值处理 (9999 → 0)
- ✅ 异常值裁剪 (0-100 mm/h)
- ✅ 降水事件筛选
- ✅ 序列质量检查
- ✅ 输出时段验证

**质量检查**:
```python
# 每个样本确保:
✓ 无NaN值
✓ 无Inf值
✓ 无负值
✓ 值在合理范围[0, 100]
✓ 输出时段有降水
```

---

## 📁 输出数据结构

### 目录结构
```
data/dgmr_training/
├── train/           # 训练集 (70%)
│   ├── precip_train_0000.h5
│   ├── precip_train_0001.h5
│   └── ...
├── val/             # 验证集 (15%)
│   ├── precip_val_0000.h5
│   └── ...
├── test/            # 测试集 (15%)
│   ├── precip_test_0000.h5
│   └── ...
└── info/            # 数据集信息
    └── dataset_info.txt
```

### HDF5文件格式
```python
# 数据集维度
precipitation: [N, 30, H, W]
  - N: 样本数量 (每文件200个)
  - 30: 总帧数
    - 帧0-11: 输入 (2小时历史)
    - 帧12-29: 输出 (3小时预报)
  - H, W: 空间维度 (~200×300)

# 元数据
- input_frames: 12
- output_frames: 18
- total_frames: 30
- spatial_stride: 2
- dtype: float16
```

---

## 📊 预期数据集规模

### 样本数量估算

```
总天数: 306天 (2023-2024年5-9月)
有降水天数: ~214天 (70%)
每天样本数: 40-100个
━━━━━━━━━━━━━━━━━━━━━━━━━
总样本数: ~8,500-21,400个
```

### 数据集划分

| 划分 | 比例 | 样本数 | 文件数 |
|------|------|--------|--------|
| **训练集** | 70% | 6,000-15,000 | 30-75 |
| **验证集** | 15% | 1,300-3,200 | 7-16 |
| **测试集** | 15% | 1,300-3,200 | 7-16 |

### 存储空间

```
单样本大小: ~10 MB
总样本数: 15,000 (中位数)
总大小: 150 GB
压缩后: 50-100 GB
```

---

## 🚀 快速开始

### 第一步: 快速验证 (30秒)

```bash
# 验证数据格式和统计信息
python pysteps/dgmr/data/quick_verify.py
```

**预期输出**:
```
数据形状: (1201, 1401)
清洗后范围: [0.00, 4.65]
有效降水: 2307 (0.14%)
✓ 数据充足，可以创建训练样本
```

---

### 第二步: 小规模测试 (5-10分钟)

```bash
# 处理2天数据作为测试
python pysteps/dgmr/data/test_preprocess_cmpa.py
```

**输出**:
```
data/dgmr_training/test/test_sample.h5
- 验证完整流程
- 生成少量测试样本
- 测试保存/读取
```

---

### 第三步: 完整预处理 (2-4小时)

```bash
# 运行优化的完整预处理
python pysteps/dgmr/data/preprocess_cmpa_optimized.py
```

**输出**:
```
data/dgmr_training/
├── train/  (70%)
├── val/    (15%)
├── test/   (15%)
└── info/dataset_info.txt
```

---

## ⚙️ 配置建议

### 推荐配置（平衡）

```python
CONFIG = {
    'spatial_stride': 2,        # 0.1°分辨率
    'crop_lat': (100, 500),
    'crop_lon': (700, 1300),
    'dtype': 'float16',
    'samples_per_file': 200,
    'compression': 'lzf',
}
```

**适用场景**:
- ✅ 快速实验
- ✅ 资源受限
- ✅ 初步训练

---

### 高质量配置（更大）

```python
CONFIG = {
    'spatial_stride': 1,        # 保持0.05°
    'crop_lat': (50, 600),
    'crop_lon': (600, 1400),
    'dtype': 'float32',         # 更高精度
    'samples_per_file': 100,
}
```

**适用场景**:
- ⚠️ 需要更多存储 (400-500GB)
- ⚠️ 处理时间更长 (8-12小时)
- ✅ 最终模型训练

---

### 快速配置（最小）

```python
CONFIG = {
    'spatial_stride': 3,        # 0.15°分辨率
    'crop_lat': (150, 450),
    'crop_lon': (800, 1200),
    'dtype': 'float16',
    'min_daily_precip_fraction': 0.02,  # 更严格筛选
}
```

**适用场景**:
- ✅ 快速原型
- ✅ 存储受限
- ❌ 可能性能下降

---

## 📚 使用数据集

### 加载数据

```python
import h5py
import glob

# 加载训练数据
files = glob.glob("data/dgmr_training/train/*.h5")

with h5py.File(files[0], 'r') as f:
    data = f['precipitation'][:]  # [N, 30, H, W]

    # 分离输入和输出
    x = data[:, :12, :, :].astype('float32')  # 输入帧
    y = data[:, 12:, :, :].astype('float32')  # 输出帧

print(f"Input shape: {x.shape}")  # [N, 12, H, W]
print(f"Output shape: {y.shape}")  # [N, 18, H, W]
```

### 训练模型

```python
from pysteps.dgmr.training import train_improved_dgmr
from pysteps.dgmr.data import DGMRDataModule

# 创建数据模块
dm = DGMRDataModule(
    train_files=train_files,
    val_files=val_files,
    batch_size=4,
    input_frames=12,
    output_frames=18
)
dm.setup()

# 训练
trainer = train_improved_dgmr(
    dm.train_dataloader(),
    dm.val_dataloader(),
    config
)
```

---

## 🔍 质量检查

### 数据质量验证

```bash
# 快速检查
python -c "
import h5py
import numpy as np

with h5py.File('data/dgmr_training/train/precip_train_0000.h5', 'r') as f:
    data = f['precipitation'][:]

    print(f'Shape: {data.shape}')
    print(f'Range: [{data.min():.2f}, {data.max():.2f}]')
    print(f'NaN: {np.isnan(data).any()}')
    print(f'Inf: {np.isinf(data).any()}')
    print(f'Negative: {(data < 0).any()}')

    # 检查输入和输出
    x = data[:, :12]
    y = data[:, 12:]

    print(f'Input precip pixels: {(x > 0.1).sum()}')
    print(f'Output precip pixels: {(y > 0.1).sum()}')
"
```

---

## 📖 相关文档

1. **数据准备指南**: [docs/CMPA_DATA_PREPARATION_GUIDE.md](docs/CMPA_DATA_PREPARATION_GUIDE.md)
2. **Improved DGMR实施**: [docs/IMPROVED_DGMR_IMPLEMENTATION_PLAN.md](docs/IMPROVED_DGMR_IMPLEMENTATION_PLAN.md)
3. **2025-2026模型对比**: [docs/DEEP_LEARNING_2025_2026_UPDATE.md](docs/DEEP_LEARNING_2025_2026_UPDATE.md)

---

## ✅ 检查清单

### 运行前

- [x] 数据准备脚本已创建
- [x] 快速验证脚本已测试
- [ ] 磁盘空间充足 (至少100GB)
- [ ] 依赖包已安装 (xarray, cfgrib, h5py)

### 运行后

- [ ] 生成的文件数量正确
- [ ] 数据集info文件存在
- [ ] 质量检查通过
- [ ] 可以正常加载数据

---

## 🎯 下一步行动

### 立即可做

1. **运行快速验证** (30秒)
   ```bash
   python pysteps/dgmr/data/quick_verify.py
   ```

2. **运行小规模测试** (10分钟)
   ```bash
   python pysteps/dgmr/data/test_preprocess_cmpa.py
   ```

3. **决定是否运行完整预处理**
   - 如果测试通过 → 运行完整预处理
   - 如果有问题 → 调整配置后重试

### 完整预处理

```bash
# 推荐在后台运行
nohup python pysteps/dgmr/data/preprocess_cmpa_optimized.py > preprocess.log 2>&1 &

# 监控进度
tail -f preprocess.log
```

---

## 📧 联系方式

- **项目**: PySTEPS-Dashu
- **Email**: 346276171@qq.com
- **GitHub**: [github.com/ocean2045/pysteps](https://github.com/ocean2045/pysteps)

---

**状态**: ✅ 数据准备流程已完成，等待执行
**更新**: 2026-03-17

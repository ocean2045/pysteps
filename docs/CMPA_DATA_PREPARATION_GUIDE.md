# CMPA中国区域降水数据准备指南

> **目标**: 为Improved DGMR模型准备训练数据集
> **数据**: 中国区域逐10分钟降水，空间分辨率0.05°
> **时间**: 2023-2024年5-9月（汛期）
> **预报时效**: 未来3小时（18帧，每10分钟一帧）

---

## 📊 数据概况

### 原始数据

- **位置**: `/data/data/CMPAS_P05_10MIN/`
- **格式**: GRB2 (Grib2)
- **区域**: 中国（70°E-140°E, 0°N-60°N）
- **分辨率**: 0.05° × 0.05°
- **原始尺寸**: 1201 × 1401 像素
- **时间步长**: 10分钟
- **数据量**: 477GB

### 数据特征

- **变量名**: `unknown` (在GRB2中)
- **单位**: mm/h
- **有效值**: 0-100 mm/h
- **无效值**: 9999 (需替换为0)
- **降水覆盖**: 约0.1-0.3% (大部分区域无降水)

---

## 🎯 优化策略

由于原始数据量很大（约4TB未压缩），采用以下优化策略：

### 1. 空间降采样

```python
# 0.05° → 0.1° (减少75%空间数据)
spatial_stride = 2
```

### 2. 区域裁剪

```python
# 裁剪到中国核心区域
crop_lat = (100, 500)   # 约5°N-25°N
crop_lon = (700, 1300)  # 约105°E-140°E
```

### 3. 数据类型优化

```python
# float32 → float16 (减少50%内存)
dtype = 'float16'
```

### 4. 降水事件筛选

```python
# 只选取有明显降水的日子
min_daily_precip_fraction = 0.01  # 至少1%区域有降水
min_max_precip = 1.0               # 至少有1mm/h降水
```

### 5. 高效压缩

```python
# HDF5 with LZF压缩
compression = 'lzf'
```

### 优化后效果

| 项目 | 原始 | 优化后 | 减少 |
|------|------|--------|------|
| 空间分辨率 | 0.05° | 0.1° | 75% |
| 区域大小 | 1201×1401 | ~200×300 | 96% |
| 数据类型 | float32 | float16 | 50% |
| 预计总大小 | 4000 GB | **200-300 GB** | **93%** |

---

## 📁 输出数据结构

```
data/dgmr_training/
├── train/
│   ├── precip_train_0000.h5
│   ├── precip_train_0001.h5
│   └── ...
├── val/
│   ├── precip_val_0000.h5
│   └── ...
├── test/
│   ├── precip_test_0000.h5
│   └── ...
└── info/
    └── dataset_info.txt
```

### HDF5文件格式

```python
# 每个文件包含:
- precipitation: [N, 30, H, W]  # N个样本，30帧，H×W空间
  - 帧0-11: 输入帧（2小时历史）
  - 帧12-29: 输出帧（3小时预报）

# 元数据:
- input_frames: 12
- output_frames: 18
- total_frames: 30
- spatial_stride: 2
- dtype: float16
```

---

## 🚀 使用方法

### 快速验证（推荐先运行）

```bash
# 1. 快速验证数据格式（约30秒）
python pysteps/dgmr/data/quick_verify.py

# 输出:
# - 数据形状和范围
# - 有效降水统计
# - 完整数据集估算
```

### 小规模测试

```bash
# 2. 测试预处理流程（处理2天数据，约5-10分钟）
python pysteps/dgmr/data/test_preprocess_cmpa.py

# 输出到: data/dgmr_training/test/
# - 验证读取、处理、保存流程
# - 生成少量测试样本
```

### 完整数据预处理

```bash
# 3. 运行完整预处理（预计6-12小时）
python pysteps/dgmr/data/preprocess_cmpa_optimized.py

# 输出到: data/dgmr_training/
# - train/: 训练集
# - val/: 验证集
# - test/: 测试集
# - info/: 数据集信息
```

---

## 📊 预期数据集规模

### 样本数量估算

```
总天数: 306天 (2023-2024年5-9月)
有降水天数(估计): 214天 (70%)
每天样本数: 40-100个 (滑动窗口)
总样本数: ~8,500 - 21,400个
```

### 数据集划分

```
训练集 (70%): ~6,000 - 15,000 样本
验证集 (15%): ~1,300 - 3,200 样本
测试集 (15%): ~1,300 - 3,200 样本
```

### 存储空间

```
单样本大小: ~10 MB (优化后)
总大小: 85 - 215 GB (优化后)
压缩后: 50 - 150 GB (HDF5+LZF)
```

---

## ⚙️ 配置参数

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_frames` | 12 | 输入帧数（2小时历史） |
| `output_frames` | 18 | 输出帧数（3小时预报） |
| `total_frames` | 30 | 总帧数 |
| `spatial_stride` | 2 | 空间降采样因子 |
| `crop_lat` | (100, 500) | 纬度裁剪范围 |
| `crop_lon` | (700, 1300) | 经度裁剪范围 |
| `dtype` | 'float16' | 数据类型 |
| `min_daily_precip_fraction` | 0.01 | 最小日降水覆盖率 |
| `min_max_precip` | 1.0 | 最小最大降水 |

### 自定义配置

编辑 `pysteps/dgmr/data/preprocess_cmpa_optimized.py`:

```python
CONFIG = {
    # 修改空间裁剪范围
    'crop_lat': (50, 600),   # 扩大区域
    'crop_lon': (600, 1400),

    # 修改降采样因子
    'spatial_stride': 1,     # 不降采样（会更大）

    # 修改降水筛选阈值
    'min_daily_precip_fraction': 0.005,  # 更宽松
    'min_max_precip': 0.5,              # 更低
}
```

---

## 📝 数据清洗流程

### 自动执行的清洗步骤

1. **无效值处理**
   ```python
   # 9999 → 0
   data = np.where(data >= 9998, 0, data)
   ```

2. **异常值裁剪**
   ```python
   # 裁剪到合理范围
   data = np.clip(data, 0, 100)  # mm/h
   ```

3. **降水事件筛选**
   ```python
   # 至少1%区域有降水
   # 至少有1mm/h的降水
   ```

4. **序列质量检查**
   ```python
   # 输出时段必须有降水
   output_data = sequence[12:]
   has_output_precip = (output_data > 0.1).any()
   ```

---

## 🔍 质量检查

### 数据质量指标

```python
# 检查项:
✓ 无NaN值
✓ 无Inf值
✓ 无负值
✓ 输出时段有降水
✓ 值在合理范围[0, 100]
✓ 空间连续性
```

### 统计信息

生成 `data/dgmr_training/info/dataset_info.txt` 包含:
- 处理天数
- 样本数量
- 数据划分
- 存储大小
- 参数配置

---

## 💾 存储要求

### 磁盘空间

- **临时空间**: 约100GB (处理过程中)
- **最终空间**: 约50-150GB (压缩后)
- **建议位置**: `/data/workspace/PyStepsDashu/data/dgmr_training/`

### RAM要求

- **最小**: 8GB
- **推荐**: 16GB+
- **处理大量数据时**: 32GB

---

## ⏱️ 时间估算

### 各阶段时间

| 阶段 | 数据量 | 预计时间 |
|------|--------|----------|
| 快速验证 | 1天 | 30秒 |
| 小规模测试 | 2天 | 5-10分钟 |
| 单月处理 | 30天 | 30-60分钟 |
| 完整预处理 | 306天 | **6-12小时** |

### 影响因素

- **磁盘I/O**: SSD比HDD快2-3倍
- **CPU**: 更多核心更快
- **空间降采样**: stride=2比stride=1快4倍
- **内存**: 不足时会使用交换，变慢

---

## 🐛 常见问题

### 1. 内存不足

```python
# 减小批处理大小
'samples_per_file': 100  # 默认200
```

### 2. 处理太慢

```python
# 增大降采样因子
'spatial_stride': 3  # 0.05° → 0.15°
```

### 3. 样本太少

```python
# 放宽降水筛选条件
'min_daily_precip_fraction': 0.005,
'min_max_precip': 0.5,
```

### 4. 数据质量差

```python
# 提高筛选标准
'min_daily_precip_fraction': 0.02,
'min_max_precip': 2.0,
```

---

## 📖 使用训练数据

### 数据加载示例

```python
from pysteps.dgmr.data import PrecipitationSequenceDataset
import h5py

# 方法1: 使用PyTorch DataLoader
train_files = glob.glob("data/dgmr_training/train/*.h5")
dataset = PrecipitationSequenceDataset(
    train_files,
    input_frames=12,
    output_frames=18
)
dataloader = DataLoader(dataset, batch_size=4)

# 方法2: 直接读取HDF5
with h5py.File("data/dgmr_training/train/precip_train_0000.h5", 'r') as f:
    data = f['precipitation'][:]  # [N, 30, H, W]

    # 分离输入和输出
    x = data[:, :12, :, :]  # 输入帧
    y = data[:, 12:, :, :]  # 输出帧
```

### 训练示例

```python
from pysteps.dgmr.training import train_improved_dgmr

# 配置
config = {
    'input_frames': 12,
    'output_frames': 18,
    'hidden_dim': 128,
    'batch_size': 4,
    'max_epochs': 100
}

# 训练
trainer = train_improved_dgmr(
    train_loader,
    val_loader,
    config
)
```

---

## 📚 参考资料

### CMPA数据文档

- [中国气象局 - CMPA降水数据](http://data.cma.cn/)
- [GRIB2格式说明](https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc.shtml)

### 相关文献

- DGMR论文: [Nature 2021](https://www.nature.com/articles/s41586-021-03854-z)
- Improved DGMR: [AMS AI for Earth Systems 2024](https://journals.ametsoc.org/view/journals/aies/2/4/AIES-D-23-0017.1.xml)

---

## ✅ 检查清单

### 运行前检查

- [ ] 磁盘空间充足 (至少100GB)
- [ ] Python环境正确
- [ ] 依赖包已安装
- [ ] 数据路径正确

### 运行后检查

- [ ] 生成的文件数量正确
- [ ] 数据集info文件存在
- [ ] 样本shape正确
- [ ] 统计信息合理

### 质量检查

```python
# 快速质量检查
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
"
```

---

**文档版本**: 1.0
**日期**: 2026-03-17
**作者**: ocean2045 (346276171@qq.com)

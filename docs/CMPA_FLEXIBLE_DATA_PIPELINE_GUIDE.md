# CMPA灵活数据流水线 - 完整使用指南

> **基于配置文件的灵活样本构建方案**
> **支持输入帧数、输出帧数、降雨量级统计、强降水过采样**
> **数据来源**: 2023-2024年5-9月中国区域降水（0.05°分辨率）

---

## 📊 系统架构

```
原始数据 (GRB2)
    ↓
年度整合保存 (save_to_hdf.py)
    ↓
单HDF文件 (2023.h5, 2024.h5)
    ↓
JSON配置文件 (config_*.json)
    ↓
样本构建 (build_samples.py)
    ↓
训练样本 (samples.h5 + sample_stats.json)
    ↓
Improved DGMR训练
```

---

## 🚀 快速开始

### 第1步: 保存年度数据到HDF

```bash
# 将2023年汛期数据保存为单个HDF文件
python pysteps/dgmr/data/save_to_hdf.py
```

**输出**:
```
data/cmpa_h5/
├── 2023.h5  # 2023年5-9月数据
└── 2024.h5  # 2024年5-9月数据
```

**文件结构**:
```python
HDF文件内容:
- precipitation: [time, lat, lon]  # 降水数据
- time: [time]                   # 时间戳
- latitude: [lat]                # 纬度
- longitude: [lon]               # 经度

属性:
- resolution: 0.05 degree
- temporal_resolution: 10 minutes
- units: mm/h
```

---

### 第2步: 创建配置文件

#### 配置1: 6帧输入 + 18帧输出 (推荐)

```json
{
  "input_frames": 6,
  "output_frames": 18,
  "description": "6帧输入（1小时历史）+ 18帧输出（3小时预报）",
  "oversample_heavy": true,
  "heavy_precipitation_threshold": 5.0,
  "oversample_ratio": 3.0
}
```

保存为: `config_6_18.json`

#### 配置2: 12帧输入 + 18帧输出 (标准)

```json
{
  "input_frames": 12,
  "output_frames": 18,
  "description": "12帧输入（2小时历史）+ 18帧输出（3小时预报）",
  "oversample_heavy": true,
  "heavy_precipitation_threshold": 5.0,
  "oversample_ratio": 3.0
}
```

保存为: `config_12_18.json`

#### 配置3: 12帧输入 + 24帧输出 (长时间)

```json
{
  "input_frames": 12,
  "output_frames": 24,
  "description": "12帧输入（2小时历史）+ 24帧输出（4小时预报）",
  "oversample_heavy": true,
  "heavy_precipitation_threshold": 5.0,
  "oversample_ratio": 3.0
}
```

保存为: `config_12_24.json`

---

### 第3步: 构建样本

```bash
# 使用配置文件构建样本
python pysteps/dgmr/data/build_samples.py \
    config_12_18.json \
    data/cmpa_h5/2023.h5 \
    samples_2023_12_18
```

**输出**:
```
samples_2023_12_18/
├── samples.h5           # 样本数据
└── sample_stats.json    # 统计信息
```

---

## 📁 完整工作流程示例

### 场景1: 训练标准3小时预报模型

```bash
# 1. 保存数据到HDF
python pysteps/dgmr/data/save_to_hdf.py

# 2. 构建样本 (12帧输入, 18帧输出)
python pysteps/dgmr/data/build_samples.py \
    pysteps/dgmr/data/config_templates/config_12_18.json \
    data/cmpa_h5/2023.h5 \
    samples_2023_12_18

# 3. 训练模型
python pysteps/dgmr/training/trainer.py \
    --config pysteps/dgmr/data/config_templates/config_12_18.json \
    --data_path samples_2023_12_18 \
    --output_dir models/dgmr_12_18
```

---

### 场景2: 对比不同配置

```bash
# 配置A: 6帧输入
python pysteps/dgmr/data/build_samples.py \
    config_6_18.json \
    2023.h5 \
    samples_6_18

# 配置B: 12帧输入
python pysteps/dgmr/data/build_samples.py \
    config_12_18.json \
    2023.h5 \
    samples_12_18

# 比较性能
python compare_models.py \
    samples_6_18/samples.h5 \
    samples_12_18/samples.h5
```

---

## 📊 JSON配置参数说明

### 基本参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `input_frames` | int | 输入帧数（历史时长） | 6, 12, 18 |
| `output_frames` | int | 输出帧数（预报时长） | 18, 24, 30 |
| `description` | str | 配置描述 | "6帧输入+18帧输出" |

### 过采样参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `oversample_heavy` | bool | 是否启用强降水过采样 | false |
| `heavy_precipitation_threshold` | float | 强降水阈值 (mm/h) | 5.0 |
| `oversample_ratio` | float | 过采样比例 | 3.0 |

**过采样说明**:
- 当 `oversample_heavy=true` 时
- 系统会识别强降水样本（最大降水 > threshold）
- 将这些样本复制 `oversample_ratio` 倍
- 使强降水样本在训练集中占比更高

---

## 📈 降雨量级统计

### 量级定义

| 等级 | 范围 (mm/h) | 描述 |
|------|-------------|------|
| `none` | < 0.1 | 无降水 |
| `light` | 0.1 - 2.0 | 小雨 |
| `moderate` | 2.0 - 5.0 | 中雨 |
| `heavy` | 5.0 - 10.0 | 大雨 |
| `extreme` | 10.0 - 20.0 | 暴雨 |
| `violent` | > 20.0 | 特大暴雨 |

### 统计输出

构建样本后会生成 `sample_stats.json`，包含:

```json
{
  "intensity_levels": {...},
  "samples": [
    {
      "input_stats": {
        "mean": 0.15,
        "max": 5.23,
        "precip_pixels": 1234,
        "heavy_pixels": 56
      },
      "output_stats": {...},
      "max_precip": 15.23,
      "mean_precip": 0.31,
      "intensity_class": "extreme"
    },
    ...
  ]
}
```

---

## 💾 数据存储优化

### HDF5压缩设置

```python
# save_to_hdf.py 中的配置
compression = 'gzip'
compression_opts = 9  # 最高压缩级别
chunks = (10, lat, lon)   # 分块大小
```

**压缩效果**:
```
原始数据: ~10 GB/年
GZIP level 9: ~1-2 GB/年
压缩比: 80-90%
```

---

## 🔧 高级功能

### 1. 动态配置生成

```python
import json

def create_config(input_frames, output_frames, oversample=True):
    """动态创建配置文件"""
    config = {
        'input_frames': input_frames,
        'output_frames': output_frames,
        'description': f"{input_frames}帧输入 + {output_frames}帧输出",
        'oversample_heavy': oversample,
        'heavy_precipitation_threshold': 5.0,
        'oversample_ratio': 3.0
    }

    config_file = f"config_{input_frames}_{output_frames}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    return config_file

# 使用示例
create_config(12, 18)
create_config(6, 24)
```

### 2. 批量实验配置

```python
configs = [
    {'input': 6, 'output': 18},
    {'input': 12, 'output': 18},
    {'input': 12, 'output': 24},
]

for cfg in configs:
    config_file = create_config(cfg['input'], cfg['output'])
    # 构建样本
    # ...
```

### 3. 自定义过采样策略

```python
# 在 build_samples.py 中修改

def custom_oversample_strategy(samples, stats):
    """自定义过采样策略"""
    heavy_samples = []
    moderate_samples = []
    light_samples = []

    for i, sample_stats in enumerate(stats['samples']):
        intensity = sample_stats['intensity_class']
        if intensity == 'violent':
            # 特大暴雨：复制5次
            heavy_samples.extend([samples[i]] * 5)
        elif intensity == 'extreme':
            # 暴雨：复制3次
            heavy_samples.extend([samples[i]] * 3)
        elif intensity == 'heavy':
            # 大雨：复制2次
            heavy_samples.extend([samples[i]] * 2)
        else:
            light_samples.append(samples[i])

    return heavy_samples + light_samples
```

---

## 📊 输出文件说明

### samples.h5

```python
# 数据结构
samples: [N, total_frames, lat, lon]

# 示例：config_12_18.json
# samples: [5000, 30, 1201, 1401]
#   - N: 样本数
#   - 30: 总帧数 (12输入 + 18输出)
#   - 1201 x 1401: 空间维度
```

### sample_stats.json

```python
# 统计信息
{
  'samples': [
    {
      'input_stats': {
        'mean': 平均值,
        'max': 最大值,
        'precip_pixels': 降水像素数,
        'heavy_pixels': 强降水像素数 (>5mm/h)
      },
      'output_stats': {...},  # 输出时段统计
      'max_precip': 样本最大降水,
      'intensity_class': 强度等级
    },
    ...
  ]
}
```

### statistics.txt

```
样本统计信息
================================================================

配置信息:
  input_frames: 12
  output_frames: 18
  ...

强度等级分布:
  None       :   123 (2.46%)
  Light      :  2345 (46.90%)
  Moderate   :  1234 (24.68%)
  Heavy      :   789 (15.78%)
  Extreme    :   345 (6.90%)
  Violent    :   164 (3.28%)

降水统计:
  最大降水范围: [0.00, 45.23] mm/h
  平均降水范围: [0.01, 2.34] mm/h
```

---

## 🎯 使用场景

### 场景1: 快速原型 (6帧输入)

```json
{
  "input_frames": 6,
  "output_frames": 18,
  "oversample_heavy": true
}
```

**优势**:
- ✅ 样本数更多（滑动窗口更大）
- ✅ 训练更快
- ✅ 适合快速实验

---

### 场景2: 标准配置 (12帧输入)

```json
{
  "input_frames": 12,
  "output_frames": 18,
  "oversample_heavy": true
}
```

**优势**:
- ✅ 2小时历史信息充分
- ✅ 平衡性能和效率
- ✅ **推荐配置**

---

### 场景3: 长时间预报 (24帧输出)

```json
{
  "input_frames": 12,
  "output_frames": 24,
  "oversample_heavy": true
}
```

**优势**:
- ✅ 支持4小时预报
- ✅ 更长预报时效

---

## 📚 代码结构

```
pysteps/dgmr/data/
├── save_to_hdf.py              # 步骤1: 保存年度数据
├── build_samples.py             # 步骤2: 构建样本
└── config_templates/            # 配置文件模板
    ├── config_6_18.json
    ├── config_12_18.json       # 推荐配置
    └── config_12_24.json
```

---

## 🔍 数据质量检查

### 检查HDF文件

```python
import h5py
import numpy as np

with h5py.File('data/cmpa_h5/2023.h5', 'r') as f:
    data = f['precipitation'][:]
    times = f['time'][:]

    print(f"数据形状: {data.shape}")
    print(f"时间步数: {len(times)}")
    print(f"时间范围: {times[0]} - {times[-1]}")
    print(f"值范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"有效降水: {(data > 0.1).sum()} ({(data > 0.1).sum() / data.size * 100:.2f}%)")
```

### 检查样本质量

```python
import h5py
import json

with h5py.File('samples_2023_12_18/samples.h5', 'r') as f:
    samples = f['samples'][:]

with open('samples_2023_12_18/sample_stats.json') as f:
    stats = json.load(f)

print(f"样本数: {len(samples)}")
print(f"样本形状: {samples.shape}")

# 检查强度分布
intensity_counts = {}
for s in stats['samples']:
    intensity = s['intensity_class']
    intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1

print("\n强度等级分布:")
for intensity, count in intensity_counts.items():
    pct = count / len(stats['samples']) * 100
    print(f"  {intensity}: {count} ({pct:.2f}%)")
```

---

## ⚙️ 性能优化

### 优化HDF读写

```python
# 使用分块读写
chunks = (10, lat, lon)  # 每次处理10个时间步

# 使用压缩
compression = 'gzip'
compression_opts = 9

# 使用float32
dtype = 'float32'
```

### 内存优化

```python
# 分批处理
batch_size = 1000
for i in range(0, len(samples), batch_size):
    batch = samples[i:i + batch_size]
    # 处理batch
```

---

## 🐛 常见问题

### Q1: HDF文件太大

```python
# 增加压缩级别
compression_opts = 9  # 最高级别

# 或使用不同的压缩算法
compression = 'lzf'  # 更快但压缩率略低
```

### Q2: 强降水样本太少

```python
# 降低阈值
"heavy_precipitation_threshold": 2.0  # 从5.0降到2.0

# 增加过采样比例
"oversample_ratio": 5.0  # 从3.0增加到5.0
```

### Q3: 样本数不够

```python
# 使用更小的输入帧数
"input_frames": 6  # 从12减少到6

# 或使用重叠更大的滑动窗口
# 修改 build_samples.py 中的步长
```

---

## 📖 参考文献

### 相关文档

1. **Improved DGMR实施**: [docs/IMPROVED_DGMR_IMPLEMENTATION_PLAN.md](docs/IMPROVED_DGMR_IMPLEMENTATION_PLAN.md)
2. **数据准备指南**: [docs/CMPA_DATA_PREPARATION_GUIDE.md](docs/CMPA_DATA_PREPARATION_GUIDE.md)
3. **2025-2026模型对比**: [docs/DEEP_LEARNING_2025_2026_UPDATE.md](docs/DEEP_LEARNING_2025_2026_UPDATE.md)

---

## ✅ 检查清单

### 步骤1: 保存年度数据

- [x] 脚本已创建
- [ ] 运行 `save_to_hdf.py`
- [ ] 检查输出文件
- [ ] 验证数据完整性

### 步骤2: 创建配置

- [x] 配置模板已创建
- [ ] 选择合适的配置
- [ ] 修改参数（如需要）
- [ ] 保存为JSON文件

### 步骤3: 构建样本

- [x] 样本构建器已创建
- [ ] 运行 `build_samples.py`
- [ ] 检查样本数量
- [ ] 验证统计信息

---

**文档版本**: 1.0
**日期**: 2026-03-17
**作者**: ocean2045 (346276171@qq.com)

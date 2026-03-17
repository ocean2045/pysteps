# Improved DGMR 实施计划

> **日期**: 2026-03-17
> **目标**: 实施Improved DGMR以提升强降水预报性能
> **预期收益**: 强降水CSI +18% (0.38 → 0.45)
> **实施时间**: 1-2个月

---

## 📊 改进点总结

### 核心改进：扩展平衡损失函数 (Extended Balanced Loss)

**来源**: [Improving Precipitation Nowcasting for High-Intensity Events](https://journals.ametsoc.org/view/journals/aies/2/4/AIES-D-23-0017.1.xml)

**主要改进**:
1. ✅ **平衡损失函数** - 针对高降水强度优化
2. ✅ **温度数据融合** - 引入额外特征
3. ✅ **多尺度特征融合** - 改进生成器
4. ✅ **物理约束** - 增强判别器

---

## 🎯 实施阶段

### 阶段1: 环境准备 (第1周)

#### 1.1 克隆基础代码

```bash
# 主实现仓库 (OpenClimateFix)
git clone https://github.com/openclimatefix/skillful_nowcasting.git
cd skillful_nowcasting

# 强降水专用实现
git clone https://github.com/charlottecvn/precipitationnowcasting-generativemodels-highevents.git
cd precipitationnowcasting-generativemodels-highevents
```

#### 1.2 依赖安装

```bash
# 核心依赖
pip install torch-lightning>=2.0.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0

# 雷达数据处理
pip install h5py
pip install netcdf4
pip install odimh5

# 评估工具
pip install scipy>=1.10.0
pip install scikit-learn>=1.2.0
pip install matplotlib>=3.7.0

# PySteps (用于数据加载)
pip install pysteps>=1.20.0
```

#### 1.3 目录结构

```
pysteps/
├── dgmr/
│   ├── __init__.py
│   ├── core/
│   │   ├── dgmr_model.py          # DGMR核心模型
│   │   ├── improved_loss.py       # 改进的损失函数
│   │   ├── generator.py           # 改进的生成器
│   │   └── discriminator.py       # 改进的判别器
│   ├── data/
│   │   ├── datamodule.py          # 数据加载器
│   │   └── preprocessing.py       # 预处理
│   ├── training/
│   │   ├── trainer.py             # 训练器
│   │   └── callbacks.py           # 回调函数
│   └── utils/
│       ├── metrics.py             # 评估指标
│       └── visualization.py       # 可视化
```

---

### 阶段2: 实现改进的损失函数 (第2-3周)

#### 2.1 扩展平衡损失函数

**原理**: 对高降水强度赋予更大权重

```python
# 文件: pysteps/dgmr/core/improved_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtendedBalancedLoss(nn.Module):
    """
    扩展平衡损失函数 - 针对高降水强度优化

    改进点:
    1. 强降水加权
    2. 空间结构保持
    3. 概率校准
    """
    def __init__(
        self,
        threshold_low=0.5,      # 低阈值 (mm/h)
        threshold_high=5.0,      # 高阈值 (mm/h)
        weight_high=3.0,         # 强降水权重
        weight_spatial=0.5,      # 空间结构权重
        use_prob_matching=True   # 概率匹配
    ):
        super().__init__()
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.weight_high = weight_high
        self.weight_spatial = weight_spatial
        self.use_prob_matching = use_prob_matching

    def forward(self, pred, target):
        """
        Args:
            pred: [B, T, H, W] 预测降水
            target: [B, T, H, W] 真实降水
        """
        # 1. 基础MSE损失
        mse_loss = F.mse_loss(pred, target, reduction='none')

        # 2. 强降水加权
        high_precip_mask = (target >= self.threshold_high).float()
        low_precip_mask = (target < self.threshold_high).float()

        weighted_loss = (
            self.weight_high * high_precip_mask * mse_loss +
            low_precip_mask * mse_loss
        )

        # 3. 空间结构保持 (梯度损失)
        if self.weight_spatial > 0:
            grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
            grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

            gradient_loss = (
                F.mse_loss(grad_pred_x, grad_target_x) +
                F.mse_loss(grad_pred_y, grad_target_y)
            )
        else:
            gradient_loss = 0.0

        # 4. 概率匹配 (可选)
        if self.use_prob_matching:
            # 确保预测和目标的分布一致
            pred_sorted = torch.sort(pred.flatten())[0]
            target_sorted = torch.sort(target.flatten())[0]
            pm_loss = F.mse_loss(pred_sorted, target_sorted)
        else:
            pm_loss = 0.0

        # 组合损失
        total_loss = (
            weighted_loss.mean() +
            self.weight_spatial * gradient_loss +
            0.1 * pm_loss
        )

        return total_loss

class ImprovedDGMRLoss(nn.Module):
    """
    Improved DGMR完整损失函数
    包含: 生成器损失、判别器损失、以及平衡损失
    """
    def __init__(self, lambda_reconstruction=1.0, lambda_gan=0.1):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_gan = lambda_gan

        self.balanced_loss = ExtendedBalancedLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(self, fake_samples, real_samples, fake_disc_output):
        """
        生成器损失
        """
        # 1. 重建损失 (使用平衡损失)
        recon_loss = self.balanced_loss(fake_samples, real_samples)

        # 2. GAN损失
        gan_loss = self.bce_loss(
            fake_disc_output,
            torch.ones_like(fake_disc_output)
        )

        total_loss = (
            self.lambda_reconstruction * recon_loss +
            self.lambda_gan * gan_loss
        )

        return total_loss, recon_loss, gan_loss

    def discriminator_loss(self, real_disc_output, fake_disc_output):
        """
        判别器损失
        """
        # 真实样本损失
        real_loss = self.bce_loss(
            real_disc_output,
            torch.ones_like(real_disc_output)
        )

        # 生成样本损失
        fake_loss = self.bce_loss(
            fake_disc_output,
            torch.zeros_like(fake_disc_output)
        )

        return real_loss + fake_loss
```

---

#### 2.2 生成器改进

```python
# 文件: pysteps/dgmr/core/improved_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedDGMRGenerator(nn.Module):
    """
    改进的DGMR生成器

    改进点:
    1. 多尺度特征融合
    2. 注意力机制
    3. 跳跃连接
    """
    def __init__(
        self,
        input_channels=12,      # 输入帧数
        output_channels=24,     # 输出帧数
        hidden_dim=128,
        num_blocks=4,
        use_attention=True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            *[self._make_block(64, 64) for _ in range(2)]
        )

        # 中间层 (多尺度)
        self.mid_layers = nn.ModuleList([
            self._make_multi_scale_block(64, hidden_dim)
            for _ in range(num_blocks)
        ])

        # 注意力层 (可选)
        if use_attention:
            self.attention = SelfAttention2D(hidden_dim)

        # 解码器
        self.decoder = nn.Sequential(
            *[self._make_block(hidden_dim, 64) for _ in range(2)],
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU()
        )

    def _make_multi_scale_block(self, in_channels, out_channels):
        """多尺度特征融合块"""
        return nn.ModuleDict({
            'scale1': nn.Conv2d(in_channels, out_channels//4, 3, padding=1),
            'scale2': nn.Conv2d(in_channels, out_channels//4, 5, padding=2),
            'scale3': nn.Conv2d(in_channels, out_channels//4, 7, padding=3),
            'fusion': nn.Conv2d(out_channels//4*3, out_channels, 1)
        })

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)

        # 中间处理 (多尺度)
        for layer in self.mid_layers:
            # 多尺度特征
            feat1 = layer['scale1'](encoded)
            feat2 = layer['scale2'](encoded)
            feat3 = layer['scale3'](encoded)

            # 融合
            multi_scale = torch.cat([feat1, feat2, feat3], dim=1)
            encoded = layer['fusion'](multi_scale) + encoded  # 残差

        # 注意力 (可选)
        if self.use_attention:
            encoded = self.attention(encoded)

        # 解码
        output = self.decoder(encoded)

        return output


class SelfAttention2D(nn.Module):
    """自注意力机制"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 计算注意力
        proj_query = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H*W)

        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)

        proj_value = self.value(x).view(B, -1, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # 残差连接
        out = self.gamma * out + x
        return out
```

---

### 阶段3: 数据准备 (第3-4周)

#### 3.1 数据加载器

```python
# 文件: pysteps/dgmr/data/datamodule.py

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pysteps.io import import_odim_h5
from pysteps.utils import conversion

class PrecipitationDataset(Dataset):
    """
    降水数据集

    支持:
    - OdimH5格式 (DWD, KNMI)
    - 多帧输入输出
    - 数据增强
    """
    def __init__(
        self,
        file_paths,
        input_frames=12,       # 12帧 = 60分钟历史
        output_frames=24,      # 24帧 = 120分钟预报
        threshold=0.1,         # 最小降水阈值
        augment=True
    ):
        self.file_paths = file_paths
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.threshold = threshold
        self.augment = augment

        # 预加载所有数据
        self.data = self._load_data()

    def _load_data(self):
        """加载HDF5格式的雷达数据"""
        all_data = []

        for file_path in self.file_paths:
            try:
                # 使用pysteps加载OdimH5
                precip, _, _ = import_odim_h5(file_path, **kwargs)

                # 转换为mm/h
                precip, _ = conversion.to_rainrate(precip)

                # 过滤低值
                precip[precip < self.threshold] = 0.0

                all_data.append(precip)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # 合并所有数据
        full_sequence = np.concatenate(all_data, axis=0)

        return full_sequence

    def __len__(self):
        return len(self.data) - self.input_frames - self.output_frames

    def __getitem__(self, idx):
        # 提取序列
        sequence = self.data[idx:idx + self.input_frames + self.output_frames]

        # 归一化到 [0, 1]
        sequence = np.clip(sequence / 100.0, 0, 1)  # 假设最大100mm/h

        # 分离输入和输出
        x = sequence[:self.input_frames]
        y = sequence[self.input_frames:]

        # 数据增强
        if self.augment and np.random.rand() > 0.5:
            # 随机翻转
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)

            # 随机旋转
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k, axes=(1, 2))

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class DGMRDataModule:
    """
    DGMR数据模块
    """
    def __init__(
        self,
        train_files,
        val_files,
        batch_size=4,
        num_workers=4,
        input_frames=12,
        output_frames=24
    ):
        self.train_files = train_files
        self.val_files = val_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_frames = input_frames
        self.output_frames = output_frames

    def setup(self):
        self.train_dataset = PrecipitationDataset(
            self.train_files,
            input_frames=self.input_frames,
            output_frames=self.output_frames
        )

        self.val_dataset = PrecipitationDataset(
            self.val_files,
            input_frames=self.input_frames,
            output_frames=self.output_frames,
            augment=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
```

---

### 阶段4: 训练实施 (第5-7周)

#### 4.1 训练脚本

```python
# 文件: pysteps/dgmr/training/train_improved_dgmr.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class ImprovedDGMRModule(pl.LightningModule):
    """
    Improved DGMR Lightning模块
    """
    def __init__(
        self,
        input_frames=12,
        output_frames=24,
        hidden_dim=128,
        learning_rate=1e-4,
        lambda_reconstruction=1.0,
        lambda_gan=0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        # 模型
        self.generator = ImprovedDGMRGenerator(
            input_channels=input_frames,
            output_channels=output_frames,
            hidden_dim=hidden_dim
        )

        self.discriminator = DGMRDiscriminator()

        # 损失函数
        self.loss_fn = ImprovedDGMRLoss(
            lambda_reconstruction=lambda_reconstruction,
            lambda_gan=lambda_gan
        )

        # 评估指标
        self.train_metrics = {}
        self.val_metrics = {}

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # 生成
        fake = self.generator(x)

        # 判别
        real_logits = self.discriminator(y)
        fake_logits = self.discriminator(fake.detach())

        # 损失
        d_loss = self.loss_fn.discriminator_loss(real_logits, fake_logits)

        # 生成器损失
        fake_logits_for_g = self.discriminator(fake)
        g_loss, recon_loss, gan_loss = self.loss_fn.generator_loss(
            fake, y, fake_logits_for_g
        )

        # 记录
        self.log('train/d_loss', d_loss)
        self.log('train/g_loss', g_loss)
        self.log('train/recon_loss', recon_loss)

        return d_loss + g_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        fake = self.generator(x)

        # 计算CSI
        csi = self._compute_csi(fake, y, threshold=0.5)
        self.log('val/csi', csi)

        # 强降水CSI
        csi_high = self._compute_csi(fake, y, threshold=5.0)
        self.log('val/csi_high', csi_high)

        return csi

    def _compute_csi(self, pred, target, threshold=0.5):
        """计算CSI"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        hits = (pred_binary * target_binary).sum()
        misses = ((1 - pred_binary) * target_binary).sum()
        false_alarms = (pred_binary * (1 - target_binary)).sum()

        csi = hits / (hits + misses + false_alarms + 1e-8)
        return csi

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999)
        )

        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.5, 0.999)
        )

        return [opt_g, opt_d]


def main():
    # 配置
    config = {
        'input_frames': 12,
        'output_frames': 24,
        'hidden_dim': 128,
        'batch_size': 4,
        'max_epochs': 100,
        'learning_rate': 1e-4
    }

    # 数据模块
    datamodule = DGMRDataModule(
        train_files=train_files,
        val_files=val_files,
        batch_size=config['batch_size']
    )
    datamodule.setup()

    # 模型
    model = ImprovedDGMRModule(**config)

    # 回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val/csi_high',  # 监控强降水CSI
        mode='max',
        save_top_k=3,
        filename='improved-dgmr-{epoch:02d}-{val/csi_high:.4f}'
    )

    early_stop = EarlyStopping(
        monitor='val/csi_high',
        patience=10,
        mode='max'
    )

    # 训练器
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        logger=TensorBoardLogger('logs/improved_dgmr'),
        gradient_clip_val=1.0
    )

    # 训练
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
```

---

### 阶段5: 评估和部署 (第8周)

#### 5.1 评估脚本

```python
# 文件: benchmarks/scripts/evaluate_improved_dgmr.py

import torch
import numpy as np
from pysteps.dgmr.core.improved_generator import ImprovedDGMRGenerator
from pysteps.verification import sprocs

def evaluate_improved_dgmr(model_path, test_files):
    """
    评估Improved DGMR性能
    """
    # 加载模型
    model = ImprovedDGMRGenerator.load_from_checkpoint(model_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    results = {
        'csi': [],
        'csi_high': []  # 强降水CSI
    }

    with torch.no_grad():
        for file_path in test_files:
            # 加载数据
            x, y_true = load_test_data(file_path)

            # 预测
            x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
            y_pred = model(x_tensor).squeeze(0).cpu().numpy()

            # 计算CSI
            for threshold in [0.1, 0.5, 1.0, 2.0, 5.0]:
                csi = compute_csi(y_pred, y_true, threshold)
                if threshold >= 5.0:
                    results['csi_high'].append(csi)
                else:
                    results['csi'].append(csi)

    # 统计
    mean_csi = np.mean(results['csi'])
    mean_csi_high = np.mean(results['csi_high'])

    print(f"Average CSI: {mean_csi:.4f}")
    print(f"High-intensity CSI (>5mm/h): {mean_csi_high:.4f}")

    return results
```

---

## 📊 预期性能

| 指标 | 原版DGMR | Improved DGMR | 提升 |
|------|----------|---------------|------|
| **0-1h CSI** | 0.89 | 0.90 | +1% |
| **1-2h CSI** | 0.68 | 0.70 | +3% |
| **2-3h CSI** | 0.42 | 0.43 | +2% |
| **强降水 CSI** | 0.38 | **0.45** | **+18%** |
| **极端事件 CSI** | 0.25 | **0.35** | **+40%** |

---

## 💻 代码资源

### 主要实现仓库

1. **OpenClimateFix DGMR**
   - URL: [github.com/openclimatefix/skillful_nowcasting](https://github.com/openclimatefix/skillful_nowcasting)
   - 基础PyTorch Lightning实现

2. **强降水专用**
   - URL: [github.com/charlottecvn/precipitationnowcasting-generativemodels-highevents](https://github.com/charlottecvn/precipitationnowcasting-generativemodels-highevents)
   - 针对强降水优化

3. **其他实现**
   - [github.com/hyungting/DGMR-pytorch](https://github.com/hyungting/DGMR-pytorch)
   - [github.com/TQRTQ/DGMR](https://github.com/TQRTQ/DGMR)

---

## 🎯 下一步行动

### 第1周任务
1. ✅ 克隆相关代码仓库
2. ✅ 安装依赖和环境
3. ✅ 准备训练数据

### 第2-3周任务
1. ✅ 实现扩展平衡损失函数
2. ✅ 改进生成器和判别器
3. ✅ 测试损失函数

### 第4周任务
1. ✅ 准备数据加载器
2. ✅ 数据预处理
3. ✅ 数据增强

### 第5-7周任务
1. ✅ 训练Improved DGMR
2. ✅ 监控训练指标
3. ✅ 调整超参数

### 第8周任务
1. ✅ 全面评估性能
2. ✅ 与原版DGMR对比
3. ✅ 部署到生产环境

---

## 📚 参考文献

1. **Improved DGMR论文**
   - [Improving Precipitation Nowcasting for High-Intensity Events](https://journals.ametsoc.org/view/journals/aies/2/4/AIES-D-23-0017.1.xml)
   - AMS AI for Earth Systems, 2024

2. **原版DGMR**
   - [Skilful precipitation nowcasting using deep generative models of radar](https://www.nature.com/articles/s41586-021-03854-z)
   - Nature, 2021

3. **代码实现**
   - [OpenClimateFix/skillful_nowcasting](https://github.com/openclimatefix/skillful_nowcasting)
   - PyTorch Lightning实现

---

**文档版本**: 1.0
**日期**: 2026-03-17
**作者**: ocean2045 (346276171@qq.com)

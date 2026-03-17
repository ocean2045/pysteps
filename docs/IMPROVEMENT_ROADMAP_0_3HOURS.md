# PySteps 0-3小时短临预报改进方案

> **日期**: 2026-03-17
> **当前状态**: 0-30分钟优秀 (CSI > 0.45)
> **目标**: 0-3小时可用 (CSI > 0.3)
> **方法**: 系统性改进方案

---

## 📊 当前性能分析

### 性能现状

| 时效段 | DWD CSI | KNMI CSI | 状态 | 主要问题 |
|--------|---------|----------|------|----------|
| 0-15分钟 | 0.92 | 0.97 | ✅ 优秀 | 几乎完美 |
| 15-30分钟 | 0.82 | 0.88 | ✅ 良好 | 轻微衰减 |
| 30-60分钟 | 0.43 | 0.61 | ⚠️ 可用 | 快速衰减 |
| 60-120分钟 | 0.14 | 0.25 | ❌ 差 | 基本失效 |
| 120-180分钟 | 0.01 | 0.05 | ❌ 不可用 | 完全失效 |

### 衰减规律

**CSI 指数衰减模型**:
```
CSI(t) = CSI₀ × exp(-β × t)
```

| 数据集 | β (低阈值) | β (高阈值) | 衰减特征 |
|--------|-----------|-----------|----------|
| DWD | 0.010 | 0.040 | 高阈值衰减快4倍 |
| KNMI | 0.007 | 0.035 | 整体衰减较慢 |

**关键发现**:
- 30分钟是性能转折点
- 60分钟后基本不可用
- 高阈值衰减速度更快

---

## 🎯 改进目标

### 短期目标 (1-2个月)
- 30-60分钟: CSI > 0.4 (当前0.43/0.61)
- 60-120分钟: CSI > 0.25 (当前0.14/0.25)

### 中期目标 (3-6个月)
- 30-60分钟: CSI > 0.5
- 60-120分钟: CSI > 0.35
- 120-180分钟: CSI > 0.2

### 长期目标 (6-12个月)
- 0-60分钟: CSI > 0.6
- 60-120分钟: CSI > 0.4
- 120-180分钟: CSI > 0.25

---

## 🔧 改进策略

### 策略总览

```
┌─────────────────────────────────────────────────────┐
│          0-3小时预报改进策略                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. 算法层改进    ████████████░░░░░░░░░░░  50%    │
│     ├── 运动估计优化 (已有DARTS优化)               │
│     ├── 多尺度运动估计 (新)                        │
│     └── 自适应时间步长 (新)                        │
│                                                     │
│  2. 数据层改进    ████████████████████░░░  80%    │
│     ├── 增加输入帧数 (当前6→12)                    │
│     ├── 多源数据融合 (雷达+卫星+地面)             │
│     ├── 历史统计信息                               │
│     └── 地形影响建模                               │
│                                                     │
│  3. 模型层改进    ███████████████████████░  90%    │
│     ├── 混合预报模式 (STEPS + NWP)                │
│     ├── 机器学习后处理                             │
│     ├── 概率阈值方法                               │
│     └── 深度学习探索                               │
│                                                     │
│  4. 集成层改进    █████████████████████████ 100%  │
│     ├── 超集合预报 (多模型融合)                    │
│     ├── 时效分段策略                               │
│     ├── 动态权重调整                               │
│     └── 不确定性量化                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📋 详细改进方案

### 方案1: 算法层改进

#### 1.1 多尺度运动估计

**当前问题**:
- 单一空间分辨率
- 无法捕捉不同尺度的运动
- 长时间累积误差大

**改进方案**:
```python
class MultiScaleMotionEstimation:
    """多尺度运动估计"""

    def __init__(self, scales=[2, 4, 8, 16]):
        self.scales = scales

    def estimate(self, precip_series):
        # 在不同尺度上估计运动
        motions = []
        for scale in self.scales:
            # 下采样
            downsampled = self.downsample(precip_series, scale)
            # 运动估计
            motion = self.estimate_motion(downsampled)
            motions.append(motion)

        # 融合多尺度运动
        return self.merge_motions(motions)
```

**预期效果**: 30-60分钟 CSI +5-10%

#### 1.2 自适应时间步长

**当前问题**:
- 固定5分钟时间步
- 长时间预报步数多，累积误差
- 计算效率低

**改进方案**:
```python
class AdaptiveTimeStepping:
    """自适应时间步长"""

    def __init__(self):
        self.step_schedule = {
            (0, 15): 5,      # 0-15分钟: 5分钟步长
            (15, 30): 5,     # 15-30分钟: 5分钟步长
            (30, 60): 10,    # 30-60分钟: 10分钟步长
            (60, 120): 15,   # 60-120分钟: 15分钟步长
            (120, 180): 30,  # 120-180分钟: 30分钟步长
        }

    def get_steps(self, lead_time):
        """根据预报时效确定步长"""
        for (tmin, tmax), step in self.step_schedule.items():
            if tmin <= lead_time < tmax:
                return step
        return 30  # 默认30分钟
```

**预期效果**: 计算效率提升30%，累积误差减少20%

#### 1.3 时间相关模型改进

**当前问题**:
- AR(2)模型过于简单
- 无法捕捉非线性演化
- 长时间预报发散

**改进方案**:
```python
class ImprovedARModel:
    """改进的自回归模型"""

    def __init__(self, order=3):
        self.order = order

    def fit(self, time_series):
        # 使用更高阶AR模型
        self.ar_coefs = self.fit_ar_model(time_series, order=self.order)

        # 添加趋势项
        self.trend = self.estimate_trend(time_series)

        # 添加周期项（如果有）
        self.cycle = self.estimate_cycle(time_series)

    def predict(self, n_steps):
        # 结合AR、趋势、周期
        prediction = self.ar_predict(n_steps)
        prediction += self.trend
        prediction += self.cycle
        return prediction
```

**预期效果**: 60-120分钟 CSI +10-15%

---

### 方案2: 数据层改进

#### 2.1 增加输入帧数

**当前配置**: 6帧 (30分钟历史)

**改进配置**:

| 应用场景 | 输入帧数 | 历史长度 | 适用性 |
|---------|---------|----------|--------|
| 快速响应 | 4 | 20分钟 | 0-30分钟预报 |
| 标准预报 | 6 | 30分钟 | 0-60分钟预报 |
| **长期预报** | **12** | **60分钟** | **0-180分钟预报** |

**实施**:
```python
# 加载更多历史帧
precip_series, metadata = load_data(n_frames=12)  # 60分钟历史
```

**预期效果**: 60-120分钟 CSI +8-12%

#### 2.2 多源数据融合

**数据源**:

| 数据类型 | 来源 | 时间分辨率 | 空间分辨率 | 作用 |
|---------|------|-----------|-----------|------|
| **雷达** | DWD/KNMI | 5分钟 | 1km | 当前降水 |
| **卫星** | MSG/GOES | 15分钟 | 3km | 云系运动 |
| **地面站** | SYNOP | 60分钟 | 稀疏 | 降水验证 |
| **数值预报** | NWP | 60分钟 | 5km | 大尺度引导 |

**融合架构**:
```python
class MultiSourceDataFusion:
    """多源数据融合"""

    def __init__(self):
        self.radar_loader = RadarDataLoader()
        self.satellite_loader = SatelliteDataLoader()
        self.nwp_loader = NWPDataLoader()

    def load_and_fuse(self, current_time):
        # 加载多源数据
        radar = self.radar_loader.load(current_time)
        satellite = self.satellite_loader.load(current_time)
        nwp = self.nwp_loader.load(current_time)

        # 时间对齐
        aligned = self.temporal_align(radar, satellite, nwp)

        # 空间对齐
        fused = self.spatial_align(aligned)

        return fused

    def assimilate(self, observations, forecast):
        """数据同化"""
        # 卡尔曼滤波同化
        analysis = self.kalman_filter(observations, forecast)
        return analysis
```

**预期效果**: 60-180分钟 CSI +15-25%

#### 2.3 历史统计信息

**气候学数据库**:

| 信息类型 | 内容 | 用途 |
|---------|------|------|
| 降水频率 | 不同时间/地点的降水概率 | 先验信息 |
| 降水强度分布 | 不同强度降水的统计 | 偏差校正 |
| 运动统计 | 历史运动场统计 | 运动约束 |
| 衰减规律 | 降水系统生命周期 | 时效调整 |

**实施**:
```python
class ClimatologicalDatabase:
    """气候学数据库"""

    def __init__(self, historical_data):
        self.precip_freq = self.compute_frequency(historical_data)
        self.intensity_dist = self.compute_intensity_dist(historical_data)
        self.motion_stats = self.compute_motion_stats(historical_data)
        self.decay_patterns = self.compute_decay_patterns(historical_data)

    def get_prior(self, location, time, threshold):
        """获取先验概率"""
        return self.precip_freq[location, time, threshold]

    def bias_correct(self, forecast, location, time):
        """偏差校正"""
        clim_mean = self.intensity_dist[location, time]['mean']
        fcst_mean = np.mean(forecast)
        bias = clim_mean - fcst_mean
        return forecast + bias
```

**预期效果**: 120-180分钟 CSI +10-20%

#### 2.4 地形影响建模

**当前问题**:
- 平流模型不考虑地形
- 山区降水预报不准
- 地形强迫未被建模

**改进方案**:
```python
class OrographicEnhancement:
    """地形增强模型"""

    def __init__(self, dem_file):
        self.dem = self.load_dem(dem_file)
        self.slope = self.compute_slope(self.dem)
        self.aspect = self.compute_aspect(self.dem)

    def compute_enhancement(self, wind_direction, wind_speed):
        """计算地形增强"""
        # 风向与坡向的关系
        angle_diff = wind_direction - self.aspect
        upslope_component = np.cos(angle_diff)

        # 地形增强因子
        enhancement = wind_speed * upslope_component * self.slope
        return enhancement

    def adjust_forecast(self, forecast, wind_field):
        """调整预报"""
        enhancement = self.compute_enhancement(wind_field)
        adjusted = forecast * (1 + 0.3 * enhancement)  # 30%地形增强
        return adjusted
```

**预期效果**: 山区 CSI +10-15%

---

### 方案3: 模型层改进

#### 3.1 混合预报模式

**原理**: 短期用雷达，长期用NWP，中间过渡

**架构**:
```python
class HybridNowcastModel:
    """混合预报模型"""

    def __init__(self):
        self.radar_model = STEPS()  # 雷达外推
        self.nwp_model = NWPModel()  # 数值预报
        self.transition_time = 60  # 60分钟过渡

    def predict(self, precip_series, lead_times):
        predictions = []

        for lead_time in lead_times:
            if lead_time <= 30:
                # 0-30分钟: 纯雷达
                weight_radar = 1.0
                weight_nwp = 0.0
            elif lead_time <= 60:
                # 30-60分钟: 过渡
                progress = (lead_time - 30) / 30
                weight_radar = 1.0 - progress
                weight_nwp = progress
            else:
                # >60分钟: 纯NWP
                weight_radar = 0.0
                weight_nwp = 1.0

            # 各模型预报
            fcst_radar = self.radar_model.predict(precip_series, lead_time)
            fcst_nwp = self.nwp_model.predict(precip_series, lead_time)

            # 加权融合
            fcst_hybrid = (weight_radar * fcst_radar +
                          weight_nwp * fcst_nwp)
            predictions.append(fcst_hybrid)

        return predictions
```

**预期效果**: 60-180分钟 CSI +20-30%

#### 3.2 机器学习后处理

**方法**: 使用ML模型校正原始预报

```python
class MLPostProcessor:
    """机器学习后处理"""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def train(self, forecasts, observations):
        """训练校正模型"""
        # 特征工程
        features = self.extract_features(forecasts)
        targets = observations

        # 训练
        self.model.fit(features, targets)

    def extract_features(self, forecast):
        """特征提取"""
        features = {
            'mean': np.mean(forecast),
            'std': np.std(forecast),
            'max': np.max(forecast),
            'gradient': np.gradient(forecast),
            'spatial_variance': self.spatial_var(forecast),
            # ... 更多特征
        }
        return features

    def correct(self, forecast):
        """校正预报"""
        features = self.extract_features(forecast)
        correction = self.model.predict(features)
        return forecast + correction
```

**预期效果**: 整体 CSI +5-10%

#### 3.3 概率阈值方法

**当前问题**: 固定阈值不适应所有情况

**改进方案**:
```python
class ProbabilisticThreshold:
    """概率阈值方法"""

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def get_probability_field(self, threshold):
        """获取 exceedance probability"""
        # 计算超过阈值的概率
        prob = np.mean(self.ensemble > threshold, axis=0)
        return prob

    def optimize_threshold(self, target_pod, target_far):
        """优化阈值以达到目标 POD/FAR"""
        # 在 ROC 曲线上找最优工作点
        best_threshold = self.optimize_roc(target_pod, target_far)
        return best_threshold

    def get_confidence_interval(self, alpha=0.9):
        """获取置信区间"""
        lower = np.percentile(self.ensemble, (1-alpha)/2*100, axis=0)
        upper = np.percentile(self.ensemble, (1+alpha)/2*100, axis=0)
        return lower, upper
```

**预期效果**: 用户可自定义风险等级

#### 3.4 深度学习探索

**模型选择**: ConvLSTM, U-Net, MetNet

```python
class DeepLearningNowcast:
    """深度学习短临预报"""

    def __init__(self):
        # 使用预训练的ConvLSTM模型
        self.model = ConvLSTM(
            input_shape=(12, 765, 700),  # 12帧输入
            hidden_dims=[64, 32, 16],
            kernel_size=(3, 3),
        )

    def train(self, training_data):
        """训练模型"""
        # 输入: 历史雷达图像序列
        # 输出: 未来雷达图像序列
        self.model.fit(training_data)

    def predict(self, input_sequence, n_steps):
        """预报"""
        return self.model.predict(input_sequence, n_steps)
```

**预期效果**: 0-60分钟 CSI +10-15%，60-180分钟需要大量训练数据

---

### 方案4: 集成层改进

#### 4.1 超集合预报

**概念**: 多模型、多参数、多初始条件的超级集合

```python
class SuperEnsembleNowcast:
    """超集合预报"""

    def __init__(self):
        # 多个预报模型
        self.models = {
            'steps_advection': STEPS(method='advection'),
            'steps_spectral': STEPS(method='spectral'),
            'linda': LINDA(),
            'rainymotion': RainyMotion(),
            'deep_learning': DeepLearningNowcast(),
        }

        # 多参数配置
        self.param_configs = [
            {'n_members': 20, 'ar_order': 2},
            {'n_members': 40, 'ar_order': 3},
            {'n_members': 50, 'ar_order': 2},
        ]

    def generate_forecasts(self, precip_series):
        """生成所有预报"""
        all_forecasts = []

        for model_name, model in self.models.items():
            for params in self.param_configs:
                forecast = model.predict(precip_series, **params)
                all_forecasts.append(forecast)

        return all_forecasts

    def combine_forecasts(self, forecasts, weights=None):
        """融合预报"""
        if weights is None:
            # 动态权重（基于近期性能）
            weights = self.compute_dynamic_weights(forecasts)

        # 加权平均
        combined = np.average(forecasts, weights=weights, axis=0)
        return combined

    def compute_dynamic_weights(self, forecasts):
        """动态权重计算"""
        # 基于最近预报性能计算权重
        weights = []
        for forecast in forecasts:
            recent_skill = self.evaluate_recent_skill(forecast)
            weights.append(recent_skill)

        weights = np.array(weights) / np.sum(weights)
        return weights
```

**预期效果**: 60-180分钟 CSI +15-25%

#### 4.2 时效分段策略

**原理**: 不同时段使用不同策略

```python
class TemporalSegmentationStrategy:
    """时效分段策略"""

    def __init__(self):
        self.strategies = {
            'very_short': (0, 15),     # 极短期: 0-15分钟
            'short': (15, 30),          # 短期: 15-30分钟
            'medium': (30, 60),         # 中期: 30-60分钟
            'long': (60, 120),          # 长期: 60-120分钟
            'very_long': (120, 180),    # 极长期: 120-180分钟
        }

        self.models = {
            'very_short': 'pure_radar',
            'short': 'radar_with_ml_correction',
            'medium': 'hybrid_radar_nwp',
            'long': 'nwp_with_radar_initialization',
            'very_long': 'pure_nwp_with_climatology',
        }

    def predict(self, precip_series, lead_time):
        """根据时效选择策略"""
        # 确定时效段
        segment = self.get_segment(lead_time)

        # 选择对应模型
        model_type = self.models[segment]
        model = self.get_model(model_type)

        # 预报
        forecast = model.predict(precip_series, lead_time)
        return forecast
```

**预期效果**: 各时效段最优策略，整体 CSI +10-20%

#### 4.3 动态权重调整

**方法**: 基于实时性能调整模型权重

```python
class DynamicWeightAdjustment:
    """动态权重调整"""

    def __init__(self, window_size=6):
        self.window_size = window_size
        self.performance_history = {}

    def update_weights(self, model_names, recent_observations):
        """更新权重"""
        # 计算各模型最近性能
        for model_name in model_names:
            recent_forecast = self.get_recent_forecast(model_name)
            skill = self.compute_skill(recent_forecast, recent_observations)

            # 更新历史
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []

            self.performance_history[model_name].append(skill)

            # 保持窗口大小
            if len(self.performance_history[model_name]) > self.window_size:
                self.performance_history[model_name].pop(0)

        # 计算权重
        weights = {}
        for model_name, skills in self.performance_history.items():
            # 使用指数加权平均
            weights[model_name] = np.exp(np.mean(skills))

        # 归一化
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return weights
```

**预期效果**: 自适应优化，CSI +5-10%

#### 4.4 不确定性量化

**目标**: 提供可信的预报不确定性

```python
class UncertaintyQuantification:
    """不确定性量化"""

    def __init__(self):
        self.ensemble_spread = None
        self.spatial_correlation = None
        self.temporal_correlation = None

    def compute_uncertainty(self, ensemble_forecast):
        """计算预报不确定性"""
        # 集合离散度
        self.ensemble_spread = np.std(ensemble_forecast, axis=0)

        # 空间相关性
        self.spatial_correlation = self.compute_spatial_correlation(ensemble_forecast)

        # 时间相关性
        self.temporal_correlation = self.compute_temporal_correlation(ensemble_forecast)

        uncertainty = {
            'spread': self.ensemble_spread,
            'spatial_corr': self.spatial_correlation,
            'temporal_corr': self.temporal_correlation,
        }

        return uncertainty

    def generate_prediction_intervals(self, forecast, uncertainty, alpha=0.9):
        """生成预报区间"""
        spread = uncertainty['spread']

        # 假设正态分布
        z_score = 1.645  # 90%置信区间
        lower = forecast - z_score * spread
        upper = forecast + z_score * spread

        return lower, upper
```

**预期效果**: 用户可量化预报风险

---

## 🗺️ 实施路线图

### 第一阶段 (1-2个月): 快速改进

**目标**: 60-120分钟 CSI > 0.25

| 任务 | 优先级 | 难度 | 预期效果 | 工作量 |
|------|--------|------|----------|--------|
| 增加输入帧数 (6→12) | 🔴 高 | 低 | +8-12% | 2天 |
| 自适应时间步长 | 🟡 中 | 低 | +5% | 3天 |
| 时效分段策略 | 🔴 高 | 中 | +10% | 1周 |
| 混合预报 (雷达+NWP) | 🔴 高 | 中 | +20% | 2周 |
| **小计** | - | - | **+30-40%** | **4周** |

### 第二阶段 (3-4个月): 中期改进

**目标**: 60-120分钟 CSI > 0.35, 120-180分钟 CSI > 0.2

| 任务 | 优先级 | 难度 | 预期效果 | 工作量 |
|------|--------|------|----------|--------|
| 多尺度运动估计 | 🟡 中 | 高 | +5-10% | 3周 |
| 机器学习后处理 | 🟡 中 | 中 | +5-10% | 2周 |
| 历史统计信息 | 🟢 低 | 中 | +10-20% | 3周 |
| 地形影响建模 | 🟢 低 | 中 | +10-15% | 2周 |
| 超集合预报 | 🟡 中 | 高 | +15-25% | 4周 |
| **小计** | - | - | **+35-60%** | **14周** |

### 第三阶段 (5-6个月): 深度改进

**目标**: 60-120分钟 CSI > 0.4, 120-180分钟 CSI > 0.25

| 任务 | 优先级 | 难度 | 预期效果 | 工作量 |
|------|--------|------|----------|--------|
| 多源数据融合 | 🟡 中 | 高 | +15-25% | 4周 |
| 深度学习模型 | 🟢 低 | 很高 | +10-20% | 6周 |
| 动态权重调整 | 🟢 低 | 中 | +5-10% | 2周 |
| 不确定性量化 | 🟢 低 | 中 | 用户价值 | 2周 |
| 概率阈值方法 | 🟢 低 | 中 | 用户价值 | 2周 |
| **小计** | - | - | **+25-45%** | **16周** |

---

## 📊 预期改进效果

### 性能提升预测

| 时效段 | 当前CSI | 第一阶段 | 第二阶段 | 第三阶段 | 总提升 |
|--------|---------|----------|----------|----------|--------|
| 30-60分钟 | 0.52 | **0.65** | **0.70** | **0.75** | **+44%** |
| 60-120分钟 | 0.20 | **0.28** | **0.32** | **0.38** | **+90%** |
| 120-180分钟 | 0.03 | **0.08** | **0.15** | **0.22** | **+633%** |

### 可信度提升

| 时效段 | 当前状态 | 第一阶段 | 第二阶段 | 第三阶段 |
|--------|----------|----------|----------|----------|
| 0-30分钟 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 30-60分钟 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 60-120分钟 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 120-180分钟 | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 💡 推荐实施策略

### 优先级排序

**立即实施** (高优先级，低难度):
1. 增加输入帧数: 6 → 12 帧
2. 自适应时间步长
3. 时效分段策略
4. 混合预报模式

**近期实施** (中优先级，中难度):
1. 机器学习后处理
2. 历史统计信息
3. 地形影响建模
4. 动态权重调整

**远期考虑** (低优先级，高难度):
1. 多尺度运动估计
2. 多源数据融合
3. 深度学习模型
4. 超集合预报

### 投入产出比

| 改进方案 | 投入 | 产出 | ROI |
|---------|------|------|-----|
| 增加输入帧 | 2天 | +8-12% | ⭐⭐⭐⭐⭐ |
| 时效分段 | 1周 | +10% | ⭐⭐⭐⭐ |
| 混合预报 | 2周 | +20% | ⭐⭐⭐⭐ |
| ML后处理 | 2周 | +5-10% | ⭐⭐⭐ |
| 多源融合 | 4周 | +15-25% | ⭐⭐⭐ |
| 深度学习 | 6周 | +10-20% | ⭐⭐ |
| 超集合 | 4周 | +15-25% | ⭐⭐⭐ |

---

## 📝 总结

### 核心策略

1. **数据为王**: 更多历史帧、多源数据融合
2. **混合模式**: 短期雷达，长期NWP，中间过渡
3. **分段优化**: 不同时段使用不同策略
4. **机器学习**: 利用ML校正和优化
5. **不确定性**: 提供可信度评估

### 预期目标

**短期 (2个月)**: 60-120分钟达到可用水平 (CSI > 0.25)
**中期 (4个月)**: 60-120分钟达到良好水平 (CSI > 0.35)
**长期 (6个月)**: 120-180分钟达到可用水平 (CSI > 0.22)

### 关键成功因素

- ✅ 系统性方法，而非单一改进
- ✅ 渐进式实施，逐步验证
- ✅ 基于数据驱动，而非理论推导
- ✅ 保持向后兼容
- ✅ 持续评估和调优

---

**文档版本**: 1.0
**日期**: 2026-03-17
**作者**: ocean2045 (346276171@qq.com)

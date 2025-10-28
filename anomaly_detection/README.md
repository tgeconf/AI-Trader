# 异常检测模块

## 概述

异常检测模块是AI-Trader系统的风险监控核心，提供全面的异常检测能力，包括市场异常检测、行为异常监控和系统健康检查。该模块采用多层次检测架构，确保交易系统的稳定性和安全性。

## 模块结构

```
anomaly_detection/
├── statistical_anomaly.py    # 统计异常检测器
├── behavioral_anomaly.py     # 行为异常检测器
├── system_health.py          # 系统健康监控器
├── __init__.py               # 模块初始化
└── README.md                 # 本文件
```

## 核心组件

### 1. 统计异常检测器 (statistical_anomaly.py)

#### 功能描述
基于统计方法的异常检测，识别市场数据中的异常模式，包括价格异常、波动率异常和相关性异常。

#### 异常类型定义

```python
class AnomalyType(Enum):
    PRICE_SPIKE = "price_spike"        # 价格异常波动
    VOLUME_SURGE = "volume_surge"      # 成交量异常
    VOLATILITY_SHIFT = "volatility_shift"  # 波动率异常
    CORRELATION_BREAK = "correlation_break"  # 相关性异常
    LIQUIDITY_DROP = "liquidity_drop"  # 流动性异常
```

#### 检测方法

##### 价格异常检测 (detect_price_anomaly)
- **输入**：价格数据、成交量数据
- **方法**：
  - **Z-score方法**：基于标准差检测异常
  - **IQR方法**：基于四分位距检测异常
  - **孤立森林**：无监督异常检测
  - **局部异常因子**：基于密度的异常检测
- **输出**：异常类型、置信度、特征分析

##### 波动率异常检测 (detect_volatility_anomaly)
- **输入**：收益率序列
- **方法**：滚动波动率分析
- **阈值**：波动率比率 > 2.0 或 < 0.5
- **输出**：波动率异常状态和变化程度

##### 相关性异常检测 (detect_correlation_anomaly)
- **输入**：多资产收益率数据
- **方法**：相关性矩阵分析
- **输出**：相关性异常状态和变化模式

#### 技术实现

##### 模型训练
```python
detector = StatisticalAnomalyDetector(contamination=0.05)
detector.fit(training_data)
```

##### 多方法融合
- **投票机制**：至少两种方法检测到异常才确认为异常
- **置信度计算**：基于异常检测方法的一致性
- **特征分析**：详细的异常特征描述

#### 配置参数
- `contamination`：异常值污染比例（默认0.1）
- 训练数据要求：最少10个观测值

### 2. 行为异常检测器 (behavioral_anomaly.py)

#### 功能描述
监控交易行为和用户行为模式，检测异常交易活动和潜在风险行为。

#### 异常类型定义

```python
class BehavioralAnomalyType(Enum):
    TRADING_FREQUENCY_ANOMALY = "trading_frequency_anomaly"  # 交易频率异常
    ORDER_SIZE_ANOMALY = "order_size_anomaly"  # 订单规模异常
    TRADING_PATTERN_CHANGE = "trading_pattern_change"  # 交易模式变化
    RISK_APPETITE_CHANGE = "risk_appetite_change"  # 风险偏好变化
    TIME_OF_DAY_ANOMALY = "time_of_day_anomaly"  # 交易时间异常
    ASSET_CONCENTRATION = "asset_concentration"  # 资产集中度异常
```

#### 检测功能

##### 交易行为分析 (analyze_trading_behavior)
- **输入**：交易数据、用户ID
- **检测项目**：
  - 交易频率异常
  - 订单规模异常
  - 交易模式变化
  - 风险偏好变化
  - 交易时间异常
  - 资产集中度
- **输出**：行为分析结果和风险评分

##### 用户行为分析 (analyze_user_behavior)
- **输入**：用户行为数据
- **检测项目**：
  - 登录频率异常
  - 操作模式异常
  - 会话时长异常
- **输出**：用户行为风险评估

#### 用户画像系统

##### 交易行为画像 (TradingBehaviorProfile)
```python
@dataclass
class TradingBehaviorProfile:
    avg_trades_per_day: float      # 日均交易次数
    avg_order_size: float          # 平均订单规模
    preferred_trading_hours: List[int]  # 偏好交易时间
    risk_tolerance: float          # 风险容忍度
    diversification_score: float   # 分散化评分
    trading_consistency: float     # 交易一致性
    last_updated: datetime         # 最后更新时间
```

##### 画像更新机制
- **数据要求**：最少10笔交易建立画像
- **动态更新**：每次交易后更新画像
- **特征计算**：基于历史交易数据

#### 检测算法

##### 交易频率异常检测
- **方法**：当前交易频率与基准频率比较
- **阈值**：频率比率 > 3.0倍
- **置信度**：基于历史数据的稳定性

##### 订单规模异常检测
- **方法**：当前订单规模与基准规模比较
- **阈值**：规模比率 > 2.5倍
- **风险评分**：基于规模异常程度

##### 交易模式变化检测
- **方法**：交易时间模式匹配度分析
- **阈值**：时间匹配度 < 0.7
- **特征**：偏好交易时间分析

#### 风险评估

##### 整体风险评估 (assess_behavioral_risk)
- **输入**：行为异常列表
- **方法**：加权平均风险评分
- **风险等级**：
  - 高风险：平均评分 > 0.8 或高风险异常 ≥ 3个
  - 中等风险：平均评分 > 0.5 或高风险异常 ≥ 1个
  - 低风险：其他情况

##### 建议生成 (generate_behavioral_recommendations)
- **基于异常类型**：针对性的风险控制建议
- **风险等级**：不同风险等级的建议策略
- **预防措施**：异常行为的预防建议

### 3. 系统健康监控器 (system_health.py)

#### 功能描述
全面监控系统资源状态、服务健康和网络连接，确保交易系统的稳定运行。

#### 健康状态定义

```python
class HealthStatus(Enum):
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"           # 良好
    FAIR = "fair"           # 一般
    POOR = "poor"           # 较差
    CRITICAL = "critical"   # 危急
```

#### 监控维度

##### 系统资源监控
- **CPU使用率**：实时CPU使用情况
- **内存使用**：内存占用和可用性
- **磁盘空间**：磁盘使用率和剩余空间
- **系统负载**：1分钟、5分钟、15分钟负载

##### 进程状态监控 (check_process_status)
- **进程统计**：总进程数、运行进程、僵尸进程
- **关键进程**：Python、Redis、PostgreSQL等
- **进程警报**：进程数量超过阈值

##### 服务健康检查 (check_service_status)
- **监控服务**：Redis、PostgreSQL、Nginx、Python
- **服务状态**：运行状态、活跃状态
- **整体状态**：基于所有服务的综合状态

##### 网络连接检查 (check_network_connectivity)
- **连接测试**：Google DNS、Cloudflare DNS、本地服务
- **延迟测量**：网络延迟和响应时间
- **健康评分**：基于连接成功率的评分

#### 警报系统

##### 警报级别
```python
class AlertLevel(Enum):
    INFO = "info"        # 信息
    WARNING = "warning"  # 警告
    ERROR = "error"      # 错误
    CRITICAL = "critical" # 危急
```

##### 警报生成
- **基于阈值**：资源使用超过预设阈值
- **基于状态**：服务异常或网络连接失败
- **基于趋势**：资源使用趋势恶化

#### 健康评估

##### 整体健康评估 (assess_overall_health)
- **输入**：CPU、内存、磁盘使用率
- **方法**：多指标加权评分
- **输出**：健康状态、评分、警报、建议

##### 健康评分计算
- **指标权重**：
  - CPU使用率：30%
  - 内存使用率：25%
  - 磁盘使用率：25%
  - 系统负载：15%
  - 其他因素：5%
- **评分范围**：0.0 - 1.0

##### 趋势分析
- **短期趋势**：最近3个周期的变化
- **长期趋势**：历史数据对比
- **趋势分类**：改善、稳定、恶化

#### 监控配置

##### 默认阈值
```python
alert_thresholds = {
    "cpu_usage": 80.0,
    "memory_usage": 85.0,
    "disk_usage": 90.0,
    "network_latency": 100.0,  # ms
    "process_count": 500,
    "load_average_1min": 4.0,
    "load_average_5min": 3.0,
    "load_average_15min": 2.0,
}
```

##### 监控服务
```python
monitored_services = [
    "redis",
    "postgresql", 
    "nginx",
    "python",
]
```

## 集成使用

### 模块初始化
```python
from anomaly_detection import (
    StatisticalAnomalyDetector,
    BehavioralAnomalyDetector,
    SystemHealthMonitor
)

# 统计异常检测器
stat_detector = StatisticalAnomalyDetector(contamination=0.05)

# 行为异常检测器
behavior_detector = BehavioralAnomalyDetector(
    baseline_period_days=30,
    anomaly_threshold=0.8
)

# 系统健康监控器
health_monitor = SystemHealthMonitor(
    check_interval=60,
    alert_thresholds=custom_thresholds
)
```

### 检测流程

#### 市场异常检测
```python
# 训练模型
stat_detector.fit(training_data)

# 检测价格异常
price_anomaly = stat_detector.detect_price_anomaly(
    price_data, 
    volume_data
)

# 检测波动率异常
vol_anomaly = stat_detector.detect_volatility_anomaly(
    returns_data,
    window=20
)
```

#### 行为异常检测
```python
# 添加交易记录
behavior_detector.add_trading_record(
    user_id="user_001",
    symbol="AAPL",
    order_type="buy",
    quantity=100,
    price=150.25,
    timestamp=datetime.now()
)

# 分析交易行为
trading_analysis = behavior_detector.analyze_trading_behavior(
    trading_data,
    user_id="user_001"
)

# 检测行为异常
behavioral_anomalies = behavior_detector.detect_behavioral_anomalies(
    trading_analysis,
    user_analysis
)
```

#### 系统健康监控
```python
# 开始监控
health_monitor.start_monitoring()

# 获取健康报告
health_report = health_monitor.assess_overall_health()

# 获取详细报告
detailed_report = health_monitor.get_detailed_health_report()

# 停止监控
health_monitor.stop_monitoring()
```

## 配置优化

### 性能调优
- **检测频率**：根据业务需求调整检测间隔
- **数据窗口**：优化历史数据窗口大小
- **阈值设置**：基于实际环境调整警报阈值

### 内存管理
- **历史数据**：限制存储的历史数据量
- **缓存策略**：优化特征计算缓存
- **资源释放**：及时释放不再使用的资源

### 扩展性
- **插件架构**：支持新的异常检测方法
- **自定义阈值**：支持动态阈值配置
- **多数据源**：支持多种数据源接入

## 最佳实践

### 部署建议
- **独立部署**：异常检测模块可独立部署
- **资源隔离**：确保足够的计算资源
- **网络配置**：优化网络连接配置

### 监控策略
- **多层次监控**：结合系统、应用、业务层监控
- **实时警报**：关键异常实时通知
- **历史分析**：定期分析异常模式

### 维护指南
- **定期更新**：更新检测模型和阈值
- **性能监控**：监控检测性能指标
- **日志分析**：分析异常检测日志

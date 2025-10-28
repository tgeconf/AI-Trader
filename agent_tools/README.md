# Agent Tools模块

## 概述

Agent Tools模块是AI-Trader系统的工具层，提供了一系列专业级的金融分析工具，通过MCP（Model Context Protocol）协议为AI Agent提供强大的分析能力。该模块采用微服务架构，支持高并发、高可用的工具调用。

## 模块结构

```
agent_tools/
├── start_mcp_services.py          # MCP服务启动管理器
├── tool_anomaly_detection.py      # 异常检测工具
├── tool_get_price_local.py        # 本地价格获取工具
├── tool_jina_search.py            # 网络搜索工具
├── tool_market_regime.py          # 市场状态识别工具
├── tool_math.py                   # 数学计算工具
├── tool_multi_timeframe.py        # 多时间框架分析工具
├── tool_performance_attribution.py # 绩效归因工具
├── tool_portfolio_optimization.py # 投资组合优化工具
├── tool_trade.py                  # 交易执行工具
└── README.md                      # 本文件
```

## 工具详解

### 1. MCP服务管理器 (start_mcp_services.py)

#### 功能描述
统一管理所有MCP工具的启动、监控和停止，提供完整的服务生命周期管理。

#### 核心特性
- **服务管理**：启动、停止、重启所有MCP服务
- **健康检查**：实时监控服务状态和端口可用性
- **日志管理**：集中式日志记录和错误追踪
- **故障恢复**：自动重启失败的服务

#### 服务配置
```python
service_configs = {
    'math': {
        'script': 'tool_math.py',
        'name': 'Math',
        'port': 8000
    },
    'search': {
        'script': 'tool_jina_search.py', 
        'name': 'Search',
        'port': 8001
    },
    'trade': {
        'script': 'tool_trade.py',
        'name': 'TradeTools', 
        'port': 8002
    },
    'price': {
        'script': 'tool_get_price_local.py',
        'name': 'LocalPrices',
        'port': 8003
    }
}
```

### 2. 异常检测工具 (tool_anomaly_detection.py)

#### 功能描述
提供市场异常检测、系统健康监控和行为异常分析功能。

#### 核心工具

##### detect_market_anomalies
- **功能**：检测市场异常行为
- **输入**：股票代码、价格数据、成交量数据
- **输出**：异常检测结果和风险评估
- **检测方法**：价格异常、波动率异常、相关性异常

##### monitor_system_health
- **功能**：监控系统健康状态
- **输出**：CPU、内存、磁盘、网络状态报告
- **警报机制**：基于阈值的自动警报

##### detect_behavioral_anomalies
- **功能**：检测交易行为异常
- **输入**：交易数据和用户行为数据
- **输出**：行为异常分析和风险评分

##### generate_risk_report
- **功能**：生成综合风险报告
- **输入**：市场异常、系统健康、行为异常数据
- **输出**：综合风险评估和建议

#### 技术实现
- **统计方法**：Z-score、IQR、孤立森林
- **机器学习**：HMM模型、聚类分析
- **实时监控**：流式数据处理

### 3. 本地价格获取工具 (tool_get_price_local.py)

#### 功能描述
从本地数据文件获取股票OHLCV（开盘价、最高价、最低价、收盘价、成交量）数据。

#### 核心工具

##### get_price_local
- **功能**：获取指定股票和日期的价格数据
- **输入**：股票代码、日期
- **输出**：OHLCV数据和元信息
- **数据源**：本地JSONL格式数据文件

#### 数据格式
```json
{
    "symbol": "AAPL",
    "date": "2025-10-15",
    "ohlcv": {
        "open": 150.25,
        "high": 152.80,
        "low": 149.50,
        "close": 151.75,
        "volume": 12500000
    }
}
```

### 4. 网络搜索工具 (tool_jina_search.py)

#### 功能描述
使用Jina AI API进行网络搜索和内容抓取，为AI Agent提供实时市场信息。

#### 核心工具

##### get_information
- **功能**：搜索和获取网络信息
- **输入**：搜索查询词
- **输出**：结构化搜索结果
- **特性**：智能日期过滤、内容提取

#### 技术特性
- **智能搜索**：基于语义的搜索优化
- **内容提取**：自动提取网页主要内容
- **日期过滤**：过滤未来日期的信息
- **格式标准化**：统一日期和时间格式

### 5. 市场状态识别工具 (tool_market_regime.py)

#### 功能描述
识别和分析市场状态（牛市、熊市、震荡市等），为交易决策提供依据。

#### 核心工具

##### detect_market_regime
- **功能**：检测当前市场状态
- **输入**：股票代码、价格数据
- **输出**：市场状态分类和概率
- **方法**：HMM、波动率分类、趋势分析

##### predict_regime_transition
- **功能**：预测市场状态转换
- **输入**：当前状态、历史收益率
- **输出**：状态转换概率

##### get_regime_based_trading_recommendation
- **功能**：基于市场状态的交易建议
- **输入**：当前状态、风险容忍度
- **输出**：交易策略建议

#### 分析方法
- **HMM模型**：隐马尔可夫模型状态识别
- **波动率分类**：基于波动率的状态分类
- **趋势分析**：线性回归和动量指标

### 6. 数学计算工具 (tool_math.py)

#### 功能描述
提供基础的数学计算功能，支持AI Agent的数值计算需求。

#### 核心工具

##### add
- **功能**：加法运算
- **输入**：两个数值
- **输出**：求和结果

##### multiply
- **功能**：乘法运算
- **输入**：两个数值
- **输出**：乘积结果

### 7. 多时间框架分析工具 (tool_multi_timeframe.py)

#### 功能描述
整合多个时间框架的分析信号，解决时间框架冲突，生成综合交易决策。

#### 核心工具

##### generate_multi_timeframe_signals
- **功能**：生成多时间框架信号
- **输入**：股票代码、价格数据、时间框架列表
- **输出**：整合信号和置信度

##### analyze_timeframe_consistency
- **功能**：分析时间框架一致性
- **输入**：各时间框架信号数据
- **输出**：一致性分析和冲突解决

##### optimize_timeframe_weights
- **功能**：优化时间框架权重
- **输入**：历史表现数据
- **输出**：优化后的权重配置

#### 时间框架支持
- **长期**：1天、1周
- **中期**：4小时、1小时
- **短期**：15分钟、5分钟

### 8. 绩效归因工具 (tool_performance_attribution.py)

#### 功能描述
分析投资组合绩效来源，识别超额收益的驱动因素。

#### 核心工具

##### analyze_brinson_attribution
- **功能**：Brinson绩效归因分析
- **输入**：组合权重、基准权重、收益率
- **输出**：配置效应、选择效应、交互效应

##### analyze_multi_factor_attribution
- **功能**：多因子绩效归因
- **输入**：组合收益率、因子收益率、因子暴露
- **输出**：因子贡献和Alpha分析

##### analyze_strategy_effectiveness
- **功能**：策略有效性分析
- **输入**：策略收益率、基准收益率
- **输出**：风险调整后收益指标

#### 分析方法
- **Brinson模型**：经典的绩效归因框架
- **Fama-French三因子**：市场、规模、价值因子
- **Carhart四因子**：增加动量因子

### 9. 投资组合优化工具 (tool_portfolio_optimization.py)

#### 功能描述
提供多种投资组合优化方法，实现风险调整后的收益最大化。

#### 核心工具

##### optimize_portfolio_mean_variance
- **功能**：均值方差优化
- **输入**：历史收益率数据
- **输出**：最优权重和风险收益指标
- **目标**：最大化夏普比率、最小化方差等

##### optimize_portfolio_risk_parity
- **功能**：风险平价优化
- **输入**：历史收益率数据
- **输出**：风险均衡权重
- **方法**：等风险贡献、逆波动率

##### black_litterman_optimization
- **功能**：Black-Litterman优化
- **输入**：市场均衡权重、主观观点
- **输出**：后验最优权重

##### calculate_efficient_frontier
- **功能**：计算有效前沿
- **输入**：历史收益率数据
- **输出**：有效前沿点集

#### 优化方法
- **经典优化**：Markowitz均值方差模型
- **风险平价**：等风险贡献模型
- **贝叶斯优化**：Black-Litterman模型

### 10. 交易执行工具 (tool_trade.py)

#### 功能描述
执行股票买卖操作，管理交易成本和头寸更新。

#### 核心工具

##### buy
- **功能**：买入股票
- **输入**：股票代码、买入数量
- **输出**：新头寸和交易成本
- **验证**：资金充足性检查

##### sell
- **功能**：卖出股票
- **输入**：股票代码、卖出数量
- **输出**：新头寸和交易成本
- **验证**：持仓充足性检查

#### 交易成本计算
- **佣金**：按比例计算的交易佣金
- **滑点**：基于成交量的滑点模型
- **市场冲击**：大额交易的市场冲击成本

## 技术架构

### MCP协议
所有工具都基于FastMCP框架实现，提供标准化的工具接口：

```python
from fastmcp import FastMCP
mcp = FastMCP("ToolName")

@mcp.tool()
def tool_function(param: type) -> Dict[str, Any]:
    # 工具实现
    return {"result": value}
```

### 服务部署
- **独立进程**：每个工具运行在独立进程中
- **HTTP接口**：通过HTTP协议提供服务
- **端口配置**：可配置的服务端口
- **健康检查**：定期服务状态检查

### 错误处理
- **异常捕获**：全面的异常处理机制
- **错误返回**：标准化的错误信息格式
- **重试机制**：网络错误的自动重试

## 配置管理

### 环境变量
```bash
MATH_HTTP_PORT=8000
SEARCH_HTTP_PORT=8001
TRADE_HTTP_PORT=8002
GETPRICE_HTTP_PORT=8003
ANOMALY_DETECTION_HTTP_PORT=8008
MARKET_REGIME_HTTP_PORT=8006
MULTI_TIMEFRAME_HTTP_PORT=8005
PERFORMANCE_ATTRIBUTION_HTTP_PORT=8007
PORTFOLIO_OPTIMIZATION_HTTP_PORT=8004
```

### 数据源配置
- **价格数据**：本地JSONL文件格式
- **搜索API**：Jina AI API密钥
- **模型服务**：OpenAI API配置

## 性能优化

- **异步处理**：基于asyncio的异步IO
- **连接池**：HTTP连接复用
- **缓存机制**：热点数据缓存
- **批量处理**：批量数据操作

## 监控与调试

- **服务日志**：详细的工具调用日志
- **性能指标**：响应时间和成功率统计
- **错误追踪**：完整的错误堆栈信息
- **健康检查**：服务可用性监控

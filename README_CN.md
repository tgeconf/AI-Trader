<div align="center">

# 🤖 AI-Trader Bench
### *让AI在金融市场中一展身手*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


**AI股票交易代理系统，让多个大语言模型在纳斯达克100股票池中完全自主决策、同台竞技！**

[*点我以查看leaderboard*](https://hkuds.github.io/AI-Trader/)


## [🏆 当前锦标赛排行榜](https://hkuds.github.io/AI-Trader/)

<div align="center">

### 🥇 **锦标赛：至2025/10/22**

| 🏆 排名 | 🤖 AI模型 | 📈 总收益 |
|---------|-------------|----------------|
| **🥇 1st** | **DeepSeek** | 🚀 +8.55% | 
| 🥈 2nd | Claude-3.7 | 📊 +1.35% | 
| Baseline | QQQ | 📊 +0.37% |
| 🥉 3rd | GPT-5 | 📊 +0.28% | 
| 4th | Qwen3-max | 📊 -2.23% |
| 5th | Gemini-2.5-flash | 📊 -2.73% |

### 📊 **实时性能仪表板**
![rank](assets/rank.png)

*每日追踪各AI模型在纳斯达克100中的表现*

</div>

> 🎯 **核心特色**: 100% AI自主决策，零人工干预，纯工具驱动架构

[🚀 快速开始](#-快速开始) • [📈 性能分析](#-性能分析) • [🛠️ 配置指南](#-配置指南)

</div>

---

## 🌟 项目介绍

> **想象一下：5个不同的AI模型，每个都有独特的投资策略，在同一个市场中完全自主决策、竞争，看谁能在纳斯达克100中赚得最多！**

### 🎯 核心特性

- **🤖 完全自主决策**: AI代理100%自主分析、决策、执行，零人工干预
- **🛠️ 纯工具驱动**: 基于MCP工具链，AI通过工具调用完成所有交易操作
- **🏆 多模型竞技场**: 运行GPT、Claude、Qwen等多个AI模型进行交易
- **📊 实时性能追踪**: 完整的交易记录、持仓变化和收益分析
- **🔍 智能信息获取**: 集成Jina搜索，获取最新市场资讯和财报信息
- **⚡ MCP工具链**: 基于Model Context Protocol的模块化工具系统
- **🔌 策略可插拔**: 支持第三方策略和自定义AI代理集成
- **⏰ 可重放设计**: 支持任意时间段的回放，自动过滤未来信息


## 🚀 项目概述

AI-Trader Bench是一个创新的AI交易代理系统，它让多个大语言模型在真实的股票交易环境中同台竞技。每个AI代理都拥有：

### 🎮 交易环境
- **💰 初始资金**: $10,000 美元
- **📈 交易标的**: 纳斯达克100成分股（100只顶级科技股）
- **⏰ 交易时间**: 工作日交易，支持历史回放
- **📊 数据来源**: Alpha Vantage API + Jina AI搜索
- **🔄 时间控制**: 支持任意时间段的历史回放和未来信息过滤

### 🧠 AI代理能力
- **📰 智能信息获取**: 自动搜索市场新闻、分析师报告，自主筛选信息
- **💡 纯AI决策**: 基于多维度分析，AI完全自主做出买卖决策
- **📝 自动记录**: 系统自动记录每笔交易的详细日志和持仓变化
- **🔄 持续学习**: AI根据市场反馈自主调整策略

### 🏁 竞技规则
每个AI模型完全独立运行，使用相同的：
- **💰 初始资金**: $10,000美元起始资金
- **📊 市场数据**: 相同的价格数据和信息源
- **⏰ 交易时间**: 相同的交易时间窗口
- **📈 评估标准**: 统一的性能评估指标
- **🛠️ 工具集**: 相同的MCP工具链

**🎯 目标：看哪个AI模型能在完全自主的情况下获得最高的投资回报！**

### 🚫 零人工干预
- ❌ **无预设策略**: 不提供任何预设的交易策略或规则
- ❌ **无人工指导**: AI完全依靠自己的推理能力做决策
- ❌ **无手动干预**: 交易过程中不允许任何人工干预
- ✅ **纯工具驱动**: AI通过调用工具完成所有操作
- ✅ **自主学习**: AI根据市场反馈自主调整行为

## ⏰ 可重放设计

AI-Trader Bench的核心特色之一是**完全可重放**的交易环境，确保AI代理在历史数据上的表现具有科学性和可重复性。

### 🔄 时间控制机制

#### 📅 灵活的时间设置
```json
{
  "date_range": {
    "init_date": "2025-01-01",  // 任意开始日期
    "end_date": "2025-01-31"    // 任意结束日期
  }
}
```

#### 🛡️ 未来信息过滤
- **📊 价格数据**: 只提供当前日期及之前的价格信息
- **📰 新闻搜索**: 自动过滤未来日期的新闻和公告
- **📈 财报信息**: 只包含已发布的财务数据
- **🔍 市场分析**: 限制在指定时间点的可用信息

### 🎯 重放优势

#### 🔬 科学研究
- **📊 市场效率研究**: 测试AI在不同市场条件下的表现
- **🧠 认知偏差分析**: 研究AI决策的时间一致性
- **📈 风险模型验证**: 验证风险管理策略的有效性

#### 🎯 竞赛公平性
- **🏆 公平竞争**: 所有AI模型使用相同的历史信息
- **📊 客观评估**: 基于相同数据集的性能比较
- **🔍 透明度**: 完全可重现的实验结果

## 📁 项目架构

```
AI-Trader Bench/
├── 🤖 核心系统
│   ├── main.py    # 🎯 主程序入口
│   ├── agent/base_agent/          # 🧠 AI代理核心
│   └── configs/                   # ⚙️ 配置文件
│
├── 🛠️ MCP工具链
│   ├── agent_tools/
│   │   ├── tool_trade.py          # 💰 交易执行
│   │   ├── tool_get_price_local.py # 📊 价格查询
│   │   ├── tool_jina_search.py   # 🔍 信息搜索
│   │   └── tool_math.py           # 🧮 数学计算
│   └── tools/                     # 🔧 辅助工具
│
├── 📊 数据系统
│   ├── data/
│   │   ├── daily_prices_*.json    # 📈 股票价格数据
│   │   ├── merged.jsonl           # 🔄 统一数据格式
│   │   └── agent_data/            # 📝 AI交易记录
│   └── calculate_performance.py   # 📈 性能分析
│
├── 🎨 前端界面
│   └── frontend/                  # 🌐 Web仪表板
│
└── 📋 配置与文档
    ├── configs/                   # ⚙️ 系统配置
    ├── prompts/                   # 💬 AI提示词
    └── calc_perf.sh              # 🚀 性能计算脚本
```

### 🔧 核心组件详解

#### 🎯 主程序 (`main.py`)
- **多模型并发**: 同时运行多个AI模型进行交易
- **配置管理**: 支持JSON配置文件和环境变量
- **日期管理**: 灵活的交易日历和日期范围设置
- **错误处理**: 完善的异常处理和重试机制

#### 🛠️ MCP工具链
| 工具 | 功能 | API |
|------|------|-----|
| **交易工具** | 买入/卖出股票，持仓管理 | `buy()`, `sell()` |
| **价格工具** | 实时和历史价格查询 | `get_price_local()` |
| **搜索工具** | 市场信息搜索 | `get_information()` |
| **数学工具** | 财务计算和分析 | 基础数学运算 |

#### 📊 数据系统
- **📈 价格数据**: 纳斯达克100成分股的完整OHLCV数据
- **📝 交易记录**: 每个AI模型的详细交易历史
- **📊 性能指标**: 夏普比率、最大回撤、年化收益等
- **🔄 数据同步**: 自动化的数据获取和更新机制

## 🚀 快速开始

### 📋 前置要求

- **Python 3.8+** 
- **API密钥**: OpenAI、Alpha Vantage、Jina AI


### ⚡ 一键安装

```bash
# 1. 克隆项目
git clone https://github.com/HKUDS/AI-Trader.git
cd AI-Trader

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 🔑 环境配置

创建 `.env` 文件并配置以下变量：

```bash
# 🤖 AI模型API配置
OPENAI_API_BASE=https://your-openai-proxy.com/v1
OPENAI_API_KEY=your_openai_key

# 📊 数据源配置
ALPHAADVANTAGE_API_KEY=your_alpha_vantage_key
JINA_API_KEY=your_jina_api_key

# ⚙️ 系统配置
RUNTIME_ENV_PATH=./runtime_env.json #推荐使用绝对路径

# 🌐 服务端口配置
MATH_HTTP_PORT=8000
SEARCH_HTTP_PORT=8001
TRADE_HTTP_PORT=8002
GETPRICE_HTTP_PORT=8003
# 🧠 AI代理配置
AGENT_MAX_STEP=30             # 最大推理步数
```

### 📦 依赖包

```bash
# 安装生产环境依赖
pip install -r requirements.txt

# 或手动安装核心依赖
pip install langchain langchain-openai langchain-mcp-adapters fastmcp python-dotenv requests numpy pandas
```

## 🎮 运行指南

### 📊 步骤1: 数据准备 (`./fresh_data.sh`)


```bash
# 📈 获取纳斯达克100股票数据
cd data
python get_daily_price.py

# 🔄 合并数据为统一格式
python merge_jsonl.py
```

### 🛠️ 步骤2: 启动MCP服务

```bash
cd ./agent_tools
python start_mcp_services.py
```

### 🚀 步骤3: 启动AI竞技场

```bash
# 🎯 运行主程序 - 让AI们开始交易！
python main.py

# 🎯 或使用自定义配置
python main.py configs/my_config.json
```

### ⏰ 时间设置示例

#### 📅 创建自定义时间配置
```json
{
  "agent_type": "BaseAgent",
  "date_range": {
    "init_date": "2024-01-01",  // 回测开始日期
    "end_date": "2024-03-31"     // 回测结束日期
  },
  "models": [
    {
      "name": "claude-3.7-sonnet",
      "basemodel": "anthropic/claude-3.7-sonnet",
      "signature": "claude-3.7-sonnet",
      "enabled": true
    }
  ]
}
```

### 📈 启动Web界面

```bash
cd docs
python3 -m http.server 8000
# 访问 http://localhost:8000
```


## 📈 性能分析

### 🏆 竞技规则

| 规则项 | 设置 | 说明 |
|--------|------|------|
| **💰 初始资金** | $10,000 | 每个AI模型起始资金 |
| **📈 交易标的** | 纳斯达克100 | 100只顶级科技股 |
| **⏰ 交易时间** | 工作日 | 周一至周五 |
| **💲 价格基准** | 开盘价 | 使用当日开盘价交易 |
| **📝 记录方式** | JSONL格式 | 完整交易历史记录 |

## ⚙️ 配置指南

### 📋 配置文件结构

```json
{
  "agent_type": "BaseAgent",
  "date_range": {
    "init_date": "2025-01-01",
    "end_date": "2025-01-31"
  },
  "models": [
    {
      "name": "claude-3.7-sonnet",
      "basemodel": "anthropic/claude-3.7-sonnet",
      "signature": "claude-3.7-sonnet",
      "enabled": true
    }
  ],
  "agent_config": {
    "max_steps": 30,
    "max_retries": 3,
    "base_delay": 1.0,
    "initial_cash": 10000.0
  },
  "log_config": {
    "log_path": "./data/agent_data"
  }
}
```

### 🔧 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `agent_type` | AI代理类型 | "BaseAgent" |
| `max_steps` | 最大推理步数 | 30 |
| `max_retries` | 最大重试次数 | 3 |
| `base_delay` | 操作延迟(秒) | 1.0 |
| `initial_cash` | 初始资金 | $10,000 |

### 📊 数据格式

#### 💰 持仓记录 (position.jsonl)
```json
{
  "date": "2025-01-20",
  "id": 1,
  "this_action": {
    "action": "buy",
    "symbol": "AAPL", 
    "amount": 10
  },
  "positions": {
    "AAPL": 10,
    "MSFT": 0,
    "CASH": 9737.6
  }
}
```

#### 📈 价格数据 (merged.jsonl)
```json
{
  "Meta Data": {
    "2. Symbol": "AAPL",
    "3. Last Refreshed": "2025-01-20"
  },
  "Time Series (Daily)": {
    "2025-01-20": {
      "1. buy price": "255.8850",
      "2. high": "264.3750", 
      "3. low": "255.6300",
      "4. sell price": "262.2400",
      "5. volume": "90483029"
    }
  }
}
```

### 📁 文件结构

```
data/agent_data/
├── claude-3.7-sonnet/
│   ├── position/
│   │   └── position.jsonl      # 📝 持仓记录
│   └── log/
│       └── 2025-01-20/
│           └── log.jsonl       # 📊 交易日志
├── gpt-4o/
│   └── ...
└── qwen3-max/
    └── ...
```

## 🔌 第三方策略集成

AI-Trader Bench采用模块化设计，支持轻松集成第三方策略和自定义AI代理。

### 🛠️ 集成方式

#### 1. 自定义AI代理
```python
# 创建新的AI代理类
class CustomAgent(BaseAgent):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        # 添加自定义逻辑
```

#### 2. 注册新代理
```python
# 在 main.py 中注册
AGENT_REGISTRY = {
    "BaseAgent": {
        "module": "agent.base_agent.base_agent",
        "class": "BaseAgent"
    },
    "CustomAgent": {  # 新增
        "module": "agent.custom.custom_agent",
        "class": "CustomAgent"
    },
}
```

#### 3. 配置文件设置
```json
{
  "agent_type": "CustomAgent",
  "models": [
    {
      "name": "your-custom-model",
      "basemodel": "your/model/path",
      "signature": "custom-signature",
      "enabled": true
    }
  ]
}
```

### 🔧 扩展工具链

#### 添加自定义工具
```python
# 创建新的MCP工具
@mcp.tools()
class CustomTool:
    def __init__(self):
        self.name = "custom_tool"
    
    def execute(self, params):
        # 实现自定义工具逻辑
        return result
```

## 🚀 路线图

### 🌟 未来计划
- [ ] **🇨🇳 A股支持** - 扩展至中国股市
- [ ] **📊 收盘后统计** - 自动收益分析
- [ ] **🔌 策略市场** - 添加第三方策略分享平台
- [ ] **🎨 炫酷前端界面** - 现代化Web仪表板
- [ ] **₿ 加密货币** - 支持数字货币交易
- [ ] **📈 更多策略** - 技术分析、量化策略
- [ ] **⏰ 高级回放** - 支持分钟级时间精度和实时回放
- [ ] **🔍 智能过滤** - 更精确的未来信息检测和过滤

## 🤝 贡献指南

我们欢迎各种形式的贡献！特别是AI交易策略和代理实现。

### 🧠 AI策略贡献
- **🎯 交易策略**: 贡献你的AI交易策略实现
- **🤖 自定义代理**: 实现新的AI代理类型
- **📊 分析工具**: 添加新的市场分析工具
- **🔍 数据源**: 集成新的数据源和API

### 🐛 问题报告
- 使用GitHub Issues报告bug
- 提供详细的复现步骤
- 包含系统环境信息

### 💡 功能建议
- 在Issues中提出新功能想法
- 详细描述使用场景
- 讨论实现方案

### 🔧 代码贡献
1. Fork项目
2. 创建功能分支
3. 实现你的策略或功能
4. 添加测试用例
5. 创建Pull Request

### 📚 文档改进
- 完善README文档
- 添加代码注释
- 编写使用教程
- 贡献策略说明文档

### 🏆 策略分享
- **📈 技术分析策略**: 基于技术指标的AI策略
- **📊 量化策略**: 多因子模型和量化分析
- **🔍 基本面策略**: 基于财务数据的分析策略
- **🌐 宏观策略**: 基于宏观经济数据的策略

## 📞 支持与社区

- **💬 讨论**: [GitHub Discussions](https://github.com/HKUDS/AI-Trader/discussions)
- **🐛 问题**: [GitHub Issues](https://github.com/HKUDS/AI-Trader/issues)
- **📧 联系**: your-email@example.com

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

感谢以下开源项目和服务：
- [LangChain](https://github.com/langchain-ai/langchain) - AI应用开发框架
- [MCP](https://github.com/modelcontextprotocol) - Model Context Protocol
- [Alpha Vantage](https://www.alphavantage.co/) - 金融数据API
- [Jina AI](https://jina.ai/) - 信息搜索服务

---

<div align="center">

**🌟 如果这个项目对你有帮助，请给我们一个Star！**

[![GitHub stars](https://img.shields.io/github/stars/HKUDS/AI-Trader?style=social)](https://github.com/HKUDS/AI-Trader)
[![GitHub forks](https://img.shields.io/github/forks/HKUDS/AI-Trader?style=social)](https://github.com/HKUDS/AI-Trader)

**🤖 让AI在金融市场中完全自主决策、一展身手！**  
**🛠️ 纯工具驱动，零人工干预，真正的AI交易竞技场！** 🚀

</div>
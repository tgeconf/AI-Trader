"""
BaseAgent class - Base class for trading agents
Encapsulates core functionality including MCP tool management, AI agent creation, and trading execution
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np  # 添加numpy导入

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

# Import project tools
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tools.general_tools import extract_conversation, extract_tool_messages, get_config_value, write_config_value
from tools.price_tools import add_no_trade_record
from prompts.agent_prompt import get_agent_system_prompt, STOP_SIGNAL

# Import professional risk management modules
from risk_management.var_model import VaRModel
from risk_management.position_sizing import PositionSizing
from risk_management.stop_loss import StopLossManager

# Load environment variables
load_dotenv()


class BaseAgent:
    """
    Base class for trading agents
    
    Main functionalities:
    1. MCP tool management and connection
    2. AI agent creation and configuration
    3. Trading execution and decision loops
    4. Logging and management
    5. Position and configuration management
    """
    
    # Default NASDAQ 100 stock symbols
    DEFAULT_STOCK_SYMBOLS = [
        "NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
        "NFLX", "PLTR", "COST", "ASML", "AMD", "CSCO", "AZN", "TMUS", "MU", "LIN",
        "PEP", "SHOP", "APP", "INTU", "AMAT", "LRCX", "PDD", "QCOM", "ARM", "INTC",
        "BKNG", "AMGN", "TXN", "ISRG", "GILD", "KLAC", "PANW", "ADBE", "HON",
        "CRWD", "CEG", "ADI", "ADP", "DASH", "CMCSA", "VRTX", "MELI", "SBUX",
        "CDNS", "ORLY", "SNPS", "MSTR", "MDLZ", "ABNB", "MRVL", "CTAS", "TRI",
        "MAR", "MNST", "CSX", "ADSK", "PYPL", "FTNT", "AEP", "WDAY", "REGN", "ROP",
        "NXPI", "DDOG", "AXON", "ROST", "IDXX", "EA", "PCAR", "FAST", "EXC", "TTWO",
        "XEL", "ZS", "PAYX", "WBD", "BKR", "CPRT", "CCEP", "FANG", "TEAM", "CHTR",
        "KDP", "MCHP", "GEHC", "VRSK", "CTSH", "CSGP", "KHC", "ODFL", "DXCM", "TTD",
        "ON", "BIIB", "LULU", "CDW", "GFS"
    ]
    
    def __init__(
        self,
        signature: str,
        basemodel: str,
        stock_symbols: Optional[List[str]] = None,
        mcp_config: Optional[Dict[str, Dict[str, Any]]] = None,
        log_path: Optional[str] = None,
        max_steps: int = 10,
        max_retries: int = 3,
        base_delay: float = 0.5,
        openai_base_url: Optional[str] = None,
        initial_cash: float = 10000.0,
        init_date: str = "2025-10-13"
    ):
        """
        Initialize BaseAgent
        
        Args:
            signature: Agent signature/name
            basemodel: Base model name
            stock_symbols: List of stock symbols, defaults to NASDAQ 100
            mcp_config: MCP tool configuration, including port and URL information
            log_path: Log path, defaults to ./data/agent_data
            max_steps: Maximum reasoning steps
            max_retries: Maximum retry attempts
            base_delay: Base delay time for retries
            openai_base_url: OpenAI API base URL
            initial_cash: Initial cash amount
            init_date: Initialization date
        """
        self.signature = signature
        self.basemodel = basemodel
        self.stock_symbols = stock_symbols or self.DEFAULT_STOCK_SYMBOLS
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.initial_cash = initial_cash
        self.init_date = init_date
        
        # Set MCP configuration
        self.mcp_config = mcp_config or self._get_default_mcp_config()
        
        # Set log path
        self.base_log_path = log_path or "./data/agent_data"
        
        # Set OpenAI configuration
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_API_BASE")
        
        # Initialize professional risk management modules
        self.var_model = VaRModel()
        self.position_sizer = PositionSizing(
            total_capital=initial_cash,
            max_risk_per_trade=0.02,  # 2% per trade
            max_portfolio_risk=0.10,  # 10% total portfolio risk
            max_leverage=2.0
        )
        self.stop_loss_manager = StopLossManager()
        
        # Initialize components
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: Optional[List] = None
        self.model: Optional[ChatOpenAI] = None
        self.agent: Optional[Any] = None
        
        # Data paths
        self.data_path = os.path.join(self.base_log_path, self.signature)
        self.position_file = os.path.join(self.data_path, "position", "position.jsonl")
        
    def _get_default_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default MCP configuration"""
        return {
            "math": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('MATH_HTTP_PORT', '8000')}/mcp",
            },
            "stock_local": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('GETPRICE_HTTP_PORT', '8003')}/mcp",
            },
            "search": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('SEARCH_HTTP_PORT', '8001')}/mcp",
            },
            "trade": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('TRADE_HTTP_PORT', '8002')}/mcp",
            },
        }
    
    async def initialize(self) -> None:
        """Initialize MCP client and AI model"""
        print(f"🚀 Initializing agent: {self.signature}")
        
        # Create MCP client
        self.client = MultiServerMCPClient(self.mcp_config)
        
        # Get tools
        self.tools = await self.client.get_tools()
        print(f"✅ Loaded {len(self.tools)} MCP tools")
        
        # Create AI model
        self.model = ChatOpenAI(
            model=self.basemodel,
            base_url=self.openai_base_url,
            max_retries=3,
            timeout=30
        )
        
        # Note: agent will be created in run_trading_session() based on specific date
        # because system_prompt needs the current date and price information
        
        print(f"✅ Agent {self.signature} initialization completed")
    
    def _setup_logging(self, today_date: str) -> str:
        """Set up log file path"""
        log_path = os.path.join(self.base_log_path, self.signature, 'log', today_date)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        return os.path.join(log_path, "log.jsonl")
    
    def _log_message(self, log_file: str, new_messages: List[Dict[str, str]]) -> None:
        """Log messages to log file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "signature": self.signature,
            "new_messages": new_messages
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    async def _ainvoke_with_retry(self, message: List[Dict[str, str]]) -> Any:
        """Agent invocation with retry"""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.agent.ainvoke(
                    {"messages": message}, 
                    {"recursion_limit": 100}
                )
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                print(f"⚠️ Attempt {attempt} failed, retrying after {self.base_delay * attempt} seconds...")
                print(f"Error details: {e}")
                await asyncio.sleep(self.base_delay * attempt)
    
    async def run_trading_session(self, today_date: str) -> None:
        """
        Run single day trading session
        
        Args:
            today_date: Trading date
        """
        print(f"📈 Starting trading session: {today_date}")
        
        # Set up logging
        log_file = self._setup_logging(today_date)
        
        # Update system prompt
        self.agent = create_agent(
            self.model,
            tools=self.tools,
            system_prompt=get_agent_system_prompt(today_date, self.signature),
        )
        
        # Initial user query
        user_query = [{"role": "user", "content": f"Please analyze and update today's ({today_date}) positions."}]
        message = user_query.copy()
        
        # Log initial message
        self._log_message(log_file, user_query)
        
        # Trading loop
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"🔄 Step {current_step}/{self.max_steps}")
            
            try:
                # Call agent
                response = await self._ainvoke_with_retry(message)
                
                # Extract agent response
                agent_response = extract_conversation(response, "final")
                
                # Check stop signal
                if STOP_SIGNAL in agent_response:
                    print("✅ Received stop signal, trading session ended")
                    print(agent_response)
                    self._log_message(log_file, [{"role": "assistant", "content": agent_response}])
                    break
                
                # Extract tool messages
                tool_msgs = extract_tool_messages(response)
                tool_response = '\n'.join([msg.content for msg in tool_msgs])
                
                # Prepare new messages
                new_messages = [
                    {"role": "assistant", "content": agent_response},
                    {"role": "user", "content": f'Tool results: {tool_response}'}
                ]
                
                # Add new messages
                message.extend(new_messages)
                
                # Log messages
                self._log_message(log_file, new_messages[0])
                self._log_message(log_file, new_messages[1])
                
            except Exception as e:
                print(f"❌ Trading session error: {str(e)}")
                print(f"Error details: {e}")
                raise
        
        # Handle trading results
        await self._handle_trading_result(today_date)
    
    async def _handle_trading_result(self, today_date: str) -> None:
        """Handle trading results"""
        if_trade = get_config_value("IF_TRADE")
        if if_trade:
            write_config_value("IF_TRADE", False)
            print("✅ Trading completed")
        else:
            print("📊 No trading, maintaining positions")
            try:
                add_no_trade_record(today_date, self.signature)
            except NameError as e:
                print(f"❌ NameError: {e}")
                raise
            write_config_value("IF_TRADE", False)
    
    def register_agent(self) -> None:
        """Register new agent, create initial positions"""
        # Check if position.jsonl file already exists
        if os.path.exists(self.position_file):
            print(f"⚠️ Position file {self.position_file} already exists, skipping registration")
            return
        
        # Ensure directory structure exists
        position_dir = os.path.join(self.data_path, "position")
        if not os.path.exists(position_dir):
            os.makedirs(position_dir)
            print(f"📁 Created position directory: {position_dir}")
        
        # Create initial positions
        init_position = {symbol: 0 for symbol in self.stock_symbols}
        init_position['CASH'] = self.initial_cash
        
        with open(self.position_file, "w") as f:  # Use "w" mode to ensure creating new file
            f.write(json.dumps({
                "date": self.init_date, 
                "id": 0, 
                "positions": init_position
            }) + "\n")
        
        print(f"✅ Agent {self.signature} registration completed")
        print(f"📁 Position file: {self.position_file}")
        print(f"💰 Initial cash: ${self.initial_cash}")
        print(f"📊 Number of stocks: {len(self.stock_symbols)}")
    
    def get_trading_dates(self, init_date: str, end_date: str) -> List[str]:
        """
        Get trading date list
        
        Args:
            init_date: Start date
            end_date: End date
            
        Returns:
            List of trading dates
        """
        dates = []
        max_date = None
        
        if not os.path.exists(self.position_file):
            self.register_agent()
            max_date = init_date
        else:
            # Read existing position file, find latest date
            with open(self.position_file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    current_date = doc['date']
                    if max_date is None:
                        max_date = current_date
                    else:
                        current_date_obj = datetime.strptime(current_date, "%Y-%m-%d")
                        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
                        if current_date_obj > max_date_obj:
                            max_date = current_date
        
        # Check if new dates need to be processed
        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        if end_date_obj <= max_date_obj:
            return []
        
        # Generate trading date list
        trading_dates = []
        current_date = max_date_obj + timedelta(days=1)
        
        while current_date <= end_date_obj:
            if current_date.weekday() < 5:  # Weekdays
                trading_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        return trading_dates
    
    async def run_with_retry(self, today_date: str) -> None:
        """Run method with retry"""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"🔄 Attempting to run {self.signature} - {today_date} (Attempt {attempt})")
                await self.run_trading_session(today_date)
                print(f"✅ {self.signature} - {today_date} run successful")
                return
            except Exception as e:
                print(f"❌ Attempt {attempt} failed: {str(e)}")
                if attempt == self.max_retries:
                    print(f"💥 {self.signature} - {today_date} all retries failed")
                    raise
                else:
                    wait_time = self.base_delay * attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
    
    async def run_date_range(self, init_date: str, end_date: str) -> None:
        """
        Run all trading days in date range
        
        Args:
            init_date: Start date
            end_date: End date
        """
        print(f"📅 Running date range: {init_date} to {end_date}")
        
        # Get trading date list
        trading_dates = self.get_trading_dates(init_date, end_date)
        
        if not trading_dates:
            print(f"ℹ️ No trading days to process")
            return
        
        print(f"📊 Trading days to process: {trading_dates}")
        
        # Process each trading day
        for date in trading_dates:
            print(f"🔄 Processing {self.signature} - Date: {date}")
            
            # Set configuration
            write_config_value("TODAY_DATE", date)
            write_config_value("SIGNATURE", self.signature)
            
            try:
                await self.run_with_retry(date)
            except Exception as e:
                print(f"❌ Error processing {self.signature} - Date: {date}")
                print(e)
                raise
        
        print(f"✅ {self.signature} processing completed")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary"""
        if not os.path.exists(self.position_file):
            return {"error": "Position file does not exist"}
        
        positions = []
        with open(self.position_file, "r") as f:
            for line in f:
                positions.append(json.loads(line))
        
        if not positions:
            return {"error": "No position records"}
        
        latest_position = positions[-1]
        return {
            "signature": self.signature,
            "latest_date": latest_position.get("date"),
            "positions": latest_position.get("positions", {}),
            "total_records": len(positions)
        }
    
    def calculate_portfolio_risk(self, portfolio_returns: List[float]) -> Dict[str, Any]:
        """
        计算投资组合风险指标
        
        Args:
            portfolio_returns: 投资组合历史收益率列表
            
        Returns:
            风险指标字典
        """
        if not portfolio_returns:
            return {"error": "No portfolio returns data"}
        
        # 计算VaR
        var_result = self.var_model.calculate_var(
            returns=portfolio_returns,
            confidence_level=0.95,
            method="historical"
        )
        
        # 计算CVaR
        cvar_result = self.var_model.calculate_cvar(
            returns=portfolio_returns,
            confidence_level=0.95
        )
        
        return {
            "var_95": var_result,
            "cvar_95": cvar_result,
            "volatility": np.std(portfolio_returns),
            "max_drawdown": self._calculate_max_drawdown(portfolio_returns)
        }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return np.min(drawdowns)
    
    def calculate_position_size(self, symbol: str, price: float, 
                              expected_return: float = 0.0,
                              volatility: float = 0.2) -> Dict[str, Any]:
        """
        计算单只股票的头寸规模
        
        Args:
            symbol: 股票代码
            price: 当前价格
            expected_return: 预期收益率
            volatility: 波动率
            
        Returns:
            头寸规模计算结果
        """
        try:
            # 使用凯利公式计算头寸规模
            kelly_result = self.position_sizer.calculate_kelly_position(
                win_rate=0.6,  # 假设胜率
                win_loss_ratio=2.0,  # 假设盈亏比
                total_capital=self.initial_cash
            )
            
            # 使用波动率调整法
            volatility_result = self.position_sizer.calculate_volatility_adjusted_position(
                symbol=symbol,
                price=price,
                volatility=volatility,
                max_risk_per_trade=0.02
            )
            
            return {
                "symbol": symbol,
                "kelly_position": kelly_result,
                "volatility_adjusted_position": volatility_result,
                "recommended_position": min(kelly_result, volatility_result),
                "max_shares": int(self.initial_cash * 0.02 / price)  # 单笔交易最大2%风险
            }
        except Exception as e:
            return {"error": f"Position sizing calculation failed: {str(e)}"}
    
    def check_stop_loss(self, symbol: str, current_price: float, 
                       entry_price: float, position_size: int) -> Dict[str, Any]:
        """
        检查止损条件
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            entry_price: 入场价格
            position_size: 持仓数量
            
        Returns:
            止损检查结果
        """
        try:
            # 使用固定百分比止损
            fixed_stop = self.stop_loss_manager.fixed_percentage_stop_loss(
                entry_price=entry_price,
                current_price=current_price,
                stop_loss_percentage=0.05  # 5%止损
            )
            
            # 使用ATR止损
            atr_stop = self.stop_loss_manager.atr_stop_loss(
                entry_price=entry_price,
                current_price=current_price,
                atr_value=current_price * 0.02,  # 假设ATR为价格的2%
                atr_multiplier=2.0
            )
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "entry_price": entry_price,
                "position_size": position_size,
                "fixed_stop_loss": fixed_stop,
                "atr_stop_loss": atr_stop,
                "should_stop": fixed_stop["should_stop"] or atr_stop["should_stop"],
                "stop_reason": "fixed_percentage" if fixed_stop["should_stop"] else "atr" if atr_stop["should_stop"] else "none"
            }
        except Exception as e:
            return {"error": f"Stop loss check failed: {str(e)}"}
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        生成风险报告
        
        Returns:
            风险报告字典
        """
        position_summary = self.get_position_summary()
        
        if "error" in position_summary:
            return {"error": "Cannot generate risk report: " + position_summary["error"]}
        
        # 这里可以添加更复杂的风险计算
        # 例如基于历史数据的VaR计算等
        
        return {
            "signature": self.signature,
            "latest_date": position_summary["latest_date"],
            "total_capital": position_summary["positions"].get("CASH", 0) + sum(
                position_summary["positions"].get(symbol, 0) * 100  # 假设每股市值100美元
                for symbol in self.stock_symbols
            ),
            "cash_balance": position_summary["positions"].get("CASH", 0),
            "stock_positions": {
                symbol: position_summary["positions"].get(symbol, 0)
                for symbol in self.stock_symbols
                if position_summary["positions"].get(symbol, 0) > 0
            },
            "risk_metrics": {
                "max_risk_per_trade": 0.02,
                "max_portfolio_risk": 0.10,
                "max_leverage": 2.0
            }
        }
    
    def __str__(self) -> str:
        return f"BaseAgent(signature='{self.signature}', basemodel='{self.basemodel}', stocks={len(self.stock_symbols)})"
    
    def __repr__(self) -> str:
        return self.__str__()

from fastmcp import FastMCP
import sys
import os
from typing import Dict, List, Optional, Any
# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from tools.price_tools import get_yesterday_date, get_open_prices, get_yesterday_open_and_close_price, get_latest_position, get_yesterday_profit
import json
from tools.general_tools import get_config_value,write_config_value

# Import transaction cost calculator
from trading_costs.transaction_cost_calculator import TransactionCostCalculator
from trading_costs.commission_model import CommissionModel, CommissionType
from trading_costs.slippage_model import SlippageModel, SlippageType

mcp = FastMCP("TradeTools")



@mcp.tool()
def buy(symbol: str, amount: int) -> Dict[str, Any]:
    """
    Buy stock function
    
    This function simulates stock buying operations, including the following steps:
    1. Get current position and operation ID
    2. Get stock opening price for the day
    3. Validate buy conditions (sufficient cash)
    4. Calculate transaction costs
    5. Update position (increase stock quantity, decrease cash)
    6. Record transaction to position.jsonl file
    
    Args:
        symbol: Stock symbol, such as "AAPL", "MSFT", etc.
        amount: Buy quantity, must be a positive integer, indicating how many shares to buy
        
    Returns:
        Dict[str, Any]:
          - Success: Returns new position dictionary (containing stock quantity and cash balance)
          - Failure: Returns {"error": error message, ...} dictionary
        
    Raises:
        ValueError: Raised when SIGNATURE environment variable is not set
        
    Example:
        >>> result = buy("AAPL", 10)
        >>> print(result)  # {"AAPL": 110, "MSFT": 5, "CASH": 5000.0, ...}
    """
    # Step 1: Get environment variables and basic information
    # Get signature (model name) from environment variable, used to determine data storage path
    signature = get_config_value("SIGNATURE")
    if signature is None:
        raise ValueError("SIGNATURE environment variable is not set")
    
    # Get current trading date from environment variable
    today_date = get_config_value("TODAY_DATE")
    
    # Step 2: Get current latest position and operation ID
    # get_latest_position returns two values: position dictionary and current maximum operation ID
    # This ID is used to ensure each operation has a unique identifier
    try:
        current_position, current_action_id = get_latest_position(today_date, signature)
    except Exception as e:
        print(e)
        print(current_position, current_action_id)
        print(today_date, signature)
    # Step 3: Get stock opening price for the day
    # Use get_open_prices function to get the opening price of specified stock for the day
    # If stock symbol does not exist or price data is missing, KeyError exception will be raised
    try:
        this_symbol_price = get_open_prices(today_date, [symbol])[f'{symbol}_price']
    except KeyError:
        # Stock symbol does not exist or price data is missing, return error message
        return {"error": f"Symbol {symbol} not found! This action will not be allowed.", "symbol": symbol, "date": today_date}

    # Step 4: Calculate transaction costs
    # Initialize transaction cost calculator
    commission_model = CommissionModel(commission_type=CommissionType.PROPORTIONAL, rate=0.001)  # 0.1% commission
    slippage_model = SlippageModel(slippage_type=SlippageType.VOLUME_ADJUSTED, base_slippage=0.001)
    cost_calculator = TransactionCostCalculator(
        commission_model=commission_model,
        slippage_model=slippage_model
    )
    
    # Calculate transaction costs
    trade_amount = this_symbol_price * amount
    cost_result = cost_calculator.calculate_single_trade_cost(
        symbol=symbol,
        price=this_symbol_price,
        volume=amount,
        market_data={
            'volatility': 0.2,  # 假设波动率20%
            'average_volume': 1000000,  # 假设平均成交量
            'current_volume': 500000  # 假设当前成交量
        }
    )
    
    total_cost = trade_amount + cost_result.total_cost
    
    # Step 5: Validate buy conditions
    # Calculate cash required for purchase: stock price × buy quantity + transaction costs
    try:
        cash_left = current_position["CASH"] - total_cost
    except Exception as e:
        print(current_position, "CASH", this_symbol_price, amount)

    # Check if cash balance is sufficient for purchase
    if cash_left < 0:
        # Insufficient cash, return error message
        return {"error": "Insufficient cash! This action will not be allowed.", "required_cash": total_cost, "cash_available": current_position.get("CASH", 0), "symbol": symbol, "date": today_date}
    else:
        # Step 6: Execute buy operation, update position
        # Create a copy of current position to avoid directly modifying original data
        new_position = current_position.copy()
        
        # Decrease cash balance (including transaction costs)
        new_position["CASH"] = cash_left
        
        # Increase stock position quantity
        new_position[symbol] += amount
        
        # Step 7: Record transaction to position.jsonl file
        # Build file path: {project_root}/data/agent_data/{signature}/position/position.jsonl
        # Use append mode ("a") to write new transaction record
        # Each operation ID increments by 1, ensuring uniqueness of operation sequence
        position_file_path = os.path.join(project_root, "data", "agent_data", signature, "position", "position.jsonl")
        with open(position_file_path, "a") as f:
            # Write JSON format transaction record, containing date, operation ID, transaction details and updated position
            transaction_record = {
                "date": today_date,
                "id": current_action_id + 1,
                "this_action": {
                    "action": "buy",
                    "symbol": symbol,
                    "amount": amount,
                    "price": this_symbol_price,
                    "transaction_costs": {
                        "commission": cost_result.commission_cost,
                        "slippage": cost_result.slippage_cost,
                        "market_impact": cost_result.market_impact_cost,
                        "total_cost": cost_result.total_cost,
                        "effective_price": cost_result.effective_price
                    }
                },
                "positions": new_position
            }
            print(f"Writing to position.jsonl: {json.dumps(transaction_record)}")
            f.write(json.dumps(transaction_record) + "\n")
        
        # Step 8: Return updated position with cost information
        write_config_value("IF_TRADE", True)
        print("IF_TRADE", get_config_value("IF_TRADE"))
        return {
            "new_position": new_position,
            "transaction_costs": {
                "commission": cost_result.commission_cost,
                "slippage": cost_result.slippage_cost,
                "market_impact": cost_result.market_impact_cost,
                "total_cost": cost_result.total_cost,
                "effective_price": cost_result.effective_price
            },
            "cost_breakdown": cost_result.breakdown
        }

@mcp.tool()
def sell(symbol: str, amount: int) -> Dict[str, Any]:
    """
    Sell stock function
    
    This function simulates stock selling operations, including the following steps:
    1. Get current position and operation ID
    2. Get stock opening price for the day
    3. Validate sell conditions (position exists, sufficient quantity)
    4. Calculate transaction costs
    5. Update position (decrease stock quantity, increase cash)
    6. Record transaction to position.jsonl file
    
    Args:
        symbol: Stock symbol, such as "AAPL", "MSFT", etc.
        amount: Sell quantity, must be a positive integer, indicating how many shares to sell
        
    Returns:
        Dict[str, Any]:
          - Success: Returns new position dictionary (containing stock quantity and cash balance)
          - Failure: Returns {"error": error message, ...} dictionary
        
    Raises:
        ValueError: Raised when SIGNATURE environment variable is not set
        
    Example:
        >>> result = sell("AAPL", 10)
        >>> print(result)  # {"AAPL": 90, "MSFT": 5, "CASH": 15000.0, ...}
    """
    # Step 1: Get environment variables and basic information
    # Get signature (model name) from environment variable, used to determine data storage path
    signature = get_config_value("SIGNATURE")
    if signature is None:
        raise ValueError("SIGNATURE environment variable is not set")
    
    # Get current trading date from environment variable
    today_date = get_config_value("TODAY_DATE")
    
    # Step 2: Get current latest position and operation ID
    # get_latest_position returns two values: position dictionary and current maximum operation ID
    # This ID is used to ensure each operation has a unique identifier
    current_position, current_action_id = get_latest_position(today_date, signature)
    
    # Step 3: Get stock opening price for the day
    # Use get_open_prices function to get the opening price of specified stock for the day
    # If stock symbol does not exist or price data is missing, KeyError exception will be raised
    try:
        this_symbol_price = get_open_prices(today_date, [symbol])[f'{symbol}_price']
    except KeyError:
        # Stock symbol does not exist or price data is missing, return error message
        return {"error": f"Symbol {symbol} not found! This action will not be allowed.", "symbol": symbol, "date": today_date}

    # Step 4: Validate sell conditions
    # Check if holding this stock
    if symbol not in current_position:
        return {"error": f"No position for {symbol}! This action will not be allowed.", "symbol": symbol, "date": today_date}

    # Check if position quantity is sufficient for selling
    if current_position[symbol] < amount:
        return {"error": "Insufficient shares! This action will not be allowed.", "have": current_position.get(symbol, 0), "want_to_sell": amount, "symbol": symbol, "date": today_date}

    # Step 5: Calculate transaction costs
    # Initialize transaction cost calculator
    commission_model = CommissionModel(commission_type=CommissionType.PROPORTIONAL, rate=0.001)  # 0.1% commission
    slippage_model = SlippageModel(slippage_type=SlippageType.VOLUME_ADJUSTED, base_slippage=0.001)
    cost_calculator = TransactionCostCalculator(
        commission_model=commission_model,
        slippage_model=slippage_model
    )
    
    # Calculate transaction costs
    trade_amount = this_symbol_price * amount
    cost_result = cost_calculator.calculate_single_trade_cost(
        symbol=symbol,
        price=this_symbol_price,
        volume=amount,
        market_data={
            'volatility': 0.2,  # 假设波动率20%
            'average_volume': 1000000,  # 假设平均成交量
            'current_volume': 500000  # 假设当前成交量
        }
    )
    
    net_proceeds = trade_amount - cost_result.total_cost
    
    # Step 6: Execute sell operation, update position
    # Create a copy of current position to avoid directly modifying original data
    new_position = current_position.copy()
    
    # Decrease stock position quantity
    new_position[symbol] -= amount
    
    # Increase cash balance: sell price × sell quantity - transaction costs
    # Use get method to ensure CASH field exists, default to 0 if not present
    new_position["CASH"] = new_position.get("CASH", 0) + net_proceeds

    # Step 7: Record transaction to position.jsonl file
    # Build file path: {project_root}/data/agent_data/{signature}/position/position.jsonl
    # Use append mode ("a") to write new transaction record
    # Each operation ID increments by 1, ensuring uniqueness of operation sequence
    position_file_path = os.path.join(project_root, "data", "agent_data", signature, "position", "position.jsonl")
    with open(position_file_path, "a") as f:
        # Write JSON format transaction record, containing date, operation ID and updated position
        transaction_record = {
            "date": today_date,
            "id": current_action_id + 1,
            "this_action": {
                "action": "sell",
                "symbol": symbol,
                "amount": amount,
                "price": this_symbol_price,
                "transaction_costs": {
                    "commission": cost_result.commission_cost,
                    "slippage": cost_result.slippage_cost,
                    "market_impact": cost_result.market_impact_cost,
                    "total_cost": cost_result.total_cost,
                    "effective_price": cost_result.effective_price
                }
            },
            "positions": new_position
        }
        print(f"Writing to position.jsonl: {json.dumps(transaction_record)}")
        f.write(json.dumps(transaction_record) + "\n")

    # Step 8: Return updated position with cost information
    write_config_value("IF_TRADE", True)
    return {
        "new_position": new_position,
        "transaction_costs": {
            "commission": cost_result.commission_cost,
            "slippage": cost_result.slippage_cost,
            "market_impact": cost_result.market_impact_cost,
            "total_cost": cost_result.total_cost,
            "effective_price": cost_result.effective_price
        },
        "cost_breakdown": cost_result.breakdown,
        "net_proceeds": net_proceeds
    }

if __name__ == "__main__":
    # new_result = buy("AAPL", 1)
    # print(new_result)
    # new_result = sell("AAPL", 1)
    # print(new_result)
    port = int(os.getenv("TRADE_HTTP_PORT", "8002"))
    mcp.run(transport="streamable-http", port=port)

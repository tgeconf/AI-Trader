"""
投资组合优化工具
提供投资组合优化功能，包括均值方差优化、风险平价优化等
"""

from fastmcp import FastMCP
import sys
import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from portfolio_optimization.mean_variance_optimizer import MeanVarianceOptimizer, OptimizationObjective
from portfolio_optimization.risk_parity_optimizer import RiskParityOptimizer
from portfolio_optimization.black_litterman_model import BlackLittermanModel
from tools.general_tools import get_config_value

mcp = FastMCP("PortfolioOptimizationTools")


@mcp.tool()
def optimize_portfolio_mean_variance(
    returns_data: Dict[str, List[float]],
    objective: str = "maximize_sharpe",
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    使用均值方差方法优化投资组合
    
    Args:
        returns_data: 各资产的历史收益率数据，格式为 {"symbol": [returns]}
        objective: 优化目标，可选 "maximize_sharpe", "minimize_variance", "maximize_return", "target_return", "target_volatility"
        target_return: 目标收益率（当objective为"target_return"时使用）
        target_volatility: 目标波动率（当objective为"target_volatility"时使用）
        constraints: 约束条件，如最大权重、最小权重等
        
    Returns:
        优化结果字典
    """
    try:
        # 准备数据
        symbols = list(returns_data.keys())
        returns_matrix = np.array([returns_data[symbol] for symbol in symbols]).T
        
        # 创建优化器
        optimizer = MeanVarianceOptimizer()
        
        # 拟合数据
        optimizer.fit(returns_matrix)
        
        # 根据目标进行优化
        if objective == "maximize_sharpe":
            result = optimizer.maximize_sharpe_ratio()
        elif objective == "minimize_variance":
            result = optimizer.minimize_variance()
        elif objective == "maximize_return":
            result = optimizer.maximize_return()
        elif objective == "target_return" and target_return is not None:
            result = optimizer.target_return_optimization(target_return)
        elif objective == "target_volatility" and target_volatility is not None:
            result = optimizer.target_volatility_optimization(target_volatility)
        else:
            return {"error": f"Invalid objective: {objective}"}
        
        # 构建权重字典
        weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.weights)}
        
        return {
            "optimization_method": "mean_variance",
            "objective": objective,
            "optimal_weights": weights_dict,
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "constraints_satisfied": result.constraints_satisfied
        }
        
    except Exception as e:
        return {"error": f"Portfolio optimization failed: {str(e)}"}


@mcp.tool()
def optimize_portfolio_risk_parity(
    returns_data: Dict[str, List[float]],
    method: str = "equal_risk_contribution"
) -> Dict[str, Any]:
    """
    使用风险平价方法优化投资组合
    
    Args:
        returns_data: 各资产的历史收益率数据
        method: 风险平价方法，可选 "equal_risk_contribution", "inverse_volatility", "minimum_variance"
        
    Returns:
        风险平价优化结果
    """
    try:
        # 准备数据
        symbols = list(returns_data.keys())
        returns_matrix = np.array([returns_data[symbol] for symbol in symbols]).T
        
        # 创建优化器
        optimizer = RiskParityOptimizer()
        
        # 拟合数据
        optimizer.fit(returns_matrix)
        
        # 根据方法进行优化
        if method == "equal_risk_contribution":
            result = optimizer.equal_risk_contribution()
        elif method == "inverse_volatility":
            result = optimizer.inverse_volatility()
        elif method == "minimum_variance":
            result = optimizer.minimum_variance()
        else:
            return {"error": f"Invalid method: {method}"}
        
        # 构建权重字典
        weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.weights)}
        
        return {
            "optimization_method": "risk_parity",
            "method": method,
            "optimal_weights": weights_dict,
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "risk_contributions": result.risk_contributions,
            "constraints_satisfied": result.constraints_satisfied
        }
        
    except Exception as e:
        return {"error": f"Risk parity optimization failed: {str(e)}"}


@mcp.tool()
def black_litterman_optimization(
    returns_data: Dict[str, List[float]],
    market_weights: Dict[str, float],
    views: Dict[str, float],
    view_confidences: Optional[Dict[str, float]] = None,
    risk_aversion: float = 2.5
) -> Dict[str, Any]:
    """
    使用Black-Litterman模型进行投资组合优化
    
    Args:
        returns_data: 各资产的历史收益率数据
        market_weights: 市场均衡权重
        views: 主观观点，格式为 {"symbol": expected_return}
        view_confidences: 观点置信度
        risk_aversion: 风险厌恶系数
        
    Returns:
        Black-Litterman优化结果
    """
    try:
        # 准备数据
        symbols = list(returns_data.keys())
        returns_matrix = np.array([returns_data[symbol] for symbol in symbols]).T
        
        # 创建模型
        model = BlackLittermanModel()
        
        # 拟合数据
        model.fit(returns_matrix)
        
        # 设置市场均衡权重
        market_weights_array = np.array([market_weights.get(symbol, 0.0) for symbol in symbols])
        
        # 设置观点
        views_array = np.array([views.get(symbol, 0.0) for symbol in symbols])
        
        # 设置观点置信度
        if view_confidences:
            confidences_array = np.array([view_confidences.get(symbol, 1.0) for symbol in symbols])
        else:
            confidences_array = np.ones(len(symbols))
        
        # 进行优化
        result = model.optimize(
            market_weights=market_weights_array,
            views=views_array,
            view_confidences=confidences_array,
            risk_aversion=risk_aversion
        )
        
        # 构建权重字典
        weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.weights)}
        
        return {
            "optimization_method": "black_litterman",
            "optimal_weights": weights_dict,
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "posterior_returns": result.posterior_returns,
            "constraints_satisfied": result.constraints_satisfied
        }
        
    except Exception as e:
        return {"error": f"Black-Litterman optimization failed: {str(e)}"}


@mcp.tool()
def calculate_efficient_frontier(
    returns_data: Dict[str, List[float]],
    num_points: int = 50
) -> Dict[str, Any]:
    """
    计算有效前沿
    
    Args:
        returns_data: 各资产的历史收益率数据
        num_points: 有效前沿点数
        
    Returns:
        有效前沿数据
    """
    try:
        # 准备数据
        symbols = list(returns_data.keys())
        returns_matrix = np.array([returns_data[symbol] for symbol in symbols]).T
        
        # 创建优化器
        optimizer = MeanVarianceOptimizer()
        
        # 拟合数据
        optimizer.fit(returns_matrix)
        
        # 计算有效前沿
        frontier = optimizer.calculate_efficient_frontier(num_points=num_points)
        
        # 构建有效前沿数据
        frontier_data = []
        for point in frontier:
            weights_dict = {symbol: weight for symbol, weight in zip(symbols, point.weights)}
            frontier_data.append({
                "return_target": point.return_target,
                "volatility": point.volatility,
                "sharpe_ratio": point.sharpe_ratio,
                "weights": weights_dict
            })
        
        return {
            "efficient_frontier": frontier_data,
            "num_points": num_points,
            "assets": symbols
        }
        
    except Exception as e:
        return {"error": f"Efficient frontier calculation failed: {str(e)}"}


@mcp.tool()
def portfolio_rebalancing_recommendation(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    transaction_costs: float = 0.001,
    min_trade_size: float = 0.01
) -> Dict[str, Any]:
    """
    生成投资组合再平衡建议
    
    Args:
        current_weights: 当前权重
        target_weights: 目标权重
        transaction_costs: 交易成本率
        min_trade_size: 最小交易规模
        
    Returns:
        再平衡建议
    """
    try:
        symbols = list(set(current_weights.keys()) | set(target_weights.keys()))
        
        rebalancing_trades = {}
        total_turnover = 0.0
        estimated_costs = 0.0
        
        for symbol in symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            
            # 计算权重变化
            weight_change = target - current
            
            # 如果变化超过最小交易规模，则建议交易
            if abs(weight_change) >= min_trade_size:
                rebalancing_trades[symbol] = {
                    "current_weight": current,
                    "target_weight": target,
                    "weight_change": weight_change,
                    "action": "buy" if weight_change > 0 else "sell",
                    "trade_size": abs(weight_change)
                }
                
                total_turnover += abs(weight_change)
                estimated_costs += abs(weight_change) * transaction_costs
        
        return {
            "rebalancing_trades": rebalancing_trades,
            "total_turnover": total_turnover,
            "estimated_transaction_costs": estimated_costs,
            "cost_rate": transaction_costs,
            "recommendations": [
                f"Total portfolio turnover: {total_turnover:.2%}",
                f"Estimated transaction costs: {estimated_costs:.4f}",
                f"Number of rebalancing trades: {len(rebalancing_trades)}"
            ]
        }
        
    except Exception as e:
        return {"error": f"Rebalancing recommendation failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("PORTFOLIO_OPTIMIZATION_HTTP_PORT", "8004"))
    mcp.run(transport="streamable-http", port=port)

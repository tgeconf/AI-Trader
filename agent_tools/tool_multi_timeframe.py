"""
多时间框架信号整合工具
提供多时间框架信号生成、整合和冲突解决功能
"""

from fastmcp import FastMCP
import sys
import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from multi_timeframe.timeframe_integration import (
    TimeframeIntegrator, Timeframe, TimeframeSignal, IntegratedSignal
)
from tools.general_tools import get_config_value

mcp = FastMCP("MultiTimeframeTools")


@mcp.tool()
def generate_multi_timeframe_signals(
    symbol: str,
    price_data: Dict[str, List[float]],
    timeframes: List[str] = ["1d", "4h", "1h", "15min"],
    indicators: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    为指定股票生成多时间框架信号
    
    Args:
        symbol: 股票代码
        price_data: 价格数据，包含open, high, low, close, volume
        timeframes: 时间框架列表
        indicators: 技术指标配置
        
    Returns:
        多时间框架信号结果
    """
    try:
        # 创建时间框架整合器
        integrator = TimeframeIntegrator()
        
        # 准备价格数据
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.date_range(end=datetime.now(), periods=len(df))))
        
        # 生成各时间框架信号
        timeframe_signals = []
        
        for tf_str in timeframes:
            try:
                timeframe = Timeframe(tf_str)
                
                # 生成该时间框架的信号
                signal = integrator.generate_timeframe_signal(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    indicators=indicators or {}
                )
                
                timeframe_signals.append(signal)
                
            except Exception as e:
                print(f"Warning: Failed to generate signal for timeframe {tf_str}: {e}")
                continue
        
        # 整合信号
        if timeframe_signals:
            integrated_signal = integrator.integrate_signals(timeframe_signals)
            
            return {
                "symbol": symbol,
                "integrated_signal": {
                    "final_signal": integrated_signal.final_signal,
                    "consensus_level": integrated_signal.consensus_level,
                    "conflict_resolved": integrated_signal.conflict_resolved,
                    "recommendation": integrated_signal.recommendation
                },
                "timeframe_signals": [
                    {
                        "timeframe": signal.timeframe.value,
                        "signal_strength": signal.signal_strength,
                        "confidence": signal.confidence,
                        "indicators": signal.indicators
                    }
                    for signal in timeframe_signals
                ],
                "timeframe_contributions": {
                    tf.value: weight
                    for tf, weight in integrated_signal.timeframe_contributions.items()
                }
            }
        else:
            return {"error": "No valid signals generated for any timeframe"}
        
    except Exception as e:
        return {"error": f"Multi-timeframe signal generation failed: {str(e)}"}


@mcp.tool()
def analyze_timeframe_consistency(
    symbol: str,
    signals_data: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    分析多时间框架信号一致性
    
    Args:
        symbol: 股票代码
        signals_data: 各时间框架信号数据
        
    Returns:
        一致性分析结果
    """
    try:
        integrator = TimeframeIntegrator()
        
        # 创建时间框架信号对象
        timeframe_signals = []
        for tf_str, signal_data in signals_data.items():
            try:
                timeframe = Timeframe(tf_str)
                signal = TimeframeSignal(
                    timeframe=timeframe,
                    symbol=symbol,
                    signal_strength=signal_data.get("signal_strength", 0.0),
                    confidence=signal_data.get("confidence", 0.5),
                    timestamp=pd.Timestamp.now(),
                    indicators=signal_data.get("indicators", {})
                )
                timeframe_signals.append(signal)
            except Exception as e:
                print(f"Warning: Invalid timeframe {tf_str}: {e}")
                continue
        
        if not timeframe_signals:
            return {"error": "No valid timeframe signals provided"}
        
        # 分析一致性
        consistency_result = integrator.analyze_consistency(timeframe_signals)
        
        # 解决冲突
        conflict_resolution = integrator.resolve_conflicts(timeframe_signals)
        
        return {
            "symbol": symbol,
            "consistency_analysis": {
                "overall_consistency": consistency_result.overall_consistency,
                "timeframe_agreement": consistency_result.timeframe_agreement,
                "conflict_count": consistency_result.conflict_count,
                "majority_direction": consistency_result.majority_direction
            },
            "conflict_resolution": {
                "resolved": conflict_resolution.resolved,
                "resolution_method": conflict_resolution.resolution_method,
                "final_recommendation": conflict_resolution.final_recommendation,
                "confidence_after_resolution": conflict_resolution.confidence_after_resolution
            },
            "recommendation": integrator.get_trading_recommendation(timeframe_signals)
        }
        
    except Exception as e:
        return {"error": f"Timeframe consistency analysis failed: {str(e)}"}


@mcp.tool()
def optimize_timeframe_weights(
    symbol: str,
    historical_performance: Dict[str, Dict[str, float]],
    optimization_method: str = "sharpe_maximization"
) -> Dict[str, Any]:
    """
    优化时间框架权重
    
    Args:
        symbol: 股票代码
        historical_performance: 各时间框架历史表现数据
        optimization_method: 优化方法
        
    Returns:
        优化后的时间框架权重
    """
    try:
        integrator = TimeframeIntegrator()
        
        # 准备性能数据
        performance_data = {}
        for tf_str, perf_data in historical_performance.items():
            try:
                timeframe = Timeframe(tf_str)
                performance_data[timeframe] = perf_data
            except Exception as e:
                print(f"Warning: Invalid timeframe {tf_str}: {e}")
                continue
        
        if not performance_data:
            return {"error": "No valid performance data provided"}
        
        # 优化权重
        if optimization_method == "sharpe_maximization":
            optimized_weights = integrator.optimize_weights_sharpe(performance_data)
        elif optimization_method == "equal_risk_contribution":
            optimized_weights = integrator.optimize_weights_risk_parity(performance_data)
        elif optimization_method == "momentum_based":
            optimized_weights = integrator.optimize_weights_momentum(performance_data)
        else:
            return {"error": f"Invalid optimization method: {optimization_method}"}
        
        return {
            "symbol": symbol,
            "optimization_method": optimization_method,
            "optimized_weights": {
                tf.value: weight for tf, weight in optimized_weights.items()
            },
            "performance_metrics": integrator.calculate_performance_metrics(performance_data, optimized_weights)
        }
        
    except Exception as e:
        return {"error": f"Timeframe weight optimization failed: {str(e)}"}


@mcp.tool()
def generate_trading_decision(
    symbol: str,
    multi_timeframe_analysis: Dict[str, Any],
    risk_tolerance: str = "medium",
    position_size: Optional[float] = None
) -> Dict[str, Any]:
    """
    基于多时间框架分析生成交易决策
    
    Args:
        symbol: 股票代码
        multi_timeframe_analysis: 多时间框架分析结果
        risk_tolerance: 风险容忍度 (low, medium, high)
        position_size: 建议头寸规模
        
    Returns:
        交易决策
    """
    try:
        integrator = TimeframeIntegrator()
        
        # 提取信号数据
        integrated_signal = multi_timeframe_analysis.get("integrated_signal", {})
        timeframe_signals = multi_timeframe_analysis.get("timeframe_signals", [])
        
        final_signal = integrated_signal.get("final_signal", 0.0)
        consensus_level = integrated_signal.get("consensus_level", 0.0)
        
        # 根据风险容忍度调整决策
        risk_adjustments = {
            "low": 0.7,    # 保守型，降低信号强度
            "medium": 1.0,  # 中等风险
            "high": 1.3     # 激进型，增强信号强度
        }
        
        risk_factor = risk_adjustments.get(risk_tolerance, 1.0)
        adjusted_signal = final_signal * risk_factor
        
        # 生成交易决策
        decision = {
            "symbol": symbol,
            "action": "BUY" if adjusted_signal > 0.2 else "SELL" if adjusted_signal < -0.2 else "HOLD",
            "signal_strength": adjusted_signal,
            "confidence": min(consensus_level * risk_factor, 1.0),
            "risk_tolerance": risk_tolerance,
            "timeframe_analysis": {
                "final_signal": final_signal,
                "consensus_level": consensus_level,
                "timeframe_count": len(timeframe_signals)
            },
            "recommendation": integrated_signal.get("recommendation", "No clear signal")
        }
        
        # 添加头寸规模建议
        if position_size is not None:
            decision["suggested_position_size"] = position_size
            
            # 根据信号强度调整头寸规模
            if abs(adjusted_signal) > 0.5:
                decision["position_adjustment"] = "increase"
            elif abs(adjusted_signal) < 0.1:
                decision["position_adjustment"] = "decrease"
            else:
                decision["position_adjustment"] = "maintain"
        
        return decision
        
    except Exception as e:
        return {"error": f"Trading decision generation failed: {str(e)}"}


@mcp.tool()
def backtest_multi_timeframe_strategy(
    symbol: str,
    historical_data: Dict[str, List[float]],
    strategy_config: Dict[str, Any],
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    回测多时间框架策略
    
    Args:
        symbol: 股票代码
        historical_data: 历史数据
        strategy_config: 策略配置
        initial_capital: 初始资本
        
    Returns:
        回测结果
    """
    try:
        integrator = TimeframeIntegrator()
        
        # 准备数据
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.date_range(end=datetime.now(), periods=len(df))))
        
        # 运行回测
        backtest_result = integrator.backtest_strategy(
            df=df,
            symbol=symbol,
            strategy_config=strategy_config,
            initial_capital=initial_capital
        )
        
        return {
            "symbol": symbol,
            "backtest_period": {
                "start_date": backtest_result.start_date.strftime("%Y-%m-%d"),
                "end_date": backtest_result.end_date.strftime("%Y-%m-%d"),
                "total_days": backtest_result.total_days
            },
            "performance_metrics": {
                "total_return": backtest_result.total_return,
                "annualized_return": backtest_result.annualized_return,
                "volatility": backtest_result.volatility,
                "sharpe_ratio": backtest_result.sharpe_ratio,
                "max_drawdown": backtest_result.max_drawdown,
                "win_rate": backtest_result.win_rate,
                "profit_factor": backtest_result.profit_factor
            },
            "trading_activity": {
                "total_trades": backtest_result.total_trades,
                "winning_trades": backtest_result.winning_trades,
                "losing_trades": backtest_result.losing_trades,
                "average_trade_return": backtest_result.average_trade_return
            }
        }
        
    except Exception as e:
        return {"error": f"Multi-timeframe strategy backtest failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("MULTI_TIMEFRAME_HTTP_PORT", "8005"))
    mcp.run(transport="streamable-http", port=port)

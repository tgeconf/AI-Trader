"""
市场状态识别工具
提供市场状态识别和预测功能
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

from market_regime.hmm_model import HMMMarketRegime, MarketRegime
from market_regime.volatility_classifier import VolatilityClassifier
from market_regime.trend_analyzer import TrendAnalyzer
from tools.general_tools import get_config_value

mcp = FastMCP("MarketRegimeTools")


@mcp.tool()
def detect_market_regime(
    symbol: str,
    price_data: Dict[str, List[float]],
    method: str = "hmm",
    lookback_period: int = 252
) -> Dict[str, Any]:
    """
    检测市场状态
    
    Args:
        symbol: 股票代码
        price_data: 价格数据
        method: 检测方法 ("hmm", "volatility", "trend")
        lookback_period: 回看期数
        
    Returns:
        市场状态检测结果
    """
    try:
        # 准备数据
        df = pd.DataFrame(price_data)
        
        if 'close' not in df.columns:
            return {"error": "Price data must contain 'close' prices"}
        
        # 计算收益率
        returns = df['close'].pct_change().dropna().values
        
        if len(returns) < lookback_period:
            return {"error": f"Insufficient data. Need at least {lookback_period} periods"}
        
        # 使用最近的数据
        recent_returns = returns[-lookback_period:]
        
        if method == "hmm":
            # 使用HMM方法
            hmm_model = HMMMarketRegime(n_regimes=3)
            hmm_model.fit(recent_returns)
            result = hmm_model.detect_regime(recent_returns)
            
            return {
                "symbol": symbol,
                "method": "hmm",
                "current_regime": result.current_regime.value,
                "regime_probabilities": {
                    regime.value: prob for regime, prob in result.regime_probabilities.items()
                },
                "regime_duration": result.regime_duration,
                "model_converged": result.model_converged,
                "log_likelihood": result.log_likelihood,
                "regime_statistics": hmm_model.get_regime_statistics()
            }
            
        elif method == "volatility":
            # 使用波动率分类方法
            vol_classifier = VolatilityClassifier()
            result = vol_classifier.classify_market_regime(recent_returns)
            
            return {
                "symbol": symbol,
                "method": "volatility",
                "current_regime": result.regime.value,
                "volatility_level": result.volatility_level,
                "volatility_regime": result.volatility_regime,
                "confidence": result.confidence,
                "volatility_metrics": result.volatility_metrics
            }
            
        elif method == "trend":
            # 使用趋势分析方法
            trend_analyzer = TrendAnalyzer()
            result = trend_analyzer.analyze_trend_regime(recent_returns)
            
            return {
                "symbol": symbol,
                "method": "trend",
                "current_regime": result.regime.value,
                "trend_strength": result.trend_strength,
                "trend_direction": result.trend_direction,
                "confidence": result.confidence,
                "trend_metrics": result.trend_metrics
            }
            
        else:
            return {"error": f"Invalid method: {method}"}
        
    except Exception as e:
        return {"error": f"Market regime detection failed: {str(e)}"}


@mcp.tool()
def predict_regime_transition(
    symbol: str,
    current_regime: str,
    historical_returns: List[float],
    prediction_horizon: int = 10
) -> Dict[str, Any]:
    """
    预测市场状态转换
    
    Args:
        symbol: 股票代码
        current_regime: 当前市场状态
        historical_returns: 历史收益率
        prediction_horizon: 预测期数
        
    Returns:
        状态转换预测结果
    """
    try:
        # 映射字符串到枚举
        try:
            current_regime_enum = MarketRegime(current_regime)
        except ValueError:
            return {"error": f"Invalid regime: {current_regime}"}
        
        # 使用HMM模型预测
        hmm_model = HMMMarketRegime(n_regimes=3)
        hmm_model.fit(np.array(historical_returns))
        
        # 预测转换概率
        transition_probs = hmm_model.predict_regime_transition(
            current_regime_enum, 
            lookback_periods=prediction_horizon
        )
        
        return {
            "symbol": symbol,
            "current_regime": current_regime,
            "prediction_horizon": prediction_horizon,
            "transition_probabilities": {
                regime.value: prob for regime, prob in transition_probs.items()
            },
            "most_likely_transition": max(transition_probs.items(), key=lambda x: x[1])[0].value if transition_probs else "unknown",
            "regime_stability": transition_probs.get(current_regime_enum, 0.0)
        }
        
    except Exception as e:
        return {"error": f"Regime transition prediction failed: {str(e)}"}


@mcp.tool()
def get_regime_based_trading_recommendation(
    symbol: str,
    current_regime: str,
    regime_analysis: Dict[str, Any],
    risk_tolerance: str = "medium"
) -> Dict[str, Any]:
    """
    基于市场状态的交易建议
    
    Args:
        symbol: 股票代码
        current_regime: 当前市场状态
        regime_analysis: 市场状态分析结果
        risk_tolerance: 风险容忍度
        
    Returns:
        交易建议
    """
    try:
        # 基于市场状态的策略建议
        regime_strategies = {
            "bull": {
                "action": "BUY",
                "position_size": "large",
                "stop_loss": "wide",
                "time_horizon": "long",
                "risk_level": "medium"
            },
            "bear": {
                "action": "SELL",
                "position_size": "small",
                "stop_loss": "tight",
                "time_horizon": "short",
                "risk_level": "high"
            },
            "sideways": {
                "action": "HOLD",
                "position_size": "medium",
                "stop_loss": "medium",
                "time_horizon": "medium",
                "risk_level": "low"
            },
            "high_vol": {
                "action": "REDUCE",
                "position_size": "small",
                "stop_loss": "tight",
                "time_horizon": "short",
                "risk_level": "high"
            },
            "low_vol": {
                "action": "ACCUMULATE",
                "position_size": "large",
                "stop_loss": "wide",
                "time_horizon": "long",
                "risk_level": "low"
            }
        }
        
        if current_regime not in regime_strategies:
            return {"error": f"Unknown regime: {current_regime}"}
        
        strategy = regime_strategies[current_regime]
        
        # 根据风险容忍度调整
        risk_adjustments = {
            "low": {"position_size": "reduce", "stop_loss": "tighter"},
            "medium": {},
            "high": {"position_size": "increase", "stop_loss": "wider"}
        }
        
        adjustment = risk_adjustments.get(risk_tolerance, {})
        
        return {
            "symbol": symbol,
            "current_regime": current_regime,
            "trading_recommendation": {
                "primary_action": strategy["action"],
                "suggested_position_size": strategy["position_size"],
                "stop_loss_strategy": strategy["stop_loss"],
                "time_horizon": strategy["time_horizon"],
                "risk_level": strategy["risk_level"],
                "risk_adjustment": adjustment
            },
            "regime_confidence": regime_analysis.get("regime_probabilities", {}).get(current_regime, 0.0),
            "additional_notes": self._get_regime_notes(current_regime)
        }
        
    except Exception as e:
        return {"error": f"Trading recommendation generation failed: {str(e)}"}


@mcp.tool()
def _get_regime_notes(regime: str) -> List[str]:
    """获取市场状态说明"""
    notes = {
        "bull": [
            "市场处于上升趋势，适合长期投资",
            "考虑增加成长股和科技股配置",
            "可以使用较宽的止损策略"
        ],
        "bear": [
            "市场处于下降趋势，谨慎操作",
            "考虑防御性资产和现金配置",
            "使用紧密的止损策略控制风险"
        ],
        "sideways": [
            "市场震荡，适合波段交易",
            "关注技术突破和支撑位",
            "控制仓位，避免过度交易"
        ],
        "high_vol": [
            "市场波动性高，风险较大",
            "减少仓位，控制风险暴露",
            "关注市场情绪和新闻事件"
        ],
        "low_vol": [
            "市场波动性低，相对稳定",
            "适合价值投资和长期持有",
            "关注基本面分析和估值"
        ]
    }
    
    return notes.get(regime, ["市场状态分析中..."])


@mcp.tool()
def analyze_multiple_regime_methods(
    symbol: str,
    price_data: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    使用多种方法分析市场状态
    
    Args:
        symbol: 股票代码
        price_data: 价格数据
        
    Returns:
        多方法分析结果
    """
    try:
        # 使用三种方法分析
        hmm_result = detect_market_regime(symbol, price_data, "hmm")
        vol_result = detect_market_regime(symbol, price_data, "volatility")
        trend_result = detect_market_regime(symbol, price_data, "trend")
        
        # 汇总结果
        methods_results = {
            "hmm": hmm_result,
            "volatility": vol_result,
            "trend": trend_result
        }
        
        # 计算一致性
        regimes = [
            hmm_result.get("current_regime") if "error" not in hmm_result else None,
            vol_result.get("current_regime") if "error" not in vol_result else None,
            trend_result.get("current_regime") if "error" not in trend_result else None
        ]
        
        valid_regimes = [r for r in regimes if r is not None]
        
        if valid_regimes:
            consensus_regime = max(set(valid_regimes), key=valid_regimes.count)
            consensus_level = valid_regimes.count(consensus_regime) / len(valid_regimes)
        else:
            consensus_regime = "unknown"
            consensus_level = 0.0
        
        return {
            "symbol": symbol,
            "consensus_regime": consensus_regime,
            "consensus_level": consensus_level,
            "method_results": methods_results,
            "recommendation": "High confidence" if consensus_level >= 0.67 else "Medium confidence" if consensus_level >= 0.33 else "Low confidence"
        }
        
    except Exception as e:
        return {"error": f"Multiple regime analysis failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("MARKET_REGIME_HTTP_PORT", "8006"))
    mcp.run(transport="streamable-http", port=port)

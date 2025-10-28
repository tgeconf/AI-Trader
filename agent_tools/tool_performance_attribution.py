"""
性能归因分析工具
提供投资组合绩效归因分析功能
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

from performance_attribution.brinson_model import BrinsonAttribution
from performance_attribution.multi_factor_attribution import MultiFactorAttribution
from performance_attribution.strategy_effectiveness import StrategyEffectivenessAnalyzer
from tools.general_tools import get_config_value

mcp = FastMCP("PerformanceAttributionTools")


@mcp.tool()
def analyze_brinson_attribution(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    portfolio_returns: Dict[str, float],
    benchmark_returns: Dict[str, float]
) -> Dict[str, Any]:
    """
    执行Brinson绩效归因分析
    
    Args:
        portfolio_weights: 投资组合权重
        benchmark_weights: 基准组合权重
        portfolio_returns: 投资组合收益率
        benchmark_returns: 基准组合收益率
        
    Returns:
        Brinson归因分析结果
    """
    try:
        # 创建Brinson归因模型
        attribution_model = BrinsonAttribution()
        
        # 拟合模型
        attribution_model.fit(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns
        )
        
        # 计算归因分析
        result = attribution_model.calculate_attribution()
        summary = attribution_model.get_attribution_summary()
        
        return {
            "analysis_method": "brinson_attribution",
            "total_excess_return": result.total_excess_return,
            "attribution_components": {
                "allocation_effect": result.allocation_effect,
                "selection_effect": result.selection_effect,
                "interaction_effect": result.interaction_effect
            },
            "relative_contributions": summary["relative_contributions"],
            "primary_driver": summary["primary_driver"],
            "attribution_quality": summary["attribution_quality"],
            "asset_contributions": result.asset_contributions,
            "attribution_breakdown": result.attribution_breakdown.to_dict('records')
        }
        
    except Exception as e:
        return {"error": f"Brinson attribution analysis failed: {str(e)}"}


@mcp.tool()
def analyze_multi_factor_attribution(
    portfolio_returns: List[float],
    factor_returns: Dict[str, List[float]],
    factor_exposures: Dict[str, List[float]],
    method: str = "fm_regression"
) -> Dict[str, Any]:
    """
    执行多因子绩效归因分析
    
    Args:
        portfolio_returns: 投资组合收益率序列
        factor_returns: 因子收益率数据
        factor_exposures: 因子暴露数据
        method: 分析方法 ("fm_regression", "carhart", "ff3")
        
    Returns:
        多因子归因分析结果
    """
    try:
        # 创建多因子归因模型
        factor_model = MultiFactorAttribution()
        
        # 拟合模型
        factor_model.fit(
            portfolio_returns=np.array(portfolio_returns),
            factor_returns={k: np.array(v) for k, v in factor_returns.items()},
            factor_exposures={k: np.array(v) for k, v in factor_exposures.items()}
        )
        
        # 计算归因分析
        if method == "fm_regression":
            result = factor_model.fama_macbeth_regression()
        elif method == "carhart":
            result = factor_model.carhart_four_factor()
        elif method == "ff3":
            result = factor_model.fama_french_three_factor()
        else:
            return {"error": f"Invalid method: {method}"}
        
        return {
            "analysis_method": f"multi_factor_{method}",
            "factor_contributions": result.factor_contributions,
            "alpha": result.alpha,
            "r_squared": result.r_squared,
            "factor_significance": result.factor_significance,
            "residual_analysis": result.residual_analysis,
            "model_summary": result.model_summary
        }
        
    except Exception as e:
        return {"error": f"Multi-factor attribution analysis failed: {str(e)}"}


@mcp.tool()
def analyze_strategy_effectiveness(
    strategy_returns: List[float],
    benchmark_returns: List[float],
    risk_free_rate: float = 0.02,
    analysis_period: str = "monthly"
) -> Dict[str, Any]:
    """
    分析策略有效性
    
    Args:
        strategy_returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 无风险利率
        analysis_period: 分析周期
        
    Returns:
        策略有效性分析结果
    """
    try:
        # 创建策略有效性分析器
        effectiveness_analyzer = StrategyEffectivenessAnalyzer()
        
        # 分析策略有效性
        result = effectiveness_analyzer.analyze_strategy(
            strategy_returns=np.array(strategy_returns),
            benchmark_returns=np.array(benchmark_returns),
            risk_free_rate=risk_free_rate,
            period=analysis_period
        )
        
        return {
            "analysis_method": "strategy_effectiveness",
            "performance_metrics": {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "calmar_ratio": result.calmar_ratio,
                "information_ratio": result.information_ratio,
                "alpha": result.alpha,
                "beta": result.beta
            },
            "risk_metrics": {
                "var_95": result.var_95,
                "cvar_95": result.cvar_95,
                "skewness": result.skewness,
                "kurtosis": result.kurtosis
            },
            "statistical_tests": result.statistical_tests,
            "strategy_quality": result.strategy_quality
        }
        
    except Exception as e:
        return {"error": f"Strategy effectiveness analysis failed: {str(e)}"}


@mcp.tool()
def generate_performance_report(
    portfolio_data: Dict[str, Any],
    benchmark_data: Dict[str, Any],
    analysis_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    生成综合绩效报告
    
    Args:
        portfolio_data: 投资组合数据
        benchmark_data: 基准数据
        analysis_config: 分析配置
        
    Returns:
        综合绩效报告
    """
    try:
        report = {
            "report_timestamp": pd.Timestamp.now().isoformat(),
            "analysis_period": analysis_config.get("period", "custom"),
            "performance_summary": {},
            "attribution_analysis": {},
            "risk_analysis": {},
            "recommendations": []
        }
        
        # 提取数据
        portfolio_returns = portfolio_data.get("returns", [])
        benchmark_returns = benchmark_data.get("returns", [])
        portfolio_weights = portfolio_data.get("weights", {})
        benchmark_weights = benchmark_data.get("weights", {})
        
        # 策略有效性分析
        if len(portfolio_returns) > 0 and len(benchmark_returns) > 0:
            effectiveness_result = analyze_strategy_effectiveness(
                strategy_returns=portfolio_returns,
                benchmark_returns=benchmark_returns,
                risk_free_rate=analysis_config.get("risk_free_rate", 0.02)
            )
            
            if "error" not in effectiveness_result:
                report["performance_summary"] = effectiveness_result["performance_metrics"]
                report["risk_analysis"] = effectiveness_result["risk_metrics"]
        
        # Brinson归因分析
        if portfolio_weights and benchmark_weights:
            portfolio_asset_returns = portfolio_data.get("asset_returns", {})
            benchmark_asset_returns = benchmark_data.get("asset_returns", {})
            
            if portfolio_asset_returns and benchmark_asset_returns:
                attribution_result = analyze_brinson_attribution(
                    portfolio_weights=portfolio_weights,
                    benchmark_weights=benchmark_weights,
                    portfolio_returns=portfolio_asset_returns,
                    benchmark_returns=benchmark_asset_returns
                )
                
                if "error" not in attribution_result:
                    report["attribution_analysis"] = attribution_result
        
        # 生成建议
        report["recommendations"] = _generate_performance_recommendations(report)
        
        return report
        
    except Exception as e:
        return {"error": f"Performance report generation failed: {str(e)}"}


@mcp.tool()
def _generate_performance_recommendations(report: Dict[str, Any]) -> List[str]:
    """生成绩效改进建议"""
    recommendations = []
    
    performance = report.get("performance_summary", {})
    attribution = report.get("attribution_analysis", {})
    
    # 基于夏普比率的建议
    sharpe_ratio = performance.get("sharpe_ratio", 0)
    if sharpe_ratio < 0.5:
        recommendations.append("夏普比率较低，建议优化风险调整后收益")
    elif sharpe_ratio > 1.5:
        recommendations.append("夏普比率优秀，继续保持当前策略")
    
    # 基于最大回撤的建议
    max_drawdown = performance.get("max_drawdown", 0)
    if abs(max_drawdown) > 0.15:
        recommendations.append("最大回撤较大，建议加强风险管理")
    
    # 基于归因分析的建议
    attribution_components = attribution.get("attribution_components", {})
    allocation_effect = attribution_components.get("allocation_effect", 0)
    selection_effect = attribution_components.get("selection_effect", 0)
    
    if allocation_effect < 0:
        recommendations.append("资产配置效应为负，建议调整资产配置权重")
    
    if selection_effect < 0:
        recommendations.append("证券选择效应为负，建议优化选股策略")
    
    # 基于信息比率的建议
    information_ratio = performance.get("information_ratio", 0)
    if information_ratio < 0:
        recommendations.append("信息比率为负，策略未能跑赢基准")
    
    # 默认建议
    if not recommendations:
        recommendations.append("策略表现良好，建议持续监控和优化")
    
    return recommendations


@mcp.tool()
def compare_multiple_strategies(
    strategies_data: Dict[str, Dict[str, Any]],
    benchmark_returns: List[float],
    comparison_metrics: List[str] = ["sharpe_ratio", "max_drawdown", "information_ratio"]
) -> Dict[str, Any]:
    """
    比较多个策略的表现
    
    Args:
        strategies_data: 多个策略的数据
        benchmark_returns: 基准收益率
        comparison_metrics: 比较指标
        
    Returns:
        策略比较结果
    """
    try:
        comparison_results = {}
        
        for strategy_name, strategy_data in strategies_data.items():
            strategy_returns = strategy_data.get("returns", [])
            
            if len(strategy_returns) > 0:
                effectiveness_result = analyze_strategy_effectiveness(
                    strategy_returns=strategy_returns,
                    benchmark_returns=benchmark_returns
                )
                
                if "error" not in effectiveness_result:
                    comparison_results[strategy_name] = {
                        "performance_metrics": effectiveness_result["performance_metrics"],
                        "strategy_quality": effectiveness_result.get("strategy_quality", "unknown")
                    }
        
        # 排名分析
        rankings = {}
        for metric in comparison_metrics:
            strategy_scores = {}
            for strategy_name, result in comparison_results.items():
                score = result["performance_metrics"].get(metric, 0)
                strategy_scores[strategy_name] = score
            
            # 根据指标特性排序（夏普比率越高越好，最大回撤越小越好）
            if metric == "max_drawdown":
                sorted_strategies = sorted(strategy_scores.items(), key=lambda x: abs(x[1]))
            else:
                sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            
            rankings[metric] = {strategy: rank + 1 for rank, (strategy, _) in enumerate(sorted_strategies)}
        
        # 综合排名
        overall_rankings = {}
        for strategy_name in comparison_results.keys():
            total_rank = sum(rankings[metric].get(strategy_name, 0) for metric in comparison_metrics)
            overall_rankings[strategy_name] = total_rank
        
        sorted_overall = sorted(overall_rankings.items(), key=lambda x: x[1])
        
        return {
            "comparison_results": comparison_results,
            "metric_rankings": rankings,
            "overall_rankings": {strategy: rank + 1 for rank, (strategy, _) in enumerate(sorted_overall)},
            "best_strategy": sorted_overall[0][0] if sorted_overall else "none",
            "comparison_metrics": comparison_metrics
        }
        
    except Exception as e:
        return {"error": f"Strategy comparison failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("PERFORMANCE_ATTRIBUTION_HTTP_PORT", "8007"))
    mcp.run(transport="streamable-http", port=port)

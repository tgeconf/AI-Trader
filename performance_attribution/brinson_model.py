"""
Brinson绩效归因模型
实现经典的Brinson归因分析，分解投资组合超额收益的来源
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class AttributionComponent(Enum):
    """归因成分枚举"""
    ALLOCATION = "allocation"      # 资产配置效应
    SELECTION = "selection"        # 证券选择效应
    INTERACTION = "interaction"    # 交互效应
    TOTAL = "total"               # 总效应


@dataclass
class AttributionResult:
    """归因分析结果"""
    total_excess_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    component_contributions: Dict[str, float]
    asset_contributions: Dict[str, Dict[str, float]]
    attribution_breakdown: pd.DataFrame


class BrinsonAttribution:
    """
    Brinson绩效归因模型
    
    实现经典的Brinson归因分析，将投资组合超额收益分解为：
    - 资产配置效应 (Allocation Effect)
    - 证券选择效应 (Selection Effect)  
    - 交互效应 (Interaction Effect)
    """
    
    def __init__(self):
        """初始化Brinson归因模型"""
        self.portfolio_weights = None
        self.benchmark_weights = None
        self.portfolio_returns = None
        self.benchmark_returns = None
        
    def fit(self, portfolio_weights: Dict[str, float], 
            benchmark_weights: Dict[str, float],
            portfolio_returns: Dict[str, float],
            benchmark_returns: Dict[str, float]) -> None:
        """
        拟合归因模型
        
        Args:
            portfolio_weights: 投资组合权重
            benchmark_weights: 基准组合权重
            portfolio_returns: 投资组合收益率
            benchmark_returns: 基准组合收益率
        """
        # 确保所有字典的键一致
        assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys()) | \
                 set(portfolio_returns.keys()) | set(benchmark_returns.keys())
        
        self.portfolio_weights = {asset: portfolio_weights.get(asset, 0.0) for asset in assets}
        self.benchmark_weights = {asset: benchmark_weights.get(asset, 0.0) for asset in assets}
        self.portfolio_returns = {asset: portfolio_returns.get(asset, 0.0) for asset in assets}
        self.benchmark_returns = {asset: benchmark_returns.get(asset, 0.0) for asset in assets}
        
    def calculate_attribution(self) -> AttributionResult:
        """
        计算Brinson归因分析
        
        Returns:
            归因分析结果
        """
        if any(x is None for x in [self.portfolio_weights, self.benchmark_weights, 
                                  self.portfolio_returns, self.benchmark_returns]):
            raise ValueError("Model must be fitted before attribution calculation")
        
        assets = list(self.portfolio_weights.keys())
        
        # 计算总超额收益
        portfolio_total_return = self._calculate_total_return(self.portfolio_weights, self.portfolio_returns)
        benchmark_total_return = self._calculate_total_return(self.benchmark_weights, self.benchmark_returns)
        total_excess_return = portfolio_total_return - benchmark_total_return
        
        # 计算各成分效应
        allocation_effect = self._calculate_allocation_effect()
        selection_effect = self._calculate_selection_effect()
        interaction_effect = self._calculate_interaction_effect()
        
        # 验证归因分解
        attribution_sum = allocation_effect + selection_effect + interaction_effect
        if not np.isclose(attribution_sum, total_excess_return, rtol=1e-6):
            print(f"Warning: Attribution decomposition doesn't sum to total excess return: {attribution_sum} vs {total_excess_return}")
        
        # 计算各资产贡献
        asset_contributions = self._calculate_asset_contributions()
        
        # 构建归因分解表
        attribution_breakdown = self._build_attribution_breakdown()
        
        return AttributionResult(
            total_excess_return=total_excess_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            component_contributions={
                "allocation": allocation_effect,
                "selection": selection_effect,
                "interaction": interaction_effect,
                "total": total_excess_return
            },
            asset_contributions=asset_contributions,
            attribution_breakdown=attribution_breakdown
        )
    
    def _calculate_total_return(self, weights: Dict[str, float], returns: Dict[str, float]) -> float:
        """计算组合总收益率"""
        total_return = 0.0
        for asset in weights:
            total_return += weights[asset] * returns[asset]
        return total_return
    
    def _calculate_allocation_effect(self) -> float:
        """计算资产配置效应"""
        allocation_effect = 0.0
        benchmark_total_return = self._calculate_total_return(self.benchmark_weights, self.benchmark_returns)
        
        for asset in self.portfolio_weights:
            weight_diff = self.portfolio_weights[asset] - self.benchmark_weights[asset]
            benchmark_asset_return = self.benchmark_returns[asset]
            allocation_effect += weight_diff * (benchmark_asset_return - benchmark_total_return)
        
        return allocation_effect
    
    def _calculate_selection_effect(self) -> float:
        """计算证券选择效应"""
        selection_effect = 0.0
        
        for asset in self.portfolio_weights:
            return_diff = self.portfolio_returns[asset] - self.benchmark_returns[asset]
            selection_effect += self.benchmark_weights[asset] * return_diff
        
        return selection_effect
    
    def _calculate_interaction_effect(self) -> float:
        """计算交互效应"""
        interaction_effect = 0.0
        
        for asset in self.portfolio_weights:
            weight_diff = self.portfolio_weights[asset] - self.benchmark_weights[asset]
            return_diff = self.portfolio_returns[asset] - self.benchmark_returns[asset]
            interaction_effect += weight_diff * return_diff
        
        return interaction_effect
    
    def _calculate_asset_contributions(self) -> Dict[str, Dict[str, float]]:
        """计算各资产的归因贡献"""
        asset_contributions = {}
        benchmark_total_return = self._calculate_total_return(self.benchmark_weights, self.benchmark_returns)
        
        for asset in self.portfolio_weights:
            # 资产配置贡献
            weight_diff = self.portfolio_weights[asset] - self.benchmark_weights[asset]
            benchmark_asset_return = self.benchmark_returns[asset]
            allocation_contribution = weight_diff * (benchmark_asset_return - benchmark_total_return)
            
            # 证券选择贡献
            return_diff = self.portfolio_returns[asset] - self.benchmark_returns[asset]
            selection_contribution = self.benchmark_weights[asset] * return_diff
            
            # 交互贡献
            interaction_contribution = weight_diff * return_diff
            
            # 总贡献
            total_contribution = allocation_contribution + selection_contribution + interaction_contribution
            
            asset_contributions[asset] = {
                "allocation": allocation_contribution,
                "selection": selection_contribution,
                "interaction": interaction_contribution,
                "total": total_contribution
            }
        
        return asset_contributions
    
    def _build_attribution_breakdown(self) -> pd.DataFrame:
        """构建归因分解表"""
        data = []
        
        for asset in self.portfolio_weights:
            row = {
                'asset': asset,
                'portfolio_weight': self.portfolio_weights[asset],
                'benchmark_weight': self.benchmark_weights[asset],
                'portfolio_return': self.portfolio_returns[asset],
                'benchmark_return': self.benchmark_returns[asset],
                'weight_diff': self.portfolio_weights[asset] - self.benchmark_weights[asset],
                'return_diff': self.portfolio_returns[asset] - self.benchmark_returns[asset]
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 计算各成分贡献
        benchmark_total_return = self._calculate_total_return(self.benchmark_weights, self.benchmark_returns)
        df['allocation_contribution'] = df['weight_diff'] * (df['benchmark_return'] - benchmark_total_return)
        df['selection_contribution'] = df['benchmark_weight'] * df['return_diff']
        df['interaction_contribution'] = df['weight_diff'] * df['return_diff']
        df['total_contribution'] = df['allocation_contribution'] + df['selection_contribution'] + df['interaction_contribution']
        
        return df
    
    def get_attribution_summary(self) -> Dict[str, Any]:
        """获取归因分析摘要"""
        result = self.calculate_attribution()
        
        # 计算各成分的相对贡献
        total_abs = abs(result.allocation_effect) + abs(result.selection_effect) + abs(result.interaction_effect)
        
        if total_abs > 0:
            allocation_pct = abs(result.allocation_effect) / total_abs * 100
            selection_pct = abs(result.selection_effect) / total_abs * 100
            interaction_pct = abs(result.interaction_effect) / total_abs * 100
        else:
            allocation_pct = selection_pct = interaction_pct = 0.0
        
        return {
            "total_excess_return": result.total_excess_return,
            "allocation_effect": result.allocation_effect,
            "selection_effect": result.selection_effect,
            "interaction_effect": result.interaction_effect,
            "relative_contributions": {
                "allocation": allocation_pct,
                "selection": selection_pct,
                "interaction": interaction_pct
            },
            "primary_driver": self._identify_primary_driver(result),
            "attribution_quality": self._assess_attribution_quality(result)
        }
    
    def _identify_primary_driver(self, result: AttributionResult) -> str:
        """识别主要驱动因素"""
        effects = {
            "allocation": abs(result.allocation_effect),
            "selection": abs(result.selection_effect),
            "interaction": abs(result.interaction_effect)
        }
        
        primary_driver = max(effects.items(), key=lambda x: x[1])[0]
        
        # 如果主要驱动因素贡献超过50%，则认为是主要驱动
        total_abs = sum(effects.values())
        if total_abs > 0 and effects[primary_driver] / total_abs > 0.5:
            return f"strong_{primary_driver}"
        else:
            return f"mixed_{primary_driver}"
    
    def _assess_attribution_quality(self, result: AttributionResult) -> str:
        """评估归因质量"""
        attribution_sum = result.allocation_effect + result.selection_effect + result.interaction_effect
        attribution_error = abs(attribution_sum - result.total_excess_return)
        
        if attribution_error < 1e-6:
            return "excellent"
        elif attribution_error < 1e-4:
            return "good"
        elif attribution_error < 1e-2:
            return "fair"
        else:
            return "poor"

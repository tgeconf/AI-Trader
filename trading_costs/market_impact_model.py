"""
市场冲击模型
实现交易对市场价格的影响计算，包括临时冲击和永久冲击
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ImpactType(Enum):
    """市场冲击类型枚举"""
    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    COMBINED = "combined"


@dataclass
class MarketImpactResult:
    """市场冲击计算结果"""
    temporary_impact: float
    permanent_impact: float
    total_impact: float
    impact_type: ImpactType
    trade_volume: float
    original_price: float
    impacted_price: float
    impact_rate: float


class MarketImpactModel:
    """
    市场冲击模型类
    
    实现市场冲击计算：
    - 临时冲击：交易执行期间的暂时价格影响
    - 永久冲击：交易导致的永久性价格变化
    - 综合冲击：临时和永久冲击的组合
    """
    
    def __init__(self, temporary_impact_coef: float = 0.01, permanent_impact_coef: float = 0.005,
                 volume_sensitivity: float = 0.0001, volatility_sensitivity: float = 0.5,
                 liquidity_sensitivity: float = 0.00001):
        """
        初始化市场冲击模型
        
        Args:
            temporary_impact_coef: 临时冲击系数
            permanent_impact_coef: 永久冲击系数
            volume_sensitivity: 成交量敏感度
            volatility_sensitivity: 波动率敏感度
            liquidity_sensitivity: 流动性敏感度
        """
        self.temporary_impact_coef = temporary_impact_coef
        self.permanent_impact_coef = permanent_impact_coef
        self.volume_sensitivity = volume_sensitivity
        self.volatility_sensitivity = volatility_sensitivity
        self.liquidity_sensitivity = liquidity_sensitivity
        
    def calculate_temporary_impact(self, original_price: float, trade_volume: float,
                                 average_volume: float, volatility: float) -> float:
        """
        计算临时市场冲击
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            average_volume: 平均成交量
            volatility: 波动率
            
        Returns:
            float: 临时冲击金额
        """
        if average_volume <= 0:
            return 0.0
            
        # 成交量占比
        volume_ratio = trade_volume / average_volume
        
        # 波动率调整
        volatility_adjustment = 1 + self.volatility_sensitivity * volatility
        
        # 临时冲击计算
        temporary_impact = (
            self.temporary_impact_coef * 
            volume_ratio * 
            volatility_adjustment * 
            original_price
        )
        
        return temporary_impact
    
    def calculate_permanent_impact(self, original_price: float, trade_volume: float,
                                 market_cap: float, liquidity: float) -> float:
        """
        计算永久市场冲击
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            market_cap: 市值
            liquidity: 流动性指标
            
        Returns:
            float: 永久冲击金额
        """
        if market_cap <= 0:
            return 0.0
            
        # 交易量相对于市值的比例
        trade_to_market_ratio = (trade_volume * original_price) / market_cap
        
        # 流动性调整
        liquidity_adjustment = 1 / (1 + self.liquidity_sensitivity * liquidity) if liquidity > 0 else 1.0
        
        # 永久冲击计算
        permanent_impact = (
            self.permanent_impact_coef * 
            trade_to_market_ratio * 
            liquidity_adjustment * 
            original_price
        )
        
        return permanent_impact
    
    def calculate_combined_impact(self, original_price: float, trade_volume: float,
                                market_data: Dict) -> MarketImpactResult:
        """
        计算综合市场冲击
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            market_data: 市场数据
            
        Returns:
            MarketImpactResult: 市场冲击计算结果
        """
        avg_volume = market_data.get('average_volume', 1000000)
        volatility = market_data.get('volatility', 0.2)
        market_cap = market_data.get('market_cap', 1000000000)
        liquidity = market_data.get('liquidity', 1000000)
        
        # 计算临时冲击
        temporary_impact = self.calculate_temporary_impact(
            original_price, trade_volume, avg_volume, volatility
        )
        
        # 计算永久冲击
        permanent_impact = self.calculate_permanent_impact(
            original_price, trade_volume, market_cap, liquidity
        )
        
        # 总冲击
        total_impact = temporary_impact + permanent_impact
        impacted_price = original_price + total_impact
        impact_rate = total_impact / original_price if original_price > 0 else 0
        
        return MarketImpactResult(
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_impact=total_impact,
            impact_type=ImpactType.COMBINED,
            trade_volume=trade_volume,
            original_price=original_price,
            impacted_price=impacted_price,
            impact_rate=impact_rate
        )
    
    def calculate_market_impact(self, original_price: float, trade_volume: float,
                              market_data: Dict, impact_type: ImpactType = ImpactType.COMBINED) -> MarketImpactResult:
        """
        计算市场冲击
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            market_data: 市场数据
            impact_type: 冲击类型
            
        Returns:
            MarketImpactResult: 市场冲击计算结果
        """
        if impact_type == ImpactType.TEMPORARY:
            avg_volume = market_data.get('average_volume', 1000000)
            volatility = market_data.get('volatility', 0.2)
            temporary_impact = self.calculate_temporary_impact(
                original_price, trade_volume, avg_volume, volatility
            )
            
            return MarketImpactResult(
                temporary_impact=temporary_impact,
                permanent_impact=0.0,
                total_impact=temporary_impact,
                impact_type=ImpactType.TEMPORARY,
                trade_volume=trade_volume,
                original_price=original_price,
                impacted_price=original_price + temporary_impact,
                impact_rate=temporary_impact / original_price if original_price > 0 else 0
            )
            
        elif impact_type == ImpactType.PERMANENT:
            market_cap = market_data.get('market_cap', 1000000000)
            liquidity = market_data.get('liquidity', 1000000)
            permanent_impact = self.calculate_permanent_impact(
                original_price, trade_volume, market_cap, liquidity
            )
            
            return MarketImpactResult(
                temporary_impact=0.0,
                permanent_impact=permanent_impact,
                total_impact=permanent_impact,
                impact_type=ImpactType.PERMANENT,
                trade_volume=trade_volume,
                original_price=original_price,
                impacted_price=original_price + permanent_impact,
                impact_rate=permanent_impact / original_price if original_price > 0 else 0
            )
            
        else:  # COMBINED
            return self.calculate_combined_impact(original_price, trade_volume, market_data)
    
    def optimize_trade_schedule(self, total_volume: float, original_price: float,
                              market_data: Dict, time_horizon: int = 10) -> Dict[str, float]:
        """
        优化交易计划以最小化市场冲击
        
        Args:
            total_volume: 总交易量
            original_price: 原始价格
            market_data: 市场数据
            time_horizon: 时间范围（分钟）
            
        Returns:
            Dict[str, float]: 优化结果
        """
        # 尝试不同的交易速度
        speed_strategies = []
        
        for minutes in range(1, time_horizon + 1):
            volume_per_minute = total_volume / minutes
            total_impact = 0.0
            current_price = original_price
            
            for minute in range(minutes):
                # 每分钟后更新价格（考虑累积冲击）
                impact_result = self.calculate_market_impact(
                    current_price, volume_per_minute, market_data
                )
                total_impact += impact_result.total_impact * volume_per_minute
                current_price = impact_result.impacted_price
            
            speed_strategies.append({
                'execution_minutes': minutes,
                'volume_per_minute': volume_per_minute,
                'total_impact': total_impact,
                'final_price': current_price,
                'avg_impact_per_minute': total_impact / minutes,
                'impact_rate': total_impact / (total_volume * original_price)
            })
        
        # 找到总冲击最小的策略
        best_strategy = min(speed_strategies, key=lambda x: x['total_impact'])
        
        return {
            'optimal_execution_minutes': best_strategy['execution_minutes'],
            'optimal_volume_per_minute': best_strategy['volume_per_minute'],
            'min_total_impact': best_strategy['total_impact'],
            'final_price': best_strategy['final_price'],
            'impact_reduction_vs_instant': (
                speed_strategies[0]['total_impact'] - best_strategy['total_impact']
            ) / speed_strategies[0]['total_impact'] if speed_strategies[0]['total_impact'] > 0 else 0,
            'all_strategies': speed_strategies
        }
    
    def calculate_portfolio_impact(self, trades: List[Dict]) -> Dict[str, float]:
        """
        计算投资组合层面的市场冲击
        
        Args:
            trades: 交易列表
            
        Returns:
            Dict[str, float]: 投资组合冲击统计
        """
        total_impact = 0.0
        impact_by_symbol = {}
        total_trade_value = 0.0
        
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            price = trade.get('price', 0)
            volume = trade.get('volume', 0)
            market_data = trade.get('market_data', {})
            
            impact_result = self.calculate_market_impact(price, volume, market_data)
            trade_impact = impact_result.total_impact * volume
            total_impact += trade_impact
            total_trade_value += price * volume
            
            if symbol in impact_by_symbol:
                impact_by_symbol[symbol] += trade_impact
            else:
                impact_by_symbol[symbol] = trade_impact
        
        return {
            'total_market_impact': total_impact,
            'impact_by_symbol': impact_by_symbol,
            'avg_impact_per_trade': total_impact / len(trades) if trades else 0,
            'total_trade_value': total_trade_value,
            'overall_impact_rate': total_impact / total_trade_value if total_trade_value > 0 else 0,
            'impact_as_percentage_of_trade': (total_impact / total_trade_value) * 100 if total_trade_value > 0 else 0
        }
    
    def estimate_price_reversion(self, temporary_impact: float, time_horizon: int,
                               volatility: float) -> Dict[str, float]:
        """
        估计价格回归
        
        Args:
            temporary_impact: 临时冲击
            time_horizon: 时间范围（分钟）
            volatility: 波动率
            
        Returns:
            Dict[str, float]: 价格回归估计
        """
        # 价格回归模型（指数衰减）
        reversion_half_life = 5  # 半衰期（分钟）
        reversion_rate = np.log(2) / reversion_half_life
        
        # 计算时间范围内的价格回归
        remaining_impact = temporary_impact * np.exp(-reversion_rate * time_horizon)
        reverted_impact = temporary_impact - remaining_impact
        reversion_percentage = reverted_impact / temporary_impact if temporary_impact > 0 else 0
        
        # 考虑波动率的影响
        volatility_adjustment = 1 - (volatility * 0.1)  # 高波动率降低回归效果
        adjusted_reversion = reversion_percentage * volatility_adjustment
        
        return {
            'initial_temporary_impact': temporary_impact,
            'remaining_impact_after_reversion': remaining_impact,
            'reverted_impact': reverted_impact,
            'reversion_percentage': reversion_percentage,
            'volatility_adjusted_reversion': adjusted_reversion,
            'effective_temporary_impact': temporary_impact * (1 - adjusted_reversion)
        }

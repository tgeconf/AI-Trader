"""
滑点模型
实现固定滑点、比例滑点、流动性调整滑点等多种滑点计算方式
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SlippageType(Enum):
    """滑点类型枚举"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_ADJUSTED = "volume_adjusted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    LIQUIDITY_ADJUSTED = "liquidity_adjusted"


@dataclass
class SlippageResult:
    """滑点计算结果"""
    slippage_amount: float
    slippage_type: SlippageType
    trade_amount: float
    slippage_rate: float
    effective_price: float
    original_price: float


class SlippageModel:
    """
    滑点模型类
    
    实现多种滑点计算方式：
    - 固定滑点
    - 比例滑点
    - 成交量调整滑点
    - 波动率调整滑点
    - 流动性调整滑点
    """
    
    def __init__(self, slippage_type: SlippageType = SlippageType.PERCENTAGE,
                 fixed_slippage: float = 0.01, percentage_slippage: float = 0.001,
                 volume_sensitivity: float = 0.0001, volatility_sensitivity: float = 0.5,
                 liquidity_sensitivity: float = 0.00001):
        """
        初始化滑点模型
        
        Args:
            slippage_type: 滑点类型
            fixed_slippage: 固定滑点金额
            percentage_slippage: 比例滑点率
            volume_sensitivity: 成交量敏感度
            volatility_sensitivity: 波动率敏感度
            liquidity_sensitivity: 流动性敏感度
        """
        self.slippage_type = slippage_type
        self.fixed_slippage = fixed_slippage
        self.percentage_slippage = percentage_slippage
        self.volume_sensitivity = volume_sensitivity
        self.volatility_sensitivity = volatility_sensitivity
        self.liquidity_sensitivity = liquidity_sensitivity
        
    def fixed_slippage_model(self, original_price: float, trade_volume: float) -> SlippageResult:
        """
        固定滑点模型
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            
        Returns:
            SlippageResult: 滑点计算结果
        """
        slippage_amount = self.fixed_slippage
        effective_price = original_price + slippage_amount
        slippage_rate = slippage_amount / original_price if original_price > 0 else 0
        
        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_type=SlippageType.FIXED,
            trade_amount=trade_volume * original_price,
            slippage_rate=slippage_rate,
            effective_price=effective_price,
            original_price=original_price
        )
    
    def percentage_slippage_model(self, original_price: float, trade_volume: float) -> SlippageResult:
        """
        比例滑点模型
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            
        Returns:
            SlippageResult: 滑点计算结果
        """
        slippage_amount = original_price * self.percentage_slippage
        effective_price = original_price + slippage_amount
        slippage_rate = self.percentage_slippage
        
        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_type=SlippageType.PERCENTAGE,
            trade_amount=trade_volume * original_price,
            slippage_rate=slippage_rate,
            effective_price=effective_price,
            original_price=original_price
        )
    
    def volume_adjusted_slippage(self, original_price: float, trade_volume: float,
                               average_volume: float) -> SlippageResult:
        """
        成交量调整滑点
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            average_volume: 平均成交量
            
        Returns:
            SlippageResult: 滑点计算结果
        """
        if average_volume <= 0:
            return self.percentage_slippage_model(original_price, trade_volume)
            
        # 成交量占比
        volume_ratio = trade_volume / average_volume
        
        # 滑点随成交量增加而增加
        adjusted_slippage = self.percentage_slippage * (1 + self.volume_sensitivity * volume_ratio)
        
        slippage_amount = original_price * adjusted_slippage
        effective_price = original_price + slippage_amount
        
        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_type=SlippageType.VOLUME_ADJUSTED,
            trade_amount=trade_volume * original_price,
            slippage_rate=adjusted_slippage,
            effective_price=effective_price,
            original_price=original_price
        )
    
    def volatility_adjusted_slippage(self, original_price: float, trade_volume: float,
                                   volatility: float) -> SlippageResult:
        """
        波动率调整滑点
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            volatility: 波动率
            
        Returns:
            SlippageResult: 滑点计算结果
        """
        # 波动率调整
        volatility_adjustment = 1 + self.volatility_sensitivity * volatility
        adjusted_slippage = self.percentage_slippage * volatility_adjustment
        
        slippage_amount = original_price * adjusted_slippage
        effective_price = original_price + slippage_amount
        
        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_type=SlippageType.VOLATILITY_ADJUSTED,
            trade_amount=trade_volume * original_price,
            slippage_rate=adjusted_slippage,
            effective_price=effective_price,
            original_price=original_price
        )
    
    def liquidity_adjusted_slippage(self, original_price: float, trade_volume: float,
                                  bid_ask_spread: float, market_depth: float) -> SlippageResult:
        """
        流动性调整滑点
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            bid_ask_spread: 买卖价差
            market_depth: 市场深度
            
        Returns:
            SlippageResult: 滑点计算结果
        """
        # 基于买卖价差和市场深度的滑点计算
        spread_component = bid_ask_spread / 2  # 一半的买卖价差
        
        # 市场深度调整
        if market_depth > 0:
            depth_ratio = trade_volume / market_depth
            depth_component = self.liquidity_sensitivity * depth_ratio * original_price
        else:
            depth_component = self.percentage_slippage * original_price
        
        slippage_amount = spread_component + depth_component
        effective_price = original_price + slippage_amount
        slippage_rate = slippage_amount / original_price if original_price > 0 else 0
        
        return SlippageResult(
            slippage_amount=slippage_amount,
            slippage_type=SlippageType.LIQUIDITY_ADJUSTED,
            trade_amount=trade_volume * original_price,
            slippage_rate=slippage_rate,
            effective_price=effective_price,
            original_price=original_price
        )
    
    def calculate_slippage(self, original_price: float, trade_volume: float,
                          market_data: Optional[Dict] = None,
                          slippage_type: Optional[SlippageType] = None) -> SlippageResult:
        """
        计算滑点
        
        Args:
            original_price: 原始价格
            trade_volume: 交易量
            market_data: 市场数据
            slippage_type: 滑点类型
            
        Returns:
            SlippageResult: 滑点计算结果
        """
        if slippage_type is None:
            slippage_type = self.slippage_type
            
        if market_data is None:
            market_data = {}
            
        if slippage_type == SlippageType.FIXED:
            return self.fixed_slippage_model(original_price, trade_volume)
        elif slippage_type == SlippageType.PERCENTAGE:
            return self.percentage_slippage_model(original_price, trade_volume)
        elif slippage_type == SlippageType.VOLUME_ADJUSTED:
            avg_volume = market_data.get('average_volume', 1000000)
            return self.volume_adjusted_slippage(original_price, trade_volume, avg_volume)
        elif slippage_type == SlippageType.VOLATILITY_ADJUSTED:
            volatility = market_data.get('volatility', 0.2)
            return self.volatility_adjusted_slippage(original_price, trade_volume, volatility)
        elif slippage_type == SlippageType.LIQUIDITY_ADJUSTED:
            spread = market_data.get('bid_ask_spread', 0.01)
            depth = market_data.get('market_depth', 1000000)
            return self.liquidity_adjusted_slippage(original_price, trade_volume, spread, depth)
        else:
            raise ValueError(f"不支持的滑点类型: {slippage_type}")
    
    def calculate_total_slippage(self, trades: List[Dict]) -> Dict[str, float]:
        """
        计算总滑点
        
        Args:
            trades: 交易列表
            
        Returns:
            Dict[str, float]: 总滑点统计
        """
        total_slippage = 0.0
        slippage_by_symbol = {}
        total_trade_amount = 0.0
        
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            price = trade.get('price', 0)
            volume = trade.get('volume', 0)
            market_data = trade.get('market_data', {})
            
            slippage_result = self.calculate_slippage(price, volume, market_data)
            total_slippage += slippage_result.slippage_amount * volume
            total_trade_amount += price * volume
            
            if symbol in slippage_by_symbol:
                slippage_by_symbol[symbol] += slippage_result.slippage_amount * volume
            else:
                slippage_by_symbol[symbol] = slippage_result.slippage_amount * volume
        
        return {
            'total_slippage': total_slippage,
            'slippage_by_symbol': slippage_by_symbol,
            'avg_slippage_per_trade': total_slippage / len(trades) if trades else 0,
            'total_trade_amount': total_trade_amount,
            'overall_slippage_rate': total_slippage / total_trade_amount if total_trade_amount > 0 else 0
        }
    
    def optimize_trade_execution(self, total_volume: float, original_price: float,
                               market_data: Dict, max_slices: int = 10) -> Dict[str, float]:
        """
        优化交易执行以减少滑点
        
        Args:
            total_volume: 总交易量
            original_price: 原始价格
            market_data: 市场数据
            max_slices: 最大分片数
            
        Returns:
            Dict[str, float]: 优化结果
        """
        # 尝试不同的分片策略
        slice_strategies = []
        
        for n_slices in range(1, max_slices + 1):
            slice_volume = total_volume / n_slices
            total_slippage = 0.0
            
            for i in range(n_slices):
                # 假设后续交易会有更高的滑点
                volume_multiplier = 1.0 + (i * 0.1)  # 每片增加10%的滑点
                adjusted_market_data = market_data.copy()
                adjusted_market_data['average_volume'] = market_data.get('average_volume', 1000000) * volume_multiplier
                
                slippage_result = self.calculate_slippage(
                    original_price, slice_volume, adjusted_market_data
                )
                total_slippage += slippage_result.slippage_amount * slice_volume
            
            slice_strategies.append({
                'n_slices': n_slices,
                'slice_volume': slice_volume,
                'total_slippage': total_slippage,
                'avg_slippage_per_slice': total_slippage / n_slices,
                'slippage_rate': total_slippage / (total_volume * original_price)
            })
        
        # 找到总滑点最小的策略
        best_strategy = min(slice_strategies, key=lambda x: x['total_slippage'])
        
        return {
            'optimal_slices': best_strategy['n_slices'],
            'optimal_slice_volume': best_strategy['slice_volume'],
            'min_total_slippage': best_strategy['total_slippage'],
            'slippage_reduction_vs_single': (
                slice_strategies[0]['total_slippage'] - best_strategy['total_slippage']
            ) / slice_strategies[0]['total_slippage'] if slice_strategies[0]['total_slippage'] > 0 else 0,
            'all_strategies': slice_strategies
        }

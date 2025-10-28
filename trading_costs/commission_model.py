"""
佣金模型
实现固定佣金、比例佣金、分层佣金等多种佣金计算方式
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class CommissionType(Enum):
    """佣金类型枚举"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    TIERED = "tiered"
    HYBRID = "hybrid"


@dataclass
class CommissionTier:
    """佣金分层"""
    min_amount: float
    max_amount: Optional[float]
    fixed_fee: float
    percentage_rate: float


@dataclass
class CommissionResult:
    """佣金计算结果"""
    commission_amount: float
    commission_type: CommissionType
    trade_amount: float
    commission_rate: float
    effective_rate: float


class CommissionModel:
    """
    佣金模型类
    
    实现多种佣金计算方式：
    - 固定佣金
    - 比例佣金
    - 分层佣金
    - 混合佣金
    """
    
    def __init__(self, commission_type: CommissionType = CommissionType.PERCENTAGE,
                 fixed_fee: float = 0.0, percentage_rate: float = 0.001,
                 min_commission: float = 1.0, max_commission: float = 50.0,
                 tiers: Optional[List[CommissionTier]] = None):
        """
        初始化佣金模型
        
        Args:
            commission_type: 佣金类型
            fixed_fee: 固定费用
            percentage_rate: 比例费率
            min_commission: 最低佣金
            max_commission: 最高佣金
            tiers: 分层佣金配置
        """
        self.commission_type = commission_type
        self.fixed_fee = fixed_fee
        self.percentage_rate = percentage_rate
        self.min_commission = min_commission
        self.max_commission = max_commission
        self.tiers = tiers or self._get_default_tiers()
        
    def _get_default_tiers(self) -> List[CommissionTier]:
        """获取默认分层佣金配置"""
        return [
            CommissionTier(min_amount=0, max_amount=10000, fixed_fee=5.0, percentage_rate=0.001),
            CommissionTier(min_amount=10000, max_amount=50000, fixed_fee=8.0, percentage_rate=0.0008),
            CommissionTier(min_amount=50000, max_amount=100000, fixed_fee=10.0, percentage_rate=0.0005),
            CommissionTier(min_amount=100000, max_amount=None, fixed_fee=15.0, percentage_rate=0.0003)
        ]
    
    def fixed_commission(self, trade_amount: float) -> CommissionResult:
        """
        固定佣金计算
        
        Args:
            trade_amount: 交易金额
            
        Returns:
            CommissionResult: 佣金计算结果
        """
        commission = self.fixed_fee
        effective_rate = commission / trade_amount if trade_amount > 0 else 0
        
        return CommissionResult(
            commission_amount=commission,
            commission_type=CommissionType.FIXED,
            trade_amount=trade_amount,
            commission_rate=self.fixed_fee,
            effective_rate=effective_rate
        )
    
    def percentage_commission(self, trade_amount: float) -> CommissionResult:
        """
        比例佣金计算
        
        Args:
            trade_amount: 交易金额
            
        Returns:
            CommissionResult: 佣金计算结果
        """
        commission = trade_amount * self.percentage_rate
        
        # 应用最低和最高限制
        commission = max(commission, self.min_commission)
        commission = min(commission, self.max_commission)
        
        effective_rate = commission / trade_amount if trade_amount > 0 else 0
        
        return CommissionResult(
            commission_amount=commission,
            commission_type=CommissionType.PERCENTAGE,
            trade_amount=trade_amount,
            commission_rate=self.percentage_rate,
            effective_rate=effective_rate
        )
    
    def tiered_commission(self, trade_amount: float) -> CommissionResult:
        """
        分层佣金计算
        
        Args:
            trade_amount: 交易金额
            
        Returns:
            CommissionResult: 佣金计算结果
        """
        applicable_tier = None
        
        # 找到适用的分层
        for tier in self.tiers:
            if tier.min_amount <= trade_amount and (tier.max_amount is None or trade_amount <= tier.max_amount):
                applicable_tier = tier
                break
        
        if applicable_tier is None:
            # 使用最高分层
            applicable_tier = self.tiers[-1]
        
        # 计算佣金
        commission = applicable_tier.fixed_fee + (trade_amount * applicable_tier.percentage_rate)
        
        # 应用最低和最高限制
        commission = max(commission, self.min_commission)
        commission = min(commission, self.max_commission)
        
        effective_rate = commission / trade_amount if trade_amount > 0 else 0
        
        return CommissionResult(
            commission_amount=commission,
            commission_type=CommissionType.TIERED,
            trade_amount=trade_amount,
            commission_rate=applicable_tier.percentage_rate,
            effective_rate=effective_rate
        )
    
    def hybrid_commission(self, trade_amount: float) -> CommissionResult:
        """
        混合佣金计算（固定+比例）
        
        Args:
            trade_amount: 交易金额
            
        Returns:
            CommissionResult: 佣金计算结果
        """
        commission = self.fixed_fee + (trade_amount * self.percentage_rate)
        
        # 应用最低和最高限制
        commission = max(commission, self.min_commission)
        commission = min(commission, self.max_commission)
        
        effective_rate = commission / trade_amount if trade_amount > 0 else 0
        
        return CommissionResult(
            commission_amount=commission,
            commission_type=CommissionType.HYBRID,
            trade_amount=trade_amount,
            commission_rate=self.percentage_rate,
            effective_rate=effective_rate
        )
    
    def calculate_commission(self, trade_amount: float, 
                           commission_type: Optional[CommissionType] = None) -> CommissionResult:
        """
        计算佣金
        
        Args:
            trade_amount: 交易金额
            commission_type: 佣金类型，如为None则使用默认类型
            
        Returns:
            CommissionResult: 佣金计算结果
        """
        if commission_type is None:
            commission_type = self.commission_type
            
        if commission_type == CommissionType.FIXED:
            return self.fixed_commission(trade_amount)
        elif commission_type == CommissionType.PERCENTAGE:
            return self.percentage_commission(trade_amount)
        elif commission_type == CommissionType.TIERED:
            return self.tiered_commission(trade_amount)
        elif commission_type == CommissionType.HYBRID:
            return self.hybrid_commission(trade_amount)
        else:
            raise ValueError(f"不支持的佣金类型: {commission_type}")
    
    def calculate_total_commission(self, trades: List[Dict]) -> Dict[str, float]:
        """
        计算总佣金
        
        Args:
            trades: 交易列表，每个交易包含'symbol', 'amount', 'price'
            
        Returns:
            Dict[str, float]: 总佣金统计
        """
        total_commission = 0.0
        commission_by_symbol = {}
        
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            amount = trade.get('amount', 0)
            price = trade.get('price', 0)
            trade_amount = amount * price
            
            commission_result = self.calculate_commission(trade_amount)
            total_commission += commission_result.commission_amount
            
            if symbol in commission_by_symbol:
                commission_by_symbol[symbol] += commission_result.commission_amount
            else:
                commission_by_symbol[symbol] = commission_result.commission_amount
        
        return {
            'total_commission': total_commission,
            'commission_by_symbol': commission_by_symbol,
            'avg_commission_per_trade': total_commission / len(trades) if trades else 0,
            'total_trade_amount': sum(trade.get('amount', 0) * trade.get('price', 0) for trade in trades),
            'overall_effective_rate': total_commission / sum(trade.get('amount', 0) * trade.get('price', 0) for trade in trades) 
                                   if sum(trade.get('amount', 0) * trade.get('price', 0) for trade in trades) > 0 else 0
        }
    
    def optimize_trade_size(self, expected_return: float, trade_amount: float,
                          min_trade_size: float = 100.0) -> Dict[str, float]:
        """
        优化交易规模以最小化佣金影响
        
        Args:
            expected_return: 预期收益率
            trade_amount: 计划交易金额
            min_trade_size: 最小交易规模
            
        Returns:
            Dict[str, float]: 优化结果
        """
        # 计算不同交易规模下的净收益
        trade_sizes = [
            min_trade_size * (2 ** i) for i in range(10) 
            if min_trade_size * (2 ** i) <= trade_amount
        ]
        
        if trade_amount not in trade_sizes:
            trade_sizes.append(trade_amount)
        
        optimization_results = []
        
        for size in trade_sizes:
            commission_result = self.calculate_commission(size)
            net_return = (expected_return * size) - commission_result.commission_amount
            net_return_rate = net_return / size if size > 0 else 0
            
            optimization_results.append({
                'trade_size': size,
                'commission': commission_result.commission_amount,
                'commission_rate': commission_result.effective_rate,
                'net_return': net_return,
                'net_return_rate': net_return_rate
            })
        
        # 找到净收益率最高的交易规模
        best_trade = max(optimization_results, key=lambda x: x['net_return_rate'])
        
        return {
            'optimal_trade_size': best_trade['trade_size'],
            'optimal_net_return_rate': best_trade['net_return_rate'],
            'commission_at_optimal': best_trade['commission'],
            'all_options': optimization_results
        }

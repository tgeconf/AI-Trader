"""
交易成本计算器
整合佣金、滑点和市场冲击模型，提供完整的交易成本分析
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from .commission_model import CommissionModel, CommissionType
from .slippage_model import SlippageModel, SlippageType
from .market_impact_model import MarketImpactModel, ImpactType


@dataclass
class TransactionCostResult:
    """交易成本计算结果"""
    commission_cost: float
    slippage_cost: float
    market_impact_cost: float
    total_cost: float
    trade_amount: float
    original_price: float
    effective_price: float
    cost_rate: float
    breakdown: Dict[str, float]


@dataclass
class TradeOptimizationResult:
    """交易优化结果"""
    optimal_trade_size: float
    optimal_slices: int
    optimal_execution_time: int
    min_total_cost: float
    cost_reduction: float
    recommendations: List[str]


class TransactionCostCalculator:
    """
    交易成本计算器类
    
    整合佣金、滑点和市场冲击模型，提供：
    - 完整的交易成本分析
    - 交易优化建议
    - 成本敏感性分析
    """
    
    def __init__(self, commission_model: Optional[CommissionModel] = None,
                 slippage_model: Optional[SlippageModel] = None,
                 market_impact_model: Optional[MarketImpactModel] = None):
        """
        初始化交易成本计算器
        
        Args:
            commission_model: 佣金模型
            slippage_model: 滑点模型
            market_impact_model: 市场冲击模型
        """
        self.commission_model = commission_model or CommissionModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.market_impact_model = market_impact_model or MarketImpactModel()
        
    def calculate_single_trade_cost(self, symbol: str, price: float, volume: float,
                                  market_data: Dict) -> TransactionCostResult:
        """
        计算单笔交易成本
        
        Args:
            symbol: 股票代码
            price: 价格
            volume: 交易量
            market_data: 市场数据
            
        Returns:
            TransactionCostResult: 交易成本计算结果
        """
        trade_amount = price * volume
        
        # 计算佣金
        commission_result = self.commission_model.calculate_commission(trade_amount)
        commission_cost = commission_result.commission_amount
        
        # 计算滑点
        slippage_result = self.slippage_model.calculate_slippage(price, volume, market_data)
        slippage_cost = slippage_result.slippage_amount * volume
        
        # 计算市场冲击
        market_impact_result = self.market_impact_model.calculate_market_impact(
            price, volume, market_data
        )
        market_impact_cost = market_impact_result.total_impact * volume
        
        # 总成本
        total_cost = commission_cost + slippage_cost + market_impact_cost
        
        # 有效价格
        effective_price = price + (slippage_result.slippage_amount + market_impact_result.total_impact)
        cost_rate = total_cost / trade_amount if trade_amount > 0 else 0
        
        # 成本分解
        breakdown = {
            'commission': commission_cost,
            'slippage': slippage_cost,
            'market_impact': market_impact_cost,
            'commission_rate': commission_cost / trade_amount if trade_amount > 0 else 0,
            'slippage_rate': slippage_cost / trade_amount if trade_amount > 0 else 0,
            'market_impact_rate': market_impact_cost / trade_amount if trade_amount > 0 else 0
        }
        
        return TransactionCostResult(
            commission_cost=commission_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=market_impact_cost,
            total_cost=total_cost,
            trade_amount=trade_amount,
            original_price=price,
            effective_price=effective_price,
            cost_rate=cost_rate,
            breakdown=breakdown
        )
    
    def calculate_portfolio_trading_cost(self, trades: List[Dict]) -> Dict[str, float]:
        """
        计算投资组合交易成本
        
        Args:
            trades: 交易列表
            
        Returns:
            Dict[str, float]: 投资组合交易成本统计
        """
        total_commission = 0.0
        total_slippage = 0.0
        total_market_impact = 0.0
        total_trade_amount = 0.0
        cost_by_symbol = {}
        
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            price = trade.get('price', 0)
            volume = trade.get('volume', 0)
            market_data = trade.get('market_data', {})
            
            cost_result = self.calculate_single_trade_cost(symbol, price, volume, market_data)
            
            total_commission += cost_result.commission_cost
            total_slippage += cost_result.slippage_cost
            total_market_impact += cost_result.market_impact_cost
            total_trade_amount += cost_result.trade_amount
            
            if symbol in cost_by_symbol:
                cost_by_symbol[symbol] += cost_result.total_cost
            else:
                cost_by_symbol[symbol] = cost_result.total_cost
        
        total_cost = total_commission + total_slippage + total_market_impact
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_market_impact': total_market_impact,
            'total_trading_cost': total_cost,
            'total_trade_amount': total_trade_amount,
            'overall_cost_rate': total_cost / total_trade_amount if total_trade_amount > 0 else 0,
            'cost_by_symbol': cost_by_symbol,
            'commission_rate': total_commission / total_trade_amount if total_trade_amount > 0 else 0,
            'slippage_rate': total_slippage / total_trade_amount if total_trade_amount > 0 else 0,
            'market_impact_rate': total_market_impact / total_trade_amount if total_trade_amount > 0 else 0
        }
    
    def optimize_trade_execution(self, symbol: str, total_volume: float, price: float,
                               market_data: Dict) -> TradeOptimizationResult:
        """
        优化交易执行
        
        Args:
            symbol: 股票代码
            total_volume: 总交易量
            price: 价格
            market_data: 市场数据
            
        Returns:
            TradeOptimizationResult: 交易优化结果
        """
        recommendations = []
        
        # 1. 优化交易规模
        trade_size_optimization = self.commission_model.optimize_trade_size(
            expected_return=0.02,  # 假设2%的预期收益
            trade_amount=total_volume * price,
            min_trade_size=1000.0
        )
        
        # 2. 优化交易分片
        execution_optimization = self.slippage_model.optimize_trade_execution(
            total_volume=total_volume,
            original_price=price,
            market_data=market_data
        )
        
        # 3. 优化交易速度
        schedule_optimization = self.market_impact_model.optimize_trade_schedule(
            total_volume=total_volume,
            original_price=price,
            market_data=market_data
        )
        
        # 计算优化后的成本
        optimal_trade_size = trade_size_optimization['optimal_trade_size']
        optimal_volume = optimal_trade_size / price
        
        optimized_cost = self.calculate_single_trade_cost(
            symbol, price, optimal_volume, market_data
        )
        
        # 原始成本（一次性交易）
        original_cost = self.calculate_single_trade_cost(
            symbol, price, total_volume, market_data
        )
        
        cost_reduction = (original_cost.total_cost - optimized_cost.total_cost) / original_cost.total_cost
        
        # 生成建议
        if cost_reduction > 0.1:
            recommendations.append(f"建议将交易分片为{execution_optimization['optimal_slices']}次执行")
        
        if trade_size_optimization['optimal_trade_size'] < total_volume * price:
            recommendations.append(f"建议调整交易规模至${trade_size_optimization['optimal_trade_size']:.2f}")
        
        if schedule_optimization['optimal_execution_minutes'] > 1:
            recommendations.append(f"建议在{schedule_optimization['optimal_execution_minutes']}分钟内完成交易")
        
        return TradeOptimizationResult(
            optimal_trade_size=optimal_trade_size,
            optimal_slices=execution_optimization['optimal_slices'],
            optimal_execution_time=schedule_optimization['optimal_execution_minutes'],
            min_total_cost=optimized_cost.total_cost,
            cost_reduction=cost_reduction,
            recommendations=recommendations
        )
    
    def sensitivity_analysis(self, symbol: str, price: float, volume: float,
                           market_data: Dict, parameters: Dict[str, List[float]]) -> Dict[str, List[Dict]]:
        """
        敏感性分析
        
        Args:
            symbol: 股票代码
            price: 价格
            volume: 交易量
            market_data: 市场数据
            parameters: 敏感性分析参数
            
        Returns:
            Dict[str, List[Dict]]: 敏感性分析结果
        """
        sensitivity_results = {}
        
        # 交易量敏感性
        if 'volume' in parameters:
            volume_sensitivity = []
            for vol_multiplier in parameters['volume']:
                adjusted_volume = volume * vol_multiplier
                cost_result = self.calculate_single_trade_cost(symbol, price, adjusted_volume, market_data)
                volume_sensitivity.append({
                    'volume_multiplier': vol_multiplier,
                    'adjusted_volume': adjusted_volume,
                    'total_cost': cost_result.total_cost,
                    'cost_rate': cost_result.cost_rate,
                    'commission': cost_result.commission_cost,
                    'slippage': cost_result.slippage_cost,
                    'market_impact': cost_result.market_impact_cost
                })
            sensitivity_results['volume'] = volume_sensitivity
        
        # 波动率敏感性
        if 'volatility' in parameters:
            volatility_sensitivity = []
            for volatility in parameters['volatility']:
                adjusted_market_data = market_data.copy()
                adjusted_market_data['volatility'] = volatility
                cost_result = self.calculate_single_trade_cost(symbol, price, volume, adjusted_market_data)
                volatility_sensitivity.append({
                    'volatility': volatility,
                    'total_cost': cost_result.total_cost,
                    'cost_rate': cost_result.cost_rate,
                    'commission': cost_result.commission_cost,
                    'slippage': cost_result.slippage_cost,
                    'market_impact': cost_result.market_impact_cost
                })
            sensitivity_results['volatility'] = volatility_sensitivity
        
        # 流动性敏感性
        if 'liquidity' in parameters:
            liquidity_sensitivity = []
            for liquidity in parameters['liquidity']:
                adjusted_market_data = market_data.copy()
                adjusted_market_data['liquidity'] = liquidity
                cost_result = self.calculate_single_trade_cost(symbol, price, volume, adjusted_market_data)
                liquidity_sensitivity.append({
                    'liquidity': liquidity,
                    'total_cost': cost_result.total_cost,
                    'cost_rate': cost_result.cost_rate,
                    'commission': cost_result.commission_cost,
                    'slippage': cost_result.slippage_cost,
                    'market_impact': cost_result.market_impact_cost
                })
            sensitivity_results['liquidity'] = liquidity_sensitivity
        
        return sensitivity_results
    
    def generate_cost_report(self, trades: List[Dict]) -> Dict[str, any]:
        """
        生成成本报告
        
        Args:
            trades: 交易列表
            
        Returns:
            Dict[str, any]: 成本报告
        """
        portfolio_cost = self.calculate_portfolio_trading_cost(trades)
        
        # 计算成本效率指标
        total_trade_amount = portfolio_cost['total_trade_amount']
        total_cost = portfolio_cost['total_trading_cost']
        
        # 成本构成分析
        cost_breakdown = {
            'commission_percentage': (portfolio_cost['total_commission'] / total_cost * 100) if total_cost > 0 else 0,
            'slippage_percentage': (portfolio_cost['total_slippage'] / total_cost * 100) if total_cost > 0 else 0,
            'market_impact_percentage': (portfolio_cost['total_market_impact'] / total_cost * 100) if total_cost > 0 else 0
        }
        
        # 识别高成本交易
        high_cost_trades = []
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            price = trade.get('price', 0)
            volume = trade.get('volume', 0)
            market_data = trade.get('market_data', {})
            
            cost_result = self.calculate_single_trade_cost(symbol, price, volume, market_data)
            
            if cost_result.cost_rate > 0.01:  # 成本率超过1%
                high_cost_trades.append({
                    'symbol': symbol,
                    'cost_rate': cost_result.cost_rate,
                    'total_cost': cost_result.total_cost,
                    'trade_amount': cost_result.trade_amount,
                    'breakdown': cost_result.breakdown
                })
        
        # 生成优化建议
        optimization_suggestions = []
        
        if portfolio_cost['commission_rate'] > 0.002:
            optimization_suggestions.append("佣金成本较高，考虑调整佣金结构或合并交易")
        
        if portfolio_cost['slippage_rate'] > 0.003:
            optimization_suggestions.append("滑点成本较高，建议优化交易执行策略")
        
        if portfolio_cost['market_impact_rate'] > 0.005:
            optimization_suggestions.append("市场冲击成本较高，建议分散交易执行")
        
        return {
            'summary': {
                'total_trades': len(trades),
                'total_trade_amount': total_trade_amount,
                'total_trading_cost': total_cost,
                'overall_cost_rate': portfolio_cost['overall_cost_rate'],
                'cost_efficiency_score': 1 - portfolio_cost['overall_cost_rate']  # 成本效率得分
            },
            'cost_breakdown': cost_breakdown,
            'high_cost_trades': high_cost_trades,
            'optimization_suggestions': optimization_suggestions,
            'detailed_costs': portfolio_cost
        }

"""
头寸规模控制模块
实现凯利公式、固定分数、波动率调整等多种头寸规模计算方法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PositionSizingMethod(Enum):
    """头寸规模计算方法枚举"""
    KELLY = "kelly"
    FIXED_FRACTION = "fixed_fraction"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"


@dataclass
class PositionSizingResult:
    """头寸规模计算结果"""
    method: PositionSizingMethod
    position_sizes: Dict[str, float]  # 各资产头寸规模
    total_risk: float  # 总风险暴露
    risk_per_trade: float  # 单笔交易风险
    leverage: float  # 杠杆倍数


class PositionSizing:
    """
    头寸规模控制类
    
    实现多种头寸规模计算方法：
    - 凯利公式
    - 固定分数法
    - 波动率调整法
    - 风险平价法
    - 最优f值法
    """
    
    def __init__(self, total_capital: float, max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.10, max_leverage: float = 2.0):
        """
        初始化头寸规模控制器
        
        Args:
            total_capital: 总资本
            max_risk_per_trade: 单笔交易最大风险比例
            max_portfolio_risk: 投资组合最大风险比例
            max_leverage: 最大杠杆倍数
        """
        self.total_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_leverage = max_leverage
        
    def kelly_criterion(self, win_rate: float, win_loss_ratio: float, 
                       confidence: float = 0.95) -> float:
        """
        凯利公式计算最优头寸规模
        
        Args:
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            confidence: 置信水平，用于调整保守程度
            
        Returns:
            float: 凯利分数
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
            
        # 标准凯利公式
        kelly_f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # 应用置信水平调整（更保守）
        adjusted_f = kelly_f * confidence
        
        # 限制在合理范围内
        return max(0.0, min(adjusted_f, self.max_risk_per_trade))
    
    def fixed_fractional(self, risk_per_trade: float, stop_loss_pct: float) -> float:
        """
        固定分数法计算头寸规模
        
        Args:
            risk_per_trade: 单笔交易风险金额
            stop_loss_pct: 止损百分比
            
        Returns:
            float: 头寸规模
        """
        if stop_loss_pct <= 0:
            return 0.0
            
        position_size = risk_per_trade / stop_loss_pct
        max_position = self.total_capital * self.max_risk_per_trade / stop_loss_pct
        
        return min(position_size, max_position)
    
    def volatility_adjusted(self, volatility: float, target_volatility: float = 0.15,
                          correlation_matrix: Optional[pd.DataFrame] = None) -> float:
        """
        波动率调整头寸规模
        
        Args:
            volatility: 资产波动率
            target_volatility: 目标波动率
            correlation_matrix: 相关性矩阵
            
        Returns:
            float: 调整后的头寸规模
        """
        if volatility <= 0:
            return 0.0
            
        # 基础波动率调整
        base_size = target_volatility / volatility
        
        # 考虑相关性调整
        if correlation_matrix is not None:
            avg_correlation = correlation_matrix.mean().mean()
            correlation_adjustment = 1 / np.sqrt(1 + avg_correlation)
            base_size *= correlation_adjustment
            
        # 应用风险限制
        max_size = self.total_capital * self.max_risk_per_trade / volatility
        
        return min(base_size, max_size)
    
    def risk_parity_allocation(self, volatilities: Dict[str, float], 
                             correlations: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        风险平价分配
        
        Args:
            volatilities: 各资产波动率
            correlations: 相关性矩阵
            
        Returns:
            Dict[str, float]: 各资产权重
        """
        assets = list(volatilities.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return {}
            
        # 构建协方差矩阵
        cov_matrix = np.zeros((n_assets, n_assets))
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i == j:
                    cov_matrix[i, j] = volatilities[asset_i] ** 2
                else:
                    corr = correlations.get(asset_i, {}).get(asset_j, 0.0)
                    cov_matrix[i, j] = volatilities[asset_i] * volatilities[asset_j] * corr
        
        # 计算风险贡献
        try:
            # 使用逆方差加权作为初始解
            inv_variances = 1 / np.diag(cov_matrix)
            weights = inv_variances / inv_variances.sum()
            
            # 迭代优化风险平价
            max_iterations = 100
            tolerance = 1e-6
            
            for _ in range(max_iterations):
                # 计算边际风险贡献
                portfolio_variance = weights.T @ cov_matrix @ weights
                marginal_contributions = (cov_matrix @ weights) / portfolio_variance
                
                # 计算风险贡献
                risk_contributions = weights * marginal_contributions
                
                # 检查收敛
                if np.allclose(risk_contributions, risk_contributions.mean(), rtol=tolerance):
                    break
                    
                # 调整权重
                adjustment = risk_contributions.mean() / risk_contributions
                weights *= adjustment
                weights /= weights.sum()
                
        except (ValueError, ZeroDivisionError):
            # 如果计算失败，使用等权重
            weights = np.ones(n_assets) / n_assets
            
        return {asset: weight for asset, weight in zip(assets, weights)}
    
    def optimal_f(self, returns: pd.Series, max_f: float = 0.5) -> float:
        """
        最优f值法（Ralph Vince方法）
        
        Args:
            returns: 收益率序列
            max_f: 最大f值
            
        Returns:
            float: 最优f值
        """
        if len(returns) < 10:
            return 0.0
            
        # 计算几何平均收益
        def geometric_growth(f):
            growth_factors = 1 + f * returns
            return np.prod(growth_factors) ** (1 / len(returns)) - 1
        
        # 网格搜索最优f值
        f_values = np.linspace(0.01, max_f, 100)
        growth_rates = [geometric_growth(f) for f in f_values]
        
        optimal_f = f_values[np.argmax(growth_rates)]
        
        # 应用风险限制
        return min(optimal_f, self.max_risk_per_trade)
    
    def calculate_comprehensive_position_sizes(self, 
                                             asset_data: Dict[str, Dict],
                                             method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_ADJUSTED) -> PositionSizingResult:
        """
        综合计算头寸规模
        
        Args:
            asset_data: 资产数据，包含波动率、相关性等信息
            method: 头寸规模计算方法
            
        Returns:
            PositionSizingResult: 头寸规模计算结果
        """
        position_sizes = {}
        
        if method == PositionSizingMethod.KELLY:
            for asset, data in asset_data.items():
                win_rate = data.get('win_rate', 0.5)
                win_loss_ratio = data.get('win_loss_ratio', 2.0)
                position_sizes[asset] = self.kelly_criterion(win_rate, win_loss_ratio)
                
        elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            volatilities = {asset: data.get('volatility', 0.2) for asset, data in asset_data.items()}
            correlations = {asset: data.get('correlations', {}) for asset, data in asset_data.items()}
            
            for asset, data in asset_data.items():
                volatility = data.get('volatility', 0.2)
                position_sizes[asset] = self.volatility_adjusted(volatility)
                
        elif method == PositionSizingMethod.RISK_PARITY:
            volatilities = {asset: data.get('volatility', 0.2) for asset, data in asset_data.items()}
            correlations = {asset: data.get('correlations', {}) for asset, data in asset_data.items()}
            
            weights = self.risk_parity_allocation(volatilities, correlations)
            for asset, weight in weights.items():
                position_sizes[asset] = weight * self.total_capital
                
        elif method == PositionSizingMethod.OPTIMAL_F:
            for asset, data in asset_data.items():
                returns = data.get('returns', pd.Series([0.0]))
                position_sizes[asset] = self.optimal_f(returns)
                
        else:  # FIXED_FRACTION
            for asset, data in asset_data.items():
                stop_loss = data.get('stop_loss', 0.05)
                position_sizes[asset] = self.fixed_fractional(
                    self.total_capital * self.max_risk_per_trade, stop_loss
                )
        
        # 计算总风险暴露
        total_position_value = sum(position_sizes.values())
        total_risk = total_position_value / self.total_capital
        
        # 计算杠杆
        leverage = total_position_value / self.total_capital
        
        # 应用杠杆限制
        if leverage > self.max_leverage:
            scaling_factor = self.max_leverage / leverage
            position_sizes = {asset: size * scaling_factor for asset, size in position_sizes.items()}
            total_risk *= scaling_factor
            leverage = self.max_leverage
        
        return PositionSizingResult(
            method=method,
            position_sizes=position_sizes,
            total_risk=total_risk,
            risk_per_trade=self.max_risk_per_trade,
            leverage=leverage
        )
    
    def validate_position_sizes(self, position_sizes: Dict[str, float], 
                              current_positions: Dict[str, float]) -> Tuple[bool, str]:
        """
        验证头寸规模是否合规
        
        Args:
            position_sizes: 建议头寸规模
            current_positions: 当前持仓
            
        Returns:
            Tuple[bool, str]: (是否合规, 错误信息)
        """
        total_new_position = sum(position_sizes.values())
        total_current_position = sum(current_positions.values())
        
        # 检查杠杆限制
        total_exposure = total_new_position + total_current_position
        leverage = total_exposure / self.total_capital
        
        if leverage > self.max_leverage:
            return False, f"杠杆倍数{leverage:.2f}超过限制{self.max_leverage}"
        
        # 检查单资产风险
        for asset, size in position_sizes.items():
            asset_risk = size / self.total_capital
            if asset_risk > self.max_risk_per_trade:
                return False, f"资产{asset}风险{asset_risk:.2%}超过单笔限制{self.max_risk_per_trade:.2%}"
        
        # 检查总风险
        total_risk = total_new_position / self.total_capital
        if total_risk > self.max_portfolio_risk:
            return False, f"总风险{total_risk:.2%}超过组合限制{self.max_portfolio_risk:.2%}"
        
        return True, "头寸规模验证通过"

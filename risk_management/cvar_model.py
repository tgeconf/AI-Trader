"""
条件风险价值(CVaR)模型
实现CVaR计算，提供比VaR更严格的风险度量
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CVaRResult:
    """CVaR计算结果"""
    cvar_95: float
    cvar_99: float
    cvar_99_5: float
    confidence_levels: List[float]
    cvar_values: Dict[float, float]
    calculation_time: datetime
    portfolio_value: float


class CVaRModel:
    """
    条件风险价值模型
    
    CVaR（条件风险价值）衡量在VaR突破情况下的平均损失，
    提供比VaR更严格的风险度量
    """
    
    def __init__(self, lookback_period: int = 252, confidence_levels: List[float] = None):
        """
        初始化CVaR模型
        
        Args:
            lookback_period: 回看期，默认252个交易日
            confidence_levels: 置信水平列表
        """
        self.lookback_period = lookback_period
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.995]
        self.returns_data = None
        
    def fit(self, returns: pd.DataFrame) -> None:
        """
        拟合模型数据
        
        Args:
            returns: 收益率数据
        """
        self.returns_data = returns.iloc[-self.lookback_period:]
        
    def historical_cvar(self, weights: np.ndarray, portfolio_value: float) -> CVaRResult:
        """
        历史模拟法计算CVaR
        
        Args:
            weights: 资产权重
            portfolio_value: 投资组合价值
            
        Returns:
            CVaRResult: CVaR计算结果
        """
        if self.returns_data is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算投资组合历史收益率
        portfolio_returns = self.returns_data.dot(weights)
        
        # 计算不同置信水平的CVaR
        cvar_values = {}
        for confidence in self.confidence_levels:
            # 计算VaR
            var = np.percentile(portfolio_returns, (1 - confidence) * 100)
            
            # 计算CVaR：VaR突破情况下的平均损失
            tail_returns = portfolio_returns[portfolio_returns <= var]
            cvar = tail_returns.mean() if len(tail_returns) > 0 else var
            cvar_values[confidence] = abs(cvar) * portfolio_value
            
        return CVaRResult(
            cvar_95=cvar_values[0.95],
            cvar_99=cvar_values[0.99],
            cvar_99_5=cvar_values[0.995],
            confidence_levels=self.confidence_levels,
            cvar_values=cvar_values,
            calculation_time=datetime.now(),
            portfolio_value=portfolio_value
        )
    
    def parametric_cvar(self, weights: np.ndarray, portfolio_value: float) -> CVaRResult:
        """
        参数法计算CVaR
        
        Args:
            weights: 资产权重
            portfolio_value: 投资组合价值
            
        Returns:
            CVaRResult: CVaR计算结果
        """
        if self.returns_data is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算投资组合均值和标准差
        portfolio_returns = self.returns_data.dot(weights)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # 计算不同置信水平的CVaR
        cvar_values = {}
        for confidence in self.confidence_levels:
            # 计算VaR
            z_score = stats.norm.ppf(confidence)
            var = z_score * std_return - mean_return
            
            # 计算CVaR
            cvar = (std_return / (1 - confidence)) * stats.norm.pdf(z_score) - mean_return
            cvar_values[confidence] = abs(cvar) * portfolio_value
            
        return CVaRResult(
            cvar_95=cvar_values[0.95],
            cvar_99=cvar_values[0.99],
            cvar_99_5=cvar_values[0.995],
            confidence_levels=self.confidence_levels,
            cvar_values=cvar_values,
            calculation_time=datetime.now(),
            portfolio_value=portfolio_value
        )
    
    def calculate_risk_contribution(self, weights: np.ndarray, portfolio_value: float, 
                                  confidence_level: float = 0.95) -> Dict[str, float]:
        """
        计算各资产对CVaR的风险贡献
        
        Args:
            weights: 资产权重
            portfolio_value: 投资组合价值
            confidence_level: 置信水平
            
        Returns:
            Dict[str, float]: 各资产的风险贡献
        """
        if self.returns_data is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算投资组合CVaR
        portfolio_cvar = self.historical_cvar(weights, portfolio_value)
        base_cvar = portfolio_cvar.cvar_values[confidence_level]
        
        # 计算边际风险贡献
        risk_contributions = {}
        epsilon = 1e-6
        
        for i, asset in enumerate(self.returns_data.columns):
            # 微小调整权重
            perturbed_weights = weights.copy()
            perturbed_weights[i] += epsilon
            perturbed_weights /= perturbed_weights.sum()  # 重新归一化
            
            # 计算调整后的CVaR
            perturbed_cvar = self.historical_cvar(perturbed_weights, portfolio_value)
            perturbed_cvar_value = perturbed_cvar.cvar_values[confidence_level]
            
            # 计算边际风险贡献
            marginal_contribution = (perturbed_cvar_value - base_cvar) / epsilon
            risk_contributions[asset] = marginal_contribution * weights[i]
            
        return risk_contributions
    
    def calculate_diversification_benefit(self, weights: np.ndarray, portfolio_value: float) -> Dict[str, float]:
        """
        计算投资组合的分散化效益
        
        Args:
            weights: 资产权重
            portfolio_value: 投资组合价值
            
        Returns:
            Dict[str, float]: 分散化效益指标
        """
        if self.returns_data is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算投资组合CVaR
        portfolio_cvar = self.historical_cvar(weights, portfolio_value)
        
        # 计算各资产单独投资的CVaR
        individual_cvars = {}
        for asset in self.returns_data.columns:
            asset_weights = np.zeros(len(self.returns_data.columns))
            asset_weights[self.returns_data.columns.get_loc(asset)] = 1.0
            asset_cvar = self.historical_cvar(asset_weights, portfolio_value)
            individual_cvars[asset] = asset_cvar.cvar_values[0.95]
        
        # 计算加权平均CVaR
        weighted_avg_cvar = sum(w * individual_cvars[asset] 
                              for w, asset in zip(weights, self.returns_data.columns))
        
        # 计算分散化效益
        diversification_benefit = weighted_avg_cvar - portfolio_cvar.cvar_95
        diversification_ratio = portfolio_cvar.cvar_95 / weighted_avg_cvar if weighted_avg_cvar > 0 else 1.0
        
        return {
            "portfolio_cvar": portfolio_cvar.cvar_95,
            "weighted_avg_cvar": weighted_avg_cvar,
            "diversification_benefit": diversification_benefit,
            "diversification_ratio": diversification_ratio,
            "diversification_percentage": (1 - diversification_ratio) * 100
        }

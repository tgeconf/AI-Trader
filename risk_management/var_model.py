"""
风险价值(VaR)模型
实现历史模拟法、参数法、蒙特卡洛模拟等多种VaR计算方法
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class VaRResult:
    """VaR计算结果"""
    var_95: float
    var_99: float
    var_99_5: float
    method: str
    confidence_levels: List[float]
    var_values: Dict[float, float]
    calculation_time: datetime
    portfolio_value: float


class VaRModel:
    """
    风险价值模型
    
    实现多种VaR计算方法：
    - 历史模拟法
    - 参数法（方差-协方差法）
    - 蒙特卡洛模拟法
    """
    
    def __init__(self, lookback_period: int = 252, confidence_levels: List[float] = None):
        """
        初始化VaR模型
        
        Args:
            lookback_period: 回看期，默认252个交易日（约1年）
            confidence_levels: 置信水平列表，默认[0.95, 0.99, 0.995]
        """
        self.lookback_period = lookback_period
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.995]
        self.returns_data = None
        self.cov_matrix = None
        
    def fit(self, returns: pd.DataFrame) -> None:
        """
        拟合模型数据
        
        Args:
            returns: 收益率数据，DataFrame格式，列为资产，行为时间
        """
        if len(returns) < self.lookback_period:
            warnings.warn(f"数据长度({len(returns)})小于回看期({self.lookback_period})")
            
        self.returns_data = returns.iloc[-self.lookback_period:]
        self.cov_matrix = self.returns_data.cov()
        
    def historical_var(self, weights: np.ndarray, portfolio_value: float) -> VaRResult:
        """
        历史模拟法计算VaR
        
        Args:
            weights: 资产权重向量
            portfolio_value: 投资组合价值
            
        Returns:
            VaRResult: VaR计算结果
        """
        if self.returns_data is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算投资组合历史收益率
        portfolio_returns = self.returns_data.dot(weights)
        
        # 计算不同置信水平的VaR
        var_values = {}
        for confidence in self.confidence_levels:
            var = np.percentile(portfolio_returns, (1 - confidence) * 100)
            var_values[confidence] = abs(var) * portfolio_value
            
        return VaRResult(
            var_95=var_values[0.95],
            var_99=var_values[0.99],
            var_99_5=var_values[0.995],
            method="历史模拟法",
            confidence_levels=self.confidence_levels,
            var_values=var_values,
            calculation_time=datetime.now(),
            portfolio_value=portfolio_value
        )
    
    def parametric_var(self, weights: np.ndarray, portfolio_value: float) -> VaRResult:
        """
        参数法（方差-协方差法）计算VaR
        
        Args:
            weights: 资产权重向量
            portfolio_value: 投资组合价值
            
        Returns:
            VaRResult: VaR计算结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算投资组合标准差
        portfolio_variance = weights.T @ self.cov_matrix @ weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # 计算不同置信水平的VaR
        var_values = {}
        for confidence in self.confidence_levels:
            z_score = stats.norm.ppf(confidence)
            var = z_score * portfolio_std
            var_values[confidence] = abs(var) * portfolio_value
            
        return VaRResult(
            var_95=var_values[0.95],
            var_99=var_values[0.99],
            var_99_5=var_values[0.995],
            method="参数法",
            confidence_levels=self.confidence_levels,
            var_values=var_values,
            calculation_time=datetime.now(),
            portfolio_value=portfolio_value
        )
    
    def monte_carlo_var(self, weights: np.ndarray, portfolio_value: float, 
                       n_simulations: int = 10000) -> VaRResult:
        """
        蒙特卡洛模拟法计算VaR
        
        Args:
            weights: 资产权重向量
            portfolio_value: 投资组合价值
            n_simulations: 模拟次数
            
        Returns:
            VaRResult: VaR计算结果
        """
        if self.returns_data is None or self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算资产收益率均值和协方差矩阵
        means = self.returns_data.mean()
        
        # 生成多元正态分布随机数
        simulated_returns = np.random.multivariate_normal(
            means, self.cov_matrix, n_simulations
        )
        
        # 计算模拟的投资组合收益率
        portfolio_simulated_returns = simulated_returns.dot(weights)
        
        # 计算不同置信水平的VaR
        var_values = {}
        for confidence in self.confidence_levels:
            var = np.percentile(portfolio_simulated_returns, (1 - confidence) * 100)
            var_values[confidence] = abs(var) * portfolio_value
            
        return VaRResult(
            var_95=var_values[0.95],
            var_99=var_values[0.99],
            var_99_5=var_values[0.995],
            method="蒙特卡洛模拟",
            confidence_levels=self.confidence_levels,
            var_values=var_values,
            calculation_time=datetime.now(),
            portfolio_value=portfolio_value
        )
    
    def calculate_comprehensive_var(self, weights: np.ndarray, portfolio_value: float) -> Dict[str, VaRResult]:
        """
        综合计算多种方法的VaR
        
        Args:
            weights: 资产权重向量
            portfolio_value: 投资组合价值
            
        Returns:
            Dict[str, VaRResult]: 各种方法的VaR结果
        """
        results = {}
        
        # 历史模拟法
        results["historical"] = self.historical_var(weights, portfolio_value)
        
        # 参数法
        results["parametric"] = self.parametric_var(weights, portfolio_value)
        
        # 蒙特卡洛模拟法
        results["monte_carlo"] = self.monte_carlo_var(weights, portfolio_value)
        
        return results
    
    def backtest_var(self, actual_returns: pd.Series, var_predictions: pd.Series, 
                    confidence_level: float = 0.95) -> Dict[str, float]:
        """
        VaR模型回测
        
        Args:
            actual_returns: 实际收益率
            var_predictions: VaR预测值
            confidence_level: 置信水平
            
        Returns:
            Dict[str, float]: 回测指标
        """
        # 计算突破次数
        violations = (actual_returns < -var_predictions).sum()
        total_observations = len(actual_returns)
        violation_rate = violations / total_observations
        
        # 计算期望突破率
        expected_violation_rate = 1 - confidence_level
        
        # Kupiec检验统计量
        if violation_rate > 0:
            lr_uc = -2 * np.log(
                (expected_violation_rate ** violations) * 
                ((1 - expected_violation_rate) ** (total_observations - violations)) /
                (violation_rate ** violations * (1 - violation_rate) ** (total_observations - violations))
            )
        else:
            lr_uc = 0
            
        return {
            "total_observations": total_observations,
            "violations": violations,
            "violation_rate": violation_rate,
            "expected_violation_rate": expected_violation_rate,
            "kupiec_test_statistic": lr_uc,
            "test_passed": abs(violation_rate - expected_violation_rate) < 0.01
        }

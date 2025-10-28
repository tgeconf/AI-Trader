"""
均值方差优化器
实现Markowitz投资组合理论，包括有效前沿计算和最优组合选择
"""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OptimizationObjective(Enum):
    """优化目标枚举"""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_RETURN = "maximize_return"
    TARGET_RETURN = "target_return"
    TARGET_VOLATILITY = "target_volatility"


@dataclass
class OptimizationResult:
    """优化结果"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective: OptimizationObjective
    assets: List[str]
    constraints_satisfied: bool


@dataclass
class EfficientFrontierPoint:
    """有效前沿点"""
    return_target: float
    volatility: float
    weights: np.ndarray
    sharpe_ratio: float


class MeanVarianceOptimizer:
    """
    均值方差优化器
    
    实现Markowitz投资组合优化：
    - 有效前沿计算
    - 夏普比率最大化
    - 方差最小化
    - 目标收益优化
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化均值方差优化器
        
        Args:
            risk_free_rate: 无风险利率
        """
        self.risk_free_rate = risk_free_rate
        self.expected_returns = None
        self.cov_matrix = None
        self.assets = None
        
    def fit(self, returns: pd.DataFrame) -> None:
        """
        拟合模型数据
        
        Args:
            returns: 收益率数据，列为资产，行为时间
        """
        self.expected_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
        
    def portfolio_statistics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        计算投资组合统计量
        
        Args:
            weights: 资产权重
            
        Returns:
            Tuple[float, float, float]: (预期收益, 波动率, 夏普比率)
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def minimize_variance(self, target_return: Optional[float] = None,
                         constraints: Optional[List] = None) -> OptimizationResult:
        """
        最小化方差
        
        Args:
            target_return: 目标收益，如为None则无条件最小化方差
            constraints: 额外约束条件
            
        Returns:
            OptimizationResult: 优化结果
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        n_assets = len(self.assets)
        
        # 目标函数：投资组合方差
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
        ]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, self.expected_returns) - target_return
            })
        
        # 添加额外约束
        if constraints:
            constraints_list.extend(constraints)
        
        # 边界条件：权重在0到1之间
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # 计算统计量
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_statistics(result.x)
        
        return OptimizationResult(
            weights=result.x,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            objective=OptimizationObjective.MINIMIZE_VARIANCE,
            assets=self.assets,
            constraints_satisfied=result.success
        )
    
    def maximize_sharpe_ratio(self, constraints: Optional[List] = None) -> OptimizationResult:
        """
        最大化夏普比率
        
        Args:
            constraints: 额外约束条件
            
        Returns:
            OptimizationResult: 优化结果
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        n_assets = len(self.assets)
        
        # 目标函数：负夏普比率（因为我们要最大化）
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else -np.inf
            return -sharpe
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
        ]
        
        # 添加额外约束
        if constraints:
            constraints_list.extend(constraints)
        
        # 边界条件
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = optimize.minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # 计算统计量
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_statistics(result.x)
        
        return OptimizationResult(
            weights=result.x,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            objective=OptimizationObjective.MAXIMIZE_SHARPE,
            assets=self.assets,
            constraints_satisfied=result.success
        )
    
    def maximize_return(self, target_volatility: Optional[float] = None,
                       constraints: Optional[List] = None) -> OptimizationResult:
        """
        最大化收益
        
        Args:
            target_volatility: 目标波动率
            constraints: 额外约束条件
            
        Returns:
            OptimizationResult: 优化结果
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        n_assets = len(self.assets)
        
        # 目标函数：负收益（因为我们要最大化）
        def negative_return(weights):
            return -np.dot(weights, self.expected_returns)
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
        ]
        
        if target_volatility is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x.T, np.dot(self.cov_matrix, x))) - target_volatility
            })
        
        # 添加额外约束
        if constraints:
            constraints_list.extend(constraints)
        
        # 边界条件
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = optimize.minimize(
            negative_return,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        # 计算统计量
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_statistics(result.x)
        
        return OptimizationResult(
            weights=result.x,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            objective=OptimizationObjective.MAXIMIZE_RETURN,
            assets=self.assets,
            constraints_satisfied=result.success
        )
    
    def calculate_efficient_frontier(self, num_points: int = 50) -> List[EfficientFrontierPoint]:
        """
        计算有效前沿
        
        Args:
            num_points: 前沿点数
            
        Returns:
            List[EfficientFrontierPoint]: 有效前沿点列表
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算最小和最大可行收益
        min_return_result = self.minimize_variance()
        max_return_result = self.maximize_return()
        
        min_return = min_return_result.expected_return
        max_return = max_return_result.expected_return
        
        # 生成目标收益点
        target_returns = np.linspace(min_return, max_return, num_points)
        
        efficient_frontier = []
        
        for target_return in target_returns:
            try:
                result = self.minimize_variance(target_return=target_return)
                if result.constraints_satisfied:
                    point = EfficientFrontierPoint(
                        return_target=target_return,
                        volatility=result.volatility,
                        weights=result.weights,
                        sharpe_ratio=result.sharpe_ratio
                    )
                    efficient_frontier.append(point)
            except:
                continue
        
        return efficient_frontier
    
    def optimize_with_constraints(self, objective: OptimizationObjective,
                                 max_weight_per_asset: float = 0.2,
                                 min_weight_per_asset: float = 0.0,
                                 sector_constraints: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        带约束优化
        
        Args:
            objective: 优化目标
            max_weight_per_asset: 单资产最大权重
            min_weight_per_asset: 单资产最小权重
            sector_constraints: 行业约束
            
        Returns:
            OptimizationResult: 优化结果
        """
        constraints = []
        
        # 单资产权重约束
        n_assets = len(self.assets)
        bounds = tuple((min_weight_per_asset, max_weight_per_asset) for _ in range(n_assets))
        
        # 行业约束
        if sector_constraints:
            # 这里需要资产到行业的映射，简化实现
            pass
        
        if objective == OptimizationObjective.MAXIMIZE_SHARPE:
            return self.maximize_sharpe_ratio(constraints=constraints)
        elif objective == OptimizationObjective.MINIMIZE_VARIANCE:
            return self.minimize_variance(constraints=constraints)
        elif objective == OptimizationObjective.MAXIMIZE_RETURN:
            return self.maximize_return(constraints=constraints)
        else:
            raise ValueError(f"不支持的优化目标: {objective}")
    
    def calculate_turnover(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """
        计算换手率
        
        Args:
            old_weights: 旧权重
            new_weights: 新权重
            
        Returns:
            float: 换手率
        """
        return np.sum(np.abs(new_weights - old_weights)) / 2
    
    def sensitivity_analysis(self, objective: OptimizationObjective,
                           parameter_ranges: Dict[str, List[float]]) -> Dict[str, List[OptimizationResult]]:
        """
        敏感性分析
        
        Args:
            objective: 优化目标
            parameter_ranges: 参数范围
            
        Returns:
            Dict[str, List[OptimizationResult]]: 敏感性分析结果
        """
        sensitivity_results = {}
        
        # 无风险利率敏感性
        if 'risk_free_rate' in parameter_ranges:
            risk_free_sensitivity = []
            for rf_rate in parameter_ranges['risk_free_rate']:
                self.risk_free_rate = rf_rate
                result = self.optimize_with_constraints(objective)
                risk_free_sensitivity.append(result)
            sensitivity_results['risk_free_rate'] = risk_free_sensitivity
            
        # 最大权重敏感性
        if 'max_weight' in parameter_ranges:
            max_weight_sensitivity = []
            for max_w in parameter_ranges['max_weight']:
                result = self.optimize_with_constraints(objective, max_weight_per_asset=max_w)
                max_weight_sensitivity.append(result)
            sensitivity_results['max_weight'] = max_weight_sensitivity
        
        return sensitivity_results

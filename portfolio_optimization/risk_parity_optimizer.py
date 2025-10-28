"""
风险平价优化器
实现风险平价投资组合分配，确保各资产对组合风险贡献相等
"""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskParityResult:
    """风险平价结果"""
    weights: np.ndarray
    risk_contributions: np.ndarray
    total_risk: float
    assets: List[str]
    risk_parity_score: float
    convergence_achieved: bool


class RiskParityOptimizer:
    """
    风险平价优化器
    
    实现风险平价投资组合分配：
    - 等风险贡献
    - 基于波动率的风险平价
    - 基于协方差的风险平价
    """
    
    def __init__(self):
        """初始化风险平价优化器"""
        self.cov_matrix = None
        self.assets = None
        
    def fit(self, returns: pd.DataFrame) -> None:
        """
        拟合模型数据
        
        Args:
            returns: 收益率数据
        """
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
        
    def calculate_risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """
        计算风险贡献
        
        Args:
            weights: 资产权重
            
        Returns:
            np.ndarray: 各资产的风险贡献
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        portfolio_variance = weights.T @ self.cov_matrix @ weights
        marginal_risk_contribution = (self.cov_matrix @ weights) / portfolio_variance
        risk_contribution = weights * marginal_risk_contribution
        
        return risk_contribution
    
    def risk_parity_objective(self, weights: np.ndarray) -> float:
        """
        风险平价目标函数
        
        Args:
            weights: 资产权重
            
        Returns:
            float: 目标函数值
        """
        risk_contributions = self.calculate_risk_contribution(weights)
        target_contribution = 1.0 / len(weights)
        
        # 计算风险贡献与目标贡献的差异
        deviation = np.sum((risk_contributions - target_contribution) ** 2)
        
        return deviation
    
    def naive_risk_parity(self) -> RiskParityResult:
        """
        朴素风险平价（基于波动率倒数）
        
        Returns:
            RiskParityResult: 风险平价结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算各资产波动率
        volatilities = np.sqrt(np.diag(self.cov_matrix))
        
        # 基于波动率倒数的权重
        inv_volatilities = 1.0 / volatilities
        weights = inv_volatilities / np.sum(inv_volatilities)
        
        # 计算风险贡献
        risk_contributions = self.calculate_risk_contribution(weights)
        total_risk = np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        # 计算风险平价得分（越低越好）
        target_contribution = 1.0 / len(weights)
        risk_parity_score = np.sum((risk_contributions - target_contribution) ** 2)
        
        return RiskParityResult(
            weights=weights,
            risk_contributions=risk_contributions,
            total_risk=total_risk,
            assets=self.assets,
            risk_parity_score=risk_parity_score,
            convergence_achieved=True
        )
    
    def optimized_risk_parity(self, max_iterations: int = 1000, 
                            tolerance: float = 1e-8) -> RiskParityResult:
        """
        优化风险平价
        
        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            
        Returns:
            RiskParityResult: 风险平价结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        n_assets = len(self.assets)
        
        # 约束条件：权重和为1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # 边界条件：权重非负
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = optimize.minimize(
            self.risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': tolerance}
        )
        
        # 计算风险贡献
        risk_contributions = self.calculate_risk_contribution(result.x)
        total_risk = np.sqrt(result.x.T @ self.cov_matrix @ result.x)
        
        # 计算风险平价得分
        target_contribution = 1.0 / n_assets
        risk_parity_score = np.sum((risk_contributions - target_contribution) ** 2)
        
        return RiskParityResult(
            weights=result.x,
            risk_contributions=risk_contributions,
            total_risk=total_risk,
            assets=self.assets,
            risk_parity_score=risk_parity_score,
            convergence_achieved=result.success
        )
    
    def ccr_risk_parity(self, max_iterations: int = 100) -> RiskParityResult:
        """
        CCR（循环坐标下降）风险平价算法
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            RiskParityResult: 风险平价结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        n_assets = len(self.assets)
        
        # 初始解：等权重
        weights = np.ones(n_assets) / n_assets
        
        for iteration in range(max_iterations):
            old_weights = weights.copy()
            
            for i in range(n_assets):
                # 计算其他资产的加权协方差
                other_weights = np.delete(weights, i)
                other_cov = np.delete(np.delete(self.cov_matrix, i, axis=0), i, axis=1)
                
                cov_with_others = np.delete(self.cov_matrix[i, :], i)
                
                # 计算最优权重
                numerator = np.sqrt(other_weights.T @ other_cov @ other_weights)
                denominator = np.sqrt(self.cov_matrix[i, i])
                
                if denominator > 0:
                    optimal_weight = numerator / denominator
                else:
                    optimal_weight = 0
                
                # 更新权重
                weights[i] = optimal_weight
                
                # 重新归一化
                weights /= np.sum(weights)
            
            # 检查收敛
            if np.max(np.abs(weights - old_weights)) < 1e-8:
                break
        
        # 计算风险贡献
        risk_contributions = self.calculate_risk_contribution(weights)
        total_risk = np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        # 计算风险平价得分
        target_contribution = 1.0 / n_assets
        risk_parity_score = np.sum((risk_contributions - target_contribution) ** 2)
        
        return RiskParityResult(
            weights=weights,
            risk_contributions=risk_contributions,
            total_risk=total_risk,
            assets=self.assets,
            risk_parity_score=risk_parity_score,
            convergence_achieved=True
        )
    
    def constrained_risk_parity(self, max_weight_per_asset: float = 0.3,
                              min_weight_per_asset: float = 0.0) -> RiskParityResult:
        """
        带约束的风险平价
        
        Args:
            max_weight_per_asset: 单资产最大权重
            min_weight_per_asset: 单资产最小权重
            
        Returns:
            RiskParityResult: 风险平价结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        n_assets = len(self.assets)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # 边界条件
        bounds = [(min_weight_per_asset, max_weight_per_asset) for _ in range(n_assets)]
        
        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = optimize.minimize(
            self.risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # 计算风险贡献
        risk_contributions = self.calculate_risk_contribution(result.x)
        total_risk = np.sqrt(result.x.T @ self.cov_matrix @ result.x)
        
        # 计算风险平价得分
        target_contribution = 1.0 / n_assets
        risk_parity_score = np.sum((risk_contributions - target_contribution) ** 2)
        
        return RiskParityResult(
            weights=result.x,
            risk_contributions=risk_contributions,
            total_risk=total_risk,
            assets=self.assets,
            risk_parity_score=risk_parity_score,
            convergence_achieved=result.success
        )
    
    def compare_methods(self) -> Dict[str, RiskParityResult]:
        """
        比较不同风险平价方法
        
        Returns:
            Dict[str, RiskParityResult]: 各种方法的结果
        """
        results = {}
        
        # 朴素风险平价
        results['naive'] = self.naive_risk_parity()
        
        # 优化风险平价
        results['optimized'] = self.optimized_risk_parity()
        
        # CCR风险平价
        results['ccr'] = self.ccr_risk_parity()
        
        # 带约束风险平价
        results['constrained'] = self.constrained_risk_parity()
        
        return results
    
    def calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """
        计算分散化比率
        
        Args:
            weights: 资产权重
            
        Returns:
            float: 分散化比率
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 加权平均波动率
        weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(self.cov_matrix)))
        
        # 投资组合波动率
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        # 分散化比率
        diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        return diversification_ratio
    
    def risk_budgeting(self, risk_budgets: np.ndarray) -> RiskParityResult:
        """
        风险预算分配
        
        Args:
            risk_budgets: 风险预算（和为1）
            
        Returns:
            RiskParityResult: 风险预算结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        n_assets = len(self.assets)
        
        if len(risk_budgets) != n_assets:
            raise ValueError("风险预算数量必须与资产数量相同")
            
        if not np.isclose(np.sum(risk_budgets), 1.0):
            raise ValueError("风险预算和必须为1")
        
        # 目标函数：风险贡献与风险预算的差异
        def risk_budget_objective(weights):
            risk_contributions = self.calculate_risk_contribution(weights)
            deviation = np.sum((risk_contributions - risk_budgets) ** 2)
            return deviation
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # 边界条件
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = optimize.minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # 计算风险贡献
        risk_contributions = self.calculate_risk_contribution(result.x)
        total_risk = np.sqrt(result.x.T @ self.cov_matrix @ result.x)
        
        # 计算风险预算得分
        risk_budget_score = np.sum((risk_contributions - risk_budgets) ** 2)
        
        return RiskParityResult(
            weights=result.x,
            risk_contributions=risk_contributions,
            total_risk=total_risk,
            assets=self.assets,
            risk_parity_score=risk_budget_score,
            convergence_achieved=result.success
        )

"""
Black-Litterman模型
实现结合市场均衡和主观观点的投资组合优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class View:
    """主观观点"""
    assets: List[str]  # 涉及资产
    coefficients: np.ndarray  # 观点系数
    return_view: float  # 观点收益
    confidence: float  # 置信度


@dataclass
class BlackLittermanResult:
    """Black-Litterman结果"""
    equilibrium_returns: np.ndarray
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    weights: np.ndarray
    assets: List[str]
    view_impact: np.ndarray


class BlackLittermanModel:
    """
    Black-Litterman模型
    
    实现结合市场均衡和主观观点的投资组合优化：
    - 市场均衡收益计算
    - 主观观点整合
    - 后验收益估计
    """
    
    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        """
        初始化Black-Litterman模型
        
        Args:
            risk_aversion: 风险厌恶系数
            tau: 不确定性缩放因子
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.cov_matrix = None
        self.assets = None
        self.market_weights = None
        
    def fit(self, returns: pd.DataFrame, market_weights: Optional[np.ndarray] = None) -> None:
        """
        拟合模型数据
        
        Args:
            returns: 收益率数据
            market_weights: 市场权重，如为None则使用等权重
        """
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
        
        if market_weights is None:
            self.market_weights = np.ones(len(self.assets)) / len(self.assets)
        else:
            self.market_weights = market_weights
            
    def calculate_equilibrium_returns(self) -> np.ndarray:
        """
        计算市场均衡收益
        
        Returns:
            np.ndarray: 均衡收益
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 均衡收益 = 风险厌恶系数 × 协方差矩阵 × 市场权重
        equilibrium_returns = self.risk_aversion * self.cov_matrix @ self.market_weights
        
        return equilibrium_returns
    
    def create_view_matrix(self, views: List[View]) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建观点矩阵
        
        Args:
            views: 主观观点列表
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (观点矩阵P, 观点收益向量Q)
        """
        n_assets = len(self.assets)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, view in enumerate(views):
            # 创建观点系数向量
            for j, asset in enumerate(view.assets):
                asset_index = self.assets.index(asset)
                P[i, asset_index] = view.coefficients[j]
            
            Q[i] = view.return_view
        
        return P, Q
    
    def calculate_omega(self, views: List[View], P: np.ndarray) -> np.ndarray:
        """
        计算观点不确定性矩阵Ω
        
        Args:
            views: 主观观点列表
            P: 观点矩阵
            
        Returns:
            np.ndarray: 观点不确定性矩阵
        """
        n_views = len(views)
        omega = np.zeros((n_views, n_views))
        
        for i, view in enumerate(views):
            # 观点不确定性 = 观点系数 × (τΣ) × 观点系数转置
            p_i = P[i, :].reshape(1, -1)
            omega_ii = p_i @ (self.tau * self.cov_matrix) @ p_i.T
            
            # 调整置信度
            if view.confidence > 0:
                omega_ii /= view.confidence
            
            omega[i, i] = omega_ii[0, 0]
        
        return omega
    
    def calculate_posterior_returns(self, views: List[View]) -> BlackLittermanResult:
        """
        计算后验收益
        
        Args:
            views: 主观观点列表
            
        Returns:
            BlackLittermanResult: Black-Litterman结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")
            
        # 计算均衡收益
        equilibrium_returns = self.calculate_equilibrium_returns()
        
        # 如果没有观点，直接返回均衡收益
        if not views:
            return BlackLittermanResult(
                equilibrium_returns=equilibrium_returns,
                posterior_returns=equilibrium_returns,
                posterior_covariance=self.cov_matrix,
                weights=self.market_weights,
                assets=self.assets,
                view_impact=np.zeros(len(self.assets))
            )
        
        # 创建观点矩阵
        P, Q = self.create_view_matrix(views)
        
        # 计算观点不确定性矩阵
        omega = self.calculate_omega(views, P)
        
        # 计算后验收益
        # E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        
        tau_sigma = self.tau * self.cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)
        
        # 计算后验收益协方差矩阵
        posterior_cov_inv = tau_sigma_inv + P.T @ omega_inv @ P
        posterior_covariance = np.linalg.inv(posterior_cov_inv)
        
        # 计算后验收益
        posterior_returns = posterior_covariance @ (tau_sigma_inv @ equilibrium_returns + P.T @ omega_inv @ Q)
        
        # 计算观点影响
        view_impact = posterior_returns - equilibrium_returns
        
        # 计算最优权重
        weights = (1 / self.risk_aversion) * np.linalg.inv(self.cov_matrix) @ posterior_returns
        weights = weights / np.sum(weights)  # 归一化
        
        return BlackLittermanResult(
            equilibrium_returns=equilibrium_returns,
            posterior_returns=posterior_returns,
            posterior_covariance=posterior_covariance,
            weights=weights,
            assets=self.assets,
            view_impact=view_impact
        )
    
    def create_absolute_view(self, asset: str, expected_return: float, 
                           confidence: float = 0.5) -> View:
        """
        创建绝对观点
        
        Args:
            asset: 资产
            expected_return: 预期收益
            confidence: 置信度
            
        Returns:
            View: 绝对观点
        """
        asset_index = self.assets.index(asset)
        coefficients = np.zeros(len(self.assets))
        coefficients[asset_index] = 1.0
        
        return View(
            assets=[asset],
            coefficients=coefficients,
            return_view=expected_return,
            confidence=confidence
        )
    
    def create_relative_view(self, outperforming_assets: List[str], 
                           underperforming_assets: List[str],
                           performance_spread: float, 
                           confidence: float = 0.5) -> View:
        """
        创建相对观点
        
        Args:
            outperforming_assets: 表现优异的资产
            underperforming_assets: 表现较差的资产
            performance_spread: 表现差异
            confidence: 置信度
            
        Returns:
            View: 相对观点
        """
        coefficients = np.zeros(len(self.assets))
        
        # 优异资产系数为正
        for asset in outperforming_assets:
            asset_index = self.assets.index(asset)
            coefficients[asset_index] = 1.0 / len(outperforming_assets)
        
        # 较差资产系数为负
        for asset in underperforming_assets:
            asset_index = self.assets.index(asset)
            coefficients[asset_index] = -1.0 / len(underperforming_assets)
        
        return View(
            assets=outperforming_assets + underperforming_assets,
            coefficients=coefficients,
            return_view=performance_spread,
            confidence=confidence
        )
    
    def sensitivity_analysis(self, views: List[View], 
                           parameter: str, values: List[float]) -> Dict[str, BlackLittermanResult]:
        """
        敏感性分析
        
        Args:
            views: 主观观点
            parameter: 参数名称
            values: 参数值
            
        Returns:
            Dict[str, BlackLittermanResult]: 敏感性分析结果
        """
        results = {}
        
        original_risk_aversion = self.risk_aversion
        original_tau = self.tau
        
        for value in values:
            if parameter == 'risk_aversion':
                self.risk_aversion = value
            elif parameter == 'tau':
                self.tau = value
            elif parameter == 'confidence':
                # 更新所有观点的置信度
                for view in views:
                    view.confidence = value
            else:
                raise ValueError(f"不支持的参数: {parameter}")
            
            result = self.calculate_posterior_returns(views)
            results[str(value)] = result
        
        # 恢复原始参数
        self.risk_aversion = original_risk_aversion
        self.tau = original_tau
        
        return results
    
    def calculate_view_implied_returns(self, views: List[View]) -> np.ndarray:
        """
        计算观点隐含收益
        
        Args:
            views: 主观观点
            
        Returns:
            np.ndarray: 观点隐含收益
        """
        if not views:
            return np.zeros(len(self.assets))
        
        P, Q = self.create_view_matrix(views)
        omega = self.calculate_omega(views, P)
        
        # 观点隐含收益 = P' × (P P')^(-1) × Q
        # 这是一个简化的计算，实际中更复杂
        
        P_pseudo_inv = np.linalg.pinv(P)
        view_implied_returns = P_pseudo_inv @ Q
        
        return view_implied_returns
    
    def compare_approaches(self, views: List[View]) -> Dict[str, np.ndarray]:
        """
        比较不同方法的结果
        
        Args:
            views: 主观观点
            
        Returns:
            Dict[str, np.ndarray]: 不同方法的结果
        """
        results = {}
        
        # 市场均衡方法
        equilibrium_returns = self.calculate_equilibrium_returns()
        results['equilibrium'] = equilibrium_returns
        
        # 纯主观方法
        if views:
            view_implied_returns = self.calculate_view_implied_returns(views)
            results['subjective'] = view_implied_returns
        
        # Black-Litterman方法
        bl_result = self.calculate_posterior_returns(views)
        results['black_litterman'] = bl_result.posterior_returns
        
        return results
    
    def calculate_information_ratio(self, views: List[View]) -> float:
        """
        计算信息比率
        
        Args:
            views: 主观观点
            
        Returns:
            float: 信息比率
        """
        if not views:
            return 0.0
        
        bl_result = self.calculate_posterior_returns(views)
        
        # 主动收益 = 后验收益 - 均衡收益
        active_returns = bl_result.posterior_returns - bl_result.equilibrium_returns
        
        # 主动风险 = 主动收益的标准差
        active_risk = np.sqrt(np.var(active_returns))
        
        # 信息比率 = 平均主动收益 / 主动风险
        information_ratio = np.mean(active_returns) / active_risk if active_risk > 0 else 0.0
        
        return information_ratio

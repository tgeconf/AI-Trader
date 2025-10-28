"""
多因子绩效归因模型
实现基于因子模型的绩效归因分析，包括Fama-French、Carhart等模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression


class FactorModel(Enum):
    """因子模型枚举"""

    FAMA_FRENCH_3F = "fama_french_3f"  # Fama-French三因子模型
    FAMA_FRENCH_5F = "fama_french_5f"  # Fama-French五因子模型
    CARHART_4F = "carhart_4f"  # Carhart四因子模型
    APT = "apt"  # 套利定价理论模型
    CUSTOM = "custom"  # 自定义因子模型


@dataclass
class FactorAttributionResult:
    """因子归因结果"""

    alpha: float  # 超额收益（Alpha）
    factor_contributions: Dict[str, float]  # 因子贡献
    r_squared: float  # 模型拟合度
    factor_significance: Dict[str, Dict[str, float]]  # 因子显著性
    residual_analysis: Dict[str, Any]  # 残差分析
    model_summary: Dict[str, Any]  # 模型摘要


@dataclass
class FactorExposure:
    """因子暴露"""

    factor_name: str
    exposure_values: np.ndarray
    exposure_weights: Optional[np.ndarray] = None


class MultiFactorAttribution:
    """
    多因子绩效归因模型

    实现基于因子模型的绩效归因分析：
    - Fama-French三因子模型
    - Fama-French五因子模型
    - Carhart四因子模型
    - 套利定价理论模型
    - 自定义因子模型
    """

    def __init__(self, significance_level: float = 0.05, min_observations: int = 30):
        """
        初始化多因子归因模型

        Args:
            significance_level: 显著性水平
            min_observations: 最小观测数
        """
        self.significance_level = significance_level
        self.min_observations = min_observations
        self.portfolio_returns = None
        self.factor_returns = None
        self.factor_exposures = None
        self.model_results = {}

    def fit(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: Dict[str, np.ndarray],
        factor_exposures: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        拟合因子模型

        Args:
            portfolio_returns: 投资组合收益率序列
            factor_returns: 因子收益率数据
            factor_exposures: 因子暴露数据
        """
        # 数据验证
        n_obs = len(portfolio_returns)
        if n_obs < self.min_observations:
            raise ValueError(
                f"Insufficient observations: {n_obs} < {self.min_observations}"
            )

        for factor_name, factor_data in factor_returns.items():
            if len(factor_data) != n_obs:
                raise ValueError(f"Factor {factor_name} has inconsistent length")

        self.portfolio_returns = portfolio_returns
        self.factor_returns = factor_returns

        # 如果没有提供因子暴露，假设为常数暴露
        if factor_exposures is None:
            self.factor_exposures = {
                factor_name: np.ones(n_obs) for factor_name in factor_returns.keys()
            }
        else:
            self.factor_exposures = factor_exposures

    def fama_french_three_factor(self) -> FactorAttributionResult:
        """
        Fama-French三因子模型归因

        Returns:
            三因子模型归因结果
        """
        required_factors = ["mkt_rf", "smb", "hml"]
        self._validate_factor_availability(required_factors)

        # 准备因子数据
        X = np.column_stack(
            [
                self.factor_returns["mkt_rf"],
                self.factor_returns["smb"],
                self.factor_returns["hml"],
            ]
        )

        # 添加常数项（Alpha）
        X = sm.add_constant(X)
        y = self.portfolio_returns

        # 执行回归
        model = sm.OLS(y, X).fit()

        return self._build_factor_result(model, ["alpha", "mkt_rf", "smb", "hml"])

    def fama_french_five_factor(self) -> FactorAttributionResult:
        """
        Fama-French五因子模型归因

        Returns:
            五因子模型归因结果
        """
        required_factors = ["mkt_rf", "smb", "hml", "rmw", "cma"]
        self._validate_factor_availability(required_factors)

        # 准备因子数据
        X = np.column_stack(
            [
                self.factor_returns["mkt_rf"],
                self.factor_returns["smb"],
                self.factor_returns["hml"],
                self.factor_returns["rmw"],
                self.factor_returns["cma"],
            ]
        )

        # 添加常数项（Alpha）
        X = sm.add_constant(X)
        y = self.portfolio_returns

        # 执行回归
        model = sm.OLS(y, X).fit()

        return self._build_factor_result(
            model, ["alpha", "mkt_rf", "smb", "hml", "rmw", "cma"]
        )

    def carhart_four_factor(self) -> FactorAttributionResult:
        """
        Carhart四因子模型归因

        Returns:
            四因子模型归因结果
        """
        required_factors = ["mkt_rf", "smb", "hml", "mom"]
        self._validate_factor_availability(required_factors)

        # 准备因子数据
        X = np.column_stack(
            [
                self.factor_returns["mkt_rf"],
                self.factor_returns["smb"],
                self.factor_returns["hml"],
                self.factor_returns["mom"],
            ]
        )

        # 添加常数项（Alpha）
        X = sm.add_constant(X)
        y = self.portfolio_returns

        # 执行回归
        model = sm.OLS(y, X).fit()

        return self._build_factor_result(
            model, ["alpha", "mkt_rf", "smb", "hml", "mom"]
        )

    def fama_macbeth_regression(self) -> FactorAttributionResult:
        """
        Fama-MacBeth两阶段回归

        Returns:
            Fama-MacBeth回归结果
        """
        if self.factor_exposures is None:
            raise ValueError("Factor exposures required for Fama-MacBeth regression")

        # 第一阶段：横截面回归估计因子暴露
        n_obs = len(self.portfolio_returns)
        n_factors = len(self.factor_returns)

        # 假设因子暴露已知，直接使用
        factor_names = list(self.factor_returns.keys())

        # 第二阶段：时间序列回归
        X = np.column_stack([self.factor_returns[factor] for factor in factor_names])
        X = sm.add_constant(X)
        y = self.portfolio_returns

        model = sm.OLS(y, X).fit()

        return self._build_factor_result(model, ["alpha"] + factor_names)

    def custom_factor_model(
        self,
        custom_factors: List[str],
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> FactorAttributionResult:
        """
        自定义因子模型归因

        Args:
            custom_factors: 自定义因子列表
            factor_weights: 因子权重

        Returns:
            自定义因子模型归因结果
        """
        self._validate_factor_availability(custom_factors)

        # 准备因子数据
        X = np.column_stack([self.factor_returns[factor] for factor in custom_factors])

        # 添加常数项（Alpha）
        X = sm.add_constant(X)
        y = self.portfolio_returns

        # 执行回归
        model = sm.OLS(y, X).fit()

        return self._build_factor_result(model, ["alpha"] + custom_factors)

    def rolling_factor_attribution(
        self, window: int = 60, model_type: FactorModel = FactorModel.FAMA_FRENCH_3F
    ) -> Dict[str, Any]:
        """
        滚动因子归因分析

        Args:
            window: 滚动窗口大小
            model_type: 因子模型类型

        Returns:
            滚动归因结果
        """
        n_obs = len(self.portfolio_returns)
        if n_obs < window:
            raise ValueError(
                f"Insufficient data for rolling analysis: {n_obs} < {window}"
            )

        rolling_results = {
            "alpha_series": [],
            "factor_contributions_series": {},
            "r_squared_series": [],
            "dates": [],  # 假设有时间索引
        }

        # 初始化因子贡献序列
        factor_names = self._get_factor_names_for_model(model_type)
        for factor in factor_names[1:]:  # 跳过alpha
            rolling_results["factor_contributions_series"][factor] = []

        # 执行滚动回归
        for i in range(window, n_obs):
            # 提取窗口数据
            portfolio_window = self.portfolio_returns[i - window : i]
            factor_window = {
                factor: returns[i - window : i]
                for factor, returns in self.factor_returns.items()
            }

            # 拟合模型
            try:
                self.fit(portfolio_window, factor_window)

                if model_type == FactorModel.FAMA_FRENCH_3F:
                    result = self.fama_french_three_factor()
                elif model_type == FactorModel.FAMA_FRENCH_5F:
                    result = self.fama_french_five_factor()
                elif model_type == FactorModel.CARHART_4F:
                    result = self.carhart_four_factor()
                else:
                    result = self.fama_french_three_factor()  # 默认

                # 存储结果
                rolling_results["alpha_series"].append(result.alpha)
                rolling_results["r_squared_series"].append(result.r_squared)

                for factor, contribution in result.factor_contributions.items():
                    if factor != "alpha":
                        rolling_results["factor_contributions_series"][factor].append(
                            contribution
                        )

            except Exception as e:
                # 处理回归失败的情况
                rolling_results["alpha_series"].append(np.nan)
                rolling_results["r_squared_series"].append(np.nan)
                for factor in factor_names[1:]:
                    rolling_results["factor_contributions_series"][factor].append(
                        np.nan
                    )

        return rolling_results

    def _validate_factor_availability(self, required_factors: List[str]) -> None:
        """验证因子数据可用性"""
        missing_factors = [
            factor for factor in required_factors if factor not in self.factor_returns
        ]
        if missing_factors:
            raise ValueError(f"Missing required factors: {missing_factors}")

    def _get_factor_names_for_model(self, model_type: FactorModel) -> List[str]:
        """获取指定模型的因子名称"""
        if model_type == FactorModel.FAMA_FRENCH_3F:
            return ["alpha", "mkt_rf", "smb", "hml"]
        elif model_type == FactorModel.FAMA_FRENCH_5F:
            return ["alpha", "mkt_rf", "smb", "hml", "rmw", "cma"]
        elif model_type == FactorModel.CARHART_4F:
            return ["alpha", "mkt_rf", "smb", "hml", "mom"]
        else:
            return ["alpha"] + list(self.factor_returns.keys())

    def _build_factor_result(
        self,
        model: sm.regression.linear_model.RegressionResults,
        factor_names: List[str],
    ) -> FactorAttributionResult:
        """构建因子归因结果"""
        # 提取系数
        coefficients = model.params
        p_values = model.pvalues
        t_stats = model.tvalues

        # 计算因子贡献
        factor_contributions = {}
        for i, factor in enumerate(factor_names):
            factor_contributions[factor] = coefficients[i]

        # 因子显著性分析
        factor_significance = {}
        for i, factor in enumerate(factor_names):
            factor_significance[factor] = {
                "coefficient": coefficients[i],
                "p_value": p_values[i],
                "t_statistic": t_stats[i],
                "significant": p_values[i] < self.significance_level,
            }

        # 残差分析
        residuals = model.resid
        residual_analysis = {
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "skewness": stats.skew(residuals),
            "kurtosis": stats.kurtosis(residuals),
            "jarque_bera_stat": stats.jarque_bera(residuals)[0],
            "jarque_bera_pvalue": stats.jarque_bera(residuals)[1],
            "durbin_watson": sm.stats.stattools.durbin_watson(residuals),
        }

        # 模型摘要
        model_summary = {
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "aic": model.aic,
            "bic": model.bic,
            "condition_number": np.linalg.cond(model.model.exog),
        }

        return FactorAttributionResult(
            alpha=coefficients[0],  # 第一个系数是alpha
            factor_contributions=factor_contributions,
            r_squared=model.rsquared,
            factor_significance=factor_significance,
            residual_analysis=residual_analysis,
            model_summary=model_summary,
        )

    def calculate_factor_contribution_breakdown(
        self, result: FactorAttributionResult
    ) -> Dict[str, Any]:
        """
        计算因子贡献分解

        Args:
            result: 因子归因结果

        Returns:
            因子贡献分解
        """
        total_contribution = sum(
            abs(contribution)
            for factor, contribution in result.factor_contributions.items()
            if factor != "alpha"
        )

        if total_contribution == 0:
            return {}

        # 计算相对贡献
        relative_contributions = {}
        for factor, contribution in result.factor_contributions.items():
            if factor != "alpha":
                relative_contributions[factor] = (
                    abs(contribution) / total_contribution * 100
                )

        # 识别主要贡献因子
        primary_contributors = sorted(
            relative_contributions.items(), key=lambda x: x[1], reverse=True
        )[
            :3
        ]  # 前三个主要贡献因子

        return {
            "relative_contributions": relative_contributions,
            "primary_contributors": primary_contributors,
            "alpha_contribution": result.alpha,
            "alpha_significance": result.factor_significance["alpha"]["significant"],
        }

    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        return {
            "significance_level": self.significance_level,
            "min_observations": self.min_observations,
            "available_factors": (
                list(self.factor_returns.keys()) if self.factor_returns else []
            ),
            "model_results_count": len(self.model_results),
        }

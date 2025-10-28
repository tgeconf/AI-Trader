"""
压力测试模块
实现机构级压力测试体系，包括情景生成、组合模拟、风险分析与报告输出
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cvar_model import CVaRModel
from .var_model import VaRModel


class ScenarioType(Enum):
    """压力情景类型"""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    CUSTOM = "custom"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class MarketScenario:
    """
    市场情景定义
    
    Attributes:
        name: 情景名称
        scenario_type: 情景类型
        description: 情景描述
        price_shocks: 价格冲击参数（__global__ 或资产级别）
        factor_shocks: 因子冲击参数（如利率、汇率等）
        liquidity_shocks: 流动性冲击参数
        volatility_multiplier: 波动率放大倍数
        correlation_shock: 冲击后的相关性矩阵
        duration: 情景持续时间
        metadata: 额外元数据（如历史时期、来源）
    """
    name: str
    scenario_type: ScenarioType
    description: str
    price_shocks: Dict[str, float] = field(default_factory=dict)
    factor_shocks: Dict[str, float] = field(default_factory=dict)
    liquidity_shocks: Dict[str, float] = field(default_factory=dict)
    volatility_multiplier: float = 1.0
    correlation_shock: Optional[pd.DataFrame] = None
    duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestConfig:
    """
    压力测试配置
    
    包含资本、流动性、市场风险阈值以及模拟参数
    """
    initial_portfolio_value: float = 1_000_000.0
    available_capital: float = 5_000_000.0
    tier1_capital: float = 4_000_000.0
    total_capital: float = 6_000_000.0
    risk_weighted_assets: float = 50_000_000.0
    total_exposure: float = 120_000_000.0
    liquidity_buffer: float = 1_500_000.0
    net_outflow_5day: float = 1_200_000.0
    stable_funding: float = 20_000_000.0
    required_stable_funding: float = 18_000_000.0
    rebalance: bool = False
    rebalance_frequency: int = 20
    transaction_cost_bp: float = 5.0
    var_confidences: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.995])
    capital_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "tier1_ratio": 0.08,
            "total_capital_ratio": 0.105,
            "leverage_ratio": 0.03,
            "capital_buffer": 0.0,
        }
    )
    liquidity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "lcr": 1.0,
            "nsfr": 1.0,
            "cash_buffer_days": 5.0,
        }
    )
    market_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "max_daily_loss": 0.02,
            "max_drawdown": 0.15,
        }
    )
    factor_exposures: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scenario_selection: Optional[List[str]] = None
    sensitivity_shock: float = 0.05


@dataclass
class PortfolioSimulationResult:
    """
    压力测试组合模拟结果
    
    Attributes:
        scenario: 对应的市场情景
        portfolio_values: 投资组合价值路径
        portfolio_returns: 投资组合收益率序列
        drawdowns: 回撤路径
        turnover: 换手率序列
        transaction_costs: 交易成本序列
        liquidity_shortfalls: 流动性缺口序列
        asset_values: 各资产持仓价值路径
        adjusted_returns: 情景下的资产收益率
        diagnostics: 诊断信息（如缺失资产、施加的冲击参数）
    """
    scenario: MarketScenario
    portfolio_values: pd.Series
    portfolio_returns: pd.Series
    drawdowns: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    liquidity_shortfalls: pd.Series
    asset_values: pd.DataFrame
    adjusted_returns: pd.DataFrame
    diagnostics: Dict[str, Any]


@dataclass
class StressTestResult:
    """压力测试分析结果"""
    scenario: MarketScenario
    risk_metrics: Dict[str, float]
    capital_metrics: Dict[str, float]
    liquidity_metrics: Dict[str, float]
    sensitivity_analysis: Dict[str, Any]
    breaches: Dict[str, bool]
    diagnostics: Dict[str, Any]


HISTORICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "2008_crisis": {
        "description": "全球金融危机：股票暴跌、信用利差急剧扩张、流动性枯竭",
        "price_shocks": {"__global__": -0.50},
        "volatility_multiplier": 5.0,
        "liquidity_shocks": {"liquidity_drop": 0.70, "volume_drop": 0.60},
        "duration_days": 126,
        "metadata": {"period": "2008-09-01/2009-03-01"},
    },
    "covid_19": {
        "description": "2020年疫情冲击：突发性跌幅、波动率飙升，快速反弹",
        "price_shocks": {"__global__": -0.35},
        "volatility_multiplier": 7.0,
        "liquidity_shocks": {"liquidity_drop": 0.45, "volume_drop": 0.40},
        "duration_days": 63,
        "metadata": {"period": "2020-02-15/2020-05-15"},
    },
    "black_monday_1987": {
        "description": "1987年黑色星期一：单日高达20%以上的极端跌幅",
        "price_shocks": {"__global__": -0.30},
        "volatility_multiplier": 4.0,
        "liquidity_shocks": {"liquidity_drop": 0.50},
        "duration_days": 20,
        "metadata": {"period": "1987-10-19/1987-11-30"},
    },
    "asia_1997": {
        "description": "亚洲金融危机：货币贬值、资本流出、区域市场系统性风险",
        "price_shocks": {"__global__": -0.25},
        "volatility_multiplier": 3.5,
        "factor_shocks": {"currency_depreciation": -0.20},
        "liquidity_shocks": {"liquidity_drop": 0.55},
        "duration_days": 84,
        "metadata": {"period": "1997-07-01/1997-12-31"},
    },
    "euro_debt_2011": {
        "description": "欧洲主权债务危机：信用利差扩张、风险偏好骤降",
        "price_shocks": {"__global__": -0.20},
        "volatility_multiplier": 3.0,
        "factor_shocks": {"credit_spread": 0.015},
        "liquidity_shocks": {"liquidity_drop": 0.40},
        "duration_days": 90,
        "metadata": {"period": "2011-08-01/2011-11-30"},
    },
}


HYPOTHETICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "interest_rate_shock": {
        "description": "利率突升300bp，长久期资产大幅调整",
        "factor_shocks": {"rate_shift_bp": 300},
        "price_shocks": {"fixed_income": -0.15},
        "volatility_multiplier": 1.8,
        "duration_days": 30,
    },
    "currency_crisis": {
        "description": "主要货币贬值20%，输入型通胀抬升8%",
        "factor_shocks": {"currency_depreciation": -0.20, "import_inflation": 0.08},
        "price_shocks": {"__global__": -0.12},
        "volatility_multiplier": 2.2,
        "duration_days": 45,
    },
    "liquidity_crunch": {
        "description": "流动性枯竭：成交量下降80%，融资被迫收缩",
        "price_shocks": {"__global__": -0.18},
        "liquidity_shocks": {"volume_drop": 0.80, "liquidity_drop": 0.65},
        "volatility_multiplier": 2.5,
        "duration_days": 40,
    },
    "correlation_collapse": {
        "description": "资产相关性崩塌，先前互补资产同步下跌",
        "price_shocks": {"__global__": -0.22},
        "volatility_multiplier": 2.8,
        "duration_days": 50,
    },
    "volatility_spike": {
        "description": "隐含波动率飙升，VIX突破80",
        "price_shocks": {"__global__": -0.14},
        "volatility_multiplier": 4.5,
        "duration_days": 25,
    },
}


class ScenarioGenerator:
    """情景生成器"""

    def __init__(
        self,
        historical_scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
        hypothetical_scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.historical_scenarios = historical_scenarios or HISTORICAL_SCENARIOS
        self.hypothetical_scenarios = hypothetical_scenarios or HYPOTHETICAL_SCENARIOS

    def generate_historical_scenario(self, crisis_period: str) -> MarketScenario:
        """生成历史情景"""
        if crisis_period not in self.historical_scenarios:
            raise ValueError(f"未定义的历史情景: {crisis_period}")

        params = self.historical_scenarios[crisis_period]
        duration = timedelta(days=params.get("duration_days", 60))
        return MarketScenario(
            name=crisis_period,
            scenario_type=ScenarioType.HISTORICAL,
            description=params.get("description", ""),
            price_shocks=params.get("price_shocks", {}),
            factor_shocks=params.get("factor_shocks", {}),
            liquidity_shocks=params.get("liquidity_shocks", {}),
            volatility_multiplier=params.get("volatility_multiplier", 1.0),
            duration=duration,
            metadata=params.get("metadata", {}),
        )

    def generate_hypothetical_scenario(
        self, name: str, shock_params: Optional[Dict[str, Any]] = None
    ) -> MarketScenario:
        """生成假设情景"""
        base_params = self.hypothetical_scenarios.get(name, {})
        if not base_params and shock_params is None:
            raise ValueError(f"未定义的假设情景: {name}")

        params = {**base_params, **(shock_params or {})}
        duration = timedelta(days=params.get("duration_days", 30))
        return MarketScenario(
            name=name,
            scenario_type=ScenarioType.HYPOTHETICAL,
            description=params.get("description", ""),
            price_shocks=params.get("price_shocks", {}),
            factor_shocks=params.get("factor_shocks", {}),
            liquidity_shocks=params.get("liquidity_shocks", {}),
            volatility_multiplier=params.get("volatility_multiplier", 1.0),
            duration=duration,
            metadata=params.get("metadata", {}),
        )

    def generate_correlation_breakdown(
        self, base_correlations: pd.DataFrame, target_level: float = 0.0, name: str = "correlation_breakdown"
    ) -> MarketScenario:
        """生成相关性崩溃情景"""
        shocked = base_correlations.copy()
        for i in shocked.index:
            for j in shocked.columns:
                if i == j:
                    continue
                shocked.loc[i, j] = target_level + 0.2 * (shocked.loc[i, j] - target_level)

        return MarketScenario(
            name=name,
            scenario_type=ScenarioType.CORRELATION_BREAKDOWN,
            description="相关性崩溃：资产协动性急剧下降，分散化失效",
            correlation_shock=shocked,
            volatility_multiplier=2.5,
            price_shocks={"__global__": -0.18},
            metadata={"base_correlation": base_correlations},
        )

    def generate_liquidity_crisis(self, volume_drop: float, name: str = "custom_liquidity_crisis") -> MarketScenario:
        """生成流动性危机情景"""
        volume_drop = np.clip(volume_drop, 0.0, 0.95)
        liquidity_drop = min(0.9, volume_drop * 0.85)
        return MarketScenario(
            name=name,
            scenario_type=ScenarioType.LIQUIDITY_CRISIS,
            description="自定义流动性危机：成交量骤降，做市深度枯竭",
            price_shocks={"__global__": -0.16},
            liquidity_shocks={"volume_drop": volume_drop, "liquidity_drop": liquidity_drop},
            volatility_multiplier=2.2,
        )

    def build_custom_scenario(
        self,
        name: str,
        description: str,
        price_shocks: Optional[Dict[str, float]] = None,
        factor_shocks: Optional[Dict[str, float]] = None,
        liquidity_shocks: Optional[Dict[str, float]] = None,
        volatility_multiplier: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MarketScenario:
        """构建自定义情景"""
        return MarketScenario(
            name=name,
            scenario_type=ScenarioType.CUSTOM,
            description=description,
            price_shocks=price_shocks or {},
            factor_shocks=factor_shocks or {},
            liquidity_shocks=liquidity_shocks or {},
            volatility_multiplier=volatility_multiplier,
            metadata=metadata or {},
        )


class PortfolioSimulator:
    """组合模拟器"""

    def __init__(self, config: StressTestConfig) -> None:
        self.config = config

    def simulate(
        self,
        portfolio_weights: Dict[str, float],
        market_returns: pd.DataFrame,
        scenario: MarketScenario,
    ) -> PortfolioSimulationResult:
        """
        在给定情景下模拟组合表现
        
        Args:
            portfolio_weights: 资产权重（可含空头，需和市场数据对齐）
            market_returns: 基础收益率数据（index为时间，columns为资产）
            scenario: 压力情景
        """
        if market_returns.empty:
            raise ValueError("市场收益率数据为空，无法执行压力测试")

        weights = pd.Series(portfolio_weights, dtype=float)
        weights = weights[weights.abs() > 0]
        weights = weights.sort_index()

        missing_assets = [asset for asset in weights.index if asset not in market_returns.columns]
        if missing_assets:
            weights = weights.drop(missing_assets, errors="ignore")
        if weights.empty:
            raise ValueError("投资组合权重为空或与市场数据无交集")

        weights /= weights.sum()
        returns = market_returns[weights.index].fillna(0.0).copy()
        adjusted_returns = self._apply_scenario_shocks(returns, weights.index, scenario)

        initial_positions = weights * self.config.initial_portfolio_value
        positions = initial_positions.copy()

        portfolio_values: List[float] = []
        drawdowns: List[float] = []
        turnover: List[float] = []
        transaction_costs: List[float] = []
        liquidity_shortfalls: List[float] = []
        asset_values_records: List[pd.Series] = []

        running_max = self.config.initial_portfolio_value
        rebalance_frequency = max(1, self.config.rebalance_frequency)

        liquidity_shortfall_amount = self._estimate_liquidity_shortfall(scenario)

        for idx, (_, asset_ret) in enumerate(adjusted_returns.iterrows()):
            asset_ret = asset_ret.fillna(0.0)
            positions = positions * (1.0 + asset_ret)
            portfolio_value = float(positions.sum())

            turnover_ratio = 0.0
            transaction_cost = 0.0

            if self.config.rebalance and ((idx + 1) % rebalance_frequency == 0):
                target_positions = weights * portfolio_value
                turnover_notional = float((target_positions - positions).abs().sum())
                turnover_ratio = turnover_notional / (2.0 * portfolio_value) if portfolio_value else 0.0
                transaction_cost = turnover_notional * (self.config.transaction_cost_bp / 10_000)
                portfolio_value = max(portfolio_value - transaction_cost, 0.0)
                if portfolio_value > 0:
                    scaling_factor = portfolio_value / target_positions.sum() if target_positions.sum() else 0.0
                    positions = target_positions * scaling_factor
                else:
                    positions = positions * 0.0
            else:
                target_positions = weights * portfolio_value
                turnover_ratio = (
                    float((positions - target_positions).abs().sum()) / (2.0 * portfolio_value) if portfolio_value else 0.0
                )

            running_max = max(running_max, portfolio_value)
            drawdown = (portfolio_value - running_max) / running_max if running_max else 0.0

            portfolio_values.append(portfolio_value)
            turnover.append(turnover_ratio)
            transaction_costs.append(transaction_cost)
            liquidity_shortfalls.append(liquidity_shortfall_amount)
            drawdowns.append(drawdown)
            asset_values_records.append(positions.copy())

        index = adjusted_returns.index
        portfolio_values_series = pd.Series(portfolio_values, index=index, name="portfolio_value")
        portfolio_returns = portfolio_values_series.pct_change().fillna(0.0)
        drawdown_series = pd.Series(drawdowns, index=index, name="drawdown")
        turnover_series = pd.Series(turnover, index=index, name="turnover")
        transaction_cost_series = pd.Series(transaction_costs, index=index, name="transaction_cost")
        liquidity_shortfall_series = pd.Series(liquidity_shortfalls, index=index, name="liquidity_shortfall")
        asset_values = pd.DataFrame(asset_values_records, index=index)

        diagnostics = {
            "missing_assets": missing_assets,
            "volatility_multiplier": scenario.volatility_multiplier,
            "applied_price_shocks": scenario.price_shocks,
            "applied_factor_shocks": scenario.factor_shocks,
            "liquidity_shortfall_amount": liquidity_shortfall_amount,
        }

        return PortfolioSimulationResult(
            scenario=scenario,
            portfolio_values=portfolio_values_series,
            portfolio_returns=portfolio_returns,
            drawdowns=drawdown_series,
            turnover=turnover_series,
            transaction_costs=transaction_cost_series,
            liquidity_shortfalls=liquidity_shortfall_series,
            asset_values=asset_values,
            adjusted_returns=adjusted_returns,
            diagnostics=diagnostics,
        )

    def _apply_scenario_shocks(
        self, returns: pd.DataFrame, assets: Sequence[str], scenario: MarketScenario
    ) -> pd.DataFrame:
        """将情景冲击施加到收益率序列"""
        shocked_returns = returns.copy()

        if scenario.volatility_multiplier != 1.0:
            shocked_returns = shocked_returns * scenario.volatility_multiplier

        global_shock = scenario.price_shocks.get("__global__")
        if global_shock is not None:
            shocked_returns.iloc[-1] += global_shock

        for asset, shock in scenario.price_shocks.items():
            if asset == "__global__":
                continue
            if asset in shocked_returns.columns:
                shocked_returns.iloc[-1, shocked_returns.columns.get_loc(asset)] += shock

        if scenario.factor_shocks and self.config.factor_exposures:
            for asset in assets:
                exposures = self.config.factor_exposures.get(asset, {})
                factor_impact = sum(
                    exposures.get(factor, 0.0) * shock for factor, shock in scenario.factor_shocks.items()
                )
                if factor_impact and asset in shocked_returns.columns:
                    shocked_returns.iloc[-1, shocked_returns.columns.get_loc(asset)] += factor_impact

        shocked_returns = shocked_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return shocked_returns

    def _estimate_liquidity_shortfall(self, scenario: MarketScenario) -> float:
        """估算情景下的流动性缺口"""
        liquidity_drop = scenario.liquidity_shocks.get("liquidity_drop", 0.0)
        volume_drop = scenario.liquidity_shocks.get("volume_drop", 0.0)

        effective_buffer = self.config.liquidity_buffer * (1.0 - liquidity_drop)
        stressed_outflow = self.config.net_outflow_5day * (1.0 + volume_drop)
        shortfall = max(0.0, stressed_outflow - effective_buffer)
        return shortfall


class RiskAnalyzer:
    """风险分析器"""

    def __init__(self, config: StressTestConfig) -> None:
        self.config = config

    def evaluate(self, simulation: PortfolioSimulationResult) -> StressTestResult:
        """计算压力测试指标体系"""
        risk_metrics = self._compute_market_risk(simulation)
        capital_metrics = self._compute_capital_metrics(simulation, risk_metrics)
        liquidity_metrics = self._compute_liquidity_metrics(simulation)
        sensitivity = self._compute_sensitivity(simulation)
        breaches = self._identify_breaches(risk_metrics, capital_metrics, liquidity_metrics)

        diagnostics = {
            **simulation.diagnostics,
            "stress_var_99": risk_metrics.get("stress_var_99"),
            "stress_cvar_99": risk_metrics.get("stress_cvar_99"),
            "max_drawdown": risk_metrics.get("max_drawdown"),
        }

        return StressTestResult(
            scenario=simulation.scenario,
            risk_metrics=risk_metrics,
            capital_metrics=capital_metrics,
            liquidity_metrics=liquidity_metrics,
            sensitivity_analysis=sensitivity,
            breaches=breaches,
            diagnostics=diagnostics,
        )

    def _compute_market_risk(self, simulation: PortfolioSimulationResult) -> Dict[str, float]:
        """计算市场风险指标"""
        returns_df = simulation.portfolio_returns.to_frame(name="portfolio")
        var_model = VaRModel(lookback_period=len(returns_df), confidence_levels=self.config.var_confidences)
        cvar_model = CVaRModel(lookback_period=len(returns_df), confidence_levels=self.config.var_confidences)

        var_model.fit(returns_df)
        cvar_model.fit(returns_df)

        weights = np.array([1.0])
        var_result = var_model.historical_var(weights, self.config.initial_portfolio_value)
        cvar_result = cvar_model.historical_cvar(weights, self.config.initial_portfolio_value)

        max_drawdown = float(simulation.drawdowns.min())
        max_daily_loss = float(simulation.portfolio_returns.min())
        liquidity_adjustment = float(simulation.liquidity_shortfalls.max())

        risk_metrics = {
            "stress_var_95": var_result.var_values.get(0.95, np.nan),
            "stress_var_99": var_result.var_values.get(0.99, np.nan),
            "stress_var_99_5": var_result.var_values.get(0.995, np.nan),
            "stress_cvar_95": cvar_result.cvar_values.get(0.95, np.nan),
            "stress_cvar_99": cvar_result.cvar_values.get(0.99, np.nan),
            "stress_cvar_99_5": cvar_result.cvar_values.get(0.995, np.nan),
            "liquidity_adjusted_var": var_result.var_values.get(0.99, np.nan) + liquidity_adjustment,
            "max_drawdown": max_drawdown,
            "max_daily_loss": max_daily_loss,
            "volatility": float(simulation.portfolio_returns.std(ddof=0)),
            "expected_shortfall": cvar_result.cvar_values.get(0.99, np.nan),
            "tail_loss_probability": float(
                (simulation.portfolio_returns < -self.config.market_thresholds["max_daily_loss"]).mean()
            ),
        }
        return risk_metrics

    def _compute_capital_metrics(
        self, simulation: PortfolioSimulationResult, risk_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """计算资本充足性指标"""
        stress_losses = max(
            0.0, self.config.initial_portfolio_value - float(simulation.portfolio_values.min())
        )
        stress_capital_requirement = stress_losses + risk_metrics.get("stress_cvar_99", 0.0)

        remaining_tier1 = max(0.0, self.config.tier1_capital - stress_capital_requirement)
        remaining_total_capital = max(0.0, self.config.total_capital - stress_capital_requirement)

        tier1_ratio = remaining_tier1 / self.config.risk_weighted_assets if self.config.risk_weighted_assets else np.nan
        total_capital_ratio = (
            remaining_total_capital / self.config.risk_weighted_assets if self.config.risk_weighted_assets else np.nan
        )
        leverage_ratio = remaining_tier1 / self.config.total_exposure if self.config.total_exposure else np.nan

        capital_metrics = {
            "stress_capital_requirement": stress_capital_requirement,
            "remaining_tier1_capital": remaining_tier1,
            "remaining_total_capital": remaining_total_capital,
            "tier1_ratio": tier1_ratio,
            "total_capital_ratio": total_capital_ratio,
            "leverage_ratio": leverage_ratio,
            "capital_buffer_adequacy": (remaining_tier1 / self.config.tier1_capital)
            if self.config.tier1_capital
            else np.nan,
            "survival_horizon_days": self._estimate_survival_horizon(simulation.portfolio_values),
        }
        return capital_metrics

    def _compute_liquidity_metrics(self, simulation: PortfolioSimulationResult) -> Dict[str, float]:
        """计算流动性风险指标"""
        liquidity_drop = simulation.scenario.liquidity_shocks.get("liquidity_drop", 0.0)
        volume_drop = simulation.scenario.liquidity_shocks.get("volume_drop", 0.0)
        liquidity_shortfall = float(simulation.liquidity_shortfalls.max())

        effective_hqla = self.config.liquidity_buffer * (1.0 - liquidity_drop)
        stressed_outflow = self.config.net_outflow_5day * (1.0 + volume_drop)

        lcr = effective_hqla / stressed_outflow if stressed_outflow else np.nan
        nsfr = (
            (self.config.stable_funding * (1.0 - liquidity_drop)) / self.config.required_stable_funding
            if self.config.required_stable_funding
            else np.nan
        )
        cash_buffer_days = (
            effective_hqla / (self.config.net_outflow_5day / 5.0) if self.config.net_outflow_5day else np.nan
        )

        liquidity_metrics = {
            "liquidity_coverage_ratio": lcr,
            "net_stable_funding_ratio": nsfr,
            "liquidity_gap": effective_hqla - stressed_outflow,
            "liquidity_shortfall": liquidity_shortfall,
            "cash_buffer_days": cash_buffer_days,
        }
        return liquidity_metrics

    def _compute_sensitivity(self, simulation: PortfolioSimulationResult) -> Dict[str, Any]:
        """进行敏感性分析"""
        asset_drawdowns: Dict[str, float] = {}
        for asset in simulation.asset_values.columns:
            series = simulation.asset_values[asset]
            if series.empty or series.iloc[0] == 0:
                asset_drawdowns[asset] = 0.0
                continue
            running_max = series.cummax()
            drawdown = ((series - running_max) / running_max).min()
            asset_drawdowns[asset] = float(drawdown)

        leverage_sensitivity = (
            float(simulation.asset_values.abs().sum(axis=1).max()) / self.config.initial_portfolio_value
            if not simulation.asset_values.empty and self.config.initial_portfolio_value
            else np.nan
        )

        liquidity_sensitivity = {
            "shortfall_to_buffer_ratio": (
                simulation.liquidity_shortfalls.max() / self.config.liquidity_buffer if self.config.liquidity_buffer else np.nan
            ),
            "volume_drop": simulation.scenario.liquidity_shocks.get("volume_drop", 0.0),
        }

        return {
            "asset_drawdowns": asset_drawdowns,
            "leverage_sensitivity": leverage_sensitivity,
            "liquidity_sensitivity": liquidity_sensitivity,
        }

    def _estimate_survival_horizon(self, portfolio_values: pd.Series) -> float:
        """估算组合在阈值下的生存期"""
        if portfolio_values.empty:
            return 0.0

        threshold = self.config.initial_portfolio_value * (1.0 - self.config.market_thresholds["max_drawdown"])
        below_threshold = portfolio_values[portfolio_values <= threshold]
        if below_threshold.empty:
            return float(len(portfolio_values))

        breach_index = below_threshold.index[0]
        start_index = portfolio_values.index[0]
        if isinstance(breach_index, pd.Timestamp):
            return float((breach_index - start_index).days or 1)
        return float(portfolio_values.index.get_loc(breach_index) + 1)

    def _identify_breaches(
        self, risk_metrics: Dict[str, float], capital_metrics: Dict[str, float], liquidity_metrics: Dict[str, float]
    ) -> Dict[str, bool]:
        """判断指标是否突破阈值"""
        breaches = {
            "max_daily_loss": abs(risk_metrics["max_daily_loss"]) > self.config.market_thresholds["max_daily_loss"],
            "max_drawdown": abs(risk_metrics["max_drawdown"]) > self.config.market_thresholds["max_drawdown"],
            "tier1_ratio": capital_metrics["tier1_ratio"] < self.config.capital_thresholds["tier1_ratio"],
            "total_capital_ratio": capital_metrics["total_capital_ratio"] < self.config.capital_thresholds["total_capital_ratio"],
            "leverage_ratio": capital_metrics["leverage_ratio"] < self.config.capital_thresholds["leverage_ratio"],
            "lcr": liquidity_metrics["liquidity_coverage_ratio"] < self.config.liquidity_thresholds["lcr"],
            "nsfr": liquidity_metrics["net_stable_funding_ratio"] < self.config.liquidity_thresholds["nsfr"],
            "cash_buffer_days": liquidity_metrics["cash_buffer_days"] < self.config.liquidity_thresholds["cash_buffer_days"],
        }
        return breaches


class ReportGenerator:
    """压力测试报告生成器"""

    def __init__(self, config: StressTestConfig) -> None:
        self.config = config

    def build_report(self, results: List[StressTestResult]) -> Dict[str, Any]:
        """生成结构化报告"""
        generated_at = datetime.utcnow().isoformat()
        summary = {
            "generated_at": generated_at,
            "total_scenarios": len(results),
            "breach_summary": self._aggregate_breaches(results),
            "scenarios": [],
        }

        for result in results:
            scenario_entry = {
                "name": result.scenario.name,
                "type": result.scenario.scenario_type.value,
                "description": result.scenario.description,
                "risk_metrics": result.risk_metrics,
                "capital_metrics": result.capital_metrics,
                "liquidity_metrics": result.liquidity_metrics,
                "sensitivity_analysis": result.sensitivity_analysis,
                "breaches": result.breaches,
                "metadata": result.scenario.metadata,
            }
            summary["scenarios"].append(scenario_entry)

        summary["recommendations"] = self._generate_recommendations(summary)
        return summary

    def to_dataframe(self, results: List[StressTestResult]) -> pd.DataFrame:
        """将结果转换为DataFrame便于分析"""
        records: List[Dict[str, Any]] = []
        for result in results:
            record = {
                "scenario": result.scenario.name,
                "type": result.scenario.scenario_type.value,
                **{f"risk_{k}": v for k, v in result.risk_metrics.items()},
                **{f"capital_{k}": v for k, v in result.capital_metrics.items()},
                **{f"liquidity_{k}": v for k, v in result.liquidity_metrics.items()},
            }
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _aggregate_breaches(self, results: List[StressTestResult]) -> Dict[str, int]:
        """统计阈值突破次数"""
        breach_counter: Dict[str, int] = {}
        for result in results:
            for key, breached in result.breaches.items():
                breach_counter[key] = breach_counter.get(key, 0) + int(breached)
        return breach_counter

    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """根据突破情况生成建议"""
        recommendations: List[str] = []
        breach_summary = summary["breach_summary"]

        if breach_summary.get("tier1_ratio", 0) > 0:
            recommendations.append("增加一级资本缓冲或压缩风险加权资产，确保Tier1资本充足率达标。")
        if breach_summary.get("lcr", 0) > 0:
            recommendations.append("提升高流动性资产储备或延长负债期限以改善流动性覆盖率。")
        if breach_summary.get("max_drawdown", 0) > 0:
            recommendations.append("重新审视资产配置与对冲策略，降低组合在极端情景下的回撤。")
        if breach_summary.get("max_daily_loss", 0) > 0:
            recommendations.append("强化日内风险限额与自动减仓机制，控制单日损失。")
        if not recommendations:
            recommendations.append("压力测试结果在设定阈值内，维持当前风控框架并持续监测。")
        return recommendations


class StressTester:
    """压力测试器"""

    def __init__(
        self,
        config: Optional[StressTestConfig] = None,
        scenario_generator: Optional[ScenarioGenerator] = None,
        portfolio_simulator: Optional[PortfolioSimulator] = None,
        risk_analyzer: Optional[RiskAnalyzer] = None,
        report_generator: Optional[ReportGenerator] = None,
    ) -> None:
        self.config = config or StressTestConfig()
        self.scenario_generator = scenario_generator or ScenarioGenerator()
        self.portfolio_simulator = portfolio_simulator or PortfolioSimulator(self.config)
        self.risk_analyzer = risk_analyzer or RiskAnalyzer(self.config)
        self.report_generator = report_generator or ReportGenerator(self.config)

    def run(
        self,
        portfolio_weights: Dict[str, float],
        market_returns: pd.DataFrame,
        scenarios: Optional[List[MarketScenario]] = None,
    ) -> Dict[str, Any]:
        """
        执行压力测试
        
        Args:
            portfolio_weights: 投资组合权重
            market_returns: 市场收益率历史数据
            scenarios: 自定义情景列表（可选）
        Returns:
            结构化压力测试报告
        """
        scenario_list = scenarios or self._build_default_scenarios(market_returns)
        results: List[StressTestResult] = []

        for scenario in scenario_list:
            simulation = self.portfolio_simulator.simulate(portfolio_weights, market_returns, scenario)
            analysis = self.risk_analyzer.evaluate(simulation)
            results.append(analysis)

        report = self.report_generator.build_report(results)
        report["dataframe"] = self.report_generator.to_dataframe(results)
        return report

    def _build_default_scenarios(self, market_returns: pd.DataFrame) -> List[MarketScenario]:
        """根据配置生成默认情景"""
        scenarios: List[MarketScenario] = []
        selection = self.config.scenario_selection or [
            "2008_crisis",
            "covid_19",
            "interest_rate_shock",
            "liquidity_crunch",
        ]

        for name in selection:
            if name in HISTORICAL_SCENARIOS:
                scenarios.append(self.scenario_generator.generate_historical_scenario(name))
            elif name in HYPOTHETICAL_SCENARIOS:
                scenarios.append(self.scenario_generator.generate_hypothetical_scenario(name))
            else:
                # 尝试根据关键词生成特殊情景
                if "correlation" in name and not market_returns.empty:
                    base_corr = market_returns.corr()
                    scenarios.append(self.scenario_generator.generate_correlation_breakdown(base_corr, target_level=0.0, name=name))
                elif "liquidity" in name:
                    scenarios.append(self.scenario_generator.generate_liquidity_crisis(volume_drop=0.8, name=name))

        return scenarios

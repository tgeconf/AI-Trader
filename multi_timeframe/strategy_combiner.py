"""
策略组合器
实现多策略信号的组合与优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .timeframe_integration import TimeframeSignal, IntegratedSignal


class StrategyType(Enum):
    """策略类型枚举"""

    TREND_FOLLOWING = "trend_following"  # 趋势跟踪策略
    MEAN_REVERSION = "mean_reversion"  # 均值回归策略
    BREAKOUT = "breakout"  # 突破策略
    MOMENTUM = "momentum"  # 动量策略
    VOLATILITY = "volatility"  # 波动率策略
    ARBITRAGE = "arbitrage"  # 套利策略


@dataclass
class StrategySignal:
    """策略信号"""

    strategy_type: StrategyType
    symbol: str
    signal_strength: float  # -1到1，负数为卖出，正数为买入
    confidence: float  # 0到1，信号置信度
    timeframe: str  # 主要时间框架
    indicators_used: List[str]  # 使用的技术指标
    performance_score: float  # 策略历史表现评分


@dataclass
class CombinedStrategySignal:
    """组合策略信号"""

    symbol: str
    final_signal: float  # 最终信号强度
    strategy_contributions: Dict[StrategyType, float]  # 各策略贡献
    correlation_matrix: np.ndarray  # 策略相关性矩阵
    diversification_score: float  # 策略分散化评分
    risk_adjusted_signal: float  # 风险调整后信号


class StrategyCombiner:
    """
    策略组合器

    实现多策略信号的组合与优化：
    - 策略权重分配
    - 策略相关性分析
    - 风险分散优化
    - 动态权重调整
    - 绩效评估
    """

    def __init__(
        self,
        base_strategies: Optional[List[StrategyType]] = None,
        correlation_threshold: float = 0.7,
        max_strategy_weight: float = 0.4,
    ):
        """
        初始化策略组合器

        Args:
            base_strategies: 基础策略类型
            correlation_threshold: 相关性阈值
            max_strategy_weight: 最大策略权重
        """
        self.correlation_threshold = correlation_threshold
        self.max_strategy_weight = max_strategy_weight
        self.strategies = base_strategies or list(StrategyType)
        self.strategy_weights = self._get_default_weights()
        self.strategy_performance = {}
        self.correlation_matrix = self._initialize_correlation_matrix()

    def _get_default_weights(self) -> Dict[StrategyType, float]:
        """获取默认策略权重"""
        return {
            StrategyType.TREND_FOLLOWING: 0.25,
            StrategyType.MEAN_REVERSION: 0.20,
            StrategyType.BREAKOUT: 0.15,
            StrategyType.MOMENTUM: 0.15,
            StrategyType.VOLATILITY: 0.15,
            StrategyType.ARBITRAGE: 0.10,
        }

    def _initialize_correlation_matrix(self) -> np.ndarray:
        """初始化策略相关性矩阵"""
        n_strategies = len(self.strategies)
        # 初始假设策略间相关性较低
        return np.eye(n_strategies) * 0.8 + np.ones((n_strategies, n_strategies)) * 0.2

    def combine_strategy_signals(
        self, strategy_signals: List[StrategySignal]
    ) -> CombinedStrategySignal:
        """
        组合策略信号

        Args:
            strategy_signals: 策略信号列表

        Returns:
            组合策略信号
        """
        if not strategy_signals:
            raise ValueError("策略信号列表不能为空")

        symbol = strategy_signals[0].symbol

        # 按策略类型分组信号
        signals_by_strategy = defaultdict(list)
        for signal in strategy_signals:
            signals_by_strategy[signal.strategy_type].append(signal)

        # 计算各策略的加权信号
        strategy_contributions = {}
        weighted_signals = []
        strategy_weights_used = []

        for strategy_type, strategy_signals_list in signals_by_strategy.items():
            if strategy_type not in self.strategy_weights:
                continue

            # 计算该策略的平均信号（考虑置信度和表现评分）
            weighted_strategy_signal = sum(
                s.signal_strength * s.confidence * s.performance_score
                for s in strategy_signals_list
            ) / sum(s.confidence * s.performance_score for s in strategy_signals_list)

            # 应用策略权重
            strategy_weight = self.strategy_weights[strategy_type]
            final_contribution = weighted_strategy_signal * strategy_weight

            strategy_contributions[strategy_type] = final_contribution
            weighted_signals.append(final_contribution)
            strategy_weights_used.append(strategy_weight)

        # 计算最终信号
        final_signal = sum(weighted_signals)

        # 计算策略相关性
        correlation_matrix = self._calculate_strategy_correlation(strategy_signals)

        # 计算分散化评分
        diversification_score = self._calculate_diversification_score(
            strategy_contributions, correlation_matrix
        )

        # 风险调整信号
        risk_adjusted_signal = self._calculate_risk_adjusted_signal(
            final_signal, diversification_score
        )

        return CombinedStrategySignal(
            symbol=symbol,
            final_signal=final_signal,
            strategy_contributions=strategy_contributions,
            correlation_matrix=correlation_matrix,
            diversification_score=diversification_score,
            risk_adjusted_signal=risk_adjusted_signal,
        )

    def _calculate_strategy_correlation(
        self, strategy_signals: List[StrategySignal]
    ) -> np.ndarray:
        """计算策略相关性"""
        n_strategies = len(self.strategies)
        correlation_matrix = np.zeros((n_strategies, n_strategies))

        # 按策略类型分组历史信号
        strategy_signals_history = {strategy: [] for strategy in self.strategies}

        for signal in strategy_signals:
            strategy_signals_history[signal.strategy_type].append(
                signal.signal_strength
            )

        # 计算相关性
        for i, strategy_i in enumerate(self.strategies):
            for j, strategy_j in enumerate(self.strategies):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    signals_i = strategy_signals_history[strategy_i]
                    signals_j = strategy_signals_history[strategy_j]

                    if len(signals_i) > 1 and len(signals_j) > 1:
                        # 确保信号长度一致
                        min_length = min(len(signals_i), len(signals_j))
                        corr = np.corrcoef(
                            signals_i[:min_length], signals_j[:min_length]
                        )[0, 1]
                        correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[i, j] = 0.0

        return correlation_matrix

    def _calculate_diversification_score(
        self,
        strategy_contributions: Dict[StrategyType, float],
        correlation_matrix: np.ndarray,
    ) -> float:
        """计算分散化评分"""
        if not strategy_contributions:
            return 0.0

        # 计算策略权重向量
        total_contribution = sum(
            abs(contribution) for contribution in strategy_contributions.values()
        )
        if total_contribution == 0:
            return 0.0

        weights = np.array(
            [
                abs(contribution) / total_contribution
                for contribution in strategy_contributions.values()
            ]
        )

        # 计算投资组合方差（简化版）
        portfolio_variance = weights.T @ correlation_matrix @ weights

        # 转换为分散化评分（方差越小，分散化越好）
        max_variance = np.max(np.diag(correlation_matrix))  # 最大单个策略方差
        diversification_score = 1.0 - (portfolio_variance / max_variance)

        return max(0.0, diversification_score)

    def _calculate_risk_adjusted_signal(
        self, final_signal: float, diversification_score: float
    ) -> float:
        """计算风险调整后信号"""
        # 分散化程度越高，信号越可靠
        risk_adjustment = 1.0 + (diversification_score * 0.5)  # 最多增强50%
        return final_signal * risk_adjustment

    def optimize_strategy_weights(
        self, historical_performance: Dict[StrategyType, pd.DataFrame]
    ) -> Dict[StrategyType, float]:
        """
        优化策略权重

        Args:
            historical_performance: 各策略历史表现数据

        Returns:
            优化后的策略权重
        """
        strategy_scores = {}

        for strategy_type, perf_data in historical_performance.items():
            if strategy_type not in self.strategies:
                continue

            # 计算策略表现评分
            score = self._calculate_strategy_score(perf_data)
            strategy_scores[strategy_type] = score

        # 归一化得分
        total_score = sum(strategy_scores.values())
        if total_score > 0:
            optimized_weights = {
                strategy: score / total_score
                for strategy, score in strategy_scores.items()
            }
        else:
            optimized_weights = self.strategy_weights

        # 应用最大权重限制
        return self._apply_weight_constraints(optimized_weights)

    def _calculate_strategy_score(self, perf_data: pd.DataFrame) -> float:
        """计算策略表现评分"""
        if perf_data.empty:
            return 0.0

        # 计算多个表现指标
        returns = perf_data.get("returns", pd.Series([0]))
        signals = perf_data.get("signals", pd.Series([0]))

        # 夏普比率（简化版）
        avg_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0

        # 胜率
        winning_trades = (returns > 0).sum()
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 信号准确性
        correct_signals = ((signals > 0) & (returns > 0)).sum() + (
            (signals < 0) & (returns < 0)
        ).sum()
        signal_accuracy = correct_signals / len(signals) if len(signals) > 0 else 0

        # 综合评分
        composite_score = sharpe_ratio * 0.4 + win_rate * 0.3 + signal_accuracy * 0.3

        return max(0.0, composite_score)

    def _apply_weight_constraints(
        self, weights: Dict[StrategyType, float]
    ) -> Dict[StrategyType, float]:
        """应用权重约束"""
        # 确保权重和为1
        total_weight = sum(weights.values())
        if total_weight == 0:
            return self.strategy_weights

        normalized_weights = {
            strategy: weight / total_weight for strategy, weight in weights.items()
        }

        # 应用最大权重限制
        constrained_weights = {}
        remaining_weight = 1.0

        # 首先分配不超过最大权重的部分
        for strategy, weight in normalized_weights.items():
            constrained_weight = min(weight, self.max_strategy_weight)
            constrained_weights[strategy] = constrained_weight
            remaining_weight -= constrained_weight

        # 如果有剩余权重，分配给表现最好的策略
        if remaining_weight > 0:
            best_strategy = max(normalized_weights.items(), key=lambda x: x[1])[0]
            constrained_weights[best_strategy] += remaining_weight

        return constrained_weights

    def update_strategy_performance(
        self, strategy_type: StrategyType, performance_data: pd.DataFrame
    ) -> None:
        """
        更新策略表现数据

        Args:
            strategy_type: 策略类型
            performance_data: 表现数据
        """
        if strategy_type not in self.strategy_performance:
            self.strategy_performance[strategy_type] = []

        self.strategy_performance[strategy_type].append(performance_data)

        # 只保留最近100条记录
        if len(self.strategy_performance[strategy_type]) > 100:
            self.strategy_performance[strategy_type] = self.strategy_performance[
                strategy_type
            ][-100:]

    def generate_strategy_report(
        self, combined_signal: CombinedStrategySignal
    ) -> Dict[str, Any]:
        """
        生成策略报告

        Args:
            combined_signal: 组合策略信号

        Returns:
            策略分析报告
        """
        # 分析策略贡献
        strategy_analysis = {}
        for (
            strategy_type,
            contribution,
        ) in combined_signal.strategy_contributions.items():
            weight = self.strategy_weights.get(strategy_type, 0.0)
            strategy_analysis[strategy_type.value] = {
                "contribution": contribution,
                "weight": weight,
                "effective_signal": contribution / weight if weight > 0 else 0.0,
            }

        # 分析策略相关性
        correlation_analysis = {}
        for i, strategy_i in enumerate(self.strategies):
            for j, strategy_j in enumerate(self.strategies):
                if i < j:  # 只分析上三角
                    corr = combined_signal.correlation_matrix[i, j]
                    correlation_analysis[f"{strategy_i.value}_{strategy_j.value}"] = {
                        "correlation": corr,
                        "status": (
                            "high" if abs(corr) > self.correlation_threshold else "low"
                        ),
                    }

        return {
            "symbol": combined_signal.symbol,
            "final_signal": combined_signal.final_signal,
            "risk_adjusted_signal": combined_signal.risk_adjusted_signal,
            "diversification_score": combined_signal.diversification_score,
            "strategy_analysis": strategy_analysis,
            "correlation_analysis": correlation_analysis,
            "recommendation": self._generate_strategy_recommendation(combined_signal),
        }

    def _generate_strategy_recommendation(
        self, combined_signal: CombinedStrategySignal
    ) -> str:
        """生成策略建议"""
        signal_strength = abs(combined_signal.final_signal)
        diversification = combined_signal.diversification_score

        if signal_strength < 0.1:
            return "策略信号较弱，建议观望"

        direction = "买入" if combined_signal.final_signal > 0 else "卖出"

        if diversification > 0.7:
            confidence = "高度可靠"
        elif diversification > 0.5:
            confidence = "中等可靠"
        else:
            confidence = "谨慎参考"

        if signal_strength > 0.3:
            strength = "强势"
        elif signal_strength > 0.15:
            strength = "中等"
        else:
            strength = "弱势"

        return f"{confidence}{direction}信号（{strength}，分散化评分：{diversification:.2f}）"

    def get_combiner_statistics(self) -> Dict[str, Any]:
        """获取组合器统计信息"""
        return {
            "total_strategies": len(self.strategies),
            "strategy_weights": {
                strategy.value: weight
                for strategy, weight in self.strategy_weights.items()
            },
            "max_strategy_weight": self.max_strategy_weight,
            "correlation_threshold": self.correlation_threshold,
            "performance_history_size": sum(
                len(data) for data in self.strategy_performance.values()
            ),
        }

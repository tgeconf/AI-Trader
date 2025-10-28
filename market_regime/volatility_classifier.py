"""
波动率状态分类器
实现机构级波动率分析、异常检测与置信度评估
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """波动率状态枚举"""
    LOW_VOL = "low_volatility"
    NORMAL_VOL = "normal_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"


@dataclass
class VolatilityMetrics:
    """波动率指标集"""
    rolling_20d: float
    rolling_60d: float
    rolling_120d: float
    realized_vol: float
    garch_forecast: float
    volatility_ratio: float
    percentile_rank: float
    ewma_vol: float


@dataclass
class VolatilityAnalysisResult:
    """波动率分析结果"""
    current_regime: VolatilityRegime
    volatility_level: float
    confidence: float
    regime_duration: int
    anomaly_flags: List[str]
    volatility_metrics: Dict[str, float]
    regime_probabilities: Dict[VolatilityRegime, float]


class VolatilityCalculator:
    """波动率计算器"""

    def __init__(self, annualization_factor: float = 252.0) -> None:
        self.annualization_factor = annualization_factor

    def calculate_rolling_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """计算滚动波动率（年化）"""
        if window <= 1:
            raise ValueError("Rolling window must be greater than 1")
        return returns.rolling(window=window).std(ddof=0) * np.sqrt(self.annualization_factor)

    def estimate_garch_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """
        简化的GARCH(1,1)波动率估计
        使用指数加权方案近似，避免额外依赖
        """
        alpha = 0.05
        beta = 0.9
        omega = returns.var(ddof=0) * (1 - alpha - beta)
        variance = returns.var(ddof=0)
        variances = []
        for ret in returns.dropna():
            variance = omega + alpha * (ret ** 2) + beta * variance
            variances.append(variance)
        if not variances:
            forecast = float(np.sqrt(variance * self.annualization_factor))
        else:
            forecast = float(np.sqrt(variances[-1] * self.annualization_factor))
        return {
            "conditional_volatility": forecast,
            "omega": float(omega),
            "alpha": alpha,
            "beta": beta,
        }

    def calculate_realized_volatility(self, returns: pd.Series) -> float:
        """计算已实现波动率"""
        return float(returns.std(ddof=0) * np.sqrt(self.annualization_factor))

    def compute_volatility_ratio(
        self, rolling_short: pd.Series, rolling_long: pd.Series
    ) -> float:
        """计算短期与长期波动率之比"""
        latest_short = float(rolling_short.iloc[-1])
        latest_long = float(rolling_long.iloc[-1]) if rolling_long.iloc[-1] != 0 else np.nan
        if np.isnan(latest_short) or np.isnan(latest_long) or latest_long == 0:
            return float("nan")
        return latest_short / latest_long

    def compute_ewma_volatility(self, returns: pd.Series, span: int = 30) -> pd.Series:
        """使用EWMA估计波动率"""
        ewma_var = returns.pow(2).ewm(span=span, adjust=False).mean()
        return np.sqrt(ewma_var) * np.sqrt(self.annualization_factor)


class RegimeClassifier:
    """波动率状态分类器"""

    def __init__(
        self,
        percentile_thresholds: Optional[Dict[VolatilityRegime, Tuple[float, float]]] = None,
        volatility_thresholds: Optional[Dict[VolatilityRegime, float]] = None,
    ) -> None:
        self.percentile_thresholds = percentile_thresholds or {
            VolatilityRegime.LOW_VOL: (0.0, 0.20),
            VolatilityRegime.NORMAL_VOL: (0.20, 0.80),
            VolatilityRegime.HIGH_VOL: (0.80, 0.95),
            VolatilityRegime.EXTREME_VOL: (0.95, 1.0),
        }
        self.volatility_thresholds = volatility_thresholds or {
            VolatilityRegime.LOW_VOL: 0.10,
            VolatilityRegime.NORMAL_VOL: 0.20,
            VolatilityRegime.HIGH_VOL: 0.30,
            VolatilityRegime.EXTREME_VOL: 0.40,
        }

    def classify(
        self, rolling_vol: pd.Series, percentile_rank: float
    ) -> VolatilityRegime:
        """根据分位数与绝对阈值分类"""
        latest_vol = float(rolling_vol.iloc[-1])
        for regime in [
            VolatilityRegime.EXTREME_VOL,
            VolatilityRegime.HIGH_VOL,
            VolatilityRegime.NORMAL_VOL,
            VolatilityRegime.LOW_VOL,
        ]:
            vol_threshold = self.volatility_thresholds.get(regime, np.inf)
            lower, upper = self.percentile_thresholds.get(regime, (0.0, 1.0))
            if percentile_rank >= lower and percentile_rank < upper:
                if latest_vol >= vol_threshold or regime in (VolatilityRegime.NORMAL_VOL, VolatilityRegime.LOW_VOL):
                    return regime
        return VolatilityRegime.NORMAL_VOL

    def compute_probabilities(
        self,
        percentile_rank: float,
        volatility_level: float,
        rolling_vol: pd.Series,
    ) -> Dict[VolatilityRegime, float]:
        """构建波动率状态概率分布"""
        z_scores = {}
        latest = float(volatility_level)
        mean = float(rolling_vol.mean())
        std = float(rolling_vol.std(ddof=0)) if rolling_vol.std(ddof=0) > 0 else 1.0
        z = (latest - mean) / std

        for regime, (lower, upper) in self.percentile_thresholds.items():
            midpoint = (lower + upper) / 2
            dist = abs(percentile_rank - midpoint)
            vol_gap = abs(latest - self.volatility_thresholds.get(regime, latest))
            score = np.exp(-5 * dist) * np.exp(-vol_gap) * np.exp(-0.5 * z ** 2)
            z_scores[regime] = score

        total = sum(z_scores.values())
        if total == 0:
            uniform = 1.0 / len(VolatilityRegime)
            return {regime: uniform for regime in VolatilityRegime}
        return {regime: score / total for regime, score in z_scores.items()}


class AnomalyDetector:
    """波动率异常检测器"""

    def __init__(self, jump_threshold: float = 3.0, clustering_window: int = 20) -> None:
        self.jump_threshold = jump_threshold
        self.clustering_window = clustering_window

    def detect(self, rolling_vol: pd.Series) -> List[str]:
        """输出异常标志列表"""
        flags: List[str] = []
        if len(rolling_vol) < self.clustering_window + 2:
            return flags

        latest = float(rolling_vol.iloc[-1])
        prev = float(rolling_vol.iloc[-2])
        diff = latest - prev
        recent_std = float(rolling_vol.diff().rolling(window=self.clustering_window).std(ddof=0).iloc[-1])
        baseline = float(rolling_vol.iloc[:-self.clustering_window].mean()) if len(rolling_vol) > self.clustering_window else float(rolling_vol.mean())

        if recent_std > 0 and diff / recent_std > self.jump_threshold:
            flags.append("vol_jump_detected")
        if latest > baseline * 1.5 and diff > 0:
            flags.append("volatility_regime_shift")
        if self._is_clustering(rolling_vol):
            flags.append("volatility_clustering")
        if latest < baseline * 0.5 and diff < 0:
            flags.append("volatility_crush")
        return flags

    def _is_clustering(self, rolling_vol: pd.Series) -> bool:
        """判断是否出现波动率聚集"""
        window = self.clustering_window
        recent = rolling_vol.iloc[-window:]
        earlier = rolling_vol.iloc[-2 * window : -window]
        if len(earlier) < window or len(recent) < window:
            return False
        recent_var = float(recent.var(ddof=0))
        earlier_var = float(earlier.var(ddof=0)) if len(earlier) > 0 else 0.0
        return earlier_var > 0 and recent_var / earlier_var > 1.8


class ConfidenceAssessor:
    """置信度评估器"""

    def assess(
        self,
        regime_probs: Dict[VolatilityRegime, float],
        anomaly_flags: Sequence[str],
        volatility_ratio: float,
    ) -> float:
        """综合评估置信度"""
        probs = np.array(list(regime_probs.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(probs))
        entropy_score = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0

        ratio_score = 1.0 - min(abs(1.0 - volatility_ratio), 1.0) if not np.isnan(volatility_ratio) else 0.5
        anomaly_penalty = 0.2 * len(anomaly_flags)
        confidence = max(0.0, min(1.0, 0.6 * entropy_score + 0.3 * ratio_score + 0.1 - anomaly_penalty))
        return float(confidence)


class VolatilityClassifier:
    """波动率分类器主类"""

    def __init__(
        self,
        lookback: int = 252,
        calculator: Optional[VolatilityCalculator] = None,
        classifier: Optional[RegimeClassifier] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        assessor: Optional[ConfidenceAssessor] = None,
    ) -> None:
        self.lookback = lookback
        self.calculator = calculator or VolatilityCalculator()
        self.classifier = classifier or RegimeClassifier()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()
        self.assessor = assessor or ConfidenceAssessor()

    def analyze(self, returns: pd.Series) -> VolatilityAnalysisResult:
        """
        对收益率序列执行波动率分析
        
        Args:
            returns: 资产或组合收益率，索引为时间
        """
        if len(returns.dropna()) < max(60, self.lookback // 2):
            raise ValueError("Insufficient data for volatility analysis")

        returns = returns.astype(float).fillna(0.0)
        returns = returns.iloc[-max(self.lookback, len(returns)) :]

        rolling_20 = self.calculator.calculate_rolling_volatility(returns, 20).fillna(method="bfill")
        rolling_60 = self.calculator.calculate_rolling_volatility(returns, 60).fillna(method="bfill")
        rolling_120 = self.calculator.calculate_rolling_volatility(returns, 120).fillna(method="bfill")
        ewma_vol = self.calculator.compute_ewma_volatility(returns)
        realized_vol = self.calculator.calculate_realized_volatility(returns)
        garch_result = self.calculator.estimate_garch_volatility(returns)
        volatility_ratio = self.calculator.compute_volatility_ratio(rolling_20, rolling_120)

        rolling_target = rolling_20.combine(ewma_vol, func=lambda x, y: 0.5 * (x + y))
        percentile_rank = float(self._percentile_rank(rolling_target.dropna(), rolling_target.iloc[-1]))

        current_regime = self.classifier.classify(rolling_target.dropna(), percentile_rank)
        regime_probs = self.classifier.compute_probabilities(percentile_rank, float(rolling_target.iloc[-1]), rolling_target.dropna())
        anomaly_flags = self.anomaly_detector.detect(rolling_target.dropna())
        confidence = self.assessor.assess(regime_probs, anomaly_flags, volatility_ratio)
        regime_duration = self._estimate_regime_duration(rolling_target.dropna(), current_regime)

        metrics = VolatilityMetrics(
            rolling_20d=float(rolling_20.iloc[-1]),
            rolling_60d=float(rolling_60.iloc[-1]),
            rolling_120d=float(rolling_120.iloc[-1]),
            realized_vol=realized_vol,
            garch_forecast=float(garch_result["conditional_volatility"]),
            volatility_ratio=float(volatility_ratio),
            percentile_rank=percentile_rank,
            ewma_vol=float(ewma_vol.iloc[-1]),
        )

        return VolatilityAnalysisResult(
            current_regime=current_regime,
            volatility_level=float(rolling_target.iloc[-1]),
            confidence=confidence,
            regime_duration=regime_duration,
            anomaly_flags=list(anomaly_flags),
            volatility_metrics={
                "rolling_20d": metrics.rolling_20d,
                "rolling_60d": metrics.rolling_60d,
                "rolling_120d": metrics.rolling_120d,
                "realized_vol": metrics.realized_vol,
                "garch_forecast": metrics.garch_forecast,
                "volatility_ratio": metrics.volatility_ratio,
                "percentile_rank": metrics.percentile_rank,
                "ewma_vol": metrics.ewma_vol,
            },
            regime_probabilities=regime_probs,
        )

    def _estimate_regime_duration(
        self, rolling_vol: pd.Series, current_regime: VolatilityRegime
    ) -> int:
        """估计当前波动率状态持续时间"""
        classifications: List[VolatilityRegime] = []
        for i in range(len(rolling_vol)):
            window_series = rolling_vol.iloc[: i + 1]
            percentile = self._percentile_rank(window_series, float(window_series.iloc[-1]))
            regime = self.classifier.classify(window_series, percentile)
            classifications.append(regime)

        count = 0
        for regime in reversed(classifications):
            if regime == current_regime:
                count += 1
            else:
                break
        return count

    def _percentile_rank(self, series: pd.Series, value: float) -> float:
        """计算百分位"""
        clean = series.dropna()
        if clean.empty:
            return 0.5
        rank = (clean <= value).mean()
        return float(rank)

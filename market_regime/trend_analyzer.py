"""
趋势分析模块
实现机构级趋势识别、强度评估与突破检测
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class TrendRegime(Enum):
    """趋势状态枚举"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    TREND_REVERSAL = "reversal"


@dataclass
class TrendSignal:
    """移动平均趋势信号"""
    direction: float  # -1 ~ 1
    short_value: float
    long_value: float
    crossover: Optional[str]
    short_series: pd.Series = field(repr=False)
    long_series: pd.Series = field(repr=False)


@dataclass
class MomentumAnalysis:
    """动量指标集合"""
    momentum: float
    rsi: float
    macd: float
    macd_signal: float
    stochastic_k: float
    composite_score: float


@dataclass
class BreakoutSignal:
    """突破信号"""
    signal_type: str
    level: float
    direction: str
    confidence: float
    timestamp: pd.Timestamp


@dataclass
class TrendAnalysisResult:
    """趋势分析结果"""
    current_regime: TrendRegime
    trend_strength: float
    trend_direction: float
    confidence: float
    duration_periods: int
    support_resistance_levels: Dict[str, float]
    breakout_signals: List[BreakoutSignal]


class TrendCalculator:
    """趋势计算器"""

    def calculate_moving_average_trend(
        self,
        prices: pd.Series,
        short_ma: int = 20,
        long_ma: int = 60,
    ) -> TrendSignal:
        if len(prices) < long_ma:
            raise ValueError("Insufficient data for moving average trend calculation")

        short_series = prices.rolling(window=short_ma, min_periods=short_ma).mean().fillna(method="bfill")
        long_series = prices.rolling(window=long_ma, min_periods=long_ma).mean().fillna(method="bfill")

        latest_short = float(short_series.iloc[-1])
        latest_long = float(long_series.iloc[-1])
        base = max(abs(latest_long), 1e-6)
        direction = np.tanh((latest_short - latest_long) / base)

        crossover = None
        if len(short_series) >= 2 and len(long_series) >= 2:
            prev_diff = short_series.iloc[-2] - long_series.iloc[-2]
            curr_diff = latest_short - latest_long
            if prev_diff <= 0 and curr_diff > 0:
                crossover = "bullish"
            elif prev_diff >= 0 and curr_diff < 0:
                crossover = "bearish"

        return TrendSignal(
            direction=float(direction),
            short_value=latest_short,
            long_value=latest_long,
            crossover=crossover,
            short_series=short_series,
            long_series=long_series,
        )

    def compute_linear_regression_trend(
        self,
        prices: pd.Series,
        use_log: bool = True,
    ) -> Dict[str, float]:
        if len(prices.dropna()) < 5:
            raise ValueError("Insufficient data for regression trend computation")

        y = np.log(prices) if use_log else prices.astype(float)
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=1)
        slope, intercept = coeffs
        fitted = slope * x + intercept
        residuals = y - fitted
        total_var = np.var(y)
        residual_var = np.var(residuals)
        r_squared = 1.0 - residual_var / total_var if total_var > 0 else 0.0
        annualized_slope = np.expm1(slope) * 252 if use_log else slope * 252
        noise_ratio = float(np.std(residuals) / (np.std(y) + 1e-6))
        return {
            "slope": float(annualized_slope),
            "intercept": float(intercept),
            "r_squared": float(max(0.0, min(1.0, r_squared))),
            "noise_ratio": float(max(0.0, min(1.0, noise_ratio))),
            "direction": float(np.sign(annualized_slope)),
        }

    def calculate_adx_trend_strength(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        if min(len(high), len(low), len(close)) < period + 1:
            return float("nan")

        high = high.astype(float)
        low = low.astype(float)
        close = close.astype(float)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        dm_plus = np.where((high - high.shift()) > (low.shift() - low), np.maximum(high - high.shift(), 0.0), 0.0)
        dm_minus = np.where((low.shift() - low) > (high - high.shift()), np.maximum(low.shift() - low, 0.0), 0.0)

        tr_smooth = pd.Series(true_range).rolling(window=period).mean()
        dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean()
        dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean()

        di_plus = 100 * (dm_plus_smooth / (tr_smooth + 1e-9))
        di_minus = 100 * (dm_minus_smooth / (tr_smooth + 1e-9))

        dx = (abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)) * 100
        adx = dx.rolling(window=period).mean().iloc[-1]
        return float(adx)

    def analyze_momentum_indicators(self, prices: pd.Series, high: Optional[pd.Series] = None, low: Optional[pd.Series] = None) -> MomentumAnalysis:
        closes = prices.astype(float)
        momentum_period = min(10, len(closes) - 1) if len(closes) > 1 else 1
        momentum = float(np.log(closes.iloc[-1] / closes.iloc[-momentum_period])) if momentum_period > 0 else 0.0

        rsi = self._compute_rsi(closes, period=14)
        macd_series, signal_series = self._compute_macd(closes)
        stochastic_k = self._compute_stochastic(closes, high, low)

        rsi_score = (rsi - 50.0) / 50.0  # -1 ~ 1
        macd_score = np.tanh(macd_series.iloc[-1])
        momentum_score = np.tanh(momentum * 10)
        stochastic_score = (stochastic_k - 50.0) / 50.0
        composite = float(np.clip((rsi_score + macd_score + momentum_score + stochastic_score) / 4.0, -1.0, 1.0))

        return MomentumAnalysis(
            momentum=float(momentum),
            rsi=float(rsi),
            macd=float(macd_series.iloc[-1]),
            macd_signal=float(signal_series.iloc[-1]),
            stochastic_k=float(stochastic_k),
            composite_score=composite,
        )

    def _compute_rsi(self, closes: pd.Series, period: int = 14) -> float:
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def _compute_macd(self, closes: pd.Series) -> (pd.Series, pd.Series):
        ema_fast = closes.ewm(span=12, adjust=False).mean()
        ema_slow = closes.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _compute_stochastic(
        self, closes: pd.Series, high: Optional[pd.Series], low: Optional[pd.Series], period: int = 14
    ) -> float:
        if high is None or low is None:
            high = closes.rolling(window=period).max()
            low = closes.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        stochastic_k = ((closes - lowest_low) / (highest_high - lowest_low + 1e-9)) * 100
        return float(stochastic_k.iloc[-1])


class StrengthAssessor:
    """趋势强度评估器"""

    def assess(
        self,
        trend_signal: TrendSignal,
        regression_metrics: Dict[str, float],
        adx_value: float,
        momentum: MomentumAnalysis,
    ) -> float:
        slope_score = self._normalize_slope(regression_metrics.get("slope", 0.0))
        r2_score = regression_metrics.get("r_squared", 0.0)
        adx_score = min(max(adx_value, 0.0) / 50.0, 1.0) if not np.isnan(adx_value) else 0.4
        momentum_score = (momentum.composite_score + 1.0) / 2.0
        direction_score = (trend_signal.direction + 1.0) / 2.0
        strength = (
            0.30 * slope_score
            + 0.25 * r2_score
            + 0.25 * adx_score
            + 0.15 * momentum_score
            + 0.05 * direction_score
        )
        return float(np.clip(strength, 0.0, 1.0))

    def _normalize_slope(self, slope: float) -> float:
        scale = 0.50  # 50% annualized slope ~ strong trend
        return float(np.clip(abs(slope) / scale, 0.0, 1.0))


class DurationAnalyzer:
    """趋势持续时间分析器"""

    def compute_duration(self, regime_series: Sequence[TrendRegime]) -> int:
        if not regime_series:
            return 0
        count = 0
        last_regime = regime_series[-1]
        for regime in reversed(regime_series):
            if regime == last_regime:
                count += 1
            else:
                break
        return count


class BreakoutDetector:
    """突破检测器"""

    def __init__(self, window: int = 20, tolerance: float = 0.005, volume_factor: float = 1.5) -> None:
        self.window = window
        self.tolerance = tolerance
        self.volume_factor = volume_factor

    def detect(
        self,
        closes: pd.Series,
        highs: Optional[pd.Series] = None,
        lows: Optional[pd.Series] = None,
        volumes: Optional[pd.Series] = None,
    ) -> List[BreakoutSignal]:
        if len(closes) < self.window + 2:
            return []

        highs = highs if highs is not None else closes
        lows = lows if lows is not None else closes

        rolling_high = highs.rolling(window=self.window).max().iloc[-1]
        rolling_low = lows.rolling(window=self.window).min().iloc[-1]
        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])

        signals: List[BreakoutSignal] = []
        timestamp = closes.index[-1] if isinstance(closes.index, pd.DatetimeIndex) else pd.Timestamp.utcnow()
        volume_confirm = 1.0
        if volumes is not None and len(volumes) >= self.window:
            avg_vol = float(volumes.iloc[-self.window :].mean())
            last_vol = float(volumes.iloc[-1])
            volume_confirm = min(last_vol / (avg_vol + 1e-9), 3.0)

        # Resistance breakout
        if last_close > rolling_high * (1 + self.tolerance) and prev_close <= rolling_high:
            confidence = min(1.0, 0.6 + 0.2 * volume_confirm)
            signals.append(
                BreakoutSignal(
                    signal_type="resistance_break",
                    level=float(rolling_high),
                    direction="up",
                    confidence=confidence,
                    timestamp=timestamp,
                )
            )

        # Support breakdown
        if last_close < rolling_low * (1 - self.tolerance) and prev_close >= rolling_low:
            confidence = min(1.0, 0.6 + 0.2 * volume_confirm)
            signals.append(
                BreakoutSignal(
                    signal_type="support_break",
                    level=float(rolling_low),
                    direction="down",
                    confidence=confidence,
                    timestamp=timestamp,
                )
            )

        return signals


class TrendAnalyzer:
    """趋势分析器主类"""

    def __init__(
        self,
        lookback: int = 252,
        short_ma: int = 20,
        long_ma: int = 60,
        calculator: Optional[TrendCalculator] = None,
        strength_assessor: Optional[StrengthAssessor] = None,
        duration_analyzer: Optional[DurationAnalyzer] = None,
        breakout_detector: Optional[BreakoutDetector] = None,
    ) -> None:
        self.lookback = lookback
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.calculator = calculator or TrendCalculator()
        self.strength_assessor = strength_assessor or StrengthAssessor()
        self.duration_analyzer = duration_analyzer or DurationAnalyzer()
        self.breakout_detector = breakout_detector or BreakoutDetector()

    def analyze(self, market_data: pd.DataFrame) -> TrendAnalysisResult:
        """
        执行趋势分析
        
        Args:
            market_data: 包含close, high, low, volume等列的DataFrame
        """
        if "close" not in market_data.columns:
            raise ValueError("market_data must include 'close' column")

        data = market_data.copy().astype(float)
        close = data["close"].dropna()
        if len(close) < max(self.long_ma + 5, 60):
            raise ValueError("Insufficient data for trend analysis")

        close = close.iloc[-self.lookback :]
        high = data["high"].reindex(close.index).fillna(close) if "high" in data.columns else close
        low = data["low"].reindex(close.index).fillna(close) if "low" in data.columns else close
        volume = data["volume"].reindex(close.index) if "volume" in data.columns else None

        trend_signal = self.calculator.calculate_moving_average_trend(close, self.short_ma, self.long_ma)
        regression_metrics = self.calculator.compute_linear_regression_trend(close.iloc[-self.long_ma :])
        adx_value = self.calculator.calculate_adx_trend_strength(high, low, close)
        momentum = self.calculator.analyze_momentum_indicators(close, high, low)
        trend_strength = self.strength_assessor.assess(trend_signal, regression_metrics, adx_value, momentum)
        trend_direction = float(np.clip((trend_signal.direction + regression_metrics.get("direction", 0.0)) / 2.0, -1.0, 1.0))

        current_regime = self._determine_regime(trend_signal, regression_metrics, adx_value, momentum, trend_strength)
        regime_series = self._build_regime_series(trend_signal.short_series, trend_signal.long_series, regression_metrics, adx_value, momentum)
        duration = self.duration_analyzer.compute_duration(regime_series)
        support_resistance = self._compute_support_resistance(high, low, close)
        breakout_signals = self.breakout_detector.detect(close, high, low, volume)
        confidence = self._compute_confidence(trend_strength, adx_value, momentum, breakout_signals, current_regime)

        return TrendAnalysisResult(
            current_regime=current_regime,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            confidence=confidence,
            duration_periods=duration,
            support_resistance_levels=support_resistance,
            breakout_signals=breakout_signals,
        )

    def _determine_regime(
        self,
        trend_signal: TrendSignal,
        regression_metrics: Dict[str, float],
        adx_value: float,
        momentum: MomentumAnalysis,
        trend_strength: float,
    ) -> TrendRegime:
        slope = regression_metrics.get("slope", 0.0)
        composite = momentum.composite_score
        direction = trend_signal.direction
        adx_threshold = 18.0 if np.isnan(adx_value) else adx_value

        if trend_strength < 0.35 or abs(slope) < 0.05 or adx_threshold < 18.0 or abs(composite) < 0.15:
            return TrendRegime.SIDEWAYS
        if direction > 0 and slope > 0 and composite > 0:
            return TrendRegime.UPTREND
        if direction < 0 and slope < 0 and composite < 0:
            return TrendRegime.DOWNTREND
        return TrendRegime.TREND_REVERSAL

    def _build_regime_series(
        self,
        short_ma_series: pd.Series,
        long_ma_series: pd.Series,
        regression_metrics: Dict[str, float],
        adx_value: float,
        momentum: MomentumAnalysis,
    ) -> List[TrendRegime]:
        regimes: List[TrendRegime] = []
        composite = momentum.composite_score
        slope = regression_metrics.get("slope", 0.0)
        adx_threshold = 18.0 if np.isnan(adx_value) else adx_value

        for short, long in zip(short_ma_series.iloc[self.long_ma - 1 :], long_ma_series.iloc[self.long_ma - 1 :]):
            direction = float(np.tanh((short - long) / (abs(long) + 1e-6)))
            strength_proxy = min(abs(short - long) / (abs(long) + 1e-6), 1.0)
            if strength_proxy < 0.05 or abs(slope) < 0.05 or adx_threshold < 18.0 or abs(composite) < 0.15:
                regimes.append(TrendRegime.SIDEWAYS)
            elif direction > 0 and slope > 0 and composite > 0:
                regimes.append(TrendRegime.UPTREND)
            elif direction < 0 and slope < 0 and composite < 0:
                regimes.append(TrendRegime.DOWNTREND)
            else:
                regimes.append(TrendRegime.TREND_REVERSAL)
        return regimes

    def _compute_support_resistance(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> Dict[str, float]:
        recent_high = float(high.rolling(window=window).max().iloc[-1])
        recent_low = float(low.rolling(window=window).min().iloc[-1])
        pivot = (recent_high + recent_low + float(close.iloc[-1])) / 3.0
        atr = self._compute_atr(high, low, close, period=window // 2)
        return {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "pivot": pivot,
            "atr": atr,
        }

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return float(atr)

    def _compute_confidence(
        self,
        strength: float,
        adx_value: float,
        momentum: MomentumAnalysis,
        breakout_signals: Sequence[BreakoutSignal],
        regime: TrendRegime,
    ) -> float:
        adx_score = min(max(adx_value, 0.0) / 50.0, 1.0) if not np.isnan(adx_value) else 0.4
        breakout_penalty = 0.05 * len([b for b in breakout_signals if b.confidence < 0.7])
        composite_score = (momentum.composite_score + 1.0) / 2.0
        confidence = 0.45 * strength + 0.30 * adx_score + 0.20 * composite_score + 0.05
        confidence = confidence - breakout_penalty
        if regime == TrendRegime.TREND_REVERSAL:
            confidence *= 0.8
        return float(np.clip(confidence, 0.0, 1.0))

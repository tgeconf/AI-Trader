"""
信号生成器
实现多时间框架技术指标信号生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import talib
from datetime import datetime, timedelta

from .timeframe_integration import Timeframe, TimeframeSignal


class SignalType(Enum):
    """信号类型枚举"""

    TREND_FOLLOWING = "trend_following"  # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"  # 均值回归
    MOMENTUM = "momentum"  # 动量
    VOLATILITY_BREAKOUT = "volatility_breakout"  # 波动率突破
    SUPPORT_RESISTANCE = "support_resistance"  # 支撑阻力


@dataclass
class TechnicalIndicator:
    """技术指标配置"""

    name: str
    parameters: Dict[str, Any]
    weight: float
    signal_type: SignalType


class SignalGenerator:
    """
    信号生成器

    实现多时间框架技术指标信号生成：
    - 趋势指标：MA, EMA, MACD, ADX
    - 动量指标：RSI, Stochastic, CCI, Williams %R
    - 波动率指标：Bollinger Bands, ATR, Keltner Channels
    - 成交量指标：OBV, Volume Profile, MFI
    - 支撑阻力：Pivot Points, Fibonacci
    """

    def __init__(
        self,
        default_indicators: Optional[List[TechnicalIndicator]] = None,
        signal_threshold: float = 0.3,
    ):
        """
        初始化信号生成器

        Args:
            default_indicators: 默认技术指标配置
            signal_threshold: 信号阈值
        """
        self.signal_threshold = signal_threshold
        self.indicators = default_indicators or self._get_default_indicators()
        self.signal_history = {}

    def _get_default_indicators(self) -> List[TechnicalIndicator]:
        """获取默认技术指标配置"""
        return [
            # 趋势指标
            TechnicalIndicator(
                name="EMA",
                parameters={"fast_period": 12, "slow_period": 26},
                weight=0.15,
                signal_type=SignalType.TREND_FOLLOWING,
            ),
            TechnicalIndicator(
                name="MACD",
                parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
                weight=0.20,
                signal_type=SignalType.TREND_FOLLOWING,
            ),
            TechnicalIndicator(
                name="ADX",
                parameters={"period": 14},
                weight=0.10,
                signal_type=SignalType.TREND_FOLLOWING,
            ),
            # 动量指标
            TechnicalIndicator(
                name="RSI",
                parameters={"period": 14},
                weight=0.15,
                signal_type=SignalType.MOMENTUM,
            ),
            TechnicalIndicator(
                name="Stochastic",
                parameters={"k_period": 14, "d_period": 3},
                weight=0.10,
                signal_type=SignalType.MOMENTUM,
            ),
            # 波动率指标
            TechnicalIndicator(
                name="Bollinger_Bands",
                parameters={"period": 20, "std_dev": 2},
                weight=0.15,
                signal_type=SignalType.VOLATILITY_BREAKOUT,
            ),
            TechnicalIndicator(
                name="ATR",
                parameters={"period": 14},
                weight=0.10,
                signal_type=SignalType.VOLATILITY_BREAKOUT,
            ),
            # 成交量指标
            TechnicalIndicator(
                name="OBV", parameters={}, weight=0.05, signal_type=SignalType.MOMENTUM
            ),
        ]

    def generate_timeframe_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        custom_indicators: Optional[List[TechnicalIndicator]] = None,
    ) -> List[TimeframeSignal]:
        """
        为指定时间框架生成技术指标信号

        Args:
            df: 价格数据DataFrame
            symbol: 股票代码
            timeframe: 时间框架
            custom_indicators: 自定义指标配置

        Returns:
            时间框架信号列表
        """
        indicators_to_use = custom_indicators or self.indicators
        signals = []

        for indicator in indicators_to_use:
            try:
                signal = self._generate_single_indicator_signal(
                    df=df, symbol=symbol, timeframe=timeframe, indicator=indicator
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"Warning: Failed to generate {indicator.name} signal: {e}")
                continue

        return signals

    def _generate_single_indicator_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        indicator: TechnicalIndicator,
    ) -> Optional[TimeframeSignal]:
        """生成单个技术指标信号"""
        # 提取价格数据
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else None

        # 根据指标类型生成信号
        if indicator.name == "EMA":
            signal_strength, confidence, indicators_dict = self._ema_signal(
                close, indicator.parameters
            )
        elif indicator.name == "MACD":
            signal_strength, confidence, indicators_dict = self._macd_signal(
                close, indicator.parameters
            )
        elif indicator.name == "ADX":
            signal_strength, confidence, indicators_dict = self._adx_signal(
                high, low, close, indicator.parameters
            )
        elif indicator.name == "RSI":
            signal_strength, confidence, indicators_dict = self._rsi_signal(
                close, indicator.parameters
            )
        elif indicator.name == "Stochastic":
            signal_strength, confidence, indicators_dict = self._stochastic_signal(
                high, low, close, indicator.parameters
            )
        elif indicator.name == "Bollinger_Bands":
            signal_strength, confidence, indicators_dict = self._bollinger_bands_signal(
                close, indicator.parameters
            )
        elif indicator.name == "ATR":
            signal_strength, confidence, indicators_dict = self._atr_signal(
                high, low, close, indicator.parameters
            )
        elif indicator.name == "OBV":
            signal_strength, confidence, indicators_dict = self._obv_signal(
                close, volume, indicator.parameters
            )
        else:
            return None

        # 应用指标权重
        weighted_signal = signal_strength * indicator.weight
        weighted_confidence = confidence * indicator.weight

        # 只有当信号强度超过阈值时才生成信号
        if abs(weighted_signal) < self.signal_threshold:
            return None

        return TimeframeSignal(
            timeframe=timeframe,
            symbol=symbol,
            signal_strength=weighted_signal,
            confidence=weighted_confidence,
            timestamp=pd.Timestamp.now(),
            indicators=indicators_dict,
        )

    def _ema_signal(
        self, close: np.ndarray, parameters: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """EMA信号生成"""
        fast_period = parameters.get("fast_period", 12)
        slow_period = parameters.get("slow_period", 26)

        # 计算EMA
        ema_fast = talib.EMA(close, timeperiod=fast_period)
        ema_slow = talib.EMA(close, timeperiod=slow_period)

        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return 0.0, 0.0, {}

        # 计算信号
        current_fast = ema_fast[-1]
        current_slow = ema_slow[-1]
        prev_fast = ema_fast[-2]
        prev_slow = ema_slow[-2]

        # 金叉死叉信号
        if current_fast > current_slow and prev_fast <= prev_slow:
            signal_strength = 1.0  # 金叉买入信号
        elif current_fast < current_slow and prev_fast >= prev_slow:
            signal_strength = -1.0  # 死叉卖出信号
        else:
            # 趋势强度
            trend_strength = (current_fast - current_slow) / current_slow
            signal_strength = np.clip(trend_strength * 10, -1.0, 1.0)

        # 置信度基于EMA距离和趋势持续性
        ema_distance = abs(current_fast - current_slow) / current_slow
        confidence = min(ema_distance * 20, 0.8)

        indicators = {
            "ema_fast": current_fast,
            "ema_slow": current_slow,
            "ema_distance": ema_distance,
        }

        return signal_strength, confidence, indicators

    def _macd_signal(
        self, close: np.ndarray, parameters: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """MACD信号生成"""
        fast_period = parameters.get("fast_period", 12)
        slow_period = parameters.get("slow_period", 26)
        signal_period = parameters.get("signal_period", 9)

        # 计算MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period,
        )

        if len(macd) < 2 or len(macd_signal) < 2:
            return 0.0, 0.0, {}

        current_macd = macd[-1]
        current_signal = macd_signal[-1]
        current_hist = macd_hist[-1]
        prev_macd = macd[-2]
        prev_signal = macd_signal[-2]

        # MACD信号逻辑
        if current_macd > current_signal and prev_macd <= prev_signal:
            signal_strength = 1.0  # MACD上穿信号线
        elif current_macd < current_signal and prev_macd >= prev_signal:
            signal_strength = -1.0  # MACD下穿信号线
        else:
            # 基于MACD柱状图的信号强度
            signal_strength = np.clip(current_hist * 10, -1.0, 1.0)

        # 置信度基于MACD柱状图大小
        hist_abs = abs(current_hist)
        confidence = min(hist_abs * 50, 0.8)

        indicators = {
            "macd": current_macd,
            "macd_signal": current_signal,
            "macd_histogram": current_hist,
        }

        return signal_strength, confidence, indicators

    def _adx_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        parameters: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, float]]:
        """ADX趋势强度信号"""
        period = parameters.get("period", 14)

        # 计算ADX
        adx = talib.ADX(high, low, close, timeperiod=period)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)

        if len(adx) < 1:
            return 0.0, 0.0, {}

        current_adx = adx[-1]
        current_plus_di = plus_di[-1]
        current_minus_di = minus_di[-1]

        # ADX信号逻辑
        if current_adx > 25:  # 强趋势
            if current_plus_di > current_minus_di:
                signal_strength = 1.0  # 上升趋势
            else:
                signal_strength = -1.0  # 下降趋势
        else:
            signal_strength = 0.0  # 震荡市场

        # 置信度基于ADX值
        confidence = min(current_adx / 50, 0.8)

        indicators = {
            "adx": current_adx,
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
        }

        return signal_strength, confidence, indicators

    def _rsi_signal(
        self, close: np.ndarray, parameters: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """RSI超买超卖信号"""
        period = parameters.get("period", 14)

        # 计算RSI
        rsi = talib.RSI(close, timeperiod=period)

        if len(rsi) < 1:
            return 0.0, 0.0, {}

        current_rsi = rsi[-1]

        # RSI信号逻辑
        if current_rsi < 30:
            signal_strength = 1.0  # 超卖，买入信号
        elif current_rsi > 70:
            signal_strength = -1.0  # 超买，卖出信号
        else:
            signal_strength = 0.0

        # 置信度基于RSI偏离程度
        if current_rsi < 30:
            confidence = (30 - current_rsi) / 30
        elif current_rsi > 70:
            confidence = (current_rsi - 70) / 30
        else:
            confidence = 0.0

        indicators = {"rsi": current_rsi}

        return signal_strength, confidence, indicators

    def _stochastic_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        parameters: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, float]]:
        """随机指标信号"""
        k_period = parameters.get("k_period", 14)
        d_period = parameters.get("d_period", 3)

        # 计算随机指标
        slowk, slowd = talib.STOCH(
            high,
            low,
            close,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0,
        )

        if len(slowk) < 2 or len(slowd) < 2:
            return 0.0, 0.0, {}

        current_k = slowk[-1]
        current_d = slowd[-1]
        prev_k = slowk[-2]
        prev_d = slowd[-2]

        # 随机指标信号逻辑
        if current_k < 20 and current_d < 20:  # 超卖区域
            if current_k > current_d and prev_k <= prev_d:  # K线上穿D线
                signal_strength = 1.0
            else:
                signal_strength = 0.5
        elif current_k > 80 and current_d > 80:  # 超买区域
            if current_k < current_d and prev_k >= prev_d:  # K线下穿D线
                signal_strength = -1.0
            else:
                signal_strength = -0.5
        else:
            signal_strength = 0.0

        # 置信度基于超买超卖程度
        if current_k < 20:
            confidence = (20 - current_k) / 20
        elif current_k > 80:
            confidence = (current_k - 80) / 20
        else:
            confidence = 0.3

        indicators = {"stochastic_k": current_k, "stochastic_d": current_d}

        return signal_strength, confidence, indicators

    def _bollinger_bands_signal(
        self, close: np.ndarray, parameters: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """布林带信号"""
        period = parameters.get("period", 20)
        std_dev = parameters.get("std_dev", 2)

        # 计算布林带
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )

        if len(upper) < 1:
            return 0.0, 0.0, {}

        current_close = close[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_middle = middle[-1]

        # 布林带信号逻辑
        if current_close > current_upper:
            signal_strength = -1.0  # 上轨突破，卖出信号
        elif current_close < current_lower:
            signal_strength = 1.0  # 下轨突破，买入信号
        else:
            # 基于中轨的趋势信号
            if current_close > current_middle:
                signal_strength = 0.3  # 中轨上方，轻微买入
            else:
                signal_strength = -0.3  # 中轨下方，轻微卖出

        # 置信度基于价格与轨道的距离
        if current_close > current_upper:
            confidence = (current_close - current_upper) / current_upper
        elif current_close < current_lower:
            confidence = (current_lower - current_close) / current_close
        else:
            confidence = 0.3

        indicators = {
            "bb_upper": current_upper,
            "bb_middle": current_middle,
            "bb_lower": current_lower,
            "bb_position": (current_close - current_lower)
            / (current_upper - current_lower),
        }

        return signal_strength, confidence, indicators

    def _atr_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        parameters: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, float]]:
        """ATR波动率信号"""
        period = parameters.get("period", 14)

        # 计算ATR
        atr = talib.ATR(high, low, close, timeperiod=period)

        if len(atr) < 2:
            return 0.0, 0.0, {}

        current_atr = atr[-1]
        prev_atr = atr[-2]

        # ATR信号逻辑：高波动率通常伴随趋势
        atr_change = (current_atr - prev_atr) / prev_atr if prev_atr > 0 else 0

        if atr_change > 0.1:  # ATR显著增加，可能开始趋势
            signal_strength = 0.5  # 轻微买入信号（趋势开始）
        elif atr_change < -0.1:  # ATR显著减少，可能进入盘整
            signal_strength = -0.3  # 轻微卖出信号（趋势结束）
        else:
            signal_strength = 0.0

        # 置信度基于ATR变化幅度
        confidence = min(abs(atr_change) * 5, 0.6)

        indicators = {"atr": current_atr, "atr_change": atr_change}

        return signal_strength, confidence, indicators

    def _obv_signal(
        self,
        close: np.ndarray,
        volume: Optional[np.ndarray],
        parameters: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, float]]:
        """能量潮信号"""
        if volume is None:
            return 0.0, 0.0, {}

        # 计算OBV
        obv = talib.OBV(close, volume)

        if len(obv) < 2:
            return 0.0, 0.0, {}

        current_obv = obv[-1]
        prev_obv = obv[-2]

        # OBV信号逻辑
        obv_change = (current_obv - prev_obv) / abs(prev_obv) if prev_obv != 0 else 0

        if obv_change > 0.02:  # OBV显著上升
            signal_strength = 0.8  # 买入信号
        elif obv_change < -0.02:  # OBV显著下降
            signal_strength = -0.8  # 卖出信号
        else:
            signal_strength = 0.0

        # 置信度基于OBV变化幅度
        confidence = min(abs(obv_change) * 25, 0.7)

        indicators = {"obv": current_obv, "obv_change": obv_change}

        return signal_strength, confidence, indicators

    def optimize_indicator_weights(
        self, performance_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        优化指标权重

        Args:
            performance_data: 性能数据，包含各指标信号和实际收益

        Returns:
            优化后的指标权重
        """
        indicator_performance = {}

        for indicator in self.indicators:
            indicator_name = indicator.name
            if indicator_name not in performance_data.columns:
                continue

            # 计算该指标信号的性能
            signals = performance_data[indicator_name]
            returns = performance_data["returns"]

            # 计算信号与收益的相关性（绝对值）
            correlation = abs(signals.corr(returns))
            indicator_performance[indicator_name] = correlation

        # 归一化性能得分
        total_performance = sum(indicator_performance.values())
        if total_performance > 0:
            optimized_weights = {
                name: perf / total_performance
                for name, perf in indicator_performance.items()
            }
        else:
            optimized_weights = {ind.name: ind.weight for ind in self.indicators}

        return optimized_weights

    def get_signal_statistics(self) -> Dict[str, Any]:
        """获取信号统计信息"""
        return {
            "total_indicators": len(self.indicators),
            "signal_threshold": self.signal_threshold,
            "indicator_types": list(
                set(ind.signal_type.value for ind in self.indicators)
            ),
            "signal_history_size": sum(
                len(signals) for signals in self.signal_history.values()
            ),
        }

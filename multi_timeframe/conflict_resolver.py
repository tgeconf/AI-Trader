"""
冲突解决器
实现多时间框架信号冲突的检测与解决
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .timeframe_integration import TimeframeSignal, Timeframe


class ConflictType(Enum):
    """冲突类型枚举"""

    DIRECTION_CONFLICT = "direction_conflict"  # 方向冲突
    STRENGTH_CONFLICT = "strength_conflict"  # 强度冲突
    TIMEFRAME_CONFLICT = "timeframe_conflict"  # 时间框架冲突
    CONFIDENCE_CONFLICT = "confidence_conflict"  # 置信度冲突


class ResolutionMethod(Enum):
    """解决方法枚举"""

    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    MAJORITY_VOTE = "majority_vote"  # 多数投票
    CONFIDENCE_BASED = "confidence_based"  # 置信度优先
    TIMEFRAME_HIERARCHY = "timeframe_hierarchy"  # 时间框架层级
    RISK_AVERSE = "risk_averse"  # 风险规避


@dataclass
class SignalConflict:
    """信号冲突"""

    conflict_type: ConflictType
    timeframe_pairs: List[Tuple[Timeframe, Timeframe]]  # 冲突的时间框架对
    conflict_score: float  # 冲突严重程度（0-1）
    description: str  # 冲突描述


@dataclass
class ConflictResolution:
    """冲突解决结果"""

    resolved: bool  # 是否解决冲突
    resolution_method: ResolutionMethod  # 解决方法
    original_signals: List[TimeframeSignal]  # 原始信号
    resolved_signals: List[TimeframeSignal]  # 解决后的信号
    conflict_analysis: Dict[str, Any]  # 冲突分析
    confidence_after_resolution: float  # 解决后的置信度


class ConflictResolver:
    """
    冲突解决器

    实现多时间框架信号冲突的检测与解决：
    - 冲突检测与分类
    - 多种解决方法
    - 冲突严重程度评估
    - 解决效果验证
    - 自适应解决方法选择
    """

    def __init__(
        self,
        conflict_threshold: float = 0.3,
        default_resolution_method: ResolutionMethod = ResolutionMethod.WEIGHTED_AVERAGE,
    ):
        """
        初始化冲突解决器

        Args:
            conflict_threshold: 冲突阈值
            default_resolution_method: 默认解决方法
        """
        self.conflict_threshold = conflict_threshold
        self.default_resolution_method = default_resolution_method
        self.resolution_history = {}
        self.conflict_patterns = {}

    def detect_conflicts(self, signals: List[TimeframeSignal]) -> List[SignalConflict]:
        """
        检测信号冲突

        Args:
            signals: 时间框架信号列表

        Returns:
            信号冲突列表
        """
        conflicts = []

        if len(signals) < 2:
            return conflicts

        # 方向冲突检测
        direction_conflicts = self._detect_direction_conflicts(signals)
        if direction_conflicts:
            conflicts.append(direction_conflicts)

        # 强度冲突检测
        strength_conflicts = self._detect_strength_conflicts(signals)
        if strength_conflicts:
            conflicts.append(strength_conflicts)

        # 时间框架冲突检测
        timeframe_conflicts = self._detect_timeframe_conflicts(signals)
        if timeframe_conflicts:
            conflicts.append(timeframe_conflicts)

        # 置信度冲突检测
        confidence_conflicts = self._detect_confidence_conflicts(signals)
        if confidence_conflicts:
            conflicts.append(confidence_conflicts)

        return conflicts

    def _detect_direction_conflicts(
        self, signals: List[TimeframeSignal]
    ) -> Optional[SignalConflict]:
        """检测方向冲突"""
        bullish_signals = [s for s in signals if s.signal_strength > 0]
        bearish_signals = [s for s in signals if s.signal_strength < 0]

        if not bullish_signals or not bearish_signals:
            return None

        # 计算冲突严重程度
        total_signals = len(signals)
        conflict_ratio = min(len(bullish_signals), len(bearish_signals)) / total_signals

        if conflict_ratio < self.conflict_threshold:
            return None

        # 识别冲突的时间框架对
        conflict_pairs = []
        for bull_signal in bullish_signals:
            for bear_signal in bearish_signals:
                conflict_pairs.append((bull_signal.timeframe, bear_signal.timeframe))

        return SignalConflict(
            conflict_type=ConflictType.DIRECTION_CONFLICT,
            timeframe_pairs=conflict_pairs,
            conflict_score=conflict_ratio,
            description=f"方向冲突：{len(bullish_signals)}个看涨 vs {len(bearish_signals)}个看跌",
        )

    def _detect_strength_conflicts(
        self, signals: List[TimeframeSignal]
    ) -> Optional[SignalConflict]:
        """检测强度冲突"""
        if len(signals) < 2:
            return None

        # 计算信号强度的标准差
        signal_strengths = [s.signal_strength for s in signals]
        strength_std = np.std(signal_strengths)

        if strength_std < 0.2:  # 强度差异不大
            return None

        # 识别强度差异大的时间框架对
        conflict_pairs = []
        for i, signal_i in enumerate(signals):
            for j, signal_j in enumerate(signals):
                if (
                    i < j
                    and abs(signal_i.signal_strength - signal_j.signal_strength) > 0.5
                ):
                    conflict_pairs.append((signal_i.timeframe, signal_j.timeframe))

        if not conflict_pairs:
            return None

        conflict_score = min(strength_std, 1.0)

        return SignalConflict(
            conflict_type=ConflictType.STRENGTH_CONFLICT,
            timeframe_pairs=conflict_pairs,
            conflict_score=conflict_score,
            description=f"强度冲突：信号强度标准差 {strength_std:.3f}",
        )

    def _detect_timeframe_conflicts(
        self, signals: List[TimeframeSignal]
    ) -> Optional[SignalConflict]:
        """检测时间框架冲突"""
        # 按时间框架层级分组
        short_term = [
            s
            for s in signals
            if s.timeframe
            in [Timeframe.MINUTE_1, Timeframe.MINUTE_5, Timeframe.MINUTE_15]
        ]
        medium_term = [
            s
            for s in signals
            if s.timeframe in [Timeframe.MINUTE_30, Timeframe.HOUR_1, Timeframe.HOUR_4]
        ]
        long_term = [
            s
            for s in signals
            if s.timeframe in [Timeframe.DAILY, Timeframe.WEEKLY, Timeframe.MONTHLY]
        ]

        # 检查不同层级间的方向一致性
        conflict_pairs = []

        for st_signal in short_term:
            for lt_signal in long_term:
                if (
                    st_signal.signal_strength > 0 and lt_signal.signal_strength < 0
                ) or (st_signal.signal_strength < 0 and lt_signal.signal_strength > 0):
                    conflict_pairs.append((st_signal.timeframe, lt_signal.timeframe))

        if not conflict_pairs:
            return None

        # 计算冲突严重程度
        total_possible_pairs = len(short_term) * len(long_term)
        conflict_ratio = (
            len(conflict_pairs) / total_possible_pairs
            if total_possible_pairs > 0
            else 0
        )

        return SignalConflict(
            conflict_type=ConflictType.TIMEFRAME_CONFLICT,
            timeframe_pairs=conflict_pairs,
            conflict_score=conflict_ratio,
            description=f"时间框架冲突：短期与长期信号方向不一致",
        )

    def _detect_confidence_conflicts(
        self, signals: List[TimeframeSignal]
    ) -> Optional[SignalConflict]:
        """检测置信度冲突"""
        if len(signals) < 2:
            return None

        # 计算置信度的标准差
        confidences = [s.confidence for s in signals]
        confidence_std = np.std(confidences)

        if confidence_std < 0.2:  # 置信度差异不大
            return None

        # 识别置信度差异大的时间框架对
        conflict_pairs = []
        for i, signal_i in enumerate(signals):
            for j, signal_j in enumerate(signals):
                if i < j and abs(signal_i.confidence - signal_j.confidence) > 0.4:
                    conflict_pairs.append((signal_i.timeframe, signal_j.timeframe))

        if not conflict_pairs:
            return None

        conflict_score = min(confidence_std, 1.0)

        return SignalConflict(
            conflict_type=ConflictType.CONFIDENCE_CONFLICT,
            timeframe_pairs=conflict_pairs,
            conflict_score=conflict_score,
            description=f"置信度冲突：置信度标准差 {confidence_std:.3f}",
        )

    def resolve_conflicts(
        self,
        signals: List[TimeframeSignal],
        resolution_method: Optional[ResolutionMethod] = None,
    ) -> ConflictResolution:
        """
        解决信号冲突

        Args:
            signals: 时间框架信号列表
            resolution_method: 解决方法

        Returns:
            冲突解决结果
        """
        method = resolution_method or self.default_resolution_method
        conflicts = self.detect_conflicts(signals)

        if not conflicts:
            # 无冲突，直接返回原始信号
            return ConflictResolution(
                resolved=True,
                resolution_method=method,
                original_signals=signals,
                resolved_signals=signals,
                conflict_analysis={"conflicts_detected": 0},
                confidence_after_resolution=self._calculate_overall_confidence(signals),
            )

        # 根据方法解决冲突
        if method == ResolutionMethod.WEIGHTED_AVERAGE:
            resolved_signals = self._resolve_weighted_average(signals)
        elif method == ResolutionMethod.MAJORITY_VOTE:
            resolved_signals = self._resolve_majority_vote(signals)
        elif method == ResolutionMethod.CONFIDENCE_BASED:
            resolved_signals = self._resolve_confidence_based(signals)
        elif method == ResolutionMethod.TIMEFRAME_HIERARCHY:
            resolved_signals = self._resolve_timeframe_hierarchy(signals)
        elif method == ResolutionMethod.RISK_AVERSE:
            resolved_signals = self._resolve_risk_averse(signals)
        else:
            resolved_signals = signals  # 默认不处理

        # 计算解决后的置信度
        final_confidence = self._calculate_resolution_confidence(
            signals, resolved_signals, conflicts
        )

        return ConflictResolution(
            resolved=len(resolved_signals) > 0,
            resolution_method=method,
            original_signals=signals,
            resolved_signals=resolved_signals,
            conflict_analysis={
                "conflicts_detected": len(conflicts),
                "conflict_types": [
                    conflict.conflict_type.value for conflict in conflicts
                ],
                "total_conflict_score": sum(
                    conflict.conflict_score for conflict in conflicts
                ),
            },
            confidence_after_resolution=final_confidence,
        )

    def _resolve_weighted_average(
        self, signals: List[TimeframeSignal]
    ) -> List[TimeframeSignal]:
        """加权平均解决方法"""
        if not signals:
            return []

        symbol = signals[0].symbol

        # 按时间框架分组
        signals_by_timeframe = defaultdict(list)
        for signal in signals:
            signals_by_timeframe[signal.timeframe].append(signal)

        resolved_signals = []
        for timeframe, tf_signals in signals_by_timeframe.items():
            # 计算加权平均信号
            total_weight = sum(signal.confidence for signal in tf_signals)
            if total_weight == 0:
                continue

            weighted_signal = (
                sum(signal.signal_strength * signal.confidence for signal in tf_signals)
                / total_weight
            )

            # 计算平均置信度
            avg_confidence = sum(signal.confidence for signal in tf_signals) / len(
                tf_signals
            )

            # 合并技术指标
            combined_indicators = {}
            for signal in tf_signals:
                for key, value in signal.indicators.items():
                    if key not in combined_indicators:
                        combined_indicators[key] = []
                    combined_indicators[key].append(value)

            # 计算指标平均值
            for key in combined_indicators:
                combined_indicators[key] = np.mean(combined_indicators[key])

            resolved_signal = TimeframeSignal(
                timeframe=timeframe,
                symbol=symbol,
                signal_strength=weighted_signal,
                confidence=avg_confidence,
                timestamp=pd.Timestamp.now(),
                indicators=combined_indicators,
            )
            resolved_signals.append(resolved_signal)

        return resolved_signals

    def _resolve_majority_vote(
        self, signals: List[TimeframeSignal]
    ) -> List[TimeframeSignal]:
        """多数投票解决方法"""
        if not signals:
            return []

        symbol = signals[0].symbol

        # 统计方向
        bullish_count = sum(1 for s in signals if s.signal_strength > 0)
        bearish_count = sum(1 for s in signals if s.signal_strength < 0)
        neutral_count = sum(1 for s in signals if s.signal_strength == 0)

        # 确定多数方向
        if bullish_count > bearish_count and bullish_count > neutral_count:
            majority_direction = 1.0
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            majority_direction = -1.0
        else:
            majority_direction = 0.0

        # 应用多数方向
        resolved_signals = []
        for signal in signals:
            # 保持原信号强度但调整方向
            new_strength = abs(signal.signal_strength) * majority_direction

            resolved_signal = TimeframeSignal(
                timeframe=signal.timeframe,
                symbol=symbol,
                signal_strength=new_strength,
                confidence=signal.confidence,
                timestamp=signal.timestamp,
                indicators=signal.indicators,
            )
            resolved_signals.append(resolved_signal)

        return resolved_signals

    def _resolve_confidence_based(
        self, signals: List[TimeframeSignal]
    ) -> List[TimeframeSignal]:
        """置信度优先解决方法"""
        if not signals:
            return []

        symbol = signals[0].symbol

        # 找到最高置信度的信号
        max_confidence_signal = max(signals, key=lambda s: s.confidence)

        # 所有信号采用最高置信度信号的方向
        resolved_signals = []
        for signal in signals:
            # 保持原信号强度但采用最高置信度信号的方向
            direction = 1.0 if max_confidence_signal.signal_strength > 0 else -1.0
            new_strength = abs(signal.signal_strength) * direction

            # 调整置信度：向最高置信度靠拢
            adjusted_confidence = (
                signal.confidence + max_confidence_signal.confidence
            ) / 2

            resolved_signal = TimeframeSignal(
                timeframe=signal.timeframe,
                symbol=symbol,
                signal_strength=new_strength,
                confidence=adjusted_confidence,
                timestamp=signal.timestamp,
                indicators=signal.indicators,
            )
            resolved_signals.append(resolved_signal)

        return resolved_signals

    def _resolve_timeframe_hierarchy(
        self, signals: List[TimeframeSignal]
    ) -> List[TimeframeSignal]:
        """时间框架层级解决方法"""
        if not signals:
            return []

        symbol = signals[0].symbol

        # 定义时间框架层级权重
        timeframe_weights = {
            Timeframe.MONTHLY: 0.25,
            Timeframe.WEEKLY: 0.20,
            Timeframe.DAILY: 0.18,
            Timeframe.HOUR_4: 0.15,
            Timeframe.HOUR_1: 0.12,
            Timeframe.MINUTE_30: 0.06,
            Timeframe.MINUTE_15: 0.03,
            Timeframe.MINUTE_5: 0.01,
            Timeframe.MINUTE_1: 0.0,
        }

        # 计算加权方向
        weighted_direction = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = timeframe_weights.get(signal.timeframe, 0.0)
            weighted_direction += signal.signal_strength * weight
            total_weight += weight

        if total_weight > 0:
            final_direction = weighted_direction / total_weight
        else:
            final_direction = 0.0

        # 应用层级调整后的方向
        resolved_signals = []
        for signal in signals:
            # 保持相对强度但调整整体方向
            relative_strength = signal.signal_strength / max(
                abs(signal.signal_strength), 0.001
            )
            new_strength = relative_strength * final_direction

            # 根据层级调整置信度
            timeframe_weight = timeframe_weights.get(signal.timeframe, 0.0)
            adjusted_confidence = signal.confidence * (0.5 + timeframe_weight * 2)

            resolved_signal = TimeframeSignal(
                timeframe=signal.timeframe,
                symbol=symbol,
                signal_strength=new_strength,
                confidence=min(adjusted_confidence, 1.0),
                timestamp=signal.timestamp,
                indicators=signal.indicators,
            )
            resolved_signals.append(resolved_signal)

        return resolved_signals

    def _resolve_risk_averse(
        self, signals: List[TimeframeSignal]
    ) -> List[TimeframeSignal]:
        """风险规避解决方法"""
        if not signals:
            return []

        symbol = signals[0].symbol

        # 检测冲突严重程度
        conflicts = self.detect_conflicts(signals)
        total_conflict_score = sum(conflict.conflict_score for conflict in conflicts)

        # 冲突严重时降低信号强度
        conflict_reduction = 1.0 - min(total_conflict_score, 0.8)

        resolved_signals = []
        for signal in signals:
            # 降低信号强度
            reduced_strength = signal.signal_strength * conflict_reduction

            # 降低置信度
            reduced_confidence = signal.confidence * conflict_reduction

            resolved_signal = TimeframeSignal(
                timeframe=signal.timeframe,
                symbol=symbol,
                signal_strength=reduced_strength,
                confidence=reduced_confidence,
                timestamp=signal.timestamp,
                indicators=signal.indicators,
            )
            resolved_signals.append(resolved_signal)

        return resolved_signals

    def _calculate_overall_confidence(self, signals: List[TimeframeSignal]) -> float:
        """计算整体置信度"""
        if not signals:
            return 0.0

        # 加权平均置信度
        total_weight = sum(abs(signal.signal_strength) for signal in signals)
        if total_weight == 0:
            return np.mean([signal.confidence for signal in signals])

        weighted_confidence = (
            sum(signal.confidence * abs(signal.signal_strength) for signal in signals)
            / total_weight
        )

        return weighted_confidence

    def _calculate_resolution_confidence(
        self,
        original_signals: List[TimeframeSignal],
        resolved_signals: List[TimeframeSignal],
        conflicts: List[SignalConflict],
    ) -> float:
        """计算解决后的置信度"""
        if not resolved_signals:
            return 0.0

        # 基础置信度
        base_confidence = self._calculate_overall_confidence(resolved_signals)

        # 冲突解决效果加成
        conflict_reduction = (
            1.0
            - sum(conflict.conflict_score for conflict in conflicts) / len(conflicts)
            if conflicts
            else 1.0
        )

        # 信号一致性加成
        signal_strengths = [s.signal_strength for s in resolved_signals]
        consistency = (
            1.0 - np.std(signal_strengths) if len(signal_strengths) > 1 else 1.0
        )

        final_confidence = base_confidence * conflict_reduction * consistency
        return min(final_confidence, 1.0)

    def select_best_resolution_method(
        self, signals: List[TimeframeSignal], market_condition: str = "normal"
    ) -> ResolutionMethod:
        """
        选择最佳解决方法

        Args:
            signals: 时间框架信号列表
            market_condition: 市场条件

        Returns:
            最佳解决方法
        """
        conflicts = self.detect_conflicts(signals)

        if not conflicts:
            return ResolutionMethod.WEIGHTED_AVERAGE

        # 分析冲突类型
        conflict_types = [conflict.conflict_type for conflict in conflicts]
        total_conflict_score = sum(conflict.conflict_score for conflict in conflicts)

        # 根据冲突类型选择方法
        if ConflictType.DIRECTION_CONFLICT in conflict_types:
            if market_condition == "trending":
                return ResolutionMethod.TIMEFRAME_HIERARCHY
            else:
                return ResolutionMethod.MAJORITY_VOTE

        elif ConflictType.STRENGTH_CONFLICT in conflict_types:
            return ResolutionMethod.WEIGHTED_AVERAGE

        elif ConflictType.TIMEFRAME_CONFLICT in conflict_types:
            return ResolutionMethod.TIMEFRAME_HIERARCHY

        elif ConflictType.CONFIDENCE_CONFLICT in conflict_types:
            return ResolutionMethod.CONFIDENCE_BASED

        # 高冲突情况下使用风险规避
        if total_conflict_score > 0.7:
            return ResolutionMethod.RISK_AVERSE

        return self.default_resolution_method

    def update_resolution_history(
        self, symbol: str, resolution: ConflictResolution
    ) -> None:
        """
        更新解决历史

        Args:
            symbol: 股票代码
            resolution: 冲突解决结果
        """
        if symbol not in self.resolution_history:
            self.resolution_history[symbol] = []

        self.resolution_history[symbol].append(
            {
                "timestamp": pd.Timestamp.now(),
                "resolution_method": resolution.resolution_method.value,
                "conflicts_detected": resolution.conflict_analysis[
                    "conflicts_detected"
                ],
                "confidence_after": resolution.confidence_after_resolution,
                "successful": resolution.resolved,
            }
        )

        # 只保留最近50条记录
        if len(self.resolution_history[symbol]) > 50:
            self.resolution_history[symbol] = self.resolution_history[symbol][-50:]

    def analyze_resolution_effectiveness(self, symbol: str) -> Dict[str, Any]:
        """
        分析解决效果

        Args:
            symbol: 股票代码

        Returns:
            解决效果分析
        """
        if symbol not in self.resolution_history:
            return {"error": "No resolution history for symbol"}

        history = self.resolution_history[symbol]
        if not history:
            return {"error": "Empty resolution history"}

        # 分析方法效果
        method_performance = {}
        for record in history:
            method = record["resolution_method"]
            if method not in method_performance:
                method_performance[method] = {
                    "count": 0,
                    "total_confidence": 0,
                    "success_count": 0,
                }

            perf = method_performance[method]
            perf["count"] += 1
            perf["total_confidence"] += record["confidence_after"]
            if record["successful"]:
                perf["success_count"] += 1

        # 计算平均表现
        for method, perf in method_performance.items():
            perf["avg_confidence"] = perf["total_confidence"] / perf["count"]
            perf["success_rate"] = perf["success_count"] / perf["count"]

        return {
            "symbol": symbol,
            "total_resolutions": len(history),
            "method_performance": method_performance,
            "best_method": (
                max(method_performance.items(), key=lambda x: x[1]["success_rate"])[0]
                if method_performance
                else None
            ),
        }

    def get_resolver_statistics(self) -> Dict[str, Any]:
        """获取解决器统计信息"""
        return {
            "conflict_threshold": self.conflict_threshold,
            "default_resolution_method": self.default_resolution_method.value,
            "resolution_history_size": sum(
                len(history) for history in self.resolution_history.values()
            ),
            "conflict_patterns_tracked": len(self.conflict_patterns),
        }

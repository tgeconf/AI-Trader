"""
时间框架整合器
实现多时间框架信号的整合与权重分配
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Timeframe(Enum):
    """时间框架枚举"""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"


@dataclass
class TimeframeSignal:
    """时间框架信号"""
    timeframe: Timeframe
    symbol: str
    signal_strength: float  # -1到1，负数为卖出，正数为买入
    confidence: float  # 0到1，信号置信度
    timestamp: pd.Timestamp
    indicators: Dict[str, float]  # 技术指标值


@dataclass
class IntegratedSignal:
    """整合信号"""
    symbol: str
    final_signal: float  # 最终信号强度
    timeframe_contributions: Dict[Timeframe, float]  # 各时间框架贡献
    consensus_level: float  # 一致性水平
    conflict_resolved: bool  # 是否解决冲突
    recommendation: str  # 交易建议


class TimeframeIntegrator:
    """
    时间框架整合器
    
    实现多时间框架信号的整合：
    - 时间框架权重分配
    - 信号一致性检查
    - 冲突解决
    - 最终信号生成
    """
    
    def __init__(self, base_timeframe: Timeframe = Timeframe.DAILY):
        """
        初始化时间框架整合器
        
        Args:
            base_timeframe: 基础时间框架
        """
        self.base_timeframe = base_timeframe
        self.timeframe_weights = self._get_default_weights()
        self.signal_history = {}
        
    def _get_default_weights(self) -> Dict[Timeframe, float]:
        """获取默认时间框架权重"""
        return {
            Timeframe.MINUTE_1: 0.05,
            Timeframe.MINUTE_5: 0.08,
            Timeframe.MINUTE_15: 0.12,
            Timeframe.MINUTE_30: 0.15,
            Timeframe.HOUR_1: 0.18,
            Timeframe.HOUR_4: 0.20,
            Timeframe.DAILY: 0.15,
            Timeframe.WEEKLY: 0.05,
            Timeframe.MONTHLY: 0.02
        }
    
    def set_custom_weights(self, weights: Dict[Timeframe, float]) -> None:
        """
        设置自定义时间框架权重
        
        Args:
            weights: 时间框架权重字典
        """
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError("权重和必须为1")
        self.timeframe_weights = weights
    
    def calculate_timeframe_consistency(self, signals: List[TimeframeSignal]) -> float:
        """
        计算时间框架一致性
        
        Args:
            signals: 时间框架信号列表
            
        Returns:
            float: 一致性得分（0到1）
        """
        if not signals:
            return 0.0
        
        # 计算信号方向的一致性
        bullish_signals = [s for s in signals if s.signal_strength > 0]
        bearish_signals = [s for s in signals if s.signal_strength < 0]
        
        if not bullish_signals and not bearish_signals:
            return 0.0
        
        # 计算主导方向
        dominant_direction = (
            "bullish" if len(bullish_signals) > len(bearish_signals) 
            else "bearish" if len(bearish_signals) > len(bullish_signals) 
            else "neutral"
        )
        
        if dominant_direction == "neutral":
            return 0.5
        
        # 计算一致性得分
        total_signals = len(signals)
        consistent_signals = (
            len(bullish_signals) if dominant_direction == "bullish" 
            else len(bearish_signals)
        )
        
        base_consistency = consistent_signals / total_signals
        
        # 考虑置信度加权
        weighted_consistency = sum(
            s.confidence for s in signals 
            if (s.signal_strength > 0 and dominant_direction == "bullish") or
               (s.signal_strength < 0 and dominant_direction == "bearish")
        ) / sum(s.confidence for s in signals)
        
        final_consistency = (base_consistency + weighted_consistency) / 2
        
        return final_consistency
    
    def integrate_signals(self, signals: List[TimeframeSignal]) -> IntegratedSignal:
        """
        整合多时间框架信号
        
        Args:
            signals: 时间框架信号列表
            
        Returns:
            IntegratedSignal: 整合信号
        """
        if not signals:
            raise ValueError("信号列表不能为空")
        
        symbol = signals[0].symbol
        
        # 按时间框架分组信号
        signals_by_timeframe = {}
        for signal in signals:
            if signal.timeframe not in signals_by_timeframe:
                signals_by_timeframe[signal.timeframe] = []
            signals_by_timeframe[signal.timeframe].append(signal)
        
        # 计算各时间框架的加权信号
        timeframe_contributions = {}
        weighted_signals = []
        
        for timeframe, tf_signals in signals_by_timeframe.items():
            if timeframe not in self.timeframe_weights:
                continue
                
            # 计算该时间框架的平均信号（考虑置信度）
            weighted_tf_signal = sum(
                s.signal_strength * s.confidence for s in tf_signals
            ) / sum(s.confidence for s in tf_signals)
            
            # 应用时间框架权重
            final_contribution = weighted_tf_signal * self.timeframe_weights[timeframe]
            timeframe_contributions[timeframe] = final_contribution
            weighted_signals.append(final_contribution)
        
        # 计算最终信号
        final_signal = sum(weighted_signals)
        
        # 计算一致性
        consensus_level = self.calculate_timeframe_consistency(signals)
        
        # 生成交易建议
        recommendation = self._generate_recommendation(final_signal, consensus_level)
        
        # 检查冲突
        conflict_resolved = self._check_conflict_resolution(signals)
        
        return IntegratedSignal(
            symbol=symbol,
            final_signal=final_signal,
            timeframe_contributions=timeframe_contributions,
            consensus_level=consensus_level,
            conflict_resolved=conflict_resolved,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, final_signal: float, consensus_level: float) -> str:
        """
        生成交易建议
        
        Args:
            final_signal: 最终信号强度
            consensus_level: 一致性水平
            
        Returns:
            str: 交易建议
        """
        signal_strength = abs(final_signal)
        
        if signal_strength < 0.1:
            return "观望"
        
        direction = "买入" if final_signal > 0 else "卖出"
        
        if consensus_level > 0.7:
            confidence = "强烈"
        elif consensus_level > 0.5:
            confidence = "中等"
        else:
            confidence = "谨慎"
        
        if signal_strength > 0.3:
            strength = "强势"
        elif signal_strength > 0.15:
            strength = "中等"
        else:
            strength = "弱势"
        
        return f"{confidence}{direction} ({strength}信号)"
    
    def _check_conflict_resolution(self, signals: List[TimeframeSignal]) -> bool:
        """
        检查冲突解决
        
        Args:
            signals: 时间框架信号列表
            
        Returns:
            bool: 是否解决冲突
        """
        if len(signals) < 2:
            return True
        
        # 检查主要时间框架的一致性
        major_timeframes = [
            Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAILY, Timeframe.WEEKLY
        ]
        
        major_signals = [s for s in signals if s.timeframe in major_timeframes]
        
        if not major_signals:
            return True
        
        # 计算主要时间框架的一致性
        major_consistency = self.calculate_timeframe_consistency(major_signals)
        
        return major_consistency > 0.6
    
    def update_signal_history(self, symbol: str, integrated_signal: IntegratedSignal) -> None:
        """
        更新信号历史
        
        Args:
            symbol: 股票代码
            integrated_signal: 整合信号
        """
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append({
            'timestamp': pd.Timestamp.now(),
            'signal': integrated_signal.final_signal,
            'consensus': integrated_signal.consensus_level,
            'recommendation': integrated_signal.recommendation
        })
        
        # 只保留最近100条记录
        if len(self.signal_history[symbol]) > 100:
            self.signal_history[symbol] = self.signal_history[symbol][-100:]
    
    def calculate_signal_persistence(self, symbol: str, window: int = 10) -> Dict[str, float]:
        """
        计算信号持续性
        
        Args:
            symbol: 股票代码
            window: 观察窗口
            
        Returns:
            Dict[str, float]: 持续性指标
        """
        if symbol not in self.signal_history or len(self.signal_history[symbol]) < window:
            return {}
        
        recent_signals = self.signal_history[symbol][-window:]
        signals = [s['signal'] for s in recent_signals]
        
        # 计算信号方向持续性
        bullish_count = sum(1 for s in signals if s > 0)
        bearish_count = sum(1 for s in signals if s < 0)
        neutral_count = sum(1 for s in signals if s == 0)
        
        total_count = len(signals)
        
        # 计算信号强度变化
        signal_changes = [
            abs(signals[i] - signals[i-1]) for i in range(1, len(signals))
        ]
        avg_signal_change = np.mean(signal_changes) if signal_changes else 0
        
        # 计算信号稳定性
        signal_std = np.std(signals)
        
        return {
            'bullish_persistence': bullish_count / total_count,
            'bearish_persistence': bearish_count / total_count,
            'neutral_persistence': neutral_count / total_count,
            'avg_signal_change': avg_signal_change,
            'signal_stability': 1 / (1 + signal_std) if signal_std > 0 else 1.0,
            'direction_consistency': max(bullish_count, bearish_count) / total_count
        }
    
    def optimize_timeframe_weights(self, performance_data: pd.DataFrame) -> Dict[Timeframe, float]:
        """
        优化时间框架权重
        
        Args:
            performance_data: 性能数据，包含各时间框架信号和实际收益
            
        Returns:
            Dict[Timeframe, float]: 优化后的权重
        """
        # 简化实现：基于历史性能调整权重
        timeframe_performance = {}
        
        for timeframe in Timeframe:
            if timeframe.value not in performance_data.columns:
                continue
            
            # 计算该时间框架信号的性能
            signals = performance_data[timeframe.value]
            returns = performance_data['returns']
            
            # 计算信号与收益的相关性（绝对值）
            correlation = abs(signals.corr(returns))
            timeframe_performance[timeframe] = correlation
        
        # 归一化性能得分
        total_performance = sum(timeframe_performance.values())
        if total_performance > 0:
            optimized_weights = {
                tf: perf / total_performance 
                for tf, perf in timeframe_performance.items()
            }
        else:
            optimized_weights = self.timeframe_weights
        
        return optimized_weights
    
    def generate_timeframe_analysis_report(self, signals: List[TimeframeSignal]) -> Dict[str, any]:
        """
        生成时间框架分析报告
        
        Args:
            signals: 时间框架信号列表
            
        Returns:
            Dict[str, any]: 分析报告
        """
        integrated_signal = self.integrate_signals(signals)
        
        # 按时间框架分析
        timeframe_analysis = {}
        for signal in signals:
            tf = signal.timeframe
            if tf not in timeframe_analysis:
                timeframe_analysis[tf] = {
                    'signal_count': 0,
                    'avg_signal_strength': 0,
                    'avg_confidence': 0,
                    'signal_direction': []
                }
            
            analysis = timeframe_analysis[tf]
            analysis['signal_count'] += 1
            analysis['avg_signal_strength'] += signal.signal_strength
            analysis['avg_confidence'] += signal.confidence
            analysis['signal_direction'].append(
                "bullish" if signal.signal_strength > 0 else 
                "bearish" if signal.signal_strength < 0 else "neutral"
            )
        
        # 计算平均值
        for tf_analysis in timeframe_analysis.values():
            if tf_analysis['signal_count'] > 0:
                tf_analysis['avg_signal_strength'] /= tf_analysis['signal_count']
                tf_analysis['avg_confidence'] /= tf_analysis['signal_count']
        
        return {
            'integrated_signal': integrated_signal,
            'timeframe_analysis': timeframe_analysis,
            'total_signals': len(signals),
            'timeframes_covered': len(timeframe_analysis),
            'consensus_level': integrated_signal.consensus_level,
            'conflict_status': "已解决" if integrated_signal.conflict_resolved else "存在冲突"
        }

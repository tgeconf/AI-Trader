"""
止损管理模块
实现多种止损策略：固定百分比、移动平均、波动率调整、时间止损等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class StopLossType(Enum):
    """止损类型枚举"""
    FIXED_PERCENTAGE = "fixed_percentage"
    TRAILING_PERCENTAGE = "trailing_percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MOVING_AVERAGE = "moving_average"
    ATR_BASED = "atr_based"
    TIME_BASED = "time_based"
    CHANDELIER_EXIT = "chandelier_exit"


@dataclass
class StopLossResult:
    """止损计算结果"""
    stop_loss_price: float
    stop_loss_type: StopLossType
    current_price: float
    stop_loss_pct: float
    trigger_distance: float
    is_triggered: bool


class StopLossManager:
    """
    止损管理类
    
    实现多种止损策略：
    - 固定百分比止损
    - 移动止损
    - 波动率调整止损
    - 移动平均止损
    - ATR止损
    - 时间止损
    - 吊灯止损
    """
    
    def __init__(self, default_stop_loss_pct: float = 0.08):
        """
        初始化止损管理器
        
        Args:
            default_stop_loss_pct: 默认止损百分比
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.position_records = {}  # 持仓记录
        
    def fixed_percentage_stop_loss(self, entry_price: float, current_price: float,
                                 stop_loss_pct: Optional[float] = None) -> StopLossResult:
        """
        固定百分比止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            stop_loss_pct: 止损百分比
            
        Returns:
            StopLossResult: 止损结果
        """
        if stop_loss_pct is None:
            stop_loss_pct = self.default_stop_loss_pct
            
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        is_triggered = current_price <= stop_loss_price
        trigger_distance = (current_price - stop_loss_price) / entry_price
        
        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_type=StopLossType.FIXED_PERCENTAGE,
            current_price=current_price,
            stop_loss_pct=stop_loss_pct,
            trigger_distance=trigger_distance,
            is_triggered=is_triggered
        )
    
    def trailing_percentage_stop_loss(self, entry_price: float, current_price: float,
                                    highest_price: float, stop_loss_pct: Optional[float] = None) -> StopLossResult:
        """
        移动百分比止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            highest_price: 最高价格
            stop_loss_pct: 止损百分比
            
        Returns:
            StopLossResult: 止损结果
        """
        if stop_loss_pct is None:
            stop_loss_pct = self.default_stop_loss_pct
            
        # 移动止损价格 = 最高价格 * (1 - 止损百分比)
        stop_loss_price = highest_price * (1 - stop_loss_pct)
        
        # 确保止损价格不低于入场止损
        entry_stop_loss = entry_price * (1 - stop_loss_pct)
        stop_loss_price = max(stop_loss_price, entry_stop_loss)
        
        is_triggered = current_price <= stop_loss_price
        trigger_distance = (current_price - stop_loss_price) / current_price
        
        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_type=StopLossType.TRAILING_PERCENTAGE,
            current_price=current_price,
            stop_loss_pct=stop_loss_pct,
            trigger_distance=trigger_distance,
            is_triggered=is_triggered
        )
    
    def volatility_adjusted_stop_loss(self, entry_price: float, current_price: float,
                                    volatility: float, multiplier: float = 2.0) -> StopLossResult:
        """
        波动率调整止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            volatility: 波动率
            multiplier: 波动率乘数
            
        Returns:
            StopLossResult: 止损结果
        """
        # 基于波动率计算止损距离
        stop_loss_distance = volatility * multiplier
        stop_loss_price = entry_price * (1 - stop_loss_distance)
        
        is_triggered = current_price <= stop_loss_price
        trigger_distance = (current_price - stop_loss_price) / entry_price
        
        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_type=StopLossType.VOLATILITY_ADJUSTED,
            current_price=current_price,
            stop_loss_pct=stop_loss_distance,
            trigger_distance=trigger_distance,
            is_triggered=is_triggered
        )
    
    def moving_average_stop_loss(self, current_price: float, moving_average: float,
                               deviation_pct: float = 0.05) -> StopLossResult:
        """
        移动平均止损
        
        Args:
            current_price: 当前价格
            moving_average: 移动平均值
            deviation_pct: 偏离百分比
            
        Returns:
            StopLossResult: 止损结果
        """
        stop_loss_price = moving_average * (1 - deviation_pct)
        is_triggered = current_price <= stop_loss_price
        trigger_distance = (current_price - stop_loss_price) / current_price
        
        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_type=StopLossType.MOVING_AVERAGE,
            current_price=current_price,
            stop_loss_pct=deviation_pct,
            trigger_distance=trigger_distance,
            is_triggered=is_triggered
        )
    
    def atr_based_stop_loss(self, entry_price: float, current_price: float,
                           atr: float, atr_multiplier: float = 2.0) -> StopLossResult:
        """
        ATR（平均真实波幅）止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            atr: 平均真实波幅
            atr_multiplier: ATR乘数
            
        Returns:
            StopLossResult: 止损结果
        """
        stop_loss_price = entry_price - (atr * atr_multiplier)
        is_triggered = current_price <= stop_loss_price
        trigger_distance = (current_price - stop_loss_price) / entry_price
        
        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_type=StopLossType.ATR_BASED,
            current_price=current_price,
            stop_loss_pct=(atr * atr_multiplier) / entry_price,
            trigger_distance=trigger_distance,
            is_triggered=is_triggered
        )
    
    def time_based_stop_loss(self, entry_price: float, current_price: float,
                           entry_time: datetime, max_holding_days: int = 30,
                           time_decay_pct: float = 0.01) -> StopLossResult:
        """
        时间止损
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            entry_time: 入场时间
            max_holding_days: 最大持有天数
            time_decay_pct: 时间衰减百分比
            
        Returns:
            StopLossResult: 止损结果
        """
        current_time = datetime.now()
        holding_days = (current_time - entry_time).days
        
        # 时间衰减止损
        time_decay = min(holding_days / max_holding_days, 1.0) * time_decay_pct
        stop_loss_pct = self.default_stop_loss_pct + time_decay
        
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        is_triggered = current_price <= stop_loss_price or holding_days >= max_holding_days
        trigger_distance = (current_price - stop_loss_price) / entry_price
        
        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_type=StopLossType.TIME_BASED,
            current_price=current_price,
            stop_loss_pct=stop_loss_pct,
            trigger_distance=trigger_distance,
            is_triggered=is_triggered
        )
    
    def chandelier_exit(self, highest_price: float, current_price: float,
                       atr: float, atr_multiplier: float = 3.0) -> StopLossResult:
        """
        吊灯止损
        
        Args:
            highest_price: 最高价格
            current_price: 当前价格
            atr: 平均真实波幅
            atr_multiplier: ATR乘数
            
        Returns:
            StopLossResult: 止损结果
        """
        stop_loss_price = highest_price - (atr * atr_multiplier)
        is_triggered = current_price <= stop_loss_price
        trigger_distance = (current_price - stop_loss_price) / current_price
        
        return StopLossResult(
            stop_loss_price=stop_loss_price,
            stop_loss_type=StopLossType.CHANDELIER_EXIT,
            current_price=current_price,
            stop_loss_pct=(atr * atr_multiplier) / highest_price,
            trigger_distance=trigger_distance,
            is_triggered=is_triggered
        )
    
    def calculate_comprehensive_stop_loss(self, position_data: Dict) -> Dict[str, StopLossResult]:
        """
        综合计算多种止损策略
        
        Args:
            position_data: 持仓数据
            
        Returns:
            Dict[str, StopLossResult]: 各种止损策略的结果
        """
        results = {}
        
        entry_price = position_data.get('entry_price')
        current_price = position_data.get('current_price')
        highest_price = position_data.get('highest_price', current_price)
        volatility = position_data.get('volatility', 0.2)
        moving_average = position_data.get('moving_average')
        atr = position_data.get('atr')
        entry_time = position_data.get('entry_time', datetime.now())
        
        # 固定百分比止损
        results['fixed_percentage'] = self.fixed_percentage_stop_loss(
            entry_price, current_price
        )
        
        # 移动止损
        results['trailing_percentage'] = self.trailing_percentage_stop_loss(
            entry_price, current_price, highest_price
        )
        
        # 波动率调整止损
        results['volatility_adjusted'] = self.volatility_adjusted_stop_loss(
            entry_price, current_price, volatility
        )
        
        # 移动平均止损
        if moving_average is not None:
            results['moving_average'] = self.moving_average_stop_loss(
                current_price, moving_average
            )
        
        # ATR止损
        if atr is not None:
            results['atr_based'] = self.atr_based_stop_loss(
                entry_price, current_price, atr
            )
            
            # 吊灯止损
            results['chandelier_exit'] = self.chandelier_exit(
                highest_price, current_price, atr
            )
        
        # 时间止损
        results['time_based'] = self.time_based_stop_loss(
            entry_price, current_price, entry_time
        )
        
        return results
    
    def get_optimal_stop_loss(self, position_data: Dict) -> Tuple[StopLossResult, str]:
        """
        获取最优止损策略
        
        Args:
            position_data: 持仓数据
            
        Returns:
            Tuple[StopLossResult, str]: (最优止损结果, 策略说明)
        """
        all_results = self.calculate_comprehensive_stop_loss(position_data)
        
        # 根据市场状态选择最优策略
        volatility = position_data.get('volatility', 0.2)
        trend_strength = position_data.get('trend_strength', 0.5)
        
        if volatility > 0.3:
            # 高波动率市场，使用ATR或波动率调整止损
            if 'atr_based' in all_results:
                return all_results['atr_based'], "高波动率市场，使用ATR止损"
            else:
                return all_results['volatility_adjusted'], "高波动率市场，使用波动率调整止损"
        elif trend_strength > 0.7:
            # 强趋势市场，使用移动止损
            return all_results['trailing_percentage'], "强趋势市场，使用移动止损"
        else:
            # 震荡市场，使用固定百分比止损
            return all_results['fixed_percentage'], "震荡市场，使用固定百分比止损"
    
    def update_position_record(self, symbol: str, position_data: Dict) -> None:
        """
        更新持仓记录
        
        Args:
            symbol: 股票代码
            position_data: 持仓数据
        """
        if symbol not in self.position_records:
            self.position_records[symbol] = []
            
        self.position_records[symbol].append({
            'timestamp': datetime.now(),
            'data': position_data,
            'stop_loss_result': self.get_optimal_stop_loss(position_data)[0]
        })
        
        # 只保留最近100条记录
        if len(self.position_records[symbol]) > 100:
            self.position_records[symbol] = self.position_records[symbol][-100:]
    
    def analyze_stop_loss_performance(self, symbol: str) -> Dict[str, float]:
        """
        分析止损策略表现
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict[str, float]: 止损表现指标
        """
        if symbol not in self.position_records:
            return {}
            
        records = self.position_records[symbol]
        
        if len(records) < 5:
            return {}
        
        triggered_count = sum(1 for r in records if r['stop_loss_result'].is_triggered)
        total_count = len(records)
        trigger_rate = triggered_count / total_count if total_count > 0 else 0
        
        # 计算平均止损距离
        avg_trigger_distance = np.mean([
            r['stop_loss_result'].trigger_distance 
            for r in records if r['stop_loss_result'].is_triggered
        ]) if triggered_count > 0 else 0
        
        # 计算避免的损失
        avoided_losses = []
        for r in records:
            if r['stop_loss_result'].is_triggered:
                entry_price = r['data']['entry_price']
                stop_price = r['stop_loss_result'].stop_loss_price
                min_price = r['data'].get('min_price_after_trigger', stop_price * 0.9)
                avoided_loss = (stop_price - min_price) / entry_price
                avoided_losses.append(avoided_loss)
        
        avg_avoided_loss = np.mean(avoided_losses) if avoided_losses else 0
        
        return {
            'total_positions': total_count,
            'triggered_positions': triggered_count,
            'trigger_rate': trigger_rate,
            'avg_trigger_distance': avg_trigger_distance,
            'avg_avoided_loss': avg_avoided_loss,
            'effectiveness_score': avg_avoided_loss - avg_trigger_distance
        }

"""
隐马尔可夫模型市场状态识别
使用HMM识别市场状态（牛市、熊市、震荡市）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from hmmlearn import hmm


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "bull"      # 牛市
    BEAR = "bear"      # 熊市
    SIDEWAYS = "sideways"  # 震荡市
    HIGH_VOL = "high_vol"  # 高波动
    LOW_VOL = "low_vol"    # 低波动


@dataclass
class RegimeDetectionResult:
    """市场状态检测结果"""
    current_regime: MarketRegime
    regime_probabilities: Dict[MarketRegime, float]
    regime_duration: int  # 当前状态持续时间
    transition_probabilities: np.ndarray
    model_converged: bool
    log_likelihood: float


class HMMMarketRegime:
    """
    隐马尔可夫模型市场状态识别器
    
    使用高斯HMM模型识别市场状态：
    - 牛市：高收益、低波动
    - 熊市：负收益、高波动  
    - 震荡市：低收益、中等波动
    - 高波动：高波动、收益不确定
    - 低波动：低波动、收益不确定
    """
    
    def __init__(self, n_regimes: int = 3, n_features: int = 2):
        """
        初始化HMM市场状态识别器
        
        Args:
            n_regimes: 市场状态数量
            n_features: 特征数量（收益率、波动率）
        """
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.is_fitted = False
        
    def prepare_features(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """
        准备特征数据
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
            
        Returns:
            特征矩阵
        """
        # 计算滚动波动率
        volatility = pd.Series(returns).rolling(window=window).std().fillna(method='bfill').values
        
        # 构建特征矩阵
        features = np.column_stack([returns, volatility])
        
        return features
    
    def fit(self, returns: np.ndarray, window: int = 20) -> None:
        """
        拟合HMM模型
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
        """
        # 准备特征
        features = self.prepare_features(returns, window)
        
        # 移除NaN值
        features = features[~np.isnan(features).any(axis=1)]
        
        if len(features) < self.n_regimes:
            raise ValueError("Insufficient data for HMM fitting")
        
        # 拟合模型
        self.model.fit(features)
        self.is_fitted = True
        
        # 分析各状态特征
        self._analyze_regimes(features)
    
    def _analyze_regimes(self, features: np.ndarray) -> None:
        """分析各市场状态的特征"""
        # 预测状态序列
        states = self.model.predict(features)
        
        # 分析每个状态的平均特征
        self.regime_characteristics = {}
        for regime in range(self.n_regimes):
            regime_mask = states == regime
            if np.sum(regime_mask) > 0:
                regime_features = features[regime_mask]
                self.regime_characteristics[regime] = {
                    'mean_return': np.mean(regime_features[:, 0]),
                    'mean_volatility': np.mean(regime_features[:, 1]),
                    'frequency': np.sum(regime_mask) / len(states),
                    'duration': self._calculate_regime_duration(states, regime)
                }
    
    def _calculate_regime_duration(self, states: np.ndarray, regime: int) -> float:
        """计算状态平均持续时间"""
        durations = []
        current_duration = 0
        
        for state in states:
            if state == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def detect_regime(self, recent_returns: np.ndarray) -> RegimeDetectionResult:
        """
        检测当前市场状态
        
        Args:
            recent_returns: 近期收益率序列
            
        Returns:
            市场状态检测结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detection")
        
        # 准备特征
        features = self.prepare_features(recent_returns)
        
        # 预测最近状态
        recent_features = features[-30:]  # 使用最近30个观测值
        recent_features = recent_features[~np.isnan(recent_features).any(axis=1)]
        
        if len(recent_features) == 0:
            raise ValueError("No valid recent data for regime detection")
        
        # 预测状态
        states = self.model.predict(recent_features)
        current_state = states[-1]
        
        # 计算状态概率
        state_probs = self.model.predict_proba(recent_features)
        current_probs = state_probs[-1]
        
        # 计算当前状态持续时间
        current_duration = self._calculate_current_duration(states)
        
        # 映射状态到市场状态
        mapped_regime = self._map_state_to_regime(current_state)
        
        # 构建概率字典
        regime_probs = {}
        for i, prob in enumerate(current_probs):
            regime = self._map_state_to_regime(i)
            regime_probs[regime] = prob
        
        return RegimeDetectionResult(
            current_regime=mapped_regime,
            regime_probabilities=regime_probs,
            regime_duration=current_duration,
            transition_probabilities=self.model.transmat_,
            model_converged=self.model.monitor_.converged,
            log_likelihood=self.model.score(recent_features)
        )
    
    def _map_state_to_regime(self, state: int) -> MarketRegime:
        """将HMM状态映射到市场状态"""
        if not hasattr(self, 'regime_characteristics'):
            return MarketRegime.SIDEWAYS
        
        if state not in self.regime_characteristics:
            return MarketRegime.SIDEWAYS
        
        chars = self.regime_characteristics[state]
        mean_return = chars['mean_return']
        mean_vol = chars['mean_volatility']
        
        # 基于均值和波动率分类
        if mean_return > 0.001 and mean_vol < 0.02:
            return MarketRegime.BULL
        elif mean_return < -0.001 and mean_vol > 0.015:
            return MarketRegime.BEAR
        elif mean_vol > 0.025:
            return MarketRegime.HIGH_VOL
        elif mean_vol < 0.01:
            return MarketRegime.LOW_VOL
        else:
            return MarketRegime.SIDEWAYS
    
    def _calculate_current_duration(self, states: np.ndarray) -> int:
        """计算当前状态持续时间"""
        if len(states) == 0:
            return 0
        
        current_state = states[-1]
        duration = 1
        
        for i in range(len(states)-2, -1, -1):
            if states[i] == current_state:
                duration += 1
            else:
                break
        
        return duration
    
    def get_regime_statistics(self) -> Dict:
        """获取市场状态统计信息"""
        if not hasattr(self, 'regime_characteristics'):
            return {}
        
        stats = {}
        for state, chars in self.regime_characteristics.items():
            regime = self._map_state_to_regime(state)
            stats[regime.value] = {
                'state_id': state,
                'mean_return': chars['mean_return'],
                'mean_volatility': chars['mean_volatility'],
                'frequency': chars['frequency'],
                'avg_duration': chars['duration']
            }
        
        return stats
    
    def predict_regime_transition(self, current_regime: MarketRegime, 
                                 lookback_periods: int = 10) -> Dict[MarketRegime, float]:
        """
        预测市场状态转换概率
        
        Args:
            current_regime: 当前市场状态
            lookback_periods: 回看期数
            
        Returns:
            各状态转换概率
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 找到当前状态对应的HMM状态
        current_state = None
        for state, chars in self.regime_characteristics.items():
            if self._map_state_to_regime(state) == current_regime:
                current_state = state
                break
        
        if current_state is None:
            return {}
        
        # 获取转换概率
        transition_probs = self.model.transmat_[current_state]
        
        # 映射到市场状态
        regime_transitions = {}
        for state, prob in enumerate(transition_probs):
            regime = self._map_state_to_regime(state)
            if regime in regime_transitions:
                regime_transitions[regime] += prob
            else:
                regime_transitions[regime] = prob
        
        return regime_transitions

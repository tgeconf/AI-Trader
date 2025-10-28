  """
统计异常检测器
实现基于统计方法的异常检测，包括Z-score、IQR、孤立森林等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


class AnomalyType(Enum):
    """异常类型枚举"""
    PRICE_SPIKE = "price_spike"        # 价格异常波动
    VOLUME_SURGE = "volume_surge"      # 成交量异常
    VOLATILITY_SHIFT = "volatility_shift"  # 波动率异常
    CORRELATION_BREAK = "correlation_break"  # 相关性异常
    LIQUIDITY_DROP = "liquidity_drop"  # 流动性异常


@dataclass
class AnomalyDetectionResult:
    """异常检测结果"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: AnomalyType
    confidence: float
    features: Dict[str, float]
    timestamp: pd.Timestamp
    explanation: str


class StatisticalAnomalyDetector:
    """
    统计异常检测器
    
    实现多种统计异常检测方法：
    - Z-score方法
    - IQR方法
    - 孤立森林
    - 局部异常因子
    - 椭圆包络
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        初始化异常检测器
        
        Args:
            contamination: 异常值污染比例
        """
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.local_outlier_factor = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
        self.elliptic_envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> None:
        """
        拟合异常检测模型
        
        Args:
            data: 训练数据
        """
        if len(data) < 10:
            raise ValueError("Insufficient data for anomaly detection")
        
        # 训练各种异常检测模型
        self.isolation_forest.fit(data)
        self.local_outlier_factor.fit(data)
        self.elliptic_envelope.fit(data)
        
        # 计算统计特征
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        self.data_median = np.median(data, axis=0)
        self.q1 = np.percentile(data, 25, axis=0)
        self.q3 = np.percentile(data, 75, axis=0)
        self.iqr = self.q3 - self.q1
        
        self.is_fitted = True
        
    def detect_price_anomaly(self, price_data: np.ndarray, 
                           volume_data: Optional[np.ndarray] = None) -> AnomalyDetectionResult:
        """
        检测价格异常
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            
        Returns:
            价格异常检测结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detection")
        
        # 计算价格特征
        price_returns = np.diff(price_data) / price_data[:-1]
        price_volatility = np.std(price_returns) if len(price_returns) > 1 else 0
        
        features = {
            'current_price': price_data[-1],
            'price_change': price_data[-1] - price_data[-2] if len(price_data) > 1 else 0,
            'price_return': price_returns[-1] if len(price_returns) > 0 else 0,
            'volatility': price_volatility
        }
        
        # 添加成交量特征
        if volume_data is not None and len(volume_data) > 1:
            volume_change = volume_data[-1] - volume_data[-2]
            volume_ratio = volume_data[-1] / np.mean(volume_data[:-1]) if np.mean(volume_data[:-1]) > 0 else 1
            features.update({
                'volume': volume_data[-1],
                'volume_change': volume_change,
                'volume_ratio': volume_ratio
            })
        
        # 使用多种方法检测异常
        anomaly_scores = []
        
        # Z-score方法
        z_score = abs((price_data[-1] - self.data_mean[0]) / self.data_std[0]) if self.data_std[0] > 0 else 0
        z_anomaly = z_score > 3
        anomaly_scores.append(z_anomaly)
        
        # IQR方法
        iqr_lower = self.q1[0] - 1.5 * self.iqr[0]
        iqr_upper = self.q3[0] + 1.5 * self.iqr[0]
        iqr_anomaly = price_data[-1] < iqr_lower or price_data[-1] > iqr_upper
        anomaly_scores.append(iqr_anomaly)
        
        # 孤立森林
        if_anomaly = self.isolation_forest.predict([[price_data[-1]]])[0] == -1
        anomaly_scores.append(if_anomaly)
        
        # 综合判断
        is_anomaly = sum(anomaly_scores) >= 2  # 至少两种方法检测到异常
        anomaly_score = sum(anomaly_scores) / len(anomaly_scores)
        
        # 确定异常类型
        anomaly_type = self._classify_anomaly_type(features)
        
        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            confidence=anomaly_score,
            features=features,
            timestamp=pd.Timestamp.now(),
            explanation=self._generate_explanation(anomaly_type, features)
        )
    
    def detect_volatility_anomaly(self, returns_data: np.ndarray, 
                                window: int = 20) -> AnomalyDetectionResult:
        """
        检测波动率异常
        
        Args:
            returns_data: 收益率数据
            window: 滚动窗口大小
            
        Returns:
            波动率异常检测结果
        """
        if len(returns_data) < window:
            return AnomalyDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type=AnomalyType.VOLATILITY_SHIFT,
                confidence=0.0,
                features={},
                timestamp=pd.Timestamp.now(),
                explanation="Insufficient data for volatility analysis"
            )
        
        # 计算滚动波动率
        rolling_volatility = pd.Series(returns_data).rolling(window=window).std().dropna().values
        
        if len(rolling_volatility) == 0:
            return AnomalyDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type=AnomalyType.VOLATILITY_SHIFT,
                confidence=0.0,
                features={},
                timestamp=pd.Timestamp.now(),
                explanation="No volatility data available"
            )
        
        current_volatility = rolling_volatility[-1]
        historical_volatility = np.mean(rolling_volatility[:-1])
        
        # 计算波动率变化
        volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1
        
        features = {
            'current_volatility': current_volatility,
            'historical_volatility': historical_volatility,
            'volatility_ratio': volatility_ratio,
            'volatility_change': current_volatility - historical_volatility
        }
        
        # 检测异常
        is_anomaly = volatility_ratio > 2.0 or volatility_ratio < 0.5
        anomaly_score = min(abs(volatility_ratio - 1), 1.0)
        
        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type=AnomalyType.VOLATILITY_SHIFT,
            confidence=anomaly_score,
            features=features,
            timestamp=pd.Timestamp.now(),
            explanation=self._generate_volatility_explanation(volatility_ratio)
        )
    
    def detect_correlation_anomaly(self, asset_returns: Dict[str, np.ndarray], 
                                 correlation_threshold: float = 0.7) -> AnomalyDetectionResult:
        """
        检测相关性异常
        
        Args:
            asset_returns: 各资产收益率数据
            correlation_threshold: 相关性阈值
            
        Returns:
            相关性异常检测结果
        """
        if len(asset_returns) < 2:
            return AnomalyDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type=AnomalyType.CORRELATION_BREAK,
                confidence=0.0,
                features={},
                timestamp=pd.Timestamp.now(),
                explanation="Insufficient assets for correlation analysis"
            )
        
        # 计算相关性矩阵
        assets = list(asset_returns.keys())
        correlation_matrix = np.corrcoef([asset_returns[asset] for asset in assets])
        
        # 检测异常相关性变化
        correlation_changes = []
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                current_corr = correlation_matrix[i, j]
                correlation_changes.append(abs(current_corr))
        
        avg_correlation = np.mean(correlation_changes) if correlation_changes else 0
        max_correlation_change = max(correlation_changes) if correlation_changes else 0
        
        features = {
            'average_correlation': avg_correlation,
            'max_correlation': max_correlation_change,
            'correlation_threshold': correlation_threshold
        }
        
        # 检测异常
        is_anomaly = avg_correlation < correlation_threshold * 0.5 or max_correlation_change > correlation_threshold * 1.5
        anomaly_score = min(abs(avg_correlation - correlation_threshold) / correlation_threshold, 1.0)
        
        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomaly_type=AnomalyType.CORRELATION_BREAK,
            confidence=anomaly_score,
            features=features,
            timestamp=pd.Timestamp.now(),
            explanation=self._generate_correlation_explanation(avg_correlation, correlation_threshold)
        )
    
    def _classify_anomaly_type(self, features: Dict[str, float]) -> AnomalyType:
        """分类异常类型"""
        price_change = features.get('price_change', 0)
        volume_ratio = features.get('volume_ratio', 1)
        
        if abs(price_change) > 0.05:  # 价格变化超过5%
            if volume_ratio > 2.0:    # 成交量翻倍
                return AnomalyType.VOLUME_SURGE
            else:
                return AnomalyType.PRICE_SPIKE
        elif volume_ratio > 3.0:      # 成交量三倍以上
            return AnomalyType.VOLUME_SURGE
        else:
            return AnomalyType.PRICE_SPIKE
    
    def _generate_explanation(self, anomaly_type: AnomalyType, features: Dict[str, float]) -> str:
        """生成异常解释"""
        explanations = {
            AnomalyType.PRICE_SPIKE: f"价格异常波动: 变化 {features.get('price_change', 0):.2%}",
            AnomalyType.VOLUME_SURGE: f"成交量异常: 成交量比率 {features.get('volume_ratio', 1):.2f}",
            AnomalyType.VOLATILITY_SHIFT: "波动率异常变化",
            AnomalyType.CORRELATION_BREAK: "资产相关性异常",
            AnomalyType.LIQUIDITY_DROP: "流动性异常下降"
        }
        return explanations.get(anomaly_type, "检测到异常行为")
    
    def _generate_volatility_explanation(self, volatility_ratio: float) -> str:
        """生成波动率异常解释"""
        if volatility_ratio > 2.0:
            return f"波动率异常升高: 当前波动率是历史平均的 {volatility_ratio:.2f} 倍"
        elif volatility_ratio < 0.5:
            return f"波动率异常降低: 当前波动率是历史平均的 {volatility_ratio:.2f} 倍"
        else:
            return "波动率正常"
    
    def _generate_correlation_explanation(self, avg_correlation: float, threshold: float) -> str:
        """生成相关性异常解释"""
        if avg_correlation < threshold * 0.5:
            return f"相关性异常降低: 平均相关性 {avg_correlation:.3f} 低于阈值 {threshold}"
        else:
            return f"相关性正常: 平均相关性 {avg_correlation:.3f}"
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        if not self.is_fitted:
            return {}
        
        return {
            "data_statistics": {
                "mean": self.data_mean.tolist(),
                "std": self.data_std.tolist(),
                "median": self.data_median.tolist(),
                "q1": self.q1.tolist(),
                "q3": self.q3.tolist(),
                "iqr": self.iqr.tolist()
            },
            "model_parameters": {
                "contamination": self.contamination,
                "is_fitted": self.is_fitted
            }
        }

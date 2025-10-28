"""
行为异常检测器
实现交易行为和用户行为异常检测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import norm, chi2, kstest, anderson
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


class BehavioralAnomalyType(Enum):
    """行为异常类型枚举"""

    TRADING_FREQUENCY_ANOMALY = "trading_frequency_anomaly"  # 交易频率异常
    ORDER_SIZE_ANOMALY = "order_size_anomaly"  # 订单规模异常
    TRADING_PATTERN_CHANGE = "trading_pattern_change"  # 交易模式变化
    RISK_APPETITE_CHANGE = "risk_appetite_change"  # 风险偏好变化
    TIME_OF_DAY_ANOMALY = "time_of_day_anomaly"  # 交易时间异常
    ASSET_CONCENTRATION = "asset_concentration"  # 资产集中度异常
    PORTFOLIO_TURNOVER_ANOMALY = "portfolio_turnover_anomaly"  # 组合换手率异常
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # 相关性异常
    LIQUIDITY_RISK = "liquidity_risk"  # 流动性风险
    MARKET_TIMING_ANOMALY = "market_timing_anomaly"  # 市场择时异常


@dataclass
class BehavioralAnalysisResult:
    """行为分析结果"""

    is_anomaly: bool
    anomaly_score: float
    anomaly_type: BehavioralAnomalyType
    confidence: float
    features: Dict[str, float]
    timestamp: pd.Timestamp
    explanation: str
    recommendations: List[str]


@dataclass
class TradingBehaviorProfile:
    """交易行为画像"""

    avg_trades_per_day: float
    avg_order_size: float
    preferred_trading_hours: List[int]
    risk_tolerance: float
    diversification_score: float
    trading_consistency: float
    last_updated: datetime
    # 新增专业字段
    var_95: float  # 95% VaR
    expected_shortfall: float  # 期望损失
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    calmar_ratio: float  # 卡玛比率
    portfolio_beta: float  # 组合贝塔
    tracking_error: float  # 跟踪误差
    information_ratio: float  # 信息比率
    herfindahl_index: float  # 赫芬达尔指数
    gini_coefficient: float  # 基尼系数
    turnover_ratio: float  # 换手率
    concentration_ratio: float  # 集中度比率
    # 时间序列特征
    autocorrelation_lag1: float  # 自相关性
    hurst_exponent: float  # 赫斯特指数
    volatility_clustering: float  # 波动率聚集性
    # 行为特征
    loss_aversion_coefficient: float  # 损失厌恶系数
    overconfidence_bias: float  # 过度自信偏差
    disposition_effect: float  # 处置效应
    # 机器学习特征
    isolation_forest_score: float  # 孤立森林异常分数
    dbscan_cluster: int  # DBSCAN聚类标签
    # 统计检验
    normality_test_pvalue: float  # 正态性检验p值
    stationarity_test_pvalue: float  # 平稳性检验p值
    regime_change_probability: float  # 市场状态变化概率


class BehavioralAnomalyDetector:
    """
    行为异常检测器

    实现交易行为和用户行为异常检测：
    - 交易频率异常检测
    - 订单规模异常检测
    - 交易模式变化检测
    - 风险偏好变化检测
    - 交易时间异常检测
    - 资产集中度检测
    """

    def __init__(
        self,
        baseline_period_days: int = 30,
        anomaly_threshold: float = 0.8,
        max_trading_history: int = 1000,
        confidence_level: float = 0.95,
        lookback_window: int = 252,
        min_trades_for_profile: int = 50,
    ):
        """
        初始化行为异常检测器

        Args:
            baseline_period_days: 基准期天数
            anomaly_threshold: 异常阈值
            max_trading_history: 最大交易历史记录数
            confidence_level: 置信水平
            lookback_window: 回看窗口
            min_trades_for_profile: 建立画像所需最小交易数
        """
        self.baseline_period_days = baseline_period_days
        self.anomaly_threshold = anomaly_threshold
        self.max_trading_history = max_trading_history
        self.confidence_level = confidence_level
        self.lookback_window = lookback_window
        self.min_trades_for_profile = min_trades_for_profile

        # 交易历史存储
        self.trading_history = deque(maxlen=max_trading_history)
        self.user_profiles: Dict[str, TradingBehaviorProfile] = {}

        # 专业级异常检测模型参数
        self.frequency_threshold_multiplier = 3.0
        self.size_threshold_multiplier = 2.5
        self.pattern_change_threshold = 0.7
        self.risk_change_threshold = 0.5
        
        # 统计检验阈值
        self.normality_threshold = 0.05
        self.stationarity_threshold = 0.05
        self.regime_change_threshold = 0.8
        
        # 风险度量参数
        self.var_confidence = 0.95
        self.es_confidence = 0.975
        
        # 机器学习模型
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # 缓存和状态
        self.feature_cache: Dict[str, List[float]] = {}
        self.model_trained = False
        self.last_training_time = None

    def add_trading_record(
        self,
        user_id: str,
        symbol: str,
        order_type: str,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """
        添加交易记录

        Args:
            user_id: 用户ID
            symbol: 交易标的
            order_type: 订单类型 (buy/sell)
            quantity: 数量
            price: 价格
            timestamp: 时间戳
        """
        trading_record = {
            "user_id": user_id,
            "symbol": symbol,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "notional": quantity * price,
        }
        self.trading_history.append(trading_record)

        # 更新用户画像
        self._update_user_profile(user_id)

    def analyze_trading_behavior(
        self, trading_data: Dict[str, Any], user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析交易行为

        Args:
            trading_data: 交易数据
            user_id: 用户ID

        Returns:
            交易行为分析结果
        """
        # 提取交易特征
        features = self._extract_trading_features(trading_data, user_id)

        # 检测各种异常
        anomalies = []

        # 交易频率异常检测
        frequency_anomaly = self._detect_trading_frequency_anomaly(features, user_id)
        if frequency_anomaly:
            anomalies.append(frequency_anomaly)

        # 订单规模异常检测
        size_anomaly = self._detect_order_size_anomaly(features, user_id)
        if size_anomaly:
            anomalies.append(size_anomaly)

        # 交易模式变化检测
        pattern_anomaly = self._detect_trading_pattern_change(features, user_id)
        if pattern_anomaly:
            anomalies.append(pattern_anomaly)

        # 风险偏好变化检测
        risk_anomaly = self._detect_risk_appetite_change(features, user_id)
        if risk_anomaly:
            anomalies.append(risk_anomaly)

        # 交易时间异常检测
        time_anomaly = self._detect_time_of_day_anomaly(features, user_id)
        if time_anomaly:
            anomalies.append(time_anomaly)

        # 资产集中度检测
        concentration_anomaly = self._detect_asset_concentration(features, user_id)
        if concentration_anomaly:
            anomalies.append(concentration_anomaly)

        return {
            "features": features,
            "anomalies": anomalies,
            "overall_risk_score": self._calculate_overall_risk_score(anomalies),
            "user_profile": self.user_profiles.get(user_id) if user_id else None,
        }

    def analyze_user_behavior(self, user_behavior: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析用户行为

        Args:
            user_behavior: 用户行为数据

        Returns:
            用户行为分析结果
        """
        # 提取用户行为特征
        features = self._extract_user_behavior_features(user_behavior)

        # 检测用户行为异常
        anomalies = []

        # 登录频率异常
        login_anomaly = self._detect_login_frequency_anomaly(features)
        if login_anomaly:
            anomalies.append(login_anomaly)

        # 操作模式异常
        operation_anomaly = self._detect_operation_pattern_anomaly(features)
        if operation_anomaly:
            anomalies.append(operation_anomaly)

        # 会话时长异常
        session_anomaly = self._detect_session_duration_anomaly(features)
        if session_anomaly:
            anomalies.append(session_anomaly)

        return {
            "features": features,
            "anomalies": anomalies,
            "behavioral_risk_score": self._calculate_behavioral_risk_score(anomalies),
        }

    def detect_behavioral_anomalies(
        self, trading_analysis: Dict[str, Any], user_analysis: Dict[str, Any]
    ) -> List[BehavioralAnalysisResult]:
        """
        检测行为异常

        Args:
            trading_analysis: 交易行为分析结果
            user_analysis: 用户行为分析结果

        Returns:
            行为异常检测结果列表
        """
        anomalies = []

        # 合并交易异常
        trading_anomalies = trading_analysis.get("anomalies", [])
        anomalies.extend(trading_anomalies)

        # 合并用户行为异常
        user_anomalies = user_analysis.get("anomalies", [])
        anomalies.extend(user_anomalies)

        # 转换为标准格式
        results = []
        for anomaly in anomalies:
            if anomaly.get("is_anomaly", False):
                results.append(
                    BehavioralAnalysisResult(
                        is_anomaly=True,
                        anomaly_score=anomaly.get("score", 0.0),
                        anomaly_type=anomaly.get("type"),
                        confidence=anomaly.get("confidence", 0.0),
                        features=anomaly.get("features", {}),
                        timestamp=pd.Timestamp.now(),
                        explanation=anomaly.get("explanation", ""),
                        recommendations=anomaly.get("recommendations", []),
                    )
                )

        return results

    def assess_behavioral_risk(
        self, behavioral_anomalies: List[BehavioralAnalysisResult]
    ) -> Dict[str, Any]:
        """
        评估行为风险

        Args:
            behavioral_anomalies: 行为异常列表

        Returns:
            行为风险评估结果
        """
        if not behavioral_anomalies:
            return {
                "overall_risk": "low",
                "risk_score": 0.0,
                "anomaly_count": 0,
                "high_risk_anomalies": [],
            }

        # 计算风险评分
        total_score = sum(anomaly.anomaly_score for anomaly in behavioral_anomalies)
        avg_score = total_score / len(behavioral_anomalies)

        # 识别高风险异常
        high_risk_anomalies = [
            anomaly for anomaly in behavioral_anomalies if anomaly.anomaly_score > 0.8
        ]

        # 确定整体风险等级
        if avg_score > 0.8 or len(high_risk_anomalies) >= 3:
            overall_risk = "high"
        elif avg_score > 0.5 or len(high_risk_anomalies) >= 1:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return {
            "overall_risk": overall_risk,
            "risk_score": avg_score,
            "anomaly_count": len(behavioral_anomalies),
            "high_risk_anomalies": [
                {
                    "type": anomaly.anomaly_type.value,
                    "score": anomaly.anomaly_score,
                    "explanation": anomaly.explanation,
                }
                for anomaly in high_risk_anomalies
            ],
        }

    def generate_behavioral_recommendations(
        self, behavioral_anomalies: List[BehavioralAnalysisResult]
    ) -> List[str]:
        """
        生成行为建议

        Args:
            behavioral_anomalies: 行为异常列表

        Returns:
            行为建议列表
        """
        recommendations = []

        for anomaly in behavioral_anomalies:
            if anomaly.anomaly_score > 0.7:
                if (
                    anomaly.anomaly_type
                    == BehavioralAnomalyType.TRADING_FREQUENCY_ANOMALY
                ):
                    recommendations.append(
                        "检测到交易频率异常，建议检查交易策略或暂停交易"
                    )
                elif anomaly.anomaly_type == BehavioralAnomalyType.ORDER_SIZE_ANOMALY:
                    recommendations.append("检测到订单规模异常，建议重新评估头寸规模")
                elif (
                    anomaly.anomaly_type == BehavioralAnomalyType.TRADING_PATTERN_CHANGE
                ):
                    recommendations.append("检测到交易模式变化，建议确认策略变更意图")
                elif anomaly.anomaly_type == BehavioralAnomalyType.RISK_APPETITE_CHANGE:
                    recommendations.append(
                        "检测到风险偏好变化，建议重新评估风险承受能力"
                    )

        if not recommendations:
            recommendations.append("行为模式正常，继续保持")

        return recommendations

    def _update_user_profile(self, user_id: str) -> None:
        """更新用户画像"""
        user_trades = [
            trade for trade in self.trading_history if trade["user_id"] == user_id
        ]

        if len(user_trades) < 10:  # 最少需要10笔交易建立画像
            return

        # 计算交易频率
        trade_dates = [trade["timestamp"] for trade in user_trades]
        date_range = (max(trade_dates) - min(trade_dates)).days
        avg_trades_per_day = len(user_trades) / max(date_range, 1)

        # 计算平均订单规模
        avg_order_size = np.mean([trade["notional"] for trade in user_trades])

        # 分析交易时间偏好
        trading_hours = [trade["timestamp"].hour for trade in user_trades]
        preferred_hours = self._analyze_trading_hours(trading_hours)

        # 计算风险容忍度
        risk_tolerance = self._calculate_risk_tolerance(user_trades)

        # 计算分散化评分
        diversification_score = self._calculate_diversification_score(user_trades)

        # 计算交易一致性
        trading_consistency = self._calculate_trading_consistency(user_trades)

        profile = TradingBehaviorProfile(
            avg_trades_per_day=avg_trades_per_day,
            avg_order_size=avg_order_size,
            preferred_trading_hours=preferred_hours,
            risk_tolerance=risk_tolerance,
            diversification_score=diversification_score,
            trading_consistency=trading_consistency,
            last_updated=datetime.now(),
        )

        self.user_profiles[user_id] = profile

    def _extract_trading_features(
        self, trading_data: Dict[str, Any], user_id: Optional[str]
    ) -> Dict[str, float]:
        """提取交易特征"""
        features = {}

        # 基础交易特征
        features["trade_count"] = trading_data.get("trade_count", 0)
        features["total_volume"] = trading_data.get("total_volume", 0)
        features["avg_trade_size"] = trading_data.get("avg_trade_size", 0)

        # 时间特征
        current_hour = datetime.now().hour
        features["current_hour"] = current_hour

        # 风险特征
        features["max_drawdown"] = trading_data.get("max_drawdown", 0)
        features["volatility"] = trading_data.get("volatility", 0)

        # 如果提供了用户ID，添加用户特定特征
        if user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            features["baseline_trades_per_day"] = profile.avg_trades_per_day
            features["baseline_order_size"] = profile.avg_order_size
            features["preferred_hours_match"] = self._calculate_hour_match(
                current_hour, profile.preferred_trading_hours
            )

        return features

    def _extract_user_behavior_features(
        self, user_behavior: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取用户行为特征"""
        features = {}

        # 登录行为特征
        features["login_frequency"] = user_behavior.get("login_frequency", 0)
        features["session_duration_avg"] = user_behavior.get("session_duration_avg", 0)
        features["operations_per_session"] = user_behavior.get(
            "operations_per_session", 0
        )

        # 操作模式特征
        features["preferred_operations"] = len(
            user_behavior.get("preferred_operations", [])
        )
        features["operation_consistency"] = user_behavior.get(
            "operation_consistency", 0
        )

        return features

    def _detect_trading_frequency_anomaly(
        self, features: Dict[str, float], user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """检测交易频率异常"""
        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]
        current_trades = features.get("trade_count", 0)
        baseline_trades = profile.avg_trades_per_day

        if baseline_trades == 0:
            return None

        frequency_ratio = current_trades / baseline_trades

        if frequency_ratio > self.frequency_threshold_multiplier:
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.TRADING_FREQUENCY_ANOMALY,
                "score": min(
                    frequency_ratio / self.frequency_threshold_multiplier, 1.0
                ),
                "confidence": 0.8,
                "features": {
                    "current_trades": current_trades,
                    "baseline_trades": baseline_trades,
                    "frequency_ratio": frequency_ratio,
                },
                "explanation": f"交易频率异常：当前{current_trades}笔，基准{baseline_trades:.1f}笔",
                "recommendations": ["检查交易策略", "确认交易意图"],
            }

        return None

    def _detect_order_size_anomaly(
        self, features: Dict[str, float], user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """检测订单规模异常"""
        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]
        current_size = features.get("avg_trade_size", 0)
        baseline_size = profile.avg_order_size

        if baseline_size == 0:
            return None

        size_ratio = current_size / baseline_size

        if size_ratio > self.size_threshold_multiplier:
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.ORDER_SIZE_ANOMALY,
                "score": min(size_ratio / self.size_threshold_multiplier, 1.0),
                "confidence": 0.7,
                "features": {
                    "current_size": current_size,
                    "baseline_size": baseline_size,
                    "size_ratio": size_ratio,
                },
                "explanation": f"订单规模异常：当前{current_size:.2f}，基准{baseline_size:.2f}",
                "recommendations": ["重新评估头寸规模", "检查风险控制"],
            }

        return None

    def _detect_trading_pattern_change(
        self, features: Dict[str, float], user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """检测交易模式变化"""
        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]
        hour_match = features.get("preferred_hours_match", 0)

        if hour_match < self.pattern_change_threshold:
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.TRADING_PATTERN_CHANGE,
                "score": 1.0 - hour_match,
                "confidence": 0.6,
                "features": {
                    "hour_match_score": hour_match,
                    "preferred_hours": profile.preferred_trading_hours,
                },
                "explanation": f"交易时间模式变化：匹配度{hour_match:.2f}",
                "recommendations": ["确认策略变更意图", "检查账户安全"],
            }

        return None

    def _detect_risk_appetite_change(
        self, features: Dict[str, float], user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        检测风险偏好变化 - 专业级实现
        
        使用多种统计检验和贝叶斯方法：
        1. 分布变化检测 (Kolmogorov-Smirnov检验)
        2. 均值变化检测 (t检验)
        3. 方差变化检测 (F检验)
        4. 贝叶斯变化点检测
        5. 机器学习异常检测
        
        Args:
            features: 交易特征
            user_id: 用户ID
            
        Returns:
            风险偏好变化异常检测结果
        """
        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]
        current_volatility = features.get("volatility", 0)

        # 获取历史风险数据
        historical_risk_data = self._get_user_historical_risk_data(user_id)
        
        if len(historical_risk_data) < 20:
            return None

        # 1. 统计检验
        statistical_tests = self._perform_statistical_tests(
            historical_risk_data, current_volatility
        )
        
        # 2. 贝叶斯变化点检测
        bayesian_change_prob = self._bayesian_change_point_detection(
            historical_risk_data, current_volatility
        )
        
        # 3. 机器学习异常检测
        ml_anomaly_score = self._ml_risk_anomaly_detection(
            historical_risk_data, current_volatility
        )
        
        # 综合风险变化评分
        risk_change_score = self._calculate_comprehensive_risk_change_score(
            statistical_tests, bayesian_change_prob, ml_anomaly_score
        )
        
        if risk_change_score > self.risk_change_threshold:
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.RISK_APPETITE_CHANGE,
                "score": risk_change_score,
                "confidence": self._calculate_risk_change_confidence(statistical_tests),
                "features": {
                    "current_volatility": current_volatility,
                    "baseline_risk": profile.risk_tolerance,
                    "risk_change_score": risk_change_score,
                    "statistical_tests": statistical_tests,
                    "bayesian_probability": bayesian_change_prob,
                    "ml_anomaly_score": ml_anomaly_score,
                },
                "explanation": self._generate_risk_change_explanation(
                    current_volatility, profile.risk_tolerance, statistical_tests
                ),
                "recommendations": [
                    "重新评估风险承受能力",
                    "调整交易策略",
                    "进行压力测试",
                    "咨询风险管理专家"
                ],
            }

        return None

    def _get_user_historical_risk_data(self, user_id: str) -> List[float]:
        """获取用户历史风险数据"""
        user_trades = [
            trade for trade in self.trading_history if trade["user_id"] == user_id
        ]
        
        if len(user_trades) < 10:
            return []
        
        # 提取价格序列计算历史波动率
        prices = [trade["price"] for trade in user_trades]
        returns = self._calculate_returns(prices, [trade["timestamp"] for trade in user_trades])
        
        if len(returns) < 10:
            return []
        
        # 计算滚动波动率作为历史风险数据
        rolling_window = min(20, len(returns) // 2)
        historical_volatility = []
        
        for i in range(rolling_window, len(returns)):
            window_returns = returns[i-rolling_window:i]
            vol = np.std(window_returns) * np.sqrt(252)
            historical_volatility.append(vol)
        
        return historical_volatility

    def _perform_statistical_tests(
        self, historical_data: List[float], current_value: float
    ) -> Dict[str, float]:
        """执行统计检验"""
        if len(historical_data) < 10:
            return {}
        
        historical_array = np.array(historical_data)
        
        # 1. Kolmogorov-Smirnov检验 (分布变化)
        try:
            ks_stat, ks_pvalue = kstest(historical_array, 'norm')
        except:
            ks_pvalue = 1.0
        
        # 2. t检验 (均值变化)
        try:
            t_stat, t_pvalue = stats.ttest_1samp(historical_array, current_value)
        except:
            t_pvalue = 1.0
        
        # 3. F检验 (方差变化)
        try:
            recent_data = historical_array[-10:] if len(historical_array) >= 10 else historical_array
            older_data = historical_array[:-10] if len(historical_array) >= 20 else historical_array
            
            if len(recent_data) > 1 and len(older_data) > 1:
                f_stat, f_pvalue = stats.levene(recent_data, older_data)
            else:
                f_pvalue = 1.0
        except:
            f_pvalue = 1.0
        
        # 4. Anderson-Darling检验 (正态性)
        try:
            anderson_stat, anderson_critical, anderson_significance = anderson(historical_array)
            ad_pvalue = anderson_significance[2]  # 使用5%显著性水平
        except:
            ad_pvalue = 1.0
        
        return {
            "ks_pvalue": ks_pvalue,
            "t_pvalue": t_pvalue,
            "f_pvalue": f_pvalue,
            "ad_pvalue": ad_pvalue,
        }

    def _bayesian_change_point_detection(
        self, historical_data: List[float], current_value: float
    ) -> float:
        """贝叶斯变化点检测"""
        if len(historical_data) < 20:
            return 0.0
        
        # 简化的贝叶斯变化点检测
        # 使用滑动窗口计算后验概率
        window_size = min(10, len(historical_data) // 2)
        recent_mean = np.mean(historical_data[-window_size:])
        historical_mean = np.mean(historical_data[:-window_size])
        
        # 计算似然比
        recent_std = np.std(historical_data[-window_size:])
        historical_std = np.std(historical_data[:-window_size])
        
        if historical_std > 0:
            # 使用t分布计算变化概率
            t_stat = abs(recent_mean - historical_mean) / (
                np.sqrt((recent_std**2 / window_size) + (historical_std**2 / window_size))
            )
            
            # 计算贝叶斯后验概率
            degrees_freedom = 2 * window_size - 2
            change_prob = 1 - stats.t.cdf(t_stat, degrees_freedom)
            return min(change_prob, 1.0)
        
        return 0.0

    def _ml_risk_anomaly_detection(
        self, historical_data: List[float], current_value: float
    ) -> float:
        """机器学习风险异常检测"""
        if len(historical_data) < 20:
            return 0.0
        
        # 准备特征数据
        features = []
        for i in range(len(historical_data)):
            window_data = historical_data[max(0, i-9):i+1]
            if len(window_data) >= 5:
                feature_vector = [
                    np.mean(window_data),
                    np.std(window_data),
                    np.percentile(window_data, 25),
                    np.percentile(window_data, 75),
                    stats.skew(window_data) if len(window_data) > 2 else 0,
                ]
                features.append(feature_vector)
        
        if len(features) < 10:
            return 0.0
        
        # 使用孤立森林检测异常
        try:
            self.isolation_forest.fit(features)
            current_feature = [
                np.mean(historical_data[-5:]),
                np.std(historical_data[-5:]),
                np.percentile(historical_data[-5:], 25),
                np.percentile(historical_data[-5:], 75),
                stats.skew(historical_data[-5:]) if len(historical_data[-5:]) > 2 else 0,
            ]
            anomaly_score = -self.isolation_forest.score_samples([current_feature])[0]
            return min(max(anomaly_score, 0.0), 1.0)
        except:
            return 0.0

    def _calculate_comprehensive_risk_change_score(
        self, 
        statistical_tests: Dict[str, float], 
        bayesian_prob: float, 
        ml_score: float
    ) -> float:
        """计算综合风险变化评分"""
        # 统计检验权重
        stat_weight = 0.4
        bayesian_weight = 0.3
        ml_weight = 0.3
        
        # 统计检验评分 (p值越小，变化可能性越大)
        stat_scores = []
        for pvalue in statistical_tests.values():
            if pvalue < 0.05:
                stat_scores.append(1.0 - (pvalue / 0.05))
            else:
                stat_scores.append(0.0)
        
        stat_score = np.mean(stat_scores) if stat_scores else 0.0
        
        # 综合评分
        comprehensive_score = (
            stat_weight * stat_score +
            bayesian_weight * bayesian_prob +
            ml_weight * ml_score
        )
        
        return min(max(comprehensive_score, 0.0), 1.0)

    def _calculate_risk_change_confidence(self, statistical_tests: Dict[str, float]) -> float:
        """计算风险变化置信度"""
        significant_tests = sum(1 for pvalue in statistical_tests.values() if pvalue < 0.05)
        total_tests = len(statistical_tests)
        
        if total_tests == 0:
            return 0.0
        
        confidence = significant_tests / total_tests
        return min(max(confidence, 0.0), 1.0)

    def _generate_risk_change_explanation(
        self, 
        current_volatility: float, 
        baseline_risk: float, 
        statistical_tests: Dict[str, float]
    ) -> str:
        """生成风险变化解释"""
        risk_change_pct = abs(current_volatility - baseline_risk) / max(baseline_risk, 0.01) * 100
        
        explanation = f"风险偏好显著变化：当前波动率{current_volatility:.4f}，基准{baseline_risk:.4f}，变化幅度{risk_change_pct:.1f}%"
        
        # 添加统计检验结果
        significant_tests = [
            test_name for test_name, pvalue in statistical_tests.items() 
            if pvalue < 0.05
        ]
        
        if significant_tests:
            explanation += f"。统计检验显示{', '.join(significant_tests)}显著(p<0.05)"
        
        return explanation

    def _detect_time_of_day_anomaly(
        self, features: Dict[str, float], user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """检测交易时间异常"""
        current_hour = features.get("current_hour", 0)

        # 检测非交易时间活动（假设交易时间为9-16点）
        if current_hour < 9 or current_hour > 16:
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.TIME_OF_DAY_ANOMALY,
                "score": 0.8,
                "confidence": 0.9,
                "features": {
                    "current_hour": current_hour,
                    "trading_hours": [9, 10, 11, 12, 13, 14, 15, 16],
                },
                "explanation": f"非交易时间活动：当前时间{current_hour}点",
                "recommendations": ["确认交易意图", "检查账户安全"],
            }

        return None

    def _detect_asset_concentration(
        self, features: Dict[str, float], user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """检测资产集中度异常"""
        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]

        # 简化的集中度检测
        if profile.diversification_score < 0.3:  # 分散化评分过低
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.ASSET_CONCENTRATION,
                "score": 1.0 - profile.diversification_score,
                "confidence": 0.8,
                "features": {
                    "diversification_score": profile.diversification_score,
                    "concentration_risk": 1.0 - profile.diversification_score,
                },
                "explanation": f"资产集中度异常：分散化评分{profile.diversification_score:.2f}",
                "recommendations": ["增加资产分散化", "降低单一资产风险暴露"],
            }

        return None

    def _detect_login_frequency_anomaly(
        self, features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """检测登录频率异常"""
        login_frequency = features.get("login_frequency", 0)

        # 假设正常登录频率为每天1-5次
        if login_frequency > 10:  # 异常高频率
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.TRADING_FREQUENCY_ANOMALY,
                "score": min(login_frequency / 10, 1.0),
                "confidence": 0.7,
                "features": {"login_frequency": login_frequency},
                "explanation": f"登录频率异常：每天{login_frequency}次",
                "recommendations": ["检查账户安全", "确认操作意图"],
            }

        return None

    def _detect_operation_pattern_anomaly(
        self, features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """检测操作模式异常"""
        operation_consistency = features.get("operation_consistency", 0)

        if operation_consistency < 0.3:  # 操作一致性过低
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.TRADING_PATTERN_CHANGE,
                "score": 1.0 - operation_consistency,
                "confidence": 0.6,
                "features": {"operation_consistency": operation_consistency},
                "explanation": f"操作模式异常：一致性评分{operation_consistency:.2f}",
                "recommendations": ["检查操作模式", "确认账户安全"],
            }

        return None

    def _detect_session_duration_anomaly(
        self, features: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """检测会话时长异常"""
        session_duration = features.get("session_duration_avg", 0)

        # 假设正常会话时长为10-60分钟
        if session_duration > 120:  # 异常长会话
            return {
                "is_anomaly": True,
                "type": BehavioralAnomalyType.TIME_OF_DAY_ANOMALY,
                "score": min(session_duration / 120, 1.0),
                "confidence": 0.7,
                "features": {"session_duration_avg": session_duration},
                "explanation": f"会话时长异常：平均{session_duration}分钟",
                "recommendations": ["检查会话活动", "确认操作意图"],
            }

        return None

    def _analyze_trading_hours(self, trading_hours: List[int]) -> List[int]:
        """分析交易时间偏好"""
        if not trading_hours:
            return []

        hour_counts = {}
        for hour in trading_hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # 返回交易频率最高的3个小时
        preferred_hours = sorted(
            hour_counts.keys(), key=lambda x: hour_counts[x], reverse=True
        )[:3]
        return preferred_hours

    def _calculate_hour_match(
        self, current_hour: int, preferred_hours: List[int]
    ) -> float:
        """计算时间匹配度"""
        if not preferred_hours:
            return 0.0

        if current_hour in preferred_hours:
            return 1.0

        # 计算与最近偏好时间的距离
        min_distance = min(abs(current_hour - hour) for hour in preferred_hours)
        return max(0.0, 1.0 - min_distance / 6.0)  # 6小时内有一定匹配度

    def _calculate_risk_tolerance(self, user_trades: List[Dict]) -> float:
        """
        计算风险容忍度 - 专业级实现
        
        使用多维度风险指标：
        1. 波动率指标
        2. 下行风险指标
        3. 极端风险指标
        4. 行为风险指标
        
        Args:
            user_trades: 用户交易记录
            
        Returns:
            综合风险容忍度评分 (0-1)
        """
        if len(user_trades) < self.min_trades_for_profile:
            return 0.0

        # 提取交易数据
        trade_sizes = [trade["notional"] for trade in user_trades]
        trade_prices = [trade["price"] for trade in user_trades]
        trade_timestamps = [trade["timestamp"] for trade in user_trades]
        
        # 计算收益率序列
        returns = self._calculate_returns(trade_prices, trade_timestamps)
        
        if len(returns) < 10:
            return 0.0

        # 1. 波动率指标 (权重: 25%)
        volatility_metrics = self._calculate_volatility_metrics(returns)
        
        # 2. 下行风险指标 (权重: 30%)
        downside_metrics = self._calculate_downside_metrics(returns)
        
        # 3. 极端风险指标 (权重: 25%)
        extreme_risk_metrics = self._calculate_extreme_risk_metrics(returns)
        
        # 4. 行为风险指标 (权重: 20%)
        behavioral_metrics = self._calculate_behavioral_risk_metrics(user_trades)
        
        # 综合风险容忍度评分
        risk_tolerance = (
            0.25 * volatility_metrics["normalized_volatility"] +
            0.30 * downside_metrics["normalized_var"] +
            0.25 * extreme_risk_metrics["normalized_es"] +
            0.20 * behavioral_metrics["normalized_behavioral_risk"]
        )
        
        return min(max(risk_tolerance, 0.0), 1.0)

    def _calculate_returns(self, prices: List[float], timestamps: List[datetime]) -> List[float]:
        """计算收益率序列"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        return returns

    def _calculate_volatility_metrics(self, returns: List[float]) -> Dict[str, float]:
        """计算波动率指标"""
        if not returns:
            return {"normalized_volatility": 0.0}
        
        returns_array = np.array(returns)
        
        # 年化波动率
        annual_volatility = np.std(returns_array) * np.sqrt(252)
        
        # 滚动波动率
        rolling_vol = self._calculate_rolling_volatility(returns_array, window=20)
        
        # 波动率聚集性
        vol_clustering = self._calculate_volatility_clustering(returns_array)
        
        # 归一化波动率 (假设正常范围0.1-0.4)
        normalized_vol = min(max((annual_volatility - 0.1) / 0.3, 0.0), 1.0)
        
        return {
            "annual_volatility": annual_volatility,
            "rolling_volatility": np.mean(rolling_vol) if rolling_vol else 0.0,
            "volatility_clustering": vol_clustering,
            "normalized_volatility": normalized_vol
        }

    def _calculate_downside_metrics(self, returns: List[float]) -> Dict[str, float]:
        """计算下行风险指标"""
        if not returns:
            return {"normalized_var": 0.0}
        
        returns_array = np.array(returns)
        
        # VaR计算 (历史模拟法)
        var_95 = np.percentile(returns_array, (1 - self.var_confidence) * 100)
        
        # 期望损失 (ES)
        es_95 = returns_array[returns_array <= var_95].mean() if len(returns_array[returns_array <= var_95]) > 0 else var_95
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(returns_array)
        
        # 索提诺比率
        sortino_ratio = self._calculate_sortino_ratio(returns_array)
        
        # 归一化VaR (假设正常范围-0.05到-0.15)
        normalized_var = min(max((abs(var_95) - 0.05) / 0.1, 0.0), 1.0)
        
        return {
            "var_95": var_95,
            "expected_shortfall": es_95,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino_ratio,
            "normalized_var": normalized_var
        }

    def _calculate_extreme_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """计算极端风险指标"""
        if not returns:
            return {"normalized_es": 0.0}
        
        returns_array = np.array(returns)
        
        # 极值理论 - 广义帕累托分布拟合
        extreme_returns = returns_array[returns_array < np.percentile(returns_array, 10)]
        
        if len(extreme_returns) > 5:
            # 使用广义帕累托分布拟合尾部
            try:
                params = stats.genpareto.fit(extreme_returns)
                tail_risk = params[0]  # 形状参数
            except:
                tail_risk = 0.0
        else:
            tail_risk = 0.0
        
        # 肥尾指标 (峰度)
        kurtosis = stats.kurtosis(returns_array)
        
        # 跳跃风险
        jump_risk = self._calculate_jump_risk(returns_array)
        
        # 归一化期望损失
        es_97_5 = np.percentile(returns_array, (1 - self.es_confidence) * 100)
        normalized_es = min(max((abs(es_97_5) - 0.08) / 0.12, 0.0), 1.0)
        
        return {
            "tail_risk": tail_risk,
            "kurtosis": kurtosis,
            "jump_risk": jump_risk,
            "normalized_es": normalized_es
        }

    def _calculate_behavioral_risk_metrics(self, user_trades: List[Dict]) -> Dict[str, float]:
        """计算行为风险指标"""
        # 损失厌恶系数
        loss_aversion = self._calculate_loss_aversion(user_trades)
        
        # 过度自信偏差
        overconfidence = self._calculate_overconfidence_bias(user_trades)
        
        # 处置效应
        disposition_effect = self._calculate_disposition_effect(user_trades)
        
        # 综合行为风险
        behavioral_risk = (loss_aversion + overconfidence + disposition_effect) / 3.0
        normalized_behavioral_risk = min(max(behavioral_risk, 0.0), 1.0)
        
        return {
            "loss_aversion": loss_aversion,
            "overconfidence": overconfidence,
            "disposition_effect": disposition_effect,
            "normalized_behavioral_risk": normalized_behavioral_risk
        }

    def _calculate_diversification_score(self, user_trades: List[Dict]) -> float:
        """
        计算分散化评分 - 专业级实现
        
        使用现代投资组合理论和多种风险度量：
        1. 赫芬达尔-赫希曼指数 (HHI)
        2. 基尼系数
        3. 集中度比率
        4. 有效资产数量
        5. 风险分散化度量
        6. 相关性分析
        
        Args:
            user_trades: 用户交易记录
            
        Returns:
            综合分散化评分 (0-1)
        """
        if not user_trades:
            return 0.0

        # 提取交易数据
        symbols = [trade["symbol"] for trade in user_trades]
        trade_sizes = [trade["notional"] for trade in user_trades]
        
        # 1. 资产集中度度量
        concentration_metrics = self._calculate_concentration_metrics(symbols, trade_sizes)
        
        # 2. 风险分散化度量
        risk_diversification_metrics = self._calculate_risk_diversification_metrics(user_trades)
        
        # 3. 相关性分析
        correlation_metrics = self._calculate_correlation_metrics(user_trades)
        
        # 综合分散化评分
        diversification_score = (
            0.4 * concentration_metrics["normalized_diversification"] +
            0.4 * risk_diversification_metrics["normalized_risk_diversification"] +
            0.2 * correlation_metrics["normalized_correlation_diversification"]
        )
        
        return min(max(diversification_score, 0.0), 1.0)

    def _calculate_concentration_metrics(
        self, symbols: List[str], trade_sizes: List[float]
    ) -> Dict[str, float]:
        """计算资产集中度度量"""
        if not symbols or not trade_sizes:
            return {"normalized_diversification": 0.0}
        
        # 1. 赫芬达尔-赫希曼指数 (HHI)
        symbol_counts = {}
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        total_trades = len(symbols)
        hhi = sum((count / total_trades) ** 2 for count in symbol_counts.values())

        # 2. 基于交易规模的HHI
        symbol_weights = {}
        for symbol, size in zip(symbols, trade_sizes):
            symbol_weights[symbol] = symbol_weights.get(symbol, 0) + size
        
        total_size = sum(symbol_weights.values())
        if total_size > 0:
            hhi_size = sum((size / total_size) ** 2 for size in symbol_weights.values())
        else:
            hhi_size = 1.0
        
        # 3. 基尼系数
        gini_coefficient = self._calculate_gini_coefficient(list(symbol_weights.values()))
        
        # 4. 集中度比率 (CR3, CR5)
        sorted_weights = sorted(symbol_weights.values(), reverse=True)
        cr3 = sum(sorted_weights[:3]) / total_size if total_size > 0 else 1.0
        cr5 = sum(sorted_weights[:5]) / total_size if total_size > 0 and len(sorted_weights) >= 5 else 1.0
        
        # 5. 有效资产数量
        effective_number = 1 / hhi if hhi > 0 else len(symbol_counts)
        
        # 归一化分散化评分
        normalized_diversification = 1.0 - (hhi_size + gini_coefficient + cr3) / 3.0
        
        return {
            "hhi": hhi,
            "hhi_size": hhi_size,
            "gini_coefficient": gini_coefficient,
            "cr3": cr3,
            "cr5": cr5,
            "effective_number": effective_number,
            "normalized_diversification": max(0.0, normalized_diversification)
        }

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values:
            return 0.0
        
        # 排序并计算累积分布
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumulative_values = np.cumsum(sorted_values)
        
        # 计算基尼系数
        if cumulative_values[-1] == 0:
            return 0.0
        
        # 使用梯形法则计算洛伦兹曲线下面积
        lorenz_curve = cumulative_values / cumulative_values[-1]
        lorenz_area = np.trapz(lorenz_curve, dx=1.0/n)
        
        # 基尼系数 = 1 - 2 * 洛伦兹曲线下面积
        gini = 1 - 2 * lorenz_area
        return max(0.0, min(gini, 1.0))

    def _calculate_risk_diversification_metrics(self, user_trades: List[Dict]) -> Dict[str, float]:
        """计算风险分散化度量"""
        if len(user_trades) < 10:
            return {"normalized_risk_diversification": 0.0}
        
        # 按资产分组计算风险指标
        symbol_groups = {}
        for trade in user_trades:
            symbol = trade["symbol"]
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(trade)
        
        # 计算各资产的波动率
        asset_volatilities = {}
        for symbol, trades in symbol_groups.items():
            if len(trades) >= 5:
                prices = [t["price"] for t in trades]
                returns = self._calculate_returns(prices, [t["timestamp"] for t in trades])
                if len(returns) >= 5:
                    asset_volatilities[symbol] = np.std(returns) * np.sqrt(252)
        
        if len(asset_volatilities) < 2:
            return {"normalized_risk_diversification": 0.0}
        
        # 计算风险集中度
        total_risk = sum(asset_volatilities.values())
        risk_weights = {symbol: vol/total_risk for symbol, vol in asset_volatilities.items()}
        
        # 风险集中度HHI
        risk_hhi = sum(weight ** 2 for weight in risk_weights.values())
        
        # 风险分散化评分
        risk_diversification = 1.0 - risk_hhi
        
        # 考虑资产数量对分散化的影响
        num_assets = len(asset_volatilities)
        diversification_bonus = min(num_assets / 10, 1.0)  # 最多10个资产的分散化收益
        
        normalized_risk_diversification = risk_diversification * diversification_bonus
        
        return {
            "risk_hhi": risk_hhi,
            "num_assets": num_assets,
            "risk_diversification": risk_diversification,
            "normalized_risk_diversification": normalized_risk_diversification
        }

    def _calculate_correlation_metrics(self, user_trades: List[Dict]) -> Dict[str, float]:
        """计算相关性度量"""
        if len(user_trades) < 20:
            return {"normalized_correlation_diversification": 0.0}
        
        # 按资产分组并计算收益率序列
        symbol_returns = {}
        for trade in user_trades:
            symbol = trade["symbol"]
            if symbol not in symbol_returns:
                symbol_returns[symbol] = []
        
        # 为每个资产构建价格序列
        symbol_prices = {}
        symbol_timestamps = {}
        
        for trade in user_trades:
            symbol = trade["symbol"]
            if symbol not in symbol_prices:
                symbol_prices[symbol] = []
                symbol_timestamps[symbol] = []
            
            symbol_prices[symbol].append(trade["price"])
            symbol_timestamps[symbol].append(trade["timestamp"])
        
        # 计算各资产的收益率序列
        for symbol in symbol_prices:
            if len(symbol_prices[symbol]) >= 5:
                returns = self._calculate_returns(
                    symbol_prices[symbol], symbol_timestamps[symbol]
                )
                if len(returns) >= 5:
                    symbol_returns[symbol] = returns
        
        # 过滤掉数据不足的资产
        valid_symbols = [symbol for symbol in symbol_returns if len(symbol_returns[symbol]) >= 5]
        
        if len(valid_symbols) < 2:
            return {"normalized_correlation_diversification": 0.0}
        
        # 对齐收益率序列长度
        min_length = min(len(symbol_returns[symbol]) for symbol in valid_symbols)
        aligned_returns = {}
        
        for symbol in valid_symbols:
            aligned_returns[symbol] = symbol_returns[symbol][-min_length:]
        
        # 计算相关性矩阵
        correlation_matrix = np.ones((len(valid_symbols), len(valid_symbols)))
        
        for i, symbol1 in enumerate(valid_symbols):
            for j, symbol2 in enumerate(valid_symbols):
                if i < j:
                    try:
                        corr = np.corrcoef(aligned_returns[symbol1], aligned_returns[symbol2])[0, 1]
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
                    except:
                        correlation_matrix[i, j] = 0.0
                        correlation_matrix[j, i] = 0.0
        
        # 计算平均相关性
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        avg_correlation = np.mean(np.abs(upper_triangle)) if len(upper_triangle) > 0 else 0.0
        
        # 相关性分散化评分 (相关性越低，分散化越好)
        correlation_diversification = 1.0 - avg_correlation
        
        # 考虑资产数量调整
        num_assets = len(valid_symbols)
        size_adjustment = min(num_assets / 5, 1.0)  # 最多5个资产的调整
        
        normalized_correlation_diversification = correlation_diversification * size_adjustment
        
        return {
            "avg_correlation": avg_correlation,
            "num_correlated_assets": num_assets,
            "correlation_diversification": correlation_diversification,
            "normalized_correlation_diversification": normalized_correlation_diversification
        }

    def _calculate_trading_consistency(self, user_trades: List[Dict]) -> float:
        """
        计算交易一致性 - 专业级实现
        
        使用时间序列分析和模式识别：
        1. 时间间隔一致性分析
        2. 交易规模稳定性
        3. 交易时间模式识别
        4. 交易方向一致性
        5. 机器学习模式识别
        
        Args:
            user_trades: 用户交易记录
            
        Returns:
            综合交易一致性评分 (0-1)
        """
        if len(user_trades) < 20:
            return 0.0

        # 1. 时间间隔一致性
        time_consistency = self._calculate_time_interval_consistency(user_trades)
        
        # 2. 交易规模稳定性
        size_consistency = self._calculate_trade_size_consistency(user_trades)
        
        # 3. 交易时间模式识别
        time_pattern_consistency = self._calculate_time_pattern_consistency(user_trades)
        
        # 4. 交易方向一致性
        direction_consistency = self._calculate_trade_direction_consistency(user_trades)
        
        # 5. 机器学习模式识别
        ml_consistency = self._calculate_ml_pattern_consistency(user_trades)
        
        # 综合一致性评分
        consistency_score = (
            0.25 * time_consistency +
            0.25 * size_consistency +
            0.20 * time_pattern_consistency +
            0.15 * direction_consistency +
            0.15 * ml_consistency
        )
        
        return min(max(consistency_score, 0.0), 1.0)

    def _calculate_time_interval_consistency(self, user_trades: List[Dict]) -> float:
        """计算时间间隔一致性"""
        if len(user_trades) < 10:
            return 0.0

        # 排序时间戳
        timestamps = sorted([trade["timestamp"] for trade in user_trades])
        intervals = []

        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600  # 小时
            intervals.append(interval)

        if not intervals:
            return 0.0

        intervals_array = np.array(intervals)
        
        # 1. 变异系数 (CV)
        cv = np.std(intervals_array) / np.mean(intervals_array) if np.mean(intervals_array) > 0 else 1.0
        cv_consistency = max(0.0, 1.0 - min(cv, 1.0))
        
        # 2. 自相关性
        if len(intervals_array) > 5:
            try:
                autocorr = np.corrcoef(intervals_array[:-1], intervals_array[1:])[0, 1]
                autocorr_consistency = max(0.0, 1.0 - abs(autocorr))
            except:
                autocorr_consistency = 0.0
        else:
            autocorr_consistency = 0.0
        
        # 3. 趋势分析
        if len(intervals_array) > 10:
            try:
                # 使用线性回归检测趋势
                x = np.arange(len(intervals_array))
                slope, _, _, _, _ = stats.linregress(x, intervals_array)
                trend_consistency = max(0.0, 1.0 - abs(slope) * 10)  # 斜率越小越一致
            except:
                trend_consistency = 0.0
        else:
            trend_consistency = 0.0
        
        # 综合时间间隔一致性
        time_consistency = (cv_consistency + autocorr_consistency + trend_consistency) / 3.0
        return time_consistency

    def _calculate_trade_size_consistency(self, user_trades: List[Dict]) -> float:
        """计算交易规模稳定性"""
        if len(user_trades) < 10:
            return 0.0

        trade_sizes = [trade["notional"] for trade in user_trades]
        sizes_array = np.array(trade_sizes)
        
        # 1. 规模变异系数
        cv_size = np.std(sizes_array) / np.mean(sizes_array) if np.mean(sizes_array) > 0 else 1.0
        cv_consistency = max(0.0, 1.0 - min(cv_size, 1.0))
        
        # 2. 规模分布正态性检验
        if len(sizes_array) >= 20:
            try:
                _, normality_pvalue = stats.normaltest(sizes_array)
                normality_consistency = min(normality_pvalue, 1.0)  # p值越大越符合正态分布
            except:
                normality_consistency = 0.0
        else:
            normality_consistency = 0.0
        
        # 3. 规模变化趋势
        if len(sizes_array) > 10:
            try:
                x = np.arange(len(sizes_array))
                slope, _, _, _, _ = stats.linregress(x, sizes_array)
                trend_consistency = max(0.0, 1.0 - abs(slope / np.mean(sizes_array)) * 100)
            except:
                trend_consistency = 0.0
        else:
            trend_consistency = 0.0
        
        # 综合规模一致性
        size_consistency = (cv_consistency + normality_consistency + trend_consistency) / 3.0
        return size_consistency

    def _calculate_time_pattern_consistency(self, user_trades: List[Dict]) -> float:
        """计算交易时间模式一致性"""
        if len(user_trades) < 20:
            return 0.0
        
        trading_hours = [trade["timestamp"].hour for trade in user_trades]
        trading_days = [trade["timestamp"].weekday() for trade in user_trades]
        
        # 1. 小时模式一致性
        hour_counts = {}
        for hour in trading_hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        total_trades = len(trading_hours)
        hour_entropy = -sum((count/total_trades) * np.log(count/total_trades) 
                           for count in hour_counts.values() if count > 0)
        max_entropy = np.log(len(hour_counts)) if len(hour_counts) > 0 else 0
        
        if max_entropy > 0:
            hour_consistency = 1.0 - (hour_entropy / max_entropy)
        else:
            hour_consistency = 0.0
        
        # 2. 星期模式一致性
        day_counts = {}
        for day in trading_days:
            day_counts[day] = day_counts.get(day, 0) + 1
        
        day_entropy = -sum((count/total_trades) * np.log(count/total_trades) 
                          for count in day_counts.values() if count > 0)
        max_day_entropy = np.log(len(day_counts)) if len(day_counts) > 0 else 0
        
        if max_day_entropy > 0:
            day_consistency = 1.0 - (day_entropy / max_day_entropy)
        else:
            day_consistency = 0.0
        
        # 3. 时间聚类分析
        time_cluster_consistency = self._analyze_time_clustering(trading_hours)
        
        # 综合时间模式一致性
        time_pattern_consistency = (hour_consistency + day_consistency + time_cluster_consistency) / 3.0
        return time_pattern_consistency

    def _analyze_time_clustering(self, trading_hours: List[int]) -> float:
        """分析时间聚类模式"""
        if len(trading_hours) < 10:
            return 0.0
        
        # 使用DBSCAN聚类分析交易时间
        hours_array = np.array(trading_hours).reshape(-1, 1)
        
        try:
            clusters = self.dbscan.fit_predict(hours_array)
            unique_clusters = set(clusters)
            
            # 排除噪声点 (-1)
            valid_clusters = [c for c in unique_clusters if c != -1]
            
            if len(valid_clusters) == 0:
                return 0.0
            
            # 计算聚类质量
            cluster_sizes = []
            for cluster in valid_clusters:
                cluster_points = hours_array[clusters == cluster]
                cluster_sizes.append(len(cluster_points))
            
            # 聚类一致性：主要聚类占比
            total_valid_points = sum(cluster_sizes)
            if total_valid_points > 0:
                main_cluster_ratio = max(cluster_sizes) / total_valid_points
                return min(main_cluster_ratio, 1.0)
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_trade_direction_consistency(self, user_trades: List[Dict]) -> float:
        """计算交易方向一致性"""
        if len(user_trades) < 10:
            return 0.0
        
        order_types = [trade["order_type"] for trade in user_trades]
        
        # 1. 买卖方向比例一致性
        buy_count = order_types.count("buy")
        sell_count = order_types.count("sell")
        total_count = len(order_types)
        
        if total_count == 0:
            return 0.0
        
        # 方向平衡性 (越接近0.5越平衡)
        buy_ratio = buy_count / total_count
        direction_balance = 1.0 - 2 * abs(buy_ratio - 0.5)
        
        # 2. 方向序列自相关性
        direction_sequence = [1 if order_type == "buy" else -1 for order_type in order_types]
        
        if len(direction_sequence) > 5:
            try:
                autocorr = np.corrcoef(direction_sequence[:-1], direction_sequence[1:])[0, 1]
                sequence_consistency = max(0.0, 1.0 - abs(autocorr))
            except:
                sequence_consistency = 0.0
        else:
            sequence_consistency = 0.0
        
        # 3. 方向变化频率
        direction_changes = 0
        for i in range(1, len(direction_sequence)):
            if direction_sequence[i] != direction_sequence[i-1]:
                direction_changes += 1
        
        change_frequency = direction_changes / (len(direction_sequence) - 1) if len(direction_sequence) > 1 else 0
        change_consistency = max(0.0, 1.0 - change_frequency)
        
        # 综合方向一致性
        direction_consistency = (direction_balance + sequence_consistency + change_consistency) / 3.0
        return direction_consistency

    def _calculate_ml_pattern_consistency(self, user_trades: List[Dict]) -> float:
        """使用机器学习计算模式一致性"""
        if len(user_trades) < 30:
            return 0.0
        
        # 构建特征矩阵
        features = []
        
        for i in range(len(user_trades)):
            if i >= 5:  # 使用前5个交易构建特征
                window_trades = user_trades[max(0, i-5):i]
                
                # 时间特征
                timestamps = [trade["timestamp"] for trade in window_trades]
                time_intervals = []
                for j in range(1, len(timestamps)):
                    interval = (timestamps[j] - timestamps[j-1]).total_seconds() / 3600
                    time_intervals.append(interval)
                
                # 规模特征
                sizes = [trade["notional"] for trade in window_trades]
                
                # 方向特征
                directions = [1 if trade["order_type"] == "buy" else -1 for trade in window_trades]
                
                # 构建特征向量
                feature_vector = [
                    np.mean(time_intervals) if time_intervals else 0,
                    np.std(time_intervals) if time_intervals else 0,
                    np.mean(sizes) if sizes else 0,
                    np.std(sizes) if sizes else 0,
                    np.mean(directions) if directions else 0,
                    len(set([trade["symbol"] for trade in window_trades])) / len(window_trades),
                ]
                features.append(feature_vector)
        
        if len(features) < 10:
            return 0.0
        
        # 使用孤立森林检测异常模式
        try:
            features_array = np.array(features)
            self.isolation_forest.fit(features_array)
            
            # 计算所有样本的异常分数
            anomaly_scores = -self.isolation_forest.score_samples(features_array)
            
            # 一致性评分：异常分数越低越一致
            consistency_score = 1.0 - np.mean(anomaly_scores)
            return max(0.0, consistency_score)
        except:
            return 0.0

    def _calculate_overall_risk_score(self, anomalies: List[Dict]) -> float:
        """计算整体风险评分"""
        if not anomalies:
            return 0.0

        total_score = sum(anomaly.get("score", 0) for anomaly in anomalies)
        return min(total_score / len(anomalies), 1.0)

    def _calculate_behavioral_risk_score(self, anomalies: List[Dict]) -> float:
        """计算行为风险评分"""
        return self._calculate_overall_risk_score(anomalies)

    def get_behavioral_statistics(self) -> Dict[str, Any]:
        """获取行为统计信息"""
        return {
            "user_profiles_count": len(self.user_profiles),
            "trading_history_size": len(self.trading_history),
            "detection_parameters": {
                "baseline_period_days": self.baseline_period_days,
                "anomaly_threshold": self.anomaly_threshold,
                "frequency_threshold_multiplier": self.frequency_threshold_multiplier,
                "size_threshold_multiplier": self.size_threshold_multiplier,
            },
        }

    def _calculate_rolling_volatility(self, returns: np.ndarray, window: int) -> List[float]:
        """计算滚动波动率"""
        if len(returns) < window:
            return []
        
        rolling_vol = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            vol = np.std(window_returns) * np.sqrt(252)
            rolling_vol.append(vol)
        
        return rolling_vol

    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """计算波动率聚集性"""
        if len(returns) < 10:
            return 0.0
        
        # 使用绝对收益率的自相关性
        abs_returns = np.abs(returns)
        if len(abs_returns) > 1:
            autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
            return max(0.0, autocorr)
        return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """计算索提诺比率"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        return mean_return / downside_std

    def _calculate_jump_risk(self, returns: np.ndarray) -> float:
        """计算跳跃风险"""
        if len(returns) < 10:
            return 0.0
        
        # 使用双幂变差检测跳跃
        bpv = np.sum(np.abs(returns[:-1] * returns[1:]))
        rv = np.sum(returns**2)
        
        if rv > 0:
            jump_ratio = max(0, 1 - bpv / rv)
            return jump_ratio
        return 0.0

    def _calculate_loss_aversion(self, user_trades: List[Dict]) -> float:
        """计算损失厌恶系数"""
        if len(user_trades) < 10:
            return 0.0
        
        # 分析盈利和亏损交易的持有时间
        profitable_trades = [t for t in user_trades if t.get('profit', 0) > 0]
        loss_trades = [t for t in user_trades if t.get('profit', 0) < 0]
        
        if len(profitable_trades) == 0 or len(loss_trades) == 0:
            return 0.0
        
        # 计算平均持有时间比率
        avg_profit_holding = np.mean([(t['timestamp'] - t.get('entry_time', t['timestamp'])).total_seconds() 
                                     for t in profitable_trades])
        avg_loss_holding = np.mean([(t['timestamp'] - t.get('entry_time', t['timestamp'])).total_seconds() 
                                   for t in loss_trades])
        
        if avg_loss_holding > 0:
            loss_aversion_ratio = avg_profit_holding / avg_loss_holding
            return min(max(loss_aversion_ratio - 1, 0.0), 1.0)
        return 0.0

    def _calculate_overconfidence_bias(self, user_trades: List[Dict]) -> float:
        """计算过度自信偏差"""
        if len(user_trades) < 10:
            return 0.0
        
        # 分析交易频率和成功率
        total_trades = len(user_trades)
        profitable_trades = len([t for t in user_trades if t.get('profit', 0) > 0])
        
        success_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
        
        # 过度自信指标：高交易频率 + 低成功率
        trading_frequency = total_trades / max(1, (max(t['timestamp'] for t in user_trades) - 
                                                  min(t['timestamp'] for t in user_trades)).days)
        
        # 归一化过度自信评分
        overconfidence = (1 - success_rate) * min(trading_frequency / 10, 1.0)
        return min(max(overconfidence, 0.0), 1.0)

    def _calculate_disposition_effect(self, user_trades: List[Dict]) -> float:
        """计算处置效应"""
        if len(user_trades) < 10:
            return 0.0
        
        # 分析盈利交易和亏损交易的卖出比例
        profitable_trades = [t for t in user_trades if t.get('profit', 0) > 0]
        loss_trades = [t for t in user_trades if t.get('profit', 0) < 0]
        
        if len(profitable_trades) == 0 or len(loss_trades) == 0:
            return 0.0
        
        # 计算卖出比例
        sell_profitable = len([t for t in profitable_trades if t.get('order_type') == 'sell'])
        sell_loss = len([t for t in loss_trades if t.get('order_type') == 'sell'])
        
        prop_sell_profitable = sell_profitable / len(profitable_trades) if len(profitable_trades) > 0 else 0.0
        prop_sell_loss = sell_loss / len(loss_trades) if len(loss_trades) > 0 else 0.0
        
        # 处置效应：过早卖出盈利交易，持有亏损交易
        disposition_effect = max(0, prop_sell_profitable - prop_sell_loss)
        return min(disposition_effect, 1.0)

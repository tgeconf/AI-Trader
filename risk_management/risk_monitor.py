"""
风险监控器
实现实时风险监控、预警和风险限额管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """警报类型枚举"""
    VAR_BREACH = "var_breach"
    CVAR_BREACH = "cvar_breach"
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_RISK = "concentration_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class RiskAlert:
    """风险警报"""
    alert_type: AlertType
    level: RiskLevel
    message: str
    timestamp: datetime
    asset: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    breach_amount: Optional[float] = None


@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation_matrix: pd.DataFrame
    concentration_ratio: float
    liquidity_metrics: Dict[str, float]


class RiskMonitor:
    """
    风险监控器
    
    实现实时风险监控功能：
    - VaR/CVaR监控和预警
    - 头寸限额监控
    - 集中度风险监控
    - 流动性风险监控
    - 波动率异常监控
    """
    
    def __init__(self, 
                 var_threshold: float = 0.05,
                 cvar_threshold: float = 0.07,
                 position_limit: float = 0.1,
                 concentration_limit: float = 0.2,
                 volatility_threshold: float = 0.3):
        """
        初始化风险监控器
        
        Args:
            var_threshold: VaR阈值（占组合价值的百分比）
            cvar_threshold: CVaR阈值
            position_limit: 单资产头寸限额
            concentration_limit: 集中度限额
            volatility_threshold: 波动率异常阈值
        """
        self.var_threshold = var_threshold
        self.cvar_threshold = cvar_threshold
        self.position_limit = position_limit
        self.concentration_limit = concentration_limit
        self.volatility_threshold = volatility_threshold
        
        self.alerts: List[RiskAlert] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        self.monitoring_start_time = datetime.now()
        
    def monitor_portfolio_risk(self, 
                             portfolio_returns: pd.DataFrame,
                             portfolio_weights: Dict[str, float],
                             portfolio_value: float,
                             var_model: Any,
                             cvar_model: Any) -> List[RiskAlert]:
        """
        监控投资组合风险
        
        Args:
            portfolio_returns: 投资组合收益率数据
            portfolio_weights: 投资组合权重
            portfolio_value: 投资组合价值
            var_model: VaR模型实例
            cvar_model: CVaR模型实例
            
        Returns:
            风险警报列表
        """
        current_alerts = []
        
        # 计算VaR和CVaR
        var_result = var_model.calculate_var(
            returns=portfolio_returns.iloc[:, 0].values,
            confidence_level=0.95,
            method="historical"
        )
        
        cvar_result = cvar_model.calculate_cvar(
            returns=portfolio_returns.iloc[:, 0].values,
            confidence_level=0.95
        )
        
        # VaR突破检查
        var_ratio = abs(var_result) / portfolio_value
        if var_ratio > self.var_threshold:
            alert = RiskAlert(
                alert_type=AlertType.VAR_BREACH,
                level=RiskLevel.HIGH,
                message=f"VaR突破阈值: {var_ratio:.2%} > {self.var_threshold:.2%}",
                timestamp=datetime.now(),
                current_value=var_ratio,
                threshold=self.var_threshold,
                breach_amount=var_ratio - self.var_threshold
            )
            current_alerts.append(alert)
        
        # CVaR突破检查
        cvar_ratio = abs(cvar_result) / portfolio_value
        if cvar_ratio > self.cvar_threshold:
            alert = RiskAlert(
                alert_type=AlertType.CVAR_BREACH,
                level=RiskLevel.CRITICAL,
                message=f"CVaR突破阈值: {cvar_ratio:.2%} > {self.cvar_threshold:.2%}",
                timestamp=datetime.now(),
                current_value=cvar_ratio,
                threshold=self.cvar_threshold,
                breach_amount=cvar_ratio - self.cvar_threshold
            )
            current_alerts.append(alert)
        
        # 头寸限额检查
        position_alerts = self._check_position_limits(portfolio_weights)
        current_alerts.extend(position_alerts)
        
        # 集中度风险检查
        concentration_alerts = self._check_concentration_risk(portfolio_weights)
        current_alerts.extend(concentration_alerts)
        
        # 波动率异常检查
        volatility_alerts = self._check_volatility_anomalies(portfolio_returns)
        current_alerts.extend(volatility_alerts)
        
        # 记录警报
        self.alerts.extend(current_alerts)
        
        return current_alerts
    
    def _check_position_limits(self, portfolio_weights: Dict[str, float]) -> List[RiskAlert]:
        """检查头寸限额"""
        alerts = []
        
        for asset, weight in portfolio_weights.items():
            if abs(weight) > self.position_limit:
                alert = RiskAlert(
                    alert_type=AlertType.POSITION_LIMIT,
                    level=RiskLevel.MEDIUM,
                    message=f"{asset}头寸超限: {weight:.2%} > {self.position_limit:.2%}",
                    timestamp=datetime.now(),
                    asset=asset,
                    current_value=weight,
                    threshold=self.position_limit,
                    breach_amount=weight - self.position_limit
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_concentration_risk(self, portfolio_weights: Dict[str, float]) -> List[RiskAlert]:
        """检查集中度风险"""
        alerts = []
        
        if not portfolio_weights:
            return alerts
        
        # 计算前N大资产集中度
        sorted_weights = sorted(portfolio_weights.values(), reverse=True)
        top_3_concentration = sum(sorted_weights[:min(3, len(sorted_weights))])
        
        if top_3_concentration > self.concentration_limit:
            alert = RiskAlert(
                alert_type=AlertType.CONCENTRATION_RISK,
                level=RiskLevel.HIGH,
                message=f"集中度风险: 前3大资产占比{top_3_concentration:.2%} > {self.concentration_limit:.2%}",
                timestamp=datetime.now(),
                current_value=top_3_concentration,
                threshold=self.concentration_limit,
                breach_amount=top_3_concentration - self.concentration_limit
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_volatility_anomalies(self, portfolio_returns: pd.DataFrame) -> List[RiskAlert]:
        """检查波动率异常"""
        alerts = []
        
        if len(portfolio_returns) < 20:
            return alerts
        
        # 计算滚动波动率
        rolling_volatility = portfolio_returns.rolling(window=20).std().iloc[-1, 0]
        
        if rolling_volatility > self.volatility_threshold:
            alert = RiskAlert(
                alert_type=AlertType.VOLATILITY_SPIKE,
                level=RiskLevel.MEDIUM,
                message=f"波动率异常: {rolling_volatility:.2%} > {self.volatility_threshold:.2%}",
                timestamp=datetime.now(),
                current_value=rolling_volatility,
                threshold=self.volatility_threshold,
                breach_amount=rolling_volatility - self.volatility_threshold
            )
            alerts.append(alert)
        
        return alerts
    
    def calculate_risk_metrics(self, 
                             portfolio_returns: pd.DataFrame,
                             portfolio_weights: Dict[str, float]) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            portfolio_returns: 投资组合收益率
            portfolio_weights: 投资组合权重
            
        Returns:
            风险指标
        """
        # 计算基本风险指标
        portfolio_volatility = portfolio_returns.std().iloc[0]
        
        # 计算最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min().iloc[0]
        
        # 计算Beta（需要基准数据）
        beta = 1.0  # 简化计算
        
        # 计算相关性矩阵
        correlation_matrix = portfolio_returns.corr()
        
        # 计算集中度比率
        weights_array = np.array(list(portfolio_weights.values()))
        concentration_ratio = np.sum(weights_array ** 2)  # Herfindahl指数
        
        # 流动性指标（简化）
        liquidity_metrics = {
            "turnover_ratio": 0.1,
            "bid_ask_spread": 0.002,
            "volume_ratio": 1.0
        }
        
        metrics = RiskMetrics(
            portfolio_var=0.0,  # 需要VaR模型计算
            portfolio_cvar=0.0,  # 需要CVaR模型计算
            max_drawdown=max_drawdown,
            volatility=portfolio_volatility,
            beta=beta,
            correlation_matrix=correlation_matrix,
            concentration_ratio=concentration_ratio,
            liquidity_metrics=liquidity_metrics
        )
        
        self.risk_metrics_history.append(metrics)
        return metrics
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        if not self.risk_metrics_history:
            return {}
        
        latest_metrics = self.risk_metrics_history[-1]
        
        # 计算风险等级
        risk_level = self._assess_overall_risk_level(latest_metrics)
        
        # 统计警报
        critical_alerts = len([a for a in self.alerts if a.level == RiskLevel.CRITICAL])
        high_alerts = len([a for a in self.alerts if a.level == RiskLevel.HIGH])
        
        return {
            "overall_risk_level": risk_level.value,
            "monitoring_duration": (datetime.now() - self.monitoring_start_time).total_seconds() / 3600,
            "total_alerts": len(self.alerts),
            "critical_alerts": critical_alerts,
            "high_alerts": high_alerts,
            "latest_metrics": {
                "volatility": latest_metrics.volatility,
                "max_drawdown": latest_metrics.max_drawdown,
                "concentration_ratio": latest_metrics.concentration_ratio
            },
            "alert_summary": self._get_alert_summary()
        }
    
    def _assess_overall_risk_level(self, metrics: RiskMetrics) -> RiskLevel:
        """评估整体风险等级"""
        risk_score = 0
        
        # 基于波动率评分
        if metrics.volatility > 0.25:
            risk_score += 2
        elif metrics.volatility > 0.15:
            risk_score += 1
        
        # 基于最大回撤评分
        if abs(metrics.max_drawdown) > 0.20:
            risk_score += 2
        elif abs(metrics.max_drawdown) > 0.10:
            risk_score += 1
        
        # 基于集中度评分
        if metrics.concentration_ratio > 0.15:
            risk_score += 1
        
        if risk_score >= 4:
            return RiskLevel.CRITICAL
        elif risk_score >= 2:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_alert_summary(self) -> Dict[str, int]:
        """获取警报摘要"""
        summary = {}
        for alert_type in AlertType:
            count = len([a for a in self.alerts if a.alert_type == alert_type])
            summary[alert_type.value] = count
        return summary
    
    def clear_alerts(self, alert_types: Optional[List[AlertType]] = None) -> None:
        """清除警报"""
        if alert_types is None:
            self.alerts.clear()
        else:
            self.alerts = [a for a in self.alerts if a.alert_type not in alert_types]
    
    def export_risk_report(self, filepath: str) -> None:
        """导出风险报告"""
        import json
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "risk_summary": self.get_risk_summary(),
            "recent_alerts": [
                {
                    "type": alert.alert_type.value,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "asset": alert.asset
                }
                for alert in self.alerts[-50:]  # 最近50个警报
            ],
            "monitoring_config": {
                "var_threshold": self.var_threshold,
                "cvar_threshold": self.cvar_threshold,
                "position_limit": self.position_limit,
                "concentration_limit": self.concentration_limit,
                "volatility_threshold": self.volatility_threshold
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

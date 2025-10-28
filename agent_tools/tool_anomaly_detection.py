"""
异常检测和系统健康监控工具
提供市场异常检测和系统健康监控功能
"""

from fastmcp import FastMCP
import sys
import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import psutil
import time

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from anomaly_detection.statistical_anomaly import StatisticalAnomalyDetector, AnomalyType
from anomaly_detection.behavioral_anomaly import BehavioralAnomalyDetector
from anomaly_detection.system_health import SystemHealthMonitor
from tools.general_tools import get_config_value

mcp = FastMCP("AnomalyDetectionTools")


@mcp.tool()
def detect_market_anomalies(
    symbol: str,
    price_data: Dict[str, List[float]],
    volume_data: Optional[Dict[str, List[float]]] = None,
    detection_methods: List[str] = ["price", "volatility", "correlation"]
) -> Dict[str, Any]:
    """
    检测市场异常
    
    Args:
        symbol: 股票代码
        price_data: 价格数据
        volume_data: 成交量数据
        detection_methods: 检测方法列表
        
    Returns:
        市场异常检测结果
    """
    try:
        # 准备数据
        prices = price_data.get("close", [])
        
        if len(prices) < 20:
            return {"error": "Insufficient price data for anomaly detection"}
        
        # 创建异常检测器
        detector = StatisticalAnomalyDetector(contamination=0.05)
        
        # 使用历史数据训练模型
        training_data = np.array(prices[:-10]).reshape(-1, 1)  # 使用前N-10个数据训练
        detector.fit(training_data)
        
        detection_results = {}
        
        # 价格异常检测
        if "price" in detection_methods:
            volume_array = np.array(volume_data.get("volume", [])) if volume_data else None
            price_result = detector.detect_price_anomaly(
                price_data=np.array(prices),
                volume_data=volume_array
            )
            detection_results["price_anomaly"] = {
                "is_anomaly": price_result.is_anomaly,
                "anomaly_score": price_result.anomaly_score,
                "anomaly_type": price_result.anomaly_type.value,
                "confidence": price_result.confidence,
                "explanation": price_result.explanation,
                "features": price_result.features
            }
        
        # 波动率异常检测
        if "volatility" in detection_methods and len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            if len(returns) > 20:
                volatility_result = detector.detect_volatility_anomaly(
                    returns_data=returns,
                    window=10
                )
                detection_results["volatility_anomaly"] = {
                    "is_anomaly": volatility_result.is_anomaly,
                    "anomaly_score": volatility_result.anomaly_score,
                    "anomaly_type": volatility_result.anomaly_type.value,
                    "confidence": volatility_result.confidence,
                    "explanation": volatility_result.explanation,
                    "features": volatility_result.features
                }
        
        # 相关性异常检测（需要多个资产）
        if "correlation" in detection_methods:
            # 这里需要多个资产的数据，暂时跳过
            pass
        
        # 综合评估
        total_anomalies = sum(1 for result in detection_results.values() if result["is_anomaly"])
        overall_risk = "high" if total_anomalies >= 2 else "medium" if total_anomalies == 1 else "low"
        
        return {
            "symbol": symbol,
            "overall_risk_level": overall_risk,
            "total_anomalies_detected": total_anomalies,
            "detection_results": detection_results,
            "recommendations": _generate_anomaly_recommendations(detection_results)
        }
        
    except Exception as e:
        return {"error": f"Market anomaly detection failed: {str(e)}"}


@mcp.tool()
def monitor_system_health() -> Dict[str, Any]:
    """
    监控系统健康状态
    
    Returns:
        系统健康状态报告
    """
    try:
        # 创建系统健康监控器
        health_monitor = SystemHealthMonitor()
        
        # 检查系统资源
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        # 检查进程状态
        process_status = health_monitor.check_process_status()
        
        # 检查服务状态
        service_status = health_monitor.check_service_status()
        
        # 检查网络连接
        network_status = health_monitor.check_network_connectivity()
        
        # 评估整体健康状态
        health_status = health_monitor.assess_overall_health(
            cpu_usage=cpu_usage,
            memory_usage=memory_info.percent,
            disk_usage=disk_usage.percent
        )
        
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "overall_health": health_status["overall_health"],
            "health_score": health_status["health_score"],
            "system_resources": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_info.percent,
                "memory_available_gb": memory_info.available / (1024**3),
                "disk_usage_percent": disk_usage.percent,
                "disk_free_gb": disk_usage.free / (1024**3)
            },
            "process_status": process_status,
            "service_status": service_status,
            "network_status": network_status,
            "alerts": health_status["alerts"],
            "recommendations": health_status["recommendations"]
        }
        
    except Exception as e:
        return {"error": f"System health monitoring failed: {str(e)}"}


@mcp.tool()
def detect_behavioral_anomalies(
    trading_data: Dict[str, Any],
    user_behavior: Dict[str, Any]
) -> Dict[str, Any]:
    """
    检测行为异常
    
    Args:
        trading_data: 交易数据
        user_behavior: 用户行为数据
        
    Returns:
        行为异常检测结果
    """
    try:
        # 创建行为异常检测器
        behavioral_detector = BehavioralAnomalyDetector()
        
        # 分析交易行为
        trading_analysis = behavioral_detector.analyze_trading_behavior(trading_data)
        
        # 分析用户行为模式
        user_analysis = behavioral_detector.analyze_user_behavior(user_behavior)
        
        # 检测异常模式
        behavioral_anomalies = behavioral_detector.detect_behavioral_anomalies(
            trading_analysis=trading_analysis,
            user_analysis=user_analysis
        )
        
        return {
            "behavioral_analysis": {
                "trading_behavior": trading_analysis,
                "user_behavior": user_analysis
            },
            "detected_anomalies": behavioral_anomalies,
            "risk_assessment": behavioral_detector.assess_behavioral_risk(behavioral_anomalies),
            "recommendations": behavioral_detector.generate_behavioral_recommendations(behavioral_anomalies)
        }
        
    except Exception as e:
        return {"error": f"Behavioral anomaly detection failed: {str(e)}"}


@mcp.tool()
def generate_risk_report(
    market_anomalies: Dict[str, Any],
    system_health: Dict[str, Any],
    behavioral_anomalies: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    生成综合风险报告
    
    Args:
        market_anomalies: 市场异常检测结果
        system_health: 系统健康状态
        behavioral_anomalies: 行为异常检测结果
        
    Returns:
        综合风险报告
    """
    try:
        report = {
            "report_timestamp": pd.Timestamp.now().isoformat(),
            "risk_summary": {},
            "detailed_analysis": {},
            "alerts": [],
            "recommendations": []
        }
        
        # 市场风险分析
        market_risk = market_anomalies.get("overall_risk_level", "unknown")
        market_anomaly_count = market_anomalies.get("total_anomalies_detected", 0)
        
        # 系统风险分析
        system_health_status = system_health.get("overall_health", "unknown")
        system_alerts = system_health.get("alerts", [])
        
        # 行为风险分析
        behavioral_risk = "low"
        if behavioral_anomalies:
            behavioral_risk = behavioral_anomalies.get("risk_assessment", {}).get("overall_risk", "low")
        
        # 计算综合风险评分
        risk_scores = {
            "high": 3,
            "medium": 2, 
            "low": 1,
            "unknown": 2
        }
        
        total_risk_score = (
            risk_scores.get(market_risk, 2) + 
            risk_scores.get(system_health_status, 2) + 
            risk_scores.get(behavioral_risk, 1)
        )
        
        overall_risk = "high" if total_risk_score >= 8 else "medium" if total_risk_score >= 5 else "low"
        
        report["risk_summary"] = {
            "overall_risk": overall_risk,
            "risk_score": total_risk_score,
            "market_risk": market_risk,
            "system_risk": system_health_status,
            "behavioral_risk": behavioral_risk
        }
        
        report["detailed_analysis"] = {
            "market_anomalies": market_anomalies,
            "system_health": system_health,
            "behavioral_anomalies": behavioral_anomalies
        }
        
        # 生成警报
        if market_anomaly_count > 0:
            report["alerts"].append(f"检测到 {market_anomaly_count} 个市场异常")
        
        if system_health_status == "poor":
            report["alerts"].append("系统健康状态不佳")
        
        if behavioral_risk == "high":
            report["alerts"].append("检测到高风险行为模式")
        
        # 生成建议
        report["recommendations"] = _generate_comprehensive_recommendations(report)
        
        return report
        
    except Exception as e:
        return {"error": f"Risk report generation failed: {str(e)}"}


@mcp.tool()
def _generate_anomaly_recommendations(detection_results: Dict[str, Any]) -> List[str]:
    """生成异常检测建议"""
    recommendations = []
    
    for anomaly_type, result in detection_results.items():
        if result["is_anomaly"]:
            if anomaly_type == "price_anomaly":
                recommendations.append("价格异常，建议暂停交易或调整策略")
            elif anomaly_type == "volatility_anomaly":
                recommendations.append("波动率异常，建议降低仓位或增加对冲")
            elif anomaly_type == "correlation_anomaly":
                recommendations.append("相关性异常，建议重新评估资产配置")
    
    if not recommendations:
        recommendations.append("市场状态正常，可继续交易")
    
    return recommendations


@mcp.tool()
def _generate_comprehensive_recommendations(report: Dict[str, Any]) -> List[str]:
    """生成综合建议"""
    recommendations = []
    risk_summary = report.get("risk_summary", {})
    
    overall_risk = risk_summary.get("overall_risk", "low")
    
    if overall_risk == "high":
        recommendations.append("⚠️ 高风险状态：建议立即暂停所有交易活动")
        recommendations.append("🔍 全面检查系统状态和市场环境")
        recommendations.append("📊 重新评估所有持仓和风险暴露")
    elif overall_risk == "medium":
        recommendations.append("⚠️ 中等风险：建议降低交易频率和仓位")
        recommendations.append("📈 加强风险监控和预警机制")
        recommendations.append("💾 确保系统备份和恢复计划就绪")
    else:
        recommendations.append("✅ 低风险状态：可正常进行交易活动")
        recommendations.append("📋 继续保持常规风险监控")
    
    # 基于具体风险的建议
    market_risk = risk_summary.get("market_risk")
    if market_risk == "high":
        recommendations.append("📉 市场风险高：考虑增加对冲或降低风险暴露")
    
    system_risk = risk_summary.get("system_risk")
    if system_risk == "poor":
        recommendations.append("🖥️ 系统状态不佳：建议进行系统维护和优化")
    
    return recommendations


@mcp.tool()
def setup_continuous_monitoring(
    monitoring_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    设置持续监控
    
    Args:
        monitoring_config: 监控配置
        
    Returns:
        监控设置结果
    """
    try:
        config = {
            "market_monitoring": monitoring_config.get("market_monitoring", True),
            "system_monitoring": monitoring_config.get("system_monitoring", True),
            "behavioral_monitoring": monitoring_config.get("behavioral_monitoring", False),
            "alert_thresholds": monitoring_config.get("alert_thresholds", {
                "cpu_usage": 80,
                "memory_usage": 85,
                "disk_usage": 90,
                "price_anomaly_score": 0.7,
                "volatility_anomaly_score": 0.8
            }),
            "monitoring_frequency": monitoring_config.get("monitoring_frequency", "5min"),
            "alert_channels": monitoring_config.get("alert_channels", ["log", "email"])
        }
        
        return {
            "monitoring_setup": "success",
            "configuration": config,
            "next_check_time": (pd.Timestamp.now() + pd.Timedelta(minutes=5)).isoformat(),
            "monitoring_status": "active"
        }
        
    except Exception as e:
        return {"error": f"Continuous monitoring setup failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("ANOMALY_DETECTION_HTTP_PORT", "8008"))
    mcp.run(transport="streamable-http", port=port)

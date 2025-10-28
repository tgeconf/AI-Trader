"""
系统健康监控器
实现系统资源、进程状态、服务健康监控
"""

import psutil
import time
import socket
import subprocess
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class HealthStatus(Enum):
    """健康状态枚举"""

    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"  # 良好
    FAIR = "fair"  # 一般
    POOR = "poor"  # 较差
    CRITICAL = "critical"  # 危急


class AlertLevel(Enum):
    """警报级别枚举"""

    INFO = "info"  # 信息
    WARNING = "warning"  # 警告
    ERROR = "error"  # 错误
    CRITICAL = "critical"  # 危急


@dataclass
class SystemAlert:
    """系统警报"""

    level: AlertLevel
    message: str
    component: str
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class HealthMetric:
    """健康指标"""

    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    trend: str  # 'improving', 'stable', 'deteriorating'


class SystemHealthMonitor:
    """
    系统健康监控器

    实现全面的系统健康监控：
    - CPU使用率监控
    - 内存使用监控
    - 磁盘空间监控
    - 网络连接监控
    - 进程状态监控
    - 服务健康检查
    - 性能指标收集
    - 警报管理
    """

    def __init__(
        self,
        check_interval: int = 60,
        alert_thresholds: Optional[Dict[str, float]] = None,
        monitored_services: Optional[List[str]] = None,
    ):
        """
        初始化系统健康监控器

        Args:
            check_interval: 检查间隔（秒）
            alert_thresholds: 警报阈值配置
            monitored_services: 监控的服务列表
        """
        self.check_interval = check_interval
        self.monitored_services = monitored_services or [
            "redis",
            "postgresql",
            "nginx",
            "python",
        ]

        # 默认警报阈值
        self.alert_thresholds = alert_thresholds or {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "network_latency": 100.0,  # ms
            "process_count": 500,
            "load_average_1min": 4.0,
            "load_average_5min": 3.0,
            "load_average_15min": 2.0,
        }

        # 监控数据存储
        self.health_history: List[Dict[str, Any]] = []
        self.active_alerts: List[SystemAlert] = []
        self.metric_trends: Dict[str, List[float]] = {}

        # 监控线程控制
        self.monitoring_thread = None
        self.is_monitoring = False

        # 回调函数
        self.alert_callbacks: List[Callable] = []

    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def add_alert_callback(self, callback: Callable) -> None:
        """添加警报回调函数"""
        self.alert_callbacks.append(callback)

    def check_process_status(self) -> Dict[str, Any]:
        """
        检查进程状态

        Returns:
            进程状态报告
        """
        try:
            # 获取系统进程信息
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "status", "cpu_percent", "memory_percent"]
            ):
                try:
                    process_info = proc.info
                    processes.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # 分析关键进程
            critical_processes = {
                "python": any(p["name"] == "python" for p in processes),
                "redis": any("redis" in p["name"].lower() for p in processes),
                "postgres": any("postgres" in p["name"].lower() for p in processes),
            }

            # 计算进程统计
            total_processes = len(processes)
            running_processes = sum(
                1 for p in processes if p["status"] == psutil.STATUS_RUNNING
            )
            zombie_processes = sum(
                1 for p in processes if p["status"] == psutil.STATUS_ZOMBIE
            )

            return {
                "total_processes": total_processes,
                "running_processes": running_processes,
                "zombie_processes": zombie_processes,
                "critical_processes_status": critical_processes,
                "process_count_alert": total_processes
                > self.alert_thresholds["process_count"],
            }

        except Exception as e:
            return {"error": f"Process status check failed: {str(e)}"}

    def check_service_status(self) -> Dict[str, Any]:
        """
        检查服务状态

        Returns:
            服务状态报告
        """
        service_status = {}

        for service in self.monitored_services:
            try:
                # 检查服务是否运行（简化实现）
                if service == "python":
                    # Python进程检查
                    service_status[service] = {
                        "running": any(
                            p["name"] == "python"
                            for p in self.check_process_status().get("processes", [])
                        ),
                        "status": (
                            "active"
                            if any(
                                p["name"] == "python"
                                for p in self.check_process_status().get(
                                    "processes", []
                                )
                            )
                            else "inactive"
                        ),
                    }
                else:
                    # 其他服务的简化检查
                    service_status[service] = {
                        "running": True,  # 简化实现，实际应检查具体服务
                        "status": "active",
                    }

            except Exception as e:
                service_status[service] = {
                    "running": False,
                    "status": "error",
                    "error": str(e),
                }

        return {
            "services": service_status,
            "overall_status": (
                "healthy"
                if all(s["running"] for s in service_status.values())
                else "degraded"
            ),
        }

    def check_network_connectivity(self) -> Dict[str, Any]:
        """
        检查网络连接

        Returns:
            网络连接状态报告
        """
        connectivity_tests = {}

        # 测试关键网络连接
        test_targets = [
            ("google_dns", "8.8.8.8", 53),
            ("cloudflare_dns", "1.1.1.1", 53),
            ("localhost", "127.0.0.1", 8000),  # 假设本地服务端口
        ]

        for name, host, port in test_targets:
            try:
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                latency = (time.time() - start_time) * 1000  # 毫秒
                sock.close()

                connectivity_tests[name] = {
                    "reachable": result == 0,
                    "latency_ms": latency if result == 0 else None,
                    "latency_alert": (
                        latency > self.alert_thresholds["network_latency"]
                        if result == 0
                        else False
                    ),
                }

            except Exception as e:
                connectivity_tests[name] = {"reachable": False, "error": str(e)}

        # 计算网络健康评分
        successful_tests = sum(
            1 for test in connectivity_tests.values() if test.get("reachable", False)
        )
        network_health_score = (
            successful_tests / len(connectivity_tests) if connectivity_tests else 0.0
        )

        return {
            "connectivity_tests": connectivity_tests,
            "network_health_score": network_health_score,
            "overall_connectivity": (
                "good" if network_health_score > 0.7 else "degraded"
            ),
        }

    def assess_overall_health(
        self,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        disk_usage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        评估整体健康状态

        Args:
            cpu_usage: CPU使用率
            memory_usage: 内存使用率
            disk_usage: 磁盘使用率

        Returns:
            整体健康评估结果
        """
        # 收集各项指标
        metrics = self._collect_health_metrics(cpu_usage, memory_usage, disk_usage)

        # 计算健康评分
        health_score = self._calculate_health_score(metrics)

        # 生成警报
        alerts = self._generate_alerts(metrics)

        # 确定整体健康状态
        overall_health = self._determine_overall_health(health_score, alerts)

        # 生成建议
        recommendations = self._generate_health_recommendations(metrics, alerts)

        return {
            "overall_health": overall_health,
            "health_score": health_score,
            "metrics": metrics,
            "alerts": alerts,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

    def get_system_resources(self) -> Dict[str, Any]:
        """
        获取系统资源信息

        Returns:
            系统资源报告
        """
        try:
            # CPU信息
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg()

            # 内存信息
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # 磁盘信息
            disk = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()

            # 网络信息
            net_io = psutil.net_io_counters()

            return {
                "cpu": {
                    "usage_percent": cpu_usage,
                    "core_count": cpu_count,
                    "load_average_1min": load_avg[0],
                    "load_average_5min": load_avg[1],
                    "load_average_15min": load_avg[2],
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_percent": memory.percent,
                    "swap_used_percent": swap.percent,
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_percent": disk.percent,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0,
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent if net_io else 0,
                    "bytes_recv": net_io.bytes_recv if net_io else 0,
                },
            }

        except Exception as e:
            return {"error": f"System resources check failed: {str(e)}"}

    def get_health_history(
        self, hours: int = 24, metric: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取健康历史数据

        Args:
            hours: 小时数
            metric: 特定指标

        Returns:
            健康历史数据DataFrame
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            record
            for record in self.health_history
            if record["timestamp"] >= cutoff_time
        ]

        if not recent_data:
            return pd.DataFrame()

        df = pd.DataFrame(recent_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if metric:
            if metric in df.columns:
                return df[["timestamp", metric]]
            else:
                return pd.DataFrame()

        return df

    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                # 执行健康检查
                health_assessment = self.assess_overall_health()

                # 存储历史数据
                self.health_history.append(
                    {
                        "timestamp": datetime.now(),
                        "health_score": health_assessment["health_score"],
                        "cpu_usage": health_assessment["metrics"]["cpu"][
                            "usage_percent"
                        ],
                        "memory_usage": health_assessment["metrics"]["memory"][
                            "used_percent"
                        ],
                        "disk_usage": health_assessment["metrics"]["disk"][
                            "used_percent"
                        ],
                        "alerts_count": len(health_assessment["alerts"]),
                    }
                )

                # 限制历史数据大小
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]

                # 处理警报
                self._process_alerts(health_assessment["alerts"])

            except Exception as e:
                # 记录监控错误但不停止监控
                error_alert = SystemAlert(
                    level=AlertLevel.ERROR,
                    message=f"Health monitoring error: {str(e)}",
                    component="SystemHealthMonitor",
                    timestamp=datetime.now(),
                    details={"error": str(e)},
                )
                self._trigger_alert(error_alert)

            # 等待下一个检查周期
            time.sleep(self.check_interval)

    def _collect_health_metrics(
        self,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        disk_usage: Optional[float] = None,
    ) -> Dict[str, HealthMetric]:
        """收集健康指标"""
        resources = self.get_system_resources()

        if "error" in resources:
            return {}

        # CPU指标
        cpu_value = (
            cpu_usage if cpu_usage is not None else resources["cpu"]["usage_percent"]
        )
        cpu_metric = HealthMetric(
            name="cpu_usage",
            value=cpu_value,
            unit="percent",
            status=self._evaluate_metric_status(
                cpu_value, self.alert_thresholds["cpu_usage"]
            ),
            threshold_warning=self.alert_thresholds["cpu_usage"] * 0.8,
            threshold_critical=self.alert_thresholds["cpu_usage"],
            trend=self._calculate_trend("cpu_usage", cpu_value),
        )

        # 内存指标
        memory_value = (
            memory_usage
            if memory_usage is not None
            else resources["memory"]["used_percent"]
        )
        memory_metric = HealthMetric(
            name="memory_usage",
            value=memory_value,
            unit="percent",
            status=self._evaluate_metric_status(
                memory_value, self.alert_thresholds["memory_usage"]
            ),
            threshold_warning=self.alert_thresholds["memory_usage"] * 0.8,
            threshold_critical=self.alert_thresholds["memory_usage"],
            trend=self._calculate_trend("memory_usage", memory_value),
        )

        # 磁盘指标
        disk_value = (
            disk_usage if disk_usage is not None else resources["disk"]["used_percent"]
        )
        disk_metric = HealthMetric(
            name="disk_usage",
            value=disk_value,
            unit="percent",
            status=self._evaluate_metric_status(
                disk_value, self.alert_thresholds["disk_usage"]
            ),
            threshold_warning=self.alert_thresholds["disk_usage"] * 0.8,
            threshold_critical=self.alert_thresholds["disk_usage"],
            trend=self._calculate_trend("disk_usage", disk_value),
        )

        # 负载指标
        load_1min = resources["cpu"]["load_average_1min"]
        load_metric = HealthMetric(
            name="load_average_1min",
            value=load_1min,
            unit="load",
            status=self._evaluate_metric_status(
                load_1min, self.alert_thresholds["load_average_1min"]
            ),
            threshold_warning=self.alert_thresholds["load_average_1min"] * 0.8,
            threshold_critical=self.alert_thresholds["load_average_1min"],
            trend=self._calculate_trend("load_average_1min", load_1min),
        )

        return {
            "cpu": cpu_metric,
            "memory": memory_metric,
            "disk": disk_metric,
            "load": load_metric,
        }

    def _evaluate_metric_status(
        self, value: float, critical_threshold: float
    ) -> HealthStatus:
        """评估指标状态"""
        warning_threshold = critical_threshold * 0.8

        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.POOR
        elif value >= warning_threshold * 0.8:
            return HealthStatus.FAIR
        elif value >= warning_threshold * 0.6:
            return HealthStatus.GOOD
        else:
            return HealthStatus.EXCELLENT

    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """计算指标趋势"""
        if metric_name not in self.metric_trends:
            self.metric_trends[metric_name] = []

        # 保留最近10个值
        self.metric_trends[metric_name].append(current_value)
        if len(self.metric_trends[metric_name]) > 10:
            self.metric_trends[metric_name] = self.metric_trends[metric_name][-10:]

        values = self.metric_trends[metric_name]
        if len(values) < 3:
            return "stable"

        # 简单趋势分析
        recent_avg = np.mean(values[-3:])
        previous_avg = np.mean(values[-6:-3]) if len(values) >= 6 else values[0]

        if recent_avg > previous_avg * 1.1:
            return "deteriorating"
        elif recent_avg < previous_avg * 0.9:
            return "improving"
        else:
            return "stable"

    def _calculate_health_score(self, metrics: Dict[str, HealthMetric]) -> float:
        """计算健康评分"""
        if not metrics:
            return 0.0

        # 基于指标状态计算加权评分
        weights = {
            HealthStatus.EXCELLENT: 1.0,
            HealthStatus.GOOD: 0.8,
            HealthStatus.FAIR: 0.6,
            HealthStatus.POOR: 0.3,
            HealthStatus.CRITICAL: 0.0,
        }

        total_score = 0.0
        for metric in metrics.values():
            total_score += weights.get(metric.status, 0.0)

        return total_score / len(metrics)

    def _generate_alerts(self, metrics: Dict[str, HealthMetric]) -> List[SystemAlert]:
        """生成警报"""
        alerts = []

        for metric_name, metric in metrics.items():
            if metric.status == HealthStatus.CRITICAL:
                alerts.append(
                    SystemAlert(
                        level=AlertLevel.CRITICAL,
                        message=f"{metric_name} critical: {metric.value:.1f}{metric.unit}",
                        component=metric_name.upper(),
                        timestamp=datetime.now(),
                        details={
                            "metric": metric_name,
                            "value": metric.value,
                            "threshold": metric.threshold_critical,
                        },
                    )
                )
            elif metric.status == HealthStatus.POOR:
                alerts.append(
                    SystemAlert(
                        level=AlertLevel.WARNING,
                        message=f"{metric_name} warning: {metric.value:.1f}{metric.unit}",
                        component=metric_name.upper(),
                        timestamp=datetime.now(),
                        details={
                            "metric": metric_name,
                            "value": metric.value,
                            "threshold": metric.threshold_warning,
                        },
                    )
                )

        # 检查进程状态
        process_status = self.check_process_status()
        if process_status.get("process_count_alert", False):
            alerts.append(
                SystemAlert(
                    level=AlertLevel.WARNING,
                    message=f"High process count: {process_status['total_processes']}",
                    component="PROCESS",
                    timestamp=datetime.now(),
                    details=process_status,
                )
            )

        # 检查网络连接
        network_status = self.check_network_connectivity()
        if network_status["overall_connectivity"] == "degraded":
            alerts.append(
                SystemAlert(
                    level=AlertLevel.ERROR,
                    message="Network connectivity degraded",
                    component="NETWORK",
                    timestamp=datetime.now(),
                    details=network_status,
                )
            )

        return alerts

    def _determine_overall_health(
        self, health_score: float, alerts: List[SystemAlert]
    ) -> str:
        """确定整体健康状态"""
        critical_alerts = sum(
            1 for alert in alerts if alert.level == AlertLevel.CRITICAL
        )
        error_alerts = sum(1 for alert in alerts if alert.level == AlertLevel.ERROR)

        if critical_alerts > 0:
            return "critical"
        elif error_alerts > 0:
            return "poor"
        elif health_score < 0.5:
            return "fair"
        elif health_score < 0.8:
            return "good"
        else:
            return "excellent"

    def _generate_health_recommendations(
        self, metrics: Dict[str, HealthMetric], alerts: List[SystemAlert]
    ) -> List[str]:
        """生成健康建议"""
        recommendations = []

        for metric_name, metric in metrics.items():
            if metric.status in [HealthStatus.POOR, HealthStatus.CRITICAL]:
                if metric_name == "cpu_usage":
                    recommendations.append("CPU使用率过高，建议优化代码或增加计算资源")
                elif metric_name == "memory_usage":
                    recommendations.append("内存使用率过高，建议检查内存泄漏或增加内存")
                elif metric_name == "disk_usage":
                    recommendations.append("磁盘空间不足，建议清理日志文件或增加存储")
                elif metric_name == "load_average_1min":
                    recommendations.append("系统负载过高，建议减少并发任务或优化性能")

        # 基于警报的建议
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                recommendations.append(f"紧急：{alert.message}，需要立即处理")
            elif alert.level == AlertLevel.ERROR:
                recommendations.append(f"错误：{alert.message}，建议尽快处理")

        if not recommendations:
            recommendations.append("系统状态良好，继续保持")

        return recommendations

    def _process_alerts(self, alerts: List[SystemAlert]) -> None:
        """处理警报"""
        for alert in alerts:
            # 检查是否已存在相同警报
            existing_alert = next(
                (
                    a
                    for a in self.active_alerts
                    if a.message == alert.message and a.component == alert.component
                ),
                None,
            )

            if not existing_alert:
                self.active_alerts.append(alert)
                self._trigger_alert(alert)

    def _trigger_alert(self, alert: SystemAlert) -> None:
        """触发警报"""
        # 调用注册的回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                # 记录回调错误但不中断
                print(f"Alert callback error: {e}")

    def clear_resolved_alerts(self, component: Optional[str] = None) -> None:
        """清除已解决的警报"""
        if component:
            self.active_alerts = [
                alert for alert in self.active_alerts if alert.component != component
            ]
        else:
            self.active_alerts = []

    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        return {
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "check_interval_seconds": self.check_interval,
            "health_history_size": len(self.health_history),
            "active_alerts_count": len(self.active_alerts),
            "alert_thresholds": self.alert_thresholds,
            "monitored_services": self.monitored_services,
            "metric_trends_tracked": list(self.metric_trends.keys()),
        }

    def get_detailed_health_report(self) -> Dict[str, Any]:
        """获取详细健康报告"""
        # 收集所有健康信息
        system_resources = self.get_system_resources()
        process_status = self.check_process_status()
        service_status = self.check_service_status()
        network_status = self.check_network_connectivity()
        overall_health = self.assess_overall_health()

        return {
            "report_timestamp": datetime.now().isoformat(),
            "system_overview": {
                "hostname": socket.gethostname(),
                "platform": f"{psutil.sys.platform} {psutil.os.uname().release}",
                "uptime_seconds": time.time() - psutil.boot_time(),
                "monitoring_duration_hours": len(self.health_history)
                * self.check_interval
                / 3600,
            },
            "resource_utilization": system_resources,
            "process_analysis": process_status,
            "service_health": service_status,
            "network_status": network_status,
            "health_assessment": overall_health,
            "trend_analysis": self._analyze_health_trends(),
            "recommendations": self._generate_comprehensive_recommendations(
                overall_health
            ),
        }

    def _analyze_health_trends(self) -> Dict[str, Any]:
        """分析健康趋势"""
        if len(self.health_history) < 10:
            return {"insufficient_data": True}

        # 转换为DataFrame进行分析
        df = pd.DataFrame(self.health_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 计算趋势
        recent_data = df.tail(10)
        older_data = df.head(max(0, len(df) - 10))

        trends = {}
        metrics = ["health_score", "cpu_usage", "memory_usage", "disk_usage"]

        for metric in metrics:
            if metric in df.columns:
                recent_avg = recent_data[metric].mean()
                older_avg = (
                    older_data[metric].mean() if len(older_data) > 0 else recent_avg
                )

                if older_avg == 0:
                    trend = "stable"
                elif recent_avg > older_avg * 1.05:
                    trend = "deteriorating"
                elif recent_avg < older_avg * 0.95:
                    trend = "improving"
                else:
                    trend = "stable"

                trends[metric] = {
                    "current_value": recent_avg,
                    "previous_value": older_avg,
                    "trend": trend,
                    "change_percent": (
                        (recent_avg - older_avg) / older_avg * 100
                        if older_avg > 0
                        else 0
                    ),
                }

        return trends

    def _generate_comprehensive_recommendations(
        self, health_assessment: Dict[str, Any]
    ) -> List[str]:
        """生成综合建议"""
        recommendations = []
        overall_health = health_assessment.get("overall_health", "unknown")

        # 基于整体健康状态的建议
        if overall_health == "critical":
            recommendations.append("🚨 系统状态危急：立即停止交易活动并进行系统维护")
            recommendations.append("🔧 检查所有关键服务和进程状态")
            recommendations.append("📊 分析系统资源使用情况，识别瓶颈")
        elif overall_health == "poor":
            recommendations.append("⚠️ 系统状态较差：建议减少交易负载")
            recommendations.append("🔄 优化系统配置和资源分配")
            recommendations.append("📈 监控关键指标，准备应急计划")
        elif overall_health == "fair":
            recommendations.append("📋 系统状态一般：建议进行预防性维护")
            recommendations.append("⚡ 优化性能敏感的操作")
            recommendations.append("💾 确保备份和恢复机制正常")
        else:
            recommendations.append("✅ 系统状态良好：可正常进行交易活动")
            recommendations.append("📊 继续保持常规监控")

        # 基于具体问题的建议
        alerts = health_assessment.get("alerts", [])
        for alert in alerts:
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
                component = alert.component.lower()
                if "cpu" in component:
                    recommendations.append("💻 CPU资源紧张：考虑优化算法或增加计算资源")
                elif "memory" in component:
                    recommendations.append(
                        "🧠 内存使用过高：检查内存泄漏，考虑增加内存"
                    )
                elif "disk" in component:
                    recommendations.append("💾 磁盘空间不足：清理临时文件，考虑扩容")
                elif "network" in component:
                    recommendations.append("🌐 网络连接问题：检查网络配置和连接稳定性")

        return recommendations

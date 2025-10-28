"""
策略有效性分析器
实现策略表现的全面评估和统计检验
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class StrategyQuality(Enum):
    """策略质量等级枚举"""
    EXCELLENT = "excellent"    # 优秀
    GOOD = "good"             # 良好
    FAIR = "fair"             # 一般
    POOR = "poor"             # 较差
    VERY_POOR = "very_poor"   # 很差


class StatisticalTest(Enum):
    """统计检验枚举"""
    T_TEST = "t_test"                    # T检验
    SHARPE_TEST = "sharpe_test"          # 夏普比率检验
    INFORMATION_RATIO_TEST = "information_ratio_test"  # 信息比率检验
    NORMALITY_TEST = "normality_test"    # 正态性检验
    STATIONARITY_TEST = "stationarity_test"  # 平稳性检验


@dataclass
class StrategyEffectivenessResult:
    """策略有效性分析结果"""
    # 绩效指标
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float
    alpha: float
    beta: float
    
    # 风险指标
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    
    # 统计检验
    statistical_tests: Dict[str, Dict[str, Any]]
    
    # 质量评估
    strategy_quality: StrategyQuality


class StrategyEffectivenessAnalyzer:
        """
    策略有效性分析器
    
    实现策略表现的全面评估和统计检验：
    - 绩效指标计算
    - 风险指标评估
    - 统计显著性检验
    - 策略质量评级
    - 稳健性分析
    """
    
    def __init__(self,
                 significance_level: float = 0.05,
                 var_confidence: float = 0.95,
                 min_observations: int = 30):
        """
        初始化策略有效性分析器
        
        Args:
            significance_level: 显著性水平
            var_confidence: VaR置信水平
            min_observations: 最小观测数
        """
        self.significance_level = significance_level
        self.var_confidence = var_confidence
        self.min_observations = min_observations
        self.analysis_history = {}
        
    def analyze_strategy(self,
                        strategy_returns: np.ndarray,
                        benchmark_returns: np.ndarray,
                        risk_free_rate: float = 0.02,
                        period: str = "monthly") -> StrategyEffectivenessResult:
        """
        分析策略有效性
        
        Args:
            strategy_returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            risk_free_rate: 无风险利率
            period: 收益率周期
            
        Returns:
            策略有效性分析结果
        """
        # 数据验证
        self._validate_input_data(strategy_returns, benchmark_returns)
        
        # 计算绩效指标
        performance_metrics = self._calculate_performance_metrics(
            strategy_returns, benchmark_returns, risk_free_rate, period
        )
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(strategy_returns)
        
        # 执行统计检验
        statistical_tests = self._perform_statistical_tests(
            strategy_returns, benchmark_returns, risk_free_rate
        )
        
        # 评估策略质量
        strategy_quality = self._assess_strategy_quality(
            performance_metrics, risk_metrics, statistical_tests
        )
        
        return StrategyEffectivenessResult(
            # 绩效指标
            total_return=performance_metrics['total_return'],
            annualized_return=performance_metrics['annualized_return'],
            volatility=performance_metrics['volatility'],
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            max_drawdown=performance_metrics['max_drawdown'],
            calmar_ratio=performance_metrics['calmar_ratio'],
            information_ratio=performance_metrics['information_ratio'],
            alpha=performance_metrics['alpha'],
            beta=performance_metrics['beta'],
            
            # 风险指标
            var_95=risk_metrics['var_95'],
            cvar_95=risk_metrics['cvar_95'],
            skewness=risk_metrics['skewness'],
            kurtosis=risk_metrics['kurtosis'],
            
            # 统计检验
            statistical_tests=statistical_tests,
            
            # 质量评估
            strategy_quality=strategy_quality
        )
    
    def _validate_input_data(self, 
                           strategy_returns: np.ndarray, 
                           benchmark_returns: np.ndarray) -> None:
        """验证输入数据"""
        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Strategy and benchmark returns must have same length")
        
        if len(strategy_returns) < self.min_observations:
            raise ValueError(f"Insufficient observations: {len(strategy_returns)} < {self.min_observations}")
        
        if np.any(np.isnan(strategy_returns)) or np.any(np.isnan(benchmark_returns)):
            raise ValueError("Input data contains NaN values")
    
    def _calculate_performance_metrics(self,
                                    strategy_returns: np.ndarray,
                                    benchmark_returns: np.ndarray,
                                    risk_free_rate: float,
                                    period: str) -> Dict[str, float]:
        """计算绩效指标"""
        # 周期调整因子
        period_factor = self._get_period_factor(period)
        
        # 总收益率
        total_return = np.prod(1 + strategy_returns) - 1
        
        # 年化收益率
        n_periods = len(strategy_returns)
        annualized_return = (1 + total_return) ** (period_factor / n_periods) - 1
        
        # 年化波动率
        volatility = np.std(strategy_returns) * np.sqrt(period_factor)
        
        # 夏普比率
        excess_returns = strategy_returns - risk_free_rate / period_factor
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(period_factor)
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(strategy_returns)
        
        # Calmar比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 信息比率
        active_returns = strategy_returns - benchmark_returns
        information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(period_factor)
        
        # Alpha和Beta
        alpha, beta = self._calculate_alpha_beta(strategy_returns, benchmark_returns, risk_free_rate, period)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'alpha': alpha,
            'beta': beta
        }
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """计算风险指标"""
        # VaR (历史模拟法)
        var_95 = np.percentile(returns, (1 - self.var_confidence) * 100)
        
        # CVaR (条件VaR)
        cvar_95 = returns[returns <= var_95].mean()
        
        # 偏度和峰度
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _perform_statistical_tests(self,
                                 strategy_returns: np.ndarray,
                                 benchmark_returns: np.ndarray,
                                 risk_free_rate: float) -> Dict[str, Dict[str, Any]]:
        """执行统计检验"""
        tests = {}
        
        # T检验：检验策略收益是否显著大于0
        t_stat, t_pvalue = stats.ttest_1samp(strategy_returns, 0)
        tests['t_test'] = {
            'statistic': t_stat,
            'p_value': t_pvalue,
            'significant': t_pvalue < self.significance_level,
            'null_hypothesis': '策略收益均值为0'
        }
        
        # 夏普比率检验
        sharpe_test = self._test_sharpe_ratio(strategy_returns, risk_free_rate)
        tests['sharpe_test'] = sharpe_test
        
        # 信息比率检验
        ir_test = self._test_information_ratio(strategy_returns, benchmark_returns)
        tests['information_ratio_test'] = ir_test
        
        # 正态性检验
        normality_test = self._test_normality(strategy_returns)
        tests['normality_test'] = normality_test
        
        # 平稳性检验 (ADF检验)
        stationarity_test = self._test_stationarity(strategy_returns)
        tests['stationarity_test'] = stationarity_test
        
        return tests
    
    def _test_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float) -> Dict[str, Any]:
        """检验夏普比率显著性"""
        excess_returns = returns - risk_free_rate / 252  # 假设日度数据
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        # Jobson-Korkie检验
        n = len(returns)
        if n > 2:
            test_stat = sharpe_ratio * np.sqrt(n)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        else:
            test_stat = 0
            p_value = 1
        
        return {
            'statistic': test_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'null_hypothesis': '夏普比率为0',
            'sharpe_ratio': sharpe_ratio
        }
    
    def _test_information_ratio(self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict[str, Any]:
        """检验信息比率显著性"""
        active_returns = strategy_returns - benchmark_returns
        information_ratio = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
        
        n = len(active_returns)
        if n > 2:
            test_stat = information_ratio * np.sqrt(n)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        else:
            test_stat = 0
            p_value = 1
        
        return {
            'statistic': test_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'null_hypothesis': '信息比率为0',
            'information_ratio': information_ratio
        }
    
    def _test_normality(self, returns: np.ndarray) -> Dict[str, Any]:
        """正态性检验"""
        # Jarque-Bera检验
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Shapiro-Wilk检验（适用于小样本）
        if len(returns) < 5000:
            sw_stat, sw_pvalue = stats.shapiro(returns)
        else:
            sw_stat, sw_pvalue = np.nan, np.nan
        
        return {
            'jarque_bera_statistic': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'shapiro_wilk_statistic': sw_stat,
            'shapiro_wilk_pvalue': sw_pvalue,
            'normal_distributed': jb_pvalue > self.significance_level,
            'null_hypothesis': '收益率服从正态分布'
        }
    
    def _test_stationarity(self, returns: np.ndarray) -> Dict[str, Any]:
        """平稳性检验"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Augmented Dickey-Fuller检验
            adf_result = adfuller(returns)
            adf_stat, adf_pvalue = adf_result[0], adf_result[1]
            
            return {
                'adf_statistic': adf_stat,
                'adf_pvalue': adf_pvalue,
                'stationary': adf_pvalue < self.significance_level,
                'null_hypothesis': '序列存在单位根（非平稳）'
            }
        except ImportError:
            return {
                'adf_statistic': np.nan,
                'adf_pvalue': np.nan,
                'stationary': True,  # 假设平稳
                'null_hypothesis': '无法检验（需要statsmodels）'
            }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return np.min(drawdowns)
    
    def _calculate_alpha_beta(self,
                            strategy_returns: np.ndarray,
                            benchmark_returns: np.ndarray,
                            risk_free_rate: float,
                            period: str) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        period_factor = self._get_period_factor(period)
        
        # 超额收益
        strategy_excess = strategy_returns - risk_free_rate / period_factor
        benchmark_excess = benchmark_returns - risk_free_rate / period_factor
        
        # 线性回归
        if len(strategy_excess) > 1:
            beta = np.cov(strategy_excess, benchmark_excess)[0, 1] / np.var(benchmark_excess)
            alpha = np.mean(strategy_excess) - beta * np.mean(benchmark_excess)
        else:
            alpha, beta = 0.0, 1.0
        
        return alpha, beta
    
    def _get_period_factor(self, period: str) -> float:
        """获取周期调整因子"""
        factors = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        return factors.get(period.lower(), 252)  # 默认日度
    
    def _assess_strategy_quality(self,
                               performance_metrics: Dict[str, float],
                               risk_metrics: Dict[str, float],
                               statistical_tests: Dict[str, Dict[str, Any]]) -> StrategyQuality:
        """评估策略质量"""
        score = 0
        
        # 夏普比率评分
        sharpe_ratio = performance_metrics['sharpe_ratio']
        if sharpe_ratio > 1.5:
            score += 3
        elif sharpe_ratio > 1.0:
            score += 2
        elif sharpe_ratio > 0.5:
            score += 1
        elif sharpe_ratio < 0:
            score -= 2
        
        # 最大回撤评分
        max_drawdown = abs(performance_metrics['max_drawdown'])
        if max_drawdown < 0.1:
            score += 3
        elif max_drawdown < 0.2:
            score += 2
        elif max_drawdown < 0.3:
            score += 1
        elif max_drawdown > 0.5:
            score -= 2
        
        # 信息比率评分
        information_ratio = performance_metrics['information_ratio']
        if information_ratio > 0.5:
            score += 2
        elif information_ratio > 0.2:
            score += 1
        elif information_ratio < 0:
            score -= 1
        
        # Alpha评分
        alpha = performance_metrics['alpha']
        if alpha > 0.05:
            score += 2
        elif alpha > 0.02:
            score += 1
        elif alpha < -0.05:
            score -= 2
        
        # 统计显著性评分
        t_test_sig = statistical_tests['t_test']['significant']
        sharpe_test_sig = statistical_tests['sharpe_test']['significant']
        
        if t_test_sig and sharpe_test_sig:
            score += 2
        elif t_test_sig or sharpe_test_sig:
            score += 1
        
        # 风险指标评分
        skewness = risk_metrics['skewness']
        if skewness > 0:  # 正偏态（右偏）通常更好
            score += 1
        
        # 确定质量等级
        if score >= 8:
            return StrategyQuality.EXCELLENT
        elif score >= 6:
            return StrategyQuality.GOOD
        elif score >= 4:
            return StrategyQuality.FAIR
        elif score >= 2:
            return StrategyQuality.POOR
        else:
            return StrategyQuality.VERY_POOR
    
    def rolling_analysis(self,
                        strategy_returns: np.ndarray,
                        benchmark_returns: np.ndarray,
                        window: int = 60,
                        step: int = 1) -> Dict[str, Any]:
        """
        滚动分析策略表现
        
        Args:
            strategy_returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            window: 滚动窗口大小
            step: 滚动步长
            
        Returns:
            滚动分析结果
        """
        n_obs = len(strategy_returns)
        if n_obs < window:
            raise ValueError(f"Insufficient data for rolling analysis: {n_obs} < {window}")
        
        rolling_results = {
            'sharpe_ratio': [],
            'max_drawdown': [],
            'information_ratio': [],
            'alpha': [],
            'beta': [],
            'strategy_quality': [],
            'dates': []  # 假设有时间索引
        }
        
        for i in range(0, n_obs - window + 1, step):
            # 提取窗口数据
            strategy_window = strategy_returns[i:i+window]
            benchmark_window = benchmark_returns[i:i+window]
            
            try:
                # 分析当前窗口
                result = self.analyze_strategy(strategy_window, benchmark_window)
                
                # 存储结果
                rolling_results['sharpe_ratio'].append(result.sharpe_ratio)
                rolling_results['max_drawdown'].append(result.max_drawdown)
                rolling_results['information_ratio'].append(result.information_ratio)
                rolling_results['alpha'].append(result.alpha)
                rolling_results['beta'].append(result.beta)
                rolling_results['strategy_quality'].append(result.strategy_quality.value)
                
            except Exception as e:
                # 处理分析失败的情况
                rolling_results['sharpe_ratio'].append(np.nan)
                rolling_results['max_drawdown'].append(np.nan)
                rolling_results['information_ratio'].append(np.nan)
                rolling_results['alpha'].append(np.nan)
                rolling_results['beta'].append(np.nan)
                rolling_results['strategy_quality'].append('unknown')
        
        return rolling_results
    
    def compare_strategies(self,
                         strategies_data: Dict[str, np.ndarray],
                         benchmark_returns: np.ndarray) -> Dict[str, Any]:
        """
        比较多个策略
        
        Args:
            strategies_data: 多个策略的收益率数据
            benchmark_returns: 基准收益率
            
        Returns:
            策略比较结果
        """
        comparison_results = {}
        
        for strategy_name, strategy_returns in strategies_data.items():
            try:
                result = self.analyze_strategy(strategy_returns, benchmark_returns)
                comparison_results[strategy_name] = {
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'information_ratio': result.information_ratio,
                    'alpha': result.alpha,
                    'strategy_quality': result.strategy_quality.value,
                    'total_return': result.total_return
                }
            except Exception as e:
                comparison_results[strategy_name] = {'error': str(e)}
        
        # 排名分析
        rankings = {}
        metrics = ['sharpe_ratio', 'information_ratio', 'alpha']
        
        for metric in metrics:
            valid_strategies = {
                name: data[metric] 
                for name, data in comparison_results.items() 
                if 'error' not in data and not np.isnan(data[metric])
            }
            
            if valid_strategies:
                sorted_strategies = sorted(valid_strategies.items(), key=lambda x: x[1], reverse=True)
                rankings[metric] = {strategy: rank + 1 for rank, (strategy, _) in enumerate(sorted_strategies)}
            else:
                rankings[metric] = {}
        
        # 综合排名
        overall_rankings = {}
        for strategy_name in strategies_data.keys():
            if strategy_name in comparison_results and 'error' not in comparison_results[strategy_name]:
                total_rank = sum(
                    rankings[metric].get(strategy_name, len(strategies_data) + 1) 
                    for metric in metrics
                )
                overall_rankings[strategy_name] = total_rank
        
        sorted_overall = sorted(overall_rankings.items(), key=lambda x: x[1])
        
        return {
            'comparison_results': comparison_results,
            'metric_rankings': rankings,
            'overall_rankings': {strategy: rank + 1 for rank, (strategy, _) in enumerate(sorted_overall)},
            'best_strategy': sorted_overall[0][0] if sorted_overall else 'none'
        }
    
    def generate_strategy_report(self,
                              result: StrategyEffectivenessResult,
                              strategy_name: str = "Unnamed Strategy") -> Dict[str, Any]:
        """
        生成策略分析报告
        
        Args:
            result: 策略有效性分析结果
            strategy_name: 策略名称
            
        Returns:
            策略分析报告
        """
        report = {
            'strategy_name': strategy_name,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'strategy_quality': result.strategy_quality.value,
            'performance_summary': {
                'total_return': f"{result.total_return:.2%}",
                'annualized_return': f"{result.annualized_return:.2%}",
                'volatility': f"{result.volatility:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.3f}",
                'max_drawdown': f"{result.max_drawdown:.2%}",
                'calmar_ratio': f"{result.calmar_ratio:.3f}",
                'information_ratio': f"{result.information_ratio:.3f}",
                'alpha': f"{result.alpha:.4f}",
                'beta': f"{result.beta:.3f}"
            },
            'risk_metrics': {
                'var_95': f"{result.var_95:.2%}",
                'cvar_95': f"{result.cvar_95:.2%}",
                'skewness': f"{result.skewness:.3f}",
                'kurtosis': f"{result.kurtosis:.3f}"
            },
            'statistical_tests': {},
            'recommendations': self._generate_strategy_recommendations(result)
        }
        
        # 格式化统计检验结果
        for test_name, test_result in result.statistical_tests.items():
            report['statistical_tests'][test_name] = {
                'significant': test_result['significant'],
                'p_value': f"{test_result['p_value']:.4f}",
                'null_hypothesis': test_result['null_hypothesis']
            }
        
        return report
    
    def _generate_strategy_recommendations(self, result: StrategyEffectivenessResult) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        # 基于夏普比率的建议
        if result.sharpe_ratio < 0.5:
            recommendations.append("夏普比率较低，建议优化风险调整后收益")
        elif result.sharpe_ratio > 1.5:
            recommendations.append("夏普比率优秀，继续保持当前策略")
        
        # 基于最大回撤的建议
        if abs(result.max_drawdown) > 0.2:
            recommendations.append("最大回撤较大，建议加强风险管理")
        
        # 基于信息比率的建议
        if result.information_ratio < 0:
            recommendations.append("信息比率为负，策略未能跑赢基准")
        elif result.information_ratio > 0.5:
            recommendations.append("信息比率良好，策略具有超额收益能力")
        
        # 基于Alpha的建议
        if result.alpha < -0.02:
            recommendations.append("Alpha为负，策略未能创造超额收益")
        elif result.alpha > 0.05:
            recommendations.append("Alpha显著为正，策略具有选股能力")
        
        # 基于统计检验的建议
        t_test_sig = result.statistical_tests['t_test']['significant']
        if not t_test_sig:
            recommendations.append("策略收益统计不显著，建议延长观察期")
        
        # 基于偏度的建议
        if result.skewness < -0.5:
            recommendations.append("收益率左偏，存在极端亏损风险")
        
        # 基于策略质量的建议
        if result.strategy_quality in [StrategyQuality.POOR, StrategyQuality.VERY_POOR]:
            recommendations.append("策略质量较差，建议重新设计或优化")
        elif result.strategy_quality == StrategyQuality.EXCELLENT:
            recommendations.append("策略质量优秀，可考虑增加资金配置")
        
        # 默认建议
        if not recommendations:
            recommendations.append("策略表现良好，建议持续监控和优化")
        
        return recommendations
    
    def update_analysis_history(self, strategy_name: str, result: StrategyEffectivenessResult) -> None:
        """
        更新分析历史
        
        Args:
            strategy_name: 策略名称
            result: 分析结果
        """
        if strategy_name not in self.analysis_history:
            self.analysis_history[strategy_name] = []
        
        self.analysis_history[strategy_name].append({
            'timestamp': pd.Timestamp.now(),
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'information_ratio': result.information_ratio,
            'alpha': result.alpha,
            'strategy_quality': result.strategy_quality.value
        })
        
        # 只保留最近50条记录
        if len(self.analysis_history[strategy_name]) > 50:
            self.analysis_history[strategy_name] = self.analysis_history[strategy_name][-50:]
    
    def analyze_strategy_evolution(self, strategy_name: str) -> Dict[str, Any]:
        """
        分析策略演变
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略演变分析
        """
        if strategy_name not in self.analysis_history:
            return {"error": "No analysis history for strategy"}
        
        history = self.analysis_history[strategy_name]
        if len(history) < 2:
            return {"error": "Insufficient history for evolution analysis"}
        
        # 转换为DataFrame
        df = pd.DataFrame(history)
        df.set_index('timestamp', inplace=True)
        
        # 计算变化趋势
        trends = {}
        metrics = ['sharpe_ratio', 'max_drawdown', 'information_ratio', 'alpha']
        
        for metric in metrics:
            values = df[metric].values
            if len(values) > 1:
                # 计算斜率（简单线性趋势）
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                trends[metric] = {
                    'slope': slope,
                    'trend': 'improving' if slope > 0 else 'deteriorating' if slope < 0 else 'stable',
                    'current_value': values[-1],
                    'initial_value': values[0],
                    'change': values[-1] - values[0]
                }
        
        # 质量变化分析
        quality_counts = df['strategy_quality'].value_counts()
        quality_trend = 'stable'
        if len(quality_counts) > 1:
            # 检查质量是否改善
            quality_mapping = {
                'very_poor': 1, 'poor': 2, 'fair': 3, 'good': 4, 'excellent': 5
            }
            initial_quality = quality_mapping.get(df['strategy_quality'].iloc[0], 3)
            current_quality = quality_mapping.get(df['strategy_quality'].iloc[-1], 3)
            if current_quality > initial_quality:
                quality_trend = 'improving'
            elif current_quality < initial_quality:
                quality_trend = 'deteriorating'
        
        return {
            'strategy_name': strategy_name,
            'analysis_period': {
                'start_date': df.index[0].strftime('%Y-%m-%d'),
                'end_date': df.index[-1].strftime('%Y-%m-%d'),
                'total_analyses': len(df)
            },
            'metric_trends': trends,
            'quality_trend': quality_trend,
            'current_quality': df['strategy_quality'].iloc[-1],
            'recommendations': self._generate_evolution_recommendations(trends, quality_trend)
        }
    
    def _generate_evolution_recommendations(self, trends: Dict[str, Any], quality_trend: str) -> List[str]:
        """生成演变分析建议"""
        recommendations = []
        
        # 基于趋势的建议
        sharpe_trend = trends.get('sharpe_ratio', {}).get('trend', 'stable')
        if sharpe_trend == 'deteriorating':
            recommendations.append("夏普比率呈下降趋势，建议检查策略适应性")
        
        max_dd_trend = trends.get('max_drawdown', {}).get('trend', 'stable')
        if max_dd_trend == 'deteriorating':
            recommendations.append("最大回撤呈恶化趋势，建议加强风险控制")
        
        alpha_trend = trends.get('alpha', {}).get('trend', 'stable')
        if alpha_trend == 'deteriorating':
            recommendations.append("Alpha呈下降趋势，建议优化选股能力")
        
        # 基于质量趋势的建议
        if quality_trend == 'deteriorating':
            recommendations.append("策略质量持续下降，建议进行策略调整")
        elif quality_trend == 'improving':
            recommendations.append("策略质量持续改善，表现良好")
        
        if not recommendations:
            recommendations.append("策略表现稳定，建议继续保持")
        
        return recommendations
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """获取分析器统计信息"""
        return {
            "significance_level": self.significance_level,
            "var_confidence": self.var_confidence,
            "min_observations": self.min_observations,
            "analysis_history_size": sum(len(history) for history in self.analysis_history.values()),
            "strategies_tracked": len(self.analysis_history)
        }
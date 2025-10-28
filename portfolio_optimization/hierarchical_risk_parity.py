"""
分层风险平价优化器
实现基于层次聚类的风险平价投资组合优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import optimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform


class LinkageMethod(Enum):
    """层次聚类方法枚举"""

    SINGLE = "single"  # 单链接
    COMPLETE = "complete"  # 全链接
    AVERAGE = "average"  # 平均链接
    WARD = "ward"  # Ward方法


@dataclass
class HRPResult:
    """分层风险平价结果"""

    weights: np.ndarray
    clusters: Dict[int, List[int]]  # 聚类结果
    linkage_matrix: np.ndarray  # 链接矩阵
    assets: List[str]
    diversification_ratio: float
    risk_contribution: np.ndarray


class HierarchicalRiskParity:
    """
    分层风险平价优化器

    实现基于层次聚类的风险平价投资组合优化：
    - 相关性矩阵层次聚类
    - 准对角化矩阵重排
    - 递归二分法权重分配
    - 风险分散化优化
    """

    def __init__(
        self,
        linkage_method: LinkageMethod = LinkageMethod.SINGLE,
        distance_metric: str = "correlation",
    ):
        """
        初始化分层风险平价优化器

        Args:
            linkage_method: 层次聚类方法
            distance_metric: 距离度量方法
        """
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.correlation_matrix = None
        self.cov_matrix = None
        self.assets = None
        self.volatilities = None

    def fit(self, returns: pd.DataFrame) -> None:
        """
        拟合模型数据

        Args:
            returns: 收益率数据，列为资产，行为时间
        """
        self.correlation_matrix = returns.corr()
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()
        self.volatilities = np.sqrt(np.diag(self.cov_matrix))

    def compute_distance_matrix(self) -> np.ndarray:
        """
        计算距离矩阵

        Returns:
            np.ndarray: 距离矩阵
        """
        if self.correlation_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")

        # 将相关性转换为距离：d = sqrt(2*(1 - ρ))
        distance_matrix = np.sqrt(2 * (1 - self.correlation_matrix))

        # 确保距离矩阵是对称且非负的
        np.fill_diagonal(distance_matrix, 0)

        return distance_matrix

    def hierarchical_clustering(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        执行层次聚类

        Returns:
            Tuple[np.ndarray, Dict[int, List[int]]]: (链接矩阵, 聚类结果)
        """
        distance_matrix = self.compute_distance_matrix()

        # 执行层次聚类
        linkage_matrix = hierarchy.linkage(
            pdist(distance_matrix), method=self.linkage_method.value
        )

        # 提取聚类结果
        clusters = self._extract_clusters(linkage_matrix)

        return linkage_matrix, clusters

    def _extract_clusters(self, linkage_matrix: np.ndarray) -> Dict[int, List[int]]:
        """提取聚类结果"""
        n_assets = len(self.assets)
        clusters = {}

        # 初始每个资产一个聚类
        for i in range(n_assets):
            clusters[i] = [i]

        # 根据链接矩阵合并聚类
        for i, (cluster1, cluster2, _, _) in enumerate(linkage_matrix):
            new_cluster_id = n_assets + i
            clusters[new_cluster_id] = clusters[int(cluster1)] + clusters[int(cluster2)]

            # 删除已合并的聚类
            del clusters[int(cluster1)]
            del clusters[int(cluster2)]

        return clusters

    def quasi_diagonalization(self, linkage_matrix: np.ndarray) -> List[int]:
        """
        准对角化矩阵重排

        Args:
            linkage_matrix: 链接矩阵

        Returns:
            List[int]: 重排后的资产索引
        """
        # 从链接矩阵中提取叶子节点顺序
        leaves = hierarchy.leaves_list(linkage_matrix)
        return leaves.tolist()

    def recursive_bisection(
        self, cov_matrix: np.ndarray, assets_indices: List[int]
    ) -> np.ndarray:
        """
        递归二分法权重分配

        Args:
            cov_matrix: 协方差矩阵
            assets_indices: 资产索引列表

        Returns:
            np.ndarray: 权重向量
        """
        n_assets = len(assets_indices)

        if n_assets == 1:
            # 单个资产，权重为1
            weights = np.array([1.0])
        else:
            # 将资产分为两个子集
            split_point = n_assets // 2
            left_indices = assets_indices[:split_point]
            right_indices = assets_indices[split_point:]

            # 提取子协方差矩阵
            left_cov = cov_matrix[np.ix_(left_indices, left_indices)]
            right_cov = cov_matrix[np.ix_(right_indices, right_indices)]

            # 递归计算子集权重
            left_weights = self.recursive_bisection(
                left_cov, list(range(len(left_indices)))
            )
            right_weights = self.recursive_bisection(
                right_cov, list(range(len(right_indices)))
            )

            # 计算子集风险
            left_risk = np.sqrt(left_weights.T @ left_cov @ left_weights)
            right_risk = np.sqrt(right_weights.T @ right_cov @ right_weights)

            # 根据风险分配权重
            alpha = (
                1 - left_risk / (left_risk + right_risk)
                if (left_risk + right_risk) > 0
                else 0.5
            )

            # 合并权重
            weights = np.zeros(n_assets)
            weights[: len(left_indices)] = alpha * left_weights
            weights[len(left_indices) :] = (1 - alpha) * right_weights

        return weights

    def optimize(self) -> HRPResult:
        """
        执行分层风险平价优化

        Returns:
            HRPResult: 分层风险平价结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")

        # 1. 层次聚类
        linkage_matrix, clusters = self.hierarchical_clustering()

        # 2. 准对角化重排
        ordered_indices = self.quasi_diagonalization(linkage_matrix)

        # 3. 递归二分法权重分配
        weights = self.recursive_bisection(self.cov_matrix, ordered_indices)

        # 4. 重新排序权重到原始资产顺序
        final_weights = np.zeros(len(self.assets))
        for new_idx, old_idx in enumerate(ordered_indices):
            final_weights[old_idx] = weights[new_idx]

        # 5. 归一化权重
        final_weights = final_weights / np.sum(final_weights)

        # 6. 计算风险贡献
        risk_contribution = self._calculate_risk_contribution(final_weights)

        # 7. 计算分散化比率
        diversification_ratio = self._calculate_diversification_ratio(final_weights)

        return HRPResult(
            weights=final_weights,
            clusters=clusters,
            linkage_matrix=linkage_matrix,
            assets=self.assets,
            diversification_ratio=diversification_ratio,
            risk_contribution=risk_contribution,
        )

    def _calculate_risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """计算风险贡献"""
        portfolio_variance = weights.T @ self.cov_matrix @ weights
        marginal_risk_contribution = (self.cov_matrix @ weights) / portfolio_variance
        risk_contribution = weights * marginal_risk_contribution
        return risk_contribution

    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """计算分散化比率"""
        weighted_avg_vol = np.sum(weights * self.volatilities)
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

    def plot_dendrogram(
        self,
        linkage_matrix: np.ndarray,
        title: str = "Hierarchical Risk Parity Dendrogram",
    ) -> None:
        """
        绘制树状图（需要matplotlib）

        Args:
            linkage_matrix: 链接矩阵
            title: 图表标题
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            hierarchy.dendrogram(linkage_matrix, labels=self.assets)
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("需要安装matplotlib来绘制树状图")

    def compare_with_traditional_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        与传统方法比较

        Returns:
            Dict[str, Dict[str, Any]]: 各种方法的结果比较
        """
        results = {}

        # HRP方法
        hrp_result = self.optimize()
        results["hrp"] = {
            "weights": hrp_result.weights,
            "diversification_ratio": hrp_result.diversification_ratio,
            "volatility": np.sqrt(
                hrp_result.weights.T @ self.cov_matrix @ hrp_result.weights
            ),
        }

        # 等权重方法
        equal_weights = np.ones(len(self.assets)) / len(self.assets)
        results["equal_weight"] = {
            "weights": equal_weights,
            "diversification_ratio": self._calculate_diversification_ratio(
                equal_weights
            ),
            "volatility": np.sqrt(equal_weights.T @ self.cov_matrix @ equal_weights),
        }

        # 最小方差方法
        try:
            from .mean_variance_optimizer import MeanVarianceOptimizer

            mv_optimizer = MeanVarianceOptimizer()
            mv_optimizer.fit(
                pd.DataFrame(self.cov_matrix, columns=self.assets, index=self.assets)
            )
            mv_result = mv_optimizer.minimize_variance()
            results["min_variance"] = {
                "weights": mv_result.weights,
                "diversification_ratio": self._calculate_diversification_ratio(
                    mv_result.weights
                ),
                "volatility": mv_result.volatility,
            }
        except ImportError:
            pass

        # 风险平价方法
        try:
            from .risk_parity_optimizer import RiskParityOptimizer

            rp_optimizer = RiskParityOptimizer()
            rp_optimizer.fit(
                pd.DataFrame(self.cov_matrix, columns=self.assets, index=self.assets)
            )
            rp_result = rp_optimizer.optimized_risk_parity()
            results["risk_parity"] = {
                "weights": rp_result.weights,
                "diversification_ratio": self._calculate_diversification_ratio(
                    rp_result.weights
                ),
                "volatility": rp_result.total_risk,
            }
        except ImportError:
            pass

        return results

    def sensitivity_analysis(
        self, n_simulations: int = 100, noise_level: float = 0.1
    ) -> Dict[str, Any]:
        """
        敏感性分析

        Args:
            n_simulations: 模拟次数
            noise_level: 噪声水平

        Returns:
            Dict[str, Any]: 敏感性分析结果
        """
        original_weights = self.optimize().weights

        weight_variations = []
        diversification_ratios = []
        volatilities = []

        for _ in range(n_simulations):
            # 添加噪声到协方差矩阵
            noise = np.random.normal(0, noise_level, self.cov_matrix.shape)
            noisy_cov = self.cov_matrix * (1 + noise)

            # 使用噪声数据重新优化
            try:
                self.cov_matrix = noisy_cov
                result = self.optimize()

                weight_variations.append(result.weights)
                diversification_ratios.append(result.diversification_ratio)
                volatilities.append(
                    np.sqrt(result.weights.T @ self.cov_matrix @ result.weights)
                )
            except:
                continue

        # 恢复原始协方差矩阵
        self.cov_matrix = self.cov_matrix / (1 + noise_level)

        if weight_variations:
            weight_variations = np.array(weight_variations)
            weight_std = np.std(weight_variations, axis=0)
            weight_stability = 1 - np.mean(weight_std)
        else:
            weight_stability = 0

        return {
            "weight_stability": weight_stability,
            "diversification_ratio_mean": (
                np.mean(diversification_ratios) if diversification_ratios else 0
            ),
            "diversification_ratio_std": (
                np.std(diversification_ratios) if diversification_ratios else 0
            ),
            "volatility_mean": np.mean(volatilities) if volatilities else 0,
            "volatility_std": np.std(volatilities) if volatilities else 0,
            "n_successful_simulations": len(weight_variations),
        }

    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        return {
            "linkage_method": self.linkage_method.value,
            "distance_metric": self.distance_metric,
            "n_assets": len(self.assets) if self.assets else 0,
            "correlation_matrix_shape": (
                self.correlation_matrix.shape
                if self.correlation_matrix is not None
                else None
            ),
        }

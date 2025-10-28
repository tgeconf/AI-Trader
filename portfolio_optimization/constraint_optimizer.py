"""
约束优化器
实现带复杂约束的投资组合优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from scipy import optimize
import cvxpy as cp


class ConstraintType(Enum):
    """约束类型枚举"""

    LINEAR_EQUALITY = "linear_equality"  # 线性等式约束
    LINEAR_INEQUALITY = "linear_inequality"  # 线性不等式约束
    QUADRATIC = "quadratic"  # 二次约束
    INTEGER = "integer"  # 整数约束
    CARDINALITY = "cardinality"  # 基数约束
    TURNOVER = "turnover"  # 换手率约束


class OptimizationMethod(Enum):
    """优化方法枚举"""

    SLSQP = "slsqp"  # 序列二次规划
    COBYLA = "cobyla"  # 约束优化线性逼近
    TRUST_CONSTR = "trust_constr"  # 信赖域约束优化
    CVXPY = "cvxpy"  # 凸优化
    GENETIC = "genetic"  # 遗传算法


@dataclass
class Constraint:
    """约束定义"""

    constraint_type: ConstraintType
    function: Callable
    bounds: Optional[Tuple[float, float]] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ConstrainedOptimizationResult:
    """约束优化结果"""

    weights: np.ndarray
    objective_value: float
    constraints_satisfied: bool
    optimization_method: OptimizationMethod
    constraint_violations: Dict[str, float]
    assets: List[str]
    iterations: int


class ConstraintOptimizer:
    """
    约束优化器

    实现带复杂约束的投资组合优化：
    - 多种约束类型支持
    - 多种优化算法选择
    - 约束违反检测
    - 复杂投资策略实现
    """

    def __init__(
        self,
        optimization_method: OptimizationMethod = OptimizationMethod.SLSQP,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ):
        """
        初始化约束优化器

        Args:
            optimization_method: 优化方法
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
        """
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.expected_returns = None
        self.cov_matrix = None
        self.assets = None
        self.constraints = []

    def fit(self, returns: pd.DataFrame) -> None:
        """
        拟合模型数据

        Args:
            returns: 收益率数据，列为资产，行为时间
        """
        self.expected_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.assets = returns.columns.tolist()

    def add_constraint(self, constraint: Constraint) -> None:
        """
        添加约束

        Args:
            constraint: 约束定义
        """
        self.constraints.append(constraint)

    def add_weight_constraint(
        self, min_weight: float = 0.0, max_weight: float = 1.0
    ) -> None:
        """
        添加权重约束

        Args:
            min_weight: 最小权重
            max_weight: 最大权重
        """
        n_assets = len(self.assets) if self.assets else 0
        if n_assets == 0:
            raise ValueError("请先调用fit方法拟合数据")

        # 权重边界约束
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

        constraint = Constraint(
            constraint_type=ConstraintType.LINEAR_INEQUALITY,
            function=lambda x: x,  # 边界约束通过bounds参数处理
            bounds=(min_weight, max_weight),
        )
        self.constraints.append(constraint)

    def add_sector_constraint(
        self, sector_weights: Dict[str, List[str]], max_sector_weight: float
    ) -> None:
        """
        添加行业约束

        Args:
            sector_weights: 行业权重映射，{"sector": [assets]}
            max_sector_weight: 最大行业权重
        """
        if self.assets is None:
            raise ValueError("请先调用fit方法拟合数据")

        for sector, assets in sector_weights.items():
            # 创建行业权重约束函数
            def sector_constraint_func(weights, sector_assets=assets):
                sector_indices = [self.assets.index(asset) for asset in sector_assets]
                return np.sum(weights[sector_indices]) - max_sector_weight

            constraint = Constraint(
                constraint_type=ConstraintType.LINEAR_INEQUALITY,
                function=sector_constraint_func,
                bounds=(-np.inf, 0),  # sum(weights) <= max_sector_weight
            )
            self.constraints.append(constraint)

    def add_turnover_constraint(
        self, current_weights: np.ndarray, max_turnover: float
    ) -> None:
        """
        添加换手率约束

        Args:
            current_weights: 当前权重
            max_turnover: 最大换手率
        """

        def turnover_constraint_func(weights):
            turnover = np.sum(np.abs(weights - current_weights)) / 2
            return turnover - max_turnover

        constraint = Constraint(
            constraint_type=ConstraintType.LINEAR_INEQUALITY,
            function=turnover_constraint_func,
            bounds=(-np.inf, 0),  # turnover <= max_turnover
        )
        self.constraints.append(constraint)

    def add_cardinality_constraint(self, max_assets: int) -> None:
        """
        添加基数约束（持有资产数量限制）

        Args:
            max_assets: 最大持有资产数量
        """

        def cardinality_constraint_func(weights):
            # 使用L0范数近似（非凸，需要特殊处理）
            non_zero_count = np.sum(weights > 1e-6)
            return non_zero_count - max_assets

        constraint = Constraint(
            constraint_type=ConstraintType.INTEGER,
            function=cardinality_constraint_func,
            bounds=(-np.inf, 0),  # non_zero_count <= max_assets
        )
        self.constraints.append(constraint)

    def add_tracking_error_constraint(
        self, benchmark_weights: np.ndarray, max_tracking_error: float
    ) -> None:
        """
        添加跟踪误差约束

        Args:
            benchmark_weights: 基准权重
            max_tracking_error: 最大跟踪误差
        """

        def tracking_error_constraint_func(weights):
            active_weights = weights - benchmark_weights
            tracking_variance = active_weights.T @ self.cov_matrix @ active_weights
            tracking_error = np.sqrt(tracking_variance)
            return tracking_error - max_tracking_error

        constraint = Constraint(
            constraint_type=ConstraintType.QUADRATIC,
            function=tracking_error_constraint_func,
            bounds=(-np.inf, 0),  # tracking_error <= max_tracking_error
        )
        self.constraints.append(constraint)

    def add_risk_constraint(self, max_volatility: float) -> None:
        """
        添加风险约束

        Args:
            max_volatility: 最大波动率
        """

        def risk_constraint_func(weights):
            portfolio_variance = weights.T @ self.cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
            return portfolio_volatility - max_volatility

        constraint = Constraint(
            constraint_type=ConstraintType.QUADRATIC,
            function=risk_constraint_func,
            bounds=(-np.inf, 0),  # volatility <= max_volatility
        )
        self.constraints.append(constraint)

    def add_return_constraint(self, min_return: float) -> None:
        """
        添加收益约束

        Args:
            min_return: 最小预期收益
        """

        def return_constraint_func(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            return min_return - portfolio_return  # portfolio_return >= min_return

        constraint = Constraint(
            constraint_type=ConstraintType.LINEAR_INEQUALITY,
            function=return_constraint_func,
            bounds=(-np.inf, 0),  # portfolio_return >= min_return
        )
        self.constraints.append(constraint)

    def optimize_mean_variance(
        self,
        objective: str = "minimize_variance",
        target_return: Optional[float] = None,
    ) -> ConstrainedOptimizationResult:
        """
        均值方差优化

        Args:
            objective: 优化目标
            target_return: 目标收益

        Returns:
            约束优化结果
        """
        if self.expected_returns is None or self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")

        n_assets = len(self.assets)

        # 定义目标函数
        if objective == "minimize_variance":

            def objective_func(weights):
                return weights.T @ self.cov_matrix @ weights

        elif objective == "maximize_sharpe":
            risk_free_rate = 0.02  # 默认无风险利率

            def objective_func(weights):
                portfolio_return = np.dot(weights, self.expected_returns)
                portfolio_variance = weights.T @ self.cov_matrix @ weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe = (
                    (portfolio_return - risk_free_rate) / portfolio_volatility
                    if portfolio_volatility > 0
                    else -np.inf
                )
                return -sharpe  # 最大化夏普比率

        elif objective == "maximize_return":

            def objective_func(weights):
                return -np.dot(weights, self.expected_returns)  # 最大化收益

        else:
            raise ValueError(f"不支持的优化目标: {objective}")

        # 构建约束条件
        scipy_constraints = self._build_scipy_constraints()

        # 权重和为1的约束
        scipy_constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})

        # 目标收益约束
        if target_return is not None:
            scipy_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.dot(x, self.expected_returns) - target_return,
                }
            )

        # 边界条件
        bounds = self._get_bounds()

        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            objective_func,
            initial_weights,
            method=self.optimization_method.value,
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )

        # 检查约束满足情况
        constraint_violations = self._check_constraint_violations(result.x)

        return ConstrainedOptimizationResult(
            weights=result.x,
            objective_value=result.fun,
            constraints_satisfied=result.success,
            optimization_method=self.optimization_method,
            constraint_violations=constraint_violations,
            assets=self.assets,
            iterations=result.nit,
        )

    def optimize_risk_parity(self) -> ConstrainedOptimizationResult:
        """
        风险平价优化

        Returns:
            约束优化结果
        """
        if self.cov_matrix is None:
            raise ValueError("请先调用fit方法拟合数据")

        n_assets = len(self.assets)

        # 风险平价目标函数
        def risk_parity_objective(weights):
            risk_contributions = self._calculate_risk_contributions(weights)
            target_contribution = 1.0 / n_assets
            deviation = np.sum((risk_contributions - target_contribution) ** 2)
            return deviation

        # 构建约束条件
        scipy_constraints = self._build_scipy_constraints()
        scipy_constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})

        # 边界条件
        bounds = self._get_bounds()

        # 初始解：等权重
        initial_weights = np.ones(n_assets) / n_assets

        # 优化
        result = optimize.minimize(
            risk_parity_objective,
            initial_weights,
            method=self.optimization_method.value,
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )

        # 检查约束满足情况
        constraint_violations = self._check_constraint_violations(result.x)

        return ConstrainedOptimizationResult(
            weights=result.x,
            objective_value=result.fun,
            constraints_satisfied=result.success,
            optimization_method=self.optimization_method,
            constraint_violations=constraint_violations,
            assets=self.assets,
            iterations=result.nit,
        )

    def optimize_with_cvxpy(
        self, objective: str = "minimize_variance"
    ) -> ConstrainedOptimizationResult:
        """
        使用CVXPY进行凸优化

        Args:
            objective: 优化目标

        Returns:
            约束优化结果
        """
        try:
            n_assets = len(self.assets)

            # 定义变量
            weights = cp.Variable(n_assets)

            # 基本约束：权重和为1，权重非负
            constraints = [cp.sum(weights) == 1, weights >= 0]

            # 添加自定义约束
            for constraint in self.constraints:
                if constraint.constraint_type == ConstraintType.LINEAR_INEQUALITY:
                    if constraint.bounds:
                        lb, ub = constraint.bounds
                        if lb is not None:
                            constraints.append(constraint.function(weights) >= lb)
                        if ub is not None:
                            constraints.append(constraint.function(weights) <= ub)
                elif constraint.constraint_type == ConstraintType.LINEAR_EQUALITY:
                    constraints.append(constraint.function(weights) == 0)
                elif constraint.constraint_type == ConstraintType.QUADRATIC:
                    # 二次约束需要特殊处理
                    pass

            # 定义目标函数
            if objective == "minimize_variance":
                objective_func = cp.quad_form(weights, self.cov_matrix.values)
                problem = cp.Problem(cp.Minimize(objective_func), constraints)
            elif objective == "maximize_return":
                objective_func = weights.T @ self.expected_returns.values
                problem = cp.Problem(cp.Maximize(objective_func), constraints)
            else:
                raise ValueError(f"不支持的优化目标: {objective}")

            # 求解
            problem.solve()

            # 检查约束满足情况
            constraint_violations = self._check_constraint_violations(weights.value)

            return ConstrainedOptimizationResult(
                weights=weights.value,
                objective_value=problem.value,
                constraints_satisfied=problem.status == cp.OPTIMAL,
                optimization_method=OptimizationMethod.CVXPY,
                constraint_violations=constraint_violations,
                assets=self.assets,
                iterations=0,  # CVXPY不提供迭代次数
            )

        except ImportError:
            raise ImportError("需要安装cvxpy来使用凸优化方法")

    def _build_scipy_constraints(self) -> List[Dict]:
        """构建scipy优化约束"""
        scipy_constraints = []

        for constraint in self.constraints:
            if constraint.constraint_type in [
                ConstraintType.LINEAR_EQUALITY,
                ConstraintType.LINEAR_INEQUALITY,
            ]:
                if constraint.bounds:
                    lb, ub = constraint.bounds
                    if lb is not None and ub is not None:
                        # 双边约束
                        scipy_constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda x, lb=lb, ub=ub, f=constraint.function: [
                                    f(x) - lb,
                                    ub - f(x),
                                ],
                            }
                        )
                    elif lb is not None:
                        # 下界约束
                        scipy_constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda x, lb=lb, f=constraint.function: f(x)
                                - lb,
                            }
                        )
                    elif ub is not None:
                        # 上界约束
                        scipy_constraints.append(
                            {
                                "type": "ineq",
                                "fun": lambda x, ub=ub, f=constraint.function: ub
                                - f(x),
                            }
                        )

        return scipy_constraints

    def _get_bounds(self) -> List[Tuple]:
        """获取边界条件"""
        bounds = []

        for constraint in self.constraints:
            if (
                constraint.constraint_type == ConstraintType.LINEAR_INEQUALITY
                and constraint.bounds
            ):
                lb, ub = constraint.bounds
                if lb is not None and ub is not None:
                    # 如果约束是针对单个权重的边界
                    if (
                        hasattr(constraint.function, "__code__")
                        and constraint.function.__code__.co_argcount == 1
                    ):
                        # 这是一个权重边界约束
                        bounds.append((lb, ub))

        # 如果没有明确的边界约束，使用默认边界
        if not bounds:
            n_assets = len(self.assets)
            bounds = [(0, 1) for _ in range(n_assets)]

        return bounds

    def _calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """计算风险贡献"""
        portfolio_variance = weights.T @ self.cov_matrix @ weights
        marginal_risk_contribution = (self.cov_matrix @ weights) / portfolio_variance
        risk_contribution = weights * marginal_risk_contribution
        return risk_contribution

    def _check_constraint_violations(self, weights: np.ndarray) -> Dict[str, float]:
        """检查约束违反情况"""
        violations = {}

        for i, constraint in enumerate(self.constraints):
            if constraint.bounds:
                lb, ub = constraint.bounds
                value = constraint.function(weights)

                if lb is not None and value < lb:
                    violations[f"constraint_{i}_lower"] = lb - value
                elif ub is not None and value > ub:
                    violations[f"constraint_{i}_upper"] = value - ub

        # 检查权重和为1的约束
        weight_sum = np.sum(weights)
        if abs(weight_sum - 1) > 1e-6:
            violations["weight_sum"] = abs(weight_sum - 1)

        return violations

    def clear_constraints(self) -> None:
        """清除所有约束"""
        self.constraints = []

    def get_constraint_summary(self) -> Dict[str, Any]:
        """获取约束摘要"""
        summary = {
            "total_constraints": len(self.constraints),
            "constraint_types": {},
            "optimization_method": self.optimization_method.value,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }

        for constraint in self.constraints:
            constraint_type = constraint.constraint_type.value
            if constraint_type not in summary["constraint_types"]:
                summary["constraint_types"][constraint_type] = 0
            summary["constraint_types"][constraint_type] += 1

        return summary

    def sensitivity_analysis(
        self,
        objective: str = "minimize_variance",
        parameter: str = "max_weight",
        values: List[float] = None,
    ) -> Dict[str, ConstrainedOptimizationResult]:
        """
        敏感性分析

        Args:
            objective: 优化目标
            parameter: 参数名称
            values: 参数值列表

        Returns:
            敏感性分析结果
        """
        if values is None:
            values = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = {}

        # 保存原始约束
        original_constraints = self.constraints.copy()

        for value in values:
            # 清除约束
            self.clear_constraints()

            # 根据参数设置约束
            if parameter == "max_weight":
                self.add_weight_constraint(max_weight=value)
            elif parameter == "max_turnover":
                current_weights = np.ones(len(self.assets)) / len(self.assets)
                self.add_turnover_constraint(current_weights, max_turnover=value)
            elif parameter == "max_volatility":
                self.add_risk_constraint(max_volatility=value)
            else:
                raise ValueError(f"不支持的参数: {parameter}")

            # 执行优化
            try:
                if objective == "minimize_variance":
                    result = self.optimize_mean_variance("minimize_variance")
                elif objective == "maximize_sharpe":
                    result = self.optimize_mean_variance("maximize_sharpe")
                elif objective == "risk_parity":
                    result = self.optimize_risk_parity()
                else:
                    raise ValueError(f"不支持的优化目标: {objective}")

                results[str(value)] = result
            except Exception as e:
                print(f"优化失败，参数值 {value}: {e}")

        # 恢复原始约束
        self.constraints = original_constraints

        return results

    def compare_optimization_methods(
        self, objective: str = "minimize_variance"
    ) -> Dict[str, ConstrainedOptimizationResult]:
        """
        Args:
            objective: 优化目标

        Returns:
            不同优化方法的结果比较
        """
        results = {}

        # 保存原始优化方法
        original_method = self.optimization_method

        # 测试不同优化方法
        methods_to_test = [
            OptimizationMethod.SLSQP,
            OptimizationMethod.TRUST_CONSTR,
            OptimizationMethod.COBYLA,
        ]

        for method in methods_to_test:
            try:
                self.optimization_method = method

                if objective == "minimize_variance":
                    result = self.optimize_mean_variance("minimize_variance")
                elif objective == "maximize_sharpe":
                    result = self.optimize_mean_variance("maximize_sharpe")
                elif objective == "risk_parity":
                    result = self.optimize_risk_parity()
                else:
                    continue

                results[method.value] = result
            except Exception as e:
                print(f"优化方法 {method.value} 失败: {e}")

        # 恢复原始优化方法
        self.optimization_method = original_method

        return results

    def generate_optimization_report(
        self, result: ConstrainedOptimizationResult
    ) -> Dict[str, Any]:
        """
        生成优化报告

        Args:
            result: 优化结果

        Returns:
            优化报告
        """
        # 计算投资组合统计量
        portfolio_return = np.dot(result.weights, self.expected_returns)
        portfolio_variance = result.weights.T @ self.cov_matrix @ result.weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        # 计算风险贡献
        risk_contributions = self._calculate_risk_contributions(result.weights)

        # 计算分散化比率
        weighted_avg_vol = np.sum(result.weights * np.sqrt(np.diag(self.cov_matrix)))
        diversification_ratio = (
            weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0
        )

        # 构建权重字典
        weights_dict = {
            asset: weight for asset, weight in zip(self.assets, result.weights)
        }

        # 构建风险贡献字典
        risk_contributions_dict = {
            asset: contribution
            for asset, contribution in zip(self.assets, risk_contributions)
        }

        return {
            "optimization_method": result.optimization_method.value,
            "objective_value": result.objective_value,
            "constraints_satisfied": result.constraints_satisfied,
            "iterations": result.iterations,
            "portfolio_statistics": {
                "expected_return": portfolio_return,
                "volatility": portfolio_volatility,
                "diversification_ratio": diversification_ratio,
            },
            "optimal_weights": weights_dict,
            "risk_contributions": risk_contributions_dict,
            "constraint_violations": result.constraint_violations,
            "recommendations": self._generate_optimization_recommendations(result),
        }

    def _generate_optimization_recommendations(
        self, result: ConstrainedOptimizationResult
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 基于约束违反的建议
        if result.constraint_violations:
            total_violation = sum(result.constraint_violations.values())
            if total_violation > 0.01:
                recommendations.append("约束违反较严重，建议放宽约束条件")
            else:
                recommendations.append("轻微约束违反，可接受")
        else:
            recommendations.append("所有约束均满足")

        # 基于优化方法的建议
        if result.optimization_method == OptimizationMethod.SLSQP:
            recommendations.append("使用SLSQP方法优化成功")
        elif result.optimization_method == OptimizationMethod.CVXPY:
            recommendations.append("使用凸优化方法，结果可靠")

        # 基于迭代次数的建议
        if result.iterations >= self.max_iterations:
            recommendations.append("达到最大迭代次数，建议增加迭代上限")

        # 基于权重分布的建议
        weights = result.weights
        if np.max(weights) > 0.3:
            recommendations.append("存在集中持仓，建议分散化")
        elif np.sum(weights > 0.01) < 5:
            recommendations.append("持仓过于集中，建议增加资产数量")

        return recommendations

    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        return {
            "optimization_method": self.optimization_method.value,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "n_assets": len(self.assets) if self.assets else 0,
            "n_constraints": len(self.constraints),
            "constraint_types": [
                constraint.constraint_type.value for constraint in self.constraints
            ],
        }

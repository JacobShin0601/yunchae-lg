"""
Common Interfaces - Standardized data structures for optimization

Provides consistent interfaces for optimization inputs and outputs across
all optimization methods (Pareto, Robust, Stochastic, etc.).

Components:
- OptimizationInput: Standardized input data structure
- OptimizationResult: Standardized result data structure
- ResultComparator: Compare results across methods
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd


class OptimizationMethod(Enum):
    """Optimization methods"""
    SINGLE_OBJECTIVE = "single_objective"
    WEIGHT_SWEEP = "weight_sweep"
    EPSILON_CONSTRAINT = "epsilon_constraint"
    NSGA2 = "nsga2"
    ROBUST = "robust"
    STOCHASTIC = "stochastic"
    HYBRID = "hybrid"


class OptimizationStatus(Enum):
    """Optimization status"""
    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class OptimizationInput:
    """
    Standardized optimization input data structure

    Encapsulates all data needed for an optimization run.
    """
    # Core data
    optimization_data: Dict[str, Any]  # Raw optimization data (scenario_df, etc.)
    cost_calculator: Any  # Cost calculator instance

    # Method configuration
    method: OptimizationMethod = OptimizationMethod.SINGLE_OBJECTIVE
    objective_type: str = 'minimize_carbon'  # 'minimize_carbon', 'minimize_cost', 'multi_objective'

    # Scenario configuration
    baseline_case: str = 'case1'
    constraint_preset: str = 'medium'
    scenario_template: Optional[str] = None

    # Solver configuration
    solver_name: str = 'auto'
    time_limit: int = 300  # seconds
    gap_tolerance: float = 0.01
    verbose: bool = False

    # Method-specific parameters
    method_params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    run_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        return {
            'method': self.method.value,
            'objective_type': self.objective_type,
            'baseline_case': self.baseline_case,
            'constraint_preset': self.constraint_preset,
            'scenario_template': self.scenario_template,
            'solver_name': self.solver_name,
            'time_limit': self.time_limit,
            'gap_tolerance': self.gap_tolerance,
            'verbose': self.verbose,
            'method_params': self.method_params,
            'run_id': self.run_id,
            'description': self.description,
            'tags': self.tags
        }


@dataclass
class OptimizationResult:
    """
    Standardized optimization result structure

    Provides consistent interface for all optimization results.
    """
    # Core results
    status: OptimizationStatus
    method: OptimizationMethod

    # Objectives
    total_carbon: float
    total_cost: float
    carbon_reduction_pct: float = 0.0
    cost_increase_pct: float = 0.0

    # Solution details
    solution: Dict[str, Any] = field(default_factory=dict)
    result_df: Optional[pd.DataFrame] = None

    # Quality metrics
    solver_time: float = 0.0
    gap: float = 0.0
    iterations: int = 0

    # Pareto-specific (optional)
    pareto_rank: Optional[int] = None
    crowding_distance: Optional[float] = None
    hypervolume_contribution: Optional[float] = None

    # Robust/Stochastic-specific (optional)
    robustness_score: Optional[float] = None
    expected_carbon: Optional[float] = None
    expected_cost: Optional[float] = None
    carbon_std: Optional[float] = None
    cost_std: Optional[float] = None
    confidence_interval: Optional[Dict[str, Tuple[float, float]]] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_config: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        return self.status == OptimizationStatus.SUCCESS

    def get_summary(self) -> Dict[str, Any]:
        """Get result summary"""
        summary = {
            'status': self.status.value,
            'method': self.method.value,
            'total_carbon': self.total_carbon,
            'total_cost': self.total_cost,
            'carbon_reduction_pct': self.carbon_reduction_pct,
            'cost_increase_pct': self.cost_increase_pct,
            'solver_time': self.solver_time,
            'timestamp': self.timestamp
        }

        # Add optional fields if present
        if self.pareto_rank is not None:
            summary['pareto_rank'] = self.pareto_rank
        if self.robustness_score is not None:
            summary['robustness_score'] = self.robustness_score
        if self.expected_carbon is not None:
            summary['expected_carbon'] = self.expected_carbon
            summary['expected_cost'] = self.expected_cost

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        result_dict = {
            'status': self.status.value,
            'method': self.method.value,
            'total_carbon': self.total_carbon,
            'total_cost': self.total_cost,
            'carbon_reduction_pct': self.carbon_reduction_pct,
            'cost_increase_pct': self.cost_increase_pct,
            'solution': self.solution,
            'solver_time': self.solver_time,
            'gap': self.gap,
            'iterations': self.iterations,
            'timestamp': self.timestamp
        }

        # Add optional fields
        optional_fields = [
            'pareto_rank', 'crowding_distance', 'hypervolume_contribution',
            'robustness_score', 'expected_carbon', 'expected_cost',
            'carbon_std', 'cost_std', 'confidence_interval',
            'input_config', 'error_message'
        ]

        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result_dict[field_name] = value

        return result_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create result from dictionary"""
        return cls(
            status=OptimizationStatus(data['status']),
            method=OptimizationMethod(data['method']),
            total_carbon=data['total_carbon'],
            total_cost=data['total_cost'],
            carbon_reduction_pct=data.get('carbon_reduction_pct', 0.0),
            cost_increase_pct=data.get('cost_increase_pct', 0.0),
            solution=data.get('solution', {}),
            solver_time=data.get('solver_time', 0.0),
            gap=data.get('gap', 0.0),
            iterations=data.get('iterations', 0),
            pareto_rank=data.get('pareto_rank'),
            crowding_distance=data.get('crowding_distance'),
            hypervolume_contribution=data.get('hypervolume_contribution'),
            robustness_score=data.get('robustness_score'),
            expected_carbon=data.get('expected_carbon'),
            expected_cost=data.get('expected_cost'),
            carbon_std=data.get('carbon_std'),
            cost_std=data.get('cost_std'),
            confidence_interval=data.get('confidence_interval'),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            input_config=data.get('input_config'),
            error_message=data.get('error_message')
        )


class ResultComparator:
    """
    Compare optimization results across different methods

    Provides side-by-side comparison and ranking of results.
    """

    @staticmethod
    def compare_two(
        result1: OptimizationResult,
        result2: OptimizationResult
    ) -> Dict[str, Any]:
        """
        Compare two optimization results

        Args:
            result1: First result
            result2: Second result

        Returns:
            Comparison dictionary
        """
        comparison = {
            'methods': [result1.method.value, result2.method.value],
            'carbon_difference': result2.total_carbon - result1.total_carbon,
            'carbon_difference_pct': (
                (result2.total_carbon - result1.total_carbon) / result1.total_carbon * 100
                if result1.total_carbon > 0 else 0
            ),
            'cost_difference': result2.total_cost - result1.total_cost,
            'cost_difference_pct': (
                (result2.total_cost - result1.total_cost) / result1.total_cost * 100
                if result1.total_cost > 0 else 0
            ),
            'time_difference': result2.solver_time - result1.solver_time,
            'better_carbon': result1.method.value if result1.total_carbon < result2.total_carbon else result2.method.value,
            'better_cost': result1.method.value if result1.total_cost < result2.total_cost else result2.method.value
        }

        return comparison

    @staticmethod
    def compare_multiple(
        results: List[OptimizationResult],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple optimization results

        Args:
            results: List of results to compare
            metrics: List of metrics to compare (default: all)

        Returns:
            DataFrame with comparison
        """
        if not results:
            return pd.DataFrame()

        if metrics is None:
            metrics = [
                'total_carbon', 'total_cost', 'carbon_reduction_pct',
                'cost_increase_pct', 'solver_time'
            ]

        data = []
        for result in results:
            row = {
                'method': result.method.value,
                'status': result.status.value
            }

            for metric in metrics:
                value = getattr(result, metric, None)
                if value is not None:
                    row[metric] = value

            data.append(row)

        df = pd.DataFrame(data)

        return df

    @staticmethod
    def rank_results(
        results: List[OptimizationResult],
        criteria: str = 'carbon'
    ) -> List[Tuple[int, OptimizationResult]]:
        """
        Rank results by specified criteria

        Args:
            results: List of results
            criteria: Ranking criteria ('carbon', 'cost', 'balanced')

        Returns:
            List of (rank, result) tuples
        """
        if not results:
            return []

        # Filter feasible results
        feasible_results = [r for r in results if r.is_feasible()]

        if not feasible_results:
            return []

        # Rank by criteria
        if criteria == 'carbon':
            sorted_results = sorted(feasible_results, key=lambda r: r.total_carbon)
        elif criteria == 'cost':
            sorted_results = sorted(feasible_results, key=lambda r: r.total_cost)
        elif criteria == 'balanced':
            # Normalize and sum
            carbon_values = [r.total_carbon for r in feasible_results]
            cost_values = [r.total_cost for r in feasible_results]

            carbon_min, carbon_max = min(carbon_values), max(carbon_values)
            cost_min, cost_max = min(cost_values), max(cost_values)

            def balanced_score(r):
                norm_carbon = (r.total_carbon - carbon_min) / (carbon_max - carbon_min) if carbon_max > carbon_min else 0
                norm_cost = (r.total_cost - cost_min) / (cost_max - cost_min) if cost_max > cost_min else 0
                return norm_carbon + norm_cost

            sorted_results = sorted(feasible_results, key=balanced_score)
        else:
            raise ValueError(f"Unknown criteria: {criteria}")

        return [(i + 1, result) for i, result in enumerate(sorted_results)]

    @staticmethod
    def create_comparison_report(
        results: List[OptimizationResult]
    ) -> str:
        """
        Create formatted comparison report

        Args:
            results: List of results to compare

        Returns:
            Formatted text report
        """
        report = "\n" + "=" * 70 + "\n"
        report += "OPTIMIZATION RESULTS COMPARISON\n"
        report += "=" * 70 + "\n\n"

        if not results:
            report += "No results to compare.\n"
            return report

        # Summary table
        df = ResultComparator.compare_multiple(results)
        report += "Summary:\n"
        report += df.to_string(index=False) + "\n\n"

        # Rankings
        report += "=" * 70 + "\n"
        report += "RANKINGS:\n\n"

        for criteria in ['carbon', 'cost', 'balanced']:
            report += f"{criteria.upper()} Ranking:\n"
            ranked = ResultComparator.rank_results(results, criteria)

            for rank, result in ranked:
                report += f"  {rank}. {result.method.value:20s} - "
                report += f"Carbon: {result.total_carbon:.2f}, "
                report += f"Cost: ${result.total_cost:,.0f}\n"

            report += "\n"

        report += "=" * 70 + "\n"

        return report

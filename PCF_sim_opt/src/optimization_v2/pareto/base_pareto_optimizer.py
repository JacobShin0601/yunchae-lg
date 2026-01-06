"""
Base Pareto Optimizer - Common functionality for all Pareto optimization methods

This module provides a unified base class for Pareto optimization algorithms,
eliminating code duplication and ensuring consistency across methods.

Key Features:
- Unified cost calculator setup
- Standardized baseline calculation
- Consistent result storage and retrieval
- Configurable Pareto filtering
- Abstract interface for optimization methods
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import json
import pandas as pd

from .config_loader import ParetoConfigLoader
from ..utils.total_cost_calculator import TotalCostCalculator


class BaseParetoOptimizer(ABC):
    """
    Base class for all Pareto optimization methods

    This abstract base class provides common functionality for:
    - Epsilon Constraint Method
    - NSGA-II
    - Weight Sweep

    Subclasses must implement the abstract run_optimization() method.
    """

    def __init__(self, user_id: str = None):
        """
        Initialize base optimizer

        Args:
            user_id: User ID for configuration management (optional)
        """
        self.user_id = user_id
        self.config_loader = ParetoConfigLoader(user_id)
        self.total_cost_calculator: Optional[TotalCostCalculator] = None
        self.zero_premium_baseline: Optional[float] = None
        self.baseline_carbon: Optional[float] = None
        self.results: List[Dict[str, Any]] = []

    def setup_cost_calculator(self, cost_calculator) -> None:
        """
        Setup unified cost calculator

        This method should be called once at the beginning of optimization
        to create a TotalCostCalculator wrapper around the RE100PremiumCalculator.

        Args:
            cost_calculator: RE100PremiumCalculator instance
        """
        self.total_cost_calculator = TotalCostCalculator(
            re100_calculator=cost_calculator,
            debug_mode=False
        )

    def calculate_baseline(
        self,
        optimization_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate baseline values (zero-premium cost and baseline carbon)

        This unified method eliminates duplication across optimizers.

        Args:
            optimization_data: Optimization data containing:
                - scenario_df: Material scenario DataFrame
                - material_classification: Material classification info
                - original_df: Original material data

        Returns:
            Dictionary with 'zero_premium_baseline' and 'baseline_carbon' keys

        Raises:
            ValueError: If cost calculator not setup or data missing
        """
        if self.total_cost_calculator is None:
            raise ValueError(
                "Cost calculator not setup. Call setup_cost_calculator() first."
            )

        scenario_df = optimization_data.get('scenario_df')
        material_classification = optimization_data.get('material_classification')
        original_df = optimization_data.get('original_df')

        if scenario_df is None or material_classification is None:
            raise ValueError(
                "optimization_data must contain 'scenario_df' and 'material_classification'"
            )

        # Calculate zero-premium baseline cost
        self.zero_premium_baseline = self.total_cost_calculator.calculate_zero_premium_baseline(
            scenario_df,
            material_classification,
            original_df
        )

        # Calculate baseline carbon
        self.baseline_carbon = sum(
            info['original_emission'] * info['quantity']
            for info in material_classification.values()
        )

        if self.baseline_carbon <= 0:
            raise ValueError(
                f"Invalid baseline carbon: {self.baseline_carbon}. "
                "Check that material_classification contains valid emission data."
            )

        return {
            'zero_premium_baseline': self.zero_premium_baseline,
            'baseline_carbon': self.baseline_carbon
        }

    @abstractmethod
    def run_optimization(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run Pareto optimization (subclass implementation)

        Each subclass must implement this method with its specific algorithm.

        Args:
            optimization_data: Optimization data
            cost_calculator: RE100PremiumCalculator instance
            **kwargs: Method-specific parameters

        Returns:
            List of Pareto points (results)
        """
        pass

    def filter_pareto_frontier(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        filter_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter results to get Pareto frontier

        This method will use the ParetoFilter class (to be implemented)
        for configurable filtering with dominance, quality, and similarity checks.

        Args:
            results: Results to filter (defaults to self.results)
            filter_config: Filter configuration (defaults to config from file)
                - dominance_enabled: bool (default True)
                - quality_threshold: float (default 1.0)
                - similarity_threshold: float (default 0.01)

        Returns:
            Filtered Pareto frontier points
        """
        if results is None:
            results = self.results

        if not results:
            return []

        # Get filter config from file if not provided
        if filter_config is None:
            filter_config = self.config_loader.config.get('pareto_filter', {})

        # Import ParetoFilter (will be implemented next)
        try:
            from .pareto_filter import ParetoFilter
            pareto_filter = ParetoFilter(filter_config)
            filtered_results = pareto_filter.apply(results)
        except ImportError:
            # Fallback to basic dominance filtering if ParetoFilter not yet implemented
            print("⚠️  ParetoFilter not available, using basic dominance filtering")
            filtered_results = self._basic_dominance_filter(results)

        return filtered_results

    def _basic_dominance_filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Basic Pareto dominance filtering (fallback)

        Args:
            results: Results to filter

        Returns:
            Non-dominated points
        """
        pareto_points = []

        for candidate in results:
            candidate_carbon = candidate['summary']['total_carbon']
            candidate_cost = candidate['summary'].get('total_cost', 0)

            is_dominated = False

            for other in results:
                if other is candidate:
                    continue

                other_carbon = other['summary']['total_carbon']
                other_cost = other['summary'].get('total_cost', 0)

                # Dominance check (lower is better for both objectives)
                if (other_carbon <= candidate_carbon and other_cost <= candidate_cost) and \
                   (other_carbon < candidate_carbon or other_cost < candidate_cost):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_points.append(candidate)

        print(f"🌟 Pareto frontier (dominance only): {len(pareto_points)}/{len(results)}")

        return pareto_points

    def save_results(
        self,
        method_name: str,
        results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Path]:
        """
        Save results in standardized format

        Args:
            method_name: Name of the optimization method ('epsilon_constraint', 'nsga2', 'weight_sweep')
            results: Results to save (defaults to self.results)

        Returns:
            Dictionary with 'json' and 'csv' keys pointing to saved files
        """
        if results is None:
            results = self.results

        if not results:
            print("⚠️  No results to save")
            return {}

        # Create output directory
        output_dir = Path(
            self.config_loader.config.get('results', {}).get('output_dir', 'log/pareto_results')
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Save full results as JSON
        json_path = output_dir / f"{method_name}_{timestamp}.json"

        simplified_results = []
        for r in results:
            # Extract common fields (method-specific fields handled by subclass)
            simplified = {
                'summary': r.get('summary', {}),
                'timestamp': r.get('timestamp', datetime.now().isoformat()),
                'method': method_name
            }

            # Add method-specific metadata
            if 'epsilon' in r:
                simplified['epsilon'] = r['epsilon']
            if 'weights' in r:
                simplified['weights'] = r['weights']
            if 'rank' in r:
                simplified['rank'] = r['rank']
            if 'crowding_distance' in r:
                simplified['crowding_distance'] = r['crowding_distance']

            # Add baseline costs
            if 'baseline_cost' in r:
                simplified['baseline_cost'] = r['baseline_cost']
            if 'zero_premium_baseline' in r:
                simplified['zero_premium_baseline'] = r['zero_premium_baseline']
            if 'baseline_carbon' in r:
                simplified['baseline_carbon'] = r['baseline_carbon']

            simplified_results.append(simplified)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)

        print(f"   💾 JSON saved: {json_path}")

        # 2. Save summary as CSV
        csv_path = output_dir / f"{method_name}_summary_{timestamp}.csv"

        summary_rows = []
        for r in results:
            row = {
                'total_carbon': r['summary']['total_carbon'],
                'total_cost': r['summary'].get('total_cost', 0),
                'timestamp': r.get('timestamp', '')
            }

            # Add method-specific columns
            if 'epsilon' in r:
                row['epsilon'] = r['epsilon']
                row['baseline_cost'] = r.get('baseline_cost', 0)
                if r.get('baseline_cost', 0) > 0:
                    row['cost_increase_pct'] = (
                        (r['summary'].get('total_cost', 0) / r['baseline_cost'] - 1) * 100
                    )
            elif 'weights' in r:
                row['carbon_weight'] = r['weights'].get('carbon_weight', 0)
                row['cost_weight'] = r['weights'].get('cost_weight', 0)
                row['baseline_cost'] = r.get('baseline_cost', 0)
                if r.get('baseline_cost', 0) > 0:
                    row['cost_increase_pct'] = (
                        (r['summary'].get('total_cost', 0) / r['baseline_cost'] - 1) * 100
                    )
            elif 'rank' in r:
                row['rank'] = r['rank']
                row['crowding_distance'] = r.get('crowding_distance', 0)

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"   💾 CSV saved: {csv_path}")

        return {
            'json': json_path,
            'csv': csv_path
        }

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of current results

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {
                'n_results': 0,
                'n_pareto': 0
            }

        # Calculate Pareto frontier
        pareto_points = self.filter_pareto_frontier()

        # Extract carbon and cost values
        carbons = [r['summary']['total_carbon'] for r in self.results]
        costs = [r['summary'].get('total_cost', 0) for r in self.results]

        return {
            'n_results': len(self.results),
            'n_pareto': len(pareto_points),
            'carbon_range': (min(carbons), max(carbons)) if carbons else (0, 0),
            'cost_range': (min(costs), max(costs)) if costs else (0, 0),
            'zero_premium_baseline': self.zero_premium_baseline,
            'baseline_carbon': self.baseline_carbon
        }

    def clear_results(self) -> None:
        """Clear stored results"""
        self.results = []

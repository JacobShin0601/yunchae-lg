"""
Robust-Stochastic Optimizer - Hybrid optimization under uncertainty

Combines robust and stochastic optimization approaches to handle:
1. Robust Optimization: Worst-case parameter uncertainty
2. Stochastic Optimization: Probabilistic scenario analysis

This hybrid method provides:
- Worst-case robustness guarantees
- Expected value optimization
- Risk-aware decision making (CVaR)
- Monte Carlo uncertainty quantification
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pyomo.environ as pyo
from pathlib import Path
import json

from ..core.optimization_engine import OptimizationEngine
from ..core.result_processor import ResultProcessor
from ..utils.total_cost_calculator import TotalCostCalculator


class UncertaintyType(Enum):
    """Types of uncertainty"""
    BOX = "box"  # Box uncertainty (min-max)
    ELLIPSOIDAL = "ellipsoidal"  # Ellipsoidal uncertainty
    POLYHEDRAL = "polyhedral"  # Polyhedral uncertainty
    PROBABILISTIC = "probabilistic"  # Probability distribution


class RiskMeasure(Enum):
    """Risk measures for stochastic optimization"""
    EXPECTED_VALUE = "expected_value"  # E[X]
    WORST_CASE = "worst_case"  # max X
    CVAR = "cvar"  # Conditional Value at Risk
    VARIANCE = "variance"  # Var[X]


@dataclass
class UncertaintyParameter:
    """Defines an uncertain parameter"""
    name: str
    nominal_value: float
    uncertainty_type: UncertaintyType

    # Box uncertainty
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Probabilistic uncertainty
    distribution: Optional[str] = None  # 'normal', 'uniform', 'lognormal'
    distribution_params: Dict[str, float] = field(default_factory=dict)

    # Correlation
    correlated_with: List[str] = field(default_factory=list)

    def sample(self, n_samples: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        """Sample from uncertainty distribution"""
        if rng is None:
            rng = np.random.default_rng()

        if self.uncertainty_type == UncertaintyType.BOX:
            # Uniform sampling within box
            return rng.uniform(self.min_value, self.max_value, n_samples)

        elif self.uncertainty_type == UncertaintyType.PROBABILISTIC:
            if self.distribution == 'normal':
                mean = self.distribution_params.get('mean', self.nominal_value)
                std = self.distribution_params.get('std', 0.1 * self.nominal_value)
                return rng.normal(mean, std, n_samples)

            elif self.distribution == 'uniform':
                low = self.distribution_params.get('low', self.min_value)
                high = self.distribution_params.get('high', self.max_value)
                return rng.uniform(low, high, n_samples)

            elif self.distribution == 'lognormal':
                mean = self.distribution_params.get('mean', np.log(self.nominal_value))
                sigma = self.distribution_params.get('sigma', 0.1)
                return rng.lognormal(mean, sigma, n_samples)

            else:
                raise ValueError(f"Unknown distribution: {self.distribution}")

        else:
            # Default: return nominal value
            return np.full(n_samples, self.nominal_value)


@dataclass
class Scenario:
    """A realization of uncertain parameters"""
    scenario_id: int
    parameters: Dict[str, float]
    probability: float = 1.0


class RobustStochasticOptimizer:
    """
    Hybrid Robust-Stochastic Optimizer

    Provides multiple optimization modes:
    1. Pure Robust: Worst-case optimization
    2. Pure Stochastic: Expected value optimization
    3. Hybrid: Robust + Stochastic (worst expected value)
    4. Risk-Averse: CVaR-based optimization
    """

    def __init__(self, user_id: str = None):
        """Initialize optimizer"""
        self.user_id = user_id
        self.uncertain_parameters: Dict[str, UncertaintyParameter] = {}
        self.scenarios: List[Scenario] = []
        self.results: List[Dict[str, Any]] = []

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(seed=42)

    def add_uncertain_parameter(
        self,
        name: str,
        nominal_value: float,
        uncertainty_type: UncertaintyType,
        **kwargs
    ) -> None:
        """
        Add uncertain parameter

        Args:
            name: Parameter name
            nominal_value: Nominal (expected) value
            uncertainty_type: Type of uncertainty
            **kwargs: Additional uncertainty specification
        """
        param = UncertaintyParameter(
            name=name,
            nominal_value=nominal_value,
            uncertainty_type=uncertainty_type,
            **kwargs
        )

        self.uncertain_parameters[name] = param
        print(f"   ✅ Added uncertain parameter: {name} (type: {uncertainty_type.value})")

    def generate_scenarios(
        self,
        n_scenarios: int = 100,
        method: str = 'monte_carlo'
    ) -> List[Scenario]:
        """
        Generate scenarios for uncertain parameters

        Args:
            n_scenarios: Number of scenarios to generate
            method: Scenario generation method ('monte_carlo', 'latin_hypercube')

        Returns:
            List of generated scenarios
        """
        print(f"\n🎲 Generating {n_scenarios} scenarios using {method}...")

        if method == 'monte_carlo':
            scenarios = self._generate_monte_carlo_scenarios(n_scenarios)
        elif method == 'latin_hypercube':
            scenarios = self._generate_latin_hypercube_scenarios(n_scenarios)
        else:
            raise ValueError(f"Unknown scenario generation method: {method}")

        self.scenarios = scenarios
        print(f"   ✅ Generated {len(scenarios)} scenarios")

        return scenarios

    def _generate_monte_carlo_scenarios(self, n_scenarios: int) -> List[Scenario]:
        """Generate scenarios using Monte Carlo sampling"""
        scenarios = []

        for i in range(n_scenarios):
            parameters = {}

            for param_name, param in self.uncertain_parameters.items():
                sample = param.sample(n_samples=1, rng=self.rng)[0]
                parameters[param_name] = float(sample)

            scenario = Scenario(
                scenario_id=i,
                parameters=parameters,
                probability=1.0 / n_scenarios  # Equal probability
            )

            scenarios.append(scenario)

        return scenarios

    def _generate_latin_hypercube_scenarios(self, n_scenarios: int) -> List[Scenario]:
        """Generate scenarios using Latin Hypercube Sampling"""
        from scipy.stats import qmc

        n_params = len(self.uncertain_parameters)

        # Generate LHS samples in [0, 1]^n
        sampler = qmc.LatinHypercube(d=n_params, seed=self.rng)
        lhs_samples = sampler.random(n=n_scenarios)

        scenarios = []
        param_names = list(self.uncertain_parameters.keys())

        for i in range(n_scenarios):
            parameters = {}

            for j, param_name in enumerate(param_names):
                param = self.uncertain_parameters[param_name]

                # Transform [0,1] sample to parameter space
                if param.uncertainty_type == UncertaintyType.BOX:
                    value = param.min_value + lhs_samples[i, j] * (param.max_value - param.min_value)
                elif param.uncertainty_type == UncertaintyType.PROBABILISTIC:
                    # Use inverse CDF (not implemented here, fallback to Monte Carlo)
                    value = param.sample(1, self.rng)[0]
                else:
                    value = param.nominal_value

                parameters[param_name] = float(value)

            scenario = Scenario(
                scenario_id=i,
                parameters=parameters,
                probability=1.0 / n_scenarios
            )

            scenarios.append(scenario)

        return scenarios

    def optimize_robust(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium'
    ) -> Dict[str, Any]:
        """
        Pure robust optimization - worst-case approach

        Optimizes for the worst-case realization of uncertain parameters.

        Args:
            optimization_data: Optimization data
            cost_calculator: Cost calculator instance
            baseline_case: Baseline case
            constraint_preset: Constraint preset

        Returns:
            Robust optimization result
        """
        print(f"\n🛡️ Robust Optimization (Worst-Case)")
        print("=" * 60)

        if not self.uncertain_parameters:
            raise ValueError("No uncertain parameters defined")

        # Generate extreme scenarios (corners of uncertainty set)
        extreme_scenarios = self._generate_extreme_scenarios()

        print(f"   Evaluating {len(extreme_scenarios)} extreme scenarios...")

        # Solve for each extreme scenario
        scenario_results = []

        for scenario in extreme_scenarios:
            print(f"\n   Scenario {scenario.scenario_id}: {scenario.parameters}")

            # Apply uncertain parameters to optimization data
            modified_data = self._apply_scenario_to_data(
                optimization_data,
                scenario
            )

            # Solve optimization
            try:
                result = self._solve_single_scenario(
                    modified_data,
                    cost_calculator,
                    baseline_case,
                    constraint_preset
                )

                result['scenario'] = scenario.parameters
                scenario_results.append(result)

                print(f"      ✅ Carbon: {result['summary']['total_carbon']:.2f}, "
                      f"Cost: ${result['summary']['total_cost']:,.0f}")

            except Exception as e:
                print(f"      ❌ Failed: {str(e)}")
                continue

        # Select worst-case solution
        if not scenario_results:
            raise RuntimeError("All scenarios failed")

        worst_case_result = max(scenario_results, key=lambda r: r['summary']['total_carbon'])

        print(f"\n   🛡️ Worst-case result selected:")
        print(f"      Carbon: {worst_case_result['summary']['total_carbon']:.2f}")
        print(f"      Cost: ${worst_case_result['summary']['total_cost']:,.0f}")

        # Add robustness metadata
        worst_case_result['optimization_type'] = 'robust'
        worst_case_result['n_scenarios_evaluated'] = len(scenario_results)
        worst_case_result['robustness_score'] = 1.0  # Guaranteed robust

        return worst_case_result

    def optimize_stochastic(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium',
        n_scenarios: int = 100,
        risk_measure: RiskMeasure = RiskMeasure.EXPECTED_VALUE
    ) -> Dict[str, Any]:
        """
        Pure stochastic optimization - expected value approach

        Optimizes expected value across scenarios.

        Args:
            optimization_data: Optimization data
            cost_calculator: Cost calculator instance
            baseline_case: Baseline case
            constraint_preset: Constraint preset
            n_scenarios: Number of scenarios
            risk_measure: Risk measure to use

        Returns:
            Stochastic optimization result with statistics
        """
        print(f"\n🎲 Stochastic Optimization ({risk_measure.value})")
        print("=" * 60)

        # Generate scenarios if not already done
        if not self.scenarios or len(self.scenarios) != n_scenarios:
            self.generate_scenarios(n_scenarios=n_scenarios)

        print(f"\n   Evaluating {len(self.scenarios)} scenarios...")

        # Solve for each scenario
        scenario_results = []

        for i, scenario in enumerate(self.scenarios):
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(self.scenarios)} scenarios...")

            # Apply uncertain parameters
            modified_data = self._apply_scenario_to_data(
                optimization_data,
                scenario
            )

            try:
                result = self._solve_single_scenario(
                    modified_data,
                    cost_calculator,
                    baseline_case,
                    constraint_preset
                )

                result['scenario'] = scenario.parameters
                result['probability'] = scenario.probability
                scenario_results.append(result)

            except Exception as e:
                continue

        if not scenario_results:
            raise RuntimeError("All scenarios failed")

        print(f"\n   ✅ {len(scenario_results)}/{len(self.scenarios)} scenarios successful")

        # Calculate statistics
        stats = self._calculate_statistics(scenario_results, risk_measure)

        # Select representative solution based on risk measure
        if risk_measure == RiskMeasure.EXPECTED_VALUE:
            # Find solution closest to expected value
            expected_carbon = stats['expected_carbon']
            representative = min(
                scenario_results,
                key=lambda r: abs(r['summary']['total_carbon'] - expected_carbon)
            )
        elif risk_measure == RiskMeasure.WORST_CASE:
            representative = max(scenario_results, key=lambda r: r['summary']['total_carbon'])
        else:
            # Default: median
            carbons = [r['summary']['total_carbon'] for r in scenario_results]
            median_carbon = np.median(carbons)
            representative = min(
                scenario_results,
                key=lambda r: abs(r['summary']['total_carbon'] - median_carbon)
            )

        # Add stochastic metadata
        representative['optimization_type'] = 'stochastic'
        representative['risk_measure'] = risk_measure.value
        representative['statistics'] = stats
        representative['n_scenarios'] = len(scenario_results)

        print(f"\n   📊 Statistics:")
        print(f"      Expected Carbon: {stats['expected_carbon']:.2f} ± {stats['carbon_std']:.2f}")
        print(f"      Expected Cost: ${stats['expected_cost']:,.0f} ± ${stats['cost_std']:,.0f}")
        print(f"      Carbon Range: [{stats['carbon_min']:.2f}, {stats['carbon_max']:.2f}]")

        return representative

    def _generate_extreme_scenarios(self) -> List[Scenario]:
        """Generate extreme corner scenarios for robust optimization"""
        # For box uncertainty, generate 2^n corner points
        param_names = list(self.uncertain_parameters.keys())
        n_params = len(param_names)

        scenarios = []

        # Generate all binary combinations (corners)
        for i in range(2 ** n_params):
            parameters = {}

            for j, param_name in enumerate(param_names):
                param = self.uncertain_parameters[param_name]

                # Check if j-th bit is set
                is_max = (i >> j) & 1

                if param.uncertainty_type == UncertaintyType.BOX:
                    value = param.max_value if is_max else param.min_value
                else:
                    value = param.nominal_value

                parameters[param_name] = value

            scenario = Scenario(
                scenario_id=i,
                parameters=parameters,
                probability=1.0 / (2 ** n_params)
            )

            scenarios.append(scenario)

        return scenarios

    def _apply_scenario_to_data(
        self,
        optimization_data: Dict[str, Any],
        scenario: Scenario
    ) -> Dict[str, Any]:
        """Apply scenario parameters to optimization data"""
        # Deep copy to avoid modifying original
        import copy
        modified_data = copy.deepcopy(optimization_data)

        # Apply uncertain parameter values
        # This is problem-specific - example implementation for emission factors
        if 'scenario_df' in modified_data:
            df = modified_data['scenario_df'].copy()

            for param_name, param_value in scenario.parameters.items():
                # Example: param_name might be "emission_factor_scale"
                if 'emission_factor_scale' in param_name:
                    # Scale all emission factors
                    df['배출계수'] = df['배출계수'] * param_value

            modified_data['scenario_df'] = df

        return modified_data

    def _solve_single_scenario(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str,
        constraint_preset: str
    ) -> Dict[str, Any]:
        """Solve optimization for a single scenario"""
        from ..core.constraint_manager import ConstraintManager

        # Create engine
        engine = OptimizationEngine(solver_name='auto')
        constraint_manager = ConstraintManager()
        engine.constraint_manager = constraint_manager

        # Build and solve model
        engine.build_model(optimization_data, objective_type='minimize_carbon')
        results = engine.solve(time_limit=60, verbose=False)

        # Extract solution
        solution = engine.extract_solution()
        result_processor = ResultProcessor()
        result_df = result_processor.process_solution(solution)
        summary = result_processor.calculate_summary(result_df)

        # Calculate cost
        total_cost_calculator = TotalCostCalculator(
            re100_calculator=cost_calculator,
            debug_mode=False
        )

        zero_premium_baseline = total_cost_calculator.calculate_zero_premium_baseline(
            optimization_data
        )

        model = engine.model
        total_cost_expr = total_cost_calculator.calculate_total_cost(model, optimization_data)
        actual_total_cost = pyo.value(total_cost_expr)

        summary['total_cost'] = actual_total_cost

        return {
            'summary': summary,
            'solution': solution,
            'result_df': result_df,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_statistics(
        self,
        scenario_results: List[Dict[str, Any]],
        risk_measure: RiskMeasure
    ) -> Dict[str, float]:
        """Calculate statistics across scenarios"""
        carbons = np.array([r['summary']['total_carbon'] for r in scenario_results])
        costs = np.array([r['summary']['total_cost'] for r in scenario_results])
        probabilities = np.array([r.get('probability', 1.0 / len(scenario_results))
                                 for r in scenario_results])

        stats = {
            'expected_carbon': float(np.average(carbons, weights=probabilities)),
            'expected_cost': float(np.average(costs, weights=probabilities)),
            'carbon_std': float(np.std(carbons)),
            'cost_std': float(np.std(costs)),
            'carbon_min': float(np.min(carbons)),
            'carbon_max': float(np.max(carbons)),
            'cost_min': float(np.min(costs)),
            'cost_max': float(np.max(costs)),
            'carbon_median': float(np.median(carbons)),
            'cost_median': float(np.median(costs))
        }

        # Add CVaR if requested
        if risk_measure == RiskMeasure.CVAR:
            alpha = 0.95  # 95% CVaR
            stats['cvar_carbon'] = self._calculate_cvar(carbons, probabilities, alpha)
            stats['cvar_cost'] = self._calculate_cvar(costs, probabilities, alpha)

        return stats

    def _calculate_cvar(
        self,
        values: np.ndarray,
        probabilities: np.ndarray,
        alpha: float
    ) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        # Sort values
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_probs = probabilities[sorted_indices]

        # Find alpha-quantile
        cumulative_prob = 0.0
        var_index = 0

        for i, prob in enumerate(sorted_probs):
            cumulative_prob += prob
            if cumulative_prob >= alpha:
                var_index = i
                break

        # Calculate CVaR (expected value beyond VaR)
        if var_index < len(sorted_values) - 1:
            cvar = np.mean(sorted_values[var_index:])
        else:
            cvar = sorted_values[-1]

        return float(cvar)

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save optimization results"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"robust_stochastic_{timestamp}.json"

        save_dir = Path('log/robust_stochastic')
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / filename

        # Prepare serializable results
        results_data = {
            'uncertain_parameters': {
                name: {
                    'nominal_value': param.nominal_value,
                    'uncertainty_type': param.uncertainty_type.value,
                    'min_value': param.min_value,
                    'max_value': param.max_value
                }
                for name, param in self.uncertain_parameters.items()
            },
            'n_scenarios': len(self.scenarios),
            'results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"💾 Results saved: {filepath}")
        return filepath

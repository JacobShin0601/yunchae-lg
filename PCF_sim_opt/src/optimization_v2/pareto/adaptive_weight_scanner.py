"""
Adaptive Weight Scanner - Intelligent weight selection for Pareto optimization

This module implements adaptive weight scanning that intelligently adds weight
combinations in sparse regions of the Pareto front, improving coverage without
unnecessary computation.

Strategy:
1. Coarse initial sweep with few points
2. Identify sparse regions (gaps) in the Pareto front
3. Adaptively add weights in gaps
4. Iteratively refine until desired coverage achieved

Benefits:
- Fewer total optimizations than dense uniform sweep
- Better coverage of non-convex regions
- Automatic adaptation to problem characteristics
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from .config_loader import WeightCombination


@dataclass
class ParetoGap:
    """Represents a gap in the Pareto front"""
    point1_idx: int
    point2_idx: int
    distance: float
    carbon1: float
    carbon2: float
    cost1: float
    cost2: float
    suggested_weight: float  # Suggested carbon weight to fill this gap


class AdaptiveWeightScanner:
    """
    Adaptive weight scanner for efficient Pareto front exploration

    Uses iterative refinement to add weights only where needed,
    achieving better coverage with fewer optimization runs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adaptive scanner

        Args:
            config: Configuration dict with:
                - initial_points: Number of initial sweep points (default: 5)
                - max_iterations: Maximum refinement iterations (default: 3)
                - gap_threshold: Normalized distance threshold for gaps (default: 0.15)
                - min_points_per_iteration: Minimum points to add per iteration (default: 2)
                - max_points_per_iteration: Maximum points to add per iteration (default: 5)
        """
        if config is None:
            config = {}

        self.initial_points = config.get('initial_points', 5)
        self.max_iterations = config.get('max_iterations', 3)
        self.gap_threshold = config.get('gap_threshold', 0.15)
        self.min_points_per_iteration = config.get('min_points_per_iteration', 2)
        self.max_points_per_iteration = config.get('max_points_per_iteration', 5)

        # History tracking
        self.iteration_history = []
        self.all_weights_tried = []
        self.current_iteration = 0

    def generate_initial_weights(self) -> List[WeightCombination]:
        """
        Generate initial coarse weight sweep

        Returns:
            List of initial weight combinations
        """
        weights = []

        for i in range(self.initial_points):
            carbon_weight = i / (self.initial_points - 1)
            cost_weight = 1.0 - carbon_weight
            weights.append(WeightCombination(carbon_weight, cost_weight))

        print(f"🎯 Initial sweep: {len(weights)} weight combinations")
        self.all_weights_tried.extend(weights)

        return weights

    def analyze_pareto_gaps(
        self,
        results: List[Dict[str, Any]]
    ) -> List[ParetoGap]:
        """
        Analyze Pareto front to identify gaps

        Args:
            results: Current optimization results

        Returns:
            List of identified gaps, sorted by distance (largest first)
        """
        if len(results) < 2:
            return []

        # Extract objectives
        objectives = []
        for r in results:
            carbon = r['summary']['total_carbon']
            cost = r['summary'].get('total_cost', 0)
            objectives.append((carbon, cost))

        # Sort by carbon (for sequential gap detection)
        sorted_indices = sorted(range(len(objectives)), key=lambda i: objectives[i][0])
        sorted_objectives = [objectives[i] for i in sorted_indices]

        # Normalize objectives for distance calculation
        carbons = np.array([o[0] for o in sorted_objectives])
        costs = np.array([o[1] for o in sorted_objectives])

        carbon_range = carbons.max() - carbons.min()
        cost_range = costs.max() - costs.min()

        if carbon_range == 0 or cost_range == 0:
            return []

        normalized_objectives = np.column_stack([
            (carbons - carbons.min()) / carbon_range,
            (costs - costs.min()) / cost_range
        ])

        # Detect gaps between consecutive points
        gaps = []
        for i in range(len(normalized_objectives) - 1):
            p1 = normalized_objectives[i]
            p2 = normalized_objectives[i + 1]

            # Euclidean distance in normalized space
            distance = np.linalg.norm(p2 - p1)

            # Get original weights if available
            weight1 = results[sorted_indices[i]].get('weights', {}).get('carbon_weight', None)
            weight2 = results[sorted_indices[i + 1]].get('weights', {}).get('carbon_weight', None)

            # Suggest midpoint weight
            if weight1 is not None and weight2 is not None:
                suggested_weight = (weight1 + weight2) / 2
            else:
                # Fallback: interpolate based on carbon values
                carbon_pos = (carbons[i] + carbons[i + 1]) / 2
                suggested_weight = (carbon_pos - carbons.min()) / carbon_range if carbon_range > 0 else 0.5

            gap = ParetoGap(
                point1_idx=sorted_indices[i],
                point2_idx=sorted_indices[i + 1],
                distance=distance,
                carbon1=sorted_objectives[i][0],
                carbon2=sorted_objectives[i + 1][0],
                cost1=sorted_objectives[i][1],
                cost2=sorted_objectives[i + 1][1],
                suggested_weight=suggested_weight
            )

            gaps.append(gap)

        # Sort by distance (largest gaps first)
        gaps.sort(key=lambda g: g.distance, reverse=True)

        return gaps

    def select_adaptive_weights(
        self,
        current_results: List[Dict[str, Any]]
    ) -> Tuple[List[WeightCombination], List[ParetoGap]]:
        """
        Select new weights to fill identified gaps

        Args:
            current_results: Current optimization results

        Returns:
            Tuple of (new weight combinations, identified gaps)
        """
        gaps = self.analyze_pareto_gaps(current_results)

        if not gaps:
            print("   ℹ️  No significant gaps detected")
            return [], []

        # Filter gaps above threshold
        significant_gaps = [g for g in gaps if g.distance >= self.gap_threshold]

        if not significant_gaps:
            print(f"   ℹ️  No gaps above threshold ({self.gap_threshold:.2f})")
            return [], gaps

        # Select top gaps (up to max_points_per_iteration)
        num_gaps_to_fill = min(
            len(significant_gaps),
            self.max_points_per_iteration
        )

        # Ensure minimum points if significant gaps exist
        if num_gaps_to_fill < self.min_points_per_iteration and len(significant_gaps) >= self.min_points_per_iteration:
            num_gaps_to_fill = self.min_points_per_iteration

        selected_gaps = significant_gaps[:num_gaps_to_fill]

        # Generate new weights
        new_weights = []
        for gap in selected_gaps:
            carbon_weight = gap.suggested_weight
            cost_weight = 1.0 - carbon_weight

            # Check if weight already tried (avoid duplicates)
            if not self._is_weight_duplicate(carbon_weight):
                weight = WeightCombination(carbon_weight, cost_weight)
                new_weights.append(weight)
                self.all_weights_tried.append(weight)

        print(f"   📍 Identified {len(gaps)} gaps, {len(significant_gaps)} significant")
        print(f"   ➕ Adding {len(new_weights)} new weight combinations")

        return new_weights, selected_gaps

    def _is_weight_duplicate(
        self,
        carbon_weight: float,
        tolerance: float = 0.01
    ) -> bool:
        """
        Check if weight is duplicate (within tolerance)

        Args:
            carbon_weight: Carbon weight to check
            tolerance: Tolerance for considering weights identical

        Returns:
            True if duplicate, False otherwise
        """
        for tried_weight in self.all_weights_tried:
            if abs(tried_weight.carbon_weight - carbon_weight) < tolerance:
                return True
        return False

    def should_continue_refinement(
        self,
        gaps: List[ParetoGap],
        iteration: int
    ) -> bool:
        """
        Determine if refinement should continue

        Args:
            gaps: Current identified gaps
            iteration: Current iteration number

        Returns:
            True if should continue, False otherwise
        """
        # Max iterations reached
        if iteration >= self.max_iterations:
            print(f"   🏁 Max iterations ({self.max_iterations}) reached")
            return False

        # No significant gaps
        significant_gaps = [g for g in gaps if g.distance >= self.gap_threshold]
        if not significant_gaps:
            print(f"   🏁 No significant gaps remaining")
            return False

        return True

    def run_adaptive_scan(
        self,
        optimizer,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium',
        scenario_template: str = None
    ) -> List[Dict[str, Any]]:
        """
        Run complete adaptive weight scanning process

        Args:
            optimizer: WeightSweepOptimizer instance
            optimization_data: Optimization data
            cost_calculator: Cost calculator instance
            baseline_case: Baseline case
            constraint_preset: Constraint preset
            scenario_template: Scenario template

        Returns:
            All optimization results from adaptive scan
        """
        print(f"\n🎯 Adaptive Weight Scanning")
        print("=" * 60)

        all_results = []

        # Phase 1: Initial coarse sweep
        print(f"\n📍 Phase 1: Initial Coarse Sweep")
        print("-" * 60)

        initial_weights = self.generate_initial_weights()

        for idx, weight in enumerate(initial_weights, 1):
            print(f"\n[{idx}/{len(initial_weights)}] Weight: {weight}")

            try:
                result = optimizer._run_single_optimization(
                    weight,
                    optimization_data,
                    cost_calculator,
                    baseline_case,
                    constraint_preset,
                    scenario_template
                )

                all_results.append(result)
                print(f"   ✅ Complete - Carbon: {result['summary']['total_carbon']:.2f}, "
                      f"Cost: ${result['summary'].get('total_cost', 0):,.0f}")

            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue

        self.iteration_history.append({
            'iteration': 0,
            'phase': 'initial',
            'weights_added': len(initial_weights),
            'total_points': len(all_results)
        })

        # Phase 2: Adaptive refinement iterations
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n📍 Phase 2: Adaptive Refinement (Iteration {iteration})")
            print("-" * 60)

            # Analyze current results
            new_weights, gaps = self.select_adaptive_weights(all_results)

            if not new_weights:
                print(f"   🏁 No new weights needed, stopping refinement")
                break

            # Check convergence
            if not self.should_continue_refinement(gaps, iteration):
                break

            # Optimize with new weights
            print(f"\n   Optimizing {len(new_weights)} new weight combinations...")

            iteration_results = []
            for idx, weight in enumerate(new_weights, 1):
                print(f"\n   [{idx}/{len(new_weights)}] Weight: {weight}")

                try:
                    result = optimizer._run_single_optimization(
                        weight,
                        optimization_data,
                        cost_calculator,
                        baseline_case,
                        constraint_preset,
                        scenario_template
                    )

                    iteration_results.append(result)
                    all_results.append(result)

                    print(f"      ✅ Complete - Carbon: {result['summary']['total_carbon']:.2f}, "
                          f"Cost: ${result['summary'].get('total_cost', 0):,.0f}")

                except Exception as e:
                    print(f"      ❌ Failed: {str(e)}")
                    continue

            self.iteration_history.append({
                'iteration': iteration,
                'phase': 'refinement',
                'weights_added': len(new_weights),
                'total_points': len(all_results),
                'largest_gap': gaps[0].distance if gaps else 0
            })

            print(f"\n   ✅ Iteration {iteration} complete: {len(iteration_results)}/{len(new_weights)} successful")

        # Summary
        print(f"\n" + "=" * 60)
        print(f"✅ Adaptive Scan Complete")
        print(f"   Total iterations: {len(self.iteration_history)}")
        print(f"   Total weight combinations tried: {len(self.all_weights_tried)}")
        print(f"   Successful optimizations: {len(all_results)}")
        print("=" * 60)

        return all_results

    def get_scan_summary(self) -> Dict[str, Any]:
        """
        Get summary of adaptive scan process

        Returns:
            Summary dictionary
        """
        return {
            'total_iterations': len(self.iteration_history),
            'total_weights_tried': len(self.all_weights_tried),
            'iteration_history': self.iteration_history,
            'config': {
                'initial_points': self.initial_points,
                'max_iterations': self.max_iterations,
                'gap_threshold': self.gap_threshold,
                'min_points_per_iteration': self.min_points_per_iteration,
                'max_points_per_iteration': self.max_points_per_iteration
            }
        }

    def print_summary(self) -> None:
        """Print formatted summary of adaptive scan"""
        summary = self.get_scan_summary()

        print("\n" + "=" * 60)
        print("ADAPTIVE WEIGHT SCAN SUMMARY")
        print("=" * 60)

        print(f"\n📊 Configuration:")
        print(f"   Initial Points: {summary['config']['initial_points']}")
        print(f"   Max Iterations: {summary['config']['max_iterations']}")
        print(f"   Gap Threshold: {summary['config']['gap_threshold']:.2f}")
        print(f"   Points per Iteration: {summary['config']['min_points_per_iteration']}-{summary['config']['max_points_per_iteration']}")

        print(f"\n📈 Results:")
        print(f"   Total Iterations: {summary['total_iterations']}")
        print(f"   Total Weights Tried: {summary['total_weights_tried']}")

        print(f"\n📋 Iteration History:")
        for hist in summary['iteration_history']:
            if hist['phase'] == 'initial':
                print(f"   Phase 1 (Initial): {hist['weights_added']} weights → {hist['total_points']} points")
            else:
                gap_info = f", Gap: {hist['largest_gap']:.3f}" if 'largest_gap' in hist else ""
                print(f"   Phase 2 (Iter {hist['iteration']}): +{hist['weights_added']} weights → {hist['total_points']} points{gap_info}")

        print("=" * 60 + "\n")


def compare_adaptive_vs_uniform(
    adaptive_results: List[Dict[str, Any]],
    uniform_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare adaptive scanning vs uniform scanning

    Args:
        adaptive_results: Results from adaptive scanning
        uniform_results: Results from uniform scanning

    Returns:
        Comparison metrics
    """
    def calculate_coverage(results):
        """Calculate hypervolume (coverage metric)"""
        if not results:
            return 0.0

        objectives = [
            (r['summary']['total_carbon'], r['summary'].get('total_cost', 0))
            for r in results
        ]

        # Simple 2D hypervolume
        sorted_objs = sorted(objectives, key=lambda x: x[0])
        max_carbon = max(o[0] for o in sorted_objs) * 1.1
        max_cost = max(o[1] for o in sorted_objs) * 1.1

        hypervolume = 0.0
        prev_carbon = 0.0

        for carbon, cost in sorted_objs:
            width = carbon - prev_carbon
            height = max_cost - cost
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_carbon = carbon

        return hypervolume

    adaptive_coverage = calculate_coverage(adaptive_results)
    uniform_coverage = calculate_coverage(uniform_results)

    comparison = {
        'adaptive': {
            'n_points': len(adaptive_results),
            'coverage': adaptive_coverage
        },
        'uniform': {
            'n_points': len(uniform_results),
            'coverage': uniform_coverage
        },
        'efficiency': {
            'point_reduction': (len(uniform_results) - len(adaptive_results)) / len(uniform_results) * 100 if uniform_results else 0,
            'coverage_ratio': adaptive_coverage / uniform_coverage if uniform_coverage > 0 else 0
        }
    }

    print(f"\n📊 Adaptive vs Uniform Comparison")
    print("=" * 60)
    print(f"Adaptive: {comparison['adaptive']['n_points']} points, Coverage: {adaptive_coverage:.2e}")
    print(f"Uniform:  {comparison['uniform']['n_points']} points, Coverage: {uniform_coverage:.2e}")
    print(f"Efficiency: {comparison['efficiency']['point_reduction']:.1f}% fewer points, "
          f"{comparison['efficiency']['coverage_ratio']:.1%} coverage")
    print("=" * 60)

    return comparison

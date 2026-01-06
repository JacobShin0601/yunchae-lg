"""
Pareto Method Comparator - Automatic comparison of Pareto optimization methods

This module provides comprehensive comparison between different Pareto optimization approaches:
- Epsilon Constraint Method
- NSGA-II (Genetic Algorithm)
- Weight Sweep Method

Comparison Metrics:
- Coverage: Hypervolume indicator (quality of Pareto front)
- Diversity: Spacing metric (distribution uniformity)
- Quality Score: Overall optimization effectiveness
- Computation Time: Runtime performance
- Number of Points: Solution density
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime


class ParetoMethodComparator:
    """
    Comprehensive comparison system for Pareto optimization methods

    Analyzes and compares multiple Pareto optimization methods to identify
    the best approach for a given problem.
    """

    def __init__(self, reference_point: Optional[Tuple[float, float]] = None):
        """
        Initialize comparator

        Args:
            reference_point: Reference point for hypervolume calculation (carbon, cost)
                           If None, will be computed from worst points
        """
        self.reference_point = reference_point
        self.comparison_results = None

    def compare_methods(
        self,
        epsilon_results: Optional[List[Dict[str, Any]]] = None,
        nsga2_results: Optional[List[Dict[str, Any]]] = None,
        weight_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare all available Pareto methods

        Args:
            epsilon_results: Results from Epsilon Constraint method
            nsga2_results: Results from NSGA-II method
            weight_results: Results from Weight Sweep method

        Returns:
            Comprehensive comparison dictionary with:
            - Individual method analyses
            - Best method selection
            - Detailed recommendations
        """
        print("\n📊 Pareto Method Comparison")
        print("=" * 60)

        comparison = {
            'timestamp': datetime.now().isoformat(),
            'methods': {},
            'best_method': None,
            'recommendation': None,
            'comparison_matrix': {}
        }

        # Analyze each method
        if epsilon_results:
            print("\n🔵 Analyzing Epsilon Constraint...")
            comparison['methods']['epsilon_constraint'] = self._analyze_method(
                epsilon_results, 'epsilon_constraint'
            )

        if nsga2_results:
            print("\n🟢 Analyzing NSGA-II...")
            comparison['methods']['nsga2'] = self._analyze_method(
                nsga2_results, 'nsga2'
            )

        if weight_results:
            print("\n🟡 Analyzing Weight Sweep...")
            comparison['methods']['weight_sweep'] = self._analyze_method(
                weight_results, 'weight_sweep'
            )

        # Select best method
        if comparison['methods']:
            comparison['best_method'] = self._select_best_method(comparison['methods'])
            comparison['recommendation'] = self._generate_recommendation(comparison)
            comparison['comparison_matrix'] = self._create_comparison_matrix(comparison['methods'])

            print("\n" + "=" * 60)
            print(f"🏆 Best Method: {comparison['best_method']}")
            print(f"📝 {comparison['recommendation']}")
            print("=" * 60)

        self.comparison_results = comparison
        return comparison

    def _analyze_method(
        self,
        results: List[Dict[str, Any]],
        method_name: str
    ) -> Dict[str, Any]:
        """
        Analyze individual Pareto method performance

        Args:
            results: Method results
            method_name: Name of the method

        Returns:
            Analysis dictionary with metrics
        """
        if not results:
            return {
                'n_points': 0,
                'coverage': 0.0,
                'diversity': 0.0,
                'quality_score': 0.0,
                'computation_time': 0.0,
                'valid': False
            }

        # Extract objectives
        objectives = []
        for r in results:
            carbon = r['summary']['total_carbon']
            cost = r['summary'].get('total_cost', 0)
            objectives.append((carbon, cost))

        # Calculate metrics
        n_points = len(objectives)
        coverage = self._calculate_hypervolume(objectives)
        diversity = self._calculate_spacing(objectives)
        quality_score = self._calculate_quality_score(results)
        computation_time = self._estimate_computation_time(results)

        analysis = {
            'n_points': n_points,
            'coverage': coverage,
            'diversity': diversity,
            'quality_score': quality_score,
            'computation_time': computation_time,
            'carbon_range': (min(o[0] for o in objectives), max(o[0] for o in objectives)),
            'cost_range': (min(o[1] for o in objectives), max(o[1] for o in objectives)),
            'valid': True
        }

        # Print summary
        print(f"   Points: {n_points}")
        print(f"   Coverage (Hypervolume): {coverage:.2e}")
        print(f"   Diversity (Spacing): {diversity:.4f}")
        print(f"   Quality Score: {quality_score:.2f}")
        print(f"   Est. Time: {computation_time:.1f}s")

        return analysis

    def _calculate_hypervolume(self, objectives: List[Tuple[float, float]]) -> float:
        """
        Calculate hypervolume indicator (coverage metric)

        Hypervolume measures the volume of objective space dominated by the Pareto front.
        Higher is better.

        Args:
            objectives: List of (carbon, cost) tuples

        Returns:
            Hypervolume value
        """
        if not objectives:
            return 0.0

        # Determine reference point if not provided
        if self.reference_point is None:
            max_carbon = max(o[0] for o in objectives) * 1.1
            max_cost = max(o[1] for o in objectives) * 1.1
            ref_point = (max_carbon, max_cost)
        else:
            ref_point = self.reference_point

        # Sort by first objective (carbon)
        sorted_objs = sorted(objectives, key=lambda x: x[0])

        # Calculate 2D hypervolume using slicing method
        hypervolume = 0.0
        prev_carbon = 0.0

        for carbon, cost in sorted_objs:
            if carbon > ref_point[0] or cost > ref_point[1]:
                continue  # Point outside reference

            width = carbon - prev_carbon
            height = ref_point[1] - cost

            if width > 0 and height > 0:
                hypervolume += width * height

            prev_carbon = carbon

        return hypervolume

    def _calculate_spacing(self, objectives: List[Tuple[float, float]]) -> float:
        """
        Calculate spacing metric (diversity measure)

        Spacing measures how evenly distributed the Pareto points are.
        Lower is better (more uniform distribution).

        Args:
            objectives: List of (carbon, cost) tuples

        Returns:
            Spacing value (0 = perfectly uniform)
        """
        if len(objectives) < 2:
            return 0.0

        # Normalize objectives to [0, 1] range
        carbons = np.array([o[0] for o in objectives])
        costs = np.array([o[1] for o in objectives])

        carbon_range = carbons.max() - carbons.min()
        cost_range = costs.max() - costs.min()

        if carbon_range == 0 or cost_range == 0:
            return 0.0

        normalized = np.column_stack([
            (carbons - carbons.min()) / carbon_range,
            (costs - costs.min()) / cost_range
        ])

        # Calculate distances to nearest neighbors
        n = len(normalized)
        distances = []

        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(normalized[i] - normalized[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)

        # Calculate spacing metric
        mean_dist = np.mean(distances)
        spacing = np.sqrt(np.sum((np.array(distances) - mean_dist) ** 2) / n)

        return spacing

    def _calculate_quality_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall quality score

        Combines multiple factors:
        - Carbon reduction effectiveness
        - Cost efficiency
        - Solution feasibility

        Args:
            results: Method results

        Returns:
            Quality score (0-100)
        """
        if not results:
            return 0.0

        scores = []

        for r in results:
            solution = r.get('solution', {})
            materials = solution.get('materials', {})

            # Check for optimization features
            feature_score = 0
            feature_count = 0

            for mat_name, mat_data in materials.items():
                # Check RE100
                tier1_re = mat_data.get('tier1_re', 0)
                tier2_re = mat_data.get('tier2_re', 0)
                if tier1_re > 0 or tier2_re > 0:
                    feature_score += min(tier1_re + tier2_re, 1.0) * 30
                    feature_count += 1

                # Check recycling
                recycle = mat_data.get('recycle_ratio', 0)
                if recycle > 0:
                    feature_score += min(recycle, 1.0) * 35
                    feature_count += 1

                # Check low-carbon
                low_carbon = mat_data.get('low_carbon_ratio', 0)
                if low_carbon > 0:
                    feature_score += min(low_carbon, 1.0) * 35
                    feature_count += 1

            if feature_count > 0:
                scores.append(feature_score / feature_count)

        if not scores:
            return 50.0  # Default middle score

        return np.mean(scores)

    def _estimate_computation_time(self, results: List[Dict[str, Any]]) -> float:
        """
        Estimate computation time

        Args:
            results: Method results

        Returns:
            Estimated time in seconds
        """
        # This is a rough estimation based on number of points
        # In production, actual timing data should be collected
        n_points = len(results)

        # Rough estimates per method type
        if any('epsilon' in r.get('method', '') for r in results):
            # Epsilon: ~1-2 min per point
            return n_points * 90.0
        elif any('nsga2' in r.get('method', '') for r in results):
            # NSGA-II: depends on generations, estimate ~5-10 min total
            return 450.0
        else:
            # Weight sweep: ~30-60 sec per point
            return n_points * 45.0

    def _select_best_method(self, methods: Dict[str, Dict[str, Any]]) -> str:
        """
        Select the best performing method

        Uses weighted scoring across multiple metrics.

        Args:
            methods: Dictionary of method analyses

        Returns:
            Name of best method
        """
        scores = {}

        for method_name, analysis in methods.items():
            if not analysis.get('valid', False):
                scores[method_name] = 0.0
                continue

            # Weighted scoring
            score = 0.0

            # Coverage (40% weight) - higher is better
            coverage = analysis.get('coverage', 0)
            if coverage > 0:
                score += 40.0 * min(coverage / 1e10, 1.0)  # Normalize

            # Diversity (20% weight) - lower is better
            spacing = analysis.get('diversity', 1.0)
            score += 20.0 * (1.0 - min(spacing, 1.0))

            # Quality (30% weight) - higher is better
            quality = analysis.get('quality_score', 0)
            score += 30.0 * (quality / 100.0)

            # Time efficiency (10% weight) - lower is better
            time = analysis.get('computation_time', 1000)
            score += 10.0 * (1.0 - min(time / 1000.0, 1.0))

            scores[method_name] = score

        # Select highest score
        best_method = max(scores.items(), key=lambda x: x[1])
        return best_method[0]

    def _generate_recommendation(self, comparison: Dict[str, Any]) -> str:
        """
        Generate human-readable recommendation

        Args:
            comparison: Comparison results

        Returns:
            Recommendation text
        """
        best = comparison['best_method']
        methods = comparison['methods']

        if best not in methods:
            return "Insufficient data for recommendation."

        analysis = methods[best]

        # Build recommendation
        rec = f"{best.replace('_', ' ').title()} is recommended based on:\n"

        if analysis['coverage'] > 0:
            rec += f"  • Best coverage (hypervolume: {analysis['coverage']:.2e})\n"

        if analysis['diversity'] < 0.5:
            rec += f"  • Good diversity (spacing: {analysis['diversity']:.4f})\n"

        if analysis['quality_score'] > 70:
            rec += f"  • High quality solutions ({analysis['quality_score']:.1f}/100)\n"

        rec += f"  • {analysis['n_points']} Pareto points found"

        return rec

    def _create_comparison_matrix(self, methods: Dict[str, Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Create comparison matrix for visualization

        Args:
            methods: Method analyses

        Returns:
            Matrix with normalized metrics
        """
        matrix = {
            'methods': list(methods.keys()),
            'n_points': [],
            'coverage': [],
            'diversity': [],
            'quality': [],
            'time': []
        }

        for method_name in matrix['methods']:
            analysis = methods[method_name]

            matrix['n_points'].append(analysis.get('n_points', 0))
            matrix['coverage'].append(analysis.get('coverage', 0))
            matrix['diversity'].append(analysis.get('diversity', 0))
            matrix['quality'].append(analysis.get('quality_score', 0))
            matrix['time'].append(analysis.get('computation_time', 0))

        return matrix

    def get_summary_report(self) -> str:
        """
        Get formatted summary report

        Returns:
            Formatted text report
        """
        if not self.comparison_results:
            return "No comparison results available. Run compare_methods() first."

        report = "\n" + "=" * 70 + "\n"
        report += "PARETO METHOD COMPARISON REPORT\n"
        report += "=" * 70 + "\n\n"

        methods = self.comparison_results.get('methods', {})

        for method_name, analysis in methods.items():
            if not analysis.get('valid', False):
                continue

            report += f"\n{method_name.upper().replace('_', ' ')}:\n"
            report += "-" * 70 + "\n"
            report += f"  Points Generated: {analysis['n_points']}\n"
            report += f"  Coverage (Hypervolume): {analysis['coverage']:.2e}\n"
            report += f"  Diversity (Spacing): {analysis['diversity']:.4f}\n"
            report += f"  Quality Score: {analysis['quality_score']:.1f}/100\n"
            report += f"  Est. Computation Time: {analysis['computation_time']:.1f}s\n"
            report += f"  Carbon Range: {analysis['carbon_range'][0]:.2f} - {analysis['carbon_range'][1]:.2f}\n"
            report += f"  Cost Range: ${analysis['cost_range'][0]:,.0f} - ${analysis['cost_range'][1]:,.0f}\n"

        report += "\n" + "=" * 70 + "\n"
        report += f"🏆 RECOMMENDED METHOD: {self.comparison_results['best_method']}\n"
        report += "=" * 70 + "\n"
        report += self.comparison_results['recommendation'] + "\n"
        report += "=" * 70 + "\n"

        return report

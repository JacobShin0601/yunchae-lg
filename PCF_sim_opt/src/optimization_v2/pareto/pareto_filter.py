"""
Pareto Filter - Configurable multi-stage filtering system for Pareto frontiers

This module provides a flexible filtering system that replaces hardcoded thresholds
with user-configurable parameters.

Filtering Stages:
1. Dominance filtering - Remove dominated solutions
2. Quality filtering - Remove solutions with insufficient optimization
3. Similarity filtering - Remove duplicate/similar solutions
"""

from typing import List, Dict, Any, Optional


class ParetoFilter:
    """
    Configurable Pareto frontier filtering system

    Applies up to three stages of filtering:
    1. Dominance: Remove solutions dominated by others
    2. Quality: Remove solutions with low optimization feature usage
    3. Similarity: Remove solutions too similar to existing ones

    All thresholds are configurable via the config dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize filter with configuration

        Args:
            config: Filter configuration dict with keys:
                - dominance_enabled: bool (default True) - Apply dominance filtering
                - quality_enabled: bool (default True) - Apply quality filtering
                - similarity_enabled: bool (default True) - Apply similarity filtering
                - quality_threshold: float (default 1.0) - Minimum % for optimization features
                - similarity_threshold: float (default 0.01) - Relative similarity threshold (1%)
                - similarity_metrics: List[str] (default ['carbon', 'cost']) - Metrics for similarity
        """
        if config is None:
            config = {}

        # Stage enablement
        self.dominance_enabled = config.get('dominance_enabled', True)
        self.quality_enabled = config.get('quality_enabled', True)
        self.similarity_enabled = config.get('similarity_enabled', True)

        # Thresholds (now configurable!)
        self.quality_threshold = config.get('quality_threshold', 1.0)  # 1% default
        self.similarity_threshold = config.get('similarity_threshold', 0.01)  # 1% default

        # Similarity metrics (which objectives to compare)
        self.similarity_metrics = config.get('similarity_metrics', ['carbon', 'cost'])

        # Debug mode
        self.debug = config.get('debug', False)

    def apply(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply all enabled filtering stages

        Args:
            results: List of optimization results to filter

        Returns:
            Filtered results (Pareto frontier)
        """
        if not results:
            return []

        print(f"\n🔍 Pareto Filtering: {len(results)} initial results")

        filtered = results.copy()

        # Stage 1: Dominance filtering
        if self.dominance_enabled:
            filtered = self._filter_dominated(filtered)
            print(f"   ✅ After dominance: {len(filtered)}/{len(results)}")

        # Stage 2: Quality filtering
        if self.quality_enabled:
            before_quality = len(filtered)
            filtered = self._filter_quality(filtered)
            print(f"   ✅ After quality: {len(filtered)}/{before_quality}")

        # Stage 3: Similarity filtering
        if self.similarity_enabled:
            before_similarity = len(filtered)
            filtered = self._filter_similarity(filtered)
            print(f"   ✅ After similarity: {len(filtered)}/{before_similarity}")

        print(f"\n🌟 Final Pareto frontier: {len(filtered)}/{len(results)} points")

        return filtered

    def _filter_dominated(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out dominated solutions

        A solution is dominated if another solution is better in all objectives
        and strictly better in at least one.

        Args:
            results: Results to filter

        Returns:
            Non-dominated solutions
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

                # Check dominance (lower is better for both)
                carbon_better_or_equal = other_carbon <= candidate_carbon
                cost_better_or_equal = other_cost <= candidate_cost
                at_least_one_strictly_better = (
                    other_carbon < candidate_carbon or
                    other_cost < candidate_cost
                )

                if carbon_better_or_equal and cost_better_or_equal and at_least_one_strictly_better:
                    is_dominated = True
                    if self.debug:
                        print(f"      Dominated: C={candidate_carbon:.2f}, $={candidate_cost:.2f}")
                    break

            if not is_dominated:
                pareto_points.append(candidate)

        return pareto_points

    def _filter_quality(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out solutions with insufficient optimization

        Removes solutions where optimization features (RE100, recycling, low-carbon)
        are barely used, indicating minimal optimization effort.

        Args:
            results: Results to filter

        Returns:
            Results with sufficient optimization quality
        """
        quality_filtered = []

        for point in results:
            # Check if solution has sufficient optimization features
            if self._check_optimization_quality(point):
                quality_filtered.append(point)
            else:
                if self.debug:
                    identifier = point.get('epsilon', point.get('weights', 'Unknown'))
                    print(f"      ⚠️  Quality filter: {identifier} removed (low feature usage)")

        return quality_filtered

    def _check_optimization_quality(self, point: Dict[str, Any]) -> bool:
        """
        Check if a solution has sufficient optimization quality

        Args:
            point: Single optimization result

        Returns:
            True if quality is sufficient, False otherwise
        """
        solution = point.get('solution', {})
        materials = solution.get('materials', {})

        # Find cathode materials (most important for optimization)
        cathode_materials = {
            name: data for name, data in materials.items()
            if 'Cathode' in name or '양극' in name or 'CAM' in name
        }

        if not cathode_materials:
            # No cathode found - assume general material (include by default)
            return True

        # Check if any optimization features are above threshold
        has_optimization = False

        for mat_name, mat_data in cathode_materials.items():
            # Extract optimization features (convert to percentage)
            tier1_re = mat_data.get('tier1_re', 0) * 100
            tier2_re = mat_data.get('tier2_re', 0) * 100
            recycle_ratio = mat_data.get('recycle_ratio', 0) * 100
            low_carbon_ratio = mat_data.get('low_carbon_ratio', 0) * 100

            # Check if any feature exceeds threshold
            if (tier1_re > self.quality_threshold or
                tier2_re > self.quality_threshold or
                recycle_ratio > self.quality_threshold or
                low_carbon_ratio > self.quality_threshold):
                has_optimization = True
                break

        return has_optimization

    def _filter_similarity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out similar/duplicate solutions

        Removes solutions that are too similar to already-selected ones
        based on relative differences in objectives.

        Args:
            results: Results to filter

        Returns:
            Deduplicated results
        """
        if len(results) <= 1:
            return results

        deduplicated = []

        for candidate in results:
            if not self._is_similar_to_existing(candidate, deduplicated):
                deduplicated.append(candidate)
            else:
                if self.debug:
                    identifier = candidate.get('epsilon', candidate.get('weights', 'Unknown'))
                    print(f"      ⚠️  Similarity filter: {identifier} removed (similar to existing)")

        return deduplicated

    def _is_similar_to_existing(
        self,
        candidate: Dict[str, Any],
        existing: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if candidate is similar to any existing solution

        Args:
            candidate: Candidate solution to check
            existing: Already-selected solutions

        Returns:
            True if candidate is similar to any existing solution
        """
        candidate_carbon = candidate['summary']['total_carbon']
        candidate_cost = candidate['summary'].get('total_cost', 0)

        for existing_point in existing:
            existing_carbon = existing_point['summary']['total_carbon']
            existing_cost = existing_point['summary'].get('total_cost', 0)

            # Calculate relative differences
            if existing_carbon > 0 and existing_cost > 0:
                carbon_diff = abs(candidate_carbon - existing_carbon) / existing_carbon
                cost_diff = abs(candidate_cost - existing_cost) / existing_cost

                # Check if both differences are below threshold
                if carbon_diff < self.similarity_threshold and cost_diff < self.similarity_threshold:
                    return True  # Too similar

        return False  # Not similar to any existing point

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current filter configuration

        Returns:
            Dictionary with configuration details
        """
        return {
            'dominance_enabled': self.dominance_enabled,
            'quality_enabled': self.quality_enabled,
            'similarity_enabled': self.similarity_enabled,
            'quality_threshold': self.quality_threshold,
            'similarity_threshold': self.similarity_threshold,
            'similarity_metrics': self.similarity_metrics,
            'debug': self.debug
        }

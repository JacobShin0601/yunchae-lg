"""
Pareto Method Recommender - Intelligent method selection based on problem characteristics

This module provides AI-based recommendations for selecting the optimal Pareto
optimization method based on problem characteristics and user preferences.

Selection Criteria:
- Problem size (number of materials)
- Problem complexity
- Time constraints
- User preferences (quality vs speed)
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum


class ProblemComplexity(Enum):
    """Problem complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ParetoMethodRecommender:
    """
    Intelligent Pareto method recommendation system

    Analyzes problem characteristics and recommends the most suitable
    Pareto optimization approach.
    """

    def __init__(self):
        """Initialize recommender"""
        # Method characteristics database
        self.method_profiles = {
            'epsilon_constraint': {
                'best_for': ['small_problems', 'thorough_search', 'guaranteed_pareto'],
                'strengths': ['Complete coverage', 'Systematic exploration', 'High quality'],
                'weaknesses': ['Slow for large problems', 'Many infeasible solves'],
                'time_per_point': 90.0,  # seconds
                'scalability': 'poor',
                'quality': 'excellent'
            },
            'nsga2': {
                'best_for': ['large_problems', 'complex_constraints', 'exploration'],
                'strengths': ['Handles complexity well', 'Parallel-friendly', 'Robust'],
                'weaknesses': ['Stochastic', 'Requires tuning', 'Convergence uncertain'],
                'time_per_generation': 30.0,  # seconds
                'scalability': 'excellent',
                'quality': 'good'
            },
            'weight_sweep': {
                'best_for': ['quick_results', 'simple_problems', 'initial_exploration'],
                'strengths': ['Fast', 'Reliable', 'Simple to use'],
                'weaknesses': ['May miss non-convex regions', 'Limited coverage'],
                'time_per_point': 45.0,  # seconds
                'scalability': 'good',
                'quality': 'good'
            }
        }

    def recommend(
        self,
        problem_characteristics: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend optimal Pareto method

        Args:
            problem_characteristics: Dict with keys:
                - n_materials: Number of materials to optimize
                - complexity: Problem complexity level (str or ProblemComplexity)
                - time_limit_min: Available time in minutes
                - has_integer_vars: Boolean for integer variables
                - constraint_types: List of constraint types
            user_preferences: Optional dict with:
                - priority: 'quality' or 'speed'
                - risk_tolerance: 'conservative' or 'aggressive'

        Returns:
            Recommendation dict with:
            - recommended_method: Method name
            - confidence: Confidence score (0-1)
            - scores: Individual scores for each method
            - reasoning: Explanation
            - alternatives: Alternative methods
        """
        # Extract problem characteristics
        n_materials = problem_characteristics.get('n_materials', 0)
        complexity = problem_characteristics.get('complexity', 'medium')
        time_limit = problem_characteristics.get('time_limit_min', 30)
        has_integer = problem_characteristics.get('has_integer_vars', False)

        # Parse user preferences
        if user_preferences is None:
            user_preferences = {}

        priority = user_preferences.get('priority', 'balanced')
        risk_tolerance = user_preferences.get('risk_tolerance', 'moderate')

        print("\n🤖 Pareto Method Recommender")
        print("=" * 60)
        print(f"Problem Characteristics:")
        print(f"  • Materials: {n_materials}")
        print(f"  • Complexity: {complexity}")
        print(f"  • Time Limit: {time_limit} min")
        print(f"  • Integer Variables: {has_integer}")
        print(f"User Preferences:")
        print(f"  • Priority: {priority}")
        print(f"  • Risk Tolerance: {risk_tolerance}")
        print()

        # Calculate scores for each method
        scores = {
            'epsilon_constraint': self._score_epsilon(
                n_materials, complexity, time_limit, priority, has_integer
            ),
            'nsga2': self._score_nsga2(
                n_materials, complexity, time_limit, priority, has_integer
            ),
            'weight_sweep': self._score_weight_sweep(
                n_materials, complexity, time_limit, priority, has_integer
            )
        }

        # Select best method
        best_method = max(scores.items(), key=lambda x: x[1])
        recommended_method = best_method[0]
        confidence = best_method[1]

        # Get alternatives (methods within 10% of best score)
        threshold = confidence * 0.9
        alternatives = [
            method for method, score in scores.items()
            if score >= threshold and method != recommended_method
        ]

        # Generate reasoning
        reasoning = self._explain_recommendation(
            recommended_method,
            problem_characteristics,
            user_preferences,
            scores
        )

        recommendation = {
            'recommended_method': recommended_method,
            'confidence': confidence,
            'scores': scores,
            'reasoning': reasoning,
            'alternatives': alternatives,
            'method_profiles': {
                method: self.method_profiles[method]
                for method in [recommended_method] + alternatives
            }
        }

        # Print recommendation
        print(f"🏆 Recommended: {recommended_method.replace('_', ' ').title()}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"\n{reasoning}")

        if alternatives:
            print(f"\n📋 Alternative methods: {', '.join(alt.replace('_', ' ').title() for alt in alternatives)}")

        print("=" * 60)

        return recommendation

    def _score_epsilon(
        self,
        n_materials: int,
        complexity: str,
        time_limit: float,
        priority: str,
        has_integer: bool
    ) -> float:
        """
        Score Epsilon Constraint method

        Args:
            n_materials: Number of materials
            complexity: Problem complexity
            time_limit: Time limit in minutes
            priority: User priority
            has_integer: Has integer variables

        Returns:
            Score (0-1)
        """
        score = 0.5  # Base score

        # Small problems favor epsilon constraint
        if n_materials < 20:
            score += 0.2
        elif n_materials < 30:
            score += 0.1
        else:
            score -= 0.1  # Penalize for large problems

        # Time availability
        if time_limit >= 30:
            score += 0.15
        elif time_limit >= 20:
            score += 0.05
        else:
            score -= 0.1  # Not enough time

        # Complexity
        if complexity in ['simple', 'medium']:
            score += 0.1
        else:
            score -= 0.05

        # Priority
        if priority == 'quality':
            score += 0.15  # Epsilon gives highest quality
        elif priority == 'speed':
            score -= 0.1

        # Integer variables are OK
        if has_integer:
            score += 0.05

        return max(0.0, min(1.0, score))

    def _score_nsga2(
        self,
        n_materials: int,
        complexity: str,
        time_limit: float,
        priority: str,
        has_integer: bool
    ) -> float:
        """
        Score NSGA-II method

        Args:
            n_materials: Number of materials
            complexity: Problem complexity
            time_limit: Time limit in minutes
            priority: User priority
            has_integer: Has integer variables

        Returns:
            Score (0-1)
        """
        score = 0.6  # Base score (generally robust)

        # Large problems favor NSGA-II
        if n_materials > 50:
            score += 0.2
        elif n_materials > 30:
            score += 0.1
        else:
            score -= 0.05  # Overkill for small problems

        # High complexity favors evolutionary approach
        if complexity in ['high', 'very_high']:
            score += 0.2
        elif complexity == 'medium':
            score += 0.1

        # Time requirements
        if time_limit >= 30:
            score += 0.1
        elif time_limit < 15:
            score -= 0.15  # Needs time to converge

        # Priority
        if priority == 'exploration':
            score += 0.15
        elif priority == 'quality':
            score += 0.05
        elif priority == 'speed':
            score -= 0.05

        # Integer variables need careful handling
        if has_integer:
            score -= 0.1  # Gene encoding is tricky

        return max(0.0, min(1.0, score))

    def _score_weight_sweep(
        self,
        n_materials: int,
        complexity: str,
        time_limit: float,
        priority: str,
        has_integer: bool
    ) -> float:
        """
        Score Weight Sweep method

        Args:
            n_materials: Number of materials
            complexity: Problem complexity
            time_limit: Time limit in minutes
            priority: User priority
            has_integer: Has integer variables

        Returns:
            Score (0-1)
        """
        score = 0.7  # Base score (most reliable)

        # Medium-sized problems are ideal
        if 20 <= n_materials <= 50:
            score += 0.15
        elif n_materials < 20:
            score += 0.1
        else:
            score -= 0.05

        # Simple to medium complexity
        if complexity in ['simple', 'medium']:
            score += 0.15
        elif complexity == 'high':
            score += 0.05
        else:
            score -= 0.1  # May struggle with very high complexity

        # Time flexibility
        if time_limit < 20:
            score += 0.1  # Good for tight deadlines
        elif time_limit >= 30:
            score -= 0.05  # Other methods may provide better results

        # Priority
        if priority == 'speed':
            score += 0.15
        elif priority == 'balanced':
            score += 0.1
        elif priority == 'quality':
            score -= 0.05

        # Handles integer variables well
        if has_integer:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _explain_recommendation(
        self,
        method: str,
        problem_chars: Dict[str, Any],
        user_prefs: Dict[str, Any],
        scores: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation

        Args:
            method: Recommended method
            problem_chars: Problem characteristics
            user_prefs: User preferences
            scores: Method scores

        Returns:
            Explanation text
        """
        n_materials = problem_chars.get('n_materials', 0)
        complexity = problem_chars.get('complexity', 'medium')
        time_limit = problem_chars.get('time_limit_min', 30)

        profile = self.method_profiles[method]

        explanation = f"Recommendation Reasoning:\n\n"

        # Why this method?
        if method == 'epsilon_constraint':
            explanation += "Epsilon Constraint is recommended because:\n"
            if n_materials < 30:
                explanation += f"  • Small problem size ({n_materials} materials) allows thorough search\n"
            if time_limit >= 20:
                explanation += f"  • Sufficient time ({time_limit} min) for systematic exploration\n"
            if complexity in ['simple', 'medium']:
                explanation += f"  • {complexity.title()} complexity is well-suited for exact methods\n"

        elif method == 'nsga2':
            explanation += "NSGA-II is recommended because:\n"
            if n_materials > 30:
                explanation += f"  • Large problem size ({n_materials} materials) benefits from evolutionary approach\n"
            if complexity in ['high', 'very_high']:
                explanation += f"  • {complexity.replace('_', ' ').title()} complexity handled well by genetic algorithms\n"
            if time_limit >= 30:
                explanation += f"  • Adequate time ({time_limit} min) for convergence\n"

        else:  # weight_sweep
            explanation += "Weight Sweep is recommended because:\n"
            if 20 <= n_materials <= 50:
                explanation += f"  • Medium problem size ({n_materials} materials) is ideal for weight methods\n"
            if time_limit < 20:
                explanation += f"  • Tight time constraint ({time_limit} min) favors fast methods\n"
            if complexity in ['simple', 'medium']:
                explanation += f"  • {complexity.title()} complexity allows effective weighted sum\n"

        # Add method strengths
        explanation += f"\nKey Strengths:\n"
        for strength in profile['strengths'][:3]:
            explanation += f"  • {strength}\n"

        # Add score comparison
        explanation += f"\nMethod Scores:\n"
        for m, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            explanation += f"  • {m.replace('_', ' ').title()}: {s:.1%}\n"

        return explanation

    def get_method_comparison_table(self) -> str:
        """
        Get formatted comparison table of all methods

        Returns:
            Formatted table string
        """
        table = "\n" + "=" * 80 + "\n"
        table += "PARETO METHOD COMPARISON TABLE\n"
        table += "=" * 80 + "\n\n"

        table += f"{'Method':<20} {'Best For':<25} {'Scalability':<15} {'Quality':<10}\n"
        table += "-" * 80 + "\n"

        for method, profile in self.method_profiles.items():
            best_for = profile['best_for'][0].replace('_', ' ')
            scalability = profile['scalability'].title()
            quality = profile['quality'].title()

            table += f"{method.replace('_', ' ').title():<20} "
            table += f"{best_for:<25} "
            table += f"{scalability:<15} "
            table += f"{quality:<10}\n"

        table += "\n" + "=" * 80 + "\n"

        return table

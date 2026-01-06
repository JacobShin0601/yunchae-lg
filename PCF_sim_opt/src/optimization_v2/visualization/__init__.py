"""
Visualization module for optimization results

Provides visualization tools for:
- Pareto front analysis
- Evolution animations (NSGA-II)
- Convergence tracking
- Interactive 3D charts
- Method comparison visualizations
- Scenario analysis
"""

from .evolution_animator import EvolutionAnimator
from .interactive_chart_builder import InteractiveChartBuilder

__all__ = [
    'EvolutionAnimator',
    'InteractiveChartBuilder'
]

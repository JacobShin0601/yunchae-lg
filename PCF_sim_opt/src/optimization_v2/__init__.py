"""
Optimization V2 - 모듈식 제약조건 기반 최적화 시스템

완전히 재설계된 최적화 시스템으로, 다음 기능을 제공합니다:
- 유연한 제약조건 시스템
- RE100 프리미엄 비용 통합
- 자재별 위치 및 관리 제약
- 확장 가능한 아키텍처
"""

__version__ = "2.0.0"
__author__ = "PCF Simulator Team"

# Core classes
from .core.constraint_base import ConstraintBase
from .core.constraint_manager import ConstraintManager
from .core.optimization_engine import OptimizationEngine
from .core.result_processor import ResultProcessor

# Constraint implementations
from .constraints.material_constraint import MaterialManagementConstraint
from .constraints.cost_constraint import CostConstraint
from .constraints.location_constraint import LocationConstraint

# Utilities
from .utils.data_loader import DataLoader

# UI Components
from .ui.constraint_configurator import ConstraintConfigurator
from .ui.results_visualizer import ResultsVisualizer
from .ui.comparison_dashboard import ComparisonDashboard

__all__ = [
    # Core
    'ConstraintBase',
    'ConstraintManager',
    'OptimizationEngine',
    'ResultProcessor',

    # Constraints
    'MaterialManagementConstraint',
    'CostConstraint',
    'LocationConstraint',

    # Utils
    'DataLoader',

    # UI
    'ConstraintConfigurator',
    'ResultsVisualizer',
    'ComparisonDashboard',
]

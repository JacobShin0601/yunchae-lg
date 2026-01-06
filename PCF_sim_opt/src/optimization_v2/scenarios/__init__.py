"""
Scenarios module - Unified scenario management for optimization

Provides common framework for defining and managing optimization scenarios,
including constraint presets, templates, and validation.
"""

from .scenario_framework import (
    ScenarioType,
    ConstraintLevel,
    ElementConstraints,
    RE100Constraints,
    CostConstraints,
    Scenario,
    ScenarioBuilder,
    ScenarioLibrary,
    ScenarioManager
)

__all__ = [
    # Enums
    'ScenarioType',
    'ConstraintLevel',

    # Constraint classes
    'ElementConstraints',
    'RE100Constraints',
    'CostConstraints',

    # Main classes
    'Scenario',
    'ScenarioBuilder',
    'ScenarioLibrary',
    'ScenarioManager'
]

"""
Common module - Shared interfaces and utilities

Provides standardized interfaces for optimization inputs and results
across all optimization methods.
"""

from .interfaces import (
    OptimizationMethod,
    OptimizationStatus,
    OptimizationInput,
    OptimizationResult,
    ResultComparator
)

__all__ = [
    # Enums
    'OptimizationMethod',
    'OptimizationStatus',

    # Main classes
    'OptimizationInput',
    'OptimizationResult',
    'ResultComparator'
]

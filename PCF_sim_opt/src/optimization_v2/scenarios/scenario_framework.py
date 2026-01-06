"""
Unified Scenario Framework - Common scenario management for optimization

This module provides a unified framework for managing optimization scenarios,
including constraint presets, scenario templates, and validation.

Components:
- ScenarioBuilder: Build scenarios from templates
- ScenarioValidator: Validate scenario configurations
- ScenarioLibrary: Catalog of pre-defined scenarios
- ScenarioComparator: Compare multiple scenarios
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import copy


class ScenarioType(Enum):
    """Scenario types"""
    BASELINE = "baseline"  # Business as usual
    AGGRESSIVE = "aggressive"  # Aggressive carbon reduction
    BALANCED = "balanced"  # Balance between carbon and cost
    COST_FOCUSED = "cost_focused"  # Cost minimization priority
    CARBON_FOCUSED = "carbon_focused"  # Carbon minimization priority
    CUSTOM = "custom"  # User-defined


class ConstraintLevel(Enum):
    """Constraint strictness levels"""
    RELAXED = "relaxed"
    MEDIUM = "medium"
    STRICT = "strict"
    VERY_STRICT = "very_strict"


@dataclass
class ElementConstraints:
    """Constraints for a single element (Ni, Co, Li)"""
    element: str  # 'Ni', 'Co', 'Li'
    recycle_min: float = 0.0
    recycle_max: float = 1.0
    low_carbon_min: float = 0.0
    low_carbon_max: float = 1.0
    virgin_min: float = 0.0
    virgin_max: float = 1.0

    def validate(self) -> Tuple[bool, str]:
        """Validate constraint consistency"""
        # Check bounds
        if not (0 <= self.recycle_min <= self.recycle_max <= 1.0):
            return False, f"{self.element}: Invalid recycle bounds"
        if not (0 <= self.low_carbon_min <= self.low_carbon_max <= 1.0):
            return False, f"{self.element}: Invalid low_carbon bounds"
        if not (0 <= self.virgin_min <= self.virgin_max <= 1.0):
            return False, f"{self.element}: Invalid virgin bounds"

        # Check feasibility (sum must be 1)
        min_sum = self.recycle_min + self.low_carbon_min + self.virgin_min
        max_sum = self.recycle_max + self.low_carbon_max + self.virgin_max

        if min_sum > 1.0:
            return False, f"{self.element}: Minimum bounds sum > 1.0 (infeasible)"
        if max_sum < 1.0:
            return False, f"{self.element}: Maximum bounds sum < 1.0 (infeasible)"

        return True, "OK"


@dataclass
class RE100Constraints:
    """RE100 constraints for Formula materials"""
    tier1_re_min: float = 0.0
    tier1_re_max: float = 1.0
    tier2_re_min: float = 0.0
    tier2_re_max: float = 1.0

    def validate(self) -> Tuple[bool, str]:
        """Validate constraint consistency"""
        if not (0 <= self.tier1_re_min <= self.tier1_re_max <= 1.0):
            return False, "Invalid Tier1 RE bounds"
        if not (0 <= self.tier2_re_min <= self.tier2_re_max <= 1.0):
            return False, "Invalid Tier2 RE bounds"
        return True, "OK"


@dataclass
class CostConstraints:
    """Cost-related constraints"""
    premium_limit_pct: Optional[float] = None  # Maximum premium % above baseline
    absolute_budget: Optional[float] = None  # Absolute cost budget

    def validate(self) -> Tuple[bool, str]:
        """Validate constraint consistency"""
        if self.premium_limit_pct is not None and self.premium_limit_pct < 0:
            return False, "Premium limit cannot be negative"
        if self.absolute_budget is not None and self.absolute_budget < 0:
            return False, "Absolute budget cannot be negative"
        return True, "OK"


@dataclass
class Scenario:
    """
    Complete optimization scenario definition

    A scenario encapsulates all configuration needed for an optimization run,
    including constraints, objectives, and metadata.
    """
    name: str
    description: str
    scenario_type: ScenarioType
    constraint_level: ConstraintLevel

    # Constraints
    element_constraints: Dict[str, ElementConstraints] = field(default_factory=dict)
    re100_constraints: Optional[RE100Constraints] = None
    cost_constraints: Optional[CostConstraints] = None

    # Objectives
    carbon_weight: float = 0.5
    cost_weight: float = 0.5

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate scenario configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate weights
        if self.carbon_weight < 0 or self.cost_weight < 0:
            errors.append("Weights cannot be negative")

        weight_sum = self.carbon_weight + self.cost_weight
        if abs(weight_sum - 1.0) > 1e-6:
            errors.append(f"Weights must sum to 1.0 (got {weight_sum})")

        # Validate element constraints
        for element, constraints in self.element_constraints.items():
            valid, msg = constraints.validate()
            if not valid:
                errors.append(msg)

        # Validate RE100 constraints
        if self.re100_constraints:
            valid, msg = self.re100_constraints.validate()
            if not valid:
                errors.append(f"RE100: {msg}")

        # Validate cost constraints
        if self.cost_constraints:
            valid, msg = self.cost_constraints.validate()
            if not valid:
                errors.append(f"Cost: {msg}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type.value,
            'constraint_level': self.constraint_level.value,
            'element_constraints': {
                elem: {
                    'element': c.element,
                    'recycle_min': c.recycle_min,
                    'recycle_max': c.recycle_max,
                    'low_carbon_min': c.low_carbon_min,
                    'low_carbon_max': c.low_carbon_max,
                    'virgin_min': c.virgin_min,
                    'virgin_max': c.virgin_max
                }
                for elem, c in self.element_constraints.items()
            },
            're100_constraints': {
                'tier1_re_min': self.re100_constraints.tier1_re_min,
                'tier1_re_max': self.re100_constraints.tier1_re_max,
                'tier2_re_min': self.re100_constraints.tier2_re_min,
                'tier2_re_max': self.re100_constraints.tier2_re_max
            } if self.re100_constraints else None,
            'cost_constraints': {
                'premium_limit_pct': self.cost_constraints.premium_limit_pct,
                'absolute_budget': self.cost_constraints.absolute_budget
            } if self.cost_constraints else None,
            'carbon_weight': self.carbon_weight,
            'cost_weight': self.cost_weight,
            'created_at': self.created_at,
            'tags': self.tags,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scenario':
        """Create scenario from dictionary"""
        # Convert element constraints
        element_constraints = {}
        if 'element_constraints' in data:
            for elem, c_data in data['element_constraints'].items():
                element_constraints[elem] = ElementConstraints(**c_data)

        # Convert RE100 constraints
        re100_constraints = None
        if data.get('re100_constraints'):
            re100_constraints = RE100Constraints(**data['re100_constraints'])

        # Convert cost constraints
        cost_constraints = None
        if data.get('cost_constraints'):
            cost_constraints = CostConstraints(**data['cost_constraints'])

        return cls(
            name=data['name'],
            description=data['description'],
            scenario_type=ScenarioType(data['scenario_type']),
            constraint_level=ConstraintLevel(data['constraint_level']),
            element_constraints=element_constraints,
            re100_constraints=re100_constraints,
            cost_constraints=cost_constraints,
            carbon_weight=data.get('carbon_weight', 0.5),
            cost_weight=data.get('cost_weight', 0.5),
            created_at=data.get('created_at', datetime.now().isoformat()),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )


class ScenarioBuilder:
    """
    Builder for creating scenarios from templates

    Provides fluent interface for scenario construction.
    """

    def __init__(self):
        self.scenario = None
        self.reset()

    def reset(self) -> 'ScenarioBuilder':
        """Reset builder to initial state"""
        self.scenario = Scenario(
            name="Untitled",
            description="",
            scenario_type=ScenarioType.CUSTOM,
            constraint_level=ConstraintLevel.MEDIUM
        )
        return self

    def set_basic_info(
        self,
        name: str,
        description: str,
        scenario_type: ScenarioType = ScenarioType.CUSTOM
    ) -> 'ScenarioBuilder':
        """Set basic scenario information"""
        self.scenario.name = name
        self.scenario.description = description
        self.scenario.scenario_type = scenario_type
        return self

    def set_constraint_level(self, level: ConstraintLevel) -> 'ScenarioBuilder':
        """Set constraint strictness level"""
        self.scenario.constraint_level = level
        return self

    def add_element_constraint(
        self,
        element: str,
        recycle_range: Tuple[float, float] = (0.0, 1.0),
        low_carbon_range: Tuple[float, float] = (0.0, 1.0),
        virgin_range: Tuple[float, float] = (0.0, 1.0)
    ) -> 'ScenarioBuilder':
        """Add element constraint"""
        constraint = ElementConstraints(
            element=element,
            recycle_min=recycle_range[0],
            recycle_max=recycle_range[1],
            low_carbon_min=low_carbon_range[0],
            low_carbon_max=low_carbon_range[1],
            virgin_min=virgin_range[0],
            virgin_max=virgin_range[1]
        )
        self.scenario.element_constraints[element] = constraint
        return self

    def set_re100_constraints(
        self,
        tier1_range: Tuple[float, float] = (0.0, 1.0),
        tier2_range: Tuple[float, float] = (0.0, 1.0)
    ) -> 'ScenarioBuilder':
        """Set RE100 constraints"""
        self.scenario.re100_constraints = RE100Constraints(
            tier1_re_min=tier1_range[0],
            tier1_re_max=tier1_range[1],
            tier2_re_min=tier2_range[0],
            tier2_re_max=tier2_range[1]
        )
        return self

    def set_cost_constraints(
        self,
        premium_limit_pct: Optional[float] = None,
        absolute_budget: Optional[float] = None
    ) -> 'ScenarioBuilder':
        """Set cost constraints"""
        self.scenario.cost_constraints = CostConstraints(
            premium_limit_pct=premium_limit_pct,
            absolute_budget=absolute_budget
        )
        return self

    def set_weights(self, carbon_weight: float, cost_weight: float) -> 'ScenarioBuilder':
        """Set objective weights"""
        self.scenario.carbon_weight = carbon_weight
        self.scenario.cost_weight = cost_weight
        return self

    def add_tags(self, *tags: str) -> 'ScenarioBuilder':
        """Add tags to scenario"""
        self.scenario.tags.extend(tags)
        return self

    def set_metadata(self, key: str, value: Any) -> 'ScenarioBuilder':
        """Set metadata field"""
        self.scenario.metadata[key] = value
        return self

    def build(self) -> Scenario:
        """Build and validate scenario"""
        is_valid, errors = self.scenario.validate()
        if not is_valid:
            raise ValueError(f"Invalid scenario: {'; '.join(errors)}")

        result = self.scenario
        self.reset()
        return result


class ScenarioLibrary:
    """
    Library of pre-defined scenarios

    Provides common scenario templates for quick use.
    """

    @staticmethod
    def baseline() -> Scenario:
        """Business as usual scenario - no constraints"""
        return ScenarioBuilder() \
            .set_basic_info(
                "Baseline",
                "Business as usual - no special constraints",
                ScenarioType.BASELINE
            ) \
            .set_constraint_level(ConstraintLevel.RELAXED) \
            .set_weights(1.0, 0.0) \
            .add_tags("baseline", "reference") \
            .build()

    @staticmethod
    def aggressive_carbon_reduction() -> Scenario:
        """Aggressive carbon reduction scenario"""
        builder = ScenarioBuilder() \
            .set_basic_info(
                "Aggressive Carbon Reduction",
                "Maximum recycling and low-carbon materials",
                ScenarioType.AGGRESSIVE
            ) \
            .set_constraint_level(ConstraintLevel.STRICT)

        # High recycling requirements for all elements
        for element in ['Ni', 'Co', 'Li']:
            builder.add_element_constraint(
                element,
                recycle_range=(0.4, 1.0),  # Min 40% recycling
                low_carbon_range=(0.2, 1.0),  # Min 20% low-carbon
                virgin_range=(0.0, 0.4)  # Max 40% virgin
            )

        return builder \
            .set_re100_constraints(
                tier1_range=(0.5, 1.0),  # Min 50% Tier1 RE
                tier2_range=(0.3, 1.0)   # Min 30% Tier2 RE
            ) \
            .set_weights(0.9, 0.1) \
            .add_tags("aggressive", "carbon_focused") \
            .build()

    @staticmethod
    def balanced() -> Scenario:
        """Balanced scenario - moderate constraints"""
        builder = ScenarioBuilder() \
            .set_basic_info(
                "Balanced",
                "Balance between carbon reduction and cost",
                ScenarioType.BALANCED
            ) \
            .set_constraint_level(ConstraintLevel.MEDIUM)

        # Moderate requirements
        for element in ['Ni', 'Co', 'Li']:
            builder.add_element_constraint(
                element,
                recycle_range=(0.2, 0.6),
                low_carbon_range=(0.1, 0.4),
                virgin_range=(0.2, 0.7)
            )

        return builder \
            .set_re100_constraints(
                tier1_range=(0.2, 0.8),
                tier2_range=(0.1, 0.6)
            ) \
            .set_weights(0.5, 0.5) \
            .add_tags("balanced", "moderate") \
            .build()

    @staticmethod
    def cost_focused() -> Scenario:
        """Cost-focused scenario - minimize cost"""
        builder = ScenarioBuilder() \
            .set_basic_info(
                "Cost Focused",
                "Minimize cost while meeting basic requirements",
                ScenarioType.COST_FOCUSED
            ) \
            .set_constraint_level(ConstraintLevel.RELAXED)

        # Minimal requirements
        for element in ['Ni', 'Co', 'Li']:
            builder.add_element_constraint(
                element,
                recycle_range=(0.0, 0.3),
                low_carbon_range=(0.0, 0.2),
                virgin_range=(0.5, 1.0)
            )

        return builder \
            .set_re100_constraints(
                tier1_range=(0.0, 0.3),
                tier2_range=(0.0, 0.2)
            ) \
            .set_weights(0.1, 0.9) \
            .add_tags("cost_focused", "economical") \
            .build()

    @staticmethod
    def get_all_templates() -> Dict[str, Scenario]:
        """Get all pre-defined templates"""
        return {
            'baseline': ScenarioLibrary.baseline(),
            'aggressive': ScenarioLibrary.aggressive_carbon_reduction(),
            'balanced': ScenarioLibrary.balanced(),
            'cost_focused': ScenarioLibrary.cost_focused()
        }


class ScenarioManager:
    """
    Scenario persistence and management

    Handles saving/loading scenarios to/from disk.
    """

    def __init__(self, scenarios_dir: Path = Path('input/scenarios')):
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)

    def save(self, scenario: Scenario, filename: Optional[str] = None) -> Path:
        """
        Save scenario to file

        Args:
            scenario: Scenario to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            safe_name = scenario.name.lower().replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{safe_name}_{timestamp}.json"

        filepath = self.scenarios_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scenario.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"💾 Scenario saved: {filepath}")
        return filepath

    def load(self, filename: str) -> Scenario:
        """Load scenario from file"""
        filepath = self.scenarios_dir / filename

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return Scenario.from_dict(data)

    def list_scenarios(self) -> List[str]:
        """List all saved scenarios"""
        return [f.name for f in self.scenarios_dir.glob('*.json')]

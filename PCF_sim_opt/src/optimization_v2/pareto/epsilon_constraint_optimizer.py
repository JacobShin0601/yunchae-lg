"""
Epsilon-Constraint 방식 파레토 최적화
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import numpy as np
import pyomo.environ as pyo

from .config_loader import ParetoConfigLoader
from .base_pareto_optimizer import BaseParetoOptimizer
from ..core.optimization_engine import OptimizationEngine
from ..core.result_processor import ResultProcessor
from ..constraints import CostConstraint
from ..utils.total_cost_calculator import TotalCostCalculator


class EpsilonConstraintOptimizer(BaseParetoOptimizer):
    """Epsilon-Constraint 최적화 실행기"""

    def __init__(self, user_id: str = None):
        super().__init__(user_id)  # Call base class init

    def run_optimization(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run Pareto optimization using epsilon-constraint method

        This implements the abstract method from BaseParetoOptimizer.

        Args:
            optimization_data: Optimization data
            cost_calculator: RE100PremiumCalculator instance
            **kwargs: Additional parameters for run_epsilon_sweep
                - baseline_case: str (default: 'case1')
                - constraint_preset: str (default: 'medium')
                - scenario_template: str (optional)
                - premium_scan_range: List[float] (optional)

        Returns:
            List of Pareto points (results)
        """
        # Extract kwargs with defaults
        baseline_case = kwargs.get('baseline_case', 'case1')
        constraint_preset = kwargs.get('constraint_preset', 'medium')
        scenario_template = kwargs.get('scenario_template', None)
        premium_scan_range = kwargs.get('premium_scan_range', None)

        # Call the standard epsilon sweep method
        return self.run_epsilon_sweep(
            optimization_data=optimization_data,
            cost_calculator=cost_calculator,
            baseline_case=baseline_case,
            constraint_preset=constraint_preset,
            scenario_template=scenario_template,
            premium_scan_range=premium_scan_range
        )

    def run_epsilon_sweep(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium',
        scenario_template: str = None,
        premium_scan_range: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Epsilon 스캔 실행

        Args:
            optimization_data: 최적화 데이터
            cost_calculator: RE100PremiumCalculator 인스턴스
            baseline_case: 기준 케이스
            constraint_preset: 제약조건 프리셋
            scenario_template: 시나리오 템플릿 (optional)
            premium_scan_range: Optional list of premium limits (%) to scan

        Returns:
            파레토 포인트 결과 리스트
        """
        # Setup cost calculator and calculate baseline using base class
        self.setup_cost_calculator(cost_calculator)
        baselines = self.calculate_baseline(optimization_data)
        zero_premium_baseline = baselines['zero_premium_baseline']

        # Epsilon 값들 생성
        epsilon_values = self._generate_epsilon_values(zero_premium_baseline, premium_scan_range)

        print(f"\n🎯 Epsilon-Constraint 최적화 시작")
        print(f"   Zero-Premium Baseline: ${zero_premium_baseline:,.2f}")
        print(f"   탐색 포인트: {len(epsilon_values)}개")
        print(f"   비용 범위: ${min(epsilon_values):,.2f} ~ ${max(epsilon_values):,.2f}")

        self.results = []

        for idx, epsilon in enumerate(epsilon_values, 1):
            print(f"\n[{idx}/{len(epsilon_values)}] Epsilon: ${epsilon:,.2f}")

            try:
                # 최적화 실행
                result = self._run_single_optimization(
                    epsilon,
                    optimization_data,
                    cost_calculator,
                    baseline_case,
                    constraint_preset,
                    scenario_template
                )

                if result:
                    self.results.append(result)
                    print(f"   ✅ 완료 - Carbon: {result['summary']['total_carbon']:.2f}, Cost: {result['summary'].get('total_cost', 0):.2f}")
                else:
                    print(f"   ⚠️  실현불가능 (Infeasible)")

            except Exception as e:
                print(f"   ❌ 실패: {str(e)}")
                continue

        print(f"\n✅ Epsilon 스캔 완료: {len(self.results)}/{len(epsilon_values)}개 성공")

        # 결과 저장
        self._save_results()

        return self.results

    def _generate_epsilon_values(
        self,
        zero_premium_baseline: float,
        premium_scan_range: Optional[List[float]] = None
    ) -> List[float]:
        """
        Epsilon 값들 생성

        Args:
            zero_premium_baseline: Zero-premium baseline cost
            premium_scan_range: Optional list of premium limits (%) to scan

        Returns:
            List of epsilon values (absolute costs)
        """
        # If premium_scan_range provided, convert % to absolute costs
        if premium_scan_range is not None:
            epsilon_values = [
                zero_premium_baseline * (1 + pct / 100)
                for pct in premium_scan_range
            ]
            return epsilon_values

        # Otherwise use config-based strategy
        config = self.config_loader.config.get('epsilon_constraint', {})
        strategy = config.get('strategy', 'linear_sweep')

        if strategy == 'linear_sweep':
            return self._generate_linear_epsilon(zero_premium_baseline, config.get('linear_sweep', {}))
        elif strategy == 'exponential_sweep':
            return self._generate_exponential_epsilon(zero_premium_baseline, config.get('exponential_sweep', {}))
        elif strategy == 'adaptive':
            return self._generate_adaptive_epsilon_initial(zero_premium_baseline, config.get('adaptive', {}))
        else:
            raise ValueError(f"Unknown epsilon strategy: {strategy}")

    def _generate_linear_epsilon(self, baseline_cost: float, config: Dict) -> List[float]:
        """선형 증가 epsilon 값들"""
        max_multiplier = config.get('cost_max_multiplier', 1.5)
        points = config.get('points', 10)

        # baseline부터 baseline × max_multiplier까지 균등 분할
        epsilon_values = np.linspace(
            baseline_cost,
            baseline_cost * max_multiplier,
            points
        )

        return epsilon_values.tolist()

    def _generate_exponential_epsilon(self, baseline_cost: float, config: Dict) -> List[float]:
        """지수 증가 epsilon 값들"""
        max_multiplier = config.get('cost_max_multiplier', 2.0)
        points = config.get('points', 8)

        # 지수 스케일로 생성
        ratios = np.logspace(0, np.log10(max_multiplier), points, base=10)
        epsilon_values = baseline_cost * ratios

        return epsilon_values.tolist()

    def _generate_adaptive_epsilon_initial(self, baseline_cost: float, config: Dict) -> List[float]:
        """적응형 epsilon 초기값 (런타임에 확장)"""
        initial_step = config.get('initial_step', 0.05)
        max_attempts = config.get('max_attempts', 20)

        # 초기 epsilon 값들 (런타임에 추가될 수 있음)
        epsilon_values = [baseline_cost * (1 + i * initial_step) for i in range(max_attempts)]

        return epsilon_values

    def _run_single_optimization(
        self,
        epsilon: float,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str,
        constraint_preset: str,
        scenario_template: str
    ) -> Dict[str, Any]:
        """단일 epsilon 값으로 최적화 실행"""
        from ..core.constraint_manager import ConstraintManager

        # 새로운 ConstraintManager 생성
        constraint_manager = ConstraintManager()

        # 제약조건 프리셋 적용 (None이 아닌 경우에만)
        if constraint_preset:
            from .weight_sweep_optimizer import WeightSweepOptimizer
            sweep_optimizer = WeightSweepOptimizer(self.user_id)
            sweep_optimizer._apply_constraint_preset(
                constraint_manager, constraint_preset, optimization_data
            )

        # 시나리오 템플릿 적용 (optional)
        if scenario_template:
            sweep_optimizer._apply_scenario_template(
                constraint_manager, scenario_template, optimization_data
            )

        # Use TotalCostCalculator from base class (already setup)
        # Create CostConstraint
        cost_constraint = CostConstraint(cost_calculator=self.total_cost_calculator)
        cost_constraint.zero_premium_baseline = self.zero_premium_baseline

        # Convert epsilon (total cost limit) to premium budget
        # epsilon = zero_premium_baseline + premium_budget
        premium_budget = max(0, epsilon - self.zero_premium_baseline)
        cost_constraint.set_absolute_premium_budget(premium_budget)

        # ConstraintManager에 비용 제약 추가
        constraint_manager.add_constraint(cost_constraint, priority=20)

        # OptimizationEngine 생성 및 실행
        engine = OptimizationEngine(solver_name='auto')
        engine.constraint_manager = constraint_manager

        # 모델 빌드 (목적함수는 탄소만)
        engine.build_model(optimization_data, objective_type='minimize_carbon')

        # 솔버 실행
        solver_config = self.config_loader.config.get('optimization', {})
        feasibility_config = self.config_loader.config.get('epsilon_constraint', {}).get('feasibility_check', {})

        timeout = feasibility_config.get('timeout', 60) if feasibility_config.get('enabled', True) else solver_config.get('time_limit', 300)

        results = engine.solve(
            time_limit=timeout,
            gap_tolerance=solver_config.get('gap_tolerance', 0.01),
            verbose=solver_config.get('verbose', False)
        )

        # 실현가능성 체크
        from pyomo.opt import SolverStatus, TerminationCondition

        is_feasible = (
            results.solver.status == SolverStatus.ok and
            results.solver.termination_condition == TerminationCondition.optimal
        )

        if not is_feasible:
            # 실현불가능한 epsilon
            return None

        # 결과 추출
        solution = engine.extract_solution()

        # 결과 처리
        result_processor = ResultProcessor()
        result_df = result_processor.process_solution(solution)
        summary = result_processor.calculate_summary(result_df)

        # Calculate actual total cost from the solved model
        model = engine.model
        actual_total_cost = pyo.value(self.total_cost_calculator.calculate_total_cost(model, optimization_data))
        summary['total_cost'] = actual_total_cost

        # 메타데이터 추가
        result = {
            'epsilon': epsilon,
            'baseline_cost': self.zero_premium_baseline,  # Updated: zero-premium baseline
            'zero_premium_baseline': self.zero_premium_baseline,
            'baseline_carbon': self.baseline_carbon,  # 베이스라인 탄소 추가
            'premium_budget': premium_budget,
            'summary': summary,
            'result_df': result_df,
            'solution': solution,
            'timestamp': datetime.now().isoformat(),
            'method': 'epsilon_constraint'
        }

        return result

    def get_pareto_frontier(self) -> List[Dict[str, Any]]:
        """
        파레토 최적 해 필터링

        Uses the base class's filter_pareto_frontier() method with ParetoFilter.
        This replaces the old hardcoded filtering logic with a configurable system.

        Returns:
            파레토 프론티어에 속하는 해들
        """
        # Use base class's configurable filtering
        return self.filter_pareto_frontier()

    def _save_results(self):
        """결과 저장 (JSON + CSV)"""
        # Use base class's standardized save_results method
        self.save_results('epsilon_constraint')

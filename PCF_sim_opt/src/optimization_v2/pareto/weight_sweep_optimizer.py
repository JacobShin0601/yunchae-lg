"""
가중치 스캔 파레토 최적화 엔진
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import pyomo.environ as pyo

from .config_loader import ParetoConfigLoader, WeightCombination
from .base_pareto_optimizer import BaseParetoOptimizer
from .adaptive_weight_scanner import AdaptiveWeightScanner
from ..core.optimization_engine import OptimizationEngine
from ..core.result_processor import ResultProcessor
from ..constraints import CostConstraint, MaterialManagementConstraint
from ..utils.total_cost_calculator import TotalCostCalculator


class WeightSweepOptimizer(BaseParetoOptimizer):
    """가중치 스캔 최적화 실행기"""

    def __init__(self, user_id: str = None):
        super().__init__(user_id)  # Call base class init

    def run_optimization(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run Pareto optimization using weighted sum method

        This implements the abstract method from BaseParetoOptimizer.

        Args:
            optimization_data: Optimization data
            cost_calculator: RE100PremiumCalculator instance
            **kwargs: Additional parameters for run_sweep
                - baseline_case: str (default: 'case1')
                - constraint_preset: str (default: 'medium')
                - scenario_template: str (optional)

        Returns:
            List of Pareto points (results)
        """
        # Extract kwargs with defaults
        baseline_case = kwargs.get('baseline_case', 'case1')
        constraint_preset = kwargs.get('constraint_preset', 'medium')
        scenario_template = kwargs.get('scenario_template', None)

        # Call the standard sweep method
        return self.run_sweep(
            optimization_data=optimization_data,
            cost_calculator=cost_calculator,
            baseline_case=baseline_case,
            constraint_preset=constraint_preset,
            scenario_template=scenario_template
        )

    def run_sweep(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium',
        scenario_template: str = None
    ) -> List[Dict[str, Any]]:
        """
        가중치 스캔 실행

        Args:
            optimization_data: 최적화 데이터
            cost_calculator: RE100PremiumCalculator 인스턴스
            baseline_case: 기준 케이스
            constraint_preset: 제약조건 프리셋
            scenario_template: 시나리오 템플릿 (optional)

        Returns:
            파레토 포인트 결과 리스트
        """
        # 가중치 조합 생성
        weight_combinations = self.config_loader.get_weight_combinations()

        print(f"\n🎯 파레토 최적화 시작: {len(weight_combinations)}개 가중치 조합")

        self.results = []

        for idx, weights in enumerate(weight_combinations, 1):
            print(f"\n[{idx}/{len(weight_combinations)}] 가중치: {weights}")

            try:
                # 최적화 실행
                result = self._run_single_optimization(
                    weights,
                    optimization_data,
                    cost_calculator,
                    baseline_case,
                    constraint_preset,
                    scenario_template
                )

                self.results.append(result)
                print(f"   ✅ 완료 - Carbon: {result['summary']['total_carbon']:.2f}, Cost: {result['summary'].get('total_cost', 0):.2f}")

            except Exception as e:
                print(f"   ❌ 실패: {str(e)}")
                continue

        print(f"\n✅ 파레토 스캔 완료: {len(self.results)}/{len(weight_combinations)}개 성공")

        # 결과 저장
        self._save_results()

        return self.results

    def run_sweep_with_premium_scan(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium',
        scenario_template: str = None,
        premium_scan_range: List[float] = None
    ) -> Dict[float, List[Dict[str, Any]]]:
        """
        Premium limit scan: Run weighted sweep for each premium limit.

        Creates a 2D Pareto frontier with different feasible regions per premium limit.

        Args:
            optimization_data: 최적화 데이터
            cost_calculator: RE100PremiumCalculator 인스턴스
            baseline_case: 기준 케이스
            constraint_preset: 제약조건 프리셋
            scenario_template: 시나리오 템플릿 (optional)
            premium_scan_range: List of premium limits (%) to scan (default: [0, 5, 10, 15, 20])

        Returns:
            Dict mapping premium_limit_pct → results list
        """
        if premium_scan_range is None:
            premium_scan_range = [0, 5, 10, 15, 20]

        # 가중치 조합 생성
        weight_combinations = self.config_loader.get_weight_combinations()

        print(f"\n🎯 Premium Scan Weighted Sum 시작")
        print(f"   Premium 레벨: {premium_scan_range}%")
        print(f"   가중치 조합: {len(weight_combinations)}개")
        print(f"   총 최적화 포인트: {len(premium_scan_range) * len(weight_combinations)}개")

        all_results = {}

        for premium_pct in premium_scan_range:
            print(f"\n{'='*60}")
            print(f"💰 Premium Limit: +{premium_pct}%")
            print(f"{'='*60}")

            premium_results = []

            for idx, weights in enumerate(weight_combinations, 1):
                print(f"\n[{idx}/{len(weight_combinations)}] 가중치: {weights}, Premium: +{premium_pct}%")

                try:
                    # 최적화 실행 with premium limit
                    result = self._run_single_optimization(
                        weights,
                        optimization_data,
                        cost_calculator,
                        baseline_case,
                        constraint_preset,
                        scenario_template,
                        premium_limit_pct=premium_pct
                    )

                    premium_results.append(result)
                    print(f"   ✅ 완료 - Carbon: {result['summary']['total_carbon']:.2f}, Cost: {result['summary'].get('total_cost', 0):.2f}")

                except Exception as e:
                    print(f"   ❌ 실패: {str(e)}")
                    continue

            all_results[premium_pct] = premium_results
            print(f"\n   ✅ Premium {premium_pct}%: {len(premium_results)}/{len(weight_combinations)}개 성공")

        # 결과 통계
        total_success = sum(len(results) for results in all_results.values())
        total_attempts = len(premium_scan_range) * len(weight_combinations)
        print(f"\n✅ Premium Scan 완료: {total_success}/{total_attempts}개 성공")

        # 결과 저장 (flatten for standard save)
        self.results = []
        for premium_results in all_results.values():
            self.results.extend(premium_results)
        self._save_results()

        return all_results

    def run_adaptive_sweep(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium',
        scenario_template: str = None,
        scanner_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run adaptive weight scanning - intelligently adds weights in sparse regions.

        This method uses AdaptiveWeightScanner to efficiently explore the Pareto front
        by starting with a coarse sweep and adaptively adding weights where gaps are detected.

        Benefits over uniform sweep:
        - Fewer total optimizations (typically 30-50% fewer points)
        - Better coverage of non-convex regions
        - Automatic adaptation to problem characteristics

        Args:
            optimization_data: 최적화 데이터
            cost_calculator: RE100PremiumCalculator 인스턴스
            baseline_case: 기준 케이스
            constraint_preset: 제약조건 프리셋
            scenario_template: 시나리오 템플릿 (optional)
            scanner_config: AdaptiveWeightScanner configuration (optional)
                - initial_points: Number of initial points (default: 5)
                - max_iterations: Max refinement iterations (default: 3)
                - gap_threshold: Normalized distance threshold (default: 0.15)
                - min_points_per_iteration: Min points to add per iteration (default: 2)
                - max_points_per_iteration: Max points to add per iteration (default: 5)

        Returns:
            파레토 포인트 결과 리스트 (adaptively selected)

        Example:
            >>> optimizer = WeightSweepOptimizer()
            >>> results = optimizer.run_adaptive_sweep(
            ...     optimization_data,
            ...     cost_calculator,
            ...     scanner_config={
            ...         'initial_points': 5,
            ...         'max_iterations': 3,
            ...         'gap_threshold': 0.15
            ...     }
            ... )
        """
        # Create adaptive scanner
        scanner = AdaptiveWeightScanner(config=scanner_config)

        # Run adaptive scan
        results = scanner.run_adaptive_scan(
            optimizer=self,
            optimization_data=optimization_data,
            cost_calculator=cost_calculator,
            baseline_case=baseline_case,
            constraint_preset=constraint_preset,
            scenario_template=scenario_template
        )

        # Store results
        self.results = results

        # Print summary
        scanner.print_summary()

        # Save results
        self._save_results()

        return results

    def _run_single_optimization(
        self,
        weights: WeightCombination,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str,
        constraint_preset: str,
        scenario_template: str,
        premium_limit_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """단일 가중치 조합으로 최적화 실행

        Args:
            premium_limit_pct: Optional premium limit percentage for cost constraints
        """
        from ..core.constraint_manager import ConstraintManager
        import streamlit as st

        # 새로운 ConstraintManager 생성 (독립적인 제약조건 세트)
        constraint_manager = ConstraintManager()

        # Session state의 제약조건들을 복사 (빠른 프리셋에서 설정한 제약조건 포함)
        if hasattr(st, 'session_state') and 'constraint_manager' in st.session_state:
            session_cm = st.session_state.constraint_manager
            for constraint_name in session_cm.list_constraints(enabled_only=True):
                constraint = session_cm.get_constraint(constraint_name)
                # CostConstraint는 나중에 별도로 처리하므로 스킵
                if constraint.name != 'cost_constraint':
                    constraint_manager.add_constraint(constraint)

        # 제약조건 프리셋 적용 (None이 아닌 경우에만)
        if constraint_preset:
            self._apply_constraint_preset(constraint_manager, constraint_preset, optimization_data)

        # 시나리오 템플릿 적용 (optional)
        if scenario_template:
            self._apply_scenario_template(constraint_manager, scenario_template, optimization_data)

        # Setup cost calculator and calculate baseline (if not already done)
        if self.total_cost_calculator is None:
            self.setup_cost_calculator(cost_calculator)
            self.calculate_baseline(optimization_data)

        # CostConstraint 처리: Premium Scan > Preset Premium Limit 우선 순위
        if premium_limit_pct is not None:
            # Premium Scan mode: 지정된 premium limit 사용
            cost_constraint = CostConstraint(cost_calculator=self.total_cost_calculator)
            cost_constraint.zero_premium_baseline = self.zero_premium_baseline
            cost_constraint.set_premium_limit(premium_limit_pct)
            constraint_manager.add_constraint(cost_constraint, priority=20)
        elif hasattr(st, 'session_state') and 'preset_premium_limit' in st.session_state:
            # Preset mode: 빠른 프리셋에서 설정한 premium limit 사용
            preset_limit = st.session_state.preset_premium_limit
            cost_constraint = CostConstraint(cost_calculator=self.total_cost_calculator)
            cost_constraint.zero_premium_baseline = self.zero_premium_baseline
            cost_constraint.set_premium_limit(preset_limit)
            constraint_manager.add_constraint(cost_constraint, priority=20)
            print(f"   📌 프리셋 비용 제약 적용: +{preset_limit}% (baseline: ${self.zero_premium_baseline:,.2f})")

        # OptimizationEngine 생성 및 실행
        engine = OptimizationEngine(solver_name='auto')
        engine.constraint_manager = constraint_manager

        # 모델 빌드 (carbon objective 먼저)
        engine.build_model(optimization_data, objective_type='minimize_carbon')
        model = engine.model

        # Get carbon expression (already in model.objective.expr)
        carbon_expr = model.objective.expr

        # Create cost expression using base class's TotalCostCalculator
        total_cost_expr = self.total_cost_calculator.calculate_total_cost(model, optimization_data)

        # Normalize objectives using base class's baseline values
        normalized_carbon = carbon_expr / self.baseline_carbon if self.baseline_carbon > 0 else carbon_expr
        normalized_cost = total_cost_expr / self.zero_premium_baseline if self.zero_premium_baseline > 0 else total_cost_expr

        # Create weighted multi-objective
        # Multi-objective logic moved from CostConstraint to optimizer
        weighted_objective = (
            weights.carbon_weight * normalized_carbon +
            weights.cost_weight * normalized_cost
        )

        # Replace objective
        model.del_component(model.objective)
        model.objective = pyo.Objective(expr=weighted_objective, sense=pyo.minimize)

        # 솔버 실행
        solver_config = self.config_loader.config.get('optimization', {})
        results = engine.solve(
            time_limit=solver_config.get('time_limit', 300),
            gap_tolerance=solver_config.get('gap_tolerance', 0.01),
            verbose=solver_config.get('verbose', False)
        )

        # 결과 추출
        solution = engine.extract_solution()

        # 결과 처리
        result_processor = ResultProcessor()
        result_df = result_processor.process_solution(solution)
        summary = result_processor.calculate_summary(result_df)

        # Calculate actual total cost from the solved model
        actual_total_cost = pyo.value(total_cost_expr)
        summary['total_cost'] = actual_total_cost

        # 메타데이터 추가
        result = {
            'weights': {
                'carbon_weight': weights.carbon_weight,
                'cost_weight': weights.cost_weight
            },
            'baseline_cost': self.zero_premium_baseline,  # Updated: zero-premium baseline
            'zero_premium_baseline': self.zero_premium_baseline,
            'baseline_carbon': self.baseline_carbon,
            'summary': summary,
            'result_df': result_df,
            'solution': solution,
            'timestamp': datetime.now().isoformat()
        }

        if premium_limit_pct is not None:
            result['premium_limit_pct'] = premium_limit_pct

        return result

    def _apply_constraint_preset(
        self,
        constraint_manager,
        preset_name: str,
        optimization_data: Dict[str, Any]
    ):
        """제약조건 프리셋 적용"""
        preset = self.config_loader.get_constraint_preset(preset_name)

        if not preset:
            return

        # 양극재 자재 찾기
        cathode_materials = [
            name for name, info in optimization_data['material_classification'].items()
            if 'Cathode' in name or '양극재' in name or 'CAM' in name
        ]

        if not cathode_materials:
            return

        # 제약조건 생성
        material_constraint = MaterialManagementConstraint()

        for material_name in cathode_materials:
            for element in ['Ni', 'Co', 'Li']:
                if element in preset:
                    element_config = preset[element]
                    recycle_range = element_config['recycle']
                    low_carbon_range = element_config['low_carbon']

                    material_constraint.add_rule(
                        'force_element_ratio_range',
                        material_name,
                        params={
                            'element': element,
                            'recycle_min': recycle_range[0],
                            'recycle_max': recycle_range[1],
                            'low_carbon_min': low_carbon_range[0],
                            'low_carbon_max': low_carbon_range[1]
                        }
                    )

        material_constraint.name = f"preset_{preset_name}"
        material_constraint.description = f"{preset_name} 프리셋"
        constraint_manager.add_constraint(material_constraint, priority=10)

    def _apply_scenario_template(
        self,
        constraint_manager,
        template_name: str,
        optimization_data: Dict[str, Any]
    ):
        """시나리오 템플릿 적용"""
        template = self.config_loader.get_scenario_template(template_name)

        if not template:
            return

        # element_constraints가 있으면 적용
        if 'element_constraints' in template:
            element_constraints = template['element_constraints']

            # 양극재 자재 찾기
            cathode_materials = [
                name for name, info in optimization_data['material_classification'].items()
                if 'Cathode' in name or '양극재' in name or 'CAM' in name
            ]

            if cathode_materials:
                material_constraint = MaterialManagementConstraint()

                for material_name in cathode_materials:
                    for element in ['Ni', 'Co', 'Li']:
                        if element in element_constraints:
                            element_config = element_constraints[element]
                            recycle_range = element_config['recycle']
                            low_carbon_range = element_config['low_carbon']

                            material_constraint.add_rule(
                                'force_element_ratio_range',
                                material_name,
                                params={
                                    'element': element,
                                    'recycle_min': recycle_range[0],
                                    'recycle_max': recycle_range[1],
                                    'low_carbon_min': low_carbon_range[0],
                                    'low_carbon_max': low_carbon_range[1]
                                }
                            )

                material_constraint.name = f"template_{template_name}"
                material_constraint.description = template.get('description', template_name)
                constraint_manager.add_constraint(material_constraint, priority=10)

        # constraint_preset이 있으면 적용
        if 'constraint_preset' in template:
            self._apply_constraint_preset(
                constraint_manager,
                template['constraint_preset'],
                optimization_data
            )

        # RE100 범위 제약 (향후 확장 가능)
        # tier1_re_range, tier2_re_range 활용

    def get_pareto_frontier(self) -> List[Dict[str, Any]]:
        """
        파레토 최적 해 필터링

        Uses the base class's filter_pareto_frontier() method with ParetoFilter.

        Returns:
            파레토 프론티어에 속하는 해들
        """
        # Use base class's configurable filtering
        return self.filter_pareto_frontier()

    def _save_results(self):
        """결과 저장 (JSON + CSV)"""
        # Use base class's standardized save_results method
        self.save_results('weight_sweep')

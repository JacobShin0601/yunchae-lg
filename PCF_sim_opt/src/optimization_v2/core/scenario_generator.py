"""
시나리오 생성 엔진

파라미터 범위를 스위핑하여 다양한 실현 가능한 시나리오를 생성합니다.
단일 최적해가 아닌, 배출량 목표 한도 내에서 여러 조합을 탐색합니다.
"""

from typing import Dict, Any, List, Tuple, Optional
import itertools
from dataclasses import dataclass, asdict
import json
import pandas as pd


@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    name: str
    recycle_ratio_range: Tuple[float, float, float]  # (min, max, step)
    low_carbon_ratio_range: Tuple[float, float, float]  # (min, max, step)
    tier1_re_range: Optional[Tuple[float, float, float]] = None
    tier2_re_range: Optional[Tuple[float, float, float]] = None


@dataclass
class ScenarioResult:
    """시나리오 결과"""
    scenario_id: int
    scenario_name: str
    parameters: Dict[str, Any]  # 입력 파라미터
    is_feasible: bool  # 실현 가능 여부
    total_emission: Optional[float]  # 총 배출량 (kgCO2eq)
    total_cost: Optional[float]  # 총 비용 ($)
    solver_status: str  # 솔버 상태
    solver_time: float  # 솔버 실행 시간 (초)
    results_data: Optional[Dict[str, Any]]  # 상세 결과 데이터


class ScenarioGenerator:
    """
    시나리오 생성 엔진 클래스

    파라미터 범위를 받아서 모든 조합을 생성하고,
    각 조합에 대해 최적화를 실행하여 실현 가능한 솔루션을 수집합니다.
    """

    def __init__(self, optimization_engine, constraint_manager):
        """
        시나리오 생성 엔진 초기화

        Args:
            optimization_engine: OptimizationEngine 인스턴스
            constraint_manager: ConstraintManager 인스턴스
        """
        self.optimization_engine = optimization_engine
        self.constraint_manager = constraint_manager
        self.results: List[ScenarioResult] = []

    def generate_parameter_combinations(
        self,
        config: ScenarioConfig
    ) -> List[Dict[str, Any]]:
        """
        파라미터 범위로부터 모든 조합 생성

        Args:
            config: 시나리오 설정

        Returns:
            파라미터 조합 리스트
        """
        combinations = []

        # 재활용 비율 범위
        recycle_min, recycle_max, recycle_step = config.recycle_ratio_range
        recycle_values = self._generate_range(recycle_min, recycle_max, recycle_step)

        # 저탄소 비율 범위
        low_carbon_min, low_carbon_max, low_carbon_step = config.low_carbon_ratio_range
        low_carbon_values = self._generate_range(low_carbon_min, low_carbon_max, low_carbon_step)

        # Tier1 RE 범위 (선택적)
        if config.tier1_re_range:
            tier1_min, tier1_max, tier1_step = config.tier1_re_range
            tier1_values = self._generate_range(tier1_min, tier1_max, tier1_step)
        else:
            tier1_values = [None]

        # Tier2 RE 범위 (선택적)
        if config.tier2_re_range:
            tier2_min, tier2_max, tier2_step = config.tier2_re_range
            tier2_values = self._generate_range(tier2_min, tier2_max, tier2_step)
        else:
            tier2_values = [None]

        # 모든 조합 생성
        for recycle, low_carbon, tier1, tier2 in itertools.product(
            recycle_values, low_carbon_values, tier1_values, tier2_values
        ):
            # 비율 합이 1을 초과하지 않는지 확인 (재활용 + 저탄소 <= 1)
            if recycle + low_carbon > 1.0:
                continue

            combo = {
                'recycle_ratio': recycle,
                'low_carbon_ratio': low_carbon,
                'virgin_ratio': 1.0 - recycle - low_carbon,
                'tier1_re': tier1,
                'tier2_re': tier2
            }
            combinations.append(combo)

        return combinations

    def _generate_range(self, min_val: float, max_val: float, step: float) -> List[float]:
        """
        범위와 스텝으로부터 값 리스트 생성

        Args:
            min_val: 최소값
            max_val: 최대값
            step: 스텝

        Returns:
            값 리스트
        """
        values = []
        current = min_val

        while current <= max_val + 1e-9:  # 부동소수점 오차 허용
            values.append(round(current, 4))
            current += step

        return values

    def generate_scenarios(
        self,
        data: Dict[str, Any],
        config: ScenarioConfig,
        max_scenarios: int = 100,
        constraint_manager=None
    ) -> List[ScenarioResult]:
        """
        시나리오 생성 및 최적화 실행

        Args:
            data: 최적화 데이터
            config: 시나리오 설정
            max_scenarios: 최대 시나리오 수
            constraint_manager: ConstraintManager (선택적, 기본값은 self.constraint_manager)

        Returns:
            시나리오 결과 리스트
        """
        if constraint_manager is None:
            constraint_manager = self.constraint_manager

        print("\n" + "="*60)
        print("🔍 시나리오 생성 시작")
        print("="*60)

        # 파라미터 조합 생성
        combinations = self.generate_parameter_combinations(config)
        print(f"총 {len(combinations)}개의 파라미터 조합 생성됨")

        # 최대 시나리오 수 제한
        if len(combinations) > max_scenarios:
            print(f"⚠️  조합이 {len(combinations)}개로 너무 많습니다. 처음 {max_scenarios}개만 실행합니다.")
            combinations = combinations[:max_scenarios]

        # 각 조합에 대해 최적화 실행
        results = []
        feasible_count = 0
        infeasible_count = 0

        for idx, params in enumerate(combinations):
            print(f"\n[{idx+1}/{len(combinations)}] 시나리오 실행 중...")
            print(f"  파라미터: {self._format_params(params)}")

            try:
                # 제약조건에 파라미터 적용
                self._apply_parameters_to_constraints(params, constraint_manager, data)

                # 모델 구축 및 최적화 실행
                result = self._run_optimization(
                    scenario_id=idx,
                    scenario_name=f"{config.name}_{idx+1}",
                    params=params,
                    data=data,
                    constraint_manager=constraint_manager
                )

                results.append(result)

                if result.is_feasible:
                    feasible_count += 1
                    print(f"  ✅ 실현 가능 | 배출량: {result.total_emission:,.2f} kgCO2eq")
                else:
                    infeasible_count += 1
                    print(f"  ❌ 실현 불가능 | 상태: {result.solver_status}")

            except Exception as e:
                print(f"  ⚠️  오류 발생: {str(e)}")
                result = ScenarioResult(
                    scenario_id=idx,
                    scenario_name=f"{config.name}_{idx+1}",
                    parameters=params,
                    is_feasible=False,
                    total_emission=None,
                    total_cost=None,
                    solver_status="error",
                    solver_time=0.0,
                    results_data=None
                )
                results.append(result)
                infeasible_count += 1

        print("\n" + "="*60)
        print("✅ 시나리오 생성 완료")
        print(f"  • 총 시나리오 수: {len(results)}개")
        print(f"  • 실현 가능: {feasible_count}개 ({feasible_count/len(results)*100:.1f}%)")
        print(f"  • 실현 불가능: {infeasible_count}개")
        print("="*60)

        self.results = results
        return results

    def _apply_parameters_to_constraints(
        self,
        params: Dict[str, Any],
        constraint_manager,
        data: Dict[str, Any]
    ) -> None:
        """
        파라미터를 제약조건에 적용

        Args:
            params: 파라미터 딕셔너리
            constraint_manager: ConstraintManager
            data: 최적화 데이터
        """
        # MaterialManagementConstraint에 비율 범위 적용
        from ..constraints import MaterialManagementConstraint

        # 기존 자재 관리 제약 제거
        material_constraints = [
            c for c in constraint_manager.list_constraints()
            if isinstance(c, MaterialManagementConstraint)
        ]

        for constraint in material_constraints:
            constraint_manager.remove_constraint(constraint.name)

        # 새로운 제약 추가 (Ni-Co-Li 자재에 대해)
        material_classification = data['material_classification']
        nicoli_materials = [
            m for m, info in material_classification.items()
            if info['is_ni_co_li']
        ]

        if nicoli_materials:
            # 모든 Ni-Co-Li 자재에 동일한 비율 범위 적용
            for material in nicoli_materials:
                constraint = MaterialManagementConstraint()
                constraint.add_material_rule(
                    material_name=material,
                    rule_type='force_ratio_range',
                    params={
                        'recycle_min': params['recycle_ratio'],
                        'recycle_max': params['recycle_ratio'],
                        'low_carbon_min': params['low_carbon_ratio'],
                        'low_carbon_max': params['low_carbon_ratio']
                    }
                )
                constraint_manager.add_constraint(constraint)

    def _run_optimization(
        self,
        scenario_id: int,
        scenario_name: str,
        params: Dict[str, Any],
        data: Dict[str, Any],
        constraint_manager
    ) -> ScenarioResult:
        """
        단일 시나리오에 대해 최적화 실행

        Args:
            scenario_id: 시나리오 ID
            scenario_name: 시나리오 이름
            params: 파라미터
            data: 최적화 데이터
            constraint_manager: ConstraintManager

        Returns:
            ScenarioResult
        """
        import time
        from pyomo.opt import SolverStatus, TerminationCondition

        start_time = time.time()

        try:
            # 모델 구축
            self.optimization_engine.constraint_manager = constraint_manager
            model = self.optimization_engine.build_model(data, objective_type='minimize_carbon')

            # 솔버 실행
            solver = self.optimization_engine.create_solver()
            solver_results = solver.solve(model, tee=False)

            solver_time = time.time() - start_time

            # 결과 추출
            is_feasible = (
                solver_results.solver.status == SolverStatus.ok and
                solver_results.solver.termination_condition == TerminationCondition.optimal
            )

            if is_feasible:
                # 배출량 계산
                total_emission = self._calculate_total_emission(model, data)

                # 비용 계산 (간단히 0으로 설정, 향후 확장 가능)
                total_cost = 0.0

                # 상세 결과 데이터 추출
                results_data = self._extract_results(model, data)
            else:
                total_emission = None
                total_cost = None
                results_data = None

            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                parameters=params,
                is_feasible=is_feasible,
                total_emission=total_emission,
                total_cost=total_cost,
                solver_status=str(solver_results.solver.termination_condition),
                solver_time=solver_time,
                results_data=results_data
            )

        except Exception as e:
            solver_time = time.time() - start_time
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                parameters=params,
                is_feasible=False,
                total_emission=None,
                total_cost=None,
                solver_status=f"error: {str(e)}",
                solver_time=solver_time,
                results_data=None
            )

    def _calculate_total_emission(self, model, data: Dict[str, Any]) -> float:
        """
        총 배출량 계산

        Args:
            model: Pyomo 모델
            data: 최적화 데이터

        Returns:
            총 배출량 (kgCO2eq)
        """
        scenario_df = data['scenario_df']
        total_emission = 0.0

        for material in model.materials:
            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            quantity = material_row['제품총소요량(kg)']

            # modified_emission 값 추출
            emission_factor = model.modified_emission[material].value

            material_emission = emission_factor * quantity
            total_emission += material_emission

        return total_emission

    def _extract_results(self, model, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        모델로부터 상세 결과 추출

        Args:
            model: Pyomo 모델
            data: 최적화 데이터

        Returns:
            결과 딕셔너리
        """
        results = {
            'materials': {}
        }

        material_classification = data['material_classification']

        for material in model.materials:
            material_type = material_classification[material]['type']

            material_result = {
                'type': material_type
            }

            if material_type == 'Formula':
                material_result['tier1_re'] = model.tier1_re[material].value
                material_result['tier2_re'] = model.tier2_re[material].value

            elif material_type == 'Ni-Co-Li':
                material_result['recycle_ratio'] = model.recycle_ratio[material].value
                material_result['low_carbon_ratio'] = model.low_carbon_ratio[material].value
                material_result['virgin_ratio'] = model.virgin_ratio[material].value

            material_result['modified_emission'] = model.modified_emission[material].value

            results['materials'][material] = material_result

        return results

    def _format_params(self, params: Dict[str, Any]) -> str:
        """파라미터 포맷팅"""
        parts = []
        if params.get('recycle_ratio') is not None:
            parts.append(f"재활용={params['recycle_ratio']*100:.1f}%")
        if params.get('low_carbon_ratio') is not None:
            parts.append(f"저탄소={params['low_carbon_ratio']*100:.1f}%")
        if params.get('tier1_re') is not None:
            parts.append(f"Tier1={params['tier1_re']*100:.1f}%")
        if params.get('tier2_re') is not None:
            parts.append(f"Tier2={params['tier2_re']*100:.1f}%")
        return ", ".join(parts)

    def export_results_to_dataframe(self) -> pd.DataFrame:
        """
        결과를 DataFrame으로 변환

        Returns:
            결과 DataFrame
        """
        rows = []

        for result in self.results:
            row = {
                'scenario_id': result.scenario_id,
                'scenario_name': result.scenario_name,
                'is_feasible': result.is_feasible,
                'total_emission': result.total_emission,
                'total_cost': result.total_cost,
                'solver_status': result.solver_status,
                'solver_time': result.solver_time,
            }

            # 파라미터 추가
            for key, value in result.parameters.items():
                row[f'param_{key}'] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self, filepath: str) -> None:
        """
        결과를 JSON 파일로 저장

        Args:
            filepath: 저장 경로
        """
        results_dict = [asdict(r) for r in self.results]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ 결과 저장됨: {filepath}")

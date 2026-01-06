"""
제약 완화 분석기 (Constraint Relaxation Analyzer)

어떤 제약조건이 최적화를 방해하는지 식별하고,
각 제약을 완화했을 때의 효과를 정량화합니다.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import copy
from ..core.optimization_engine import OptimizationEngine
from ..utils.data_loader import DataLoader


class ConstraintRelaxationAnalyzer:
    """
    제약 완화 분석기

    주요 기능:
    - Binding constraint 식별 (slack = 0인 제약)
    - Shadow price (dual value) 추출
    - 제약 완화 시뮬레이션
    - 제약 우선순위화 및 추천
    """

    def __init__(
        self,
        optimization_engine: OptimizationEngine,
        data_loader: DataLoader
    ):
        """
        제약 완화 분석기 초기화

        Args:
            optimization_engine: 최적화 엔진 (dual value 지원 필요)
            data_loader: 데이터 로더
        """
        self.engine = optimization_engine
        self.data_loader = data_loader
        self.analysis_results = {}

    def identify_binding_constraints(
        self,
        slack_threshold: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Binding constraint 식별

        Args:
            slack_threshold: Slack이 이 값보다 작으면 binding으로 간주

        Returns:
            Binding constraint 정보
        """
        print("\n" + "=" * 60)
        print("🔍 Binding Constraint 식별")
        print("=" * 60)

        # Slack value 추출
        try:
            slack_values = self.engine.get_constraint_slack()
        except Exception as e:
            print(f"❌ Slack value 추출 실패: {e}")
            return {}

        # Dual value 추출 (IPOPT 사용 시)
        dual_values = {}
        try:
            if self.engine.solver_name == 'ipopt' and self.engine.enable_dual_values:
                dual_values = self.engine.get_dual_values()
                print(f"✅ Dual value 추출 완료: {len(dual_values)}개")
        except Exception as e:
            print(f"⚠️  Dual value 추출 실패 (무시 가능): {e}")

        # Binding constraint 필터링
        binding_constraints = {}

        for constraint_name, slack_value in slack_values.items():
            if slack_value < slack_threshold:
                dual_value = dual_values.get(constraint_name, None)

                binding_constraints[constraint_name] = {
                    'slack': slack_value,
                    'dual_value': dual_value,
                    'is_binding': True
                }

        print(f"\n✅ Binding constraint 식별 완료: {len(binding_constraints)}개")

        return binding_constraints

    def analyze_relaxation_impact(
        self,
        base_data: Dict[str, Any],
        constraint_specs: List[Dict[str, Any]],
        relaxation_levels: List[float] = [5, 10, 15, 20],
        objective_type: str = 'minimize_carbon'
    ) -> Dict[str, Any]:
        """
        제약 완화 영향 분석

        각 제약조건을 단계별로 완화하며 목적함수 개선 효과 측정

        Args:
            base_data: 기준 최적화 데이터
            constraint_specs: 분석할 제약조건 스펙 리스트
                각 스펙 형식: {
                    'name': str,  # 제약조건 이름
                    'type': str,  # 제약 유형 (예: 'premium_limit')
                    'current_value': float,  # 현재 값
                    'relaxation_direction': str  # 'increase' or 'decrease'
                }
            relaxation_levels: 완화 비율 리스트 (%, 예: [5, 10, 15, 20])
            objective_type: 목적함수 유형

        Returns:
            제약 완화 분석 결과
        """
        print("\n" + "=" * 60)
        print("🔧 제약 완화 영향 분석")
        print("=" * 60)

        # 기준 케이스 최적화
        print("\n📊 기준 케이스 최적화...")
        base_model = self.engine.build_model(base_data, objective_type)
        base_solution = self.engine.solve()

        if not base_solution:
            raise ValueError("기준 케이스 최적화 실패")

        from pyomo.opt import TerminationCondition
        import pyomo.environ as pyo

        if base_solution.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise ValueError(f"기준 케이스 최적화 실패: {base_solution.solver.termination_condition}")

        base_objective = pyo.value(self.engine.model.objective)
        print(f"✅ 기준 목적함수 값: {base_objective:.4f}")

        # 각 제약조건에 대해 완화 분석
        relaxation_results = {}

        for constraint_spec in constraint_specs:
            constraint_name = constraint_spec['name']
            current_value = constraint_spec['current_value']
            direction = constraint_spec['relaxation_direction']

            print(f"\n📈 분석 중: {constraint_name} (현재: {current_value:.2f})")

            constraint_result = {
                'constraint_name': constraint_name,
                'constraint_spec': constraint_spec,
                'base_objective': base_objective,
                'relaxation_levels': [],
                'relaxed_values': [],
                'objective_values': [],
                'objective_improvements': [],
                'marginal_benefits': []
            }

            prev_objective = base_objective

            # 각 완화 수준에 대해 최적화
            for relax_pct in relaxation_levels:
                try:
                    # 완화된 값 계산
                    if direction == 'increase':
                        relaxed_value = current_value * (1 + relax_pct / 100)
                    else:  # decrease
                        relaxed_value = current_value * (1 - relax_pct / 100)

                    # 데이터 복사 및 제약 수정
                    relaxed_data = copy.deepcopy(base_data)
                    self._apply_constraint_relaxation(
                        relaxed_data,
                        constraint_spec,
                        relaxed_value
                    )

                    # 최적화 실행
                    model = self.engine.build_model(relaxed_data, objective_type)
                    solution = self.engine.solve()

                    if solution.solver.termination_condition == pyo.TerminationCondition.optimal:
                        obj_value = pyo.value(self.engine.model.objective)
                        improvement = ((base_objective - obj_value) / base_objective) * 100
                        marginal_benefit = obj_value - prev_objective

                        constraint_result['relaxation_levels'].append(relax_pct)
                        constraint_result['relaxed_values'].append(relaxed_value)
                        constraint_result['objective_values'].append(obj_value)
                        constraint_result['objective_improvements'].append(improvement)
                        constraint_result['marginal_benefits'].append(abs(marginal_benefit))

                        prev_objective = obj_value

                        print(f"  {relax_pct:+6.1f}% → 목적함수: {obj_value:.4f} (개선: {improvement:+.2f}%)")
                    else:
                        print(f"  {relax_pct:+6.1f}% → 최적화 실패")

                except Exception as e:
                    print(f"  {relax_pct:+6.1f}% → 오류: {str(e)}")
                    continue

            relaxation_results[constraint_name] = constraint_result

        self.analysis_results = {
            'base_objective': base_objective,
            'objective_type': objective_type,
            'constraint_results': relaxation_results
        }

        print("\n" + "=" * 60)
        print("✅ 제약 완화 영향 분석 완료")
        print("=" * 60)

        return self.analysis_results

    def _apply_constraint_relaxation(
        self,
        data: Dict[str, Any],
        constraint_spec: Dict[str, Any],
        new_value: float
    ):
        """
        데이터에 제약 완화 적용

        Args:
            data: 최적화 데이터
            constraint_spec: 제약조건 스펙
            new_value: 새로운 제약 값
        """
        constraint_type = constraint_spec['type']

        # Constraint manager를 통해 제약조건 수정
        for constraint in self.engine.constraint_manager.constraints:
            if constraint.name == constraint_spec['name']:
                # 제약조건 파라미터 수정
                if hasattr(constraint, 'premium_limit_pct'):
                    constraint.premium_limit_pct = new_value
                elif hasattr(constraint, 'absolute_premium_budget'):
                    constraint.absolute_premium_budget = new_value
                # 필요한 경우 다른 제약 속성 추가
                break

    def prioritize_constraints(
        self,
        method: str = 'marginal_benefit'
    ) -> pd.DataFrame:
        """
        제약조건 우선순위화

        Args:
            method: 우선순위 기준
                - 'marginal_benefit': 한계 편익 기준
                - 'total_improvement': 총 개선 효과 기준
                - 'dual_value': Shadow price 기준 (IPOPT 사용 시)

        Returns:
            우선순위화된 제약조건 데이터프레임
        """
        if not self.analysis_results:
            raise ValueError("먼저 analyze_relaxation_impact()를 실행하세요.")

        priority_data = []

        for constraint_name, result in self.analysis_results['constraint_results'].items():
            if len(result['objective_improvements']) == 0:
                continue

            # 최대 개선 효과
            max_improvement = max(result['objective_improvements']) if result['objective_improvements'] else 0

            # 평균 한계 편익
            avg_marginal_benefit = np.mean(result['marginal_benefits']) if result['marginal_benefits'] else 0

            # 첫 단계 한계 편익 (가장 민감)
            first_marginal_benefit = result['marginal_benefits'][0] if result['marginal_benefits'] else 0

            priority_data.append({
                '제약조건': constraint_name,
                '현재_값': result['constraint_spec']['current_value'],
                '최대_개선(%)': max_improvement,
                '평균_한계편익': avg_marginal_benefit,
                '초기_한계편익': first_marginal_benefit,
                '완화_방향': result['constraint_spec']['relaxation_direction']
            })

        priority_df = pd.DataFrame(priority_data)

        # 우선순위 결정
        if method == 'marginal_benefit':
            priority_df = priority_df.sort_values('평균_한계편익', ascending=False)
        elif method == 'total_improvement':
            priority_df = priority_df.sort_values('최대_개선(%)', ascending=False)
        else:
            priority_df = priority_df.sort_values('평균_한계편익', ascending=False)

        priority_df = priority_df.reset_index(drop=True)
        priority_df.insert(0, '우선순위', range(1, len(priority_df) + 1))

        return priority_df

    def generate_recommendations(
        self,
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        제약 완화 추천 생성

        Args:
            top_n: 상위 N개 제약조건 추천

        Returns:
            추천 리스트
        """
        priority_df = self.prioritize_constraints()

        recommendations = []

        for idx, row in priority_df.head(top_n).iterrows():
            constraint_name = row['제약조건']
            result = self.analysis_results['constraint_results'][constraint_name]

            # 최적 완화 수준 찾기 (marginal benefit이 급감하는 지점)
            marginal_benefits = result['marginal_benefits']
            relaxation_levels = result['relaxation_levels']

            if len(marginal_benefits) >= 2:
                # 한계 편익 변화율 계산
                benefit_changes = np.diff(marginal_benefits)
                # 가장 큰 감소가 있는 지점 (수확 체감 시작)
                optimal_idx = np.argmin(benefit_changes) if len(benefit_changes) > 0 else 0
                optimal_relaxation = relaxation_levels[optimal_idx]
            else:
                optimal_relaxation = relaxation_levels[0] if relaxation_levels else 5

            recommendation = {
                '우선순위': row['우선순위'],
                '제약조건': constraint_name,
                '현재_값': row['현재_값'],
                '추천_완화_수준(%)': optimal_relaxation,
                '예상_개선(%)': result['objective_improvements'][optimal_idx] if len(result['objective_improvements']) > optimal_idx else 0,
                '설명': self._generate_explanation(row, result, optimal_relaxation)
            }

            recommendations.append(recommendation)

        return recommendations

    def _generate_explanation(
        self,
        priority_row: pd.Series,
        result: Dict[str, Any],
        optimal_relaxation: float
    ) -> str:
        """
        추천 설명 생성

        Args:
            priority_row: 우선순위 데이터프레임 행
            result: 제약 완화 결과
            optimal_relaxation: 최적 완화 수준

        Returns:
            설명 텍스트
        """
        constraint_name = priority_row['제약조건']
        max_improvement = priority_row['최대_개선(%)']
        direction = priority_row['완화_방향']

        direction_text = "증가" if direction == 'increase' else "감소"

        explanation = (
            f"**{constraint_name}** 제약을 {optimal_relaxation}% {direction_text}시키면 "
            f"목적함수가 약 {max_improvement:.2f}% 개선될 것으로 예상됩니다.\n\n"
            f"이 제약조건은 현재 최적화에서 가장 큰 방해 요소이며, "
            f"완화 시 높은 효과를 기대할 수 있습니다."
        )

        return explanation

    def export_results(self) -> Dict[str, Any]:
        """
        분석 결과 Export

        Returns:
            직렬화 가능한 결과 딕셔너리
        """
        if not self.analysis_results:
            return {}

        export_data = {
            'base_objective': self.analysis_results['base_objective'],
            'objective_type': self.analysis_results['objective_type'],
            'constraints': {}
        }

        for constraint_name, result in self.analysis_results['constraint_results'].items():
            export_data['constraints'][constraint_name] = {
                'relaxation_levels': result['relaxation_levels'],
                'objective_values': result['objective_values'],
                'objective_improvements': result['objective_improvements'],
                'marginal_benefits': result['marginal_benefits']
            }

        return export_data

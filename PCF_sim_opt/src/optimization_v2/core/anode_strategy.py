"""
음극재 최적화 전략

기존 Anode Active Material (Natural/Artificial Graphite) 자재의
composition 비율을 최적화하고 RE100을 적용합니다.
"""

from typing import Dict, List, Any, Optional
import pyomo.environ as pyo
from .material_strategy import MaterialOptimizationStrategy


class AnodeOptimizationStrategy(MaterialOptimizationStrategy):
    """
    음극재 최적화 전략

    - Composition 최적화: Natural Graphite vs Artificial Graphite 비율
    - RE100 적용: Tier1 (Anode 제조), Tier2 (전구체 제조)
    """

    def __init__(
        self,
        material_name: str,
        material_data: Dict[str, Any],
        anode_data: Dict[str, Any]
    ):
        """
        음극재 전략 초기화

        Args:
            material_name: 자재명
            material_data: 자재 데이터 (classification에서 추출)
            anode_data: 음극재 전용 데이터
                - natural_emission: Natural Graphite 배출계수
                - artificial_emission: Artificial Graphite 배출계수
                - natural_quantity: Natural Graphite 총 소요량
                - artificial_quantity: Artificial Graphite 총 소요량
                - natural_materials: Natural Graphite 자재 리스트
                - artificial_materials: Artificial Graphite 자재 리스트
        """
        super().__init__(material_name, material_data)
        self.anode_data = anode_data

    def get_optimization_type(self) -> str:
        return "composition"  # Composition 최적화

    def define_variables(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> List[str]:
        """
        음극재 최적화 변수 정의

        Returns:
            정의된 변수명 리스트
        """
        # 음극재 composition 변수는 전역으로 정의 (모든 음극재가 공유)
        # 이미 정의되어 있으면 스킵
        if hasattr(model, 'natural_graphite_ratio'):
            return [
                'natural_graphite_ratio',
                'artificial_graphite_ratio',
                'anode_emission_factor'
            ]

        # 현재 데이터 기반 초기 비율 계산
        natural_qty = self.anode_data.get('natural_quantity', 0)
        artificial_qty = self.anode_data.get('artificial_quantity', 0)
        total_qty = natural_qty + artificial_qty

        if total_qty > 0:
            initial_natural_ratio = natural_qty / total_qty
            initial_artificial_ratio = artificial_qty / total_qty
        else:
            # Fallback: 균등 분배
            initial_natural_ratio = 0.5
            initial_artificial_ratio = 0.5

        print(f"   📊 음극재 초기 비율: Natural={initial_natural_ratio:.1%}, Artificial={initial_artificial_ratio:.1%}")

        # Natural vs Artificial Graphite 비율
        model.natural_graphite_ratio = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            initialize=initial_natural_ratio,  # 초기값 설정
            doc="Natural Graphite 비율"
        )

        model.artificial_graphite_ratio = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            initialize=initial_artificial_ratio,  # 초기값 설정
            doc="Artificial Graphite 비율"
        )

        # 초기 배출계수 추정
        natural_emission = self.anode_data.get('natural_emission', 0)
        artificial_emission = self.anode_data.get('artificial_emission', 0)
        initial_emission = (
            initial_natural_ratio * natural_emission +
            initial_artificial_ratio * artificial_emission
        )

        # 음극재 전체 배출계수 (계산값)
        model.anode_emission_factor = pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=initial_emission,  # 초기값 설정
            doc="음극재 전체 배출계수"
        )

        print(f"   ✅ 음극재 전용 변수 추가 (Natural/Artificial composition)")
        print(f"      • Natural 배출계수: {natural_emission:.3f} kgCO2eq/kg")
        print(f"      • Artificial 배출계수: {artificial_emission:.3f} kgCO2eq/kg")
        print(f"      • 초기 가중평균 배출계수: {initial_emission:.3f} kgCO2eq/kg")

        return [
            'natural_graphite_ratio',
            'artificial_graphite_ratio',
            'anode_emission_factor'
        ]

    def add_constraints(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> None:
        """
        음극재 제약 조건 추가
        """
        # 이미 제약조건이 추가되어 있으면 스킵 (전역 제약)
        if hasattr(model, 'anode_ratio_sum_constraint'):
            return

        # 1. Natural + Artificial = 1
        def anode_ratio_sum_rule(model):
            return (
                model.natural_graphite_ratio +
                model.artificial_graphite_ratio == 1.0
            )

        model.anode_ratio_sum_constraint = pyo.Constraint(
            rule=anode_ratio_sum_rule,
            doc="음극재 비율 합 = 1"
        )

        print(f"   ✅ 음극재 비율 제약조건 추가")

    def _calculate_avg_energy_ratio(
        self,
        materials: List[str],
        tier: str,
        material_classification: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        자재 그룹의 평균 에너지 비율 계산

        Args:
            materials: 자재명 리스트
            tier: 'tier1' 또는 'tier2'
            material_classification: 자재 분류 정보

        Returns:
            평균 에너지 비율 (0~1 소수)
        """
        if not materials:
            return 0.0

        ratios = [
            material_classification[m].get(f'{tier}_energy_ratio', 0)
            for m in materials
        ]

        # 평균 계산
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        return avg_ratio

    def _add_uniform_re_constraints(
        self,
        model: pyo.ConcreteModel,
        anode_materials: List[str]
    ) -> None:
        """
        모든 음극재가 동일한 RE100 비율을 사용하도록 제약 추가 (호환성 유지용)

        Args:
            model: Pyomo 모델
            anode_materials: 음극재 자재 리스트
        """
        if len(anode_materials) <= 1:
            return

        representative = anode_materials[0]

        def same_re_tier1_rule(model, anode_mat):
            if anode_mat == representative:
                return pyo.Constraint.Skip
            return model.tier1_re[anode_mat] == model.tier1_re[representative]

        def same_re_tier2_rule(model, anode_mat):
            if anode_mat == representative:
                return pyo.Constraint.Skip
            return model.tier2_re[anode_mat] == model.tier2_re[representative]

        # 음극재 집합 정의
        model.anode_materials_set = pyo.Set(initialize=anode_materials)

        model.same_anode_re_tier1_constraint = pyo.Constraint(
            model.anode_materials_set,
            rule=same_re_tier1_rule,
            doc="모든 음극재가 같은 tier1_re 사용"
        )

        model.same_anode_re_tier2_constraint = pyo.Constraint(
            model.anode_materials_set,
            rule=same_re_tier2_rule,
            doc="모든 음극재가 같은 tier2_re 사용"
        )

        print(f"   ℹ️  [호환성 모드] 음극재 {len(anode_materials)}개 → 모두 동일한 RE100 비율 사용")

    def add_anode_emission_constraint(
        self,
        model: pyo.ConcreteModel,
        anode_materials: List[str],
        material_classification: Dict[str, Dict[str, Any]],
        force_uniform_re100: bool = False
    ) -> None:
        """
        음극재 전체 배출계수 제약조건 추가 (RE100 포함)

        이 함수는 별도로 호출되어야 합니다 (material_classification 필요)

        Args:
            model: Pyomo 모델
            anode_materials: 음극재 자재 리스트
            material_classification: 자재 분류 정보
            force_uniform_re100: True이면 모든 음극재가 동일한 RE100 사용 (기본값: False)
                                False이면 Natural/Artificial 타입별 독립적 RE100 사용
        """
        natural_emission = self.anode_data['natural_emission']
        artificial_emission = self.anode_data['artificial_emission']

        if not anode_materials:
            # 음극재가 없으면 RE100 없이 계산
            def anode_emission_rule(model):
                return model.anode_emission_factor == (
                    model.natural_graphite_ratio * natural_emission +
                    model.artificial_graphite_ratio * artificial_emission
                )
        else:
            # Natural과 Artificial 자재 분리
            natural_materials = [m for m in anode_materials if 'Natural' in m]
            artificial_materials = [
                m for m in anode_materials
                if 'Artificial' in m or 'Synthetic' in m
            ]

            # 각 타입별 평균 에너지 비율 계산
            natural_tier1_ratio = self._calculate_avg_energy_ratio(
                natural_materials, 'tier1', material_classification
            )
            natural_tier2_ratio = self._calculate_avg_energy_ratio(
                natural_materials, 'tier2', material_classification
            )

            artificial_tier1_ratio = self._calculate_avg_energy_ratio(
                artificial_materials, 'tier1', material_classification
            )
            artificial_tier2_ratio = self._calculate_avg_energy_ratio(
                artificial_materials, 'tier2', material_classification
            )

            print(f"\n   🔋 음극재 RE100 독립성 모드:")
            print(f"      • Natural Graphite ({len(natural_materials)}개): "
                  f"Tier1={natural_tier1_ratio*100:.1f}%, Tier2={natural_tier2_ratio*100:.1f}%")
            print(f"      • Artificial Graphite ({len(artificial_materials)}개): "
                  f"Tier1={artificial_tier1_ratio*100:.1f}%, Tier2={artificial_tier2_ratio*100:.1f}%")

            def anode_emission_rule(model):
                # Step 1: Composition으로 계산된 기본 배출계수
                base_emission = (
                    model.natural_graphite_ratio * natural_emission +
                    model.artificial_graphite_ratio * artificial_emission
                )

                # Step 2: 타입별 독립적인 RE100 감축
                natural_re_reduction = 0.0
                if natural_materials:
                    # Natural Graphite 대표 자재의 RE100 변수 사용
                    natural_re_reduction = (
                        model.tier1_re[natural_materials[0]] * natural_tier1_ratio +
                        model.tier2_re[natural_materials[0]] * natural_tier2_ratio
                    ) * model.natural_graphite_ratio

                artificial_re_reduction = 0.0
                if artificial_materials:
                    # Artificial Graphite 대표 자재의 RE100 변수 사용
                    artificial_re_reduction = (
                        model.tier1_re[artificial_materials[0]] * artificial_tier1_ratio +
                        model.tier2_re[artificial_materials[0]] * artificial_tier2_ratio
                    ) * model.artificial_graphite_ratio

                total_re_reduction = natural_re_reduction + artificial_re_reduction

                # Step 3: 최종 배출계수 = 기본 배출계수 × (1 - RE100 감축)
                return model.anode_emission_factor == base_emission * (1 - total_re_reduction)

            # 호환성 유지: force_uniform_re100=True이면 기존 동작 (모든 음극재 동일 RE100)
            if force_uniform_re100:
                self._add_uniform_re_constraints(model, anode_materials)

        model.anode_emission_calc = pyo.Constraint(
            rule=anode_emission_rule,
            doc="음극재 전체 배출계수 계산 (composition+RE100)"
        )

    def calculate_emission_factor(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> float:
        """음극재 배출계수 반환"""
        return float(pyo.value(model.anode_emission_factor))

    def extract_solution(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> Dict[str, Any]:
        """
        음극재 솔루션 추출

        Returns:
            추출된 결과 딕셔너리
        """
        natural_ratio = pyo.value(model.natural_graphite_ratio)
        artificial_ratio = pyo.value(model.artificial_graphite_ratio)

        # 진단 로그
        print(f"   ✓ [ANODE SOLUTION] {self.material_name[:40]}")
        print(f"      natural_graphite_ratio={natural_ratio:.4f}")
        print(f"      artificial_graphite_ratio={artificial_ratio:.4f}")

        return {
            'natural_graphite_ratio': natural_ratio,
            'artificial_graphite_ratio': artificial_ratio
        }

"""
양극재 최적화 전략

기존 optimization_engine.py의 양극재 로직을 추출한 Strategy 구현체입니다.
Element-level 최적화 + RE100을 동시에 적용합니다.
"""

from typing import Dict, List, Any
import pyomo.environ as pyo
from .material_strategy import MaterialOptimizationStrategy


class CathodeOptimizationStrategy(MaterialOptimizationStrategy):
    """
    양극재 최적화 전략

    - Element-level 최적화: Ni, Co, Mn, Li 원소별 재활용/저탄소/버진 비율
    - RE100 적용: Tier1 (CAM 제조), Tier2 (pCAM 제조)
    """

    def __init__(
        self,
        material_name: str,
        material_data: Dict[str, Any],
        cathode_data: Dict[str, Any]
    ):
        """
        양극재 전략 초기화

        Args:
            material_name: 자재명
            material_data: 자재 데이터 (classification에서 추출)
            cathode_data: 양극재 전용 데이터
                - cathode_composition: 원소 조성비
                - recycle_impact: 재활용재 impact
                - low_carb_emission: 저탄소 배출계수
                - virgin_emission: 버진 배출계수
        """
        super().__init__(material_name, material_data)
        self.cathode_data = cathode_data

    def get_optimization_type(self) -> str:
        return "hybrid"  # Element-level + RE100

    def define_variables(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> List[str]:
        """
        양극재 최적화 변수 정의

        기존 optimization_engine.py lines 161-222 로직 추출
        """
        # 양극재는 전역 변수로 정의 (모든 양극재가 공유)
        # 이미 정의되어 있으면 스킵
        if hasattr(model, 'elements'):
            return [
                'element_virgin_ratio',
                'element_recycle_ratio',
                'element_low_carb_ratio',
                'element_emission',
                'cathode_emission_factor'
            ]

        cathode_composition = self.cathode_data['cathode_composition']

        # 조성비가 0보다 큰 원소만 포함
        elements = {e for e, comp in cathode_composition.items() if comp > 0}

        # 재활용/저탄소 옵션이 있는 원소 추가
        if self.cathode_data.get('recycle_impact'):
            for e in self.cathode_data['recycle_impact'].keys():
                if e not in cathode_composition or cathode_composition.get(e, 0) > 0:
                    elements.add(e)

        if self.cathode_data.get('low_carb_emission'):
            for e in self.cathode_data['low_carb_emission'].keys():
                if e not in cathode_composition or cathode_composition.get(e, 0) > 0:
                    elements.add(e)

        if self.cathode_data.get('virgin_emission'):
            for e in self.cathode_data['virgin_emission'].keys():
                if e not in cathode_composition or cathode_composition.get(e, 0) > 0:
                    elements.add(e)

        # Set을 리스트로 변환하여 정렬
        elements = sorted(list(elements))

        model.elements = pyo.Set(initialize=elements, doc="양극재 원소 집합")

        # 원소별 신재/재활용/저탄소 비율
        model.element_virgin_ratio = pyo.Var(
            model.elements,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="원소별 신재 비율"
        )

        model.element_recycle_ratio = pyo.Var(
            model.elements,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="원소별 재활용 비율"
        )

        model.element_low_carb_ratio = pyo.Var(
            model.elements,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="원소별 저탄소메탈 비율"
        )

        # 원소별 배출계수 (계산값)
        model.element_emission = pyo.Var(
            model.elements,
            domain=pyo.NonNegativeReals,
            doc="원소별 배출계수"
        )

        # 양극재 전체 배출계수 (계산값)
        model.cathode_emission_factor = pyo.Var(
            domain=pyo.NonNegativeReals,
            doc="양극재 전체 배출계수"
        )

        print(f"   ✅ 양극재 전용 변수 추가 (원소: {elements})")

        return [
            'element_virgin_ratio',
            'element_recycle_ratio',
            'element_low_carb_ratio',
            'element_emission',
            'cathode_emission_factor'
        ]

    def add_constraints(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> None:
        """
        양극재 제약 조건 추가

        기존 optimization_engine.py lines 275-454 로직 추출
        """
        # 이미 제약조건이 추가되어 있으면 스킵 (전역 제약)
        if hasattr(model, 'element_ratio_sum_constraint'):
            return

        cathode_composition = self.cathode_data['cathode_composition']
        recycle_impact = self.cathode_data['recycle_impact']
        low_carb_emission = self.cathode_data['low_carb_emission']
        virgin_emission = self.cathode_data['virgin_emission']

        # 1. 원소별 비율 합 = 1
        def element_ratio_sum_rule(model, e):
            if e in cathode_composition and cathode_composition[e] == 0:
                return pyo.Constraint.Skip

            has_recycle = e in recycle_impact
            has_low_carb = e in low_carb_emission and low_carb_emission[e] != virgin_emission.get(e, 0)

            if not has_recycle and not has_low_carb:
                return model.element_virgin_ratio[e] == 1.0
            else:
                return (
                    model.element_virgin_ratio[e] +
                    model.element_recycle_ratio[e] +
                    model.element_low_carb_ratio[e] == 1.0
                )

        model.element_ratio_sum_constraint = pyo.Constraint(
            model.elements,
            rule=element_ratio_sum_rule,
            doc="원소별 비율 합 = 1"
        )

        # 1-1. 재활용 옵션이 없는 원소는 재활용 비율 = 0
        def recycle_unavailable_rule(model, e):
            if e not in recycle_impact:
                return model.element_recycle_ratio[e] == 0.0
            return pyo.Constraint.Skip

        model.recycle_unavailable_constraint = pyo.Constraint(
            model.elements,
            rule=recycle_unavailable_rule,
            doc="재활용 옵션이 없는 원소는 재활용 비율 0"
        )

        # 1-2. 저탄소 옵션이 없는 원소는 저탄소 비율 = 0
        def low_carb_unavailable_rule(model, e):
            has_low_carb = e in low_carb_emission and low_carb_emission[e] != virgin_emission.get(e, 0)
            if not has_low_carb:
                return model.element_low_carb_ratio[e] == 0.0
            return pyo.Constraint.Skip

        model.low_carb_unavailable_constraint = pyo.Constraint(
            model.elements,
            rule=low_carb_unavailable_rule,
            doc="저탄소 옵션이 없는 원소는 저탄소 비율 0"
        )

        # 2. 원소별 배출계수 계산
        def element_emission_rule(model, e):
            virgin_ef = virgin_emission.get(e, 0)
            recycle_imp = recycle_impact.get(e, 1.0)
            low_carb_ef = low_carb_emission.get(e, virgin_ef)

            return model.element_emission[e] == (
                model.element_virgin_ratio[e] * virgin_ef +
                model.element_recycle_ratio[e] * (virgin_ef * recycle_imp) +
                model.element_low_carb_ratio[e] * low_carb_ef
            )

        model.element_emission_calc = pyo.Constraint(
            model.elements,
            rule=element_emission_rule,
            doc="원소별 배출계수 계산"
        )

        # 3. 양극재 전체 배출계수 계산 (RE100 포함)
        # 주의: 이 부분은 material_classification이 필요하므로
        # OptimizationEngine에서 호출할 때 처리됩니다.

    def add_cathode_emission_constraint(
        self,
        model: pyo.ConcreteModel,
        cathode_materials: List[str],
        material_classification: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        양극재 전체 배출계수 제약조건 추가 (RE100 포함)

        이 함수는 별도로 호출되어야 합니다 (material_classification 필요)

        Args:
            model: Pyomo 모델
            cathode_materials: 양극재 자재 리스트
            material_classification: 자재 분류 정보
        """
        cathode_composition = self.cathode_data['cathode_composition']

        if not cathode_materials:
            # RE100 없이 계산
            def cathode_emission_rule(model):
                return model.cathode_emission_factor == sum(
                    cathode_composition[e] * model.element_emission[e]
                    for e in cathode_composition.keys()
                    if cathode_composition[e] > 0
                )
        else:
            # RE100 적용
            cathode_material = cathode_materials[0]

            def cathode_emission_rule(model):
                # Step 1: Element-level 배출계수
                base_emission = sum(
                    cathode_composition[e] * model.element_emission[e]
                    for e in cathode_composition.keys()
                    if cathode_composition[e] > 0
                )

                # Step 2: RE100 감축 계수
                tier1_ratio = material_classification[cathode_material].get('tier1_energy_ratio', 0)
                tier2_ratio = material_classification[cathode_material].get('tier2_energy_ratio', 0)

                re100_reduction_factor = (
                    1 - model.tier1_re[cathode_material] * tier1_ratio
                      - model.tier2_re[cathode_material] * tier2_ratio
                )

                # Step 3: 최종 배출계수
                return model.cathode_emission_factor == base_emission * re100_reduction_factor

            # 모든 양극재가 같은 RE100 비율 사용
            if len(cathode_materials) > 1:
                def same_re_tier1_rule(model, cath_mat):
                    if cath_mat == cathode_material:
                        return pyo.Constraint.Skip
                    return model.tier1_re[cath_mat] == model.tier1_re[cathode_material]

                def same_re_tier2_rule(model, cath_mat):
                    if cath_mat == cathode_material:
                        return pyo.Constraint.Skip
                    return model.tier2_re[cath_mat] == model.tier2_re[cathode_material]

                model.cathode_materials_set = pyo.Set(initialize=cathode_materials)

                model.same_re_tier1_constraint = pyo.Constraint(
                    model.cathode_materials_set,
                    rule=same_re_tier1_rule,
                    doc="모든 양극재가 같은 tier1_re 사용"
                )

                model.same_re_tier2_constraint = pyo.Constraint(
                    model.cathode_materials_set,
                    rule=same_re_tier2_rule,
                    doc="모든 양극재가 같은 tier2_re 사용"
                )

                print(f"   ℹ️  양극재 {len(cathode_materials)}개 → 모두 동일한 RE100 비율 사용")

        model.cathode_emission_calc = pyo.Constraint(
            rule=cathode_emission_rule,
            doc="양극재 전체 배출계수 계산 (재활용+저탄소+RE100)"
        )

    def calculate_emission_factor(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> float:
        """양극재 배출계수 반환"""
        return float(pyo.value(model.cathode_emission_factor))

    def extract_solution(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> Dict[str, Any]:
        """
        양극재 솔루션 추출

        기존 optimization_engine.py lines 1009-1062 로직 추출
        """
        cathode_composition = self.cathode_data['cathode_composition']

        # 원소별 비율의 가중평균 계산
        total_virgin = 0.0
        total_recycle = 0.0
        total_low_carb = 0.0

        for e in model.elements:
            if e in cathode_composition and cathode_composition[e] > 0:
                comp_ratio = cathode_composition[e]
                total_virgin += comp_ratio * pyo.value(model.element_virgin_ratio[e])
                total_recycle += comp_ratio * pyo.value(model.element_recycle_ratio[e])
                total_low_carb += comp_ratio * pyo.value(model.element_low_carb_ratio[e])

        # 진단 로그
        print(f"   ✓ [POINT A - RATIOS] {self.material_name[:40]}")
        print(f"      virgin_ratio={total_virgin}")
        print(f"      recycle_ratio={total_recycle}")
        print(f"      low_carbon_ratio={total_low_carb}")

        return {
            'virgin_ratio': total_virgin,
            'recycle_ratio': total_recycle,
            'low_carbon_ratio': total_low_carb
        }

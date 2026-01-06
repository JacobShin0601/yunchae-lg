"""
최적화 엔진

Pyomo를 사용하여 최적화 모델을 구축하고 실행하는 핵심 엔진입니다.
"""

from typing import Dict, Any, Optional, List
import pyomo.environ as pyo
from .constraint_manager import ConstraintManager


class OptimizationEngine:
    """
    최적화 엔진 클래스

    - Pyomo 모델 구축
    - 제약조건 적용
    - 목적함수 설정
    - 솔버 실행
    - 결과 추출
    """

    def __init__(self, solver_name: str = 'glpk'):
        """
        최적화 엔진 초기화

        Args:
            solver_name: 솔버 이름 ('glpk', 'cbc', 'ipopt', 'auto')
        """
        self.solver_name = solver_name
        self.model: Optional[pyo.ConcreteModel] = None
        self.solver: Optional[Any] = None
        self.results: Optional[Any] = None
        self.constraint_manager = ConstraintManager()
        self.data: Optional[Dict[str, Any]] = None
        self.is_nonlinear: bool = False  # 모델 비선형 여부

    def build_model(
        self,
        data: Dict[str, Any],
        objective_type: str = 'minimize_carbon'
    ) -> pyo.ConcreteModel:
        """
        Pyomo 최적화 모델 구축

        Args:
            data: 최적화 데이터 (DataLoader에서 제공)
            objective_type: 목적함수 유형
                - 'minimize_carbon': 탄소배출 최소화
                - 'minimize_cost': 비용 최소화
                - 'multi_objective': 다목적 최적화

        Returns:
            Pyomo ConcreteModel
        """
        print("\n" + "=" * 60)
        print("🔨 최적화 모델 구축 시작")
        print("=" * 60)

        self.data = data
        self.model = pyo.ConcreteModel()

        # 자재 집합 정의
        target_materials = data['target_materials']
        material_classification = data['material_classification']

        self.model.materials = pyo.Set(initialize=target_materials)
        print(f"📦 대상 자재: {len(target_materials)}개")

        # 결정 변수 정의
        self._define_variables(material_classification)

        # 기본 제약조건 추가 (비율 합 = 1)
        self._add_basic_constraints(material_classification)

        # 양극재 관련 제약조건 추가
        self._add_cathode_constraints(material_classification)

        # 배출계수 계산 (목적함수용)
        self._define_emission_expressions(data, material_classification)

        # 사용자 정의 제약조건 적용
        if len(self.constraint_manager.list_constraints(enabled_only=True)) > 0:
            print(f"\n⚙️  사용자 정의 제약조건 적용 중...")
            self.constraint_manager.apply_all_to_model(self.model, data)

        # 목적함수 설정
        self._set_objective(data, objective_type)

        # 모델 비선형 여부 체크
        self.is_nonlinear = self._check_nonlinearity()

        print(f"\n✅ 모델 구축 완료!")
        print(f"   • 변수 수: {len(list(self.model.component_objects(pyo.Var, active=True)))}개")
        print(f"   • 제약조건 수: {len(list(self.model.component_objects(pyo.Constraint, active=True)))}개")
        print(f"   • 목적함수: {objective_type}")
        print(f"   • 문제 유형: {'비선형 (NLP)' if self.is_nonlinear else '선형 (LP)'}")

        return self.model

    def _define_variables(self, material_classification: Dict[str, Dict[str, Any]]) -> None:
        """
        결정 변수 정의

        Args:
            material_classification: 자재 분류 정보
        """
        print("\n📝 결정 변수 정의 중...")

        # Formula 자재용 변수: Tier1/Tier2 RE 비율
        self.model.tier1_re = pyo.Var(
            self.model.materials,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="Tier1 RE 적용 비율"
        )

        self.model.tier2_re = pyo.Var(
            self.model.materials,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="Tier2 RE 적용 비율"
        )

        # Ni/Co/Li 자재용 변수: 재활용/저탄소/버진 비율
        self.model.recycle_ratio = pyo.Var(
            self.model.materials,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="재활용재 비율"
        )

        self.model.low_carbon_ratio = pyo.Var(
            self.model.materials,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="저탄소메탈 비율"
        )

        self.model.virgin_ratio = pyo.Var(
            self.model.materials,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="버진 자재 비율"
        )

        # 수정된 배출계수
        self.model.modified_emission = pyo.Var(
            self.model.materials,
            domain=pyo.NonNegativeReals,
            doc="수정된 배출계수"
        )

        # 양극재 전용 변수: 원소별 비율 및 배출계수
        if self.data and self.data.get('cathode_composition'):
            # 조성비가 0보다 큰 원소만 포함 (조성비 0인 원소는 최적화 불필요)
            cathode_composition = self.data['cathode_composition']
            elements = {e for e, comp in cathode_composition.items() if comp > 0}

            # 재활용/저탄소 옵션이 있는 원소 추가 (Li 포함)
            # 단, 이들도 조성비가 0보다 큰 경우에만 포함
            if self.data.get('recycle_impact'):
                for e in self.data['recycle_impact'].keys():
                    if e not in cathode_composition or cathode_composition.get(e, 0) > 0:
                        elements.add(e)
            if self.data.get('low_carb_emission'):
                for e in self.data['low_carb_emission'].keys():
                    if e not in cathode_composition or cathode_composition.get(e, 0) > 0:
                        elements.add(e)
            if self.data.get('virgin_emission'):
                for e in self.data['virgin_emission'].keys():
                    if e not in cathode_composition or cathode_composition.get(e, 0) > 0:
                        elements.add(e)

            # Set을 리스트로 변환하여 정렬 (일관성 유지)
            elements = sorted(list(elements))

            self.model.elements = pyo.Set(initialize=elements, doc="양극재 원소 집합")

            # 원소별 신재/재활용/저탄소 비율
            self.model.element_virgin_ratio = pyo.Var(
                self.model.elements,
                domain=pyo.NonNegativeReals,
                bounds=(0, 1),
                doc="원소별 신재 비율"
            )

            self.model.element_recycle_ratio = pyo.Var(
                self.model.elements,
                domain=pyo.NonNegativeReals,
                bounds=(0, 1),
                doc="원소별 재활용 비율"
            )

            self.model.element_low_carb_ratio = pyo.Var(
                self.model.elements,
                domain=pyo.NonNegativeReals,
                bounds=(0, 1),
                doc="원소별 저탄소메탈 비율"
            )

            # 원소별 배출계수 (계산값)
            self.model.element_emission = pyo.Var(
                self.model.elements,
                domain=pyo.NonNegativeReals,
                doc="원소별 배출계수"
            )

            # 양극재 전체 배출계수 (계산값)
            self.model.cathode_emission_factor = pyo.Var(
                domain=pyo.NonNegativeReals,
                doc="양극재 전체 배출계수"
            )

            print(f"   ✅ 양극재 전용 변수 추가 (원소: {elements})")

        print("   ✅ 변수 정의 완료")

    def _add_basic_constraints(self, material_classification: Dict[str, Dict[str, Any]]) -> None:
        """
        기본 제약조건 추가 (비율 합 = 1)

        Args:
            material_classification: 자재 분류 정보
        """
        print("\n🔧 기본 제약조건 추가 중...")

        # Ni/Co/Li 자재: recycle + low_carbon + virgin = 1
        def ratio_sum_rule(model, m):
            if material_classification[m]['is_ni_co_li']:
                return (model.recycle_ratio[m] +
                       model.low_carbon_ratio[m] +
                       model.virgin_ratio[m]) == 1
            return pyo.Constraint.Skip

        self.model.ratio_sum_constraint = pyo.Constraint(
            self.model.materials,
            rule=ratio_sum_rule,
            doc="Ni/Co/Li 자재 비율 합 = 1"
        )

        print("   ✅ 기본 제약조건 추가 완료")

    def _add_cathode_constraints(self, material_classification: Dict[str, Dict[str, Any]]) -> None:
        """
        양극재 관련 제약조건 추가

        - 원소별 비율 합 = 1
        - 원소별 배출계수 계산
        - 양극재 전체 배출계수 계산 (RE100 포함)
        - 양극재 자재 레벨 tier1_re/tier2_re를 0으로 고정 (element-level에서만 사용)

        Args:
            material_classification: 자재 분류 정보
        """
        if not hasattr(self.model, 'elements'):
            return  # 양극재 데이터가 없으면 스킵

        print("\n🔋 양극재 제약조건 추가 중...")

        # 양극재 자재 식별
        cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
        cathode_materials = [m for m in self.model.materials
                            if any(kw in m for kw in cathode_keywords)]

        # 양극재는 tier1_re/tier2_re를 최적화 변수로 유지
        # (element-level 최적화 + RE100 감축 모두 적용)
        if cathode_materials:
            print(f"   ℹ️  양극재 {len(cathode_materials)}개: Element-level 최적화 + RE100 동시 적용")

        cathode_composition = self.data['cathode_composition']
        recycle_impact = self.data['recycle_impact']
        low_carb_emission = self.data['low_carb_emission']
        virgin_emission = self.data['virgin_emission']

        # 1. 원소별 비율 합 = 1 (재활용/저탄소 옵션이 있는 원소만)
        def element_ratio_sum_rule(model, e):
            # 조성비가 0인 원소는 제약조건 스킵
            composition = self.data['cathode_composition']
            if e in composition and composition[e] == 0:
                return pyo.Constraint.Skip

            # 재활용/저탄소 옵션 확인
            has_recycle = e in recycle_impact
            has_low_carb = e in low_carb_emission and low_carb_emission[e] != virgin_emission.get(e, 0)

            if not has_recycle and not has_low_carb:
                # 재활용/저탄소 옵션이 없으면 신재 100% 고정
                return model.element_virgin_ratio[e] == 1.0
            else:
                # 재활용/저탄소 옵션이 있으면 비율 합 = 1
                return (
                    model.element_virgin_ratio[e] +
                    model.element_recycle_ratio[e] +
                    model.element_low_carb_ratio[e] == 1.0
                )

        self.model.element_ratio_sum_constraint = pyo.Constraint(
            self.model.elements,
            rule=element_ratio_sum_rule,
            doc="원소별 비율 합 = 1 (또는 신재 100%)"
        )

        # 1-1. 재활용 옵션이 없는 원소는 재활용 비율 = 0 고정
        def recycle_unavailable_rule(model, e):
            if e not in recycle_impact:
                return model.element_recycle_ratio[e] == 0.0
            return pyo.Constraint.Skip

        self.model.recycle_unavailable_constraint = pyo.Constraint(
            self.model.elements,
            rule=recycle_unavailable_rule,
            doc="재활용 옵션이 없는 원소는 재활용 비율 0"
        )

        # 1-2. 저탄소 옵션이 없는 원소는 저탄소 비율 = 0 고정
        def low_carb_unavailable_rule(model, e):
            has_low_carb = e in low_carb_emission and low_carb_emission[e] != virgin_emission.get(e, 0)
            if not has_low_carb:
                return model.element_low_carb_ratio[e] == 0.0
            return pyo.Constraint.Skip

        self.model.low_carb_unavailable_constraint = pyo.Constraint(
            self.model.elements,
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

        self.model.element_emission_calc = pyo.Constraint(
            self.model.elements,
            rule=element_emission_rule,
            doc="원소별 배출계수 계산"
        )

        # 3. 양극재 전체 배출계수 = Σ(원소 조성비 × 원소 배출계수) × RE100 감축
        # 주의: cathode_composition에 있는 원소만 사용 (Li는 조성비에 포함 안됨)

        # 양극재 자재 찾기 (Cathode Active Material)
        cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
        cathode_materials = [m for m in self.model.materials
                            if any(kw in m for kw in cathode_keywords)]

        if not cathode_materials:
            # 양극재가 없으면 RE100 없이 계산
            def cathode_emission_rule(model):
                # 조성비가 0보다 큰 원소만 계산
                return model.cathode_emission_factor == sum(
                    cathode_composition[e] * model.element_emission[e]
                    for e in cathode_composition.keys()
                    if cathode_composition[e] > 0
                )
        else:
            # 양극재가 있으면 RE100 적용
            cathode_material = cathode_materials[0]  # 대표 양극재 (RE100 계산용)

            def cathode_emission_rule(model):
                # Step 1: Element-level로 계산된 기본 배출계수
                # 조성비가 0보다 큰 원소만 계산
                base_emission = sum(
                    cathode_composition[e] * model.element_emission[e]
                    for e in cathode_composition.keys()
                    if cathode_composition[e] > 0
                )

                # Step 2: RE100 감축 계수 (에너지 기여도만큼 감축)
                tier1_ratio = material_classification[cathode_material].get('tier1_energy_ratio', 0)
                tier2_ratio = material_classification[cathode_material].get('tier2_energy_ratio', 0)

                re100_reduction_factor = (
                    1 - model.tier1_re[cathode_material] * tier1_ratio
                      - model.tier2_re[cathode_material] * tier2_ratio
                )

                # Step 3: 최종 배출계수 = 기본 배출계수 × RE100 감축 계수
                return model.cathode_emission_factor == base_emission * re100_reduction_factor

            # 모든 양극재가 같은 RE100 비율을 사용하도록 강제
            if len(cathode_materials) > 1:
                def same_re_tier1_rule(model, cath_mat):
                    # 첫 번째 양극재와 동일한 tier1_re 사용
                    if cath_mat == cathode_material:
                        return pyo.Constraint.Skip
                    return model.tier1_re[cath_mat] == model.tier1_re[cathode_material]

                def same_re_tier2_rule(model, cath_mat):
                    # 첫 번째 양극재와 동일한 tier2_re 사용
                    if cath_mat == cathode_material:
                        return pyo.Constraint.Skip
                    return model.tier2_re[cath_mat] == model.tier2_re[cathode_material]

                # 양극재 집합 정의
                self.model.cathode_materials_set = pyo.Set(initialize=cathode_materials)

                self.model.same_re_tier1_constraint = pyo.Constraint(
                    self.model.cathode_materials_set,
                    rule=same_re_tier1_rule,
                    doc="모든 양극재가 같은 tier1_re 사용"
                )

                self.model.same_re_tier2_constraint = pyo.Constraint(
                    self.model.cathode_materials_set,
                    rule=same_re_tier2_rule,
                    doc="모든 양극재가 같은 tier2_re 사용"
                )

                print(f"   ℹ️  양극재 {len(cathode_materials)}개 → 모두 동일한 RE100 비율 사용")

        self.model.cathode_emission_calc = pyo.Constraint(
            rule=cathode_emission_rule,
            doc="양극재 전체 배출계수 계산 (재활용+저탄소+RE100)"
        )

        print("   ✅ 양극재 제약조건 추가 완료")

    def _define_emission_expressions(
        self,
        data: Dict[str, Any],
        material_classification: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        배출계수 계산 표현식 정의

        Args:
            data: 최적화 데이터
            material_classification: 자재 분류 정보
        """
        print("\n📊 배출계수 표현식 정의 중...")

        scenario_df = data['scenario_df']

        # 양극재 키워드 (element-level optimization 사용)
        cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']

        # 각 자재의 배출계수 계산 제약조건
        def emission_calculation_rule(model, m):
            material_info = material_classification[m]
            original_emission = material_info['original_emission']

            # 양극재 여부 확인
            is_cathode = any(keyword in m for keyword in cathode_keywords)

            if material_info['is_formula_applicable'] and not is_cathode:
                # Formula 자재 (양극재 제외): modified = original × (1 - tier1_re × tier1_ratio - tier2_re × tier2_ratio)
                # tier1/tier2_energy_ratio: 베이스라인에서 추출한 실제 에너지 기여도 (소수 형태)
                tier1_ratio = material_info.get('tier1_energy_ratio', 0)
                tier2_ratio = material_info.get('tier2_energy_ratio', 0)

                # 진단 로그 추가
                if tier1_ratio > 0 or tier2_ratio > 0:
                    print(f"      ✅ Formula 자재 RE100 적용: {m[:50]}...")
                    print(f"         Tier1: {tier1_ratio*100:.1f}%, Tier2: {tier2_ratio*100:.1f}%")

                reduction_factor = (1 - model.tier1_re[m] * tier1_ratio - model.tier2_re[m] * tier2_ratio)
                return model.modified_emission[m] == original_emission * reduction_factor

            elif is_cathode:
                # 양극재: element-level optimization 사용 (별도 제약조건에서 처리)
                # modified_emission은 사용되지 않지만 변수는 정의되어야 함
                return model.modified_emission[m] == original_emission

            elif material_info['is_ni_co_li']:
                # Ni/Co/Li 자재: 가중평균
                # modified = virgin_ratio × virgin_emission +
                #           recycle_ratio × recycle_emission +
                #           low_carbon_ratio × low_carbon_emission
                virgin_emission = original_emission

                # 자재명에서 원소 추출 (Ni, Co, Li 중 하나)
                element = None
                for elem in ['Ni', 'Co', 'Li']:
                    if elem in m:
                        element = elem
                        break

                # 데이터에서 재활용 영향도 및 저탄소 배출계수 가져오기
                recycle_impact_data = data.get('recycle_impact', {})
                low_carb_emission_data = data.get('low_carb_emission', {})

                # 재활용재 배출계수 계산
                if element and element in recycle_impact_data:
                    # recycle_impact는 감축 비율 (예: 0.1 = 90% 감축)
                    recycle_impact_factor = recycle_impact_data[element]
                    recycle_emission = original_emission * recycle_impact_factor
                else:
                    # 기본값: 30% 감축 (0.7)
                    recycle_emission = original_emission * 0.7

                # 저탄소메탈 배출계수
                if element and element in low_carb_emission_data:
                    # 절대값 배출계수 사용
                    low_carbon_emission = low_carb_emission_data[element]
                else:
                    # 기본값: 50% 감축 (0.5)
                    low_carbon_emission = original_emission * 0.5

                return model.modified_emission[m] == (
                    model.virgin_ratio[m] * virgin_emission +
                    model.recycle_ratio[m] * recycle_emission +
                    model.low_carbon_ratio[m] * low_carbon_emission
                )

            else:
                # 일반 자재: 변화 없음
                return model.modified_emission[m] == original_emission

        self.model.emission_calculation = pyo.Constraint(
            self.model.materials,
            rule=emission_calculation_rule,
            doc="배출계수 계산"
        )

        print("   ✅ 배출계수 표현식 정의 완료")

    def _set_objective(self, data: Dict[str, Any], objective_type: str) -> None:
        """
        목적함수 설정

        Args:
            data: 최적화 데이터
            objective_type: 목적함수 유형
        """
        print(f"\n🎯 목적함수 설정: {objective_type}")

        scenario_df = data['scenario_df']
        material_classification = data['material_classification']

        # [DEPRECATED] Legacy multi-objective mode removed in Phase 4.3
        # Multi-objective optimization now handled by WeightSweepOptimizer
        # 복합 최적화 모드는 WeightSweepOptimizer에서 처리합니다.

        if objective_type == 'minimize_carbon':
            # 총 탄소배출 최소화 (기본)
            def carbon_objective_rule(model):
                total_carbon = 0

                # 양극재 관련 변수 존재 여부 확인
                has_cathode_vars = hasattr(model, 'cathode_emission_factor')

                # 양극재 자재 식별 키워드
                cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']

                for m in model.materials:
                    quantity = material_classification[m]['quantity']

                    # 양극재 자재 여부 확인
                    is_cathode_material = any(keyword in m for keyword in cathode_keywords)

                    if is_cathode_material and has_cathode_vars:
                        # 양극재는 원소별 계산된 배출계수 사용
                        total_carbon += model.cathode_emission_factor * quantity
                        print(f"      • {m}: 양극재 전용 배출계수 사용")
                    else:
                        # 일반 자재는 기존 수정 배출계수 사용
                        total_carbon += model.modified_emission[m] * quantity

                return total_carbon

            self.model.objective = pyo.Objective(
                rule=carbon_objective_rule,
                sense=pyo.minimize,
                doc="총 탄소배출 최소화"
            )
            print("   ✅ 목적함수 설정 완료 (탄소 최소화)")

        elif objective_type == 'minimize_cost':
            # 비용 최소화
            print("   ⚠️  비용 최소화는 아직 구현되지 않았습니다. 탄소 최소화로 대체합니다.")
            self._set_objective(data, 'minimize_carbon')

        elif objective_type == 'multi_objective':
            # 다목적 최적화 (레거시 - cost constraint로 대체됨)
            print("   ⚠️  다목적 최적화는 CostConstraint의 복합 최적화 모드를 사용하세요.")
            self._set_objective(data, 'minimize_carbon')

        else:
            raise ValueError(f"알 수 없는 목적함수 유형: {objective_type}")

    def _check_nonlinearity(self) -> bool:
        """
        모델의 비선형 여부 확인

        Returns:
            True if 모델이 비선형 문제
        """
        if not self.model:
            return False

        # 양극재 전용 변수가 있고 RE100 곱셈 제약이 있으면 비선형
        has_cathode_vars = hasattr(self.model, 'cathode_emission_factor')
        has_cathode_constraint = hasattr(self.model, 'cathode_emission_calc')

        # RE100과 원소별 배출계수의 곱셈이 있으면 비선형
        is_nonlinear = has_cathode_vars and has_cathode_constraint

        return is_nonlinear

    def _auto_select_solver(self) -> str:
        """
        모델 특성에 따라 최적 솔버 자동 선택

        Returns:
            선택된 솔버 이름
        """
        # 비선형 문제는 IPOPT 필요
        if self.is_nonlinear:
            # IPOPT 사용 가능 여부 확인
            if pyo.SolverFactory('ipopt').available():
                print("   🔍 비선형 문제 감지 → IPOPT 선택")
                return 'ipopt'
            else:
                print("   ⚠️  비선형 문제이지만 IPOPT 불가 → GLPK 시도 (실패 가능)")
                return 'glpk'

        # 선형 문제는 GLPK (빠르고 안정적)
        if pyo.SolverFactory('glpk').available():
            print("   🔍 선형 문제 감지 → GLPK 선택")
            return 'glpk'
        elif pyo.SolverFactory('cbc').available():
            print("   🔍 선형 문제 감지 → CBC 선택 (GLPK 불가)")
            return 'cbc'
        else:
            # Fallback
            return 'ipopt'

    def solve(
        self,
        time_limit: int = 300,
        gap_tolerance: float = 0.01,
        verbose: bool = True
    ) -> Any:
        """
        최적화 문제 해결

        Args:
            time_limit: 시간 제한 (초)
            gap_tolerance: 갭 허용 오차
            verbose: 솔버 출력 표시 여부

        Returns:
            솔버 결과 객체
        """
        if not self.model:
            raise Exception("모델이 구축되지 않았습니다. build_model()을 먼저 호출하세요.")

        # 자동 솔버 선택 ('auto'인 경우)
        if self.solver_name == 'auto':
            self.solver_name = self._auto_select_solver()

        print("\n" + "=" * 60)
        print(f"🚀 최적화 실행: {self.solver_name}")
        print("=" * 60)

        # 솔버 생성
        self.solver = pyo.SolverFactory(self.solver_name)

        if not self.solver.available():
            available_solvers = [s for s in ['glpk', 'cbc', 'ipopt'] if pyo.SolverFactory(s).available()]
            raise Exception(
                f"솔버 '{self.solver_name}'을(를) 사용할 수 없습니다. "
                f"사용 가능한 솔버: {available_solvers}"
            )

        # 솔버 옵션 설정
        solver_options = {}

        if self.solver_name in ['glpk', 'cbc']:
            solver_options['mipgap'] = gap_tolerance
            solver_options['tmlim'] = time_limit  # GLPK의 시간 제한

        elif self.solver_name == 'ipopt':
            solver_options['max_cpu_time'] = time_limit
            solver_options['tol'] = gap_tolerance

        for key, value in solver_options.items():
            self.solver.options[key] = value

        print(f"📋 솔버 옵션:")
        for key, value in solver_options.items():
            print(f"   • {key}: {value}")

        # 최적화 실행
        print(f"\n⏳ 최적화 실행 중...")
        self.results = self.solver.solve(self.model, tee=verbose)

        # 결과 요약
        print(f"\n" + "=" * 60)
        print(f"📊 최적화 결과")
        print("=" * 60)
        print(f"   • 상태: {self.results.solver.termination_condition}")
        print(f"   • 솔버 상태: {self.results.solver.status}")

        # 목적함수 값은 최적해를 찾았을 때만 출력
        if self.results.solver.termination_condition == pyo.TerminationCondition.optimal:
            try:
                obj_value = pyo.value(self.model.objective)
                print(f"   • 목적함수 값: {obj_value:.4f}")
                print(f"   ✅ 최적해 발견!")
            except ValueError as e:
                print(f"   ⚠️  목적함수 값 계산 실패: {e}")
                print(f"   ❌ 일부 변수가 초기화되지 않았습니다.")
                # 변수 초기화 상태 확인
                self._check_variable_initialization()
        elif self.results.solver.termination_condition == pyo.TerminationCondition.feasible:
            print(f"   ⚠️  실현가능해 발견 (최적은 아님)")
            try:
                obj_value = pyo.value(self.model.objective)
                print(f"   • 목적함수 값: {obj_value:.4f}")
            except ValueError as e:
                print(f"   ⚠️  목적함수 값 계산 실패: {e}")
                self._check_variable_initialization()
        elif self.results.solver.termination_condition == pyo.TerminationCondition.infeasible:
            print(f"   ❌ 최적화 실패: 제약조건이 서로 충돌합니다 (Infeasible)")
            print(f"\n   💡 가능한 원인:")
            print(f"      1. 제약조건들이 동시에 만족될 수 없음")
            print(f"      2. 변수 범위가 너무 제한적")
            print(f"      3. 비용 제약이 너무 엄격")
            print(f"      4. 사이트 변경으로 인한 데이터 불일치")
            self._check_constraint_conflicts()
        elif self.results.solver.termination_condition == pyo.TerminationCondition.unbounded:
            print(f"   ❌ 최적화 실패: 목적함수가 무한대로 발산합니다 (Unbounded)")
            print(f"   💡 목적함수나 제약조건을 확인하세요.")
        elif self.results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
            print(f"   ⏱️  최적화 시간 초과")
            print(f"   💡 시간 제한을 늘리거나 문제를 단순화하세요.")
        else:
            print(f"   ❌ 최적화 실패")
            print(f"   • 종료 조건: {self.results.solver.termination_condition}")
            if hasattr(self.results.solver, 'message'):
                print(f"   • 메시지: {self.results.solver.message}")

            # 추가 진단 정보
            print(f"\n   🔍 진단 정보:")
            self._check_variable_initialization()

        return self.results

    def _check_variable_initialization(self) -> None:
        """변수 초기화 상태 확인 (디버깅용)"""
        print(f"\n   🔍 변수 초기화 상태 확인:")

        # tier1_re, tier2_re 변수 확인
        for material in list(self.model.materials)[:5]:  # 처음 5개만 확인
            try:
                tier1_val = pyo.value(self.model.tier1_re[material])
                tier2_val = pyo.value(self.model.tier2_re[material])
                print(f"      ✅ {material[:50]}: tier1_re={tier1_val:.6f}, tier2_re={tier2_val:.6f}")
            except ValueError:
                print(f"      ❌ {material[:50]}: 초기화되지 않음")
                break

    def _check_constraint_conflicts(self) -> None:
        """제약조건 충돌 확인 (디버깅용)"""
        print(f"\n   🔍 제약조건 충돌 진단:")

        # 활성화된 제약조건 목록
        active_constraints = list(self.model.component_objects(pyo.Constraint, active=True))
        print(f"      • 활성 제약조건 수: {len(active_constraints)}개")

        # 변수 범위 확인
        print(f"\n      📊 변수 범위 샘플:")
        for var_obj in list(self.model.component_objects(pyo.Var, active=True))[:3]:
            var_name = var_obj.name
            # 첫 번째 변수의 범위만 확인
            first_var = list(var_obj.values())[0]
            if first_var.bounds:
                print(f"         • {var_name}: bounds = {first_var.bounds}")
            else:
                print(f"         • {var_name}: no bounds")

        # 제약조건 유형 확인
        print(f"\n      🔗 제약조건 유형:")
        constraint_types = {}
        for const in active_constraints:
            const_type = str(type(const).__name__)
            constraint_types[const_type] = constraint_types.get(const_type, 0) + 1

        for const_type, count in constraint_types.items():
            print(f"         • {const_type}: {count}개")

        # Material classification 정보
        if self.data and 'material_classification' in self.data:
            material_classification = self.data['material_classification']
            formula_count = sum(1 for m in material_classification.values() if m['is_formula_applicable'])
            nicoli_count = sum(1 for m in material_classification.values() if m['is_ni_co_li'])
            print(f"\n      📦 자재 분류:")
            print(f"         • Formula 자재: {formula_count}개")
            print(f"         • Ni/Co/Li 자재: {nicoli_count}개")
            print(f"         • 총 자재: {len(material_classification)}개")

    def extract_solution(self) -> Dict[str, Any]:
        """
        최적화 결과 추출

        Returns:
            결과 딕셔너리
        """
        if not self.results:
            raise Exception("결과가 없습니다. solve()를 먼저 호출하세요.")

        print("\n📤 결과 추출 중...")

        # 종료 조건 확인
        termination_condition = self.results.solver.termination_condition
        is_optimal = termination_condition == pyo.TerminationCondition.optimal
        is_feasible = termination_condition in [
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible
        ]

        print(f"   • 종료 조건: {termination_condition}")
        print(f"   • 최적해 여부: {'✅' if is_optimal else '❌'}")

        # Objective 값 안전하게 평가
        objective_value = None
        if is_feasible:
            try:
                objective_value = pyo.value(self.model.objective)
                print(f"   • 목적함수 값: {objective_value:.4f}")
            except ValueError as e:
                print(f"   ⚠️ 목적함수 값 평가 실패: {e}")
                print(f"   일부 변수가 초기화되지 않았습니다.")
                self._check_variable_initialization()
                # 최적해가 아니므로 None 유지
                objective_value = None
        else:
            print(f"   ⚠️ 최적해를 찾지 못해 목적함수 값을 평가할 수 없습니다.")

        solution = {
            'status': str(termination_condition),
            'objective_value': objective_value,
            'materials': {},
            'summary': {
                'total_carbon': objective_value if objective_value is not None else 0.0,
                'material_count': len(self.model.materials),
            },
            'debug_logs': []  # 디버깅 로그 추가
        }

        # 최적해를 찾지 못한 경우 여기서 반환
        if not is_feasible:
            print(f"\n   ❌ 실현가능한 해를 찾지 못했습니다.")
            print(f"   빈 결과를 반환합니다.")
            return solution

        # 양극재 관련 변수 존재 여부 확인
        has_cathode_vars = hasattr(self.model, 'cathode_emission_factor')

        # 양극재 전체 결과 추출 (element-level)
        if has_cathode_vars:
            try:
                cathode_emission = pyo.value(self.model.cathode_emission_factor)
                solution['cathode'] = {
                    'cathode_emission_factor': cathode_emission,
                    'elements': {}
                }
                cathode_composition = self.data['cathode_composition']
                extracted_elements = []
                for e in self.model.elements:
                    # 조성비가 0인 원소는 결과에서 제외
                    if e in cathode_composition and cathode_composition[e] == 0:
                        continue

                    try:
                        # 원소별 비율 추출
                        virgin_val = pyo.value(self.model.element_virgin_ratio[e])
                        recycle_val = pyo.value(self.model.element_recycle_ratio[e])
                        low_carb_val = pyo.value(self.model.element_low_carb_ratio[e])
                        emission_val = pyo.value(self.model.element_emission[e])

                        # 수치 정밀도 문제 처리 (1e-6 미만은 0으로)
                        solution['cathode']['elements'][e] = {
                            'virgin_ratio': 0.0 if abs(virgin_val) < 1e-6 else virgin_val,
                            'recycle_ratio': 0.0 if abs(recycle_val) < 1e-6 else recycle_val,
                            'low_carbon_ratio': 0.0 if abs(low_carb_val) < 1e-6 else low_carb_val,
                            'emission_factor': emission_val
                        }
                        extracted_elements.append(e)
                    except ValueError as e_err:
                        print(f"   ⚠️ 원소 {e} 변수 추출 실패: {e_err}")
                        continue

                if extracted_elements:
                    print(f"   ✅ 양극재 원소별 결과 추출: {extracted_elements}")
                else:
                    print(f"   ⚠️ 양극재 원소별 결과를 추출할 수 없습니다.")

            except ValueError as cath_err:
                print(f"   ⚠️ 양극재 변수 추출 실패: {cath_err}")
                has_cathode_vars = False  # 양극재 결과 사용 안 함

        # 양극재 자재 식별 키워드
        cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']

        # 각 자재별 결과 추출
        for m in self.model.materials:
            try:
                material_info = self.data['material_classification'][m]

                # 양극재 자재 여부 확인
                is_cathode_material = any(keyword in m for keyword in cathode_keywords)

                # 배출계수 결정 (양극재는 cathode_emission_factor 사용)
                try:
                    if is_cathode_material and has_cathode_vars:
                        modified_emission = pyo.value(self.model.cathode_emission_factor)
                    else:
                        modified_emission = pyo.value(self.model.modified_emission[m])
                except ValueError as em_err:
                    print(f"   ⚠️ {m}: 배출계수 추출 실패 - 원본값 사용")
                    modified_emission = material_info['original_emission']

                material_result = {
                    'modified_emission': modified_emission,
                    'original_emission': material_info['original_emission'],
                    'quantity': material_info['quantity'],
                    'type': material_info['type'],
                    'is_cathode': is_cathode_material  # 양극재 여부 표시
                }

                # 자재 타입에 따라 해당 변수만 추출
                # 양극재도 RE100 적용하므로 tier1_re/tier2_re 추출
                if material_info['is_formula_applicable']:
                    try:
                        tier1_value = pyo.value(self.model.tier1_re[m])
                        tier2_value = pyo.value(self.model.tier2_re[m])

                        # 디버깅 로그 수집
                        log_entry = {
                            'material': m,
                            'tier1_raw': tier1_value,
                            'tier2_raw': tier2_value,
                            'is_cathode': is_cathode_material,
                            'is_formula': material_info['is_formula_applicable']
                        }
                        solution['debug_logs'].append(log_entry)

                        # 디버깅: 실제 값 출력 (콘솔)
                        if abs(tier1_value) > 1e-6 or abs(tier2_value) > 1e-6:
                            print(f"   DEBUG: {m[:50]}... tier1={tier1_value:.6f}, tier2={tier2_value:.6f}, is_cathode={is_cathode_material}")

                        # 수치 정밀도 문제 처리 (1e-6 미만은 0으로)
                        # 양극재도 RE100을 사용하므로 실제 값 표시
                        material_result['tier1_re'] = 0.0 if abs(tier1_value) < 1e-6 else tier1_value
                        material_result['tier2_re'] = 0.0 if abs(tier2_value) < 1e-6 else tier2_value
                        log_entry['tier1_final'] = material_result['tier1_re']
                        log_entry['tier2_final'] = material_result['tier2_re']

                        if is_cathode_material:
                            log_entry['note'] = '양극재 - Element-level + RE100 동시 적용'
                        else:
                            log_entry['note'] = 'Formula 자재 - RE100 적용'
                    except ValueError:
                        # 변수가 초기화되지 않은 경우 기본값 사용
                        material_result['tier1_re'] = 0.0
                        material_result['tier2_re'] = 0.0
                        print(f"   ⚠️  {m}: tier1_re/tier2_re 초기화되지 않음 (기본값 0 사용)")

                if material_info['is_ni_co_li']:
                    # 양극재는 원소별 비율의 가중평균 계산
                    if is_cathode_material and has_cathode_vars:
                        try:
                            cathode_composition = self.data['cathode_composition']

                            total_virgin = 0.0
                            total_recycle = 0.0
                            total_low_carb = 0.0

                            # 조성비가 0보다 큰 원소들에 대해 가중평균 계산
                            for e in self.model.elements:
                                if e in cathode_composition and cathode_composition[e] > 0:
                                    comp_ratio = cathode_composition[e]
                                    total_virgin += comp_ratio * pyo.value(self.model.element_virgin_ratio[e])
                                    total_recycle += comp_ratio * pyo.value(self.model.element_recycle_ratio[e])
                                    total_low_carb += comp_ratio * pyo.value(self.model.element_low_carb_ratio[e])

                            # 수치 정밀도 문제 처리 (1e-6 미만은 0으로)
                            material_result['virgin_ratio'] = 0.0 if abs(total_virgin) < 1e-6 else total_virgin
                            material_result['recycle_ratio'] = 0.0 if abs(total_recycle) < 1e-6 else total_recycle
                            material_result['low_carbon_ratio'] = 0.0 if abs(total_low_carb) < 1e-6 else total_low_carb
                        except ValueError as ratio_err:
                            print(f"   ⚠️ {m}: 원소별 비율 추출 실패 - 기본값 0 사용")
                            material_result['virgin_ratio'] = 0.0
                            material_result['recycle_ratio'] = 0.0
                            material_result['low_carbon_ratio'] = 0.0
                    else:
                        # 일반 Ni/Co/Li 자재는 자재별 변수 사용
                        try:
                            recycle_val = pyo.value(self.model.recycle_ratio[m])
                            low_carb_val = pyo.value(self.model.low_carbon_ratio[m])
                            virgin_val = pyo.value(self.model.virgin_ratio[m])

                            # 수치 정밀도 문제 처리
                            material_result['recycle_ratio'] = 0.0 if abs(recycle_val) < 1e-6 else recycle_val
                            material_result['low_carbon_ratio'] = 0.0 if abs(low_carb_val) < 1e-6 else low_carb_val
                            material_result['virgin_ratio'] = 0.0 if abs(virgin_val) < 1e-6 else virgin_val
                        except ValueError as ratio_err:
                            print(f"   ⚠️ {m}: 비율 변수 추출 실패 - 기본값 0 사용")
                            material_result['virgin_ratio'] = 0.0
                            material_result['recycle_ratio'] = 0.0
                            material_result['low_carbon_ratio'] = 0.0

                # 감축률 계산
                original = material_result['original_emission']
                modified = material_result['modified_emission']
                if original > 0:
                    reduction_pct = (1 - modified / original) * 100
                    material_result['reduction_pct'] = reduction_pct
                else:
                    material_result['reduction_pct'] = 0

                solution['materials'][m] = material_result

            except Exception as mat_err:
                print(f"   ⚠️ 자재 {m} 처리 중 오류 발생: {mat_err}")
                # 해당 자재는 건너뜀
                continue

        print(f"   ✅ {len(solution['materials'])}개 자재 결과 추출 완료")

        return solution

    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환

        Returns:
            모델 정보 딕셔너리
        """
        if not self.model:
            return {'status': 'not_built'}

        return {
            'status': 'built',
            'materials_count': len(self.model.materials),
            'variables_count': len(list(self.model.component_objects(pyo.Var, active=True))),
            'constraints_count': len(list(self.model.component_objects(pyo.Constraint, active=True))),
            'objective_defined': hasattr(self.model, 'objective'),
            'solver': self.solver_name
        }

    def __repr__(self) -> str:
        """OptimizationEngine 문자열 표현"""
        return f"<OptimizationEngine(solver='{self.solver_name}', model_built={self.model is not None})>"

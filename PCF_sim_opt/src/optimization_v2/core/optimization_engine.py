"""
최적화 엔진

Pyomo를 사용하여 최적화 모델을 구축하고 실행하는 핵심 엔진입니다.
"""

from typing import Dict, Any, Optional, List
import pyomo.environ as pyo
from .constraint_manager import ConstraintManager
# Phase 2: Strategy Pattern 도입
from .cathode_strategy import CathodeOptimizationStrategy
from .material_strategy import DefaultOptimizationStrategy
# Phase 3: [DEPRECATED] 음극재는 일반 자재로 처리 (AnodeOptimizationStrategy 사용 안 함)
# from .anode_strategy import AnodeOptimizationStrategy
# Phase 1 (Cost System Redesign): 일반 자재 전략 추가
from .general_strategy import GeneralMaterialOptimizationStrategy
# Phase 2: Electrolyte 및 Separator 전략 추가
from .electrolyte_strategy import ElectrolyteOptimizationStrategy
from .separator_strategy import SeparatorOptimizationStrategy


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
        self.enable_dual_values: bool = False  # Dual value 계산 활성화 여부

        # Phase 2: Strategy Pattern 도입
        self.material_strategies: Dict[str, Any] = {}  # 자재별 최적화 전략
        self.cathode_strategy: Optional[Any] = None  # 양극재 전략 (전역)
        # Phase 3: [DEPRECATED] 음극재는 일반 자재로 처리
        # self.anode_strategy: Optional[Any] = None  # 음극재 전략 (전역)

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

        # Phase 2: 자재별 최적화 전략 초기화
        self._initialize_material_strategies(material_classification)

        # 결정 변수 정의
        self._define_variables(material_classification)

        # 기본 제약조건 추가 (비율 합 = 1)
        self._add_basic_constraints(material_classification)

        # 양극재 관련 제약조건 추가
        self._add_cathode_constraints(material_classification)

        # Phase 3: 음극재 관련 제약조건 추가
        self._add_anode_constraints(material_classification)

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

        # Dual value 추출을 위한 suffix 추가 (IPOPT용)
        if self.enable_dual_values:
            self.model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
            print(f"   ✅ Dual value suffix 추가 (constraint relaxation 분석용)")

        print(f"\n✅ 모델 구축 완료!")
        print(f"   • 변수 수: {len(list(self.model.component_objects(pyo.Var, active=True)))}개")
        print(f"   • 제약조건 수: {len(list(self.model.component_objects(pyo.Constraint, active=True)))}개")
        print(f"   • 목적함수: {objective_type}")
        print(f"   • 문제 유형: {'비선형 (NLP)' if self.is_nonlinear else '선형 (LP)'}")

        # 제약조건 검사
        self._inspect_constraints()

        return self.model

    def _initialize_material_strategies(
        self,
        material_classification: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        자재별 최적화 전략 초기화 (Phase 2)

        Args:
            material_classification: 자재 분류 정보
        """
        # 양극재 감지
        cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
        cathode_materials = [
            m for m in self.model.materials
            if any(kw in m for kw in cathode_keywords)
        ]

        # 양극재가 있으면 CathodeOptimizationStrategy 생성
        if cathode_materials and self.data.get('cathode_composition'):
            cathode_data = {
                'cathode_composition': self.data['cathode_composition'],
                'recycle_impact': self.data.get('recycle_impact', {}),
                'low_carb_emission': self.data.get('low_carb_emission', {}),
                'virgin_emission': self.data.get('virgin_emission', {})
            }

            # 첫 번째 양극재를 대표로 사용 (전역 전략)
            representative_cathode = cathode_materials[0]
            material_data = material_classification[representative_cathode]

            self.cathode_strategy = CathodeOptimizationStrategy(
                representative_cathode,
                material_data,
                cathode_data
            )

            # 모든 양극재에 동일한 전략 할당
            for cathode_mat in cathode_materials:
                self.material_strategies[cathode_mat] = self.cathode_strategy

            print(f"   ✅ Phase 2: CathodeStrategy 초기화 (양극재 {len(cathode_materials)}개)")

        # Phase 3: 음극재는 독립적인 자재로 처리
        # [DEPRECATED] AnodeOptimizationStrategy는 사용하지 않음
        # Natural Graphite와 Artificial Graphite는 서로 다른 자재이므로
        # 각각 독립적으로 RE100 등을 최적화해야 함
        # GeneralMaterialOptimizationStrategy로 처리됨
        anode_keywords = ['Anode', '음극재', 'Graphite']
        anode_materials = [
            m for m in self.model.materials
            if any(kw in m for kw in anode_keywords)
        ]

        if anode_materials:
            print(f"   ℹ️  음극재 {len(anode_materials)}개 감지 → 각각 독립적인 자재로 최적화")
            for anode_mat in anode_materials:
                material_data = material_classification[anode_mat]
                print(f"      • {anode_mat}: {material_data['quantity']:.2f}kg (고정), "
                      f"{material_data['original_emission']:.3f} kgCO2eq/kg")

        # Phase 2: Electrolyte 전용 전략 초기화
        electrolyte_keywords = ['Electrolyte', 'electrolyte', '전해액', '전해질']
        electrolyte_materials = [
            m for m in self.model.materials
            if (
                any(kw in m for kw in electrolyte_keywords) and
                m not in self.material_strategies
            )
        ]

        if electrolyte_materials and self.data.get('electrolyte_composition'):
            electrolyte_config = self.data['electrolyte_composition']

            for electrolyte_mat in electrolyte_materials:
                material_data = material_classification[electrolyte_mat]

                # Electrolyte 전략 생성
                strategy = ElectrolyteOptimizationStrategy(
                    electrolyte_mat,
                    material_data,
                    electrolyte_config
                )

                # 전략 할당
                self.material_strategies[electrolyte_mat] = strategy

            print(f"   ✅ Phase 2: ElectrolyteStrategy 초기화 ({len(electrolyte_materials)}개)")
            print(f"      • 용매 종류: {len(electrolyte_config.get('solvents', []))}개")

        # Phase 2: Separator 전용 전략 초기화
        separator_keywords = ['Separator', 'separator', '분리막', '세퍼레이터']
        separator_materials = [
            m for m in self.model.materials
            if (
                any(kw in m for kw in separator_keywords) and
                m not in self.material_strategies
            )
        ]

        if separator_materials and self.data.get('separator_composition'):
            separator_config = self.data['separator_composition']

            for separator_mat in separator_materials:
                material_data = material_classification[separator_mat]

                # Separator 전략 생성
                strategy = SeparatorOptimizationStrategy(
                    separator_mat,
                    material_data,
                    separator_config
                )

                # 전략 할당
                self.material_strategies[separator_mat] = strategy

            print(f"   ✅ Phase 2: SeparatorStrategy 초기화 ({len(separator_materials)}개)")
            print(f"      • Dry 배출계수: {separator_config.get('dry_emission', 2.5):.2f} kgCO2eq/kg")
            print(f"      • Wet 배출계수: {separator_config.get('wet_emission', 3.2):.2f} kgCO2eq/kg")

        # Phase 1 (Cost System Redesign): 일반 자재 전략 초기화
        # Current Collector, Binder 등 기타 일반 자재에 RE100 적용
        general_material_keywords = [
            'Current Collector', 'Collector', '집전체',
            'Binder', '바인더',
            'Additive', '첨가제',
            'Electrolyte', 'electrolyte', '전해액', '전해질',  # Fallback
            'Separator', 'separator', '분리막', '세퍼레이터'   # Fallback
        ]

        general_materials = [
            m for m in self.model.materials
            if (
                any(kw in m for kw in general_material_keywords) and
                m not in self.material_strategies  # 이미 할당된 자재는 제외
            )
        ]

        # 일반 자재에 GeneralMaterialOptimizationStrategy 할당
        if general_materials:
            general_with_re100 = 0
            general_without_re100 = 0

            for mat in general_materials:
                material_data = material_classification[mat]

                # GeneralStrategy 생성
                strategy = GeneralMaterialOptimizationStrategy(mat, material_data)

                # 전략 할당
                self.material_strategies[mat] = strategy

                # RE100 적용 가능 여부 카운트
                if strategy.has_re100:
                    general_with_re100 += 1
                else:
                    general_without_re100 += 1

            print(f"   ✅ Phase 1: GeneralStrategy 초기화 (일반 자재 {len(general_materials)}개)")
            print(f"      • RE100 적용 가능: {general_with_re100}개")
            print(f"      • RE100 데이터 없음: {general_without_re100}개")

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

        # Phase 2: 양극재 전용 변수는 Strategy를 통해 정의
        if self.cathode_strategy:
            self.cathode_strategy.define_variables(self.model, 0)  # material_idx는 전역이므로 0 사용

        # Phase 3: [REMOVED] 음극재는 일반 자재로 처리 (AnodeStrategy 사용 안 함)

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

        # Fix tier1_re/tier2_re to 0 for materials that don't use them
        # This prevents ValueError when accessing uninitialized variables
        fixed_count = 0
        for m in self.model.materials:
            material_info = material_classification[m]

            # Only apply to Formula materials
            if not material_info['is_formula_applicable']:
                continue

            # Check if material actually uses RE100 (non-zero energy ratios)
            tier1_ratio = material_info.get('tier1_energy_ratio', 0)
            tier2_ratio = material_info.get('tier2_energy_ratio', 0)

            # If both ratios are 0, fix variables to 0
            if tier1_ratio == 0 and tier2_ratio == 0:
                # Fix both variables to 0 for this material
                self.model.tier1_re[m].fix(0)
                self.model.tier2_re[m].fix(0)
                fixed_count += 1
                print(f"      • {m[:60]}: tier1_re/tier2_re fixed to 0")

        if fixed_count > 0:
            print(f"   ✅ Tier RE 변수 고정 완료 ({fixed_count}개 자재)")

        print("   ✅ 기본 제약조건 추가 완료")

    def _add_cathode_constraints(self, material_classification: Dict[str, Dict[str, Any]]) -> None:
        """
        양극재 관련 제약조건 추가 (Phase 2: Strategy Pattern 적용)

        - CathodeOptimizationStrategy에 위임
        - 원소별 비율 합 = 1
        - 원소별 배출계수 계산
        - 양극재 전체 배출계수 계산 (RE100 포함)

        Args:
            material_classification: 자재 분류 정보
        """
        if not self.cathode_strategy:
            return  # 양극재 전략이 없으면 스킵

        print("\n🔋 양극재 제약조건 추가 중...")

        # Phase 2: CathodeStrategy에 element-level 제약조건 위임
        self.cathode_strategy.add_constraints(self.model, 0)

        # 양극재 자재 식별
        cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
        cathode_materials = [m for m in self.model.materials
                            if any(kw in m for kw in cathode_keywords)]

        # Phase 2: CathodeStrategy에 RE100 배출계수 제약조건 위임
        if cathode_materials:
            print(f"   ℹ️  양극재 {len(cathode_materials)}개: Element-level 최적화 + RE100 동시 적용")
            self.cathode_strategy.add_cathode_emission_constraint(
                self.model,
                cathode_materials,
                material_classification
            )

        print("   ✅ 양극재 제약조건 추가 완료")

    def _add_anode_constraints(self, material_classification: Dict[str, Dict[str, Any]]) -> None:
        """
        [DEPRECATED] 음극재 관련 제약조건 추가

        음극재는 더 이상 특별한 처리 없이 일반 자재로 최적화됩니다.
        각 음극재(Natural/Artificial)는 독립적인 자재로 고정된 수량을 가집니다.

        Args:
            material_classification: 자재 분류 정보
        """
        # AnodeOptimizationStrategy는 더 이상 사용하지 않음
        return

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
        # Phase 3: 음극재 키워드 (composition optimization 사용)
        anode_keywords = ['Anode', '음극재', 'Graphite']

        # 각 자재의 배출계수 계산 제약조건
        def emission_calculation_rule(model, m):
            material_info = material_classification[m]
            original_emission = material_info['original_emission']

            # 양극재 여부 확인
            is_cathode = any(keyword in m for keyword in cathode_keywords)
            # Phase 3: 음극재 여부 확인
            is_anode = any(keyword in m for keyword in anode_keywords)

            if material_info['is_formula_applicable'] and not is_cathode and not is_anode:
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

            elif is_anode:
                # Phase 3: 음극재는 일반 자재로 처리 (composition 최적화 안 함)
                # RE100만 적용하여 배출계수 감축
                tier1_ratio = material_info.get('tier1_energy_ratio', 0)
                tier2_ratio = material_info.get('tier2_energy_ratio', 0)

                # RE100 적용된 배출계수 계산
                reduction_factor = (1 - model.tier1_re[m] * tier1_ratio - model.tier2_re[m] * tier2_ratio)
                return model.modified_emission[m] == original_emission * reduction_factor

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
                        # 음극재 포함 일반 자재는 모두 modified_emission 사용
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

    def _inspect_constraints(self) -> Dict[str, Any]:
        """
        모델의 제약조건 검사 (디버깅 및 검증용)

        Returns:
            검사 결과 딕셔너리
        """
        print(f"\n{'='*70}")
        print("🔍 모델 제약조건 검사")
        print(f"{'='*70}")

        inspection = {
            'total_constraints': 0,
            'element_ratio_constraints': [],
            'material_ratio_constraints': [],
            'has_element_vars': False,
            'elements': [],
            'has_material_vars': False
        }

        # 전체 제약조건 수
        all_constraints = list(self.model.component_objects(pyo.Constraint, active=True))
        inspection['total_constraints'] = len(all_constraints)
        print(f"총 제약조건: {len(all_constraints)}개")

        # 원소 Set 확인
        if hasattr(self.model, 'elements'):
            inspection['has_element_vars'] = True
            inspection['elements'] = list(self.model.elements)
            print(f"\n✅ 양극재 원소 Set 존재: {inspection['elements']}")
        else:
            print(f"\n⚠️  양극재 원소 Set 없음")

        # 원소별 변수 확인
        if hasattr(self.model, 'element_recycle_ratio'):
            print(f"✅ element_recycle_ratio 변수 존재")
        else:
            print(f"⚠️  element_recycle_ratio 변수 없음")

        if hasattr(self.model, 'element_low_carb_ratio'):
            print(f"✅ element_low_carb_ratio 변수 존재")
        else:
            print(f"⚠️  element_low_carb_ratio 변수 없음")

        # 자재별 변수 확인
        if hasattr(self.model, 'recycle_ratio'):
            inspection['has_material_vars'] = True
            print(f"✅ 자재별 recycle_ratio 변수 존재")
        else:
            print(f"⚠️  자재별 recycle_ratio 변수 없음")

        # 원소별 제약조건 찾기
        for constraint in all_constraints:
            constraint_name = str(constraint)

            # 원소별 비율 제약조건 패턴 매칭
            if any(pattern in constraint_name for pattern in [
                'cathode_element_recycle',
                'cathode_element_low_carbon',
                'element_recycle',
                'element_low_carb'
            ]):
                inspection['element_ratio_constraints'].append(constraint_name)

            # 자재별 비율 제약조건 패턴 매칭
            elif any(pattern in constraint_name for pattern in [
                'recycle_min_',
                'recycle_max_',
                'low_carbon_min_',
                'low_carbon_max_',
                'exclude_recycle',
                'exclude_low_carbon',
                'virgin_only',
                'recycle_only',
                'low_carbon_only'
            ]):
                inspection['material_ratio_constraints'].append(constraint_name)

        # 원소별 제약조건 출력
        if inspection['element_ratio_constraints']:
            print(f"\n✅ 원소별 비율 제약조건: {len(inspection['element_ratio_constraints'])}개")
            for c in inspection['element_ratio_constraints'][:10]:  # 처음 10개만
                print(f"   • {c}")
            if len(inspection['element_ratio_constraints']) > 10:
                print(f"   ... 외 {len(inspection['element_ratio_constraints'])-10}개")
        else:
            print(f"\n⚠️  원소별 비율 제약조건 없음")
            print(f"   MaterialManagementConstraint의 force_element_ratio_range 규칙이")
            print(f"   제대로 적용되지 않았을 수 있습니다.")

        # 자재별 제약조건 출력
        if inspection['material_ratio_constraints']:
            print(f"\n✅ 자재별 비율 제약조건: {len(inspection['material_ratio_constraints'])}개")
            for c in inspection['material_ratio_constraints'][:10]:  # 처음 10개만
                print(f"   • {c}")
            if len(inspection['material_ratio_constraints']) > 10:
                print(f"   ... 외 {len(inspection['material_ratio_constraints'])-10}개")
        else:
            print(f"\n⚠️  자재별 비율 제약조건 없음")

        print(f"{'='*70}\n")

        return inspection

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

                # RE100-비용 관계 검증 (verbose=True이고 비용 제약이 있을 때만)
                self._run_re100_cost_validation(verbose)

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

                # 양극재 원본 배출계수 찾기 (scenario_df에서)
                cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
                original_cathode_emission = None
                for m in self.model.materials:
                    if any(keyword in m for keyword in cathode_keywords):
                        # scenario_df에서 원본 배출계수 가져오기
                        scenario_df = self.data.get('scenario_df')
                        if scenario_df is not None:
                            material_row = scenario_df[scenario_df['자재명'] == m]
                            if not material_row.empty:
                                original_cathode_emission = material_row['배출계수'].values[0]
                                break

                solution['cathode'] = {
                    'cathode_emission_factor': cathode_emission,
                    'original_cathode_emission': original_cathode_emission,
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

        # Phase 3: [REMOVED] 음극재는 일반 자재로 처리 (composition 최적화 안 함)

        # 양극재 자재 식별 키워드
        cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']

        # 각 자재별 결과 추출
        for m in self.model.materials:
            try:
                material_info = self.data['material_classification'][m]

                # 양극재 자재 여부 확인
                is_cathode_material = any(keyword in m for keyword in cathode_keywords)

                # 배출계수 결정 (양극재는 전용 배출계수 사용, 음극재 포함 나머지는 modified_emission)
                try:
                    if is_cathode_material and has_cathode_vars:
                        modified_emission = pyo.value(self.model.cathode_emission_factor)
                    else:
                        # 음극재 포함 모든 일반 자재
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

                        # 🔍 DIAGNOSTIC POINT A: 솔버 추출 직후 (threshold 적용 전)
                        if is_cathode_material:
                            print(f"\n🔍 [POINT A] Cathode {m[:40]}")
                            print(f"  Raw tier1_re from solver: {tier1_value:.10f} (before threshold)")
                            print(f"  Raw tier2_re from solver: {tier2_value:.10f} (before threshold)")
                            print(f"  Threshold: 1e-10 (adjusted from 1e-6 for better precision)")

                        # 디버깅: 실제 값 출력 (콘솔) - threshold 조정
                        if abs(tier1_value) > 1e-10 or abs(tier2_value) > 1e-10:
                            print(f"   DEBUG: {m[:50]}... tier1={tier1_value:.10f}, tier2={tier2_value:.10f}, is_cathode={is_cathode_material}")

                        # 수치 정밀도 문제 처리 (1e-10 미만은 0으로) - threshold 완화
                        # 양극재도 RE100을 사용하므로 더 작은 값도 보존
                        THRESHOLD = 1e-10  # 1e-6에서 1e-10으로 완화
                        material_result['tier1_re'] = 0.0 if abs(tier1_value) < THRESHOLD else tier1_value
                        material_result['tier2_re'] = 0.0 if abs(tier2_value) < THRESHOLD else tier2_value
                        log_entry['tier1_final'] = material_result['tier1_re']
                        log_entry['tier2_final'] = material_result['tier2_re']

                        # 🔍 DIAGNOSTIC POINT B: Threshold 적용 후
                        if is_cathode_material:
                            print(f"\n🔍 [POINT B] After threshold")
                            print(f"  tier1_re: {material_result['tier1_re']:.10f}")
                            print(f"  tier2_re: {material_result['tier2_re']:.10f}")

                        # 🔍 DIAGNOSTIC POINT C: 에너지 기여도 확인
                        if is_cathode_material:
                            tier1_energy = material_info.get('tier1_energy_ratio', 0)
                            tier2_energy = material_info.get('tier2_energy_ratio', 0)
                            print(f"\n🔍 [POINT C] Energy contribution")
                            print(f"  tier1_energy_ratio: {tier1_energy:.6f}")
                            print(f"  tier2_energy_ratio: {tier2_energy:.6f}")
                            if tier1_energy == 0 and tier2_energy == 0:
                                print("  ⚠️ WARNING: Both energy ratios are 0! RE100 will have no effect")
                            else:
                                re100_effect = tier1_value * tier1_energy + tier2_value * tier2_energy
                                print(f"  Total RE100 effect: {re100_effect:.6f} ({re100_effect*100:.2f}% reduction)")

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
                    # 양극재는 원소별 비율의 가중평균 계산 (Phase 2: Strategy Pattern 적용)
                    if is_cathode_material and has_cathode_vars and self.cathode_strategy:
                        try:
                            # Phase 2: CathodeStrategy에 위임
                            ratio_result = self.cathode_strategy.extract_solution(self.model, 0)
                            material_result.update(ratio_result)

                            # DIAGNOSTIC POINT A: After ratio extraction (cathode element-level)
                            print(f"   ✓ [POINT A - RATIOS] {m[:40]}")
                            print(f"      virgin_ratio={material_result.get('virgin_ratio', 'MISSING')}")
                            print(f"      recycle_ratio={material_result.get('recycle_ratio', 'MISSING')}")
                            print(f"      low_carbon_ratio={material_result.get('low_carbon_ratio', 'MISSING')}")
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

                # Phase 3: [REMOVED] 음극재는 일반 자재로 처리 (composition 추출 안 함)

                # 감축률 계산
                original = material_result['original_emission']
                modified = material_result['modified_emission']
                if original > 0:
                    reduction_pct = (1 - modified / original) * 100
                    material_result['reduction_pct'] = reduction_pct
                else:
                    material_result['reduction_pct'] = 0

                # 🔍 감축률 검증 로그 (양극재/음극재)
                if is_cathode_material or any(kw in m for kw in ['Anode', '음극재', 'Graphite']):
                    material_type_label = "양극재" if is_cathode_material else "음극재"
                    print(f"\n   🔍 [{material_type_label}] {m[:50]}")
                    print(f"      • Original emission: {original:.6f} kgCO2eq/kg")
                    print(f"      • Modified emission: {modified:.6f} kgCO2eq/kg")
                    print(f"      • Reduction: {reduction_pct:.2f}%")

                    # RE100 기여도 계산
                    if material_info['is_formula_applicable']:
                        tier1_re_val = material_result.get('tier1_re', 0)
                        tier2_re_val = material_result.get('tier2_re', 0)
                        tier1_energy = material_info.get('tier1_energy_ratio', 0)
                        tier2_energy = material_info.get('tier2_energy_ratio', 0)

                        re100_effect = tier1_re_val * tier1_energy + tier2_re_val * tier2_energy
                        print(f"      • RE100 values: Tier1={tier1_re_val*100:.2f}%, Tier2={tier2_re_val*100:.2f}%")
                        print(f"      • RE100 effect: {re100_effect*100:.2f}% reduction")

                        # Element-level 효과 추정 (양극재만)
                        if is_cathode_material and has_cathode_vars:
                            element_effect = reduction_pct - (re100_effect * 100)
                            print(f"      • Element-level effect: ~{element_effect:.2f}% reduction")

                # DIAGNOSTIC POINT B: Before storing in solution dict
                if is_cathode_material:
                    print(f"   ✓ [POINT B - STORING CATHODE] {m[:40]}")
                    print(f"      Keys in material_result: {list(material_result.keys())}")
                    for key in ['tier1_re', 'tier2_re', 'recycle_ratio', 'low_carbon_ratio', 'virgin_ratio', 'reduction_pct']:
                        val = material_result.get(key, 'KEY_MISSING')
                        if val != 'KEY_MISSING':
                            print(f"      {key}: {val}")
                        else:
                            print(f"      {key}: {val}")

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

    def get_dual_values(self) -> Dict[str, float]:
        """
        제약조건의 dual value (shadow price) 추출

        Dual value는 제약조건을 완화했을 때 목적함수가 얼마나 개선되는지를 나타냅니다.

        Returns:
            제약조건 이름과 dual value 딕셔너리

        Notes:
            - IPOPT 솔버를 사용했을 때만 사용 가능
            - enable_dual_values=True로 설정하고 모델을 구축해야 함
            - 최적화를 실행한 후 호출해야 함
        """
        if not self.enable_dual_values:
            raise ValueError(
                "Dual value 추출이 활성화되지 않았습니다. "
                "OptimizationEngine.enable_dual_values=True로 설정하고 모델을 다시 구축하세요."
            )

        if not self.model or not hasattr(self.model, 'dual'):
            raise ValueError(
                "모델이 구축되지 않았거나 dual suffix가 없습니다."
            )

        if not self.results:
            raise ValueError(
                "최적화를 실행하지 않았습니다. solve()를 먼저 호출하세요."
            )

        if self.solver_name != 'ipopt':
            raise ValueError(
                f"Dual value는 IPOPT 솔버만 지원합니다. 현재 솔버: {self.solver_name}"
            )

        # 최적해가 아니면 경고
        if self.results.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(
                f"⚠️  경고: 최적해가 아닌 상태에서 dual value를 추출합니다. "
                f"종료 조건: {self.results.solver.termination_condition}"
            )

        # Dual value 추출
        dual_values = {}

        for constraint in self.model.component_objects(pyo.Constraint, active=True):
            constraint_name = constraint.name

            # Indexed constraint의 경우
            if constraint.is_indexed():
                for index in constraint:
                    dual_value = self.model.dual.get(constraint[index], None)
                    if dual_value is not None:
                        key = f"{constraint_name}[{index}]"
                        dual_values[key] = float(dual_value)
            else:
                # Simple constraint
                dual_value = self.model.dual.get(constraint, None)
                if dual_value is not None:
                    dual_values[constraint_name] = float(dual_value)

        return dual_values

    def get_constraint_slack(self) -> Dict[str, float]:
        """
        제약조건의 slack value 추출

        Slack value는 제약조건이 얼마나 여유가 있는지를 나타냅니다.
        slack = 0이면 binding constraint (tight)

        Returns:
            제약조건 이름과 slack value 딕셔너리
        """
        if not self.model:
            raise ValueError("모델이 구축되지 않았습니다.")

        if not self.results:
            raise ValueError("최적화를 실행하지 않았습니다.")

        slack_values = {}

        for constraint in self.model.component_objects(pyo.Constraint, active=True):
            constraint_name = constraint.name

            # Indexed constraint의 경우
            if constraint.is_indexed():
                for index in constraint:
                    try:
                        slack = pyo.value(constraint[index].body) - pyo.value(constraint[index].upper or constraint[index].lower or 0)
                        key = f"{constraint_name}[{index}]"
                        slack_values[key] = float(abs(slack))
                    except:
                        continue
            else:
                # Simple constraint
                try:
                    slack = pyo.value(constraint.body) - pyo.value(constraint.upper or constraint.lower or 0)
                    slack_values[constraint_name] = float(abs(slack))
                except:
                    continue

        return slack_values

    def _run_re100_cost_validation(self, verbose: bool) -> None:
        """
        RE100-비용 관계 검증 실행

        최적화 결과에서 RE100 비율이 높을 때, 실제로 비용 제약 내에서
        달성된 것인지 검증합니다.

        Args:
            verbose: 상세 출력 여부
        """
        if not verbose:
            return  # verbose가 False면 검증 건너뛰기

        # 필수 조건 확인
        if not self.data:
            return  # 데이터가 없으면 검증 불가

        # Cost constraint 확인
        cost_constraints = [
            c for c in self.constraint_manager.list_constraints(enabled_only=True)
            if 'CostConstraint' in str(type(c))
        ]

        if not cost_constraints:
            return  # 비용 제약이 없으면 검증 불필요

        # Cost calculator 확인
        cost_constraint = cost_constraints[0]
        if not hasattr(cost_constraint, 'cost_calculator'):
            return

        cost_calculator = cost_constraint.cost_calculator
        zero_premium_baseline = getattr(cost_constraint, 'zero_premium_baseline', None)

        if zero_premium_baseline is None:
            return  # Baseline이 없으면 검증 불가

        # 비용 한도 계산
        if hasattr(cost_constraint, 'absolute_premium_budget'):
            # Absolute budget 사용
            premium_budget = cost_constraint.absolute_premium_budget
            cost_limit = zero_premium_baseline + premium_budget
        elif hasattr(cost_constraint, 'premium_limit_pct'):
            # Percentage limit 사용
            premium_pct = cost_constraint.premium_limit_pct
            cost_limit = zero_premium_baseline * (1 + premium_pct / 100)
        else:
            return  # 한도 정보가 없으면 검증 불가

        # RE100CostValidator 실행
        try:
            from ..utils.re100_cost_validator import RE100CostValidator

            validator = RE100CostValidator()
            validation_result = validator.validate_solution(
                model=self.model,
                optimization_data=self.data,
                cost_calculator=cost_calculator,
                cost_baseline=zero_premium_baseline,
                cost_limit=cost_limit
            )

            # 검증 결과는 validator 내부에서 이미 출력됨
            # 추가 처리가 필요하면 여기서 수행

        except Exception as e:
            print(f"\n⚠️  RE100-비용 검증 실행 실패: {e}")
            # 검증 실패는 치명적이지 않으므로 계속 진행

    def __repr__(self) -> str:
        """OptimizationEngine 문자열 표현"""
        return f"<OptimizationEngine(solver='{self.solver_name}', model_built={self.model is not None})>"

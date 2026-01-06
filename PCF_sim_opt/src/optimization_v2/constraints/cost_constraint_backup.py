"""
비용 제약조건

RE100 프리미엄 비용을 기준으로 한 비용 제약조건입니다.
RE100PremiumCalculator와 통합하여 동작합니다.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class CostConstraint(ConstraintBase):
    """
    비용 제약조건 클래스

    RE100PremiumCalculator를 사용하여:
    - Premium 상한 설정 (기준 대비 +X%)
    - 절대 예산 한도 설정
    - 자재별 비용 제한
    """

    def __init__(self, cost_calculator=None):
        """
        비용 제약조건 초기화

        Args:
            cost_calculator: RE100PremiumCalculator 인스턴스
        """
        super().__init__(
            name="cost_constraint",
            description="RE100 프리미엄 비용 제약"
        )
        self.cost_calculator = cost_calculator
        self.baseline_cost: Optional[float] = None
        self.premium_limit_pct: Optional[float] = None  # 예: 10 (10%)
        self.absolute_budget: Optional[float] = None  # 예: 50000 (USD)
        self.material_cost_limits: Dict[str, float] = {}  # {material: max_cost}

        # 복합 최적화 모드 설정
        self.optimization_mode: str = 'constraint'  # 'constraint', 'multi_objective', or 'epsilon_constraint'
        self.carbon_weight: float = 0.7  # α: 탄소 배출 가중치
        self.cost_weight: float = 0.3  # β: 비용 가중치

        # Epsilon-Constraint 모드 설정
        self.epsilon_limit: Optional[float] = None  # 비용 상한 (epsilon)

    def set_cost_calculator(self, calculator) -> None:
        """
        RE100PremiumCalculator 설정

        Args:
            calculator: RE100PremiumCalculator 인스턴스
        """
        self.cost_calculator = calculator
        print("✅ Cost calculator 설정됨")

    def calculate_baseline_cost(
        self,
        scenario_df: pd.DataFrame,
        original_df: pd.DataFrame,
        case_name: str = 'case1'
    ) -> float:
        """
        기준 비용 계산 (RE100 프리미엄 포함)

        Args:
            scenario_df: 시나리오 DataFrame
            original_df: 원본 테이블 DataFrame
            case_name: 케이스 이름 (기본값: case1)

        Returns:
            기준 총 비용 (USD)
        """
        if self.cost_calculator is None:
            raise ValueError("Cost calculator가 설정되지 않았습니다.")

        # 입력 데이터 검증
        if scenario_df is None:
            raise ValueError("scenario_df가 None입니다. 데이터를 먼저 로딩하세요.")
        if original_df is None:
            raise ValueError("original_df가 None입니다. 데이터를 먼저 로딩하세요.")

        print(f"\n🔍 기준 비용 계산 디버깅:")
        print(f"   • scenario_df shape: {scenario_df.shape}")
        print(f"   • original_df shape: {original_df.shape}")
        print(f"   • case_name: {case_name}")

        try:
            # RE100 프리미엄 계산
            results_df = self.cost_calculator.calculate_scenario_premiums(
                scenario_df, original_df
            )

            # 결과 검증
            if results_df is None:
                raise ValueError("calculate_scenario_premiums가 None을 반환했습니다.")

            print(f"   • results_df shape: {results_df.shape}")
            print(f"   • results_df columns: {list(results_df.columns)}")

            # 해당 케이스의 총 프리미엄
            premium_col = f'{case_name}_premium($)'

            if premium_col not in results_df.columns:
                available = [c for c in results_df.columns if 'premium($)' in c]
                raise ValueError(f"컬럼 '{premium_col}'을 찾을 수 없습니다. 사용 가능: {available}")

            self.baseline_cost = results_df[premium_col].sum()
            print(f"✅ 기준 비용 계산 완료: ${self.baseline_cost:,.2f}")

            return self.baseline_cost

        except Exception as e:
            print(f"\n❌ 기준 비용 계산 중 예외 발생:")
            print(f"   예외 타입: {type(e).__name__}")
            print(f"   예외 메시지: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"기준 비용 계산 실패: {str(e)}")

    def set_premium_limit(self, pct: float) -> None:
        """
        프리미엄 한도 설정 (% 단위)

        Args:
            pct: 프리미엄 한도 (예: 10 = +10%)
        """
        if pct < 0:
            raise ValueError("프리미엄 한도는 0 이상이어야 합니다.")

        self.premium_limit_pct = pct
        print(f"✅ 프리미엄 한도 설정: +{pct}%")

        if self.baseline_cost:
            max_cost = self.baseline_cost * (1 + pct / 100)
            print(f"   기준: ${self.baseline_cost:,.2f} → 상한: ${max_cost:,.2f}")

    def set_absolute_budget(self, budget: float) -> None:
        """
        절대 예산 한도 설정

        Args:
            budget: 예산 한도 (USD)
        """
        if budget <= 0:
            raise ValueError("예산은 0보다 커야 합니다.")

        self.absolute_budget = budget
        print(f"✅ 절대 예산 한도 설정: ${budget:,.2f}")

    def set_material_cost_limit(self, material_name: str, max_cost: float) -> None:
        """
        자재별 비용 제한 설정

        Args:
            material_name: 자재명
            max_cost: 최대 비용 (USD)
        """
        self.material_cost_limits[material_name] = max_cost
        print(f"✅ {material_name} 비용 제한: ${max_cost:,.2f}")

    def set_multi_objective_mode(
        self,
        carbon_weight: float = 0.7,
        cost_weight: float = 0.3
    ) -> None:
        """
        복합 최적화 모드 설정 (비용-탄소 동시 최적화)

        목적함수: minimize (α × carbon + β × normalized_cost)

        Args:
            carbon_weight: 탄소 배출 가중치 α (0.0 ~ 1.0)
            cost_weight: 비용 가중치 β (0.0 ~ 1.0)
        """
        if carbon_weight < 0 or carbon_weight > 1:
            raise ValueError("carbon_weight는 0.0~1.0 사이여야 합니다.")
        if cost_weight < 0 or cost_weight > 1:
            raise ValueError("cost_weight는 0.0~1.0 사이여야 합니다.")

        self.optimization_mode = 'multi_objective'
        self.carbon_weight = carbon_weight
        self.cost_weight = cost_weight

        print(f"✅ 복합 최적화 모드 활성화")
        print(f"   탄소 가중치 (α): {carbon_weight:.2f}")
        print(f"   비용 가중치 (β): {cost_weight:.2f}")
        print(f"   목적함수: minimize ({carbon_weight}×탄소 + {cost_weight}×정규화비용)")

    def set_constraint_mode(self) -> None:
        """
        제약 모드로 전환 (비용을 제약조건으로만 사용)
        """
        self.optimization_mode = 'constraint'
        print(f"✅ 제약 모드 활성화 (비용은 제약조건으로만 작동)")

    def set_epsilon_constraint_mode(self, epsilon_limit: float) -> None:
        """
        Epsilon-Constraint 모드 설정

        비용을 목적함수가 아닌 제약조건으로만 사용

        Args:
            epsilon_limit: 비용 상한 ($)
        """
        if epsilon_limit <= 0:
            raise ValueError("epsilon_limit은 0보다 커야 합니다.")

        self.optimization_mode = 'epsilon_constraint'
        self.epsilon_limit = epsilon_limit

        print(f"✅ Epsilon-Constraint 모드 활성화")
        print(f"   비용 상한 (ε): ${epsilon_limit:,.2f}")

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            (is_valid, message)
        """
        # Cost calculator 필수
        if self.cost_calculator is None:
            return False, "Cost calculator가 설정되지 않았습니다."

        # 최소한 하나의 제약 필요
        has_constraint = (
            self.premium_limit_pct is not None or
            self.absolute_budget is not None or
            len(self.material_cost_limits) > 0
        )

        if not has_constraint:
            return False, "프리미엄 한도, 절대 예산, 자재별 제한 중 하나는 설정되어야 합니다."

        # Premium 한도 사용 시 기준 비용 필요
        if self.premium_limit_pct is not None and self.baseline_cost is None:
            return False, "프리미엄 한도 사용 시 기준 비용(baseline_cost)이 필요합니다."

        return True, "검증 완료"

    def check_feasibility(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        실현 가능성 확인

        Args:
            data: 시뮬레이션 데이터

        Returns:
            (is_feasible, message)
        """
        scenario_df = data.get('scenario_df')
        original_df = data.get('original_df')

        if scenario_df is None or original_df is None:
            return False, "scenario_df 또는 original_df가 없습니다."

        # Cost calculator 확인
        if self.cost_calculator is None:
            return False, "Cost calculator가 설정되지 않았습니다."

        # 자재별 제한이 있는 경우, 해당 자재가 데이터에 있는지 확인
        available_materials = set(scenario_df['자재명'].unique())
        for material in self.material_cost_limits.keys():
            if material not in available_materials:
                return False, f"자재 '{material}'이(가) 데이터에 없습니다."

        return True, "실현 가능"

    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터
        """
        # 진단 로그: 비용 제약조건 적용 시작
        print(f"\n{'='*60}")
        print(f"💰 비용 제약조건 적용 (CostConstraint)")
        print(f"{'='*60}")
        print(f"   • 최적화 모드: {self.optimization_mode}")
        print(f"   • 기준 비용: ${self.baseline_cost:,.2f}" if self.baseline_cost else "   • 기준 비용: 미설정")

        if self.optimization_mode == 'constraint':
            print(f"   • 제약 모드 - 비용을 hard constraint로 사용")
            if self.premium_limit_pct is not None:
                print(f"     - 프리미엄 한도: {self.premium_limit_pct}%")
            if self.absolute_budget is not None:
                print(f"     - 절대 예산: ${self.absolute_budget:,.2f}")
        elif self.optimization_mode == 'multi_objective':
            print(f"   • 복합 최적화 모드 - 비용을 목적함수에 포함")
            print(f"     - 탄소 가중치 (α): {self.carbon_weight}")
            print(f"     - 비용 가중치 (β): {self.cost_weight}")
            print(f"     → 목적함수에서 처리되므로 여기서는 제약조건 추가 없음")
        elif self.optimization_mode == 'epsilon_constraint':
            print(f"   • Epsilon-Constraint 모드 - 비용을 제약조건으로만 사용")
            print(f"     - Epsilon (비용 상한): ${self.epsilon_limit:,.2f}")

        # 비용 표현식 계산
        try:
            total_cost_expr = self._calculate_total_cost_expression(model, data)
        except Exception as e:
            print(f"    ⚠️ 비용 표현식 계산 실패: {e}")
            print(f"    → 비용 제약조건을 적용하지 않습니다.")
            import traceback
            traceback.print_exc()
            return

        # Epsilon-Constraint 모드: epsilon 상한 제약 추가
        if self.optimization_mode == 'epsilon_constraint' and self.epsilon_limit is not None:
            model.add_component(
                'epsilon_cost_constraint',
                pyo.Constraint(expr=total_cost_expr <= self.epsilon_limit)
            )

            print(f"    • Epsilon 제약: 비용 ≤ ${self.epsilon_limit:,.2f}")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        # 기존 제약 모드의 프리미엄/예산 제약
        if self.premium_limit_pct is not None and self.baseline_cost is not None:
            # 프리미엄 한도 제약
            max_premium = self.baseline_cost * (1 + self.premium_limit_pct / 100)

            model.add_component(
                'premium_limit_constraint',
                pyo.Constraint(expr=total_cost_expr <= max_premium)
            )

            print(f"    • 프리미엄 한도: ${max_premium:,.2f} (기준 대비 +{self.premium_limit_pct}%)")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        if self.absolute_budget is not None:
            # 절대 예산 제약
            model.add_component(
                'absolute_budget_constraint',
                pyo.Constraint(expr=total_cost_expr <= self.absolute_budget)
            )

            print(f"    • 절대 예산: ${self.absolute_budget:,.2f}")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        if self.material_cost_limits:
            # 자재별 비용 제한
            scenario_df = data['scenario_df']
            original_df = data['original_df']
            material_classification = data['material_classification']

            for material_name, max_cost in self.material_cost_limits.items():
                if material_name not in model.materials:
                    print(f"    ⚠️ 자재 '{material_name}'이(가) 모델에 없습니다. 스킵합니다.")
                    continue

                # 자재 정보 가져오기
                material_row = scenario_df[scenario_df['자재명'] == material_name].iloc[0]
                material_category = material_row['자재품목']
                quantity = material_row['제품총소요량(kg)']

                # 국가 정보
                country = self.cost_calculator._get_material_country(material_name, original_df)

                # 자재품목 매핑
                opt_material = self.cost_calculator._map_material_category(material_category)

                # 자재 타입
                material_type = material_classification[material_name]['type']

                # 자재별 비용 표현식
                if material_type == 'Formula':
                    tier1_conversion = self.cost_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier1", country
                    )
                    tier2_conversion = self.cost_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier2", country
                    )

                    material_cost_expr = quantity * (
                        tier1_conversion * model.tier1_re[material_name] +
                        tier2_conversion * model.tier2_re[material_name]
                    )
                else:
                    basic_cost = self.cost_calculator._get_basic_cost(opt_material, "Tier1")
                    material_cost_expr = quantity * basic_cost

                # 제약조건 추가
                constraint_name = f'material_cost_limit_{material_name.replace(" ", "_")}'
                model.add_component(
                    constraint_name,
                    pyo.Constraint(expr=material_cost_expr <= max_cost)
                )

                print(f"    • {material_name} 비용 한도: ${max_cost:,.2f}")
                print(f"      ✅ Pyomo 제약조건 추가 완료")

    def _calculate_total_cost_expression(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any]
    ) -> pyo.Expression:
        """
        모델의 총 비용 표현식 계산

        Args:
            model: Pyomo 모델
            data: 시뮬레이션 데이터

        Returns:
            비용 표현식 (Pyomo Expression)
        """
        if self.cost_calculator is None:
            raise ValueError("Cost calculator가 설정되지 않았습니다.")

        scenario_df = data['scenario_df']
        original_df = data['original_df']
        material_classification = data['material_classification']

        # 비용 표현식 생성
        total_cost = 0.0

        # 진단 로그: 샘플 비용 계산 정보 수집
        sample_logged = False
        cathode_re100_prices = []

        for material in model.materials:
            # 자재 정보 가져오기
            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            material_category = material_row['자재품목']
            quantity = material_row['제품총소요량(kg)']

            # 국가 정보 가져오기
            country = self.cost_calculator._get_material_country(material, original_df)

            # 자재품목 매핑
            opt_material = self.cost_calculator._map_material_category(material_category)

            # 자재 타입 확인
            material_type = material_classification[material]['type']

            if material_type == 'Formula':
                # 양극재 여부 확인
                cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
                is_cathode = any(keyword in material for keyword in cathode_keywords)

                if is_cathode and hasattr(model, 'element_recycle_ratio'):
                    # 양극재: 원소별 비용 + RE100 비용 계산
                    # 1. 원소별 기본 비용
                    cathode_composition = data.get('cathode_composition', {})
                    material_cost_premiums = data.get('material_cost_premiums', {})

                    # 원소별 비용 계산
                    element_cost_expr = 0
                    for element in model.elements:
                        if element not in cathode_composition or cathode_composition[element] == 0:
                            continue

                        # 조성비
                        comp_ratio = cathode_composition[element]

                        # 원소별 기본 비용 (basic_cost를 원소별로 분배)
                        basic_cost = self.cost_calculator._get_basic_cost(opt_material, "Tier1")
                        element_basic_cost = basic_cost * comp_ratio

                        # 프리미엄 가져오기
                        if element in material_cost_premiums:
                            premiums = material_cost_premiums[element]
                        elif material_cost_premiums.get('default'):
                            premiums = material_cost_premiums['default']
                        else:
                            premiums = {"recycle_premium_pct": 30.0, "low_carbon_premium_pct": 50.0}

                        recycle_premium_pct = premiums.get('recycle_premium_pct', 30.0)
                        low_carbon_premium_pct = premiums.get('low_carbon_premium_pct', 50.0)

                        # 원소별 신재/재활용/저탄소 비용
                        virgin_cost = element_basic_cost
                        recycle_cost = element_basic_cost * (1 + recycle_premium_pct / 100)
                        low_carbon_cost = element_basic_cost * (1 + low_carbon_premium_pct / 100)

                        # 원소별 비용 = 가중평균
                        element_cost_expr += (
                            virgin_cost * model.element_virgin_ratio[element] +
                            recycle_cost * model.element_recycle_ratio[element] +
                            low_carbon_cost * model.element_low_carb_ratio[element]
                        )

                    # 2. RE100 비용 (양극재 전체에 적용)
                    tier1_conversion = self.cost_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier1", country
                    )
                    tier2_conversion = self.cost_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier2", country
                    )

                    # 진단 로그: 양극재 RE100 비용 정보 수집
                    cathode_re100_prices.append({
                        'material': material,
                        'quantity': quantity,
                        'tier1_conversion': tier1_conversion,
                        'tier2_conversion': tier2_conversion,
                        'country': country,
                        'opt_material': opt_material
                    })

                    re100_cost_expr = (
                        tier1_conversion * model.tier1_re[material] +
                        tier2_conversion * model.tier2_re[material]
                    )

                    # 총 비용 = (원소별 비용 + RE100 비용) × 수량
                    material_cost = quantity * (element_cost_expr + re100_cost_expr)

                    # 진단 로그: 첫 번째 양극재의 상세 비용 정보만 출력
                    if not sample_logged:
                        print(f"\n   🔍 샘플 비용 계산 (첫 번째 양극재: {material[:50]}...)")
                        print(f"      • 수량: {quantity:.4f} kg")
                        print(f"      • 국가: {country}")
                        print(f"      • 자재품목 매핑: {opt_material}")
                        print(f"      • RE100 Tier1 전환가격: ${tier1_conversion:.6f}/kg")
                        print(f"      • RE100 Tier2 전환가격: ${tier2_conversion:.6f}/kg")
                        print(f"      • RE100 비용 표현식: {quantity:.4f} × (${tier1_conversion:.6f} × tier1_re + ${tier2_conversion:.6f} × tier2_re)")
                        print(f"      💡 RE100=100%일 때 비용: ${quantity * (tier1_conversion + tier2_conversion):.2f}")
                        sample_logged = True

                else:
                    # 일반 Formula 자재: RE 비율에 따른 비용 계산
                    # Tier1 RE100 전환가격
                    tier1_conversion = self.cost_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier1", country
                    )
                    # Tier2 RE100 전환가격
                    tier2_conversion = self.cost_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier2", country
                    )

                    # 비용 = 수량 × (tier1_conversion × tier1_re + tier2_conversion × tier2_re)
                    material_cost = quantity * (
                        tier1_conversion * model.tier1_re[material] +
                        tier2_conversion * model.tier2_re[material]
                    )

                total_cost += material_cost

            elif material_type == 'Ni-Co-Li':
                # Ni-Co-Li 자재: 재활용/저탄소메탈 비용 프리미엄 반영
                basic_cost = self.cost_calculator._get_basic_cost(opt_material, "Tier1")

                # 비용 프리미엄 데이터 가져오기
                material_cost_premiums = data.get('material_cost_premiums', {})

                # 자재명에서 원소 추출 (예: "Ni-Sulfate" → "Ni")
                element = None
                for elem in ['Ni', 'Co', 'Li']:
                    if elem in material:
                        element = elem
                        break

                if element and element in material_cost_premiums:
                    premiums = material_cost_premiums[element]
                    recycle_premium_pct = premiums.get('recycle_premium_pct', 30.0)
                    low_carbon_premium_pct = premiums.get('low_carbon_premium_pct', 50.0)
                elif material_cost_premiums.get('default'):
                    # 기본값 사용
                    premiums = material_cost_premiums['default']
                    recycle_premium_pct = premiums.get('recycle_premium_pct', 30.0)
                    low_carbon_premium_pct = premiums.get('low_carbon_premium_pct', 50.0)
                else:
                    # 프리미엄 데이터가 없으면 기본값
                    recycle_premium_pct = 30.0
                    low_carbon_premium_pct = 50.0

                # 각 비율별 비용 계산
                virgin_cost = basic_cost
                recycle_cost = basic_cost * (1 + recycle_premium_pct / 100)
                low_carbon_cost = basic_cost * (1 + low_carbon_premium_pct / 100)

                # 비율에 따른 가중평균 비용
                material_cost = quantity * (
                    virgin_cost * model.virgin_ratio[material] +
                    recycle_cost * model.recycle_ratio[material] +
                    low_carbon_cost * model.low_carbon_ratio[material]
                )

                total_cost += material_cost

            else:
                # 일반 자재: 기본 비용만
                basic_cost = self.cost_calculator._get_basic_cost(opt_material, "Tier1")
                material_cost = quantity * basic_cost
                total_cost += material_cost

        # 진단 로그: 양극재 RE100 비용 정보 요약
        if cathode_re100_prices:
            print(f"\n   📊 양극재 RE100 비용 정보 요약 ({len(cathode_re100_prices)}개 양극재)")
            print(f"   {'='*60}")
            for idx, info in enumerate(cathode_re100_prices, 1):
                material_name = info['material']
                tier1_conv = info['tier1_conversion']
                tier2_conv = info['tier2_conversion']
                qty = info['quantity']
                max_re100_cost = qty * (tier1_conv + tier2_conv)

                print(f"   {idx}. {material_name[:50]}...")
                print(f"      • Tier1: ${tier1_conv:.6f}/kg, Tier2: ${tier2_conv:.6f}/kg")
                print(f"      • RE100=100% 시 비용: ${max_re100_cost:.2f}")
            print(f"   {'='*60}")

        return total_cost

    def get_multi_objective_expression(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any],
        carbon_expr: pyo.Expression,
        baseline_carbon: Optional[float] = None
    ) -> pyo.Expression:
        """
        복합 최적화 목적함수 생성 (정규화 개선)

        목적함수 = α × (carbon / baseline_carbon) + β × (cost / baseline_cost)

        Args:
            model: Pyomo 모델
            data: 시뮬레이션 데이터
            carbon_expr: 기존 탄소 배출 목적함수 표현식
            baseline_carbon: 기준 탄소 배출량 (정규화용)

        Returns:
            복합 목적함수 표현식 (Pyomo Expression)
        """
        if self.optimization_mode != 'multi_objective':
            raise ValueError("복합 최적화 모드가 아닙니다.")

        if self.baseline_cost is None or self.baseline_cost == 0:
            raise ValueError("복합 최적화에는 baseline_cost가 필요합니다.")

        print(f"\n   🎯 복합 최적화 목적함수 생성 시작")
        print(f"   {'='*60}")

        # 비용 표현식 계산
        cost_expr = self._calculate_total_cost_expression(model, data)

        # 비용 정규화 (baseline 대비 비율)
        normalized_cost = cost_expr / self.baseline_cost
        print(f"   • 기준 비용 (baseline_cost): ${self.baseline_cost:.2f}")
        print(f"   • 정규화 비용: cost_expr / ${self.baseline_cost:.2f}")

        # 탄소 정규화 (baseline 대비 비율)
        if baseline_carbon and baseline_carbon > 0:
            normalized_carbon = carbon_expr / baseline_carbon
            print(f"   • 기준 탄소 (baseline_carbon): {baseline_carbon:.2f} kgCO2eq")
            print(f"   • 정규화 탄소: carbon_expr / {baseline_carbon:.2f}")
        else:
            # baseline_carbon이 없으면 정규화 없이 사용 (레거시 호환)
            normalized_carbon = carbon_expr
            print(f"   ⚠️  기준 탄소 (baseline_carbon): 없음 → 정규화 미적용")

        # 복합 목적함수 (두 항 모두 정규화)
        multi_obj_expr = (
            self.carbon_weight * normalized_carbon +
            self.cost_weight * normalized_cost
        )

        print(f"\n   📐 목적함수 구성:")
        print(f"      Objective = {self.carbon_weight}×정규화탄소 + {self.cost_weight}×정규화비용")
        print(f"      Objective = {self.carbon_weight}×(carbon/{baseline_carbon if baseline_carbon else '1'}) + {self.cost_weight}×(cost/${self.baseline_cost:.2f})")

        # 가중치 비율 계산
        total_weight = self.carbon_weight + self.cost_weight
        if total_weight > 0:
            carbon_pct = (self.carbon_weight / total_weight) * 100
            cost_pct = (self.cost_weight / total_weight) * 100
            print(f"      정규화 비율: 탄소 {carbon_pct:.1f}% / 비용 {cost_pct:.1f}%")

        print(f"   {'='*60}")
        print(f"   ✅ 복합 목적함수 생성 완료")

        return multi_obj_expr

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        return {
            'type': 'cost_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'baseline_cost': self.baseline_cost,
            'premium_limit_pct': self.premium_limit_pct,
            'absolute_budget': self.absolute_budget,
            'material_cost_limits': self.material_cost_limits,
            'optimization_mode': self.optimization_mode,
            'carbon_weight': self.carbon_weight,
            'cost_weight': self.cost_weight
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        self.name = config.get('name', 'cost_constraint')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.baseline_cost = config.get('baseline_cost')
        self.premium_limit_pct = config.get('premium_limit_pct')
        self.absolute_budget = config.get('absolute_budget')
        self.material_cost_limits = config.get('material_cost_limits', {})
        self.optimization_mode = config.get('optimization_mode', 'constraint')
        self.carbon_weight = config.get('carbon_weight', 0.7)
        self.cost_weight = config.get('cost_weight', 0.3)

    def get_summary(self) -> str:
        """
        제약조건 요약

        Returns:
            요약 문자열
        """
        base_summary = super().get_summary()
        details = []

        # 최적화 모드
        mode_str = "복합 최적화" if self.optimization_mode == 'multi_objective' else "제약 모드"
        details.append(f"모드: {mode_str}")

        if self.optimization_mode == 'multi_objective':
            details.append(f"가중치 α={self.carbon_weight:.2f}, β={self.cost_weight:.2f}")

        if self.baseline_cost:
            details.append(f"기준 비용: ${self.baseline_cost:,.2f}")

        if self.premium_limit_pct is not None:
            details.append(f"프리미엄 한도: +{self.premium_limit_pct}%")

        if self.absolute_budget is not None:
            details.append(f"절대 예산: ${self.absolute_budget:,.2f}")

        if self.material_cost_limits:
            details.append(f"자재별 제한: {len(self.material_cost_limits)}개")

        detail_str = " | ".join(details) if details else "설정 없음"
        return f"{base_summary}\n  💰 {detail_str}"

    def get_display_info(self) -> Dict[str, Any]:
        """UI 표시용 비용 제약조건 정보"""
        info = {
            'mode': self.optimization_mode,
            'mode_description': self._get_mode_description(),
            'baseline_cost': self.baseline_cost,
            'settings': [],
            'pareto_warning': self._get_pareto_warning()
        }

        if self.optimization_mode == 'constraint':
            if self.premium_limit_pct is not None:
                info['settings'].append(f"프리미엄 한도: +{self.premium_limit_pct}%")
            if self.absolute_budget is not None:
                info['settings'].append(f"절대 예산: ${self.absolute_budget:,.2f}")
            if self.material_cost_limits:
                info['settings'].append(f"자재별 제한: {len(self.material_cost_limits)}개")
        elif self.optimization_mode == 'multi_objective':
            info['settings'].append(f"탄소 가중치 (α): {self.carbon_weight:.2f}")
            info['settings'].append(f"비용 가중치 (β): {self.cost_weight:.2f}")
        elif self.optimization_mode == 'epsilon_constraint':
            if self.epsilon_limit is not None:
                info['settings'].append(f"비용 상한 (ε): ${self.epsilon_limit:,.2f}")

        return info

    def _get_mode_description(self) -> str:
        """모드별 설명"""
        descriptions = {
            'constraint': '제약 조건 모드 - 비용을 hard constraint로 사용',
            'multi_objective': '복합 최적화 모드 - 비용을 목적함수에 포함',
            'epsilon_constraint': 'Epsilon-Constraint 모드 - 비용 상한 제약'
        }
        return descriptions.get(self.optimization_mode, '알 수 없는 모드')

    def _get_pareto_warning(self) -> str:
        """Pareto 최적화 시 경고 메시지"""
        if self.optimization_mode == 'constraint':
            return "⚠️ 파레토 최적화 실행 시 이 비용 제약은 무시되고 최적화 엔진이 생성한 비용 제약으로 대체됩니다."
        elif self.optimization_mode == 'multi_objective':
            return "✅ 복합 최적화 모드는 Weighted Sum 파레토 최적화에 사용됩니다."
        elif self.optimization_mode == 'epsilon_constraint':
            return "✅ Epsilon-Constraint 모드는 Epsilon 파레토 최적화에 사용됩니다."
        return ""

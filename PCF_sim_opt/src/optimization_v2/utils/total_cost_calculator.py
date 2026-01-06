"""
통합 비용 계산 모듈 (Total Cost Calculator)

이 모듈은 최적화 시스템의 모든 비용 계산을 통합하여 처리합니다:
- Zero-Premium Baseline: 환경 개선 활동 없는 상태의 기준 비용
- Total Premium: RE100 + 재활용재 + 저탄소메탈 프리미엄
- Total Cost: Zero-Premium Baseline + Total Premium
"""

from typing import Dict, Any, Optional
import pandas as pd
import pyomo.environ as pyo
from ..costs.re100_cost_calculator import RE100CostCalculator


class TotalCostCalculator:
    """
    통합 비용 계산 클래스

    모든 비용 계산을 일관되게 처리하여 Pareto optimizer 간 일관성 보장
    """

    def __init__(
        self,
        re100_calculator=None,
        re100_cost_calculator: Optional[RE100CostCalculator] = None,
        debug_mode: bool = False
    ):
        """
        TotalCostCalculator 초기화

        Args:
            re100_calculator: (Legacy) RE100PremiumCalculator 인스턴스 (기본 비용 계산용)
            re100_cost_calculator: (신규) RE100CostCalculator 인스턴스 (지역별 RE100 프리미엄)
            debug_mode: 디버그 모드 (상세 로그 출력)
        """
        self.re100_calculator = re100_calculator  # Legacy calculator (for basic costs)
        self.re100_cost_calculator = re100_cost_calculator  # New RE100 cost calculator
        self.debug_mode = debug_mode
        self.zero_premium_baseline: Optional[float] = None

        # RE100CostCalculator가 없으면 생성 (기본 설정)
        if self.re100_cost_calculator is None:
            self.re100_cost_calculator = RE100CostCalculator(debug_mode=debug_mode)

    def calculate_zero_premium_baseline(
        self,
        scenario_df: pd.DataFrame,
        material_classification: Dict[str, Any],
        original_df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Zero-Premium Baseline 계산

        환경 개선 활동 없는 상태의 총 비용:
        - RE100 = 0%
        - 재활용재 = 0%
        - 저탄소메탈 = 0%

        각 자재의 basic_cost × quantity를 합산

        Args:
            scenario_df: 시나리오 DataFrame
            material_classification: 자재 분류 정보
            original_df: 원본 DataFrame (국가 정보 추출용, optional)

        Returns:
            float: Zero-Premium Baseline 비용 (USD)
        """
        if self.debug_mode:
            print("\n" + "="*60)
            print("💰 Zero-Premium Baseline 계산 시작")
            print("="*60)

        total_baseline_cost = 0.0
        material_count = 0

        # 각 자재별로 기본 비용 계산
        for material_name in material_classification.keys():
            # 자재 정보 가져오기
            material_rows = scenario_df[scenario_df['자재명'] == material_name]

            if material_rows.empty:
                if self.debug_mode:
                    print(f"⚠️ 자재를 찾을 수 없음: {material_name}")
                continue

            material_row = material_rows.iloc[0]
            material_category = material_row['자재품목']
            quantity = material_row['제품총소요량(kg)']

            # 자재품목 매핑
            opt_material = self.re100_calculator._map_material_category(material_category)

            # 기본 비용 조회 (Tier1 기준)
            # RE100 = 0%, 재활용 = 0%, 저탄소 = 0% 상태의 비용
            basic_cost = self.re100_calculator._get_basic_cost(opt_material, "Tier1")

            # 자재별 총 비용
            material_total_cost = basic_cost * quantity
            total_baseline_cost += material_total_cost

            material_count += 1

            # 디버그 로그 (첫 5개만 출력)
            if self.debug_mode and material_count <= 5:
                print(f"\n{material_count}. {material_name[:40]}...")
                print(f"   • 자재품목: {material_category} → {opt_material}")
                print(f"   • 수량: {quantity:.4f} kg")
                print(f"   • 기본단가: ${basic_cost:.6f}/kg")
                print(f"   • 총 비용: ${material_total_cost:.2f}")

        self.zero_premium_baseline = total_baseline_cost

        if self.debug_mode:
            if material_count > 5:
                print(f"\n... 외 {material_count - 5}개 자재")
            print("\n" + "="*60)
            print(f"✅ Zero-Premium Baseline 계산 완료")
            print(f"   총 자재 수: {material_count}개")
            print(f"   Zero-Premium Baseline: ${total_baseline_cost:,.2f}")
            print("="*60)

        return total_baseline_cost

    def calculate_re100_premium(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any]
    ) -> pyo.Expression:
        """
        RE100 프리미엄 비용 표현식 계산

        새로운 RE100CostCalculator를 사용하여 지역별 RE100 프리미엄 계산

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터

        Returns:
            pyo.Expression: RE100 프리미엄 비용 표현식
        """
        # 새로운 RE100CostCalculator 사용
        if self.re100_cost_calculator:
            re100_premium_expr = self.re100_cost_calculator.calculate_re100_premium_expression(
                model, data
            )

            if self.debug_mode:
                print("\n💰 RE100 프리미엄 계산 (RE100CostCalculator 사용)")
                print("   지역별 프리미엄 및 에너지 기반 비용 모델 적용")

            return re100_premium_expr

        # Fallback: Legacy calculator 사용 (하위 호환성)
        elif self.re100_calculator:
            if self.debug_mode:
                print("\n⚠️  Legacy RE100PremiumCalculator 사용 (하위 호환 모드)")

            scenario_df = data['scenario_df']
            original_df = data.get('original_df')
            material_classification = data['material_classification']

            re100_premium_expr = 0.0

            for material in model.materials:
                # 자재 정보 가져오기
                material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
                material_category = material_row['자재품목']
                quantity = material_row['제품총소요량(kg)']

                # 국가 정보
                if original_df is not None:
                    country = self.re100_calculator._get_material_country(material, original_df)
                else:
                    country = "한국"  # Fallback

                # 자재품목 매핑
                opt_material = self.re100_calculator._map_material_category(material_category)

                # 자재 타입 확인
                material_type = material_classification[material]['type']

                # Formula 자재만 RE100 적용
                if material_type == 'Formula':
                    # RE100 전환가격 ($/kg)
                    tier1_conversion = self.re100_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier1", country
                    )
                    tier2_conversion = self.re100_calculator.calculate_re100_conversion_price(
                        opt_material, "Tier2", country
                    )

                    # RE100 프리미엄 = 수량 × (tier1_conversion × tier1_re + tier2_conversion × tier2_re)
                    material_re100_premium = quantity * (
                        tier1_conversion * model.tier1_re[material] +
                        tier2_conversion * model.tier2_re[material]
                    )

                    re100_premium_expr += material_re100_premium

            return re100_premium_expr

        else:
            # 둘 다 없으면 0 반환
            print("\n⚠️  RE100 calculator가 없습니다. RE100 프리미엄 = 0")
            return 0.0

    def calculate_recycling_premium(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any]
    ) -> pyo.Expression:
        """
        재활용재 프리미엄 비용 표현식 계산

        - 양극재: 원소별 recycle_premium_pct 적용
        - Ni/Co/Li 자재: 자재별 recycle_premium_pct 적용

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터

        Returns:
            pyo.Expression: 재활용재 프리미엄 비용 표현식
        """
        scenario_df = data['scenario_df']
        material_classification = data['material_classification']
        material_cost_premiums = data.get('material_cost_premiums', {})
        cathode_composition = data.get('cathode_composition', {})

        recycling_premium_expr = 0.0

        for material in model.materials:
            # 자재 정보 가져오기
            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            material_category = material_row['자재품목']
            quantity = material_row['제품총소요량(kg)']

            # 자재품목 매핑
            opt_material = self.re100_calculator._map_material_category(material_category)

            # 자재 타입 확인
            material_type = material_classification[material]['type']

            # 기본 비용
            basic_cost = self.re100_calculator._get_basic_cost(opt_material, "Tier1")

            if material_type == 'Formula':
                # 양극재: 원소별 재활용 프리미엄
                cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
                is_cathode = any(keyword in material for keyword in cathode_keywords)

                if is_cathode and hasattr(model, 'element_recycle_ratio'):
                    # 원소별 재활용 프리미엄 계산
                    element_recycling_premium = 0

                    for element in model.elements:
                        if element not in cathode_composition or cathode_composition[element] == 0:
                            continue

                        # 조성비
                        comp_ratio = cathode_composition[element]

                        # 원소별 기본 비용
                        element_basic_cost = basic_cost * comp_ratio

                        # 프리미엄 비율
                        if element in material_cost_premiums:
                            recycle_premium_pct = material_cost_premiums[element].get('recycle_premium_pct', 30.0)
                        elif 'default' in material_cost_premiums:
                            recycle_premium_pct = material_cost_premiums['default'].get('recycle_premium_pct', 30.0)
                        else:
                            recycle_premium_pct = 30.0

                        # 재활용재 프리미엄 = (기본 비용 × 프리미엄 비율) × 재활용 비율
                        element_premium = element_basic_cost * (recycle_premium_pct / 100) * model.element_recycle_ratio[element]
                        element_recycling_premium += element_premium

                    recycling_premium_expr += quantity * element_recycling_premium

            elif material_type == 'Ni-Co-Li':
                # Ni-Co-Li 자재: 자재별 재활용 프리미엄
                # 자재명에서 원소 추출
                element = None
                for elem in ['Ni', 'Co', 'Li']:
                    if elem in material:
                        element = elem
                        break

                # 프리미엄 비율
                if element and element in material_cost_premiums:
                    recycle_premium_pct = material_cost_premiums[element].get('recycle_premium_pct', 30.0)
                elif 'default' in material_cost_premiums:
                    recycle_premium_pct = material_cost_premiums['default'].get('recycle_premium_pct', 30.0)
                else:
                    recycle_premium_pct = 30.0

                # 재활용재 프리미엄 = 수량 × 기본 비용 × (프리미엄 비율 / 100) × 재활용 비율
                material_recycling_premium = quantity * basic_cost * (recycle_premium_pct / 100) * model.recycle_ratio[material]
                recycling_premium_expr += material_recycling_premium

        return recycling_premium_expr

    def calculate_low_carbon_premium(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any]
    ) -> pyo.Expression:
        """
        저탄소메탈 프리미엄 비용 표현식 계산

        재활용재와 유사한 방식으로 계산

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터

        Returns:
            pyo.Expression: 저탄소메탈 프리미엄 비용 표현식
        """
        scenario_df = data['scenario_df']
        material_classification = data['material_classification']
        material_cost_premiums = data.get('material_cost_premiums', {})
        cathode_composition = data.get('cathode_composition', {})

        low_carbon_premium_expr = 0.0

        for material in model.materials:
            # 자재 정보 가져오기
            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            material_category = material_row['자재품목']
            quantity = material_row['제품총소요량(kg)']

            # 자재품목 매핑
            opt_material = self.re100_calculator._map_material_category(material_category)

            # 자재 타입 확인
            material_type = material_classification[material]['type']

            # 기본 비용
            basic_cost = self.re100_calculator._get_basic_cost(opt_material, "Tier1")

            if material_type == 'Formula':
                # 양극재: 원소별 저탄소메탈 프리미엄
                cathode_keywords = ['CAM', '양극활물질', 'Cathode Active Material']
                is_cathode = any(keyword in material for keyword in cathode_keywords)

                if is_cathode and hasattr(model, 'element_low_carb_ratio'):
                    # 원소별 저탄소메탈 프리미엄 계산
                    element_low_carbon_premium = 0

                    for element in model.elements:
                        if element not in cathode_composition or cathode_composition[element] == 0:
                            continue

                        # 조성비
                        comp_ratio = cathode_composition[element]

                        # 원소별 기본 비용
                        element_basic_cost = basic_cost * comp_ratio

                        # 프리미엄 비율
                        if element in material_cost_premiums:
                            low_carbon_premium_pct = material_cost_premiums[element].get('low_carbon_premium_pct', 50.0)
                        elif 'default' in material_cost_premiums:
                            low_carbon_premium_pct = material_cost_premiums['default'].get('low_carbon_premium_pct', 50.0)
                        else:
                            low_carbon_premium_pct = 50.0

                        # 저탄소메탈 프리미엄 = (기본 비용 × 프리미엄 비율) × 저탄소 비율
                        element_premium = element_basic_cost * (low_carbon_premium_pct / 100) * model.element_low_carb_ratio[element]
                        element_low_carbon_premium += element_premium

                    low_carbon_premium_expr += quantity * element_low_carbon_premium

            elif material_type == 'Ni-Co-Li':
                # Ni-Co-Li 자재: 자재별 저탄소메탈 프리미엄
                # 자재명에서 원소 추출
                element = None
                for elem in ['Ni', 'Co', 'Li']:
                    if elem in material:
                        element = elem
                        break

                # 프리미엄 비율
                if element and element in material_cost_premiums:
                    low_carbon_premium_pct = material_cost_premiums[element].get('low_carbon_premium_pct', 50.0)
                elif 'default' in material_cost_premiums:
                    low_carbon_premium_pct = material_cost_premiums['default'].get('low_carbon_premium_pct', 50.0)
                else:
                    low_carbon_premium_pct = 50.0

                # 저탄소메탈 프리미엄 = 수량 × 기본 비용 × (프리미엄 비율 / 100) × 저탄소 비율
                material_low_carbon_premium = quantity * basic_cost * (low_carbon_premium_pct / 100) * model.low_carbon_ratio[material]
                low_carbon_premium_expr += material_low_carbon_premium

        return low_carbon_premium_expr

    def calculate_total_premium(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any]
    ) -> pyo.Expression:
        """
        총 프리미엄 계산

        Total Premium = RE100 Premium + Recycling Premium + Low-Carbon Premium

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터

        Returns:
            pyo.Expression: 총 프리미엄 비용 표현식
        """
        re100_premium = self.calculate_re100_premium(model, data)
        recycling_premium = self.calculate_recycling_premium(model, data)
        low_carbon_premium = self.calculate_low_carbon_premium(model, data)

        total_premium = re100_premium + recycling_premium + low_carbon_premium

        if self.debug_mode:
            print("\n💰 총 프리미엄 구성:")
            print(f"   • RE100 프리미엄: (Pyomo Expression)")
            print(f"   • 재활용재 프리미엄: (Pyomo Expression)")
            print(f"   • 저탄소메탈 프리미엄: (Pyomo Expression)")
            print(f"   → 총 프리미엄 = RE100 + 재활용 + 저탄소")

        return total_premium

    def calculate_total_cost(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any]
    ) -> pyo.Expression:
        """
        총 비용 계산

        Total Cost = Zero-Premium Baseline + Total Premium

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터

        Returns:
            pyo.Expression: 총 비용 표현식
        """
        if self.zero_premium_baseline is None:
            raise ValueError("Zero-Premium Baseline이 계산되지 않았습니다. calculate_zero_premium_baseline()을 먼저 호출하세요.")

        total_premium = self.calculate_total_premium(model, data)

        # 총 비용 = 기준 비용 + 프리미엄
        total_cost = self.zero_premium_baseline + total_premium

        if self.debug_mode:
            print("\n💰 총 비용 계산:")
            print(f"   • Zero-Premium Baseline: ${self.zero_premium_baseline:,.2f}")
            print(f"   • Total Premium: (Pyomo Expression)")
            print(f"   → Total Cost = Baseline + Premium")

        return total_cost

    def get_summary(self) -> Dict[str, Any]:
        """
        비용 계산 요약 정보 반환

        Returns:
            Dict: 요약 정보
        """
        return {
            'zero_premium_baseline': self.zero_premium_baseline,
            'debug_mode': self.debug_mode
        }

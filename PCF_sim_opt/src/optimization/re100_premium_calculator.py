"""
RE100 프리미엄 비용 계산 모듈

이 모듈은 RE100 적용 시 발생하는 프리미엄 비용을 계산합니다.
자재별, 국가별, Tier별, Case별로 상세한 비용 분석을 제공합니다.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import sys

# OptimizationCostsManager import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))
from src.utils.optimization_costs_manager import OptimizationCostsManager
from src.utils.file_operations import FileOperations


class RE100PremiumCalculator:
    """
    RE100 프리미엄 비용 계산 클래스

    데이터 소스:
    - electricity_usage_per_material.json: 자재품목+Tier별 전력사용량(kWh/kg)
    - unit_cost_per_country.json: 국가별 전력단가($/kWh)
    - basic_cost_per_material.json: 자재품목+Tier별 기본단가($/kg)
    - material_category_mapping.json: 자재품목 매핑 룰
    - cathode_national_code.json: 지역 코드 → 국가명 변환
    """

    def __init__(self, user_id: Optional[str] = None, debug_mode: bool = False):
        """
        RE100PremiumCalculator 초기화

        Args:
            user_id: 사용자 ID (사용자별 데이터 사용시)
            debug_mode: 디버그 모드 (상세 로그 출력)
        """
        self.user_id = user_id
        self.debug_mode = debug_mode

        # OptimizationCostsManager 초기화
        self.costs_manager = OptimizationCostsManager()

        # 사용자별 데이터 초기화 (필요시)
        if self.user_id:
            if not self.costs_manager.check_user_costs_exist(user_id):
                if self.debug_mode:
                    print(f"📁 사용자 {user_id}의 optimization_costs 파일 초기화 중...")
                self.costs_manager.initialize_user_costs(user_id)

        # 데이터 로드
        self._load_all_data()

        if self.debug_mode:
            self._print_debug_info()

    def _load_all_data(self) -> None:
        """모든 필요한 데이터 로드"""
        # electricity_usage_per_material.json
        self.electricity_usage_data = self.costs_manager.load_user_file(
            self.user_id or "default",
            "electricity_usage_per_material.json",
            fallback_to_template=True
        ) or []

        # unit_cost_per_country.json
        self.unit_cost_data = self.costs_manager.load_user_file(
            self.user_id or "default",
            "unit_cost_per_country.json",
            fallback_to_template=True
        ) or []

        # basic_cost_per_material.json (아직 없으면 빈 리스트)
        self.basic_cost_data = self.costs_manager.load_user_file(
            self.user_id or "default",
            "basic_cost_per_material.json",
            fallback_to_template=True
        ) or []

        # material_category_mapping.json
        self.mapping_data = self.costs_manager.load_user_file(
            self.user_id or "default",
            "material_category_mapping.json",
            fallback_to_template=True
        ) or {}

        # cathode_national_code.json (지역 코드 → 국가명 변환)
        try:
            self.national_code_data = FileOperations.load_json(
                "stable_var/cathode_national_code.json",
                user_id=self.user_id
            ) or {}
        except:
            self.national_code_data = {}

        # 데이터 인덱싱 (빠른 조회를 위해)
        self._build_indices()

    def _build_indices(self) -> None:
        """데이터를 빠른 조회가 가능한 형태로 인덱싱"""
        # 전력사용량 인덱스: {자재품목: {Tier: 전력사용량}}
        self.electricity_index = {}
        for item in self.electricity_usage_data:
            material = item.get('자재품목')
            tier = item.get('Tier')
            usage = item.get('전력사용량(kWh/kg)', 0)

            if material not in self.electricity_index:
                self.electricity_index[material] = {}
            self.electricity_index[material][tier] = usage

        # 전력단가 인덱스: {국가: 단가}
        self.unit_cost_index = {}
        for item in self.unit_cost_data:
            country = item.get('국가')
            cost = item.get('금액($/kWh)', 0)
            self.unit_cost_index[country] = cost

        # 기본단가 인덱스: {자재품목: {Tier: 기본단가}}
        self.basic_cost_index = {}
        for item in self.basic_cost_data:
            material = item.get('자재품목')
            tier = item.get('Tier')
            cost = item.get('기본단가($/kg)', 0)

            if material not in self.basic_cost_index:
                self.basic_cost_index[material] = {}
            self.basic_cost_index[material][tier] = cost

        # 자재품목 매핑 (reverse_mappings 사용)
        self.material_mapping = self.mapping_data.get('reverse_mappings', {})

    def _print_debug_info(self) -> None:
        """디버그 정보 출력"""
        print("===== RE100PremiumCalculator 초기화 정보 =====")
        print(f"• 전력사용량 데이터: {len(self.electricity_usage_data)}개")
        print(f"• 전력단가 데이터: {len(self.unit_cost_data)}개 국가")
        print(f"• 기본단가 데이터: {len(self.basic_cost_data)}개")
        print(f"• 자재품목 매핑: {len(self.material_mapping)}개")

        # 전력사용량 데이터 상세
        if len(self.electricity_index) > 0:
            print("\n전력사용량 인덱스:")
            for material, tiers in self.electricity_index.items():
                print(f"  • {material}: {tiers}")

        # 전력단가 데이터 상세
        if len(self.unit_cost_index) > 0:
            print("\n전력단가 인덱스:")
            for country, cost in self.unit_cost_index.items():
                print(f"  • {country}: ${cost}/kWh")

    def _map_material_category(self, brm_category: str) -> str:
        """
        BRM 자재품목 → optimization_costs 자재품목 매핑

        Args:
            brm_category: BRM 테이블의 자재품목 (예: "Cu Foil")

        Returns:
            str: optimization_costs의 자재품목 (예: "Cu-Foil")
        """
        # 직접 매핑 테이블 확인
        if brm_category in self.material_mapping:
            return self.material_mapping[brm_category]

        # 부분 일치 검색 (대소문자 무시)
        brm_lower = brm_category.lower()
        for brm_name, opt_name in self.material_mapping.items():
            if brm_name.lower() in brm_lower or brm_lower in brm_name.lower():
                return opt_name

        # 매핑되지 않으면 원본 반환
        if self.debug_mode:
            print(f"⚠️ 매핑되지 않은 자재품목: {brm_category}")
        return brm_category

    def _convert_region_to_country(self, region_code: str) -> str:
        """
        지역 코드 → 국가명 변환

        Args:
            region_code: 지역 코드 (예: "CN", "KR")

        Returns:
            str: 국가명 (예: "중국", "한국")
        """
        national_code = self.national_code_data.get('national_code', {})
        country = national_code.get(region_code, "미분류")

        if country == "미분류" and self.debug_mode:
            print(f"⚠️ 매핑되지 않은 지역 코드: {region_code}")

        return country

    def _get_electricity_usage(self, material_category: str, tier: str) -> float:
        """
        전력사용량 조회

        Args:
            material_category: 자재품목
            tier: Tier (예: "Tier1", "Tier2")

        Returns:
            float: 전력사용량 (kWh/kg)
        """
        if material_category in self.electricity_index:
            return self.electricity_index[material_category].get(tier, 0.0)
        return 0.0

    def _get_unit_cost(self, country: str) -> float:
        """
        국가별 전력단가 조회

        Args:
            country: 국가명

        Returns:
            float: 전력단가 ($/kWh)
        """
        # 국가별 전력단가 조회
        unit_cost = self.unit_cost_index.get(country)

        if unit_cost is not None and unit_cost > 0:
            return unit_cost

        # Fallback 1: "미분류" 또는 없는 국가는 한국 값 사용
        fallback_cost = self.unit_cost_index.get('한국')
        if fallback_cost is not None and fallback_cost > 0:
            if self.debug_mode:
                print(f"⚠️ 국가 '{country}' 전력단가 없음 → 한국 값(${fallback_cost}/kWh) 사용")
            return fallback_cost

        # Fallback 2: 평균값 사용
        if len(self.unit_cost_index) > 0:
            valid_costs = [c for c in self.unit_cost_index.values() if c > 0]
            if valid_costs:
                avg_cost = sum(valid_costs) / len(valid_costs)
                if self.debug_mode:
                    print(f"⚠️ 국가 '{country}' 전력단가 없음 → 평균값(${avg_cost:.6f}/kWh) 사용")
                return avg_cost

        # 최종 Fallback: 0.1 $/kWh (글로벌 평균 근사치)
        if self.debug_mode:
            print(f"⚠️ 국가 '{country}' 전력단가 없음 → 기본값($0.10/kWh) 사용")
        return 0.1

    def _get_basic_cost(self, material_category: str, tier: str) -> float:
        """
        기본단가 조회

        Args:
            material_category: 자재품목
            tier: Tier

        Returns:
            float: 기본단가 ($/kg)
        """
        if material_category in self.basic_cost_index:
            return self.basic_cost_index[material_category].get(tier, 0.0)
        return 0.0

    def calculate_re100_conversion_price(
        self,
        material_category: str,
        tier: str,
        country: str
    ) -> float:
        """
        RE100 전환가격 계산

        공식: 전력사용량(kWh/kg) × 전력단가($/kWh)

        Args:
            material_category: 자재품목
            tier: Tier
            country: 국가명

        Returns:
            float: RE100 전환가격 ($/kg)
        """
        usage = self._get_electricity_usage(material_category, tier)
        unit_cost = self._get_unit_cost(country)

        conversion_price = usage * unit_cost

        if self.debug_mode:
            print(f"RE100 전환가격: {material_category} {tier} {country}")
            print(f"  전력사용량: {usage} kWh/kg")
            print(f"  전력단가: ${unit_cost}/kWh")
            print(f"  전환가격: ${conversion_price:.6f}/kg")

        return conversion_price

    def calculate_premium_rate(
        self,
        material_category: str,
        tier: str,
        country: str
    ) -> float:
        """
        상승률 계산

        공식: (RE100_전환가격 / 기본단가) × 100

        Args:
            material_category: 자재품목
            tier: Tier
            country: 국가명

        Returns:
            float: 상승률 (%)
        """
        conversion_price = self.calculate_re100_conversion_price(material_category, tier, country)
        basic_cost = self._get_basic_cost(material_category, tier)

        if basic_cost == 0:
            if self.debug_mode:
                print(f"⚠️ 기본단가가 0입니다: {material_category} {tier}")
            return 0.0

        rate = (conversion_price / basic_cost) * 100

        if self.debug_mode:
            print(f"상승률: {material_category} {tier} {country}")
            print(f"  RE100 전환가격: ${conversion_price:.6f}/kg")
            print(f"  기본단가: ${basic_cost:.2f}/kg")
            print(f"  상승률: {rate:.2f}%")

        return rate

    def calculate_case_premium(
        self,
        material_name: str,
        material_category: str,
        quantity: float,
        country: str,
        case_config: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        특정 Case의 프리미엄 비용 계산

        Args:
            material_name: 자재명
            material_category: 자재품목 (BRM 형식)
            quantity: 제품총소요량(kg)
            country: 국가명
            case_config: {"tier1_re": 1.0, "tier2_re": 0.5} 형태의 RE 적용률

        Returns:
            Dict: {
                "tier1_premium": float,
                "tier2_premium": float,
                "total_premium": float,
                "premium_rate": float
            }
        """
        # 자재품목 매핑
        opt_material = self._map_material_category(material_category)

        # Tier별 프리미엄 계산
        tier1_premium = 0.0
        tier2_premium = 0.0

        # Tier1
        if 'tier1_re' in case_config and case_config['tier1_re'] > 0:
            tier1_conversion = self.calculate_re100_conversion_price(opt_material, "Tier1", country)
            tier1_premium = quantity * tier1_conversion * case_config['tier1_re']

        # Tier2
        if 'tier2_re' in case_config and case_config['tier2_re'] > 0:
            tier2_conversion = self.calculate_re100_conversion_price(opt_material, "Tier2", country)
            tier2_premium = quantity * tier2_conversion * case_config['tier2_re']

        # 총 프리미엄
        total_premium = tier1_premium + tier2_premium

        # 기본 비용 (참고용)
        tier1_basic = self._get_basic_cost(opt_material, "Tier1") * quantity
        tier2_basic = self._get_basic_cost(opt_material, "Tier2") * quantity
        total_basic = tier1_basic + tier2_basic

        # 상승률
        if total_basic > 0:
            premium_rate = (total_premium / total_basic) * 100
        else:
            premium_rate = 0.0

        return {
            "material_name": material_name,
            "material_category": material_category,
            "opt_material": opt_material,
            "quantity": quantity,
            "country": country,
            "tier1_premium": tier1_premium,
            "tier2_premium": tier2_premium,
            "total_premium": total_premium,
            "total_basic_cost": total_basic,
            "premium_rate": premium_rate
        }

    def calculate_scenario_premiums(
        self,
        scenario_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        시나리오 전체 자재의 Case별 프리미엄 계산

        Args:
            scenario_df: 시나리오 테이블 (Tier_RE_case 컬럼 포함)
            original_df: BRM original 테이블 (지역 정보 포함)

        Returns:
            DataFrame: 자재별, Case별 프리미엄 비용 테이블
        """
        if self.debug_mode:
            print("\n===== 시나리오 프리미엄 계산 시작 =====")
            print(f"시나리오 자재 수: {len(scenario_df)}")
            print(f"원본 테이블 자재 수: {len(original_df)}")

        # Case 컬럼 추출
        case_columns = [col for col in scenario_df.columns if 'Tier' in col and 'RE_case' in col]

        # Case 컬럼이 없으면 에러
        if not case_columns:
            raise ValueError(
                "시나리오 데이터에 RE100 Case 컬럼이 없습니다.\n"
                "필요한 컬럼: Tier1_RE_case1, Tier2_RE_case1, Tier1_RE_case2, ...\n"
                "→ '시나리오 설정' 페이지에서 RE100 케이스를 추가하세요."
            )

        # Case 번호 추출 (Tier1_RE_case1, Tier1_RE_case2, ...)
        case_numbers = set()
        for col in case_columns:
            if 'case' in col:
                case_num = col.split('case')[1]
                case_numbers.add(int(case_num))
        case_numbers = sorted(list(case_numbers))

        if self.debug_mode:
            print(f"Case 개수: {len(case_numbers)}")
            print(f"Case 번호: {case_numbers}")

        # 결과 저장 리스트
        results = []

        # 자재별 계산
        for idx, scenario_row in scenario_df.iterrows():
            material_name = scenario_row.get('자재명')
            material_category = scenario_row.get('자재품목')
            quantity = scenario_row.get('제품총소요량(kg)', 0)

            # 저감활동 적용 여부 확인
            apply_reduction = scenario_row.get('저감활동_적용여부', 0)
            if apply_reduction != 1:
                continue  # 적용하지 않는 자재는 스킵

            # 국가 정보 찾기 (original_df에서)
            country = self._get_material_country(material_name, original_df)

            # 기본 정보
            result_row = {
                '자재명': material_name,
                '자재품목': material_category,
                '지역(국가)': country,
                '제품총소요량(kg)': quantity
            }

            # Case별 프리미엄 계산
            for case_num in case_numbers:
                # Case 설정 추출
                tier1_col = f'Tier1_RE_case{case_num}'
                tier2_col = f'Tier2_RE_case{case_num}'

                tier1_re = self._parse_percentage(scenario_row.get(tier1_col, '0%'))
                tier2_re = self._parse_percentage(scenario_row.get(tier2_col, '0%'))

                case_config = {
                    'tier1_re': tier1_re,
                    'tier2_re': tier2_re
                }

                # 프리미엄 계산
                premium_result = self.calculate_case_premium(
                    material_name,
                    material_category,
                    quantity,
                    country,
                    case_config
                )

                # 결과 저장
                result_row[f'case{case_num}_tier1_re(%)'] = tier1_re * 100
                result_row[f'case{case_num}_tier2_re(%)'] = tier2_re * 100
                result_row[f'case{case_num}_premium($)'] = premium_result['total_premium']
                result_row[f'case{case_num}_rate(%)'] = premium_result['premium_rate']

            results.append(result_row)

        # DataFrame 생성
        result_df = pd.DataFrame(results)

        if self.debug_mode:
            print(f"\n계산 완료: {len(result_df)}개 자재")
            print(f"총 프리미엄 비용:")
            for case_num in case_numbers:
                total = result_df[f'case{case_num}_premium($)'].sum()
                print(f"  Case{case_num}: ${total:,.2f}")

        return result_df

    def _get_material_country(self, material_name: str, original_df: pd.DataFrame) -> str:
        """
        자재의 국가 정보 추출

        Args:
            material_name: 자재명
            original_df: BRM original 테이블

        Returns:
            str: 국가명
        """
        # 자재명으로 필터링
        material_rows = original_df[original_df['자재명'] == material_name]

        if len(material_rows) == 0:
            if self.debug_mode:
                print(f"⚠️ 자재를 찾을 수 없습니다: {material_name}")
            return "미분류"

        # 첫 번째 행의 지역 정보
        region_code = material_rows.iloc[0].get('지역', '')

        # 지역 코드 → 국가명 변환
        country = self._convert_region_to_country(region_code)

        return country

    def _parse_percentage(self, value: Any) -> float:
        """
        퍼센트 문자열을 비율로 변환

        Args:
            value: "50%" 또는 50 또는 0.5

        Returns:
            float: 0.0 ~ 1.0 비율
        """
        if isinstance(value, str):
            # "50%" → 0.5
            value = value.replace('%', '').strip()
            try:
                return float(value) / 100.0
            except:
                return 0.0
        elif isinstance(value, (int, float)):
            # 100 이상이면 퍼센트로 간주
            if value >= 1.0:
                return value / 100.0
            else:
                return value
        return 0.0

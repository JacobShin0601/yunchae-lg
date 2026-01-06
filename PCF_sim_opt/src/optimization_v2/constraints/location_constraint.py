"""
위치(국가) 제약조건

특정 자재의 소싱 국가를 제한하는 제약조건입니다.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class LocationConstraint(ConstraintBase):
    """
    위치 제약조건 클래스

    특정 자재는 특정 국가에서만 소싱하도록 제한합니다.
    예: Cu Foil은 한국과 일본에서만 조달
    """

    def __init__(self, national_code_mapping: Optional[Dict[str, str]] = None):
        """
        위치 제약조건 초기화

        Args:
            national_code_mapping: 국가 코드 매핑 딕셔너리
                예: {'KR': '한국', 'JP': '일본', 'CN': '중국'}
        """
        super().__init__(
            name="location_constraint",
            description="자재별 소싱 국가 제한 제약"
        )
        self.national_code_mapping = national_code_mapping or {}
        self.location_rules: List[Dict[str, Any]] = []

    def set_national_code_mapping(self, mapping: Dict[str, str]) -> None:
        """
        국가 코드 매핑 설정

        Args:
            mapping: {code: country_name} 매핑
        """
        self.national_code_mapping = mapping
        print(f"✅ 국가 코드 매핑 설정: {len(mapping)}개 국가")

    def add_location_rule(
        self,
        material_name: str,
        allowed_countries: List[str],
        rule_type: str = 'whitelist'
    ) -> bool:
        """
        자재별 위치 규칙 추가

        Args:
            material_name: 대상 자재명
            allowed_countries: 허용 국가 리스트
            rule_type: 규칙 유형
                - 'whitelist': 화이트리스트 (허용 목록만 가능)
                - 'blacklist': 블랙리스트 (금지 목록 제외 모두 가능)

        Returns:
            성공 여부
        """
        if not allowed_countries:
            print("⚠️  국가 리스트가 비어있습니다.")
            return False

        if rule_type not in ['whitelist', 'blacklist']:
            print(f"⚠️  알 수 없는 규칙 유형: {rule_type}")
            return False

        rule = {
            'material': material_name,
            'countries': allowed_countries,
            'type': rule_type
        }

        self.location_rules.append(rule)

        rule_desc = "허용" if rule_type == 'whitelist' else "금지"
        print(f"✅ 위치 규칙 추가: {material_name} - {rule_desc} {len(allowed_countries)}개국")

        return True

    def remove_location_rule(self, material_name: str) -> int:
        """
        자재의 위치 규칙 제거

        Args:
            material_name: 대상 자재명

        Returns:
            제거된 규칙 개수
        """
        removed_count = 0
        new_rules = []

        for rule in self.location_rules:
            if rule['material'] == material_name:
                removed_count += 1
            else:
                new_rules.append(rule)

        self.location_rules = new_rules

        if removed_count > 0:
            print(f"✅ {material_name}의 위치 규칙 {removed_count}개 제거됨")
        else:
            print(f"⚠️  제거할 규칙을 찾지 못했습니다: {material_name}")

        return removed_count

    def get_rules_for_material(self, material_name: str) -> List[Dict[str, Any]]:
        """
        특정 자재에 대한 위치 규칙 조회

        Args:
            material_name: 자재명

        Returns:
            규칙 리스트
        """
        return [rule for rule in self.location_rules if rule['material'] == material_name]

    def get_allowed_countries(self, material_name: str, all_countries: Set[str]) -> Set[str]:
        """
        자재에 대해 허용된 국가 집합 반환

        Args:
            material_name: 자재명
            all_countries: 전체 가능한 국가 집합

        Returns:
            허용된 국가 집합
        """
        rules = self.get_rules_for_material(material_name)

        if not rules:
            # 규칙이 없으면 모든 국가 허용
            return all_countries

        allowed = all_countries.copy()

        for rule in rules:
            rule_countries = set(rule['countries'])
            rule_type = rule['type']

            if rule_type == 'whitelist':
                # 화이트리스트: 교집합
                allowed = allowed.intersection(rule_countries)
            elif rule_type == 'blacklist':
                # 블랙리스트: 차집합
                allowed = allowed.difference(rule_countries)

        return allowed

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            (is_valid, message)
        """
        # 규칙이 없으면 유효하지 않음
        if not self.location_rules:
            return False, "위치 규칙이 설정되지 않았습니다."

        # 각 규칙의 국가 리스트 검증
        for i, rule in enumerate(self.location_rules):
            if not rule.get('countries'):
                return False, f"규칙 #{i+1}: 국가 리스트가 비어있습니다."

            if rule.get('type') not in ['whitelist', 'blacklist']:
                return False, f"규칙 #{i+1}: 유효하지 않은 규칙 유형"

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
        if scenario_df is None:
            return False, "scenario_df가 없습니다."

        # 규칙에 지정된 자재가 데이터에 존재하는지 확인
        available_materials = set(scenario_df['자재명'].unique())

        for rule in self.location_rules:
            material = rule['material']
            if material not in available_materials:
                return False, f"자재 '{material}'이(가) 데이터에 없습니다."

        # 데이터에 국가 정보가 있는지 확인
        if '지역' not in scenario_df.columns and 'country' not in scenario_df.columns:
            return False, "국가 정보 컬럼('지역' 또는 'country')이 없습니다."

        return True, "실현 가능"

    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        LocationConstraint는 자재의 현재 소싱 국가가 허용 목록에 있는지 검증하고,
        허용되지 않는 국가에서 소싱되는 자재는 최적화 변수를 초기값(0)으로 고정합니다.

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터
        """
        scenario_df = data.get('scenario_df')
        original_df = data.get('original_df')
        material_classification = data.get('material_classification')

        # 데이터에서 사용 가능한 모든 국가 추출
        country_column = '지역' if '지역' in scenario_df.columns else 'country'
        all_countries = set(scenario_df[country_column].unique())

        print(f"    사용 가능 국가: {len(all_countries)}개")

        # 제약조건이 적용된 자재 수 카운트
        applied_count = 0
        restricted_count = 0

        for rule in self.location_rules:
            material = rule['material']
            rule_type = rule['type']
            rule_countries = set(rule['countries'])

            # 자재가 모델에 있는지 확인
            if material not in model.materials:
                print(f"  ⚠️  자재 '{material}'이(가) 모델에 없어 규칙 스킵")
                continue

            # 허용된 국가 계산
            allowed_countries = self.get_allowed_countries(material, all_countries)

            if not allowed_countries:
                print(f"  ⚠️  {material}: 허용된 국가가 없습니다!")
                continue

            # 자재의 현재 소싱 국가 확인
            current_country = self._get_material_current_country(
                material, scenario_df, original_df, country_column
            )

            # 현재 국가가 허용 목록에 있는지 확인
            is_allowed = current_country in allowed_countries

            rule_desc = "허용" if rule_type == 'whitelist' else "제외"
            print(f"    • {material}: {rule_desc} {len(rule_countries)}개국 → "
                  f"최종 허용 {len(allowed_countries)}개국")
            print(f"      현재 국가: {current_country} → ", end="")

            if is_allowed:
                print("✅ 허용됨")
                applied_count += 1
            else:
                print("❌ 허용되지 않음")

                # 허용되지 않는 국가의 자재는 최적화 변수를 0으로 고정
                # (저감 활동을 적용하지 않음)
                material_type = material_classification[material]['type']

                if material_type == 'Formula':
                    # Formula 자재: RE 비율을 0으로 고정
                    model.tier1_re[material].fix(0)
                    model.tier2_re[material].fix(0)
                    constraint_name = f'location_restrict_formula_{material.replace(" ", "_")}'
                    print(f"      → Tier1/Tier2 RE 비율을 0으로 고정 (제약조건: {constraint_name})")

                elif material_type == 'Ni-Co-Li':
                    # Ni-Co-Li 자재: 재활용/저탄소를 0으로, 버진을 1로 고정
                    model.recycle_ratio[material].fix(0)
                    model.low_carbon_ratio[material].fix(0)
                    model.virgin_ratio[material].fix(1)
                    constraint_name = f'location_restrict_nicoli_{material.replace(" ", "_")}'
                    print(f"      → 재활용/저탄소 비율을 0으로 고정 (제약조건: {constraint_name})")

                else:
                    # 일반 자재: 변경 불가
                    print(f"      → 일반 자재로 변경 불가")

                restricted_count += 1

        if applied_count > 0:
            print(f"\n    ✅ 위치 제약조건 적용 완료: {applied_count}개 자재")
        if restricted_count > 0:
            print(f"    ⚠️  허용되지 않는 국가의 자재: {restricted_count}개 (변수 고정됨)")

    def _get_material_current_country(
        self,
        material_name: str,
        scenario_df,
        original_df,
        country_column: str
    ) -> str:
        """
        자재의 현재 소싱 국가 조회

        Args:
            material_name: 자재명
            scenario_df: 시나리오 DataFrame
            original_df: 원본 DataFrame
            country_column: 국가 컬럼명

        Returns:
            국가명
        """
        # scenario_df에서 먼저 찾기
        material_rows = scenario_df[scenario_df['자재명'] == material_name]
        if len(material_rows) > 0:
            country = material_rows.iloc[0].get(country_column, '미분류')
            if country and country != '미분류':
                return str(country)

        # original_df에서 찾기
        if original_df is not None:
            material_rows = original_df[original_df['자재명'] == material_name]
            if len(material_rows) > 0:
                country = material_rows.iloc[0].get(country_column, '미분류')
                if country and country != '미분류':
                    # 국가 코드 변환 (필요시)
                    if country in self.national_code_mapping:
                        return self.national_code_mapping[country]
                    return str(country)

        return '미분류'

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        return {
            'type': 'location_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'national_code_mapping': self.national_code_mapping,
            'location_rules': self.location_rules
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        self.name = config.get('name', 'location_constraint')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.national_code_mapping = config.get('national_code_mapping', {})
        self.location_rules = config.get('location_rules', [])

    def get_summary(self) -> str:
        """
        제약조건 요약

        Returns:
            요약 문자열
        """
        base_summary = super().get_summary()
        rule_count = len(self.location_rules)
        material_count = len(set(r['material'] for r in self.location_rules))

        return f"{base_summary}\n  🌍 규칙 {rule_count}개 (자재 {material_count}개)"

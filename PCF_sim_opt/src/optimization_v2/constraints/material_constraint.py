"""
자재 관리 제약조건

자재별로 감축 활동 유형, 재활용/저탄소 비율 등을 제한하는 제약조건입니다.
"""

from typing import Dict, Any, List, Tuple, Optional
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class MaterialManagementConstraint(ConstraintBase):
    """
    자재 관리 제약조건 클래스

    지원하는 규칙 유형:
    - exclude_low_carbon: 저탄소메탈 제외
    - exclude_recycle: 재활용재 제외
    - virgin_only: 신재만 사용 (재활용, 저탄소 모두 제외)
    - recycle_only: 재활용재만 사용 (100%)
    - low_carbon_only: 저탄소메탈만 사용 (100%)
    - force_ratio_range: 재활용/저탄소 비율 범위 강제
    - force_element_ratio_range: 원소별(Ni/Co/Li) 재활용/저탄소 비율 범위 강제
    - limit_activities: 감축 활동 유형 개수 제한 (미구현)
    - regional_preference: 특정 지역 선호 (미구현)
    """

    def __init__(self):
        """자재 관리 제약조건 초기화"""
        super().__init__(
            name="material_management",
            description="자재별 감축 활동 및 비율 관리 제약"
        )
        self.material_rules: List[Dict[str, Any]] = []
        self.material_groups: Dict[str, List[str]] = {}  # 그룹명 -> 자재 리스트

    def add_rule(
        self,
        rule_type: str,
        material_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        자재별 규칙 추가

        Args:
            rule_type: 규칙 유형
                - 'exclude_low_carbon': 저탄소메탈 제외
                - 'exclude_recycle': 재활용재 제외
                - 'virgin_only': 신재만 사용
                - 'recycle_only': 재활용재만 사용 (100%)
                - 'low_carbon_only': 저탄소메탈만 사용 (100%)
                - 'force_ratio_range': 비율 범위 강제
                - 'limit_activities': 감축 활동 최대 개수 제한
                - 'regional_preference': 지역 선호도
            material_name: 대상 자재명
            params: 규칙별 파라미터
                - exclude_low_carbon, exclude_recycle, virgin_only,
                  recycle_only, low_carbon_only: {}
                - limit_activities: {'max_count': int}
                - force_ratio_range: {
                    'recycle_min': float, 'recycle_max': float,
                    'low_carbon_min': float, 'low_carbon_max': float
                  }
                - force_element_ratio_range: {
                    'element': str ('Ni', 'Co', 'Li'),
                    'recycle_min': float, 'recycle_max': float,
                    'low_carbon_min': float, 'low_carbon_max': float
                  }
                - regional_preference: {'preferred_regions': List[str]}

        Returns:
            성공 여부
        """
        if params is None:
            params = {}

        # 규칙 유효성 검증
        valid_types = [
            'exclude_low_carbon',      # 저탄소메탈 제외
            'exclude_recycle',         # 재활용재 제외 (NEW)
            'virgin_only',             # 신재만 사용 (NEW)
            'recycle_only',            # 재활용재만 사용 (NEW)
            'low_carbon_only',         # 저탄소메탈만 사용 (NEW)
            'limit_activities',        # 활동 유형 제한
            'force_ratio_range',       # 비율 범위 강제
            'force_element_ratio_range',  # 원소별 비율 범위 강제 (NEW)
            'regional_preference'      # 지역 선호도
        ]
        if rule_type not in valid_types:
            print(f"⚠️  알 수 없는 규칙 유형: {rule_type}")
            return False

        rule = {
            'type': rule_type,
            'material': material_name,
            'params': params
        }

        self.material_rules.append(rule)
        print(f"✅ 규칙 추가: {material_name} - {rule_type}")
        return True

    def remove_rule(self, material_name: str, rule_type: Optional[str] = None) -> int:
        """
        자재의 규칙 제거

        Args:
            material_name: 대상 자재명
            rule_type: 규칙 유형 (None이면 해당 자재의 모든 규칙 제거)

        Returns:
            제거된 규칙 개수
        """
        removed_count = 0
        new_rules = []

        for rule in self.material_rules:
            # 자재 이름과 규칙 유형 모두 일치하는 경우만 제거
            should_remove = rule['material'] == material_name
            if rule_type:
                should_remove = should_remove and rule['type'] == rule_type

            if should_remove:
                removed_count += 1
            else:
                new_rules.append(rule)

        self.material_rules = new_rules

        if removed_count > 0:
            print(f"✅ {material_name}의 규칙 {removed_count}개 제거됨")
        else:
            print(f"⚠️  제거할 규칙을 찾지 못했습니다: {material_name}")

        return removed_count

    def get_rules_for_material(self, material_name: str) -> List[Dict[str, Any]]:
        """
        특정 자재에 대한 규칙 목록 조회

        Args:
            material_name: 자재명

        Returns:
            규칙 리스트
        """
        return [rule for rule in self.material_rules if rule['material'] == material_name]

    # ========== 그룹 관리 메서드 ==========

    def add_material_group(
        self,
        group_name: str,
        materials: List[str],
        description: str = ""
    ) -> bool:
        """
        자재 그룹 정의

        Args:
            group_name: 그룹 이름
            materials: 그룹에 포함할 자재 리스트
            description: 그룹 설명 (선택)

        Returns:
            성공 여부
        """
        if not materials:
            print(f"⚠️  자재 리스트가 비어있습니다.")
            return False

        self.material_groups[group_name] = materials
        print(f"✅ 자재 그룹 '{group_name}' 생성: {len(materials)}개 자재")
        if description:
            print(f"   설명: {description}")
        return True

    def add_rule_to_group(
        self,
        group_name: str,
        rule_type: str,
        params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        그룹의 모든 자재에 규칙 적용

        Args:
            group_name: 그룹 이름
            rule_type: 규칙 유형
            params: 규칙 파라미터

        Returns:
            적용된 자재 개수
        """
        if group_name not in self.material_groups:
            print(f"⚠️  그룹 '{group_name}'을 찾을 수 없습니다.")
            return 0

        materials = self.material_groups[group_name]
        count = 0

        for material in materials:
            if self.add_rule(rule_type, material, params):
                count += 1

        print(f"✅ 그룹 '{group_name}': {count}개 자재에 규칙 적용됨")
        return count

    def add_rule_to_materials(
        self,
        rule_type: str,
        materials: List[str],
        params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        여러 자재에 동일한 규칙 일괄 적용

        Args:
            rule_type: 규칙 유형
            materials: 자재 리스트
            params: 규칙 파라미터

        Returns:
            적용된 자재 개수
        """
        if not materials:
            print(f"⚠️  자재 리스트가 비어있습니다.")
            return 0

        count = 0
        for material in materials:
            if self.add_rule(rule_type, material, params):
                count += 1

        print(f"✅ {count}개 자재에 규칙 일괄 적용됨")
        return count

    def create_predefined_groups(self, scenario_df) -> Dict[str, List[str]]:
        """
        사전정의 그룹 자동 생성 (자재품목별, 원소별)

        Args:
            scenario_df: 시나리오 DataFrame

        Returns:
            생성된 그룹 딕셔너리
        """
        import pandas as pd

        if not isinstance(scenario_df, pd.DataFrame):
            print("⚠️  scenario_df가 DataFrame이 아닙니다.")
            return {}

        groups = {}

        # 1. 자재품목별 그룹
        if '자재품목' in scenario_df.columns:
            for category in scenario_df['자재품목'].unique():
                if pd.notna(category):
                    materials = scenario_df[scenario_df['자재품목'] == category]['자재명'].tolist()
                    group_name = f"그룹_{category}"
                    groups[group_name] = materials
                    self.material_groups[group_name] = materials

        # 2. 원소별 그룹 (이름에 원소가 포함된 경우)
        for element in ['Ni', 'Co', 'Li', 'Al', 'Cu', 'Fe', 'Mn']:
            materials = scenario_df[
                scenario_df['자재명'].str.contains(element, case=False, na=False)
            ]['자재명'].tolist()

            if materials:
                group_name = f"그룹_{element}_포함"
                groups[group_name] = materials
                self.material_groups[group_name] = materials

        # 3. 저감활동 적용 여부별 그룹
        if '저감활동_적용여부' in scenario_df.columns:
            # 적용 대상 자재
            applied_materials = scenario_df[
                scenario_df['저감활동_적용여부'] == 1
            ]['자재명'].tolist()

            if applied_materials:
                group_name = "그룹_저감활동_적용"
                groups[group_name] = applied_materials
                self.material_groups[group_name] = applied_materials

            # 미적용 자재
            not_applied_materials = scenario_df[
                scenario_df['저감활동_적용여부'] == 0
            ]['자재명'].tolist()

            if not_applied_materials:
                group_name = "그룹_저감활동_미적용"
                groups[group_name] = not_applied_materials
                self.material_groups[group_name] = not_applied_materials

        print(f"✅ 사전정의 그룹 {len(groups)}개 생성됨")
        for group_name, materials in groups.items():
            print(f"   • {group_name}: {len(materials)}개 자재")

        return groups

    def list_groups(self) -> Dict[str, int]:
        """
        현재 정의된 그룹 목록 조회

        Returns:
            {그룹명: 자재 개수} 딕셔너리
        """
        return {name: len(materials) for name, materials in self.material_groups.items()}

    def get_group_materials(self, group_name: str) -> List[str]:
        """
        특정 그룹의 자재 목록 조회

        Args:
            group_name: 그룹 이름

        Returns:
            자재 리스트
        """
        return self.material_groups.get(group_name, [])

    # ========== 기존 메서드 ==========

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            (is_valid, message)
        """
        # 규칙이 없으면 유효하지 않음
        if not self.material_rules:
            return False, "규칙이 설정되지 않았습니다."

        # 각 규칙의 파라미터 검증
        for i, rule in enumerate(self.material_rules):
            rule_type = rule['type']
            params = rule['params']

            if rule_type == 'limit_activities':
                if 'max_count' not in params:
                    return False, f"규칙 #{i+1}: limit_activities에 max_count 필요"
                if params['max_count'] < 1:
                    return False, f"규칙 #{i+1}: max_count는 1 이상이어야 함"

            elif rule_type == 'force_ratio_range':
                required = ['recycle_min', 'recycle_max', 'low_carbon_min', 'low_carbon_max']
                missing = [p for p in required if p not in params]
                if missing:
                    return False, f"규칙 #{i+1}: 누락된 파라미터 - {missing}"

                # 범위 검증
                if not (0 <= params['recycle_min'] <= params['recycle_max'] <= 1):
                    return False, f"규칙 #{i+1}: 재활용 비율 범위 오류"
                if not (0 <= params['low_carbon_min'] <= params['low_carbon_max'] <= 1):
                    return False, f"규칙 #{i+1}: 저탄소 비율 범위 오류"

            elif rule_type == 'force_element_ratio_range':
                required = ['element', 'recycle_min', 'recycle_max', 'low_carbon_min', 'low_carbon_max']
                missing = [p for p in required if p not in params]
                if missing:
                    return False, f"규칙 #{i+1}: 누락된 파라미터 - {missing}"

                # 원소 검증
                if params['element'] not in ['Ni', 'Co', 'Li']:
                    return False, f"규칙 #{i+1}: 잘못된 원소 - {params['element']}"

                # 범위 검증
                if not (0 <= params['recycle_min'] <= params['recycle_max'] <= 1):
                    return False, f"규칙 #{i+1}: 재활용 비율 범위 오류"
                if not (0 <= params['low_carbon_min'] <= params['low_carbon_max'] <= 1):
                    return False, f"규칙 #{i+1}: 저탄소 비율 범위 오류"

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

        for rule in self.material_rules:
            material = rule['material']
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
        print(f"\n{'='*70}")
        print(f"🔧 MaterialManagementConstraint 적용: {self.name}")
        print(f"{'='*70}")
        print(f"총 규칙: {len(self.material_rules)}개")
        print(f"활성화: {self.enabled}")

        if not self.material_rules:
            print("⚠️  규칙이 없습니다 - 제약이 적용되지 않습니다")
            print(f"{'='*70}\n")
            return

        applied_count = 0
        skipped_count = 0
        skipped_reasons = []

        for rule_idx, rule in enumerate(self.material_rules):
            material = rule['material']
            rule_type = rule['type']
            params = rule['params']

            # force_element_ratio_range는 원소 자재에 적용되므로 material 체크 스킵
            if rule_type != 'force_element_ratio_range':
                # 자재가 모델에 있는지 확인
                if material not in model.materials:
                    print(f"  ⚠️  자재 '{material}'이(가) 모델에 없어 규칙 스킵")
                    continue

            if rule_type == 'exclude_low_carbon':
                # 저탄소메탈 비율 = 0으로 강제
                constraint_name = f'exclude_low_carbon_{material}_r{rule_idx}'.replace(' ', '_')

                if hasattr(model, 'low_carbon_ratio'):
                    model.add_component(
                        constraint_name,
                        pyo.Constraint(expr=model.low_carbon_ratio[material] == 0)
                    )
                    print(f"    • {material}: 저탄소메탈 제외")

            elif rule_type == 'exclude_recycle':
                # 재활용재 비율 = 0으로 강제
                constraint_name = f'exclude_recycle_{material}_r{rule_idx}'.replace(' ', '_')

                if hasattr(model, 'recycle_ratio'):
                    model.add_component(
                        constraint_name,
                        pyo.Constraint(expr=model.recycle_ratio[material] == 0)
                    )
                    print(f"    • {material}: 재활용재 제외")

            elif rule_type == 'virgin_only':
                # 신재만 사용 (재활용 = 0, 저탄소 = 0)
                if hasattr(model, 'recycle_ratio') and hasattr(model, 'low_carbon_ratio'):
                    model.add_component(
                        f'virgin_only_recycle_{material}_r{rule_idx}'.replace(' ', '_'),
                        pyo.Constraint(expr=model.recycle_ratio[material] == 0)
                    )
                    model.add_component(
                        f'virgin_only_low_carbon_{material}_r{rule_idx}'.replace(' ', '_'),
                        pyo.Constraint(expr=model.low_carbon_ratio[material] == 0)
                    )
                    print(f"    • {material}: 신재만 사용")

            elif rule_type == 'recycle_only':
                # 재활용재만 사용 (재활용 = 1)
                constraint_name = f'recycle_only_{material}_r{rule_idx}'.replace(' ', '_')

                if hasattr(model, 'recycle_ratio'):
                    model.add_component(
                        constraint_name,
                        pyo.Constraint(expr=model.recycle_ratio[material] == 1)
                    )
                    print(f"    • {material}: 재활용재만 사용 (100%)")

            elif rule_type == 'low_carbon_only':
                # 저탄소메탈만 사용 (저탄소 = 1)
                constraint_name = f'low_carbon_only_{material}_r{rule_idx}'.replace(' ', '_')

                if hasattr(model, 'low_carbon_ratio'):
                    model.add_component(
                        constraint_name,
                        pyo.Constraint(expr=model.low_carbon_ratio[material] == 1)
                    )
                    print(f"    • {material}: 저탄소메탈만 사용 (100%)")

            elif rule_type == 'limit_activities':
                # 활동 유형 개수 제한 (이진 변수 사용)
                max_count = params['max_count']
                # TODO: 활동 유형 이진 변수가 정의되면 구현
                print(f"    • {material}: 활동 제한 {max_count}개 (미구현)")

            elif rule_type == 'force_ratio_range':
                # 비율 범위 강제
                recycle_min = params['recycle_min']
                recycle_max = params['recycle_max']
                low_carbon_min = params['low_carbon_min']
                low_carbon_max = params['low_carbon_max']

                if hasattr(model, 'recycle_ratio'):
                    # 재활용 비율 범위
                    model.add_component(
                        f'recycle_min_{material}_r{rule_idx}'.replace(' ', '_'),
                        pyo.Constraint(expr=model.recycle_ratio[material] >= recycle_min)
                    )
                    model.add_component(
                        f'recycle_max_{material}_r{rule_idx}'.replace(' ', '_'),
                        pyo.Constraint(expr=model.recycle_ratio[material] <= recycle_max)
                    )

                if hasattr(model, 'low_carbon_ratio'):
                    # 저탄소 비율 범위
                    model.add_component(
                        f'low_carbon_min_{material}_r{rule_idx}'.replace(' ', '_'),
                        pyo.Constraint(expr=model.low_carbon_ratio[material] >= low_carbon_min)
                    )
                    model.add_component(
                        f'low_carbon_max_{material}_r{rule_idx}'.replace(' ', '_'),
                        pyo.Constraint(expr=model.low_carbon_ratio[material] <= low_carbon_max)
                    )

                print(f"    • {material}: 재활용 [{recycle_min:.1%}~{recycle_max:.1%}], "
                      f"저탄소 [{low_carbon_min:.1%}~{low_carbon_max:.1%}]")

            elif rule_type == 'force_element_ratio_range':
                # 원소별 비율 범위 강제 (양극재 전용 변수 사용)
                element = params['element']
                recycle_min = params['recycle_min']
                recycle_max = params['recycle_max']
                low_carbon_min = params['low_carbon_min']
                low_carbon_max = params['low_carbon_max']

                # 양극재 원소별 변수가 있는지 확인
                if hasattr(model, 'elements') and element in model.elements:
                    if hasattr(model, 'element_recycle_ratio'):
                        # 재활용 비율 범위 (rule_idx로 고유 이름 생성)
                        model.add_component(
                            f'cathode_element_recycle_min_{element}_r{rule_idx}'.replace(' ', '_'),
                            pyo.Constraint(expr=model.element_recycle_ratio[element] >= recycle_min)
                        )
                        model.add_component(
                            f'cathode_element_recycle_max_{element}_r{rule_idx}'.replace(' ', '_'),
                            pyo.Constraint(expr=model.element_recycle_ratio[element] <= recycle_max)
                        )

                    if hasattr(model, 'element_low_carb_ratio'):
                        # 저탄소 비율 범위 (rule_idx로 고유 이름 생성)
                        model.add_component(
                            f'cathode_element_low_carbon_min_{element}_r{rule_idx}'.replace(' ', '_'),
                            pyo.Constraint(expr=model.element_low_carb_ratio[element] >= low_carbon_min)
                        )
                        model.add_component(
                            f'cathode_element_low_carbon_max_{element}_r{rule_idx}'.replace(' ', '_'),
                            pyo.Constraint(expr=model.element_low_carb_ratio[element] <= low_carbon_max)
                        )

                    applied_count += 1
                    print(f"  ✅ [{rule_idx+1}] {element}: "
                          f"재활용 [{recycle_min*100:.1f}%~{recycle_max*100:.1f}%], "
                          f"저탄소 [{low_carbon_min*100:.1f}%~{low_carbon_max*100:.1f}%]")
                else:
                    skipped_count += 1
                    reason = f"[{rule_idx+1}] {element}: elements Set 없거나 원소 미포함"
                    skipped_reasons.append(reason)
                    print(f"  ⚠️  {reason}")

            elif rule_type == 'regional_preference':
                # 지역 선호도 (위치 제약과 연계)
                print(f"    • {material}: 지역 선호도 (미구현)")

        # 적용 결과 요약
        print(f"\n📊 적용 결과:")
        print(f"   ✅ 성공: {applied_count}개")
        print(f"   ⚠️  스킵: {skipped_count}개")

        if skipped_reasons:
            print(f"\n⚠️  스킵된 규칙 상세:")
            for reason in skipped_reasons:
                print(f"   • {reason}")

        print(f"{'='*70}\n")

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        return {
            'type': 'material_management_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'material_rules': self.material_rules
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        self.name = config.get('name', 'material_management')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.material_rules = config.get('material_rules', [])

    def get_summary(self) -> str:
        """
        제약조건 요약

        Returns:
            요약 문자열
        """
        base_summary = super().get_summary()
        rule_count = len(self.material_rules)
        material_count = len(set(r['material'] for r in self.material_rules))

        return f"{base_summary}\n  📋 규칙 {rule_count}개 (자재 {material_count}개)"

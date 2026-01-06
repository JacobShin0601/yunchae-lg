"""
기능 옵션 제약조건

재활용재, 저탄소메탈, 사이트 변경 등의 기능을 활성화/비활성화하는 제약조건입니다.
"""

from typing import Dict, Any, Tuple
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class RecyclingOptionConstraint(ConstraintBase):
    """
    재활용재 사용 옵션 제약조건

    - 활성화: 재활용재 비율을 최적화 변수로 사용 (0~100% 자유)
    - 비활성화: 재활용재 비율을 0%로 고정
    """

    def __init__(self, enabled: bool = True):
        """
        재활용재 옵션 제약조건 초기화

        Args:
            enabled: True면 재활용재 사용 허용, False면 금지
        """
        super().__init__(
            name="recycling_option",
            description="재활용재 사용 활성화/비활성화",
            enabled=enabled
        )

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """설정 유효성 검증 (특별한 설정 없음)"""
        return True, "유효한 설정입니다."

    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        비활성화된 경우:
        - 모든 자재의 recycle_ratio를 0으로 고정
        - 양극재 원소의 element_recycle_ratio를 0으로 고정
        """
        if not self.enabled:
            print(f"   🚫 재활용재 사용 금지 - recycle_ratio를 0으로 고정")

            # 일반 자재의 재활용 비율 = 0
            def no_recycle_rule(m, material):
                return m.recycle_ratio[material] == 0

            model.no_recycle_constraint = pyo.Constraint(
                model.materials,
                rule=no_recycle_rule,
                doc="재활용재 사용 금지"
            )

            # 양극재 원소의 재활용 비율 = 0
            if hasattr(model, 'elements'):
                def no_element_recycle_rule(m, element):
                    return m.element_recycle_ratio[element] == 0

                model.no_element_recycle_constraint = pyo.Constraint(
                    model.elements,
                    rule=no_element_recycle_rule,
                    doc="양극재 원소 재활용 금지"
                )
        else:
            print(f"   ✅ 재활용재 사용 허용 - 최적화 변수로 사용")

    def check_feasibility(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """실현 가능성 확인 (항상 가능)"""
        return True, "재활용재 옵션은 항상 적용 가능합니다."

    def to_dict(self) -> Dict[str, Any]:
        """제약조건 설정을 딕셔너리로 직렬화"""
        return {
            'type': 'recycling_option_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """딕셔너리에서 제약조건 설정 로드"""
        self.enabled = config.get('enabled', True)

    def get_display_summary(self) -> str:
        """UI 표시용 요약"""
        if self.enabled:
            return "✅ 허용 | recycling_option: 재활용재 사용 허용"
        else:
            return "🚫 금지 | recycling_option: 재활용재 사용 금지 (비율=0)"


class LowCarbonOptionConstraint(ConstraintBase):
    """
    저탄소메탈 사용 옵션 제약조건

    - 활성화: 저탄소메탈 비율을 최적화 변수로 사용 (0~100% 자유)
    - 비활성화: 저탄소메탈 비율을 0%로 고정
    """

    def __init__(self, enabled: bool = True):
        """
        저탄소메탈 옵션 제약조건 초기화

        Args:
            enabled: True면 저탄소메탈 사용 허용, False면 금지
        """
        super().__init__(
            name="low_carbon_option",
            description="저탄소메탈 사용 활성화/비활성화",
            enabled=enabled
        )

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """설정 유효성 검증 (특별한 설정 없음)"""
        return True, "유효한 설정입니다."

    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        비활성화된 경우:
        - 모든 자재의 low_carbon_ratio를 0으로 고정
        - 양극재 원소의 element_low_carb_ratio를 0으로 고정
        """
        if not self.enabled:
            print(f"   🚫 저탄소메탈 사용 금지 - low_carbon_ratio를 0으로 고정")

            # 일반 자재의 저탄소 비율 = 0
            def no_low_carbon_rule(m, material):
                return m.low_carbon_ratio[material] == 0

            model.no_low_carbon_constraint = pyo.Constraint(
                model.materials,
                rule=no_low_carbon_rule,
                doc="저탄소메탈 사용 금지"
            )

            # 양극재 원소의 저탄소 비율 = 0
            if hasattr(model, 'elements'):
                def no_element_low_carbon_rule(m, element):
                    return m.element_low_carb_ratio[element] == 0

                model.no_element_low_carbon_constraint = pyo.Constraint(
                    model.elements,
                    rule=no_element_low_carbon_rule,
                    doc="양극재 원소 저탄소메탈 금지"
                )
        else:
            print(f"   ✅ 저탄소메탈 사용 허용 - 최적화 변수로 사용")

    def check_feasibility(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """실현 가능성 확인 (항상 가능)"""
        return True, "저탄소메탈 옵션은 항상 적용 가능합니다."

    def to_dict(self) -> Dict[str, Any]:
        """제약조건 설정을 딕셔너리로 직렬화"""
        return {
            'type': 'low_carbon_option_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """딕셔너리에서 제약조건 설정 로드"""
        self.enabled = config.get('enabled', True)

    def get_display_summary(self) -> str:
        """UI 표시용 요약"""
        if self.enabled:
            return "✅ 허용 | low_carbon_option: 저탄소메탈 사용 허용"
        else:
            return "🚫 금지 | low_carbon_option: 저탄소메탈 사용 금지 (비율=0)"


class SiteChangeOptionConstraint(ConstraintBase):
    """
    생산지 변경 옵션 제약조건

    - 활성화: 변경된 전력 배출계수 사용 (cathode_site.json의 'after' 사이트)
    - 비활성화: 기본 전력 배출계수 사용 (cathode_site.json의 'before' 사이트)

    주의: 이 제약조건은 DataLoader에서 데이터를 로드할 때 영향을 미치므로
    모델 빌드 전에 적용되어야 합니다.
    """

    def __init__(self, enabled: bool = False):
        """
        생산지 변경 옵션 제약조건 초기화

        Args:
            enabled: True면 사이트 변경 허용, False면 기본 사이트 유지
        """
        super().__init__(
            name="site_change_option",
            description="생산지 변경 활성화/비활성화",
            enabled=enabled
        )
        self.site = 'after' if enabled else 'before'

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """설정 유효성 검증"""
        # cathode_site.json 파일 존재 확인
        import os
        import json

        try:
            cathode_site_path = "input/cathode_site.json"
            if not os.path.exists(cathode_site_path):
                return False, "cathode_site.json 파일이 없습니다."

            with open(cathode_site_path, 'r', encoding='utf-8') as f:
                site_data = json.load(f)

            # 필수 키 확인
            if 'CAM' not in site_data or 'pCAM' not in site_data:
                return False, "cathode_site.json에 CAM 또는 pCAM 설정이 없습니다."

            return True, "유효한 설정입니다."

        except Exception as e:
            return False, f"cathode_site.json 검증 실패: {str(e)}"

    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        주의: 이 제약조건은 실제로는 DataLoader 단계에서 처리되어야 하므로
        여기서는 로깅만 수행합니다.
        """
        if self.enabled:
            print(f"   🌍 생산지 변경 허용 - 'after' 사이트 전력계수 사용")
        else:
            print(f"   🏠 생산지 유지 - 'before' 사이트 전력계수 사용")

    def check_feasibility(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """실현 가능성 확인"""
        return self.validate_config({})

    def to_dict(self) -> Dict[str, Any]:
        """제약조건 설정을 딕셔너리로 직렬화"""
        return {
            'type': 'site_change_option_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'site': self.site
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """딕셔너리에서 제약조건 설정 로드"""
        self.enabled = config.get('enabled', False)
        self.site = 'after' if self.enabled else 'before'

    def get_site(self) -> str:
        """현재 사이트 설정 반환 ('before' 또는 'after')"""
        return self.site

    def get_display_summary(self) -> str:
        """UI 표시용 요약"""
        if self.enabled:
            return f"🌍 허용 | site_change_option: 생산지 변경 허용 (사이트: {self.site})"
        else:
            return f"🏠 금지 | site_change_option: 생산지 변경 금지 (사이트: {self.site})"

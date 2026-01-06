"""
제약조건 관리자

모든 제약조건을 관리하고 조정하는 중앙 관리자입니다.
"""

from typing import Dict, List, Tuple, Any, Optional
import pyomo.environ as pyo
from .constraint_base import ConstraintBase


class ConstraintManager:
    """
    제약조건 관리자 클래스

    - 제약조건 추가/제거
    - 우선순위 기반 실행 순서 관리
    - 제약조건 검증 파이프라인
    - 설정 직렬화/역직렬화
    """

    def __init__(self):
        """제약조건 관리자 초기화"""
        self.constraints: Dict[str, ConstraintBase] = {}
        self.execution_order: List[Tuple[int, str]] = []  # [(priority, name), ...]

    def add_constraint(
        self,
        constraint: ConstraintBase,
        priority: int = 100,
        replace_if_exists: bool = False,
        auto_save: bool = True,
        user_id: str = 'default'
    ) -> bool:
        """
        제약조건 추가

        Args:
            constraint: 추가할 제약조건 객체
            priority: 우선순위 (낮을수록 먼저 실행, 기본값 100)
            replace_if_exists: 동일 이름 제약조건이 있을 때 교체 여부
            auto_save: 자동 저장 여부
            user_id: 사용자 ID (auto_save=True일 때 사용)

        Returns:
            성공 여부
        """
        if constraint.name in self.constraints and not replace_if_exists:
            print(f"⚠️  제약조건 '{constraint.name}'이(가) 이미 존재합니다.")
            return False

        # 기존 제약조건 제거 (교체 모드)
        if constraint.name in self.constraints:
            self.remove_constraint(constraint.name, auto_save=False)  # 교체 시에는 auto_save 비활성화

        # 새 제약조건 추가
        self.constraints[constraint.name] = constraint
        self.execution_order.append((priority, constraint.name))
        self.execution_order.sort()  # 우선순위 순으로 정렬

        print(f"✅ 제약조건 '{constraint.name}' 추가됨 (우선순위: {priority})")

        # 자동 저장
        if auto_save:
            self.save_to_file(user_id=user_id)

        return True

    def remove_constraint(self, name: str, auto_save: bool = True, user_id: str = 'default') -> bool:
        """
        제약조건 제거

        Args:
            name: 제거할 제약조건 이름
            auto_save: 자동 저장 여부
            user_id: 사용자 ID

        Returns:
            성공 여부
        """
        if name not in self.constraints:
            print(f"⚠️  제약조건 '{name}'을(를) 찾을 수 없습니다.")
            return False

        del self.constraints[name]
        self.execution_order = [(p, n) for p, n in self.execution_order if n != name]

        print(f"✅ 제약조건 '{name}' 제거됨")

        # 자동 저장
        if auto_save:
            self.save_to_file(user_id=user_id)

        return True

    def get_constraint(self, name: str) -> Optional[ConstraintBase]:
        """
        이름으로 제약조건 가져오기

        Args:
            name: 제약조건 이름

        Returns:
            제약조건 객체 또는 None
        """
        return self.constraints.get(name)

    def list_constraints(self, enabled_only: bool = False) -> List[str]:
        """
        제약조건 목록 반환

        Args:
            enabled_only: True면 활성화된 제약조건만 반환

        Returns:
            제약조건 이름 리스트
        """
        if enabled_only:
            return [name for name in self.constraints if self.constraints[name].enabled]
        return list(self.constraints.keys())

    def list_constraint_objects(self, enabled_only: bool = False) -> List[ConstraintBase]:
        """
        제약조건 객체 목록 반환

        Args:
            enabled_only: True면 활성화된 제약조건만 반환

        Returns:
            제약조건 객체 리스트
        """
        if enabled_only:
            return [c for c in self.constraints.values() if c.enabled]
        return list(self.constraints.values())

    def enable_constraint(self, name: str) -> bool:
        """
        제약조건 활성화

        Args:
            name: 제약조건 이름

        Returns:
            성공 여부
        """
        constraint = self.get_constraint(name)
        if constraint:
            constraint.enable()
            print(f"✅ 제약조건 '{name}' 활성화됨")
            return True
        return False

    def disable_constraint(self, name: str) -> bool:
        """
        제약조건 비활성화

        Args:
            name: 제약조건 이름

        Returns:
            성공 여부
        """
        constraint = self.get_constraint(name)
        if constraint:
            constraint.disable()
            print(f"❌ 제약조건 '{name}' 비활성화됨")
            return True
        return False

    def validate_all(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        모든 활성화된 제약조건 검증

        Args:
            data: 시뮬레이션 데이터

        Returns:
            (is_valid, errors): 전체 유효성과 오류 메시지 리스트
        """
        errors = []

        for _, name in self.execution_order:
            constraint = self.constraints[name]

            if not constraint.enabled:
                continue

            # 제약조건 설정 검증
            is_valid, msg = constraint.validate_config(constraint.config)
            if not is_valid:
                errors.append(f"[{name}] 설정 오류: {msg}")
                continue

            # 실현 가능성 검증
            is_feasible, msg = constraint.check_feasibility(data)
            if not is_feasible:
                errors.append(f"[{name}] 실현 불가능: {msg}")

        if errors:
            print(f"❌ 제약조건 검증 실패: {len(errors)}개 오류")
            for error in errors:
                print(f"  • {error}")
        else:
            print(f"✅ 모든 제약조건 검증 통과 ({len(self.list_constraints(enabled_only=True))}개)")

        return len(errors) == 0, errors

    def validate_constraint_compatibility(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        제약조건 간 충돌 검사

        MaterialManagementConstraint와 CostConstraint의 조합이 실현 가능한지 검증합니다.

        Args:
            data: 시뮬레이션 데이터

        Returns:
            (is_compatible, warnings): 호환성 여부와 경고 메시지 리스트
        """
        warnings = []

        # MaterialManagementConstraint와 CostConstraint 찾기
        from ..constraints import MaterialManagementConstraint, CostConstraint

        material_constraints = [
            c for c in self.list_constraint_objects(enabled_only=True)
            if isinstance(c, MaterialManagementConstraint)
        ]
        cost_constraints = [
            c for c in self.list_constraint_objects(enabled_only=True)
            if isinstance(c, CostConstraint)
        ]

        if not material_constraints or not cost_constraints:
            # 두 제약조건이 함께 사용되지 않으면 충돌 없음
            return True, warnings

        # 비용 프리미엄 데이터 확인
        material_cost_premiums = data.get('material_cost_premiums', {})
        if not material_cost_premiums:
            warnings.append(
                "⚠️ 비용 프리미엄 데이터가 없습니다. 재활용/저탄소 비용이 반영되지 않을 수 있습니다."
            )

        # MaterialManagementConstraint 규칙 검사
        for mc in material_constraints:
            for rule in mc.material_rules:
                rule_type = rule['type']
                material = rule['material']

                # 재활용/저탄소 강제 규칙 확인
                if rule_type in ['recycle_only', 'low_carbon_only', 'force_ratio_range', 'force_element_ratio_range']:
                    # 원소 추출
                    element = None
                    for elem in ['Ni', 'Co', 'Li']:
                        if elem in material:
                            element = elem
                            break

                    if element and element in material_cost_premiums:
                        premiums = material_cost_premiums[element]
                        recycle_premium = premiums.get('recycle_premium_pct', 0)
                        low_carbon_premium = premiums.get('low_carbon_premium_pct', 0)

                        if rule_type == 'recycle_only' and recycle_premium > 0:
                            warnings.append(
                                f"💡 {material}: 'recycle_only' 규칙 적용 시 비용이 +{recycle_premium}% 증가합니다."
                            )
                        elif rule_type == 'low_carbon_only' and low_carbon_premium > 0:
                            warnings.append(
                                f"💡 {material}: 'low_carbon_only' 규칙 적용 시 비용이 +{low_carbon_premium}% 증가합니다."
                            )

        # CostConstraint 제한 확인
        for cc in cost_constraints:
            if cc.premium_limit_pct is not None:
                warnings.append(
                    f"💡 비용 프리미엄 한도: +{cc.premium_limit_pct}% "
                    f"(재활용/저탄소 강제 시 초과 가능성 있음)"
                )

            if cc.absolute_premium_budget is not None:
                warnings.append(
                    f"💡 절대 예산 한도: ${cc.absolute_premium_budget:,.2f} "
                    f"(재활용/저탄소 강제 시 초과 가능성 있음)"
                )

        return len(warnings) == 0 or all('💡' in w for w in warnings), warnings

    def apply_all_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        모든 활성화된 제약조건을 Pyomo 모델에 적용

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터
        """
        # 제약조건 충돌 검사
        is_compatible, warnings = self.validate_constraint_compatibility(data)

        if warnings:
            print("\n⚠️  제약조건 호환성 경고:")
            for warning in warnings:
                print(f"  {warning}")
            print()

        applied_count = 0

        for _, name in self.execution_order:
            constraint = self.constraints[name]

            if not constraint.enabled:
                continue

            try:
                print(f"⚙️  제약조건 적용 중: {name}")
                constraint.apply_to_model(model, data)
                applied_count += 1
                print(f"  ✅ 완료")

            except Exception as e:
                error_msg = f"제약조건 '{name}' 적용 중 오류: {str(e)}"
                print(f"  ❌ {error_msg}")
                raise Exception(error_msg) from e

        print(f"\n✅ 총 {applied_count}개 제약조건이 모델에 적용되었습니다.")

    def to_config(self) -> Dict[str, Any]:
        """
        모든 제약조건을 설정 딕셔너리로 내보내기

        Returns:
            설정 딕셔너리
        """
        config = {
            'constraints': {},
            'execution_order': {}
        }

        for name, constraint in self.constraints.items():
            config['constraints'][name] = constraint.to_dict()

        for priority, name in self.execution_order:
            config['execution_order'][name] = priority

        return config

    def from_config(self, config: Dict[str, Any], constraint_registry: Dict[str, type]) -> None:
        """
        설정 딕셔너리에서 제약조건 로드

        Args:
            config: 설정 딕셔너리
            constraint_registry: {constraint_type: ConstraintClass} 매핑
        """
        self.constraints.clear()
        self.execution_order.clear()

        for name, constraint_config in config.get('constraints', {}).items():
            constraint_type = constraint_config.get('type')

            if constraint_type not in constraint_registry:
                print(f"⚠️  알 수 없는 제약조건 타입: {constraint_type}")
                print(f"   사용 가능한 타입: {list(constraint_registry.keys())}")
                continue

            # 제약조건 인스턴스 생성
            constraint_class = constraint_registry[constraint_type]
            constraint = constraint_class()
            constraint.from_dict(constraint_config)

            # 우선순위 가져오기
            priority = config.get('execution_order', {}).get(name, 100)

            # 추가
            self.add_constraint(constraint, priority=priority)

    def save_to_file(self, user_id: str = 'default') -> bool:
        """
        현재 제약조건들을 파일에 저장합니다.

        Args:
            user_id: 사용자 ID

        Returns:
            성공 여부
        """
        try:
            from src.utils.file_operations import FileOperations
            from datetime import datetime

            # 설정 생성
            config = {
                'version': '1.0',
                'saved_at': datetime.now().isoformat(),
                'constraints': []
            }

            # to_config() 결과를 리스트 형태로 변환
            config_dict = self.to_config()

            # 제약조건 타입 검증
            from ..constraints import (
                MaterialManagementConstraint,
                CostConstraint,
                RecyclingOptionConstraint,
                LowCarbonOptionConstraint,
                SiteChangeOptionConstraint
            )

            constraint_registry = {
                'cost_constraint': CostConstraint,
                'material_management_constraint': MaterialManagementConstraint,
                'recycling_option_constraint': RecyclingOptionConstraint,
                'low_carbon_option_constraint': LowCarbonOptionConstraint,
                'site_change_option_constraint': SiteChangeOptionConstraint
            }

            # 타입 검증
            invalid_types = []
            for name, constraint_data in config_dict['constraints'].items():
                constraint_type = constraint_data.get('type')
                if constraint_type not in constraint_registry:
                    invalid_types.append((name, constraint_type))

            if invalid_types:
                print(f"\n⚠️  경고: {len(invalid_types)}개 제약조건이 잘못된 타입을 가지고 있습니다:")
                for name, ctype in invalid_types:
                    print(f"   • '{name}': type='{ctype}'")
                print(f"   이 제약조건들은 로드되지 않을 수 있습니다!")
                print(f"   유효한 타입: {list(constraint_registry.keys())}")

            for name, constraint_data in config_dict['constraints'].items():
                constraint_data['priority'] = config_dict['execution_order'].get(name, 100)
                config['constraints'].append(constraint_data)

            # 저장 전 로깅
            print(f"\n💾 제약조건 저장 중: {len(config['constraints'])}개")
            for c in config['constraints']:
                print(f"   • {c.get('name')} (type: {c.get('type')})")

            # 파일 저장
            file_path = f'input/{user_id}/constraint_config.json'
            FileOperations.save_json(file_path, config, user_id=user_id)

            print(f"✅ 제약조건 저장 완료: {file_path}")
            return True

        except Exception as e:
            print(f"❌ 제약조건 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_from_file(self, user_id: str = 'default', constraint_registry: Optional[Dict[str, type]] = None) -> bool:
        """
        파일에서 제약조건들을 불러옵니다.

        Args:
            user_id: 사용자 ID
            constraint_registry: 제약조건 타입 매핑

        Returns:
            성공 여부
        """
        try:
            from src.utils.file_operations import FileOperations

            # 파일 로드
            file_path = f'input/{user_id}/constraint_config.json'
            config = FileOperations.load_json(
                file_path,
                default={'version': '1.0', 'constraints': []},
                user_id=user_id
            )

            # 버전 확인
            if config.get('version') != '1.0':
                print(f"⚠️ 호환되지 않는 설정 버전: {config.get('version')}")
                return False

            # 제약조건 불러오기
            constraints_list = config.get('constraints', [])
            print(f"📂 파일에서 {len(constraints_list)}개 제약조건 발견")
            if constraints_list:
                # 리스트를 from_config가 기대하는 딕셔너리 형태로 변환
                converted_config = {
                    'constraints': {},
                    'execution_order': {}
                }

                for constraint_data in constraints_list:
                    name = constraint_data.get('name')
                    priority = constraint_data.get('priority', 100)
                    converted_config['constraints'][name] = constraint_data
                    converted_config['execution_order'][name] = priority

                # constraint_registry가 없으면 기본 레지스트리 사용
                if constraint_registry is None:
                    from ..constraints import (
                        CostConstraint,
                        MaterialManagementConstraint,
                        RecyclingOptionConstraint,
                        LowCarbonOptionConstraint,
                        SiteChangeOptionConstraint
                    )
                    constraint_registry = {
                        'cost_constraint': CostConstraint,
                        'material_management_constraint': MaterialManagementConstraint,
                        'recycling_option_constraint': RecyclingOptionConstraint,
                        'low_carbon_option_constraint': LowCarbonOptionConstraint,
                        'site_change_option_constraint': SiteChangeOptionConstraint
                    }

                self.from_config(converted_config, constraint_registry)

                # CostConstraint에 cost_calculator 주입
                from ..constraints import CostConstraint
                from ..utils.total_cost_calculator import TotalCostCalculator
                from src.optimization.re100_premium_calculator import RE100PremiumCalculator

                for constraint in self.list_constraint_objects():
                    if isinstance(constraint, CostConstraint) and constraint.cost_calculator is None:
                        # TotalCostCalculator 생성 및 주입
                        re100_calc = RE100PremiumCalculator(user_id=user_id)
                        total_cost_calc = TotalCostCalculator(re100_calculator=re100_calc, debug_mode=False)
                        constraint.set_cost_calculator(total_cost_calc)
                        print(f"   ℹ️  CostConstraint에 cost_calculator 자동 주입됨")

                print(f"✅ 제약조건 {len(constraints_list)}개 불러오기 완료")
                return True
            else:
                print(f"ℹ️  저장된 제약조건이 없습니다.")
                return False

        except Exception as e:
            print(f"❌ 제약조건 불러오기 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_summary(self) -> str:
        """
        제약조건 관리자 요약 정보

        Returns:
            요약 문자열
        """
        total = len(self.constraints)
        enabled = len(self.list_constraints(enabled_only=True))
        disabled = total - enabled

        summary = f"📊 제약조건 관리자 요약\n"
        summary += f"  • 전체: {total}개\n"
        summary += f"  • 활성화: {enabled}개\n"
        summary += f"  • 비활성화: {disabled}개\n\n"

        if self.constraints:
            summary += "제약조건 목록 (우선순위 순):\n"
            for priority, name in self.execution_order:
                constraint = self.constraints[name]
                summary += f"  [{priority}] {constraint}\n"

        return summary

    def __repr__(self) -> str:
        """제약조건 관리자 문자열 표현"""
        return f"<ConstraintManager(constraints={len(self.constraints)})>"

    def __str__(self) -> str:
        """제약조건 관리자 사용자 친화적 문자열"""
        return self.get_summary()

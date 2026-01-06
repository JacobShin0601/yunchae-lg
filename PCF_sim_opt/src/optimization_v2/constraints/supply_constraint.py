"""
공급량 제약조건

재활용재, 저탄소메탈 등의 가용량을 제한하는 제약조건입니다.
"""

from typing import Dict, Any, Tuple, Optional
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class SupplyConstraint(ConstraintBase):
    """
    공급량 제약조건 클래스

    다음 유형의 공급량 제한을 설정할 수 있습니다:
    - 재활용재 총 가용량 제한
    - 저탄소메탈 총 가용량 제한
    - 자재별 최대 공급량 제한
    """

    def __init__(self):
        """
        공급량 제약조건 초기화
        """
        super().__init__(
            name="supply_constraint",
            description="재활용재/저탄소메탈 공급량 제약"
        )
        self.total_recycle_supply: Optional[float] = None  # kg
        self.total_low_carbon_supply: Optional[float] = None  # kg
        self.material_supply_limits: Dict[str, float] = {}  # {material: max_supply_kg}

    def set_total_recycle_supply(self, max_supply: float) -> None:
        """
        재활용재 총 가용량 설정

        Args:
            max_supply: 최대 공급량 (kg)
        """
        if max_supply <= 0:
            raise ValueError("공급량은 0보다 커야 합니다.")

        self.total_recycle_supply = max_supply
        print(f"✅ 재활용재 총 가용량 설정: {max_supply:,.2f} kg")

    def set_total_low_carbon_supply(self, max_supply: float) -> None:
        """
        저탄소메탈 총 가용량 설정

        Args:
            max_supply: 최대 공급량 (kg)
        """
        if max_supply <= 0:
            raise ValueError("공급량은 0보다 커야 합니다.")

        self.total_low_carbon_supply = max_supply
        print(f"✅ 저탄소메탈 총 가용량 설정: {max_supply:,.2f} kg")

    def set_material_supply_limit(self, material_name: str, max_supply: float) -> None:
        """
        자재별 최대 공급량 설정

        Args:
            material_name: 자재명
            max_supply: 최대 공급량 (kg)
        """
        if max_supply < 0:
            raise ValueError("공급량은 0 이상이어야 합니다.")

        self.material_supply_limits[material_name] = max_supply
        print(f"✅ {material_name} 최대 공급량 설정: {max_supply:,.2f} kg")

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            (is_valid, message)
        """
        # 최소한 하나의 제한 필요
        has_limit = (
            self.total_recycle_supply is not None or
            self.total_low_carbon_supply is not None or
            len(self.material_supply_limits) > 0
        )

        if not has_limit:
            return False, "재활용재 가용량, 저탄소메탈 가용량, 자재별 제한 중 하나는 설정되어야 합니다."

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

        material_classification = data.get('material_classification')
        if material_classification is None:
            return False, "material_classification이 없습니다."

        # 자재별 제한이 있는 경우, 해당 자재가 데이터에 있는지 확인
        available_materials = set(scenario_df['자재명'].unique())
        for material in self.material_supply_limits.keys():
            if material not in available_materials:
                return False, f"자재 '{material}'이(가) 데이터에 없습니다."

        # Ni-Co-Li 자재의 총 수요량 계산 (간단한 확인)
        nicoli_materials = [
            m for m, info in material_classification.items()
            if info['is_ni_co_li']
        ]

        if nicoli_materials and '제품총소요량(kg)' in scenario_df.columns:
            total_demand = 0
            for material in nicoli_materials:
                material_row = scenario_df[scenario_df['자재명'] == material]
                if len(material_row) > 0:
                    total_demand += material_row.iloc[0]['제품총소요량(kg)']

            # 재활용재 가용량 확인
            if self.total_recycle_supply is not None:
                if self.total_recycle_supply < total_demand * 0.05:  # 최소 5% 적용 가능해야 함
                    return False, f"재활용재 가용량({self.total_recycle_supply:,.2f}kg)이 총 수요량({total_demand:,.2f}kg)의 5%보다 적습니다. 제약이 너무 엄격할 수 있습니다."

            # 저탄소메탈 가용량 확인
            if self.total_low_carbon_supply is not None:
                if self.total_low_carbon_supply < total_demand * 0.05:
                    return False, f"저탄소메탈 가용량({self.total_low_carbon_supply:,.2f}kg)이 총 수요량({total_demand:,.2f}kg)의 5%보다 적습니다. 제약이 너무 엄격할 수 있습니다."

        return True, "실현 가능"

    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터
        """
        scenario_df = data['scenario_df']
        material_classification = data['material_classification']

        # Ni-Co-Li 자재만 필터링
        nicoli_materials = [
            m for m in model.materials
            if material_classification[m]['is_ni_co_li']
        ]

        if not nicoli_materials:
            print("    ⚠️ Ni-Co-Li 자재가 없어 공급량 제약을 적용하지 않습니다.")
            return

        # 1. 재활용재 총 가용량 제약
        if self.total_recycle_supply is not None:
            # 재활용재 총 사용량 = Σ(재활용 비율 × 수량)
            total_recycle_usage = 0.0

            for material in nicoli_materials:
                material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
                quantity = material_row['제품총소요량(kg)']

                # 재활용재 사용량 = recycle_ratio × 수량
                recycle_usage = model.recycle_ratio[material] * quantity
                total_recycle_usage += recycle_usage

            # 제약조건 추가
            model.add_component(
                'total_recycle_supply_constraint',
                pyo.Constraint(expr=total_recycle_usage <= self.total_recycle_supply)
            )

            print(f"    • 재활용재 총 가용량: {self.total_recycle_supply:,.2f} kg")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        # 2. 저탄소메탈 총 가용량 제약
        if self.total_low_carbon_supply is not None:
            # 저탄소메탈 총 사용량 = Σ(저탄소 비율 × 수량)
            total_low_carbon_usage = 0.0

            for material in nicoli_materials:
                material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
                quantity = material_row['제품총소요량(kg)']

                # 저탄소메탈 사용량 = low_carbon_ratio × 수량
                low_carbon_usage = model.low_carbon_ratio[material] * quantity
                total_low_carbon_usage += low_carbon_usage

            # 제약조건 추가
            model.add_component(
                'total_low_carbon_supply_constraint',
                pyo.Constraint(expr=total_low_carbon_usage <= self.total_low_carbon_supply)
            )

            print(f"    • 저탄소메탈 총 가용량: {self.total_low_carbon_supply:,.2f} kg")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        # 3. 자재별 최대 공급량 제약
        if self.material_supply_limits:
            for material_name, max_supply in self.material_supply_limits.items():
                if material_name not in model.materials:
                    print(f"    ⚠️ 자재 '{material_name}'이(가) 모델에 없습니다. 스킵합니다.")
                    continue

                # 자재 정보
                material_row = scenario_df[scenario_df['자재명'] == material_name].iloc[0]
                quantity = material_row['제품총소요량(kg)']

                # 자재의 총 사용량이 최대 공급량을 초과할 수 없음
                # (실제로는 수량이 고정이므로, 이 제약은 실현 불가능 확인용)
                if quantity > max_supply:
                    print(f"    ⚠️ {material_name}: 현재 수요량({quantity:,.2f}kg)이 최대 공급량({max_supply:,.2f}kg)을 초과합니다!")
                    print(f"       → 변수를 0으로 고정하여 저감 활동 제한")

                    # 저감 활동을 제한 (Formula 자재의 경우)
                    if material_classification[material_name]['type'] == 'Formula':
                        model.tier1_re[material_name].fix(0)
                        model.tier2_re[material_name].fix(0)

                    print(f"    • {material_name} 최대 공급량: {max_supply:,.2f} kg")
                    print(f"      ✅ 공급량 초과로 변수 고정")
                else:
                    print(f"    • {material_name} 최대 공급량: {max_supply:,.2f} kg (현재: {quantity:,.2f} kg)")
                    print(f"      ✅ 공급량 제약 확인 완료")

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        return {
            'type': 'supply_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'total_recycle_supply': self.total_recycle_supply,
            'total_low_carbon_supply': self.total_low_carbon_supply,
            'material_supply_limits': self.material_supply_limits
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        self.name = config.get('name', 'supply_constraint')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.total_recycle_supply = config.get('total_recycle_supply')
        self.total_low_carbon_supply = config.get('total_low_carbon_supply')
        self.material_supply_limits = config.get('material_supply_limits', {})

    def get_summary(self) -> str:
        """
        제약조건 요약

        Returns:
            요약 문자열
        """
        base_summary = super().get_summary()
        details = []

        if self.total_recycle_supply is not None:
            details.append(f"재활용재: {self.total_recycle_supply:,.0f} kg")

        if self.total_low_carbon_supply is not None:
            details.append(f"저탄소메탈: {self.total_low_carbon_supply:,.0f} kg")

        if self.material_supply_limits:
            details.append(f"자재별 제한: {len(self.material_supply_limits)}개")

        detail_str = " | ".join(details) if details else "설정 없음"
        return f"{base_summary}\n  📦 {detail_str}"

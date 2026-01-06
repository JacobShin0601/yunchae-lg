"""
배출량 목표 제약조건

탄소 배출량 목표를 설정하는 제약조건입니다.
전체 배출량 상한, 자재별 배출량 상한, 감축률 목표 등을 설정할 수 있습니다.
"""

from typing import Dict, Any, Tuple, Optional
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class EmissionTargetConstraint(ConstraintBase):
    """
    배출량 목표 제약조건 클래스

    다음 유형의 배출량 목표를 설정할 수 있습니다:
    - 전체 배출량 상한 (absolute_emission_cap)
    - 감축률 목표 (reduction_target_pct)
    - 자재별 배출량 상한 (material_emission_limits)
    """

    def __init__(self):
        """
        배출량 목표 제약조건 초기화
        """
        super().__init__(
            name="emission_target_constraint",
            description="탄소 배출량 목표 제약"
        )
        self.baseline_emission: Optional[float] = None
        self.absolute_emission_cap: Optional[float] = None  # kgCO2eq
        self.reduction_target_pct: Optional[float] = None  # 예: 30 (30% 감축)
        self.material_emission_limits: Dict[str, float] = {}  # {material: max_emission}

    def set_baseline_emission(self, baseline: float) -> None:
        """
        기준 배출량 설정

        Args:
            baseline: 기준 배출량 (kgCO2eq)
        """
        if baseline <= 0:
            raise ValueError("기준 배출량은 0보다 커야 합니다.")

        self.baseline_emission = baseline
        print(f"✅ 기준 배출량 설정: {baseline:,.2f} kgCO2eq")

    def set_absolute_cap(self, cap: float) -> None:
        """
        절대 배출량 상한 설정

        Args:
            cap: 배출량 상한 (kgCO2eq)
        """
        if cap <= 0:
            raise ValueError("배출량 상한은 0보다 커야 합니다.")

        self.absolute_emission_cap = cap
        print(f"✅ 배출량 상한 설정: {cap:,.2f} kgCO2eq")

        if self.baseline_emission:
            reduction = (1 - cap / self.baseline_emission) * 100
            print(f"   기준 대비 감축률: {reduction:.1f}%")

    def set_reduction_target(self, target_pct: float) -> None:
        """
        감축률 목표 설정

        Args:
            target_pct: 감축률 목표 (예: 30 = 30% 감축)
        """
        if target_pct < 0 or target_pct > 100:
            raise ValueError("감축률은 0~100 사이여야 합니다.")

        self.reduction_target_pct = target_pct
        print(f"✅ 감축률 목표 설정: {target_pct:.1f}%")

        if self.baseline_emission:
            cap = self.baseline_emission * (1 - target_pct / 100)
            print(f"   배출량 상한: {cap:,.2f} kgCO2eq")

    def set_material_emission_limit(self, material_name: str, max_emission: float) -> None:
        """
        자재별 배출량 제한 설정

        Args:
            material_name: 자재명
            max_emission: 최대 배출량 (kgCO2eq)
        """
        if max_emission < 0:
            raise ValueError("배출량은 0 이상이어야 합니다.")

        self.material_emission_limits[material_name] = max_emission
        print(f"✅ {material_name} 배출량 제한: {max_emission:,.2f} kgCO2eq")

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            (is_valid, message)
        """
        # 최소한 하나의 목표 필요
        has_target = (
            self.absolute_emission_cap is not None or
            self.reduction_target_pct is not None or
            len(self.material_emission_limits) > 0
        )

        if not has_target:
            return False, "배출량 상한, 감축률 목표, 자재별 제한 중 하나는 설정되어야 합니다."

        # 감축률 사용 시 기준 배출량 필요
        if self.reduction_target_pct is not None and self.baseline_emission is None:
            return False, "감축률 목표 사용 시 기준 배출량(baseline_emission)이 필요합니다."

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

        # 자재별 제한이 있는 경우, 해당 자재가 데이터에 있는지 확인
        available_materials = set(scenario_df['자재명'].unique())
        for material in self.material_emission_limits.keys():
            if material not in available_materials:
                return False, f"자재 '{material}'이(가) 데이터에 없습니다."

        # 현재 배출량 계산 (간단한 확인)
        if '배출계수' in scenario_df.columns and '제품총소요량(kg)' in scenario_df.columns:
            current_emission = (scenario_df['배출계수'] * scenario_df['제품총소요량(kg)']).sum()

            # 절대 상한 확인
            if self.absolute_emission_cap is not None:
                if current_emission > self.absolute_emission_cap * 1.5:  # 1.5배 여유
                    return False, f"현재 배출량({current_emission:,.2f})이 목표({self.absolute_emission_cap:,.2f})의 150%를 초과합니다. 목표가 너무 엄격할 수 있습니다."

            # 감축률 목표 확인
            if self.reduction_target_pct is not None and self.baseline_emission is not None:
                target_emission = self.baseline_emission * (1 - self.reduction_target_pct / 100)
                if current_emission > target_emission * 1.5:
                    return False, f"현재 배출량이 목표의 150%를 초과합니다. 감축률 목표가 너무 높을 수 있습니다."

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

        # 전체 배출량 표현식 계산
        total_emission_expr = self._calculate_total_emission_expression(
            model, scenario_df, material_classification
        )

        # 1. 절대 배출량 상한 제약
        if self.absolute_emission_cap is not None:
            model.add_component(
                'absolute_emission_cap_constraint',
                pyo.Constraint(expr=total_emission_expr <= self.absolute_emission_cap)
            )
            print(f"    • 배출량 상한: {self.absolute_emission_cap:,.2f} kgCO2eq")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        # 2. 감축률 목표 제약
        if self.reduction_target_pct is not None and self.baseline_emission is not None:
            target_emission = self.baseline_emission * (1 - self.reduction_target_pct / 100)

            model.add_component(
                'reduction_target_constraint',
                pyo.Constraint(expr=total_emission_expr <= target_emission)
            )
            print(f"    • 감축률 목표: {self.reduction_target_pct:.1f}%")
            print(f"      기준: {self.baseline_emission:,.2f} → 목표: {target_emission:,.2f} kgCO2eq")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        # 3. 자재별 배출량 제한
        if self.material_emission_limits:
            for material_name, max_emission in self.material_emission_limits.items():
                if material_name not in model.materials:
                    print(f"    ⚠️ 자재 '{material_name}'이(가) 모델에 없습니다. 스킵합니다.")
                    continue

                # 자재 정보
                material_row = scenario_df[scenario_df['자재명'] == material_name].iloc[0]
                quantity = material_row['제품총소요량(kg)']

                # 자재별 배출량 표현식
                material_emission_expr = model.modified_emission[material_name] * quantity

                # 제약조건 추가
                constraint_name = f'material_emission_limit_{material_name.replace(" ", "_")}'
                model.add_component(
                    constraint_name,
                    pyo.Constraint(expr=material_emission_expr <= max_emission)
                )

                print(f"    • {material_name} 배출량 한도: {max_emission:,.2f} kgCO2eq")
                print(f"      ✅ Pyomo 제약조건 추가 완료")

    def _calculate_total_emission_expression(
        self,
        model: pyo.ConcreteModel,
        scenario_df,
        material_classification: Dict[str, Dict[str, Any]]
    ):
        """
        전체 배출량 표현식 계산

        Args:
            model: Pyomo 모델
            scenario_df: 시나리오 DataFrame
            material_classification: 자재 분류 정보

        Returns:
            전체 배출량 표현식 (Pyomo Expression)
        """
        total_emission = 0.0

        for material in model.materials:
            # 자재 정보
            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            quantity = material_row['제품총소요량(kg)']

            # 배출량 = modified_emission × 수량
            material_emission = model.modified_emission[material] * quantity

            total_emission += material_emission

        return total_emission

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        return {
            'type': 'emission_target_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'baseline_emission': self.baseline_emission,
            'absolute_emission_cap': self.absolute_emission_cap,
            'reduction_target_pct': self.reduction_target_pct,
            'material_emission_limits': self.material_emission_limits
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        self.name = config.get('name', 'emission_target_constraint')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.baseline_emission = config.get('baseline_emission')
        self.absolute_emission_cap = config.get('absolute_emission_cap')
        self.reduction_target_pct = config.get('reduction_target_pct')
        self.material_emission_limits = config.get('material_emission_limits', {})

    def get_summary(self) -> str:
        """
        제약조건 요약

        Returns:
            요약 문자열
        """
        base_summary = super().get_summary()
        details = []

        if self.baseline_emission:
            details.append(f"기준: {self.baseline_emission:,.0f} kgCO2eq")

        if self.absolute_emission_cap is not None:
            details.append(f"상한: {self.absolute_emission_cap:,.0f} kgCO2eq")

        if self.reduction_target_pct is not None:
            details.append(f"감축률: {self.reduction_target_pct:.1f}%")

        if self.material_emission_limits:
            details.append(f"자재별 제한: {len(self.material_emission_limits)}개")

        detail_str = " | ".join(details) if details else "설정 없음"
        return f"{base_summary}\n  🎯 {detail_str}"

"""
System-Level RE100 제약조건

시스템 전체의 RE100 적용률 목표를 설정하고 최적 배분을 수행합니다.
개별 자재의 RE100 최적화가 아닌, 전체 시스템 관점에서 RE100 목표를 달성하도록 조정합니다.
"""

from typing import Dict, Any, Tuple, Optional
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class SystemRE100Constraint(ConstraintBase):
    """
    System-Level RE100 제약조건 클래스

    특징:
    - 시스템 전체 RE100 목표 설정 (예: 50% RE100 달성)
    - 에너지 가중 평균 RE100 적용률 계산
    - 비용 효율적인 RE100 배분 (자동 최적화)
    - Tier1/Tier2 별도 목표 설정 가능

    수식:
        system_re100_rate = Σ(energy_i × re100_i) / Σ(energy_i) >= target_rate

        where:
        - energy_i = material i의 에너지 소비량 (tier1 + tier2)
        - re100_i = material i의 RE100 적용률 (가중 평균)
    """

    def __init__(self):
        """System-Level RE100 제약조건 초기화"""
        super().__init__(
            name="system_re100_constraint",
            description="시스템 전체 RE100 목표 제약"
        )

        # 시스템 레벨 RE100 목표
        self.system_re100_target: Optional[float] = None  # 0.0 ~ 1.0 (예: 0.5 = 50%)

        # Tier별 목표 (선택사항)
        self.tier1_re100_target: Optional[float] = None  # Tier1 RE100 목표 (0~1)
        self.tier2_re100_target: Optional[float] = None  # Tier2 RE100 목표 (0~1)

        # 자재별 최소/최대 RE100 제한 (선택사항)
        self.material_re100_min: Dict[str, float] = {}  # {material: min_rate}
        self.material_re100_max: Dict[str, float] = {}  # {material: max_rate}

        # 우선순위 가중치 (비용 효율성 고려)
        self.use_cost_efficiency: bool = True  # 비용 효율적 배분 사용 여부

        # Tier1 RE >= Tier2 RE 제약 (Tier1이 더 높은 RE 비율을 가지도록 강제)
        self.enforce_tier1_higher_than_tier2: bool = True  # 기본값: 활성화

    def set_system_re100_target(self, target_rate: float) -> None:
        """
        시스템 전체 RE100 목표 설정

        Args:
            target_rate: 목표 RE100 적용률 (0.0 ~ 1.0)
                        예: 0.5 = 시스템 전체 에너지의 50%를 RE100으로 전환
        """
        if target_rate < 0 or target_rate > 1:
            raise ValueError("RE100 목표는 0~1 사이여야 합니다.")

        self.system_re100_target = target_rate
        print(f"✅ System-Level RE100 목표 설정: {target_rate*100:.1f}%")

    def set_tier_targets(self, tier1_target: Optional[float] = None,
                        tier2_target: Optional[float] = None) -> None:
        """
        Tier별 RE100 목표 설정

        Args:
            tier1_target: Tier1 RE100 목표 (0~1), None이면 설정 안함
            tier2_target: Tier2 RE100 목표 (0~1), None이면 설정 안함
        """
        if tier1_target is not None:
            if tier1_target < 0 or tier1_target > 1:
                raise ValueError("Tier1 목표는 0~1 사이여야 합니다.")
            self.tier1_re100_target = tier1_target
            print(f"✅ Tier1 RE100 목표: {tier1_target*100:.1f}%")

        if tier2_target is not None:
            if tier2_target < 0 or tier2_target > 1:
                raise ValueError("Tier2 목표는 0~1 사이여야 합니다.")
            self.tier2_re100_target = tier2_target
            print(f"✅ Tier2 RE100 목표: {tier2_target*100:.1f}%")

    def set_material_re100_bounds(self, material_name: str,
                                  min_rate: float = 0.0,
                                  max_rate: float = 1.0) -> None:
        """
        자재별 RE100 적용률 범위 설정

        Args:
            material_name: 자재명
            min_rate: 최소 RE100 적용률 (0~1)
            max_rate: 최대 RE100 적용률 (0~1)
        """
        if min_rate < 0 or min_rate > 1:
            raise ValueError("최소 RE100 적용률은 0~1 사이여야 합니다.")
        if max_rate < 0 or max_rate > 1:
            raise ValueError("최대 RE100 적용률은 0~1 사이여야 합니다.")
        if min_rate > max_rate:
            raise ValueError("최소값이 최대값보다 클 수 없습니다.")

        self.material_re100_min[material_name] = min_rate
        self.material_re100_max[material_name] = max_rate
        print(f"✅ {material_name} RE100 범위: {min_rate*100:.0f}~{max_rate*100:.0f}%")

    def set_tier1_higher_than_tier2(self, enforce: bool = True) -> None:
        """
        Tier1 RE >= Tier2 RE 제약 설정

        각 자재에 대해 Tier1의 RE100 비율이 Tier2보다 높거나 같도록 강제합니다.
        이는 일반적으로 Tier1(직접 공급업체)이 Tier2(간접 공급업체)보다
        RE100 전환이 더 용이하고 영향력이 크다는 현실을 반영합니다.

        Args:
            enforce: True면 제약 활성화, False면 비활성화
        """
        self.enforce_tier1_higher_than_tier2 = enforce
        if enforce:
            print(f"✅ Tier1 RE >= Tier2 RE 제약 활성화")
        else:
            print(f"❌ Tier1 RE >= Tier2 RE 제약 비활성화")

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            (is_valid, message)
        """
        # 최소한 시스템 목표 또는 Tier 목표 중 하나는 있어야 함
        has_target = (
            self.system_re100_target is not None or
            self.tier1_re100_target is not None or
            self.tier2_re100_target is not None
        )

        if not has_target:
            return False, "System RE100 목표 또는 Tier별 목표 중 하나는 설정되어야 합니다."

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

        material_classification = data.get('material_classification', {})
        if not material_classification:
            return False, "material_classification이 없습니다."

        # RE100 적용 가능한 자재가 있는지 확인
        re100_applicable_count = sum(
            1 for mat_info in material_classification.values()
            if mat_info.get('tier1_energy_ratio', 0) > 0 or
               mat_info.get('tier2_energy_ratio', 0) > 0
        )

        if re100_applicable_count == 0:
            return False, "RE100 적용 가능한 자재가 없습니다. 에너지 비율이 설정된 자재가 필요합니다."

        print(f"   ℹ️  RE100 적용 가능 자재: {re100_applicable_count}개")

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

        print(f"\n🔋 System-Level RE100 제약조건 적용:")

        # 1. System-Level RE100 목표 제약
        if self.system_re100_target is not None:
            self._apply_system_re100_constraint(
                model, scenario_df, material_classification
            )

        # 2. Tier별 RE100 목표 제약
        if self.tier1_re100_target is not None:
            self._apply_tier1_re100_constraint(
                model, scenario_df, material_classification
            )

        if self.tier2_re100_target is not None:
            self._apply_tier2_re100_constraint(
                model, scenario_df, material_classification
            )

        # 3. 자재별 RE100 범위 제약
        if self.material_re100_min or self.material_re100_max:
            self._apply_material_re100_bounds(model)

        # 4. Tier1 >= Tier2 제약
        if self.enforce_tier1_higher_than_tier2:
            self._apply_tier1_higher_than_tier2_constraint(model)

    def _apply_system_re100_constraint(
        self,
        model: pyo.ConcreteModel,
        scenario_df,
        material_classification: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        시스템 전체 RE100 목표 제약 적용

        시스템 RE100 적용률 = Σ(에너지_i × RE100_i) / Σ(에너지_i) >= target
        """
        # 분자: 총 RE100 적용 에너지
        total_re100_energy = 0.0

        # 분모: 총 에너지 소비량
        total_energy = 0.0

        for material in model.materials:
            mat_info = material_classification.get(material, {})

            # 에너지 비율
            tier1_ratio = mat_info.get('tier1_energy_ratio', 0)
            tier2_ratio = mat_info.get('tier2_energy_ratio', 0)

            if tier1_ratio == 0 and tier2_ratio == 0:
                continue  # RE100 적용 불가 자재는 스킵

            # 자재 정보
            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            original_emission = material_row['배출계수']
            quantity = material_row['제품총소요량(kg)']

            # 에너지 소비량 추정 (배출계수 기반)
            # Note: 실제 electricity_usage 데이터가 있으면 더 정확
            EMISSION_TO_ENERGY_FACTOR = 2.0  # 1 kgCO2eq ≈ 2 kWh
            estimated_total_energy = original_emission * EMISSION_TO_ENERGY_FACTOR * quantity

            # Tier별 에너지
            tier1_energy = estimated_total_energy * tier1_ratio
            tier2_energy = estimated_total_energy * tier2_ratio

            # RE100 변수
            tier1_re = model.tier1_re[material]
            tier2_re = model.tier2_re[material]

            # RE100 적용 에너지
            material_re100_energy = tier1_energy * tier1_re + tier2_energy * tier2_re

            total_re100_energy += material_re100_energy
            total_energy += (tier1_energy + tier2_energy)

        # 제약조건: RE100 적용률 >= 목표
        # total_re100_energy / total_energy >= target
        # => total_re100_energy >= target * total_energy

        model.add_component(
            'system_re100_constraint',
            pyo.Constraint(
                expr=total_re100_energy >= self.system_re100_target * total_energy
            )
        )

        print(f"    ✅ System-Level RE100 목표: {self.system_re100_target*100:.1f}%")
        print(f"       제약식: Σ(RE100 에너지) / Σ(총 에너지) >= {self.system_re100_target:.3f}")

    def _apply_tier1_re100_constraint(
        self,
        model: pyo.ConcreteModel,
        scenario_df,
        material_classification: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Tier1 RE100 목표 제약 적용

        Tier1 RE100 적용률 = Σ(Tier1 에너지_i × tier1_re_i) / Σ(Tier1 에너지_i) >= target
        """
        total_tier1_re100_energy = 0.0
        total_tier1_energy = 0.0

        for material in model.materials:
            mat_info = material_classification.get(material, {})
            tier1_ratio = mat_info.get('tier1_energy_ratio', 0)

            if tier1_ratio == 0:
                continue

            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            original_emission = material_row['배출계수']
            quantity = material_row['제품총소요량(kg)']

            EMISSION_TO_ENERGY_FACTOR = 2.0
            estimated_total_energy = original_emission * EMISSION_TO_ENERGY_FACTOR * quantity
            tier1_energy = estimated_total_energy * tier1_ratio

            tier1_re = model.tier1_re[material]
            total_tier1_re100_energy += tier1_energy * tier1_re
            total_tier1_energy += tier1_energy

        if total_tier1_energy > 0:
            model.add_component(
                'tier1_re100_constraint',
                pyo.Constraint(
                    expr=total_tier1_re100_energy >= self.tier1_re100_target * total_tier1_energy
                )
            )
            print(f"    ✅ Tier1 RE100 목표: {self.tier1_re100_target*100:.1f}%")

    def _apply_tier2_re100_constraint(
        self,
        model: pyo.ConcreteModel,
        scenario_df,
        material_classification: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Tier2 RE100 목표 제약 적용
        """
        total_tier2_re100_energy = 0.0
        total_tier2_energy = 0.0

        for material in model.materials:
            mat_info = material_classification.get(material, {})
            tier2_ratio = mat_info.get('tier2_energy_ratio', 0)

            if tier2_ratio == 0:
                continue

            material_row = scenario_df[scenario_df['자재명'] == material].iloc[0]
            original_emission = material_row['배출계수']
            quantity = material_row['제품총소요량(kg)']

            EMISSION_TO_ENERGY_FACTOR = 2.0
            estimated_total_energy = original_emission * EMISSION_TO_ENERGY_FACTOR * quantity
            tier2_energy = estimated_total_energy * tier2_ratio

            tier2_re = model.tier2_re[material]
            total_tier2_re100_energy += tier2_energy * tier2_re
            total_tier2_energy += tier2_energy

        if total_tier2_energy > 0:
            model.add_component(
                'tier2_re100_constraint',
                pyo.Constraint(
                    expr=total_tier2_re100_energy >= self.tier2_re100_target * total_tier2_energy
                )
            )
            print(f"    ✅ Tier2 RE100 목표: {self.tier2_re100_target*100:.1f}%")

    def _apply_material_re100_bounds(self, model: pyo.ConcreteModel) -> None:
        """
        자재별 RE100 범위 제약 적용
        """
        applied_count = 0

        for material in model.materials:
            min_rate = self.material_re100_min.get(material)
            max_rate = self.material_re100_max.get(material)

            # Tier1 최소/최대 제약
            if min_rate is not None and min_rate > 0:
                model.add_component(
                    f'material_tier1_re100_min_{material.replace(" ", "_")}',
                    pyo.Constraint(expr=model.tier1_re[material] >= min_rate)
                )
                applied_count += 1

            if max_rate is not None and max_rate < 1.0:
                model.add_component(
                    f'material_tier1_re100_max_{material.replace(" ", "_")}',
                    pyo.Constraint(expr=model.tier1_re[material] <= max_rate)
                )
                applied_count += 1

            # Tier2 최소/최대 제약
            if min_rate is not None and min_rate > 0:
                model.add_component(
                    f'material_tier2_re100_min_{material.replace(" ", "_")}',
                    pyo.Constraint(expr=model.tier2_re[material] >= min_rate)
                )
                applied_count += 1

            if max_rate is not None and max_rate < 1.0:
                model.add_component(
                    f'material_tier2_re100_max_{material.replace(" ", "_")}',
                    pyo.Constraint(expr=model.tier2_re[material] <= max_rate)
                )
                applied_count += 1

        if applied_count > 0:
            print(f"    ✅ 자재별 RE100 범위 제약: {applied_count}개 적용")

    def _apply_tier1_higher_than_tier2_constraint(self, model: pyo.ConcreteModel) -> None:
        """
        Tier1 RE >= Tier2 RE 제약 적용

        각 자재에 대해 Tier1의 RE100 비율이 Tier2 RE100 비율보다
        높거나 같도록 제약조건을 추가합니다.

        제약식: tier1_re[material] >= tier2_re[material], ∀ material
        """
        constraint_count = 0

        for material in model.materials:
            # 제약조건 추가: tier1_re >= tier2_re
            constraint_name = f'tier1_ge_tier2_{material.replace(" ", "_").replace("(", "").replace(")", "")}'
            model.add_component(
                constraint_name,
                pyo.Constraint(
                    expr=model.tier1_re[material] >= model.tier2_re[material]
                )
            )
            constraint_count += 1

        print(f"    ✅ Tier1 >= Tier2 제약: {constraint_count}개 자재에 적용")

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        return {
            'type': 'system_re100_constraint',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'system_re100_target': self.system_re100_target,
            'tier1_re100_target': self.tier1_re100_target,
            'tier2_re100_target': self.tier2_re100_target,
            'material_re100_min': self.material_re100_min,
            'material_re100_max': self.material_re100_max,
            'use_cost_efficiency': self.use_cost_efficiency,
            'enforce_tier1_higher_than_tier2': self.enforce_tier1_higher_than_tier2
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        self.name = config.get('name', 'system_re100_constraint')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.system_re100_target = config.get('system_re100_target')
        self.tier1_re100_target = config.get('tier1_re100_target')
        self.tier2_re100_target = config.get('tier2_re100_target')
        self.material_re100_min = config.get('material_re100_min', {})
        self.material_re100_max = config.get('material_re100_max', {})
        self.use_cost_efficiency = config.get('use_cost_efficiency', True)
        self.enforce_tier1_higher_than_tier2 = config.get('enforce_tier1_higher_than_tier2', False)

    def get_summary(self) -> str:
        """
        제약조건 요약

        Returns:
            요약 문자열
        """
        base_summary = super().get_summary()
        details = []

        if self.system_re100_target is not None:
            details.append(f"System: {self.system_re100_target*100:.0f}%")

        if self.tier1_re100_target is not None:
            details.append(f"Tier1: {self.tier1_re100_target*100:.0f}%")

        if self.tier2_re100_target is not None:
            details.append(f"Tier2: {self.tier2_re100_target*100:.0f}%")

        if self.material_re100_min or self.material_re100_max:
            details.append(f"자재별 범위: {len(self.material_re100_min | self.material_re100_max)}개")

        if self.enforce_tier1_higher_than_tier2:
            details.append("Tier1 >= Tier2")

        detail_str = " | ".join(details) if details else "설정 없음"
        return f"{base_summary}\n  🔋 {detail_str}"

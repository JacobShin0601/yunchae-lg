"""
분리막 최적화 전략

Separator의 타입 비율(Dry vs Wet) 및 코팅 특성을 최적화하고 RE100을 적용합니다.
"""

from typing import Dict, List, Any, Optional
import pyomo.environ as pyo
from .material_strategy import MaterialOptimizationStrategy


class SeparatorOptimizationStrategy(MaterialOptimizationStrategy):
    """
    분리막 최적화 전략

    최적화 대상:
    - Dry Type vs Wet Type 비율
    - RE100 적용 (Tier1/Tier2)

    제약 조건:
    - Dry + Wet 비율 = 1
    - 타입별 최소/최대 비율 (성능 요구사항)
    """

    def __init__(
        self,
        material_name: str,
        material_data: Dict[str, Any],
        separator_config: Dict[str, Any]
    ):
        """
        분리막 전략 초기화

        Args:
            material_name: 자재명
            material_data: 자재 데이터 (classification에서 추출)
            separator_config: 분리막 설정
                - dry_emission: Dry Type 배출계수 (kgCO2eq/kg)
                - wet_emission: Wet Type 배출계수 (kgCO2eq/kg)
                - min_dry_ratio: Dry Type 최소 비율
                - max_dry_ratio: Dry Type 최대 비율
        """
        super().__init__(material_name, material_data)
        self.config = separator_config
        self.dry_emission = self.config.get('dry_emission', 2.5)
        self.wet_emission = self.config.get('wet_emission', 3.2)
        self.min_dry_ratio = self.config.get('min_dry_ratio', 0.0)
        self.max_dry_ratio = self.config.get('max_dry_ratio', 1.0)

    def get_optimization_type(self) -> str:
        """최적화 타입 반환"""
        return "separator_composition"

    def define_variables(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> List[str]:
        """
        분리막 최적화 변수 정의

        Returns:
            정의된 변수명 리스트
        """
        variables = []

        # Dry Type 비율 변수
        if not hasattr(model, 'separator_dry_ratio'):
            model.separator_dry_ratio = pyo.Var(
                domain=pyo.NonNegativeReals,
                bounds=(self.min_dry_ratio, self.max_dry_ratio),
                doc="Separator Dry Type 비율"
            )
            print(f"   ✅ 분리막 Dry Type 변수 추가 (범위: {self.min_dry_ratio:.1%} ~ {self.max_dry_ratio:.1%})")

        # Wet Type 비율 변수
        if not hasattr(model, 'separator_wet_ratio'):
            model.separator_wet_ratio = pyo.Var(
                domain=pyo.NonNegativeReals,
                bounds=(0, 1),
                doc="Separator Wet Type 비율"
            )
            print(f"   ✅ 분리막 Wet Type 변수 추가")

        variables.extend(['separator_dry_ratio', 'separator_wet_ratio'])

        # RE100 변수 (이미 전역으로 정의됨)
        variables.extend(['tier1_re', 'tier2_re'])

        return variables

    def add_constraints(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> None:
        """
        분리막 제약 조건 추가

        1. Dry + Wet 비율 = 1
        """
        # 이미 제약조건이 추가되어 있으면 스킵
        if hasattr(model, 'separator_ratio_sum_constraint'):
            return

        # Dry + Wet 비율 합 = 1
        def separator_sum_rule(model):
            return model.separator_dry_ratio + model.separator_wet_ratio == 1.0

        model.separator_ratio_sum_constraint = pyo.Constraint(
            rule=separator_sum_rule,
            doc="분리막 타입 비율 합 = 1"
        )

        print(f"   ✅ 분리막 비율 제약조건 추가 (Dry + Wet = 1)")

    def calculate_emission_factor(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> float:
        """
        분리막 배출계수 계산

        공식:
        1. 타입 조성 기반 배출계수 = Dry_비율 × Dry_배출계수 + Wet_비율 × Wet_배출계수
        2. RE100 감축 적용 = 기본_배출계수 × (1 - RE100_감축)

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스

        Returns:
            최적화된 배출계수
        """
        # Step 1: 타입 조성에 따른 기본 배출계수
        dry_ratio = pyo.value(model.separator_dry_ratio)
        wet_ratio = pyo.value(model.separator_wet_ratio)

        base_emission = (
            dry_ratio * self.dry_emission +
            wet_ratio * self.wet_emission
        )

        # Step 2: RE100 감축 적용
        tier1_ratio = self.material_data.get('tier1_energy_ratio', 0)
        tier2_ratio = self.material_data.get('tier2_energy_ratio', 0)

        material_name = self.material_name

        tier1_re = pyo.value(model.tier1_re[material_name])
        tier2_re = pyo.value(model.tier2_re[material_name])

        re100_reduction = tier1_re * tier1_ratio + tier2_re * tier2_ratio

        final_emission = base_emission * (1 - re100_reduction)

        return final_emission

    def extract_solution(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> Dict[str, Any]:
        """
        분리막 솔루션 추출

        Returns:
            추출된 결과 딕셔너리
        """
        # 타입 비율
        dry_ratio = pyo.value(model.separator_dry_ratio)
        wet_ratio = pyo.value(model.separator_wet_ratio)

        # RE100 비율
        tier1_re = pyo.value(model.tier1_re[self.material_name])
        tier2_re = pyo.value(model.tier2_re[self.material_name])

        # 진단 로그
        print(f"   ✓ [SEPARATOR] {self.material_name[:40]}")
        print(f"      타입 조성:")
        print(f"        • Dry Type: {dry_ratio*100:.2f}% (배출계수: {self.dry_emission:.3f} kgCO2eq/kg)")
        print(f"        • Wet Type: {wet_ratio*100:.2f}% (배출계수: {self.wet_emission:.3f} kgCO2eq/kg)")

        tier1_ratio = self.material_data.get('tier1_energy_ratio', 0)
        tier2_ratio = self.material_data.get('tier2_energy_ratio', 0)
        re100_reduction_pct = (tier1_re * tier1_ratio + tier2_re * tier2_ratio) * 100

        print(f"      RE100: tier1={tier1_re:.4f}, tier2={tier2_re:.4f} (총 감축: {re100_reduction_pct:.2f}%)")

        return {
            'dry_type_ratio': dry_ratio,
            'wet_type_ratio': wet_ratio,
            'tier1_re': tier1_re,
            'tier2_re': tier2_re,
            'tier1_energy_ratio': tier1_ratio,
            'tier2_energy_ratio': tier2_ratio,
            're100_reduction_pct': re100_reduction_pct
        }

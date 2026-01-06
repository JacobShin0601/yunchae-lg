"""
전해액 최적화 전략

Electrolyte의 용매(solvent) 조성 비율을 최적화하고 RE100을 적용합니다.
"""

from typing import Dict, List, Any, Optional
import pyomo.environ as pyo
from .material_strategy import MaterialOptimizationStrategy


class ElectrolyteOptimizationStrategy(MaterialOptimizationStrategy):
    """
    전해액 최적화 전략

    최적화 대상:
    - 용매 비율 (EC, DMC, EMC 등)
    - RE100 적용 (Tier1/Tier2)

    제약 조건:
    - 용매 비율 합 = 1
    - 각 용매별 최소/최대 비율 (안정성 및 성능 요구사항)
    """

    def __init__(
        self,
        material_name: str,
        material_data: Dict[str, Any],
        electrolyte_config: Dict[str, Any]
    ):
        """
        전해액 전략 초기화

        Args:
            material_name: 자재명
            material_data: 자재 데이터 (classification에서 추출)
            electrolyte_config: 전해액 설정
                - solvents: 용매 리스트 (예: ['EC', 'DMC', 'EMC'])
                - solvent_emissions: 용매별 배출계수 dict (예: {'EC': 3.45, 'DMC': 2.87})
                - min_ratios: 용매별 최소 비율 dict (예: {'EC': 0.2})
                - max_ratios: 용매별 최대 비율 dict (예: {'EC': 0.5})
        """
        super().__init__(material_name, material_data)
        self.config = electrolyte_config
        self.solvents = self.config.get('solvents', ['EC', 'DMC', 'EMC'])
        self.solvent_emissions = self.config.get('solvent_emissions', {})
        self.min_ratios = self.config.get('min_ratios', {})
        self.max_ratios = self.config.get('max_ratios', {})

    def get_optimization_type(self) -> str:
        """최적화 타입 반환"""
        return "electrolyte_composition"

    def define_variables(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> List[str]:
        """
        전해액 최적화 변수 정의

        Returns:
            정의된 변수명 리스트
        """
        variables = []

        # 용매 비율 변수 (각 용매별)
        for solvent in self.solvents:
            var_name = f'electrolyte_{solvent}_ratio'

            # 이미 정의되어 있으면 스킵
            if hasattr(model, var_name):
                variables.append(var_name)
                continue

            # 최소/최대 범위
            min_ratio = self.min_ratios.get(solvent, 0.0)
            max_ratio = self.max_ratios.get(solvent, 1.0)

            # 변수 정의
            setattr(
                model,
                var_name,
                pyo.Var(
                    domain=pyo.NonNegativeReals,
                    bounds=(min_ratio, max_ratio),
                    doc=f"Electrolyte {solvent} 비율"
                )
            )
            variables.append(var_name)

            print(f"   ✅ 전해액 용매 변수 추가: {solvent} (범위: {min_ratio:.1%} ~ {max_ratio:.1%})")

        # RE100 변수 (이미 전역으로 정의됨)
        variables.extend(['tier1_re', 'tier2_re'])

        return variables

    def add_constraints(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> None:
        """
        전해액 제약 조건 추가

        1. 용매 비율 합 = 1
        2. 각 용매별 최소/최대 비율 (이미 변수 bounds에 적용됨)
        """
        # 이미 제약조건이 추가되어 있으면 스킵
        if hasattr(model, 'electrolyte_ratio_sum_constraint'):
            return

        # 용매 비율 합 = 1
        def solvent_sum_rule(model):
            return sum(
                getattr(model, f'electrolyte_{solvent}_ratio')
                for solvent in self.solvents
            ) == 1.0

        model.electrolyte_ratio_sum_constraint = pyo.Constraint(
            rule=solvent_sum_rule,
            doc="전해액 용매 비율 합 = 1"
        )

        print(f"   ✅ 전해액 비율 제약조건 추가 (용매 {len(self.solvents)}개 합 = 1)")

    def calculate_emission_factor(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> float:
        """
        전해액 배출계수 계산

        공식:
        1. 용매 조성 기반 배출계수 = Σ(용매_비율 × 용매_배출계수)
        2. RE100 감축 적용 = 기본_배출계수 × (1 - RE100_감축)

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스

        Returns:
            최적화된 배출계수
        """
        # Step 1: 용매 조성에 따른 기본 배출계수
        base_emission = 0.0
        for solvent in self.solvents:
            solvent_ratio = pyo.value(getattr(model, f'electrolyte_{solvent}_ratio'))
            solvent_emission = self.solvent_emissions.get(solvent, 0)
            base_emission += solvent_ratio * solvent_emission

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
        전해액 솔루션 추출

        Returns:
            추출된 결과 딕셔너리
        """
        # 용매 비율
        solvent_ratios = {}
        for solvent in self.solvents:
            ratio = pyo.value(getattr(model, f'electrolyte_{solvent}_ratio'))
            solvent_ratios[solvent] = ratio

        # RE100 비율
        tier1_re = pyo.value(model.tier1_re[self.material_name])
        tier2_re = pyo.value(model.tier2_re[self.material_name])

        # 진단 로그
        print(f"   ✓ [ELECTROLYTE] {self.material_name[:40]}")
        print(f"      용매 조성:")
        for solvent, ratio in solvent_ratios.items():
            emission = self.solvent_emissions.get(solvent, 0)
            print(f"        • {solvent}: {ratio*100:.2f}% (배출계수: {emission:.3f} kgCO2eq/kg)")

        tier1_ratio = self.material_data.get('tier1_energy_ratio', 0)
        tier2_ratio = self.material_data.get('tier2_energy_ratio', 0)
        re100_reduction_pct = (tier1_re * tier1_ratio + tier2_re * tier2_ratio) * 100

        print(f"      RE100: tier1={tier1_re:.4f}, tier2={tier2_re:.4f} (총 감축: {re100_reduction_pct:.2f}%)")

        return {
            'solvent_ratios': solvent_ratios,
            'tier1_re': tier1_re,
            'tier2_re': tier2_re,
            'tier1_energy_ratio': tier1_ratio,
            'tier2_energy_ratio': tier2_ratio,
            're100_reduction_pct': re100_reduction_pct
        }

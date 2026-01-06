"""
일반 자재 최적화 전략

Electrolyte, Separator, Current Collector, Binder 등 일반 자재에 대한
RE100 최적화를 적용합니다. (Phase 1: RE100만 적용)
"""

from typing import Dict, List, Any
import pyomo.environ as pyo
from .material_strategy import MaterialOptimizationStrategy


class GeneralMaterialOptimizationStrategy(MaterialOptimizationStrategy):
    """
    일반 자재 최적화 전략 (Phase 1: RE100만 적용)

    대상 자재:
    - Electrolyte (전해액)
    - Separator (분리막)
    - Current Collector (집전체)
    - Binder (바인더)
    - Additives (첨가제)
    - 기타 Formula 자재

    최적화:
    - RE100 적용 (에너지 데이터가 있는 경우)
    - Phase 2에서 자재별 전용 전략으로 확장 예정
    """

    def __init__(self, material_name: str, material_data: Dict[str, Any]):
        """
        일반 자재 전략 초기화

        Args:
            material_name: 자재명
            material_data: 자재 데이터 (classification에서 추출)
                - original_emission: 원래 배출계수
                - tier1_energy_ratio: Tier1 에너지 비율 (0~1)
                - tier2_energy_ratio: Tier2 에너지 비율 (0~1)
                - quantity: 소요량
                - country: 생산 지역
        """
        super().__init__(material_name, material_data)
        self.has_re100 = self._check_re100_availability()

    def get_optimization_type(self) -> str:
        """최적화 타입 반환"""
        return "general_re100" if self.has_re100 else "none"

    def _check_re100_availability(self) -> bool:
        """
        RE100 적용 가능 여부 확인

        에너지 비율이 0.1% 이상이면 RE100 적용 가능

        Returns:
            RE100 적용 가능 여부
        """
        tier1_ratio = self.material_data.get('tier1_energy_ratio', 0)
        tier2_ratio = self.material_data.get('tier2_energy_ratio', 0)

        total_ratio = tier1_ratio + tier2_ratio

        # 0.1% 임계값 (너무 낮으면 최적화 의미 없음)
        return total_ratio > 0.001

    def define_variables(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> List[str]:
        """
        일반 자재 최적화 변수 정의

        RE100 변수는 이미 전역으로 정의되어 있으므로
        새로운 변수를 추가하지 않습니다.

        Returns:
            사용할 변수명 리스트 (RE100 변수)
        """
        if self.has_re100:
            # tier1_re, tier2_re는 전역 변수로 이미 정의됨
            return ['tier1_re', 'tier2_re']
        return []

    def add_constraints(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> None:
        """
        일반 자재 제약 조건 추가

        RE100 관련 제약은 OptimizationEngine에서 전역으로 관리하므로
        자재별 추가 제약이 없습니다.
        """
        # 일반 자재는 별도 제약이 없음
        # RE100 변수의 범위 제약 (0~1)은 이미 전역 정의에 포함
        pass

    def calculate_emission_factor(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> float:
        """
        일반 자재 배출계수 계산

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스

        Returns:
            최적화된 배출계수
        """
        base_emission = self.material_data['original_emission']

        if not self.has_re100:
            # RE100 데이터 없음 → 원래 배출계수 그대로 반환
            return base_emission

        # RE100 적용
        tier1_ratio = self.material_data.get('tier1_energy_ratio', 0)
        tier2_ratio = self.material_data.get('tier2_energy_ratio', 0)

        material_name = self.material_name

        # RE100 변수 값 가져오기
        tier1_re = pyo.value(model.tier1_re[material_name])
        tier2_re = pyo.value(model.tier2_re[material_name])

        # RE100 감축 계산
        # 공식: final_emission = base_emission × (1 - tier1_re × tier1_ratio - tier2_re × tier2_ratio)
        re100_reduction = tier1_re * tier1_ratio + tier2_re * tier2_ratio

        final_emission = base_emission * (1 - re100_reduction)

        return final_emission

    def extract_solution(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> Dict[str, Any]:
        """
        일반 자재 솔루션 추출

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스

        Returns:
            추출된 결과 딕셔너리
        """
        if not self.has_re100:
            return {}

        material_name = self.material_name

        tier1_re = pyo.value(model.tier1_re[material_name])
        tier2_re = pyo.value(model.tier2_re[material_name])

        # 진단 로그
        tier1_ratio = self.material_data.get('tier1_energy_ratio', 0)
        tier2_ratio = self.material_data.get('tier2_energy_ratio', 0)

        reduction_pct = (tier1_re * tier1_ratio + tier2_re * tier2_ratio) * 100

        print(f"   ✓ [GENERAL] {self.material_name[:40]}")
        print(f"      tier1_re={tier1_re:.4f} (ratio={tier1_ratio*100:.1f}%)")
        print(f"      tier2_re={tier2_re:.4f} (ratio={tier2_ratio*100:.1f}%)")
        print(f"      RE100 총 감축: {reduction_pct:.2f}%")

        return {
            'tier1_re': tier1_re,
            'tier2_re': tier2_re,
            'tier1_energy_ratio': tier1_ratio,
            'tier2_energy_ratio': tier2_ratio,
            're100_reduction_pct': reduction_pct
        }

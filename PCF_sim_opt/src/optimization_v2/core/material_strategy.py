"""
자재별 최적화 전략 추상 클래스

각 자재 타입(양극재, 음극재, 분리막, 전해액)에 대한
최적화 전략을 정의하는 추상 베이스 클래스입니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pyomo.environ as pyo


class MaterialOptimizationStrategy(ABC):
    """
    자재별 최적화 전략 추상 클래스

    각 자재 타입은 이 클래스를 상속받아 자재별 최적화 로직을 구현합니다.
    """

    def __init__(self, material_name: str, material_data: Dict[str, Any]):
        """
        전략 초기화

        Args:
            material_name: 자재명
            material_data: 자재 데이터 (classification에서 추출)
        """
        self.material_name = material_name
        self.material_data = material_data

    @abstractmethod
    def get_optimization_type(self) -> str:
        """
        최적화 타입 반환

        Returns:
            최적화 타입
            - 'element': Element-level 최적화 (양극재)
            - 'composition': Composition 최적화 (음극재 등)
            - 're100': RE100만 적용
            - 'hybrid': Element-level + RE100 (양극재)
        """
        pass

    @abstractmethod
    def define_variables(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> List[str]:
        """
        최적화 변수 정의

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스

        Returns:
            정의된 변수명 리스트
        """
        pass

    @abstractmethod
    def add_constraints(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> None:
        """
        제약 조건 추가

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스
        """
        pass

    @abstractmethod
    def calculate_emission_factor(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> float:
        """
        최적화된 배출계수 계산

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스

        Returns:
            최적화된 배출계수
        """
        pass

    @abstractmethod
    def extract_solution(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> Dict[str, Any]:
        """
        솔버 결과에서 해당 자재의 최적화 변수 추출

        Args:
            model: Pyomo 모델
            material_idx: 자재 인덱스

        Returns:
            추출된 결과 딕셔너리
        """
        pass

    def get_material_info(self) -> Dict[str, Any]:
        """
        자재 메타정보 반환

        Returns:
            자재 정보 딕셔너리
        """
        return {
            'name': self.material_name,
            'type': self.get_optimization_type(),
            'data': self.material_data
        }

    def __repr__(self) -> str:
        """문자열 표현"""
        return f"<{self.__class__.__name__}(material={self.material_name[:30]})>"


class DefaultOptimizationStrategy(MaterialOptimizationStrategy):
    """
    기본 최적화 전략 (최적화 없음)

    최적화 대상이 아닌 자재에 사용됩니다.
    """

    def get_optimization_type(self) -> str:
        return "none"

    def define_variables(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> List[str]:
        """변수 정의 없음"""
        return []

    def add_constraints(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> None:
        """제약 조건 없음"""
        pass

    def calculate_emission_factor(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> float:
        """원본 배출계수 그대로 반환"""
        return self.material_data.get('original_emission', 0)

    def extract_solution(
        self,
        model: pyo.ConcreteModel,
        material_idx: int
    ) -> Dict[str, Any]:
        """
        최적화 없음 - 원본 데이터만 반환

        Returns:
            원본 배출계수 및 기본 정보
        """
        return {
            'modified_emission': self.material_data.get('original_emission', 0),
            'original_emission': self.material_data.get('original_emission', 0),
            'quantity': self.material_data.get('quantity', 0),
            'type': 'General',
            'is_cathode': False,
            'reduction_pct': 0.0
        }

"""
제약조건 기반 클래스 (Abstract Base Class)

모든 제약조건 타입이 상속해야 하는 추상 클래스입니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import pyomo.environ as pyo


class ConstraintBase(ABC):
    """
    제약조건 추상 기반 클래스

    모든 제약조건(자재 관리, 비용, 위치 등)은 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, name: str, description: str = "", enabled: bool = True):
        """
        제약조건 초기화

        Args:
            name: 제약조건 고유 이름
            description: 제약조건 설명
            enabled: 활성화 여부
        """
        self.name = name
        self.description = description
        self.enabled = enabled
        self.config = {}  # 제약조건별 설정 저장

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정의 유효성 검증

        Args:
            config: 제약조건 설정 딕셔너리

        Returns:
            (is_valid, message): 유효성 여부와 메시지
        """
        pass

    @abstractmethod
    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터 (scenario_df, original_df 등)
        """
        pass

    @abstractmethod
    def check_feasibility(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        주어진 데이터에서 제약조건이 실현 가능한지 확인

        Args:
            data: 시뮬레이션 데이터

        Returns:
            (is_feasible, message): 실현 가능성 여부와 메시지
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        제약조건 설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        pass

    @abstractmethod
    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 제약조건 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        pass

    def get_summary(self) -> str:
        """
        제약조건 요약 정보 반환

        Returns:
            요약 문자열
        """
        status = "✅ 활성화" if self.enabled else "❌ 비활성화"
        return f"{status} | {self.name}: {self.description}"

    def enable(self) -> None:
        """제약조건 활성화"""
        self.enabled = True

    def disable(self) -> None:
        """제약조건 비활성화"""
        self.enabled = False

    def toggle(self) -> None:
        """제약조건 활성화 상태 토글"""
        self.enabled = not self.enabled

    def __repr__(self) -> str:
        """제약조건 문자열 표현"""
        status = "enabled" if self.enabled else "disabled"
        return f"<{self.__class__.__name__}(name='{self.name}', {status})>"

    def __str__(self) -> str:
        """제약조건 사용자 친화적 문자열"""
        return self.get_summary()

    def get_display_summary(self) -> str:
        """UI 표시용 요약 (서브클래스에서 오버라이드 가능)"""
        return self.get_summary()

    def is_feature_option_constraint(self) -> bool:
        """Feature Option 제약조건 여부 확인"""
        return self.name in ['recycling_option', 'low_carbon_option', 'site_change_option']

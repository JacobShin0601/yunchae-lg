"""
PCF 최적화를 위한 기본 최적화 클래스
모든 솔버별 최적화 클래스는 이 클래스를 상속받아 구현됩니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pyomo.environ import ConcreteModel, Var, Constraint, Objective
import pandas as pd


class BaseOptimizer(ABC):
    """
    최적화 문제를 해결하기 위한 추상 기본 클래스
    
    각 솔버별 구현체는 이 클래스를 상속받아 특정 솔버에 맞는
    모델 구축 및 해결 방법을 구현해야 합니다.
    """
    
    def __init__(self, config: Dict[str, Any], stable_var_data: Dict[str, Any]):
        """
        Args:
            config: 최적화 설정 (목적함수, 의사결정변수, 제약조건 등)
            stable_var_data: stable_var 디렉토리에서 로드한 고정 데이터
        """
        self.config = config
        self.stable_var_data = stable_var_data
        self.model = None
        self.results = None
        self.solver_name = None
        
    @abstractmethod
    def build_model(self) -> ConcreteModel:
        """
        Pyomo 모델 생성 및 변수, 제약조건, 목적함수 정의
        
        Returns:
            ConcreteModel: 구축된 Pyomo 모델
        """
        pass
        
    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        """
        최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과 (status, objective_value, variables 등)
        """
        pass
        
    def get_problem_type(self) -> str:
        """
        문제 유형 판별 (선형, 비선형, 정수 등)
        
        Returns:
            str: 문제 유형 ('linear', 'nonlinear', 'integer')
        """
        # 간단한 휴리스틱으로 문제 유형 판별
        objective_type = self.config.get('objective')
        cathode_type = self.config.get('decision_vars', {}).get('cathode', {}).get('type')
        
        # Type A는 원료구성이 변수이므로 비선형 문제
        if cathode_type == 'A':
            return 'nonlinear'
        
        # maximize_ease는 정수 계획 문제
        if objective_type == 'maximize_ease':
            return 'integer'
            
        # 기본적으로 선형 문제로 간주
        return 'linear'
        
    def extract_variables(self) -> Dict[str, float]:
        """
        모델에서 변수값 추출
        
        Returns:
            Dict: 변수명과 값의 매핑
        """
        if not self.model:
            return {}
            
        variables = {}
        for var in list(self.model.component_objects(Var, active=True)):
            var_name = str(var)
            if var.is_indexed():
                variables[var_name] = {
                    str(idx): var[idx].value for idx in var
                }
            else:
                variables[var_name] = var.value
                
        return variables
        
    def get_results(self) -> Dict[str, Any]:
        """
        최적화 결과 반환
        
        Returns:
            Dict: 최적화 결과
        """
        return self.results
        
    def validate_config(self) -> bool:
        """
        설정 유효성 검증
        
        Returns:
            bool: 유효한 설정이면 True
        """
        required_keys = ['objective', 'decision_vars', 'constraints']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"필수 설정 '{key}'가 누락되었습니다.")
                
        return True
        
    def get_reduction_vars(self) -> Dict[str, float]:
        """
        감축비율 변수 정보 반환
        
        Returns:
            Dict: Tier별 감축비율 변수
        """
        return self.config.get('decision_vars', {}).get('reduction_rates', {})
        
    def get_cathode_config(self) -> Dict[str, Any]:
        """
        양극재 구성 설정 반환
        
        Returns:
            Dict: 양극재 구성 관련 설정
        """
        return self.config.get('decision_vars', {}).get('cathode', {})
        
    def get_location(self) -> str:
        """
        생산지 정보 반환
        
        Returns:
            str: 선택된 생산지
        """
        return self.config.get('decision_vars', {}).get('location', '한국')
        
    def calculate_location_factor(self) -> float:
        """
        생산지 전력배출계수 계산
        
        Returns:
            float: 해당 생산지의 전력배출계수
        """
        location = self.get_location()
        electricity_coefs = self.stable_var_data.get('electricity_coef', {})
        return electricity_coefs.get(location, 0.637)  # 기본값은 한국
        
    def calculate_recycle_impact(self, element: str = 'Ni') -> float:
        """
        재활용재 환경영향 계수 계산
        
        Args:
            element: 재활용재 원소 (Ni, Co, Li 등)
            
        Returns:
            float: 환경영향 계수
        """
        recycle_impact = self.stable_var_data.get('recycle_impact', {}).get('재활용재', {})
        return recycle_impact.get(element, 0.1)  # 기본값 0.1
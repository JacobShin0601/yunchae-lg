"""
RE 적용 Case들 간의 비중 제약조건 관리 모듈
"""

from typing import Dict, Any, Optional, List
from pyomo.environ import ConcreteModel, Constraint

class CaseConstraintManager:
    """
    Tier 내 Case1, Case2, Case3 간의 RE 비중 제약조건을 관리하는 클래스
    
    Case1 > Case2 > Case3 순으로 RE 비중이 높아지도록 제약조건을 설정합니다.
    """
    
    def __init__(self, model: Optional[ConcreteModel] = None):
        """
        Args:
            model: Pyomo 모델 (None이면 추후에 set_model로 설정)
        """
        self.model = model
        self.case_variables = {}
        self.case_constraints = []
    
    def set_model(self, model: ConcreteModel) -> None:
        """
        모델 설정
        
        Args:
            model: Pyomo 모델
        """
        self.model = model
    
    def add_case_variable(self, tier: str, case_number: int, variable_name: str) -> None:
        """
        Case 변수 등록
        
        Args:
            tier: 티어명 (예: tier1, tier2, tier3)
            case_number: 케이스 번호 (1, 2, 3)
            variable_name: 변수명
        """
        if tier not in self.case_variables:
            self.case_variables[tier] = {}
        
        self.case_variables[tier][case_number] = variable_name
    
    def add_case_variables_from_dict(self, variables_dict: Dict[str, Dict[int, str]]) -> None:
        """
        사전에서 Case 변수 일괄 등록
        
        Args:
            variables_dict: 티어와 케이스별 변수명이 담긴 사전
            예시: {
                'tier1': {1: 'tier1_case1', 2: 'tier1_case2', 3: 'tier1_case3'},
                'tier2': {1: 'tier2_case1', 2: 'tier2_case2', 3: 'tier2_case3'}
            }
        """
        for tier, cases in variables_dict.items():
            for case_number, variable_name in cases.items():
                self.add_case_variable(tier, case_number, variable_name)
    
    def create_case_ordering_constraints(self) -> None:
        """
        모든 티어에 대해 Case1 > Case2 > Case3 순으로 제약조건 생성
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
        
        constraint_count = 1
        
        for tier, cases in self.case_variables.items():
            # 각 티어 내에서 Case1 > Case2 제약
            if 1 in cases and 2 in cases:
                var1 = cases[1]
                var2 = cases[2]
                
                if hasattr(self.model, var1) and hasattr(self.model, var2):
                    # 제약조건 이름 생성
                    constraint_name = f"case_order_{tier}_1_2"
                    
                    # 제약조건 생성 (Case1 >= Case2)
                    setattr(self.model, constraint_name, 
                           Constraint(expr=getattr(self.model, var1) >= getattr(self.model, var2)))
                    
                    self.case_constraints.append(constraint_name)
                    constraint_count += 1
            
            # 각 티어 내에서 Case2 > Case3 제약
            if 2 in cases and 3 in cases:
                var2 = cases[2]
                var3 = cases[3]
                
                if hasattr(self.model, var2) and hasattr(self.model, var3):
                    # 제약조건 이름 생성
                    constraint_name = f"case_order_{tier}_2_3"
                    
                    # 제약조건 생성 (Case2 >= Case3)
                    setattr(self.model, constraint_name, 
                           Constraint(expr=getattr(self.model, var2) >= getattr(self.model, var3)))
                    
                    self.case_constraints.append(constraint_name)
                    constraint_count += 1
            
            # Case1 > Case3 제약 (옵션)
            if 1 in cases and 3 in cases:
                var1 = cases[1]
                var3 = cases[3]
                
                if hasattr(self.model, var1) and hasattr(self.model, var3):
                    # 제약조건 이름 생성
                    constraint_name = f"case_order_{tier}_1_3"
                    
                    # 제약조건 생성 (Case1 >= Case3)
                    setattr(self.model, constraint_name, 
                           Constraint(expr=getattr(self.model, var1) >= getattr(self.model, var3)))
                    
                    self.case_constraints.append(constraint_name)
                    constraint_count += 1
        
        return constraint_count - 1
    
    def create_minimum_difference_constraints(self, min_difference: float = 0.05) -> int:
        """
        Case 간 최소 차이를 설정하는 제약조건 생성
        
        Args:
            min_difference: 케이스 간 최소 차이 비율 (기본값: 0.05 또는 5%)
            
        Returns:
            int: 생성된 제약조건 수
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
        
        constraint_count = 0
        
        for tier, cases in self.case_variables.items():
            # Case1과 Case2 간 최소 차이
            if 1 in cases and 2 in cases:
                var1 = cases[1]
                var2 = cases[2]
                
                if hasattr(self.model, var1) and hasattr(self.model, var2):
                    # 제약조건 이름 생성
                    constraint_name = f"case_diff_{tier}_1_2"
                    
                    # 제약조건 생성 (Case1 - Case2 >= min_difference)
                    setattr(self.model, constraint_name, 
                           Constraint(expr=getattr(self.model, var1) - getattr(self.model, var2) >= min_difference))
                    
                    self.case_constraints.append(constraint_name)
                    constraint_count += 1
            
            # Case2와 Case3 간 최소 차이
            if 2 in cases and 3 in cases:
                var2 = cases[2]
                var3 = cases[3]
                
                if hasattr(self.model, var2) and hasattr(self.model, var3):
                    # 제약조건 이름 생성
                    constraint_name = f"case_diff_{tier}_2_3"
                    
                    # 제약조건 생성 (Case2 - Case3 >= min_difference)
                    setattr(self.model, constraint_name, 
                           Constraint(expr=getattr(self.model, var2) - getattr(self.model, var3) >= min_difference))
                    
                    self.case_constraints.append(constraint_name)
                    constraint_count += 1
        
        return constraint_count

    def get_all_case_constraints(self) -> List[str]:
        """
        생성된 모든 Case 제약조건의 이름 반환
        
        Returns:
            List[str]: 제약조건 이름 목록
        """
        return self.case_constraints
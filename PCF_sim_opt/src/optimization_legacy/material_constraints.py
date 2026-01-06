"""
재활용재 및 저탄소 메탈 사용 비율 제약조건 관리 모듈
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pyomo.environ import ConcreteModel, Constraint, Var

class MaterialConstraintManager:
    """
    재활용재 및 저탄소 메탈 사용 비율에 대한 제약조건을 관리하는 클래스
    
    주요 기능:
    - 재활용재 사용비율 상한/하한 제약
    - 저탄소 메탈 사용비중 상한/하한 제약
    - 원료 조합 제약
    """
    
    def __init__(self, model: Optional[ConcreteModel] = None):
        """
        Args:
            model: Pyomo 모델 (None이면 추후에 set_model로 설정)
        """
        self.model = model
        self.material_constraints = []
    
    def set_model(self, model: ConcreteModel) -> None:
        """
        모델 설정
        
        Args:
            model: Pyomo 모델
        """
        self.model = model
    
    def create_recycle_ratio_constraints(self, min_ratio: float = 0.1, max_ratio: float = 0.5) -> int:
        """
        재활용재 사용비율에 대한 상한/하한 제약조건 생성
        
        Args:
            min_ratio: 최소 재활용재 사용비율 (기본값: 0.1 또는 10%)
            max_ratio: 최대 재활용재 사용비율 (기본값: 0.5 또는 50%)
            
        Returns:
            int: 생성된 제약조건 수
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        constraint_count = 0
        
        # 재활용재 비율 변수가 있는지 확인
        if not hasattr(self.model, 'recycle_ratio'):
            return 0
        
        # 하한 제약 추가
        constraint_name = 'recycle_ratio_min'
        setattr(self.model, constraint_name,
                Constraint(expr=self.model.recycle_ratio >= min_ratio))
        self.material_constraints.append(constraint_name)
        constraint_count += 1
        
        # 상한 제약 추가
        constraint_name = 'recycle_ratio_max'
        setattr(self.model, constraint_name,
                Constraint(expr=self.model.recycle_ratio <= max_ratio))
        self.material_constraints.append(constraint_name)
        constraint_count += 1
        
        return constraint_count
    
    def create_low_carbon_ratio_constraints(self, min_ratio: float = 0.05, max_ratio: float = 0.3) -> int:
        """
        저탄소 메탈 사용비율에 대한 상한/하한 제약조건 생성
        
        Args:
            min_ratio: 최소 저탄소 메탈 사용비율 (기본값: 0.05 또는 5%)
            max_ratio: 최대 저탄소 메탈 사용비율 (기본값: 0.3 또는 30%)
            
        Returns:
            int: 생성된 제약조건 수
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        constraint_count = 0
        
        # 저탄소 메탈 비율 변수가 있는지 확인
        if not hasattr(self.model, 'low_carbon_ratio'):
            return 0
        
        # 하한 제약 추가
        constraint_name = 'low_carbon_ratio_min'
        setattr(self.model, constraint_name,
                Constraint(expr=self.model.low_carbon_ratio >= min_ratio))
        self.material_constraints.append(constraint_name)
        constraint_count += 1
        
        # 상한 제약 추가
        constraint_name = 'low_carbon_ratio_max'
        setattr(self.model, constraint_name,
                Constraint(expr=self.model.low_carbon_ratio <= max_ratio))
        self.material_constraints.append(constraint_name)
        constraint_count += 1
        
        return constraint_count
    
    def create_material_balance_constraint(self, max_total_ratio: float = 0.7) -> int:
        """
        재활용재와 저탄소 메탈 비율 합에 대한 제약조건 생성
        
        Args:
            max_total_ratio: 두 비율의 합 최대값 (기본값: 0.7 또는 70%)
            
        Returns:
            int: 생성된 제약조건 수
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        # 두 변수 모두 있는지 확인
        if not (hasattr(self.model, 'recycle_ratio') and hasattr(self.model, 'low_carbon_ratio')):
            return 0
        
        # 두 비율의 합 제약 추가
        constraint_name = 'material_total_ratio_max'
        setattr(self.model, constraint_name,
                Constraint(expr=self.model.recycle_ratio + self.model.low_carbon_ratio <= max_total_ratio))
        self.material_constraints.append(constraint_name)
        
        return 1
    
    def create_material_proportion_constraints(self, proportions: Dict[str, float]) -> int:
        """
        원료 간 비율 제약조건 생성
        
        Args:
            proportions: 원료 간 비율 사전 (예: {'recycle_to_low_carbon': 2.0} - 재활용재 비율이 저탄소 메탈의 2배)
            
        Returns:
            int: 생성된 제약조건 수
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        constraint_count = 0
        
        # 재활용재 대 저탄소 메탈 비율
        if 'recycle_to_low_carbon' in proportions and hasattr(self.model, 'recycle_ratio') and hasattr(self.model, 'low_carbon_ratio'):
            ratio = proportions['recycle_to_low_carbon']
            
            if ratio > 0:
                # recycle_ratio = ratio * low_carbon_ratio
                constraint_name = 'recycle_to_low_carbon_ratio'
                setattr(self.model, constraint_name,
                        Constraint(expr=self.model.recycle_ratio == ratio * self.model.low_carbon_ratio))
                self.material_constraints.append(constraint_name)
                constraint_count += 1
        
        # 기타 비율 제약 추가 가능
        
        return constraint_count
    
    def get_all_material_constraints(self) -> List[str]:
        """
        생성된 모든 원료 제약조건의 이름 반환
        
        Returns:
            List[str]: 제약조건 이름 목록
        """
        return self.material_constraints
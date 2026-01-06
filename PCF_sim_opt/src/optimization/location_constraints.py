"""
자재 생산국가 제약조건 관리 모듈
"""

from typing import Dict, Any, Optional, List, Set
from pyomo.environ import ConcreteModel, Constraint, Var

class LocationConstraintManager:
    """
    자재별 생산국가 제약조건을 관리하는 클래스
    
    특정 자재의 생산국가를 제한하거나 지정할 수 있는 기능을 제공합니다.
    """
    
    def __init__(self, model: Optional[ConcreteModel] = None):
        """
        Args:
            model: Pyomo 모델 (None이면 추후에 set_model로 설정)
        """
        self.model = model
        self.location_variables = {}
        self.location_constraints = []
        self.material_location_mapping = {}
    
    def set_model(self, model: ConcreteModel) -> None:
        """
        모델 설정
        
        Args:
            model: Pyomo 모델
        """
        self.model = model
    
    def add_location_variable(self, material: str, variable_name: str) -> None:
        """
        위치 변수 등록
        
        Args:
            material: 자재명
            variable_name: 변수명
        """
        self.location_variables[material] = variable_name
    
    def set_material_location_constraint(self, material: str, allowed_locations: List[str]) -> None:
        """
        자재별 허용 위치 설정
        
        Args:
            material: 자재명
            allowed_locations: 허용된 생산국가 목록
        """
        self.material_location_mapping[material] = set(allowed_locations)
    
    def set_material_location_constraints_from_dict(self, constraints_dict: Dict[str, List[str]]) -> None:
        """
        사전에서 자재별 허용 위치 일괄 설정
        
        Args:
            constraints_dict: 자재별 허용 위치 사전
            예시: {
                '양극재': ['한국', '중국', '일본'],
                '분리막': ['한국', '폴란드']
            }
        """
        for material, locations in constraints_dict.items():
            self.set_material_location_constraint(material, locations)
    
    def create_location_constraints(self) -> int:
        """
        자재별 생산국가 제약조건 생성
        
        Returns:
            int: 생성된 제약조건 수
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
        
        constraint_count = 0
        
        for material, locations in self.material_location_mapping.items():
            if material not in self.location_variables:
                continue
            
            variable_name = self.location_variables[material]
            if not hasattr(self.model, variable_name):
                continue
            
            # 제약조건 이름 생성
            constraint_name = f"location_{material}"
            
            # 허용된 위치 목록 확인
            location_var = getattr(self.model, variable_name)
            
            # 제약조건 생성
            if isinstance(location_var, Var):
                # 위치 변수가 이산 변수인 경우 (예: Binary)
                if hasattr(location_var, 'domain') and location_var.domain in ['Binary', 'Integer']:
                    def location_rule(model):
                        return location_var in locations
                    
                    setattr(self.model, constraint_name, Constraint(rule=location_rule))
                    self.location_constraints.append(constraint_name)
                    constraint_count += 1
                    
                # 위치 변수가 연속 변수인 경우 (위치 인덱스)
                else:
                    # 위치를 숫자 인덱스로 매핑하는 사전 생성
                    location_indices = {loc: i for i, loc in enumerate(locations)}
                    
                    def location_range_rule(model):
                        return (location_var >= 0) & (location_var < len(locations))
                    
                    setattr(self.model, constraint_name, Constraint(rule=location_range_rule))
                    self.location_constraints.append(constraint_name)
                    constraint_count += 1
        
        return constraint_count
    
    def create_material_location_constraints(self) -> int:
        """
        자재별 생산국가 고정 제약조건 생성
        
        Returns:
            int: 생성된 제약조건 수
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
        
        constraint_count = 0
        
        # 모든 자재-위치 조합에 대한 변수 이름 형식: material_location
        material_location_vars = {}
        
        # 모델의 모든 변수 중 자재-위치 변수 찾기
        for var_name in dir(self.model):
            if not var_name.startswith('_') and '_location_' in var_name:
                parts = var_name.split('_location_')
                if len(parts) == 2:
                    material = parts[0]
                    location = parts[1]
                    
                    if material not in material_location_vars:
                        material_location_vars[material] = []
                        
                    material_location_vars[material].append((location, var_name))
        
        # 자재별 제약조건 생성
        for material, locations in self.material_location_mapping.items():
            if material not in material_location_vars:
                continue
                
            # 해당 자재에 대한 모든 위치 변수
            material_vars = material_location_vars[material]
            
            # 허용된 위치만 1, 나머지는 0으로 설정
            for location, var_name in material_vars:
                if hasattr(self.model, var_name):
                    # 제약조건 이름 생성
                    constraint_name = f"location_constraint_{var_name}"
                    
                    if location in locations:
                        # 허용된 위치: 값 = 1
                        setattr(self.model, constraint_name,
                               Constraint(expr=getattr(self.model, var_name) == 1))
                    else:
                        # 허용되지 않은 위치: 값 = 0
                        setattr(self.model, constraint_name,
                               Constraint(expr=getattr(self.model, var_name) == 0))
                    
                    self.location_constraints.append(constraint_name)
                    constraint_count += 1
        
        return constraint_count
    
    def get_all_location_constraints(self) -> List[str]:
        """
        생성된 모든 위치 제약조건의 이름 반환
        
        Returns:
            List[str]: 제약조건 이름 목록
        """
        return self.location_constraints
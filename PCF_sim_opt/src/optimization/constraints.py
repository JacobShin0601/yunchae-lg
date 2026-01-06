"""
최적화 모델의 제약조건을 전문적으로 관리하는 모듈
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
from pyomo.environ import (
    ConcreteModel, Constraint, Var, Expression
)
from .constant import OptimizationConstants
from .variable import OptimizationVariables
from .input import OptimizationInput


class ConstraintManager:
    """
    최적화 모델의 제약조건을 정의하고 관리하는 클래스
    
    주요 기능:
    - 다양한 제약조건 생성 및 등록
    - 제약조건 활성화/비활성화
    - 제약조건 그룹 관리
    """
    
    def __init__(self, 
                opt_input: OptimizationInput,
                model: Optional[ConcreteModel] = None):
        """
        Args:
            opt_input: 최적화 입력 객체
            model: Pyomo 모델 (None이면 새로 생성)
        """
        self.opt_input = opt_input
        self.constants = opt_input.get_constants()
        self.variables = opt_input.get_variables()
        self.stable_var_data = opt_input.get_stable_var_data()
        self.model = model
        
        # 제약조건 레지스트리
        self._constraint_registry = {}
        self._constraint_groups = {}
    
    def set_model(self, model: ConcreteModel) -> None:
        """
        모델 설정
        
        Args:
            model: Pyomo 모델
        """
        self.model = model
    
    def register_constraint(self, name: str, constraint_func: Callable, group: Optional[str] = None) -> None:
        """
        제약조건 등록
        
        Args:
            name: 제약조건 이름
            constraint_func: 제약조건 생성 함수
            group: 제약조건 그룹 (선택 사항)
        """
        self._constraint_registry[name] = constraint_func
        
        if group:
            if group not in self._constraint_groups:
                self._constraint_groups[group] = []
            self._constraint_groups[group].append(name)
    
    def apply_all_constraints(self) -> None:
        """모든 등록된 제약조건 적용"""
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        for name, constraint_func in self._constraint_registry.items():
            constraint_func(self.model)
    
    def apply_constraint(self, name: str) -> None:
        """
        특정 제약조건 적용
        
        Args:
            name: 제약조건 이름
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        if name not in self._constraint_registry:
            raise ValueError(f"제약조건 '{name}'이 등록되지 않았습니다.")
            
        self._constraint_registry[name](self.model)
    
    def apply_constraint_group(self, group: str) -> None:
        """
        제약조건 그룹 적용
        
        Args:
            group: 제약조건 그룹
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        if group not in self._constraint_groups:
            raise ValueError(f"제약조건 그룹 '{group}'이 존재하지 않습니다.")
            
        for name in self._constraint_groups[group]:
            self._constraint_registry[name](self.model)
    
    def get_available_constraints(self) -> List[str]:
        """
        사용 가능한 모든 제약조건 이름 반환
        
        Returns:
            List[str]: 제약조건 이름 목록
        """
        return list(self._constraint_registry.keys())
    
    def get_available_groups(self) -> List[str]:
        """
        사용 가능한 모든 제약조건 그룹 반환
        
        Returns:
            List[str]: 제약조건 그룹 목록
        """
        return list(self._constraint_groups.keys())
    
    def register_standard_constraints(self) -> None:
        """표준 제약조건 등록"""
        # 1. 비율 제약조건
        self.register_constraint(
            'ratio_sum_constraint',
            self._create_ratio_constraint,
            'cathode'
        )
        
        # 2. 탄소발자국 제약조건
        self.register_constraint(
            'carbon_constraint',
            self._create_carbon_constraint,
            'environmental'
        )
        
        # 3. 최대 활동 수 제약조건
        self.register_constraint(
            'max_activities_constraint',
            self._create_max_activities_constraint,
            'operational'
        )
        
        # 4. 비용 제약조건
        self.register_constraint(
            'budget_constraint',
            self._create_budget_constraint,
            'economic'
        )
        
        # 5. 희귀 금속 제한 제약조건
        self.register_constraint(
            'nickel_constraint',
            self._create_nickel_constraint,
            'resource'
        )
        
        # 6. 최소 재활용재 비율 제약조건
        self.register_constraint(
            'min_recycle_constraint',
            self._create_min_recycle_constraint,
            'environmental'
        )
    
    def _create_ratio_constraint(self, model: ConcreteModel) -> None:
        """
        양극재 비율 합 제약조건 생성
        
        Args:
            model: Pyomo 모델
        """
        cathode_type = self.variables.get_cathode_type()
        
        if cathode_type == 'B':
            @model.Constraint()
            def ratio_sum_constraint(m):
                return m.recycle_ratio + m.low_carbon_ratio <= 1
    
    def _create_carbon_constraint(self, model: ConcreteModel) -> None:
        """
        탄소발자국 제약조건 생성
        
        Args:
            model: Pyomo 모델
        """
        target_carbon = self.opt_input.get_constraint('target_carbon')
        objective_type = self.opt_input.get_objective()
        
        # 목적함수가 탄소발자국 최소화인 경우에는 제약조건 불필요
        if objective_type == 'minimize_carbon':
            return
        
        if target_carbon is not None:
            cathode_config = self.variables.get_cathode_config()
            cathode_type = self.variables.get_cathode_type()
            reduction_vars = self.variables.get_reduction_variables()
            base_emission = self.constants.get('base_emission', 80)
            location_factor = self.constants.get_location_factor(
                self.opt_input.config.get('decision_vars', {}).get('location', '한국')
            )
            
            @model.Constraint()
            def carbon_constraint(m):
                # 기본 배출량
                base = base_emission * location_factor
                
                # Tier별 감축 효과
                reduction_effect = sum(
                    getattr(m, var_name) * self.constants.get('reduction_effect_coefficient', 0.1)
                    for var_name in reduction_vars.keys()
                    if hasattr(m, var_name)
                )
                
                # 양극재 효과 계산
                if cathode_type == 'A':
                    cathode_effect = m.low_carbon_emission * m.low_carbon_ratio
                    recycle_effect = m.recycle_ratio * (1 - self.constants.get_recycle_impact())
                else:
                    cathode_effect = m.low_carbon_emission * m.low_carbon_ratio
                    recycle_effect = m.recycle_ratio * (1 - self.constants.get_recycle_impact())
                
                # 총 배출량
                total_emission = base - reduction_effect - cathode_effect - recycle_effect
                
                return total_emission <= target_carbon
    
    def _create_max_activities_constraint(self, model: ConcreteModel) -> None:
        """
        최대 활동 수 제약조건 생성
        
        Args:
            model: Pyomo 모델
        """
        max_activities = self.opt_input.get_constraint('max_activities')
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        if max_activities is not None and use_binary:
            reduction_vars = self.variables.get_reduction_variables()
            
            @model.Constraint()
            def max_activities_constraint(m):
                active_vars = [getattr(m, f"{var_name}_active") 
                              for var_name in reduction_vars.keys() 
                              if hasattr(m, f"{var_name}_active")]
                if active_vars:
                    return sum(active_vars) <= max_activities
                else:
                    return Constraint.Skip
    
    def _create_budget_constraint(self, model: ConcreteModel) -> None:
        """
        비용 제약조건 생성
        
        Args:
            model: Pyomo 모델
        """
        max_cost = self.opt_input.get_constraint('max_cost')
        
        if max_cost is not None:
            reduction_vars = self.variables.get_reduction_variables()
            cathode_config = self.variables.get_cathode_config()
            cathode_type = self.variables.get_cathode_type()
            use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
            
            @model.Constraint()
            def budget_constraint(m):
                # 감축 비율에 따른 비용
                if use_binary:
                    # 이진 변수가 있는 경우 고정 비용 + 가변 비용
                    fixed_costs = sum(
                        getattr(m, f"{var_name}_active") * self.constants.get_activity_cost(var_name)
                        for var_name in reduction_vars.keys()
                        if hasattr(m, f"{var_name}_active")
                    )
                    
                    variable_costs = sum(
                        getattr(m, var_name) * self.constants.get('variable_cost_per_percent', 50)
                        for var_name in reduction_vars.keys()
                    )
                    
                    reduction_cost = fixed_costs + variable_costs
                else:
                    # 단순 비율 기반 비용
                    reduction_cost = sum(
                        getattr(m, var_name) * self.constants.get('variable_cost_per_percent', 50)
                        for var_name in reduction_vars.keys()
                    )
                
                # 양극재 구성에 따른 비용
                if cathode_type == 'B':
                    material_cost = (
                        m.recycle_ratio * self.constants.get('material_costs.recycle_material_cost', 500) +
                        m.low_carbon_ratio * self.constants.get('material_costs.low_carbon_material_cost', 800)
                    )
                else:
                    material_cost = (
                        m.recycle_ratio * self.constants.get('material_costs.recycle_material_cost', 500) +
                        m.low_carbon_ratio * self.constants.get('material_costs.low_carbon_material_cost', 800)
                    )
                
                total_cost = reduction_cost + material_cost
                return total_cost <= max_cost
    
    def _create_nickel_constraint(self, model: ConcreteModel) -> None:
        """
        니켈 관련 제약조건 생성
        
        Args:
            model: Pyomo 모델
        """
        ni_max = self.opt_input.get_constraint('ni_max')
        
        if ni_max is not None:
            reduction_vars = self.variables.get_reduction_variables()
            tier3_vars = [var for var in reduction_vars.keys() if 'tier3' in var]
            
            if tier3_vars:
                @model.Constraint()
                def nickel_constraint(m):
                    return sum(
                        getattr(m, var_name) for var_name in tier3_vars
                    ) <= ni_max
    
    def _create_min_recycle_constraint(self, model: ConcreteModel) -> None:
        """
        최소 재활용재 비율 제약조건 생성
        
        Args:
            model: Pyomo 모델
        """
        min_recycle = self.opt_input.get_constraint('min_recycle')
        cathode_type = self.variables.get_cathode_type()
        
        if min_recycle is not None and cathode_type == 'B':
            @model.Constraint()
            def min_recycle_constraint(m):
                return m.recycle_ratio >= min_recycle / 100  # 퍼센트로 입력된 값을 비율로 변환
    
    def create_custom_constraint(self, name: str, constraint_expr: Callable, group: Optional[str] = None) -> None:
        """
        사용자 정의 제약조건 생성 및 등록
        
        Args:
            name: 제약조건 이름
            constraint_expr: 제약조건 표현식을 생성하는 함수
            group: 제약조건 그룹 (선택 사항)
        """
        def constraint_func(model):
            setattr(model, name, Constraint(expr=constraint_expr(model)))
            
        self.register_constraint(name, constraint_func, group)
    
    def define_all_constraints(self) -> None:
        """
        모든 표준 제약조건 등록 및 적용
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        # 표준 제약조건 등록
        self.register_standard_constraints()
        
        # 모든 제약조건 적용
        self.apply_all_constraints()
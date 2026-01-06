"""
최적화 모델의 수식을 정의하는 클래스
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, 
    minimize, maximize, Expression
)
from .constant import OptimizationConstants
from .variable import OptimizationVariables
from .input import OptimizationInput


class OptimizationFormula:
    """
    최적화 모델의 제약조건과 목적함수를 정의하는 클래스
    
    주요 기능:
    - 목적함수 정의 (탄소발자국 최소화, 비용 최소화, 용이성 최대화)
    - 제약조건 정의 (탄소발자국 제한, 최대 활동 수 제한, 예산 제한 등)
    - 다양한 문제 유형 지원 (선형, 비선형, 정수)
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
        
        if model is None:
            self.model = ConcreteModel()
        else:
            self.model = model
    
    def define_objective(self) -> None:
        """모델에 목적함수 정의"""
        objective_type = self.opt_input.get_objective()
        
        # 기존 목적함수 삭제 (재정의를 위해)
        if hasattr(self.model, 'objective'):
            self.model.del_component('objective')
        
        if objective_type == 'minimize_carbon':
            self._define_minimize_carbon_objective()
        elif objective_type == 'minimize_cost':
            self._define_minimize_cost_objective()
        elif objective_type == 'maximize_ease':
            self._define_maximize_ease_objective()
        elif objective_type == 'multi_objective':
            self._define_multi_objective()
        else:
            raise ValueError(f"지원하지 않는 목적함수 유형: {objective_type}")
    
    def _define_minimize_carbon_objective(self) -> None:
        """탄소발자국 최소화 목적함수 정의"""
        # 필요한 데이터 준비
        cathode_config = self.variables.get_cathode_config()
        cathode_type = self.variables.get_cathode_type()
        reduction_vars = self.variables.get_reduction_variables()
        base_emission = self.constants.get('base_emission', 80)
        location_factor = self.constants.get_location_factor(
            self.opt_input.config.get('decision_vars', {}).get('location', '한국')
        )
        
        @self.model.Objective(sense=minimize)
        def objective(m):
            # 기본 배출량
            base = base_emission * location_factor
            
            # Tier별 감축 효과
            reduction_effect = sum(
                getattr(m, var_name) * self.constants.get('reduction_effect_coefficient', 0.1)
                for var_name in reduction_vars.keys()
            )
            
            # 양극재 효과 계산
            if cathode_type == 'A':
                # Type A: 원료구성이 변수, 비율은 고정
                cathode_effect = m.low_carbon_emission * m.low_carbon_ratio
                recycle_effect = m.recycle_ratio * (1 - self.constants.get_recycle_impact())
            else:
                # Type B: 원료구성이 고정, 비율이 변수
                cathode_effect = m.low_carbon_emission * m.low_carbon_ratio
                recycle_effect = m.recycle_ratio * (1 - self.constants.get_recycle_impact())
            
            # 총 배출량
            total_emission = base - reduction_effect - cathode_effect - recycle_effect
            return total_emission
    
    def _define_minimize_cost_objective(self) -> None:
        """비용 최소화 목적함수 정의"""
        reduction_vars = self.variables.get_reduction_variables()
        cathode_config = self.variables.get_cathode_config()
        cathode_type = self.variables.get_cathode_type()
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        @self.model.Objective(sense=minimize)
        def objective(m):
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
            
            return reduction_cost + material_cost
    
    def _define_maximize_ease_objective(self) -> None:
        """구현 용이성 최대화 목적함수 정의 (활동 수 최소화)"""
        reduction_vars = self.variables.get_reduction_variables()
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        @self.model.Objective(sense=minimize)
        def objective(m):
            if use_binary:
                # 이진 변수 사용 - 활성화된 활동 수 최소화
                return sum(
                    getattr(m, f"{var_name}_active")
                    for var_name in reduction_vars.keys()
                    if hasattr(m, f"{var_name}_active")
                )
            else:
                # 이진 변수 없음 - 근사 방법 사용
                # 작은 상수를 더해 0으로 나누기 방지
                return sum(
                    getattr(m, var_name) / (getattr(m, var_name) + 0.001)
                    for var_name in reduction_vars.keys()
                )
    
    def _define_multi_objective(self) -> None:
        """다목적 최적화 목적함수 정의 (가중합 방식)"""
        # 가중치 설정
        weights = self.opt_input.config.get('objective_options', {}).get('multi_objective_weights', {})
        carbon_weight = weights.get('carbon', 0.7)
        cost_weight = weights.get('cost', 0.3)
        
        # 별도 모델에서 개별 목적함수 계산을 위한 임시 모델
        temp_model = ConcreteModel()
        
        # 탄소발자국 목적함수
        carbon_obj = self._create_carbon_objective_expr(self.model)
        
        # 비용 목적함수
        cost_obj = self._create_cost_objective_expr(self.model)
        
        # 정규화를 위한 참조값 (대략적 스케일)
        carbon_ref = 80.0  # 기본 배출량 참조
        cost_ref = 100000.0  # 비용 참조값
        
        @self.model.Objective(sense=minimize)
        def objective(m):
            normalized_carbon = carbon_obj(m) / carbon_ref
            normalized_cost = cost_obj(m) / cost_ref
            return carbon_weight * normalized_carbon + cost_weight * normalized_cost
    
    def _create_carbon_objective_expr(self, model: ConcreteModel) -> Callable:
        """탄소발자국 목적함수 표현식 생성"""
        cathode_config = self.variables.get_cathode_config()
        cathode_type = self.variables.get_cathode_type()
        reduction_vars = self.variables.get_reduction_variables()
        base_emission = self.constants.get('base_emission', 80)
        location_factor = self.constants.get_location_factor(
            self.opt_input.config.get('decision_vars', {}).get('location', '한국')
        )
        
        def carbon_obj_expr(m):
            # 기본 배출량
            base = base_emission * location_factor
            
            # Tier별 감축 효과
            reduction_effect = sum(
                getattr(m, var_name) * self.constants.get('reduction_effect_coefficient', 0.1)
                for var_name in reduction_vars.keys()
            )
            
            # 양극재 효과 계산
            if cathode_type == 'A':
                cathode_effect = m.low_carbon_emission * m.low_carbon_ratio
                recycle_effect = m.recycle_ratio * (1 - self.constants.get_recycle_impact())
            else:
                cathode_effect = m.low_carbon_emission * m.low_carbon_ratio
                recycle_effect = m.recycle_ratio * (1 - self.constants.get_recycle_impact())
            
            return base - reduction_effect - cathode_effect - recycle_effect
        
        return carbon_obj_expr
    
    def _create_cost_objective_expr(self, model: ConcreteModel) -> Callable:
        """비용 목적함수 표현식 생성"""
        reduction_vars = self.variables.get_reduction_variables()
        cathode_config = self.variables.get_cathode_config()
        cathode_type = self.variables.get_cathode_type()
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        def cost_obj_expr(m):
            # 감축 비율에 따른 비용
            if use_binary:
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
            
            return reduction_cost + material_cost
        
        return cost_obj_expr
    
    def define_constraints(self) -> None:
        """모델에 제약조건 정의"""
        # 1. 비율 제약조건 (Type B일 때만)
        self._define_ratio_constraint()
        
        # 2. 탄소발자국 제약조건
        self._define_carbon_constraint()
        
        # 3. 최대 활동 수 제약 (선택적)
        self._define_max_activities_constraint()
        
        # 4. 비용 제약 (선택적)
        self._define_budget_constraint()
        
        # 5. 니켈 제약 (선택적)
        self._define_nickel_constraint()
    
    def _define_ratio_constraint(self) -> None:
        """양극재 비율 제약조건 (Type B일 때만)"""
        cathode_type = self.variables.get_cathode_type()
        
        if cathode_type == 'B':
            @self.model.Constraint()
            def ratio_sum_constraint(m):
                return m.recycle_ratio + m.low_carbon_ratio <= 1
    
    def _define_carbon_constraint(self) -> None:
        """탄소발자국 제약조건"""
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
            
            @self.model.Constraint()
            def carbon_constraint(m):
                # 기본 배출량
                base = base_emission * location_factor
                
                # Tier별 감축 효과
                reduction_effect = sum(
                    getattr(m, var_name) * self.constants.get('reduction_effect_coefficient', 0.1)
                    for var_name in reduction_vars.keys()
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
    
    def _define_max_activities_constraint(self) -> None:
        """최대 활동 수 제약조건 (선택적)"""
        max_activities = self.opt_input.get_constraint('max_activities')
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        if max_activities is not None and use_binary:
            reduction_vars = self.variables.get_reduction_variables()
            
            @self.model.Constraint()
            def max_activities_constraint(m):
                return sum(
                    getattr(m, f"{var_name}_active")
                    for var_name in reduction_vars.keys()
                    if hasattr(m, f"{var_name}_active")
                ) <= max_activities
    
    def _define_budget_constraint(self) -> None:
        """비용 제약조건 (선택적)"""
        max_cost = self.opt_input.get_constraint('max_cost')
        
        if max_cost is not None:
            cost_expr = self._create_cost_objective_expr(self.model)
            
            @self.model.Constraint()
            def budget_constraint(m):
                return cost_expr(m) <= max_cost
    
    def _define_nickel_constraint(self) -> None:
        """니켈 관련 제약조건 (선택적)"""
        ni_max = self.opt_input.get_constraint('ni_max')
        
        if ni_max is not None:
            reduction_vars = self.variables.get_reduction_variables()
            tier3_vars = [var for var in reduction_vars.keys() if 'tier3' in var]
            
            if tier3_vars:
                @self.model.Constraint()
                def nickel_constraint(m):
                    return sum(
                        getattr(m, var_name) for var_name in tier3_vars
                    ) <= ni_max
    
    def calculate_carbon_footprint(self, variables: Dict[str, float]) -> float:
        """
        주어진 변수값으로 탄소발자국 계산
        
        Args:
            variables: 변수값 딕셔너리
            
        Returns:
            float: 계산된 탄소발자국
        """
        try:
            print(f"DEBUG: calculate_carbon_footprint 시작, variables={variables}")
            
            # 기본 배출량
            base_emission = self.constants.get('base_emission', 80)
            print(f"DEBUG: base_emission={base_emission}")
            
            # 생산지 전력배출계수
            location = self.opt_input.config.get('decision_vars', {}).get('location', '한국')
            location_factor = self.constants.get_location_factor(location)
            print(f"DEBUG: location={location}, location_factor={location_factor}")
            
            # Tier별 감축 효과
            reduction_effect = 0
            reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
            for var_name in reduction_vars.keys():
                if var_name in variables:
                    reduction_effect += variables[var_name] * self.constants.get('reduction_effect_coefficient', 0.1)
            print(f"DEBUG: reduction_effect={reduction_effect}")
            
            # 양극재 효과
            cathode_config = self.variables.get_cathode_config()
            cathode_type = self.variables.get_cathode_type()
            print(f"DEBUG: cathode_type={cathode_type}")
            
            if cathode_type == 'A':
                low_carbon_emission = variables.get('low_carbon_emission', 10)
                low_carbon_ratio = cathode_config.get('type_A_config', {}).get('low_carbon_ratio_fixed', 0.1)
                recycle_ratio = cathode_config.get('type_A_config', {}).get('recycle_ratio_fixed', 0.2)
            else:
                low_carbon_emission = cathode_config.get('type_B_config', {}).get('emission_fixed', 10)
                low_carbon_ratio = variables.get('low_carbon_ratio', 0.1)
                recycle_ratio = variables.get('recycle_ratio', 0.2)
            
            print(f"DEBUG: low_carbon_emission={low_carbon_emission}, low_carbon_ratio={low_carbon_ratio}, recycle_ratio={recycle_ratio}")
            
            cathode_effect = low_carbon_emission * low_carbon_ratio
            recycle_effect = recycle_ratio * (1 - self.constants.get_recycle_impact())
            print(f"DEBUG: cathode_effect={cathode_effect}, recycle_effect={recycle_effect}")
            
            # 총 배출량
            total_emission = (base_emission * location_factor) - reduction_effect - cathode_effect - recycle_effect
            print(f"DEBUG: 계산된 total_emission={total_emission}")
            
            return float(total_emission)  # 명시적으로 float 타입 반환
        except Exception as e:
            print(f"ERROR: calculate_carbon_footprint 예외 발생: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # 오류 발생시 기본값 반환
    
    def calculate_total_cost(self, variables: Dict[str, float]) -> float:
        """
        주어진 변수값으로 총 비용 계산
        
        Args:
            variables: 변수값 딕셔너리
            
        Returns:
            float: 계산된 총 비용
        """
        # 감축 비용
        reduction_cost = 0
        reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
        
        for var_name in reduction_vars.keys():
            if var_name in variables:
                # 가변 비용 (감축 비율에 따른)
                reduction_cost += variables[var_name] * self.constants.get('variable_cost_per_percent', 50)
                
                # 고정 비용 (활성화 여부에 따른)
                if f"{var_name}_active" in variables and variables[f"{var_name}_active"]:
                    reduction_cost += self.constants.get_activity_cost(var_name)
        
        # 재료 비용
        cathode_config = self.variables.get_cathode_config()
        cathode_type = self.variables.get_cathode_type()
        
        if cathode_type == 'A':
            recycle_ratio = cathode_config.get('type_A_config', {}).get('recycle_ratio_fixed', 0.2)
            low_carbon_ratio = cathode_config.get('type_A_config', {}).get('low_carbon_ratio_fixed', 0.1)
        else:
            recycle_ratio = variables.get('recycle_ratio', 0.2)
            low_carbon_ratio = variables.get('low_carbon_ratio', 0.1)
        
        material_cost = (
            recycle_ratio * self.constants.get('material_costs.recycle_material_cost', 500) +
            low_carbon_ratio * self.constants.get('material_costs.low_carbon_material_cost', 800)
        )
        
        return reduction_cost + material_cost
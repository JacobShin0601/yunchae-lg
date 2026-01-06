"""
CBC 솔버를 사용한 정수 계획법 최적화 구현
주로 이진 변수나 정수 변수가 포함된 문제에 사용됩니다.
"""

from typing import Dict, Any
from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, Binary, 
    minimize, value, TerminationCondition, summation
)
from pyomo.opt import SolverFactory
from .base_optimizer import BaseOptimizer


class CBCOptimizer(BaseOptimizer):
    """
    CBC(COIN-OR Branch and Cut)를 사용한 정수 계획 최적화 클래스
    
    CBC는 혼합 정수 선형 계획법(MILP) 문제를 해결하는 오픈소스 솔버입니다.
    감축활동 선택(on/off)과 같은 이진 결정이 필요한 문제에 적합합니다.
    """
    
    def __init__(self, config: Dict[str, Any], stable_var_data: Dict[str, Any]):
        super().__init__(config, stable_var_data)
        self.solver_name = 'cbc'
        
    def build_model(self) -> ConcreteModel:
        """
        정수 계획법을 위한 Pyomo 모델 구축
        
        Returns:
            ConcreteModel: 구축된 최적화 모델
        """
        self.validate_config()
        
        # 모델 생성
        self.model = ConcreteModel()
        
        # 1. 의사결정 변수 정의
        self._define_variables()
        
        # 2. 제약조건 정의
        self._define_constraints()
        
        # 3. 목적함수 정의
        self._define_objective()
        
        return self.model
        
    def _define_variables(self):
        """의사결정 변수 정의"""
        # Tier별 감축활동 선택 변수 (이진 변수)
        reduction_vars = self.get_reduction_vars()
        
        # 감축활동 활성화 여부 (이진 변수)
        for var_name in reduction_vars.keys():
            setattr(self.model, f"{var_name}_active", Var(domain=Binary))
            # 실제 감축비율 변수 (0-100%)
            setattr(self.model, var_name, Var(bounds=(0, 100)))
            
        # 양극재 구성 변수
        cathode_config = self.get_cathode_config()
        
        if cathode_config['type'] == 'A':
            # Type A: 원료구성이 변수
            self.model.low_carbon_emission = Var(
                bounds=cathode_config['emission_range']
            )
            # 비율은 고정값으로 설정
            self.model.recycle_ratio = cathode_config['recycle_ratio_fixed']
            self.model.low_carbon_ratio = cathode_config['low_carbon_ratio_fixed']
            self.model.new_material_ratio = 1 - self.model.recycle_ratio - self.model.low_carbon_ratio
            
        else:  # Type B
            # Type B: 비율이 변수
            self.model.recycle_ratio = Var(bounds=cathode_config['recycle_range'])
            self.model.low_carbon_ratio = Var(bounds=cathode_config['low_carbon_range'])
            self.model.low_carbon_emission = cathode_config['emission_fixed']
            
            # 신재 비율은 표현식으로 정의
            @self.model.Expression
            def new_material_ratio(m):
                return 1 - m.recycle_ratio - m.low_carbon_ratio
                
    def _define_constraints(self):
        """제약조건 정의"""
        cathode_config = self.get_cathode_config()
        reduction_vars = self.get_reduction_vars()
        
        # 1. 감축활동과 감축비율 연계 제약
        for var_name in reduction_vars.keys():
            # 활성화되지 않으면 감축비율은 0
            @self.model.Constraint
            def activity_link_lower(m, var=var_name):
                return getattr(m, var) >= 0
                
            # 활성화되면 감축비율은 최소값 이상
            @self.model.Constraint  
            def activity_link_upper(m, var=var_name):
                return getattr(m, var) <= 100 * getattr(m, f"{var}_active")
                
        # 2. 최대 활성화 가능 활동 수 제한
        if 'max_activities' in self.config.get('constraints', {}):
            @self.model.Constraint
            def max_activities_constraint(m):
                return sum(
                    getattr(m, f"{var}_active") for var in reduction_vars.keys()
                ) <= self.config['constraints']['max_activities']
        
        # 3. 비율 합 = 1 (Type B만 해당)
        if cathode_config['type'] == 'B':
            @self.model.Constraint
            def ratio_sum_constraint(m):
                return m.recycle_ratio + m.low_carbon_ratio <= 1
                
        # 4. 탄소발자국 제한
        @self.model.Constraint
        def carbon_constraint(m):
            # 기본 배출량
            base_emission = 80
            
            # 생산지 전력배출계수
            location_factor = self.calculate_location_factor()
            
            # Tier별 감축 효과
            reduction_effect = sum(
                getattr(m, var) * 0.1 for var in reduction_vars.keys()
            )
            
            # 양극재 효과 계산
            if cathode_config['type'] == 'A':
                cathode_effect = m.low_carbon_emission * self.model.low_carbon_ratio
                recycle_effect = self.model.recycle_ratio * (1 - self.calculate_recycle_impact())
            else:
                cathode_effect = self.model.low_carbon_emission * m.low_carbon_ratio
                recycle_effect = m.recycle_ratio * (1 - self.calculate_recycle_impact())
                
            # 총 배출량 계산
            total_emission = (base_emission * location_factor) - reduction_effect - cathode_effect - recycle_effect
            
            return total_emission <= self.config['constraints']['target_carbon']
            
        # 5. 예산 제약 (활성화된 활동에 대한 비용)
        if 'max_cost' in self.config.get('constraints', {}):
            @self.model.Constraint
            def budget_constraint(m):
                # 각 활동별 고정 비용 (예시)
                activity_costs = {
                    'tier1_양극재': 10000,
                    'tier1_분리막': 8000,
                    'tier1_전해액': 7000,
                    'tier2_양극재': 12000,
                    'tier2_저탄소원료': 15000,
                    'tier2_전구체': 9000,
                    'tier3_니켈원료': 20000,
                    'tier3_코발트': 25000
                }
                
                total_cost = sum(
                    getattr(m, f"{var}_active") * activity_costs.get(var, 10000)
                    for var in reduction_vars.keys()
                )
                
                return total_cost <= self.config['constraints']['max_cost']
                
    def _define_objective(self):
        """목적함수 정의"""
        objective_type = self.config['objective']
        cathode_config = self.get_cathode_config()
        reduction_vars = self.get_reduction_vars()
        
        if objective_type == 'minimize_carbon':
            @self.model.Objective(sense=minimize)
            def objective(m):
                # 기본 배출량
                base_emission = 80
                
                # 생산지 전력배출계수
                location_factor = self.calculate_location_factor()
                
                # Tier별 감축 효과
                reduction_effect = sum(
                    getattr(m, var) * 0.1 for var in reduction_vars.keys()
                )
                
                # 양극재 효과 계산
                if cathode_config['type'] == 'A':
                    cathode_effect = m.low_carbon_emission * self.model.low_carbon_ratio
                    recycle_effect = self.model.recycle_ratio * (1 - self.calculate_recycle_impact())
                else:
                    cathode_effect = self.model.low_carbon_emission * m.low_carbon_ratio
                    recycle_effect = m.recycle_ratio * (1 - self.calculate_recycle_impact())
                    
                # 총 배출량 최소화
                return (base_emission * location_factor) - reduction_effect - cathode_effect - recycle_effect
                
        elif objective_type == 'maximize_ease':
            @self.model.Objective(sense=minimize)
            def objective(m):
                # 활성화된 감축활동 수 최소화 (용이성 최대화)
                return sum(
                    getattr(m, f"{var}_active") for var in reduction_vars.keys()
                )
                
        elif objective_type == 'minimize_cost':
            @self.model.Objective(sense=minimize)
            def objective(m):
                # 활동별 고정 비용과 가변 비용 합계
                activity_costs = {
                    'tier1_양극재': 10000,
                    'tier1_분리막': 8000,
                    'tier1_전해액': 7000,
                    'tier2_양극재': 12000,
                    'tier2_저탄소원료': 15000,
                    'tier2_전구체': 9000,
                    'tier3_니켈원료': 20000,
                    'tier3_코발트': 25000
                }
                
                # 고정 비용 (활성화 비용)
                fixed_cost = sum(
                    getattr(m, f"{var}_active") * activity_costs.get(var, 10000)
                    for var in reduction_vars.keys()
                )
                
                # 가변 비용 (감축 비율에 따른 비용)
                variable_cost = sum(
                    getattr(m, var) * 50  # 감축 비율당 비용
                    for var in reduction_vars.keys()
                )
                
                return fixed_cost + variable_cost
                
    def solve(self) -> Dict[str, Any]:
        """
        CBC를 사용하여 최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과
        """
        if not self.model:
            self.build_model()
            
        # CBC 솔버 생성
        solver = SolverFactory('cbc')
        
        # 솔버 옵션 설정
        solver.options['seconds'] = 300      # 시간 제한
        solver.options['ratio'] = 0.01       # Gap 허용치
        solver.options['threads'] = 4        # 병렬 스레드 수
        
        try:
            # 최적화 실행
            results = solver.solve(self.model, tee=True)
            
            if results.solver.termination_condition == TerminationCondition.optimal:
                # 최적해 찾음
                self.results = {
                    'status': 'optimal',
                    'solver': 'cbc',
                    'objective_value': value(self.model.objective),
                    'variables': self.extract_variables(),
                    'termination_condition': str(results.solver.termination_condition),
                    'solver_time': results.solver.time if hasattr(results.solver, 'time') else None,
                    'active_reductions': self._get_active_reductions()
                }
            else:
                # 최적해를 찾지 못함
                self.results = {
                    'status': 'failed',
                    'solver': 'cbc',
                    'message': f"Termination condition: {results.solver.termination_condition}",
                    'termination_condition': str(results.solver.termination_condition)
                }
                
        except Exception as e:
            self.results = {
                'status': 'error',
                'solver': 'cbc',
                'message': str(e)
            }
            
        return self.results
        
    def _get_active_reductions(self) -> Dict[str, bool]:
        """활성화된 감축활동 목록 반환"""
        reduction_vars = self.get_reduction_vars()
        active_reductions = {}
        
        for var_name in reduction_vars.keys():
            active_var = getattr(self.model, f"{var_name}_active", None)
            if active_var is not None:
                active_reductions[var_name] = bool(value(active_var))
                
        return active_reductions
        
    def get_solver_info(self) -> Dict[str, Any]:
        """
        CBC 솔버 정보 반환
        
        Returns:
            Dict: 솔버 정보
        """
        return {
            'name': 'CBC',
            'full_name': 'COIN-OR Branch and Cut',
            'type': 'Mixed Integer Linear Programming (MILP)',
            'suitable_for': [
                '정수 변수',
                '이진 변수 (on/off 결정)',
                '선형 목적함수/제약조건',
                '조합 최적화 문제'
            ],
            'limitations': [
                '비선형 문제 미지원',
                '대규모 문제에서 시간 소요',
                'Type A 문제의 비선형 항 처리 제한'
            ]
        }
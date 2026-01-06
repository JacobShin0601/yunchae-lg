"""
IPOPT 솔버를 사용한 비선형 최적화 구현
주로 복잡한 비선형 목적함수나 제약조건이 있는 문제에 사용됩니다.
"""

from typing import Dict, Any
from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, 
    minimize, value, TerminationCondition
)
from pyomo.opt import SolverFactory
from .base_optimizer import BaseOptimizer


class IPOPTOptimizer(BaseOptimizer):
    """
    IPOPT(Interior Point OPTimizer)를 사용한 비선형 최적화 클래스
    
    IPOPT는 대규모 비선형 최적화 문제를 해결하는 데 특화된 솔버입니다.
    양극재 구성비율과 배출계수의 곱셈 항이 포함된 비선형 문제에 적합합니다.
    """
    
    def __init__(self, config: Dict[str, Any], stable_var_data: Dict[str, Any]):
        super().__init__(config, stable_var_data)
        self.solver_name = 'ipopt'
        
    def build_model(self) -> ConcreteModel:
        """
        비선형 최적화를 위한 Pyomo 모델 구축
        
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
        # Tier별 감축비율 변수 (0-100%)
        reduction_vars = self.get_reduction_vars()
        for var_name in reduction_vars.keys():
            setattr(self.model, var_name, Var(bounds=(0, 100)))
            
        # 양극재 구성 변수
        cathode_config = self.get_cathode_config()
        
        if cathode_config['type'] == 'A':
            # Type A: 원료구성이 변수, 비율은 고정
            self.model.low_carbon_emission = Var(
                bounds=cathode_config['emission_range']
            )
            # 비율은 고정값으로 설정
            self.model.recycle_ratio = cathode_config['recycle_ratio_fixed']
            self.model.low_carbon_ratio = cathode_config['low_carbon_ratio_fixed']
            self.model.new_material_ratio = 1 - self.model.recycle_ratio - self.model.low_carbon_ratio
            
        else:  # Type B
            # Type B: 비율이 변수, 원료구성은 고정
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
        
        # 1. 비율 합 = 1 (Type B만 해당)
        if cathode_config['type'] == 'B':
            @self.model.Constraint
            def ratio_sum_constraint(m):
                return m.recycle_ratio + m.low_carbon_ratio <= 1
                
        # 2. 탄소발자국 제한
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
            
            # 양극재 효과 계산 (비선형 항 포함)
            if cathode_config['type'] == 'A':
                # Type A: 원료구성(변수) × 비율(고정)
                cathode_effect = m.low_carbon_emission * self.model.low_carbon_ratio
                recycle_effect = self.model.recycle_ratio * (1 - self.calculate_recycle_impact())
            else:
                # Type B: 원료구성(고정) × 비율(변수)
                cathode_effect = self.model.low_carbon_emission * m.low_carbon_ratio
                recycle_effect = m.recycle_ratio * (1 - self.calculate_recycle_impact())
                
            # 총 배출량 계산
            total_emission = (base_emission * location_factor) - reduction_effect - cathode_effect - recycle_effect
            
            return total_emission <= self.config['constraints']['target_carbon']
            
        # 3. 희귀 금속 제한 (추가 가능)
        # 4. 비용 제한 (추가 가능)
        
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
                
                # 양극재 효과 계산 (비선형 항 포함)
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
                # IPOPT는 연속 변수만 다루므로 근사화 필요
                active_reductions = sum(
                    getattr(m, var) / (getattr(m, var) + 0.001) 
                    for var in reduction_vars.keys()
                )
                return active_reductions
                
    def solve(self) -> Dict[str, Any]:
        """
        IPOPT를 사용하여 최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과
        """
        if not self.model:
            self.build_model()
            
        # IPOPT 솔버 생성
        solver = SolverFactory('ipopt')
        
        # 솔버 옵션 설정
        solver.options['max_iter'] = 3000
        solver.options['tol'] = 1e-6
        solver.options['print_level'] = 5  # 출력 레벨
        
        try:
            # 최적화 실행
            results = solver.solve(self.model, tee=True)
            
            if results.solver.termination_condition == TerminationCondition.optimal:
                # 최적해 찾음
                self.results = {
                    'status': 'optimal',
                    'solver': 'ipopt',
                    'objective_value': value(self.model.objective),
                    'variables': self.extract_variables(),
                    'termination_condition': str(results.solver.termination_condition),
                    'solver_time': results.solver.time
                }
            else:
                # 최적해를 찾지 못함
                self.results = {
                    'status': 'failed',
                    'solver': 'ipopt',
                    'message': f"Termination condition: {results.solver.termination_condition}",
                    'termination_condition': str(results.solver.termination_condition)
                }
                
        except Exception as e:
            self.results = {
                'status': 'error',
                'solver': 'ipopt',
                'message': str(e)
            }
            
        return self.results
        
    def get_solver_info(self) -> Dict[str, Any]:
        """
        IPOPT 솔버 정보 반환
        
        Returns:
            Dict: 솔버 정보
        """
        return {
            'name': 'IPOPT',
            'full_name': 'Interior Point OPTimizer',
            'type': 'Nonlinear Programming (NLP)',
            'suitable_for': [
                '비선형 목적함수',
                '비선형 제약조건',
                '대규모 문제',
                '연속 변수'
            ],
            'limitations': [
                '정수 변수 미지원',
                '초기값에 민감할 수 있음',
                '국소 최적해만 보장'
            ]
        }
"""
GLPK 솔버를 사용한 선형 최적화 구현
주로 선형 목적함수와 선형 제약조건만 있는 문제에 사용됩니다.
"""

from typing import Dict, Any
from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, 
    minimize, value, TerminationCondition
)
from pyomo.opt import SolverFactory
from .base_optimizer import BaseOptimizer


class GLPKOptimizer(BaseOptimizer):
    """
    GLPK(GNU Linear Programming Kit)를 사용한 선형 최적화 클래스
    
    GLPK는 선형 계획법(LP)과 혼합 정수 선형 계획법(MIP) 문제를 해결하는 솔버입니다.
    Type B와 같이 양극재 비율이 변수인 선형 문제에 적합합니다.
    """
    
    def __init__(self, config: Dict[str, Any], stable_var_data: Dict[str, Any]):
        super().__init__(config, stable_var_data)
        self.solver_name = 'glpk'
        
    def build_model(self) -> ConcreteModel:
        """
        선형 최적화를 위한 Pyomo 모델 구축
        
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
            # Type A는 원료구성이 변수이므로 GLPK에 적합하지 않음
            # 경고 메시지 출력
            import warnings
            warnings.warn(
                "Type A 문제는 비선형 항을 포함하므로 GLPK보다 IPOPT가 더 적합합니다.",
                UserWarning
            )
            # 그래도 선형 근사화하여 처리
            self.model.low_carbon_emission = Var(
                bounds=cathode_config['emission_range']
            )
            # 비율은 고정값으로 설정
            self.model.recycle_ratio = cathode_config['recycle_ratio_fixed']
            self.model.low_carbon_ratio = cathode_config['low_carbon_ratio_fixed']
            self.model.new_material_ratio = 1 - self.model.recycle_ratio - self.model.low_carbon_ratio
            
        else:  # Type B
            # Type B: 비율이 변수 (선형 문제)
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
            
            # Tier별 감축 효과 (선형)
            reduction_effect = sum(
                getattr(m, var) * 0.1 for var in reduction_vars.keys()
            )
            
            # 양극재 효과 계산
            if cathode_config['type'] == 'A':
                # Type A: 선형 근사화 (고정 비율 사용)
                cathode_effect = m.low_carbon_emission * self.model.low_carbon_ratio
                recycle_effect = self.model.recycle_ratio * (1 - self.calculate_recycle_impact())
            else:
                # Type B: 원래부터 선형
                cathode_effect = self.model.low_carbon_emission * m.low_carbon_ratio
                recycle_effect = m.recycle_ratio * (1 - self.calculate_recycle_impact())
                
            # 총 배출량 계산
            total_emission = (base_emission * location_factor) - reduction_effect - cathode_effect - recycle_effect
            
            return total_emission <= self.config['constraints']['target_carbon']
            
        # 3. 희귀 금속 제한 (선택적)
        if 'ni_max' in self.config['constraints']:
            @self.model.Constraint
            def nickel_constraint(m):
                # 간단한 선형 제약
                return sum(getattr(m, var) for var in reduction_vars.keys() if 'tier1' in var) <= 300
        
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
                
        elif objective_type == 'minimize_cost':
            @self.model.Objective(sense=minimize)
            def objective(m):
                # 비용 함수 (선형)
                reduction_cost = sum(
                    getattr(m, var) * 100 for var in reduction_vars.keys()  # 감축 비율당 비용
                )
                
                if cathode_config['type'] == 'B':
                    material_cost = (
                        m.recycle_ratio * 500 +  # 재활용재 비용
                        m.low_carbon_ratio * 800  # 저탄소원료 비용
                    )
                else:
                    material_cost = (
                        self.model.recycle_ratio * 500 + 
                        self.model.low_carbon_ratio * 800
                    )
                    
                return reduction_cost + material_cost
                
    def solve(self) -> Dict[str, Any]:
        """
        GLPK를 사용하여 최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과
        """
        if not self.model:
            self.build_model()
            
        # GLPK 솔버 생성
        solver = SolverFactory('glpk')
        
        # 솔버 옵션 설정
        solver.options['mipgap'] = 0.01  # MIP gap
        solver.options['tmlim'] = 300    # 시간 제한 (초)
        
        try:
            # 최적화 실행
            results = solver.solve(self.model, tee=True)
            
            if results.solver.termination_condition == TerminationCondition.optimal:
                # 최적해 찾음
                self.results = {
                    'status': 'optimal',
                    'solver': 'glpk',
                    'objective_value': value(self.model.objective),
                    'variables': self.extract_variables(),
                    'termination_condition': str(results.solver.termination_condition),
                    'solver_time': results.solver.time if hasattr(results.solver, 'time') else None
                }
            else:
                # 최적해를 찾지 못함
                self.results = {
                    'status': 'failed',
                    'solver': 'glpk',
                    'message': f"Termination condition: {results.solver.termination_condition}",
                    'termination_condition': str(results.solver.termination_condition)
                }
                
        except Exception as e:
            self.results = {
                'status': 'error',
                'solver': 'glpk',
                'message': str(e)
            }
            
        return self.results
        
    def get_solver_info(self) -> Dict[str, Any]:
        """
        GLPK 솔버 정보 반환
        
        Returns:
            Dict: 솔버 정보
        """
        return {
            'name': 'GLPK',
            'full_name': 'GNU Linear Programming Kit',
            'type': 'Linear Programming (LP) / Mixed Integer Programming (MIP)',
            'suitable_for': [
                '선형 목적함수',
                '선형 제약조건',
                '중소규모 문제',
                '연속/정수 변수'
            ],
            'limitations': [
                '비선형 문제 미지원',
                'Type A 문제에는 근사화 필요',
                '대규모 문제에서 성능 저하 가능'
            ]
        }
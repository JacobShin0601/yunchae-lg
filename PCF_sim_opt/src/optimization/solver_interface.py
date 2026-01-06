"""
다양한 최적화 솔버를 위한 통합 인터페이스
"""

from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from pyomo.environ import (
    ConcreteModel, value, TerminationCondition, SolverFactory
)
from .input import OptimizationInput
from .formula import OptimizationFormula


class SolverInterface(ABC):
    """
    최적화 솔버를 위한 추상 기본 클래스
    
    모든 솔버 구현체는 이 클래스를 상속받아야 합니다.
    """
    
    def __init__(self, opt_input: OptimizationInput, model: Optional[ConcreteModel] = None):
        """
        Args:
            opt_input: 최적화 입력 객체
            model: Pyomo 모델 (None이면 새로 생성)
        """
        self.opt_input = opt_input
        
        if model is None:
            self.model = ConcreteModel()
        else:
            self.model = model
            
        self.formula = OptimizationFormula(opt_input, self.model)
        self.results = None
    
    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        """
        최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과
        """
        pass
    
    def build_model(self) -> ConcreteModel:
        """
        최적화 모델 구축
        
        Returns:
            ConcreteModel: 구축된 Pyomo 모델
        """
        # 변수 정의
        self.opt_input.get_variables().define_variables(self.model)
        
        # 목적함수 정의
        self.formula.define_objective()
        
        # 제약조건 정의
        self.formula.define_constraints()
        
        return self.model
    
    def extract_variables(self) -> Dict[str, Any]:
        """
        모델에서 변수값 추출
        
        Returns:
            Dict: 변수명과 값의 매핑
        """
        if not self.model:
            return {}
        
        variables = {}
        
        # 감축비율 변수
        reduction_vars = self.opt_input.get_variables().get_reduction_variables()
        for var_name in reduction_vars.keys():
            var = getattr(self.model, var_name, None)
            if var is not None:
                variables[var_name] = value(var)
            
            # 이진 변수 (있는 경우)
            binary_var = getattr(self.model, f"{var_name}_active", None)
            if binary_var is not None:
                variables[f"{var_name}_active"] = value(binary_var)
        
        # 양극재 구성 변수
        cathode_type = self.opt_input.get_variables().get_cathode_type()
        
        if cathode_type == 'A':
            var = getattr(self.model, 'low_carbon_emission', None)
            if var is not None:
                variables['low_carbon_emission'] = value(var)
            # 비율은 고정값이므로 모델에서 추출하지 않음
        else:
            var = getattr(self.model, 'recycle_ratio', None)
            if var is not None:
                variables['recycle_ratio'] = value(var)
                
            var = getattr(self.model, 'low_carbon_ratio', None)
            if var is not None:
                variables['low_carbon_ratio'] = value(var)
        
        return variables
    
    def get_results(self) -> Dict[str, Any]:
        """최적화 결과 반환"""
        return self.results


class GLPKSolver(SolverInterface):
    """
    GLPK 솔버 구현
    
    선형 계획법(LP)과 혼합 정수 선형 계획법(MIP) 문제에 적합합니다.
    """
    
    def __init__(self, opt_input: OptimizationInput, model: Optional[ConcreteModel] = None):
        super().__init__(opt_input, model)
        self.solver_name = 'glpk'
    
    def solve(self) -> Dict[str, Any]:
        """
        GLPK 솔버를 사용하여 최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과
        """
        if not hasattr(self.model, 'objective'):
            self.build_model()
        
        # 솔버 생성
        from pyomo.environ import SolverFactory as PyomoSolverFactory
        solver = PyomoSolverFactory('glpk')
        
        # 솔버 옵션 설정
        solver_options = self.opt_input.get_solver_options('glpk')
        for key, value in solver_options.items():
            if key != 'description':
                solver.options[key] = value
        
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
                    'solver_time': results.solver.time if hasattr(results.solver, 'time') else None,
                    # Debug 정보 추가 및 오류 처리 강화
                    'carbon_footprint': 0.0  # 기본값
                }
                
                # carbon_footprint 안전하게 계산 시도
                try:
                    variables = self.extract_variables()
                    carbon_footprint = self.formula.calculate_carbon_footprint(variables)
                    print(f"DEBUG: carbon_footprint = {carbon_footprint}, 타입: {type(carbon_footprint)}")
                    if isinstance(carbon_footprint, (int, float)):
                        self.results['carbon_footprint'] = float(carbon_footprint)
                    else:
                        print(f"WARNING: carbon_footprint가 숫자가 아닙니다: {carbon_footprint}")
                except Exception as e:
                    print(f"ERROR: carbon_footprint 계산 중 오류: {e}")
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


class IPOPTSolver(SolverInterface):
    """
    IPOPT 솔버 구현
    
    비선형 최적화 문제에 적합합니다.
    """
    
    def __init__(self, opt_input: OptimizationInput, model: Optional[ConcreteModel] = None):
        super().__init__(opt_input, model)
        self.solver_name = 'ipopt'
    
    def solve(self) -> Dict[str, Any]:
        """
        IPOPT 솔버를 사용하여 최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과
        """
        if not hasattr(self.model, 'objective'):
            self.build_model()
        
        # 솔버 생성
        from pyomo.environ import SolverFactory as PyomoSolverFactory
        solver = PyomoSolverFactory('ipopt')
        
        # 솔버 옵션 설정
        solver_options = self.opt_input.get_solver_options('ipopt')
        for key, value in solver_options.items():
            if key != 'description':
                solver.options[key] = value
        
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
                    'solver_time': results.solver.time if hasattr(results.solver, 'time') else None,
                    # Debug 정보 추가 및 오류 처리 강화
                    'carbon_footprint': 0.0  # 기본값
                }
                
                # carbon_footprint 안전하게 계산 시도
                try:
                    variables = self.extract_variables()
                    carbon_footprint = self.formula.calculate_carbon_footprint(variables)
                    print(f"DEBUG: carbon_footprint = {carbon_footprint}, 타입: {type(carbon_footprint)}")
                    if isinstance(carbon_footprint, (int, float)):
                        self.results['carbon_footprint'] = float(carbon_footprint)
                    else:
                        print(f"WARNING: carbon_footprint가 숫자가 아닙니다: {carbon_footprint}")
                except Exception as e:
                    print(f"ERROR: carbon_footprint 계산 중 오류: {e}")
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


class CBCSolver(SolverInterface):
    """
    CBC 솔버 구현
    
    혼합 정수 계획법(MIP) 문제에 적합합니다.
    """
    
    def __init__(self, opt_input: OptimizationInput, model: Optional[ConcreteModel] = None):
        super().__init__(opt_input, model)
        self.solver_name = 'cbc'
    
    def solve(self) -> Dict[str, Any]:
        """
        CBC 솔버를 사용하여 최적화 문제 해결
        
        Returns:
            Dict: 최적화 결과
        """
        if not hasattr(self.model, 'objective'):
            self.build_model()
        
        # 솔버 생성
        from pyomo.environ import SolverFactory as PyomoSolverFactory
        solver = PyomoSolverFactory('cbc')
        
        # 솔버 옵션 설정
        solver_options = self.opt_input.get_solver_options('cbc')
        for key, value in solver_options.items():
            if key != 'description':
                solver.options[key] = value
        
        try:
            # 최적화 실행
            results = solver.solve(self.model, tee=True)
            
            if results.solver.termination_condition == TerminationCondition.optimal:
                # 최적해 찾음
                variables = self.extract_variables()
                
                # 활성화된 감축활동 목록 추출
                active_reductions = {}
                for var_name, var_value in variables.items():
                    if var_name.endswith('_active') and var_value > 0.5:
                        activity_name = var_name.replace('_active', '')
                        active_reductions[activity_name] = True
                
                self.results = {
                    'status': 'optimal',
                    'solver': 'cbc',
                    'objective_value': value(self.model.objective),
                    'variables': variables,
                    'termination_condition': str(results.solver.termination_condition),
                    'solver_time': results.solver.time if hasattr(results.solver, 'time') else None,
                    # Debug 정보 추가 및 오류 처리 강화
                    'carbon_footprint': 0.0,  # 기본값
                    'active_reductions': active_reductions
                }
                
                # carbon_footprint 안전하게 계산 시도
                try:
                    carbon_footprint = self.formula.calculate_carbon_footprint(variables)
                    print(f"DEBUG: carbon_footprint (CBC) = {carbon_footprint}, 타입: {type(carbon_footprint)}")
                    if isinstance(carbon_footprint, (int, float)):
                        self.results['carbon_footprint'] = float(carbon_footprint)
                    else:
                        print(f"WARNING: carbon_footprint가 숫자가 아닙니다: {carbon_footprint}")
                except Exception as e:
                    print(f"ERROR: carbon_footprint 계산 중 오류: {e}")
                
                # active_reductions 기존 설정
                # active_reductions is already added
                # No need to modify self.results again

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


class SolverFactory:
    """
    솔버 팩토리 클래스
    
    적절한 솔버 인스턴스를 생성합니다.
    """
    
    @staticmethod
    def create_solver(
        solver_name: str,
        opt_input: OptimizationInput,
        model: Optional[ConcreteModel] = None
    ) -> SolverInterface:
        """
        지정된 이름의 솔버 인스턴스 생성
        
        Args:
            solver_name: 솔버 이름
            opt_input: 최적화 입력 객체
            model: Pyomo 모델 (선택 사항)
            
        Returns:
            SolverInterface: 솔버 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 솔버
        """
        if solver_name.lower() == 'glpk':
            return GLPKSolver(opt_input, model)
        elif solver_name.lower() == 'ipopt':
            return IPOPTSolver(opt_input, model)
        elif solver_name.lower() == 'cbc':
            return CBCSolver(opt_input, model)
        else:
            raise ValueError(f"지원하지 않는 솔버: {solver_name}")
    
    @staticmethod
    def create_recommended_solver(
        opt_input: OptimizationInput,
        model: Optional[ConcreteModel] = None
    ) -> SolverInterface:
        """
        설정에 기반한 추천 솔버 인스턴스 생성
        
        Args:
            opt_input: 최적화 입력 객체
            model: Pyomo 모델 (선택 사항)
            
        Returns:
            SolverInterface: 추천 솔버 인스턴스
        """
        # 문제 특성에 따라 솔버 추천
        solver_name = opt_input.get_solver_recommendation()
        return SolverFactory.create_solver(solver_name, opt_input, model)
    
    @staticmethod
    def get_available_solvers() -> List[str]:
        """
        사용 가능한 솔버 목록 반환
        
        Returns:
            List[str]: 솔버 이름 목록
        """
        return ['glpk', 'ipopt', 'cbc']
    
    @staticmethod
    def get_solver_info() -> Dict[str, Dict[str, Any]]:
        """
        사용 가능한 솔버 정보 반환
        
        Returns:
            Dict: 솔버별 정보 딕셔너리
        """
        return {
            'glpk': {
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
            },
            'ipopt': {
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
            },
            'cbc': {
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
        }

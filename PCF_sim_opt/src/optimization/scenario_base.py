"""
최적화 시나리오를 위한 기본 인터페이스 모듈
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import json
from datetime import datetime
from pyomo.environ import ConcreteModel, value

from .input import OptimizationInput
from .constant import OptimizationConstants
from .variable import OptimizationVariables
from .validator import Validator
from .results_processor import ResultsProcessor
from .solver_interface import SolverInterface, SolverFactory
from .constraints import ConstraintManager
from .objective import ObjectiveManager
from .material_specific_objective import MaterialSpecificObjective


class OptimizationScenario(ABC):
    """
    최적화 시나리오를 위한 추상 기본 클래스
    
    모든 시나리오 구현체는 이 클래스를 상속받아야 합니다.
    """
    
    def __init__(self, 
                config_path: Optional[str] = None,
                name: str = "default_scenario",
                description: str = "기본 최적화 시나리오"):
        """
        Args:
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
            name: 시나리오 이름
            description: 시나리오 설명
        """
        self.name = name
        self.description = description
        
        # 모듈 초기화
        self.opt_input = OptimizationInput()
        self.validator = Validator(self.opt_input)
        self.results_processor = ResultsProcessor(self.opt_input)
        self.material_objective = MaterialSpecificObjective(self.opt_input)
        
        # 모델 및 결과
        self.model = None
        self.solver = None
        self.results = None
        
        # 설정 파일 로드 (있는 경우)
        if config_path:
            self.load_config(config_path)
            
        # 시나리오 구성
        self._configure_scenario()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            Dict: 로드된 설정
        """
        config = self.opt_input.load_config(config_path)
        self.validator.set_opt_input(self.opt_input)
        self.results_processor.set_opt_input(self.opt_input)
        return config
    
    @abstractmethod
    def _configure_scenario(self) -> None:
        """
        시나리오 특화 설정을 적용합니다.
        각 시나리오는 이 메서드를 구현하여 고유한 설정을 적용해야 합니다.
        """
        pass
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        설정 유효성 검증
        
        Returns:
            Tuple[bool, List[str]]: (유효성 여부, 에러 메시지 목록)
        """
        return self.validator.validate_config()
    
    def build_model(self) -> ConcreteModel:
        """
        최적화 모델 구축
        
        Returns:
            ConcreteModel: 구축된 Pyomo 모델
        """
        self.model = ConcreteModel(name=f"{self.name}_model")
        
        # 1. 변수 정의
        variables = self.opt_input.get_variables()
        variables.define_variables(self.model)
        
        # 2. 제약조건 정의
        constraint_manager = ConstraintManager(self.opt_input, self.model)
        constraint_manager.register_standard_constraints()
        constraint_manager.apply_all_constraints()
        
        # 3. 목적함수 정의
        if self.opt_input.is_material_specific_enabled():
            # 소재별 최적화가 활성화된 경우
            self.material_objective.set_model(self.model)
            self.material_objective.define_material_objectives()
        else:
            # 기본 목적함수 정의
            objective_manager = ObjectiveManager(self.opt_input, self.model)
            objective_manager.register_standard_objectives()
            objective_manager.apply_objective_from_config()
        
        # 4. 추가 시나리오별 모델 설정
        self._configure_model(self.model)
        
        return self.model
    
    def _configure_model(self, model: ConcreteModel) -> None:
        """
        시나리오별 모델 구성 추가
        
        Args:
            model: Pyomo 모델
        """
        # 기본 구현은 아무것도 하지 않음
        # 자식 클래스에서 필요에 따라 재정의
        pass
    
    def select_solver(self) -> str:
        """
        시나리오에 적합한 솔버 선택
        
        Returns:
            str: 선택된 솔버 이름
        """
        return self.opt_input.get_solver_recommendation()
    
    def solve(self, solver_name: Optional[str] = None) -> Dict[str, Any]:
        """
        최적화 문제 해결
        
        Args:
            solver_name: 사용할 솔버 이름 (None이면 자동 선택)
            
        Returns:
            Dict: 최적화 결과
        """
        # 모델이 없으면 구축
        if self.model is None:
            self.build_model()
        
        # 솔버 선택
        if solver_name is None:
            solver_name = self.select_solver()
        
        # 솔버 생성
        self.solver = SolverFactory.create_solver(solver_name, self.opt_input, self.model)
        
        # 최적화 실행
        self.results = self.solver.solve()
        
        # 결과 처리
        if self.results.get('status') == 'optimal':
            self.results_processor.process_results(self.results)
        
        return self.results
    
    def run_scenario(self) -> Dict[str, Any]:
        """
        시나리오 실행 (모든 단계 통합)
        
        Returns:
            Dict: 포맷팅된 최적화 결과
        """
        try:
            # 1. 설정 유효성 검증
            is_valid, errors = self.validate_config()
            if not is_valid:
                return {
                    'status': 'error',
                    'message': '설정 유효성 검증 실패',
                    'errors': errors
                }
            
            # 2. 모델 구축
            self.build_model()
            
            # 3. 최적화 실행
            try:
                results = self.solve()
            except TypeError as e:
                # 'int' object is not callable 오류 직접 처리
                import traceback
                traceback.print_exc()
                # 양극재 타입 확인
                cathode_type = "unknown"
                try:
                    cathode_config = self.opt_input.get_config().get('decision_vars', {}).get('cathode', {})
                    cathode_type = cathode_config.get('type', 'unknown')
                except Exception:
                    pass
                return {
                    'status': 'error',
                    'message': f"'int' object is not callable 오류 (양극재 타입: {cathode_type}): {str(e)}"
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f"시나리오 실행 오류: {str(e)}"
            }
        
        # 4. 결과 처리
        if results.get('status') == 'optimal':
            formatted_results = self.results_processor.process_results(results)
            
            # 소재별 최적화가 활성화된 경우 소재별 기여도 추가
            if self.opt_input.is_material_specific_enabled():
                material_contributions = {}
                for material in self.opt_input.get_all_materials():
                    contribution = self.material_objective.get_material_contribution(
                        material, results.get('variables', {}))
                    if contribution:
                        material_contributions[material] = contribution
                        
                formatted_results['material_contributions'] = material_contributions
                
            return formatted_results
        else:
            return {
                'status': results.get('status', 'error'),
                'message': results.get('message', '최적화 실패')
            }
    
    def export_results(self, file_path: Optional[str] = None, format: str = 'json') -> str:
        """
        결과를 파일로 내보내기
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            format: 파일 형식 ('json' 또는 'yaml')
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.results:
            raise ValueError("내보낼 결과가 없습니다. run_scenario() 또는 solve()를 먼저 호출하세요.")
            
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.name}_results_{timestamp}.{format}"
        
        # 결과 형식에 따라 내보내기
        if format.lower() == 'json':
            return self.results_processor.export_to_json(file_path)
        else:
            return self.results_processor.export_to_yaml(file_path)
    
    def create_report(self, file_path: Optional[str] = None) -> str:
        """
        결과 보고서 생성
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.results:
            raise ValueError("보고서를 위한 결과가 없습니다. run_scenario() 또는 solve()를 먼저 호출하세요.")
            
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.name}_report_{timestamp}.md"
            
        return self.results_processor.create_report(file_path)
    
    def visualize_results(self, file_path: Optional[str] = None) -> str:
        """
        결과 시각화
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.results:
            raise ValueError("시각화할 결과가 없습니다. run_scenario() 또는 solve()를 먼저 호출하세요.")
            
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.name}_visualization_{timestamp}.png"
            
        return self.results_processor.visualize_results(file_path)
    
    def export_scenario_config(self, file_path: Optional[str] = None, format: str = 'yaml') -> str:
        """
        현재 시나리오 설정을 파일로 내보내기
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            format: 파일 형식 ('yaml' 또는 'json')
            
        Returns:
            str: 저장된 파일 경로
        """
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.name}_config_{timestamp}.{format}"
            
        # 설정에 시나리오 정보 추가
        config = self.opt_input.get_config()
        
        if 'metadata' not in config:
            config['metadata'] = {}
            
        config['metadata']['scenario_name'] = self.name
        config['metadata']['scenario_description'] = self.description
        config['metadata']['export_time'] = datetime.now().isoformat()
        
        # 형식에 따라 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:  # json으로 저장
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                
        return str(path)
    
    def get_name(self) -> str:
        """시나리오 이름 반환"""
        return self.name
    
    def get_description(self) -> str:
        """시나리오 설명 반환"""
        return self.description
    
    @staticmethod
    def load_scenario_from_config(config_path: str) -> 'OptimizationScenario':
        """
        설정 파일로부터 시나리오 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            OptimizationScenario: 시나리오 인스턴스
        """
        # 설정 파일 로드
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        
        # 파일 형식에 따라 로드
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:  # JSON으로 간주
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 시나리오 유형 결정
        objective = config.get('objective', 'minimize_carbon')
        
        if objective == 'minimize_carbon':
            from .carbon_minimization import CarbonMinimization
            return CarbonMinimization(config_path)
        elif objective == 'minimize_cost':
            from .cost_minimization import CostMinimization
            return CostMinimization(config_path)
        elif objective == 'multi_objective':
            from .multi_objective import MultiObjective
            return MultiObjective(config_path)
        elif objective == 'maximize_ease':
            from .implementation_ease import ImplementationEase
            return ImplementationEase(config_path)
        else:
            # 기본 시나리오는 현재 구현되지 않았으므로 NotImplementedError 발생
            raise NotImplementedError(f"목적함수 '{objective}'에 대한 시나리오가 아직 구현되지 않았습니다.")


class ScenarioFactory:
    """
    시나리오 생성을 위한 팩토리 클래스
    """
    
    @staticmethod
    def create_scenario(scenario_type: str, config_path: Optional[str] = None) -> OptimizationScenario:
        """
        지정된 유형의 시나리오 인스턴스 생성
        
        Args:
            scenario_type: 시나리오 유형
            config_path: 설정 파일 경로 (선택 사항)
            
        Returns:
            OptimizationScenario: 시나리오 인스턴스
        """
        if scenario_type == 'carbon_minimization':
            from .carbon_minimization import CarbonMinimization
            return CarbonMinimization(config_path)
        elif scenario_type == 'cost_minimization':
            from .cost_minimization import CostMinimization
            return CostMinimization(config_path)
        elif scenario_type == 'multi_objective':
            from .multi_objective import MultiObjective
            return MultiObjective(config_path)
        elif scenario_type == 'implementation_ease':
            from .implementation_ease import ImplementationEase
            return ImplementationEase(config_path)
        elif scenario_type == 'regional_optimization':
            from .regional_optimization import RegionalOptimization
            return RegionalOptimization(config_path)
        else:
            raise ValueError(f"지원하지 않는 시나리오 유형: {scenario_type}")
    
    @staticmethod
    def get_available_scenario_types() -> List[str]:
        """
        사용 가능한 시나리오 유형 목록 반환
        
        Returns:
            List[str]: 시나리오 유형 목록
        """
        return [
            'carbon_minimization',
            'cost_minimization',
            'multi_objective',
            'implementation_ease',
            'regional_optimization'
        ]
    
    @staticmethod
    def get_scenario_info() -> Dict[str, Dict[str, str]]:
        """
        시나리오 정보 반환
        
        Returns:
            Dict: 시나리오 유형별 정보
        """
        return {
            'carbon_minimization': {
                'name': '탄소배출 최소화',
                'description': '탄소발자국을 최소화하는 최적화 시나리오',
                'objective': 'minimize_carbon'
            },
            'cost_minimization': {
                'name': '비용 최소화',
                'description': '총 구현 비용을 최소화하는 최적화 시나리오',
                'objective': 'minimize_cost'
            },
            'multi_objective': {
                'name': '다목적 최적화',
                'description': '탄소발자국과 비용을 동시에 고려하는 다목적 최적화 시나리오',
                'objective': 'multi_objective'
            },
            'implementation_ease': {
                'name': '구현 용이성 최적화',
                'description': '구현 복잡도를 최소화하는 최적화 시나리오',
                'objective': 'maximize_ease'
            },
            'regional_optimization': {
                'name': '지역별 최적화',
                'description': '생산 위치와 물류를 고려한 지역별 최적화 시나리오',
                'objective': 'minimize_carbon'
            }
        }
"""
최적화 실행을 위한 통합 컨트롤러 클래스
모든 최적화 모듈을 관리하고 사용자 인터페이스를 제공합니다.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import yaml
import json
import os
from datetime import datetime
from pyomo.environ import ConcreteModel

from .input import OptimizationInput
from .constant import OptimizationConstants
from .variable import OptimizationVariables
from .formula import OptimizationFormula
from .carbon_minimization import CarbonMinimization
from .cost_minimization import CostMinimization
from .multi_objective import MultiObjective
from .implementation_ease import ImplementationEase
from .regional_optimization import RegionalOptimization
from .solver_interface import SolverFactory, SolverInterface
from .material_specific_objective import MaterialSpecificObjective


class OptimizationController:
    """
    최적화 실행을 위한 통합 컨트롤러
    
    모든 최적화 구성요소를 관리하고 사용자 인터페이스 제공
    """
    
    def __init__(self, config_path: Optional[str] = None, stable_var_dir: str = "stable_var"):
        """
        Args:
            config_path: 설정 파일 경로 (YAML 또는 JSON)
            stable_var_dir: stable_var 디렉토리 경로
        """
        # 모듈 초기화
        self.opt_input = OptimizationInput(stable_var_dir=stable_var_dir)
        self.model = None
        self.solver = None
        self.scenario = None
        self.material_objective = MaterialSpecificObjective(self.opt_input)
        
        # 설정 파일 로드 (있는 경우)
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로 (YAML 또는 JSON)
            
        Returns:
            Dict: 로드된 설정
        """
        return self.opt_input.load_config(config_path)
    
    def apply_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        시나리오 적용
        
        Args:
            scenario_name: 시나리오 이름
            
        Returns:
            Dict: 적용된 시나리오 설정
        """
        return self.opt_input.apply_scenario(scenario_name)
    
    def get_available_scenarios(self) -> Dict[str, str]:
        """
        사용 가능한 시나리오 정보 반환
        
        Returns:
            Dict: 시나리오 이름과 설명의 매핑
        """
        return self.opt_input.get_available_scenarios()
    
    def create_custom_config(self, **overrides) -> Dict[str, Any]:
        """
        커스텀 설정 생성
        
        Args:
            **overrides: 덮어쓸 설정값들
            
        Returns:
            Dict: 커스텀 설정
        """
        return self.opt_input.create_custom_config(**overrides)
    
    def build_model(self) -> ConcreteModel:
        """
        최적화 모델 구축
        
        Returns:
            ConcreteModel: 구축된 Pyomo 모델
        """
        self.model = ConcreteModel()
        
        # 변수 정의
        variables = self.opt_input.get_variables()
        variables.define_variables(self.model)
        
        # 소재별 최적화 설정
        if self.opt_input.is_material_specific_enabled():
            # 소재별 목적함수 설정
            self.material_objective.set_model(self.model)
            self.material_objective.define_material_objectives()
        else:
            # 기본 수식 정의
            formula = OptimizationFormula(self.opt_input, self.model)
            formula.define_objective()
            formula.define_constraints()
        
        return self.model
    
    def get_solver(self, solver_name: Optional[str] = None) -> SolverInterface:
        """
        솔버 인스턴스 생성
        
        Args:
            solver_name: 솔버 이름 (None이면 자동 선택)
            
        Returns:
            SolverInterface: 솔버 인스턴스
        """
        if solver_name is None:
            self.solver = SolverFactory.create_recommended_solver(self.opt_input, self.model)
        else:
            self.solver = SolverFactory.create_solver(solver_name, self.opt_input, self.model)
        
        return self.solver
    
    def solve(self, solver_name: Optional[str] = None) -> Dict[str, Any]:
        """
        최적화 문제 해결
        
        Args:
            solver_name: 솔버 이름 (None이면 자동 선택)
            
        Returns:
            Dict: 최적화 결과
        """
        # 모델이 없으면 구축
        if self.model is None:
            self.build_model()
        
        # 솔버 가져오기
        solver = self.get_solver(solver_name)
        
        # 최적화 실행
        results = solver.solve()
        
        return results
    
    def run_optimization(self, 
                        config_path: Optional[str] = None, 
                        scenario_name: Optional[str] = None,
                        solver_name: Optional[str] = None) -> Dict[str, Any]:
        """
        설정 로드부터 결과 반환까지 전체 최적화 프로세스 실행
        
        Args:
            config_path: 설정 파일 경로 (선택 사항)
            scenario_name: 시나리오 이름 (선택 사항)
            solver_name: 솔버 이름 (선택 사항)
            
        Returns:
            Dict: 최적화 결과
        """
        # 1. 설정 로드
        if config_path:
            self.load_config(config_path)
        
        # 2. 시나리오 적용
        if scenario_name:
            self.apply_scenario(scenario_name)
        
        # 3. 모델 구축
        self.build_model()
        
        # 4. 최적화 실행
        results = self.solve(solver_name)
        
        # 5. 결과 반환
        return results
    
    def run_carbon_minimization(self, 
                              config_path: Optional[str] = None,
                              solver_name: Optional[str] = None) -> Dict[str, Any]:
        """
        탄소배출 최소화 시나리오 실행
        
        Args:
            config_path: 설정 파일 경로 (선택 사항)
            solver_name: 솔버 이름 (선택 사항)
            
        Returns:
            Dict: 최적화 결과
        """
        # 설정 파일이 있으면 로드
        if config_path:
            self.load_config(config_path)
        
        # 탄소배출 최소화 시나리오 생성 및 실행
        self.scenario = CarbonMinimization(
            config_path=config_path if config_path else None
        )
        
        # 시나리오 실행
        results = self.scenario.run_scenario()
        
        return results
    
    def run_cost_minimization(self, 
                            config_path: Optional[str] = None,
                            solver_name: Optional[str] = None) -> Dict[str, Any]:
        """
        비용 최소화 시나리오 실행
        
        Args:
            config_path: 설정 파일 경로 (선택 사항)
            solver_name: 솔버 이름 (선택 사항)
            
        Returns:
            Dict: 최적화 결과
        """
        # 설정 파일이 있으면 로드
        if config_path:
            self.load_config(config_path)
        
        # 비용 최소화 시나리오 생성 및 실행
        self.scenario = CostMinimization(
            config_path=config_path if config_path else None
        )
        
        # 시나리오 실행
        results = self.scenario.run_scenario()
        
        return results
    
    def run_multi_objective(self, 
                          config_path: Optional[str] = None,
                          carbon_weight: float = 0.7,
                          cost_weight: float = 0.3) -> Dict[str, Any]:
        """
        다목적 최적화 시나리오 실행
        
        Args:
            config_path: 설정 파일 경로 (선택 사항)
            carbon_weight: 탄소발자국 목적함수의 가중치 (0.0-1.0)
            cost_weight: 비용 목적함수의 가중치 (0.0-1.0)
            
        Returns:
            Dict: 최적화 결과
        """
        # 설정 파일이 있으면 로드
        if config_path:
            self.load_config(config_path)
        
        # 다목적 최적화 시나리오 생성 및 실행
        self.scenario = MultiObjective(
            config_path=config_path if config_path else None,
            carbon_weight=carbon_weight,
            cost_weight=cost_weight
        )
        
        # 시나리오 실행
        results = self.scenario.run_scenario()
        
        return results
    
    def run_implementation_ease(self, 
                              config_path: Optional[str] = None,
                              carbon_target: float = 45.0) -> Dict[str, Any]:
        """
        구현 용이성 최적화 시나리오 실행
        
        Args:
            config_path: 설정 파일 경로 (선택 사항)
            carbon_target: 달성해야 할 탄소발자국 목표 (kg CO2eq/kWh)
            
        Returns:
            Dict: 최적화 결과
        """
        # 설정 파일이 있으면 로드
        if config_path:
            self.load_config(config_path)
        
        # 구현 용이성 최적화 시나리오 생성 및 실행
        self.scenario = ImplementationEase(
            config_path=config_path if config_path else None,
            carbon_target=carbon_target
        )
        
        # 시나리오 실행
        results = self.scenario.run_scenario()
        
        return results
    
    def run_regional_optimization(self, 
                                config_path: Optional[str] = None,
                                target_regions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        지역별 최적화 시나리오 실행
        
        Args:
            config_path: 설정 파일 경로 (선택 사항)
            target_regions: 고려할 지역 목록 (None이면 모든 가능한 지역 사용)
            
        Returns:
            Dict: 최적화 결과
        """
        # 설정 파일이 있으면 로드
        if config_path:
            self.load_config(config_path)
        
        # 지역별 최적화 시나리오 생성 및 실행
        self.scenario = RegionalOptimization(
            config_path=config_path if config_path else None,
            target_regions=target_regions
        )
        
        # 시나리오 실행
        results = self.scenario.run_scenario()
        
        return results
    
    def run_multi_solver_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        여러 솔버로 동일한 문제 해결 및 비교
        
        Returns:
            Dict: 솔버별 결과
        """
        # 모델이 없으면 구축
        if self.model is None:
            self.build_model()
        
        # 사용 가능한 솔버 목록
        solvers = SolverFactory.get_available_solvers()
        results = {}
        
        # 각 솔버로 최적화 실행
        for solver_name in solvers:
            try:
                # 모델 복제 (솔버간 간섭 방지)
                model_copy = ConcreteModel()
                variables = self.opt_input.get_variables()
                variables.define_variables(model_copy)
                
                # 소재별 최적화 설정 또는 기본 수식 정의
                if self.opt_input.is_material_specific_enabled():
                    material_objective_copy = MaterialSpecificObjective(self.opt_input)
                    material_objective_copy.set_model(model_copy)
                    material_objective_copy.define_material_objectives()
                else:
                    formula = OptimizationFormula(self.opt_input, model_copy)
                    formula.define_objective()
                    formula.define_constraints()
                
                # 솔버 생성 및 문제 해결
                solver = SolverFactory.create_solver(solver_name, self.opt_input, model_copy)
                results[solver_name] = solver.solve()
                
            except Exception as e:
                results[solver_name] = {
                    'status': 'error',
                    'solver': solver_name,
                    'message': str(e)
                }
        
        return results
    
    def export_results(self, results: Dict[str, Any], file_path: Optional[str] = None, 
                     format: str = 'json') -> str:
        """
        최적화 결과를 파일로 내보내기
        
        Args:
            results: 최적화 결과
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            format: 파일 형식 ('json' 또는 'yaml')
            
        Returns:
            str: 저장된 파일 경로
        """
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"optimization_results_{timestamp}.{format}"
        
        # 결과 데이터 구성
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.opt_input.get_config(),
            'results': results
        }
        
        # 파일 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
        else:  # json으로 저장
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return str(path)
    
    def get_formatted_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 친화적 형태로 포맷팅된 결과 반환
        
        Args:
            results: 최적화 결과
            
        Returns:
            Dict: 포맷팅된 결과
        """
        if not results or results.get('status') != 'optimal':
            return {'status': results.get('status', 'not_solved'), 
                    'message': results.get('message', '최적화가 실행되지 않았거나 실패했습니다.')}
        
        variables = results.get('variables', {})
        
        # 소재별 최적화가 활성화된 경우 소재별 기여도 분석 추가
        material_contributions = {}
        if self.opt_input.is_material_specific_enabled() and self.material_objective:
            for material in self.opt_input.get_all_materials():
                contribution = self.material_objective.get_material_contribution(
                    material, variables)
                if contribution:
                    material_contributions[material] = contribution
        
        # 감축비율 결과
        reduction_results = {}
        for var_name, var_value in variables.items():
            if 'tier' in var_name and not var_name.endswith('_active'):
                parts = var_name.split('_')
                tier = parts[0].upper()
                item = '_'.join(parts[1:])
                
                if tier not in reduction_results:
                    reduction_results[tier] = {}
                    
                reduction_results[tier][item] = f"{var_value:.2f}%"
        
        # 양극재 구성 결과
        cathode_results = {}
        
        if 'recycle_ratio' in variables:
            cathode_results['재활용재_비율'] = f"{variables['recycle_ratio'] * 100:.2f}%"
            
        if 'low_carbon_ratio' in variables:
            cathode_results['저탄소원료_비율'] = f"{variables['low_carbon_ratio'] * 100:.2f}%"
            cathode_results['신재_비율'] = f"{(1 - variables.get('recycle_ratio', 0) - variables.get('low_carbon_ratio', 0)) * 100:.2f}%"
            
        if 'low_carbon_emission' in variables:
            cathode_results['저탄소원료_배출계수'] = f"{variables['low_carbon_emission']:.2f}"
        
        # 최종 결과 구성
        formatted_results = {
            'status': 'optimal',
            'objective': self.opt_input.get_objective(),
            'carbon_footprint': f"{float(results.get('carbon_footprint', 0)):.4f} kg CO2eq/kWh",
            'solver': results.get('solver'),
            'solver_time': f"{results.get('solver_time', 0):.2f} 초" if results.get('solver_time') else 'N/A',
            'reduction_ratios': reduction_results,
            'cathode_composition': cathode_results
        }
        
        # 소재별 최적화 결과 추가
        if material_contributions:
            formatted_results['material_contributions'] = material_contributions
            
        return formatted_results
    
    def save_config(self, file_path: Optional[str] = None, format: str = 'yaml') -> str:
        """
        현재 설정 저장
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            format: 파일 형식 ('yaml' 또는 'json')
            
        Returns:
            str: 저장된 파일 경로
        """
        return self.opt_input.export_config(
            file_path if file_path else f"config_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}", 
            format
        )
    
    def get_solver_info(self) -> Dict[str, Dict[str, Any]]:
        """
        사용 가능한 솔버 정보 반환
        
        Returns:
            Dict: 솔버별 정보
        """
        return SolverFactory.get_solver_info()
    
    def get_recommended_solver(self) -> str:
        """
        문제 유형에 따른 추천 솔버 이름 반환
        
        Returns:
            str: 추천 솔버 이름
        """
        return self.opt_input.get_solver_recommendation()
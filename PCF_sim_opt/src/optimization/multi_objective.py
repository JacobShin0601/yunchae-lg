"""
다목적 최적화 시나리오 구현 모듈
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from .scenario_base import OptimizationScenario


class MultiObjective(OptimizationScenario):
    """
    다목적 최적화 시나리오
    
    이 클래스는 탄소발자국과 비용을 동시에 고려하는 다목적 최적화 문제를 구성하고 해결합니다.
    가중합(weighted sum) 방식을 사용하여 여러 목적함수를 통합합니다.
    """
    
    def __init__(self, 
                config_path: Optional[str] = None,
                carbon_weight: float = 0.7,
                cost_weight: float = 0.3):
        """
        Args:
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
            carbon_weight: 탄소발자국 목적함수의 가중치 (0.0-1.0)
            cost_weight: 비용 목적함수의 가중치 (0.0-1.0)
        """
        self.carbon_weight = carbon_weight
        self.cost_weight = cost_weight
        
        super().__init__(
            config_path=config_path,
            name="multi_objective",
            description="탄소발자국과 비용을 동시에 고려하는 다목적 최적화 시나리오"
        )
    
    def _configure_scenario(self) -> None:
        """다목적 최적화 시나리오 설정 적용"""
        # 기본 설정에 시나리오가 있는지 확인하고 적용
        available_scenarios = self.opt_input.get_available_scenarios()
        
        if 'multi_objective' in available_scenarios:
            self.opt_input.apply_scenario('multi_objective')
        else:
            # 시나리오가 없으면 수동으로 설정
            custom_config = {
                'objective': 'multi_objective',
                'objective_options': {
                    'multi_objective_weights': {
                        'carbon': self.carbon_weight,
                        'cost': self.cost_weight
                    }
                },
                'constraints': {
                    'target_carbon': 60.0,  # 기준값
                    'max_cost': 50000.0,    # 최대 비용 제한
                    'max_activities': 10    # 활동 수 제한
                },
                'decision_vars': {
                    'cathode': {
                        'type': 'B'  # 선형 문제로 시작
                    },
                    'use_binary_variables': True  # 이진 변수 활성화
                }
            }
            self.opt_input.create_custom_config(**custom_config)
    
    def _configure_model(self, model) -> None:
        """
        다목적 최적화 시나리오를 위한 모델 추가 설정
        
        Args:
            model: Pyomo 모델
        """
        # 기본 다목적 최적화는 기본 클래스에서 이미 처리됨
        pass
    
    def select_solver(self) -> str:
        """
        다목적 최적화에 적합한 솔버 선택
        
        Returns:
            str: 선택된 솔버 이름
        """
        # 이진 변수가 있으면 MIP 솔버, 없으면 LP 솔버
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        if use_binary:
            # 정수 프로그래밍 (MILP)
            return 'cbc'
        else:
            # 선형 문제
            return 'glpk'
    
    def set_objective_weights(self, carbon_weight: float, cost_weight: float) -> None:
        """
        목적함수 가중치 설정
        
        Args:
            carbon_weight: 탄소발자국 목적함수의 가중치 (0.0-1.0)
            cost_weight: 비용 목적함수의 가중치 (0.0-1.0)
        """
        # 가중치 합이 1이 되도록 정규화
        total = carbon_weight + cost_weight
        if total > 0:
            carbon_weight = carbon_weight / total
            cost_weight = cost_weight / total
        else:
            # 기본값 설정
            carbon_weight = 0.7
            cost_weight = 0.3
        
        self.carbon_weight = carbon_weight
        self.cost_weight = cost_weight
        
        # 설정 업데이트
        if 'objective_options' not in self.opt_input.config:
            self.opt_input.config['objective_options'] = {}
        
        if 'multi_objective_weights' not in self.opt_input.config['objective_options']:
            self.opt_input.config['objective_options']['multi_objective_weights'] = {}
            
        self.opt_input.config['objective_options']['multi_objective_weights']['carbon'] = carbon_weight
        self.opt_input.config['objective_options']['multi_objective_weights']['cost'] = cost_weight
    
    def get_objective_components(self) -> Tuple[float, float]:
        """
        최적해의 개별 목적함수 값 계산 및 반환
        
        Returns:
            Tuple[float, float]: (탄소발자국, 비용)
        """
        if not self.results or self.results.get('status') != 'optimal':
            return (0.0, 0.0)
        
        # 결과에서 값 추출
        carbon_footprint = self.results.get('carbon_footprint', 0.0)
        cost = self.results.get('cost', 0.0)
        
        # 탄소발자국이 없으면 formatted_results에서 추출 시도
        if carbon_footprint == 0.0 and self.results_processor.formatted_results:
            carbon_str = self.results_processor.formatted_results.get('carbon_footprint', '0.0')
            try:
                carbon_footprint = float(carbon_str.split()[0])
            except:
                pass
        
        # 비용이 없으면 계산 시도
        if cost == 0.0:
            from .formula import OptimizationFormula
            
            formula = OptimizationFormula(self.opt_input)
            variables = self.results.get('variables', {})
            
            if variables:
                cost = formula.calculate_total_cost(variables)
        
        return (carbon_footprint, cost)
    
    def get_formatted_results(self) -> Dict[str, Any]:
        """
        사용자 친화적 결과 반환
        
        Returns:
            Dict: 포맷팅된 결과
        """
        if not self.results_processor.formatted_results:
            if self.results:
                self.results_processor.process_results(self.results)
            else:
                return {'status': 'not_solved'}
                
        results = self.results_processor.formatted_results
        
        # 다목적 최적화 결과 추가
        if results.get('status') == 'optimal':
            carbon_footprint, cost = self.get_objective_components()
            
            # 파레토 효율성 분석 (단순화된 버전)
            # 정규화된 값들을 계산
            base_carbon = self.opt_input.get_constants().get('base_emission', 80.0)
            base_cost = 100000.0  # 기준 비용
            
            norm_carbon = carbon_footprint / base_carbon if base_carbon > 0 else 0
            norm_cost = cost / base_cost if base_cost > 0 else 0
            
            # 가중합으로 계산된 다목적 목적함수 값
            weighted_sum = (self.carbon_weight * norm_carbon + 
                           self.cost_weight * norm_cost)
            
            results['multi_objective_analysis'] = {
                'carbon_footprint': carbon_footprint,
                'cost': cost,
                'weights': {
                    'carbon': self.carbon_weight,
                    'cost': self.cost_weight
                },
                'normalized': {
                    'carbon': norm_carbon,
                    'cost': norm_cost
                },
                'weighted_sum': weighted_sum
            }
        
        return results
    
    def generate_pareto_front(self, num_points: int = 10) -> Dict[str, Any]:
        """
        파레토 프론트 근사치 생성
        
        다양한 가중치 조합으로 최적화를 수행하여 파레토 프론트 근사치를 생성합니다.
        
        Args:
            num_points: 생성할 파레토 프론트의 점 수
            
        Returns:
            Dict: 파레토 프론트 데이터
        """
        if num_points < 2:
            num_points = 2
            
        # 원래 가중치 저장
        original_carbon_weight = self.carbon_weight
        original_cost_weight = self.cost_weight
        
        pareto_points = []
        
        # 다양한 가중치로 최적화 수행
        for i in range(num_points):
            # 가중치 설정 (탄소: 1.0→0.0, 비용: 0.0→1.0)
            carbon_weight = 1.0 - (i / (num_points - 1))
            cost_weight = 1.0 - carbon_weight
            
            self.set_objective_weights(carbon_weight, cost_weight)
            
            # 모델 재구축 및 최적화
            self.model = None  # 모델 재설정
            results = self.solve()
            
            if results.get('status') == 'optimal':
                carbon_footprint, cost = self.get_objective_components()
                
                pareto_points.append({
                    'carbon_weight': carbon_weight,
                    'cost_weight': cost_weight,
                    'carbon_footprint': carbon_footprint,
                    'cost': cost
                })
        
        # 원래 가중치 복원
        self.set_objective_weights(original_carbon_weight, original_cost_weight)
        
        # 파레토 효율성 검사
        efficient_points = []
        for i, point in enumerate(pareto_points):
            is_efficient = True
            
            for j, other in enumerate(pareto_points):
                if i != j:
                    # point가 other에 지배당하는지 검사
                    if (other['carbon_footprint'] <= point['carbon_footprint'] and 
                        other['cost'] <= point['cost'] and
                        (other['carbon_footprint'] < point['carbon_footprint'] or 
                         other['cost'] < point['cost'])):
                        is_efficient = False
                        break
            
            if is_efficient:
                efficient_points.append(point)
        
        return {
            'all_points': pareto_points,
            'efficient_points': efficient_points,
            'status': 'success' if pareto_points else 'failed'
        }
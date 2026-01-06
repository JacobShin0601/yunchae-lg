"""
소재별 최적화 전략 구현 모듈

이 모듈은 소재별로 다른 최적화 전략을 적용할 수 있는 기능을 제공합니다.
"""

from typing import Dict, Any, Optional, List, Union
import pyomo.environ as pyo
from pyomo.environ import Constraint, Objective, minimize, maximize, ConcreteModel


class MaterialSpecificObjective:
    """
    소재별 최적화 목적함수 및 제약조건 관리 클래스
    
    다양한 소재별로 독립적인 최적화 목적과 제약조건을 정의합니다.
    """
    
    def __init__(self, opt_input):
        """
        Args:
            opt_input: 최적화 입력 객체
        """
        self.opt_input = opt_input
        self.model = None
        self.material_objectives = {}
        
    def set_model(self, model: ConcreteModel) -> None:
        """
        Pyomo 모델 설정
        
        Args:
            model: Pyomo 모델
        """
        self.model = model
    
    def define_material_objectives(self) -> None:
        """각 소재별 목적함수 정의"""
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        if not self.opt_input.is_material_specific_enabled():
            return
            
        # 소재별 목적함수 생성
        material_weights = {}  # 소재별 가중치
        total_weight = 0
        
        for material in self.opt_input.get_all_materials():
            material_config = self.opt_input.get_material_config(material)
            if material_config:
                # 소재 가중치 (기본값 1.0)
                weight = material_config.get('weight', 1.0)
                material_weights[material] = weight
                total_weight += weight
                
                # 소재별 목적함수 표현식 생성
                objective_expr = self._create_objective_expression(material, material_config)
                # 객체 검사 대신 직접 저장
                self.material_objectives[material] = objective_expr
        
        # 가중치 정규화 (모든 가중치 합이 1이 되도록)
        if total_weight > 0:
            for material in material_weights:
                material_weights[material] /= total_weight
                
        # 통합 목적함수 정의 (가중합)
        if self.material_objectives:
            @self.model.Objective(sense=pyo.minimize)
            def combined_objective(m):
                return sum(
                    material_weights.get(material, 1.0) * obj_expr 
                    for material, obj_expr in self.material_objectives.items()
                )
    
    def _create_objective_expression(self, material: str, material_config: Dict[str, Any]):
        """
        특정 소재에 대한 목적함수 표현식 생성
        
        Args:
            material: 소재 이름
            material_config: 소재 설정
            
        Returns:
            목적함수 표현식
        """
        strategy = material_config.get('strategy', 'minimize_carbon')
        
        # 해당 소재와 관련된 변수만 추출
        material_vars = []
        reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
        
        for var_name in reduction_vars.keys():
            if hasattr(self.model, var_name) and material in var_name:
                material_vars.append(var_name)
                
        if not material_vars:
            # 변수가 없으면 None 대신 0 반환 (Pyomo에서 안전)으로 처리
            from pyomo.environ import Expression
            return 0
            
        # 전략별 목적함수 표현식
        if strategy == 'minimize_carbon':
            # 탄소발자국 최소화 (감축비율 최대화)
            return -1 * sum(getattr(self.model, var_name) for var_name in material_vars)
            
        elif strategy == 'minimize_cost':
            # 비용 최소화
            use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
            
            if use_binary:
                # 이진 변수가 있는 경우 고정비 + 가변비
                fixed_costs = sum(
                    getattr(self.model, f"{var_name}_active") * 5000  # 활동별 고정 비용 (예시값)
                    for var_name in material_vars
                    if hasattr(self.model, f"{var_name}_active")
                )
                
                variable_costs = sum(
                    getattr(self.model, var_name) * 50  # 변수별 가변 비용 (예시값)
                    for var_name in material_vars
                )
                
                return fixed_costs + variable_costs
            else:
                # 단순 비율 기반 비용
                return sum(
                    getattr(self.model, var_name) * 50  # 변수별 가변 비용 (예시값)
                    for var_name in material_vars
                )
                
        elif strategy == 'maximize_ease':
            # 구현 용이성 최대화 (활동 수 최소화)
            use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
            
            if use_binary:
                return sum(
                    getattr(self.model, f"{var_name}_active")
                    for var_name in material_vars
                    if hasattr(self.model, f"{var_name}_active")
                )
            else:
                # 이진 변수가 없는 경우 활성화 대체 지표로 임계값 사용
                # > 0.1과 같은 비교는 Pyomo에서 직접 사용 불가
                active_count = 0
                for var_name in material_vars:
                    var_val = getattr(self.model, var_name)
                    # 임계값을 양수로 설정 (비율이 최소값보다 크면 활동 간주)
                    active_count += var_val / 100  # 비율에 비례하여 카운트
                return active_count
        
        # 기본 표현식
        return sum(getattr(self.model, var_name) for var_name in material_vars)
    
    def get_material_contribution(self, material: str, variable_values: Dict[str, float]) -> Dict[str, Any]:
        """
        특정 소재의 기여도 분석
        
        Args:
            material: 소재 이름
            variable_values: 최적화 결과 변수 값 딕셔너리
            
        Returns:
            Dict: 소재별 기여도 분석 결과
        """
        material_config = self.opt_input.get_material_config(material)
        if not material_config:
            return {}
            
        strategy = material_config.get('strategy', 'minimize_carbon')
        
        # 해당 소재와 관련된 변수만 추출
        material_vars = {}
        for var_name, value in variable_values.items():
            if material in var_name and not var_name.endswith('_active'):
                material_vars[var_name] = value
                
        # 분석 결과
        analysis = {
            'material': material,
            'strategy': strategy,
            'variables': material_vars,
            'total_reduction': sum(material_vars.values()),
            'avg_reduction': sum(material_vars.values()) / len(material_vars) if material_vars else 0
        }
        
        # 전략별 추가 분석
        if strategy == 'minimize_carbon':
            analysis['carbon_reduction'] = sum(val * 0.1 for val in material_vars.values())  # 예시 계산
            
        elif strategy == 'minimize_cost':
            analysis['cost_saving'] = sum(val * 50 for val in material_vars.values())  # 예시 계산
            
        elif strategy == 'maximize_ease':
            active_count = sum(1 for val in material_vars.values() if val > 0.1)
            analysis['active_activities'] = active_count
            analysis['ease_score'] = 100 - (active_count * 10)  # 활동이 적을수록 높은 점수
            
        return analysis
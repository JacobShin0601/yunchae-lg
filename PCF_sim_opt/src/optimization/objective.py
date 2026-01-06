"""
최적화 모델의 목적함수를 전문적으로 관리하는 모듈
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from pyomo.environ import (
    ConcreteModel, Var, Objective, Expression, 
    minimize, maximize, value
)
from .constant import OptimizationConstants
from .variable import OptimizationVariables
from .input import OptimizationInput

# 시뮬레이션 정렬 목적함수 임포트 (선택적)
try:
    from .simulation_aligned_carbon_objective import SimulationAlignedCarbonObjective
    from .simulation_aligned_cost_objective import SimulationAlignedCostObjective
    from .simulation_aligned_regional_objective import SimulationAlignedRegionalObjective
    SIMULATION_ALIGNED_AVAILABLE = True
except ImportError:
    SIMULATION_ALIGNED_AVAILABLE = False
    print("Warning: SimulationAligned objectives not available. Using simplified calculations.")


class ObjectiveManager:
    """
    최적화 모델의 목적함수를 정의하고 관리하는 클래스
    
    주요 기능:
    - 다양한 목적함수 생성 및 등록
    - 목적함수 전환
    - 다중 목적함수 지원
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
        
        # 목적함수 레지스트리
        self._objective_registry = {}
        self._current_objective = None
    
    def set_model(self, model: ConcreteModel) -> None:
        """
        모델 설정
        
        Args:
            model: Pyomo 모델
        """
        self.model = model
    
    def register_objective(self, name: str, objective_func: Callable, sense: str = "minimize") -> None:
        """
        목적함수 등록
        
        Args:
            name: 목적함수 이름
            objective_func: 목적함수 생성 함수
            sense: 최소화/최대화 ("minimize" 또는 "maximize")
        """
        if sense not in ["minimize", "maximize"]:
            raise ValueError("sense는 'minimize' 또는 'maximize'여야 합니다.")
        
        self._objective_registry[name] = {
            "func": objective_func,
            "sense": sense
        }
    
    def get_available_objectives(self) -> List[str]:
        """
        사용 가능한 목적함수 목록 반환
        
        Returns:
            List[str]: 목적함수 이름 목록
        """
        return list(self._objective_registry.keys())
    
    def get_objective_info(self) -> Dict[str, Dict[str, Any]]:
        """
        목적함수 정보 반환
        
        Returns:
            Dict: 목적함수 정보
        """
        result = {}
        
        for name, obj_info in self._objective_registry.items():
            result[name] = {
                "sense": obj_info["sense"],
                "description": self._get_objective_description(name)
            }
        
        return result
    
    def _get_objective_description(self, name: str) -> str:
        """
        목적함수 설명 반환
        
        Args:
            name: 목적함수 이름
            
        Returns:
            str: 목적함수 설명
        """
        descriptions = {
            "minimize_carbon": "탄소발자국 최소화",
            "minimize_cost": "총 비용 최소화",
            "maximize_ease": "구현 용이성 최대화 (활성화 활동 수 최소화)",
            "multi_objective": "다목적 최적화 (탄소발자국과 비용의 가중합)",
            "minimize_carbon_simulation": "탄소발자국 최소화 (시뮬레이션 정렬)",
            "minimize_cost_simulation": "총 비용 최소화 (시뮬레이션 정렬)",
            "regional_optimization": "지역별 최적화 (시뮬레이션 정렬)"
        }
        
        return descriptions.get(name, f"목적함수: {name}")
    
    def apply_objective(self, name: str) -> None:
        """
        특정 목적함수 적용
        
        Args:
            name: 목적함수 이름
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        if name not in self._objective_registry:
            raise ValueError(f"목적함수 '{name}'이 등록되지 않았습니다.")
        
        # 기존 목적함수 삭제
        if hasattr(self.model, 'objective'):
            self.model.del_component('objective')
        
        # 새 목적함수 적용
        obj_info = self._objective_registry[name]
        obj_func = obj_info["func"]
        obj_sense = obj_info["sense"]
        
        pyomo_sense = minimize if obj_sense == "minimize" else maximize
        
        @self.model.Objective(sense=pyomo_sense)
        def objective(m):
            return obj_func(m)
            
        self._current_objective = name
    
    def get_current_objective(self) -> Optional[str]:
        """
        현재 적용된 목적함수 이름 반환
        
        Returns:
            Optional[str]: 목적함수 이름
        """
        return self._current_objective
    
    def register_standard_objectives(self) -> None:
        """
        표준 목적함수 등록
        """
        # 1. 탄소발자국 최소화
        self.register_objective(
            "minimize_carbon",
            self._create_carbon_objective_expr,
            "minimize"
        )
        
        # 2. 비용 최소화
        self.register_objective(
            "minimize_cost",
            self._create_cost_objective_expr,
            "minimize"
        )
        
        # 3. 용이성 최대화 (활동 수 최소화)
        self.register_objective(
            "maximize_ease",
            self._create_ease_objective_expr,
            "minimize"  # 활동 수를 최소화하므로 minimize
        )
        
        # 4. 다목적 최적화
        self.register_objective(
            "multi_objective",
            self._create_multi_objective_expr,
            "minimize"
        )
        
        # 5. 시뮬레이션 정렬 목적함수들 (사용 가능한 경우)
        if SIMULATION_ALIGNED_AVAILABLE and self._has_simulation_data():
            self.register_objective(
                "minimize_carbon_simulation",
                self._create_simulation_carbon_expr,
                "minimize"
            )
            
            self.register_objective(
                "minimize_cost_simulation",
                self._create_simulation_cost_expr,
                "minimize"
            )
            
            self.register_objective(
                "regional_optimization",
                self._create_simulation_regional_expr,
                "minimize"
            )
    
    def _create_carbon_objective_expr(self, model: ConcreteModel) -> Any:
        """
        탄소발자국 최소화 목적함수 표현식 생성
        시뮬레이션 로직과 정렬된 계산 방식 사용
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        # 시뮬레이션 데이터가 제공된 경우 정밀 계산 사용
        if hasattr(self.opt_input, 'scenario_df') and self.opt_input.scenario_df is not None:
            return self._create_simulation_aligned_carbon_expr(model)
        else:
            # 기존 단순화 방식 (하위 호환성)
            return self._create_simplified_carbon_expr(model)
    
    def _create_simulation_aligned_carbon_expr(self, model: ConcreteModel) -> Any:
        """
        시뮬레이션 로직과 정렬된 탄소발자국 계산
        rule_based.py의 실제 계산 로직 반영
        """
        from .simulation_aligned_carbon_objective import SimulationAlignedCarbonObjective
        
        # 시뮬레이션 정렬 목적함수 객체 생성
        sim_objective = SimulationAlignedCarbonObjective(
            scenario_df=self.opt_input.scenario_df,
            ref_formula_df=self.opt_input.ref_formula_df,
            ref_proportions_df=self.opt_input.ref_proportions_df,
            original_df=getattr(self.opt_input, 'original_df', None)
        )
        
        # 정렬된 탄소발자국 표현식 생성
        return sim_objective.create_carbon_objective_expression(model)
    
    def _create_simplified_carbon_expr(self, model: ConcreteModel) -> Any:
        """
        기존 단순화된 탄소발자국 계산 (하위 호환성)
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        cathode_config = self.variables.get_cathode_config()
        cathode_type = self.variables.get_cathode_type()
        reduction_vars = self.variables.get_reduction_variables()
        base_emission = self.constants.get('base_emission', 80)
        location_factor = self.constants.get_location_factor(
            self.opt_input.config.get('decision_vars', {}).get('location', '한국')
        )
        
        # 기본 배출량
        base = base_emission * location_factor
        
        # Tier별 감축 효과 (단순화된 계산)
        reduction_effect = sum(
            getattr(model, var_name) * self.constants.get('reduction_effect_coefficient', 0.1)
            for var_name in reduction_vars.keys()
        )
        
        # 양극재 효과 계산
        if cathode_type == 'A':
            # Type A: 원료구성이 변수, 비율은 고정
            cathode_effect = model.low_carbon_emission * model.low_carbon_ratio
            recycle_effect = model.recycle_ratio * (1 - self.constants.get_recycle_impact())
        else:
            # Type B: 원료구성이 고정, 비율이 변수
            cathode_effect = model.low_carbon_emission * model.low_carbon_ratio
            recycle_effect = model.recycle_ratio * (1 - self.constants.get_recycle_impact())
        
        # 총 배출량 (기존 단순화 방식)
        total_emission = base - reduction_effect - cathode_effect - recycle_effect
        return total_emission
    
    def _create_cost_objective_expr(self, model: ConcreteModel) -> Any:
        """
        비용 최소화 목적함수 표현식 생성
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        reduction_vars = self.variables.get_reduction_variables()
        cathode_type = self.variables.get_cathode_type()
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        # 감축 비율에 따른 비용
        if use_binary:
            # 이진 변수가 있는 경우 고정 비용 + 가변 비용
            fixed_costs = sum(
                getattr(model, f"{var_name}_active") * self.constants.get_activity_cost(var_name)
                for var_name in reduction_vars.keys()
                if hasattr(model, f"{var_name}_active")
            )
            
            variable_costs = sum(
                getattr(model, var_name) * self.constants.get('variable_cost_per_percent', 50)
                for var_name in reduction_vars.keys()
            )
            
            reduction_cost = fixed_costs + variable_costs
        else:
            # 단순 비율 기반 비용
            reduction_cost = sum(
                getattr(model, var_name) * self.constants.get('variable_cost_per_percent', 50)
                for var_name in reduction_vars.keys()
            )
        
        # 양극재 구성에 따른 비용
        if cathode_type == 'B':
            material_cost = (
                model.recycle_ratio * self.constants.get('material_costs.recycle_material_cost', 500) +
                model.low_carbon_ratio * self.constants.get('material_costs.low_carbon_material_cost', 800)
            )
        else:
            material_cost = (
                model.recycle_ratio * self.constants.get('material_costs.recycle_material_cost', 500) +
                model.low_carbon_ratio * self.constants.get('material_costs.low_carbon_material_cost', 800)
            )
        
        # 총 비용
        total_cost = reduction_cost + material_cost
        return total_cost
    
    def _create_ease_objective_expr(self, model: ConcreteModel) -> Any:
        """
        용이성 최대화 목적함수 표현식 생성 (활동 수 최소화)
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        reduction_vars = self.variables.get_reduction_variables()
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        if use_binary:
            # 이진 변수 사용 - 활성화된 활동 수 최소화
            return sum(
                getattr(model, f"{var_name}_active")
                for var_name in reduction_vars.keys()
                if hasattr(model, f"{var_name}_active")
            )
        else:
            # 이진 변수 없음 - 근사 방법 사용
            # 작은 상수를 더해 0으로 나누기 방지
            return sum(
                getattr(model, var_name) / (getattr(model, var_name) + 0.001)
                for var_name in reduction_vars.keys()
            )
    
    def _create_multi_objective_expr(self, model: ConcreteModel) -> Any:
        """
        다목적 최적화 목적함수 표현식 생성
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        # 가중치 설정
        weights = self.opt_input.config.get('objective_options', {}).get('multi_objective_weights', {})
        carbon_weight = weights.get('carbon', 0.7)
        cost_weight = weights.get('cost', 0.3)
        
        # 탄소발자국 목적함수
        carbon_obj = self._create_carbon_objective_expr(model)
        
        # 비용 목적함수
        cost_obj = self._create_cost_objective_expr(model)
        
        # 정규화를 위한 참조값 (대략적 스케일)
        carbon_ref = 80.0  # 기본 배출량 참조
        cost_ref = 100000.0  # 비용 참조값
        
        # 정규화된 다목적 함수
        normalized_carbon = carbon_obj / carbon_ref
        normalized_cost = cost_obj / cost_ref
        
        return carbon_weight * normalized_carbon + cost_weight * normalized_cost
    
    def _has_simulation_data(self) -> bool:
        """시뮬레이션 데이터 존재 여부 확인"""
        return (hasattr(self.opt_input, 'scenario_df') and 
                self.opt_input.scenario_df is not None and
                len(self.opt_input.scenario_df) > 0)
    
    def _create_simulation_carbon_expr(self, model: ConcreteModel) -> Any:
        """
        시뮬레이션 정렬 탄소발자국 목적함수 표현식 생성
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        if not SIMULATION_ALIGNED_AVAILABLE:
            return self._create_simplified_carbon_expr(model)
        
        # 시뮬레이션 정렬 목적함수 객체 생성
        sim_objective = SimulationAlignedCarbonObjective(
            scenario_df=self.opt_input.scenario_df,
            ref_formula_df=getattr(self.opt_input, 'ref_formula_df', None),
            ref_proportions_df=getattr(self.opt_input, 'ref_proportions_df', None),
            original_df=getattr(self.opt_input, 'original_df', None)
        )
        
        return sim_objective.create_carbon_objective_expression(model)
    
    def _create_simulation_cost_expr(self, model: ConcreteModel) -> Any:
        """
        시뮬레이션 정렬 비용 목적함수 표현식 생성
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        if not SIMULATION_ALIGNED_AVAILABLE:
            return self._create_cost_objective_expr(model)
        
        # 자재 매칭 정보 생성
        material_matching_info = self._create_material_matching_info()
        
        # 시뮬레이션 정렬 비용 목적함수 객체 생성
        sim_cost_objective = SimulationAlignedCostObjective(
            opt_input=self.opt_input,
            material_matching_info=material_matching_info
        )
        
        return sim_cost_objective.create_cost_objective_expression(model)
    
    def _create_simulation_regional_expr(self, model: ConcreteModel) -> Any:
        """
        시뮬레이션 정렬 지역 목적함수 표현식 생성
        
        Args:
            model: Pyomo 모델
            
        Returns:
            Any: 목적함수 표현식
        """
        if not SIMULATION_ALIGNED_AVAILABLE:
            return self._create_carbon_objective_expr(model)
        
        # 자재 매칭 정보 생성
        material_matching_info = self._create_material_matching_info()
        
        # 시뮬레이션 정렬 지역 목적함수 객체 생성
        sim_regional_objective = SimulationAlignedRegionalObjective(
            opt_input=self.opt_input,
            material_matching_info=material_matching_info,
            target_regions=["한국", "중국", "일본", "미국", "독일"]
        )
        
        return sim_regional_objective.create_regional_objective_expression(model)
    
    def _create_material_matching_info(self) -> Dict[str, Dict]:
        """자재 매칭 정보 생성"""
        material_matching_info = {}
        
        if hasattr(self.opt_input, 'scenario_df') and self.opt_input.scenario_df is not None:
            scenario_df = self.opt_input.scenario_df
            
            for idx, row in scenario_df.iterrows():
                material_key = f"{row.get('자재명', '')}_{row.get('자재품목', '')}_{idx}"
                
                material_info = {
                    'material_name': row.get('자재명', ''),
                    'material_category': row.get('자재품목', ''),
                    'baseline_amount': row.get('제품총소요량(kg)', 0.0),
                    'baseline_emission': row.get('배출계수', 0.0),
                    'baseline_cost': row.get('재료단가', 10.0),
                    'index': idx
                }
                
                material_matching_info[material_key] = material_info
        
        return material_matching_info
    
    def create_custom_objective(self, name: str, objective_expr: Callable, sense: str = "minimize") -> None:
        """
        사용자 정의 목적함수 생성 및 등록
        
        Args:
            name: 목적함수 이름
            objective_expr: 목적함수 표현식을 생성하는 함수
            sense: 최소화/최대화 ("minimize" 또는 "maximize")
        """
        self.register_objective(name, objective_expr, sense)
    
    def apply_objective_from_config(self) -> None:
        """
        설정에서 지정된 목적함수 적용
        """
        objective_type = self.opt_input.get_objective()
        
        # 등록된 목적함수 중에 있는지 확인
        if objective_type in self._objective_registry:
            self.apply_objective(objective_type)
        else:
            raise ValueError(f"설정에 지정된 목적함수 '{objective_type}'이 등록되지 않았습니다.")
    
    def define_objective_from_name(self, name: str) -> None:
        """
        이름으로 목적함수 정의
        
        Args:
            name: 목적함수 이름
        """
        if not self.model:
            raise ValueError("모델이 설정되지 않았습니다. set_model()을 먼저 호출하세요.")
            
        # 표준 목적함수 등록 (아직 등록되지 않은 경우)
        if not self._objective_registry:
            self.register_standard_objectives()
        
        # 목적함수 적용
        self.apply_objective(name)
    
    def define_all_standard_objectives(self) -> None:
        """
        모든 표준 목적함수 등록
        """
        self.register_standard_objectives()
"""
비용 최소화 시나리오 구현 모듈
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .scenario_base import OptimizationScenario
from .input import OptimizationInput
import pyomo.environ
from pyomo.environ import Constraint

# 시뮬레이션 정렬 목적함수 임포트 (선택적)
try:
    from .simulation_aligned_cost_objective import SimulationAlignedCostObjective
    SIMULATION_ALIGNED_COST_AVAILABLE = True
except ImportError:
    SIMULATION_ALIGNED_COST_AVAILABLE = False
    print("Warning: SimulationAlignedCostObjective not available. Using simplified cost calculation.")


class CostMinimization(OptimizationScenario):
    """
    비용 최소화 시나리오
    
    이 클래스는 총 구현 비용을 최소화하는 최적화 문제를 구성하고 해결합니다.
    시뮬레이션 데이터가 제공되면 rule_based.py의 실제 비용 계산 로직을 사용합니다.
    """
    
    def __init__(self, opt_input: OptimizationInput = None, config_path: Optional[str] = None):
        """
        Args:
            opt_input: 최적화 입력 객체 (시뮬레이션 데이터 포함 가능)
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
        """
        # OptimizationInput 객체 설정
        if opt_input is not None:
            self.opt_input = opt_input
        
        super().__init__(
            config_path=config_path,
            name="cost_minimization",
            description="총 비용 최소화 시나리오 (시뮬레이션 정렬 지원)"
        )
        
        # 시뮬레이션 정렬 목적함수 초기화
        self.simulation_aligned_cost_objective = None
        if SIMULATION_ALIGNED_COST_AVAILABLE and self._is_simulation_aligned():
            self._initialize_simulation_aligned_objective()
    
    def _configure_scenario(self) -> None:
        """비용 최소화 시나리오 설정 적용"""
        # 기본 설정에 시나리오가 있는지 확인하고 적용
        available_scenarios = self.opt_input.get_available_scenarios()
        
        if 'cost_minimization' in available_scenarios:
            self.opt_input.apply_scenario('cost_minimization')
        else:
            # 시나리오가 없으면 수동으로 설정
            custom_config = {
                'objective': 'minimize_cost',
                'constraints': {
                    'target_carbon': 50.0,       # 탄소발자국 제한
                    'max_activities': 8,        # 활동 수 제한
                    'max_premium_ratio': 0.05   # 프리미엄 비용 비율 제한 (5%)
                },
                'decision_vars': {
                    'cathode': {
                        'type': 'B'  # 선형 문제로 시작
                    },
                    'use_binary_variables': True,  # 이진 변수 활성화 (고정 비용 포함)
                    'optimize_reduction_rates': True,  # 새로 추가: tier별 감축률을 최적화 변수로 사용
                    # tier별 감축률에 cap 적용
                    'reduction_rates': {
                        'tier1_양극재': {
                            'min': 0,
                            'max': 100,
                            'default': 20,
                            'cap': 30.0,  # 최대 30%까지만 가능
                            'description': "Tier1 양극재 감축비율 (%)"
                        },
                        'tier1_분리막': {
                            'min': 0,
                            'max': 100,
                            'default': 15,
                            'cap': 25.0,  # 최대 25%까지만 가능
                            'description': "Tier1 분리막 감축비율 (%)"
                        },
                        'tier2_양극재': {
                            'min': 0,
                            'max': 100,
                            'default': 30,
                            'cap': 40.0,  # 최대 40%까지만 가능
                            'description': "Tier2 양극재 감축비율 (%)"
                        },
                        'tier2_저탄소원료': {
                            'min': 0,
                            'max': 100,
                            'default': 35,
                            'cap': 45.0,  # 최대 45%까지만 가능
                            'description': "Tier2 저탄소원료 감축비율 (%)"
                        }
                    },
                    # 시뮬레이션 정렬 지원
                    'simulation_aligned': self._is_simulation_aligned()
                }
            }
            self.opt_input.create_custom_config(**custom_config)
    
    def _is_simulation_aligned(self) -> bool:
        """시뮬레이션 정렬 모드인지 확인"""
        return (hasattr(self.opt_input, 'scenario_df') and 
                self.opt_input.scenario_df is not None and
                len(self.opt_input.scenario_df) > 0)
    
    def _initialize_simulation_aligned_objective(self):
        """시뮬레이션 정렬 목적함수 초기화"""
        try:
            # 자재 매칭 정보 생성
            material_matching_info = self._create_material_matching_info()
            
            # 시뮬레이션 정렬 목적함수 생성
            self.simulation_aligned_cost_objective = SimulationAlignedCostObjective(
                opt_input=self.opt_input,
                material_matching_info=material_matching_info
            )
            
            print("✅ 시뮬레이션 정렬 비용 목적함수 초기화 완료")
            
        except Exception as e:
            print(f"⚠️ 시뮬레이션 정렬 목적함수 초기화 실패: {e}")
            self.simulation_aligned_cost_objective = None
    
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
                    'baseline_cost': row.get('재료단가', 10.0),  # 기본값
                    'index': idx
                }
                
                material_matching_info[material_key] = material_info
        
        return material_matching_info
    
    def _configure_model(self, model) -> None:
        """
        비용 최소화 시나리오를 위한 모델 추가 설정
        
        Args:
            model: Pyomo 모델
        """
        # 프리미엄 비율 제약조건 추가
        self._add_premium_ratio_constraint(model)
        
        # tier별 총 감축량 제약조건 추가
        self._add_total_reduction_constraint(model)
        
        # 소재별 최적화 전략이 활성화되어 있으면 소재별 제약조건 추가
        if self.opt_input.is_material_specific_enabled():
            self._add_material_specific_constraints(model)
    
    def _add_premium_ratio_constraint(self, model) -> None:
        """
        프리미엄 비용 비율 제약조건 추가
        
        Args:
            model: Pyomo 모델
        """
        # 프리미엄 비율 제한 가져오기
        max_premium_ratio = self.opt_input.get_constraint('max_premium_ratio', 0.05)
        
        # 기본 제품 비용 (예시값)
        base_product_cost = 1.0
        
        # cost_by_tier.json의 데이터 가져오기
        constants = self.opt_input.get_constants()
        cost_by_tier_data = constants.get('cost_by_tier', {}).get('tier_data', [])
        
        # 1. 감축 비율에 따른 프리미엄 비용
        reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
        
        # 단순화된 제약조건 추가 - 각 tier별로 감축 비율에 cap 적용
        for var_name, config in reduction_vars.items():
            if hasattr(model, var_name) and 'cap' in config:
                cap_val = config.get('cap')
                if cap_val is not None:
                    # 해당 변수에 cap 제약조건 추가
                    def make_cap_rule(var_name, cap):
                        def cap_rule(m):
                            return getattr(m, var_name) <= cap
                        return cap_rule
                    
                    constraint_name = f"{var_name}_cap_constraint"
                    model.add_component(constraint_name, 
                                        pyomo.environ.Constraint(rule=make_cap_rule(var_name, cap_val)))
                    
        # 총 프리미엄 제한 추가 - 단순 테스트용 제약조건
        # 고정 본툰 제약조건을 생략하고 변수에 cap만 적용
    
    def _add_total_reduction_constraint(self, model) -> None:
        """
        tier별 총 감축량 제약조건 추가
        
        Args:
            model: Pyomo 모델
        """
        # 설정에서 tier별 제약조개 가져오기
        reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
        
        # 소재별 최적화가 활성화되어 있다면 여기서는 전체 tier 감축량을 제한하지 않음
        if self.opt_input.is_material_specific_enabled():
            return
            
        # tier별 변수 그룹화
        tier_groups = {}
        
        for var_name in reduction_vars.keys():
            if hasattr(model, var_name):
                # tier 추출 (예: tier1_양극재 -> tier1)
                tier = var_name.split('_')[0]
                
                if tier not in tier_groups:
                    tier_groups[tier] = []
                
                tier_groups[tier].append(var_name)
        
        # tier별 총 감축량 제약조건 추가
        constraint_counter = 0
        for tier, var_names in tier_groups.items():
            # 각 tier별 최대 총 감축량 설정 (예: tier1은 총 50% 이하)
            max_total = 50.0 if tier == 'tier1' else 70.0
            constraint_name = f'tier_total_constraint_{constraint_counter}'
            
            def make_constraint_rule(tier_vars, max_val):
                def constraint_rule(m):
                    return sum(getattr(m, var_name) for var_name in tier_vars) <= max_val
                return constraint_rule
            
            model.add_component(constraint_name, 
                             pyomo.environ.Constraint(rule=make_constraint_rule(var_names, max_total)))
            constraint_counter += 1
    
    def _add_material_specific_constraints(self, model) -> None:
        """
        소재별 최적화 제약조건 추가
        
        Args:
            model: Pyomo 모델
        """
        # 모든 소재 목록 가져오기
        materials = self.opt_input.get_all_materials()
        
        for material in materials:
            # 소재별 설정 가져오기
            material_config = self.opt_input.get_material_config(material)
            if not material_config:
                continue
                
            # 1. 최대 감축량 제약조건 추가 (material별 곥통)
            max_reduction = material_config.get('constraints', {}).get('max_reduction', 50.0)
            
            # 해당 소재와 관련된 변수만 찾기
            material_vars = []
            reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
            
            for var_name in reduction_vars.keys():
                if hasattr(model, var_name) and material in var_name:
                    material_vars.append(var_name)
            
            # 소재별 최대 감축량 제약조건 추가
            if material_vars:
                constraint_name = f"{material}_max_reduction_constraint"
                
                def make_constraint_rule(material_vars, max_val):
                    def constraint_rule(m):
                        return sum(getattr(m, var_name) for var_name in material_vars) <= max_val
                    return constraint_rule
                
                model.add_component(constraint_name, 
                                 pyomo.environ.Constraint(rule=make_constraint_rule(material_vars, max_reduction)))
            
            # 2. 소재별 추가 제약조건
            strategy = material_config.get('strategy', 'minimize_carbon')
            constraints = material_config.get('constraints', {})
            
            # 전략에 따른 추가 제약조건
            if strategy == 'minimize_cost':
                # 탄소발자국 제한
                if 'target_carbon' in constraints and material_vars:
                    target_carbon = constraints['target_carbon']
                    constraint_name = f"{material}_target_carbon_constraint"
                    
                    # 해당 소재의 탄소발자국 계산 직접 정의
                    def make_carbon_constraint_rule(material_vars, target):
                        def constraint_rule(m):
                            # 단순화된 탄소발자국 계산 (Tier별 감축률에 비례)
                            material_carbon = 80.0 - sum(getattr(m, var_name) * 0.1 for var_name in material_vars)
                            return material_carbon <= target
                        return constraint_rule
                    
                    model.add_component(constraint_name, 
                                     pyomo.environ.Constraint(rule=make_carbon_constraint_rule(material_vars, target_carbon)))
                    
            elif strategy == 'maximize_ease':
                # 용이성 최대화는 활동 수 최소화를 의미
                if 'max_activities' in constraints and material_vars:
                    max_activities = constraints['max_activities']
                    
                    # 이진 변수가 있을 때만 적용 가능
                    use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
                    if use_binary:
                        active_vars = [f"{var_name}_active" for var_name in material_vars 
                                     if hasattr(model, f"{var_name}_active")]
                        
                        if active_vars:
                            constraint_name = f"{material}_max_activities_constraint"
                            
                            def make_activities_constraint_rule(active_vars, max_val):
                                def constraint_rule(m):
                                    return sum(getattr(m, var_name) for var_name in active_vars) <= max_val
                                return constraint_rule
                            
                            model.add_component(constraint_name, 
                                             pyomo.environ.Constraint(rule=make_activities_constraint_rule(active_vars, max_activities)))
    
    def select_solver(self) -> str:
        """
        비용 최소화에 적합한 솔버 선택
        
        Returns:
            str: 선택된 솔버 이름
        """
        # 이진 변수가 있으면 MIP 솔버 선택
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        
        # 시뮬레이션 정렬 모드에서는 더 정교한 솔버 선택
        if self._is_simulation_aligned():
            return 'glpk'  # 시뮬레이션 정렬에서는 GLPK 사용
        
        # 테스트용으로 항상 GLPK 사용 (테스트 환경에서 CBC가 없을 수 있음)
        return 'glpk'
    
    def get_total_cost(self) -> float:
        """
        최적해의 총 비용 값 계산 및 반환
        
        Returns:
            float: 총 비용
        """
        if not self.results or self.results.get('status') != 'optimal':
            return 0.0
            
        # 결과에 목적함수 값이 총 비용
        return self.results.get('objective_value', 0.0)
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """
        비용 항목별 분석 결과 반환
        
        Returns:
            Dict: 비용 분석 결과
        """
        if not self.results or self.results.get('status') != 'optimal':
            return {'status': 'error', 'message': '최적화 결과가 없습니다.'}
        
        # 시뮬레이션 정렬 모드인 경우 정교한 비용 분석 사용
        if self.simulation_aligned_cost_objective:
            try:
                return self.simulation_aligned_cost_objective.get_cost_breakdown_from_results(self.results)
            except Exception as e:
                print(f"⚠️ 시뮬레이션 정렬 비용 분석 실패, 기본 분석 사용: {e}")
                # 기본 분석으로 폴백
                pass
            
        # 비용 분석을 위한 변수값 추출
        variables = self.results.get('variables', {})
        
        # 항목별 비용 계산
        fixed_costs = 0.0
        variable_costs = 0.0
        material_costs = 0.0
        tier_costs = {
            'tier1': 0.0,
            'tier2': 0.0,
            'tier3': 0.0
        }
        
        # 기본 제품 비용
        base_cost = 1.0
        
        # Tier별 상세 비용 계산
        constants = self.opt_input.get_constants()
        cost_by_tier_data = constants.get('cost_by_tier', {}).get('tier_data', [])
        
        # Tier별 국가별 비용 계산
        country_costs = {}
        
        # 프리미엄 비용 계산
        premium_costs = {}
        total_premium_cost = 0.0
        
        # 활성화된 활동에 대한 고정 비용
        use_binary = self.opt_input.config.get('decision_vars', {}).get('use_binary_variables', False)
        if use_binary:
            reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
            for var_name in reduction_vars.keys():
                active_var = f"{var_name}_active"
                if active_var in variables and variables[active_var] > 0.5:
                    # tier와 material 추출 (예: tier1_양극재 -> Tier1, 양극재)
                    parts = var_name.split('_', 1)
                    tier = f"Tier{parts[0][4:]}" if parts[0].startswith('tier') else parts[0]
                    material = parts[1] if len(parts) > 1 else ''
                    
                    # 기본 비용 계산
                    activity_cost = constants.get_activity_cost(var_name)
                    fixed_costs += activity_cost
                    
                    # tier별 비용 분류
                    tier_key = tier.lower()
                    if tier_key in tier_costs:
                        tier_costs[tier_key] += activity_cost
                    
                    # 프리미엄 비용 계산
                    if var_name in variables:
                        # 해당 조합에 맞는 프리미엄 비용 찾기
                        for item in cost_by_tier_data:
                            if item.get('tier') == tier and material in item.get('material', ''):
                                premium = variables[var_name] * item.get('expected_cost', 0.0) / 100.0
                                
                                # 프리미엄 비용 기록
                                premium_costs[var_name] = premium
                                total_premium_cost += premium
                                break
                    
                    # 국가별 비용 추가 반영 (생산 위치 정보가 있는 경우)
                    production_location = variables.get('production_location', None)
                    if production_location and material:
                        cost_from_tier = constants.get_tier_cost(tier, material, production_location)
                        
                        # 국가별 비용 합계
                        if production_location not in country_costs:
                            country_costs[production_location] = 0.0
                        country_costs[production_location] += cost_from_tier
        
        # 감축 비율에 따른 가변 비용
        reduction_vars = self.opt_input.config.get('decision_vars', {}).get('reduction_rates', {})
        variable_cost_per_percent = constants.get('variable_cost_per_percent', 50)
        for var_name in reduction_vars.keys():
            if var_name in variables:
                variable_costs += variables[var_name] * variable_cost_per_percent
        
        # 양극재 재료 비용
        recycle_material_cost = constants.get('material_costs.recycle_material_cost', 500)
        low_carbon_material_cost = constants.get('material_costs.low_carbon_material_cost', 800)
        
        if 'recycle_ratio' in variables:
            material_costs += variables['recycle_ratio'] * recycle_material_cost
            
        if 'low_carbon_ratio' in variables:
            material_costs += variables['low_carbon_ratio'] * low_carbon_material_cost
        
        # 총 비용
        total_cost = fixed_costs + variable_costs + material_costs
        
        # 비용 분석 결과 구성
        cost_analysis = {
            'base_cost': base_cost,
            'premium_cost': total_premium_cost,
            'premium_ratio': total_premium_cost / base_cost if base_cost > 0 else 0,
            'premium_percentage': f"{(total_premium_cost / base_cost if base_cost > 0 else 0) * 100:.2f}%",
            'total_cost': total_cost,
            'fixed_costs': fixed_costs,
            'variable_costs': variable_costs,
            'material_costs': material_costs,
            'tier_costs': tier_costs,
            'breakdown': {
                '기본 비용': base_cost,
                '프리미엄 비용': total_premium_cost,
                '고정 비용 (활동 시작비용)': fixed_costs,
                '가변 비용 (감축 비율에 따른)': variable_costs,
                '재료 비용 (재활용재, 저탄소원료)': material_costs
            }
        }
        
        # 프리미엄 비용 상세 항목
        if premium_costs:
            premium_breakdown = {}
            for var_name, premium in premium_costs.items():
                premium_breakdown[var_name] = premium
            cost_analysis['premium_breakdown'] = premium_breakdown
        
        # 국가별 비용이 있으면 추가
        if country_costs:
            cost_analysis['country_costs'] = country_costs
            
            # 비용 분석 결과에 국가별 항목 추가
            country_cost_breakdown = {}
            for country, cost in country_costs.items():
                country_cost_breakdown[f'생산국가 {country}'] = cost
            cost_analysis['breakdown'].update(country_cost_breakdown)
        
        # tier 데이터가 있으면 상세 분석 추가
        if cost_by_tier_data:
            tier_material_breakdown = {}
            for tier in ['Tier1', 'Tier2']:
                materials = set(item['material'] for item in cost_by_tier_data if item['tier'] == tier)
                for material in materials:
                    material_info = constants.get_material_cost_by_tier(tier, material)
                    if material_info and material_info['costs']:
                        key = f"{tier} {material}"
                        value = sum(item['cost'] for item in material_info['costs'])
                        tier_material_breakdown[key] = value
            
            cost_analysis['tier_material_costs'] = tier_material_breakdown
        
        cost_analysis['status'] = 'success'
        # 시뮬레이션 정렬 추가 정보
        if self._is_simulation_aligned():
            cost_analysis['simulation_aligned'] = True
            cost_analysis['simulation_data_size'] = len(self.opt_input.scenario_df) if hasattr(self.opt_input, 'scenario_df') else 0
        else:
            cost_analysis['simulation_aligned'] = False
        
        return cost_analysis
    
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
        
        # 비용 분석 추가
        if results.get('status') == 'optimal':
            cost_breakdown = self.get_cost_breakdown()
            if cost_breakdown.get('status') == 'success':
                results['cost_analysis'] = cost_breakdown
        
        # 시뮬레이션 정렬 정보 추가
        results['simulation_aligned'] = self._is_simulation_aligned()
        if self._is_simulation_aligned():
            results['simulation_info'] = {
                'data_source': 'rule_based_simulation',
                'material_count': len(self.opt_input.scenario_df) if hasattr(self.opt_input, 'scenario_df') else 0,
                'cost_calculation': 'simulation_aligned'
            }
        
        return results
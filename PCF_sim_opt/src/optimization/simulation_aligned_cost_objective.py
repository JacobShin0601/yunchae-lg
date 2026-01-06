"""
시뮬레이션 정렬 비용 목적함수 구현

이 모듈은 rule_based.py의 실제 비용 계산 로직을 반영하여
비용 최소화 최적화 문제의 목적함수를 구성합니다.
"""

import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pyomo.environ import ConcreteModel
import json
import os


class SimulationAlignedCostObjective:
    """
    시뮬레이션과 정렬된 비용 목적함수 클래스
    
    rule_based.py의 실제 비용 계산 로직을 반영하여 
    정확한 비용 최소화 최적화를 수행합니다.
    """
    
    def __init__(self, opt_input, material_matching_info: Dict[str, Dict] = None):
        """
        초기화
        
        Args:
            opt_input: 최적화 입력 객체
            material_matching_info: 자재 매칭 정보
        """
        self.opt_input = opt_input
        self.material_matching_info = material_matching_info or {}
        
        # cost_by_tier.json 데이터 로드
        self.cost_by_tier_data = self._load_cost_by_tier_data()
        
        # 기본 비용 계수들
        self.base_cost_coefficients = self._initialize_base_cost_coefficients()
        
    def _load_cost_by_tier_data(self) -> Dict[str, Any]:
        """cost_by_tier.json 파일 로드"""
        try:
            # stable_var 디렉토리에서 cost_by_tier.json 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            cost_file_path = os.path.join(project_root, 'stable_var', 'cost_by_tier.json')
            
            if os.path.exists(cost_file_path):
                with open(cost_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"Warning: cost_by_tier.json not found at {cost_file_path}")
                return self._get_default_cost_data()
                
        except Exception as e:
            print(f"Error loading cost_by_tier.json: {e}")
            return self._get_default_cost_data()
    
    def _get_default_cost_data(self) -> Dict[str, Any]:
        """기본 비용 데이터 반환"""
        return {
            'tier_data': [
                {
                    'tier': 'Tier1',
                    'material': '양극재',
                    'country': '한국',
                    'expected_cost': 15.0,
                    'min_cost': 10.0,
                    'max_cost': 20.0
                },
                {
                    'tier': 'Tier1',
                    'material': '분리막',
                    'country': '한국', 
                    'expected_cost': 12.0,
                    'min_cost': 8.0,
                    'max_cost': 16.0
                },
                {
                    'tier': 'Tier2',
                    'material': '양극재',
                    'country': '한국',
                    'expected_cost': 25.0,
                    'min_cost': 20.0,
                    'max_cost': 30.0
                },
                {
                    'tier': 'Tier2',
                    'material': '저탄소원료',
                    'country': '한국',
                    'expected_cost': 28.0,
                    'min_cost': 22.0,
                    'max_cost': 35.0
                }
            ],
            'base_costs': {
                'activity_startup_cost': 1000.0,  # 활동 시작 고정 비용
                'variable_cost_per_percent': 50.0,  # 감축률 1%당 가변 비용
                'material_premium_base': 100.0,  # 재료 프리미엄 기본 비용
                'certification_cost_multiplier': 1.2  # RE 인증 비용 배수
            }
        }
    
    def _initialize_base_cost_coefficients(self) -> Dict[str, float]:
        """기본 비용 계수 초기화"""
        base_costs = self.cost_by_tier_data.get('base_costs', {})
        
        return {
            'activity_startup_cost': base_costs.get('activity_startup_cost', 1000.0),
            'variable_cost_per_percent': base_costs.get('variable_cost_per_percent', 50.0),
            'material_premium_base': base_costs.get('material_premium_base', 100.0),
            'certification_cost_multiplier': base_costs.get('certification_cost_multiplier', 1.2),
            'recycling_material_cost': 500.0,  # 재활용 재료 비용
            'low_carbon_material_cost': 800.0,  # 저탄소 재료 비용
            'transportation_cost_per_km': 0.5,  # km당 운송 비용
        }
    
    def create_cost_objective_expression(self, model: ConcreteModel) -> Any:
        """
        비용 최소화 목적함수 표현식 생성
        
        Args:
            model: Pyomo 모델
            
        Returns:
            비용 목적함수 표현식
        """
        total_cost = 0
        
        # 1. 기본 비용 (baseline PCF 기반)
        baseline_cost = self._calculate_baseline_cost()
        total_cost += baseline_cost
        
        # 2. 자재별 비용 계산 (시뮬레이션 정렬)
        for material_key, info in self.material_matching_info.items():
            material_cost = self._calculate_material_cost(model, info)
            total_cost += material_cost
        
        # 3. RE 인증 비용 (tier별)
        re_certification_cost = self._calculate_re_certification_cost(model)
        total_cost += re_certification_cost
        
        # 4. 활동 고정비용 (이진 변수 기반)
        activity_fixed_cost = self._calculate_activity_fixed_cost(model)
        total_cost += activity_fixed_cost
        
        return total_cost
    
    def _calculate_baseline_cost(self) -> float:
        """기본 비용 계산"""
        if hasattr(self.opt_input, 'scenario_df') and self.opt_input.scenario_df is not None:
            # 시뮬레이션 데이터가 있는 경우
            scenario_df = self.opt_input.scenario_df
            
            # 기본 제품 비용 = 총 소요량 * 재료 단가
            if '제품총소요량(kg)' in scenario_df.columns and '재료단가' in scenario_df.columns:
                baseline_cost = (scenario_df['제품총소요량(kg)'] * scenario_df['재료단가']).sum()
            else:
                # 재료단가가 없으면 기본값 사용
                baseline_cost = scenario_df.get('제품총소요량(kg)', pd.Series([0])).sum() * 10.0  # 기본 단가
            
            return baseline_cost
        else:
            # 기본값
            return 10000.0
    
    def _calculate_material_cost(self, model: ConcreteModel, material_info: Dict[str, Any]) -> Any:
        """
        개별 자재의 비용 계산
        
        Args:
            model: Pyomo 모델
            material_info: 자재 정보
            
        Returns:
            자재 비용 표현식
        """
        material_cost = 0
        material_name = material_info.get('material_name', '')
        material_category = material_info.get('material_category', '')
        baseline_amount = material_info.get('baseline_amount', 0.0)
        
        # 1. 기본 재료 비용
        base_material_cost = baseline_amount * self.base_cost_coefficients['material_premium_base']
        material_cost += base_material_cost
        
        # 2. Tier별 프리미엄 비용 (감축률에 따라)
        for tier_num in [1, 2]:
            re_var_name = f'tier{tier_num}_re_application_rate'
            if hasattr(model, re_var_name):
                re_var = getattr(model, re_var_name)
                
                # 해당 tier와 자재에 맞는 비용 찾기
                tier_cost = self._get_tier_material_cost(f'Tier{tier_num}', material_category)
                if tier_cost > 0:
                    # 감축률에 비례한 프리미엄 비용
                    tier_premium = re_var * (tier_cost / 100.0) * baseline_amount
                    material_cost += tier_premium
        
        # 3. 재활용/저탄소 재료 추가 비용
        if '양극재' in material_category:
            # 재활용 비율 비용
            if hasattr(model, 'recycle_ratio'):
                recycle_cost = model.recycle_ratio * baseline_amount * self.base_cost_coefficients['recycling_material_cost'] / 1000.0
                material_cost += recycle_cost
            
            # 저탄소 재료 비용
            if hasattr(model, 'low_carbon_ratio'):
                low_carbon_cost = model.low_carbon_ratio * baseline_amount * self.base_cost_coefficients['low_carbon_material_cost'] / 1000.0
                material_cost += low_carbon_cost
        
        return material_cost
    
    def _get_tier_material_cost(self, tier: str, material_category: str) -> float:
        """
        특정 tier와 자재에 대한 비용 반환
        
        Args:
            tier: Tier 이름 (Tier1, Tier2)
            material_category: 자재 카테고리
            
        Returns:
            해당 조합의 예상 비용
        """
        tier_data = self.cost_by_tier_data.get('tier_data', [])
        
        for item in tier_data:
            if item.get('tier') == tier and material_category in item.get('material', ''):
                return item.get('expected_cost', 0.0)
        
        # 기본값 반환
        if tier == 'Tier1':
            return 15.0  # 기본 Tier1 비용
        else:
            return 25.0  # 기본 Tier2 비용
    
    def _calculate_re_certification_cost(self, model: ConcreteModel) -> Any:
        """
        RE 인증 비용 계산
        
        Args:
            model: Pyomo 모델
            
        Returns:
            RE 인증 비용 표현식
        """
        re_certification_cost = 0
        certification_multiplier = self.base_cost_coefficients['certification_cost_multiplier']
        
        # Tier별 RE 적용률에 따른 인증 비용
        for tier_num in [1, 2]:
            re_var_name = f'tier{tier_num}_re_application_rate'
            if hasattr(model, re_var_name):
                re_var = getattr(model, re_var_name)
                
                # 기본 인증 비용 (tier에 따라 차등)
                base_cert_cost = 1000.0 if tier_num == 1 else 1500.0
                
                # RE 적용률에 비례한 인증 비용
                tier_cert_cost = re_var * base_cert_cost * certification_multiplier / 100.0
                re_certification_cost += tier_cert_cost
        
        return re_certification_cost
    
    def _calculate_activity_fixed_cost(self, model: ConcreteModel) -> Any:
        """
        활동 고정비용 계산 (이진 변수 기반)
        
        Args:
            model: Pyomo 모델
            
        Returns:
            활동 고정비용 표현식
        """
        activity_fixed_cost = 0
        startup_cost = self.base_cost_coefficients['activity_startup_cost']
        
        # 이진 변수로 활성화된 활동에 대한 고정비용
        binary_vars = []
        
        # RE 적용 여부 이진 변수들
        for tier_num in [1, 2]:
            binary_var_name = f'tier{tier_num}_re_active'
            if hasattr(model, binary_var_name):
                binary_var = getattr(model, binary_var_name)
                binary_vars.append(binary_var)
        
        # 재활용/저탄소 재료 사용 여부
        for material_var in ['recycle_active', 'low_carbon_active']:
            if hasattr(model, material_var):
                binary_var = getattr(model, material_var)
                binary_vars.append(binary_var)
        
        # 각 활성화된 활동에 대한 고정비용
        for binary_var in binary_vars:
            activity_cost = binary_var * startup_cost
            activity_fixed_cost += activity_cost
        
        return activity_fixed_cost
    
    def get_cost_breakdown_from_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        최적화 결과로부터 비용 분석 생성
        
        Args:
            results: 최적화 결과
            
        Returns:
            비용 분석 결과
        """
        if not results or results.get('status') != 'optimal':
            return {'status': 'error', 'message': '최적화 결과가 없습니다.'}
        
        variables = results.get('variables', {})
        
        # 각 비용 항목 계산
        cost_breakdown = {
            'baseline_cost': self._calculate_baseline_cost(),
            'material_costs': 0.0,
            're_certification_costs': 0.0,
            'activity_fixed_costs': 0.0,
            'total_cost': results.get('objective_value', 0.0)
        }
        
        # 자재별 비용 계산
        for material_key, info in self.material_matching_info.items():
            material_cost = self._estimate_material_cost_from_variables(variables, info)
            cost_breakdown['material_costs'] += material_cost
        
        # RE 인증 비용 계산
        cost_breakdown['re_certification_costs'] = self._estimate_re_certification_cost_from_variables(variables)
        
        # 활동 고정비용 계산
        cost_breakdown['activity_fixed_costs'] = self._estimate_activity_fixed_cost_from_variables(variables)
        
        # 비용 분석 추가 정보
        cost_breakdown['cost_efficiency'] = self._calculate_cost_efficiency(results)
        cost_breakdown['status'] = 'success'
        
        return cost_breakdown
    
    def _estimate_material_cost_from_variables(self, variables: Dict[str, float], material_info: Dict[str, Any]) -> float:
        """변수 값으로부터 자재 비용 추정"""
        material_cost = 0.0
        baseline_amount = material_info.get('baseline_amount', 0.0)
        material_category = material_info.get('material_category', '')
        
        # 기본 재료 비용
        base_cost = baseline_amount * self.base_cost_coefficients['material_premium_base']
        material_cost += base_cost
        
        # Tier별 프리미엄 비용
        for tier_num in [1, 2]:
            re_var_name = f'tier{tier_num}_re_application_rate'
            if re_var_name in variables:
                re_rate = variables[re_var_name]
                tier_cost = self._get_tier_material_cost(f'Tier{tier_num}', material_category)
                
                tier_premium = re_rate * (tier_cost / 100.0) * baseline_amount
                material_cost += tier_premium
        
        # 재활용/저탄소 재료 비용
        if '양극재' in material_category:
            if 'recycle_ratio' in variables:
                recycle_cost = variables['recycle_ratio'] * baseline_amount * self.base_cost_coefficients['recycling_material_cost'] / 1000.0
                material_cost += recycle_cost
                
            if 'low_carbon_ratio' in variables:
                low_carbon_cost = variables['low_carbon_ratio'] * baseline_amount * self.base_cost_coefficients['low_carbon_material_cost'] / 1000.0
                material_cost += low_carbon_cost
        
        return material_cost
    
    def _estimate_re_certification_cost_from_variables(self, variables: Dict[str, float]) -> float:
        """변수 값으로부터 RE 인증 비용 추정"""
        re_cost = 0.0
        certification_multiplier = self.base_cost_coefficients['certification_cost_multiplier']
        
        for tier_num in [1, 2]:
            re_var_name = f'tier{tier_num}_re_application_rate'
            if re_var_name in variables:
                re_rate = variables[re_var_name]
                base_cert_cost = 1000.0 if tier_num == 1 else 1500.0
                
                tier_cert_cost = re_rate * base_cert_cost * certification_multiplier / 100.0
                re_cost += tier_cert_cost
        
        return re_cost
    
    def _estimate_activity_fixed_cost_from_variables(self, variables: Dict[str, float]) -> float:
        """변수 값으로부터 활동 고정비용 추정"""
        fixed_cost = 0.0
        startup_cost = self.base_cost_coefficients['activity_startup_cost']
        
        # 이진 변수들 확인
        binary_vars = []
        for tier_num in [1, 2]:
            binary_var_name = f'tier{tier_num}_re_active'
            if binary_var_name in variables and variables[binary_var_name] > 0.5:
                binary_vars.append(binary_var_name)
        
        for material_var in ['recycle_active', 'low_carbon_active']:
            if material_var in variables and variables[material_var] > 0.5:
                binary_vars.append(material_var)
        
        fixed_cost = len(binary_vars) * startup_cost
        return fixed_cost
    
    def _calculate_cost_efficiency(self, results: Dict[str, Any]) -> Dict[str, float]:
        """비용 효율성 계산"""
        total_cost = results.get('objective_value', 0.0)
        
        # 탄소 감축량 추정
        carbon_reduction = 0.0
        variables = results.get('variables', {})
        
        for tier_num in [1, 2]:
            re_var_name = f'tier{tier_num}_re_application_rate'
            if re_var_name in variables:
                # 간단한 탄소 감축 추정 (실제로는 더 복잡한 계산 필요)
                tier_reduction = variables[re_var_name] * (0.5 if tier_num == 1 else 0.8)  # tier별 감축 효율
                carbon_reduction += tier_reduction
        
        # 비용 효율성 지표
        cost_per_carbon_reduction = total_cost / (carbon_reduction + 0.001)  # 0으로 나누기 방지
        
        return {
            'total_cost': total_cost,
            'carbon_reduction_estimated': carbon_reduction,
            'cost_per_carbon_reduction': cost_per_carbon_reduction,
            'cost_efficiency_score': 100.0 / (cost_per_carbon_reduction + 0.001)  # 높을수록 효율적
        }
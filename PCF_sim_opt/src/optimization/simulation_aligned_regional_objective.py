"""
시뮬레이션 정렬 지역 목적함수 구현

이 모듈은 rule_based.py의 실제 지역별 계산 로직을 반영하여
지역 최적화 문제의 목적함수를 구성합니다.
"""

import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pyomo.environ import ConcreteModel
import json
import os


class SimulationAlignedRegionalObjective:
    """
    시뮬레이션과 정렬된 지역 목적함수 클래스
    
    rule_based.py의 실제 지역별 계산 로직을 반영하여 
    정확한 지역 최적화를 수행합니다.
    """
    
    def __init__(self, opt_input, material_matching_info: Dict[str, Dict] = None, target_regions: List[str] = None):
        """
        초기화
        
        Args:
            opt_input: 최적화 입력 객체
            material_matching_info: 자재 매칭 정보
            target_regions: 대상 지역 목록
        """
        self.opt_input = opt_input
        self.material_matching_info = material_matching_info or {}
        self.target_regions = target_regions or ["한국", "중국", "일본", "미국", "독일"]
        
        # 지역별 전력 배출계수 데이터 로드
        self.regional_emission_factors = self._load_regional_emission_factors()
        
        # 지역별 비용 데이터 로드
        self.regional_costs = self._load_regional_costs()
        
        # 물류 거리 매트릭스
        self.logistics_matrix = self._initialize_logistics_matrix()
        
    def _load_regional_emission_factors(self) -> Dict[str, float]:
        """지역별 전력 배출계수 로드"""
        try:
            # stable_var 디렉토리에서 electricity_coef_by_country.json 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            
            # 사용자별 디렉토리 확인
            for user_dir in ['sooyoun', 'yunchae', 'yunchae2']:
                emission_file_path = os.path.join(project_root, 'stable_var', user_dir, 'electricity_coef_by_country.json')
                
                if os.path.exists(emission_file_path):
                    with open(emission_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 국가별 전력 배출계수 추출
                        emission_factors = {}
                        for country, info in data.items():
                            emission_factors[country] = info.get('배출계수', 0.5)  # 기본값 0.5
                        return emission_factors
            
            print("Warning: electricity_coef_by_country.json not found, using default values")
            return self._get_default_emission_factors()
                
        except Exception as e:
            print(f"Error loading regional emission factors: {e}")
            return self._get_default_emission_factors()
    
    def _get_default_emission_factors(self) -> Dict[str, float]:
        """기본 지역별 전력 배출계수 반환"""
        return {
            '한국': 0.4781,
            '중국': 0.5810,
            '일본': 0.4570,
            '미국': 0.4120,
            '독일': 0.3380,
            'Korea': 0.4781,
            'China': 0.5810,
            'Japan': 0.4570,
            'USA': 0.4120,
            'Germany': 0.3380
        }
    
    def _load_regional_costs(self) -> Dict[str, Dict[str, float]]:
        """지역별 비용 데이터 로드"""
        return {
            '한국': {
                'labor_cost': 25.0,        # 시간당 인건비 (USD)
                'energy_cost': 0.12,       # kWh당 전력비 (USD)
                'land_cost': 500.0,        # m²당 토지비 (USD/월)
                'logistics_hub_bonus': 0.9  # 물류 허브 보너스 (비용 감소)
            },
            '중국': {
                'labor_cost': 8.0,
                'energy_cost': 0.08,
                'land_cost': 150.0,
                'logistics_hub_bonus': 0.8
            },
            '일본': {
                'labor_cost': 28.0,
                'energy_cost': 0.18,
                'land_cost': 800.0,
                'logistics_hub_bonus': 0.95
            },
            '미국': {
                'labor_cost': 30.0,
                'energy_cost': 0.10,
                'land_cost': 400.0,
                'logistics_hub_bonus': 0.85
            },
            '독일': {
                'labor_cost': 35.0,
                'energy_cost': 0.25,
                'land_cost': 600.0,
                'logistics_hub_bonus': 0.9
            }
        }
    
    def _initialize_logistics_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """물류 거리 및 비용 매트릭스 초기화"""
        # 거리 매트릭스 (km)
        distance_matrix = {
            '한국': {'한국': 0, '중국': 600, '일본': 800, '미국': 9000, '독일': 8000},
            '중국': {'한국': 600, '중국': 0, '일본': 1200, '미국': 10000, '독일': 7500},
            '일본': {'한국': 800, '일본': 0, '중국': 1200, '미국': 8500, '독일': 9000},
            '미국': {'한국': 9000, '중국': 10000, '일본': 8500, '미국': 0, '독일': 6000},
            '독일': {'한국': 8000, '중국': 7500, '일본': 9000, '미국': 6000, '독일': 0}
        }
        
        # 물류 비용 및 배출량 계산
        logistics_matrix = {}
        
        for origin in self.target_regions:
            logistics_matrix[origin] = {}
            
            for destination in self.target_regions:
                distance = distance_matrix.get(origin, {}).get(destination, 0)
                
                # 거리 기반 물류 비용 및 배출량 계산
                cost_per_km = 0.5  # USD per km per ton
                emission_per_km = 0.1  # kg CO2 per km per ton
                
                logistics_matrix[origin][destination] = {
                    'distance': distance,
                    'cost_per_ton': distance * cost_per_km,
                    'emission_per_ton': distance * emission_per_km,
                    'transit_time': distance / 50.0  # 평균 50km/h 가정
                }
        
        return logistics_matrix
    
    def create_regional_objective_expression(self, model: ConcreteModel) -> Any:
        """
        지역 최적화 목적함수 표현식 생성
        
        Args:
            model: Pyomo 모델
            
        Returns:
            지역 목적함수 표현식 (탄소발자국 최소화 기반)
        """
        total_carbon_footprint = 0
        
        # 1. 기본 탄소발자국 (지역 무관)
        baseline_carbon = self._calculate_baseline_carbon()
        total_carbon_footprint += baseline_carbon
        
        # 2. 자재별 지역 의존 탄소발자국
        for material_key, info in self.material_matching_info.items():
            material_carbon = self._calculate_material_regional_carbon(model, info)
            total_carbon_footprint += material_carbon
        
        # 3. 지역별 전력 사용 탄소발자국
        regional_electricity_carbon = self._calculate_regional_electricity_carbon(model)
        total_carbon_footprint += regional_electricity_carbon
        
        # 4. 물류 탄소발자국 (지역 간 운송)
        logistics_carbon = self._calculate_logistics_carbon(model)
        total_carbon_footprint += logistics_carbon
        
        return total_carbon_footprint
    
    def _calculate_baseline_carbon(self) -> float:
        """기본 탄소발자국 계산 (지역 무관 부분)"""
        if hasattr(self.opt_input, 'scenario_df') and self.opt_input.scenario_df is not None:
            scenario_df = self.opt_input.scenario_df
            
            # 지역에 독립적인 기본 배출량
            baseline_carbon = 0.0
            
            for _, row in scenario_df.iterrows():
                material_emission = row.get('배출계수', 0.0) * row.get('제품총소요량(kg)', 0.0)
                
                # 지역 독립적 부분만 계산 (예: 원자재 자체 배출량의 70%)
                region_independent_ratio = 0.7
                baseline_carbon += material_emission * region_independent_ratio
            
            return baseline_carbon
        else:
            return 50.0  # 기본값
    
    def _calculate_material_regional_carbon(self, model: ConcreteModel, material_info: Dict[str, Any]) -> Any:
        """
        자재별 지역 의존 탄소발자국 계산
        
        Args:
            model: Pyomo 모델
            material_info: 자재 정보
            
        Returns:
            자재의 지역별 탄소발자국 표현식
        """
        material_carbon = 0
        baseline_amount = material_info.get('baseline_amount', 0.0)
        baseline_emission = material_info.get('baseline_emission', 0.0)
        
        # 지역에 의존하는 부분 (예: 전력 사용량의 30%)
        region_dependent_ratio = 0.3
        base_regional_emission = baseline_emission * baseline_amount * region_dependent_ratio
        
        # 각 지역의 생산 비율과 해당 지역의 전력 배출계수를 곱함
        for region in self.target_regions:
            # 지역별 생산 비율 변수 (이진 변수 또는 연속 변수)
            production_var_name = f'production_ratio_{region.replace(" ", "_")}'
            
            if hasattr(model, production_var_name):
                production_ratio = getattr(model, production_var_name)
                regional_emission_factor = self.regional_emission_factors.get(region, 0.5)
                
                # 해당 지역에서 생산될 때의 추가 배출량
                regional_carbon = production_ratio * base_regional_emission * regional_emission_factor / 0.5  # 0.5는 기준값
                material_carbon += regional_carbon
        
        return material_carbon
    
    def _calculate_regional_electricity_carbon(self, model: ConcreteModel) -> Any:
        """
        지역별 전력 사용 탄소발자국 계산
        
        Args:
            model: Pyomo 모델
            
        Returns:
            지역별 전력 탄소발자국 표현식
        """
        electricity_carbon = 0
        
        # Tier별 전력 사용량 추정
        for tier_num in [1, 2]:
            re_var_name = f'tier{tier_num}_re_application_rate'
            
            if hasattr(model, re_var_name):
                re_var = getattr(model, re_var_name)
                
                # 각 지역에서의 전력 사용 탄소발자국
                for region in self.target_regions:
                    region_var_name = f'tier{tier_num}_{region.replace(" ", "_")}_active'
                    
                    if hasattr(model, region_var_name):
                        region_active = getattr(model, region_var_name)
                        regional_emission_factor = self.regional_emission_factors.get(region, 0.5)
                        
                        # 기본 전력 사용량 (kWh per kg)
                        base_electricity_usage = 2.5 if tier_num == 1 else 4.0
                        
                        # RE 적용률에 따른 실제 배출계수 조정
                        effective_emission_factor = regional_emission_factor * (1 - re_var / 100.0)
                        
                        # 전력 사용 탄소발자국
                        tier_electricity_carbon = (region_active * base_electricity_usage * 
                                                 effective_emission_factor * 100.0)  # 규모 조정
                        electricity_carbon += tier_electricity_carbon
        
        return electricity_carbon
    
    def _calculate_logistics_carbon(self, model: ConcreteModel) -> Any:
        """
        물류 탄소발자국 계산 (지역 간 운송)
        
        Args:
            model: Pyomo 모델
            
        Returns:
            물류 탄소발자국 표현식
        """
        logistics_carbon = 0
        
        # CAM 및 pCAM 위치 변수 기반 물류 계산
        for cam_region in self.target_regions:
            for pcam_region in self.target_regions:
                
                cam_var_name = f'cam_location_{cam_region.replace(" ", "_")}'
                pcam_var_name = f'pcam_location_{pcam_region.replace(" ", "_")}'
                
                if hasattr(model, cam_var_name) and hasattr(model, pcam_var_name):
                    cam_var = getattr(model, cam_var_name)
                    pcam_var = getattr(model, pcam_var_name)
                    
                    # 두 지역이 모두 선택되었을 때의 물류 배출량
                    logistics_info = self.logistics_matrix.get(cam_region, {}).get(pcam_region, {})
                    emission_per_ton = logistics_info.get('emission_per_ton', 0.0)
                    
                    # 예상 물류량 (톤)
                    estimated_logistics_volume = 10.0  # 기본값
                    
                    # 물류 탄소발자국
                    route_logistics_carbon = cam_var * pcam_var * emission_per_ton * estimated_logistics_volume
                    logistics_carbon += route_logistics_carbon
        
        return logistics_carbon
    
    def get_regional_analysis_from_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        최적화 결과로부터 지역별 분석 생성
        
        Args:
            results: 최적화 결과
            
        Returns:
            지역별 분석 결과
        """
        if not results or results.get('status') != 'optimal':
            return {'status': 'error', 'message': '최적화 결과가 없습니다.'}
        
        variables = results.get('variables', {})
        
        # 최적 위치 추출
        optimal_locations = self._extract_optimal_locations(variables)
        
        # 지역별 성능 분석
        regional_performance = self._analyze_regional_performance(variables, optimal_locations)
        
        # 물류 분석
        logistics_analysis = self._analyze_logistics_impact(optimal_locations)
        
        # 탄소발자국 분석
        carbon_analysis = self._analyze_carbon_footprint_by_region(variables, optimal_locations)
        
        # 비용 분석 (보조적)
        cost_analysis = self._analyze_regional_costs(optimal_locations)
        
        return {
            'optimal_locations': optimal_locations,
            'regional_performance': regional_performance,
            'logistics_analysis': logistics_analysis,
            'carbon_analysis': carbon_analysis,
            'cost_analysis': cost_analysis,
            'total_carbon_footprint': results.get('objective_value', 0.0),
            'status': 'success'
        }
    
    def _extract_optimal_locations(self, variables: Dict[str, float]) -> Dict[str, str]:
        """변수 값으로부터 최적 위치 추출"""
        optimal_locations = {}
        
        # CAM 위치
        for region in self.target_regions:
            cam_var_name = f'cam_location_{region.replace(" ", "_")}'
            if cam_var_name in variables and variables[cam_var_name] > 0.5:
                optimal_locations['CAM'] = region
                break
        
        # pCAM 위치
        for region in self.target_regions:
            pcam_var_name = f'pcam_location_{region.replace(" ", "_")}'
            if pcam_var_name in variables and variables[pcam_var_name] > 0.5:
                optimal_locations['pCAM'] = region
                break
        
        # 기본값 설정
        if 'CAM' not in optimal_locations:
            optimal_locations['CAM'] = '한국'
        if 'pCAM' not in optimal_locations:
            optimal_locations['pCAM'] = '한국'
        
        return optimal_locations
    
    def _analyze_regional_performance(self, variables: Dict[str, float], locations: Dict[str, str]) -> Dict[str, Any]:
        """지역별 성능 분석"""
        performance = {}
        
        for facility, region in locations.items():
            emission_factor = self.regional_emission_factors.get(region, 0.5)
            cost_info = self.regional_costs.get(region, {})
            
            performance[f'{facility}_{region}'] = {
                'emission_factor': emission_factor,
                'labor_cost': cost_info.get('labor_cost', 20.0),
                'energy_cost': cost_info.get('energy_cost', 0.15),
                'land_cost': cost_info.get('land_cost', 300.0),
                'logistics_bonus': cost_info.get('logistics_hub_bonus', 1.0),
                'competitiveness_score': self._calculate_regional_competitiveness(region)
            }
        
        return performance
    
    def _calculate_regional_competitiveness(self, region: str) -> float:
        """지역별 경쟁력 점수 계산"""
        emission_factor = self.regional_emission_factors.get(region, 0.5)
        cost_info = self.regional_costs.get(region, {})
        
        # 낮은 배출계수는 좋음 (가중치: 40%)
        emission_score = max(0, (0.6 - emission_factor) / 0.3 * 40)
        
        # 낮은 비용은 좋음 (가중치: 30%)
        total_cost = cost_info.get('labor_cost', 20) + cost_info.get('energy_cost', 0.15) * 100
        cost_score = max(0, (50 - total_cost) / 30 * 30)
        
        # 물류 허브 보너스 (가중치: 30%)
        logistics_score = cost_info.get('logistics_hub_bonus', 1.0) * 30
        
        total_score = emission_score + cost_score + logistics_score
        return min(100, max(0, total_score))
    
    def _analyze_logistics_impact(self, locations: Dict[str, str]) -> Dict[str, Any]:
        """물류 영향 분석"""
        cam_region = locations.get('CAM', '한국')
        pcam_region = locations.get('pCAM', '한국')
        
        logistics_info = self.logistics_matrix.get(cam_region, {}).get(pcam_region, {})
        
        return {
            'transport_distance': logistics_info.get('distance', 0),
            'transport_cost_per_ton': logistics_info.get('cost_per_ton', 0),
            'transport_emission_per_ton': logistics_info.get('emission_per_ton', 0),
            'transit_time': logistics_info.get('transit_time', 0),
            'logistics_efficiency': self._calculate_logistics_efficiency(cam_region, pcam_region)
        }
    
    def _calculate_logistics_efficiency(self, cam_region: str, pcam_region: str) -> float:
        """물류 효율성 계산"""
        distance = self.logistics_matrix.get(cam_region, {}).get(pcam_region, {}).get('distance', 0)
        
        # 거리가 짧을수록 효율적 (100점 만점)
        if distance == 0:
            return 100.0
        elif distance < 1000:
            return 90.0
        elif distance < 3000:
            return 70.0
        elif distance < 6000:
            return 50.0
        else:
            return 30.0
    
    def _analyze_carbon_footprint_by_region(self, variables: Dict[str, float], locations: Dict[str, str]) -> Dict[str, Any]:
        """지역별 탄소발자국 분석"""
        carbon_breakdown = {
            'baseline_carbon': self._calculate_baseline_carbon(),
            'regional_electricity': 0.0,
            'logistics_carbon': 0.0,
            'total_reduction': 0.0
        }
        
        # 지역별 전력 탄소발자국
        for facility, region in locations.items():
            emission_factor = self.regional_emission_factors.get(region, 0.5)
            
            # 예상 전력 사용량 기반 계산
            estimated_electricity = 1000.0  # kWh
            regional_electricity_carbon = estimated_electricity * emission_factor
            carbon_breakdown['regional_electricity'] += regional_electricity_carbon
        
        # 물류 탄소발자국
        cam_region = locations.get('CAM', '한국')
        pcam_region = locations.get('pCAM', '한국')
        logistics_info = self.logistics_matrix.get(cam_region, {}).get(pcam_region, {})
        
        estimated_logistics_volume = 10.0  # 톤
        logistics_carbon = logistics_info.get('emission_per_ton', 0) * estimated_logistics_volume
        carbon_breakdown['logistics_carbon'] = logistics_carbon
        
        # RE 적용에 따른 감축량
        total_reduction = 0.0
        for tier_num in [1, 2]:
            re_var_name = f'tier{tier_num}_re_application_rate'
            if re_var_name in variables:
                re_rate = variables[re_var_name]
                tier_reduction = re_rate * (5.0 if tier_num == 1 else 8.0)  # 예상 감축량
                total_reduction += tier_reduction
        
        carbon_breakdown['total_reduction'] = total_reduction
        
        return carbon_breakdown
    
    def _analyze_regional_costs(self, locations: Dict[str, str]) -> Dict[str, Any]:
        """지역별 비용 분석 (보조적 정보)"""
        total_cost_analysis = {
            'production_costs': {},
            'logistics_costs': {},
            'total_estimated_cost': 0.0
        }
        
        # 생산 비용
        for facility, region in locations.items():
            cost_info = self.regional_costs.get(region, {})
            
            estimated_production_cost = (
                cost_info.get('labor_cost', 20.0) * 100 +  # 100시간 가정
                cost_info.get('energy_cost', 0.15) * 1000 + # 1000kWh 가정  
                cost_info.get('land_cost', 300.0) * 0.1     # 10% 할당 가정
            )
            
            total_cost_analysis['production_costs'][f'{facility}_{region}'] = estimated_production_cost
            total_cost_analysis['total_estimated_cost'] += estimated_production_cost
        
        # 물류 비용
        cam_region = locations.get('CAM', '한국')
        pcam_region = locations.get('pCAM', '한국')
        logistics_info = self.logistics_matrix.get(cam_region, {}).get(pcam_region, {})
        
        estimated_logistics_volume = 10.0  # 톤
        logistics_cost = logistics_info.get('cost_per_ton', 0) * estimated_logistics_volume
        
        total_cost_analysis['logistics_costs']['cam_to_pcam'] = logistics_cost
        total_cost_analysis['total_estimated_cost'] += logistics_cost
        
        return total_cost_analysis
    
    def generate_regional_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """지역 선택 권고사항 생성"""
        analysis = self.get_regional_analysis_from_results(results)
        
        if analysis.get('status') != 'success':
            return analysis
        
        optimal_locations = analysis['optimal_locations']
        regional_performance = analysis['regional_performance']
        
        recommendations = {
            'primary_recommendation': {
                'cam_region': optimal_locations.get('CAM'),
                'pcam_region': optimal_locations.get('pCAM'),
                'rationale': f"최적 조합: 탄소발자국 최소화를 통해 CAM을 {optimal_locations.get('CAM')}에서, pCAM을 {optimal_locations.get('pCAM')}에서 생산"
            },
            'performance_summary': {},
            'alternative_scenarios': []
        }
        
        # 성능 요약
        for key, perf in regional_performance.items():
            recommendations['performance_summary'][key] = {
                'competitiveness_score': perf['competitiveness_score'],
                'emission_factor': perf['emission_factor'],
                'estimated_cost_index': perf['labor_cost'] + perf['energy_cost'] * 100
            }
        
        # 대안 시나리오 생성
        for region in self.target_regions:
            if region != optimal_locations.get('CAM'):
                carbon_diff = self._estimate_carbon_difference(optimal_locations.get('CAM'), region, 'CAM')
                
                recommendations['alternative_scenarios'].append({
                    'type': 'CAM_alternative',
                    'region': region,
                    'carbon_impact': f"{carbon_diff:+.1f}%",
                    'description': f"CAM을 {region}에서 생산 시 탄소발자국 변화"
                })
        
        return recommendations
    
    def _estimate_carbon_difference(self, current_region: str, alternative_region: str, facility_type: str) -> float:
        """지역 변경 시 탄소발자국 차이 추정"""
        current_factor = self.regional_emission_factors.get(current_region, 0.5)
        alternative_factor = self.regional_emission_factors.get(alternative_region, 0.5)
        
        # 단순 비례 계산 (실제로는 더 복잡한 계산 필요)
        factor_diff = (alternative_factor - current_factor) / current_factor
        
        # 해당 시설이 전체 탄소발자국에 미치는 영향 (추정)
        facility_weight = 0.4 if facility_type == 'CAM' else 0.3
        
        return factor_diff * facility_weight * 100
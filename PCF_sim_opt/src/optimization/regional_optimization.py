"""
지역별 최적화 시나리오 구현 모듈
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd

from .scenario_base import OptimizationScenario
from .input import OptimizationInput

# 시뮬레이션 정렬 목적함수 임포트 (선택적)
try:
    from .simulation_aligned_regional_objective import SimulationAlignedRegionalObjective
    SIMULATION_ALIGNED_REGIONAL_AVAILABLE = True
except ImportError:
    SIMULATION_ALIGNED_REGIONAL_AVAILABLE = False
    print("Warning: SimulationAlignedRegionalObjective not available. Using simplified regional calculation.")


class RegionalOptimization(OptimizationScenario):
    """
    지역별 최적화 시나리오
    
    이 클래스는 생산 위치를 최적화하여 전력 배출계수와 물류 비용을 
    고려한 최적의 지역 구성을 찾습니다.
    시뮬레이션 데이터가 제공되면 rule_based.py의 실제 지역별 계산 로직을 사용합니다.
    """
    
    def __init__(self, 
                opt_input: OptimizationInput = None,
                config_path: Optional[str] = None,
                target_regions: Optional[List[str]] = None):
        """
        Args:
            opt_input: 최적화 입력 객체 (시뮬레이션 데이터 포함 가능)
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
            target_regions: 고려할 지역 목록 (None이면 모든 가능한 지역 사용)
        """
        self.target_regions = target_regions or ["한국", "중국", "일본", "미국", "독일"]
        
        # OptimizationInput 객체 설정
        if opt_input is not None:
            self.opt_input = opt_input
        
        super().__init__(
            config_path=config_path,
            name="regional_optimization",
            description="생산 위치와 물류를 고려한 지역별 최적화 시나리오 (시뮬레이션 정렬 지원)"
        )
        
        # 시뮬레이션 정렬 목적함수 초기화
        self.simulation_aligned_regional_objective = None
        if SIMULATION_ALIGNED_REGIONAL_AVAILABLE and self._is_simulation_aligned():
            self._initialize_simulation_aligned_objective()
    
    def _configure_scenario(self) -> None:
        """지역별 최적화 시나리오 설정 적용"""
        # 기본 설정에 시나리오가 있는지 확인하고 적용
        available_scenarios = self.opt_input.get_available_scenarios()
        
        if 'regional_optimization' in available_scenarios:
            self.opt_input.apply_scenario('regional_optimization')
        else:
            # 시나리오가 없으면 수동으로 설정
            # 지역별 최적화를 위한 기본 설정 (탄소발자국 최소화 기반)
            custom_config = {
                'objective': 'minimize_carbon',  # 탄소발자국 최소화
                'constraints': {
                    'max_cost': 70000.0,  # 비용 제한
                    'max_transport_distance': 5000.0  # 최대 물류 거리 (km)
                },
                'decision_vars': {
                    'cathode': {
                        'type': 'B'  # 선형 문제로 시작
                    },
                    # 지역 변수는 추가 설정 필요
                    'regions': {
                        'cam_location': self.target_regions,  # CAM 생산 위치 옵션
                        'pcam_location': self.target_regions,  # pCAM 생산 위치 옵션
                        'use_region_binary': True,  # 지역 선택을 위한 이진 변수
                    # 시뮬레이션 정렬 지원
                    'simulation_aligned': self._is_simulation_aligned()
                    }
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
            self.simulation_aligned_regional_objective = SimulationAlignedRegionalObjective(
                opt_input=self.opt_input,
                material_matching_info=material_matching_info,
                target_regions=self.target_regions
            )
            
            print("✅ 시뮬레이션 정렬 지역 목적함수 초기화 완료")
            
        except Exception as e:
            print(f"⚠️ 시뮬레이션 정렬 목적함수 초기화 실패: {e}")
            self.simulation_aligned_regional_objective = None
    
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
                    'regional_dependency': self._calculate_regional_dependency(row),
                    'index': idx
                }
                
                material_matching_info[material_key] = material_info
        
        return material_matching_info
    
    def _calculate_regional_dependency(self, row: pd.Series) -> float:
        """자재의 지역 의존도 계산"""
        material_category = row.get('자재품목', '')
        
        # 자재 유형별 지역 의존도 (전력 사용량 기반)
        dependency_map = {
            '양극재': 0.8,  # 높은 전력 사용
            '음극재': 0.4,  # 중간 전력 사용
            '분리막': 0.6,  # 중간-높음 전력 사용
            '전해액': 0.3,  # 낮은 전력 사용
            '동박': 0.7,    # 높은 전력 사용
            '알박': 0.7     # 높은 전력 사용
        }
        
        return dependency_map.get(material_category, 0.5)  # 기본값 0.5
    
    def _configure_model(self, model) -> None:
        """
        지역별 최적화 시나리오를 위한 모델 추가 설정
        
        Args:
            model: Pyomo 모델
        """
        from pyomo.environ import Var, Binary, Constraint, Set
        
        # 1. 지역 집합 정의
        model.REGIONS = Set(initialize=self.target_regions)
        
        # 2. 지역 선택 이진 변수 정의
        model.cam_location = Var(model.REGIONS, within=Binary, initialize=0)
        model.pcam_location = Var(model.REGIONS, within=Binary, initialize=0)
        
        # 3. 지역 제약조건: 각 시설은 정확히 한 지역에만 위치
        @model.Constraint()
        def cam_one_location_constraint(m):
            return sum(m.cam_location[r] for r in m.REGIONS) == 1
            
        @model.Constraint()
        def pcam_one_location_constraint(m):
            return sum(m.pcam_location[r] for r in m.REGIONS) == 1
            
        # 4. 전력 배출계수를 지역 선택에 연동
        # (이 부분은 실제 구현에서는 더 복잡할 수 있음)
        # 현재 모델에서는 단순화를 위해 생략
        
        # 5. 물류 비용/거리 제약
        # (이 부분은 실제 구현에서는 더 복잡할 수 있음)
        # 현재 모델에서는 단순화를 위해 생략
    
    def select_solver(self) -> str:
        """
        지역별 최적화에 적합한 솔버 선택
        
        Returns:
            str: 선택된 솔버 이름
        """
        # 지역 선택을 위한 이진 변수가 있으므로 MIP 솔버 사용
        # 시뮬레이션 정렬 모드에서는 더 정교한 솔버 선택
        if self._is_simulation_aligned():
            return 'glpk'  # 시뮬레이션 정렬에서는 GLPK 사용
        
        return 'cbc'  # MILP 문제에 적합
    
    def get_optimal_locations(self) -> Dict[str, str]:
        """
        최적 생산 위치 반환
        
        Returns:
            Dict[str, str]: 시설별 최적 위치
        """
        if not self.results or self.results.get('status') != 'optimal':
            return {}
            
        variables = self.results.get('variables', {})
        
        # 최적 위치 찾기
        cam_location = None
        pcam_location = None
        
        for var_name, var_value in variables.items():
            # CAM 위치 변수 검색
            if var_name.startswith('cam_location[') and var_value > 0.5:
                region = var_name[13:-1]  # 'cam_location['와 ']' 제거
                cam_location = region
                
            # pCAM 위치 변수 검색
            if var_name.startswith('pcam_location[') and var_value > 0.5:
                region = var_name[14:-1]  # 'pcam_location['와 ']' 제거
                pcam_location = region
        
        return {
            'CAM': cam_location,
            'pCAM': pcam_location
        }
    
    def get_regional_analysis(self) -> Dict[str, Any]:
        """
        지역별 분석 결과 반환
        
        Returns:
            Dict: 지역별 분석 결과
        """
        if not self.results or self.results.get('status') != 'optimal':
            return {'status': 'error', 'message': '최적화 결과가 없습니다.'}
        
        # 시뮬레이션 정렬 모드인 경우 정교한 지역 분석 사용
        if self.simulation_aligned_regional_objective:
            try:
                return self.simulation_aligned_regional_objective.get_regional_analysis_from_results(self.results)
            except Exception as e:
                print(f"⚠️ 시뮬레이션 정렬 지역 분석 실패, 기본 분석 사용: {e}")
                # 기본 분석으로 폴백
                pass
            
        # 최적 위치
        optimal_locations = self.get_optimal_locations()
        
        # 지역별 전력 배출계수 가져오기
        electricity_factors = {}
        
        for region in self.target_regions:
            factor = self.opt_input.get_constants().get_location_factor(region)
            electricity_factors[region] = factor
        
        # 최적 위치의 전력 배출계수
        cam_factor = electricity_factors.get(optimal_locations.get('CAM', ''), 0.0)
        pcam_factor = electricity_factors.get(optimal_locations.get('pCAM', ''), 0.0)
        
        # 전체 탄소발자국 계산
        carbon_footprint = 0.0
        if 'carbon_footprint' in self.results:
            carbon_footprint = self.results['carbon_footprint']
        elif hasattr(self.results_processor, 'formatted_results') and self.results_processor.formatted_results:
            carbon_str = self.results_processor.formatted_results.get('carbon_footprint', '0.0')
            try:
                carbon_footprint = float(carbon_str.split()[0])
            except:
                pass
        
        # 물류 거리 및 비용 분석 (실제 구현에서는 더 정교한 계산 필요)
        # 여기서는 가상의 값으로 대체
        logistics_data = self._calculate_logistics_data(optimal_locations)
        
        # 시뮬레이션 정렬 추가 정보
        analysis_result = {
            'optimal_locations': optimal_locations,
            'electricity_factors': {
                'CAM': cam_factor,
                'pCAM': pcam_factor
            },
            'carbon_footprint': carbon_footprint,
            'logistics': logistics_data,
            'regional_comparison': self._generate_regional_comparison(),
            'status': 'success'
        }
        
        if self._is_simulation_aligned():
            analysis_result['simulation_aligned'] = True
            analysis_result['simulation_data_size'] = len(self.opt_input.scenario_df) if hasattr(self.opt_input, 'scenario_df') else 0
        else:
            analysis_result['simulation_aligned'] = False
        
        return analysis_result
    
    def _calculate_logistics_data(self, locations: Dict[str, str]) -> Dict[str, Any]:
        """
        물류 데이터 계산 (예시)
        
        Args:
            locations: 시설별 위치
            
        Returns:
            Dict: 물류 관련 데이터
        """
        # 예시 데이터 - 실제 구현에서는 더 정교한 계산이 필요
        distance_matrix = {
            '한국': {'한국': 0, '중국': 600, '일본': 800, '미국': 9000, '독일': 8000},
            '중국': {'한국': 600, '중국': 0, '일본': 1200, '미국': 10000, '독일': 7500},
            '일본': {'한국': 800, '일본': 0, '중국': 1200, '미국': 8500, '독일': 9000},
            '미국': {'한국': 9000, '중국': 10000, '일본': 8500, '미국': 0, '독일': 6000},
            '독일': {'한국': 8000, '중국': 7500, '일본': 9000, '미국': 6000, '독일': 0}
        }
        
        transport_cost_per_km = 0.5  # 예시 비용 (km당)
        
        cam_location = locations.get('CAM', '한국')  # 기본값 '한국'
        pcam_location = locations.get('pCAM', '한국')  # 기본값 '한국'
        
        # 물류 거리 및 비용
        transport_distance = distance_matrix.get(cam_location, {}).get(pcam_location, 0)
        transport_cost = transport_distance * transport_cost_per_km
        
        return {
            'transport_distance': transport_distance,
            'transport_cost': transport_cost,
            'carbon_emissions_from_transport': transport_distance * 0.1  # 예시 값 (km당 0.1 kg CO2)
        }
    
    def _generate_regional_comparison(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        지역별 성능 비교표 생성
        
        Returns:
            Dict: 지역별 비교 데이터
        """
        # 각 지역 조합의 성능 추정
        combinations = []
        
        for cam_region in self.target_regions:
            for pcam_region in self.target_regions:
                # 전력 배출계수
                cam_factor = self.opt_input.get_constants().get_location_factor(cam_region)
                pcam_factor = self.opt_input.get_constants().get_location_factor(pcam_region)
                
                # 물류 데이터
                logistics = self._calculate_logistics_data({
                    'CAM': cam_region,
                    'pCAM': pcam_region
                })
                
                # 탄소발자국 대략적 추정 (예시 - 실제 구현에서는 더 정교한 계산 필요)
                base_emission = 80.0  # 기본값
                if hasattr(self, 'constants'):
                    base_emission = self.constants.get('base_emission', 80.0)
                
                # 전력 배출계수의 기준 값 (예: 한국)
                base_factor = self.opt_input.get_constants().get_location_factor('한국')
                
                # 대략적인 탄소발자국 추정
                carbon_footprint = base_emission * (
                    0.6 * (cam_factor / base_factor) +  # CAM 영향
                    0.4 * (pcam_factor / base_factor)   # pCAM 영향
                ) + logistics['carbon_emissions_from_transport']
                
                combinations.append({
                    'cam_region': cam_region,
                    'pcam_region': pcam_region,
                    'carbon_footprint': carbon_footprint,
                    'transport_distance': logistics['transport_distance'],
                    'transport_cost': logistics['transport_cost']
                })
        
        # 결과 정렬 (탄소발자국 기준)
        combinations.sort(key=lambda x: x['carbon_footprint'])
        
        # 최적 조합과 기타 조합 분리
        optimal = None
        if self.results and self.results.get('status') == 'optimal':
            optimal_locations = self.get_optimal_locations()
            
            for combo in combinations:
                if (combo['cam_region'] == optimal_locations.get('CAM') and 
                    combo['pcam_region'] == optimal_locations.get('pCAM')):
                    optimal = combo
                    break
        
        return {
            'all_combinations': combinations,
            'optimal': optimal,
            'top_5': combinations[:5]  # 상위 5개 조합
        }
    
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
        
        # 지역 분석 결과 추가
        if results.get('status') == 'optimal':
            regional_analysis = self.get_regional_analysis()
            if regional_analysis.get('status') == 'success':
                results['regional_analysis'] = regional_analysis
        
        # 시뮬레이션 정렬 정보 추가
        results['simulation_aligned'] = self._is_simulation_aligned()
        if self._is_simulation_aligned():
            results['simulation_info'] = {
                'data_source': 'rule_based_simulation',
                'material_count': len(self.opt_input.scenario_df) if hasattr(self.opt_input, 'scenario_df') else 0,
                'regional_calculation': 'simulation_aligned',
                'target_regions': self.target_regions
            }
        
        return results
    
    def run_location_sensitivity(self) -> Dict[str, Any]:
        """
        생산 위치에 대한 민감도 분석 수행
        
        Returns:
            Dict: 민감도 분석 결과
        """
        if not self.opt_input:
            return {'status': 'error', 'message': '최적화 입력이 설정되지 않았습니다.'}
            
        sensitivity_results = []
        
        # 각 지역 쌍에 대해 최적화 수행
        for cam_region in self.target_regions:
            for pcam_region in self.target_regions:
                # 원래 설정 백업
                original_config = self.opt_input.get_config()
                
                # 위치 고정 설정
                if 'decision_vars' not in self.opt_input.config:
                    self.opt_input.config['decision_vars'] = {}
                
                if 'regions' not in self.opt_input.config['decision_vars']:
                    self.opt_input.config['decision_vars']['regions'] = {}
                    
                self.opt_input.config['decision_vars']['regions']['cam_location_fixed'] = cam_region
                self.opt_input.config['decision_vars']['regions']['pcam_location_fixed'] = pcam_region
                
                # 새 모델로 최적화
                self.model = None  # 모델 재설정
                results = self.solve()
                
                if results.get('status') == 'optimal':
                    # 탄소발자국
                    carbon_footprint = 0.0
                    if 'carbon_footprint' in results:
                        carbon_footprint = results['carbon_footprint']
                    
                    # 물류 데이터
                    logistics = self._calculate_logistics_data({
                        'CAM': cam_region,
                        'pCAM': pcam_region
                    })
                    
                    sensitivity_results.append({
                        'cam_region': cam_region,
                        'pcam_region': pcam_region,
                        'carbon_footprint': carbon_footprint,
                        'transport_distance': logistics['transport_distance'],
                        'transport_cost': logistics['transport_cost'],
                        'objective_value': results.get('objective_value', 0.0)
                    })
                
                # 원래 설정 복원
                self.opt_input.config = original_config
        
        # 결과 정렬 (탄소발자국 기준)
        sensitivity_results.sort(key=lambda x: x['carbon_footprint'])
        
        return {
            'results': sensitivity_results,
            'status': 'success' if sensitivity_results else 'failed'
        }
    
    def generate_location_recommendations(self) -> Dict[str, Any]:
        """
        최적 생산 위치 조합 추천
        
        Returns:
            Dict: 위치 추천 결과
        """
        if not self.results or self.results.get('status') != 'optimal':
            return {'status': 'error', 'message': '최적화 결과가 없습니다.'}
        
        # 시뮬레이션 정렬 모드인 경우 정교한 추천 사용
        if self.simulation_aligned_regional_objective:
            try:
                return self.simulation_aligned_regional_objective.generate_regional_recommendations(self.results)
            except Exception as e:
                print(f"⚠️ 시뮬레이션 정렬 추천 실패, 기본 추천 사용: {e}")
                # 기본 추천으로 폴백
                pass
            
        # 지역 분석 결과 가져오기
        regional_analysis = self.get_regional_analysis()
        if regional_analysis.get('status') != 'success':
            return {'status': 'error', 'message': '지역 분석에 실패했습니다.'}
            
        # 최적 위치
        optimal_locations = regional_analysis.get('optimal_locations', {})
        
        # 지역별 비교 데이터
        comparisons = regional_analysis.get('regional_comparison', {})
        top_combinations = comparisons.get('top_5', [])
        
        # 추천 생성
        recommendations = {
            'primary': {
                'cam_region': optimal_locations.get('CAM'),
                'pcam_region': optimal_locations.get('pCAM'),
                'description': f"최적 조합: CAM을 {optimal_locations.get('CAM')}에서, pCAM을 {optimal_locations.get('pCAM')}에서 생산"
            },
            'alternatives': []
        }
        
        # 대안 추천 (최적 조합과 다른 상위 조합)
        for combo in top_combinations:
            if (combo['cam_region'] != optimal_locations.get('CAM') or 
                combo['pcam_region'] != optimal_locations.get('pCAM')):
                
                # 최적해와의 차이 계산
                carbon_diff = 100 * (combo['carbon_footprint'] - regional_analysis['carbon_footprint']) / regional_analysis['carbon_footprint']
                
                recommendations['alternatives'].append({
                    'cam_region': combo['cam_region'],
                    'pcam_region': combo['pcam_region'],
                    'carbon_footprint': combo['carbon_footprint'],
                    'carbon_difference': f"{carbon_diff:.2f}%",
                    'transport_distance': combo['transport_distance'],
                    'description': f"대안 조합: CAM을 {combo['cam_region']}에서, pCAM을 {combo['pcam_region']}에서 생산 (탄소발자국 {carbon_diff:.1f}% 증가)"
                })
                
                # 최대 3개 대안만 포함
                if len(recommendations['alternatives']) >= 3:
                    break
        
        # 시뮬레이션 정렬 추가 정보
        if self._is_simulation_aligned():
            recommendations['simulation_aligned'] = True
            recommendations['calculation_method'] = 'rule_based_simulation'
        else:
            recommendations['simulation_aligned'] = False
            recommendations['calculation_method'] = 'simplified_estimation'
        
        return recommendations
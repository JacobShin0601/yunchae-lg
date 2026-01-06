"""
구현 용이성 최적화 시나리오 구현 모듈
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from .scenario_base import OptimizationScenario
from .input import OptimizationInput


class ImplementationEase(OptimizationScenario):
    """
    구현 용이성 최적화 시나리오
    
    이 클래스는 탄소 감축에 필요한 활동 수를 최소화하면서
    탄소발자국 목표를 달성하는 가장 용이한 구현 경로를 찾습니다.
    시뮬레이션 데이터가 제공되면 rule_based.py의 실제 구현 복잡도를 반영합니다.
    """
    
    def __init__(self, 
                opt_input: OptimizationInput = None,
                config_path: Optional[str] = None,
                carbon_target: float = 45.0):
        """
        Args:
            opt_input: 최적화 입력 객체 (시뮬레이션 데이터 포함 가능)
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
            carbon_target: 달성해야 할 탄소발자국 목표 (kg CO2eq/kWh)
        """
        self.carbon_target = carbon_target
        
        # OptimizationInput 객체 설정
        if opt_input is not None:
            self.opt_input = opt_input
        
        super().__init__(
            config_path=config_path,
            name="implementation_ease",
            description="구현 용이성 최적화 시나리오 (활동 수 최소화, 시뮬레이션 지원)"
        )
    
    def _configure_scenario(self) -> None:
        """구현 용이성 최적화 시나리오 설정 적용"""
        # 기본 설정에 시나리오가 있는지 확인하고 적용
        available_scenarios = self.opt_input.get_available_scenarios()
        
        if 'implementation_ease' in available_scenarios:
            self.opt_input.apply_scenario('implementation_ease')
        else:
            # 시나리오가 없으면 수동으로 설정
            custom_config = {
                'objective': 'maximize_ease',  # 구현 용이성 최대화 (활동 수 최소화)
                'constraints': {
                    'target_carbon': self.carbon_target,  # 탄소발자국 목표
                    'max_cost': 80000.0   # 비용 제약 (높게 설정)
                },
                'decision_vars': {
                    'cathode': {
                        'type': 'B'  # 선형 문제로 시작
                    },
                    'use_binary_variables': True  # 이진 변수 필수 (활동 수 세기 위해)
                }
            }
            self.opt_input.create_custom_config(**custom_config)
    
    def _is_simulation_aligned(self) -> bool:
        """시뮬레이션 정렬 모드인지 확인"""
        return (hasattr(self.opt_input, 'scenario_df') and 
                self.opt_input.scenario_df is not None and
                len(self.opt_input.scenario_df) > 0)
    
    def _configure_model(self, model) -> None:
        """
        구현 용이성 최적화 시나리오를 위한 모델 추가 설정
        
        Args:
            model: Pyomo 모델
        """
        # 필요한 추가 제약조건 설정
        # 예: 특정 활동 간의 의존성 제약, 최소한의 탄소 감축 보장 등
        pass
    
    def select_solver(self) -> str:
        """
        구현 용이성 최적화에 적합한 솔버 선택
        
        Returns:
            str: 선택된 솔버 이름
        """
        # 이진 변수가 필수이므로 MIP 솔버 사용
        return 'cbc'  # MILP 문제에 적합
    
    def set_carbon_target(self, target: float) -> None:
        """
        탄소발자국 목표 설정
        
        Args:
            target: 탄소발자국 목표 (kg CO2eq/kWh)
        """
        self.carbon_target = target
        
        # 설정에 반영
        if 'constraints' not in self.opt_input.config:
            self.opt_input.config['constraints'] = {}
            
        self.opt_input.config['constraints']['target_carbon'] = target
    
    def get_active_activities(self) -> List[str]:
        """
        최적해에서 활성화된 활동 목록 추출
        
        Returns:
            List[str]: 활성화된 활동 이름 목록
        """
        if not self.results or self.results.get('status') != 'optimal':
            return []
        
        active_activities = []
        variables = self.results.get('variables', {})
        
        for var_name, var_value in variables.items():
            # 이진 활성화 변수 검색 (_active 접미사)
            if var_name.endswith('_active') and var_value > 0.5:
                # 접미사 제거하여 원래 활동명 추출
                activity_name = var_name.replace('_active', '')
                active_activities.append(activity_name)
        
        return active_activities
    
    def get_implementation_complexity(self) -> Dict[str, Any]:
        """
        구현 복잡도 분석 결과 반환
        
        Returns:
            Dict: 구현 복잡도 분석 결과
        """
        if not self.results or self.results.get('status') != 'optimal':
            return {'status': 'error', 'message': '최적화 결과가 없습니다.'}
        
        # 시뮬레이션 정렬 모드인 경우 실제 매칭 복잡도 반영
        if self._is_simulation_aligned():
            return self._get_simulation_aligned_complexity()
        else:
            return self._get_basic_complexity()
    
    def _get_simulation_aligned_complexity(self) -> Dict[str, Any]:
        """시뮬레이션 정렬 복잡도 분석"""
        variables = self.results.get('variables', {})
        
        # 자재별 복잡도 분석
        material_complexity = self._analyze_material_complexity()
        
        # 매칭 복잡도 분석
        matching_complexity = self._analyze_matching_complexity()
        
        # 활동별 복잡도 분석
        activity_complexity = self._analyze_activity_complexity(variables)
        
        # 시뮬레이션 기반 전체 복잡도 계산
        total_complexity_score = self._calculate_simulation_complexity_score(
            material_complexity, matching_complexity, activity_complexity
        )
        
        return {
            'num_activities': len(self.get_active_activities()),
            'active_activities': self.get_active_activities(),
            'material_complexity': material_complexity,
            'matching_complexity': matching_complexity,
            'activity_complexity': activity_complexity,
            'carbon_footprint': self.results.get('objective_value', 0.0),
            'carbon_target': self.carbon_target,
            'total_cost': self._estimate_total_cost(variables),
            'complexity_score': total_complexity_score,
            'implementation_ease': 100 - total_complexity_score,
            'simulation_aligned': True,
            'status': 'success'
        }
    
    def _get_basic_complexity(self) -> Dict[str, Any]:
        """기본 복잡도 분석 (기존 로직)"""
        
        # 활성화된 활동 목록
        active_activities = self.get_active_activities()
        num_activities = len(active_activities)
        
        # 각 활동의 감축 기여도
        variables = self.results.get('variables', {})
        activity_contributions = {}
        
        for activity in active_activities:
            # 해당 활동의 감축률
            reduction_rate = variables.get(activity, 0.0)
            activity_contributions[activity] = reduction_rate
        
        # 각 활동의 비용 기여도
        activity_costs = {}
        
        for activity in active_activities:
            # 고정 비용
            fixed_cost = self.opt_input.get_constants().get_activity_cost(activity)
            
            # 가변 비용
            reduction_rate = variables.get(activity, 0.0)
            variable_cost_per_percent = self.opt_input.get_constants().get('variable_cost_per_percent', 50)
            variable_cost = reduction_rate * variable_cost_per_percent
            
            # 총 비용
            total_cost = fixed_cost + variable_cost
            activity_costs[activity] = total_cost
        
        # 탄소발자국 값
        carbon_footprint = 0.0
        if 'carbon_footprint' in self.results:
            carbon_footprint = self.results['carbon_footprint']
        elif hasattr(self.results_processor, 'formatted_results') and self.results_processor.formatted_results:
            carbon_str = self.results_processor.formatted_results.get('carbon_footprint', '0.0')
            try:
                carbon_footprint = float(carbon_str.split()[0])
            except:
                pass
        
        # 구현 복잡도 점수 (활동 수 기반, 낮을수록 간단)
        # 가중치: 활동 수(70%) + 총비용(30%)
        max_activities = 10  # 예상 최대 활동 수
        max_cost = 100000.0  # 예상 최대 비용
        
        # 총 비용
        total_cost = sum(activity_costs.values())
        
        # 복잡도 점수 (0-100, 낮을수록 간단)
        complexity_score = (
            0.7 * (num_activities / max_activities) * 100 +
            0.3 * (total_cost / max_cost) * 100
        )
        
        return {
            'num_activities': num_activities,
            'active_activities': active_activities,
            'activity_contributions': activity_contributions,
            'activity_costs': activity_costs,
            'carbon_footprint': carbon_footprint,
            'carbon_target': self.carbon_target,
            'total_cost': total_cost,
            'complexity_score': complexity_score,
            'implementation_ease': 100 - complexity_score,  # 용이성 점수 (100-복잡도)
            'simulation_aligned': False,
            'status': 'success'
        }
    
    def _analyze_material_complexity(self) -> Dict[str, Any]:
        """자재별 복잡도 분석"""
        if not hasattr(self.opt_input, 'scenario_df'):
            return {'total_materials': 0, 'complexity_factors': {}}
        
        scenario_df = self.opt_input.scenario_df
        
        material_complexity = {
            'total_materials': len(scenario_df),
            'material_categories': scenario_df['자재품목'].nunique() if '자재품목' in scenario_df.columns else 0,
            'complexity_factors': {}
        }
        
        # 자재품목별 복잡도
        if '자재품목' in scenario_df.columns:
            category_counts = scenario_df['자재품목'].value_counts().to_dict()
            
            for category, count in category_counts.items():
                # 복잡도 요인: 자재 개수, 특수성
                complexity_factor = self._get_material_category_complexity(category, count)
                material_complexity['complexity_factors'][category] = complexity_factor
        
        return material_complexity
    
    def _get_material_category_complexity(self, category: str, count: int) -> Dict[str, float]:
        """자재 카테고리별 복잡도 계산"""
        # 자재별 기본 복잡도 (높을수록 복잡)
        base_complexity = {
            '양극재': 8.0,    # 매우 복잡
            '음극재': 4.0,    # 중간 복잡도
            '분리막': 6.0,    # 높은 복잡도
            '전해액': 3.0,    # 낮은 복잡도
            '동박': 5.0,      # 중간-높음 복잡도
            '알박': 5.0       # 중간-높음 복잡도
        }.get(category, 4.0)
        
        # 개수에 따른 가중치 (많을수록 복잡)
        count_multiplier = min(2.0, 1.0 + (count - 1) * 0.1)
        
        total_complexity = base_complexity * count_multiplier
        
        return {
            'base_complexity': base_complexity,
            'count': count,
            'count_multiplier': count_multiplier,
            'total_complexity': total_complexity
        }
    
    def _analyze_matching_complexity(self) -> Dict[str, Any]:
        """매칭 복잡도 분석"""
        if not hasattr(self.opt_input, 'scenario_df'):
            return {'matching_success_rate': 1.0, 'complexity_score': 0.0}
        
        scenario_df = self.opt_input.scenario_df
        
        # 저감활동_적용여부가 1인 행들에 대한 매칭 분석
        if '저감활동_적용여부' in scenario_df.columns:
            applicable_rows = scenario_df[scenario_df['저감활동_적용여부'] == 1.0]
        else:
            applicable_rows = scenario_df
        
        total_applicable = len(applicable_rows)
        
        if total_applicable == 0:
            return {'matching_success_rate': 1.0, 'complexity_score': 0.0}
        
        # 매칭 복잡도 요인 분석
        matching_complexity = {
            'total_applicable_materials': total_applicable,
            'nan_materials': 0,
            'complex_matching_required': 0,
            'formula_vs_proportions': {'formula_preferred': 0, 'proportions_preferred': 0},
            'matching_success_rate': 0.85,  # 추정 성공률
            'complexity_score': 0.0
        }
        
        # NaN 자재명 개수 (복잡도 증가)
        if '자재명' in applicable_rows.columns:
            nan_count = applicable_rows['자재명'].isna().sum()
            matching_complexity['nan_materials'] = nan_count
        
        # 음극재 관련 복잡 매칭 (예: artificial/natural)
        if '자재이름' in applicable_rows.columns and '자재품목' in applicable_rows.columns:
            cathode_materials = applicable_rows[applicable_rows['자재품목'] == '음극재']
            if len(cathode_materials) > 0:
                complex_matching = len(cathode_materials[cathode_materials['자재이름'].str.contains('artificial|natural', case=False, na=False)])
                matching_complexity['complex_matching_required'] = complex_matching
        
        # 양극재의 formula vs proportions 매칭 선호도
        if '자재품목' in applicable_rows.columns:
            cathode_count = len(applicable_rows[applicable_rows['자재품목'] == '양극재'])
            # 양극재는 시나리오에 따라 proportions 또는 formula 우선
            matching_complexity['formula_vs_proportions']['proportions_preferred'] = cathode_count
        
        # 전체 매칭 복잡도 점수 계산
        complexity_factors = [
            matching_complexity['nan_materials'] * 2.0,  # NaN 자재명
            matching_complexity['complex_matching_required'] * 3.0,  # 복잡 매칭
            total_applicable * 0.5  # 기본 매칭 대상 수
        ]
        
        total_complexity = sum(complexity_factors)
        normalized_complexity = min(100.0, total_complexity / max(1, total_applicable) * 10)
        
        matching_complexity['complexity_score'] = normalized_complexity
        
        return matching_complexity
    
    def _analyze_activity_complexity(self, variables: Dict[str, float]) -> Dict[str, Any]:
        """활동별 복잡도 분석"""
        active_activities = self.get_active_activities()
        
        activity_complexity = {
            'total_activities': len(active_activities),
            'tier_distribution': {'tier1': 0, 'tier2': 0, 'other': 0},
            'coordination_complexity': 0.0,
            'implementation_sequence_complexity': 0.0,
            'resource_conflict_potential': 0.0
        }
        
        # Tier별 분류
        for activity in active_activities:
            if 'tier1' in activity.lower():
                activity_complexity['tier_distribution']['tier1'] += 1
            elif 'tier2' in activity.lower():
                activity_complexity['tier_distribution']['tier2'] += 1
            else:
                activity_complexity['tier_distribution']['other'] += 1
        
        # 조정 복잡도 (다양한 tier가 많을수록 복잡)
        tier_diversity = len([v for v in activity_complexity['tier_distribution'].values() if v > 0])
        activity_complexity['coordination_complexity'] = tier_diversity * 10.0
        
        # 구현 순서 복잡도 (동시 구현 vs 단계적 구현)
        if len(active_activities) > 3:
            activity_complexity['implementation_sequence_complexity'] = 25.0  # 높음
        elif len(active_activities) > 1:
            activity_complexity['implementation_sequence_complexity'] = 15.0  # 중간
        else:
            activity_complexity['implementation_sequence_complexity'] = 5.0   # 낮음
        
        # 자원 충돌 가능성
        tier1_count = activity_complexity['tier_distribution']['tier1']
        tier2_count = activity_complexity['tier_distribution']['tier2']
        
        if tier1_count > 0 and tier2_count > 0:
            activity_complexity['resource_conflict_potential'] = 20.0  # 높음 (다양한 tier)
        elif tier1_count > 2 or tier2_count > 2:
            activity_complexity['resource_conflict_potential'] = 15.0  # 중간
        else:
            activity_complexity['resource_conflict_potential'] = 5.0   # 낮음
        
        return activity_complexity
    
    def _calculate_simulation_complexity_score(self, material_complexity: Dict, matching_complexity: Dict, activity_complexity: Dict) -> float:
        """시뮬레이션 기반 전체 복잡도 점수 계산"""
        # 가중치 설정
        material_weight = 0.3
        matching_weight = 0.4
        activity_weight = 0.3
        
        # 자재 복잡도 점수
        material_score = 0
        if material_complexity.get('complexity_factors'):
            total_material_complexity = sum(
                factor.get('total_complexity', 0) 
                for factor in material_complexity['complexity_factors'].values()
            )
            material_score = min(100, total_material_complexity)
        
        # 매칭 복잡도 점수
        matching_score = matching_complexity.get('complexity_score', 0)
        
        # 활동 복잡도 점수
        activity_score = (
            activity_complexity.get('coordination_complexity', 0) +
            activity_complexity.get('implementation_sequence_complexity', 0) +
            activity_complexity.get('resource_conflict_potential', 0)
        )
        
        # 최종 복잡도 점수 (0-100)
        total_score = (
            material_weight * material_score +
            matching_weight * matching_score +
            activity_weight * activity_score
        )
        
        return min(100, max(0, total_score))
    
    def _estimate_total_cost(self, variables: Dict[str, float]) -> float:
        """총 비용 추정"""
        # 기본적인 비용 추정 (시뮬레이션 정렬 없이)
        active_activities = self.get_active_activities()
        
        # 고정 비용
        fixed_cost = len(active_activities) * 1000.0
        
        # 가변 비용 (감축률에 비례)
        variable_cost = 0.0
        for activity in active_activities:
            reduction_rate = variables.get(activity, 0.0)
            variable_cost += reduction_rate * 50.0  # 감축률 1%당 50 달러
        
        return fixed_cost + variable_cost
    
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
        
        # 구현 용이성 분석 추가
        if results.get('status') == 'optimal':
            implementation_analysis = self.get_implementation_complexity()
            if implementation_analysis.get('status') == 'success':
                results['implementation_analysis'] = implementation_analysis
                
                # 구현 단계 추천
                results['implementation_steps'] = self._generate_implementation_steps()
        
        # 시뮬레이션 정렬 정보 추가
        results['simulation_aligned'] = self._is_simulation_aligned()
        if self._is_simulation_aligned():
            results['simulation_info'] = {
                'data_source': 'rule_based_simulation',
                'material_count': len(self.opt_input.scenario_df) if hasattr(self.opt_input, 'scenario_df') else 0,
                'complexity_calculation': 'simulation_aligned'
            }
        
        return results
    
    def _generate_implementation_steps(self) -> List[Dict[str, Any]]:
        """
        구현 단계 생성 (구현 용이성 기반)
        
        Returns:
            List[Dict]: 구현 단계 목록
        """
        if not self.results or self.results.get('status') != 'optimal':
            return []
            
        # 활성화된 활동 목록
        active_activities = self.get_active_activities()
        variables = self.results.get('variables', {})
        
        # 각 활동의 비용/효과 비율 계산
        activity_scores = []
        
        for activity in active_activities:
            # 감축률
            reduction_rate = variables.get(activity, 0.0)
            
            # 비용
            fixed_cost = self.opt_input.get_constants().get_activity_cost(activity)
            variable_cost_per_percent = self.opt_input.get_constants().get('variable_cost_per_percent', 50)
            variable_cost = reduction_rate * variable_cost_per_percent
            total_cost = fixed_cost + variable_cost
            
            # 점수 (낮을수록 우선순위 높음) - 비용 대비 효과
            score = total_cost / (reduction_rate + 0.001)  # 0으로 나누기 방지
            
            activity_scores.append({
                'activity': activity,
                'score': score,
                'reduction_rate': reduction_rate,
                'cost': total_cost
            })
        
        # 점수에 따라 정렬 (낮은 점수부터)
        activity_scores.sort(key=lambda x: x['score'])
        
        # 구현 단계 생성
        implementation_steps = []
        
        for i, activity_data in enumerate(activity_scores):
            activity = activity_data['activity']
            reduction_rate = activity_data['reduction_rate']
            cost = activity_data['cost']
            
            implementation_steps.append({
                'step': i + 1,
                'activity': activity,
                'reduction_rate': f"{reduction_rate:.2f}%",
                'cost': cost,
                'description': f"{activity} 활동을 {reduction_rate:.1f}% 감축률로 구현"
            })
        
        return implementation_steps
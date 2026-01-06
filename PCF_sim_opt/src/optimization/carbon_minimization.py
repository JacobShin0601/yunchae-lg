"""
탄소배출 최소화 시나리오 구현 모듈
"""

from typing import Dict, Any, Optional
import json
import yaml
from pathlib import Path
from datetime import datetime
from src.utils.logging_migration import log_info, log_warning, log_error

from .scenario_base import OptimizationScenario
from .input import OptimizationInput
from .constraints import ConstraintManager
from .objective import ObjectiveManager
from .results_processor import ResultsProcessor
from .validator import Validator
from .case_constraints import CaseConstraintManager
from .location_constraints import LocationConstraintManager
from .material_constraints import MaterialConstraintManager
from .location_recommender import LocationRecommender


class CarbonMinimization(OptimizationScenario):
    """
    탄소발자국 최소화 시나리오
    
    이 클래스는 탄소발자국을 최소화하는 최적화 문제를 구성하고 해결합니다.
    양극재 구성, 재활용 비율, 저탄소 원료 비율, 그리고 Tier 1/2/3 감축률을 최적화합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None, debug_mode: bool = True):
        """
        Args:
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
            debug_mode: 디버그 모드 사용 여부
        """
        super().__init__(
            config_path=config_path,
            name="carbon_minimization",
            description="탄소발자국 최소화 시나리오"
        )
        
        # 디버그 모드 설정
        self.debug_mode = debug_mode
        
        # 디버그 모드인 경우 추가 정보 출력
        if self.debug_mode:
            self._print(f"CarbonMinimization 초기화 (디버그 모드)")
    
    def _configure_scenario(self) -> None:
        """탄소배출 최소화 시나리오 설정 적용"""
        # 기본 설정에 시나리오가 있는지 확인하고 적용
        available_scenarios = self.opt_input.get_available_scenarios()
        
        if 'carbon_minimization' in available_scenarios:
            self.opt_input.apply_scenario('carbon_minimization')
        else:
            # 시나리오가 없으면 수동으로 설정
            custom_config = {
                'objective': 'minimize_carbon',
                'constraints': {
                    'target_carbon': 30.0,  # 도전적인 목표
                    'max_activities': 10    # 활동 제한 완화
                },
                'decision_vars': {
                    'cathode': {
                        'type': 'B'  # 선형 문제로 시작
                    }
                }
            }
            self.opt_input.create_custom_config(**custom_config)
    
    def _configure_model(self, model) -> None:
        """
        탄소배출 최소화 시나리오를 위한 모델 추가 설정
        
        Args:
            model: Pyomo 모델
        """
        # Case1 > Case2 > Case3 순으로 RE 비중 제약 조건 추가
        case_config = self.opt_input.get_config().get('case_constraints', {})
        if case_config.get('enabled', False):
            self._add_case_ordering_constraints(model, case_config)
            
        # 자재 생산국가 제약 조건 추가
        location_config = self.opt_input.get_config().get('location_constraints', {})
        if location_config.get('enabled', False):
            self._add_location_constraints(model, location_config)
        
        # 재활용재 및 저탄소 메탈 사용비율 제약 조건 추가
        material_config = self.opt_input.get_config().get('material_constraints', {})
        if material_config.get('enabled', False):
            self._add_material_constraints(model, material_config)
    
    def select_solver(self) -> str:
        """
        탄소배출 최소화에 적합한 솔버 선택
        
        Returns:
            str: 선택된 솔버 이름
        """
        # 양극재 타입에 따른 솔버 선택
        try:
            # 타입 A는 비선형이므로 IPOPT 필요
            cathode_config = self.opt_input.get_config().get('decision_vars', {}).get('cathode', {})
            cathode_type = cathode_config.get('type', 'B')
            
            if cathode_type == 'A':
                # 비선형 문제
                return 'ipopt'
            else:
                # 선형 문제
                return 'glpk'
        except Exception as e:
            # 오류 발생시 기본 솔버 반환
            print(f"솔버 선택 중 오류 발생: {e}")
            return 'glpk'
    
    def get_carbon_footprint(self) -> float:
        """
        최적해의 탄소발자국 값 계산 및 반환
        
        Returns:
            float: 탄소발자국
        """
        if not self.results or self.results.get('status') != 'optimal':
            return 0.0
            
        # 결과에 탄소발자국이 있으면 반환
        if 'carbon_footprint' in self.results:
            return self.results['carbon_footprint']
            
        # 없으면 formatted_results에서 추출
        formatted = self.results_processor.formatted_results
        if formatted and 'carbon_footprint' in formatted:
            try:
                # 문자열이 아닌 값의 경우 그대로 반환
                if not isinstance(formatted['carbon_footprint'], str):
                    return float(formatted['carbon_footprint'])
                
                # 문자열인 경우 분리 후 처리
                return float(formatted['carbon_footprint'].split()[0])
            except:
                return 0.0
                
        return 0.0
    
    def compare_to_baseline(self) -> Dict[str, Any]:
        """
        기준 시나리오 대비 개선율 계산
        
        Returns:
            Dict: 비교 결과
        """
        if not self.results or self.results.get('status') != 'optimal':
            return {'status': 'error', 'message': '최적화 결과가 없습니다.'}
            
        # 현재 탄소발자국
        current_carbon = self.get_carbon_footprint()
        
        # 기준 탄소발자국 (기본 배출량)
        base_emission = 80.0  # 기본값
        if hasattr(self, 'constants'):
            base_emission = self.constants.get('base_emission', 80.0)
            
        # 개선율 계산
        reduction = base_emission - current_carbon
        reduction_percentage = (reduction / base_emission) * 100 if base_emission > 0 else 0
        
        # 최적 국가 추천 정보 추가
        country_recommendations = self.get_country_recommendations()
        
        return {
            'baseline': base_emission,
            'optimized': current_carbon,
            'reduction': reduction,
            'reduction_percentage': f"{reduction_percentage:.2f}%",
            'country_recommendations': country_recommendations,
            'status': 'success'
        }
    
    def _print(self, message: str, level: str = "info") -> None:
        """
        로깅 메시지 출력
        
        Args:
            message: 로깅할 메시지
            level: 로깅 레벨 ('info', 'warning', 'error', 'debug')
        """
        if level == "info":
            log_info(message)
        elif level == "warning":
            log_warning(message)
        elif level == "error":
            log_error(message)
        elif level == "debug":
            # debug는 기본 log_info로 처리
            log_info(f"[DEBUG] {message}")
            
    def _add_case_ordering_constraints(self, model, case_config: Dict[str, Any]) -> None:
        """
        Tier 내 Case1 > Case2 > Case3 순으로 RE 비중 제약 조건 추가
        
        Args:
            model: Pyomo 모델
            case_config: Case 제약조건 설정 사전
        """
        # CaseConstraintManager 초기화
        case_manager = CaseConstraintManager(model)
        
        # Case 변수 등록
        case_variables = case_config.get('variables', {})
        case_manager.add_case_variables_from_dict(case_variables)
        
        # 순서 제약조건 생성
        constraints_added = case_manager.create_case_ordering_constraints()
        
        # 최소 차이 제약조건 생성 (설정된 경우)
        min_difference = case_config.get('min_difference')
        if min_difference is not None:
            case_manager.create_minimum_difference_constraints(min_difference)
        
        self._print(f"Case 제약조건 {constraints_added}개가 추가되었습니다.", level="info")
    
    def _add_location_constraints(self, model, location_config: Dict[str, Any]) -> None:
        """
        자재 생산국가 제약조건 추가
        
        Args:
            model: Pyomo 모델
            location_config: 생산국가 제약조건 설정 사전
        """
        # LocationConstraintManager 초기화
        location_manager = LocationConstraintManager(model)
        
        # 자재별 위치 변수 등록
        location_variables = location_config.get('variables', {})
        for material, variable in location_variables.items():
            location_manager.add_location_variable(material, variable)
        
        # 자재별 생산국가 제약조건 설정
        material_locations = location_config.get('material_locations', {})
        location_manager.set_material_location_constraints_from_dict(material_locations)
        
        # 제약조건 생성
        constraints_added = location_manager.create_location_constraints()
        
        # 가능한 경우 자재별 생산국가 고정 제약조건도 생성
        if location_config.get('use_fixed_constraints', False):
            fixed_constraints = location_manager.create_material_location_constraints()
            constraints_added += fixed_constraints
        
        self._print(f"자재 생산국가 제약조건 {constraints_added}개가 추가되었습니다.", level="info")
        
    def _add_material_constraints(self, model, material_config: Dict[str, Any]) -> None:
        """
        재활용재 및 저탄소 메탈 사용비율 제약조건 추가
        
        Args:
            model: Pyomo 모델
            material_config: 원료 제약조건 설정 사전
        """
        # 양극재 타입 확인
        try:
            # 먼저 Type A 양극재 구성인지 확인
            try:
                from .type_a_constraint_fix import is_type_a_cathode, add_type_a_material_constraints
                if is_type_a_cathode(model):
                    self._print("양극재 Type A 구성 감지: 전용 제약조건 적용", level="info")
                    add_type_a_material_constraints(model)
                    return  # Type A 양극재에 대한 처리를 완료했으므로 이후 로직은 실행하지 않음
            except ImportError:
                self._print("양극재 Type A 제약조건 수정 모듈을 로드할 수 없음", level="warning")
                
            # 기본 구현 사용 - Type B 양극재 혹은 수정 모듈을 사용하지 않는 경우
            # MaterialConstraintManager 초기화
            material_manager = MaterialConstraintManager(model)
            
            # 재활용재 사용비율 제약조건
            recycle_config = material_config.get('recycle_ratio', {})
            recycle_constraints = 0
            if recycle_config.get('enabled', False):
                min_ratio = recycle_config.get('min', 0.1)
                max_ratio = recycle_config.get('max', 0.5)
                recycle_constraints = material_manager.create_recycle_ratio_constraints(min_ratio, max_ratio)
            
            # 저탄소 메탈 사용비율 제약조건
            low_carbon_config = material_config.get('low_carbon_ratio', {})
            low_carbon_constraints = 0
            if low_carbon_config.get('enabled', False):
                min_ratio = low_carbon_config.get('min', 0.05)
                max_ratio = low_carbon_config.get('max', 0.3)
                low_carbon_constraints = material_manager.create_low_carbon_ratio_constraints(min_ratio, max_ratio)
            
            # 원료 합계 제약조건
            balance_config = material_config.get('material_balance', {})
            balance_constraints = 0
            if balance_config.get('enabled', False):
                max_total = balance_config.get('max_total', 0.7)
                balance_constraints = material_manager.create_material_balance_constraint(max_total)
            
            # 원료 간 비율 제약조건
            proportion_config = material_config.get('proportions', {})
            proportion_constraints = 0
            if proportion_config.get('enabled', False):
                proportions = proportion_config.get('values', {})
                proportion_constraints = material_manager.create_material_proportion_constraints(proportions)
            
            total_constraints = recycle_constraints + low_carbon_constraints + balance_constraints + proportion_constraints
            self._print(f"원료 제약조건 {total_constraints}개가 추가되었습니다. (재활용재: {recycle_constraints}, 저탄소메탈: {low_carbon_constraints}, 원료합계: {balance_constraints}, 가용비: {proportion_constraints})", level="info")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._print(f"원료 제약조건 추가 중 오류 발생: {e}", level="error")
    
    def get_country_recommendations(self) -> Dict[str, Any]:
        """
        최적 국가 추천 정보 반환
        
        Returns:
            Dict: 추천 국가 정보
        """
        # LocationRecommender 초기화
        stable_var_dir = self.opt_input.stable_var_dir
        user_id = getattr(self.opt_input, 'user_id', None)
        recommender = LocationRecommender(stable_var_dir=str(stable_var_dir), user_id=user_id)
        
        # 기본 추천 가중치
        carbon_weight = 0.6
        cost_weight = 0.3
        logistics_weight = 0.1
        
        # 추천 보고서 생성
        recommendation_report = recommender.generate_recommendation_report(
            carbon_weight=carbon_weight,
            cost_weight=cost_weight,
            logistics_weight=logistics_weight,
            top_n=3
        )
        
        # 비용 차이 추정
        # (최고/최저 지역 차이를 활용한 찾아구현 프리미엄)
        cost_data = recommendation_report.get('상세데이터', [])
        if cost_data and len(cost_data) > 0:
            max_cost = max([item.get('생산비용지수', 1.0) for item in cost_data])
            min_cost = min([item.get('생산비용지수', 1.0) for item in cost_data])
            best_country = recommendation_report.get('최종추천국가', [''])[0]
            best_country_cost = next((item.get('생산비용지수', 1.0) for item in cost_data if item.get('국가') == best_country), min_cost)
            
            # 최적 지역으로 변경 시 추가 비용 계산
            cost_premium = (best_country_cost - min_cost) * 100  # 퍼센트로 표시
            recommendation_report['비용_프리미엄'] = cost_premium
        
        return recommendation_report
    
    def solve(self, solver_name: Optional[str] = None) -> Dict[str, Any]:
        """
        최적화 문제 해결 (부모 클래스의 메서드 재정의)
        
        Args:
            solver_name: 사용할 솔버 이름 (None이면 자동 선택)
            
        Returns:
            Dict: 최적화 결과
        """
        try:
            # 디버그 모드인 경우 추가 정보 출력
            if self.debug_mode:
                # 양극재 타입 확인
                cathode_config = self.opt_input.get_config().get('decision_vars', {}).get('cathode', {})
                cathode_type = cathode_config.get('type', 'unknown')
                self._print(f"solve() 실행: 양극재 타입={cathode_type}, 솔버={solver_name or self.select_solver()}")
                
                # 모델 확인
                if self.model:
                    self._print("Model attributes:")
                    model_attributes = [attr for attr in dir(self.model) if not attr.startswith('_')]
                    self._print(f"Model has {len(model_attributes)} attributes")
                    self._print(f"recycle_ratio in model: {hasattr(self.model, 'recycle_ratio')}")
                    if hasattr(self.model, 'recycle_ratio'):
                        self._print(f"recycle_ratio type: {type(self.model.recycle_ratio)}")
            
            # 부모 클래스의 solve 메서드 호출
            return super().solve(solver_name)
        except TypeError as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            
            # 양극재 타입 관련 오류 확인
            cathode_type = "unknown"
            try:
                cathode_config = self.opt_input.get_config().get('decision_vars', {}).get('cathode', {})
                cathode_type = cathode_config.get('type', 'unknown')
            except Exception:
                pass
                
            return {
                'status': 'error',
                'message': f"'int' object is not callable 오류 발생 (양극재 타입: {cathode_type}): {error_msg}"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f"최적화 중 오류 발생: {str(e)}"
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
                
        return self.results_processor.formatted_results
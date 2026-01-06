"""
그리드서치 최적화 모듈

이 모듈은 주어진 제약조건 내에서 가능한 모든 해결책을 탐색하는
그리드서치 최적화 기능을 제공합니다. 다양한 파라미터 조합에 대해
시뮬레이션을 수행하고 최적의 조합을 찾습니다.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import os
import json
from pathlib import Path

from .material_based_optimizer import MaterialBasedOptimizer


class GridSearchOptimizer:
    """
    그리드서치 최적화 클래스
    
    이 클래스는 여러 파라미터 값의 조합을 탐색하여 최적의 파라미터 조합을 찾습니다.
    주어진 제약 조건을 만족하면서 탄소 배출량을 최소화하는 해결책을 탐색합니다.
    """
    
    def __init__(self, 
                 simulation_data: Dict[str, pd.DataFrame] = None,
                 base_config: Dict[str, Any] = None,
                 stable_var_dir: str = "stable_var",
                 user_id: Optional[str] = None,
                 debug_mode: bool = False):
        """
        GridSearchOptimizer 초기화
        
        Args:
            simulation_data: 시뮬레이션 데이터 (시나리오 및 참조 데이터프레임)
            base_config: 기본 최적화 설정
            stable_var_dir: stable_var 디렉토리 경로
            user_id: 사용자 ID (사용자별 데이터 사용시)
            debug_mode: 디버그 모드 사용 여부
        """
        self.debug_mode = debug_mode
        self.user_id = user_id
        self.stable_var_dir = stable_var_dir
        self.simulation_data = simulation_data or {}
        self.base_config = base_config or self._get_default_base_config()
        
        # 그리드서치 결과 저장
        self.results = []
        self.pareto_front = []
        
        # 기본 MaterialBasedOptimizer 인스턴스 생성
        self.base_optimizer = MaterialBasedOptimizer(
            simulation_data=self.simulation_data,
            config=self.base_config,
            stable_var_dir=self.stable_var_dir,
            user_id=self.user_id,
            debug_mode=False  # 그리드서치시 개별 최적화 디버그 출력은 비활성화
        )
        
        # 그리드서치 파라미터 초기화
        self.grid_params = {}
        
        # 디버그 정보 출력
        if self.debug_mode:
            self._print_debug_info()
    
    def _get_default_base_config(self) -> Dict[str, Any]:
        """기본 최적화 설정 반환"""
        return {
            'reduction_target': {
                'min': -5,   # 최소 감축률 (%)
                'max': -10,  # 최대 감축률 (%)
            },
            're_rates': {
                'tier1': {'min': 0.1, 'max': 0.9},  # Tier1 RE 적용률 범위
                'tier2': {'min': 0.1, 'max': 0.9},  # Tier2 RE 적용률 범위
                'tier3': {'min': 0.1, 'max': 0.9},  # Tier3 RE 적용률 범위
            },
            'material_ratios': {
                'recycle': {'min': 0.05, 'max': 0.5},      # 재활용 비율 범위
                'low_carbon': {'min': 0.05, 'max': 0.3},  # 저탄소메탈 비율 범위
            }
        }
    
    def _print_debug_info(self) -> None:
        """디버그 정보 출력"""
        print("===== GridSearchOptimizer 초기화 정보 =====")
        
        # 시뮬레이션 데이터 정보
        scenario_count = len(self.simulation_data.get('scenario_df', pd.DataFrame()))
        formula_count = len(self.simulation_data.get('ref_formula_df', pd.DataFrame()))
        proportions_count = len(self.simulation_data.get('ref_proportions_df', pd.DataFrame()))
        
        print(f"• 시뮬레이션 데이터:")
        print(f"  - 시나리오 자재: {scenario_count}개")
        print(f"  - Formula 참조 데이터: {formula_count}개")
        print(f"  - Proportions 참조 데이터: {proportions_count}개")
        
        # 기본 설정 정보
        min_reduction = self.base_config['reduction_target']['min']
        max_reduction = self.base_config['reduction_target']['max']
        
        print(f"• 기본 감축 목표: {min_reduction}% ~ {max_reduction}%")
        
        # 기준 PCF 계산
        if hasattr(self.base_optimizer, 'original_pcf'):
            print(f"• 기준 PCF: {self.base_optimizer.original_pcf:.4f} kgCO2eq")
    
    def set_grid_params(self, grid_params: Dict[str, List[Any]]) -> None:
        """
        그리드서치 파라미터 설정
        
        Args:
            grid_params: 파라미터별 탐색 값 목록
            예: {
                'reduction_min': [-3, -5, -7],
                'reduction_max': [-8, -10, -12],
                'tier1_re': [0.2, 0.3, 0.4],
                'recycle_ratio': [0.1, 0.2, 0.3]
            }
        """
        self.grid_params = grid_params
        
        if self.debug_mode:
            print(f"✅ 그리드서치 파라미터 설정됨:")
            for param_name, values in grid_params.items():
                print(f"  • {param_name}: {values}")
            
            # 총 조합 수 계산
            total_combinations = 1
            for values in grid_params.values():
                total_combinations *= len(values)
            print(f"• 총 파라미터 조합 수: {total_combinations}개")
    
    def run_grid_search(self, max_iterations: int = 1000, progress_callback: Callable[[int, int], None] = None) -> List[Dict[str, Any]]:
        """
        그리드서치 최적화 실행
        
        Args:
            max_iterations: 최대 탐색 조합 수 (기본값: 1000)
            progress_callback: 진행 상황 콜백 함수 (현재 진행, 총 진행)
            
        Returns:
            List[Dict[str, Any]]: 그리드서치 결과 리스트
        """
        # 파라미터 목록 및 조합 생성
        param_names = list(self.grid_params.keys())
        param_values = list(self.grid_params.values())
        
        # 모든 조합 생성
        all_combinations = list(itertools.product(*param_values))
        
        # 최대 조합 수 제한
        combinations_to_run = all_combinations[:max_iterations]
        total_combinations = len(combinations_to_run)
        
        if self.debug_mode:
            print(f"🔍 그리드서치 시작 (총 {total_combinations}개 조합 탐색)")
            print(f"• 최대 조합 수: {max_iterations}개")
            print(f"• 실제 탐색 조합 수: {total_combinations}개")
        
        # 결과 초기화
        self.results = []
        
        # 조합별 최적화 실행
        for i, combo in enumerate(combinations_to_run):
            # 현재 진행 상황 콜백
            if progress_callback:
                progress_callback(i + 1, total_combinations)
            
            # 현재 조합으로 설정 구성
            current_config = self._create_config_for_combination(param_names, combo)
            
            # 최적화 실행
            result = self._run_single_optimization(current_config)
            
            # 결과 저장
            if result and result.get('status') == 'optimal':
                self.results.append({
                    'params': dict(zip(param_names, combo)),
                    'result': result
                })
            
            # 디버그 정보 출력
            if self.debug_mode and (i+1) % 10 == 0:
                print(f"  • 진행 상황: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
        
        # 파레토 최적해 계산
        self._calculate_pareto_front()
        
        # 결과 요약
        if self.debug_mode:
            print(f"✅ 그리드서치 완료:")
            print(f"• 총 실행 조합: {len(combinations_to_run)}개")
            print(f"• 최적해 찾은 조합: {len(self.results)}개")
            print(f"• 파레토 최적해: {len(self.pareto_front)}개")
        
        return self.results
    
    def _create_config_for_combination(self, param_names: List[str], combo: Tuple) -> Dict[str, Any]:
        """파라미터 조합으로 설정 사전 생성"""
        # 기본 설정 복사
        config = self._deep_copy_config(self.base_config)
        
        # 조합의 파라미터 값 적용
        for param_name, param_value in zip(param_names, combo):
            self._update_config_param(config, param_name, param_value)
        
        return config
    
    def _deep_copy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """설정 사전 깊은 복사"""
        import copy
        return copy.deepcopy(config)
    
    def _update_config_param(self, config: Dict[str, Any], param_name: str, param_value: Any) -> None:
        """설정 사전에 특정 파라미터 업데이트"""
        # 감축률 파라미터
        if param_name == 'reduction_min':
            config['reduction_target']['min'] = param_value
        elif param_name == 'reduction_max':
            config['reduction_target']['max'] = param_value
        
        # Tier RE 파라미터
        elif param_name == 'tier1_re_min':
            config['re_rates']['tier1']['min'] = param_value
        elif param_name == 'tier1_re_max':
            config['re_rates']['tier1']['max'] = param_value
        elif param_name == 'tier2_re_min':
            config['re_rates']['tier2']['min'] = param_value
        elif param_name == 'tier2_re_max':
            config['re_rates']['tier2']['max'] = param_value
        elif param_name == 'tier3_re_min':
            config['re_rates']['tier3']['min'] = param_value
        elif param_name == 'tier3_re_max':
            config['re_rates']['tier3']['max'] = param_value
            
        # 단일 값으로 min/max 모두 설정하는 경우
        elif param_name == 'tier1_re':
            config['re_rates']['tier1']['min'] = param_value
            config['re_rates']['tier1']['max'] = param_value
        elif param_name == 'tier2_re':
            config['re_rates']['tier2']['min'] = param_value
            config['re_rates']['tier2']['max'] = param_value
        elif param_name == 'tier3_re':
            config['re_rates']['tier3']['min'] = param_value
            config['re_rates']['tier3']['max'] = param_value
        
        # 자재 비율 파라미터
        elif param_name == 'recycle_min':
            config['material_ratios']['recycle']['min'] = param_value
        elif param_name == 'recycle_max':
            config['material_ratios']['recycle']['max'] = param_value
        elif param_name == 'low_carbon_min':
            config['material_ratios']['low_carbon']['min'] = param_value
        elif param_name == 'low_carbon_max':
            config['material_ratios']['low_carbon']['max'] = param_value
            
        # 단일 값으로 min/max 모두 설정하는 경우
        elif param_name == 'recycle_ratio':
            config['material_ratios']['recycle']['min'] = param_value
            config['material_ratios']['recycle']['max'] = param_value
        elif param_name == 'low_carbon_ratio':
            config['material_ratios']['low_carbon']['min'] = param_value
            config['material_ratios']['low_carbon']['max'] = param_value
        
        # 기타 커스텀 파라미터
        else:
            # 파라미터 이름을 점으로 분리 (예: 'constraints.apply_formula_first')
            parts = param_name.split('.')
            if len(parts) > 1:
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = param_value
            else:
                config[param_name] = param_value
    
    def _run_single_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """단일 최적화 실행"""
        # 최적화기 생성
        optimizer = MaterialBasedOptimizer(
            simulation_data=self.simulation_data,
            config=config,
            stable_var_dir=self.stable_var_dir,
            user_id=self.user_id,
            debug_mode=False  # 개별 최적화는 디버그 출력 비활성화
        )
        
        # 최적화 모델 구성
        optimizer.build_optimization_model()
        
        # 최적화 실행
        return optimizer.solve('glpk')
    
    def _calculate_pareto_front(self) -> None:
        """파레토 최적해 계산"""
        if not self.results:
            self.pareto_front = []
            return
        
        # 모든 결과에서 PCF와 비용(가정) 추출
        points = []
        for i, result_item in enumerate(self.results):
            result = result_item['result']
            if result.get('status') == 'optimal':
                # PCF
                pcf = result.get('optimized_pcf', 0)
                
                # 비용 (RE 적용률 + 재활용/저탄소메탈 비율에 비례)
                cost = self._calculate_implementation_cost(result)
                
                points.append((i, pcf, cost))
        
        # 파레토 최적해 계산
        self.pareto_front = self._find_pareto_front(points)
    
    def _calculate_implementation_cost(self, result: Dict[str, Any]) -> float:
        """구현 비용 계산 (간단한 휴리스틱 사용)"""
        # 티어별 RE 적용률에 비례하는 비용
        re_cost = 0
        tier1_sum = 0
        tier2_sum = 0
        tier3_sum = 0
        material_count = 0
        
        # 변수에서 RE 적용률 추출
        variables = result.get('variables', {})
        for var_name, var_value in variables.items():
            if var_name.startswith('tier1_re_'):
                tier1_sum += var_value
                material_count += 1
            elif var_name.startswith('tier2_re_'):
                tier2_sum += var_value
            elif var_name.startswith('tier3_re_'):
                tier3_sum += var_value
        
        if material_count > 0:
            # 가중치 설정 (Tier별로 비용이 다름)
            re_cost = (tier1_sum * 1.0 + tier2_sum * 1.5 + tier3_sum * 2.0) / material_count
        
        # 재활용/저탄소메탈 비율에 비례하는 비용
        recycle_sum = 0
        low_carbon_sum = 0
        ni_co_li_count = 0
        
        for var_name, var_value in variables.items():
            if var_name.startswith('recycle_ratio_'):
                recycle_sum += var_value
                ni_co_li_count += 1
            elif var_name.startswith('low_carbon_ratio_'):
                low_carbon_sum += var_value
        
        material_cost = 0
        if ni_co_li_count > 0:
            # 저탄소메탈은 재활용보다 비용이 더 높다고 가정
            material_cost = (recycle_sum * 1.2 + low_carbon_sum * 1.8) / ni_co_li_count
        
        # 전체 비용 계산
        total_cost = re_cost * 0.7 + material_cost * 0.3
        
        return total_cost * 100  # 0-100 범위로 정규화
    
    def _find_pareto_front(self, points: List[Tuple[int, float, float]]) -> List[Tuple[int, float, float]]:
        """PCF와 비용에 대한 파레토 최적해 계산"""
        pareto_front = []
        
        for i, pcf, cost in points:
            is_dominated = False
            for j, other_pcf, other_cost in points:
                if i != j and other_pcf <= pcf and other_cost <= cost and (other_pcf < pcf or other_cost < cost):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append((i, pcf, cost))
        
        return pareto_front
    
    def get_pareto_results(self) -> List[Dict[str, Any]]:
        """파레토 최적해 결과 반환"""
        pareto_results = []
        
        for i, pcf, cost in self.pareto_front:
            result_item = self.results[i].copy()
            result_item['pcf'] = pcf
            result_item['cost'] = cost
            pareto_results.append(result_item)
        
        return pareto_results
    
    def export_results_to_csv(self, file_path: str = None) -> str:
        """결과를 CSV 파일로 내보내기"""
        if not self.results:
            raise ValueError("그리드서치 결과가 없습니다.")
        
        # 결과를 DataFrame으로 변환
        rows = []
        for result_item in self.results:
            params = result_item['params']
            result = result_item['result']
            
            if result.get('status') == 'optimal':
                row = {
                    'optimized_pcf': result.get('optimized_pcf', 0),
                    'reduction_percentage': result.get('reduction_percentage', 0),
                    'implementation_cost': self._calculate_implementation_cost(result)
                }
                
                # 파라미터 추가
                for param_name, param_value in params.items():
                    row[f"param_{param_name}"] = param_value
                
                # 중요 변수 값 추가
                variables = result.get('variables', {})
                for var_name, var_value in variables.items():
                    # 모든 변수를 포함하면 너무 커지므로, 중요 변수만 선택
                    if any(var_name.startswith(prefix) for prefix in ['tier1_re_', 'tier2_re_', 'recycle_ratio_', 'low_carbon_ratio_']):
                        material_name = var_name.split('_', 1)[1]  # 'tier1_re_' 이후 부분
                        row[var_name] = var_value
                
                rows.append(row)
        
        # DataFrame 생성
        results_df = pd.DataFrame(rows)
        
        # 파일 경로 설정
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"grid_search_results_{timestamp}.csv"
        
        # CSV로 저장
        results_df.to_csv(file_path, index=False)
        
        return file_path
    
    def export_pareto_results_to_csv(self, file_path: str = None) -> str:
        """파레토 최적해 결과를 CSV 파일로 내보내기"""
        pareto_results = self.get_pareto_results()
        
        if not pareto_results:
            raise ValueError("파레토 최적해 결과가 없습니다.")
        
        # 결과를 DataFrame으로 변환
        rows = []
        for result_item in pareto_results:
            params = result_item['params']
            result = result_item['result']
            
            row = {
                'optimized_pcf': result.get('optimized_pcf', 0),
                'reduction_percentage': result.get('reduction_percentage', 0),
                'implementation_cost': result_item.get('cost', 0)
            }
            
            # 파라미터 추가
            for param_name, param_value in params.items():
                row[f"param_{param_name}"] = param_value
            
            # 중요 변수 값 추가 (모든 변수를 포함하면 너무 커짐)
            variables = result.get('variables', {})
            for var_name, var_value in variables.items():
                if any(var_name.startswith(prefix) for prefix in ['tier1_re_', 'tier2_re_', 'recycle_ratio_', 'low_carbon_ratio_']):
                    row[var_name] = var_value
            
            rows.append(row)
        
        # DataFrame 생성
        results_df = pd.DataFrame(rows)
        
        # 파일 경로 설정
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"pareto_results_{timestamp}.csv"
        
        # CSV로 저장
        results_df.to_csv(file_path, index=False)
        
        return file_path
    
    def generate_visualization_data(self) -> Dict[str, Any]:
        """시각화를 위한 데이터 생성"""
        if not self.results:
            return {
                'all_points': [],
                'pareto_points': [],
                'axis_ranges': {
                    'pcf': [0, 1],
                    'cost': [0, 1]
                }
            }
        
        # 모든 결과 포인트
        all_points = []
        for i, result_item in enumerate(self.results):
            result = result_item['result']
            if result.get('status') == 'optimal':
                pcf = result.get('optimized_pcf', 0)
                cost = self._calculate_implementation_cost(result)
                reduction = result.get('reduction_percentage', 0)
                
                all_points.append({
                    'index': i,
                    'pcf': pcf,
                    'cost': cost,
                    'reduction': reduction,
                    'params': result_item['params']
                })
        
        # 파레토 최적해 포인트
        pareto_points = []
        for i, pcf, cost in self.pareto_front:
            result_item = self.results[i]
            result = result_item['result']
            reduction = result.get('reduction_percentage', 0)
            
            pareto_points.append({
                'index': i,
                'pcf': pcf,
                'cost': cost,
                'reduction': reduction,
                'params': result_item['params']
            })
        
        # 축 범위 계산
        pcf_values = [p['pcf'] for p in all_points]
        cost_values = [p['cost'] for p in all_points]
        
        pcf_min = min(pcf_values) if pcf_values else 0
        pcf_max = max(pcf_values) if pcf_values else 1
        cost_min = min(cost_values) if cost_values else 0
        cost_max = max(cost_values) if cost_values else 1
        
        # 여백 추가
        pcf_range = pcf_max - pcf_min
        cost_range = cost_max - cost_min
        pcf_min -= pcf_range * 0.05
        pcf_max += pcf_range * 0.05
        cost_min -= cost_range * 0.05
        cost_max += cost_range * 0.05
        
        return {
            'all_points': all_points,
            'pareto_points': pareto_points,
            'axis_ranges': {
                'pcf': [pcf_min, pcf_max],
                'cost': [cost_min, cost_max]
            }
        }
    
    def get_best_result_by_pcf(self) -> Dict[str, Any]:
        """PCF 기준 최적해 반환"""
        if not self.results:
            return {'status': 'error', 'message': '그리드서치 결과가 없습니다.'}
        
        # PCF가 가장 작은 결과 찾기
        best_index = -1
        best_pcf = float('inf')
        
        for i, result_item in enumerate(self.results):
            result = result_item['result']
            if result.get('status') == 'optimal':
                pcf = result.get('optimized_pcf', 0)
                if pcf < best_pcf:
                    best_pcf = pcf
                    best_index = i
        
        if best_index >= 0:
            return self.results[best_index]
        else:
            return {'status': 'error', 'message': '최적해를 찾지 못했습니다.'}
    
    def get_best_result_by_cost(self) -> Dict[str, Any]:
        """비용 기준 최적해 반환"""
        if not self.results:
            return {'status': 'error', 'message': '그리드서치 결과가 없습니다.'}
        
        # 비용이 가장 작은 결과 찾기
        best_index = -1
        best_cost = float('inf')
        
        for i, result_item in enumerate(self.results):
            result = result_item['result']
            if result.get('status') == 'optimal':
                cost = self._calculate_implementation_cost(result)
                if cost < best_cost:
                    best_cost = cost
                    best_index = i
        
        if best_index >= 0:
            return self.results[best_index]
        else:
            return {'status': 'error', 'message': '최적해를 찾지 못했습니다.'}
    
    def get_best_compromise_result(self) -> Dict[str, Any]:
        """PCF와 비용 사이의 최적 타협안 반환"""
        if not self.get_pareto_results():
            return {'status': 'error', 'message': '파레토 최적해가 없습니다.'}
        
        # PCF와 비용을 정규화하고 가중합이 가장 작은 결과 찾기
        pareto_results = self.get_pareto_results()
        
        # PCF와 비용 범위
        pcf_values = [result['pcf'] for result in pareto_results]
        cost_values = [result['cost'] for result in pareto_results]
        
        pcf_min = min(pcf_values)
        pcf_max = max(pcf_values)
        pcf_range = pcf_max - pcf_min if pcf_max > pcf_min else 1
        
        cost_min = min(cost_values)
        cost_max = max(cost_values)
        cost_range = cost_max - cost_min if cost_max > cost_min else 1
        
        # 각 결과의 정규화된 점수 계산
        best_index = -1
        best_score = float('inf')
        
        for i, result in enumerate(pareto_results):
            # 정규화된 PCF와 비용 (0~1 범위)
            norm_pcf = (result['pcf'] - pcf_min) / pcf_range
            norm_cost = (result['cost'] - cost_min) / cost_range
            
            # PCF와 비용의 가중합 (0.6:0.4 가중치)
            score = norm_pcf * 0.6 + norm_cost * 0.4
            
            if score < best_score:
                best_score = score
                best_index = i
        
        if best_index >= 0:
            return pareto_results[best_index]
        else:
            return {'status': 'error', 'message': '최적 타협안을 찾지 못했습니다.'}
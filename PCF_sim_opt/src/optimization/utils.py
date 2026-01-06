"""
최적화 관련 유틸리티 함수들
"""

from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_stable_var_data() -> Dict[str, Any]:
    """
    stable_var 디렉토리에서 필요한 데이터 로드
    
    Returns:
        Dict: 전력배출계수, 재활용재 환경영향 등의 데이터
    """
    data = {}
    stable_var_path = Path("stable_var")
    
    # 국가별 전력배출계수 로드
    electricity_coef_path = stable_var_path / "electricity_coef_by_country.json"
    if electricity_coef_path.exists():
        with open(electricity_coef_path, 'r', encoding='utf-8') as f:
            data['electricity_coef'] = json.load(f)
    else:
        # 기본값
        data['electricity_coef'] = {
            "한국": 0.637420635,
            "중국": 0.8825,
            "일본": 0.667861719,
            "폴란드": 0.948984701,
            "미국": 0.522,
            "유럽": 0.432
        }
    
    # 재활용재 환경영향 로드
    recycle_impact_path = stable_var_path / "recycle_material_impact.json"
    if recycle_impact_path.exists():
        with open(recycle_impact_path, 'r', encoding='utf-8') as f:
            data['recycle_impact'] = json.load(f)
    else:
        # 기본값
        data['recycle_impact'] = {
            "신재": 1.0,
            "재활용재": {"Ni": 0.1, "Co": 0.15, "Li": 0.1}
        }
    
    # 양극재 Tier1 입력 데이터 로드
    cathode_tier1_path = stable_var_path / "cathode_tier1_input.json"
    if cathode_tier1_path.exists():
        with open(cathode_tier1_path, 'r', encoding='utf-8') as f:
            data['cathode_tier1'] = json.load(f)
    
    # 양극재 Tier2 입력 데이터 로드
    cathode_tier2_path = stable_var_path / "cathode_tier2_input.json"
    if cathode_tier2_path.exists():
        with open(cathode_tier2_path, 'r', encoding='utf-8') as f:
            data['cathode_tier2'] = json.load(f)
    
    return data


def select_optimal_solver(config: Dict[str, Any]) -> str:
    """
    문제 유형에 따라 최적의 솔버 선택
    
    Args:
        config: 최적화 설정
        
    Returns:
        str: 추천 솔버 이름
    """
    objective_type = config.get('objective')
    cathode_config = config.get('decision_vars', {}).get('cathode', {})
    
    # Type A는 비선형 문제
    if cathode_config.get('type') == 'A':
        return 'ipopt'
    
    # 용이성 최대화는 정수 계획 문제
    if objective_type == 'maximize_ease':
        return 'cbc'
    
    # 그 외는 선형 문제로 간주
    return 'glpk'


def validate_optimization_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    최적화 설정 유효성 검증
    
    Args:
        config: 최적화 설정
        
    Returns:
        Tuple[bool, Optional[str]]: (유효성 여부, 에러 메시지)
    """
    # 필수 키 확인
    required_keys = ['objective', 'decision_vars', 'constraints']
    for key in required_keys:
        if key not in config:
            return False, f"필수 설정 '{key}'가 누락되었습니다."
    
    # 목적함수 유형 확인
    valid_objectives = ['minimize_carbon', 'minimize_cost', 'multi_objective', 'maximize_ease']
    if config['objective'] not in valid_objectives:
        return False, f"유효하지 않은 목적함수: {config['objective']}"
    
    # 의사결정변수 확인
    decision_vars = config.get('decision_vars', {})
    if 'cathode' not in decision_vars:
        return False, "양극재 구성 설정이 누락되었습니다."
    
    cathode_config = decision_vars['cathode']
    if 'type' not in cathode_config:
        return False, "양극재 프로젝트 유형이 지정되지 않았습니다."
    
    # Type별 필수 설정 확인
    if cathode_config['type'] == 'A':
        required_a = ['recycle_ratio_fixed', 'low_carbon_ratio_fixed', 'emission_range']
        for key in required_a:
            if key not in cathode_config:
                return False, f"Type A 필수 설정 '{key}'가 누락되었습니다."
    elif cathode_config['type'] == 'B':
        required_b = ['emission_fixed', 'recycle_range', 'low_carbon_range']
        for key in required_b:
            if key not in cathode_config:
                return False, f"Type B 필수 설정 '{key}'가 누락되었습니다."
    
    # 제약조건 확인
    constraints = config.get('constraints', {})
    if 'target_carbon' not in constraints:
        return False, "목표 탄소발자국이 설정되지 않았습니다."
    
    return True, None


def format_optimization_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    최적화 결과를 보기 좋은 DataFrame으로 변환
    
    Args:
        results: 최적화 결과
        
    Returns:
        pd.DataFrame: 정리된 결과 테이블
    """
    if results.get('status') != 'optimal':
        return pd.DataFrame()
    
    # 변수 결과 정리
    variables = results.get('variables', {})
    data = []
    
    # 감축비율 변수
    for var_name, value in variables.items():
        if 'tier' in var_name and not var_name.endswith('_active'):
            tier = var_name.split('_')[0].upper()
            item = ' '.join(var_name.split('_')[1:])
            data.append({
                '구분': '감축비율',
                'Tier': tier,
                '항목': item,
                '값': f"{value:.2f}%"
            })
    
    # 양극재 구성
    if 'recycle_ratio' in variables:
        data.append({
            '구분': '양극재 구성',
            'Tier': '-',
            '항목': '재활용 비율',
            '값': f"{variables['recycle_ratio']:.3f}"
        })
    
    if 'low_carbon_ratio' in variables:
        data.append({
            '구분': '양극재 구성',
            'Tier': '-',
            '항목': '저탄소원료 비율',
            '값': f"{variables['low_carbon_ratio']:.3f}"
        })
    
    if 'low_carbon_emission' in variables:
        data.append({
            '구분': '양극재 구성',
            'Tier': '-',
            '항목': '저탄소 원료구성',
            '값': f"{variables['low_carbon_emission']:.2f}"
        })
    
    return pd.DataFrame(data)


def calculate_carbon_footprint(
    config: Dict[str, Any],
    variables: Dict[str, float],
    stable_var_data: Dict[str, Any]
) -> float:
    """
    주어진 변수값으로 탄소발자국 계산
    
    Args:
        config: 최적화 설정
        variables: 변수값 딕셔너리
        stable_var_data: stable_var 데이터
        
    Returns:
        float: 계산된 탄소발자국
    """
    # 기본 배출량
    base_emission = 80
    
    # 생산지 전력배출계수
    location = config.get('decision_vars', {}).get('location', '한국')
    electricity_coefs = stable_var_data.get('electricity_coef', {})
    location_factor = electricity_coefs.get(location, 0.637)
    
    # Tier별 감축 효과
    reduction_effect = 0
    reduction_vars = config.get('decision_vars', {}).get('reduction_rates', {})
    for var_name in reduction_vars.keys():
        if var_name in variables:
            reduction_effect += variables[var_name] * 0.1
    
    # 양극재 효과
    cathode_config = config.get('decision_vars', {}).get('cathode', {})
    recycle_impact = stable_var_data.get('recycle_impact', {}).get('재활용재', {})
    
    if cathode_config.get('type') == 'A':
        low_carbon_emission = variables.get('low_carbon_emission', 10)
        low_carbon_ratio = cathode_config.get('low_carbon_ratio_fixed', 0.1)
        recycle_ratio = cathode_config.get('recycle_ratio_fixed', 0.2)
    else:
        low_carbon_emission = cathode_config.get('emission_fixed', 10)
        low_carbon_ratio = variables.get('low_carbon_ratio', 0.1)
        recycle_ratio = variables.get('recycle_ratio', 0.2)
    
    cathode_effect = low_carbon_emission * low_carbon_ratio
    recycle_effect = recycle_ratio * (1 - recycle_impact.get('Ni', 0.1))
    
    # 총 배출량
    total_emission = (base_emission * location_factor) - reduction_effect - cathode_effect - recycle_effect
    
    return total_emission


def export_results_to_json(
    results: Dict[str, Any],
    config: Dict[str, Any],
    filename: Optional[str] = None
) -> str:
    """
    최적화 결과를 JSON 파일로 저장
    
    Args:
        results: 최적화 결과
        config: 최적화 설정
        filename: 저장할 파일명 (없으면 자동 생성)
        
    Returns:
        str: 저장된 파일 경로
    """
    from datetime import datetime
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.json"
    
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': config,
        'results': results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    return filename


def compare_solver_results(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    여러 솔버의 결과를 비교하는 테이블 생성
    
    Args:
        results_dict: 솔버 이름을 키로 하는 결과 딕셔너리
        
    Returns:
        pd.DataFrame: 비교 테이블
    """
    comparison_data = []
    
    for solver_name, results in results_dict.items():
        if results.get('status') == 'optimal':
            comparison_data.append({
                '솔버': solver_name,
                '상태': results['status'],
                '목적함수값': f"{results.get('objective_value', 0):.4f}",
                '해결 시간': f"{results.get('solver_time', 0):.2f}s" if results.get('solver_time') else 'N/A',
                '종료 조건': results.get('termination_condition', 'N/A')
            })
        else:
            comparison_data.append({
                '솔버': solver_name,
                '상태': results.get('status', 'error'),
                '목적함수값': 'N/A',
                '해결 시간': 'N/A',
                '종료 조건': results.get('message', 'N/A')
            })
    
    return pd.DataFrame(comparison_data)
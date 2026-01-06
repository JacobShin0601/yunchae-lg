import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import yaml
import json
from pathlib import Path
from datetime import datetime
import os
import sys
from typing import Dict, Any, Optional, List, Tuple

# src 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 최적화 모듈 임포트
from src.optimization import (
    # 기본 클래스 및 함수
    create_optimizer,
    run_optimization,
    run_multi_solver_optimization,
    select_optimal_solver,
    format_optimization_results,
    export_results_to_json,
    compare_solver_results,
    load_stable_var_data,
    
    # 시나리오 관련
    create_scenario,
    get_available_scenario_types,
    SCENARIO_INFO,
    
    # 특정 시나리오 실행 함수
    run_carbon_minimization,
    run_cost_minimization,
    run_multi_objective,
    run_implementation_ease,
    run_regional_optimization,
    
    # 컨트롤러
    get_controller,
    
    # 솔버 관련
    get_available_solvers,
    
    # 자재 기반 최적화 및 비용 계산
    MaterialPremiumCostCalculator,
    ReductionConstraintManager
)

# 시뮬레이션 정렬 최적화 모듈 임포트
try:
    from src.optimization.input import OptimizationInput
    from src.optimization.carbon_minimization import CarbonMinimization
    from src.optimization.simulation_aligned_carbon_objective import SimulationAlignedCarbonObjective
    SIMULATION_ALIGNED_AVAILABLE = True
except ImportError as e:
    print(f"시뮬레이션 정렬 최적화 모듈을 로드할 수 없습니다: {e}")
    SIMULATION_ALIGNED_AVAILABLE = False

# RuleBasedSim 임포트 (시뮬레이션 데이터 생성용)
try:
    from src.rule_based import RuleBasedSim
    RULE_BASED_AVAILABLE = True
except ImportError as e:
    print(f"RuleBasedSim 모듈을 로드할 수 없습니다: {e}")
    RULE_BASED_AVAILABLE = False
from src.utils.file_operations import FileOperations, FileLoadError, FileSaveError
from src.utils.logging_migration import log_button_click, log_input_change, log_info, log_error, log_warning
from src.utils.styles import get_page_styles

# 최적화 설정 파일 경로 설정
def get_config_yaml_path(user_id: Optional[str] = None) -> str:
    """최적화 설정 YAML 파일 경로를 반환합니다."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    
    if user_id:
        # 사용자별 설정 파일 경로
        config_dir = os.path.join(project_root, "input", user_id)
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "config_opt.yaml")
    else:
        # 기본 설정 파일 경로
        return os.path.join(project_root, "input", "config_opt.yaml")

# 최적화 설정 저장
def save_optimization_config(config_data: dict, user_id: Optional[str] = None) -> bool:
    """최적화 설정을 YAML 파일로 저장합니다."""
    try:
        config_path = get_config_yaml_path(user_id)
        
        # YAML 파일로 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        log_info(f"최적화 설정 저장됨: {config_path}")
        return True
    except Exception as e:
        log_error(f"최적화 설정 저장 실패: {e}")
        return False

# 최적화 설정 로드
def load_optimization_config(user_id: Optional[str] = None) -> dict:
    """최적화 설정을 YAML 파일에서 로드합니다."""
    try:
        config_path = get_config_yaml_path(user_id)
        
        if os.path.exists(config_path):
            # YAML 파일에서 로드
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            log_info(f"최적화 설정 로드됨: {config_path}")
            return config_data or {}
        else:
            log_info(f"최적화 설정 파일 없음: {config_path}")
            return {}
    except Exception as e:
        log_error(f"최적화 설정 로드 실패: {e}")
        return {}

# 기본 최적화 설정
def get_default_optimization_config() -> dict:
    """기본 최적화 설정을 반환합니다."""
    return {
        'scenario_type': 'carbon_minimization',
        'solver': 'glpk',
        'scenario_params': {},
        'advanced_options': {
            'compare_solvers': False,
            'time_limit': 300,  # 5분
            'gap_tolerance': 0.01  # 1%
        },
        'results': None,
        'last_run': None
    }

# 최적화 페이지
def get_dynamic_default_parameters(num_tier=2):
    """num_tier에 따른 동적 기본 파라미터 생성"""
    
    # 기본 tier별 RE 적용률
    tier_rates = {}
    tier_ranges = {}
    
    for tier in range(1, num_tier + 1):
        # tier별 기본 RE 적용률 (tier가 높을수록 더 높게)
        default_rate = min(0.3 + (tier - 1) * 0.2, 0.9)
        tier_rates[f'tier{tier}_re_rate'] = default_rate
        tier_ranges[f'tier{tier}'] = {'min': 0.1, 'max': 0.9}
    
    return {
        'carbon_minimization': {
            **tier_rates,
            'target_carbon': 12.0,  # 목표 탄소발자국 (kgCO2eq)
            'use_simulation_data': True
        },
        'cost_minimization': {
            'target_carbon': 45.0,
            **tier_rates,
            'max_cost': 50000.0,
            'use_simulation_data': True,
            'simulation_scenario': 'baseline'
        },
        'multi_objective': {
            **tier_rates,
            'target_carbon': 12.0,  # 목표 탄소발자국 (kgCO2eq)
            'max_cost': 50000.0,
            'use_simulation_data': True,
            'simulation_scenario': 'baseline'
        },
        'material_based': {
            'reduction_target': {
                'min': 5,    # 최소 감축률 (%)
                'max': 15,   # 최대 감축률 (%)
            },
            're_rates': tier_ranges,
            'material_ratios': {
                'recycle': {'min': 0.05, 'max': 0.5},      # 재활용 비율 범위
                'low_carbon': {'min': 0.05, 'max': 0.3},  # 저탄소메탈 비율 범위
            },
            'constraints': {
                'apply_formula_first': True,  # Formula 로직 우선 적용
            },
            'use_simulation_data': True,
            'simulation_scenario': 'baseline'
        }
    }

def _display_material_target_validation(material_name, material_results, debug_container):
    """
    자재별 감축 목표 준수 여부를 UI에 표시
    
    Args:
        material_name: 자재명
        material_results: MaterialBasedOptimizer 자재 결과
        debug_container: 디버그 표시용 Streamlit 컨테이너
    """
    try:
        if not material_results or material_results.get('status') != 'optimal':
            return
            
        # 각 자재별 결과에서 material_target_compliance 확인
        materials_data = material_results.get('materials', {})
        
        for mat_name, mat_result in materials_data.items():
            if mat_name != material_name or mat_result.get('status') not in ['optimal', 'suboptimal', 'exceeding_target']:
                continue
            
            target_compliance = mat_result.get('material_target_compliance', {})
            if not target_compliance:
                continue
            
            target_used = target_compliance.get('target_used', 'unknown')
            meets_min = target_compliance.get('meets_min_target', True)
            within_max = target_compliance.get('within_max_target', True)
            actual_reduction = target_compliance.get('actual_reduction_pct', 0)
            min_target = target_compliance.get('min_target_pct', 0)
            max_target = target_compliance.get('max_target_pct', 0)
            compliance_status = target_compliance.get('compliance_status', 'unknown')
            
            # UI 표시
            if meets_min and within_max:
                st.success(f"✅ **{material_name}**: 자재별 감축 목표 달성")
                st.caption(f"🎯 목표: {min_target:.1f}%-{max_target:.1f}% | 달성: {actual_reduction:.1f}% ({target_used} 설정 사용)")
            elif not meets_min:
                st.warning(f"⚠️ **{material_name}**: 최소 감축 목표 미달")
                st.caption(f"🎯 목표: {min_target:.1f}%-{max_target:.1f}% | 달성: {actual_reduction:.1f}% (부족: {min_target - actual_reduction:.1f}%)")
            elif not within_max:
                st.info(f"📈 **{material_name}**: 최대 감축 목표 초과 (양호)")
                st.caption(f"🎯 목표: {min_target:.1f}%-{max_target:.1f}% | 달성: {actual_reduction:.1f}% (초과: {actual_reduction - max_target:.1f}%)")
            
            # 디버그 정보 표시
            if debug_container:
                with debug_container:
                    st.write(f"**자재별 감축 목표 상세 ({material_name})**")
                    st.json({
                        'target_type': target_used,
                        'target_range': f"{min_target:.1f}%-{max_target:.1f}%",
                        'actual_reduction': f"{actual_reduction:.1f}%",
                        'compliance_status': compliance_status,
                        'meets_minimum': meets_min,
                        'within_maximum': within_max
                    })
            break
                    
    except Exception as e:
        st.error(f"자재별 목표 검증 표시 중 오류: {e}")

def _display_constraint_validation(material_name, material_results, debug_container):
    """
    제약조건 준수 여부를 UI에 표시
    
    Args:
        material_name: 자재명
        material_results: MaterialBasedOptimizer 자재 결과
        debug_container: Streamlit 디버그 컨테이너
    """
    try:
        # 자재 결과에서 제약조건 검증 정보 추출
        for result_key, result_data in material_results.items():
            if result_data.get('status') == 'optimal' and 'constraint_validation' in result_data:
                validation = result_data['constraint_validation']
                re_config = result_data.get('re_config_summary', {})
                
                if validation.get('all_constraints_satisfied', True):
                    # 모든 제약조건 준수
                    with debug_container:
                        st.info(f"✅ **{material_name}**: 모든 UI 제약조건 준수")
                        
                        # 상세 정보 표시
                        constraint_details = []
                        for tier, details in validation.get('compliance_details', {}).items():
                            tier_re = details['actual_value']
                            tier_min = details['min_constraint']
                            tier_max = details['max_constraint']
                            constraint_details.append(f"**{tier.upper()}**: {tier_re:.3f} (범위: [{tier_min:.3f}, {tier_max:.3f}])")
                        
                        if constraint_details:
                            st.write("📊 **적용된 RE 비율**:")
                            for detail in constraint_details:
                                st.write(f"  • {detail}")
                else:
                    # 제약조건 위반
                    with debug_container:
                        st.warning(f"⚠️ **{material_name}**: 제약조건 위반 감지")
                        
                        for violation in validation.get('violations', []):
                            tier = violation['tier']
                            actual = violation['actual_value']
                            constraint_range = violation['constraint_range']
                            violation_type = violation['violation_type']
                            
                            if violation_type == 'below_min':
                                st.error(f"❌ **{tier.upper()}**: {actual:.3f} < {constraint_range[0]:.3f} (최소값 미달)")
                            else:
                                st.error(f"❌ **{tier.upper()}**: {actual:.3f} > {constraint_range[1]:.3f} (최대값 초과)")
                break
                
    except Exception as e:
        with debug_container:
            st.warning(f"⚠️ 제약조건 검증 표시 중 오류: {e}")


def optimization_page():
    """PCF 최적화 페이지"""
    
    # 현재 사용자의 num_tier 설정 확인
    user_id = st.session_state.get('user_id', None)
    from app_helper import load_simulation_config
    config = load_simulation_config(user_id=user_id)
    num_tier = config.get('num_tier', 2) if config else 2  # 기본값 2
    
    print(f"🔧 DEBUG - optimization_page에서 num_tier: {num_tier}")
    
    # 페이지 스타일 적용
    st.markdown(get_page_styles('optimization'), unsafe_allow_html=True)
    
    # 개선된 페이지 스타일
    st.markdown("""
    <style>
    .scenario-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: 2px solid transparent;
        border-radius: 12px;
        padding: 24px;
        margin: 15px 0;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .scenario-card:hover {
        transform: translateY(-2px);
        border-color: var(--primary-color);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .scenario-card.selected {
        border-color: #2E8B57;
        background: linear-gradient(135deg, #e8f5e8 0%, #c3e9c3 100%);
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.3);
    }
    .scenario-icon {
        font-size: 2rem;
        margin-bottom: 10px;
        display: block;
    }
    .scenario-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 12px;
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 8px;
    }
    .scenario-description {
        color: #5a6c7d;
        font-size: 0.95rem;
        margin-bottom: 15px;
        line-height: 1.5;
    }
    .scenario-features {
        list-style: none;
        padding-left: 0;
        margin-bottom: 15px;
    }
    .scenario-features li {
        margin-bottom: 5px;
        color: #34495e;
        font-size: 0.9rem;
    }
    .scenario-features li:before {
        content: "✓ ";
        color: #27ae60;
        font-weight: bold;
    }
    .config-section {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .config-title {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .config-title:before {
        content: "⚙️ ";
        margin-right: 8px;
    }
    .result-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 5px solid var(--primary-color);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .solver-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    .solver-card:hover {
        transform: translateY(-2px);
        border-color: var(--primary-color);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .solver-card.selected {
        border-color: #2E8B57;
        background: linear-gradient(145deg, #e8f5e8 0%, #d4ecd4 100%);
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.3);
    }
    .progress-indicator {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 10px 0;
        text-align: center;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #856404;
    }
    .success-box {
        background: linear-gradient(135deg, #d1f2eb 0%, #a7f3d0 100%);
        border: 1px solid #27ae60;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #155724;
    }
    .step-indicator {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .step {
        flex: 1;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin: 0 5px;
        transition: all 0.3s ease;
    }
    .step.active {
        background: rgba(255,255,255,0.2);
        font-weight: bold;
    }
    .step.completed {
        background: rgba(46, 139, 87, 0.8);
    }
    .data-summary {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="page-title">
        <h1>🚀 PCF 최적화 엔진</h1>
        <p>첨단 알고리즘과 시뮬레이션 기반 탄소 발자국 최적화 플랫폼</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사용자 ID 가져오기
    user_id = st.session_state.get('user_id', None)
    
    # 최적화 컨트롤러 초기화
    if 'optimization_controller' not in st.session_state:
        st.session_state.optimization_controller = get_controller()
    controller = st.session_state.optimization_controller
    
    # 최적화 설정 상태 초기화
    if 'optimization_config' not in st.session_state:
        # 저장된 설정 로드 시도
        loaded_config = load_optimization_config(user_id=user_id)
        if loaded_config:
            st.session_state.optimization_config = loaded_config
            log_info(f"사용자별 최적화 설정 로드됨 (user_id: {user_id})")
        else:
            # 기본 설정 사용
            st.session_state.optimization_config = get_default_optimization_config()
            log_info("기본 최적화 설정 사용")
    
    # 진행 단계 표시
    current_step = 1
    if st.session_state.optimization_config.get('scenario_type'):
        current_step = 2
    if st.session_state.optimization_config.get('results'):
        current_step = 3
    
    st.markdown(f"""
    <div class="step-indicator">
        <div class="step {'completed' if current_step > 1 else 'active' if current_step == 1 else ''}">
            <span>1️⃣ 시나리오 선택</span>
        </div>
        <div class="step {'completed' if current_step > 2 else 'active' if current_step == 2 else ''}">
            <span>2️⃣ 설정 구성</span>
        </div>
        <div class="step {'completed' if current_step > 3 else 'active' if current_step == 3 else ''}">
            <span>3️⃣ 결과 확인</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 탭 생성 - 시나리오 선택 / 설정 구성 / 결과 시각화
    tab1, tab2, tab3 = st.tabs(["시나리오 선택", "설정 구성", "결과 시각화"])
    
    with tab1:
        st.markdown('<div class="section-title">🎯 시나리오 선택</div>', unsafe_allow_html=True)
        
        # 사용 가능한 시나리오 목록
        available_scenarios = get_available_scenario_types()
        scenario_info = SCENARIO_INFO
        
        # 현재 선택된 시나리오
        current_scenario = st.session_state.optimization_config.get('scenario_type', 'carbon_minimization')
        
        # 시나리오 선택 도움말
        st.markdown("""
        <div class="success-box">
            <strong>💡 시나리오 선택 가이드</strong><br>
            • <strong>탄소발자국 최소화</strong>: 환경 친화적 목표 중심<br>
            • <strong>비용 최소화</strong>: 경제성 우선 고려<br>
            • <strong>다목적 최적화</strong>: 환경과 경제의 균형<br>
            • <strong>자재 기반 최적화</strong>: 자재별 맞춤형 접근<br>
            • <strong>시뮬레이션 정렬</strong>: 실제 데이터 기반 정밀 계산
        </div>
        """, unsafe_allow_html=True)
        
        # 시나리오 카드 레이아웃
        st.markdown("### 🔍 최적화 시나리오 선택")
        st.markdown("아래 시나리오 중 하나를 선택하여 최적화 설정을 구성하세요.")
        
        # 2열 레이아웃으로 시나리오 카드 표시
        col1, col2, col3 = st.columns(3)
        
        # 첫번째 열 - 탄소 최소화 & 비용 최소화
        with col1:
            # 탄소발자국 최소화 시나리오
            selected_class = "selected" if current_scenario == "carbon_minimization" else ""
            st.markdown(f"""
            <div class="scenario-card {selected_class}">
                <span class="scenario-icon">🌿</span>
                <div class="scenario-title">탄소발자국 최소화</div>
                <div class="scenario-description">
                    양극재 생산과정에서 발생하는 탄소배출량을 최소화하기 위한 최적 구성을 찾습니다.
                    환경 친화적 목표를 우선시하는 기업에게 적합합니다.
                </div>
                <ul class="scenario-features">
                    <li>시뮬레이션과 동일한 자재별 배출계수 계산</li>
                    <li>Tier별 RE 적용률 최적화</li>
                    <li>Formula/Proportions 방식 구분 적용</li>
                    <li>정밀한 탄소발자국 감축 계산</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button("🎯 탄소발자국 최소화 선택", 
                      key="select_carbon_min",
                      type="primary" if current_scenario == "carbon_minimization" else "secondary",
                      use_container_width=True):
                st.session_state.optimization_config['scenario_type'] = "carbon_minimization"
                st.success("✅ 탄소발자국 최소화 시나리오가 선택되었습니다!")
                # 시나리오 변경시 결과 초기화
                if 'results' in st.session_state.optimization_config:
                    st.session_state.optimization_config['results'] = None
                st.rerun()
            
            st.markdown("---")
            
            # 다목적 최적화 시나리오
            selected_class = "selected" if current_scenario == "multi_objective" else ""
            st.markdown(f"""
            <div class="scenario-card {selected_class}">
                <span class="scenario-icon">⚖️</span>
                <div class="scenario-title">다목적 최적화</div>
                <div class="scenario-description">
                    탄소발자국과 비용을 모두 고려한 균형 잡힌 최적화를 수행합니다.
                    경제성과 환경성을 동시에 추구하는 기업에게 적합합니다.
                </div>
                <ul class="scenario-features">
                    <li>탄소발자국과 비용의 가중 합 최소화</li>
                    <li>사용자 정의 가중치로 우선순위 조정</li>
                    <li>파레토 효율적 솔루션 탐색</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button("⚖️ 다목적 최적화 선택", 
                      key="select_multi_obj",
                      type="primary" if current_scenario == "multi_objective" else "secondary",
                      use_container_width=True):
                st.session_state.optimization_config['scenario_type'] = "multi_objective"
                st.success("✅ 다목적 최적화 시나리오가 선택되었습니다!")
                # 시나리오 변경시 결과 초기화
                if 'results' in st.session_state.optimization_config:
                    st.session_state.optimization_config['results'] = None
                st.rerun()
        
        # 두번째 열 - 구현 용이성 & 비용 최소화 & 지역별 최적화 & 시뮬레이션 정렬
        with col2:
            # 시뮬레이션 정렬 탄소발자국 최적화 시나리오
            if SIMULATION_ALIGNED_AVAILABLE:
                selected_class = "selected" if current_scenario == "simulation_aligned_carbon" else ""
                st.markdown(f"""
                <div class="scenario-card {selected_class}">
                    <span class="scenario-icon">🎯</span>
                    <div class="scenario-title">시뮬레이션 정렬 최적화</div>
                    <div class="scenario-description">
                        실제 시뮬레이션과 동일한 계산 로직을 사용하여 정확한 탄소발자국 최적화를 수행합니다.
                        정밀도가 중요한 연구 및 분석 용도에 적합합니다.
                    </div>
                    <ul class="scenario-features">
                        <li>rule_based.py와 동일한 계산 로직 사용</li>
                        <li>자재별 정밀한 배출계수 수정 계산</li>
                        <li>Formula/Proportions 방식 구분 적용</li>
                        <li>Tier별 RE 적용률 최적화</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # 선택 버튼
                if st.button("🎯 시뮬레이션 정렬 최적화 선택", 
                          key="select_sim_aligned",
                          type="primary" if current_scenario == "simulation_aligned_carbon" else "secondary",
                          use_container_width=True):
                    st.session_state.optimization_config['scenario_type'] = "simulation_aligned_carbon"
                    st.success("✅ 시뮬레이션 정렬 최적화 시나리오가 선택되었습니다!")
                    # 시나리오 변경시 결과 초기화
                    if 'results' in st.session_state.optimization_config:
                        st.session_state.optimization_config['results'] = None
                    st.rerun()
                
                st.markdown("---")
            
            # 비용 최소화 시나리오
            selected_class = "selected" if current_scenario == "cost_minimization" else ""
            st.markdown(f"""
            <div class="scenario-card {selected_class}">
                <span class="scenario-icon">💰</span>
                <div class="scenario-title">비용 최소화</div>
                <div class="scenario-description">
                    목표 탄소발자국을 달성하면서 최소 비용으로 구현할 수 있는 최적 구성을 찾습니다.
                    예산 제약이 있는 프로젝트에 적합합니다.
                </div>
                <ul class="scenario-features">
                    <li>구현 비용 최소화 목적함수</li>
                    <li>탄소발자국 상한선 제약조건</li>
                    <li>비용 효율적인 감축 활동 식별</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button("💰 비용 최소화 선택", 
                      key="select_cost_min",
                      type="primary" if current_scenario == "cost_minimization" else "secondary",
                      use_container_width=True):
                st.session_state.optimization_config['scenario_type'] = "cost_minimization"
                st.success("✅ 비용 최소화 시나리오가 선택되었습니다!")
                # 시나리오 변경시 결과 초기화
                if 'results' in st.session_state.optimization_config:
                    st.session_state.optimization_config['results'] = None
                st.rerun()
            
            st.markdown("---")
            
            # 구현 용이성 최적화 시나리오
            selected_class = "selected" if current_scenario == "implementation_ease" else ""
            st.markdown(f"""
            <div class="scenario-card {selected_class}">
                <span class="scenario-icon">🛠️</span>
                <div class="scenario-title">구현 용이성 최적화</div>
                <div class="scenario-description">
                    목표 탄소발자국을 달성하면서 최소한의 변화로 구현할 수 있는 방안을 찾습니다.
                    기존 운영 체계를 크게 바꾸고 싶지 않은 경우에 적합합니다.
                </div>
                <ul class="scenario-features">
                    <li>변화해야 하는 활동 수 최소화</li>
                    <li>탄소발자국 목표 제약 조건</li>
                    <li>최소한의 변화로 최대 효과</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button("🛠️ 구현 용이성 최적화 선택", 
                      key="select_ease",
                      type="primary" if current_scenario == "implementation_ease" else "secondary",
                      use_container_width=True):
                st.session_state.optimization_config['scenario_type'] = "implementation_ease"
                st.success("✅ 구현 용이성 최적화 시나리오가 선택되었습니다!")
                # 시나리오 변경시 결과 초기화
                if 'results' in st.session_state.optimization_config:
                    st.session_state.optimization_config['results'] = None
                st.rerun()
        
        # 세번째 열 - 자재 기반 최적화 & 파레토 최적화
        with col3:
            
            # 지역별 최적화 시나리오
            selected_class = "selected" if current_scenario == "regional_optimization" else ""
            st.markdown(f"""
            <div class="scenario-card {selected_class}">
                <span class="scenario-icon">🌎</span>
                <div class="scenario-title">지역별 최적화</div>
                <div class="scenario-description">
                    생산 지역 변경에 따른 탄소발자국 감소 효과를 최대화하는 최적 위치를 찾습니다.
                    글로벌 공급망 재구성을 고려하는 기업에게 적합합니다.
                </div>
                <ul class="scenario-features">
                    <li>지역별 전력 배출계수 기반 최적화</li>
                    <li>물류 비용과 탄소발자국 균형 고려</li>
                    <li>지역별 제약조건 반영</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button("🌎 지역별 최적화 선택", 
                      key="select_regional",
                      type="primary" if current_scenario == "regional_optimization" else "secondary",
                      use_container_width=True):
                st.session_state.optimization_config['scenario_type'] = "regional_optimization"
                st.success("✅ 지역별 최적화 시나리오가 선택되었습니다!")
                # 시나리오 변경시 결과 초기화
                if 'results' in st.session_state.optimization_config:
                    st.session_state.optimization_config['results'] = None
                st.rerun()
            
            st.markdown("---")
            
            # 자재 기반 최적화 시나리오
            selected_class = "selected" if current_scenario == "material_based" else ""
            st.markdown(f"""
            <div class="scenario-card {selected_class}">
                <span class="scenario-icon">🧩</span>
                <div class="scenario-title">자재 기반 최적화</div>
                <div class="scenario-description">
                    자재별 특성을 고려한 맞춤형 최적화로 정밀한 탄소 배출량 감축 방안을 도출합니다.
                    자재 중심의 세밀한 분석이 필요한 경우에 적합합니다.
                </div>
                <ul class="scenario-features">
                    <li>자재 유형 자동 감지 및 분류</li>
                    <li>자재별 최적 저감 활동 도출</li>
                    <li>Formula/Proportions 방식 자동 구분 적용</li>
                    <li>그리드서치를 통한 파라미터 탐색</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button("🧩 자재 기반 최적화 선택", 
                      key="select_material_based",
                      type="primary" if current_scenario == "material_based" else "secondary",
                      use_container_width=True):
                st.session_state.optimization_config['scenario_type'] = "material_based"
                st.success("✅ 자재 기반 최적화 시나리오가 선택되었습니다!")
                # 시나리오 변경시 결과 초기화
                if 'results' in st.session_state.optimization_config:
                    st.session_state.optimization_config['results'] = None
                st.rerun()
            
            st.markdown("---")
                
            # 파레토 최적화 시나리오
            selected_class = "selected" if current_scenario == "pareto_optimization" else ""
            st.markdown(f"""
            <div class="scenario-card {selected_class}">
                <span class="scenario-icon">📊</span>
                <div class="scenario-title">파레토 최적화</div>
                <div class="scenario-description">
                    그리드서치를 통해 다양한 파라미터 조합을 탐색하고 파레토 최적해를 찾습니다.
                    여러 솔루션을 비교 검토하고 싶은 경우에 적합합니다.
                </div>
                <ul class="scenario-features">
                    <li>다수의 파라미터 조합 탐색</li>
                    <li>PCF와 비용 사이의 최적 균형점 발견</li>
                    <li>시각화 기반 의사결정 지원</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 선택 버튼
            if st.button("📊 파레토 최적화 선택", 
                      key="select_pareto",
                      type="primary" if current_scenario == "pareto_optimization" else "secondary",
                      use_container_width=True):
                st.session_state.optimization_config['scenario_type'] = "pareto_optimization"
                st.success("✅ 파레토 최적화 시나리오가 선택되었습니다!")
                # 시나리오 변경시 결과 초기화
                if 'results' in st.session_state.optimization_config:
                    st.session_state.optimization_config['results'] = None
                st.rerun()
        
        # 시나리오 선택 후 다음 단계 안내
        st.markdown("---")
        
        if current_scenario:
            scenario_title = SCENARIO_INFO.get(current_scenario, {}).get('title', current_scenario)
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ 선택된 시나리오: {scenario_title}</strong><br>
                이제 '설정 구성' 탭에서 세부 파라미터를 조정하고 최적화를 실행하세요.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ 시나리오를 선택해주세요</strong><br>
                위의 시나리오 카드 중 하나를 선택하여 최적화를 시작하세요.
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-title">⚙️ 설정 구성</div>', unsafe_allow_html=True)
        
        # 현재 선택된 시나리오
        current_scenario = st.session_state.optimization_config.get('scenario_type', 'carbon_minimization')
        
        # 시나리오 미선택 시 안내
        if not current_scenario:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ 시나리오를 먼저 선택해주세요</strong><br>
                '시나리오 선택' 탭에서 원하는 최적화 시나리오를 선택한 후 설정을 구성하세요.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # 선택된 시나리오 표시
        scenario_title = SCENARIO_INFO.get(current_scenario, {}).get('title', current_scenario)
        st.markdown(f"""
        <div class="data-summary">
            <h3>🎯 현재 선택된 시나리오: {scenario_title}</h3>
            <p>아래 설정들을 조정하여 최적화 조건을 세밀하게 구성할 수 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 시나리오별 설명 및 파라미터
        scenario_descriptions = {
            'carbon_minimization': "시뮬레이션과 동일한 계산 로직을 사용하여 탄소발자국을 최소화합니다. 자재별 배출계수 수정을 통한 정밀한 최적화를 수행합니다.",
            'cost_minimization': "목표 탄소발자국을 달성하면서 구현 비용을 최소화합니다. 시뮬레이션 데이터 사용 여부를 선택할 수 있습니다.",
            'multi_objective': "탄소발자국과 비용을 모두 고려한 균형 잡힌 최적화를 수행합니다.",
            'implementation_ease': "목표 탄소발자국을 달성하면서 변화해야 할 활동의 수를 최소화합니다.",
            'regional_optimization': "생산 지역 변경에 따른 탄소발자국 감소 효과를 최대화합니다.",
            'simulation_aligned_carbon': "실제 시뮬레이션과 동일한 계산 로직을 사용하여 정확한 탄소발자국 최적화를 수행합니다. 자재별 배출계수 수정 과정을 정밀하게 모델링합니다.",
            'material_based': "자재별 특성을 고려한 맞춤형 최적화로 정밀한 탄소 배출량 감축 방안을 도출합니다. 자재 유형에 따라 최적화 접근 방식을 자동으로 다르게 적용합니다.",
            'pareto_optimization': "그리드서치를 통해 다양한 파라미터 조합을 탐색하고 PCF와 비용 사이의 파레토 최적해를 찾습니다. 시각화 기반 의사결정을 지원합니다."
        }
        
        # 시나리오별 기본 파라미터 (동적 생성)
        default_parameters = get_dynamic_default_parameters(num_tier)
        
        
        # 시나리오 파라미터가 없으면 기본값 설정
        if 'scenario_params' not in st.session_state.optimization_config:
            st.session_state.optimization_config['scenario_params'] = {}
        
        # 현재 시나리오에 대한 파라미터가 없으면 기본값 설정
        if current_scenario not in st.session_state.optimization_config['scenario_params']:
            st.session_state.optimization_config['scenario_params'][current_scenario] = default_parameters.get(current_scenario, {})
        
        # 현재 파라미터 값
        current_params = st.session_state.optimization_config['scenario_params'].get(current_scenario, {})
        
        # 시나리오 제목 및 설명
        st.markdown(f"## {SCENARIO_INFO.get(current_scenario, {}).get('title', current_scenario)}")
        st.markdown(f"*{scenario_descriptions.get(current_scenario, '')}*")
        
        # 시나리오별 파라미터 UI
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-title">시나리오 파라미터</div>', unsafe_allow_html=True)
        
        if current_scenario == 'carbon_minimization':
            # 탄소발자국 최소화 파라미터 - 시뮬레이션 정렬 버전
            st.markdown("### 시뮬레이션 데이터 로드")
            
            # 시뮬레이션 데이터 로드 방법 선택
            data_source = st.radio(
                "시뮬레이션 데이터 소스",
                options=["현재 세션 데이터 사용", "샘플 데이터 사용"],
                index=0,  # 기본값: 현재 세션 데이터
                help="최적화에 사용할 시뮬레이션 데이터 소스를 선택하세요",
                key=f"data_source_carbon_{current_scenario}"
            )
            
            # 데이터 로드 상태 표시
            if 'carbon_simulation_data_loaded' not in st.session_state:
                st.session_state.carbon_simulation_data_loaded = False
                st.session_state.carbon_simulation_data = None
            
            if data_source == "샘플 데이터 사용":
                if st.button("📊 샘플 데이터 로드", key="load_sample_data_carbon"):
                    try:
                        with st.spinner("샘플 데이터 생성 중..."):
                            sample_data = generate_sample_simulation_data()
                            
                            # 데이터 유효성 검사
                            if not sample_data or not all(key in sample_data for key in ['scenario_df', 'ref_formula_df', 'ref_proportions_df', 'original_df']):
                                st.error("❌ 샘플 데이터 생성에 실패했습니다. 필수 데이터프레임이 누락되었습니다.")
                            elif any(df.empty for df in sample_data.values() if hasattr(df, 'empty')):
                                st.warning("⚠️ 일부 데이터프레임이 비어있습니다. 최적화 결과에 영향을 줄 수 있습니다.")
                                st.session_state.carbon_simulation_data = sample_data
                                st.session_state.carbon_simulation_data_loaded = True
                                st.success("✅ 샘플 시뮬레이션 데이터가 로드되었습니다!")
                            else:
                                st.session_state.carbon_simulation_data = sample_data
                                st.session_state.carbon_simulation_data_loaded = True
                                st.success("✅ 샘플 시뮬레이션 데이터가 로드되었습니다!")
                                
                                # 데이터 품질 정보 표시
                                total_materials = len(sample_data['scenario_df'])
                                st.info(f"📊 로드된 데이터: {total_materials}개 자재, 완전한 데이터셋")
                    except Exception as e:
                        st.error(f"❌ 샘플 데이터 로드 중 오류 발생: {str(e)}")
                        st.markdown("**해결 방법**: 페이지를 새로고침하고 다시 시도해주세요.")
            
            elif data_source == "현재 세션 데이터 사용":
                # 디버깅: 세션 상태 정보 표시
                with st.expander("🔍 세션 디버깅 정보"):
                    st.write("**현재 세션 키들:**")
                    session_keys = list(st.session_state.keys())
                    st.write(session_keys)
                    
                    if 'simulation_results' in st.session_state:
                        sim_results = st.session_state.simulation_results
                        st.write(f"**simulation_results 타입:** {type(sim_results)}")
                        if isinstance(sim_results, dict):
                            st.write(f"**simulation_results 키들:** {list(sim_results.keys())}")
                            for key, value in sim_results.items():
                                st.write(f"**{key}:** {type(value)}")
                                if isinstance(value, dict):
                                    st.write(f"  - 하위 키들: {list(value.keys())}")
                        else:
                            st.write(f"**simulation_results 내용:** {sim_results}")
                    else:
                        st.write("**simulation_results 없음**")
                
                if check_session_simulation_data():
                    if st.button("📊 세션 데이터 로드", key="load_session_data_carbon"):
                        try:
                            with st.spinner("세션 데이터 로드 중..."):
                                session_data = get_session_simulation_data()
                                
                                # 데이터 유효성 검사
                                if not session_data:
                                    st.error("❌ 세션 데이터가 비어있습니다.")
                                elif not all(key in session_data for key in ['scenario_df', 'ref_formula_df', 'ref_proportions_df', 'original_df']):
                                    st.warning("⚠️ 일부 필수 데이터가 누락되었습니다. 최적화가 제한될 수 있습니다.")
                                    st.session_state.carbon_simulation_data = session_data
                                    st.session_state.carbon_simulation_data_loaded = True
                                    st.success("✅ 세션 시뮬레이션 데이터가 로드되었습니다! (일부 제한)")
                                else:
                                    # 데이터 품질 검사
                                    scenario_count = len(session_data['scenario_df']) if not session_data['scenario_df'].empty else 0
                                    if scenario_count == 0:
                                        st.warning("⚠️ 시나리오 데이터가 비어있습니다. 샘플 데이터 사용을 권장합니다.")
                                    
                                    st.session_state.carbon_simulation_data = session_data
                                    st.session_state.carbon_simulation_data_loaded = True
                                    st.success("✅ 세션 시뮬레이션 데이터가 로드되었습니다!")
                                    
                                    if scenario_count > 0:
                                        st.info(f"📊 로드된 데이터: {scenario_count}개 자재 시나리오")
                        except Exception as e:
                            st.error(f"❌ 세션 데이터 로드 중 오류 발생: {str(e)}")
                            st.markdown("""
                            **해결 방법**:
                            1. PCF 시뮬레이션 페이지에서 시뮬레이션을 다시 실행해주세요
                            2. 또는 '샘플 데이터 사용' 옵션을 선택해주세요
                            """)
                else:
                    st.warning("⚠️ **현재 세션에서 시뮬레이션 데이터를 찾을 수 없습니다**")
                    st.markdown("""
                    **가능한 원인**:
                    - PCF 시뮬레이션이 아직 실행되지 않음
                    - 세션이 만료되었거나 데이터가 초기화됨
                    - 다른 탭에서 데이터가 삭제됨
                    
                    **해결 방법**:
                    1. **PCF 시뮬레이션** 페이지로 이동
                    2. 원하는 시나리오로 시뮬레이션 실행
                    3. 이 페이지로 돌아와서 세션 데이터 로드 시도
                    4. 또는 아래 '샘플 데이터 사용' 선택
                    """)
            
            # 데이터 로드 상태에 따른 UI
            if st.session_state.carbon_simulation_data_loaded and st.session_state.carbon_simulation_data:
                sim_data = st.session_state.carbon_simulation_data
                
                # 데이터 요약 표시
                st.markdown("#### 로드된 시뮬레이션 데이터 요약")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    scenario_count = len(sim_data['scenario_df']) if 'scenario_df' in sim_data else 0
                    st.metric("시나리오 자재 수", scenario_count)
                
                with col2:
                    applicable_count = 0
                    if 'scenario_df' in sim_data and '저감활동_적용여부' in sim_data['scenario_df'].columns:
                        applicable_count = len(sim_data['scenario_df'][sim_data['scenario_df']['저감활동_적용여부'] == 1.0])
                    elif 'scenario_df' in sim_data:
                        # 저감활동_적용여부 컬럼이 없으면 전체 자재 수로 표시
                        applicable_count = len(sim_data['scenario_df'])
                    st.metric("저감활동 적용 자재", applicable_count)
                
                with col3:
                    formula_count = len(sim_data['ref_formula_df']) if 'ref_formula_df' in sim_data else 0
                    st.metric("Formula 참조 데이터", formula_count)
                
                with col4:
                    proportions_count = len(sim_data['ref_proportions_df']) if 'ref_proportions_df' in sim_data else 0
                    st.metric("Proportions 참조 데이터", proportions_count)
                
                # 기준 PCF 계산 - 시뮬레이션과 동일한 기준 사용
                if 'scenario_df' in sim_data:
                    scenario_df = sim_data['scenario_df']
                    # 1순위: PCF_reference 컬럼 사용 (시뮬레이션과 동일한 기준)
                    if 'PCF_reference' in scenario_df.columns:
                        baseline_pcf = scenario_df['PCF_reference'].sum()
                    # 2순위: 배출량(kgCO2eq) 컬럼 사용
                    elif '배출량(kgCO2eq)' in scenario_df.columns:
                        baseline_pcf = scenario_df['배출량(kgCO2eq)'].sum()
                    # 3순위: 배출계수 × 제품총소요량으로 계산
                    elif '배출계수' in scenario_df.columns and '제품총소요량(kg)' in scenario_df.columns:
                        baseline_pcf = (scenario_df['배출계수'] * scenario_df['제품총소요량(kg)']).sum()
                    else:
                        baseline_pcf = 0.0
                    st.metric("기준 PCF", f"{baseline_pcf:.4f} kgCO2eq")
                
                # RE 적용률 설정
                st.markdown("### RE 적용률 범위 설정")
                st.markdown("각 Tier별 재생에너지(RE) 적용률 범위를 설정하세요. (0: 미적용, 1: 100% 적용)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Tier1 RE 적용률 범위 설정
                    tier1_re_min, tier1_re_max = st.slider(
                        "Tier1 RE 적용률 범위",
                        min_value=0.0,
                        max_value=1.0,
                        value=(float(current_params.get('tier1_re_min', 0.2)), float(current_params.get('tier1_re_max', 0.4))),
                        step=0.05,
                        format="%.2f",
                        help="Tier1 재생에너지 적용 범위 (0: 미적용, 1: 100% 적용)",
                        key="tier1_carbon_range"
                    )
                    current_params['tier1_re_min'] = tier1_re_min
                    current_params['tier1_re_max'] = tier1_re_max
                    # 이전 단일 값 호환을 위해 중간값으로 저장
                    current_params['tier1_re_rate'] = (tier1_re_min + tier1_re_max) / 2
                
                with col2:
                    # Tier2 RE 적용률 범위 설정
                    tier2_re_min, tier2_re_max = st.slider(
                        "Tier2 RE 적용률 범위",
                        min_value=0.0,
                        max_value=1.0,
                        value=(float(current_params.get('tier2_re_min', 0.4)), float(current_params.get('tier2_re_max', 0.6))),
                        step=0.05,
                        format="%.2f",
                        help="Tier2 재생에너지 적용 범위 (0: 미적용, 1: 100% 적용)",
                        key="tier2_carbon_range"
                    )
                    current_params['tier2_re_min'] = tier2_re_min
                    current_params['tier2_re_max'] = tier2_re_max
                    # 이전 단일 값 호환을 위해 중간값으로 저장
                    current_params['tier2_re_rate'] = (tier2_re_min + tier2_re_max) / 2
                
                # 시뮬레이션 데이터에서 num_tier 확인
                num_tier = 3  # 기본값
                
                # 시뮬레이션 데이터가 로드되었는지 확인
                if st.session_state.get('carbon_simulation_data_loaded', False) and st.session_state.get('carbon_simulation_data'):
                    sim_data = st.session_state.carbon_simulation_data
                    if 'original_df' in sim_data and 'num_tier' in sim_data['original_df'].columns:
                        # 데이터프레임에서 num_tier 값 추출 (첫 번째 값 사용)
                        num_tier_values = sim_data['original_df']['num_tier'].unique()
                        if len(num_tier_values) > 0:
                            num_tier = int(num_tier_values[0])
                    elif 'num_tier' in sim_data:
                        # 직접 num_tier 키가 있는 경우
                        num_tier = int(sim_data['num_tier'])
                
                # Tier3는 num_tier가 3일 때만 표시
                if num_tier >= 3:
                    with col3:
                        # Tier3 RE 적용률 범위 설정
                        tier3_re_min, tier3_re_max = st.slider(
                            "Tier3 RE 적용률 범위",
                            min_value=0.0,
                            max_value=1.0,
                            value=(float(current_params.get('tier3_re_min', 0.6)), float(current_params.get('tier3_re_max', 0.8))),
                            step=0.05,
                            format="%.2f",
                            help="Tier3 재생에너지 적용 범위 (0: 미적용, 1: 100% 적용)",
                            key="tier3_carbon_range"
                        )
                        current_params['tier3_re_min'] = tier3_re_min
                        current_params['tier3_re_max'] = tier3_re_max
                        # 이전 단일 값 호환을 위해 중간값으로 저장
                        current_params['tier3_re_rate'] = (tier3_re_min + tier3_re_max) / 2
                
                # 제약조건 설정
                st.markdown("### 제약조건 설정")
                col1, col2 = st.columns(2)
                
                with col1:
                    # 탄소발자국 상한값
                    target_carbon = st.number_input(
                        "탄소발자국 상한값 (kgCO2eq)",
                        min_value=5.0,
                        max_value=50.0,
                        value=float(current_params.get('target_carbon', 12.0)),
                        step=0.5,
                        format="%.1f",
                        help="최소화 과정에서 허용되는 최대 탄소발자국 값. 더 낮은 값을 설정할수록 최적화가 더 엄격해집니다.",
                        key="target_carbon_min"
                    )
                    current_params['target_carbon'] = target_carbon
                
                with col2:
                    use_simulation_data = st.checkbox(
                        "시뮬레이션 데이터 사용",
                        value=current_params.get('use_simulation_data', True),
                        help="로드된 시뮬레이션 데이터를 최적화에 사용",
                        key="use_sim_data_carbon"
                    )
                    current_params['use_simulation_data'] = use_simulation_data
                
                # 양극재 설정 영역
                st.markdown("### 양극재 설정")
                st.markdown("양극재 타입과 관련 설정을 지정하세요.")
                
                # 양극재 타입 선택
                cathode_type = st.selectbox(
                    "양극재 타입",
                    options=["A", "B"],
                    index=0,
                    help="양극재 타입 (A: 비율 고정/배출계수 변경, B: 배출계수 고정/비율 변경)",
                    key="cathode_type"
                )
                
                # 양극재 설정
                if cathode_type == "A":
                    col1, col2 = st.columns(2)
                    with col1:
                        # 저탄소 원료 고정 비율
                        low_carbon_fixed = st.slider(
                            "저탄소 원료 고정 비율",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=0.05,
                            format="%.2f",
                            help="저탄소 원료의 고정 비율 (0-1)",
                            key="low_carbon_fixed"
                        )
                    
                    with col2:
                        # 재활용 원료 고정 비율
                        recycle_fixed = st.slider(
                            "재활용 원료 고정 비율",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.2,
                            step=0.05,
                            format="%.2f",
                            help="재활용 원료의 고정 비율 (0-1)",
                            key="recycle_fixed"
                        )
                    
                    # 배출계수 범위
                    emission_min, emission_max = st.slider(
                        "배출계수 범위",
                        min_value=1.0,
                        max_value=100.0,
                        value=(5.0, 15.0),
                        step=1.0,
                        format="%.1f",
                        help="Type A 양극재의 배출계수 범위",
                        key="emission_range"
                    )
                    
                    # 양극재 설정을 시나리오 파라미터에 저장
                    if 'cathode' not in current_params:
                        current_params['cathode'] = {}
                    current_params['cathode']['type'] = "A"
                    current_params['cathode']['type_A_config'] = {
                        "emission_range": [emission_min, emission_max],
                        "low_carbon_ratio_fixed": low_carbon_fixed,
                        "recycle_ratio_fixed": recycle_fixed
                    }
                elif cathode_type == "B":
                    col1, col2 = st.columns(2)
                    with col1:
                        # 저탄소 원료 범위
                        low_carbon_min, low_carbon_max = st.slider(
                            "저탄소 원료 비율 범위",
                            min_value=0.0,
                            max_value=1.0,
                            value=(0.05, 0.3),
                            step=0.05,
                            format="%.2f",
                            help="저탄소 원료의 비율 범위 (0-1)",
                            key="low_carbon_range"
                        )
                    
                    with col2:
                        # 재활용 원료 범위
                        recycle_min, recycle_max = st.slider(
                            "재활용 원료 비율 범위",
                            min_value=0.0,
                            max_value=1.0,
                            value=(0.1, 0.5),
                            step=0.05,
                            format="%.2f",
                            help="재활용 원료의 비율 범위 (0-1)",
                            key="recycle_range"
                        )
                    
                    # 고정 배출계수
                    emission_fixed = st.number_input(
                        "고정 배출계수",
                        min_value=1.0,
                        max_value=100.0,
                        value=10.0,
                        step=1.0,
                        format="%.1f",
                        help="Type B 양극재의 고정 배출계수",
                        key="emission_fixed"
                    )
                    
                    # 양극재 설정을 시나리오 파라미터에 저장
                    if 'cathode' not in current_params:
                        current_params['cathode'] = {}
                    current_params['cathode']['type'] = "B"
                    current_params['cathode']['type_B_config'] = {
                        "emission_fixed": emission_fixed,
                        "low_carbon_range": [low_carbon_min, low_carbon_max],
                        "recycle_range": [recycle_min, recycle_max]
                    }
                    
                
                # 설명 추가
                st.info("""
                **양극재 타입:**
                - Type A: 원료 비율 고정, 배출계수 최적화
                - Type B: 배출계수 고정, 원료 비율 최적화
                
                **주의**: 모든 비율의 합은 1을 초과할 수 없습니다.
                """)
                
                # 기본 제약조건 안내
                st.markdown("#### 기본 제약조건")
                st.info("• 탄소발자국 ≤ 탄소발자국 상한값")
                st.info("• RE 적용률 범위: 설정한 최소~최대 범위 내에서 최적화")
                st.info("• 시뮬레이션과 동일한 자재별 배출계수 계산")
                st.info("• Formula/Proportions 방식 구분 적용")
                
                # 추가 제약조건 설정
                st.markdown("### 추가 제약조건 설정")
                
                # Case 순서 제약조건
                with st.expander("Case 순서 제약조건 (Case1>Case2>Case3)", expanded=False):
                    case_enabled = st.checkbox(
                        "Case 순서 제약조건 활성화", 
                        value=current_params.get('case_constraints', {}).get('enabled', False),
                        key="case_constraints_enabled"
                    )
                    
                    if case_enabled:
                        # 최소 차이 설정
                        min_difference = st.slider(
                            "Case 간 최소 차이 (%)",
                            min_value=1.0,
                            max_value=20.0,
                            value=float(current_params.get('case_constraints', {}).get('min_difference', 5.0)),
                            step=1.0,
                            format="%.1f",
                            help="Case 간 최소 RE 비중 차이 (%)"
                        )
                        
                        st.info("• Case1 > Case2 > Case3 순으로 RE 비중이 높아지도록 제약")
                        st.info(f"• Case 간 최소 차이: {min_difference}%")
                        
                        # 설정 저장
                        if 'case_constraints' not in current_params:
                            current_params['case_constraints'] = {}
                        current_params['case_constraints']['enabled'] = case_enabled
                        current_params['case_constraints']['min_difference'] = min_difference
                    else:
                        if 'case_constraints' not in current_params:
                            current_params['case_constraints'] = {}
                        current_params['case_constraints']['enabled'] = False
                
                # 자재 생산국가 제약조건
                with st.expander("자재 생산국가 제약조건", expanded=False):
                    location_enabled = st.checkbox(
                        "생산국가 제약조건 활성화", 
                        value=current_params.get('location_constraints', {}).get('enabled', False),
                        key="location_constraints_enabled"
                    )
                    
                    if location_enabled:
                        # 고정 제약 사용 여부
                        use_fixed_constraints = st.checkbox(
                            "자재별 생산국가 고정 제약 사용",
                            value=current_params.get('location_constraints', {}).get('use_fixed_constraints', True),
                            help="특정 자재의 생산 국가를 고정하려면 선택",
                            key="location_fixed_constraints"
                        )
                        
                        # 자재별 생산국가 설정
                        st.markdown("##### 자재별 가능 생산국가")
                        
                        # 시뮬레이션 데이터에서 국가 정보 추출
                        if 'simulation_data_loaded' in st.session_state and st.session_state.get('simulation_data', None):
                            sim_data = st.session_state.simulation_data
                            available_countries = []
                            
                            # 전력배출계수 데이터에서 기본 국가 목록 추출
                            try:
                                stable_var_data = load_stable_var_data()
                                electricity_coefs = stable_var_data.get('electricity_coef', {})
                                available_countries = list(electricity_coefs.keys())
                            except Exception as e:
                                available_countries = ['한국', '중국', '일본', '폴란드', '독일', '미국']
                                
                            # 시뮬레이션 데이터에서 자재별 국가 추출
                            material_countries = {}
                            if 'original_df' in sim_data and '자재품목' in sim_data['original_df'].columns and '지역' in sim_data['original_df'].columns:
                                df = sim_data['original_df']
                                for material, group in df.groupby('자재품목'):
                                    if not pd.isna(material):
                                        # 해당 자재가 생산되는 국가들
                                        countries = group['지역'].dropna().unique().tolist()
                                        if countries:
                                            material_countries[material] = countries
                            
                            # 자재별 국가 정보 출력
                            if material_countries:
                                for material, countries in material_countries.items():
                                    country_list = ', '.join(countries)
                                    st.info(f"• {material}: {country_list}")
                            else:
                                # 시뮬레이션 데이터에서 국가 정보를 추출할 수 없을 때 기본값 사용
                                st.info("• 양극재: 한국, 중국, 일본")
                                st.info("• 분리막: 한국, 중국, 폴란드")
                                st.info("• 전해액: 한국, 중국, 일본")
                        else:
                            # 데이터가 없을 때 기본값 사용
                            st.info("• 양극재: 한국, 중국, 일본")
                            st.info("• 분리막: 한국, 중국, 폴란드")
                            st.info("• 전해액: 한국, 중국, 일본")
                        
                        # 설정 저장
                        if 'location_constraints' not in current_params:
                            current_params['location_constraints'] = {}
                        current_params['location_constraints']['enabled'] = location_enabled
                        current_params['location_constraints']['use_fixed_constraints'] = use_fixed_constraints
                    else:
                        if 'location_constraints' not in current_params:
                            current_params['location_constraints'] = {}
                        current_params['location_constraints']['enabled'] = False
                
                # 재활용재 및 저탄소 메탈 제약조건
                with st.expander("재활용재 및 저탄소 메탈 사용비율 제약조건", expanded=False):
                    material_enabled = st.checkbox(
                        "원료 구성 제약조건 활성화", 
                        value=current_params.get('material_constraints', {}).get('enabled', False),
                        key="material_constraints_enabled"
                    )
                    
                    if material_enabled:
                        # 재활용재 사용비율 제약
                        col1, col2 = st.columns(2)
                        with col1:
                            recycle_enabled = st.checkbox(
                                "재활용재 사용비율 제약 활성화", 
                                value=current_params.get('material_constraints', {}).get('recycle_ratio', {}).get('enabled', True),
                                key="recycle_enabled"
                            )
                        
                        with col2:
                            low_carbon_enabled = st.checkbox(
                                "저탄소 메탈 사용비율 제약 활성화", 
                                value=current_params.get('material_constraints', {}).get('low_carbon_ratio', {}).get('enabled', True),
                                key="low_carbon_enabled"
                            )
                        
                        # 재활용재 비율 범위
                        if recycle_enabled:
                            st.markdown("##### 재활용재 사용비율 설정")
                            # 기본값 설정 및 100% 범위로 변환
                            # 이미 material_constraints가 있는지 확인
                            if 'material_constraints' not in current_params:
                                current_params['material_constraints'] = {}
                            if 'recycle_ratio' not in current_params['material_constraints']:
                                current_params['material_constraints']['recycle_ratio'] = {'min': 0.1, 'max': 0.5}
                            
                            # 퍼센트로 변환 (0-1 범위 -> 0-100 범위)
                            min_val = float(current_params['material_constraints']['recycle_ratio'].get('min', 0.1))
                            max_val = float(current_params['material_constraints']['recycle_ratio'].get('max', 0.5))
                            
                            # 유효값 확인 및 퍼센트 변환
                            if min_val > 1.0:  # 이미 퍼센트로 저장되어 있는 경우
                                min_val_pct = min_val
                            else:
                                min_val_pct = min_val * 100
                                
                            if max_val > 1.0:  # 이미 퍼센트로 저장되어 있는 경우
                                max_val_pct = max_val
                            else:
                                max_val_pct = max_val * 100
                            
                            # 구간 유효성 검사
                            if min_val_pct > max_val_pct:
                                min_val_pct, max_val_pct = max_val_pct, min_val_pct
                                
                            # 범위를 0-100% 내로 제한
                            min_val_pct = max(0.0, min(min_val_pct, 100.0))
                            max_val_pct = max(0.0, min(max_val_pct, 100.0))
                            
                            # 슬라이더 표시
                            recycle_min, recycle_max = st.slider(
                                "재활용재 사용비율 범위 (%)",
                                min_value=0.0,
                                max_value=100.0,
                                value=(float(min_val_pct), float(max_val_pct)),
                                step=5.0,
                                help="재활용재 사용비율 범위 (%)"                                
                            )
                        
                        # 저탄소 메탈 비율 범위
                        if low_carbon_enabled:
                            st.markdown("##### 저탄소 메탈 사용비율 설정")
                            # 기본값 설정 및 100% 범위로 변환
                            # 이미 material_constraints가 있는지 확인
                            if 'material_constraints' not in current_params:
                                current_params['material_constraints'] = {}
                            if 'low_carbon_ratio' not in current_params['material_constraints']:
                                current_params['material_constraints']['low_carbon_ratio'] = {'min': 0.05, 'max': 0.3}
                            
                            # 퍼센트로 변환 (0-1 범위 -> 0-100 범위)
                            min_val = float(current_params['material_constraints']['low_carbon_ratio'].get('min', 0.05))
                            max_val = float(current_params['material_constraints']['low_carbon_ratio'].get('max', 0.3))
                            
                            # 유효값 확인 및 퍼센트 변환
                            if min_val > 1.0:  # 이미 퍼센트로 저장되어 있는 경우
                                min_val_pct = min_val
                            else:
                                min_val_pct = min_val * 100
                                
                            if max_val > 1.0:  # 이미 퍼센트로 저장되어 있는 경우
                                max_val_pct = max_val
                            else:
                                max_val_pct = max_val * 100
                            
                            # 구간 유효성 검사
                            if min_val_pct > max_val_pct:
                                min_val_pct, max_val_pct = max_val_pct, min_val_pct
                                
                            # 범위를 0-100% 내로 제한
                            min_val_pct = max(0.0, min(min_val_pct, 100.0))
                            max_val_pct = max(0.0, min(max_val_pct, 100.0))
                            
                            # 슬라이더 표시
                            low_carbon_min, low_carbon_max = st.slider(
                                "저탄소 메탈 사용비율 범위 (%)",
                                min_value=0.0,
                                max_value=100.0,
                                value=(float(min_val_pct), float(max_val_pct)),
                                step=5.0,
                                help="저탄소 메탈 사용비율 범위 (%)"                                
                            )
                            
                        # 원료 합계 제약
                        balance_enabled = st.checkbox(
                            "원료 합계 제약 활성화", 
                            value=current_params.get('material_constraints', {}).get('material_balance', {}).get('enabled', True),
                            help="재활용재와 저탄소 메탈 합계에 제약 설정",
                            key="balance_enabled"
                        )
                        
                        if balance_enabled:
                            # 기본값 설정 및 100% 범위로 변환
                            # 이미 material_constraints가 있는지 확인
                            if 'material_constraints' not in current_params:
                                current_params['material_constraints'] = {}
                            if 'material_balance' not in current_params['material_constraints']:
                                current_params['material_constraints']['material_balance'] = {'max_total': 0.7}
                            
                            # 퍼센트로 변환 (0-1 범위 -> 0-100 범위)
                            max_val = float(current_params['material_constraints']['material_balance'].get('max_total', 0.7))
                            
                            # 유효값 확인 및 퍼센트 변환
                            if max_val > 1.0:  # 이미 퍼센트로 저장되어 있는 경우
                                max_val_pct = max_val
                            else:
                                max_val_pct = max_val * 100
                                
                            # 범위를 10-100% 내로 제한
                            max_val_pct = max(10.0, min(max_val_pct, 100.0))
                            
                            # 슬라이더 표시
                            max_total = st.slider(
                                "재활용재+저탄소메탈 최대 합계 (%)",
                                min_value=10.0,
                                max_value=100.0,
                                value=float(max_val_pct),
                                step=5.0,
                                help="두 원료의 최대 합계 비율 (%)"                                
                            )
                        
                        # 설정 저장
                        if 'material_constraints' not in current_params:
                            current_params['material_constraints'] = {}
                        current_params['material_constraints']['enabled'] = material_enabled
                        
                        # 재활용재 설정
                        if 'recycle_ratio' not in current_params['material_constraints']:
                            current_params['material_constraints']['recycle_ratio'] = {}
                        current_params['material_constraints']['recycle_ratio']['enabled'] = recycle_enabled
                        if recycle_enabled:
                            current_params['material_constraints']['recycle_ratio']['min'] = recycle_min / 100.0
                            current_params['material_constraints']['recycle_ratio']['max'] = recycle_max / 100.0
                        
                        # 저탄소 메탈 설정
                        if 'low_carbon_ratio' not in current_params['material_constraints']:
                            current_params['material_constraints']['low_carbon_ratio'] = {}
                        current_params['material_constraints']['low_carbon_ratio']['enabled'] = low_carbon_enabled
                        if low_carbon_enabled:
                            current_params['material_constraints']['low_carbon_ratio']['min'] = low_carbon_min / 100.0
                            current_params['material_constraints']['low_carbon_ratio']['max'] = low_carbon_max / 100.0
                        
                        # 합계 제약 설정
                        if 'material_balance' not in current_params['material_constraints']:
                            current_params['material_constraints']['material_balance'] = {}
                        current_params['material_constraints']['material_balance']['enabled'] = balance_enabled
                        if balance_enabled:
                            current_params['material_constraints']['material_balance']['max_total'] = max_total / 100.0
                    else:
                        if 'material_constraints' not in current_params:
                            current_params['material_constraints'] = {}
                        current_params['material_constraints']['enabled'] = False
                
            else:
                st.warning("⚠️ 시뮬레이션 데이터를 먼저 로드해주세요.")
                
                # 데이터 로드 안내
                st.markdown("""
                **시뮬레이션 정렬 탄소최소화 사용 방법:**
                1. 위에서 데이터 소스를 선택하세요
                2. 'PCF 시뮬레이션' 페이지에서 시뮬레이션을 실행하거나 샘플 데이터를 로드하세요
                3. RE 적용률과 목표 탄소발자국을 설정하세요
                4. 최적화를 실행하세요
                """)
            
        elif current_scenario == 'cost_minimization':
            # 비용 최소화 파라미터 - 시뮬레이션 정렬 버전
            st.markdown("### 목표 설정")
            
            col1, col2 = st.columns(2)
            with col1:
                # 탄소발자국 목표
                target_carbon = st.number_input(
                    "목표 탄소발자국 (kgCO2eq)", 
                    min_value=5.0,
                    max_value=50.0,
                    value=float(current_params.get('target_carbon', 45.0)),
                    step=0.5,
                    format="%.1f",
                    help="달성하고자 하는 탄소발자국 목표 값",
                    key="target_carbon_cost"
                )
                current_params['target_carbon'] = target_carbon
            
            with col2:
                use_simulation_data = st.checkbox(
                    "시뮬레이션 데이터 사용",
                    value=current_params.get('use_simulation_data', False),
                    help="시뮬레이션 데이터를 사용한 정밀 계산 (미선택시 단순화된 계산)",
                    key="use_sim_data_cost"
                )
                current_params['use_simulation_data'] = use_simulation_data
            
            if use_simulation_data:
                st.markdown("### RE 적용률 설정")
                st.markdown("각 Tier별 재생에너지(RE) 적용률을 설정하세요.")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tier1_re_rate = st.slider(
                        "Tier1 RE 적용률",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_params.get('tier1_re_rate', 0.3)),
                        step=0.05,
                        format="%.2f",
                        key="tier1_cost_min"
                    )
                    current_params['tier1_re_rate'] = tier1_re_rate
                
                with col2:
                    tier2_re_rate = st.slider(
                        "Tier2 RE 적용률",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_params.get('tier2_re_rate', 0.5)),
                        step=0.05,
                        format="%.2f",
                        key="tier2_cost_min"
                    )
                    current_params['tier2_re_rate'] = tier2_re_rate
                
                # 시뮬레이션 데이터에서 num_tier 확인
                num_tier = 3  # 기본값
                
                # 시뮬레이션 데이터가 로드되었는지 확인
                if 'simulation_data' in st.session_state and st.session_state['simulation_data']:
                    sim_data = st.session_state['simulation_data']
                    if 'original_df' in sim_data and 'num_tier' in sim_data['original_df'].columns:
                        # 데이터프레임에서 num_tier 값 추출 (첫 번째 값 사용)
                        num_tier_values = sim_data['original_df']['num_tier'].unique()
                        if len(num_tier_values) > 0:
                            num_tier = int(num_tier_values[0])
                    elif 'num_tier' in sim_data:
                        # 직접 num_tier 키가 있는 경우
                        num_tier = int(sim_data['num_tier'])
                
                # Tier3는 num_tier가 3일 때만 표시
                if num_tier >= 3:
                    with col3:
                        tier3_re_rate = st.slider(
                            "Tier3 RE 적용률",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(current_params.get('tier3_re_rate', 0.7)),
                            step=0.05,
                            format="%.2f",
                            key="tier3_cost_min"
                        )
                        current_params['tier3_re_rate'] = tier3_re_rate
            
            # 기본 제약조건 표시
            st.markdown("#### 기본 제약조건")
            st.info("• 탄소발자국 ≤ 목표 탄소발자국")
            if use_simulation_data:
                st.info("• RE 적용률 범위: 0-100%")
                st.info("• 시뮬레이션과 동일한 자재별 배출계수 계산")
            else:
                st.info("• 모든 비율의 합 = 1 (기본 제약)")
                st.info("• 감축비율 범위는 0-100%")
            
        elif current_scenario == 'multi_objective':
            # 다목적 최적화 파라미터
            st.markdown("### 가중치 설정")
            st.markdown("탄소발자국과 비용의 중요도를 설정하세요. 두 가중치의 합은 1이 됩니다.")
            
            col1, col2 = st.columns(2)
            with col1:
                # 탄소발자국 가중치
                carbon_weight = st.slider(
                    "탄소발자국 가중치", 
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_params.get('carbon_weight', 0.7)),
                    step=0.05,
                    help="탄소발자국 감축의 중요도 (0: 중요하지 않음, 1: 가장 중요)"
                )
                current_params['carbon_weight'] = carbon_weight
            
            with col2:
                # 비용 가중치 (자동 계산)
                cost_weight = 1.0 - carbon_weight
                st.metric("비용 가중치", f"{cost_weight:.2f}")
                current_params['cost_weight'] = cost_weight
            
            # 비용 제약
            st.markdown("### 제약조건 설정")
            max_cost = st.number_input(
                "최대 비용 (USD)", 
                min_value=1000.0,
                max_value=1000000.0,
                value=float(current_params.get('max_cost', 100000.0)),
                step=1000.0,
                format="%.1f",
                help="허용되는 최대 구현 비용"
            )
            current_params['max_cost'] = max_cost
            
            # 가중치 시각화
            st.markdown("### 가중치 시각화")
            fig = go.Figure(go.Pie(
                labels=["탄소발자국", "비용"], 
                values=[carbon_weight, cost_weight],
                hole=.3,
                marker_colors=['#2E8B57', '#4682B4'],
                hoverinfo='label+percent',
                textinfo='label+percent'
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif current_scenario == 'implementation_ease':
            # 구현 용이성 최적화 파라미터
            st.markdown("### 목표 설정")
            
            # 탄소발자국 목표
            target_carbon = st.number_input(
                "목표 탄소발자국 (kg CO2/kWh)", 
                min_value=10.0,
                max_value=100.0,
                value=float(current_params.get('target_carbon', 45.0)),
                step=1.0,
                format="%.1f",
                help="달성하고자 하는 탄소발자국 목표 값"
            )
            current_params['target_carbon'] = target_carbon
            
            # 구현 난이도 가중치
            implementation_factor = st.slider(
                "구현 난이도 가중치", 
                min_value=0.1,
                max_value=1.0,
                value=float(current_params.get('implementation_factor', 0.8)),
                step=0.05,
                help="구현 난이도의 중요도 (낮을수록 더 적은 활동에 집중)"
            )
            current_params['implementation_factor'] = implementation_factor
            
            # 설명
            st.info("""
            **구현 난이도 가중치**란?
            - 값이 높을수록 (1에 가까울수록) 더 많은 활동에 작은 변화를 적용
            - 값이 낮을수록 (0에 가까울수록) 적은 수의 활동에 큰 변화를 집중
            """)
            
            # 기본 제약조건
            st.markdown("#### 기본 제약조건")
            st.info("• 탄소발자국 ≤ 목표 탄소발자국")
            st.info("• 활성화된 감축 활동 수 최소화")
            
        elif current_scenario == 'regional_optimization':
            # 지역별 최적화 파라미터
            st.markdown("### 지역 설정")
            
            # stable_var에서 전력배출계수 데이터 로드
            electricity_coefs = None
            try:
                stable_var_data = load_stable_var_data()
                electricity_coefs = stable_var_data['electricity_coef']
                available_regions = list(electricity_coefs.keys())
            except:
                available_regions = ['한국', '중국', '일본', '폴란드']
            
            # 고려할 지역 선택
            selected_regions = st.multiselect(
                "고려할 지역", 
                options=available_regions,
                default=current_params.get('target_regions', ['한국', '중국', '일본', '폴란드']),
                help="최적화에 고려할 생산 가능 지역"
            )
            current_params['target_regions'] = selected_regions
            
            # 가중치 설정
            st.markdown("### 가중치 설정")
            col1, col2 = st.columns(2)
            
            with col1:
                # 탄소발자국 가중치
                carbon_weight = st.slider(
                    "탄소발자국 가중치", 
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_params.get('carbon_weight', 0.7)),
                    step=0.05,
                    help="탄소발자국 감축의 중요도"
                )
                current_params['carbon_weight'] = carbon_weight
            
            with col2:
                # 물류 가중치 (자동 계산)
                logistics_weight = 1.0 - carbon_weight
                st.metric("물류 가중치", f"{logistics_weight:.2f}")
                current_params['logistics_weight'] = logistics_weight
            
            # 지역별 전력배출계수 시각화
            if electricity_coefs and selected_regions:
                st.markdown("### 지역별 전력배출계수")
                
                # 선택된 지역의 전력배출계수만 필터링
                region_data = {region: electricity_coefs[region] for region in selected_regions if region in electricity_coefs}
                
                # 바 차트 생성
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(region_data.keys()),
                    y=list(region_data.values()),
                    text=[f"{val:.4f}" for val in region_data.values()],
                    textposition='auto',
                    marker_color='#4682B4'
                ))
                
                fig.update_layout(
                    title="지역별 전력배출계수",
                    xaxis_title="지역",
                    yaxis_title="배출계수",
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif current_scenario == 'simulation_aligned_carbon':
            # 시뮬레이션 정렬 탄소발자국 최소화 파라미터
            if SIMULATION_ALIGNED_AVAILABLE:
                st.markdown("### 시뮬레이션 데이터 로드")
                
                # 시뮬레이션 데이터 로드 방법 선택
                data_source = st.radio(
                    "시뮬레이션 데이터 소스",
                    options=["현재 세션 데이터 사용", "파일에서 로드", "샘플 데이터 사용"],
                    index=2,  # 기본값: 샘플 데이터
                    help="최적화에 사용할 시뮬레이션 데이터 소스를 선택하세요"
                )
                
                # 데이터 로드 상태 표시
                if 'simulation_data_loaded' not in st.session_state:
                    st.session_state.simulation_data_loaded = False
                    st.session_state.simulation_data = None
                
                if data_source == "샘플 데이터 사용":
                    if st.button("📊 샘플 데이터 로드", key="load_sample_data"):
                        with st.spinner("샘플 데이터 생성 중..."):
                            # 샘플 데이터 생성
                            st.session_state.simulation_data = generate_sample_simulation_data()
                            st.session_state.simulation_data_loaded = True
                            st.success("✅ 샘플 시뮬레이션 데이터가 로드되었습니다!")
                
                elif data_source == "현재 세션 데이터 사용":
                    # PCF 시뮬레이션 페이지의 데이터 확인
                    if check_session_simulation_data():
                        if st.button("📊 세션 데이터 로드", key="load_session_data"):
                            st.session_state.simulation_data = get_session_simulation_data()
                            st.session_state.simulation_data_loaded = True
                            st.success("✅ 세션 시뮬레이션 데이터가 로드되었습니다!")
                    else:
                        st.warning("⚠️ 현재 세션에서 시뮬레이션 데이터를 찾을 수 없습니다. PCF 시뮬레이션 페이지에서 먼저 시뮬레이션을 실행해주세요.")
                
                # 데이터 로드 상태에 따른 UI
                if st.session_state.simulation_data_loaded and st.session_state.simulation_data:
                    sim_data = st.session_state.simulation_data
                    
                    # 데이터 요약 표시
                    st.markdown("#### 로드된 시뮬레이션 데이터 요약")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        scenario_count = len(sim_data['scenario_df']) if 'scenario_df' in sim_data else 0
                        st.metric("시나리오 자재 수", scenario_count)
                    
                    with col2:
                        applicable_count = 0
                        if 'scenario_df' in sim_data and '저감활동_적용여부' in sim_data['scenario_df'].columns:
                            applicable_count = len(sim_data['scenario_df'][sim_data['scenario_df']['저감활동_적용여부'] == 1.0])
                        elif 'scenario_df' in sim_data:
                            # 저감활동_적용여부 컬럼이 없으면 전체 자재 수로 표시
                            applicable_count = len(sim_data['scenario_df'])
                        st.metric("저감활동 적용 자재", applicable_count)
                    
                    with col3:
                        formula_count = len(sim_data['ref_formula_df']) if 'ref_formula_df' in sim_data else 0
                        st.metric("Formula 참조 데이터", formula_count)
                    
                    with col4:
                        proportions_count = len(sim_data['ref_proportions_df']) if 'ref_proportions_df' in sim_data else 0
                        st.metric("Proportions 참조 데이터", proportions_count)
                    
                    # 기준 PCF 계산
                    if 'scenario_df' in sim_data:
                        baseline_pcf = sim_data['scenario_df']['배출량(kgCO2eq)'].sum()
                        st.metric("기준 PCF", f"{baseline_pcf:.4f} kgCO2eq")
                        
                        # 국가 추천 설정
                        with st.expander("국가 추천 설정", expanded=False):
                            recommend_countries = st.checkbox(
                                "최적 국가 추천 활성화",
                                value=current_params.get('recommend_countries', False),
                                help="최적화 결과에서 국가 추천 정보를 제공합니다"
                            )
                            
                            if recommend_countries:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    carbon_weight = st.slider(
                                        "탄소배출 가중치",
                                        min_value=0.1,
                                        max_value=1.0,
                                        value=float(current_params.get('country_weights', {}).get('carbon', 0.6)),
                                        step=0.1,
                                        help="국가 추천시 탄소배출 중요도"
                                    )
                                
                                with col2:
                                    cost_weight = st.slider(
                                        "비용 가중치",
                                        min_value=0.1,
                                        max_value=1.0,
                                        value=float(current_params.get('country_weights', {}).get('cost', 0.3)),
                                        step=0.1,
                                        help="국가 추천시 비용 중요도"
                                    )
                                    
                                with col3:
                                    logistics_weight = st.slider(
                                        "물류 가중치",
                                        min_value=0.1,
                                        max_value=1.0,
                                        value=float(current_params.get('country_weights', {}).get('logistics', 0.1)),
                                        step=0.1,
                                        help="국가 추천시 물류비용 중요도"
                                    )
                                
                                # 가중치 합 정규화
                                total_weight = carbon_weight + cost_weight + logistics_weight
                                carbon_weight_norm = carbon_weight / total_weight
                                cost_weight_norm = cost_weight / total_weight
                                logistics_weight_norm = logistics_weight / total_weight
                                
                                st.info(f"정규화된 가중치: 탄소({carbon_weight_norm:.2f}), 비용({cost_weight_norm:.2f}), 물류({logistics_weight_norm:.2f})")
                                
                                # 결과에 표시할 추천 국가 수
                                top_n = st.number_input(
                                    "표시할 추천 국가 수",
                                    min_value=1,
                                    max_value=10,
                                    value=int(current_params.get('top_n_countries', 3)),
                                    step=1
                                )
                                
                                # 설정 저장
                                current_params['recommend_countries'] = recommend_countries
                                if 'country_weights' not in current_params:
                                    current_params['country_weights'] = {}
                                current_params['country_weights']['carbon'] = carbon_weight_norm
                                current_params['country_weights']['cost'] = cost_weight_norm
                                current_params['country_weights']['logistics'] = logistics_weight_norm
                                current_params['top_n_countries'] = top_n
                            else:
                                current_params['recommend_countries'] = False
                    
                    # RE 적용률 설정
                    st.markdown("### RE 적용률 설정")
                    st.markdown("각 Tier별 재생에너지(RE) 적용률을 설정하세요. (0: 미적용, 1: 100% 적용)")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        tier1_re_rate = st.slider(
                            "Tier1 RE 적용률",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(current_params.get('tier1_re_rate', 0.3)),
                            step=0.05,
                            format="%.2f",
                            help="Tier1 재생에너지 적용 비율"
                        )
                        current_params['tier1_re_rate'] = tier1_re_rate
                    
                    with col2:
                        tier2_re_rate = st.slider(
                            "Tier2 RE 적용률",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(current_params.get('tier2_re_rate', 0.5)),
                            step=0.05,
                            format="%.2f",
                            help="Tier2 재생에너지 적용 비율"
                        )
                        current_params['tier2_re_rate'] = tier2_re_rate
                    
                    # 시뮬레이션 데이터에서 num_tier 확인
                    num_tier = 3  # 기본값
                    
                    # 시뮬레이션 데이터가 로드되었는지 확인
                    if 'original_df' in sim_data and 'num_tier' in sim_data['original_df'].columns:
                        # 데이터프레임에서 num_tier 값 추출 (첫 번째 값 사용)
                        num_tier_values = sim_data['original_df']['num_tier'].unique()
                        if len(num_tier_values) > 0:
                            num_tier = int(num_tier_values[0])
                    elif 'num_tier' in sim_data:
                        # 직접 num_tier 키가 있는 경우
                        num_tier = int(sim_data['num_tier'])
                    
                    # Tier3는 num_tier가 3일 때만 표시
                    if num_tier >= 3:
                        with col3:
                            tier3_re_rate = st.slider(
                                "Tier3 RE 적용률",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(current_params.get('tier3_re_rate', 0.7)),
                                step=0.05,
                                format="%.2f",
                                help="Tier3 재생에너지 적용 비율"
                            )
                            current_params['tier3_re_rate'] = tier3_re_rate
                    
                    # 제약조건 설정
                    st.markdown("### 제약조건 설정")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 목표 탄소발자국
                        target_carbon = st.number_input(
                            "목표 탄소발자국 (kgCO2eq)",
                            min_value=5.0,
                            max_value=50.0,
                            value=float(current_params.get('target_carbon', 12.0)),
                            step=0.5,
                            format="%.1f",
                            help="달성하고자 하는 탄소발자국 목표 값"
                        )
                        current_params['target_carbon'] = target_carbon
                    
                    with col2:
                        # 최대 비용
                        max_cost = st.number_input(
                            "최대 비용 (USD)",
                            min_value=1000.0,
                            max_value=200000.0,
                            value=float(current_params.get('max_cost', 50000.0)),
                            step=1000.0,
                            format="%.0f",
                            help="허용되는 최대 구현 비용"
                        )
                        current_params['max_cost'] = max_cost
                    
                    # 기본 제약조건 안내
                    st.markdown("#### 기본 제약조건")
                    st.info("• 탄소발자국 ≤ 목표 탄소발자국")
                    st.info("• 총 비용 ≤ 최대 비용") 
                    st.info("• RE 적용률 범위: 0-100%")
                    st.info("• 시뮬레이션과 동일한 자재별 배출계수 계산")
                    
                else:
                    st.warning("⚠️ 시뮬레이션 데이터를 먼저 로드해주세요.")
                    
                    # 데이터 로드 안내
                    st.markdown("""
                    **시뮬레이션 정렬 최적화 사용 방법:**
                    1. 위에서 데이터 소스를 선택하세요
                    2. 'PCF 시뮬레이션' 페이지에서 시뮬레이션을 실행하거나 샘플 데이터를 로드하세요
                    3. RE 적용률과 제약조건을 설정하세요
                    4. 최적화를 실행하세요
                    """)
            
            else:
                st.error("❌ 시뮬레이션 정렬 최적화 모듈을 사용할 수 없습니다. 필요한 모듈이 설치되었는지 확인해주세요.")
        
        elif current_scenario == 'material_based':
            # 자재 기반 최적화 - 시뮬레이션 데이터 로드
            st.markdown("### 시뮬레이션 데이터 로드")
            
            # 데이터 로드 상태 초기화
            if 'material_simulation_data_loaded' not in st.session_state:
                st.session_state.material_simulation_data_loaded = False
                st.session_state.material_simulation_data = None
            
            # 시뮬레이션 데이터 로드 방법 선택
            data_source = st.radio(
                "시뮬레이션 데이터 소스",
                options=["현재 세션 데이터 사용", "샘플 데이터 사용"],
                index=0,  # 기본값: 현재 세션 데이터
                help="최적화에 사용할 시뮬레이션 데이터 소스를 선택하세요",
                key=f"data_source_material_based"
            )
            
            if data_source == "샘플 데이터 사용":
                if st.button("📊 샘플 데이터 로드", key="load_sample_material_data"):
                    with st.spinner("샘플 데이터 생성 중..."):
                        st.session_state.material_simulation_data = generate_sample_simulation_data()
                        st.session_state.material_simulation_data_loaded = True
                        st.success("✅ 샘플 시뮬레이션 데이터가 로드되었습니다!")
            
            elif data_source == "현재 세션 데이터 사용":
                # 세션 데이터 확인
                if check_session_simulation_data():
                    if st.button("📊 세션 데이터 로드", key="load_session_material_data"):
                        st.session_state.material_simulation_data = get_session_simulation_data()
                        st.session_state.material_simulation_data_loaded = True
                        st.success("✅ 세션 시뮬레이션 데이터가 로드되었습니다!")
                else:
                    st.warning("⚠️ 현재 세션에서 시뮬레이션 데이터를 찾을 수 없습니다. PCF 시뮬레이션 페이지에서 먼저 시뮬레이션을 실행해주세요.")
            
            # 데이터 로드 상태에 따른 expander 표시
            if st.session_state.get('material_simulation_data_loaded', False) and st.session_state.get('material_simulation_data'):
                sim_data = st.session_state.material_simulation_data
                scenario_df = sim_data.get('scenario_df', pd.DataFrame())
                
                # 시나리오 데이터프레임 expander 추가
                if not scenario_df.empty:
                    with st.expander(f"📊 시나리오 데이터 확인 : {len(scenario_df)}개 자재", expanded=False):
                        st.write("**현재 로드된 시나리오 데이터:**")
                        st.dataframe(scenario_df, use_container_width=True, height=400)
                        
                        # 데이터 통계 정보
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 자재 수", f"{len(scenario_df):,}개")
                        with col2:
                            applied_count = len(scenario_df[scenario_df.get('저감활동_적용여부', 0) == 1.0]) if '저감활동_적용여부' in scenario_df.columns else 0
                            st.metric("🎯 저감활동 적용", f"{applied_count}개")
                        with col3:
                            st.metric("📋 특성 수", f"{len(scenario_df.columns)}개")
                
                # 저감활동 적용 자재 정보 expander
                if not scenario_df.empty and '저감활동_적용여부' in scenario_df.columns:
                    # 저감활동이 적용된 자재만 필터링
                    applied_materials = scenario_df[scenario_df['저감활동_적용여부'] == 1.0].copy()
                    
                    if len(applied_materials) > 0:
                        # 저감활동 적용 자재 상세 정보 (expander)
                        with st.expander(f"저감활동 적용 자재 상세 정보 : {len(applied_materials)}개", expanded=False):
                            # 표시할 컬럼 정의
                            base_columns = ['자재명', '자재품목', '제품총소요량(kg)', '배출계수명', '배출계수', '배출량(kgCO2eq)', '저감활동_적용여부']
                            
                            # 실제 존재하는 컬럼만 필터링
                            available_columns = [col for col in base_columns if col in applied_materials.columns]
                            
                            # 데이터프레임 표시
                            st.write("**저감활동 적용 자재 상세 정보:**")
                            
                            # 데이터프레임을 더 보기 좋게 표시
                            display_df = applied_materials[available_columns].copy()
                            
                            # 숫자 컬럼 포맷팅
                            numeric_columns = ['제품총소요량(kg)', '배출계수', '배출량(kgCO2eq)']
                            for col in numeric_columns:
                                if col in display_df.columns:
                                    display_df[col] = display_df[col].round(4)
                            
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                height=400
                            )
                            
                            # 자재품목별 분포
                            st.write("**자재품목별 분포:**")
                            material_distribution = applied_materials['자재품목'].value_counts()
                            for category, count in material_distribution.items():
                                st.write(f"• {category}: {count}개")
            
            else:
                st.error("❌ 시뮬레이션 정렬 최적화 모듈을 사용할 수 없습니다. 필요한 모듈이 설치되었는지 확인해주세요.")
        
        # 파라미터 업데이트
        st.session_state.optimization_config['scenario_params'][current_scenario] = current_params
        
        # 자재 기반 최적화 특수 설정
        if current_scenario == 'material_based':
            # 자재 기반 최적화 파라미터 설정
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.markdown('<div class="config-title">자재 기반 최적화 파라미터</div>', unsafe_allow_html=True)
            
            # 자재 기반 최적화 설명
            st.markdown("""
            **자재별 특성을 고려한 맞춤형 최적화로 정밀한 탄소 배출량 감축 방안을 도출합니다.**  
            자재 유형에 따라 최적화 접근 방식을 자동으로 다르게 적용합니다.
            """)
            
            # 데이터 로드 상태에 따른 파라미터 설정 UI
            if st.session_state.get('material_simulation_data_loaded', False) and st.session_state.get('material_simulation_data'):
                # 최적화 시나리오 선택
                st.markdown("### 최적화 시나리오 설정")
                optimization_scenario = st.selectbox(
                    "최적화 시나리오",
                    options=['baseline', 'recycling', 'site_change', 'both'],
                    format_func=lambda x: {
                        'baseline': '기준 시나리오 (변경 없음)',
                        'recycling': '재활용&저탄소메탈 시나리오',
                        'site_change': '생산지 변경 시나리오',
                        'both': '종합 시나리오 (재활용&저탄소메탈 + 생산지 변경)'
                    }[x],
                    index=0,
                    help="시뮬레이션 로직과 일치하는 최적화 시나리오를 선택하세요",
                    key=f"optimization_scenario_material_based"
                )
                
                # 선택된 시나리오 정보 표시
                scenario_descriptions = {
                    'baseline': '🔵 기준 시나리오: 양극재는 기본 설정, 일반 자재는 tier-RE만 적용',
                    'recycling': '🟢 재활용&저탄소메탈: cathode_configuration.py 재활용 설정 적용',
                    'site_change': '🟡 생산지 변경: after 사이트 전력배출계수 적용',
                    'both': '🔴 종합 시나리오: 재활용&저탄소메탈 + 생산지 변경 모두 적용'
                }
                st.info(scenario_descriptions[optimization_scenario])
                
                # 자재별 감축 목표 설정
                st.markdown("---")
                st.markdown("### 자재별 감축 목표 설정")
                st.markdown("저감활동이 적용된 자재별로 개별 감축 목표를 설정하세요. 자재의 특성에 따라 감축 가능성이 다릅니다.")
                
                # 시뮬레이션 데이터에서 자재 목록 추출
                sim_data = st.session_state.material_simulation_data
                scenario_df = sim_data.get('scenario_df', pd.DataFrame())
                
                if not scenario_df.empty and '자재품목' in scenario_df.columns and '저감활동_적용여부' in scenario_df.columns:
                    # 저감활동이 적용된 자재만 필터링
                    applied_materials_df = scenario_df[scenario_df['저감활동_적용여부'] == 1.0]
                    
                    if applied_materials_df.empty:
                        st.warning("⚠️ 저감활동이 적용된 자재가 없습니다. 시나리오 설정에서 저감활동을 적용할 자재를 선택해주세요.")
                        st.stop()
                    
                    unique_materials = applied_materials_df['자재품목'].unique().tolist()
                    st.info(f"🎯 최적화 대상 자재: {', '.join(unique_materials)}")
                    
                    # 자재별 설정 초기화
                    if 'material_specific_targets' not in current_params:
                        current_params['material_specific_targets'] = {}
                    
                    # 탭 UI로 자재별 감축 목표 설정
                    if len(unique_materials) > 0:
                        material_tabs = st.tabs([f"📦 {material}" for material in unique_materials])
                        
                        for i, material in enumerate(unique_materials):
                            with material_tabs[i]:
                                st.markdown(f"### 🎯 {material} 감축 목표 설정")
                                
                                # 자재별 기본값 설정
                                default_values = {
                                    '양극재': {'min': 15.0, 'max': 25.0},  # 높은 감축 가능성
                                    '분리막': {'min': 5.0, 'max': 10.0},   # 중간 감축 가능성
                                    '전해액': {'min': 8.0, 'max': 15.0},   # 중간 감축 가능성
                                    '음극재': {'min': 10.0, 'max': 18.0},  # 높은 감축 가능성
                                    '동박': {'min': 3.0, 'max': 8.0},     # 낮은 감축 가능성
                                    'Al Foil': {'min': 5.0, 'max': 10.0}, # 알루미늄 호일
                                    'Cu Foil': {'min': 5.0, 'max': 10.0}  # 구리 호일
                                }
                                
                                material_defaults = default_values.get(material, {'min': 5.0, 'max': 10.0})
                                
                                # 현재 배출량 표시 (참고용)
                                if '배출량(kgCO2eq)' in applied_materials_df.columns:
                                    material_emission = applied_materials_df[applied_materials_df['자재품목'] == material]['배출량(kgCO2eq)'].sum()
                                    st.metric(f"현재 {material} 배출량", f"{material_emission:.4f} kgCO2eq")
                                
                                # 자재별 감축률 설정 (2열 레이아웃)
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # 기존 세션에서 음수 값이 있으면 양수로 변환
                                    saved_min = current_params.get('material_specific_targets', {}).get(material, {}).get('min', material_defaults['min'])
                                    saved_max = current_params.get('material_specific_targets', {}).get(material, {}).get('max', material_defaults['max'])
                                    
                                    # 음수를 양수로 자동 변환
                                    if saved_min < 0:
                                        saved_min = abs(saved_min)
                                    if saved_max < 0:
                                        saved_max = abs(saved_max)
                                        
                                    # min/max 순서 보정
                                    if saved_min > saved_max:
                                        saved_min, saved_max = saved_max, saved_min
                                    
                                    # 최소 감축률
                                    min_reduction = st.slider(
                                        f"최소 감축률 (%)",
                                        min_value=0.0,
                                        max_value=50.0,
                                        value=float(saved_min),
                                        step=1.0,
                                        format="%.1f",
                                        help=f"{material}의 최소 PCF 감축률 (% 단위 감축률)",
                                        key=f"min_reduction_{material}"
                                    )
                                
                                with col2:
                                    # 최대 감축률
                                    max_reduction = st.slider(
                                        f"최대 감축률 (%)",
                                        min_value=0.0,
                                        max_value=50.0,
                                        value=float(saved_max),
                                        step=1.0,
                                        format="%.1f",
                                        help=f"{material}의 최대 PCF 감축률 (% 단위 감축률)",
                                        key=f"max_reduction_{material}"
                                    )
                                
                                # 자재별 설정 저장
                                if material not in current_params['material_specific_targets']:
                                    current_params['material_specific_targets'][material] = {}
                                current_params['material_specific_targets'][material]['min'] = min_reduction
                                current_params['material_specific_targets'][material]['max'] = max_reduction
                                
                                # 예상 감축량 계산 및 표시
                                if '배출량(kgCO2eq)' in applied_materials_df.columns:
                                    material_emission = applied_materials_df[applied_materials_df['자재품목'] == material]['배출량(kgCO2eq)'].sum()
                                    min_reduction_amount = material_emission * (min_reduction / 100)
                                    max_reduction_amount = material_emission * (max_reduction / 100)
                                    
                                    st.markdown("#### 📈 예상 감축 효과")
                                    reduction_col1, reduction_col2 = st.columns(2)
                                    with reduction_col1:
                                        st.metric("최소 감축량", f"{min_reduction_amount:.4f} kgCO2eq", f"{min_reduction:.1f}%")
                                    with reduction_col2:
                                        st.metric("최대 감축량", f"{max_reduction_amount:.4f} kgCO2eq", f"{max_reduction:.1f}%")
                    
                    # 전체 감축 목표 요약
                    st.markdown("#### 📊 전체 감축 목표 요약")
                    summary_data = []
                    total_current_emission = 0
                    total_min_target = 0
                    total_max_target = 0
                    
                    for material in unique_materials:
                        if material in current_params['material_specific_targets']:
                            material_data = applied_materials_df[applied_materials_df['자재품목'] == material]
                            current_emission = material_data['배출량(kgCO2eq)'].sum() if '배출량(kgCO2eq)' in applied_materials_df.columns else 0
                            
                            min_target = current_params['material_specific_targets'][material]['min']
                            max_target = current_params['material_specific_targets'][material]['max']
                            
                            min_target_emission = current_emission * (1 - min_target/100)
                            max_target_emission = current_emission * (1 - max_target/100)
                            
                            total_current_emission += current_emission
                            total_min_target += min_target_emission
                            total_max_target += max_target_emission
                            
                            summary_data.append({
                                "자재": material,
                                "현재 배출량": f"{current_emission:.4f}",
                                "최소 목표": f"{min_target_emission:.4f}",
                                "최대 목표": f"{max_target_emission:.4f}",
                                "감축률 범위": f"{min_target:.1f}% ~ {max_target:.1f}%"
                            })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # 전체 요약
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("전체 현재 배출량", f"{total_current_emission:.4f} kgCO2eq")
                        with col2:
                            st.metric("전체 최소 목표", f"{total_min_target:.4f} kgCO2eq", 
                                    f"{((total_min_target - total_current_emission)/total_current_emission*100):.1f}%")
                        with col3:
                            st.metric("전체 최대 목표", f"{total_max_target:.4f} kgCO2eq", 
                                    f"{((total_max_target - total_current_emission)/total_current_emission*100):.1f}%")
                else:
                    st.warning("⚠️ 시뮬레이션 데이터에서 자재 정보를 찾을 수 없습니다.")
                    
                    # 기존 전체 설정으로 fallback
                    st.markdown("**Fallback: 전체 감축 목표 설정**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        min_reduction = st.slider(
                            "최소 감축률 (%)",
                            min_value=0.0,
                            max_value=30.0,
                            value=float(current_params.get('reduction_target', {}).get('min', 5.0)),
                            step=1.0,
                            format="%.1f",
                            help="목표로 하는 최소 PCF 감축률 (양수는 감축을 의미)",
                            key="min_reduction_material_fallback"
                        )
                        
                    with col2:
                        max_reduction = st.slider(
                            "최대 감축률 (%)",
                            min_value=0.0,
                            max_value=30.0,
                            value=float(current_params.get('reduction_target', {}).get('max', 10.0)),
                            step=1.0,
                            format="%.1f",
                            help="목표로 하는 최대 PCF 감축률 (양수는 감축을 의미)",
                            key="max_reduction_material_fallback"
                        )
                    
                    # 감축 목표 저장 
                    # UI에서는 양수로 입력받지만, 내부 계산에서는 음수로 저장 (감축 = 마이너스 값)
                    if 'reduction_target' not in current_params:
                        current_params['reduction_target'] = {}
                    current_params['reduction_target']['min'] = -min_reduction  # 5% 감축 → -5%로 저장
                    current_params['reduction_target']['max'] = -max_reduction  # 10% 감축 → -10%로 저장
                
                # 자재별 RE 적용률 범위 설정
                st.markdown("---")
                st.markdown("### 자재별 RE 적용률 범위 설정")
                st.markdown("저감활동이 적용된 자재별로 Tier별 재생에너지(RE) 적용률 범위를 설정하세요. 자재의 공급망 특성에 따라 RE 적용 가능성이 다릅니다.")
                
                if not applied_materials_df.empty and '자재품목' in applied_materials_df.columns:
                    # 자재별 RE 설정 초기화
                    if 'material_specific_re_rates' not in current_params:
                        current_params['material_specific_re_rates'] = {}
                    
                    # 현재 시나리오 설정에서 tier 수 가져오기
                    current_num_tier = st.session_state.get('num_tier', 2)
                    
                    # 자재별 기본 RE 적용률 범위 (산업 특성 반영) - 동적 tier 수에 맞춤
                    # Tier 1이 가장 높고, Tier가 올라갈수록 낮아짐 (공급망 상류일수록 RE 적용 어려움)
                    def get_material_re_defaults(material, num_tiers):
                        """자재와 tier 수에 따른 동적 RE 기본값 생성"""
                        base_values = {
                            '양극재': {'base_min': 0.9, 'base_max': 1.0, 'decrease_rate': 0.12},  # Tier1 최고, 점진적 감소
                            '분리막': {'base_min': 0.8, 'base_max': 0.9, 'decrease_rate': 0.1},   # Tier1 높음, 완만한 감소
                            '전해액': {'base_min': 0.85, 'base_max': 0.95, 'decrease_rate': 0.11}, # Tier1 매우 높음, 점진적 감소
                            '음극재': {'base_min': 0.88, 'base_max': 0.98, 'decrease_rate': 0.11}, # Tier1 매우 높음, 점진적 감소
                            '동박': {'base_min': 0.85, 'base_max': 0.95, 'decrease_rate': 0.15}    # Al Foil - Tier1 매우 높게, 뚜렷한 감소
                        }
                        
                        material_base = base_values.get(material, {
                            'base_min': 0.8, 'base_max': 0.9, 'decrease_rate': 0.1  # 기본값 - Tier1 높게
                        })
                        
                        tier_values = {}
                        for tier in range(1, num_tiers + 1):
                            # Tier 1이 가장 높고, Tier가 올라갈수록 RE 적용률 감소
                            decrease_factor = (tier - 1) * material_base['decrease_rate']
                            tier_min = max(0.05, material_base['base_min'] - decrease_factor)
                            tier_max = max(0.1, material_base['base_max'] - decrease_factor)
                            
                            tier_values[f'tier{tier}'] = {
                                'min': round(tier_min, 2),
                                'max': round(tier_max, 2)
                            }
                        
                        return tier_values
                    
                    # 자재별 RE 기본값 딕셔너리 생성
                    re_default_values = {}
                    for material in unique_materials:
                        re_default_values[material] = get_material_re_defaults(material, current_num_tier)
                    
                    # 탭으로 자재별 설정 구분
                    material_tabs = st.tabs([f"📋 {material}" for material in unique_materials])
                    
                    for tab_idx, (material, tab) in enumerate(zip(unique_materials, material_tabs)):
                        with tab:
                            st.markdown(f"### {material} RE 적용률 설정")
                            
                            # 자재별 기본값 (동적 생성된 값 사용)
                            material_re_defaults = re_default_values.get(material, 
                                get_material_re_defaults(material, current_num_tier)
                            )
                            
                            # 자재별 설정 초기화
                            if material not in current_params['material_specific_re_rates']:
                                current_params['material_specific_re_rates'][material] = {}
                            
                            # Tier별 설정 (동적 tier 수에 맞춤)
                            tier_names = [f'tier{i}' for i in range(1, current_num_tier + 1)]
                            tier_cols = st.columns(current_num_tier)
                            
                            for tier_idx, tier_name in enumerate(tier_names):
                                with tier_cols[tier_idx]:
                                    st.markdown(f"#### {tier_name.upper()}")
                                    
                                    # 현재 자재의 tier별 기본값
                                    tier_defaults = material_re_defaults[tier_name]
                                    
                                    # 최소 RE 적용률 - 새로운 기본값 강제 적용
                                    min_re = st.slider(
                                        f"최소 RE 적용률",
                                        min_value=0.0,
                                        max_value=1.0,
                                        value=float(tier_defaults['min']),  # 항상 새로운 기본값 사용
                                        step=0.05,
                                        format="%.2f",
                                        help=f"{material} {tier_name.upper()} 최소 재생에너지 적용률",
                                        key=f"{material}_{tier_name}_re_min"
                                    )
                                    
                                    # 최대 RE 적용률 - 새로운 기본값 강제 적용
                                    max_re = st.slider(
                                        f"최대 RE 적용률",
                                        min_value=min_re,  # 최소값 이상으로 제한
                                        max_value=1.0,
                                        value=float(tier_defaults['max']),  # 항상 새로운 기본값 사용
                                        step=0.05,
                                        format="%.2f",
                                        help=f"{material} {tier_name.upper()} 최대 재생에너지 적용률",
                                        key=f"{material}_{tier_name}_re_max"
                                    )
                                    
                                    # 설정 저장
                                    if tier_name not in current_params['material_specific_re_rates'][material]:
                                        current_params['material_specific_re_rates'][material][tier_name] = {}
                                    
                                    current_params['material_specific_re_rates'][material][tier_name]['min'] = min_re
                                    current_params['material_specific_re_rates'][material][tier_name]['max'] = max_re
                                    
                                    # 범위 표시
                                    st.info(f"범위: {min_re:.0%} ~ {max_re:.0%}")
                            
                            # 자재별 RE 적용률 요약 차트
                            st.markdown("#### 📊 RE 적용률 범위 요약")
                            
                            # 차트 데이터 준비 (동적 tier 수에 맞춤)
                            tiers = [f'TIER{i}' for i in range(1, current_num_tier + 1)]
                            min_values = []
                            max_values = []
                            
                            for tier_name in tier_names:
                                tier_config = current_params['material_specific_re_rates'][material][tier_name]
                                min_values.append(tier_config['min'] * 100)
                                max_values.append(tier_config['max'] * 100)
                            
                            # Plotly 차트 생성
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            
                            # 범위 영역 표시
                            fig.add_trace(go.Scatter(
                                x=tiers + tiers[::-1],
                                y=min_values + max_values[::-1],
                                fill='toself',
                                fillcolor='rgba(46, 139, 87, 0.3)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='RE 적용률 범위',
                                hoverinfo="skip"
                            ))
                            
                            # 최소값 선
                            fig.add_trace(go.Scatter(
                                x=tiers,
                                y=min_values,
                                mode='lines+markers',
                                name='최소 RE 적용률',
                                line=dict(color='#2E8B57', width=3),
                                marker=dict(size=8)
                            ))
                            
                            # 최대값 선
                            fig.add_trace(go.Scatter(
                                x=tiers,
                                y=max_values,
                                mode='lines+markers',
                                name='최대 RE 적용률',
                                line=dict(color='#4682B4', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig.update_layout(
                                title=f"{material} RE 적용률 범위",
                                xaxis_title="Tier",
                                yaxis_title="RE 적용률 (%)",
                                yaxis=dict(range=[0, 100]),
                                height=400,
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # 전체 RE 적용률 설정 요약
                    st.markdown("### 📋 전체 자재별 RE 설정 요약")
                    
                    re_summary_data = []
                    for material in unique_materials:
                        if material in current_params['material_specific_re_rates']:
                            material_config = current_params['material_specific_re_rates'][material]
                            
                            # 동적 tier 데이터 생성
                            summary_row = {"자재": material}
                            max_values = []
                            
                            for tier_idx in range(1, current_num_tier + 1):
                                tier_key = f'tier{tier_idx}'
                                if tier_key in material_config:
                                    tier_range = f"{material_config[tier_key]['min']:.0%} ~ {material_config[tier_key]['max']:.0%}"
                                    summary_row[f"Tier{tier_idx} 범위"] = tier_range
                                    max_values.append(material_config[tier_key]['max'])
                            
                            # 평균 최대값 계산
                            if max_values:
                                avg_max = sum(max_values) / len(max_values)
                                summary_row["평균 최대"] = f"{avg_max:.0%}"
                            
                            re_summary_data.append(summary_row)
                    
                    if re_summary_data:
                        re_summary_df = pd.DataFrame(re_summary_data)
                        st.dataframe(re_summary_df, use_container_width=True)
                else:
                    # Fallback: 기존 전체 RE 설정
                    st.warning("⚠️ 자재별 설정을 사용할 수 없습니다. 전체 RE 설정을 사용합니다.")
                    
                    # 현재 사용자의 num_tier 설정 확인
                    user_id = st.session_state.get('user_id', None)
                    from app_helper import load_simulation_config
                    config = load_simulation_config(user_id=user_id)
                    num_tier = config.get('num_tier', 2) if config else 2  # 기본값 2
                    
                    print(f"🔧 DEBUG - Fallback RE 설정에서 num_tier: {num_tier}")
                    
                    # 동적 컬럼 생성 (num_tier에 따라)
                    if num_tier == 1:
                        col1, = st.columns(1)
                        cols = [col1]
                    elif num_tier == 2:
                        col1, col2 = st.columns(2)
                        cols = [col1, col2]
                    elif num_tier == 3:
                        col1, col2, col3 = st.columns(3)
                        cols = [col1, col2, col3]
                    elif num_tier == 4:
                        col1, col2, col3, col4 = st.columns(4)
                        cols = [col1, col2, col3, col4]
                    else:  # num_tier >= 5
                        col1, col2, col3, col4, col5 = st.columns(5)
                        cols = [col1, col2, col3, col4, col5]
                    
                    # 동적 슬라이더 생성 (num_tier에 따라)
                    tier_values = {}  # tier별 min/max 값을 저장
                    
                    for tier_idx in range(num_tier):
                        tier_num = tier_idx + 1
                        tier_key = f'tier{tier_num}'
                        
                        with cols[tier_idx]:
                            # 각 tier의 RE 적용률 범위 슬라이더
                            tier_re_min = st.slider(
                                f"Tier{tier_num} 최소 RE 적용률",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(current_params.get('re_rates', {}).get(tier_key, {}).get('min', 0.1)),
                                step=0.05,
                                format="%.2f",
                                help=f"Tier{tier_num} 최소 재생에너지 적용률 (0: 미적용, 1: 100% 적용)",
                                key=f"{tier_key}_re_min_material_fallback"
                            )
                            
                            tier_re_max = st.slider(
                                f"Tier{tier_num} 최대 RE 적용률",
                                min_value=0.0,
                                max_value=1.0,
                                value=float(current_params.get('re_rates', {}).get(tier_key, {}).get('max', 0.9)),
                                step=0.05,
                                format="%.2f",
                                help=f"Tier{tier_num} 최대 재생에너지 적용률 (0: 미적용, 1: 100% 적용)",
                                key=f"{tier_key}_re_max_material_fallback"
                            )
                            
                            tier_values[tier_key] = {'min': tier_re_min, 'max': tier_re_max}
                    
                    # RE 적용률 저장 (동적 방식)
                    if 're_rates' not in current_params:
                        current_params['re_rates'] = {}
                    
                    for tier_key, values in tier_values.items():
                        current_params['re_rates'][tier_key] = values
                
                # 시나리오별 constraint 설정
                if optimization_scenario in ['recycling', 'both']:
                    # 자재 비율 설정 (재활용&저탄소메탈 시나리오)
                    st.markdown("---")
                    st.markdown("### 자재 비율 설정")
                    st.markdown("Ni, Co, Li 등 자재에 적용되는 비율 설정 (재활용&저탄소메탈 시나리오)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 재활용 비율 범위
                        recycle_min = st.slider(
                            "재활용 최소 비율",
                            min_value=0.0,
                            max_value=0.5,
                            value=float(current_params.get('material_ratios', {}).get('recycle', {}).get('min', 0.05)),
                            step=0.05,
                            format="%.2f",
                            help="재활용 자재의 최소 비율",
                            key="recycle_min_material"
                        )
                        
                        recycle_max = st.slider(
                            "재활용 최대 비율",
                            min_value=0.0,
                            max_value=0.8,
                            value=float(current_params.get('material_ratios', {}).get('recycle', {}).get('max', 0.5)),
                            step=0.05,
                            format="%.2f",
                            help="재활용 자재의 최대 비율",
                            key="recycle_max_material"
                        )
                    
                    with col2:
                        # 저탄소 메탈 비율 범위
                        low_carbon_min = st.slider(
                            "저탄소 메탈 최소 비율",
                            min_value=0.0,
                            max_value=0.5,
                            value=float(current_params.get('material_ratios', {}).get('low_carbon', {}).get('min', 0.05)),
                            step=0.05,
                            format="%.2f",
                            help="저탄소 메탈의 최소 비율",
                            key="low_carbon_min_material"
                        )
                        
                        low_carbon_max = st.slider(
                            "저탄소 메탈 최대 비율",
                            min_value=0.0,
                            max_value=0.8,
                            value=float(current_params.get('material_ratios', {}).get('low_carbon', {}).get('max', 0.3)),
                            step=0.05,
                            format="%.2f",
                            help="저탄소 메탈의 최대 비율",
                            key="low_carbon_max_material"
                        )
                    
                    # 자재 비율 저장
                    if 'material_ratios' not in current_params:
                        current_params['material_ratios'] = {}
                    current_params['material_ratios']['recycle'] = {'min': recycle_min, 'max': recycle_max}
                    current_params['material_ratios']['low_carbon'] = {'min': low_carbon_min, 'max': low_carbon_max}
                
                if optimization_scenario in ['site_change', 'both']:
                    # 양극재 생산지 설정 (생산지 변경 시나리오)
                    st.markdown("---")
                    st.markdown("### 양극재 생산지 설정")
                    st.markdown("양극재 생산지 변경에 따른 전력 배출계수 적용 (생산지 변경 시나리오)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 현재 생산지 표시
                        st.info("**현재 생산지 (Before):** 기본 전력 배출계수 적용")
                        
                        # 실제 전력 배출계수 데이터 로드
                        def load_emission_factor_data():
                            """사용자별 또는 기본 전력 배출계수 데이터 로드"""
                            try:
                                import os
                                import json
                                current_dir = os.path.dirname(os.path.abspath(__file__))
                                project_root = os.path.join(current_dir, "..")
                                
                                # 사용자 ID 가져오기
                                user_id = st.session_state.get('user_id', None)
                                
                                electricity_coef = {}
                                national_code_mapping = {}
                                
                                # 1. 사용자별 전력 배출계수 데이터 우선 확인
                                if user_id:
                                    user_electricity_path = os.path.join(project_root, "stable_var", user_id, "electricity_coef_by_country.json")
                                    user_national_code_path = os.path.join(project_root, "stable_var", user_id, "cathode_national_code.json")
                                    
                                    if os.path.exists(user_electricity_path):
                                        with open(user_electricity_path, 'r', encoding='utf-8') as f:
                                            electricity_coef = json.load(f)
                                        st.info(f"📁 사용자별 전력 배출계수 데이터 사용: {user_id}")
                                    
                                    if os.path.exists(user_national_code_path):
                                        with open(user_national_code_path, 'r', encoding='utf-8') as f:
                                            national_data = json.load(f)
                                            national_code_mapping = national_data.get('national_code', {})
                                
                                # 2. 사용자별 데이터가 없으면 기본 데이터 사용
                                if not electricity_coef or not national_code_mapping:
                                    default_electricity_path = os.path.join(project_root, "stable_var", "electricity_coef_by_country.json")
                                    default_national_code_path = os.path.join(project_root, "stable_var", "cathode_national_code.json")
                                    
                                    if not electricity_coef and os.path.exists(default_electricity_path):
                                        with open(default_electricity_path, 'r', encoding='utf-8') as f:
                                            electricity_coef = json.load(f)
                                        if not user_id:
                                            st.info("📁 기본 전력 배출계수 데이터 사용")
                                    
                                    if not national_code_mapping and os.path.exists(default_national_code_path):
                                        with open(default_national_code_path, 'r', encoding='utf-8') as f:
                                            national_data = json.load(f)
                                            national_code_mapping = national_data.get('national_code', {})
                                
                                # 국가코드 매핑 확장 (헝가리 추가)
                                extended_national_mapping = national_code_mapping.copy()
                                extended_national_mapping['HU'] = '헝가리'
                                
                                # 국가코드 -> 국가명 -> 배출계수 매핑
                                site_emission_factors = {}
                                for code, country_name in extended_national_mapping.items():
                                    if country_name in electricity_coef:
                                        # 전력 배출계수 데이터가 있는 경우
                                        site_emission_factors[country_name] = {
                                            'code': code,
                                            'factor': electricity_coef[country_name],
                                            'description': f'{country_name} 전력 배출계수',
                                            'has_emission_data': True
                                        }
                                    else:
                                        # 헝가리의 경우: 배출계수 조정 없음, 기본값 사용
                                        if code == 'HU':
                                            site_emission_factors[country_name] = {
                                                'code': code,
                                                'factor': 0.637420635,  # 한국 기준값 사용 (조정 없음)
                                                'description': f'{country_name} (배출계수 조정 없음)',
                                                'has_emission_data': False
                                            }
                                
                                return site_emission_factors, extended_national_mapping
                                
                            except Exception as e:
                                st.warning(f"전력 배출계수 데이터 로드 오류: {e}")
                                # 오류 시 기본값 반환 (5개국 포함)
                                return {
                                    '한국': {'code': 'KR', 'factor': 0.637420635, 'description': '한국 전력 배출계수', 'has_emission_data': True},
                                    '중국': {'code': 'CN', 'factor': 0.8825, 'description': '중국 전력 배출계수', 'has_emission_data': True},
                                    '일본': {'code': 'JP', 'factor': 0.667861719, 'description': '일본 전력 배출계수', 'has_emission_data': True},
                                    '폴란드': {'code': 'PL', 'factor': 0.948984701, 'description': '폴란드 전력 배출계수', 'has_emission_data': True},
                                    '헝가리': {'code': 'HU', 'factor': 0.637420635, 'description': '헝가리 (배출계수 조정 없음)', 'has_emission_data': False}
                                }, {'KR': '한국', 'CN': '중국', 'JP': '일본', 'PL': '폴란드', 'HU': '헝가리'}
                        
                        site_emission_factors, national_code_mapping = load_emission_factor_data()
                        
                        # 변경 후 생산지 선택
                        available_sites = list(site_emission_factors.keys())
                        default_index = available_sites.index('한국') if '한국' in available_sites else 0
                        
                        target_site = st.selectbox(
                            "목표 생산지 (After)",
                            options=available_sites,
                            index=default_index,
                            help="변경할 생산지를 선택하세요. 각 지역의 실제 전력 배출계수가 적용됩니다.",
                            key="target_production_site"
                        )
                    
                    with col2:
                        
                        selected_info = site_emission_factors.get(target_site, {})
                        st.metric(
                            f"{target_site} 전력 배출계수",
                            f"{selected_info.get('factor', 0.0):.6f} kgCO2eq/kWh",
                            help=selected_info.get('description', '')
                        )
                        
                        # 전력 배출계수 조정 로직
                        has_emission_data = selected_info.get('has_emission_data', True)
                        
                        if has_emission_data:
                            # 배출계수 데이터가 있는 경우: 변화율 계산 및 표시
                            korea_factor = site_emission_factors.get('한국', {}).get('factor', 0.637420635)  # 실제 한국 기준값
                            target_factor = selected_info.get('factor', korea_factor)
                            change_rate = ((target_factor - korea_factor) / korea_factor) * 100
                            
                            if change_rate > 0:
                                st.error(f"⬆️ 전력 배출량 {change_rate:.1f}% 증가")
                            elif change_rate < 0:
                                st.success(f"⬇️ 전력 배출량 {abs(change_rate):.1f}% 감소")
                            else:
                                st.info("배출량 변화 없음")
                        else:
                            # 헝가리의 경우: 배출계수 조정 없음
                            st.warning("⚠️ 전력 배출계수 조정 데이터 없음 - 기본값 사용")
                    
                    # 생산지 설정 저장
                    if 'production_site' not in current_params:
                        current_params['production_site'] = {}
                    current_params['production_site']['target'] = target_site
                    current_params['production_site']['target_code'] = site_emission_factors.get(target_site, {}).get('code', 'KR')
                    current_params['production_site']['emission_factor'] = site_emission_factors.get(target_site, {}).get('factor', 0.637420635)
                
                # 최적화 시나리오 저장
                current_params['optimization_scenario'] = optimization_scenario
                
                # decision_vars 설정 추가 (필수 설정)
                if 'decision_vars' not in current_params:
                    current_params['decision_vars'] = {}
                
                # 양극재 타입 설정 (material_based에서는 B타입이 기본)
                if 'cathode' not in current_params['decision_vars']:
                    current_params['decision_vars']['cathode'] = {}
                current_params['decision_vars']['cathode']['type'] = 'B'  # B타입: 비율 최적화
                
                # 이진 변수 설정
                current_params['decision_vars']['use_binary_variables'] = False  # 연속 변수 사용
                
                # 그리드 서치 옵션
                st.markdown("---")
                st.markdown("### 그리드 서치 옵션")
                use_grid_search = st.checkbox(
                    "그리드 서치 사용", 
                    value=current_params.get('use_grid_search', False),
                    help="다양한 파라미터 조합을 탐색하여 최적의 조합을 찾습니다",
                    key="use_grid_search_material"
                )
                current_params['use_grid_search'] = use_grid_search
            else:
                st.warning("⚠️ 시뮬레이션 데이터를 먼저 로드해주세요.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 프리미엄 비용 제약조건 설정
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.markdown('<div class="config-title">프리미엄 비용 제약조건</div>', unsafe_allow_html=True)
            
            # 프리미엄 비용 제약 활성화 체크박스
            premium_cost_enabled = st.checkbox(
                "프리미엄 비용 제약조건 활성화", 
                value=current_params.get('premium_cost', {}).get('enabled', False),
                help="프리미엄 비용에 대한 제약조건을 적용합니다",
                key="premium_cost_enabled"
            )
            
            if premium_cost_enabled:
                # 프리미엄 비용 계산 함수
                def load_premium_cost_data():
                    """pcf_ref_cost_for_cert.csv 파일을 로드하고 전처리합니다."""
                    try:
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.join(current_dir, "..")
                        cost_file_path = os.path.join(project_root, "data", "pcf_ref_cost_for_cert.csv")
                        
                        if os.path.exists(cost_file_path):
                            cost_df = pd.read_csv(cost_file_path, encoding='utf-8-sig')
                            # 컬럼명 정리 (BOM 제거 등)
                            cost_df.columns = cost_df.columns.str.strip()
                            return cost_df
                        else:
                            st.warning(f"프리미엄 비용 데이터 파일이 없습니다: {cost_file_path}")
                            return pd.DataFrame()
                    except Exception as e:
                        st.warning(f"프리미엄 비용 데이터 로드 오류: {e}")
                        return pd.DataFrame()
                
                def get_material_mapping():
                    """자재품목과 소재 구분 매핑 테이블"""
                    return {
                        '양극재': '양극재',
                        '음극재': '음극재(천연)',  # 기본값, 인조도 가능
                        'Al Foil': '동박',
                        'Cu Foil': '동박',
                        '동박': '동박',
                        '분리막': '분리막_코팅',
                        '전해액': None  # 데이터 없음
                    }
                
                def check_token_based_match(material_name: str, target_name: str) -> bool:
                    """
                    토큰 기반 매칭: target_name의 모든 토큰이 material_name에 포함되는지 확인
                    (rule_based.py에서 추출한 로직)
                    
                    Args:
                        material_name: 원본 자재명 (소문자)
                        target_name: 매핑 대상 자재명 (소문자)
                        
                    Returns:
                        bool: 모든 토큰이 포함되면 True, 아니면 False
                    """
                    if not material_name or not target_name:
                        return False
                    
                    material_tokens = set(material_name.lower().split())
                    target_tokens = set(target_name.lower().split())
                    
                    # target_name의 모든 토큰이 material_name에 포함되는지 확인
                    return len(target_tokens) > 0 and target_tokens.issubset(material_tokens)
                
                def check_material_category_match(category1: str, category2: str) -> bool:
                    """
                    자재품목 일치 확인 (부분 일치 포함)
                    (rule_based.py에서 추출한 로직)
                    
                    Args:
                        category1: 첫 번째 자재품목 (소문자)
                        category2: 두 번째 자재품목 (소문자)
                        
                    Returns:
                        bool: 일치하면 True, 아니면 False
                    """
                    if not category1 or not category2:
                        return False
                    
                    category1 = category1.lower()
                    category2 = category2.lower()
                    
                    # 정확한 일치 확인
                    if category1 == category2:
                        return True
                    
                    # 부분 일치 확인 (예: "al foil" vs "foil")
                    if category1 in category2 or category2 in category1:
                        return True
                    
                    return False
                
                def enhanced_material_mapping(material_name: str, material_category: str, cost_ref_df, debug_mode: bool = False) -> str:
                    """
                    향상된 자재 매핑 로직 (rule_based.py 방식 적용)
                    
                    Args:
                        material_name: 자재명
                        material_category: 자재품목
                        cost_ref_df: 비용 참조 데이터프레임
                        
                    Returns:
                        str: 매칭된 소재 구분, 없으면 None
                    """
                    import pandas as pd
                    
                    # NaN 값 처리
                    if pd.isna(material_name):
                        material_name = ''
                    if pd.isna(material_category):
                        material_category = ''
                    
                    material_name_lower = str(material_name).lower()
                    material_category_lower = str(material_category).lower()
                    
                    if debug_mode:
                        print(f"🔍 [Enhanced Mapping] 자재명: '{material_name}', 자재품목: '{material_category}'")
                    
                    # 특별 케이스 처리 - 음극재 (artificial/natural 구분)
                    if material_category_lower == '음극재':
                        if 'artificial' in material_name_lower:
                            if debug_mode:
                                print(f"  ✅ 음극재(인조) 매칭 성공 (Artificial 감지)")
                            return '음극재(인조)'
                        elif 'natural' in material_name_lower:
                            if debug_mode:
                                print(f"  ✅ 음극재(천연) 매칭 성공 (Natural 감지)")
                            return '음극재(천연)'
                        else:
                            # 기본값은 천연
                            if debug_mode:
                                print(f"  ✅ 음극재(천연) 매칭 성공 (기본값)")
                            return '음극재(천연)'
                    
                    # 특별 케이스 처리 - 양극재
                    if material_category_lower == '양극재':
                        # 자재명이 빈값/NaN이거나 cathode 관련 이름인 경우
                        if (material_name_lower in ['', 'nan', 'n/a'] or 
                            'cathode' in material_name_lower):
                            return '양극재'
                        else:
                            return '양극재'
                    
                    # cost_ref_df의 '소재 구분' 컬럼을 순회하며 매칭 시도
                    if '소재 구분' in cost_ref_df.columns:
                        for idx, row in cost_ref_df.iterrows():
                            ref_material = str(row['소재 구분']).lower()
                            
                            # 1단계: 정확한 포함 관계 확인
                            if (ref_material in material_name_lower or 
                                material_name_lower in ref_material or
                                ref_material in material_category_lower or 
                                material_category_lower in ref_material):
                                return row['소재 구분']
                            
                            # 2단계: 토큰 기반 매칭
                            if (check_token_based_match(material_name_lower, ref_material) or
                                check_token_based_match(material_category_lower, ref_material)):
                                return row['소재 구분']
                    
                    # 기본 매핑 테이블로 폴백
                    basic_mapping = get_material_mapping()
                    for key, value in basic_mapping.items():
                        if key.lower() in material_category_lower:
                            return value
                    
                    # 매칭 실패
                    return None
                
                def calculate_premium_cost(scenario_df, cost_ref_df, production_site='KR', re_ratio=1.0):
                    """
                    RE 적용에 따른 프리미엄 비용 계산
                    - 저감활동_적용여부 == 1인 자재만 대상
                    - RE 100% 기준으로 계산 (re_ratio=1.0)
                    """
                    if scenario_df.empty or cost_ref_df.empty:
                        return 0.0, []
                    
                    material_mapping = get_material_mapping()
                    total_premium_cost = 0.0
                    cost_breakdown = []
                    
                    # 저감활동이 적용된 자재만 필터링
                    applied_materials = scenario_df[scenario_df.get('저감활동_적용여부', 0) == 1.0]
                    
                    for idx, row in applied_materials.iterrows():
                        material_name = row.get('자재명', '')
                        material_category = row.get('자재품목', '')
                        quantity = row.get('제품총소요량(kg)', 0)
                        
                        if quantity <= 0:
                            continue
                        
                        # 향상된 자재 매핑 로직 사용 (rule_based.py 방식 적용)
                        # 디버그 모드는 개발환경에서만 활성화
                        debug_mode = False  # 프로덕션에서는 False로 설정
                        mapped_material = enhanced_material_mapping(material_name, material_category, cost_ref_df, debug_mode)
                        
                        if not mapped_material:
                            continue  # 매핑되지 않은 자재는 건너뛰기
                        
                        # 해당 소재와 국가에 맞는 비용 찾기 (향상된 매칭)
                        cost_match = pd.DataFrame()
                        
                        # 1단계: 정확한 일치 확인
                        exact_match = cost_ref_df[
                            (cost_ref_df['소재 구분'] == mapped_material) &
                            (cost_ref_df['국가'] == production_site)
                        ]
                        
                        if not exact_match.empty:
                            cost_match = exact_match
                        else:
                            # 2단계: 부분 일치 확인 (contains 사용)
                            partial_match = cost_ref_df[
                                (cost_ref_df['소재 구분'].str.contains(mapped_material, na=False)) &
                                (cost_ref_df['국가'] == production_site)
                            ]
                            
                            if not partial_match.empty:
                                cost_match = partial_match
                            else:
                                # 3단계: 토큰 기반 매칭
                                for idx, cost_row in cost_ref_df[cost_ref_df['국가'] == production_site].iterrows():
                                    cost_material = str(cost_row['소재 구분'])
                                    if check_token_based_match(cost_material.lower(), mapped_material.lower()):
                                        cost_match = pd.DataFrame([cost_row])
                                        break
                        
                        if not cost_match.empty:
                            # 예상 비용 컬럼에서 값 가져오기
                            unit_premium_cost = cost_match.iloc[0]['예상 비용 ($/kg, $/m2)']
                            
                            # RE 100% 기준 프리미엄 비용 계산
                            material_premium = quantity * unit_premium_cost * re_ratio
                            total_premium_cost += material_premium
                            
                            cost_breakdown.append({
                                '자재명': material_name,
                                '자재품목': material_category,
                                '소재구분': mapped_material,
                                '소요량(kg)': quantity,
                                '단위비용($/kg)': unit_premium_cost,
                                'RE비율': f"{re_ratio*100:.1f}%",
                                '프리미엄비용($)': material_premium
                            })
                    
                    return total_premium_cost, cost_breakdown
                
                # 프리미엄 비용 데이터 로드
                cost_ref_df = load_premium_cost_data()
                
                # 기준 프리미엄 비용 계산
                calculated_baseline_cost = 0.0
                cost_breakdown = []
                
                if st.session_state.get('material_simulation_data_loaded', False) and st.session_state.get('material_simulation_data'):
                    try:
                        sim_data = st.session_state.material_simulation_data
                        scenario_df = sim_data.get('scenario_df', pd.DataFrame())
                        
                        # 현재 생산지 설정 가져오기 (생산지 변경 시나리오 고려)
                        current_site = 'KR'  # 기본값
                        if optimization_scenario in ['site_change', 'both']:
                            current_site = current_params.get('production_site', {}).get('target_code', 'KR')
                        
                        if not scenario_df.empty:
                            calculated_baseline_cost, cost_breakdown = calculate_premium_cost(
                                scenario_df, cost_ref_df, current_site, re_ratio=1.0
                            )
                    except Exception as e:
                        st.warning(f"프리미엄 비용 계산 오류: {e}")
                        calculated_baseline_cost = 0.0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 계산된 기준 프리미엄 비용 표시 (RE 100% 기준)
                    st.metric(
                        "기준 프리미엄 비용 (RE 100%)", 
                        f"${calculated_baseline_cost:.6f}",
                        help="저감활동 적용 자재의 RE 100% 적용 시 총 프리미엄 비용"
                    )
                    
                    # 생산지 정보 표시
                    if optimization_scenario in ['site_change', 'both']:
                        display_site = current_params.get('production_site', {}).get('target', '한국')
                        site_code = current_params.get('production_site', {}).get('target_code', 'KR')
                        st.info(f"📍 적용 생산지: {display_site} ({site_code})")
                    else:
                        st.info(f"📍 적용 생산지: 한국 (KR)")
                
                with col2:
                    # 프리미엄 비용 감축률 설정
                    reduction_target = st.slider(
                        "프리미엄 비용 감축 목표 (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_params.get('premium_cost', {}).get('reduction_target', 20.0)),
                        step=5.0,
                        format="%.1f",
                        help="프리미엄 비용 감축 목표 비율. RE 적용률을 줄여서 달성됩니다."
                    )
                    
                    # 목표 프리미엄 비용 표시
                    target_cost = calculated_baseline_cost * (1 - reduction_target / 100)
                    st.metric(
                        "목표 프리미엄 비용", 
                        f"${target_cost:.6f}",
                        delta=f"-${calculated_baseline_cost - target_cost:.6f}",
                        help="감축 목표 달성 시 프리미엄 비용"
                    )
                
                # 자재별 프리미엄 비용 상세 정보 (expander)
                if cost_breakdown:
                    with st.expander(f"자재별 프리미엄 비용 상세 ({len(cost_breakdown)}개 자재)", expanded=False):
                        breakdown_df = pd.DataFrame(cost_breakdown)
                        st.dataframe(breakdown_df, use_container_width=True, height=300)
                        
                        # 요약 정보
                        st.markdown("**요약 정보:**")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            total_quantity = breakdown_df['소요량(kg)'].sum()
                            st.metric("총 자재 소요량", f"{total_quantity:.3f} kg")
                        with col_b:
                            avg_unit_cost = breakdown_df['단위비용($/kg)'].mean()
                            st.metric("평균 단위 비용", f"${avg_unit_cost:.3f}/kg")
                        with col_c:
                            total_premium = breakdown_df['프리미엄비용($)'].sum()
                            st.metric("총 프리미엄 비용", f"${total_premium:.6f}")
                
                # 설명
                st.info("📌 **프리미엄 비용 제약조건**: RE 인증서 비용을 기반으로 계산됩니다. 감축 목표는 RE 적용률을 조정하여 달성됩니다.")
                
                # 설정 저장
                if 'premium_cost' not in current_params:
                    current_params['premium_cost'] = {}
                current_params['premium_cost']['enabled'] = premium_cost_enabled
                current_params['premium_cost']['baseline_cost'] = calculated_baseline_cost
                current_params['premium_cost']['reduction_target'] = reduction_target
                current_params['premium_cost']['production_site'] = current_site
            else:
                # 비활성화 상태에서도 설정 저장
                if 'premium_cost' not in current_params:
                    current_params['premium_cost'] = {}
                current_params['premium_cost']['enabled'] = False
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 솔버 설정
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<div class="config-title">솔버 설정</div>', unsafe_allow_html=True)
        
        # 사용 가능한 솔버 목록
        available_solvers = get_available_solvers()
        
        # 현재 선택된 솔버
        current_solver = st.session_state.optimization_config.get('solver', 'glpk')
        
        # 시나리오 유형에 따른 추천 솔버
        recommended_solver = select_optimal_solver({
            'objective': current_scenario.replace('_', '-'),
            'constraints': current_params
        })
        
        st.markdown(f"💡 추천 솔버: **{recommended_solver.upper()}** (현재 시나리오 유형에 최적화됨)")
        
        # 솔버 선택 UI
        solver_cols = st.columns(len(available_solvers))
        
        for i, solver_name in enumerate(available_solvers):
            with solver_cols[i]:
                solver_info = {
                    'glpk': {
                        'name': 'GLPK',
                        'icon': '📊',
                        'desc': '선형 계획법(LP)에 최적화된 솔버'
                    },
                    'ipopt': {
                        'name': 'IPOPT',
                        'icon': '🔄',
                        'desc': '비선형 최적화 문제 해결'
                    },
                    'cbc': {
                        'name': 'CBC',
                        'icon': '🧮',
                        'desc': '정수 계획법(MIP)에 최적화'
                    }
                }.get(solver_name, {'name': solver_name.upper(), 'icon': '🔧', 'desc': '일반 최적화 솔버'})
                
                st.markdown(f"""
                <div class="solver-card {'selected' if solver_name == current_solver else ''}" 
                     onclick="document.querySelector('#select_{solver_name}').click()">
                    <h4>{solver_info['icon']} {solver_info['name']}</h4>
                    <p style="font-size: 0.8rem;">{solver_info['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 숨겨진 버튼 (스타일링된 카드와 연동)
                if st.button(f"Select {solver_name}", key=f"select_{solver_name}"):
                    st.session_state.optimization_config['solver'] = solver_name
                    st.success(f"✓ {solver_name.upper()} 솔버가 선택되었습니다!")
                    st.rerun()
        
        # 고급 옵션
        with st.expander("고급 옵션"):
            # 솔버 비교 모드
            compare_solvers = st.checkbox(
                "솔버 비교 모드", 
                value=st.session_state.optimization_config.get('advanced_options', {}).get('compare_solvers', False),
                help="모든 솔버로 최적화를 실행하여 결과를 비교합니다"
            )
            
            # 시간 제한
            time_limit = st.number_input(
                "시간 제한 (초)", 
                min_value=10,
                max_value=3600,
                value=st.session_state.optimization_config.get('advanced_options', {}).get('time_limit', 300),
                step=10,
                help="최적화 실행 시간 제한 (초 단위)"
            )
            
            # 갭 허용치
            gap_tolerance = st.number_input(
                "갭 허용치 (%)", 
                min_value=0.0001,
                max_value=10.0,
                value=st.session_state.optimization_config.get('advanced_options', {}).get('gap_tolerance', 0.01),
                format="%.4f",
                step=0.001,
                help="최적해와의 허용 오차 (낮을수록 더 정확한 결과)"
            )
            
            # 고급 옵션 업데이트
            if 'advanced_options' not in st.session_state.optimization_config:
                st.session_state.optimization_config['advanced_options'] = {}
            
            st.session_state.optimization_config['advanced_options']['compare_solvers'] = compare_solvers
            st.session_state.optimization_config['advanced_options']['time_limit'] = time_limit
            st.session_state.optimization_config['advanced_options']['gap_tolerance'] = gap_tolerance
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 설정 저장 및 최적화 실행 버튼
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # 설정 저장 버튼
            if st.button("💾 설정 저장", use_container_width=True):
                try:
                    # 설정 유효성 검사
                    config = st.session_state.optimization_config
                    if not config.get('scenario_type'):
                        st.warning("⚠️ 시나리오가 선택되지 않았습니다. 저장 전에 시나리오를 선택해주세요.")
                    elif not config.get('scenario_params', {}).get(config['scenario_type']):
                        st.warning("⚠️ 시나리오 파라미터가 설정되지 않았습니다. 기본값으로 저장됩니다.")
                        
                    # 저장 실행
                    if save_optimization_config(config, user_id=user_id):
                        st.success("✅ 최적화 설정이 성공적으로 저장되었습니다!")
                        
                        # 저장된 설정 요약 표시
                        scenario_type = config.get('scenario_type', 'none')
                        solver = config.get('solver', 'glpk')
                        st.info(f"📋 저장된 설정: {scenario_type} 시나리오, {solver.upper()} 솔버")
                        log_info(f"최적화 설정 저장됨 (user_id: {user_id}, scenario: {scenario_type})")
                    else:
                        st.error("❌ 설정 저장 중 오류가 발생했습니다.")
                        st.markdown("""
                        **가능한 원인**:
                        - 파일 쓰기 권한 부족
                        - 디스크 공간 부족
                        - 네트워크 연결 문제
                        
                        **해결 방법**: 잠시 후 다시 시도하거나 관리자에게 문의하세요.
                        """)
                except Exception as e:
                    st.error(f"❌ 설정 저장 실패: {str(e)}")
                    st.markdown("**해결 방법**: 페이지를 새로고침하고 설정을 다시 구성해주세요.")
        
        with col3:
            # 설정 초기화 버튼
            if st.button("🔄 초기화", use_container_width=True):
                st.session_state.optimization_config = get_default_optimization_config()
                st.success("✅ 설정이 초기화되었습니다!")
                st.rerun()
        
        with col2:
            # 최적화 실행 버튼
            if st.button("🚀 최적화 실행", type="primary", use_container_width=True):
                log_button_click("optimization_run", "run_optimization_btn")
                
                # 실행 전 설정 저장
                if save_optimization_config(st.session_state.optimization_config, user_id=user_id):
                    log_info("최적화 실행 전 설정 자동 저장 완료")
                
                # 최적화 실행
                with st.spinner("최적화 계산 중..."):
                    try:
                        # 현재 설정
                        config = st.session_state.optimization_config
                        scenario_type = config['scenario_type']
                        solver = config['solver']
                        scenario_params = config['scenario_params'].get(scenario_type, {})
                        compare_solvers = config.get('advanced_options', {}).get('compare_solvers', False)
                        
                        # 시간 측정 시작
                        start_time = datetime.now()
                        
                        # 자재 기반 최적화 (신규 시나리오)
                        if scenario_type == 'material_based':
                            
                            from src.optimization.material_based_optimizer import MaterialBasedOptimizer
                            from src.optimization.grid_search_optimizer import GridSearchOptimizer
                            
                            # 시뮬레이션 데이터 확인
                            st.write(f"🔍 **DEBUG**: 시뮬레이션 데이터 확인 - material_simulation_data_loaded: {st.session_state.get('material_simulation_data_loaded', False)}")
                            if 'material_simulation_data_loaded' in st.session_state and st.session_state.material_simulation_data_loaded:
                                simulation_data = st.session_state.material_simulation_data
                                
                                # 그리드 서치 사용 여부 확인
                                use_grid_search = scenario_params.get('use_grid_search', False)
                                
                                if use_grid_search and 'grid_search_params' in scenario_params:
                                    # 그리드 서치 최적화
                                    with st.status("그리드 서치 최적화 진행 중...", expanded=True) as status:
                                        # 그리드 서치 파라미터 설정
                                        grid_params = scenario_params['grid_search_params']
                                        param_selection = grid_params.get('param_selection', {})
                                        max_iterations = grid_params.get('max_iterations', 100)
                                        
                                        if not param_selection:
                                            status.update(label="❌ 그리드 서치 파라미터가 설정되지 않았습니다.", state="error")
                                            st.error("그리드 서치 파라미터를 설정해주세요.")
                                            results = {'status': 'error', 'message': '그리드 서치 파라미터 설정 필요'}
                                        else:
                                            # 그리드 서치 실행
                                            status.update(label="▶️ 그리드 서치 초기화 중...")
                                            
                                            # 기본 설정 구성 (동적 tier 지원)
                                            dynamic_re_rates = {}
                                            for tier in range(1, num_tier + 1):
                                                dynamic_re_rates[f'tier{tier}'] = {'min': 0.1, 'max': 0.9}
                                            
                                            base_config = {
                                                'reduction_target': scenario_params.get('reduction_target', {'min': -10, 'max': -5}),
                                                're_rates': scenario_params.get('re_rates', dynamic_re_rates),
                                                'material_ratios': scenario_params.get('material_ratios', {
                                                    'recycle': {'min': 0.05, 'max': 0.5},
                                                    'low_carbon': {'min': 0.05, 'max': 0.3},
                                                    'max_total': 0.7
                                                })
                                            }
                                            
                                            # 그리드서치 최적화 객체 생성
                                            grid_optimizer = GridSearchOptimizer(
                                                simulation_data=simulation_data,
                                                base_config=base_config,
                                                debug_mode=False
                                            )
                                            
                                            # 그리드 파라미터 설정
                                            grid_optimizer.set_grid_params(param_selection)
                                            
                                            # 진행 상황 표시 함수
                                            progress_bar = st.progress(0)
                                            
                                            def update_progress(current, total):
                                                progress = current / total
                                                progress_bar.progress(progress)
                                                status.update(label=f"▶️ 그리드 서치 진행 중... ({current}/{total})")
                                            
                                            # 그리드 서치 실행
                                            status.update(label="▶️ 그리드 서치 실행 중...")
                                            grid_results = grid_optimizer.run_grid_search(
                                                max_iterations=max_iterations,
                                                progress_callback=update_progress
                                            )
                                            
                                            # 파레토 최적해 결과
                                            status.update(label="▶️ 파레토 최적해 계산 중...")
                                            pareto_results = grid_optimizer.get_pareto_results()
                                            
                                            # 최적해 선택 (가장 좋은 PCF 결과)
                                            best_result = grid_optimizer.get_best_result_by_pcf()
                                            
                                            # 결과 저장
                                            if grid_params.get('visualize_results', True):
                                                # 시각화 데이터 생성
                                                status.update(label="▶️ 결과 시각화 준비 중...")
                                                visualization_data = grid_optimizer.generate_visualization_data()
                                                
                                                # 최적화 결과에 시각화 데이터 추가
                                                best_result['visualization_data'] = visualization_data
                                            
                                            # CSV 결과 저장 옵션이 있는 경우
                                            try:
                                                if 'save_results' in grid_params and grid_params['save_results'].get('to_csv', False):
                                                    status.update(label="▶️ 결과를 CSV 파일로 저장 중...")
                                                    if grid_params['save_results'].get('pareto_only', False):
                                                        csv_path = grid_optimizer.export_pareto_results_to_csv()
                                                        st.success(f"파레토 최적해가 {csv_path}에 저장되었습니다.")
                                                    else:
                                                        csv_path = grid_optimizer.export_results_to_csv()
                                                        st.success(f"모든 그리드 서치 결과가 {csv_path}에 저장되었습니다.")
                                            except Exception as e:
                                                st.warning(f"CSV 파일 저장 중 오류 발생: {e}")
                                            
                                            # 최적화 성공
                                            status.update(label="✅ 그리드 서치 최적화 완료", state="complete")
                                            
                                            # 그리드 서치 결과 요약
                                            st.markdown(f"### 그리드 서치 결과 요약")
                                            st.markdown(f"총 {len(grid_results)}개의 유효한 조합을 탐색했습니다.")
                                            st.markdown(f"파레토 최적해: {len(pareto_results)}개")
                                            
                                            # 최적 조합 정보 표시
                                            if best_result.get('status') == 'optimal':
                                                st.markdown(f"#### 최적 조합 (최소 PCF 기준)")
                                                
                                                # 파라미터 요약
                                                params = best_result.get('params', {})
                                                col1, col2, col3, col4 = st.columns(4)
                                                with col1:
                                                    if 'reduction_min' in params:
                                                        st.metric("최소 감축률", f"{params['reduction_min']:.1f}%")
                                                with col2:
                                                    if 'reduction_max' in params:
                                                        st.metric("최대 감축률", f"{params['reduction_max']:.1f}%")
                                                with col3:
                                                    if 'tier1_re' in params:
                                                        st.metric("Tier1 RE 적용률", f"{params['tier1_re']:.2f}")
                                                with col4:
                                                    if 'recycle_ratio' in params:
                                                        st.metric("재활용 비율", f"{params['recycle_ratio']:.2f}")
                                            
                                            # 최종 결과 설정
                                            results = best_result
                                                
                                else:
                                    # 자재별 개별 최적화 + 결과 합산
                                    
                                    with st.status("자재별 최적화 진행 중...", expanded=True) as status:
                                        status.update(label="▶️ 자재별 최적화 초기화 중...")
                                        
                                        # 시뮬레이션 데이터에서 저감활동 적용자재만 추출
                                        scenario_df = simulation_data.get('scenario_df', pd.DataFrame())
                                        
                                        st.write(f"📊 DEBUG: scenario_df 상태 - shape: {scenario_df.shape if not scenario_df.empty else 'Empty'}")
                                        if not scenario_df.empty:
                                            st.write(f"📊 DEBUG: scenario_df 컬럼: {list(scenario_df.columns)}")
                                            if '자재품목' in scenario_df.columns:
                                                unique_materials = scenario_df['자재품목'].unique()
                                                st.write(f"📊 DEBUG: 시나리오 자재 목록 ({len(unique_materials)}개): {list(unique_materials)}")
                                        
                                        if scenario_df.empty or '자재품목' not in scenario_df.columns or '저감활동_적용여부' not in scenario_df.columns:
                                            status.update(label="❌ 유효한 자재 데이터가 없습니다.", state="error")
                                            st.write("❌ DEBUG: 자재 데이터 검증 실패")
                                            results = {'status': 'error', 'message': '자재 데이터 없음'}
                                        else:
                                            # 저감활동이 적용된 자재만 필터링
                                            applied_materials_df = scenario_df[scenario_df['저감활동_적용여부'] == 1.0]
                                            
                                            if applied_materials_df.empty:
                                                status.update(label="❌ 저감활동이 적용된 자재가 없습니다.", state="error")
                                                results = {'status': 'error', 'message': '저감활동 적용 자재 없음'}
                                            else:
                                                unique_materials = applied_materials_df['자재품목'].unique().tolist()
                                                total_materials = len(unique_materials)
                                                
                                                status.update(label=f"▶️ {total_materials}개 저감활동 적용 자재 개별 최적화 시작...")
                                                
                                                # 자재별 최적화 결과 저장
                                                material_results = {}
                                                total_optimized_emission = 0.0
                                                total_baseline_emission = 0.0
                                                total_cost = 0.0
                                                optimization_success_count = 0
                                                
                                                # 진행 상황 표시
                                                progress_bar = st.progress(0)
                                                
                                                # 자재별 순차 최적화
                                                for i, material in enumerate(unique_materials):
                                                    progress = (i + 1) / total_materials
                                                    progress_bar.progress(progress)
                                                    status.update(label=f"▶️ {material} 최적화 중... ({i+1}/{total_materials})")
                                                    
                                                    # 해당 자재의 데이터만 추출 (저감활동 적용자재에서)
                                                    material_data = applied_materials_df[applied_materials_df['자재품목'] == material].copy()
                                                    
                                                    if material_data.empty:
                                                        continue
                                                    
                                                    # 자재별 설정 가져오기
                                                    material_targets = scenario_params.get('material_specific_targets', {}).get(material, {})
                                                    material_re_rates = scenario_params.get('material_specific_re_rates', {}).get(material, {})
                                                    
                                                    # 기본값 설정 (설정이 없는 경우)
                                                    if not material_targets:
                                                        default_values = {
                                                            '양극재': {'min': 15.0, 'max': 25.0},
                                                            '분리막': {'min': 5.0, 'max': 10.0},
                                                            '전해액': {'min': 8.0, 'max': 15.0},
                                                            '음극재': {'min': 10.0, 'max': 18.0},
                                                            '동박': {'min': 5.0, 'max': 10.0},
                                                            'Al Foil': {'min': 5.0, 'max': 10.0},
                                                            'Cu Foil': {'min': 5.0, 'max': 10.0}
                                                        }
                                                        material_targets = default_values.get(material, {'min': 5.0, 'max': 10.0})
                                                    
                                                    if not material_re_rates:
                                                        # 동적 tier 수를 고려한 기본값 생성 함수 재사용
                                                        current_num_tier_for_sample = st.session_state.get('num_tier', 2)
                                                        
                                                        def get_sample_material_re_defaults(material, num_tiers):
                                                            """샘플 데이터용 자재별 RE 기본값 생성"""
                                                            base_values = {
                                                                '양극재': {'base_min': 0.2, 'base_max': 0.4, 'decrease_rate': 0.05},
                                                                '분리막': {'base_min': 0.25, 'base_max': 0.45, 'decrease_rate': 0.05},
                                                                '전해액': {'base_min': 0.3, 'base_max': 0.5, 'decrease_rate': 0.05},
                                                                '음극재': {'base_min': 0.25, 'base_max': 0.45, 'decrease_rate': 0.05},
                                                                '동박': {'base_min': 0.2, 'base_max': 0.4, 'decrease_rate': 0.05}
                                                            }
                                                            
                                                            material_base = base_values.get(material, {
                                                                'base_min': 0.3, 'base_max': 0.5, 'decrease_rate': 0.05
                                                            })
                                                            
                                                            tier_values = {}
                                                            for tier in range(1, num_tiers + 1):
                                                                decrease_factor = (tier - 1) * material_base['decrease_rate']
                                                                tier_min = max(0.05, material_base['base_min'] - decrease_factor)
                                                                tier_max = max(0.1, material_base['base_max'] - decrease_factor)
                                                                
                                                                tier_values[f'tier{tier}'] = {
                                                                    'min': round(tier_min, 2),
                                                                    'max': round(tier_max, 2)
                                                                }
                                                            
                                                            return tier_values
                                                        
                                                        re_default_values = {}
                                                        for mat_name in ['양극재', '분리막', '전해액', '음극재', '동박']:
                                                            re_default_values[mat_name] = get_sample_material_re_defaults(mat_name, current_num_tier_for_sample)
                                                        material_re_rates = re_default_values.get(material, 
                                                            get_sample_material_re_defaults(material, current_num_tier_for_sample)
                                                        )
                                                    
                                                    # 현재 자재의 기준 배출량 계산
                                                    baseline_emission = 0.0
                                                    if '배출량(kgCO2eq)' in material_data.columns:
                                                        baseline_emission = material_data['배출량(kgCO2eq)'].sum()
                                                        st.write(f"🔍 **DEBUG - {material} 배출량 계산:** 배출량(kgCO2eq) 컬럼 사용 = {baseline_emission:.4f}")
                                                    elif '배출계수' in material_data.columns and '제품총소요량(kg)' in material_data.columns:
                                                        emission_factors = material_data['배출계수']
                                                        quantities = material_data['제품총소요량(kg)']
                                                        baseline_emission = (emission_factors * quantities).sum()
                                                        st.write(f"🔍 **DEBUG - {material} 배출량 계산:** 배출계수 × 제품총소요량 = {baseline_emission:.4f}")
                                                        st.write(f"  - 배출계수 범위: {emission_factors.min():.4f} ~ {emission_factors.max():.4f}")
                                                        st.write(f"  - 제품총소요량 범위: {quantities.min():.4f} ~ {quantities.max():.4f}")
                                                    else:
                                                        st.warning(f"⚠️ **{material}**: 배출량 계산에 필요한 컬럼이 없습니다.")
                                                    
                                                    # 자재별 최적화 설정 구성 - UI 감축 목표 동적 반영
                                                    if baseline_emission > 0:
                                                        # UI에서 설정한 감축률 가져오기 (더 엄격한 max 감축률 사용)
                                                        max_reduction_rate = material_targets.get('max', 10.0)  # 기본값 10%
                                                        # 감축률을 적용한 목표 탄소 제한 계산
                                                        target_carbon_limit = baseline_emission * (1 - max_reduction_rate/100)
                                                    else:
                                                        target_carbon_limit = 100.0
                                                    
                                                    material_config = {
                                                        'reduction_target': material_targets,
                                                        're_rates': material_re_rates,
                                                        'material_ratios': scenario_params.get('material_ratios', {
                                                            'recycle': {'min': 0.0, 'max': 0.8},    # 더 넓은 범위 허용
                                                            'low_carbon': {'min': 0.0, 'max': 0.5}  # 더 넓은 범위 허용
                                                        }),
                                                        'optimization_scenario': scenario_params.get('optimization_scenario', 'baseline'),
                                                        # 호환성을 위한 최적화 설정 추가
                                                        'objective': 'minimize_carbon',
                                                        'decision_vars': {
                                                            'cathode': {
                                                                'type': 'B',
                                                                'type_B_config': {
                                                                    'emission_fixed': 10.0,
                                                                    'recycle_range': [0.1, 0.5],
                                                                    'low_carbon_range': [0.05, 0.3]
                                                                }
                                                            },
                                                            'use_binary_variables': False
                                                        },
                                                        'constraints': {
                                                            'target_carbon': target_carbon_limit,  # UI 감축 목표를 반영한 동적 탄소 제한
                                                            'max_cost': 500000.0  # 비용 제한을 더 여유롭게
                                                        }
                                                    }
                                                    
                                                    # 디버그: 자재별 설정 상세 로그
                                                    st.write(f"🔍 **DEBUG - {material} 최적화 설정 상세:**")
                                                    st.write(f"  - 기준 배출량: {baseline_emission:.4f} kgCO2eq")
                                                    st.write(f"  - 목표 탄소 제한: {target_carbon_limit:.4f} kgCO2eq")
                                                    st.write(f"  - 감축 목표: {material_targets.get('min', 'N/A')}% ~ {material_targets.get('max', 'N/A')}%")
                                                    st.write(f"  - 자재 데이터 행 수: {len(material_data)}")
                                                    st.write(f"  - 최적화 시나리오: {material_config['optimization_scenario']}")
                                                    
                                                    # RE 적용률 상세 로그
                                                    if material_re_rates:
                                                        st.write(f"  - RE 적용률:")
                                                        for tier, rates in material_re_rates.items():
                                                            if isinstance(rates, dict):
                                                                st.write(f"    • {tier}: {rates.get('min', 'N/A')} ~ {rates.get('max', 'N/A')}")
                                                    
                                                    # 자재 비율 상세 로그  
                                                    material_ratios = material_config['material_ratios']
                                                    st.write(f"  - 자재 비율:")
                                                    st.write(f"    • 재활용: {material_ratios['recycle']['min']} ~ {material_ratios['recycle']['max']}")
                                                    st.write(f"    • 저탄소메탈: {material_ratios['low_carbon']['min']} ~ {material_ratios['low_carbon']['max']}")
                                                    
                                                    # 최적화가 가능한지 사전 체크
                                                    if baseline_emission == 0:
                                                        st.error(f"❌ **{material}**: 기준 배출량이 0이므로 최적화를 건너뜁니다.")
                                                        continue
                                                    
                                                    # 자재별 시뮬레이션 데이터 구성
                                                    material_simulation_data = {
                                                        'scenario_df': material_data,
                                                        'ref_formula_df': simulation_data.get('ref_formula_df', pd.DataFrame()),
                                                        'ref_proportions_df': simulation_data.get('ref_proportions_df', pd.DataFrame()),
                                                        'original_df': simulation_data.get('original_df', pd.DataFrame())
                                                    }
                                                    
                                                    try:
                                                        # 디버그: 자재별 설정 로그
                                                        st.write(f"🔍 **DEBUG - {material} 자재 설정:**")
                                                        st.json({
                                                            "material_name": material,
                                                            "has_decision_vars": "decision_vars" in material_config,
                                                            "has_objective": "objective" in material_config,
                                                            "has_constraints": "constraints" in material_config,
                                                            "config_keys": list(material_config.keys())
                                                        })
                                                        
                                                        # 실행가능성 사전 검사
                                                        def check_feasibility(material_config, material_name):
                                                            """최적화 실행 전 실행가능성 검사"""
                                                            warnings = []
                                                            
                                                            # 감축 목표 검사
                                                            reduction_target = material_config.get('reduction_target', {})
                                                            min_reduction = reduction_target.get('min', 0)
                                                            max_reduction = reduction_target.get('max', 0)
                                                            
                                                            if abs(min_reduction) > 50 or abs(max_reduction) > 50:
                                                                warnings.append(f"감축 목표가 너무 큼 (50% 초과): {min_reduction}% ~ {max_reduction}%")
                                                            
                                                            if min_reduction > max_reduction:
                                                                warnings.append(f"감축 목표 범위 오류: min({min_reduction}%) > max({max_reduction}%)")
                                                            
                                                            # RE 적용률 검사
                                                            re_rates = material_config.get('re_rates', {})
                                                            for tier_key, tier_config in re_rates.items():
                                                                if isinstance(tier_config, dict):
                                                                    tier_min = tier_config.get('min', 0)
                                                                    tier_max = tier_config.get('max', 1)
                                                                    
                                                                    if tier_min > tier_max:
                                                                        warnings.append(f"{tier_key} RE 범위 오류: min({tier_min:.2f}) > max({tier_max:.2f})")
                                                                    
                                                                    if tier_min > 0.9 or tier_max > 1.0:
                                                                        warnings.append(f"{tier_key} RE 적용률이 너무 높음: {tier_min:.0%} ~ {tier_max:.0%}")
                                                            
                                                            # 자재 비율 검사
                                                            material_ratios = material_config.get('material_ratios', {})
                                                            recycle_min = material_ratios.get('recycle', {}).get('min', 0)
                                                            recycle_max = material_ratios.get('recycle', {}).get('max', 1)
                                                            low_carbon_min = material_ratios.get('low_carbon', {}).get('min', 0)
                                                            low_carbon_max = material_ratios.get('low_carbon', {}).get('max', 1)
                                                            
                                                            if recycle_min + low_carbon_min > 1.0:
                                                                warnings.append(f"재활용({recycle_min:.0%}) + 저탄소메탈({low_carbon_min:.0%}) > 100%")
                                                            
                                                            return warnings
                                                        
                                                        feasibility_warnings = check_feasibility(material_config, material)
                                                        if feasibility_warnings:
                                                            st.warning(f"⚠️ **{material} 실행가능성 경고:**")
                                                            for warning in feasibility_warnings:
                                                                st.write(f"• {warning}")
                                                        
                                                        # 자재별 최적화 실행 (간소화된 로그)
                                                        st.info(f"🔧 **{material}** 최적화 시작...")
                                                        
                                                        # 디버그 로그 표시용 컨테이너 생성 (접힌 상태로 시작)
                                                        debug_container = st.expander(f"🔍 {material} 상세 로그", expanded=False)
                                                        
                                                        material_optimizer = MaterialBasedOptimizer(
                                                            simulation_data=material_simulation_data,
                                                            config=material_config,
                                                            ui_params=current_params,
                                                            user_id=user_id,
                                                            scenario=material_config['optimization_scenario'],
                                                            debug_mode=False,  # 📝 간소화: 디버그 모드 비활성화
                                                            streamlit_container=debug_container
                                                        )
                                                        
                                                        # 최적화 모델 구성 및 실행 (간소화)
                                                        material_optimizer.build_optimization_model()
                                                        material_result = material_optimizer.solve(solver_name=solver)
                                                        
                                                        if material_result.get('status') == 'optimal':
                                                            # MaterialBasedOptimizer 결과에서 해당 자재 결과 추출
                                                            material_specific_results = material_result.get('materials', {})
                                                            
                                                            # 자재명 매핑 처리 - 시뮬레이션 변환된 이름 찾기 (enhanced mapping 적용)
                                                            actual_material_key = None
                                                            best_match_score = 0
                                                            
                                                            for key in material_specific_results.keys():
                                                                match_score = 0
                                                                
                                                                # 1. 정확한 매칭 (우선순위 최고)
                                                                if key.lower() == material.lower():
                                                                    actual_material_key = key
                                                                    break
                                                                
                                                                # 2. 토큰 기반 매칭 (rule_based.py 로직)
                                                                material_tokens = set(material.lower().split())
                                                                key_tokens = set(key.lower().split())
                                                                
                                                                # 공통 토큰 개수로 점수 계산
                                                                common_tokens = material_tokens & key_tokens
                                                                if len(common_tokens) > 0:
                                                                    match_score = len(common_tokens) / max(len(material_tokens), len(key_tokens))
                                                                
                                                                # 3. 부분 문자열 매칭
                                                                if material.lower() in key.lower() or key.lower() in material.lower():
                                                                    match_score = max(match_score, 0.7)
                                                                
                                                                # 4. 특별 케이스 매칭 (Al Foil ↔ Foil Al 등)
                                                                if ('al' in material.lower() and 'foil' in material.lower() and 
                                                                    'al' in key.lower() and 'foil' in key.lower()):
                                                                    match_score = max(match_score, 0.9)
                                                                
                                                                if match_score > best_match_score:
                                                                    best_match_score = match_score
                                                                    actual_material_key = key
                                                            
                                                            st.write(f"🔍 DEBUG: {material} 매칭 점수: {best_match_score:.2f} → {actual_material_key}")
                                                            st.write(f"🔍 DEBUG: {material} 사용 가능한 모든 키: {list(material_specific_results.keys())}")
                                                            
                                                            # 매칭 임계값을 낮춰서 더 유연하게 처리
                                                            if actual_material_key and best_match_score > 0.1 and material_specific_results[actual_material_key].get('status') == 'optimal':
                                                                specific_result = material_specific_results[actual_material_key]
                                                                optimized_emission = specific_result.get('optimized_pcf', baseline_emission)
                                                                
                                                                st.write(f"🎯 DEBUG: {material} 실제 매칭된 자재명: {actual_material_key}")
                                                                st.write(f"🎯 DEBUG: {material} 감축 결과 - {specific_result.get('reduction_percentage', 0):.1f}%")
                                                            else:
                                                                # 전체 결과에서 계산된 값 사용하거나 강제로 첫 번째 자재 결과 사용
                                                                if material_specific_results:
                                                                    # 첫 번째 자재 결과 강제 사용 (단일 자재 시나리오이므로)
                                                                    first_key = list(material_specific_results.keys())[0]
                                                                    first_result = material_specific_results[first_key]
                                                                    if first_result.get('status') == 'optimal':
                                                                        optimized_emission = first_result.get('optimized_pcf', baseline_emission)
                                                                        debug_container.info(f"자재명 매칭 실패 - 첫 번째 결과 사용: {first_key}")
                                                                    else:
                                                                        optimized_emission = material_result.get('optimized_pcf', baseline_emission)
                                                                        debug_container.warning("첫 번째 결과도 실패 - 전체 결과 사용")
                                                                else:
                                                                    optimized_emission = material_result.get('optimized_pcf', baseline_emission)
                                                                    debug_container.warning("개별 자재 결과 없음 - 전체 결과 사용")
                                                            
                                                            # 자재별 결과 저장 및 성공 표시
                                                            reduction_pct = ((baseline_emission - optimized_emission) / baseline_emission * 100) if baseline_emission > 0 else 0
                                                            st.success(f"✅ **{material}**: {reduction_pct:.1f}% 감축 달성")
                                                            
                                                            # 🎯 제약조건 준수 여부 표시
                                                            self._display_constraint_validation(material, material_specific_results, debug_container)
                                                            
                                                            # 🎯 자재별 감축 목표 준수 여부 표시
                                                            _display_material_target_validation(material, material_specific_results, debug_container)
                                                            
                                                            material_results[material] = {
                                                                'baseline_emission': baseline_emission,
                                                                'optimized_emission': optimized_emission,
                                                                'reduction_amount': baseline_emission - optimized_emission,
                                                                'reduction_percentage': reduction_pct,
                                                                'variables': material_result.get('variables', {}),
                                                                'cost': material_result.get('total_cost', 0),
                                                                'status': 'optimal'
                                                            }
                                                            
                                                            # 전체 합산
                                                            total_baseline_emission += baseline_emission
                                                            total_optimized_emission += optimized_emission
                                                            total_cost += material_result.get('total_cost', 0)
                                                            optimization_success_count += 1
                                                            
                                                        else:
                                                            # 최적화 실패시 기준값 사용 (간소화)
                                                            st.error(f"❌ **{material}** 최적화 실패: {material_result.get('message', '최적화 실패')}")
                                                            
                                                            # 상세 정보는 접힌 상태로 표시
                                                            with debug_container:
                                                                # 결과 요약 표시
                                                                status = material_result.get('status', 'unknown')
                                                                message = material_result.get('message', '상세 정보 없음')
                                                                
                                                                result_summary = {
                                                                    "status": status,
                                                                    "message": message,
                                                                    "solver_time": material_result.get('solver_time', 'N/A')
                                                                }
                                                                
                                                                # 추가 정보가 있는 경우 포함
                                                                if 'failed_materials' in material_result:
                                                                    result_summary["failed_materials"] = material_result['failed_materials']
                                                                if 'success_rate' in material_result:
                                                                    result_summary["success_rate"] = f"{material_result['success_rate']:.1f}%"
                                                                if 'error_summary' in material_result and material_result['error_summary']:
                                                                    result_summary["error_details"] = material_result['error_summary'][:3]  # 상위 3개만
                                                                
                                                                st.json(result_summary)
                                                            
                                                            # 진단 정보 표시 (간소화 - 접힌 상태)
                                                            if 'diagnostic_info' in material_result:
                                                                diagnostic = material_result['diagnostic_info']
                                                                
                                                                with st.expander(f"🔬 {material} 진단 정보", expanded=False):
                                                                    # 실행가능성 분석 표시
                                                                    if 'feasibility_analysis' in diagnostic:
                                                                        feasibility = diagnostic['feasibility_analysis']
                                                                        st.markdown("#### 🎯 실행가능성 분석")
                                                                        
                                                                        col1, col2, col3 = st.columns(3)
                                                                        with col1:
                                                                            st.metric("이론적 최대 감축", f"{feasibility.get('max_possible_reduction_pct', 0):.1f}%")
                                                                        with col2:
                                                                            st.metric("목표 감축", f"{feasibility.get('required_reduction_pct', 0):.1f}%")
                                                                        with col3:
                                                                            gap = feasibility.get('reduction_gap_pct', 0)
                                                                            if gap > 0:
                                                                                st.metric("부족분", f"{gap:.1f}%", delta=f"-{gap:.1f}%")
                                                                            else:
                                                                                st.metric("여유분", f"{-gap:.1f}%", delta=f"+{-gap:.1f}%")
                                                                        
                                                                        is_feasible = feasibility.get('is_theoretically_feasible', False)
                                                                        if is_feasible:
                                                                            st.success("✅ 이론적으로 달성 가능한 목표입니다.")
                                                                        else:
                                                                            st.error("❌ 이론적으로 달성 불가능한 목표입니다.")
                                                                    
                                                                    # 잠재적 문제점들
                                                                    if 'potential_issues' in diagnostic and diagnostic['potential_issues']:
                                                                        st.markdown("#### 🚨 발견된 문제점들")
                                                                        for i, issue in enumerate(diagnostic['potential_issues'], 1):
                                                                            st.warning(f"{i}. {issue}")
                                                                    
                                                                    # 권장 해결책들
                                                                    if 'recommended_fixes' in diagnostic and diagnostic['recommended_fixes']:
                                                                        st.markdown("#### 💡 권장 해결책들")
                                                                        for i, fix in enumerate(diagnostic['recommended_fixes'], 1):
                                                                            st.info(f"{i}. {fix}")
                                                                    
                                                                    # 제약조건 분석
                                                                    if 'constraint_analysis' in diagnostic:
                                                                        st.markdown("#### ⚖️ 제약조건 분석")
                                                                        constraint_analysis = diagnostic['constraint_analysis']
                                                                        
                                                                        for constraint_name, analysis in constraint_analysis.items():
                                                                            if constraint_name == 'error':
                                                                                continue
                                                                            
                                                                            constraint_type = analysis.get('type', constraint_name)
                                                                            violation = analysis.get('violation', False)
                                                                            
                                                                            if violation:
                                                                                st.error(f"❌ **{constraint_type}**: 제약조건 위반")
                                                                                if 'violation_amount' in analysis:
                                                                                    st.write(f"   위반 정도: {analysis['violation_amount']}")
                                                                                if 'recommendation' in analysis:
                                                                                    st.write(f"   권장사항: {analysis['recommendation']}")
                                                                            else:
                                                                                st.success(f"✅ **{constraint_type}**: 제약조건 만족")
                                                                    
                                                                    # 상세 제약조건 위반 정보 (새로 추가)
                                                                    if 'constraint_violations' in diagnostic and diagnostic['constraint_violations']:
                                                                        st.markdown("#### 🚨 제약조건 위반 상세 정보")
                                                                        
                                                                        violations = diagnostic['constraint_violations']
                                                                        violated_constraints = [v for v in violations if v.get('violated', False)]
                                                                        
                                                                        if violated_constraints:
                                                                            # 위반 요약
                                                                            constraint_summary = diagnostic.get('constraint_summary', {})
                                                                            total_constraints = constraint_summary.get('total_constraints', 0)
                                                                            violated_count = constraint_summary.get('violated_constraints', len(violated_constraints))
                                                                            
                                                                            st.error(f"총 {total_constraints}개 제약조건 중 {violated_count}개 위반")
                                                                            
                                                                            # 위반 상세 테이블
                                                                            violation_data = []
                                                                            for violation in violated_constraints:
                                                                                violation_type = violation.get('violation_type', 'unknown')
                                                                                violation_type_kr = {
                                                                                    'lower_bound': '하한 위반',
                                                                                    'upper_bound': '상한 위반',
                                                                                    'evaluation_error': '계산 오류'
                                                                                }.get(violation_type, violation_type)
                                                                                
                                                                                violation_data.append({
                                                                                    '제약조건': violation.get('constraint_name', 'N/A'),
                                                                                    '인덱스': violation.get('index', 'N/A'),
                                                                                    '위반 유형': violation_type_kr,
                                                                                    '현재값': f"{violation.get('body_value', 'N/A'):.4f}" if isinstance(violation.get('body_value'), (int, float)) else str(violation.get('body_value', 'N/A')),
                                                                                    '제한값': f"{violation.get('limit_value', 'N/A'):.4f}" if isinstance(violation.get('limit_value'), (int, float)) else str(violation.get('limit_value', 'N/A')),
                                                                                    '위반량': f"{violation.get('violation_amount', 'N/A'):.4f}" if isinstance(violation.get('violation_amount'), (int, float)) else str(violation.get('violation_amount', 'N/A'))
                                                                                })
                                                                            
                                                                            if violation_data:
                                                                                violation_df = pd.DataFrame(violation_data)
                                                                                st.dataframe(violation_df, use_container_width=True)
                                                                                
                                                                                # 위반 제약조건별 그룹화
                                                                                constraint_groups = {}
                                                                                for v in violated_constraints:
                                                                                    constraint_name = v.get('constraint_name', 'unknown')
                                                                                    if constraint_name not in constraint_groups:
                                                                                        constraint_groups[constraint_name] = []
                                                                                    constraint_groups[constraint_name].append(v)
                                                                                
                                                                                st.write("**위반된 제약조건별 요약:**")
                                                                                for constraint_name, group_violations in constraint_groups.items():
                                                                                    st.write(f"• **{constraint_name}**: {len(group_violations)}개 위반")
                                                                        else:
                                                                            st.success("✅ 모든 제약조건이 만족되었습니다.")
                                                                    
                                                                    # 자재 배출량 정보
                                                                    if 'material_emissions' in diagnostic:
                                                                        st.markdown("#### 📊 자재 배출량 정보")
                                                                        emissions = diagnostic['material_emissions']
                                                                        total_emission = diagnostic.get('total_original_emission', 0)
                                                                        
                                                                        for mat_name, emission in emissions.items():
                                                                            percentage = (emission / total_emission * 100) if total_emission > 0 else 0
                                                                            st.write(f"• **{mat_name}**: {emission:.4f} kgCO2eq ({percentage:.1f}%)")
                                                                    
                                                                    # 설정 요약
                                                                    if 'config_summary' in diagnostic:
                                                                        st.markdown("#### ⚙️ 설정 요약")
                                                                        with st.expander("설정 상세보기", expanded=False):
                                                                            st.json(diagnostic['config_summary'])
                                                            
                                                            # 디버그 로그 표시 (있는 경우)
                                                            if 'debug_logs' in material_result and material_result['debug_logs']:
                                                                st.write(f"🔍 **상세 디버그 로그 - {material}:**")
                                                                with st.expander(f"디버그 로그 상세보기 - {material}", expanded=False):
                                                                    
                                                                    # 로그 레벨별 필터링 옵션
                                                                    debug_logs = material_result['debug_logs']
                                                                    
                                                                    # 로그 레벨 통계
                                                                    log_levels = [log.get('level', 'INFO') for log in debug_logs]
                                                                    level_counts = {level: log_levels.count(level) for level in set(log_levels)}
                                                                    
                                                                    st.write(f"📊 **로그 요약:** 총 {len(debug_logs)}개")
                                                                    log_summary = " | ".join([f"{level}: {count}개" for level, count in level_counts.items()])
                                                                    st.write(f"   - {log_summary}")
                                                                    
                                                                    # 로그 레벨 필터
                                                                    show_levels = st.multiselect(
                                                                        f"표시할 로그 레벨 선택 - {material}:",
                                                                        options=['INFO', 'WARNING', 'ERROR'],
                                                                        default=['INFO', 'WARNING', 'ERROR'],
                                                                        key=f"log_filter_{material}"
                                                                    )
                                                                    
                                                                    # 필터링된 로그 표시
                                                                    filtered_logs = [log for log in debug_logs if log.get('level', 'INFO') in show_levels]
                                                                    
                                                                    if filtered_logs:
                                                                        # 스타일에 따른 로그 표시
                                                                        log_container = st.container()
                                                                        
                                                                        with log_container:
                                                                            for i, log_entry in enumerate(filtered_logs):
                                                                                level = log_entry.get('level', 'INFO')
                                                                                message = log_entry.get('message', '')
                                                                                timestamp = log_entry.get('timestamp', '')
                                                                                
                                                                                # 타임스탬프 간소화 (시간만 표시)
                                                                                if timestamp:
                                                                                    try:
                                                                                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                                                                        time_str = dt.strftime("%H:%M:%S")
                                                                                    except:
                                                                                        time_str = timestamp[-8:]  # 마지막 8자리 (시간 부분)
                                                                                else:
                                                                                    time_str = ""
                                                                                
                                                                                # 레벨별 스타일 적용
                                                                                if level == 'ERROR':
                                                                                    st.error(f"🕐 {time_str} | ❌ {message}")
                                                                                elif level == 'WARNING':
                                                                                    st.warning(f"🕐 {time_str} | ⚠️ {message}")
                                                                                else:  # INFO
                                                                                    st.info(f"🕐 {time_str} | ℹ️ {message}")
                                                                    else:
                                                                        st.write("선택한 레벨의 로그가 없습니다.")
                                                                    
                                                                    # 로그 다운로드 버튼
                                                                    if debug_logs:
                                                                        log_text = "\n".join([
                                                                            f"[{log.get('timestamp', 'N/A')}] {log.get('level', 'INFO')}: {log.get('message', '')}"
                                                                            for log in debug_logs
                                                                        ])
                                                                        
                                                                        st.download_button(
                                                                            label=f"📥 디버그 로그 다운로드 - {material}",
                                                                            data=log_text,
                                                                            file_name=f"debug_log_{material}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                                                            mime="text/plain",
                                                                            key=f"download_log_{material}"
                                                                        )
                                                            
                                                            material_results[material] = {
                                                                'baseline_emission': baseline_emission,
                                                                'optimized_emission': baseline_emission,
                                                                'reduction_amount': 0,
                                                                'reduction_percentage': 0,
                                                                'variables': {},
                                                                'cost': 0,
                                                                'status': 'failed',
                                                                'error': material_result.get('message', '최적화 실패')
                                                            }
                                                            
                                                            total_baseline_emission += baseline_emission
                                                            total_optimized_emission += baseline_emission
                                                            
                                                    except Exception as e:
                                                        # 오류 발생시 기준값 사용
                                                        import traceback
                                                        error_traceback = traceback.format_exc()
                                                        
                                                        st.error(f"💥 **{material}** 예외 발생: {str(e)}")
                                                        st.write(f"🔍 **DEBUG - {material} 예외 상세:**")
                                                        st.code(error_traceback, language="python")
                                                        
                                                        material_results[material] = {
                                                            'baseline_emission': baseline_emission,
                                                            'optimized_emission': baseline_emission,
                                                            'reduction_amount': 0,
                                                            'reduction_percentage': 0,
                                                            'variables': {},
                                                            'cost': 0,
                                                            'status': 'error',
                                                            'error': str(e)
                                                        }
                                                        
                                                        total_baseline_emission += baseline_emission
                                                        total_optimized_emission += baseline_emission
                                                
                                                # 🚨 FIXED: 중복 결과 처리 로직 비활성화
                                                # MaterialBasedOptimizer에서 이미 올바른 results를 생성했으므로 덮어쓰지 않음
                                                if optimization_success_count > 0:
                                                    # 전체 감축률 계산 (표시용)
                                                    total_reduction_percentage = ((total_baseline_emission - total_optimized_emission) / total_baseline_emission * 100) if total_baseline_emission > 0 else 0
                                                    
                                                    # ✅ MaterialBasedOptimizer 결과를 그대로 사용 (results 변수 덮어쓰지 않음)
                                                    # 대신 요약 정보만 업데이트
                                                    if 'results' in locals() and isinstance(results, dict):
                                                        # MaterialBasedOptimizer 결과에 요약 정보 추가
                                                        results['legacy_summary'] = {
                                                            'total_materials': total_materials,
                                                            'successful_optimizations': optimization_success_count,
                                                            'failed_optimizations': total_materials - optimization_success_count,
                                                            'success_rate': (optimization_success_count / total_materials * 100) if total_materials > 0 else 0,
                                                            'solver_used': solver,
                                                            'optimization_type': 'material_based_individual'
                                                        }
                                                    else:
                                                        st.info("ℹ️ MaterialBasedOptimizer 결과 없음 - 기본 결과 구성")
                                                        # Fallback: MaterialBasedOptimizer 결과가 없는 경우에만 생성
                                                        results = {
                                                            'status': 'optimal',
                                                            'objective_value': total_optimized_emission,
                                                            'carbon_footprint': total_optimized_emission,
                                                            'total_cost': total_cost,
                                                            'baseline_emission': total_baseline_emission,
                                                            'total_reduction_amount': total_baseline_emission - total_optimized_emission,
                                                            'total_reduction_percentage': total_reduction_percentage,
                                                            'material_results': material_results,
                                                            'optimization_summary': {
                                                                'total_materials': total_materials,
                                                                'successful_optimizations': optimization_success_count,
                                                                'failed_optimizations': total_materials - optimization_success_count,
                                                                'success_rate': (optimization_success_count / total_materials * 100) if total_materials > 0 else 0
                                                            },
                                                            'solver_used': solver,
                                                            'optimization_type': 'material_based_individual'
                                                        }
                                                    
                                                    status.update(label=f"✅ 자재별 최적화 완료 (성공: {optimization_success_count}/{total_materials})", state="complete")
                                                    
                                                    # 결과 요약 표시
                                                    st.markdown("### 🎯 자재별 최적화 결과 요약")
                                                    
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("전체 기준 배출량", f"{total_baseline_emission:.4f} kgCO2eq")
                                                    with col2:
                                                        st.metric("전체 최적화 배출량", f"{total_optimized_emission:.4f} kgCO2eq", 
                                                                f"{total_reduction_percentage:.1f}%")
                                                    with col3:
                                                        st.metric("성공한 최적화", f"{optimization_success_count}/{total_materials}", 
                                                                f"{(optimization_success_count/total_materials*100):.1f}%")
                                                    
                                                    # 자재별 상세 결과 테이블
                                                    st.markdown("### 📊 자재별 상세 결과")
                                                    
                                                    # 자재별 결과 분석
                                                    zero_reduction_materials = []
                                                    successful_materials = []
                                                    suboptimal_materials = []
                                                    failed_materials = []
                                                    
                                                    material_summary_data = []
                                                    
                                                    for material, result in material_results.items():
                                                        # 감축률 분석
                                                        reduction_pct = result['reduction_percentage']
                                                        
                                                        # 상태 분류
                                                        if result['status'] == 'optimal':
                                                            if reduction_pct <= 0.01:  # 0.01% 이하는 사실상 0
                                                                zero_reduction_materials.append(material)
                                                                status_text = "⚠️ 감축 없음"
                                                            else:
                                                                successful_materials.append(material)
                                                                status_text = "✅ 성공"
                                                        elif result['status'] == 'suboptimal':
                                                            suboptimal_materials.append(material)
                                                            status_text = "🔶 요구사항 미달"
                                                        else:
                                                            failed_materials.append(material)
                                                            status_text = "❌ 실패"
                                                        
                                                        material_summary_data.append({
                                                            "자재": material,
                                                            "기준 배출량": f"{result['baseline_emission']:.4f}",
                                                            "최적화 배출량": f"{result['optimized_emission']:.4f}",
                                                            "감축량": f"{result['reduction_amount']:.4f}",
                                                            "감축률 (%)": f"{result['reduction_percentage']:.2f}%",
                                                            "비용 ($)": f"{result['cost']:.2f}",
                                                            "상태": status_text
                                                        })
                                                    
                                                    # 문제 자재들에 대한 경고 표시
                                                    warning_materials = zero_reduction_materials + suboptimal_materials
                                                    if warning_materials:
                                                        if zero_reduction_materials:
                                                            st.warning(f"""
                                                            ⚠️ **감축량이 0에 가까운 자재들이 발견되었습니다!** ({len(zero_reduction_materials)}개)
                                                            
                                                            **해당 자재들:** {', '.join(zero_reduction_materials)}
                                                            
                                                            **가능한 원인:**
                                                            - 자재별 RE 적용률 설정이 너무 낮음
                                                            - proportion 자재의 매칭 실패  
                                                            - 자재의 원본 배출계수 또는 소요량 문제
                                                            
                                                            **해결 방법:**
                                                            1. RE 적용률을 더 높게 설정해보세요
                                                            2. 시뮬레이션 데이터를 다시 로드해보세요
                                                            3. 감축 목표를 조정해보세요
                                                            """)
                                                        
                                                        if suboptimal_materials:
                                                            st.info(f"""
                                                            🔶 **최소 요구사항에 미달하는 자재들이 있습니다.** ({len(suboptimal_materials)}개)
                                                            
                                                            **해당 자재들:** {', '.join(suboptimal_materials)}
                                                            
                                                            이 자재들은 감축이 가능하지만 설정된 최소 요구사항에 미달합니다.
                                                            실제 계산된 감축률이 표시되며, 값은 조작되지 않았습니다.
                                                            """)
                                                        
                                                        # 디버그 정보 표시 (접을 수 있는 형태)
                                                        with st.expander("🔧 디버그 정보 (개발자용)"):
                                                            st.markdown("**0 감축 자재들의 최적화 결과:**")
                                                            for material in zero_reduction_materials:
                                                                if material in material_results:
                                                                    result = material_results[material]
                                                                    st.json({
                                                                        'material': material,
                                                                        'status': result.get('status', 'unknown'),
                                                                        'baseline_emission': result.get('baseline_emission', 0),
                                                                        'optimized_emission': result.get('optimized_emission', 0),
                                                                        'reduction_amount': result.get('reduction_amount', 0),
                                                                        'reduction_percentage': result.get('reduction_percentage', 0),
                                                                        'debug_info': result.get('debug_info', {})
                                                                    })
                                                    
                                                    # 성공 통계 표시
                                                    if successful_materials or zero_reduction_materials or suboptimal_materials or failed_materials:
                                                        col1, col2, col3, col4 = st.columns(4)
                                                        with col1:
                                                            st.success(f"🎯 **정상 감축:** {len(successful_materials)}개")
                                                        with col2:
                                                            if suboptimal_materials:
                                                                st.info(f"🔶 **요구사항 미달:** {len(suboptimal_materials)}개")
                                                        with col3:
                                                            if zero_reduction_materials:
                                                                st.warning(f"⚠️ **감축 없음:** {len(zero_reduction_materials)}개")
                                                        with col4:
                                                            if failed_materials:
                                                                st.error(f"❌ **최적화 실패:** {len(failed_materials)}개")
                                                    
                                                    # 결과 테이블 표시
                                                    if material_summary_data:
                                                        material_summary_df = pd.DataFrame(material_summary_data)
                                                        st.dataframe(material_summary_df, use_container_width=True)
                                                    
                                                else:
                                                    # ✅ MaterialBasedOptimizer 결과가 있는지 확인 후 처리
                                                    if 'results' not in locals() or not isinstance(results, dict):
                                                        # MaterialBasedOptimizer 결과가 없는 경우에만 오류 결과 생성
                                                        results = {
                                                            'status': 'error', 
                                                            'message': '모든 자재의 최적화가 실패했습니다.',
                                                            'material_results': material_results
                                                        }
                                                    else:
                                                        # 최적화 성공한 자재들로 결과 구성
                                                        status.update(label="✅ 자재별 최적화 완료", state="complete")
                            else:
                                st.write("🔍 **DEBUG**: material_based 시나리오에서 시뮬레이션 데이터 없음")
                                st.error("시뮬레이션 데이터가 로드되지 않았습니다. 먼저 시뮬레이션 데이터를 로드해주세요.")
                                results = {'status': 'error', 'message': '시뮬레이션 데이터 로드 필요'}
                        
                            st.write("🔍 **DEBUG**: material_based 시나리오 블록 종료")
                            st.write(f"🔍 **DEBUG**: material_based 후 results 상태: {results.get('status', 'None') if 'results' in locals() else 'results 변수 없음'}")
                        
                        # 파레토 최적화 (신규 시나리오)
                        elif scenario_type == 'pareto_optimization':
                            from src.optimization.material_based_optimizer import MaterialBasedOptimizer
                            from src.optimization.grid_search_optimizer import GridSearchOptimizer
                            
                            # 시뮬레이션 데이터 확인
                            if 'pareto_simulation_data_loaded' in st.session_state and st.session_state.pareto_simulation_data_loaded:
                                simulation_data = st.session_state.pareto_simulation_data
                                
                                # 그리드 서치 파라미터 설정
                                grid_params = scenario_params.get('grid_params', {})
                                max_iterations = grid_params.get('max_iterations', 100)
                                
                                # 파레토 최적화를 위한 그리드 파라미터 구성
                                param_selection = {}
                                
                                if 'reduction_min' in grid_params and grid_params['reduction_min']:
                                    param_selection['reduction_min'] = grid_params['reduction_min']
                                    
                                if 'reduction_max' in grid_params and grid_params['reduction_max']:
                                    param_selection['reduction_max'] = grid_params['reduction_max']
                                    
                                if 'tier1_re' in grid_params and grid_params['tier1_re']:
                                    param_selection['tier1_re'] = grid_params['tier1_re']
                                    
                                if 'recycle_ratio' in grid_params and grid_params['recycle_ratio']:
                                    param_selection['recycle_ratio'] = grid_params['recycle_ratio']
                                
                                if not param_selection:
                                    st.error("파레토 최적화를 위한 그리드 서치 파라미터가 설정되지 않았습니다.")
                                    results = {'status': 'error', 'message': '그리드 서치 파라미터 설정 필요'}
                                else:
                                    # 그리드 서치 실행
                                    with st.status("파레토 최적화 진행 중...", expanded=True) as status:
                                        status.update(label="▶️ 파레토 최적화 초기화 중...")
                                        
                                        # 기본 설정 구성 (동적 tier 지원)
                                        dynamic_re_rates = {}
                                        for tier in range(1, num_tier + 1):
                                            dynamic_re_rates[f'tier{tier}'] = {'min': 0.1, 'max': 0.9}
                                        
                                        base_config = {
                                            'reduction_target': {'min': -10, 'max': -5},
                                            're_rates': dynamic_re_rates,
                                            'material_ratios': {
                                                'recycle': {'min': 0.05, 'max': 0.5},
                                                'low_carbon': {'min': 0.05, 'max': 0.3},
                                                'max_total': 0.7
                                            }
                                        }
                                        
                                        # 그리드서치 최적화 객체 생성
                                        grid_optimizer = GridSearchOptimizer(
                                            simulation_data=simulation_data,
                                            base_config=base_config,
                                            debug_mode=False
                                        )
                                        
                                        # 그리드 파라미터 설정
                                        grid_optimizer.set_grid_params(param_selection)
                                        
                                        # 진행 상황 표시 함수
                                        progress_bar = st.progress(0)
                                        
                                        def update_progress(current, total):
                                            progress = current / total
                                            progress_bar.progress(progress)
                                            status.update(label=f"▶️ 그리드 서치 진행 중... ({current}/{total})")
                                        
                                        # 그리드 서치 실행
                                        status.update(label="▶️ 파레토 해 탐색 중...")
                                        grid_results = grid_optimizer.run_grid_search(
                                            max_iterations=max_iterations,
                                            progress_callback=update_progress
                                        )
                                        
                                        # 파레토 최적해 결과
                                        status.update(label="▶️ 파레토 최적해 계산 중...")
                                        pareto_results = grid_optimizer.get_pareto_results()
                                        
                                        # 최적 타협안 선택
                                        best_compromise = grid_optimizer.get_best_compromise_result()
                                        
                                        # 결과 시각화 준비
                                        status.update(label="▶️ 파레토 프론트 시각화 준비 중...")
                                        visualization_data = grid_optimizer.generate_visualization_data()
                                        
                                        # 최적화 결과에 시각화 데이터 추가
                                        best_compromise['visualization_data'] = visualization_data
                                        best_compromise['pareto_results'] = pareto_results
                                        best_compromise['grid_results'] = grid_results
                                        
                                        # CSV 결과 저장 옵션이 있는 경우
                                        if scenario_params.get('save_results', {}).get('to_csv', False):
                                            status.update(label="▶️ 결과를 CSV 파일로 저장 중...")
                                            if scenario_params['save_results'].get('pareto_only', False):
                                                csv_path = grid_optimizer.export_pareto_results_to_csv()
                                                st.success(f"파레토 최적해가 {csv_path}에 저장되었습니다.")
                                            else:
                                                csv_path = grid_optimizer.export_results_to_csv()
                                                st.success(f"모든 그리드 서치 결과가 {csv_path}에 저장되었습니다.")
                                        
                                        # 최적화 완료
                                        status.update(label="✅ 파레토 최적화 완료", state="complete")
                                        
                                        # 그리드 서치 결과 요약
                                        st.markdown(f"### 파레토 최적화 결과 요약")
                                        st.markdown(f"총 {len(grid_results)}개의 유효한 조합을 탐색했습니다.")
                                        st.markdown(f"파레토 최적해: {len(pareto_results)}개")
                                        
                                        # 최적 조합 정보 표시
                                        if best_compromise.get('status') == 'optimal':
                                            st.markdown(f"#### 최적 타협안 (PCF {scenario_params.get('weights', {}).get('pcf', 0.6):.1f} : 비용 {scenario_params.get('weights', {}).get('cost', 0.4):.1f} 가중치 기준)")
                                            
                                            # 파라미터 요약
                                            params = best_compromise.get('params', {})
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                if 'reduction_min' in params:
                                                    st.metric("최소 감축률", f"{params['reduction_min']:.1f}%")
                                            with col2:
                                                if 'reduction_max' in params:
                                                    st.metric("최대 감축률", f"{params['reduction_max']:.1f}%")
                                            with col3:
                                                if 'tier1_re' in params:
                                                    st.metric("Tier1 RE 적용률", f"{params['tier1_re']:.2f}")
                                            with col4:
                                                if 'recycle_ratio' in params:
                                                    st.metric("재활용 비율", f"{params['recycle_ratio']:.2f}")
                                        
                                        # 최종 결과 설정
                                        results = best_compromise
                            else:
                                st.error("시뮬레이션 데이터가 로드되지 않았습니다. 먼저 시뮬레이션 데이터를 로드해주세요.")
                                results = {'status': 'error', 'message': '시뮬레이션 데이터 로드 필요'}
                        
                        # material_based와 pareto_optimization 시나리오는 이미 처리됨
                        if scenario_type not in ['material_based', 'pareto_optimization'] and compare_solvers:
                            # 여러 솔버로 실행
                            results = None
                            
                            if scenario_type == 'carbon_minimization':
                                # 탄소발자국 최소화 시나리오 (여러 솔버)
                                controller = get_controller()
                                results_dict = controller.run_multi_solver_comparison()
                                
                                # 비교 표 표시
                                comparison_df = compare_solver_results(results_dict)
                                st.markdown("### 솔버별 결과 비교")
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # 최적 솔버 결과 선택
                                best_solver = None
                                best_value = float('inf')
                                
                                for solver_name, result in results_dict.items():
                                    if result.get('status') == 'optimal' and result.get('objective_value', float('inf')) < best_value:
                                        best_value = result['objective_value']
                                        best_solver = solver_name
                                        results = result
                                
                                if best_solver:
                                    st.success(f"✓ 최적 솔버: {best_solver.upper()} (목적함수 값: {best_value:.4f})")
                                else:
                                    st.warning("⚠️ 최적해를 찾은 솔버가 없습니다.")
                                    
                            elif scenario_type in ['material_based', 'pareto_optimization']:
                                # material_based와 pareto_optimization은 솔버 비교 미지원
                                st.warning(f"⚠️ {SCENARIO_INFO.get(scenario_type, {}).get('title', scenario_type)} 시나리오는 솔버 비교를 지원하지 않습니다.")
                                st.info("💡 단일 솔버 모드로 자동 전환됩니다.")
                                
                                # compare_solvers를 False로 설정하여 단일 솔버 모드로 진행
                                compare_solvers = False
                                    
                            else:
                                # 다른 시나리오에 대한 다중 솔버 비교 구현
                                # decision_vars 설정 추가
                                comparison_config = {
                                    'objective': scenario_type.replace('_', '-'),
                                    'constraints': scenario_params,
                                    'decision_vars': current_params.get('decision_vars', {
                                        'cathode': {
                                            'type': 'B',
                                            'type_B_config': {
                                                'emission_fixed': 10.0,
                                                'recycle_range': [0.1, 0.5],
                                                'low_carbon_range': [0.05, 0.3]
                                            }
                                        },
                                        'use_binary_variables': False
                                    })
                                }
                                
                                results_dict = run_multi_solver_optimization(comparison_config)
                                
                                # 비교 표 표시
                                comparison_df = compare_solver_results(results_dict)
                                st.markdown("### 솔버별 결과 비교")
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # 최적 결과 선택
                                optimal_results = None
                                for solver_name, result in results_dict.items():
                                    if result.get('status') == 'optimal':
                                        if optimal_results is None or result['objective_value'] < optimal_results['objective_value']:
                                            optimal_results = result
                                            optimal_results['solver'] = solver_name
                                
                                if optimal_results:
                                    st.success(f"✓ 최적화가 성공적으로 완료되었습니다! (최적 솔버: {optimal_results['solver'].upper()})")
                                    results = optimal_results
                                else:
                                    st.error("모든 솔버에서 최적해를 찾지 못했습니다.")
                            
                        else:
                            # 단일 솔버로 실행
                            if scenario_type == 'carbon_minimization':
                                # 탄소발자국 최소화 - 시뮬레이션 정렬 실행
                                if scenario_params.get('use_simulation_data', True) and st.session_state.get('carbon_simulation_data_loaded', False):
                                    sim_data = st.session_state.carbon_simulation_data
                                    
                                    # OptimizationInput 객체 생성 (시뮬레이션 데이터 포함)
                                    opt_input = OptimizationInput(
                                        scenario_df=sim_data['scenario_df'],
                                        ref_formula_df=sim_data['ref_formula_df'],
                                        ref_proportions_df=sim_data['ref_proportions_df'],
                                        original_df=sim_data['original_df']
                                    )
                                    
                                    # 시뮬레이션 정렬 설정 업데이트
                                    sim_config = {
                                        "metadata": {"name": "탄소발자국 최소화 (시뮬레이션 정렬)", "version": "2.0"},
                                        "objective": "minimize_carbon",
                                        "decision_vars": {
                                            "cathode": current_params.get('cathode', {
                                                "type": "B",
                                                "type_B_config": {
                                                    "emission_fixed": 10.0,
                                                    "recycle_range": [0.1, 0.5],
                                                    "low_carbon_range": [0.05, 0.3]
                                                }
                                            }),
                                            "reduction_rates": {
                                                **{f"tier{tier}_re_rate": {
                                                    "min": scenario_params.get(f'tier{tier}_re_min', 0.2 + (tier-1)*0.2), 
                                                    "max": scenario_params.get(f'tier{tier}_re_max', 0.4 + (tier-1)*0.2), 
                                                    "default": (scenario_params.get(f'tier{tier}_re_min', 0.2 + (tier-1)*0.2) + scenario_params.get(f'tier{tier}_re_max', 0.4 + (tier-1)*0.2)) / 2
                                                } for tier in range(1, num_tier + 1)}
                                            }
                                        },
                                        "constraints": {
                                            "target_carbon": scenario_params.get('target_carbon', 12.0),
                                            "feasibility_threshold": 0.8
                                        },
                                        "case_constraints": {"enabled": False},
                                        "location_constraints": {"enabled": False},
                                        "material_constraints": {"enabled": False}
                                    }
                                    opt_input.update_config(sim_config)
                                    
                                    # CarbonMinimization 실행
                                    carbon_optimizer = CarbonMinimization()  # 설정 파일 경로 없이 초기화
                                    carbon_optimizer.opt_input = opt_input  # opt_input 직접 설정
                                    # solve 메서드가 callable인지 확인
                                    if hasattr(carbon_optimizer, 'solve') and callable(getattr(carbon_optimizer, 'solve')):
                                        results = carbon_optimizer.solve()
                                    else:
                                        # 폴백: run_scenario 메서드 사용
                                        results = carbon_optimizer.run_scenario()
                                else:
                                    # 기존 방식으로 실행
                                    results = run_carbon_minimization()
                            elif scenario_type == 'cost_minimization':
                                results = run_cost_minimization()
                            elif scenario_type == 'multi_objective':
                                results = run_multi_objective(
                                    carbon_weight=scenario_params.get('carbon_weight', 0.7),
                                    cost_weight=scenario_params.get('cost_weight', 0.3)
                                )
                            elif scenario_type == 'implementation_ease':
                                results = run_implementation_ease(
                                    carbon_target=scenario_params.get('target_carbon', 45.0)
                                )
                            elif scenario_type == 'regional_optimization':
                                results = run_regional_optimization(
                                    target_regions=scenario_params.get('target_regions', ['한국', '중국', '일본', '폴란드'])
                                )
                            elif scenario_type == 'simulation_aligned_carbon':
                                # 시뮬레이션 정렬 최적화 실행
                                if SIMULATION_ALIGNED_AVAILABLE and st.session_state.get('simulation_data_loaded', False):
                                    sim_data = st.session_state.simulation_data
                                    
                                    # OptimizationInput 객체 생성 (시뮬레이션 데이터 포함)
                                    opt_input = OptimizationInput(
                                        scenario_df=sim_data['scenario_df'],
                                        ref_formula_df=sim_data['ref_formula_df'],
                                        ref_proportions_df=sim_data['ref_proportions_df'],
                                        original_df=sim_data['original_df']
                                    )
                                    
                                    # 시뮬레이션 정렬 설정 업데이트
                                    sim_config = {
                                        "metadata": {"name": "시뮬레이션 정렬 PCF 최적화", "version": "2.0"},
                                        "objective": "minimize_carbon",
                                        "decision_vars": {
                                            "cathode": current_params.get('cathode', {
                                                "type": "B",
                                                "type_B_config": {
                                                    "emission_fixed": 10.0,
                                                    "recycle_range": [0.1, 0.5],
                                                    "low_carbon_range": [0.05, 0.3]
                                                }
                                            }),
                                            "reduction_rates": {
                                                **{f"tier{tier}_re_rate": {
                                                    "min": scenario_params.get(f'tier{tier}_re_min', 0.2 + (tier-1)*0.2), 
                                                    "max": scenario_params.get(f'tier{tier}_re_max', 0.4 + (tier-1)*0.2), 
                                                    "default": (scenario_params.get(f'tier{tier}_re_min', 0.2 + (tier-1)*0.2) + scenario_params.get(f'tier{tier}_re_max', 0.4 + (tier-1)*0.2)) / 2
                                                } for tier in range(1, num_tier + 1)}
                                            }
                                        },
                                        "constraints": {
                                            "target_carbon": scenario_params.get('target_carbon', 12.0),
                                            "max_cost": scenario_params.get('max_cost', 50000),
                                            "feasibility_threshold": 0.8
                                        },
                                        "case_constraints": {"enabled": False},
                                        "location_constraints": {"enabled": False},
                                        "material_constraints": {"enabled": False}
                                    }
                                    opt_input.update_config(sim_config)
                                    
                                    # CarbonMinimization 실행
                                    carbon_optimizer = CarbonMinimization()  # 설정 파일 경로 없이 초기화
                                    carbon_optimizer.opt_input = opt_input  # opt_input 직접 설정
                                    # solve 메서드가 callable인지 확인
                                    if hasattr(carbon_optimizer, 'solve') and callable(getattr(carbon_optimizer, 'solve')):
                                        results = carbon_optimizer.solve()
                                    else:
                                        # 폴백: run_scenario 메서드 사용
                                        results = carbon_optimizer.run_scenario()
                                else:
                                    results = {'status': 'error', 'message': '시뮬레이션 데이터가 로드되지 않았습니다.'}
                        
                        
                        # material_based와 pareto_optimization이 아닌 시나리오 처리
                        if scenario_type not in ['material_based', 'pareto_optimization']:
                            if not compare_solvers:
                                # 단일 솔버 실행
                                st.write(f"🔍 **DEBUG**: 일반 최적화 경로 진입 - scenario_type: {scenario_type}")
                                
                                results = run_optimization({
                                    'objective': scenario_type.replace('_', '-'),
                                    'constraints': scenario_params,
                                }, solver)
                        
                        # 시간 측정 종료
                        end_time = datetime.now()
                        elapsed_time = (end_time - start_time).total_seconds()
                        
                        # 결과 저장
                        if results and results.get('status') == 'optimal':
                            # 실행 시간 추가
                            if 'solver_time' not in results:
                                results['solver_time'] = elapsed_time
                            
                            # 실행 날짜 추가
                            results['run_date'] = datetime.now().isoformat()
                            
                            # 결과 저장
                            st.session_state.optimization_config['results'] = results
                            st.session_state.optimization_config['last_run'] = {
                                'date': datetime.now().isoformat(),
                                'elapsed_time': elapsed_time,
                                'scenario_type': scenario_type,
                                'solver': solver if not compare_solvers else results.get('solver', 'unknown'),
                                'status': 'success'
                            }
                            
                            # 설정 저장
                            save_optimization_config(st.session_state.optimization_config, user_id=user_id)
                            
                            st.success(f"✓ 최적화가 성공적으로 완료되었습니다! (소요시간: {elapsed_time:.2f}초)")
                            st.markdown("결과 확인을 위해 '결과 시각화' 탭으로 이동하세요.")
                            
                        elif results:
                            st.error(f"최적화 실패: {results.get('message', '알 수 없는 오류')}")
                            st.session_state.optimization_config['last_run'] = {
                                'date': datetime.now().isoformat(),
                                'elapsed_time': elapsed_time,
                                'scenario_type': scenario_type,
                                'solver': solver,
                                'status': 'failed',
                                'error': results.get('message', '알 수 없는 오류')
                            }
                        else:
                            st.error("최적화 결과를 받지 못했습니다.")
                            st.session_state.optimization_config['last_run'] = {
                                'date': datetime.now().isoformat(),
                                'elapsed_time': elapsed_time,
                                'scenario_type': scenario_type,
                                'solver': solver,
                                'status': 'error',
                                'error': '결과 없음'
                            }
                    
                    except Exception as e:
                        # 개선된 에러 메시지 표시
                        error_msg = str(e)
                        
                        # 에러 유형별 맞춤형 메시지
                        if "simulation_data" in error_msg.lower():
                            st.error("🔍 **시뮬레이션 데이터 오류**")
                            st.markdown("""
                            **문제**: 시뮬레이션 데이터를 찾을 수 없거나 형식이 올바르지 않습니다.
                            
                            **해결 방법**:
                            1. PCF 시뮬레이션 페이지에서 시뮬레이션을 먼저 실행하세요
                            2. 또는 '샘플 데이터 사용' 옵션을 선택하세요
                            3. 데이터 형식이 올바른지 확인하세요
                            """)
                        elif "solver" in error_msg.lower():
                            st.error("⚙️ **솔버 실행 오류**")
                            st.markdown("""
                            **문제**: 선택한 솔버에서 문제가 발생했습니다.
                            
                            **해결 방법**:
                            1. 다른 솔버를 선택해보세요 (GLPK → CBC 또는 IPOPT)
                            2. 제약조건을 완화해보세요
                            3. 파라미터 범위를 조정해보세요
                            """)
                        elif "memory" in error_msg.lower() or "size" in error_msg.lower():
                            st.error("💾 **메모리 부족 오류**")
                            st.markdown("""
                            **문제**: 계산 과정에서 메모리가 부족합니다.
                            
                            **해결 방법**:
                            1. 그리드 서치의 반복 횟수를 줄여보세요
                            2. 파라미터 범위를 좁혀보세요
                            3. 단순한 시나리오부터 시작해보세요
                            """)
                        elif "permission" in error_msg.lower():
                            st.error("🔐 **권한 오류**")
                            st.markdown("""
                            **문제**: 파일 접근 권한이 없습니다.
                            
                            **해결 방법**:
                            1. 사용자 ID가 올바르게 설정되었는지 확인하세요
                            2. 파일 경로에 쓰기 권한이 있는지 확인하세요
                            """)
                        else:
                            st.error("❌ **예상치 못한 오류가 발생했습니다**")
                            st.markdown(f"""
                            **오류 상세 정보**: `{error_msg}`
                            
                            **해결 방법**:
                            1. 페이지를 새로고침하고 다시 시도해보세요
                            2. 설정을 초기화하고 단계별로 진행해보세요
                            3. 문제가 지속되면 관리자에게 문의하세요
                            """)
                        
                        # 기술적 상세 정보 (확장 가능한 섹션)
                        with st.expander("🔧 기술적 상세 정보 (개발자용)"):
                            st.code(f"Error Type: {type(e).__name__}")
                            st.code(f"Error Message: {error_msg}")
                            st.exception(e)
                        
                        log_error(f"최적화 실행 오류: {e}")
                        
                        st.session_state.optimization_config['last_run'] = {
                            'date': datetime.now().isoformat(),
                            'scenario_type': scenario_type,
                            'solver': solver,
                            'status': 'error',
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
    
    with tab3:
        st.markdown('<div class="section-title">📊 결과 시각화</div>', unsafe_allow_html=True)
        
        # 최적화 결과 확인
        results = st.session_state.optimization_config.get('results', None)
        last_run = st.session_state.optimization_config.get('last_run', None)
        
        if not results or 'status' not in results or results['status'] != 'optimal':
            # 결과가 없는 경우
            st.markdown("""
            <div class="warning-box">
                <strong>📈 최적화 결과가 없습니다</strong><br>
                '설정 구성' 탭에서 최적화를 실행한 후 결과를 확인하세요.
            </div>
            """, unsafe_allow_html=True)
            
            if last_run and last_run.get('status') == 'failed':
                error_type = last_run.get('error_type', 'Unknown')
                error_msg = last_run.get('error', '알 수 없는 오류')
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffe6e6 0%, #ffb3b3 100%); 
                           border: 1px solid #ff4444; border-radius: 8px; padding: 15px; margin: 15px 0; color: #cc0000;">
                    <strong>❌ 마지막 실행 실패</strong><br>
                    <strong>오류 유형:</strong> {error_type}<br>
                    <strong>오류 내용:</strong> {error_msg}
                </div>
                """, unsafe_allow_html=True)
                
                # 오류별 맞춤형 해결책 제시
                if "simulation_data" in error_msg.lower():
                    st.info("💡 **해결 제안**: 시뮬레이션 데이터를 먼저 로드하거나 샘플 데이터를 사용해보세요.")
                elif "solver" in error_msg.lower():
                    st.info("💡 **해결 제안**: 다른 솔버를 선택하거나 제약조건을 조정해보세요.")
                elif "memory" in error_msg.lower():
                    st.info("💡 **해결 제안**: 파라미터 범위를 줄이거나 간단한 시나리오부터 시작해보세요.")
                else:
                    st.info("💡 **해결 제안**: 설정을 초기화하고 단계별로 다시 진행해보세요.")
            
            # 빠른 실행 안내
            st.markdown("""
            <div class="success-box">
                <strong>💡 빠른 시작 가이드</strong><br>
                1. '시나리오 선택' 탭에서 원하는 최적화 시나리오 선택<br>
                2. '설정 구성' 탭에서 파라미터 조정<br>
                3. 최적화 실행 버튼 클릭<br>
                4. 이곳에서 결과 확인 및 분석
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # 결과가 있는 경우 - 성공 메시지 표시
            scenario_type = st.session_state.optimization_config.get('scenario_type', 'carbon_minimization')
            scenario_title = SCENARIO_INFO.get(scenario_type, {}).get('title', scenario_type)
            
            st.markdown(f"""
            <div class="success-box">
                <strong>🎉 최적화 완료!</strong><br>
                {scenario_title} 시나리오의 최적화가 성공적으로 완료되었습니다.
            </div>
            """, unsafe_allow_html=True)
            
            # 결과 요약 카드
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            # 시나리오 타입에 따른 제목 조정
            if scenario_type == 'material_based':
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2>🧩 자재 기반 최적화 결과</h2>
                </div>
                """, unsafe_allow_html=True)
            elif scenario_type == 'pareto_optimization':
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2>📊 파레토 최적화 결과</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                icon_map = {
                    'carbon_minimization': '🌿',
                    'cost_minimization': '💰', 
                    'multi_objective': '⚖️',
                    'implementation_ease': '🛠️',
                    'regional_optimization': '🌎',
                    'simulation_aligned_carbon': '🎯'
                }
                icon = icon_map.get(scenario_type, '📈')
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2>{icon} {scenario_title} 결과</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # 실행 정보를 카드 형태로 표시
            if last_run:
                st.markdown("### 📋 실행 정보")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    run_date = datetime.fromisoformat(last_run.get('date', datetime.now().isoformat()))
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.5em;">📅</div>
                        <div style="font-size: 0.8em; color: #666;">실행 일시</div>
                        <div style="font-weight: bold; font-size: 1.1em;">{run_date.strftime("%Y-%m-%d %H:%M")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    solver = last_run.get('solver', 'unknown')
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.5em;">⚙️</div>
                        <div style="font-size: 0.8em; color: #666;">사용 솔버</div>
                        <div style="font-weight: bold; font-size: 1.1em;">{solver.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    elapsed_time = last_run.get('elapsed_time', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.5em;">⏱️</div>
                        <div style="font-size: 0.8em; color: #666;">소요 시간</div>
                        <div style="font-weight: bold; font-size: 1.1em;">{elapsed_time:.2f}초</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    status = "성공" if results.get('status') == 'optimal' else "실패"
                    status_emoji = "✅" if results.get('status') == 'optimal' else "❌"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.5em;">{status_emoji}</div>
                        <div style="font-size: 0.8em; color: #666;">실행 상태</div>
                        <div style="font-weight: bold; font-size: 1.1em;">{status}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 주요 결과 메트릭스
            st.markdown("### 🎯 주요 결과")
            
            # 목적함수 값과 추가 정보를 카드 형태로 표시
            result_cols = st.columns(3)
            
            with result_cols[0]:
                objective_value = results.get('objective_value', 0)
                obj_label = "목적함수 값"
                obj_emoji = "🎯"
                
                if scenario_type == 'carbon_minimization':
                    obj_label = "최소 탄소발자국"
                    obj_emoji = "🌿"
                    obj_unit = "kgCO2eq"
                elif scenario_type == 'cost_minimization':
                    obj_label = "최소 비용"
                    obj_emoji = "💰"
                    obj_unit = "USD"
                elif scenario_type == 'multi_objective':
                    obj_label = "가중 목적함수"
                    obj_emoji = "⚖️"
                    obj_unit = ""
                elif scenario_type == 'implementation_ease':
                    obj_label = "활성화 감축활동"
                    obj_emoji = "🛠️"
                    obj_unit = "개"
                elif scenario_type == 'regional_optimization':
                    obj_label = "최적 지역 점수"
                    obj_emoji = "🌎"
                    obj_unit = ""
                elif scenario_type == 'simulation_aligned_carbon':
                    obj_label = "최소 탄소발자국"
                    obj_emoji = "🎯"
                    obj_unit = "kgCO2eq"
                else:
                    obj_unit = ""
                
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #e8f5e8 0%, #c3e9c3 100%); border: 2px solid #27ae60;">
                    <div style="font-size: 2em;">{obj_emoji}</div>
                    <div style="font-size: 0.9em; color: #2c3e50; margin-bottom: 5px;">{obj_label}</div>
                    <div style="font-weight: bold; font-size: 1.4em; color: #27ae60;">{objective_value:.4f}</div>
                    <div style="font-size: 0.8em; color: #7f8c8d;">{obj_unit}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_cols[1]:
                # 탄소발자국 정보
                if 'carbon_footprint' in results:
                    carbon_value = results['carbon_footprint']
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #e8f4fd 0%, #c3e6fc 100%); border: 2px solid #3498db;">
                        <div style="font-size: 2em;">🌱</div>
                        <div style="font-size: 0.9em; color: #2c3e50; margin-bottom: 5px;">탄소발자국</div>
                        <div style="font-weight: bold; font-size: 1.4em; color: #3498db;">{carbon_value:.4f}</div>
                        <div style="font-size: 0.8em; color: #7f8c8d;">kg CO2eq/kWh</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #f8f9fa; border: 1px solid #dee2e6;">
                        <div style="font-size: 2em;">📊</div>
                        <div style="font-size: 0.9em; color: #6c757d;">탄소발자국</div>
                        <div style="font-weight: bold; font-size: 1.2em; color: #6c757d;">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_cols[2]:
                # 총 비용 정보
                if 'total_cost' in results:
                    cost_value = results['total_cost']
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #fdf2e9 0%, #fad7ac 100%); border: 2px solid #e67e22;">
                        <div style="font-size: 2em;">💳</div>
                        <div style="font-size: 0.9em; color: #2c3e50; margin-bottom: 5px;">총 비용</div>
                        <div style="font-weight: bold; font-size: 1.4em; color: #e67e22;">${cost_value:,.2f}</div>
                        <div style="font-size: 0.8em; color: #7f8c8d;">USD</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #f8f9fa; border: 1px solid #dee2e6;">
                        <div style="font-size: 2em;">💰</div>
                        <div style="font-size: 0.9em; color: #6c757d;">총 비용</div>
                        <div style="font-weight: bold; font-size: 1.2em; color: #6c757d;">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 프리미엄 비용 계산 및 표시
            with st.expander("프리미엄 비용 분석", expanded=True):
                if st.session_state.get('simulation_data_loaded', False) and st.session_state.get('simulation_data'):
                    # 프리미엄 비용 계산기 초기화
                    cost_calculator = MaterialPremiumCostCalculator(
                        simulation_data=st.session_state.simulation_data,
                        stable_var_dir="stable_var",
                        user_id=user_id,
                        debug_mode=False
                    )
                    
                    # 기준 프리미엄 비용 계산
                    baseline_costs = cost_calculator.calculate_baseline_premium_costs()
                    
                    # 최적화 후 프리미엄 비용 계산
                    optimized_costs = cost_calculator.calculate_optimized_premium_costs(results)
                    
                    # 비용 감축 계산
                    cost_reduction = cost_calculator.calculate_cost_reduction(
                        baseline_costs, optimized_costs
                    )
                    
                    # 결과 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("기준 프리미엄 비용", f"${baseline_costs['total']:.2f}")
                    with col2:
                        st.metric("최적화 후 프리미엄 비용", f"${optimized_costs['total']:.2f}")
                    with col3:
                        st.metric("프리미엄 비용 감축률", f"{cost_reduction['reduction_percentage']:.2f}%")
                    
                    # 자재별 프리미엄 비용 테이블
                    if 'material_reductions' in cost_reduction:
                        st.markdown("#### 자재별 프리미엄 비용 변화")
                        
                        # 데이터 테이블 생성
                        material_data = []
                        for material, data in cost_reduction['material_reductions'].items():
                            material_data.append({
                                "자재": material,
                                "기준 비용 ($)": f"{data['baseline']:.4f}",
                                "최적화 후 비용 ($)": f"{data['optimized']:.4f}",
                                "절감액 ($)": f"{data['reduction_amount']:.4f}",
                                "절감률 (%)": f"{data['reduction_percentage']:.2f}"
                            })
                        
                        # DataFrame으로 변환 및 표시
                        if material_data:
                            cost_df = pd.DataFrame(material_data)
                            st.dataframe(cost_df, use_container_width=True)
                else:
                    st.info("프리미엄 비용을 계산하려면 시뮬레이션 데이터가 필요합니다. 먼저 데이터를 로드해주세요.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 변수값 시각화
            if 'variables' in results:
                st.markdown("### 최적 변수 값")
                
                variables = results['variables']
                
                # 변수 종류에 따른 시각화
                var_categories = {}
                
                # 감축 비율 변수
                reduction_vars = {}
                for var_name, var_value in variables.items():
                    if 'tier' in var_name and not var_name.endswith('_active'):
                        # 티어 추출 (예: tier1_양극재 -> TIER1)
                        tier = var_name.split('_')[0].upper()
                        # 항목명 추출 (예: tier1_양극재 -> 양극재)
                        item = '_'.join(var_name.split('_')[1:])
                        
                        if tier not in reduction_vars:
                            reduction_vars[tier] = {}
                        
                        reduction_vars[tier][item] = var_value
                
                # 재활용 및 저탄소 메탈 변수
                material_vars = {}
                for var_name in ['recycle_ratio', 'low_carbon_ratio', 'low_carbon_emission']:
                    if var_name in variables:
                        material_vars[var_name] = variables[var_name]
                
                # 지역 변수
                location_vars = {}
                for var_name, var_value in variables.items():
                    if 'location_' in var_name:
                        region = var_name.replace('location_', '')
                        location_vars[region] = var_value
                
                # 1. 감축 비율 시각화
                if reduction_vars:
                    st.markdown("#### 티어별 감축 비율")
                    
                    # 데이터 준비
                    plot_data = []
                    for tier, items in reduction_vars.items():
                        for item, value in items.items():
                            plot_data.append({
                                'Tier': tier,
                                '항목': item,
                                '감축률(%)': value
                            })
                    
                    if plot_data:
                        plot_df = pd.DataFrame(plot_data)
                        
                        # 막대 그래프
                        fig = px.bar(
                            plot_df, 
                            x='항목', 
                            y='감축률(%)',
                            color='Tier',
                            barmode='group',
                            text_auto=True,
                            labels={'감축률(%)': '감축률 (%)', '항목': '항목', 'Tier': '티어'},
                            height=400,
                            color_discrete_map={
                                'TIER1': '#4682B4',
                                'TIER2': '#2E8B57',
                                'TIER3': '#B47846'
                            }
                        )
                        
                        fig.update_layout(
                            title="티어별 항목 감축률",
                            xaxis_title="항목",
                            yaxis_title="감축률 (%)",
                            legend_title="티어",
                            yaxis=dict(range=[0, 100]),
                            margin=dict(l=10, r=10, t=50, b=10),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 데이터 표로도 표시
                        st.markdown("**감축률 상세 데이터:**")
                        st.dataframe(plot_df, hide_index=True, use_container_width=True)
                
                # 2. 재활용 및 저탄소 메탈 변수 시각화
                if material_vars:
                    st.markdown("#### 재활용 및 저탄소 메탈 비율")
                    
                    # 신재 비율 계산
                    recycle_ratio = material_vars.get('recycle_ratio', 0)
                    low_carbon_ratio = material_vars.get('low_carbon_ratio', 0)
                    virgin_ratio = 1 - recycle_ratio - low_carbon_ratio
                    
                    # 파이 차트
                    labels = ["재활용재", "저탄소원료", "신재"]
                    values = [recycle_ratio, low_carbon_ratio, virgin_ratio]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        textinfo='label+percent',
                        insidetextorientation='radial',
                        marker_colors=['#4682B4', '#2E8B57', '#B47846'],
                        hole=.3
                    )])
                    
                    fig.update_layout(
                        title="원료 구성 비율",
                        height=400,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 데이터 표로도 표시
                    material_df = pd.DataFrame([
                        {"원료 유형": "재활용재", "비율(%)": f"{recycle_ratio*100:.2f}%"},
                        {"원료 유형": "저탄소원료", "비율(%)": f"{low_carbon_ratio*100:.2f}%"},
                        {"원료 유형": "신재", "비율(%)": f"{virgin_ratio*100:.2f}%"}
                    ])
                    
                    st.markdown("**원료 구성 상세 데이터:**")
                    st.dataframe(material_df, hide_index=True, use_container_width=True)
                    
                    # 저탄소 원료 배출계수
                    if 'low_carbon_emission' in material_vars:
                        low_carbon_emission = material_vars['low_carbon_emission']
                        st.metric("저탄소원료 배출계수", f"{low_carbon_emission:.2f}")
                
                # 3. 지역 변수 시각화
                if location_vars:
                    st.markdown("#### 지역별 할당")
                    
                    # 활성화된 지역 (값이 1인 지역)
                    active_locations = {region: value for region, value in location_vars.items() if value > 0.99}
                    
                    if active_locations:
                        # 바 차트 생성
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(location_vars.keys()),
                            y=list(location_vars.values()),
                            text=[f"{val:.2f}" for val in location_vars.values()],
                            textposition='auto',
                            marker_color=['#2E8B57' if val > 0.99 else '#B0C4DE' for val in location_vars.values()]
                        ))
                        
                        fig.update_layout(
                            title="지역별 할당 비율",
                            xaxis_title="지역",
                            yaxis_title="할당 비율",
                            height=400,
                            margin=dict(l=10, r=10, t=50, b=10),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 최적 지역 메시지
                        optimal_locations = [region for region, value in active_locations.items() if value > 0.99]
                        if optimal_locations:
                            st.success(f"✓ 최적 생산 지역: {', '.join(optimal_locations)}")
            
            # 제약조건 만족 여부
            if 'constraint_values' in results:
                st.markdown("### 제약조건 만족 여부")
                
                constraints = results['constraint_values']
                constraint_data = []
                
                for name, value in constraints.items():
                    limit = st.session_state.optimization_config.get('scenario_params', {}).get(
                        st.session_state.optimization_config.get('scenario_type', 'carbon_minimization'), 
                        {}
                    ).get(name, 'N/A')
                    
                    satisfied = "✅" if value <= limit else "❌"
                    constraint_data.append({
                        "제약조건": name,
                        "현재값": f"{value:.2f}",
                        "제한값": f"{limit:.2f}" if isinstance(limit, (int, float)) else limit,
                        "만족여부": satisfied
                    })
                
                if constraint_data:
                    st.dataframe(pd.DataFrame(constraint_data), hide_index=True, use_container_width=True)
            
            # 결과 저장
            st.markdown("### 결과 저장")
            
            # JSON으로 저장 버튼
            if st.button("결과 JSON으로 저장"):
                try:
                    # 결과 파일 저장
                    filename = export_results_to_json(
                        results, 
                        st.session_state.optimization_config
                    )
                    st.success(f"✓ 결과가 {filename}으로 저장되었습니다.")
                except Exception as e:
                    st.error(f"저장 중 오류 발생: {e}")
            
            # YAML로 저장 버튼
            if st.button("결과 YAML로 저장"):
                try:
                    # YAML 파일명 생성
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    scenario_type = st.session_state.optimization_config.get('scenario_type', 'optimization')
                    filename = f"optimization_results_{scenario_type}_{timestamp}.yaml"
                    
                    # 결과 데이터 구성
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'scenario_type': scenario_type,
                        'configuration': st.session_state.optimization_config,
                        'results': results
                    }
                    
                    # YAML 파일 저장
                    with open(filename, 'w', encoding='utf-8') as f:
                        yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
                    
                    st.success(f"✓ 결과가 {filename}으로 저장되었습니다.")
                except Exception as e:
                    st.error(f"저장 중 오류 발생: {e}")


# 시뮬레이션 정렬 최적화를 위한 헬퍼 함수들
def generate_sample_simulation_data() -> Dict[str, pd.DataFrame]:
    """
    샘플 시뮬레이션 데이터 생성
    
    Returns:
        Dict[str, pd.DataFrame]: 시뮬레이션 데이터 딕셔너리
    """
    # 샘플 scenario_df
    scenario_df = pd.DataFrame({
        '자재명': ['Cathode Active Material', 'Separator', 'Electrolyte', 'Anode Natural', 'Cu Foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        '배출계수': [12.5, 2.3, 4.1, 3.8, 15.2],
        '제품총소요량(kg)': [0.856, 0.025, 0.098, 0.103, 0.067],
        '저감활동_적용여부': [1.0, 1.0, 1.0, 1.0, 1.0],
        '배출량(kgCO2eq)': [10.7, 0.0575, 0.4018, 0.3914, 1.0184]
    })
    
    # 샘플 ref_formula_df
    ref_formula_df = pd.DataFrame({
        '자재명': ['Cathode Active Material', 'Separator', 'Electrolyte', 'Anode Natural', 'Cu Foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        '자재코드': ['CAM01', 'SEP01', 'ELY01', 'ANG01', 'CUF01'],
        '지역': ['중국', '중국', '중국', '중국', '중국'],
        'Tier1_RE100(kgCO2eq/kg)': [0.8, 0.15, 0.25, 0.18, 1.2],
        'Tier2_RE100(kgCO2eq/kg)': [0.6, 0.12, 0.18, 0.14, 0.9],
        'Tier3_RE100(kgCO2eq/kg)': [0.4, 0.08, 0.12, 0.09, 0.6]
    })
    
    # 샘플 ref_proportions_df
    ref_proportions_df = pd.DataFrame({
        '자재명(포함)': ['cathode active material', 'separator', 'electrolyte', 'natural', 'foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        'Tier1_RE100(%)': [15.0, 8.0, 12.0, 10.0, 20.0],
        'Tier2_RE100(%)': [25.0, 12.0, 18.0, 15.0, 30.0],
        'Tier3_RE100(%)': [35.0, 15.0, 25.0, 20.0, 40.0]
    })
    
    # 샘플 original_df - num_tier=2 포함
    original_df = pd.DataFrame({
        '자재명': ['Cathode Active Material', 'Separator', 'Electrolyte', 'Anode Natural', 'Cu Foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        '자재코드': ['CAM01', 'SEP01', 'ELY01', 'ANG01', 'CUF01'],
        '지역': ['중국', '중국', '중국', '중국', '중국'],
        '배출계수': [12.5, 2.3, 4.1, 3.8, 15.2],
        'num_tier': [2, 2, 2, 2, 2]  # num_tier=2 설정
    })
    
    return {
        'scenario_df': scenario_df,
        'ref_formula_df': ref_formula_df,
        'ref_proportions_df': ref_proportions_df,
        'original_df': original_df,
        'num_tier': 2  # 직접 num_tier 값도 설정
    }

def check_session_simulation_data() -> bool:
    """
    현재 세션에 시뮬레이션 데이터가 있는지 확인
    
    Returns:
        bool: 시뮬레이션 데이터 존재 여부
    """
    # 방법 1: 직접적인 키로 확인
    required_keys = ['scenario_df', 'ref_formula_df', 'ref_proportions_df', 'original_df']
    
    direct_check = True
    for key in required_keys:
        if key not in st.session_state or st.session_state[key] is None:
            direct_check = False
            break
    
    if direct_check:
        print(f"[DEBUG] 직접 키로 데이터 확인: True")
        return True
    
    # 방법 2: PCF 시뮬레이션 결과에서 확인
    if 'simulation_results' not in st.session_state:
        print(f"[DEBUG] simulation_results 키 없음")
        return False
    
    sim_results = st.session_state.simulation_results
    print(f"[DEBUG] simulation_results 타입: {type(sim_results)}")
    
    if not sim_results or not isinstance(sim_results, dict):
        print(f"[DEBUG] simulation_results가 비어있거나 dict가 아님")
        return False
    
    print(f"[DEBUG] simulation_results 키들: {list(sim_results.keys())}")
    
    # 시나리오 중 하나라도 유효한 결과가 있으면 True
    for scenario_key, scenario_data in sim_results.items():
        print(f"[DEBUG] 시나리오 {scenario_key} 확인 중...")
        if scenario_data and isinstance(scenario_data, dict):
            print(f"[DEBUG] {scenario_key}의 키들: {list(scenario_data.keys())}")
            if 'all_data' in scenario_data:
                # all_data가 DataFrame이고 비어있지 않으면 유효
                all_data = scenario_data['all_data']
                print(f"[DEBUG] {scenario_key}의 all_data 타입: {type(all_data)}, 길이: {len(all_data) if hasattr(all_data, '__len__') else 'N/A'}")
                if hasattr(all_data, '__len__') and len(all_data) > 0:
                    print(f"[DEBUG] 유효한 시뮬레이션 데이터 발견: {scenario_key}")
                    return True
    
    print(f"[DEBUG] 유효한 시뮬레이션 데이터 없음")
    return False

def get_session_simulation_data() -> Dict[str, pd.DataFrame]:
    """
    세션에서 시뮬레이션 데이터 가져오기
    
    Returns:
        Dict[str, pd.DataFrame]: 시뮬레이션 데이터 딕셔너리
    """
    # 방법 1: 직접적인 키로 확인
    required_keys = ['scenario_df', 'ref_formula_df', 'ref_proportions_df', 'original_df']
    
    direct_data = {}
    all_direct_available = True
    
    for key in required_keys:
        if key in st.session_state and st.session_state[key] is not None:
            direct_data[key] = st.session_state[key]
        else:
            all_direct_available = False
            break
    
    if all_direct_available:
        return direct_data
    
    # 방법 2: PCF 시뮬레이션 결과에서 추출
    if 'simulation_results' in st.session_state:
        sim_results = st.session_state.simulation_results
        print(f"[DEBUG] simulation_results 존재, 타입: {type(sim_results)}")
        
        if sim_results and isinstance(sim_results, dict):
            print(f"[DEBUG] simulation_results 키들: {list(sim_results.keys())}")
            # baseline 시나리오를 우선적으로 사용
            preferred_scenarios = ['baseline', 'both', 'site_change', 'material_change']
            
            for scenario_key in preferred_scenarios:
                print(f"[DEBUG] 시나리오 '{scenario_key}' 확인 중...")
                if scenario_key in sim_results:
                    scenario_data = sim_results[scenario_key]
                    print(f"[DEBUG] {scenario_key} 데이터 타입: {type(scenario_data)}")
                    if scenario_data and isinstance(scenario_data, dict):
                        print(f"[DEBUG] {scenario_key} 하위 키들: {list(scenario_data.keys())}")
                        if 'all_data' in scenario_data:
                            all_data = scenario_data['all_data']
                            print(f"[DEBUG] {scenario_key} all_data 타입: {type(all_data)}, 길이: {len(all_data) if hasattr(all_data, '__len__') else 'N/A'}")
                            if hasattr(all_data, 'columns'):
                                print(f"[DEBUG] {scenario_key} all_data 컬럼들: {list(all_data.columns)}")
                            
                            # all_data에서 필요한 데이터프레임들 생성
                            try:
                                result = _extract_optimization_data_from_simulation(all_data, scenario_data)
                                print(f"[DEBUG] {scenario_key}에서 추출 성공!")
                                return result
                            except Exception as e:
                                print(f"[DEBUG] 시뮬레이션 데이터 추출 실패 ({scenario_key}): {e}")
                                continue
                        else:
                            print(f"[DEBUG] {scenario_key}에 all_data 키 없음")
                    else:
                        print(f"[DEBUG] {scenario_key} 데이터가 비어있거나 dict가 아님")
                else:
                    print(f"[DEBUG] {scenario_key} 키가 simulation_results에 없음")
            
            # preferred 시나리오가 없으면 첫 번째 유효한 시나리오 사용
            for scenario_key, scenario_data in sim_results.items():
                if scenario_data and isinstance(scenario_data, dict) and 'all_data' in scenario_data:
                    all_data = scenario_data['all_data']
                    try:
                        return _extract_optimization_data_from_simulation(all_data, scenario_data)
                    except Exception as e:
                        print(f"시뮬레이션 데이터 추출 실패: {e}")
                        continue
    
    # 모든 방법이 실패하면 빈 딕셔너리 반환
    return {
        'scenario_df': pd.DataFrame(),
        'ref_formula_df': pd.DataFrame(),
        'ref_proportions_df': pd.DataFrame(),
        'original_df': pd.DataFrame()
    }


def _extract_optimization_data_from_simulation(all_data: pd.DataFrame, scenario_data: Dict) -> Dict[str, pd.DataFrame]:
    """
    시뮬레이션 결과에서 최적화에 필요한 데이터 추출
    
    Args:
        all_data: 시뮬레이션 결과 데이터프레임
        scenario_data: 시나리오 분석 결과
    
    Returns:
        Dict[str, pd.DataFrame]: 최적화용 데이터
    """
    # scenario_df: 저감활동 적용 가능한 자재 (all_data 기반)
    scenario_df = all_data.copy()
    
    # 필수 컬럼 확인 및 추가 (이미 있으면 그대로 유지)
    if '저감활동_적용여부' not in scenario_df.columns:
        # 자재품목이 특정 항목이면 저감활동 적용 가능으로 설정
        if '자재품목' in scenario_df.columns:
            applicable_items = ['양극재', '분리막', '전해액', '음극재', '동박']
            scenario_df['저감활동_적용여부'] = scenario_df['자재품목'].apply(
                lambda x: 1.0 if x in applicable_items else 0.0
            )
        else:
            # 자재품목 컬럼도 없으면 모든 자재에 적용 가능으로 설정
            scenario_df['저감활동_적용여부'] = 1.0
    else:
        # 컬럼이 이미 있으면 그대로 유지 (시뮬레이션 결과의 원본 데이터)
        pass
    
    # 배출량 컬럼 확인 및 추가
    if '배출량(kgCO2eq)' not in scenario_df.columns:
        if '배출계수' in scenario_df.columns and '제품총소요량(kg)' in scenario_df.columns:
            scenario_df['배출량(kgCO2eq)'] = scenario_df['배출계수'] * scenario_df['제품총소요량(kg)']
        else:
            scenario_df['배출량(kgCO2eq)'] = 0.0
    
    # original_df: 원본 배출계수 정보
    base_columns = []
    for col in ['자재명', '자재품목', '배출계수']:
        if col in scenario_df.columns:
            base_columns.append(col)
    
    if base_columns:
        original_df = scenario_df[base_columns].copy()
    else:
        original_df = pd.DataFrame()
    
    # 선택적 컬럼 추가
    for col in ['자재코드', '지역']:
        if col in scenario_df.columns:
            original_df[col] = scenario_df[col]
    
    # ref_formula_df: Formula 방식 참조 데이터 (샘플 생성)
    if len(scenario_df) > 0 and '자재품목' in scenario_df.columns:
        material_items = scenario_df['자재품목'].tolist() if '자재품목' in scenario_df.columns else ['기타'] * len(scenario_df)
        
        # 자재코드와 지역 컬럼 안전하게 처리
        if '자재코드' in scenario_df.columns:
            material_codes = scenario_df['자재코드'].tolist()
        else:
            material_codes = [''] * len(scenario_df)
            
        if '지역' in scenario_df.columns:
            regions = scenario_df['지역'].tolist()
        else:
            regions = ['중국'] * len(scenario_df)
        
        ref_formula_df = pd.DataFrame({
            '자재명': scenario_df['자재명'].tolist() if '자재명' in scenario_df.columns else [''] * len(scenario_df),
            '자재품목': material_items,
            '자재코드': material_codes,
            '지역': regions,
            'Tier1_RE100(kgCO2eq/kg)': [0.8 if item == '양극재' else 0.15 for item in material_items],
            'Tier2_RE100(kgCO2eq/kg)': [0.6 if item == '양극재' else 0.12 for item in material_items],
            'Tier3_RE100(kgCO2eq/kg)': [0.4 if item == '양극재' else 0.08 for item in material_items]
        })
    else:
        ref_formula_df = pd.DataFrame()
    
    # ref_proportions_df: Proportions 방식 참조 데이터 (샘플 생성)
    if len(scenario_df) > 0 and '자재품목' in scenario_df.columns:
        unique_items = scenario_df['자재품목'].unique()
        ref_proportions_df = pd.DataFrame({
            '자재명(포함)': [item.lower() for item in unique_items],
            '자재품목': unique_items,
            'Tier1_RE100(%)': [15.0 if item == '양극재' else 8.0 for item in unique_items],
            'Tier2_RE100(%)': [25.0 if item == '양극재' else 12.0 for item in unique_items], 
            'Tier3_RE100(%)': [35.0 if item == '양극재' else 15.0 for item in unique_items]
        })
    else:
        ref_proportions_df = pd.DataFrame()
    
    return {
        'scenario_df': scenario_df,
        'ref_formula_df': ref_formula_df,
        'ref_proportions_df': ref_proportions_df,
        'original_df': original_df,
        'num_tier': 2  # 직접 num_tier 값도 설정
    }
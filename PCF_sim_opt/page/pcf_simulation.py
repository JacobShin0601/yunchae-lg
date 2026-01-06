import streamlit as st
import pandas as pd
import os
import sys
from typing import Dict, Any
from io import BytesIO

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from src.rule_based import RuleBasedSim
from app_helper import (
    display_simulation_overview,
    display_pcf_analysis,
    display_matching_analysis,
    display_reduction_activity_analysis,
    display_material_analysis,
    display_comprehensive_analysis,
    load_simulation_config,
    get_default_simulation_config
)
from src.logger import log_button_click, log_input_change, log_info, log_error, log_warning
from src.utils.styles import get_page_styles

def load_csv_file(file_path: str) -> pd.DataFrame:
    """CSV 파일을 로드합니다."""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path, encoding='utf-8')
        else:
            st.error(f"파일을 찾을 수 없습니다: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"파일 로드 오류 ({file_path}): {e}")
        return pd.DataFrame()

def pcf_simulation_page():
    # Apply centralized styles
    st.markdown(get_page_styles('pcf_simulation'), unsafe_allow_html=True)
    
    # Add additional page-specific styles
    st.markdown("""
    <style>
    .simulation-section {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .simulation-title {
        color: var(--text-color);
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 3px;
    }
    .info-box {
        background-color: var(--bg-secondary);
        border-left: 4px solid var(--secondary-color);
        padding: 8px;
        margin: 8px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: var(--success-bg);
        border-left: 4px solid var(--primary-color);
        padding: 8px;
        margin: 8px 0;
        border-radius: 5px;
        color: var(--primary-color);
    }
    .error-box {
        background-color: var(--error-bg);
        border-left: 4px solid var(--error-color);
        padding: 8px;
        margin: 8px 0;
        border-radius: 5px;
        color: var(--error-color);
    }
    .data-section {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .summary-card {
        background-color: var(--bg-secondary);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px var(--shadow-color);
        transition: all 0.3s ease;
    }
    .summary-card:hover {
        border-color: var(--secondary-color);
        box-shadow: 0 6px 12px var(--shadow-color);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simulation-section">
        <h2 style="color: var(--text-color);">PCF Simulator</h2>
        <p style="color: var(--text-secondary);">규칙 기반 PCF 시뮬레이션을 통해 탄소배출량을 분석할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)

    # 시뮬레이터 페이지 리프레시 버튼
    st.markdown('<div style="margin-top: -10px; margin-bottom: 5px;">', unsafe_allow_html=True)

    if st.button("🔄 Refresh", type="secondary"):
        log_button_click("refresh_simulator", "refresh_simulator_page_btn")
        log_info("시뮬레이터 페이지 리프레시 실행")

        # PCF 시뮬레이션 페이지의 세션 상태 초기화
        if 'simulation_results' in st.session_state:
            st.session_state.simulation_results = {}
            log_info("simulation_results 초기화됨")

        if 'is_loading' in st.session_state:
            st.session_state.is_loading = False
            log_info("is_loading 상태 초기화됨")

        st.success("✅ 시뮬레이터 페이지가 초기화되었습니다! 새로운 시뮬레이션을 실행할 수 있습니다.")
        log_info("시뮬레이터 페이지 리프레시 완료")

    st.info("💡 이전 시뮬레이션 결과를 초기화하고 새로운 시뮬레이션을 실행하고 싶을 때 사용하세요.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    data_dir = os.path.join(project_root, "data")
    
    # 사용자별 파일 경로 설정
    user_id = st.session_state.get('user_id', None)
    if user_id:
        # 사용자별 데이터 경로
        user_data_dir = os.path.join(data_dir, user_id)
        saved_scenario_path = os.path.join(user_data_dir, "pcf_scenario_saved.csv")
        original_path = os.path.join(user_data_dir, "pcf_original_table_updated.csv")
        # 사용자별 파일이 없으면 sample 파일 사용
        if not os.path.exists(original_path):
            sample_path = os.path.join(user_data_dir, "pcf_original_table_sample.csv")
            if os.path.exists(sample_path):
                original_path = sample_path
            else:
                # 공통 파일로 폴백
                original_path = os.path.join(data_dir, "pcf_original_table_updated.csv")
    else:
        # 기본 데이터 경로
        saved_scenario_path = os.path.join(data_dir, "pcf_scenario_saved.csv")
        original_path = os.path.join(data_dir, "pcf_original_table_updated.csv")
    ref_formula_path = os.path.join(data_dir, "pcf_ref_formula_table.csv")
    ref_proportions_path = os.path.join(data_dir, "pcf_ref_proportions_table.csv")
    
    # 데이터 로드
    saved_scenario_df = load_csv_file(saved_scenario_path)
    ref_formula_df = load_csv_file(ref_formula_path)
    ref_proportions_df = load_csv_file(ref_proportions_path)
    original_df = load_csv_file(original_path)

    # saved_scenario_df에 배출계수명 고유화 적용 (original_df와 일치시키기 위해)
    if not saved_scenario_df.empty and '배출계수명' in saved_scenario_df.columns:
        from app_helper import apply_unique_emission_factor_names_with_index
        log_info("🔧 saved_scenario_df 배출계수명 고유화 시작...")
        saved_scenario_df = apply_unique_emission_factor_names_with_index(saved_scenario_df)
        log_info(f"✅ saved_scenario_df 배출계수명 고유화 완료: {saved_scenario_df['배출계수명'].nunique()}개 고유 배출계수명")

        # 음극재 배출계수명 확인 (디버깅용)
        anode_scenario = saved_scenario_df[saved_scenario_df['자재품목'] == '음극재']
        if not anode_scenario.empty:
            log_info("📊 saved_scenario_df 음극재 배출계수명:")
            for idx, row in anode_scenario.iterrows():
                log_info(f"  • {row['자재명']}: 원본='{row.get('배출계수명_원본', 'N/A')}', 고유='{row['배출계수명']}'")

    # 사용자별 데이터 검증: 양극재/음극재 소요량이 0인지 확인
    if user_id and not saved_scenario_df.empty:
        cathode_materials = saved_scenario_df[saved_scenario_df['자재품목'] == '양극재']
        anode_materials = saved_scenario_df[saved_scenario_df['자재품목'] == '음극재']
        
        # 양극재나 음극재의 총 소요량이 모두 0인 경우 메인 데이터로 fallback
        cathode_total = cathode_materials['제품총소요량(kg)'].sum() if not cathode_materials.empty else 0
        anode_total = anode_materials['제품총소요량(kg)'].sum() if not anode_materials.empty else 0
        
        if cathode_total == 0 or anode_total == 0:
            missing_materials = []
            if cathode_total == 0:
                missing_materials.append("양극재")
            if anode_total == 0:
                missing_materials.append("음극재")

            st.warning(
                f"⚠️ **데이터 검증**: 사용자별 시나리오 데이터에서 {', '.join(missing_materials)} 소요량이 0입니다.\n\n"
                f"현재 값: 양극재 {cathode_total:.6f}kg, 음극재 {anode_total:.6f}kg\n\n"
                f"💡 **해결 방법**: 먼저 '**양극재 세부 설정**' 페이지에서 설정을 완료하고 'Apply 설정' 버튼을 클릭해주세요.\n\n"
                f"현재는 기본 샘플 데이터로 시뮬레이션을 진행합니다."
            )

            # 메인 데이터 파일로 fallback
            main_scenario_path = os.path.join(data_dir, "pcf_scenario_saved.csv")
            main_original_path = os.path.join(data_dir, "pcf_original_table_updated.csv")

            saved_scenario_df = load_csv_file(main_scenario_path)
            original_df = load_csv_file(main_original_path)

            # fallback 후 배출계수명 고유화도 다시 적용
            if not saved_scenario_df.empty and '배출계수명' in saved_scenario_df.columns:
                from app_helper import apply_unique_emission_factor_names_with_index
                log_info("🔧 fallback saved_scenario_df 배출계수명 고유화 시작...")
                saved_scenario_df = apply_unique_emission_factor_names_with_index(saved_scenario_df)
                log_info(f"✅ fallback saved_scenario_df 배출계수명 고유화 완료: {saved_scenario_df['배출계수명'].nunique()}개 고유 배출계수명")

            # fallback 후 재검증
            if not saved_scenario_df.empty:
                cathode_materials_main = saved_scenario_df[saved_scenario_df['자재품목'] == '양극재']
                anode_materials_main = saved_scenario_df[saved_scenario_df['자재품목'] == '음극재']
                cathode_total_main = cathode_materials_main['제품총소요량(kg)'].sum() if not cathode_materials_main.empty else 0
                anode_total_main = anode_materials_main['제품총소요량(kg)'].sum() if not anode_materials_main.empty else 0

                st.info(f"📊 **기본 샘플 데이터 로드**: 양극재 {cathode_total_main:.6f}kg, 음극재 {anode_total_main:.6f}kg")
    
    if saved_scenario_df.empty:
        st.error("pcf_scenario_saved.csv 파일을 로드할 수 없습니다.")
        return
        
    # 현재 설정에 맞게 데이터프레임 동적 컬럼 조정
    try:
        from page.scenario_configuration import create_dynamic_columns
        current_max_case = st.session_state.max_case
        current_num_tier = st.session_state.num_tier
        
        # 로드된 CSV 파일의 현재 컬럼 상태 로깅
        tier_cols = [col for col in saved_scenario_df.columns if 'Tier' in col and 'RE_case' in col]
        log_info(f"CSV 로드 전 컬럼 상태: {len(tier_cols)}개의 Tier 컬럼, 설정은 {current_num_tier} Tier × {current_max_case} Case")
        
        # 현재 설정에 맞게 동적 컬럼 조정
        saved_scenario_df = create_dynamic_columns(saved_scenario_df, current_max_case, current_num_tier)
        
        # 컬럼 조정 결과 로깅
        if saved_scenario_df is not None:
            new_tier_cols = [col for col in saved_scenario_df.columns if 'Tier' in col and 'RE_case' in col]
            log_info(f"CSV 로드 후 컬럼 조정 완료: {len(new_tier_cols)}개의 Tier 컬럼")
        else:
            log_error("동적 컬럼 조정 실패")
            # 실패한 경우 원본 데이터프레임 유지
            saved_scenario_df = load_csv_file(saved_scenario_path)
    except Exception as e:
        log_error(f"동적 컬럼 조정 중 오류 발생: {e}")
        # 오류 발생 시 원본 데이터프레임 유지
    
    # 세션 상태 초기화
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = {}
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False

    # 시나리오 설정 변경 감지 및 시뮬레이터 초기화
    if st.session_state.get('scenario_settings_changed', False):
        log_info("시나리오 설정 변경 감지 - 시뮬레이션 결과 초기화 및 데이터 재로드")
        st.session_state.simulation_results = {}
        st.session_state.is_loading = False
        st.session_state.scenario_settings_changed = False
        st.session_state.show_reload_message = True
        log_info("시뮬레이션 결과 초기화 완료 - 페이지 재로드하여 최신 데이터 가져옴")

        # 페이지 재로드하여 최신 CSV 데이터 로드
        st.rerun()

    # 데이터 재로드 후 안내 메시지 표시
    if st.session_state.get('show_reload_message', False):
        st.info("⚠️ 시나리오 설정이 변경되었습니다. 새로운 시뮬레이션을 실행해주세요.")
        st.session_state.show_reload_message = False

    # 세션 상태에서 설정값 가져오기 (세션 상태 우선)
    if 'config_settings' in st.session_state:
        # 세션에 저장된 설정값 사용
        max_case = st.session_state.config_settings.get('max_case', 3)
        num_tier = st.session_state.config_settings.get('num_tier', 2)
        log_info(f"[세션 상태] 설정 값 로드: case={max_case}, tier={num_tier}")
    else:
        # 세션 상태에 없는 경우 파일에서 로드 
        config = load_simulation_config(user_id=user_id)
        if config:
            max_case = config.get('max_case', 3)
            num_tier = config.get('num_tier', 2)
            log_info(f"[파일] 설정 값 로드: case={max_case}, tier={num_tier}")
        else:
            # 기본값 설정
            default_config = get_default_simulation_config()
            max_case = default_config['max_case']
            num_tier = default_config['num_tier']
            log_info(f"[기본값] 설정 값 로드: case={max_case}, tier={num_tier}")
    
    # 세션 상태 업데이트 (현재 값으로)
    st.session_state.max_case = max_case
    st.session_state.num_tier = num_tier
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown('<div class="simulation-title">시뮬레이션 설정</div>', unsafe_allow_html=True)
        
        # 현재 설정 표시
        st.subheader("현재 설정")
        config_source = "세션 상태"
        if 'config_settings' not in st.session_state:
            if config:
                config_source = "설정 파일"
            else:
                config_source = "기본값"
        
        st.info(f"**Case 수:** {max_case} ({config_source})")
        st.info(f"**Tier 수:** {num_tier} ({config_source})")
        
        # 설정 변경 안내
        st.markdown("---")
        st.markdown("**설정 변경:**")
        st.markdown("시나리오 설정 페이지에서 변경 후 Apply 버튼을 눌러주세요.")
        
        # 상세 로그 옵션
        st.markdown("---")
        st.subheader("로그 설정")
        verbose_option = st.selectbox(
            "로그 레벨:",
            options=['info', 'debug', 'warning'],
            index=0,
            help="info: 기본 정보만, debug: 상세 정보, warning: 경고만"
        )
    
    # 메인 컨텐츠
    st.markdown('<div class="simulation-title">시나리오 데이터</div>', unsafe_allow_html=True)
    
    # 저감활동이 적용된 자재만 필터링
    applied_materials = saved_scenario_df[saved_scenario_df['저감활동_적용여부'] == 1.0].copy()
    
    # 디버그: 양극재와 음극재 소요량 확인
    cathode_materials = saved_scenario_df[saved_scenario_df['자재품목'] == '양극재']
    anode_materials = saved_scenario_df[saved_scenario_df['자재품목'] == '음극재']
    
    log_info("=== 양극재/음극재 소요량 디버그 ===")
    log_info(f"전체 자료 중 양극재 개수: {len(cathode_materials)}개")
    log_info(f"전체 자료 중 음극재 개수: {len(anode_materials)}개")
    
    if not cathode_materials.empty:
        log_info("양극재 상세:")
        for idx, row in cathode_materials.iterrows():
            material_name = row.get('자재명', 'N/A')
            quantity = row.get('제품총소요량(kg)', 0)
            emission_coef = row.get('배출계수', 0)
            reduction_applied = row.get('저감활동_적용여부', 0)
            log_info(f"  • {material_name}: 소요량={quantity:.6f}kg, 배출계수={emission_coef:.6f}, 저감활동={reduction_applied}")
    
    if not anode_materials.empty:
        log_info("음극재 상세:")
        for idx, row in anode_materials.iterrows():
            material_name = row.get('자재명', 'N/A')
            quantity = row.get('제품총소요량(kg)', 0)
            emission_coef = row.get('배출계수', 0)
            reduction_applied = row.get('저감활동_적용여부', 0)
            log_info(f"  • {material_name}: 소요량={quantity:.6f}kg, 배출계수={emission_coef:.6f}, 저감활동={reduction_applied}")
    
    log_info(f"저감활동 적용된 자재 중 양극재 개수: {len(applied_materials[applied_materials['자재품목'] == '양극재'])}개")
    log_info(f"저감활동 적용된 자재 중 음극재 개수: {len(applied_materials[applied_materials['자재품목'] == '음극재'])}개")
    log_info("================================")
    
    if len(applied_materials) > 0:
        # st.markdown('<div class="success-box">', unsafe_allow_html=True)
        # st.success(f"✅ 저감활동이 적용된 자재: {len(applied_materials)}개")
        # st.markdown('</div>', unsafe_allow_html=True)
        
        # 저감활동 적용 자재 상세 정보 (expander)
        with st.expander(f"저감활동 적용 자재 상세 정보 : {len(applied_materials)}개", expanded=False):
            # 표시할 컬럼 정의 (설정된 tier/case에 맞게 동적 생성)
            base_columns = ['자재명', '자재품목', '제품총소요량(kg)', '배출계수명', '배출계수', '배출량(kgCO2eq)', '저감활동_적용여부']
            
            # 세션에서 max_case와 num_tier 가져오기
            current_max_case = st.session_state.max_case
            current_num_tier = st.session_state.num_tier
            
            # 동적 컬럼 생성
            tier_case_columns = []
            for tier in range(1, current_num_tier + 1):
                for case in range(1, current_max_case + 1):
                    tier_case_columns.append(f'Tier{tier}_RE_case{case}')
            
            display_columns = base_columns + tier_case_columns
            log_info(f"저감활동 적용 자재 상세정보: {current_num_tier} Tier 와 {current_max_case} Case 기준으로 표시함")
            
            # PCF 관련 열 제외
            exclude_columns = ['PCF_reference', 'PCF_case1', 'PCF_case2', 'PCF_case3']
            display_columns = [col for col in display_columns if col not in exclude_columns]
            
            # 실제 존재하고 현재 설정에 맞는 컬럼만 필터링
            available_columns = [col for col in display_columns if col in applied_materials.columns]
            
            # Tier 컬럼의 경우 현재 설정 범위만 유지
            filtered_columns = []
            for col in available_columns:
                if 'Tier' in col and 'RE_case' in col:
                    # 현재 설정 범위 내의 Tier 컬럼인지 확인
                    is_valid_tier = False
                    for tier in range(1, current_num_tier + 1):
                        for case in range(1, current_max_case + 1):
                            if col == f'Tier{tier}_RE_case{case}':
                                is_valid_tier = True
                                break
                        if is_valid_tier:
                            break
                    if is_valid_tier:
                        filtered_columns.append(col)
                else:
                    # Tier 컬럼이 아닌 경우 그대로 포함
                    filtered_columns.append(col)
            
            available_columns = filtered_columns
            
            # 데이터프레임 표시
            # st.markdown('<div class="data-section">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 자재품목별 분포 (expander)
        with st.expander("자재품목별 분포", expanded=False):
            # st.markdown('<div class="data-section">', unsafe_allow_html=True)
            st.write("**자재품목별 분포:**")
            
            material_distribution = applied_materials['자재품목'].value_counts()
            for category, count in material_distribution.items():
                st.write(f"• {category}: {count}개")
            
            st.markdown("---")
            
            # PCF 합계 (PCF 관련 열이 있는 경우) - 전체 자재 기준 사용
            if 'PCF_reference' in saved_scenario_df.columns:
                total_reference = saved_scenario_df['PCF_reference'].sum()
                st.write(f"**총 PCF Reference: {total_reference:.3f} kgCO2eq**")
            elif 'PCF_reference' in applied_materials.columns:
                # 전체 자재에 PCF_reference가 없으면 저감활동 적용 자재만 표시
                applied_reference = applied_materials['PCF_reference'].sum()
                st.write(f"**저감활동 적용 자재 PCF Reference: {applied_reference:.3f} kgCO2eq**")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.warning("⚠️ 저감활동이 적용된 자재가 없습니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 로딩 상태 처리
    if st.session_state.is_loading:
        st.markdown("""
        <div class="simulation-section">
            <h3 style="color: var(--text-color);">⏳ 시뮬레이션 실행 중...</h3>
            <p style="color: var(--text-secondary);">모든 시나리오에 대해 시뮬레이션을 실행하고 있습니다.</p>
            <div style="margin: 20px 0;">
                <div style="display: inline-block; width: 30px; height: 30px; border: 4px solid #333; border-top: 4px solid #2196F3; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # CSS 애니메이션 추가
        st.markdown("""
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    # 시뮬레이션 결과 표시
    elif st.session_state.simulation_results:

        # 시나리오 이름 정의 (전역적으로 사용)
        scenario_names = {
            'baseline': '기본 시나리오',
            'recycling': '재활용&저탄소메탈 시나리오',
            'site_change': '생산지 변경 시나리오',
            'both': '종합 시나리오'
        }

        # ===== 통합 디버그 정보 섹션 =====
        st.markdown("---")
        st.markdown('<div class="simulation-title">🔍 디버그 정보</div>', unsafe_allow_html=True)
        st.info("📌 시뮬레이션 계산 과정의 상세 정보를 확인할 수 있습니다. 각 expander를 열어 원하는 정보를 확인하세요.")

        # PCF 계산 결과 디버그 정보 표시
        pcf_debug_scenarios = [s for s in ['baseline', 'recycling', 'site_change', 'both'] if f'pcf_debug_{s}' in st.session_state]
        if pcf_debug_scenarios:
            with st.expander("🔍 PCF 계산 결과 상세 디버그", expanded=False):
                st.markdown("#### 각 시나리오의 양극재 PCF 계산 결과")

                for scenario in pcf_debug_scenarios:
                    pcf_debug = st.session_state[f'pcf_debug_{scenario}']
                    scenario_name = pcf_debug['scenario_name']

                    st.markdown(f"### {scenario_name}")

                    # 양극재 상세 정보
                    if pcf_debug['cathode_materials']:
                        st.markdown("**양극재 자재별 PCF:**")
                        for material in pcf_debug['cathode_materials']:
                            st.markdown(f"**• {material['자재명']}**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"배출계수: {material['배출계수']:.6f}")
                                st.write(f"소요량: {material['제품총소요량(kg)']:.6f} kg")
                            with col2:
                                st.write(f"PCF_reference: {material['PCF_reference']:.6f}")
                                st.write(f"PCF_case1: {material['PCF_case1']:.6f}")
                            with col3:
                                st.write(f"PCF_case2: {material['PCF_case2']:.6f}")
                                st.write(f"PCF_case3: {material['PCF_case3']:.6f}")

                    # 전체 PCF 합계
                    st.markdown("---")
                    st.markdown("**전체 PCF 합계:**")
                    pcf_col1, pcf_col2, pcf_col3, pcf_col4 = st.columns(4)
                    with pcf_col1:
                        st.metric("PCF Reference", f"{pcf_debug['total_pcf_reference']:.6f}")
                    with pcf_col2:
                        st.metric("PCF Case1", f"{pcf_debug['total_pcf_case1']:.6f}")
                    with pcf_col3:
                        st.metric("PCF Case2", f"{pcf_debug['total_pcf_case2']:.6f}")
                    with pcf_col4:
                        st.metric("PCF Case3", f"{pcf_debug['total_pcf_case3']:.6f}")

                    st.markdown("---")

                # 시나리오 간 비교
                if 'pcf_debug_baseline' in st.session_state and 'pcf_debug_site_change' in st.session_state:
                    st.markdown("### 📊 Baseline vs Site_change 비교")

                    baseline_debug = st.session_state['pcf_debug_baseline']
                    site_change_debug = st.session_state['pcf_debug_site_change']

                    comp_col1, comp_col2 = st.columns(2)

                    with comp_col1:
                        st.markdown("**Baseline 시나리오**")
                        st.write(f"• Total PCF_case1: {baseline_debug['total_pcf_case1']:.6f} kgCO2eq")

                    with comp_col2:
                        st.markdown("**Site_change 시나리오**")
                        st.write(f"• Total PCF_case1: {site_change_debug['total_pcf_case1']:.6f} kgCO2eq")

                    # 차이 계산
                    diff = baseline_debug['total_pcf_case1'] - site_change_debug['total_pcf_case1']
                    diff_pct = (diff / baseline_debug['total_pcf_case1']) * 100 if baseline_debug['total_pcf_case1'] > 0 else 0

                    if diff > 0:
                        st.success(f"✅ 논리적으로 올바름: Baseline이 Site_change보다 {diff:.6f} kgCO2eq 높음 ({diff_pct:.2f}% 감축)")
                    elif diff < 0:
                        st.error(f"❌ 논리적 오류: Baseline이 Site_change보다 {-diff:.6f} kgCO2eq 낮음 ({-diff_pct:.2f}% 증가)")
                    else:
                        st.info("ℹ️ 동일한 PCF 값")

        # 전력계수 디버그 정보 표시
        if 'electricity_coef_baseline' in st.session_state or 'electricity_coef_site_change' in st.session_state:
            with st.expander("🔍 전력 배출계수 디버그 정보", expanded=False):
                st.markdown("#### 각 시나리오가 사용하는 전력 배출계수")

                col_debug1, col_debug2 = st.columns(2)

                with col_debug1:
                    if 'electricity_coef_baseline' in st.session_state:
                        baseline_elec = st.session_state['electricity_coef_baseline']
                        st.markdown("**기본 시나리오 (Baseline)**")
                        if 'tier1' in baseline_elec:
                            st.write(f"• Energy(Tier-1): {baseline_elec['tier1']:.6f} kgCO2eq/kWh")
                        if 'tier2' in baseline_elec:
                            st.write(f"• Energy(Tier-2): {baseline_elec['tier2']:.6f} kgCO2eq/kWh")

                with col_debug2:
                    if 'electricity_coef_site_change' in st.session_state:
                        site_change_elec = st.session_state['electricity_coef_site_change']
                        st.markdown("**생산지 변경 시나리오 (Site Change)**")
                        if 'tier1' in site_change_elec:
                            st.write(f"• Energy(Tier-1): {site_change_elec['tier1']:.6f} kgCO2eq/kWh")
                        if 'tier2' in site_change_elec:
                            st.write(f"• Energy(Tier-2): {site_change_elec['tier2']:.6f} kgCO2eq/kWh")

                # 비교 분석
                if 'electricity_coef_baseline' in st.session_state and 'electricity_coef_site_change' in st.session_state:
                    st.markdown("---")
                    st.markdown("**📊 비교 분석**")

                    baseline_elec = st.session_state['electricity_coef_baseline']
                    site_change_elec = st.session_state['electricity_coef_site_change']

                    if 'tier1' in baseline_elec and 'tier1' in site_change_elec:
                        tier1_diff = baseline_elec['tier1'] - site_change_elec['tier1']
                        tier1_change_pct = (tier1_diff / baseline_elec['tier1']) * 100 if baseline_elec['tier1'] > 0 else 0

                        if tier1_diff > 0:
                            st.success(f"✅ Tier-1: Site Change가 Baseline보다 {tier1_diff:.6f} kgCO2eq/kWh 낮음 ({tier1_change_pct:.2f}% 감소)")
                        elif tier1_diff < 0:
                            st.error(f"❌ Tier-1: Site Change가 Baseline보다 {-tier1_diff:.6f} kgCO2eq/kWh 높음 ({-tier1_change_pct:.2f}% 증가)")
                        else:
                            st.info("ℹ️ Tier-1: 동일한 전력계수 사용")

                    if 'tier2' in baseline_elec and 'tier2' in site_change_elec:
                        tier2_diff = baseline_elec['tier2'] - site_change_elec['tier2']
                        tier2_change_pct = (tier2_diff / baseline_elec['tier2']) * 100 if baseline_elec['tier2'] > 0 else 0

                        if tier2_diff > 0:
                            st.success(f"✅ Tier-2: Site Change가 Baseline보다 {tier2_diff:.6f} kgCO2eq/kWh 낮음 ({tier2_change_pct:.2f}% 감소)")
                        elif tier2_diff < 0:
                            st.error(f"❌ Tier-2: Site Change가 Baseline보다 {-tier2_diff:.6f} kgCO2eq/kWh 높음 ({-tier2_change_pct:.2f}% 증가)")
                        else:
                            st.info("ℹ️ Tier-2: 동일한 전력계수 사용")

        # 전체 simulation_results 구조 확인 (디버깅용)
        with st.expander("🔍 전체 Simulation Results 구조 확인", expanded=False):
            st.write("### session_state.simulation_results 키들:")
            if st.session_state.simulation_results:
                for key in st.session_state.simulation_results.keys():
                    st.write(f"- **{key}**")
                    result = st.session_state.simulation_results[key]
                    if isinstance(result, dict):
                        st.write(f"  - Type: dict")
                        st.write(f"  - Keys: {list(result.keys())}")

                        # all_data 확인
                        if 'all_data' in result:
                            all_data = result['all_data']
                            if all_data is not None:
                                st.write(f"  - all_data: ✅ (shape: {all_data.shape})")
                                st.write(f"  - all_data columns: {list(all_data.columns)}")
                                st.write(f"  - PCF_reference 존재: {'PCF_reference' in all_data.columns}")
                                if 'PCF_reference' in all_data.columns:
                                    st.write(f"  - PCF_reference 합계: {all_data['PCF_reference'].sum():.3f} kgCO2eq")
                            else:
                                st.write(f"  - all_data: ❌ None")
                        else:
                            st.write(f"  - all_data 키 없음: ❌")
                    else:
                        st.write(f"  - Type: {type(result)}")
            else:
                st.write("simulation_results가 비어있습니다.")

            st.write("\n### pcf_summary 데이터 (각 시나리오별)")
            for scenario in scenario_names.keys():
                if scenario in st.session_state.simulation_results:
                    result = st.session_state.simulation_results[scenario]
                    if isinstance(result, dict) and 'pcf_summary' in result:
                        st.write(f"\n**{scenario_names[scenario]}**:")
                        st.dataframe(result['pcf_summary'])
                    else:
                        st.write(f"\n**{scenario_names[scenario]}**: pcf_summary 없음")

        # 통합 Fallback 경고 표시 (모든 시나리오의 경고를 한 번에)
        all_fallback_warnings = {}
        for scenario in scenario_names.keys():
            if f'fallback_warnings_{scenario}' in st.session_state:
                warnings = st.session_state[f'fallback_warnings_{scenario}']
                if warnings:
                    all_fallback_warnings[scenario] = warnings

        if all_fallback_warnings:
            with st.expander("⚠️ Fallback 경고 (배출계수 음수 감지)", expanded=False):
                st.markdown("""
                <div style="background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                    <div style="color: #856404; font-weight: bold; font-size: 1.1rem; margin-bottom: 10px;">⚠️ Fallback 경고</div>
                    <div style="color: #856404; font-size: 0.9rem;">
                        배출계수 음수 감지로 인해 <strong>모든 Case에 Proportion 로직</strong>이 적용된 자재가 있습니다.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                warning_text = ""
                for scenario, warnings in all_fallback_warnings.items():
                    scenario_name = scenario_names[scenario]
                    warning_text += f"**[{scenario_name}]**\n\n"
                    for warning in warnings:
                        warning_text += f"- **{warning['자재명']}** ({warning['자재품목']}): Case {warning['trigger_case']}에서 음수 발생 (원본 배출계수: {warning['원본_배출계수']:.6f})\n"
                    warning_text += "\n"

                st.warning(warning_text)

        st.markdown("---")
        st.markdown('<div class="simulation-title">시뮬레이션 결과 요약</div>', unsafe_allow_html=True)

        # 2x2 그리드로 요약 박스 배치
        col1, col2 = st.columns(2)
        
        for i, (scenario, scenario_name) in enumerate(scenario_names.items()):
            if scenario in st.session_state.simulation_results:
                result_dict = st.session_state.simulation_results[scenario]
                
                # PCF 요약 데이터 가져오기
                pcf_summary = result_dict.get('pcf_summary', pd.DataFrame())
                
                if not pcf_summary.empty:
                    # 시나리오별 Case 1, 2, 3 요약 표 생성
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"""
                        <div class="summary-card">
                            <h4 style="color: var(--text-color); margin-bottom: 15px; text-align: center; border-bottom: 2px solid var(--secondary-color); padding-bottom: 10px;">{scenario_name}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Case별 데이터 추출
                        case_data = []
                        reference_data = None
                        
                        for _, row in pcf_summary.iterrows():
                            if 'PCF Reference' in row['Case']:
                                reference_data = row
                            else:
                                case_data.append(row)
                        
                        # Case별 요약 표 생성
                        if case_data:
                            # DataFrame 스타일링을 위한 CSS
                            st.markdown("""
                            <style>
                            .stDataFrame {
                                background-color: var(--bg-secondary) !important;
                                border-radius: 8px !important;
                                overflow: hidden !important;
                            }
                            .stDataFrame > div {
                                background-color: var(--bg-secondary) !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # 가장 높은 감축률 찾기
                            best_reduction = -1
                            for case in case_data:
                                if case['감축률(%)'] != '-' and float(case['감축률(%)']) > best_reduction:
                                    best_reduction = float(case['감축률(%)'])
                            
                            # === 모든 시나리오의 기준은 baseline 시나리오의 PCF_reference ===
                            baseline_all_data = st.session_state.simulation_results.get('baseline', {}).get('all_data')

                            if baseline_all_data is not None and 'PCF_reference' in baseline_all_data.columns:
                                baseline_pcf = baseline_all_data['PCF_reference'].sum()
                            else:
                                # fallback: reference_data 사용
                                baseline_pcf = reference_data['PCF_총합(kgCO2eq)'] if reference_data is not None else 0

                            # 표 생성 - 모든 시나리오에서 동일한 baseline_pcf 기준 사용
                            table_data = []

                            # 모든 시나리오에서 동일한 참조 행 추가
                            table_data.append({
                                'Case': '기본 시나리오 상 PCF 기준 (저감활동X)',
                                'PCF (kgCO2eq)': f"{baseline_pcf:.3f}",
                                '감축률 (%)': '0.0%',
                                '효과': '-'
                            })

                            # Case 1, 2, 3 데이터 추가
                            for case in case_data:
                                # 최고 감축률인지 확인
                                is_best = (case['감축률(%)'] != '-' and
                                          float(case['감축률(%)']) == best_reduction)

                                # 효과 텍스트를 기반으로 이모지 표시 (일관성 유지)
                                effect_text = case.get('효과', '')
                                effect_emoji = ""
                                if effect_text == "높음":
                                    effect_emoji = "🟢"
                                elif effect_text == "중간":
                                    effect_emoji = "🟡"
                                elif effect_text == "낮음":
                                    effect_emoji = "🔴"
                                else:
                                    # 효과 텍스트가 없는 경우 감축률 기준으로 fallback
                                    if case['감축률(%)'] != '-':
                                        reduction = float(case['감축률(%)'])
                                        if reduction >= 15:
                                            effect_emoji = "🟢"
                                        elif reduction >= 5:
                                            effect_emoji = "🟡"
                                        else:
                                            effect_emoji = "🔴"

                                # Case 이름에 최고 효과 표시
                                case_name = case['Case']
                                if is_best:
                                    case_name = f"🏆 {case_name}"

                                table_data.append({
                                    'Case': case_name,
                                    'PCF (kgCO2eq)': f"{case['PCF_총합(kgCO2eq)']:.3f}",
                                    '감축률 (%)': f"{case['감축률(%)']}%",
                                    '효과': f"{effect_emoji} {case['효과']}"
                                })

                            # DataFrame으로 변환하여 표시
                            df_display = pd.DataFrame(table_data)
                            st.dataframe(
                                df_display,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Case": st.column_config.TextColumn("Case", width="large"),
                                    "PCF (kgCO2eq)": st.column_config.TextColumn("PCF (kgCO2eq)", width="medium"),
                                    "감축률 (%)": st.column_config.TextColumn("감축률 (%)", width="medium"),
                                    "효과": st.column_config.TextColumn("효과", width="small")
                                }
                            )
                        else:
                            st.markdown(f"""
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">
                                <p>Case 데이터 없음</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"""
                        <div class="summary-card">
                            <h4 style="color: var(--text-color); margin-bottom: 15px; text-align: center; border-bottom: 2px solid var(--secondary-color); padding-bottom: 10px;">{scenario_name}</h4>
                            <div style="color: var(--text-secondary); font-size: 0.9rem; text-align: center; padding: 20px;">
                                <p>📊 PCF 요약 데이터 없음</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # ===== 시나리오별 상세 디버그 로그 섹션 (코멘트 아웃) =====
        # st.markdown("---")
        # st.markdown('<div class="simulation-title">🔍 시나리오별 상세 디버그 로그</div>', unsafe_allow_html=True)
        # st.info("📌 각 시나리오의 상세 계산 과정을 확인하세요. 아래 탭에서 시나리오를 선택하면 7단계 계산 프로세스, 자재별 배출계수 변경 내역, 배출계수 매칭 로그를 확인할 수 있습니다.")

        # # 시나리오별 탭 생성 (디버그 로그용)
        # scenario_names = {
        #     'baseline': '기본 시나리오',
        #     'recycling': '재활용&저탄소메탈 시나리오',
        #     'site_change': '생산지 변경 시나리오',
        #     'both': '종합 시나리오'
        # }

        # available_scenarios = [scenario for scenario in scenario_names.keys()
        #                      if scenario in st.session_state.simulation_results]

        # if available_scenarios:
        #     debug_tabs = st.tabs([f"🔍 {scenario_names[scenario]}" for scenario in available_scenarios])

        #     for i, scenario in enumerate(available_scenarios):
        #         with debug_tabs[i]:
        #             try:
        #                 result_dict = st.session_state.simulation_results[scenario]

        #                 if result_dict:
        #                     # 데이터 준비
        #                     all_data = result_dict.get('all_data', pd.DataFrame())
        #                     if not all_data.empty:
        #                         sorted_data = all_data.copy()
        #                         if '저감활동_적용여부' in sorted_data.columns:
        #                             if '자재명' in sorted_data.columns:
        #                                 sorted_data = sorted_data.sort_values(['저감활동_적용여부', '자재명'], ascending=[False, True])
        #                             else:
        #                                 sorted_data = sorted_data.sort_values('저감활동_적용여부', ascending=False)
        #                     else:
        #                         sorted_data = pd.DataFrame()

        #             # 1. 시나리오 개요
        #             with st.expander(f"📋 {scenario_names[scenario]} 계산 개요", expanded=True):
        #                         st.markdown("#### 시나리오 설명")
        #                         if scenario == 'baseline':
        #                                                             st.info("**기본 시나리오**: 저감활동을 적용하지 않은 기준 상태입니다.")
        #                                                             st.write("- 변경 사항 없음")
        #                                                             st.write("- 모든 자재의 기본 배출계수 사용")
        #                                                         elif scenario == 'recycling':
        #                                                             st.info("**재활용&저탄소메탈 시나리오**: 양극재에 재활용 소재 및 저탄소 메탈을 적용합니다.")
        #                                                             st.write("- 양극재 배출계수 수정 적용")
        #                                                             st.write("- 재활용 소재 비율 증가로 인한 배출량 감소")
        #                                                             st.write("- 저탄소 메탈 적용으로 인한 배출량 감소")
        #                                                         elif scenario == 'site_change':
        #                                                             st.info("**생산지 변경 시나리오**: 생산지를 변경하여 전력 배출계수를 개선합니다.")
        #                                                             st.write("- Tier1, Tier2 전력 배출계수 변경")
        #                                                             st.write("- Energy(Tier-1), Energy(Tier-2) 카테고리 배출량 감소")
        #                         
        #                                                             # 생산지 변경 상세 정보 추가
        #                                                             st.markdown("#### 📍 생산지 변경 상세 정보")
        #                         
        #                                                             try:
        #                                                                 # cathode_site.json과 electricity_coef_by_country.json 로드
        #                                                                 from src.utils.file_operations import FileOperations
        #                         
        #                                                                 # 사용자별 파일 로드
        #                                                                 site_file_path = os.path.join(project_root, "input", "cathode_site.json")
        #                                                                 electricity_file_path = os.path.join(project_root, "stable_var", "electricity_coef_by_country.json")
        #                         
        #                                                                 site_data = FileOperations.load_json(site_file_path, user_id=user_id)
        #                                                                 electricity_coef = FileOperations.load_json(electricity_file_path, user_id=user_id)
        #                         
        #                                                                 # CAM (Tier-1) 정보
        #                                                                 cam_before = site_data.get('CAM', {}).get('before', '중국')
        #                                                                 cam_after = site_data.get('CAM', {}).get('after', '한국')
        #                         
        #                                                                 # 배출계수 조회 (없으면 기본 파일에서 fallback)
        #                                                                 cam_before_coef = electricity_coef.get(cam_before, None)
        #                                                                 if cam_before_coef is None:
        #                                                                     # 사용자별 파일에 없으면 기본 파일에서 로드
        #                                                                     default_electricity_coef = FileOperations.load_json(electricity_file_path, user_id=None)
        #                                                                     cam_before_coef = default_electricity_coef.get(cam_before, 0)
        #                         
        #                                                                 cam_after_coef = electricity_coef.get(cam_after, None)
        #                                                                 if cam_after_coef is None:
        #                                                                     default_electricity_coef = FileOperations.load_json(electricity_file_path, user_id=None)
        #                                                                     cam_after_coef = default_electricity_coef.get(cam_after, 0)
        #                         
        #                                                                 # 변화율 계산 (Before가 0이면 계산 불가)
        #                                                                 if cam_before_coef > 0:
        #                                                                     cam_change = ((cam_before_coef - cam_after_coef) / cam_before_coef) * 100
        #                                                                 else:
        #                                                                     cam_change = 0 if cam_after_coef == 0 else float('inf')
        #                                                                 cam_diff = cam_before_coef - cam_after_coef
        #                         
        #                                                                 # pCAM (Tier-2) 정보
        #                                                                 pcam_before = site_data.get('pCAM', {}).get('before', '한국')
        #                                                                 pcam_after = site_data.get('pCAM', {}).get('after', '한국')
        #                         
        #                                                                 # 배출계수 조회 (없으면 기본 파일에서 fallback)
        #                                                                 pcam_before_coef = electricity_coef.get(pcam_before, None)
        #                                                                 if pcam_before_coef is None:
        #                                                                     default_electricity_coef = FileOperations.load_json(electricity_file_path, user_id=None)
        #                                                                     pcam_before_coef = default_electricity_coef.get(pcam_before, 0)
        #                         
        #                                                                 pcam_after_coef = electricity_coef.get(pcam_after, None)
        #                                                                 if pcam_after_coef is None:
        #                                                                     default_electricity_coef = FileOperations.load_json(electricity_file_path, user_id=None)
        #                                                                     pcam_after_coef = default_electricity_coef.get(pcam_after, 0)
        #                         
        #                                                                 # 변화율 계산 (Before가 0이면 계산 불가)
        #                                                                 if pcam_before_coef > 0:
        #                                                                     pcam_change = ((pcam_before_coef - pcam_after_coef) / pcam_before_coef) * 100
        #                                                                 else:
        #                                                                     pcam_change = 0 if pcam_after_coef == 0 else float('inf')
        #                                                                 pcam_diff = pcam_before_coef - pcam_after_coef
        #                         
        #                                                                 # 상세 정보 표시
        #                                                                 st.markdown(f"""
        #                         **━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
        #                         
        #                         **🏭 CAM (Tier-1) 생산지 변경**
        #                         
        #                         - **Before 사이트**: {cam_before}
        #                           - 전력 배출계수: `{cam_before_coef:.4f} kgCO2eq/kWh`
        #                         - **After 사이트**: {cam_after}
        #                           - 전력 배출계수: `{cam_after_coef:.4f} kgCO2eq/kWh`
        #                         - **변화**: `{cam_change:+.2f}%` ({"↓" if cam_diff > 0 else "↑"} `{abs(cam_diff):.4f} kgCO2eq/kWh`)
        #                         
        #                         **━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
        #                         
        #                         **🏭 pCAM (Tier-2) 생산지 변경**
        #                         
        #                         - **Before 사이트**: {pcam_before}
        #                           - 전력 배출계수: `{pcam_before_coef:.4f} kgCO2eq/kWh`
        #                         - **After 사이트**: {pcam_after}
        #                           - 전력 배출계수: `{pcam_after_coef:.4f} kgCO2eq/kWh`
        #                         - **변화**: `{pcam_change:+.2f}%` ({"↓" if pcam_diff > 0 else "↑"} `{abs(pcam_diff):.4f} kgCO2eq/kWh`)
        #                         
        #                         **━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
        #                                                                 """)
        #                         
        #                                                             except Exception as e:
        #                                                                 st.warning(f"⚠️ 생산지 변경 상세 정보를 불러올 수 없습니다: {e}")
        #                                                         elif scenario == 'both':
        #                                                             st.info("**종합 시나리오**: 재활용&저탄소메탈 + 생산지 변경을 동시에 적용합니다.")
        #                                                             st.write("- 양극재 배출계수 수정 적용")
        #                                                             st.write("- Tier1, Tier2 전력 배출계수 변경")
        #                                                             st.write("- 두 저감활동의 시너지 효과")
        #                         
        #                                                         st.markdown("---")
        #                                                         st.markdown("#### PCF 계산 결과 요약")
        #                         
        #                                                         # PCF 계산 결과 표시 - 모든 시나리오에서 baseline PCF_reference 기준 사용
        #                                                         if not all_data.empty:
        #                                                             # === 모든 시나리오의 기준은 baseline 시나리오의 PCF_reference ===
        #                                                             baseline_all_data = st.session_state.simulation_results.get('baseline', {}).get('all_data')
        #                         
        #                                                             if baseline_all_data is not None and 'PCF_reference' in baseline_all_data.columns:
        #                                                                 baseline_pcf_ref = baseline_all_data['PCF_reference'].sum()
        #                                                             else:
        #                                                                 # fallback: 현재 시나리오의 PCF_reference 사용
        #                                                                 baseline_pcf_ref = all_data['PCF_reference'].sum() if 'PCF_reference' in all_data.columns else 0
        #                         
        #                                                             st.metric("PCF Reference (Baseline 기준)", f"{baseline_pcf_ref:.3f} kgCO2eq")
        #                         
        #                                                             pcf_cols = [col for col in all_data.columns if col.startswith('PCF_case')]
        #                                                             if pcf_cols:
        #                                                                 cols = st.columns(len(pcf_cols))
        #                                                                 for idx, pcf_col in enumerate(pcf_cols):
        #                                                                     pcf_val = all_data[pcf_col].sum()
        #                                                                     with cols[idx]:
        #                                                                         # baseline PCF_reference를 기준으로 감축률 계산
        #                                                                         if baseline_pcf_ref > 0:
        #                                                                             reduction = ((baseline_pcf_ref - pcf_val) / baseline_pcf_ref) * 100
        #                                                                         else:
        #                                                                             reduction = 0
        #                                                                         st.metric(
        #                                                                             pcf_col.replace('PCF_', '').upper(),
        #                                                                             f"{pcf_val:.3f} kgCO2eq",
        #                                                                             f"{-reduction:+.2f}%"
        #                                                                         )
        #                         
        #                                                     # 2. 단계별 계산 과정
        #                                                     with st.expander(f"🔢 단계별 계산 과정 상세", expanded=False):
        #                                                         st.markdown("#### 7단계 계산 프로세스")
        #                         
        #                                                         st.markdown("##### 1️⃣ 저감활동 적용 가능한 행 추출")
        #                                                         if not sorted_data.empty:
        #                                                             applied_materials = sorted_data[sorted_data['저감활동_적용여부'] == 1.0]
        #                                                             st.write(f"- 전체 자재 수: {len(sorted_data)}개")
        #                                                             st.write(f"- 저감활동 적용 자재 수: {len(applied_materials)}개")
        #                                                             if not applied_materials.empty:
        #                                                                 st.write("- 적용 대상 자재품목:")
        #                                                                 for category in applied_materials['자재품목'].unique():
        #                                                                     count = len(applied_materials[applied_materials['자재품목'] == category])
        #                                                                     st.write(f"  • {category}: {count}개")
        #                         
        #                                                         st.markdown("##### 2️⃣ 공식 데이터 매칭")
        #                                                         st.write("- ref_formula_df와 매칭하여 배출계수 공식 적용")
        #                                                         st.write("- 배출계수명 기반으로 정확한 매칭 수행")
        #                         
        #                                                         st.markdown("##### 3️⃣ 배출계수 수정 계산")
        #                                                         if scenario == 'baseline':
        #                                                             st.write("- 기본 시나리오: 배출계수 수정 없음")
        #                                                         elif scenario == 'recycling':
        #                                                             st.write("- 양극재 배출계수 수정")
        #                                                             st.write("- 재활용 소재 비율 적용")
        #                                                             st.write("- 저탄소 메탈 배출계수 적용")
        #                                                         elif scenario == 'site_change':
        #                                                             st.write("- 전력 배출계수 변경")
        #                                                             st.write("- Tier1, Tier2 RE100 비율 반영")
        #                                                         elif scenario == 'both':
        #                                                             st.write("- 양극재 배출계수 수정")
        #                                                             st.write("- 전력 배출계수 변경")
        #                         
        #                                                         st.markdown("##### 4️⃣ PCF 값 계산")
        #                                                         st.write("- 수정된 배출계수로 PCF_case1, PCF_case2, PCF_case3 계산")
        #                                                         st.write("- 계산 공식: PCF = 배출계수 × 제품총소요량(kg)")
        #                         
        #                                                         st.markdown("##### 5️⃣ 전체 데이터 병합")
        #                                                         st.write("- 저감활동 적용 자재와 미적용 자재 병합")
        #                                                         st.write("- scenario_df와 계산 결과 통합")
        #                         
        #                                                         st.markdown("##### 6️⃣ PCF_case 열 NaN 값 처리")
        #                                                         st.write("- NaN 값은 PCF_reference 값으로 대체")
        #                                                         st.write("- 모든 행에 완전한 PCF 값 보장")
        #                         
        #                                                         st.markdown("##### 7️⃣ 최종 PCF 합계 출력")
        #                                                         st.write("- 시나리오별 최종 PCF 합계 계산")
        #                                                         st.write("- 감축률 계산 및 출력")
        #                         
        #                                                     # 3. 자재별 배출계수 변경 내역
        #                                                     with st.expander(f"📊 자재별 배출계수 변경 내역", expanded=False):
        #                                                         if scenario == 'baseline':
        #                                                             st.markdown("#### 기본 시나리오 자재 목록 (배출계수 변경 없음)")
        #                                                             st.info("기본 시나리오는 배출계수 변경이 없지만, 저감활동 적용 대상 자재들의 기준 배출계수를 확인할 수 있습니다.")
        #                                                         else:
        #                                                             st.markdown("#### 배출계수 변경 자재 목록")
        #                         
        #                                                         # 저감활동 적용 자재만 필터링 (baseline 포함 모든 시나리오)
        #                                                         if not sorted_data.empty:
        #                                                             changed_materials = sorted_data[sorted_data['저감활동_적용여부'] == 1.0].copy()
        #                                                         else:
        #                                                             changed_materials = pd.DataFrame()
        #                         
        #                                                         if not changed_materials.empty:
        #                                                             # 자재품목별로 그룹화
        #                                                             for category in changed_materials['자재품목'].unique():
        #                                                                 st.markdown(f"##### {category}")
        #                                                                 category_data = changed_materials[changed_materials['자재품목'] == category]
        #                         
        #                                                                 # 변경 내역 표시
        #                                                                 for idx, row in category_data.iterrows():
        #                                                                     material_name = row.get('자재명', 'N/A')
        #                                                                     emission_coef = row.get('배출계수', 0)
        #                                                                     quantity = row.get('제품총소요량(kg)', 0)
        #                                                                     emission = row.get('배출량(kgCO2eq)', 0)
        #                         
        #                                                                     st.write(f"**{material_name}**")
        #                                                                     st.write(f"  - 배출계수: {emission_coef:.6f}")
        #                                                                     st.write(f"  - 소요량: {quantity:.6f} kg")
        #                                                                     st.write(f"  - 배출량: {emission:.6f} kgCO2eq")
        #                         
        #                                                                     # PCF 값들 비교
        #                                                                     if 'PCF_reference' in row.index:
        #                                                                         pcf_ref = row['PCF_reference']
        #                                                                         st.write(f"  - PCF_reference: {pcf_ref:.6f} kgCO2eq")
        #                         
        #                                                                     # baseline 시나리오에서는 PCF_case들이 모두 동일하므로 표시만 함
        #                                                                     pcf_cases = [col for col in row.index if col.startswith('PCF_case')]
        #                                                                     for pcf_col in pcf_cases:
        #                                                                         pcf_val = row[pcf_col]
        #                                                                         if scenario != 'baseline' and 'PCF_reference' in row.index:
        #                                                                             # baseline이 아닌 경우에만 감축률 계산
        #                                                                             reduction = ((pcf_ref - pcf_val) / pcf_ref) * 100 if pcf_ref > 0 else 0
        #                                                                             st.write(f"  - {pcf_col}: {pcf_val:.6f} kgCO2eq ({reduction:.2f}% 감축)")
        #                                                                         else:
        #                                                                             # baseline인 경우 감축률 없이 값만 표시
        #                                                                             st.write(f"  - {pcf_col}: {pcf_val:.6f} kgCO2eq")
        #                                                                     st.markdown("---")
        #                                                         else:
        #                                                             st.info("배출계수 변경이 적용된 자재가 없습니다.")
        #                         
        #                                                     # 4. 배출계수 매칭 디버그 로그
        #                                                     debug_log_key = f'debug_logs_{scenario}'
        #                                                     if debug_log_key in st.session_state and st.session_state[debug_log_key]:
        #                                                         with st.expander(f"🔍 배출계수 매칭 상세 로그 ({len(st.session_state[debug_log_key])}개 항목)", expanded=False):
        #                                                             st.info(f"{scenario_names[scenario]} 시뮬레이터 실행 중 배출계수 매칭 과정의 상세 로그입니다.")
        #                         
        #                                                             # 로그 레벨별 필터링 옵션
        #                                                             log_levels = ['전체', 'info', 'debug', 'warning', 'error']
        #                                                             selected_level = st.selectbox(
        #                                                                 "로그 레벨 필터:",
        #                                                                 log_levels,
        #                                                                 key=f"log_filter_{scenario}"
        #                                                             )
        #                         
        #                                                             # 필터링된 로그 표시
        #                                                             filtered_logs = st.session_state[debug_log_key]
        #                                                             if selected_level != '전체':
        #                                                                 filtered_logs = [log for log in filtered_logs if log['level'] == selected_level]
        #                         
        #                                                             st.write(f"표시된 로그: {len(filtered_logs)}개")
        #                         
        #                                                             for log_entry in filtered_logs:
        #                                                                 level = log_entry['level']
        #                                                                 message = log_entry['message']
        #                                                                 if level == "warning":
        #                                                                     st.warning(message)
        #                                                                 elif level == "error":
        #                                                                     st.error(message)
        #                                                                 elif level == "debug":
        #                                                                     st.code(message)
        #                                                                 else:
        #                                                                     st.write(message)
        #                                             except Exception as e:
        #                                 st.error(f"{scenario_names[scenario]} 디버그 로그 표시 중 오류: {e}")

        st.markdown("---")
        st.markdown('<div class="simulation-title">시뮬레이션 결과 분석</div>', unsafe_allow_html=True)

        # 시나리오별 데이터 탭 생성
        available_scenarios = [scenario for scenario in scenario_names.keys()
                             if scenario in st.session_state.simulation_results]
        
        if available_scenarios:
            tabs = st.tabs([scenario_names[scenario] for scenario in available_scenarios])
            
            for i, scenario in enumerate(available_scenarios):
                with tabs[i]:
                    try:
                        result_dict = st.session_state.simulation_results[scenario]

                        if result_dict:
                            # 데이터 준비 (코멘트 아웃된 섹션에서 이동)
                            all_data = result_dict.get('all_data', pd.DataFrame())
                            if not all_data.empty:
                                sorted_data = all_data.copy()
                                if '저감활동_적용여부' in sorted_data.columns:
                                    if '자재명' in sorted_data.columns:
                                        sorted_data = sorted_data.sort_values(['저감활동_적용여부', '자재명'], ascending=[False, True])
                                    else:
                                        sorted_data = sorted_data.sort_values('저감활동_적용여부', ascending=False)
                            else:
                                sorted_data = pd.DataFrame()

                            # ===== 하단: 데이터 상세 정보 탭 =====
                            # st.markdown("---")
                            # st.write("📊 데이터 상세 정보")

                            # 데이터프레임별 탭 생성
                            data_tabs = st.tabs([
                                "📋 전체 데이터",
                                "🔧 수정된 데이터",
                                "📊 자재 요약",
                                "📈 PCF 요약",
                                "🎯 매칭 요약"
                            ])

                            # 전체 데이터 탭
                            with data_tabs[0]:
                                if not all_data.empty:
                                    st.markdown('<div class="data-section">', unsafe_allow_html=True)
                                    st.write(f"**{scenario_names[scenario]} - 전체 데이터 ({len(all_data)}개 행):**")
                                    st.dataframe(sorted_data, use_container_width=True, height=400)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.info("전체 데이터가 없습니다.")

                            # 수정된 데이터 탭
                            with data_tabs[1]:
                                if 'modified_data' in result_dict and not result_dict['modified_data'].empty:
                                    st.markdown('<div class="data-section">', unsafe_allow_html=True)
                                    st.write(f"**{scenario_names[scenario]} - 수정된 데이터 ({len(result_dict['modified_data'])}개 행):**")
                                    # 저감활동 적용 여부로 정렬 (1이 먼저 나오도록), 그 다음 자재명으로 정렬
                                    sorted_modified_data = result_dict['modified_data'].copy()
                                    if '저감활동_적용여부' in sorted_modified_data.columns:
                                        if '자재명' in sorted_modified_data.columns:
                                            sorted_modified_data = sorted_modified_data.sort_values(['저감활동_적용여부', '자재명'], ascending=[False, True])
                                        else:
                                            sorted_modified_data = sorted_modified_data.sort_values('저감활동_적용여부', ascending=False)
                                    st.dataframe(sorted_modified_data, use_container_width=True, height=400)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.info("수정된 데이터가 없습니다.")
                            
                            # 자재 요약 탭
                            with data_tabs[2]:
                                if 'material_data' in result_dict and not result_dict['material_data'].empty:
                                    st.markdown('<div class="data-section">', unsafe_allow_html=True)
                                    st.write(f"**{scenario_names[scenario]} - 자재품목별 요약:**")
                                    st.dataframe(result_dict['material_data'], use_container_width=True, height=400)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.info("자재 요약 데이터가 없습니다.")
                            
                            # PCF 요약 탭
                            with data_tabs[3]:
                                if 'pcf_summary' in result_dict and not result_dict['pcf_summary'].empty:
                                    st.markdown('<div class="data-section">', unsafe_allow_html=True)
                                    st.write(f"**{scenario_names[scenario]} - PCF 요약:**")
                                    st.dataframe(result_dict['pcf_summary'], use_container_width=True, height=400)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.info("PCF 요약 데이터가 없습니다.")
                            
                            # 매칭 요약 탭
                            with data_tabs[4]:
                                if 'matching_summary' in result_dict and not result_dict['matching_summary'].empty:
                                    st.markdown('<div class="data-section">', unsafe_allow_html=True)
                                    st.write(f"**{scenario_names[scenario]} - 매칭 결과 요약:**")
                                    st.dataframe(result_dict['matching_summary'], use_container_width=True, height=400)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.info("매칭 요약 데이터가 없습니다.")
                            

                            
                            # 결과 다운로드
                            st.markdown("---")
                            st.subheader("📥 결과 다운로드")
                            
                            # 다운로드 버튼들을 2열로 배치
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # 전체 데이터 다운로드
                                if 'all_data' in result_dict and not result_dict['all_data'].empty:
                                    # XLSX 파일로 변환
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        result_dict['all_data'].to_excel(writer, index=False, sheet_name='전체 데이터')
                                    xlsx_data = output.getvalue()
                                    st.download_button(
                                        label="📥 전체 데이터 (XLSX)",
                                        data=xlsx_data,
                                        file_name=f"pcf_all_data_{scenario}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                
                                # 수정된 데이터 다운로드
                                if 'modified_data' in result_dict and not result_dict['modified_data'].empty:
                                    # XLSX 파일로 변환
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        result_dict['modified_data'].to_excel(writer, index=False, sheet_name='수정된 데이터')
                                    xlsx_data = output.getvalue()
                                    st.download_button(
                                        label="📥 수정된 데이터 (XLSX)",
                                        data=xlsx_data,
                                        file_name=f"pcf_modified_data_{scenario}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                
                                # 자재 요약 다운로드
                                if 'material_data' in result_dict and not result_dict['material_data'].empty:
                                    # XLSX 파일로 변환
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        result_dict['material_data'].to_excel(writer, index=False, sheet_name='자재 요약')
                                    xlsx_data = output.getvalue()
                                    st.download_button(
                                        label="📥 자재 요약 (XLSX)",
                                        data=xlsx_data,
                                        file_name=f"pcf_summary_{scenario}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                            
                            with col2:
                                # PCF 요약 다운로드
                                if 'pcf_summary' in result_dict and not result_dict['pcf_summary'].empty:
                                    # XLSX 파일로 변환
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        result_dict['pcf_summary'].to_excel(writer, index=False, sheet_name='PCF 요약')
                                    xlsx_data = output.getvalue()
                                    st.download_button(
                                        label="📥 PCF 요약 (XLSX)",
                                        data=xlsx_data,
                                        file_name=f"pcf_pcf_summary_{scenario}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                
                                # 매칭 요약 다운로드
                                if 'matching_summary' in result_dict and not result_dict['matching_summary'].empty:
                                    # XLSX 파일로 변환
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        result_dict['matching_summary'].to_excel(writer, index=False, sheet_name='매칭 요약')
                                    xlsx_data = output.getvalue()
                                    st.download_button(
                                        label="📥 매칭 요약 (XLSX)",
                                        data=xlsx_data,
                                        file_name=f"pcf_matching_summary_{scenario}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )
                                

                        else:
                            st.error(f"{scenario_names[scenario]} 시뮬레이션 결과가 비어있습니다.")
                            
                    except Exception as e:
                        st.error(f"{scenario_names[scenario]} 결과 분석 중 오류가 발생했습니다: {e}")
                        st.exception(e)
        else:
            st.error("시뮬레이션 결과가 없습니다.")
    
    elif not st.session_state.is_loading:
        # 초기 안내
        st.markdown("---")
        st.markdown("""
        <div class="simulation-section">
            <h3 style="color: var(--text-color);">📋 시뮬레이션 안내</h3>
            <p style="color: var(--text-secondary);">PCF 시뮬레이션을 실행하기 전에 다음 사항을 확인해주세요:</p>
            <ol style="color: var(--text-secondary);">
                <li><strong>데이터 준비:</strong> BRM 원본 데이터와 시나리오 데이터가 업로드되어 있어야 합니다</li>
                <li><strong>설정 확인:</strong> 사이드바에서 현재 Case 수와 Tier 수를 확인하세요</li>
                <li><strong>저감활동 적용:</strong> 시나리오 설정에서 저감활동을 적용할 자재를 선택해야 합니다</li>
                <li><strong>로그 레벨 선택:</strong> 사이드바에서 원하는 로그 상세도를 선택하세요</li>
            </ol>
            <p style="color: var(--text-secondary);"><strong>실행 시나리오:</strong> baseline(기준), recycling(재활용&저탄소메탈), site_change(생산지 변경), both(동시적용) 총 4개 시나리오가 자동으로 실행됩니다.</p>
            <p style="color: var(--text-secondary);"><strong>결과 제공:</strong> 각 시나리오별로 6개의 상세 분석 탭과 요약 데이터를 제공합니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 시뮬레이션 실행 버튼
        st.markdown("---")
        st.markdown('<div class="simulation-title">🚀 시뮬레이션 실행</div>', unsafe_allow_html=True)
        
        # 저감활동이 적용된 자재 확인
        applied_materials = saved_scenario_df[saved_scenario_df['저감활동_적용여부'] == 1.0]
        
        if len(applied_materials) == 0:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.error("❌ 저감활동이 적용된 자재가 없어 시뮬레이션을 실행할 수 없습니다.")
            st.markdown("시나리오 설정 페이지에서 저감활동을 적용할 자재를 선택해주세요.", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # st.markdown('<div class="success-box">', unsafe_allow_html=True)
            # st.success(f"✅ 저감활동이 적용된 자재: {len(applied_materials)}개")
            # st.markdown('</div>', unsafe_allow_html=True)
            
            # 시뮬레이션 실행 버튼
            if st.button("🚀 Execution", type="primary", use_container_width=False):
                log_button_click("🚀 Execution", "execution_btn")
                log_info("PCF 시뮬레이션 실행 시작")
                
                # 로딩 상태 시작
                st.session_state.is_loading = True
                
                try:
                    # CathodeSimulator 초기화 및 기본 분석 실행
                    # st.info("🔄 CathodeSimulator 초기화 및 기본 분석 실행 중...")

                    from src.cathode_simulator import CathodeSimulator

                    # Baseline용 CathodeSimulator 생성 (site='before' 사용)
                    cathode_simulator = CathodeSimulator(verbose=verbose_option, user_id=user_id)

                    # Baseline 시나리오용 전력계수 업데이트 (site='before' 명시적 호출)
                    baseline_coefficient_data = cathode_simulator.update_electricity_emission_factor(site='before')
                    if baseline_coefficient_data:
                        cathode_simulator.coefficient_data = baseline_coefficient_data
                        log_info(f"✅ Baseline 시나리오용 전력계수 업데이트 완료")
                        if 'Energy(Tier-1)' in baseline_coefficient_data and '전력' in baseline_coefficient_data['Energy(Tier-1)']:
                            log_info(f"   - Energy(Tier-1) 전력계수: {baseline_coefficient_data['Energy(Tier-1)']['전력']['배출계수']:.6f}")
                        if 'Energy(Tier-2)' in baseline_coefficient_data and '전력' in baseline_coefficient_data['Energy(Tier-2)']:
                            log_info(f"   - Energy(Tier-2) 전력계수: {baseline_coefficient_data['Energy(Tier-2)']['전력']['배출계수']:.6f}")
                    else:
                        log_error(f"❌ Baseline 시나리오용 전력계수 업데이트 실패")
                    
                    # 디버그: CathodeSimulator 계수 데이터 확인
                    log_info("=== CathodeSimulator 계수 데이터 디버그 ===")
                    if cathode_simulator.coefficient_data:
                        for category, data in cathode_simulator.coefficient_data.items():
                            if isinstance(data, dict):
                                log_info(f"카테고리: {category}")
                                for material, info in data.items():
                                    if isinstance(info, dict) and '소요량' in info:
                                        quantity = info.get('소요량', 0)
                                        emission_coef = info.get('배출계수', 0)
                                        log_info(f"  • {material}: 소요량={quantity:.6f}, 배출계수={emission_coef:.6f}")
                    log_info("===================================")
                    
                    # 모든 시나리오의 데이터 생성
                    all_scenarios_data = cathode_simulator.generate_all_scenarios_data()
                    
                    if not all_scenarios_data:
                        st.error("❌ CathodeSimulator에서 시나리오 데이터 생성에 실패했습니다.")
                        st.session_state.is_loading = False
                        return
                    
                    # st.success("✅ CathodeSimulator 기본 분석 완료!")
                    
                    # basic_df 생성 (CathodeSimulator 결과를 데이터프레임으로 변환)
                    basic_df_data = []
                    
                    # Baseline 데이터
                    if all_scenarios_data.get('baseline'):
                        baseline = all_scenarios_data['baseline']
                        basic_df_data.append({
                            '시나리오': 'Baseline',
                            '총_배출량': baseline['emission_data']['총_배출량'],
                            '감축률_퍼센트': 0.0,  # Baseline은 감축 없음
                            '재활용_비율_퍼센트': 0.0,  # Baseline은 재활용&저탄소메탈 없음
                            'Energy_Tier1_전력_기여도_퍼센트': baseline['emission_data']['카테고리별_기여도'].get('Energy(Tier-1)', 0),
                            'Energy_Tier2_전력_기여도_퍼센트': baseline['emission_data']['카테고리별_기여도'].get('Energy(Tier-2)', 0)
                        })
                    
                    # 재활용&저탄소메탈만 적용 데이터
                    if all_scenarios_data.get('recycling_only'):
                        recycling = all_scenarios_data['recycling_only']
                        simulation_result = recycling['simulation_result']
                        basic_df_data.append({
                            '시나리오': '재활용&저탄소메탈 적용',
                            '총_배출량': simulation_result['total_emission'],
                            '감축률_퍼센트': simulation_result['reduction_rate'],
                            '재활용_비율_퍼센트': simulation_result['recycling_ratio_percent'],
                            'Energy_Tier1_전력_기여도_퍼센트': simulation_result['category_contributions'].get('Energy(Tier-1)', 0),
                            'Energy_Tier2_전력_기여도_퍼센트': simulation_result['category_contributions'].get('Energy(Tier-2)', 0)
                        })
                    
                    # 사이트 변경만 데이터
                    if all_scenarios_data.get('site_change_only'):
                        site_change = all_scenarios_data['site_change_only']
                        basic_df_data.append({
                            '시나리오': '사이트 변경',
                            '총_배출량': site_change['after_data']['emission_data']['총_배출량'],
                            '감축률_퍼센트': site_change['emission_change_rate'],
                            '재활용_비율_퍼센트': 0.0,  # 사이트 변경만 적용
                            'Energy_Tier1_전력_기여도_퍼센트': site_change['after_data']['emission_data']['카테고리별_기여도'].get('Energy(Tier-1)', 0),
                            'Energy_Tier2_전력_기여도_퍼센트': site_change['after_data']['emission_data']['카테고리별_기여도'].get('Energy(Tier-2)', 0)
                        })
                    
                    # 종합 데이터
                    if all_scenarios_data.get('combined'):
                        combined = all_scenarios_data['combined']
                        basic_df_data.append({
                            '시나리오': '재활용&저탄소메탈 + 사이트 변경',
                            '총_배출량': combined['after_recycling']['simulation_result']['total_emission'],
                            '감축률_퍼센트': combined['emission_change_rate'],
                            '재활용_비율_퍼센트': combined['recycling_ratio'] * 100,
                            'Energy_Tier1_전력_기여도_퍼센트': combined['after_recycling']['simulation_result']['category_contributions'].get('Energy(Tier-1)', 0),
                            'Energy_Tier2_전력_기여도_퍼센트': combined['after_recycling']['simulation_result']['category_contributions'].get('Energy(Tier-2)', 0)
                        })
                    
                    basic_df = pd.DataFrame(basic_df_data)
                    
                    # basic_df 디버깅 정보 출력
                    log_info(f"📊 basic_df 생성 완료: {len(basic_df)}개 시나리오")
                    log_info(f"📊 basic_df 컬럼: {list(basic_df.columns)}")
                    log_info(f"📊 basic_df 시나리오 목록:")
                    for idx, row in basic_df.iterrows():
                        log_info(f"  • {row['시나리오']}: 감축률 {row['감축률_퍼센트']:.2f}%")
                    
                    # st.success(f"✅ 기본 분석 데이터 생성 완료: {len(basic_df)}개 시나리오")
                    
                    # 모든 시나리오에 대해 시뮬레이션 실행
                    scenarios = ['baseline', 'recycling', 'site_change', 'both']
                    scenario_names = {
                        'baseline': '기준 시나리오',
                        'recycling': '재활용&저탄소메탈 시나리오',
                        'site_change': '생산지 변경 시나리오',
                        'both': '종합 시나리오'
                    }
                    results = {}
                    baseline_pcf_reference = None  # 모든 시나리오의 기준값 저장
                    
                    # original_df에 배출계수명 고유화 및 중복 제거 적용
                    if original_df is not None and '배출계수명' in original_df.columns:
                        from app_helper import apply_unique_emission_factor_names_with_index

                        # 1단계: 중복 검사 (고유화 전)
                        log_info("🔍 original_df 중복 데이터 검사 시작...")
                        original_row_count = len(original_df)
                        log_info(f"   - 고유화 전 총 행 수: {original_row_count}개")

                        # 2단계: 배출계수명 고유화 (인덱스 포함)
                        log_info("🔧 original_df 배출계수명 고유화 시작...")
                        original_df = apply_unique_emission_factor_names_with_index(original_df)
                        log_info(f"✅ 배출계수명 고유화 완료: {original_df['배출계수명'].nunique()}개 고유 배출계수명")

                        # 3단계: 중복 행 제거 (고유화 후) - 더 많은 컬럼으로 정확한 중복 검사
                        # 고유 ID 생성 (행 인덱스 + 자재명 + 자재품목 + 배출량)
                        # 배출량을 포함하여 같은 자재가 다른 반제품에 사용될 때 구분
                        original_df['_unique_row_id'] = (
                            original_df.index.astype(str) + '_' +
                            original_df['자재명'].astype(str) + '_' +
                            original_df['자재품목'].astype(str) + '_' +
                            original_df['배출량(kgCO2eq)'].astype(str)
                        )

                        # 확장된 중복 검사 키 (더 많은 컬럼 포함)
                        key_columns = ['자재명', '자재품목', '배출계수명', '배출계수', '제품총소요량(kg)', '배출량(kgCO2eq)', '지역', '자재코드']
                        available_key_columns = [col for col in key_columns if col in original_df.columns]

                        if available_key_columns:
                            log_info(f"🔍 중복 제거 기준 컬럼: {available_key_columns}")

                            # 중복 행 확인
                            duplicates = original_df[original_df.duplicated(subset=available_key_columns, keep=False)]
                            if not duplicates.empty:
                                log_warning(f"⚠️ 확장된 기준으로 중복 행 발견: {len(duplicates)}개")

                                # 자재품목별 중복 상세 정보
                                for category in duplicates['자재품목'].unique():
                                    category_dups = duplicates[duplicates['자재품목'] == category]
                                    log_warning(f"   - {category}: {len(category_dups)}개 중복")

                                    # 음극재의 경우 상세 정보 출력
                                    if category == '음극재':
                                        log_info(f"   📋 음극재 중복 상세:")
                                        for idx, dup_row in category_dups.iterrows():
                                            log_info(f"      • {dup_row['자재명']}: 배출계수={dup_row.get('배출계수', 'N/A')}, "
                                                    f"소요량={dup_row.get('제품총소요량(kg)', 'N/A')}, "
                                                    f"배출량={dup_row.get('배출량(kgCO2eq)', 'N/A')}")

                            # 중복 제거 (첫 번째 행만 유지) - 단, _unique_row_id도 함께 사용하여 완전 동일한 행만 제거
                            combined_key_columns = available_key_columns + ['_unique_row_id']
                            original_df = original_df.drop_duplicates(subset=combined_key_columns, keep='first')

                            # _unique_row_id 컬럼 제거
                            original_df = original_df.drop(columns=['_unique_row_id'])

                            deduplicated_row_count = len(original_df)
                            removed_count = original_row_count - deduplicated_row_count

                            if removed_count > 0:
                                log_info(f"✅ 중복 제거 완료: {removed_count}개 행 제거됨")
                                log_info(f"   - 제거 후 총 행 수: {deduplicated_row_count}개")
                            else:
                                log_info(f"✅ 중복 없음: 모든 행이 고유함")

                            # 음극재 개수 재확인
                            anode_count_after = len(original_df[original_df['자재품목'] == '음극재'])
                            log_info(f"📊 중복 제거 후 음극재 개수: {anode_count_after}개")

                        # 4단계: 고유화 결과 로그 (음극재만)
                        anode_rows = original_df[original_df['자재품목'] == '음극재']
                        if not anode_rows.empty:
                            log_info("📊 original_df 음극재 배출계수명 고유화 결과:")
                            for idx, row in anode_rows.iterrows():
                                log_info(f"  • {row['자재명']}")
                                log_info(f"    - 원본: {row.get('배출계수명_원본', 'N/A')}")
                                log_info(f"    - 고유: {row['배출계수명']}")
                                log_info(f"    - 배출계수: {row['배출계수']:.6f}")

                    with st.spinner("모든 시나리오 시뮬레이션을 실행하고 있습니다..."):
                        for scenario in scenarios:
                            # st.info(f"🔄 {scenario_names[scenario]} 실행 중...")
                            log_info(f"시나리오 시작: {scenario_names[scenario]} ({scenario})")

                            # 각 시나리오마다 완전히 새로운 RuleBasedSim 인스턴스 생성
                            # DataFrame들을 깊은 복사하여 시나리오 간 독립성 보장
                            simulator = RuleBasedSim(
                                scenario_df=saved_scenario_df.copy(deep=True),  # 깊은 복사
                                ref_formula_df=ref_formula_df.copy(deep=True),  # 깊은 복사
                                ref_proportions_df=ref_proportions_df.copy(deep=True),  # 깊은 복사
                                original_df=original_df.copy(deep=True) if original_df is not None else None,  # 깊은 복사
                                verbose="debug",  # 상세 로그를 위해 debug 레벨로 설정
                                user_id=user_id  # 사용자별 설정 파일 로드를 위해 user_id 전달
                            )

                            # 각 시나리오마다 새로운 CathodeSimulator 생성 (완전 독립)
                            # 이전 시나리오의 coefficient_data가 영향을 주지 않도록 함
                            from src.cathode_simulator import CathodeSimulator
                            simulator.cathode_simulator = CathodeSimulator(verbose=False, user_id=user_id)
                            log_info(f"✅ {scenario} 시나리오용 새로운 CathodeSimulator 생성")
                            
                            # 시나리오별로 다른 데이터와 설정 사용
                            if scenario == 'site_change':
                                log_info(f"🔧 {scenario} 시나리오: 전력 배출계수가 변경된 데이터로 시뮬레이션")

                                # site_change 시나리오용 CathodeSimulator 생성 (전력 배출계수 변경 적용)
                                from src.cathode_simulator import CathodeSimulator
                                site_change_simulator = CathodeSimulator(verbose=False, user_id=user_id)
                                
                                # 사이트 변경 시나리오의 경우 전력 배출계수가 변경된 데이터 사용
                                if all_scenarios_data.get('site_change_only'):
                                    site_change_data = all_scenarios_data['site_change_only']
                                    
                                    # after 사이트의 전력 배출계수로 업데이트된 데이터 사용
                                    after_site_data = site_change_simulator.generate_baseline_data(site='after')
                                    if after_site_data:
                                        # after 사이트 데이터를 기반으로 시뮬레이션
                                        log_info(f"📊 사이트 변경 시나리오: after 사이트 데이터 사용")
                                        log_info(f"📊 사이트 변경 시나리오 감축률: {site_change_data['emission_change_rate']:.2f}%")
                                        
                                        # after 사이트의 전력 배출계수 정보를 simulator에 적용
                                        simulator.cathode_simulator = site_change_simulator
                                        simulator.cathode_simulator.coefficient_data = after_site_data['updated_data']
                                        
                                        # 전력 배출계수 변경을 반영하기 위해 basic_df의 감축률을 조정
                                        # 사이트 변경 시나리오의 실제 감축률을 사용
                                        after_emission = after_site_data['emission_data']['총_배출량']
                                        baseline_emission = basic_df[basic_df['시나리오'] == 'Baseline']['총_배출량'].iloc[0]
                                        site_change_reduction_rate = ((baseline_emission - after_emission) / baseline_emission) * 100
                                        log_info(f"📊 사이트 변경 시나리오 실제 감축률: {site_change_reduction_rate:.2f}%")
                                        
                                        # basic_df에서 사이트 변경 시나리오의 감축률을 업데이트
                                        site_change_mask = basic_df['시나리오'] == '사이트 변경'
                                        if site_change_mask.any():
                                            basic_df.loc[site_change_mask, '감축률_퍼센트'] = site_change_reduction_rate
                                            log_info(f"✅ basic_df에서 사이트 변경 시나리오 감축률 업데이트: {site_change_reduction_rate:.2f}%")
                                        
                                        # ref_formula_df도 after 사이트 데이터로 업데이트
                                        simulator.ref_formula_df = simulator._update_ref_formula_with_site_data(
                                            after_site_data['updated_data']
                                        )
                                        
                                        # site_change 시나리오는 전력계수만 변경하고 RE100 비율은 유지
                                        # 따라서 ref_proportions_df를 업데이트하지 않음
                                        # (update_ref_proportions_with_tier_contributions는 Tier 기여도를 RE100 비율로 착각하는 버그가 있음)
                                        log_info(f"📊 사이트 변경 시나리오: ref_proportions_df는 변경하지 않음 (RE100 비율 유지)")
                                        
                                        site_change_tier1_elec = after_site_data['updated_data']['Energy(Tier-1)']['전력']['배출계수']
                                        site_change_tier2_elec = after_site_data['updated_data']['Energy(Tier-2)']['전력']['배출계수']
                                        log_info(f"📊 사이트 변경 시나리오 Energy(Tier-1) 전력 배출계수: {site_change_tier1_elec}")
                                        log_info(f"📊 사이트 변경 시나리오 Energy(Tier-2) 전력 배출계수: {site_change_tier2_elec}")
                                        log_info(f"📊 사이트 변경 시나리오 ref_proportions_df 업데이트 완료")

                                        # 세션 상태에 저장 (st.rerun() 후에도 유지)
                                        st.session_state['electricity_coef_site_change'] = {
                                            'tier1': site_change_tier1_elec,
                                            'tier2': site_change_tier2_elec
                                        }

                                        # original_df도 after 사이트의 전력 배출계수로 업데이트
                                        simulator.original_df = simulator._update_original_df_with_site_data(
                                            simulator.original_df.copy() if simulator.original_df is not None else None,
                                            after_site_data['updated_data']
                                        )
                                        log_info(f"✅ 사이트 변경 시나리오: original_df 전력 배출계수 업데이트 완료")

                                    scenario_basic_df = basic_df
                                else:
                                    scenario_basic_df = basic_df
                                    
                            elif scenario == 'both':
                                log_info(f"🔧 {scenario} 시나리오: 재활용&저탄소메탈 + 사이트 변경 모두 적용")

                                # both 시나리오용 CathodeSimulator 생성 (전력 배출계수 변경 + 재활용&저탄소메탈 적용)
                                from src.cathode_simulator import CathodeSimulator
                                both_simulator = CathodeSimulator(verbose=False, user_id=user_id)
                                
                                # 사이트 변경과 재활용&저탄소메탈이 모두 적용된 데이터 사용
                                if all_scenarios_data.get('combined'):
                                    combined_data = all_scenarios_data['combined']
                                    
                                    # after 사이트의 전력 배출계수로 업데이트된 데이터 사용
                                    after_site_data = both_simulator.generate_baseline_data(site='after')
                                    if after_site_data:
                                        # after 사이트 데이터를 기반으로 시뮬레이션
                                        log_info(f"📊 종합 시나리오: after 사이트 데이터 사용")
                                        log_info(f"📊 종합 시나리오 감축률: {combined_data['emission_change_rate']:.2f}%")
                                        
                                        # after 사이트의 전력 배출계수 정보를 simulator에 적용
                                        simulator.cathode_simulator = both_simulator
                                        simulator.cathode_simulator.coefficient_data = after_site_data['updated_data']
                                        
                                        # 전력 배출계수 변경을 반영하기 위해 basic_df의 감축률을 조정
                                        # 종합 시나리오의 실제 감축률을 사용 (after 사이트 + 재활용&저탄소메탈 데이터에서 계산)
                                        after_emission = after_site_data['emission_data']['총_배출량']
                                        baseline_emission = basic_df[basic_df['시나리오'] == 'Baseline']['총_배출량'].iloc[0]
                                        recycling_reduction_rate = basic_df[basic_df['시나리오'] == '재활용&저탄소메탈 적용']['감축률_퍼센트'].iloc[0]
                                        
                                        # 전력 배출계수 변경으로 인한 감축률
                                        power_reduction_rate = ((baseline_emission - after_emission) / baseline_emission) * 100
                                        
                                        # 종합 감축률 (전력 + 재활용&저탄소메탈)
                                        combined_reduction_rate = power_reduction_rate + recycling_reduction_rate
                                        log_info(f"📊 종합 시나리오 실제 감축률: {combined_reduction_rate:.2f}% (전력: {power_reduction_rate:.2f}% + 재활용&저탄소메탈: {recycling_reduction_rate:.2f}%)")
                                        
                                        # basic_df에서 종합 시나리오의 감축률을 업데이트
                                        combined_mask = basic_df['시나리오'] == '재활용&저탄소메탈 + 사이트 변경'
                                        if combined_mask.any():
                                            basic_df.loc[combined_mask, '감축률_퍼센트'] = combined_reduction_rate
                                            log_info(f"✅ basic_df에서 종합 시나리오 감축률 업데이트: {combined_reduction_rate:.2f}%")
                                        
                                        # ref_formula_df도 after 사이트 데이터로 업데이트
                                        simulator.ref_formula_df = simulator._update_ref_formula_with_site_data(
                                            after_site_data['updated_data']
                                        )

                                        # both 시나리오도 site_change와 동일하게 전력계수만 변경하고 RE100 비율은 유지
                                        # 따라서 ref_proportions_df를 업데이트하지 않음
                                        # (update_ref_proportions_with_tier_contributions는 Tier 기여도를 RE100 비율로 착각하는 버그가 있음)
                                        log_info(f"📊 종합 시나리오: ref_proportions_df는 변경하지 않음 (RE100 비율 유지)")

                                        log_info(f"📊 종합 시나리오 Energy(Tier-1) 전력 배출계수: {after_site_data['updated_data']['Energy(Tier-1)']['전력']['배출계수']}")
                                        log_info(f"📊 종합 시나리오 Energy(Tier-2) 전력 배출계수: {after_site_data['updated_data']['Energy(Tier-2)']['전력']['배출계수']}")

                                        # original_df도 after 사이트의 전력 배출계수로 업데이트
                                        simulator.original_df = simulator._update_original_df_with_site_data(
                                            simulator.original_df.copy() if simulator.original_df is not None else None,
                                            after_site_data['updated_data']
                                        )
                                        log_info(f"✅ 종합 시나리오: original_df 전력 배출계수 업데이트 완료")

                                    scenario_basic_df = basic_df
                                else:
                                    scenario_basic_df = basic_df
                                    
                            elif scenario == 'recycling':
                                log_info(f"🔧 {scenario} 시나리오: 재활용&저탄소메탈만 적용 (전력 배출계수 변경 없음)")
                                # recycling 시나리오는 기존 전력 배출계수 사용 (양극재 배출계수만 변경)
                                scenario_basic_df = basic_df
                                
                            else:  # baseline
                                log_info(f"🔧 {scenario} 시나리오: 기본 시나리오 (변경 없음)")
                                # baseline 시나리오는 모든 기본값 사용
                                scenario_basic_df = basic_df

                                # Baseline 시나리오용 CathodeSimulator의 전력계수 업데이트
                                if simulator.cathode_simulator:
                                    log_info("🔧 Baseline 시나리오: 전력계수 업데이트 시작")
                                    # site='before'로 전력계수 업데이트
                                    baseline_updated_data = simulator.cathode_simulator.update_electricity_emission_factor(site='before')
                                    if baseline_updated_data:
                                        simulator.cathode_simulator.coefficient_data = baseline_updated_data
                                        log_info("✅ Baseline 시나리오: 전력계수 업데이트 완료")

                                # 디버그: baseline 시나리오가 사용하는 전력계수 확인 및 저장
                                if simulator.cathode_simulator and simulator.cathode_simulator.coefficient_data:
                                    coef_data = simulator.cathode_simulator.coefficient_data
                                    baseline_elec_debug = {}
                                    if 'Energy(Tier-1)' in coef_data and '전력' in coef_data['Energy(Tier-1)']:
                                        tier1_elec = coef_data['Energy(Tier-1)']['전력']['배출계수']
                                        baseline_elec_debug['tier1'] = tier1_elec
                                        log_info(f"📊 Baseline 시나리오 Energy(Tier-1) 전력 배출계수: {tier1_elec:.6f}")
                                    if 'Energy(Tier-2)' in coef_data and '전력' in coef_data['Energy(Tier-2)']:
                                        tier2_elec = coef_data['Energy(Tier-2)']['전력']['배출계수']
                                        baseline_elec_debug['tier2'] = tier2_elec
                                        log_info(f"📊 Baseline 시나리오 Energy(Tier-2) 전력 배출계수: {tier2_elec:.6f}")

                                    # 세션 상태에 저장 (st.rerun() 후에도 유지)
                                    st.session_state['electricity_coef_baseline'] = baseline_elec_debug
                            
                            # 시나리오별 업데이트는 calculate_modified_coefficients 함수에서 수행됨
                            log_info(f"🔧 {scenario} 시나리오: calculate_modified_coefficients에서 시나리오별 업데이트 수행")
                            
                            # run_simulation 실행 (시나리오별 업데이트는 calculate_modified_coefficients에서 수행)
                            log_info(f"시뮬레이션 실행: {scenario} 시나리오")
                            sim_result = simulator.run_simulation(
                                scenario=scenario,
                                basic_df=scenario_basic_df,  # 시나리오별로 다른 basic_df 사용
                                max_case=max_case,
                                num_tier=num_tier,
                                verbose=verbose_option,
                                skip_updates=False  # calculate_modified_coefficients에서 시나리오별 업데이트 수행
                            )
                            
                            # 디버깅: 결과 데이터 확인
                            if not sim_result.empty:
                                log_info(f"📊 시뮬레이션 결과: {scenario} - {len(sim_result)}개 행")
                                # PCF 열들 확인
                                pcf_columns = [col for col in sim_result.columns if col.startswith('PCF_case')]
                                if pcf_columns:
                                    log_info(f"📈 {scenario} 시나리오 PCF 결과:")
                                    for col in pcf_columns:
                                        total_pcf = sim_result[col].sum()
                                        if 'PCF_reference' in sim_result.columns:
                                            reference_total = sim_result['PCF_reference'].sum()
                                            reduction = reference_total - total_pcf
                                            reduction_rate = (reduction / reference_total) * 100
                                            log_info(f"  • {col}: {total_pcf:.3f} kgCO2eq ({reduction_rate:.2f}% 감소)")
                                        else:
                                            log_info(f"  • {col}: {total_pcf:.3f} kgCO2eq")
                                
                                # 시나리오별 차이점 로깅
                                if scenario != 'baseline' and 'PCF_reference' in sim_result.columns:
                                    baseline_pcf = sim_result['PCF_reference'].sum()
                                    if pcf_columns:
                                        case1_pcf = sim_result['PCF_case1'].sum() if 'PCF_case1' in sim_result.columns else 0
                                        case1_reduction = baseline_pcf - case1_pcf
                                        case1_rate = (case1_reduction / baseline_pcf) * 100
                                        log_info(f"📉 {scenario} vs Baseline: Case1 감소율 = {case1_rate:.2f}% ({case1_reduction:.3f} kgCO2eq)")
                            else:
                                log_warning(f"시뮬레이션 결과가 비어있음: {scenario}")
                            
                            # analyze_simulation_results로 결과 분석
                            # baseline 시나리오의 경우 PCF_reference 저장
                            if scenario == 'baseline' and 'PCF_reference' in sim_result.columns:
                                baseline_pcf_reference = sim_result['PCF_reference'].sum()
                                log_info(f"📊 Baseline PCF_reference 저장: {baseline_pcf_reference:.3f} kgCO2eq")
                                result_dict = simulator.analyze_simulation_results(sim_result)
                            else:
                                # 다른 시나리오는 baseline_pcf_reference를 사용
                                result_dict = simulator.analyze_simulation_results(
                                    sim_result,
                                    baseline_reference=baseline_pcf_reference
                                )

                            results[scenario] = result_dict

                            # PCF 값 디버그 정보 저장 (양극재 중심)
                            if not sim_result.empty:
                                pcf_debug_info = {
                                    'scenario': scenario,
                                    'scenario_name': scenario_names[scenario],
                                    'cathode_materials': []
                                }

                                # 양극재 데이터만 추출
                                cathode_data = sim_result[sim_result['자재품목'] == '양극재']

                                for idx, row in cathode_data.iterrows():
                                    material_info = {
                                        '자재명': row.get('자재명', 'N/A'),
                                        '배출계수': row.get('배출계수', 0),
                                        '제품총소요량(kg)': row.get('제품총소요량(kg)', 0),
                                        'PCF_reference': row.get('PCF_reference', 0),
                                        'PCF_case1': row.get('PCF_case1', 0),
                                        'PCF_case2': row.get('PCF_case2', 0),
                                        'PCF_case3': row.get('PCF_case3', 0)
                                    }
                                    pcf_debug_info['cathode_materials'].append(material_info)

                                # 전체 PCF 합계
                                pcf_debug_info['total_pcf_reference'] = sim_result['PCF_reference'].sum() if 'PCF_reference' in sim_result.columns else 0
                                pcf_debug_info['total_pcf_case1'] = sim_result['PCF_case1'].sum() if 'PCF_case1' in sim_result.columns else 0
                                pcf_debug_info['total_pcf_case2'] = sim_result['PCF_case2'].sum() if 'PCF_case2' in sim_result.columns else 0
                                pcf_debug_info['total_pcf_case3'] = sim_result['PCF_case3'].sum() if 'PCF_case3' in sim_result.columns else 0

                                # 세션 상태에 저장
                                st.session_state[f'pcf_debug_{scenario}'] = pcf_debug_info
                                log_info(f"✅ {scenario} 시나리오 PCF 디버그 정보 저장 완료")

                            # 디버그 로그를 session state에 저장 (st.rerun() 후에도 유지)
                            if hasattr(simulator, 'debug_logs') and simulator.debug_logs:
                                st.session_state[f'debug_logs_{scenario}'] = simulator.debug_logs.copy()
                                log_info(f"✅ {scenario} 시나리오 디버그 로그 저장 완료: {len(simulator.debug_logs)}개 항목")

                            # Fallback 경고 저장 (st.rerun() 후에도 유지)
                            if hasattr(simulator, 'fallback_warnings') and simulator.fallback_warnings:
                                st.session_state[f'fallback_warnings_{scenario}'] = simulator.fallback_warnings.copy()
                                log_info(f"⚠️ {scenario} 시나리오 Fallback 경고 저장: {len(simulator.fallback_warnings)}개 항목")

                            st.success(f"✅ {scenario_names[scenario]} 완료!")
                            log_info(f"시나리오 완료: {scenario_names[scenario]} ({scenario})")
                    
                    # 결과 저장
                    st.session_state.simulation_results = results
                    st.session_state.is_loading = False

                    # original_df와 ref 데이터도 session_state에 저장 (최적화 페이지에서 사용)
                    if simulator and hasattr(simulator, 'original_df'):
                        st.session_state['original_df'] = simulator.original_df
                        log_info(f"✅ original_df를 session_state에 저장 ({len(simulator.original_df)} rows)")

                    if simulator and hasattr(simulator, 'ref_formula_df'):
                        st.session_state['ref_formula_df'] = simulator.ref_formula_df
                        log_info(f"✅ ref_formula_df를 session_state에 저장")

                    if simulator and hasattr(simulator, 'ref_proportions_df'):
                        st.session_state['ref_proportions_df'] = simulator.ref_proportions_df
                        log_info(f"✅ ref_proportions_df를 session_state에 저장")

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ 모든 시나리오 시뮬레이션이 완료되었습니다!")
                    st.markdown('</div>', unsafe_allow_html=True)

                    log_info("모든 시나리오 시뮬레이션 완료")
                    
                    # 페이지 새로고침
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.is_loading = False
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error(f"❌ 시뮬레이션 실행 중 오류가 발생했습니다: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    log_error(f"PCF 시뮬레이션 실행 오류: {e}")
                    st.exception(e)
import streamlit as st
import pandas as pd
import os
import sys
from typing import Tuple

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from app_helper import save_simulation_config, load_simulation_config, get_default_simulation_config
from src.logger import log_button_click, log_input_change, log_info, log_error, log_warning
from src.utils.file_operations import FileOperations, FileLoadError, FileSaveError
from src.utils.styles import get_page_styles

def scenario_configuration_page():
    # Apply centralized styles
    st.markdown(get_page_styles('scenario_configuration'), unsafe_allow_html=True)
    
    st.title("PCF 시나리오 설정")
    st.write("시뮬레이션 할 시나리오를 설정합니다. 수정 후 저장버튼을 누르고 페이지를 refresh 해주세요.")

    # 세션 상태 초기화 (기본값: case=3, tier=2)
    if 'max_case' not in st.session_state:
        st.session_state.max_case = 3
    if 'num_tier' not in st.session_state:
        st.session_state.num_tier = 2
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'edited_df' not in st.session_state:
        st.session_state.edited_df = None
    if 'show_config_message' not in st.session_state:
        st.session_state.show_config_message = False
    
    # 설정 파일에서 초기값 로드 (사용자별) - 기본값 3, 2 우선
    user_id = st.session_state.get('user_id', None)
    config = load_simulation_config(user_id=user_id)
    if config:
        # 저장된 설정이 있으면 로드하되, 기본값이 없으면 3, 2 사용
        if 'max_case' in config:
            st.session_state.max_case = config['max_case']
        else:
            st.session_state.max_case = 3
        if 'num_tier' in config:
            st.session_state.num_tier = config['num_tier']
        else:
            st.session_state.num_tier = 2
    else:
        # 설정 파일이 없으면 기본값 3, 2 사용
        st.session_state.max_case = 3
        st.session_state.num_tier = 2
    
    # 사이드바 설정
    with st.sidebar:
        st.header("시뮬레이션 설정")
        
        # 시뮬레이션 경우의 수 설정
        max_case = st.number_input(
            "시뮬레이션 경우의 수",
            min_value=1,
            max_value=10,
            value=st.session_state.max_case,
            help="시뮬레이션할 case의 최대 개수",
            key="max_case_input"
        )
        
        # Tier의 수 설정
        num_tier = st.number_input(
            "Tier의 수",
            min_value=1,
            max_value=5,
            value=st.session_state.num_tier,
            help="사용할 Tier의 최대 개수",
            key="num_tier_input"
        )
        
        # Apply 버튼
        apply_settings = st.button("apply", type="primary")

        if apply_settings:
            log_button_click("apply", "apply_scenario_settings_btn")
            log_info(f"시나리오 설정 적용: max_case={max_case}, num_tier={num_tier}")

            # 항상 세션 상태 업데이트 (이전과 같더라도 명시적 업데이트)
            st.session_state.max_case = max_case
            st.session_state.num_tier = num_tier

            # 시나리오 설정 변경 플래그 설정 - PCF 시뮬레이터 페이지 초기화용
            st.session_state.scenario_settings_changed = True
            log_info("시나리오 설정 변경 플래그 설정됨 - PCF 시뮬레이터 초기화 예정")
            
            # 변경사항 로깅
            if (st.session_state.get('max_case_prev', max_case) != max_case or st.session_state.get('num_tier_prev', num_tier) != num_tier):
                log_info(f"설정 변경됨: Case {max_case}, Tier {num_tier}")
                # 변경 전 값 저장
                st.session_state.max_case_prev = max_case
                st.session_state.num_tier_prev = num_tier
            else:
                log_info("설정 변경사항 없음 - 저장만 수행")
            
            # 데이터프레임 재생성을 위해 플래그 설정 (항상 업데이트 체크)
            st.session_state.df_needs_update = True
            
            # 설정 전역 상태 업데이트 - 다른 페이지에서 참조하기 위함
            st.session_state.config_settings = {
                "max_case": max_case,
                "num_tier": num_tier,
                "updated_at": st.session_state.get('current_time', 'Unknown')
            }
            
            # 설정을 JSON 파일로 저장
            config_data = {
                "max_case": max_case,
                "num_tier": num_tier,
                "description": f"Case: 1~{max_case}, Tier: 1~{num_tier}",
                "last_updated": st.session_state.get('current_time', 'Unknown')
            }
            
            if save_simulation_config(config_data, user_id=user_id):
                # 설정 파일 저장 성공
                if user_id:
                    user_file_path = f"input/{user_id}/sim_config.json"
                    st.success(f"✅ 설정이 적용되고 저장되었습니다! (Case: 1~{max_case}, Tier: 1~{num_tier})")
                    st.info(f"📁 {user_file_path}에 저장됨")
                    log_info(f"시나리오 설정 성공적으로 저장됨: {user_file_path}")
                    
                    # 사용자 CSV 파일 업데이트 시도
                    saved_file_path = os.path.join("data", user_id, "pcf_scenario_saved.csv")
                    if os.path.exists(saved_file_path):
                        try:
                            # 기존 파일 읽기
                            csv_df = pd.read_csv(saved_file_path)
                            
                            # 동적 열 조정
                            updated_df = create_dynamic_columns(csv_df, max_case, num_tier)
                            if updated_df is not None:
                                # 업데이트된 파일 저장
                                updated_df.to_csv(saved_file_path, index=False, encoding='utf-8-sig')
                                log_info(f"CSV 파일 업데이트 성공: {saved_file_path}")
                            else:
                                log_error(f"CSV 파일 업데이트 실패 - 동적 열 생성 오류: {saved_file_path}")
                        except Exception as e:
                            log_error(f"CSV 파일 업데이트 중 오류: {e}")
                else:
                    st.success(f"✅ 설정이 적용되고 저장되었습니다! (Case: 1~{max_case}, Tier: 1~{num_tier})")
                    st.info("📁 input/sim_config.json에 저장됨")
                    log_info("시나리오 설정 성공적으로 저장됨: sim_config.json")
            else:
                st.error("❌ 설정 저장 중 오류가 발생했습니다.")
                log_error("시나리오 설정 저장 실패")
            
            # 설정 파일 상태 메시지 표시 플래그 설정
            st.session_state.show_config_message = True
            
            st.rerun()
        
        # 현재 설정 표시
        st.info(f"**현재 설정:**\n- Case: 1~{st.session_state.max_case}\n- Tier: 1~{st.session_state.num_tier}")
        
        # Apply 버튼을 눌렀을 때만 설정 파일 정보 표시
        if st.session_state.show_config_message:
            config = load_simulation_config(user_id=user_id)
            if config:
                st.success("📁 설정 파일 로드됨")
            else:
                st.warning("⚠️ 설정 파일이 없습니다. 기본값을 사용합니다.")
        
        # 데이터 초기화 기능
        st.markdown("---")
        st.subheader("📋 데이터 관리")
        
        # 현재 데이터 상태 확인
        if user_id:
            data_file_path = os.path.join("data", user_id, "pcf_scenario_saved.csv")
        else:
            data_file_path = os.path.join("data", "pcf_scenario_saved.csv")
        
        if os.path.exists(data_file_path):
            st.success("✅ 사용자 데이터 파일 존재")
            if st.button("🔄 데이터 재초기화", help="현재 데이터를 삭제하고 기본 템플릿으로 재설정"):
                log_button_click("reinitialize", "reinitialize_data_btn")
                try:
                    os.remove(data_file_path)
                    st.success("✅ 기존 데이터가 삭제되었습니다. 페이지를 새로고침하세요.")
                    log_info(f"데이터 파일 삭제됨: {data_file_path}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 데이터 삭제 중 오류: {e}")
                    log_error(f"데이터 파일 삭제 오류: {e}")
        else:
            st.info("📁 사용자 데이터 파일 없음")
    

    
    # 메인 컨텐츠
    st.header("시나리오 데이터 확인")
    
    # 저장된 파일 확인 및 로드 (사용자별 경로)
    if user_id:
        saved_file_path = os.path.join("data", user_id, "pcf_scenario_saved.csv")
    else:
        saved_file_path = os.path.join("data", "pcf_scenario_saved.csv")
    
    # 파일이 없는 경우 초기화 옵션 제공
    if not os.path.exists(saved_file_path):
        st.warning("⚠️ 저장된 파일이 없습니다.")
        
        # 기본 시나리오 템플릿 파일 경로
        template_file_path = os.path.join("data", "pcf_scenario_saved.csv")
        
        if os.path.exists(template_file_path):
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🔄 기본 템플릿으로 초기화", type="primary"):
                    log_button_click("initialize", "initialize_with_template_btn")
                    log_info(f"기본 시나리오 템플릿으로 초기화 시작: {template_file_path}")
                    
                    try:
                        # 기본 시나리오 템플릿 파일 읽기
                        template_df = pd.read_csv(template_file_path)
                        if template_df is not None and len(template_df) > 0:
                            # 사용자별 디렉토리 생성
                            os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)
                            
                            # 현재 설정에 맞게 동적 열 조정
                            current_max_case = st.session_state.get('max_case', 3)
                            current_num_tier = st.session_state.get('num_tier', 2)
                            
                            # 동적 열 생성/조정
                            initialized_df = create_dynamic_columns(template_df, current_max_case, current_num_tier)
                            if initialized_df is not None:
                                # 초기화된 파일 저장
                                initialized_df.to_csv(saved_file_path, index=False, encoding='utf-8-sig')
                                st.success("✅ 기본 시나리오 템플릿으로 초기화되었습니다!")
                                st.info(f"📁 파일 생성됨: {saved_file_path}")
                                st.info(f"⚙️ 설정: Case 1~{current_max_case}, Tier 1~{current_num_tier}")
                                log_info(f"시나리오 템플릿 초기화 성공: {saved_file_path}")
                                st.rerun()
                            else:
                                st.error("❌ 시나리오 데이터 처리 중 오류가 발생했습니다.")
                                log_error("시나리오 템플릿 데이터 create_dynamic_columns 실패")
                        else:
                            st.error("❌ 시나리오 템플릿 파일이 비어있습니다.")
                            log_error("시나리오 템플릿 파일 데이터가 비어있음")
                    except Exception as template_error:
                        st.error(f"❌ 초기화 중 오류가 발생했습니다: {template_error}")
                        log_error(f"시나리오 템플릿 초기화 오류: {template_error}")
            
            with col2:
                st.info("💡 기본 시나리오 템플릿으로 초기화하면 설정된 Tier/Case 값으로 시작할 수 있습니다.")
        else:
            st.error(f"❌ 기본 시나리오 템플릿 파일을 찾을 수 없습니다: {template_file_path}")
            st.info("📁 시나리오 템플릿 파일이 없습니다. 관리자에게 문의하세요.")
        
        df = None
    elif os.path.exists(saved_file_path):
        try:
            # df_needs_update 플래그 확인 또는 df가 세션에 없을 때 로드
            if st.session_state.get('df_needs_update', False) or 'df' not in st.session_state or st.session_state.df is None:
                try:
                    df = pd.read_csv(saved_file_path)
                    if df is not None and len(df) > 0:
                        st.success("✅ 저장된 시나리오 데이터를 불러옵니다.")
                        
                        # 동적 열 생성/제거 (현재 세션 상태의 값 사용)
                        df = create_dynamic_columns(df, st.session_state.max_case, st.session_state.num_tier)
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.df_needs_update = False
                        else:
                            st.error("❌ 동적 열 생성 중 오류가 발생했습니다.")
                            df = None
                    else:
                        st.error("❌ 로드된 데이터가 비어있습니다.")
                        df = None
                except Exception as read_error:
                    st.error(f"❌ 파일 읽기 오류: {read_error}")
                    df = None
            else:
                df = st.session_state.df
                # 세션에서 가져온 df도 None일 수 있으므로 체크
                if df is None or (hasattr(df, 'empty') and df.empty):
                    st.warning("⚠️ 세션에 저장된 데이터가 없습니다.")
                    df = None
            
            # 데이터프레임을 expander 안에 표시
            with st.expander("📊 저장된 시나리오 데이터 확인", expanded=False):
                if df is not None:
                    # 저감활동_적용여부 컬럼이 있는 경우 정렬
                    if '저감활동_적용여부' in df.columns:
                        # 저감활동_적용여부 = 1인 항목들이 위로 오도록 정렬 (내림차순)
                        sorted_df = df.sort_values('저감활동_적용여부', ascending=False).reset_index(drop=True)
                        st.dataframe(sorted_df, use_container_width=True)
                    else:
                        # 컬럼이 없으면 원본 그대로 표시
                        st.dataframe(df, use_container_width=True)
                    
                    # 데이터 통계 정보
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 자재 수", f"{len(df):,}개")
                    with col2:
                        st.metric("📋 특성 수", f"{len(df.columns)}개")
                    with col3:
                        file_size = os.path.getsize(saved_file_path)
                        st.metric("💾 파일 크기", f"{file_size:,} bytes")
                else:
                    st.error("❌ 데이터프레임을 생성하는 중 오류가 발생했습니다.")
                    
        except Exception as e:
            st.error(f"저장된 파일 로드 중 오류가 발생했습니다: {e}")
            df = None
    
    if df is not None and len(df) > 0:
        # 데이터 편집
        st.markdown("---")
        st.subheader("시나리오 데이터 편집")
        st.markdown("<small>(수정 후 save → refresh 해주세요)</small>", unsafe_allow_html=True)
        
        # 항상 현재 세션 상태의 max_case와 num_tier 사용
        edited_df = edit_scenario_data(df, st.session_state.max_case, st.session_state.num_tier)
        if edited_df is not None:
            st.session_state.edited_df = edited_df
        else:
            st.error("❌ 데이터 편집 중 오류가 발생했습니다.")
            return
        
        # 저장 기능
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("save", type="primary"):
                log_button_click("save", "save_scenario_data_btn")
                log_info("시나리오 데이터 저장 시작")

                # 즉시 저장 처리 (rerun 전에 실행)
                if edited_df is not None:
                    with st.spinner("데이터를 저장하고 있습니다..."):
                        save_scenario_data(edited_df)

                    # 세션 상태 초기화 - 다음 로드에서 최신 CSV 읽도록
                    st.session_state.df = None
                    st.session_state.edited_df = None
                    st.session_state.df_needs_update = True
                    log_info("세션 데이터프레임 초기화 - 다음 로드에서 최신 CSV 읽음")

                    # 저장 완료 후 플래그 설정
                    st.session_state.scenario_settings_changed = True
                    log_info("시나리오 데이터 저장 완료 및 변경 플래그 설정됨 - PCF 시뮬레이터 초기화 예정")

                    # 저장 완료 후 페이지 재로드
                    st.rerun()
                else:
                    st.error("편집된 데이터가 없습니다.")

        with col2:
            if st.button("refresh"):
                log_button_click("refresh", "refresh_scenario_data_btn")
                log_info("시나리오 데이터 새로고침")
                st.session_state.df = None
                st.session_state.edited_df = None
                st.rerun()

def create_dynamic_columns(df: pd.DataFrame, max_case: int, num_tier: int) -> pd.DataFrame:
    """
    설정된 max_case와 num_tier에 따라 동적으로 열을 생성하거나 제거합니다.
    """
    try:
        if df is None:
            log_error("create_dynamic_columns: df is None")
            return None
        
        if hasattr(df, 'empty') and df.empty:
            log_error("create_dynamic_columns: df is empty")
            return None
        
        if len(df) == 0:
            log_error("create_dynamic_columns: df has no rows")
            return None
        
        # 기존 Tier_RE_case 열들 찾기
        existing_tier_cols = [col for col in df.columns if 'Tier' in col and 'RE_case' in col]
        
        # 필요한 열들 생성
        required_cols = []
        for tier in range(1, num_tier + 1):
            for case in range(1, max_case + 1):
                col_name = f'Tier{tier}_RE_case{case}'
                required_cols.append(col_name)
        
        # 새로운 데이터프레임 생성
        new_df = df.copy()
        
        # 필수 컬럼이 없는 경우 추가 (템플릿 파일 대응)
        if '저감활동_적용여부' not in new_df.columns:
            new_df['저감활동_적용여부'] = 0  # 기본값 0 (False)
            log_info("저감활동_적용여부 컬럼 추가됨")
        
        # 기존 열들 제거 (Tier_RE_case 관련)
        for col in existing_tier_cols:
            if col in new_df.columns:
                new_df = new_df.drop(columns=[col])
        
        # 필요한 열들 추가 (기존 값 유지 또는 기본값 설정)
        for col in required_cols:
            if col not in new_df.columns:
                # 기존 데이터에서 해당 열이 있는지 확인
                if col in df.columns:
                    # 기존 값 유지
                    new_df[col] = df[col]
                else:
                    # 기본값 설정
                    new_df[col] = '0%'
        
        # # PCF_case 열들도 동적으로 생성
        # for case in range(1, max_case + 1):
        #     pcf_col = f'PCF_case{case}'
        #     if pcf_col not in new_df.columns:
        #         # 기존 데이터에서 해당 열이 있는지 확인
        #         if pcf_col in df.columns:
        #             # 기존 값 유지
        #             new_df[pcf_col] = df[pcf_col]
        #         else:
        #             # PCF_reference 값으로 초기화
        #             if 'PCF_reference' in new_df.columns:
        #                 new_df[pcf_col] = new_df['PCF_reference']
        #             else:
        #                 new_df[pcf_col] = 0.0
        
        return new_df
    except Exception as e:
        log_error(f"create_dynamic_columns에서 오류 발생: {e}")
        return None

def edit_scenario_data(df: pd.DataFrame, max_case: int, num_tier: int) -> pd.DataFrame:
    """
    데이터프레임을 정렬된 테이블 형태로 표시하여 편집할 수 있도록 합니다.
    """
    if df is None:
        log_error("edit_scenario_data: df is None")
        st.error("❌ 편집할 데이터가 없습니다.")
        return None
    
    if hasattr(df, 'empty') and df.empty:
        log_error("edit_scenario_data: df is empty")
        st.error("❌ 데이터프레임이 비어있습니다.")
        return None
    
    if len(df) == 0:
        log_error("edit_scenario_data: df has no rows")
        st.error("❌ 데이터에 행이 없습니다.")
        return None
    
    # 항상 포함할 컬럼 (실제 존재하는지 확인)
    base_columns_required = ['자재명', '자재품목', '저감활동_적용여부']
    base_columns = [col for col in base_columns_required if col in df.columns]
    
    # 필수 컬럼이 없는 경우 오류 처리
    missing_base_cols = [col for col in base_columns_required if col not in df.columns]
    if missing_base_cols:
        log_error(f"필수 컬럼 누락: {missing_base_cols}")
        st.error(f"❌ 필수 컬럼이 누락되었습니다: {missing_base_cols}")
        return None
    
    # 동적으로 추가할 editable 컬럼 (case -> tier 순으로 정렬)
    dynamic_columns = []
    for case in range(1, max_case + 1):
        for tier in range(1, num_tier + 1):
            dynamic_columns.append(f'Tier{tier}_RE_case{case}')
    # PCF 관련 컬럼도 추가 (편집은 하지 않지만 한 행에 표시)
    if 'PCF_reference' in df.columns:
        dynamic_columns.append('PCF_reference')
    for case in range(1, max_case + 1):
        pcf_col = f'PCF_case{case}'
        if pcf_col in df.columns:
            dynamic_columns.append(pcf_col)
    # 실제 존재하는 컬럼만 사용
    all_columns = base_columns + [col for col in dynamic_columns if col in df.columns]
    editable_df = df[all_columns].copy()

    # 정렬: 먼저 저감활동_적용여부(내림차순 - 체크된 것이 먼저), 그 다음 자재품목(오름차순 - 한글->영어)
    editable_df = editable_df.sort_values(['저감활동_적용여부', '자재품목'], ascending=[False, True]).reset_index(drop=True)
    
    # Apply page-specific table styling
    st.markdown("""
    <style>
        /* Data table specific styling */
        .data-table-container {
            background: var(--background-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            box-shadow: 0 2px 8px var(--shadow-color);
            overflow: hidden;
            margin: 10px 0;
        }
        
        /* CSS 변수로 다크/라이트 모드 호환 색상 정의 */
        :root {
            --background-color: #ffffff;
            --text-color: #1a202c;
            --secondary-text-color: #4a5568;
            --border-color: #e2e8f0;
            --hover-color: #f7fafc;
            --header-bg-start: #4299e1;
            --header-bg-end: #3182ce;
            --header-text-color: #ffffff;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --readonly-text-color: #718096;
            --info-background: #ebf8ff;
        }
        
        /* 다크모드 감지 및 색상 변경 */
        @media (prefers-color-scheme: dark) {
            :root {
                --background-color: #1a202c;
                --text-color: #f7fafc;
                --secondary-text-color: #cbd5e0;
                --border-color: #2d3748;
                --hover-color: #2d3748;
                --header-bg-start: #2b6cb0;
                --header-bg-end: #2c5282;
                --header-text-color: #ffffff;
                --shadow-color: rgba(0, 0, 0, 0.3);
                --readonly-text-color: #a0aec0;
                --info-background: #2a4365;
            }
        }
        
        /* Streamlit 다크모드 클래스 감지 */
        .stApp[data-theme="dark"] {
            --background-color: #1a202c;
            --text-color: #f7fafc;
            --secondary-text-color: #cbd5e0;
            --border-color: #2d3748;
            --hover-color: #2d3748;
            --header-bg-start: #2b6cb0;
            --header-bg-end: #2c5282;
            --header-text-color: #ffffff;
            --shadow-color: rgba(0, 0, 0, 0.3);
            --readonly-text-color: #a0aec0;
            --info-background: #2a4365;
        }
        
        /* 테이블 헤더 */
        .data-table-header {
            background: linear-gradient(135deg, var(--header-bg-start) 0%, var(--header-bg-end) 100%);
            color: var(--header-text-color);
            font-weight: 600;
            font-size: 13px;
            padding: 12px 0;
            text-align: center;
            border-bottom: 2px solid var(--header-bg-end);
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        /* 헤더 셀 */
        .header-cell {
            display: inline-block;
            padding: 0 8px;
            vertical-align: middle;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* 데이터 행 컨테이너 */
        .data-row-container {
            border-bottom: 1px solid var(--border-color);
            transition: background-color 0.2s ease;
            background: var(--background-color);
        }
        
        .data-row-container:hover {
            background-color: var(--hover-color);
        }
        
        .data-row-container:last-child {
            border-bottom: none;
        }
        
        /* Streamlit 컴포넌트 정렬 */
        .stColumns > div {
            padding-left: 4px !important;
            padding-right: 4px !important;
        }
        
        /* 체크박스 중앙 정렬 */
        .stCheckbox {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        /* 번호 입력 필드 스타일 */
        .stNumberInput > div > div > input {
            text-align: center;
            font-size: 12px;
            padding: 4px 8px;
            background-color: var(--background-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }
        
        /* 텍스트 셀 스타일 */
        .text-cell {
            padding: 8px;
            font-size: 13px;
            line-height: 1.4;
            word-break: break-word;
            color: var(--text-color);
        }
        
        .material-name {
            font-weight: 600;
            color: var(--text-color);
        }
        
        .material-category {
            color: var(--secondary-text-color);
        }
        
        .readonly-text {
            color: var(--readonly-text-color);
            font-size: 11px;
        }
        
        /* 컬럼 너비 최적화 */
        .col-material { width: 20%; }
        .col-category { width: 15%; }
        .col-checkbox { width: 8%; }
        .col-tier { width: 7%; }
        .col-pcf { width: 10%; }
        
        /* Streamlit info 박스 스타일 조정 */
        .stAlert > div {
            background-color: var(--info-background) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("#### 자재별 시나리오 입력")
    st.write("각 자재의 Tier별 RE Case 값을 퍼센트로 입력하세요.")
    
    # 컬럼 너비 계산
    num_dynamic_cols = len(all_columns) - 3
    col_widths = [3, 2, 1] + [1] * num_dynamic_cols
    
    # 테이블 컨테이너 시작
    st.markdown('<div class="data-table-container">', unsafe_allow_html=True)
    
    # 헤더 생성
    header_cols = st.columns(col_widths)
    with header_cols[0]:
        st.markdown('<div class="data-table-header header-cell">자재명</div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown('<div class="data-table-header header-cell">자재품목</div>', unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown('<div class="data-table-header header-cell">저감활동</div>', unsafe_allow_html=True)
    
    # 동적 컬럼 헤더
    col_idx = 3
    for col in all_columns[3:]:
        with header_cols[col_idx]:
            if col.startswith('Tier'):
                # Tier1_RE_case1 → T1_C1
                tier_num = col.split('_')[0].replace('Tier', '')
                case_num = col.split('case')[1]
                display_name = f"T{tier_num}_C{case_num}"
                st.markdown(f'<div class="data-table-header header-cell">{display_name}</div>', unsafe_allow_html=True)
            elif col.startswith('PCF'):
                if 'reference' in col:
                    st.markdown('<div class="data-table-header header-cell">PCF기준</div>', unsafe_allow_html=True)
                else:
                    case_num = col.split('case')[1]
                    st.markdown(f'<div class="data-table-header header-cell">PCF{case_num}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="data-table-header header-cell">{col}</div>', unsafe_allow_html=True)
        col_idx += 1
    
    # 데이터 행들
    for idx in range(len(editable_df)):
        row = editable_df.iloc[idx]
        
        # 행 컨테이너 시작
        st.markdown(f'<div class="data-row-container">', unsafe_allow_html=True)
        
        # 행 데이터
        data_cols = st.columns(col_widths)
        
        # 자재명 (굵게 표시)
        with data_cols[0]:
            st.markdown(f'<div class="text-cell material-name">{row["자재명"]}</div>', unsafe_allow_html=True)
        
        # 자재품목
        with data_cols[1]:
            st.markdown(f'<div class="text-cell material-category">{row["자재품목"]}</div>', unsafe_allow_html=True)
        
        # 저감활동 적용여부 체크박스
        with data_cols[2]:
            current_value = row['저감활동_적용여부']
            if pd.isna(current_value) or current_value == '' or current_value == 0:
                default_checked = False
            else:
                try:
                    default_checked = bool(current_value)
                except:
                    default_checked = False
                    
            # 고유한 키를 위해 자재명 사용
            unique_key = f"checkbox_{row['자재명']}_{idx}"
            checkbox_value = st.checkbox(
                f"저감활동 적용 - {row['자재명']}",
                value=default_checked,
                key=unique_key,
                label_visibility="collapsed"
            )
            editable_df.at[idx, '저감활동_적용여부'] = 1 if checkbox_value else 0
        
        # 동적 컬럼들
        col_idx = 3
        for col in all_columns[3:]:
            with data_cols[col_idx]:
                if col.startswith('Tier'):
                    # Tier RE Case 값 편집
                    current_value = str(row[col]).replace('%', '')
                    try:
                        current_value = float(current_value)
                    except:
                        current_value = 0.0
                        
                    # 고유한 키를 위해 자재명과 컬럼명 사용
                    unique_key = f"{col}_{row['자재명']}_{idx}"
                    new_value = st.number_input(
                        f"{col} - {row['자재명']}",
                        min_value=0.0,
                        max_value=100.0,
                        value=current_value,
                        step=1.0,
                        key=unique_key,
                        format="%.1f",
                        label_visibility="collapsed"
                    )
                    editable_df.at[idx, col] = f"{new_value}%"
                else:
                    # PCF 관련 열 (읽기 전용)
                    if isinstance(row[col], (int, float)):
                        display_value = f"{row[col]:.3f}"
                    else:
                        display_value = str(row[col])
                    st.markdown(f'<div class="text-cell readonly-text">{display_value}</div>', unsafe_allow_html=True)
            col_idx += 1
        
        # 행 컨테이너 종료
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 테이블 컨테이너 종료
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 추가 정보
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📊 **총 자재 수:** {len(editable_df)}개")
    with col2:
        tier_cols = len([col for col in all_columns if col.startswith('Tier')])
        st.info(f"🔧 **편집 가능한 Tier 컬럼:** {tier_cols}개")
    with col3:
        st.info(f"⚙️ **설정:** {num_tier} Tier × {max_case} Case")
    
    # 최종 데이터프레임 생성
    final_df = df.copy()
    for col in all_columns:
        if col in final_df.columns:
            final_df[col] = editable_df[col]
    return final_df

def save_scenario_data(df: pd.DataFrame):
    """
    편집된 데이터를 CSV 파일로 저장합니다.
    """
    try:
        # 현재 작업 디렉토리 확인
        import os
        current_dir = os.getcwd()
        
        # 저장 경로 (사용자별 절대 경로 사용)
        if st.session_state.get('user_id'):
            user_id = st.session_state.get('user_id')
            save_path = os.path.join(current_dir, "data", user_id, "pcf_scenario_saved.csv")
        else:
            save_path = os.path.join(current_dir, "data", "pcf_scenario_saved.csv")
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # CSV로 저장
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        # 파일이 실제로 생성되었는지 확인
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            
            # 성공 메시지와 함께 다운로드 버튼 표시
            st.success(f"✅ 데이터가 성공적으로 저장되었습니다!")
            st.info(f"📁 저장 위치: {save_path}")
            st.info(f"📊 파일 크기: {file_size:,} bytes")
            
            # 저장된 파일 다운로드 링크 제공
            with open(save_path, 'r', encoding='utf-8-sig') as f:
                st.download_button(
                    label="📥 저장된 파일 다운로드",
                    data=f.read(),
                    file_name="pcf_scenario_saved.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.error("❌ 파일이 생성되지 않았습니다.")
            
    except Exception as e:
        st.error(f"❌ 저장 중 오류가 발생했습니다: {e}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    scenario_configuration_page()

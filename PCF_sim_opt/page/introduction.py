import streamlit as st
from app_helper import validate_brm_csv, load_default_brm_data, save_brm_data, save_scenario_data, load_default_scenario_data, generate_scenario_from_brm
import pandas as pd
from src.logger import log_button_click, log_file_upload, log_info, log_error

def introduction_page():
    # Apply centralized styles
    from src.utils.styles import get_page_styles
    st.markdown(get_page_styles('introduction'), unsafe_allow_html=True)
    
    # 인트로 섹션
    st.markdown("""
    <div class="intro-section">
        <h2>제품 탄소발자국(PCF) 시뮬레이터</h2>
        <p>본 시뮬레이터는 제품 생산 과정에서 발생하는 탄소배출량을 분석하고 최적화하는 도구입니다. 특히 양극재 생산 과정의 탄소배출량을 계산하고 다양한 시나리오를 통해 저감 방안을 모색합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 주요 기능 섹션
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown('<h3>주요 기능</h3>', unsafe_allow_html=True)
    
    st.markdown('<h4>1. 양극재 세부 설정</h4>', unsafe_allow_html=True)
    st.markdown('''
    - 양극재 소재 구성 비율 설정 (Ni, Co, Mn, Al 등)
    - 생산지 변경을 통한 전력 배출계수 조정
    - 재활용&저탄소메탈 비율 설정을 통한 탄소배출량 감축 시뮬레이션
    ''')
    
    st.markdown('<h4>2. 시나리오 설정</h4>', unsafe_allow_html=True)
    st.markdown('''
    - 저감활동 적용 자재 선택
    - Tier 및 Case별 재생에너지 사용 비율 설정
    - 다양한 저감 시나리오 구성 및 관리
    ''')
    
    st.markdown('<h4>3. PCF 시뮬레이션</h4>', unsafe_allow_html=True)
    st.markdown('''
    - 기본 시나리오: 현재 상태의 탄소배출량 계산
    - 재활용&저탄소메탈 시나리오: 재활용 소재 및 저탄소메탈 적용에 따른 영향 분석
    - 생산지 변경 시나리오: 생산 위치 변경에 따른 전력 배출계수 변화 영향 분석
    - 종합 시나리오: 모든 저감활동을 동시에 적용한 최적 시나리오 분석
    ''')
    
    st.markdown('<h4>4. 결과 분석 및 시각화</h4>', unsafe_allow_html=True)
    st.markdown('''
    - 시나리오별 PCF 감소율 비교
    - 자재별 탄소배출 기여도 분석
    - Tier별, 카테고리별 배출량 분석
    - 상세 데이터 다운로드 및 보고서 생성
    ''')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 작업 순서 섹션
    st.markdown('<div class="workflow-section">', unsafe_allow_html=True)
    st.markdown('<h3>작업 순서</h3>', unsafe_allow_html=True)
    
    st.markdown('''
    1. **데이터 업로드**: BRM 원본 데이터 및 시나리오 테이블 업로드
    2. **양극재 세부 설정**: 소재 구성 및 생산 조건 설정
    3. **시나리오 설정**: 저감활동 적용 자재 및 조건 설정
    4. **시뮬레이션 실행**: 모든 시나리오에 대한 PCF 계산
    5. **결과 분석**: 시나리오별 PCF 감소 효과 비교 및 최적 대안 도출
    ''')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # BRM 원본 업로드 섹션
    st.markdown("---")
    st.markdown("## 📁 BRM 원본 업로드")
    
    # 샘플 다운로드 버튼
    with open('data/pcf_original_table_sample.csv', 'r', encoding='utf-8-sig') as f:
        sample_brm = f.read()
    st.download_button(
        label="📥 BRM 샘플 파일 다운로드",
        data=sample_brm.encode('utf-8-sig'),
        file_name="pcf_original_table_sample.csv",
        mime="text/csv",
        help="BRM 데이터 형식의 샘플 파일을 다운로드합니다."
    )
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "BRM 원본 파일을 업로드하세요 (CSV 또는 Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="자재명, 자재품목, 제품총소요량, 배출계수명, 배출계수, 배출량(kgCO2eq), 지역, 자재코드 열이 포함된 CSV 또는 Excel 파일을 업로드하세요."
    )
    
    # Upload 버튼
    if st.button("upload BRM", type="primary"):
        log_button_click("upload BRM", "upload_brm_btn")
        
        with st.spinner("파일을 처리하는 중..."):
            if uploaded_file is not None:
                # 파일 확장자 확인
                file_ext = uploaded_file.name.split('.')[-1].lower()
                # 파일 업로드 로깅
                log_file_upload(uploaded_file.name, len(uploaded_file.getvalue()), file_ext)
                log_info(f"BRM 파일 업로드 시도: {uploaded_file.name}")

                # 업로드된 파일이 있는 경우
                is_valid, missing_columns, df = validate_brm_csv(uploaded_file)
                
                if is_valid:
                    # 파일을 pcf_original_table_updated.csv로 저장 (사용자별)
                    user_id = st.session_state.get('user_id', None)
                    if save_brm_data(df, user_id=user_id):
                        if user_id:
                            user_file_path = f"data/{user_id}/pcf_original_table_updated.csv"
                            st.success(f"✅ 업로드된 파일 '{uploaded_file.name}'이 '{user_file_path}'로 저장되었습니다!")
                        else:
                            user_file_path = "pcf_original_table_updated.csv"
                            st.success(f"✅ 업로드된 파일 '{uploaded_file.name}'이 '{user_file_path}'로 저장되었습니다!")
                        st.session_state.brm_data = df
                        st.session_state.brm_file_name = user_file_path
                        log_info(f"BRM 파일 성공적으로 저장: {uploaded_file.name} -> {user_file_path}")

                        # 🆕 BRM 데이터에서 자동으로 시나리오 생성
                        with st.spinner("BRM 데이터에서 기본 시나리오를 자동 생성하는 중..."):
                            auto_scenario_df = generate_scenario_from_brm(df)

                            if not auto_scenario_df.empty:
                                # 자동 생성된 시나리오 저장
                                if save_scenario_data(auto_scenario_df, filename="pcf_scenario_auto_generated.csv", user_id=user_id):
                                    st.info("🤖 BRM 데이터를 기반으로 기본 시나리오가 자동 생성되었습니다!")
                                    st.session_state.scenario_data = auto_scenario_df
                                    scenario_file_name = f"data/{user_id}/pcf_scenario_auto_generated.csv" if user_id else "pcf_scenario_auto_generated.csv"
                                    st.session_state.scenario_file_name = scenario_file_name
                                    log_info(f"시나리오 자동 생성 성공: {scenario_file_name}")

                                    # 생성된 시나리오 미리보기
                                    with st.expander("🔍 자동 생성된 시나리오 전체보기", expanded=False):
                                        # 저감활동 적용 자재 (저감활동_적용여부=1) 우선 표시
                                        if '저감활동_적용여부' in auto_scenario_df.columns:
                                            # 저감활동 적용 자재를 위에, 비적용 자재를 아래에 정렬
                                            sorted_scenario_df = auto_scenario_df.sort_values(
                                                by='저감활동_적용여부',
                                                ascending=False
                                            ).reset_index(drop=True)

                                            # 저감활동 적용 자재 통계
                                            applicable_count = len(auto_scenario_df[auto_scenario_df['저감활동_적용여부'] == 1])
                                            total_count = len(auto_scenario_df)

                                            st.info(f"💡 저감활동 적용 자재: {applicable_count}개 / 전체: {total_count}개 (위에서부터 저감활동 적용 자재 표시)")
                                            st.dataframe(sorted_scenario_df, use_container_width=True)
                                        else:
                                            st.dataframe(auto_scenario_df, use_container_width=True)

                                        st.info("💡 시나리오 설정 페이지에서 세부 조정이 가능합니다.")
                                else:
                                    st.warning("⚠️ 시나리오 자동 생성은 성공했으나 저장 중 오류가 발생했습니다.")
                            else:
                                st.warning("⚠️ BRM 데이터에서 시나리오를 자동 생성할 수 없습니다. 수동으로 시나리오 파일을 업로드해주세요.")

                    else:
                        st.error("❌ 파일 저장 중 오류가 발생했습니다.")
                        log_error("BRM 파일 저장 실패")
                else:
                    st.error("❌ CSV 파일 검증에 실패했습니다.")
                    log_error(f"BRM CSV 파일 검증 실패: 누락된 열 - {missing_columns}")
                    if missing_columns:
                        st.warning("⚠️ 다음 필수 열이 누락되었습니다:")
                        for col in missing_columns:
                            st.write(f"• {col}")
            else:
                # 업로드된 파일이 없는 경우 - 기본 파일 사용 (사용자별)
                user_id = st.session_state.get('user_id', None)
                log_info(f"기본 BRM 파일 사용 시도 (user_id: {user_id})")
                default_df = load_default_brm_data(user_id=user_id)

                if not default_df.empty:
                    if user_id:
                        user_file_path = f"data/{user_id}/pcf_original_table_updated.csv (또는 sample 파일)"
                        st.success(f"✅ 사용자별 저장된 파일을 사용합니다! (경로: {user_file_path})")
                    else:
                        user_file_path = "pcf_original_table_updated.csv (또는 sample 파일)"
                        st.success(f"✅ 이전에 저장된 파일 '{user_file_path}'를 사용합니다!")
                    st.session_state.brm_data = default_df
                    st.session_state.brm_file_name = user_file_path
                    log_info(f"기본 BRM 파일 성공적으로 로드: {user_file_path}")

                    # 🆕 기본 파일 사용시에도 시나리오가 없으면 자동 생성
                    if 'scenario_data' not in st.session_state or st.session_state.scenario_data is None or st.session_state.scenario_data.empty:
                        with st.spinner("기본 BRM 데이터에서 시나리오를 자동 생성하는 중..."):
                            auto_scenario_df = generate_scenario_from_brm(default_df)

                            if not auto_scenario_df.empty:
                                if save_scenario_data(auto_scenario_df, filename="pcf_scenario_auto_generated.csv", user_id=user_id):
                                    st.info("🤖 기본 BRM 데이터를 기반으로 시나리오가 자동 생성되었습니다!")
                                    st.session_state.scenario_data = auto_scenario_df
                                    scenario_file_name = f"data/{user_id}/pcf_scenario_auto_generated.csv" if user_id else "pcf_scenario_auto_generated.csv"
                                    st.session_state.scenario_file_name = scenario_file_name
                                    log_info(f"기본 파일 기반 시나리오 자동 생성 성공: {scenario_file_name}")

                                    # 생성된 시나리오 미리보기
                                    with st.expander("🔍 자동 생성된 시나리오 전체보기", expanded=False):
                                        # 저감활동 적용 자재 (저감활동_적용여부=1) 우선 표시
                                        if '저감활동_적용여부' in auto_scenario_df.columns:
                                            # 저감활동 적용 자재를 위에, 비적용 자재를 아래에 정렬
                                            sorted_scenario_df = auto_scenario_df.sort_values(
                                                by='저감활동_적용여부',
                                                ascending=False
                                            ).reset_index(drop=True)

                                            # 저감활동 적용 자재 통계
                                            applicable_count = len(auto_scenario_df[auto_scenario_df['저감활동_적용여부'] == 1])
                                            total_count = len(auto_scenario_df)

                                            st.info(f"💡 저감활동 적용 자재: {applicable_count}개 / 전체: {total_count}개 (위에서부터 저감활동 적용 자재 표시)")
                                            st.dataframe(sorted_scenario_df, use_container_width=True)
                                        else:
                                            st.dataframe(auto_scenario_df, use_container_width=True)

                                        st.info("💡 시나리오 설정 페이지에서 세부 조정이 가능합니다.")

                else:
                    st.error("❌ 저장된 파일을 찾을 수 없습니다. CSV 파일을 업로드해주세요.")
                    st.session_state.brm_data = None
                    st.session_state.brm_file_name = None
                    log_error(f"기본 BRM 파일을 찾을 수 없음 (user_id: {user_id})")
    
    # 전체 데이터프레임 확인 (expander)
    if 'brm_data' in st.session_state and st.session_state.brm_data is not None:
        st.markdown("---")
        st.markdown("### 📊 데이터 확인")
        
        with st.expander("🔍 전체 데이터프레임 보기", expanded=False):
            st.dataframe(st.session_state.brm_data, use_container_width=True)
            
            # 데이터 통계 정보
            st.markdown("#### 📈 데이터 통계")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 총 행 수", f"{len(st.session_state.brm_data):,}개")
            with col2:
                st.metric("📋 총 열 수", f"{len(st.session_state.brm_data.columns)}개")
            with col3:
                st.metric("📁 파일명", st.session_state.brm_file_name)
            with col4:
                # 메모리 사용량 추정 (대략적)
                memory_usage = st.session_state.brm_data.memory_usage(deep=True).sum()
                st.metric("💾 메모리 사용량", f"{memory_usage / 1024:.1f} KB")
    
    # 시나리오 테이블 업로드 섹션 (Expander로 처리)
    st.markdown("---")

    with st.expander("📋 시나리오 테이블 업로드 (선택사항)", expanded=False):
        st.info("💡 시나리오 테이블은 선택사항입니다. BRM 데이터 업로드 시 자동으로 기본 시나리오가 생성됩니다. 커스텀 시나리오를 사용하려면 여기에 업로드하세요.")

        # 샘플 다운로드 버튼
        with open('data/pcf_scenario_sample.csv', 'r', encoding='utf-8-sig') as f:
            sample_scenario = f.read()
        st.download_button(
            label="📥 시나리오 샘플 파일 다운로드",
            data=sample_scenario.encode('utf-8-sig'),
            file_name="pcf_scenario_sample.csv",
            mime="text/csv",
            help="시나리오 데이터 형식의 샘플 파일을 다운로드합니다."
        )

        # 파일 업로더
        scenario_uploaded_file = st.file_uploader(
            "시나리오 테이블 파일을 업로드하세요 (CSV 또는 Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="시나리오 설정이 포함된 CSV 또는 Excel 파일을 업로드하세요."
        )

        # Upload 버튼
        if st.button("upload scenario", type="primary"):
            log_button_click("upload scenario", "upload_scenario_btn")

            with st.spinner("시나리오 파일을 처리하는 중..."):
                if scenario_uploaded_file is not None:
                    # 파일 확장자 확인
                    file_ext = scenario_uploaded_file.name.split('.')[-1].lower()
                    # 파일 업로드 로깅
                    log_file_upload(scenario_uploaded_file.name, len(scenario_uploaded_file.getvalue()), file_ext)
                    log_info(f"시나리오 파일 업로드 시도: {scenario_uploaded_file.name}")

                    # 업로드된 파일이 있는 경우
                    try:
                        # 파일 확장자에 따라 적절한 방법으로 읽기
                        if file_ext == 'csv':
                            scenario_df = pd.read_csv(scenario_uploaded_file)
                        else:  # xlsx, xls
                            scenario_df = pd.read_excel(scenario_uploaded_file)
                        # 파일을 pcf_scenario_saved.csv로 저장 (사용자별)
                        user_id = st.session_state.get('user_id', None)
                        if save_scenario_data(scenario_df, user_id=user_id):
                            if user_id:
                                user_file_path = f"data/{user_id}/pcf_scenario_saved.csv"
                                st.success(f"✅ 업로드된 시나리오 파일 '{scenario_uploaded_file.name}'이 '{user_file_path}'로 저장되었습니다!")
                            else:
                                user_file_path = "pcf_scenario_saved.csv"
                                st.success(f"✅ 업로드된 시나리오 파일 '{scenario_uploaded_file.name}'이 '{user_file_path}'로 저장되었습니다!")
                            st.session_state.scenario_data = scenario_df
                            st.session_state.scenario_file_name = user_file_path
                            log_info(f"시나리오 파일 성공적으로 저장: {scenario_uploaded_file.name} -> {user_file_path}")
                        else:
                            st.error("❌ 시나리오 파일 저장 중 오류가 발생했습니다.")
                            log_error("시나리오 파일 저장 실패")
                    except Exception as e:
                        st.error(f"❌ 시나리오 파일 처리 중 오류가 발생했습니다: {e}")
                        log_error(f"시나리오 파일 처리 오류: {e}")
                else:
                    # 업로드된 파일이 없는 경우 - 기본 파일 사용 (사용자별)
                    user_id = st.session_state.get('user_id', None)
                    log_info(f"기본 시나리오 파일 사용 시도 (user_id: {user_id})")
                    default_scenario_df = load_default_scenario_data(user_id=user_id)

                    if not default_scenario_df.empty:
                        if user_id:
                            user_file_path = f"data/{user_id}/pcf_scenario_saved.csv (또는 sample 파일)"
                            st.success(f"✅ 사용자별 시나리오 파일을 사용합니다! (경로: {user_file_path})")
                        else:
                            user_file_path = "pcf_scenario_saved.csv (또는 sample 파일)"
                            st.success(f"✅ 이전에 저장된 시나리오 파일 '{user_file_path}'를 사용합니다!")
                        st.session_state.scenario_data = default_scenario_df
                        st.session_state.scenario_file_name = user_file_path
                        log_info(f"기본 시나리오 파일 성공적으로 로드: {user_file_path}")
                    else:
                        st.warning("⚠️ 저장된 시나리오 파일을 찾을 수 없습니다. CSV 파일을 업로드해주세요.")
                        st.session_state.scenario_data = None
                        st.session_state.scenario_file_name = None
                        log_error(f"기본 시나리오 파일을 찾을 수 없음 (user_id: {user_id})")
    
    # 시나리오 데이터프레임 확인 (expander)
    if 'scenario_data' in st.session_state and st.session_state.scenario_data is not None:
        st.markdown("---")
        st.markdown("### 📊 시나리오 데이터 확인")
        
        with st.expander("🔍 시나리오 데이터프레임 보기", expanded=False):
            st.dataframe(st.session_state.scenario_data, use_container_width=True)
            
            # 데이터 통계 정보
            st.markdown("#### 📈 시나리오 데이터 통계")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 총 행 수", f"{len(st.session_state.scenario_data):,}개")
            with col2:
                st.metric("📋 총 열 수", f"{len(st.session_state.scenario_data.columns)}개")
            with col3:
                st.metric("📁 파일명", st.session_state.scenario_file_name)
            with col4:
                # 메모리 사용량 추정 (대략적)
                memory_usage = st.session_state.scenario_data.memory_usage(deep=True).sum()
                st.metric("💾 메모리 사용량", f"{memory_usage / 1024:.1f} KB")
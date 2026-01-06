import streamlit as st
import os
import shutil
import re
import logging
import json
from pathlib import Path
from page.cathode_configuration import cathode_configuration_page
from page.scenario_configuration import scenario_configuration_page
from page.pcf_simulation import pcf_simulation_page
from page.introduction import introduction_page
from page.optimization import optimization_page
from page.cost_variable_settings import cost_variable_settings_page
from page.re100_premium_test import re100_premium_test_page
from src.logger import get_logger, log_button_click, log_menu_change, log_page_view, log_info
from datetime import datetime
from src.utils.file_operations import FileOperations

# Streamlit 관련 로깅 레벨 설정 (파일 감시 에러 숨기기)
logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.watcher.event_based_path_watcher").setLevel(logging.CRITICAL)
logging.getLogger("watchdog").setLevel(logging.ERROR)
logging.getLogger("watchdog.observers").setLevel(logging.ERROR)

# 페이지 설정
st.set_page_config(
    page_title="PCF Simulator",  # 브라우저 탭에 표시될 제목
    page_icon="🔋",  # 파비콘 (이모지 또는 이미지 경로 사용 가능)
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### PCF Simulator\n"  # About 섹션의 내용 (마크다운 지원)
                "This page is a tool for analyzing PCF."
    }
)



# pages 숨기기 함수
def hide_pages():
    st.markdown("""
        <style>
            div[data-testid="stSidebarNav"] {display: none !important;}
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="collapsedControl"] {display: none}
        </style>
    """, unsafe_allow_html=True)



# CSS를 사용하여 Pages 섹션 숨기기
st.markdown("""
    <style>
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #1E88E5;
            --primary-hover: #1565C0;
            --text-color: #FFFFFF;
            --text-secondary: #B0B0B0;
            --bg-color: #1E1E1E;
            --bg-secondary: #2D2D2D;
            --bg-tertiary: #3D3D3D;
            --border-color: #555555;
            --shadow-color: rgba(0, 0, 0, 0.3);
            --info-bg: #1A3A52;
            --content-bg: #2A2A2A;
        }
    }
    
    @media (prefers-color-scheme: light) {
        :root {
            --primary-color: #1E88E5;
            --primary-hover: #1565C0;
            --text-color: #212121;
            --text-secondary: #666666;
            --bg-color: #FFFFFF;
            --bg-secondary: #F5F5F5;
            --bg-tertiary: #E0E0E0;
            --border-color: #DDDDDD;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --info-bg: #E3F2FD;
            --content-bg: #F8F9FA;
        }
    }
    
    /* Force light mode colors for Streamlit's light theme */
    [data-theme="light"] {
        --primary-color: #1E88E5;
        --primary-hover: #1565C0;
        --text-color: #212121;
        --text-secondary: #666666;
        --bg-color: #FFFFFF;
        --bg-secondary: #F5F5F5;
        --bg-tertiary: #E0E0E0;
        --border-color: #DDDDDD;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --info-bg: #E3F2FD;
        --content-bg: #F8F9FA;
    }
    
    /* Force dark mode colors for Streamlit's dark theme */
    [data-theme="dark"] {
        --primary-color: #1E88E5;
        --primary-hover: #1565C0;
        --text-color: #FFFFFF;
        --text-secondary: #B0B0B0;
        --bg-color: #1E1E1E;
        --bg-secondary: #2D2D2D;
        --bg-tertiary: #3D3D3D;
        --border-color: #555555;
        --shadow-color: rgba(0, 0, 0, 0.3);
        --info-bg: #1A3A52;
        --content-bg: #2A2A2A;
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    section[data-testid="stSidebar"] > div:nth-child(2) {
        display: none;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: var(--primary-color);
    }
    .sub-header {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-bottom: 0.2rem;
    }
    .nav-container {
        padding: 15px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--border-color);
    }
    .content-div {
        background-color: var(--content-bg);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px var(--shadow-color);
        margin-bottom: 1.5rem;
    }
    .info-div {
        background-color: var(--info-bg);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        margin: 0.8rem 0;
    }
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: var(--primary-hover);
        transform: translateY(-2px);
        box-shadow: 0 2px 5px var(--shadow-color);
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    .sidebar-button {
        width: 100%;
        margin: 5px 0;
        text-align: left !important;
        padding: 0.75rem 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_user_directories(user_id):
    """사용자별 디렉토리 구조 초기화"""
    # 사용자별 디렉토리 생성 (data, input은 사용자별 폴더 방식 사용)
    user_dirs = {
        'data': Path("data/cached") / user_id,
        'input': Path("input/cached") / user_id,
        'stable_var': Path("stable_var/cached") / user_id
    }
    
    # 각 디렉토리 생성
    for dir_path in user_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    # cathode_preprocessor와 cathode_simulator에서 사용하는 stable_var/user_id 폴더도 생성
    stable_var_user_dir = Path("stable_var") / user_id
    stable_var_user_dir.mkdir(parents=True, exist_ok=True)
    
    # data/user_id 폴더도 생성 (FileOperations에서 사용)
    data_user_dir = Path("data") / user_id
    data_user_dir.mkdir(parents=True, exist_ok=True)
    
    # original table을 사용자 폴더로 기본 복사
    original_table_files = [
        ("data/pcf_original_table_sample.csv", "pcf_original_table_sample.csv"),
        ("data/pcf_scenario_sample.csv", "pcf_scenario_sample.csv")
    ]
    
    import shutil
    for source_path, file_name in original_table_files:
        source_file = Path(source_path)
        dest_file = data_user_dir / file_name
        
        if source_file.exists() and not dest_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"사용자 {user_id}에게 {file_name} 기본 파일 복사 완료")
    
    # stable_var 기본 파일들을 사용자 폴더로 복사 (파일이 존재하지 않을 경우에만)
    stable_var_base = Path("stable_var")
    stable_var_files = [
        "cathode_coef_table.json",
        "electricity_coef_by_country.json",
        "recycle_material_impact.json",
        "cathode_national_code.json",
        "CAM.json",
        "pCAM.json",
        "cathode_tier1_input.json",
        "cathode_tier2_input.json"
    ]
    
    for file_name in stable_var_files:
        source_file = stable_var_base / file_name
        dest_file = stable_var_user_dir / file_name

        if source_file.exists() and not dest_file.exists():
            shutil.copy2(source_file, dest_file)

    # optimization_costs 파일들을 사용자 폴더로 복사 (파일이 존재하지 않을 경우에만)
    try:
        from src.utils.optimization_costs_manager import initialize_user_optimization_costs
        initialize_user_optimization_costs(user_id, force=False)
    except Exception as e:
        print(f"⚠️ optimization_costs 파일 복사 중 오류: {e}")

    # input 기본 파일들을 사용자 폴더로 복사 (파일이 존재하지 않을 경우에만)
    input_base = Path("input")
    input_user_dir = Path("input") / user_id
    input_user_dir.mkdir(parents=True, exist_ok=True)
    
    input_files = [
        "recycle_material_ratio.json",
        "low_carb_metal.json",
        "cathode_ratio.json",
        "cathode_site.json",
        "sim_config.json"
    ]
    
    for file_name in input_files:
        source_file = input_base / file_name
        dest_file = input_user_dir / file_name
        
        if source_file.exists() and not dest_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"사용자 {user_id}에게 input/{file_name} 기본 파일 복사 완료")
        
    return user_dirs

def save_user_state(user_id):
    """사용자 상태 저장"""
    print(f"사용자 {user_id}의 상태 저장 중...")
    
    # 저장할 디렉토리 경로
    user_dirs = {
        'data': Path("data/cached") / user_id,
        'input': Path("input/cached") / user_id,
        'stable_var': Path("stable_var/cached") / user_id
    }
    
    # 데이터 디렉토리 파일 저장
    data_files = list(Path("data").glob(f"*_{user_id}.*"))
    for file_path in data_files:
        dest_path = user_dirs['data'] / file_path.name
        shutil.copy2(file_path, dest_path)
    
    # 입력 디렉토리 파일 저장
    input_files = list(Path("input").glob(f"*_{user_id}.*"))
    for file_path in input_files:
        dest_path = user_dirs['input'] / file_path.name
        shutil.copy2(file_path, dest_path)
    
    # stable_var 디렉토리 파일 저장
    stable_var_files = list(Path("stable_var").glob(f"*_{user_id}.*"))
    for file_path in stable_var_files:
        dest_path = user_dirs['stable_var'] / file_path.name
        shutil.copy2(file_path, dest_path)
    
    print(f"사용자 {user_id}의 상태 저장 완료")

def load_user_state(user_id):
    """사용자 상태 로드"""
    print(f"사용자 {user_id}의 상태 로드 중...")
    
    # 캐시 디렉토리 경로
    user_dirs = {
        'data': Path("data/cached") / user_id,
        'input': Path("input/cached") / user_id,
        'stable_var': Path("stable_var/cached") / user_id
    }
    
    # 각 디렉토리에서 파일 로드
    for dir_type, dir_path in user_dirs.items():
        if dir_path.exists():
            # 디렉토리 내 모든 파일 복사
            for file_path in dir_path.glob("*"):
                dest_path = Path(dir_type) / file_path.name
                shutil.copy2(file_path, dest_path)
    
    print(f"사용자 {user_id}의 상태 로드 완료")

def load_allowed_users():
    """허용된 사용자 목록을 JSON 파일에서 로드"""
    users_file = Path("data/users.json")
    try:
        if users_file.exists():
            with open(users_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('allowed_users', [])
        else:
            # 파일이 없으면 경고 메시지 출력
            print("⚠️ data/users.json 파일이 없습니다. 모든 사용자 접근이 차단됩니다.")
            return []
    except Exception as e:
        print(f"⚠️ 사용자 목록 로드 중 오류: {e}")
        return []

def is_authorized_user(user_id):
    """사용자가 허용된 사용자인지 확인"""
    allowed_users = load_allowed_users()
    is_authorized = user_id in allowed_users
    print(f"🔐 인증 확인 - 사용자: {user_id}, 허용된 사용자 목록: {allowed_users}, 인증 결과: {is_authorized}")
    return is_authorized

def validate_user_id(user_id):
    """사용자 ID의 유효성 검사"""
    # 영문, 숫자, 언더스코어, 점만 허용, 2-20자 제한
    pattern = re.compile(r'^[a-zA-Z0-9_.]{2,20}$')
    return bool(pattern.match(user_id))

# 전역 로거 초기화 (로그인 전에도 오류 로깅을 위해 필요)
global logger
logger = get_logger()

def main():
    
    # 사용자 로그인 상태 관리
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        
    # 세션 ID가 없는 경우 초기화
    if 'session_id' not in st.session_state:
        # 세션 초기화
        session_id = logger.initialize_session()
    
    # 로그인 화면 표시
    if not st.session_state.logged_in:
        st.title("PCF Simulator - 로그인")
        st.markdown("시뮬레이터를 사용하려면 사용자 ID를 입력하세요.")
        
        with st.form("login_form"):
            user_id = st.text_input("사용자 ID (영문, 숫자, 언더스코어, 점 사용 가능)")
            submit_button = st.form_submit_button("로그인")

            if submit_button:
                if not validate_user_id(user_id):
                    st.error("유효하지 않은 사용자 ID입니다. 영문, 숫자, 언더스코어, 점만 사용해주세요. (2-20자)")
                elif not is_authorized_user(user_id):
                    st.error("🚫 등록되지 않은 사용자입니다. 관리자에게 문의하세요.")
                else:
                    st.session_state.user_id = user_id
                    st.session_state.logged_in = True

                    # 사용자 디렉토리 초기화
                    initialize_user_directories(user_id)

                    # 사용자 상태 로드
                    load_user_state(user_id)

                    # 로그인 성공 메시지
                    st.success(f"✅ {user_id}님, 환영합니다!")

                    # 페이지 새로고침
                    st.rerun()
        
        return
    
    # 로그인 성공 후 메인 앱 실행
    # 사용자 ID가 있는 경우 로그에 기록
    if 'user_id' in st.session_state and st.session_state.user_id:
        try:
            log_info(f"사용자 ID: {st.session_state.user_id} 로그인")            
        except Exception as e:
            print(f"로그에 사용자 ID 기록 실패: {e}")
    
    # 페이지 숨기기 적용
    hide_pages()
    
    # 사이드바에 세션 정보 표시
    st.sidebar.title("메뉴")
    
    # 사용자 ID 표시
    st.sidebar.markdown(f"**사용자 ID: {st.session_state.user_id}**")
    st.sidebar.markdown("---")
    
    # 세션 정보 표시
    session_info = logger.get_session_info()
    if session_info:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 세션 정보")
        st.sidebar.markdown(f"**세션 ID:** {session_info['session_id'][:8]}...")
        st.sidebar.markdown(f"**시작 시간:** {session_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.sidebar.markdown(f"**경과 시간:** {str(session_info['session_duration']).split('.')[0]}")
        st.sidebar.markdown(f"**총 액션:** {session_info['total_actions']}")
        st.sidebar.markdown("---")
    
    # 이전 메뉴 상태 저장
    previous_menu = st.session_state.get('menu', None)
    
    # 버튼 메뉴 생성 및 로깅
    if st.sidebar.button("개요 및 데이터 업로드", key="intro_btn", use_container_width=True):
        log_button_click("개요 및 데이터 업로드", "intro_btn")
        st.session_state.menu = "개요"
    if st.sidebar.button("양극재 세부 설정", key="cathode_btn", use_container_width=True):
        log_button_click("양극재 세부 설정", "cathode_btn")
        st.session_state.menu = "양극재 세부 설정"
    if st.sidebar.button("시나리오 설정", key="scenario_btn", use_container_width=True):
        log_button_click("시나리오 설정", "scenario_btn")
        st.session_state.menu = "시나리오 설정"
    if st.sidebar.button("PCF 시뮬레이터", key="pcf_sim_btn", use_container_width=True):
        log_button_click("PCF 시뮬레이터", "pcf_sim_btn")
        st.session_state.menu = "PCF 시뮬레이터"
    if st.sidebar.button("💰 비용 변수 설정", key="cost_settings_btn", use_container_width=True):
        log_button_click("비용 변수 설정", "cost_settings_btn")
        st.session_state.menu = "비용 변수 설정"
    if st.sidebar.button("⚡ RE100 프리미엄 계산", key="re100_btn", use_container_width=True):
        log_button_click("RE100 프리미엄 계산", "re100_btn")
        st.session_state.menu = "RE100 프리미엄 계산"
    if st.sidebar.button("최적화", key="optimization_btn", use_container_width=True):
        log_button_click("최적화", "optimization_btn")
        st.session_state.menu = "최적화"
    
    # 초기 메뉴 상태 설정
    if 'menu' not in st.session_state:
        st.session_state.menu = "개요"
    
    # 메뉴 변경 로깅
    if previous_menu and previous_menu != st.session_state.menu:
        log_menu_change(previous_menu, st.session_state.menu)
    
    st.markdown("""
            <style>
                /* Remove blank space at top and bottom */ 
                .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    }
            </style>
            """, unsafe_allow_html=True)

    st.markdown(
        """
        <h1 style="text-align: center; margin-top: 0;">
            PCF Simulator
        </h1>
        """,
        unsafe_allow_html=True
        )
    st.markdown('<p class="sub-header" style="text-align: center;">Product Carbon Footprint Simulator</p>', unsafe_allow_html=True)
    st.markdown('<hr style="height:2px;border:none;background-color:var(--border-color);margin:2rem 0;">', unsafe_allow_html=True)
    
    # 페이지 전환 로직 및 로깅
    if st.session_state.menu == "양극재 세부 설정":
        log_page_view("양극재 세부 설정")
        cathode_configuration_page()
    elif st.session_state.menu == "시나리오 설정":
        log_page_view("시나리오 설정")
        scenario_configuration_page()
    elif st.session_state.menu == "PCF 시뮬레이터":
        log_page_view("PCF 시뮬레이터")
        pcf_simulation_page()
    elif st.session_state.menu == "비용 변수 설정":
        log_page_view("비용 변수 설정")
        cost_variable_settings_page()
    elif st.session_state.menu == "RE100 프리미엄 계산":
        log_page_view("RE100 프리미엄 계산")
        re100_premium_test_page()
    elif st.session_state.menu == "최적화":
        log_page_view("최적화")
        optimization_page()
    else:
        log_page_view("개요 및 데이터 업로드")
        introduction_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 에러 로깅
        try:
            logger.log_error(f"애플리케이션 실행 중 오류: {e}")
        except Exception as log_err:
            print(f"로깅 실패: {log_err}")
        raise e
    finally:
        # 세션 종료 로깅 및 상태 저장
        try:
            # 로그인 상태인 경우 사용자 상태 저장
            if 'logged_in' in st.session_state and st.session_state.logged_in:
                save_user_state(st.session_state.user_id)
            
            try:
                logger.log_session_end()
            except Exception as log_err:
                print(f"세션 종료 로깅 실패: {log_err}")
        except Exception as e:
            print(f"세션 종료 처리 중 오류: {e}")
            pass  # 로깅 실패 시 무시

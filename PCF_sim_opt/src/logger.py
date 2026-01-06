import uuid
import logging
import os
from datetime import datetime
import streamlit as st
from typing import Any, Dict, Optional
import json

class SessionLogger:
    def __init__(self):
        self.session_id = None
        self.logger = None
        self.session_start_time = None
        self.user_actions = []
        
    def initialize_session(self) -> str:
        """새로운 세션을 초기화하고 UUID를 생성합니다."""
        if 'session_id' not in st.session_state:
            # UUID 생성
            session_id = str(uuid.uuid4())
            st.session_state.session_id = session_id
            
            # 세션 시작 시간 기록
            session_start_time = datetime.now()
            st.session_state.session_start_time = session_start_time
            
            # 로거 설정
            self._setup_logger(session_id, session_start_time)
            
            # 세션 정보 로그
            self.log_session_start(session_id, session_start_time)
            
            return session_id
        else:
            # 세션이 이미 존재하면 로거가 없는지 확인하고 필요하면 다시 설정
            if 'logger' not in st.session_state:
                self._setup_logger(st.session_state.session_id, st.session_state.session_start_time)
            
            return st.session_state.session_id
    
    def _setup_logger(self, session_id: str, start_time: datetime):
        """로거를 설정합니다."""
        # log 폴더가 없으면 생성
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 로그 파일명 생성 (날짜시간_UUID.log 형식)
        date_str = start_time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{date_str}_{session_id}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        
        # 로거 설정
        logger = logging.getLogger(f"session_{session_id}")
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 핸들러 추가
        logger.addHandler(file_handler)
        
        # 세션 상태에 저장
        st.session_state.logger = logger
        st.session_state.log_filepath = log_filepath
    
    def log_session_start(self, session_id: str, start_time: datetime):
        """세션 시작을 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.info("=" * 80)
        logger.info(f"새로운 세션 시작")
        logger.info(f"세션 ID: {session_id}")
        logger.info(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"사용자 IP: {self._get_client_ip()}")
        logger.info(f"사용자 에이전트: {self._get_user_agent()}")
        # 사용자 ID 기록 (있는 경우)
        if 'user_id' in st.session_state:
            logger.info(f"사용자 ID: {st.session_state.user_id}")
        logger.info("=" * 80)
    
    def log_button_click(self, button_name: str, button_key: str):
        """버튼 클릭을 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.info(f"버튼 클릭: {button_name} (키: {button_key})")
        self._add_user_action("button_click", {
            "button_name": button_name,
            "button_key": button_key
        })
    
    def log_menu_change(self, old_menu: str, new_menu: str):
        """메뉴 변경을 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.info(f"메뉴 변경: {old_menu} -> {new_menu}")
        self._add_user_action("menu_change", {
            "old_menu": old_menu,
            "new_menu": new_menu
        })
    
    def log_input_change(self, input_name: str, input_key: str, old_value: Any, new_value: Any):
        """입력값 변경을 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.info(f"입력 변경: {input_name} (키: {input_key})")
        logger.info(f"  이전 값: {old_value}")
        logger.info(f"  새로운 값: {new_value}")
        self._add_user_action("input_change", {
            "input_name": input_name,
            "input_key": input_key,
            "old_value": str(old_value),
            "new_value": str(new_value)
        })
    
    def log_file_upload(self, file_name: str, file_size: int, file_type: str):
        """파일 업로드를 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.info(f"파일 업로드: {file_name}")
        logger.info(f"  파일 크기: {file_size} bytes")
        logger.info(f"  파일 타입: {file_type}")
        self._add_user_action("file_upload", {
            "file_name": file_name,
            "file_size": file_size,
            "file_type": file_type
        })
    
    def log_error(self, error_message: str, error_type: str = "ERROR"):
        """에러를 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.error(f"에러 발생: {error_message}")
        self._add_user_action("error", {
            "error_message": error_message,
            "error_type": error_type
        })
    
    def log_info(self, message: str):
        """일반 정보를 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.info(message)
    
    def log_warning(self, message: str):
        """경고를 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.warning(message)
    
    def log_page_view(self, page_name: str):
        """페이지 조회를 로그에 기록합니다."""
        logger = st.session_state.logger
        logger.info(f"페이지 조회: {page_name}")
        self._add_user_action("page_view", {
            "page_name": page_name
        })
    
    def log_session_end(self):
        """세션 종료를 로그에 기록합니다."""
        if 'logger' in st.session_state:
            logger = st.session_state.logger
            end_time = datetime.now()
            start_time = st.session_state.session_start_time
            duration = end_time - start_time
            
            logger.info("=" * 80)
            logger.info(f"세션 종료")
            logger.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"세션 지속 시간: {duration}")
            logger.info(f"총 사용자 액션 수: {len(self.user_actions)}")
            logger.info("=" * 80)
            
            # 사용자 액션 요약 저장
            self._save_user_actions_summary()
    
    def _add_user_action(self, action_type: str, action_data: Dict[str, Any]):
        """사용자 액션을 내부 리스트에 추가합니다."""
        action = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "action_data": action_data
        }
        self.user_actions.append(action)
        
        # 세션 상태에도 저장
        if 'user_actions' not in st.session_state:
            st.session_state.user_actions = []
        st.session_state.user_actions.append(action)
    
    def _save_user_actions_summary(self):
        """사용자 액션 요약을 JSON 파일로 저장합니다."""
        if 'log_filepath' in st.session_state:
            log_filepath = st.session_state.log_filepath
            summary_filepath = log_filepath.replace('.log', '_summary.json')
            
            summary = {
                "session_id": st.session_state.session_id,
                "start_time": st.session_state.session_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_actions": len(self.user_actions),
                "user_actions": self.user_actions
            }
            
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def _get_client_ip(self) -> str:
        """클라이언트 IP를 가져옵니다."""
        try:
            # Streamlit에서 클라이언트 정보 가져오기
            return getattr(st, '_client_ip', 'Unknown')
        except:
            return 'Unknown'
    
    def _get_user_agent(self) -> str:
        """사용자 에이전트를 가져옵니다."""
        try:
            # Streamlit에서 사용자 에이전트 정보 가져오기
            return getattr(st, '_user_agent', 'Unknown')
        except:
            return 'Unknown'
    
    def get_session_info(self) -> Dict[str, Any]:
        """현재 세션 정보를 반환합니다."""
        if 'session_id' in st.session_state:
            return {
                "session_id": st.session_state.session_id,
                "start_time": st.session_state.session_start_time,
                "current_time": datetime.now(),
                "session_duration": datetime.now() - st.session_state.session_start_time,
                "total_actions": len(self.user_actions)
            }
        return {}

# 전역 로거 인스턴스
session_logger = SessionLogger()

def get_logger() -> SessionLogger:
    """전역 로거 인스턴스를 반환합니다."""
    return session_logger

def log_button_click(button_name: str, button_key: str):
    """버튼 클릭을 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_button_click(button_name, button_key)
    except Exception as e:
        print(f"[BUTTON] {button_name} ({button_key})")

def log_input_change(input_name: str, input_key: str, old_value: Any, new_value: Any):
    """입력값 변경을 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_input_change(input_name, input_key, old_value, new_value)
    except Exception as e:
        print(f"[INPUT CHANGE] {input_name} ({input_key}): {old_value} -> {new_value}")

def log_file_upload(file_name: str, file_size: int, file_type: str):
    """파일 업로드를 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_file_upload(file_name, file_size, file_type)
    except Exception as e:
        print(f"[FILE UPLOAD] {file_name} ({file_size} bytes, {file_type})")

def log_error(error_message: str, error_type: str = "ERROR"):
    """에러를 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_error(error_message, error_type)
    except Exception as e:
        print(f"[ERROR] {error_type}: {error_message}")

def log_info(message: str):
    """일반 정보를 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_info(message)
    except Exception as e:
        # Fallback 로깅 (세션 로거를 사용할 수 없는 경우)
        print(f"[INFO] {message}")

def log_warning(message: str):
    """경고를 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_warning(message)
    except Exception as e:
        print(f"[WARNING] {message}")

def log_page_view(page_name: str):
    """페이지 조회를 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_page_view(page_name)
    except Exception as e:
        print(f"[PAGE VIEW] {page_name}")

def log_menu_change(old_menu: str, new_menu: str):
    """메뉴 변경을 로그에 기록하는 편의 함수입니다."""
    try:
        session_logger.log_menu_change(old_menu, new_menu)
    except Exception as e:
        print(f"[MENU CHANGE] {old_menu} -> {new_menu}")
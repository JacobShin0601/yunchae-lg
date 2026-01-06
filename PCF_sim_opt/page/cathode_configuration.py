import streamlit as st
import json
import os
import pandas as pd
from typing import Dict, Any, Tuple
import sys

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from src.cathode_simulator import CathodeSimulator
from src.helper import CathodeHelper
from app_helper import create_basic_scenarios_visualizations
from src.utils.logging_migration import log_button_click, log_input_change, log_info, log_error, log_warning
from src.utils.file_operations import FileOperations, FileLoadError, FileSaveError
from src.utils.styles import get_page_styles


def validate_numeric_inputs(data: Dict[str, Any], field_name: str) -> Tuple[bool, str]:
    """숫자 입력값을 검증합니다."""
    for key, value in data.items():
        if not isinstance(value, (int, float)) or value < 0:
            return False, f"{field_name}의 {key}: 유효하지 않은 숫자값 ({value})"
    return True, ""


def validate_ratio_sum(data: Dict[str, Any], field_name: str, expected_sum: float = 1.0) -> Tuple[bool, str]:
    """비율 합계를 검증합니다."""
    total = sum(data.values())
    if abs(total - expected_sum) > 0.01:  # 1% 오차 허용
        return False, f"{field_name}의 합계가 {expected_sum}이 아닙니다 (현재: {total:.3f})"
    return True, ""


def cathode_configuration_page():
    """양극재 설정 페이지"""
    
    # Apply centralized styles
    st.markdown(get_page_styles('cathode_configuration'), unsafe_allow_html=True)
    
    # Add page-specific styles if needed
    st.markdown("""
    <style>
    .scrollable-config {
        max-height: 80vh;
        overflow-y: auto;
        padding-right: 10px;
        scrollbar-width: thin;
        scrollbar-color: var(--primary-color) var(--bg-secondary);
    }
    .scrollable-config::-webkit-scrollbar {
        width: 8px;
    }
    .scrollable-config::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    .scrollable-config::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    .scrollable-config::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
        opacity: 0.8;
    }
    .loading-section {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 40px 20px;
        margin: 10px 0;
        text-align: center;
    }
    .loading-text {
        color: var(--primary-color);
        font-size: 1.1rem;
        margin-top: 20px;
    }
    .spinner {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 4px solid var(--border-color);
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="config-section">
        <h2 style="color: var(--text-color);">양극재 세부 설정</h2>
        <p style="color: var(--text-secondary);">양극재의 조성비, 생산지, 재활용재 사용비율 등을 설정할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    
    cathode_ratio_path = os.path.join(project_root, "input", "cathode_ratio.json")
    cathode_site_path = os.path.join(project_root, "input", "cathode_site.json")
    cathode_coef_table_path = os.path.join(project_root, "stable_var", "cathode_coef_table.json")
    recycle_ratio_path = os.path.join(project_root, "input", "recycle_material_ratio.json")
    recycle_impact_path = os.path.join(project_root, "stable_var", "recycle_material_impact.json")
    low_carb_metal_path = os.path.join(project_root, "input", "low_carb_metal.json")
    electricity_coef_path = os.path.join(project_root, "stable_var", "electricity_coef_by_country.json")
    
    # 초기 데이터 로드 (사용자별)
    user_id = st.session_state.get('user_id', None)
    
    # 사용자별 파일 경로 확인 및 초기화 처리
    def check_and_initialize_files():
        """사용자별 설정 파일들이 없으면 기본 템플릿으로 초기화"""
        if not user_id:
            return True  # user_id가 없으면 기본 파일 사용
        
        user_cathode_ratio_path = os.path.join(project_root, "input", user_id, "cathode_ratio.json")
        user_cathode_site_path = os.path.join(project_root, "input", user_id, "cathode_site.json")
        user_recycle_ratio_path = os.path.join(project_root, "input", user_id, "recycle_material_ratio.json")
        user_low_carb_metal_path = os.path.join(project_root, "input", user_id, "low_carb_metal.json")
        
        user_files_exist = all([
            os.path.exists(user_cathode_ratio_path),
            os.path.exists(user_cathode_site_path),
            os.path.exists(user_recycle_ratio_path),
            os.path.exists(user_low_carb_metal_path)
        ])
        
        if not user_files_exist:
            # 기본 템플릿 파일들 경로
            default_files = {
                user_cathode_ratio_path: cathode_ratio_path,
                user_cathode_site_path: cathode_site_path,
                user_recycle_ratio_path: recycle_ratio_path,
                user_low_carb_metal_path: low_carb_metal_path
            }
            
            # 사용자 디렉토리 생성
            os.makedirs(os.path.dirname(user_cathode_ratio_path), exist_ok=True)
            
            # 기본 파일들 복사
            for user_file, default_file in default_files.items():
                if os.path.exists(default_file) and not os.path.exists(user_file):
                    try:
                        import shutil
                        shutil.copy2(default_file, user_file)
                        log_info(f"기본 설정 파일 복사됨: {default_file} -> {user_file}")
                    except Exception as e:
                        log_error(f"파일 복사 실패: {default_file} -> {user_file}, 오류: {e}")
                        return False
            
            return True
        return True
    
    # 파일 초기화 처리
    files_initialized = check_and_initialize_files()
    
    try:
        cathode_ratio_data = FileOperations.load_json(cathode_ratio_path, user_id=user_id)
    except FileLoadError as e:
        if not files_initialized:
            st.error(f"설정 파일 초기화 중 오류가 발생했습니다: {e}")
        cathode_ratio_data = {}
    
    # Primary logic을 사용하여 양극재 생산지 기본값 설정
    try:
        # CathodeSimulator 인스턴스 생성하여 Primary logic 사용 (사용자별)
        temp_simulator = CathodeSimulator(verbose=False, user_id=user_id)
        default_cathode_site_config = temp_simulator.get_default_cathode_site_config()
        
        # 기존 파일과 Primary logic 결과 병합
        existing_cathode_site_data = FileOperations.load_json(cathode_site_path, default={}, user_id=user_id)
        if existing_cathode_site_data:
            # 기존 파일이 있으면 after 값만 유지하고 before는 Primary logic 사용
            cathode_site_data = {
                "CAM": {
                    "before": default_cathode_site_config["CAM"]["before"],  # Primary logic
                    "after": existing_cathode_site_data.get("CAM", {}).get("after", default_cathode_site_config["CAM"]["after"])
                },
                "pCAM": {
                    "before": default_cathode_site_config["pCAM"]["before"],  # Primary logic
                    "after": existing_cathode_site_data.get("pCAM", {}).get("after", default_cathode_site_config["pCAM"]["after"])
                }
            }
        else:
            # 기존 파일이 없으면 Primary logic 결과 전체 사용
            cathode_site_data = default_cathode_site_config
            
    except Exception as e:
        # Primary logic 실패 시 기존 방식으로 폴백
        st.warning(f"Primary logic 실행 중 오류가 발생하여 기존 설정을 사용합니다: {e}")
        cathode_site_data = FileOperations.load_json(cathode_site_path, default={}, user_id=user_id)
        if not cathode_site_data:
            cathode_site_data = {
                "CAM": {"before": "미분류", "after": "한국"},
                "pCAM": {"before": "미분류", "after": "한국"}
            }
    
    cathode_coef_table_data = FileOperations.load_json(cathode_coef_table_path, default={}, user_id=user_id)
    recycle_ratio_data = FileOperations.load_json(recycle_ratio_path, default={}, user_id=user_id)
    recycle_impact_data = FileOperations.load_json(recycle_impact_path, default={}, user_id=user_id)
    low_carb_metal_data = FileOperations.load_json(low_carb_metal_path, default={}, user_id=user_id)
    electricity_coef_data = FileOperations.load_json(electricity_coef_path, default={}, user_id=user_id)
    
    # 세션 상태 초기화
    if 'config_updated' not in st.session_state:
        st.session_state.config_updated = False
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
    
    # 2분할 레이아웃: 왼쪽 설정, 오른쪽 결과
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="config-title">⚙️ 설정</div>', unsafe_allow_html=True)
        
        # 스크롤 가능한 설정 영역 시작
        st.markdown('<div class="scrollable-config">', unsafe_allow_html=True)
        
        # 1. 양극재 조성비
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("1. 양극재 조성비")
        st.info("각 원소의 비율을 입력하세요 (합계: 100%)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        cathode_ratio_updated = {}
        
        # 2열로 구성
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            for element in ["Ni", "Co"]:
                default_value = cathode_ratio_data.get(element, 0.0) * 100  # 소수점을 퍼센트로 변환
                value = st.number_input(
                    f"{element} 비율 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_value),
                    step=0.0001,
                    format="%.4f",
                    key=f"cathode_ratio_{element}"
                )
                cathode_ratio_updated[element] = value / 100  # 퍼센트를 소수점으로 변환하여 저장

        with col1_2:
            for element in ["Mn", "Al"]:
                default_value = cathode_ratio_data.get(element, 0.0) * 100  # 소수점을 퍼센트로 변환
                value = st.number_input(
                    f"{element} 비율 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_value),
                    step=0.0001,
                    format="%.4f",
                    key=f"cathode_ratio_{element}"
                )
                cathode_ratio_updated[element] = value / 100  # 퍼센트를 소수점으로 변환하여 저장
        
        # 2. 양극재 생산지
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("2. 양극재 생산지")
        st.info("CAM과 pCAM의 생산지를 선택하세요")
        
        # Primary logic 정보 표시
        try:
            temp_simulator = CathodeSimulator(verbose=False, user_id=user_id)
            detected_site = temp_simulator._get_primary_cathode_site()
            if detected_site == "미분류":
                st.markdown(f"💡 **자동 감지된 Before 사이트**: {detected_site} (RoW, GLO 등 매핑되지 않는 지역)")
                st.markdown("🔍 **미분류 선택 시**: 원본 BRM/시나리오 데이터의 배출계수를 그대로 사용합니다.")
            else:
                st.markdown(f"💡 **자동 감지된 Before 사이트**: {detected_site} (PCF 테이블 기반)")
        except:
            st.markdown("💡 **자동 감지**: PCF 테이블에서 Before 사이트를 자동으로 설정합니다")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        countries = ["한국", "중국", "일본", "폴란드", "미분류"]
        
        # 2열로 구성
        col2_1, col2_2 = st.columns(2)
        
        # 안전한 index 찾기 헬퍼 함수
        def safe_get_country_index(country_name, default_country="한국"):
            try:
                return countries.index(country_name)
            except ValueError:
                # 리스트에 없는 경우 기본값 사용
                try:
                    return countries.index(default_country)
                except ValueError:
                    return 0  # 최종적으로 첫 번째 항목 사용

        with col2_1:
            cam_before_country = cathode_site_data.get("CAM", {}).get("before", "미분류")
            cam_before = st.selectbox(
                "CAM (before)",
                countries,
                index=safe_get_country_index(cam_before_country, "미분류"),
                key="cam_before"
            )

            cam_after_country = cathode_site_data.get("CAM", {}).get("after", "한국")
            cam_after = st.selectbox(
                "CAM (after)",
                countries,
                index=safe_get_country_index(cam_after_country, "한국"),
                key="cam_after"
            )

        with col2_2:
            pcam_before_country = cathode_site_data.get("pCAM", {}).get("before", "미분류")
            pcam_before = st.selectbox(
                "pCAM (before)",
                countries,
                index=safe_get_country_index(pcam_before_country, "미분류"),
                key="pcam_before"
            )

            pcam_after_country = cathode_site_data.get("pCAM", {}).get("after", "한국")
            pcam_after = st.selectbox(
                "pCAM (after)",
                countries,
                index=safe_get_country_index(pcam_after_country, "한국"),
                key="pcam_after"
            )
        
        cathode_site_updated = {
            "CAM": {"before": cam_before, "after": cam_after},
            "pCAM": {"before": pcam_before, "after": pcam_after}
        }

        # 3. 원재료 배출계수
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("3. 원재료 배출계수")
        st.info("각 원재료의 배출계수를 입력하세요")
        st.markdown('</div>', unsafe_allow_html=True)

        # 원재료 배출계수 데이터 초기화
        cathode_coef_updated = {"원재료": {}}

        # 원재료 리스트
        raw_materials = ["NiSO4", "CoSO4", "MnSO4", "Al(OH3)", "NaOH", "LiOH.H2O"]

        # 3열로 구성
        col_coef_1, col_coef_2, col_coef_3 = st.columns(3)

        for idx, material in enumerate(raw_materials):
            # 기본값 로드
            default_value = cathode_coef_table_data.get("원재료", {}).get(material, {}).get("배출계수", 0.0)

            # 3개씩 나누어서 배치
            if idx % 3 == 0:
                col = col_coef_1
            elif idx % 3 == 1:
                col = col_coef_2
            else:
                col = col_coef_3

            with col:
                value = st.number_input(
                    f"{material} 배출계수",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_value),
                    step=0.01,
                    format="%.2f",
                    key=f"raw_material_coef_{material}"
                )
                # 원래 구조 유지하면서 배출계수만 업데이트
                if material not in cathode_coef_updated["원재료"]:
                    cathode_coef_updated["원재료"][material] = cathode_coef_table_data.get("원재료", {}).get(material, {}).copy()
                cathode_coef_updated["원재료"][material]["배출계수"] = value

        # Energy 데이터도 유지 (수정하지 않고 그대로 복사)
        for energy_type in ["Energy(Tier-1)", "Energy(Tier-2)"]:
            if energy_type in cathode_coef_table_data:
                cathode_coef_updated[energy_type] = cathode_coef_table_data[energy_type]

        # 4. 재활용재 설정
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("4. 재활용재 설정")
        st.info("재활용재의 사용 비율과 환경영향 계수를 입력하세요")
        st.markdown('</div>', unsafe_allow_html=True)

        # 재활용재 데이터 초기화
        recycle_ratio_updated = {}
        recycle_impact_updated = {"신재": 1.0, "재활용재": {}}

        # 2열로 구성 (사용비율 / 환경영향계수)
        col3_1, col3_2 = st.columns(2)

        with col3_1:
            st.write("**재활용재 사용비율 (%)**")
            for element in ["Ni", "Co", "Li"]:
                default_value = recycle_ratio_data.get(element, 0.0) * 100
                value = st.number_input(
                    f"{element} 재활용 비율 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_value),
                    step=0.1,
                    format="%.1f",
                    key=f"recycle_ratio_{element}"
                )
                recycle_ratio_updated[element] = value / 100

        with col3_2:
            st.write("**재활용재 환경영향 계수 (%)**")
            for element in ["Ni", "Co", "Li"]:
                default_value = recycle_impact_data.get("재활용재", {}).get(element, 0.1) * 100
                value = st.number_input(
                    f"{element} 환경영향 계수 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_value),
                    step=1.0,
                    format="%.1f",
                    key=f"recycle_impact_{element}"
                )
                recycle_impact_updated["재활용재"][element] = value / 100
        
        # 5. 저탄소메탈 설정
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("5. 저탄소메탈 설정")
        st.info("저탄소메탈의 사용 비중과 배출계수를 입력하세요")
        st.markdown('</div>', unsafe_allow_html=True)

        # 저탄소메탈 데이터 초기화
        low_carb_metal_updated = {
            "비중": {},
            "배출계수": {}
        }

        # 2열로 구성 (비중 / 배출계수)
        col4_1, col4_2 = st.columns(2)

        with col4_1:
            st.write("**저탄소메탈 사용비중 (%)**")
            for element in ["Ni", "Co", "Li"]:
                default_value = low_carb_metal_data.get("비중", {}).get(element, 5.0)
                value = st.number_input(
                    f"{element} 저탄소메탈 비중 (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_value),
                    step=0.1,
                    format="%.1f",
                    key=f"low_carb_ratio_{element}"
                )
                low_carb_metal_updated["비중"][element] = value

        with col4_2:
            st.write("**저탄소메탈 배출계수**")
            # 원소별 신재 원재료 매핑
            element_to_material = {
                "Ni": "NiSO4",
                "Co": "CoSO4",
                "Li": "LiOH.H2O"
            }

            for element in ["Ni", "Co", "Li"]:
                default_value = low_carb_metal_data.get("배출계수", {}).get(element, 2.0)

                # 신재 배출계수 가져오기
                material_name = element_to_material.get(element, "")
                virgin_coef = cathode_coef_table_data.get("원재료", {}).get(material_name, {}).get("배출계수", 0.0)

                value = st.number_input(
                    f"{element} 저탄소메탈 배출계수 (신재 {virgin_coef:.2f})",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(default_value),
                    step=0.1,
                    format="%.3f",
                    key=f"low_carb_emission_{element}"
                )
                low_carb_metal_updated["배출계수"][element] = value

        # 3원 분할 비율 검증 및 표시
        st.markdown("---")
        st.write("**💡 3원 분할 비율 확인 (신재 + 재활용재 + 저탄소메탈 = 100%)**")

        for element in ["Ni", "Co", "Li"]:
            recycling_pct = recycle_ratio_updated.get(element, 0) * 100
            low_carb_pct = low_carb_metal_updated["비중"].get(element, 0)
            virgin_pct = 100 - recycling_pct - low_carb_pct

            if virgin_pct < 0:
                st.error(f"⚠️ {element}: 재활용({recycling_pct:.1f}%) + 저탄소메탈({low_carb_pct:.1f}%) = {recycling_pct + low_carb_pct:.1f}% > 100%")
            else:
                st.success(f"✅ {element}: 신재({virgin_pct:.1f}%) + 재활용({recycling_pct:.1f}%) + 저탄소메탈({low_carb_pct:.1f}%) = {virgin_pct + recycling_pct + low_carb_pct:.1f}%")

        # 6. 전력배출계수
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("6. 전력배출계수")
        st.info("각 국가의 전력배출계수를 입력하세요")
        st.markdown('</div>', unsafe_allow_html=True)
        
        electricity_coef_updated = {}
        
        # 2열로 구성
        col5_1, col5_2 = st.columns(2)
        
        with col5_1:
            for country in ["한국", "중국"]:
                default_value = electricity_coef_data.get(country, 0.5)
                value = st.number_input(
                    f"{country} 전력배출계수",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(default_value),
                    step=0.01,
                    format="%.6f",
                    key=f"electricity_coef_{country}"
                )
                electricity_coef_updated[country] = value
        
        with col5_2:
            for country in ["일본", "폴란드"]:
                default_value = electricity_coef_data.get(country, 0.5)
                value = st.number_input(
                    f"{country} 전력배출계수",
                    min_value=0.0,
                    max_value=2.0,
                    value=float(default_value),
                    step=0.01,
                    format="%.6f",
                    key=f"electricity_coef_{country}"
                )
                electricity_coef_updated[country] = value

        # 미분류 입력 필드 추가 (새로운 행)
        col5_3, col5_4 = st.columns(2)

        with col5_3:
            # 4개국 평균 계산
            avg_coef = sum([
                electricity_coef_updated.get("한국", electricity_coef_data.get("한국", 0.6374)),
                electricity_coef_updated.get("중국", electricity_coef_data.get("중국", 0.8825)),
                electricity_coef_updated.get("일본", electricity_coef_data.get("일본", 0.6679)),
                electricity_coef_updated.get("폴란드", electricity_coef_data.get("폴란드", 0.9490))
            ]) / 4

            default_value = electricity_coef_data.get("미분류", avg_coef)
            value = st.number_input(
                "미분류 전력배출계수 (기본값은 평균)",
                min_value=0.0,
                max_value=2.0,
                value=float(default_value),
                step=0.01,
                format="%.6f",
                key="electricity_coef_미분류",
                help="생산지가 명확하지 않은 경우 사용되는 전력배출계수입니다. 기본값은 4개국 평균값입니다."
            )
            electricity_coef_updated["미분류"] = value

        with col5_4:
            # 평균 정보 표시
            st.info(f"💡 현재 4개국 평균: {avg_coef:.6f} kgCO2eq/kWh")

        # Apply 및 Refresh 버튼
        st.markdown("---")

        # 2열 레이아웃: Apply (50%) + Refresh (50%)
        col_apply, col_refresh = st.columns(2)

        with col_apply:
            apply_clicked = st.button("⚙️ Apply 설정", type="primary", use_container_width=True, help="현재 설정을 저장하고 시뮬레이션을 실행합니다")

        with col_refresh:
            refresh_clicked = st.button("🔄 Refresh 페이지", use_container_width=True, help="페이지와 결과 패널을 초기 상태로 리셋합니다")

        # Refresh 버튼 처리
        if refresh_clicked:
            log_button_click("refresh_page", "refresh_cathode_page_btn")
            log_info("양극재 설정 페이지 새로고침 - 결과 패널 초기화")

            # 오른쪽 결과 패널 상태 초기화
            st.session_state.config_updated = False
            st.session_state.is_loading = False

            st.rerun()

        # Apply 버튼 처리
        if apply_clicked:
            log_button_click("apply", "apply_config_btn")
            log_info("양극재 설정 적용 시작")
            
            # 로딩 상태 시작
            st.session_state.is_loading = True
            st.session_state.config_updated = False
            
            # 입력값 검증
            validation_errors = []

            # 1. 양극재 조성비 검증
            is_valid, error_msg = validate_ratio_sum(cathode_ratio_updated, "양극재 조성비")
            if not is_valid:
                validation_errors.append(error_msg)

            # 2. 원재료 배출계수 검증
            raw_material_coefs = {k: v["배출계수"] for k, v in cathode_coef_updated["원재료"].items()}
            is_valid, error_msg = validate_numeric_inputs(raw_material_coefs, "원재료 배출계수")
            if not is_valid:
                validation_errors.append(error_msg)

            # 3. 재활용재 사용비율 검증
            is_valid, error_msg = validate_numeric_inputs(recycle_ratio_updated, "재활용재 사용비율")
            if not is_valid:
                validation_errors.append(error_msg)

            # 4. 재활용재 환경영향 검증
            is_valid, error_msg = validate_numeric_inputs(recycle_impact_updated["재활용재"], "재활용재 환경영향")
            if not is_valid:
                validation_errors.append(error_msg)

            # 5. 저탄소메탈 비중 검증
            is_valid, error_msg = validate_numeric_inputs(low_carb_metal_updated["비중"], "저탄소메탈 비중")
            if not is_valid:
                validation_errors.append(error_msg)

            # 6. 저탄소메탈 배출계수 검증
            is_valid, error_msg = validate_numeric_inputs(low_carb_metal_updated["배출계수"], "저탄소메탈 배출계수")
            if not is_valid:
                validation_errors.append(error_msg)

            # 7. 3원 분할 비율 합계 검증 (재활용 + 저탄소메탈 <= 100%)
            for element in ["Ni", "Co", "Li"]:
                recycling_pct = recycle_ratio_updated.get(element, 0) * 100
                low_carb_pct = low_carb_metal_updated["비중"].get(element, 0)
                total_pct = recycling_pct + low_carb_pct
                if total_pct > 100:
                    validation_errors.append(f"{element}: 재활용({recycling_pct:.1f}%) + 저탄소메탈({low_carb_pct:.1f}%) = {total_pct:.1f}% > 100%")

            # 8. 전력배출계수 검증
            is_valid, error_msg = validate_numeric_inputs(electricity_coef_updated, "전력배출계수")
            if not is_valid:
                validation_errors.append(error_msg)
            
            if validation_errors:
                # 에러 메시지들을 하나로 합치기
                error_message = "**입력값 검증 오류:**\n\n" + "\n".join([f"• {error}" for error in validation_errors])

                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error(error_message)
                st.markdown('</div>', unsafe_allow_html=True)
                log_error(f"양극재 설정 검증 실패: {validation_errors}")
                st.session_state.is_loading = False
            else:
                # 파일 저장
                save_success = True
                
                try:
                    FileOperations.save_json(cathode_ratio_path, cathode_ratio_updated, user_id=user_id)
                except FileSaveError:
                    save_success = False
                
                try:
                    FileOperations.save_json(cathode_site_path, cathode_site_updated, user_id=user_id)
                except FileSaveError:
                    save_success = False

                try:
                    FileOperations.save_json(cathode_coef_table_path, cathode_coef_updated, user_id=user_id)
                except FileSaveError:
                    save_success = False

                try:
                    FileOperations.save_json(recycle_ratio_path, recycle_ratio_updated, user_id=user_id)
                except FileSaveError:
                    save_success = False
                
                try:
                    FileOperations.save_json(recycle_impact_path, recycle_impact_updated, user_id=user_id)
                except FileSaveError:
                    save_success = False
                
                try:
                    FileOperations.save_json(low_carb_metal_path, low_carb_metal_updated, user_id=user_id)
                except FileSaveError:
                    save_success = False
                
                try:
                    FileOperations.save_json(electricity_coef_path, electricity_coef_updated, user_id=user_id)
                except FileSaveError:
                    save_success = False
                
                if save_success:
                    # 시나리오 CSV 자동 생성 (BRM 원본 데이터가 있는 경우)
                    try:
                        from app_helper import load_default_brm_data, generate_scenario_from_brm, save_scenario_data

                        # BRM 원본 데이터 로드
                        brm_df = load_default_brm_data(user_id=user_id)

                        if not brm_df.empty:
                            # 시나리오 자동 생성
                            scenario_df = generate_scenario_from_brm(brm_df)

                            if not scenario_df.empty:
                                # 사용자별로 저장
                                save_scenario_data(scenario_df, user_id=user_id)
                                log_info(f"시나리오 데이터 자동 생성 완료 (user_id: {user_id})")
                            else:
                                log_warning("시나리오 자동 생성 실패: 빈 데이터프레임")
                        else:
                            log_warning("BRM 원본 데이터 없음 - 시나리오 자동 생성 건너뜀")
                    except Exception as scenario_error:
                        log_error(f"시나리오 자동 생성 중 오류: {scenario_error}")
                        # 시나리오 생성 실패는 치명적이지 않으므로 계속 진행

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ 설정이 성공적으로 저장되었습니다!")
                    st.info("💡 **양극재 설정이 변경되었습니다.** PCF 시뮬레이터 페이지에서 새로운 설정으로 시뮬레이션을 다시 실행해주세요.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    log_info("양극재 설정 성공적으로 저장됨")
                    st.session_state.config_updated = True
                    st.session_state.is_loading = False

                    # PCF 시뮬레이터 초기화 플래그 설정 (시나리오 설정 변경과 동일한 로직)
                    st.session_state.scenario_settings_changed = True
                    log_info("양극재 설정 변경으로 인한 PCF 시뮬레이터 초기화 플래그 설정")
                else:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error("❌ 파일 저장 중 오류가 발생했습니다.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    log_error("양극재 설정 파일 저장 실패")
                    st.session_state.is_loading = False
        
        # 스크롤 가능한 설정 영역 끝
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 오른쪽 결과 영역
    with col2:
        st.markdown('<div class="result-title">📊 결과</div>', unsafe_allow_html=True)
        
        # # 리프레시 버튼 추가
        # col_refresh, col_space = st.columns([1, 4])
        # with col_refresh:
        #     if st.button("🔄 refresh", help="최신 설정으로 결과를 다시 로드합니다", key="refresh_results_btn"):
        #         log_button_click("refresh", "refresh_results_btn")
        #         st.session_state.config_updated = True
        #         st.rerun()
        
        # 로딩 상태 처리
        if st.session_state.is_loading:
            st.markdown("""
            <div class="loading-section">
                <h3 style="color: var(--text-color);">⏳ 처리 중...</h3>
                <p style="color: var(--text-secondary);">설정을 저장하고 시뮬레이션을 실행하고 있습니다.</p>
                <div style="margin: 20px 0;">
                    <div class="spinner"></div>
                </div>
                <p class="loading-text">잠시만 기다려주세요...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # CSS animation already included in the page-specific styles above
            
            # 로딩 상태를 False로 설정하여 다음 실행에서 결과 표시
            st.session_state.is_loading = False
        
        elif st.session_state.config_updated:
            st.markdown("""
            <div class="result-section">
                <h3 style="color: var(--text-color);">📊 시뮬레이션 결과</h3>
                <p style="color: var(--text-secondary);">설정이 업데이트되었습니다. 아래에서 시뮬레이션 결과를 확인하세요.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 시뮬레이션 실행
            try:
                # CathodeSimulator 초기화 (사용자별 설정 파일 사용)
                simulator = CathodeSimulator(verbose=False, user_id=user_id)
                helper = CathodeHelper(simulator, verbose=False)

                # 🔍 디버그 로그 Expander 추가
                with st.expander("🔍 디버그 로그 (계산 과정 확인)", expanded=False):
                    st.info("💡 이 섹션은 계산 과정의 중간 결과를 확인할 수 있는 디버그 정보입니다.")

                    # 1. 설정 파일 로드 확인
                    st.write("### 1. 로드된 설정 파일 내용")
                    col_debug1, col_debug2 = st.columns(2)

                    with col_debug1:
                        st.write("**양극재 조성비:**")
                        st.json(cathode_ratio_updated)

                        st.write("**재활용재 사용비율:**")
                        st.json(recycle_ratio_updated)

                        st.write("**재활용재 환경영향:**")
                        st.json(recycle_impact_updated)

                    with col_debug2:
                        st.write("**양극재 생산지:**")
                        st.json(cathode_site_updated)

                        st.write("**저탄소메탈 설정:**")
                        st.json(low_carb_metal_updated)

                        st.write("**전력배출계수:**")
                        st.json(electricity_coef_updated)

                    # 2. Simulator 초기화 정보
                    st.write("---")
                    st.write("### 2. Simulator 초기화 정보")
                    st.write(f"- **User ID**: {user_id if user_id else 'None (기본 설정 사용)'}")
                    st.write(f"- **Verbose 모드**: False")

                    # 3. Primary Logic 감지 결과
                    st.write("---")
                    st.write("### 3. Primary Logic 감지 결과")
                    try:
                        detected_site = simulator._get_primary_cathode_site()
                        st.write(f"- **자동 감지된 사이트**: {detected_site}")

                        # 감지 로직 상세 - PCF 테이블에서 직접 읽기
                        st.write("- **감지 로직 상세:**")

                        # PCF 테이블 파일 경로 확인
                        current_dir = os.path.dirname(os.path.abspath(__file__))

                        if user_id:
                            pcf_table_path = os.path.join(current_dir, "..", "data", user_id, "pcf_original_table_updated.csv")
                            if not os.path.exists(pcf_table_path):
                                pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_updated.csv")
                                if not os.path.exists(pcf_table_path):
                                    pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_sample.csv")
                        else:
                            pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_updated.csv")
                            if not os.path.exists(pcf_table_path):
                                pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_sample.csv")

                        # PCF 테이블 읽기 및 분석
                        if os.path.exists(pcf_table_path):
                            df = pd.read_csv(pcf_table_path, encoding='utf-8-sig')
                            st.write(f"  - PCF 테이블 경로: {os.path.basename(pcf_table_path)}")
                            st.write(f"  - 전체 자재 수: {len(df)}")

                            # 양극재 관련 자재 분석
                            cathode_by_category = df[df['자재품목'] == '양극재']
                            cathode_by_name = df[df['자재명'].str.contains('양극재|cathode|Cathode', na=False, case=False)]

                            st.write(f"  - 자재품목이 '양극재'인 자재: {len(cathode_by_category)}개")
                            st.write(f"  - 자재명에 '양극재/cathode' 포함된 자재: {len(cathode_by_name)}개")

                            # 양극재의 지역 분석
                            if not cathode_by_category.empty:
                                regions = cathode_by_category['지역'].value_counts()
                                st.write("  - 양극재 지역별 분포:")
                                for region, count in regions.items():
                                    st.write(f"    - {region}: {count}개")

                                # 첫 번째 양극재의 지역 정보
                                first_cathode_region = cathode_by_category.iloc[0]['지역']
                                st.write(f"  - 첫 번째 양극재 지역: {first_cathode_region}")

                            # 국가 코드 변환 정보
                            national_code_path = os.path.join(current_dir, "..", "stable_var", "cathode_national_code.json")
                            if os.path.exists(national_code_path):
                                with open(national_code_path, 'r', encoding='utf-8') as f:
                                    national_codes = json.load(f)
                                    st.write(f"  - 국가 코드 매핑 테이블 로드: {len(national_codes)}개 코드")
                        else:
                            st.write(f"  - PCF 테이블 파일을 찾을 수 없습니다: {pcf_table_path}")

                    except Exception as e:
                        st.write(f"- **Primary Logic 오류**: {e}")
                        # traceback을 사용하려면 파일 상단에 import해야 함
                        # st.write(f"  - 상세 오류: {traceback.format_exc()}")

                    # 4. 3원 분할 비율 계산
                    st.write("---")
                    st.write("### 4. 3원 분할 비율 계산 결과")
                    for element in ["Ni", "Co", "Li"]:
                        recycling_pct = recycle_ratio_updated.get(element, 0) * 100
                        low_carb_pct = low_carb_metal_updated["비중"].get(element, 0)
                        virgin_pct = 100 - recycling_pct - low_carb_pct

                        st.write(f"**{element}:**")
                        st.write(f"- 신재: {virgin_pct:.1f}%")
                        st.write(f"- 재활용재: {recycling_pct:.1f}%")
                        st.write(f"- 저탄소메탈: {low_carb_pct:.1f}%")
                        st.write(f"- 합계: {virgin_pct + recycling_pct + low_carb_pct:.1f}%")

                        if virgin_pct < 0:
                            st.error(f"⚠️ 비율 오류: 합계가 100%를 초과합니다!")

                # 모든 시나리오 데이터 생성
                all_scenarios = simulator.generate_all_scenarios_data()

                # 디버그 로그에 시나리오 생성 결과 추가
                with st.expander("🔍 디버그 로그 (계산 과정 확인)", expanded=False):
                    # 5. 시나리오 생성 결과
                    st.write("---")
                    st.write("### 5. 시나리오 생성 결과")
                    if all_scenarios:
                        st.write(f"- **생성된 시나리오 수**: {len(all_scenarios)}")
                        st.write("- **시나리오 목록:**")

                        for scenario_name, scenario_data in all_scenarios.items():
                            if scenario_name == 'summary':
                                # summary는 전체 요약 정보이므로 별도 처리
                                continue

                            if scenario_data:
                                # 각 시나리오별로 다른 데이터 구조 처리
                                if scenario_name == 'baseline':
                                    # baseline은 emission_data 구조
                                    if 'emission_data' in scenario_data:
                                        total_pcf = scenario_data['emission_data'].get('총_배출량', 0)
                                        st.write(f"  - **{scenario_name}**: PCF = {total_pcf:.2f} kgCO2eq")
                                    else:
                                        st.write(f"  - **{scenario_name}**: 데이터 구조 확인 필요 (키: {list(scenario_data.keys())[:5]})")

                                elif scenario_name in ['recycling_only', 'low_carb_only']:
                                    # 재활용/저탄소메탈 only는 simulation_result 구조
                                    if 'simulation_result' in scenario_data:
                                        sim_result = scenario_data['simulation_result']
                                        if sim_result:
                                            total_emission = sim_result.get('total_emission', 0)
                                            reduction_rate = sim_result.get('reduction_rate', 0)
                                            reduction_amount = sim_result.get('reduction_amount', 0)
                                            st.write(f"  - **{scenario_name}**: PCF = {total_emission:.2f} kgCO2eq (감축량: {reduction_amount:.2f}, 감축률: {reduction_rate:.1f}%)")
                                        else:
                                            st.write(f"  - **{scenario_name}**: simulation_result가 None")
                                    else:
                                        st.write(f"  - **{scenario_name}**: 데이터 구조 확인 필요 (키: {list(scenario_data.keys())[:5]})")

                                elif scenario_name == 'combined_recycling':
                                    # 재활용+저탄소메탈 동시 적용
                                    if 'simulation_result' in scenario_data:
                                        sim_result = scenario_data['simulation_result']
                                        if sim_result and 'emission_data' in sim_result:
                                            total_pcf = sim_result['emission_data'].get('총_배출량', 0)
                                            st.write(f"  - **{scenario_name}**: PCF = {total_pcf:.2f} kgCO2eq")
                                        elif sim_result:
                                            # emission_data가 없고 직접 total_emission이 있는 경우
                                            total_emission = sim_result.get('total_emission', 0)
                                            reduction_rate = sim_result.get('reduction_rate', 0)
                                            st.write(f"  - **{scenario_name}**: PCF = {total_emission:.2f} kgCO2eq (감축률: {reduction_rate:.1f}%)")
                                        else:
                                            st.write(f"  - **{scenario_name}**: simulation_result가 None")
                                    else:
                                        st.write(f"  - **{scenario_name}**: 데이터 구조 확인 필요 (키: {list(scenario_data.keys())[:5]})")

                                elif scenario_name == 'site_change_only':
                                    # 사이트 변경만
                                    if 'after_data' in scenario_data and scenario_data['after_data']:
                                        after_data = scenario_data['after_data']
                                        if 'emission_data' in after_data:
                                            total_pcf = after_data['emission_data'].get('총_배출량', 0)
                                            st.write(f"  - **{scenario_name}**: PCF = {total_pcf:.2f} kgCO2eq (after site)")
                                        else:
                                            st.write(f"  - **{scenario_name}**: after_data 구조 확인 필요")
                                    else:
                                        st.write(f"  - **{scenario_name}**: 데이터 구조 확인 필요 (키: {list(scenario_data.keys())[:5]})")

                                elif scenario_name == 'combined':
                                    # 종합 시나리오 (재활용 + 저탄소메탈 + 사이트변경)
                                    if 'after_recycling' in scenario_data and scenario_data['after_recycling']:
                                        after_recycling = scenario_data['after_recycling']
                                        # after_recycling 구조: {'site': ..., 'simulation_result': {...}, ...}
                                        if 'simulation_result' in after_recycling:
                                            sim_result = after_recycling['simulation_result']
                                            if 'emission_data' in sim_result:
                                                total_pcf = sim_result['emission_data'].get('총_배출량', 0)
                                                st.write(f"  - **{scenario_name}**: PCF = {total_pcf:.2f} kgCO2eq (종합)")
                                            elif 'total_emission' in sim_result:
                                                total_emission = sim_result.get('total_emission', 0)
                                                reduction_rate = sim_result.get('reduction_rate', 0)
                                                st.write(f"  - **{scenario_name}**: PCF = {total_emission:.2f} kgCO2eq (감축률: {reduction_rate:.1f}%) (종합)")
                                            else:
                                                st.write(f"  - **{scenario_name}**: simulation_result 구조 확인 필요 (키: {list(sim_result.keys())[:5]})")
                                        else:
                                            st.write(f"  - **{scenario_name}**: after_recycling에 simulation_result 없음 (키: {list(after_recycling.keys())[:5]})")
                                    else:
                                        st.write(f"  - **{scenario_name}**: 데이터 구조 확인 필요 (키: {list(scenario_data.keys())[:5]})")

                                else:
                                    st.write(f"  - **{scenario_name}**: 알 수 없는 시나리오 타입")
                            else:
                                st.write(f"  - **{scenario_name}**: None (데이터 없음)")

                        # summary 정보 표시
                        if 'summary' in all_scenarios:
                            summary_info = all_scenarios['summary']
                            st.write("\n- **전체 요약:**")
                            st.write(f"  - 총 시나리오 수: {summary_info.get('total_scenarios', 0)}")
                            st.write(f"  - 재활용 비율: {summary_info.get('recycling_ratio', 'N/A')}")
                            st.write(f"  - 저탄소메탈 비율: {summary_info.get('low_carb_ratio', 'N/A')}")
                    else:
                        st.write("- **오류**: 시나리오 생성 실패")

                    # 6. Helper 초기화 및 데이터프레임 생성
                    st.write("---")
                    st.write("### 6. Helper 데이터프레임 생성 정보")
                    try:
                        basic_df = helper.get_basic_scenarios_dataframe(all_scenarios)
                        st.write(f"- **기본 시나리오 데이터프레임 shape**: {basic_df.shape}")
                        st.write(f"- **컬럼 목록**: {', '.join(basic_df.columns.tolist())}")
                    except Exception as e:
                        st.write(f"- **데이터프레임 생성 오류**: {e}")

                    # 7. 상세 계산 결과 (Baseline 시나리오)
                    st.write("---")
                    st.write("### 7. Baseline 시나리오 상세 계산 결과")
                    if 'baseline' in all_scenarios and all_scenarios['baseline']:
                        baseline_data = all_scenarios['baseline']

                        # emission_data에서 정보 추출
                        if 'emission_data' in baseline_data:
                            st.write("**Emission Data:**")
                            emission_data = baseline_data['emission_data']
                            col_sum1, col_sum2 = st.columns(2)
                            with col_sum1:
                                st.write(f"- Total PCF: {emission_data.get('총_배출량', 0):.2f} kgCO2eq")
                                st.write(f"- 원재료 배출량: {emission_data.get('원재료', {}).get('총_배출량', 0):.2f} kgCO2eq")
                                st.write(f"- 원재료 기여도: {emission_data.get('카테고리별_기여도', {}).get('원재료', 0):.1f}%")
                            with col_sum2:
                                energy_tier1 = emission_data.get('Energy(Tier-1)', {}).get('총_배출량', 0)
                                energy_tier2 = emission_data.get('Energy(Tier-2)', {}).get('총_배출량', 0)
                                st.write(f"- Energy Tier1 PCF: {energy_tier1:.2f} kgCO2eq")
                                st.write(f"- Energy Tier2 PCF: {energy_tier2:.2f} kgCO2eq")
                                tier1_contrib = emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-1)', 0)
                                tier2_contrib = emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-2)', 0)
                                st.write(f"- 에너지 기여도: {tier1_contrib + tier2_contrib:.1f}%")

                            # 카테고리별 배출량 상세
                            st.write("\n**카테고리별 상세:**")
                            for category in ["원재료", "Energy(Tier-1)", "Energy(Tier-2)"]:
                                if category in emission_data:
                                    cat_data = emission_data[category]
                                    if isinstance(cat_data, dict) and '총_배출량' in cat_data:
                                        st.write(f"- {category}: {cat_data['총_배출량']:.2f} kgCO2eq")
                        else:
                            st.write("❌ emission_data가 없습니다.")

                    # 8. 재활용 시나리오 계산 상세
                    st.write("---")
                    st.write("### 8. 재활용 Only 시나리오 계산 과정")
                    if 'recycling_only' in all_scenarios and all_scenarios['recycling_only']:
                        recycling_data = all_scenarios['recycling_only']
                        if 'simulation_result' in recycling_data:
                            result = recycling_data['simulation_result']
                            st.write(f"- **재활용 적용 후 PCF**: {result.get('total_emission', 0):.2f} kgCO2eq")
                            st.write(f"- **감축량**: {result.get('reduction_amount', 0):.2f} kgCO2eq")
                            st.write(f"- **감축률**: {result.get('reduction_rate', 0):.2f}%")

                            # 재활용 비율 정보
                            st.write("- **적용된 재활용 비율:**")
                            st.write(f"  - 평균 재활용 비율: {result.get('recycling_ratio_percent', 0):.1f}%")
                            for element in ["Ni", "Co", "Li"]:
                                ratio = recycle_ratio_updated.get(element, 0) * 100
                                if ratio > 0:
                                    st.write(f"  - {element}: {ratio:.1f}%")
                        else:
                            st.write("❌ simulation_result가 없습니다.")
                    else:
                        st.write("❌ recycling_only 시나리오가 없습니다.")

                    # 9. 저탄소메탈 시나리오 계산 상세
                    st.write("---")
                    st.write("### 9. 저탄소메탈 Only 시나리오 계산 과정")
                    if 'low_carb_only' in all_scenarios and all_scenarios['low_carb_only']:
                        lcm_data = all_scenarios['low_carb_only']
                        if 'simulation_result' in lcm_data:
                            result = lcm_data['simulation_result']
                            st.write(f"- **저탄소메탈 적용 후 PCF**: {result.get('total_emission', 0):.2f} kgCO2eq")
                            st.write(f"- **감축량**: {result.get('reduction_amount', 0):.2f} kgCO2eq")
                            st.write(f"- **감축률**: {result.get('reduction_rate', 0):.2f}%")

                            # 저탄소메탈 비중 정보
                            st.write("- **적용된 저탄소메탈 비중:**")
                            st.write(f"  - 평균 저탄소메탈 비율: {result.get('low_carb_ratio_percent', 0):.1f}%")
                            for element in ["Ni", "Co", "Li"]:
                                ratio = low_carb_metal_updated["비중"].get(element, 0)
                                if ratio > 0:
                                    st.write(f"  - {element}: {ratio:.1f}%")
                        else:
                            st.write("❌ simulation_result가 없습니다.")
                    else:
                        st.write("❌ low_carb_only 시나리오가 없습니다.")

                    # 10. 종합 시나리오 계산 상세
                    st.write("---")
                    st.write("### 10. 종합 시나리오 (재활용 + 저탄소메탈 + 사이트변경)")
                    if 'combined' in all_scenarios and all_scenarios['combined']:
                        comp_data = all_scenarios['combined']
                        st.write(f"- **사이트 변경 여부**: {'있음' if comp_data.get('has_site_change', False) else '없음'}")

                        if 'after_recycling' in comp_data and comp_data['after_recycling']:
                            after_recycling = comp_data['after_recycling']
                            if 'simulation_result' in after_recycling:
                                sim_result = after_recycling['simulation_result']
                                if 'emission_data' in sim_result:
                                    emission_data = sim_result['emission_data']
                                    st.write(f"- **종합 시나리오 PCF**: {emission_data.get('총_배출량', 0):.2f} kgCO2eq")
                                elif 'total_emission' in sim_result:
                                    st.write(f"- **종합 시나리오 PCF**: {sim_result.get('total_emission', 0):.2f} kgCO2eq")
                                    st.write(f"- **감축량**: {sim_result.get('reduction_amount', 0):.2f} kgCO2eq")
                                    st.write(f"- **감축률**: {sim_result.get('reduction_rate', 0):.2f}%")

                        # 사이트 변경 추가 효과
                        if comp_data.get('has_site_change', False):
                            st.write(f"- **사이트 변경 추가 감축량**: {comp_data.get('emission_change', 0):.2f} kgCO2eq")
                            st.write(f"- **사이트 변경 추가 감축률**: {comp_data.get('emission_change_rate', 0):.2f}%")

                        # 각 요소별 기여도
                        st.write("- **감축 요인별 효과:**")
                        st.write("  - 재활용 + 저탄소메탈 효과: combined_recycling 시나리오 참조")
                        st.write("  - 사이트 변경 효과: site_change_only 시나리오 참조")
                        st.info("💡 개별 기여도는 시나리오별 결과를 비교하여 확인하세요.")
                    else:
                        st.write("❌ combined 시나리오가 없습니다.")

                    # 11. 📊 계산 과정 데이터프레임 (엑셀 형식)
                    st.write("---")
                    st.write("### 11. 📊 계산 과정 상세 데이터프레임")

                    # 11-1. 시나리오별 요약 데이터프레임
                    st.write("#### 11-1. 시나리오별 요약 비교")
                    scenario_summary_data = []

                    # Baseline PCF 먼저 구하기
                    baseline_pcf = 0
                    if 'baseline' in all_scenarios and all_scenarios['baseline']:
                        baseline_data = all_scenarios['baseline']
                        if 'emission_data' in baseline_data:
                            baseline_pcf = baseline_data['emission_data'].get('총_배출량', 0)

                    for scenario_name, scenario_data in all_scenarios.items():
                        if scenario_name == 'summary':
                            continue

                        if scenario_data:
                            # 각 시나리오별로 다른 데이터 구조 처리
                            scenario_info = {'시나리오': scenario_name}

                            if scenario_name == 'baseline':
                                if 'emission_data' in scenario_data:
                                    emission_data = scenario_data['emission_data']
                                    scenario_info.update({
                                        'Total PCF (kgCO2eq)': emission_data.get('총_배출량', 0),
                                        '원재료 PCF (kgCO2eq)': emission_data.get('원재료_배출량', 0),
                                        'Energy Tier1 PCF (kgCO2eq)': emission_data.get('Energy_Tier1_전력_배출량', 0),
                                        'Energy Tier2 PCF (kgCO2eq)': emission_data.get('Energy_Tier2_전력_배출량', 0),
                                        '원재료 기여도 (%)': emission_data.get('원재료_기여도', 0),
                                        '에너지 기여도 (%)': emission_data.get('Energy_Tier1_전력_기여도', 0) + emission_data.get('Energy_Tier2_전력_기여도', 0),
                                        '감축량 (kgCO2eq)': 0,
                                        '감축률 (%)': 0
                                    })

                            elif scenario_name in ['recycling_only', 'low_carb_only']:
                                if 'simulation_result' in scenario_data and scenario_data['simulation_result']:
                                    result = scenario_data['simulation_result']
                                    total_emission = result.get('total_emission', 0)
                                    reduction_amount = result.get('reduction_amount', 0)
                                    reduction_rate = result.get('reduction_rate', 0)
                                    category_contributions = result.get('category_contributions', {})
                                    scenario_info.update({
                                        'Total PCF (kgCO2eq)': total_emission,
                                        '원재료 PCF (kgCO2eq)': 0,  # 상세 데이터 없음
                                        'Energy Tier1 PCF (kgCO2eq)': 0,
                                        'Energy Tier2 PCF (kgCO2eq)': 0,
                                        '원재료 기여도 (%)': category_contributions.get('원재료', 0),
                                        '에너지 기여도 (%)': category_contributions.get('Energy(Tier-1)', 0) + category_contributions.get('Energy(Tier-2)', 0),
                                        '감축량 (kgCO2eq)': reduction_amount,
                                        '감축률 (%)': reduction_rate
                                    })

                            elif scenario_name == 'combined_recycling':
                                if 'simulation_result' in scenario_data and scenario_data['simulation_result']:
                                    result = scenario_data['simulation_result']
                                    if 'emission_data' in result:
                                        emission_data = result['emission_data']
                                        total_pcf = emission_data.get('총_배출량', 0)
                                        reduction_amount = baseline_pcf - total_pcf
                                        reduction_rate = (reduction_amount / baseline_pcf * 100) if baseline_pcf > 0 else 0
                                        scenario_info.update({
                                            'Total PCF (kgCO2eq)': total_pcf,
                                            '원재료 PCF (kgCO2eq)': emission_data.get('원재료_배출량', 0),
                                            'Energy Tier1 PCF (kgCO2eq)': emission_data.get('Energy_Tier1_전력_배출량', 0),
                                            'Energy Tier2 PCF (kgCO2eq)': emission_data.get('Energy_Tier2_전력_배출량', 0),
                                            '원재료 기여도 (%)': emission_data.get('원재료_기여도', 0),
                                            '에너지 기여도 (%)': emission_data.get('Energy_Tier1_전력_기여도', 0) + emission_data.get('Energy_Tier2_전력_기여도', 0),
                                            '감축량 (kgCO2eq)': reduction_amount,
                                            '감축률 (%)': reduction_rate
                                        })
                                    else:
                                        # emission_data가 없고 직접 total_emission이 있는 경우
                                        total_emission = result.get('total_emission', 0)
                                        reduction_amount = result.get('reduction_amount', 0)
                                        reduction_rate = result.get('reduction_rate', 0)
                                        category_contributions = result.get('category_contributions', {})
                                        scenario_info.update({
                                            'Total PCF (kgCO2eq)': total_emission,
                                            '원재료 PCF (kgCO2eq)': 0,
                                            'Energy Tier1 PCF (kgCO2eq)': 0,
                                            'Energy Tier2 PCF (kgCO2eq)': 0,
                                            '원재료 기여도 (%)': category_contributions.get('원재료', 0),
                                            '에너지 기여도 (%)': category_contributions.get('Energy(Tier-1)', 0) + category_contributions.get('Energy(Tier-2)', 0),
                                            '감축량 (kgCO2eq)': reduction_amount,
                                            '감축률 (%)': reduction_rate
                                        })

                            elif scenario_name == 'site_change_only':
                                if 'after_data' in scenario_data and scenario_data['after_data']:
                                    after_data = scenario_data['after_data']
                                    if 'emission_data' in after_data:
                                        emission_data = after_data['emission_data']
                                        total_pcf = emission_data.get('총_배출량', 0)
                                        reduction_amount = baseline_pcf - total_pcf
                                        reduction_rate = (reduction_amount / baseline_pcf * 100) if baseline_pcf > 0 else 0
                                        scenario_info.update({
                                            'Total PCF (kgCO2eq)': total_pcf,
                                            '원재료 PCF (kgCO2eq)': emission_data.get('원재료_배출량', 0),
                                            'Energy Tier1 PCF (kgCO2eq)': emission_data.get('Energy_Tier1_전력_배출량', 0),
                                            'Energy Tier2 PCF (kgCO2eq)': emission_data.get('Energy_Tier2_전력_배출량', 0),
                                            '원재료 기여도 (%)': emission_data.get('원재료_기여도', 0),
                                            '에너지 기여도 (%)': emission_data.get('Energy_Tier1_전력_기여도', 0) + emission_data.get('Energy_Tier2_전력_기여도', 0),
                                            '감축량 (kgCO2eq)': reduction_amount,
                                            '감축률 (%)': reduction_rate
                                        })

                            elif scenario_name == 'combined':
                                if 'after_recycling' in scenario_data and scenario_data['after_recycling']:
                                    after_recycling = scenario_data['after_recycling']
                                    # after_recycling 구조: {'site': ..., 'simulation_result': {...}, ...}
                                    if 'simulation_result' in after_recycling:
                                        sim_result = after_recycling['simulation_result']
                                        if 'emission_data' in sim_result:
                                            emission_data = sim_result['emission_data']
                                            total_pcf = emission_data.get('총_배출량', 0)
                                            reduction_amount = baseline_pcf - total_pcf
                                            reduction_rate = (reduction_amount / baseline_pcf * 100) if baseline_pcf > 0 else 0
                                            scenario_info.update({
                                                'Total PCF (kgCO2eq)': total_pcf,
                                                '원재료 PCF (kgCO2eq)': emission_data.get('원재료_배출량', 0),
                                                'Energy Tier1 PCF (kgCO2eq)': emission_data.get('Energy_Tier1_전력_배출량', 0),
                                                'Energy Tier2 PCF (kgCO2eq)': emission_data.get('Energy_Tier2_전력_배출량', 0),
                                                '원재료 기여도 (%)': emission_data.get('원재료_기여도', 0),
                                                '에너지 기여도 (%)': emission_data.get('Energy_Tier1_전력_기여도', 0) + emission_data.get('Energy_Tier2_전력_기여도', 0),
                                                '감축량 (kgCO2eq)': reduction_amount,
                                                '감축률 (%)': reduction_rate
                                            })
                                        elif 'total_emission' in sim_result:
                                            total_emission = sim_result.get('total_emission', 0)
                                            reduction_amount = sim_result.get('reduction_amount', 0)
                                            reduction_rate = sim_result.get('reduction_rate', 0)
                                            category_contributions = sim_result.get('category_contributions', {})
                                            scenario_info.update({
                                                'Total PCF (kgCO2eq)': total_emission,
                                                '원재료 PCF (kgCO2eq)': 0,
                                                'Energy Tier1 PCF (kgCO2eq)': 0,
                                                'Energy Tier2 PCF (kgCO2eq)': 0,
                                                '원재료 기여도 (%)': category_contributions.get('원재료', 0),
                                                '에너지 기여도 (%)': category_contributions.get('Energy(Tier-1)', 0) + category_contributions.get('Energy(Tier-2)', 0),
                                                '감축량 (kgCO2eq)': reduction_amount,
                                                '감축률 (%)': reduction_rate
                                            })

                            # 기본값 설정 (데이터가 없는 경우)
                            if len(scenario_info) == 1:  # 시나리오 이름만 있는 경우
                                scenario_info.update({
                                    'Total PCF (kgCO2eq)': 0,
                                    '원재료 PCF (kgCO2eq)': 0,
                                    'Energy Tier1 PCF (kgCO2eq)': 0,
                                    'Energy Tier2 PCF (kgCO2eq)': 0,
                                    '원재료 기여도 (%)': 0,
                                    '에너지 기여도 (%)': 0,
                                    '감축량 (kgCO2eq)': 0,
                                    '감축률 (%)': 0
                                })

                            scenario_summary_data.append(scenario_info)

                    if scenario_summary_data:
                        scenario_summary_df = pd.DataFrame(scenario_summary_data)
                        st.dataframe(scenario_summary_df, use_container_width=True)

                        # CSV 다운로드 버튼
                        csv = scenario_summary_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 시나리오 요약 데이터 다운로드 (CSV)",
                            data=csv,
                            file_name="scenario_summary_debug.csv",
                            mime="text/csv"
                        )

                    # 11-2. 원소별 3원 분할 상세 데이터프레임
                    st.write("#### 11-2. 원소별 3원 분할 상세")
                    element_split_data = []
                    for element in ["Ni", "Co", "Li"]:
                        recycling_pct = recycle_ratio_updated.get(element, 0) * 100
                        low_carb_pct = low_carb_metal_updated["비중"].get(element, 0)
                        virgin_pct = 100 - recycling_pct - low_carb_pct

                        element_split_data.append({
                            '원소': element,
                            '신재 비율 (%)': virgin_pct,
                            '재활용재 비율 (%)': recycling_pct,
                            '저탄소메탈 비율 (%)': low_carb_pct,
                            '합계 (%)': virgin_pct + recycling_pct + low_carb_pct,
                            '재활용재 환경영향계수': recycle_impact_updated["재활용재"].get(element, 0),
                            '저탄소메탈 배출계수': low_carb_metal_updated["배출계수"].get(element, 0),
                            '검증': '✅ OK' if abs((virgin_pct + recycling_pct + low_carb_pct) - 100) < 0.01 else '❌ ERROR'
                        })

                    element_split_df = pd.DataFrame(element_split_data)
                    st.dataframe(element_split_df, use_container_width=True)

                    # 11-3. 양극재 구성 및 생산지 데이터프레임
                    st.write("#### 11-3. 양극재 구성 및 생산지 설정")
                    cathode_config_data = []

                    # 양극재 조성비
                    for element, ratio in cathode_ratio_updated.items():
                        cathode_config_data.append({
                            '구분': '양극재 조성비',
                            '항목': element,
                            '값': f"{ratio * 100:.1f}%",
                            '비고': '원소 비율'
                        })

                    # 생산지 정보
                    for material in ["CAM", "pCAM"]:
                        for timing in ["before", "after"]:
                            site = cathode_site_updated.get(material, {}).get(timing, "N/A")
                            cathode_config_data.append({
                                '구분': f'{material} 생산지',
                                '항목': timing,
                                '값': site,
                                '비고': f'전력배출계수: {electricity_coef_updated.get(site, "N/A")}'
                            })

                    cathode_config_df = pd.DataFrame(cathode_config_data)
                    st.dataframe(cathode_config_df, use_container_width=True)

                    # 11-4. Baseline 상세 자재별 배출량 데이터프레임 (상위 20개)
                    if 'baseline' in all_scenarios and all_scenarios['baseline']:
                        baseline_data = all_scenarios['baseline']
                        if 'detailed' in baseline_data and 'materials' in baseline_data['detailed']:
                            st.write("#### 11-4. Baseline 자재별 배출량 상세 (상위 20개)")

                            materials = baseline_data['detailed']['materials']
                            top_materials = sorted(materials, key=lambda x: x.get('emission', 0), reverse=True)[:20]

                            material_emission_data = []
                            for mat in top_materials:
                                material_emission_data.append({
                                    '자재명': mat.get('name', 'N/A'),
                                    '카테고리': mat.get('category', 'N/A'),
                                    '배출량 (kgCO2eq)': mat.get('emission', 0),
                                    '배출계수': mat.get('emission_factor', 0),
                                    '사용량': mat.get('quantity', 0),
                                    '단위': mat.get('unit', 'N/A'),
                                    '지역': mat.get('region', 'N/A')
                                })

                            material_emission_df = pd.DataFrame(material_emission_data)
                            st.dataframe(material_emission_df, use_container_width=True)

                            # CSV 다운로드 버튼
                            csv = material_emission_df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="📥 자재별 배출량 데이터 다운로드 (CSV)",
                                data=csv,
                                file_name="material_emissions_debug.csv",
                                mime="text/csv"
                            )

                    # 11-5. 시나리오별 감축 효과 분석 데이터프레임
                    st.write("#### 11-5. 시나리오별 감축 효과 분석")

                    # baseline_pcf는 이미 위에서 계산됨
                    if baseline_pcf > 0:
                        reduction_analysis_data = []

                        for scenario_name, scenario_data in all_scenarios.items():
                            if scenario_name in ['baseline', 'summary']:
                                continue

                            if scenario_data:
                                # 각 시나리오별로 PCF 값 추출
                                scenario_pcf = 0
                                material_contribution = 0
                                energy_contribution = 0

                                if scenario_name in ['recycling_only', 'low_carb_only']:
                                    if 'simulation_result' in scenario_data and scenario_data['simulation_result']:
                                        result = scenario_data['simulation_result']
                                        scenario_pcf = result.get('total_emission', 0)
                                        material_contribution = result.get('category_contributions', {}).get('원재료', 0)
                                        energy_contribution = result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)

                                elif scenario_name == 'combined_recycling':
                                    if 'simulation_result' in scenario_data and scenario_data['simulation_result']:
                                        result = scenario_data['simulation_result']
                                        if 'emission_data' in result:
                                            scenario_pcf = result['emission_data'].get('총_배출량', 0)
                                            material_contribution = result['emission_data'].get('원재료_기여도', 0)
                                            energy_contribution = result['emission_data'].get('Energy_Tier1_전력_기여도', 0) + result['emission_data'].get('Energy_Tier2_전력_기여도', 0)
                                        else:
                                            scenario_pcf = result.get('total_emission', 0)
                                            material_contribution = result.get('category_contributions', {}).get('원재료', 0)
                                            energy_contribution = result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)

                                elif scenario_name == 'site_change_only':
                                    if 'after_data' in scenario_data and scenario_data['after_data']:
                                        after_data = scenario_data['after_data']
                                        if 'emission_data' in after_data:
                                            scenario_pcf = after_data['emission_data'].get('총_배출량', 0)
                                            material_contribution = after_data['emission_data'].get('원재료_기여도', 0)
                                            energy_contribution = after_data['emission_data'].get('Energy_Tier1_전력_기여도', 0) + after_data['emission_data'].get('Energy_Tier2_전력_기여도', 0)

                                elif scenario_name == 'combined':
                                    if 'after_recycling' in scenario_data and scenario_data['after_recycling']:
                                        after_recycling = scenario_data['after_recycling']
                                        # after_recycling 구조: {'site': ..., 'simulation_result': {...}, ...}
                                        if 'simulation_result' in after_recycling:
                                            sim_result = after_recycling['simulation_result']
                                            if 'emission_data' in sim_result:
                                                scenario_pcf = sim_result['emission_data'].get('총_배출량', 0)
                                                material_contribution = sim_result['emission_data'].get('원재료_기여도', 0)
                                                energy_contribution = sim_result['emission_data'].get('Energy_Tier1_전력_기여도', 0) + sim_result['emission_data'].get('Energy_Tier2_전력_기여도', 0)
                                            elif 'total_emission' in sim_result:
                                                scenario_pcf = sim_result.get('total_emission', 0)
                                                material_contribution = sim_result.get('category_contributions', {}).get('원재료', 0)
                                                energy_contribution = sim_result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + sim_result.get('category_contributions', {}).get('Energy(Tier-2)', 0)

                                # 감축 효과 계산
                                if scenario_pcf > 0:
                                    reduction = baseline_pcf - scenario_pcf
                                    reduction_rate = (reduction / baseline_pcf * 100) if baseline_pcf > 0 else 0

                                    reduction_analysis_data.append({
                                        '시나리오': scenario_name,
                                        'Baseline PCF (kgCO2eq)': baseline_pcf,
                                        '시나리오 PCF (kgCO2eq)': scenario_pcf,
                                        '감축량 (kgCO2eq)': reduction,
                                        '감축률 (%)': reduction_rate,
                                        '원재료 기여도 변화 (%)': material_contribution,
                                        '에너지 기여도 변화 (%)': energy_contribution
                                    })

                        if reduction_analysis_data:
                            reduction_analysis_df = pd.DataFrame(reduction_analysis_data)
                            reduction_analysis_df = reduction_analysis_df.sort_values('감축률 (%)', ascending=False)
                            st.dataframe(reduction_analysis_df, use_container_width=True)

                    # 11-6. 재활용 및 저탄소메탈 효과 상세 분석
                    st.write("#### 11-6. 재활용 및 저탄소메탈 효과 상세")
                    effect_analysis_data = []

                    # 재활용 효과
                    for element in ["Ni", "Co", "Li"]:
                        recycling_ratio = recycle_ratio_updated.get(element, 0)
                        impact_factor = recycle_impact_updated["재활용재"].get(element, 0)
                        effect = recycling_ratio * (1 - impact_factor)  # 감축 효과 추정

                        effect_analysis_data.append({
                            '구분': '재활용',
                            '원소': element,
                            '적용 비율 (%)': recycling_ratio * 100,
                            '환경영향계수': impact_factor,
                            '예상 감축 효과 (%)': effect * 100,
                            '비고': f'감축 = 비율 × (1 - 환경영향계수)'
                        })

                    # 저탄소메탈 효과
                    for element in ["Ni", "Co", "Li"]:
                        low_carb_ratio = low_carb_metal_updated["비중"].get(element, 0) / 100
                        emission_factor = low_carb_metal_updated["배출계수"].get(element, 0)

                        effect_analysis_data.append({
                            '구분': '저탄소메탈',
                            '원소': element,
                            '적용 비율 (%)': low_carb_ratio * 100,
                            '환경영향계수': emission_factor,
                            '예상 감축 효과 (%)': None,  # Use None instead of 'N/A' to maintain type consistency with numeric column
                            '비고': f'배출계수: {emission_factor}'
                        })

                    effect_analysis_df = pd.DataFrame(effect_analysis_data)
                    st.dataframe(effect_analysis_df, use_container_width=True)

                    st.success("✅ 디버그 데이터프레임 생성 완료! 위 표들을 통해 계산 과정의 세부 내용을 확인할 수 있습니다.")
                
                if all_scenarios:
                    # 1. 기본 시나리오 분석 결과
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.subheader("📊 1. 기본 시나리오 분석 결과")
                    
                    # 기본 시나리오 데이터프레임 생성
                    basic_df = helper.get_basic_scenarios_dataframe(all_scenarios)
                    if not basic_df.empty:
                        # 데이터프레임 표시
                        st.dataframe(basic_df, use_container_width=True)
                        
                        # (임시) basic_df 구조를 터미널에 출력
                        print("[DEBUG] basic_df columns:", basic_df.columns.tolist())
                        print("[DEBUG] basic_df head:\n", basic_df.head())
                        
                        # 시각화 생성 및 표시
                        st.markdown("---")
                        st.markdown("### 📈 시각화 분석")
                        
                        # 시각화 생성
                        visualizations = create_basic_scenarios_visualizations(basic_df)
                        
                        # 시각화 표시
                        if visualizations:
                            # 각 시각화를 직접 표시
                            for name, fig in visualizations.items():
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("시각화를 생성할 수 없습니다.")
                        
                        # 요약 통계 섹션 제거
                        # st.markdown("---")
                        # st.markdown("### 📋 요약 통계")
                        # summary_stats = create_summary_statistics(basic_df)
                        # if not summary_stats.empty:
                        #     st.dataframe(summary_stats, use_container_width=True)
                        # else:
                        #     st.info("요약 통계를 생성할 수 없습니다.")
                    else:
                        st.warning("기본 시나리오 데이터를 생성할 수 없습니다.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 2. 재활용 only 상세분석 결과
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.subheader("♻️ 2. 재활용 only 상세분석 결과")
                    st.info("재활용만 적용하고 저탄소메탈은 제외한 분석 결과입니다.")
                    
                    # 재활용 only 데이터프레임 생성
                    recycling_only_df = helper.get_recycling_only_detail_dataframe(all_scenarios)
                    if not recycling_only_df.empty:
                        st.dataframe(recycling_only_df, use_container_width=True)
                        
                        # 재활용 효과 요약
                        if len(recycling_only_df) > 1:
                            recycling_row = recycling_only_df[recycling_only_df['시나리오'] == '재활용 적용']
                            if not recycling_row.empty:
                                recycling_ratio = recycling_row.iloc[0]['재활용_비율_퍼센트']
                                reduction_rate = recycling_row.iloc[0]['감축률_퍼센트']
                                st.success(f"💡 **재활용 효과**: 평균 {recycling_ratio:.1f}% 재활용 적용으로 {reduction_rate:.2f}% 감축")
                    else:
                        st.warning("재활용 only 데이터를 생성할 수 없습니다.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 3. 저탄소메탈 only 상세분석 결과
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.subheader("🌱 3. 저탄소메탈 only 상세분석 결과")
                    st.info("저탄소메탈만 적용하고 재활용은 제외한 분석 결과입니다.")
                    
                    # 저탄소메탈 only 데이터프레임 생성
                    low_carb_only_df = helper.get_low_carb_metal_only_detail_dataframe(all_scenarios)
                    if not low_carb_only_df.empty:
                        st.dataframe(low_carb_only_df, use_container_width=True)
                        
                        # 저탄소메탈 효과 요약
                        if len(low_carb_only_df) > 1:
                            low_carb_row = low_carb_only_df[low_carb_only_df['시나리오'] == '저탄소메탈 적용']
                            if not low_carb_row.empty:
                                total_low_carb = low_carb_row.iloc[0]['총_저탄소메탈_비중_퍼센트']
                                reduction_rate = low_carb_row.iloc[0]['감축률_퍼센트']
                                if total_low_carb > 0:
                                    st.success(f"💡 **저탄소메탈 효과**: 평균 {total_low_carb:.1f}% 저탄소메탈 적용으로 {reduction_rate:.2f}% 감축")
                                else:
                                    st.warning("💡 **저탄소메탈 효과**: 저탄소메탈 비중 0% - 효과 없음")
                    else:
                        st.warning("저탄소메탈 only 데이터를 생성할 수 없습니다.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 4. 재활용 및 저탄소메탈 동시적용 상세분석 결과
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.subheader("🔄 4. 재활용 및 저탄소메탈 동시적용 상세분석 결과")
                    st.info("재활용과 저탄소메탈을 모두 동시에 적용한 분석 결과입니다.")
                    
                    # 재활용 + 저탄소메탈 동시적용 데이터프레임 생성
                    combined_df = helper.get_low_carb_metal_detail_dataframe(all_scenarios)
                    if not combined_df.empty:
                        st.dataframe(combined_df, use_container_width=True)
                        
                        # 동시적용 효과 요약
                        if len(combined_df) > 1:
                            baseline_row = combined_df[combined_df['시나리오'] == 'Baseline']
                            if not baseline_row.empty:
                                baseline_emission = baseline_row.iloc[0]['총_배출량_kg_CO2e']
                                
                                st.markdown("---")
                                st.markdown("### 💡 동시적용 효과 요약")
                                
                                for _, row in combined_df.iterrows():
                                    if row['시나리오'] != 'Baseline':
                                        total_low_carb = row['총_저탄소메탈_비중_퍼센트']
                                        reduction_rate = row['감축률_퍼센트']
                                        if '재활용 + 저탄소메탈' in row['시나리오']:
                                            st.success(f"**{row['시나리오']}**: 저탄소메탈 평균 비중 {total_low_carb:.1f}% + 재활용 동시 적용으로 {reduction_rate:.2f}% 감축")
                                        else:
                                            st.info(f"**{row['시나리오']}**: {reduction_rate:.2f}% 감축")
                    else:
                        st.warning("재활용 및 저탄소메탈 동시적용 데이터를 생성할 수 없습니다.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 5. 사이트 변경 상세 분석 결과
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.subheader("📊 5. 사이트 변경 상세 분석 결과")
                    
                    # 사이트 변경 상세 데이터프레임 생성
                    site_change_df = helper.get_site_change_detail_dataframe(all_scenarios)
                    if not site_change_df.empty:
                        st.dataframe(site_change_df, use_container_width=True)
                    else:
                        st.warning("사이트 변경 상세 데이터를 생성할 수 없습니다.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 📥 엑셀 다운로드 버튼 추가
                    st.markdown("---")
                    st.markdown('<div class="result-section">', unsafe_allow_html=True)
                    st.subheader("📥 데이터 다운로드")
                    st.info("시나리오별 데이터와 원본 데이터를 엑셀 파일로 다운로드할 수 있습니다.")

                    # 엑셀 파일 생성
                    try:
                        import io
                        from datetime import datetime

                        # BytesIO 객체 생성
                        output = io.BytesIO()

                        # ExcelWriter 객체 생성
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            # 1. 기본 시나리오 분석 결과 (탭1)
                            if not basic_df.empty:
                                basic_df.to_excel(writer, sheet_name='기본_시나리오_분석', index=False)

                            # 2. 재활용 only 상세분석 (탭2)
                            if not recycling_only_df.empty:
                                recycling_only_df.to_excel(writer, sheet_name='재활용_Only', index=False)

                            # 3. 저탄소메탈 only 상세분석 (탭3)
                            if not low_carb_only_df.empty:
                                low_carb_only_df.to_excel(writer, sheet_name='저탄소메탈_Only', index=False)

                            # 4. 재활용+저탄소메탈 동시적용 (탭4)
                            if not combined_df.empty:
                                combined_df.to_excel(writer, sheet_name='동시적용', index=False)

                            # 5. 사이트 변경 상세 분석 (탭5)
                            if not site_change_df.empty:
                                site_change_df.to_excel(writer, sheet_name='사이트_변경', index=False)

                            # 6. 설정값 요약 (탭6)
                            config_summary_data = []

                            # 양극재 조성비
                            for element, ratio in cathode_ratio_updated.items():
                                config_summary_data.append({
                                    '구분': '양극재_조성비',
                                    '항목': element,
                                    '값': f"{ratio * 100:.1f}%"
                                })

                            # 양극재 생산지
                            for material in ["CAM", "pCAM"]:
                                for timing in ["before", "after"]:
                                    site = cathode_site_updated.get(material, {}).get(timing, "N/A")
                                    config_summary_data.append({
                                        '구분': f'{material}_생산지',
                                        '항목': timing,
                                        '값': site
                                    })

                            # 재활용재 사용비율
                            for element, ratio in recycle_ratio_updated.items():
                                config_summary_data.append({
                                    '구분': '재활용재_사용비율',
                                    '항목': element,
                                    '값': f"{ratio * 100:.1f}%"
                                })

                            # 재활용재 환경영향
                            for element, impact in recycle_impact_updated["재활용재"].items():
                                config_summary_data.append({
                                    '구분': '재활용재_환경영향',
                                    '항목': element,
                                    '값': f"{impact * 100:.1f}%"
                                })

                            # 저탄소메탈 비중
                            for element, ratio in low_carb_metal_updated["비중"].items():
                                config_summary_data.append({
                                    '구분': '저탄소메탈_비중',
                                    '항목': element,
                                    '값': f"{ratio:.1f}%"
                                })

                            # 저탄소메탈 배출계수
                            for element, emission in low_carb_metal_updated["배출계수"].items():
                                config_summary_data.append({
                                    '구분': '저탄소메탈_배출계수',
                                    '항목': element,
                                    '값': f"{emission:.3f}"
                                })

                            # 전력배출계수
                            for country, coef in electricity_coef_updated.items():
                                config_summary_data.append({
                                    '구분': '전력배출계수',
                                    '항목': country,
                                    '값': f"{coef:.6f}"
                                })

                            config_summary_df = pd.DataFrame(config_summary_data)
                            config_summary_df.to_excel(writer, sheet_name='설정값_요약', index=False)

                            # 7. 원본 데이터 (BRM 테이블) - 사용자별 파일 로드
                            try:
                                current_dir = os.path.dirname(os.path.abspath(__file__))
                                if user_id:
                                    pcf_table_path = os.path.join(current_dir, "..", "data", user_id, "pcf_original_table_updated.csv")
                                    if not os.path.exists(pcf_table_path):
                                        pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_updated.csv")
                                        if not os.path.exists(pcf_table_path):
                                            pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_sample.csv")
                                else:
                                    pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_updated.csv")
                                    if not os.path.exists(pcf_table_path):
                                        pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_sample.csv")

                                if os.path.exists(pcf_table_path):
                                    brm_df = pd.read_csv(pcf_table_path, encoding='utf-8-sig')
                                    brm_df.to_excel(writer, sheet_name='원본_BRM_데이터', index=False)
                            except Exception as e:
                                log_warning(f"원본 BRM 데이터 로드 실패: {e}")

                        # ExcelWriter가 닫힌 후 BytesIO에서 값을 가져옴
                        excel_data = output.getvalue()

                        # 다운로드 버튼 생성
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"양극재_시나리오_분석_{timestamp}.xlsx"

                        st.download_button(
                            label="📥 전체 데이터 다운로드 (Excel)",
                            data=excel_data,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="시나리오별 분석 결과, 설정값, 원본 데이터를 포함한 엑셀 파일을 다운로드합니다."
                        )

                        st.success("✅ 엑셀 파일이 준비되었습니다. 다운로드 버튼을 클릭하세요.")

                    except Exception as e:
                        st.error(f"엑셀 파일 생성 중 오류가 발생했습니다: {e}")
                        log_error(f"엑셀 파일 생성 오류: {e}")

                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error("시뮬레이션 실행 중 오류가 발생했습니다.")
                        
            except Exception as e:
                st.error(f"시뮬레이션 실행 중 오류가 발생했습니다: {e}")
                st.exception(e)
        
        else:
            st.markdown("""
            <div class="result-section">
                <h3 style="color: var(--text-color);">📋 설정 안내</h3>
                <p style="color: var(--text-secondary);">왼쪽에서 다음 항목들을 설정하세요:</p>
                <ul style="color: var(--text-secondary);">
                    <li><strong>1. 양극재 조성비:</strong> Ni, Co, Mn, Al의 비율 (합계: 100%)</li>
                    <li><strong>2. 양극재 생산지:</strong> CAM과 pCAM의 before/after 사이트</li>
                    <li><strong>3. 원재료 배출계수:</strong> NiSO4, CoSO4, MnSO4, Al(OH3), NaOH, LiOH.H2O의 배출계수</li>
                    <li><strong>4. 재활용재 설정:</strong> Ni, Co, Li의 사용비율 및 환경영향 계수</li>
                    <li><strong>5. 저탄소메탈 설정:</strong> Ni, Co, Li의 사용비중과 배출계수</li>
                    <li><strong>6. 전력배출계수:</strong> 각 국가의 전력배출계수</li>
                </ul>
                <p style="color: var(--text-secondary);">설정 완료 후 <strong>Apply 설정</strong> 버튼을 클릭하세요.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 현재 설정값 표시
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.subheader("📊 현재 설정값")
            
            # 1. 양극재 조성비 표
            st.write("**1. 양극재 조성비**")
            cathode_ratio_df = pd.DataFrame([
                {"원소": element, "비율 (%)": f"{ratio*100:.4f}%"}
                for element, ratio in cathode_ratio_data.items()
            ])
            st.dataframe(cathode_ratio_df, use_container_width=True, hide_index=True)
            
            # 2. 양극재 생산지 표
            st.write("**2. 양극재 생산지**")
            site_df = pd.DataFrame([
                {"구분": "CAM (before)", "생산지": cathode_site_data.get('CAM', {}).get('before', 'N/A')},
                {"구분": "CAM (after)", "생산지": cathode_site_data.get('CAM', {}).get('after', 'N/A')},
                {"구분": "pCAM (before)", "생산지": cathode_site_data.get('pCAM', {}).get('before', 'N/A')},
                {"구분": "pCAM (after)", "생산지": cathode_site_data.get('pCAM', {}).get('after', 'N/A')}
            ])
            st.dataframe(site_df, use_container_width=True, hide_index=True)

            # 3. 원재료 배출계수 표
            st.write("**3. 원재료 배출계수**")
            raw_materials_coef_df = pd.DataFrame([
                {"원재료": material, "배출계수": f"{cathode_coef_table_data.get('원재료', {}).get(material, {}).get('배출계수', 0.0):.2f}"}
                for material in ["NiSO4", "CoSO4", "MnSO4", "Al(OH3)", "NaOH", "LiOH.H2O"]
            ])
            st.dataframe(raw_materials_coef_df, use_container_width=True, hide_index=True)

            # 4. 재활용재 설정 표
            st.write("**4. 재활용재 설정**")

            # 재활용재 사용비율과 환경영향을 2열로 구성
            col_rec1, col_rec2 = st.columns(2)

            with col_rec1:
                st.write("📊 **사용비율**")
                recycle_ratio_df = pd.DataFrame([
                    {"원소": element, "재활용 비율 (%)": f"{ratio*100:.1f}%"}
                    for element, ratio in recycle_ratio_data.items()
                ])
                st.dataframe(recycle_ratio_df, use_container_width=True, hide_index=True)

            with col_rec2:
                st.write("🔥 **환경영향 계수**")
                recycle_impact_df = pd.DataFrame([
                    {"원소": element, "환경영향 계수 (%)": f"{impact*100:.1f}%"}
                    for element, impact in recycle_impact_data.get("재활용재", {}).items()
                ])
                st.dataframe(recycle_impact_df, use_container_width=True, hide_index=True)

            # 5. 저탄소메탈 설정 표
            st.write("**5. 저탄소메탈 설정**")
            
            # 저탄소메탈 비중 표
            col1, col2 = st.columns(2)
            with col1:
                st.write("📊 **사용비중**")
                low_carb_ratio_df = pd.DataFrame([
                    {"원소": element, "저탄소메탈 비중 (%)": f"{ratio:.1f}%"}
                    for element, ratio in low_carb_metal_data.get("비중", {}).items()
                ])
                st.dataframe(low_carb_ratio_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("🔥 **배출계수**")
                low_carb_emission_df = pd.DataFrame([
                    {"원소": element, "배출계수": f"{emission:.3f}"}
                    for element, emission in low_carb_metal_data.get("배출계수", {}).items()
                ])
                st.dataframe(low_carb_emission_df, use_container_width=True, hide_index=True)
            
            # 3원 분할 비율 요약 표
            st.write("🔄 **3원 분할 비율 요약**")
            three_way_data = []
            for element in ["Ni", "Co", "Li"]:
                recycling_pct = recycle_ratio_data.get(element, 0) * 100
                low_carb_pct = low_carb_metal_data.get("비중", {}).get(element, 0)
                virgin_pct = 100 - recycling_pct - low_carb_pct
                three_way_data.append({
                    "원소": element,
                    "신재 (%)": f"{virgin_pct:.1f}%",
                    "재활용재 (%)": f"{recycling_pct:.1f}%",
                    "저탄소메탈 (%)": f"{low_carb_pct:.1f}%",
                    "합계 (%)": f"{virgin_pct + recycling_pct + low_carb_pct:.1f}%"
                })
            
            three_way_df = pd.DataFrame(three_way_data)
            st.dataframe(three_way_df, use_container_width=True, hide_index=True)
            
            # 6. 전력배출계수 표
            st.write("**6. 전력배출계수**")
            electricity_df = pd.DataFrame([
                {"국가": country, "전력배출계수": f"{coef:.6f}"}
                for country, coef in electricity_coef_data.items()
            ])
            st.dataframe(electricity_df, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

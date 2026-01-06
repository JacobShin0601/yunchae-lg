"""
비용 변수 설정 페이지

RE100 프리미엄 계산에 사용되는 비용 관련 데이터를 설정하고 수정합니다.
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from src.utils.optimization_costs_manager import OptimizationCostsManager
from src.logger import log_info, log_error, log_button_click


def cost_variable_settings_page():
    """비용 변수 설정 페이지"""

    st.title("💰 비용 변수 설정")
    st.markdown("RE100 프리미엄 계산에 사용되는 비용 데이터를 설정합니다.")
    st.markdown("---")

    # 사용자 ID 가져오기
    user_id = st.session_state.get('user_id', None)

    if not user_id:
        st.warning("⚠️ 사용자 ID가 설정되지 않았습니다.")
        return

    st.info(f"👤 현재 사용자: **{user_id}**")

    # OptimizationCostsManager 초기화
    try:
        costs_manager = OptimizationCostsManager()

        # 사용자 파일 존재 확인
        if not costs_manager.check_user_costs_exist(user_id):
            st.warning("⚠️ 사용자별 비용 데이터가 없습니다. 초기화를 진행합니다.")
            costs_manager.initialize_user_costs(user_id)
            st.success("✅ 비용 데이터 초기화 완료")
            st.rerun()

    except Exception as e:
        st.error(f"❌ OptimizationCostsManager 초기화 실패: {e}")
        log_error(f"OptimizationCostsManager 초기화 실패: {e}")
        return

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ 전력사용량 데이터",
        "💵 전력단가 데이터",
        "💰 기본단가 데이터",
        "🔄 자재품목 매핑"
    ])

    # ==================== 탭 1: 전력사용량 데이터 ====================
    with tab1:
        st.header("⚡ 전력사용량 데이터")
        st.markdown("자재품목 및 Tier별 제조 공정의 전력사용량(kWh/kg)을 설정합니다.")

        # 데이터 로드
        try:
            data = costs_manager.load_user_file(
                user_id,
                "electricity_usage_per_material.json",
                fallback_to_template=True
            )

            if not data:
                st.warning("⚠️ 데이터가 없습니다.")
                return

            # DataFrame 변환
            df = pd.DataFrame(data)

            # 편집 가능한 데이터 테이블
            st.markdown("### 📝 데이터 편집")

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",  # 행 추가/삭제 가능
                column_config={
                    "자재품목": st.column_config.TextColumn(
                        "자재품목",
                        help="자재 품목 이름",
                        required=True
                    ),
                    "Tier": st.column_config.SelectboxColumn(
                        "Tier",
                        help="Tier 레벨",
                        options=["Tier1", "Tier2", "Tier3"],
                        required=True
                    ),
                    "전력사용량(kWh/kg)": st.column_config.NumberColumn(
                        "전력사용량(kWh/kg)",
                        help="제조 공정의 전력사용량",
                        min_value=0.0,
                        format="%.6f",
                        required=True
                    ),
                    "공정설명": st.column_config.TextColumn(
                        "공정설명",
                        help="제조 공정 설명"
                    )
                },
                key="electricity_usage_editor"
            )

            # 버튼 영역
            col1, col2, col3 = st.columns([1, 1, 3])

            with col1:
                if st.button("💾 저장", key="save_electricity", type="primary"):
                    log_button_click("저장", "save_electricity_btn")
                    try:
                        # DataFrame을 JSON 형태로 변환
                        data_to_save = edited_df.to_dict(orient='records')

                        # 저장
                        success = costs_manager.update_user_file(
                            user_id,
                            "electricity_usage_per_material.json",
                            data_to_save
                        )

                        if success:
                            st.success("✅ 저장되었습니다!")
                            log_info(f"전력사용량 데이터 저장 완료: {user_id}")
                            st.rerun()
                        else:
                            st.error("❌ 저장 실패")

                    except Exception as e:
                        st.error(f"❌ 저장 중 오류: {e}")
                        log_error(f"전력사용량 데이터 저장 오류: {e}")

            with col2:
                if st.button("🔄 초기화", key="reset_electricity"):
                    log_button_click("초기화", "reset_electricity_btn")
                    try:
                        # 템플릿에서 다시 로드
                        costs_manager.initialize_user_costs(user_id, force=True)
                        st.success("✅ 템플릿으로 초기화되었습니다!")
                        log_info(f"전력사용량 데이터 초기화: {user_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 초기화 오류: {e}")
                        log_error(f"전력사용량 데이터 초기화 오류: {e}")

            # 통계 정보
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📊 총 항목 수", f"{len(edited_df)}개")

            with col2:
                unique_materials = edited_df['자재품목'].nunique()
                st.metric("📦 자재품목 수", f"{unique_materials}개")

            with col3:
                unique_tiers = edited_df['Tier'].nunique()
                st.metric("🎯 Tier 수", f"{unique_tiers}개")

        except Exception as e:
            st.error(f"❌ 데이터 로드 오류: {e}")
            log_error(f"전력사용량 데이터 로드 오류: {e}")

    # ==================== 탭 2: 전력단가 데이터 ====================
    with tab2:
        st.header("💵 전력단가 데이터")
        st.markdown("국가별 전력 단가($/kWh)를 설정합니다.")

        # 데이터 로드
        try:
            data = costs_manager.load_user_file(
                user_id,
                "unit_cost_per_country.json",
                fallback_to_template=True
            )

            if not data:
                st.warning("⚠️ 데이터가 없습니다.")
                return

            # DataFrame 변환
            df = pd.DataFrame(data)

            # 편집 가능한 데이터 테이블
            st.markdown("### 📝 데이터 편집")

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "국가": st.column_config.TextColumn(
                        "국가",
                        help="국가명",
                        required=True
                    ),
                    "금액($/MWh)": st.column_config.NumberColumn(
                        "금액($/MWh)",
                        help="전력 단가 ($/MWh)",
                        min_value=0.0,
                        format="%.2f"
                    ),
                    "금액($/kWh)": st.column_config.NumberColumn(
                        "금액($/kWh)",
                        help="전력 단가 ($/kWh) - 계산에 사용됨",
                        min_value=0.0,
                        format="%.6f",
                        required=True
                    )
                },
                key="unit_cost_editor"
            )

            # $/MWh에서 $/kWh 자동 계산 안내
            st.info("💡 **Tip:** `금액($/kWh) = 금액($/MWh) / 1000` 입니다. 편집 시 참고하세요.")

            # 버튼 영역
            col1, col2, col3 = st.columns([1, 1, 3])

            with col1:
                if st.button("💾 저장", key="save_unit_cost", type="primary"):
                    log_button_click("저장", "save_unit_cost_btn")
                    try:
                        # DataFrame을 JSON 형태로 변환
                        data_to_save = edited_df.to_dict(orient='records')

                        # 저장
                        success = costs_manager.update_user_file(
                            user_id,
                            "unit_cost_per_country.json",
                            data_to_save
                        )

                        if success:
                            st.success("✅ 저장되었습니다!")
                            log_info(f"전력단가 데이터 저장 완료: {user_id}")
                            st.rerun()
                        else:
                            st.error("❌ 저장 실패")

                    except Exception as e:
                        st.error(f"❌ 저장 중 오류: {e}")
                        log_error(f"전력단가 데이터 저장 오류: {e}")

            with col2:
                if st.button("🔄 초기화", key="reset_unit_cost"):
                    log_button_click("초기화", "reset_unit_cost_btn")
                    try:
                        costs_manager.initialize_user_costs(user_id, force=True)
                        st.success("✅ 템플릿으로 초기화되었습니다!")
                        log_info(f"전력단가 데이터 초기화: {user_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 초기화 오류: {e}")
                        log_error(f"전력단가 데이터 초기화 오류: {e}")

            # 통계 정보
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🌍 국가 수", f"{len(edited_df)}개")

            with col2:
                avg_cost = edited_df['금액($/kWh)'].mean()
                st.metric("📊 평균 단가", f"${avg_cost:.6f}/kWh")

            with col3:
                max_cost = edited_df['금액($/kWh)'].max()
                st.metric("📈 최고 단가", f"${max_cost:.6f}/kWh")

        except Exception as e:
            st.error(f"❌ 데이터 로드 오류: {e}")
            log_error(f"전력단가 데이터 로드 오류: {e}")

    # ==================== 탭 3: 기본단가 데이터 ====================
    with tab3:
        st.header("💰 기본단가 데이터")
        st.markdown("자재품목 및 Tier별 기본 조달 단가($/kg)를 설정합니다.")

        # 데이터 로드
        try:
            data = costs_manager.load_user_file(
                user_id,
                "basic_cost_per_material.json",
                fallback_to_template=True
            )

            if not data:
                st.warning("⚠️ 데이터가 없습니다.")
                return

            # DataFrame 변환
            df = pd.DataFrame(data)

            # 편집 가능한 데이터 테이블
            st.markdown("### 📝 데이터 편집")

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "자재품목": st.column_config.TextColumn(
                        "자재품목",
                        help="자재 품목 이름",
                        required=True
                    ),
                    "Tier": st.column_config.SelectboxColumn(
                        "Tier",
                        help="Tier 레벨",
                        options=["Tier1", "Tier2", "Tier3"],
                        required=True
                    ),
                    "기본단가($/kg)": st.column_config.NumberColumn(
                        "기본단가($/kg)",
                        help="자재의 기본 조달 단가",
                        min_value=0.0,
                        format="%.3f",
                        required=True
                    ),
                    "공정설명": st.column_config.TextColumn(
                        "공정설명",
                        help="제조 공정 설명"
                    )
                },
                key="basic_cost_editor"
            )

            # 버튼 영역
            col1, col2, col3 = st.columns([1, 1, 3])

            with col1:
                if st.button("💾 저장", key="save_basic_cost", type="primary"):
                    log_button_click("저장", "save_basic_cost_btn")
                    try:
                        # DataFrame을 JSON 형태로 변환
                        data_to_save = edited_df.to_dict(orient='records')

                        # 저장
                        success = costs_manager.update_user_file(
                            user_id,
                            "basic_cost_per_material.json",
                            data_to_save
                        )

                        if success:
                            st.success("✅ 저장되었습니다!")
                            log_info(f"기본단가 데이터 저장 완료: {user_id}")
                            st.rerun()
                        else:
                            st.error("❌ 저장 실패")

                    except Exception as e:
                        st.error(f"❌ 저장 중 오류: {e}")
                        log_error(f"기본단가 데이터 저장 오류: {e}")

            with col2:
                if st.button("🔄 초기화", key="reset_basic_cost"):
                    log_button_click("초기화", "reset_basic_cost_btn")
                    try:
                        costs_manager.initialize_user_costs(user_id, force=True)
                        st.success("✅ 템플릿으로 초기화되었습니다!")
                        log_info(f"기본단가 데이터 초기화: {user_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 초기화 오류: {e}")
                        log_error(f"기본단가 데이터 초기화 오류: {e}")

            # 통계 정보
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📊 총 항목 수", f"{len(edited_df)}개")

            with col2:
                avg_cost = edited_df['기본단가($/kg)'].mean()
                st.metric("💵 평균 단가", f"${avg_cost:.2f}/kg")

            with col3:
                max_cost = edited_df['기본단가($/kg)'].max()
                st.metric("📈 최고 단가", f"${max_cost:.2f}/kg")

        except Exception as e:
            st.error(f"❌ 데이터 로드 오류: {e}")
            log_error(f"기본단가 데이터 로드 오류: {e}")

    # ==================== 탭 4: 자재품목 매핑 ====================
    with tab4:
        st.header("🔄 자재품목 매핑")
        st.markdown("BRM 테이블의 자재품목과 최적화 모듈의 자재품목 매핑을 설정합니다.")

        # 데이터 로드
        try:
            data = costs_manager.load_user_file(
                user_id,
                "material_category_mapping.json",
                fallback_to_template=True
            )

            if not data:
                st.warning("⚠️ 데이터가 없습니다.")
                return

            # reverse_mappings를 DataFrame으로 변환
            reverse_mappings = data.get('reverse_mappings', {})

            if not reverse_mappings:
                st.warning("⚠️ 매핑 데이터가 없습니다.")
                return

            # DataFrame 생성
            mapping_data = [
                {"BRM 자재품목": k, "최적화 자재품목": v}
                for k, v in reverse_mappings.items()
            ]
            df = pd.DataFrame(mapping_data)

            # 편집 가능한 데이터 테이블
            st.markdown("### 📝 매핑 편집")

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "BRM 자재품목": st.column_config.TextColumn(
                        "BRM 자재품목",
                        help="BRM 테이블의 자재품목 이름",
                        required=True
                    ),
                    "최적화 자재품목": st.column_config.TextColumn(
                        "최적화 자재품목",
                        help="최적화 모듈의 자재품목 이름",
                        required=True
                    )
                },
                key="mapping_editor"
            )

            st.info("💡 **사용법:** BRM 테이블의 자재품목이 여러 이름으로 표현될 수 있습니다. 각 이름을 최적화 모듈의 표준 자재품목으로 매핑합니다.")

            # 버튼 영역
            col1, col2, col3 = st.columns([1, 1, 3])

            with col1:
                if st.button("💾 저장", key="save_mapping", type="primary"):
                    log_button_click("저장", "save_mapping_btn")
                    try:
                        # DataFrame을 reverse_mappings 형태로 변환
                        new_reverse_mappings = {
                            row['BRM 자재품목']: row['최적화 자재품목']
                            for _, row in edited_df.iterrows()
                        }

                        # mappings도 재구성 (최적화 자재품목을 기준으로 그룹화)
                        from collections import defaultdict
                        new_mappings = defaultdict(lambda: {"optimization_name": "", "brm_names": [], "description": ""})

                        for brm_name, opt_name in new_reverse_mappings.items():
                            new_mappings[opt_name]["optimization_name"] = opt_name
                            new_mappings[opt_name]["brm_names"].append(brm_name)

                        # 기존 description 유지
                        original_mappings = data.get('mappings', {})
                        for opt_name, info in new_mappings.items():
                            if opt_name in original_mappings:
                                info["description"] = original_mappings[opt_name].get("description", "")

                        # 저장할 데이터 구성
                        data_to_save = {
                            "mappings": dict(new_mappings),
                            "reverse_mappings": new_reverse_mappings
                        }

                        # 저장
                        success = costs_manager.update_user_file(
                            user_id,
                            "material_category_mapping.json",
                            data_to_save
                        )

                        if success:
                            st.success("✅ 저장되었습니다!")
                            log_info(f"자재품목 매핑 데이터 저장 완료: {user_id}")
                            st.rerun()
                        else:
                            st.error("❌ 저장 실패")

                    except Exception as e:
                        st.error(f"❌ 저장 중 오류: {e}")
                        log_error(f"자재품목 매핑 데이터 저장 오류: {e}")

            with col2:
                if st.button("🔄 초기화", key="reset_mapping"):
                    log_button_click("초기화", "reset_mapping_btn")
                    try:
                        costs_manager.initialize_user_costs(user_id, force=True)
                        st.success("✅ 템플릿으로 초기화되었습니다!")
                        log_info(f"자재품목 매핑 데이터 초기화: {user_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 초기화 오류: {e}")
                        log_error(f"자재품목 매핑 데이터 초기화 오류: {e}")

            # 통계 정보
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("🔄 매핑 항목 수", f"{len(edited_df)}개")

            with col2:
                unique_opt = edited_df['최적화 자재품목'].nunique()
                st.metric("📦 고유 최적화 자재품목", f"{unique_opt}개")

        except Exception as e:
            st.error(f"❌ 데이터 로드 오류: {e}")
            log_error(f"자재품목 매핑 데이터 로드 오류: {e}")


if __name__ == "__main__":
    cost_variable_settings_page()

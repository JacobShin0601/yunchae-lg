"""
RE100 프리미엄 계산 테스트 페이지

RE100PremiumCalculator의 기능을 테스트하고 검증하는 페이지입니다.
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

from src.optimization.re100_premium_calculator import RE100PremiumCalculator
from src.logger import log_info, log_error


def re100_premium_test_page():
    """RE100 프리미엄 계산 테스트 페이지"""

    st.title("⚡ RE100 프리미엄 계산 테스트")
    st.markdown("---")

    # 사용자 ID 가져오기
    user_id = st.session_state.get('user_id', None)

    if not user_id:
        st.warning("⚠️ 사용자 ID가 설정되지 않았습니다.")
        return

    st.info(f"👤 현재 사용자: **{user_id}**")

    # Calculator 초기화
    try:
        calculator = RE100PremiumCalculator(user_id=user_id, debug_mode=False)
        st.success("✅ RE100PremiumCalculator 초기화 완료")
    except Exception as e:
        st.error(f"❌ Calculator 초기화 실패: {e}")
        log_error(f"RE100PremiumCalculator 초기화 실패: {e}")
        return

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 데이터 확인",
        "💰 RE100 전환가격",
        "📈 상승률 계산",
        "🎯 Case별 프리미엄",
        "📋 시나리오 전체 계산"
    ])

    # ==================== 탭 1: 데이터 확인 ====================
    with tab1:
        st.header("📊 로드된 데이터 확인")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "전력사용량 데이터",
                f"{len(calculator.electricity_usage_data)}개",
                help="자재품목+Tier별 전력사용량"
            )

        with col2:
            st.metric(
                "전력단가 데이터",
                f"{len(calculator.unit_cost_data)}개",
                help="국가별 전력단가"
            )

        with col3:
            st.metric(
                "기본단가 데이터",
                f"{len(calculator.basic_cost_data)}개",
                help="자재품목+Tier별 기본단가"
            )

        with col4:
            st.metric(
                "자재품목 매핑",
                f"{len(calculator.material_mapping)}개",
                help="BRM → 최적화 자재품목 매핑"
            )

        st.markdown("---")

        # 전력사용량 데이터 표시
        with st.expander("🔋 전력사용량 데이터 (electricity_usage_per_material.json)"):
            if calculator.electricity_usage_data:
                df = pd.DataFrame(calculator.electricity_usage_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("데이터가 없습니다.")

        # 전력단가 데이터 표시
        with st.expander("💵 전력단가 데이터 (unit_cost_per_country.json)"):
            if calculator.unit_cost_data:
                df = pd.DataFrame(calculator.unit_cost_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("데이터가 없습니다.")

        # 기본단가 데이터 표시
        with st.expander("💰 기본단가 데이터 (basic_cost_per_material.json)"):
            if calculator.basic_cost_data:
                df = pd.DataFrame(calculator.basic_cost_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("데이터가 없습니다.")

        # 자재품목 매핑 표시
        with st.expander("🔄 자재품목 매핑 (material_category_mapping.json)"):
            if calculator.material_mapping:
                mapping_df = pd.DataFrame([
                    {"BRM 자재품목": k, "최적화 자재품목": v}
                    for k, v in calculator.material_mapping.items()
                ])
                st.dataframe(mapping_df, use_container_width=True)
            else:
                st.warning("매핑 데이터가 없습니다.")

    # ==================== 탭 2: RE100 전환가격 ====================
    with tab2:
        st.header("💰 RE100 전환가격 계산 테스트")
        st.markdown("**공식:** `전력사용량(kWh/kg) × 전력단가($/kWh)`")

        col1, col2, col3 = st.columns(3)

        with col1:
            material_options = list(calculator.electricity_index.keys())
            material = st.selectbox("자재품목", material_options, key="conv_material")

        with col2:
            if material and material in calculator.electricity_index:
                tier_options = list(calculator.electricity_index[material].keys())
                tier = st.selectbox("Tier", tier_options, key="conv_tier")
            else:
                tier = st.selectbox("Tier", ["Tier1", "Tier2"], key="conv_tier")

        with col3:
            country_options = list(calculator.unit_cost_index.keys())
            country = st.selectbox("국가", country_options, key="conv_country")

        if st.button("계산", key="calc_conversion", type="primary"):
            try:
                # 전환가격 계산
                conversion_price = calculator.calculate_re100_conversion_price(
                    material, tier, country
                )

                # 세부 정보 표시
                usage = calculator._get_electricity_usage(material, tier)
                unit_cost = calculator._get_unit_cost(country)

                st.markdown("---")
                st.subheader("📊 계산 결과")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("전력사용량", f"{usage:.4f} kWh/kg")

                with col2:
                    st.metric("전력단가", f"${unit_cost:.6f}/kWh")

                with col3:
                    st.metric("RE100 전환가격", f"${conversion_price:.6f}/kg")

                with col4:
                    # 예상 값 (sample_calculation.csv 참고)
                    expected_values = {
                        ("양극재", "Tier1", "한국"): 0.055440,
                        ("양극재", "Tier2", "한국"): 0.023030,
                        ("Cu-Foil", "Tier1", "한국"): 0.102200,
                        ("분리막", "Tier1", "한국"): 0.065000,
                    }
                    expected = expected_values.get((material, tier, country), None)
                    if expected:
                        diff = abs(conversion_price - expected)
                        match = "✅" if diff < 0.001 else "⚠️"
                        st.metric("검증", f"{match} 예상값", f"${expected:.6f}/kg")

                # 계산 과정 표시
                st.markdown("**계산 과정:**")
                st.code(f"{usage:.4f} kWh/kg × ${unit_cost:.6f}/kWh = ${conversion_price:.6f}/kg")

            except Exception as e:
                st.error(f"❌ 계산 오류: {e}")
                log_error(f"RE100 전환가격 계산 오류: {e}")

    # ==================== 탭 3: 상승률 계산 ====================
    with tab3:
        st.header("📈 상승률 계산 테스트")
        st.markdown("**공식:** `(RE100_전환가격 / 기본단가) × 100`")

        col1, col2, col3 = st.columns(3)

        with col1:
            material_options = list(calculator.electricity_index.keys())
            material = st.selectbox("자재품목", material_options, key="rate_material")

        with col2:
            if material and material in calculator.electricity_index:
                tier_options = list(calculator.electricity_index[material].keys())
                tier = st.selectbox("Tier", tier_options, key="rate_tier")
            else:
                tier = st.selectbox("Tier", ["Tier1", "Tier2"], key="rate_tier")

        with col3:
            country_options = list(calculator.unit_cost_index.keys())
            country = st.selectbox("국가", country_options, key="rate_country")

        if st.button("계산", key="calc_rate", type="primary"):
            try:
                # 상승률 계산
                rate = calculator.calculate_premium_rate(material, tier, country)

                # 세부 정보
                conversion_price = calculator.calculate_re100_conversion_price(material, tier, country)
                basic_cost = calculator._get_basic_cost(material, tier)

                st.markdown("---")
                st.subheader("📊 계산 결과")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("RE100 전환가격", f"${conversion_price:.6f}/kg")

                with col2:
                    st.metric("기본단가", f"${basic_cost:.3f}/kg")

                with col3:
                    st.metric("상승률", f"{rate:.2f}%")

                with col4:
                    # 예상 값
                    expected_rates = {
                        ("양극재", "Tier1", "한국"): 0.75,
                        ("양극재", "Tier2", "한국"): 0.31,
                        ("Cu-Foil", "Tier1", "한국"): 1.93,
                        ("분리막", "Tier1", "한국"): 0.20,
                    }
                    expected = expected_rates.get((material, tier, country), None)
                    if expected:
                        diff = abs(rate - expected)
                        match = "✅" if diff < 0.05 else "⚠️"
                        st.metric("검증", f"{match} 예상값", f"{expected:.2f}%")

                # 계산 과정
                st.markdown("**계산 과정:**")
                st.code(f"({conversion_price:.6f} / {basic_cost:.3f}) × 100 = {rate:.2f}%")

            except Exception as e:
                st.error(f"❌ 계산 오류: {e}")
                log_error(f"상승률 계산 오류: {e}")

    # ==================== 탭 4: Case별 프리미엄 ====================
    with tab4:
        st.header("🎯 Case별 프리미엄 계산 테스트")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("자재 정보")
            material_name = st.text_input("자재명", value="Foil Cu General", key="case_mat_name")
            material_category = st.text_input("자재품목", value="Cu Foil", key="case_mat_cat")
            quantity = st.number_input("제품총소요량(kg)", value=0.085396943, format="%.9f", key="case_qty")
            country = st.selectbox("국가", list(calculator.unit_cost_index.keys()), key="case_country")

        with col2:
            st.subheader("RE 적용률")
            tier1_re = st.slider("Tier1 RE (%)", 0, 100, 100, key="case_tier1") / 100.0
            tier2_re = st.slider("Tier2 RE (%)", 0, 100, 0, key="case_tier2") / 100.0

        if st.button("계산", key="calc_case", type="primary"):
            try:
                case_config = {
                    "tier1_re": tier1_re,
                    "tier2_re": tier2_re
                }

                premium = calculator.calculate_case_premium(
                    material_name, material_category, quantity, country, case_config
                )

                st.markdown("---")
                st.subheader("📊 계산 결과")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Tier1 프리미엄", f"${premium['tier1_premium']:.6f}")

                with col2:
                    st.metric("Tier2 프리미엄", f"${premium['tier2_premium']:.6f}")

                with col3:
                    st.metric("총 프리미엄", f"${premium['total_premium']:.6f}")

                with col4:
                    st.metric("상승률", f"{premium['premium_rate']:.2f}%")

                # 상세 정보
                with st.expander("📋 상세 정보"):
                    st.json(premium)

            except Exception as e:
                st.error(f"❌ 계산 오류: {e}")
                log_error(f"Case별 프리미엄 계산 오류: {e}")

    # ==================== 탭 5: 시나리오 전체 계산 ====================
    with tab5:
        st.header("📋 시나리오 전체 계산 테스트")

        # 파일 경로 확인
        scenario_path = Path(f"data/{user_id}/pcf_scenario_saved.csv")
        original_path = Path(f"data/{user_id}/pcf_original_table_sample.csv")

        col1, col2 = st.columns(2)

        with col1:
            scenario_exists = scenario_path.exists()
            if scenario_exists:
                st.success(f"✅ 시나리오 파일 존재")
            else:
                st.error(f"❌ 시나리오 파일 없음")
            st.caption(str(scenario_path))

        with col2:
            original_exists = original_path.exists()
            if original_exists:
                st.success(f"✅ 원본 테이블 존재")
            else:
                st.error(f"❌ 원본 테이블 없음")
            st.caption(str(original_path))

        if scenario_exists and original_exists:
            if st.button("시나리오 전체 계산 실행", key="calc_scenario", type="primary"):
                with st.spinner("계산 중..."):
                    try:
                        # 데이터 로드
                        scenario_df = pd.read_csv(scenario_path, encoding='utf-8-sig')
                        original_df = pd.read_csv(original_path, encoding='utf-8-sig')

                        st.info(f"📊 시나리오 자재 수: {len(scenario_df)}개")
                        st.info(f"📊 원본 테이블 자재 수: {len(original_df)}개")

                        # 전체 계산
                        result_df = calculator.calculate_scenario_premiums(scenario_df, original_df)

                        st.success(f"✅ 계산 완료: {len(result_df)}개 자재")

                        # 결과 표시
                        st.markdown("---")
                        st.subheader("📊 계산 결과")

                        # 요약 통계
                        case_cols = [col for col in result_df.columns if 'premium($)' in col]

                        if case_cols:
                            st.markdown("**Case별 총 프리미엄:**")
                            summary_data = []
                            for col in case_cols:
                                case_num = col.split('case')[1].split('_')[0]
                                total = result_df[col].sum()
                                summary_data.append({
                                    "Case": f"Case {case_num}",
                                    "총 프리미엄": f"${total:,.2f}"
                                })

                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)

                        # 전체 결과 테이블
                        st.markdown("---")
                        st.markdown("**전체 결과 테이블:**")
                        st.dataframe(result_df, use_container_width=True)

                        # CSV 다운로드
                        csv = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button(
                            label="📥 결과 다운로드 (CSV)",
                            data=csv,
                            file_name=f"re100_premium_results_{user_id}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ 계산 오류: {e}")
                        log_error(f"시나리오 전체 계산 오류: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ 시나리오 파일 또는 원본 테이블 파일이 없습니다. PCF 시뮬레이터를 먼저 실행하세요.")


if __name__ == "__main__":
    re100_premium_test_page()

"""
결과 시각화 UI

최적화 결과를 사용자 친화적으로 표시하고 시각화하는 컴포넌트입니다.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional


class ResultsVisualizer:
    """결과 시각화 UI 클래스"""

    def __init__(self):
        """초기화"""
        pass

    def render(
        self,
        result_df: pd.DataFrame,
        summary: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> None:
        """
        결과 전체 렌더링

        Args:
            result_df: 결과 DataFrame (ResultProcessor에서 생성)
            summary: 요약 통계
            solution: 원본 solution 딕셔너리
        """
        st.header("📊 최적화 결과")

        # 상태 표시
        self._render_status(solution)

        st.markdown("---")

        # 요약 카드
        self._render_summary_cards(summary)

        st.markdown("---")

        # 상세 결과 테이블
        self._render_detail_table(result_df)

        st.markdown("---")

        # 시각화
        self._render_charts(result_df, summary, solution)

    def _render_status(self, solution: Dict[str, Any]) -> None:
        """최적화 상태 표시"""
        status = solution.get('status', 'unknown')

        if 'optimal' in status.lower():
            st.success("✅ 최적화 성공: 최적해 발견!")
        elif 'feasible' in status.lower():
            st.warning("⚠️ 최적화 완료: 실현가능해 발견 (최적은 아님)")
        else:
            st.error(f"❌ 최적화 실패: {status}")

    def _render_summary_cards(self, summary: Dict[str, Any]) -> None:
        """요약 카드 렌더링"""
        st.subheader("📈 요약 통계")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "대상 자재",
                f"{summary['material_count']}개",
                help="최적화 대상 자재 개수"
            )

        with col2:
            baseline = summary['baseline_total_emission']
            st.metric(
                "기준 배출량",
                f"{baseline:,.1f} kg",
                help="최적화 전 총 탄소배출량"
            )

        with col3:
            optimized = summary['optimized_total_emission']
            reduction_pct = summary['total_reduction_pct']
            st.metric(
                "최적 배출량",
                f"{optimized:,.1f} kg",
                f"-{reduction_pct:.1f}%",
                delta_color="inverse",
                help="최적화 후 총 탄소배출량"
            )

        with col4:
            reduction = summary['total_reduction']
            st.metric(
                "총 감축량",
                f"{reduction:,.1f} kg",
                help="절대 감축량 (kgCO2eq)"
            )

        # 두 번째 줄
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "평균 감축률",
                f"{summary['average_reduction_pct']:.1f}%",
                help="자재별 평균 감축률"
            )

        with col2:
            st.metric(
                "평균 Tier1 RE",
                f"{summary['avg_tier1_re']:.1f}%",
                help="Tier1 RE 평균 적용률"
            )

        with col3:
            st.metric(
                "평균 재활용 비율",
                f"{summary['avg_recycle_ratio']:.1f}%",
                help="재활용재 평균 비율"
            )

        with col4:
            st.metric(
                "평균 저탄소 비율",
                f"{summary['avg_low_carbon_ratio']:.1f}%",
                help="저탄소메탈 평균 비율"
            )

    def _render_detail_table(self, result_df: pd.DataFrame) -> None:
        """상세 결과 테이블 렌더링"""
        st.subheader("📋 자재별 상세 결과")

        # 컬럼 선택
        display_columns = [
            '자재명',
            '제품총소요량(kg)',
            '원본_배출계수',
            '최적_배출계수',
            '감축률(%)',
            'Tier1_RE(%)',
            'Tier2_RE(%)',
            '재활용_비율(%)',
            '저탄소_비율(%)',
            '원본_배출량(kgCO2eq)',
            '최적_배출량(kgCO2eq)',
            '배출량_감축(kgCO2eq)'
        ]

        # 존재하는 컬럼만 선택
        available_columns = [col for col in display_columns if col in result_df.columns]

        # 데이터 편집기로 표시
        st.dataframe(
            result_df[available_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                '감축률(%)': st.column_config.ProgressColumn(
                    '감축률(%)',
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                ),
                '원본_배출량(kgCO2eq)': st.column_config.NumberColumn(
                    '원본_배출량(kgCO2eq)',
                    format="%.2f"
                ),
                '최적_배출량(kgCO2eq)': st.column_config.NumberColumn(
                    '최적_배출량(kgCO2eq)',
                    format="%.2f"
                ),
                '배출량_감축(kgCO2eq)': st.column_config.NumberColumn(
                    '배출량_감축(kgCO2eq)',
                    format="%.2f"
                )
            }
        )

        # CSV 다운로드
        csv = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 결과 다운로드 (CSV)",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    def _render_charts(self, result_df: pd.DataFrame, summary: Dict[str, Any], solution: Dict[str, Any]) -> None:
        """차트 렌더링"""
        st.subheader("📊 시각화")

        # 탭으로 구분
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "감축 분석",
            "RE 적용",
            "자재 구성",
            "원소별 비율",
            "비교 차트"
        ])

        with tab1:
            self._render_reduction_charts(result_df)

        with tab2:
            self._render_re_charts(result_df)

        with tab3:
            self._render_composition_charts(result_df)

        with tab4:
            self._render_element_ratio_charts(solution)

        with tab5:
            self._render_comparison_charts(result_df, summary)

    def _render_reduction_charts(self, result_df: pd.DataFrame) -> None:
        """감축 분석 차트"""
        st.markdown("### 📉 감축 분석")

        # 자재별 감축률 막대 차트
        fig = px.bar(
            result_df.sort_values('감축률(%)', ascending=True),
            x='감축률(%)',
            y='자재명',
            orientation='h',
            title='자재별 감축률',
            labels={'감축률(%)': '감축률 (%)', '자재명': '자재'},
            color='감축률(%)',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=max(400, len(result_df) * 30))
        st.plotly_chart(fig, use_container_width=True)

        # 배출량 감축 상위 10개
        top10 = result_df.nlargest(10, '배출량_감축(kgCO2eq)')
        fig = px.bar(
            top10,
            x='자재명',
            y='배출량_감축(kgCO2eq)',
            title='배출량 감축 상위 10개 자재',
            labels={'배출량_감축(kgCO2eq)': '감축량 (kgCO2eq)', '자재명': '자재'},
            color='배출량_감축(kgCO2eq)',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_re_charts(self, result_df: pd.DataFrame) -> None:
        """RE 적용 차트"""
        st.markdown("### ⚡ RE 적용 현황")

        # Tier1 vs Tier2 RE 산점도
        fig = px.scatter(
            result_df,
            x='Tier1_RE(%)',
            y='Tier2_RE(%)',
            size='제품총소요량(kg)',
            color='감축률(%)',
            hover_name='자재명',
            title='Tier1 vs Tier2 RE 적용률',
            labels={
                'Tier1_RE(%)': 'Tier1 RE (%)',
                'Tier2_RE(%)': 'Tier2 RE (%)',
                '감축률(%)': '감축률 (%)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        # RE 평균 비교
        avg_data = pd.DataFrame({
            'Tier': ['Tier1', 'Tier2'],
            'Average RE (%)': [
                result_df['Tier1_RE(%)'].mean(),
                result_df['Tier2_RE(%)'].mean()
            ]
        })

        fig = px.bar(
            avg_data,
            x='Tier',
            y='Average RE (%)',
            title='Tier별 평균 RE 적용률',
            color='Tier',
            text='Average RE (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    def _render_composition_charts(self, result_df: pd.DataFrame) -> None:
        """자재 구성 차트"""
        st.markdown("### 🔄 자재 구성")

        # 전체 평균 구성
        avg_composition = {
            '재활용': result_df['재활용_비율(%)'].mean(),
            '저탄소': result_df['저탄소_비율(%)'].mean(),
            '버진': result_df['버진_비율(%)'].mean()
        }

        fig = go.Figure(data=[go.Pie(
            labels=list(avg_composition.keys()),
            values=list(avg_composition.values()),
            hole=0.3,
            marker=dict(colors=['#2ecc71', '#3498db', '#95a5a6'])
        )])
        fig.update_layout(title='전체 평균 자재 구성')
        st.plotly_chart(fig, use_container_width=True)

        # 자재별 스택 바 차트
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='재활용',
            x=result_df['자재명'],
            y=result_df['재활용_비율(%)'],
            marker_color='#2ecc71'
        ))

        fig.add_trace(go.Bar(
            name='저탄소',
            x=result_df['자재명'],
            y=result_df['저탄소_비율(%)'],
            marker_color='#3498db'
        ))

        fig.add_trace(go.Bar(
            name='버진',
            x=result_df['자재명'],
            y=result_df['버진_비율(%)'],
            marker_color='#95a5a6'
        ))

        fig.update_layout(
            barmode='stack',
            title='자재별 구성 비율',
            xaxis_title='자재',
            yaxis_title='비율 (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_element_ratio_charts(self, solution: Dict[str, Any]) -> None:
        """원소별 비율 차트 (양극재 전용)"""
        st.markdown("### 🧪 원소별 비율 분석 (양극재)")

        # 양극재 원소 데이터 확인
        cathode_data = solution.get('cathode', {})
        elements_data = cathode_data.get('elements', {})

        if not elements_data:
            st.warning("⚠️ 양극재 원소별 데이터가 없습니다. 양극재가 포함된 시나리오에서만 사용 가능합니다.")
            return

        # 데이터 준비
        element_names = []
        virgin_ratios = []
        recycle_ratios = []
        low_carbon_ratios = []

        for element, ratios in elements_data.items():
            element_names.append(element)
            virgin_ratios.append(ratios.get('virgin_ratio', 0) * 100)
            recycle_ratios.append(ratios.get('recycle_ratio', 0) * 100)
            low_carbon_ratios.append(ratios.get('low_carbon_ratio', 0) * 100)

        # 1. 스택 바 차트 (원소별 구성)
        st.markdown("#### 원소별 자재 구성")
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='재활용',
            x=element_names,
            y=recycle_ratios,
            marker_color='#2ecc71',
            text=[f"{v:.1f}%" for v in recycle_ratios],
            textposition='inside'
        ))

        fig.add_trace(go.Bar(
            name='저탄소',
            x=element_names,
            y=low_carbon_ratios,
            marker_color='#3498db',
            text=[f"{v:.1f}%" for v in low_carbon_ratios],
            textposition='inside'
        ))

        fig.add_trace(go.Bar(
            name='버진',
            x=element_names,
            y=virgin_ratios,
            marker_color='#95a5a6',
            text=[f"{v:.1f}%" for v in virgin_ratios],
            textposition='inside'
        ))

        fig.update_layout(
            barmode='stack',
            title='원소별 자재 구성 비율',
            xaxis_title='원소',
            yaxis_title='비율 (%)',
            height=400,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)

        # 2. 재활용 vs 저탄소 산점도
        st.markdown("#### 재활용 vs 저탄소메탈 비율")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=recycle_ratios,
            y=low_carbon_ratios,
            mode='markers+text',
            marker=dict(size=20, color=['#e74c3c', '#3498db', '#2ecc71']),
            text=element_names,
            textposition='top center',
            textfont=dict(size=14, color='black')
        ))

        fig.update_layout(
            title='원소별 재활용 vs 저탄소 비율',
            xaxis_title='재활용 비율 (%)',
            yaxis_title='저탄소 비율 (%)',
            height=400,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100])
        )

        # 45도 선 추가 (재활용=저탄소 라인)
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        st.plotly_chart(fig, use_container_width=True)

        # 3. 상세 데이터 테이블
        st.markdown("#### 원소별 상세 비율")
        element_df = pd.DataFrame({
            '원소': element_names,
            '재활용 (%)': [f"{v:.2f}" for v in recycle_ratios],
            '저탄소 (%)': [f"{v:.2f}" for v in low_carbon_ratios],
            '버진 (%)': [f"{v:.2f}" for v in virgin_ratios],
            '배출계수': [f"{elements_data[e].get('emission_factor', 0):.4f}" for e in element_names]
        })

        st.dataframe(element_df, use_container_width=True, hide_index=True)

        # 제약조건 체크 안내
        st.info("""
        💡 **원소별 비율 제약조건 확인**

        Step 2 제약조건 설정에서 설정한 원소별 비율 범위가 올바르게 적용되었는지 확인할 수 있습니다.
        - 설정한 범위 내에 비율이 있는지 확인하세요
        - 범위를 벗어난 경우, 제약조건이 제대로 적용되지 않았을 수 있습니다
        """)

    def _render_comparison_charts(self, result_df: pd.DataFrame, summary: Dict[str, Any]) -> None:
        """비교 차트"""
        st.markdown("### 🔄 Before & After 비교")

        # 워터폴 차트
        fig = go.Figure(go.Waterfall(
            name="배출량 변화",
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["기준", "감축", "최적화"],
            textposition="outside",
            text=[
                f"{summary['baseline_total_emission']:.1f}",
                f"-{summary['total_reduction']:.1f}",
                f"{summary['optimized_total_emission']:.1f}"
            ],
            y=[
                summary['baseline_total_emission'],
                -summary['total_reduction'],
                summary['optimized_total_emission']
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="총 배출량 변화 (Waterfall)",
            showlegend=False,
            yaxis_title="배출량 (kgCO2eq)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Before/After 막대 비교
        comparison_data = pd.DataFrame({
            '자재명': result_df['자재명'],
            '기준': result_df['원본_배출량(kgCO2eq)'],
            '최적화': result_df['최적_배출량(kgCO2eq)']
        })

        # 상위 10개만 표시
        top10_comparison = comparison_data.nlargest(10, '기준')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='기준',
            x=top10_comparison['자재명'],
            y=top10_comparison['기준'],
            marker_color='#e74c3c'
        ))
        fig.add_trace(go.Bar(
            name='최적화',
            x=top10_comparison['자재명'],
            y=top10_comparison['최적화'],
            marker_color='#2ecc71'
        ))

        fig.update_layout(
            barmode='group',
            title='자재별 배출량 비교 (상위 10개)',
            xaxis_title='자재',
            yaxis_title='배출량 (kgCO2eq)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_compact(self, solution: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """간단한 결과 표시 (다른 탭에서 사용)"""
        status = solution.get('status', 'unknown')

        if 'optimal' in status.lower():
            st.success("✅ 최적화 성공!")
        else:
            st.warning(f"⚠️ 상태: {status}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("목적함수 값", f"{solution['objective_value']:.2f}")

        with col2:
            reduction = summary['total_reduction']
            st.metric("총 감축량", f"{reduction:.1f} kg")

        with col3:
            reduction_pct = summary['total_reduction_pct']
            st.metric("감축률", f"{reduction_pct:.1f}%")

"""
시나리오 비교 대시보드

여러 최적화 시나리오를 비교하고 분석하는 UI 컴포넌트입니다.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from datetime import datetime


class ComparisonDashboard:
    """시나리오 비교 대시보드 클래스"""

    def __init__(self):
        """초기화"""
        self.scenarios: Dict[str, Dict[str, Any]] = {}

    def add_scenario(
        self,
        name: str,
        result_df: pd.DataFrame,
        summary: Dict[str, Any],
        solution: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        비교할 시나리오 추가

        Args:
            name: 시나리오 이름
            result_df: 결과 DataFrame
            summary: 요약 통계
            solution: 원본 solution 딕셔너리
            timestamp: 생성 시간 (기본값: 현재 시간)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.scenarios[name] = {
            'result_df': result_df,
            'summary': summary,
            'solution': solution,
            'timestamp': timestamp
        }

    def render(self) -> None:
        """비교 대시보드 전체 렌더링"""
        st.header("🔄 시나리오 비교 대시보드")

        # 시나리오가 없는 경우
        if not self.scenarios:
            st.info("비교할 시나리오가 없습니다. 먼저 최적화를 실행하세요.")
            return

        # 시나리오가 1개만 있는 경우
        if len(self.scenarios) == 1:
            st.warning("비교하려면 최소 2개의 시나리오가 필요합니다.")
            scenario_name = list(self.scenarios.keys())[0]
            self._render_single_scenario_summary(scenario_name)
            return

        # 시나리오 선택
        selected_scenarios = self._render_scenario_selector()

        if len(selected_scenarios) < 2:
            st.warning("비교하려면 최소 2개의 시나리오를 선택하세요.")
            return

        st.markdown("---")

        # 비교 요약 카드
        self._render_comparison_summary(selected_scenarios)

        st.markdown("---")

        # 상세 비교 테이블
        self._render_comparison_table(selected_scenarios)

        st.markdown("---")

        # 비교 차트
        self._render_comparison_charts(selected_scenarios)

    def _render_scenario_selector(self) -> List[str]:
        """시나리오 선택 UI"""
        st.subheader("📋 시나리오 선택")

        col1, col2 = st.columns([3, 1])

        with col1:
            scenario_options = list(self.scenarios.keys())
            selected = st.multiselect(
                "비교할 시나리오를 선택하세요 (2개 이상)",
                options=scenario_options,
                default=scenario_options[:min(2, len(scenario_options))],
                help="최대 4개까지 선택 가능"
            )

        with col2:
            # 전체 선택/해제
            if st.button("🔘 전체 선택", use_container_width=True):
                selected = scenario_options

        # 선택된 시나리오 정보
        if selected:
            st.caption(f"선택됨: {len(selected)}개 시나리오")
            cols = st.columns(len(selected))
            for idx, scenario_name in enumerate(selected):
                with cols[idx]:
                    scenario = self.scenarios[scenario_name]
                    timestamp = scenario['timestamp'].strftime("%Y-%m-%d %H:%M")
                    status = scenario['solution'].get('status', 'unknown')
                    status_icon = "✅" if 'optimal' in status.lower() else "⚠️"
                    st.info(f"{status_icon} **{scenario_name}**\n\n{timestamp}")

        return selected[:4]  # 최대 4개 제한

    def _render_comparison_summary(self, selected_scenarios: List[str]) -> None:
        """비교 요약 카드"""
        st.subheader("📊 비교 요약")

        # 베이스라인 선택 (첫 번째 시나리오)
        baseline_name = selected_scenarios[0]
        baseline_summary = self.scenarios[baseline_name]['summary']

        # 메트릭 비교
        num_scenarios = len(selected_scenarios)
        cols = st.columns(num_scenarios)

        for idx, scenario_name in enumerate(selected_scenarios):
            scenario = self.scenarios[scenario_name]
            summary = scenario['summary']

            with cols[idx]:
                st.markdown(f"**{scenario_name}**")

                # 총 배출량
                optimized = summary['optimized_total_emission']
                if idx == 0:
                    st.metric(
                        "총 배출량",
                        f"{optimized:,.1f} kg",
                        help="최적화 후 총 배출량"
                    )
                else:
                    delta = optimized - baseline_summary['optimized_total_emission']
                    delta_pct = (delta / baseline_summary['optimized_total_emission'] * 100)
                    st.metric(
                        "총 배출량",
                        f"{optimized:,.1f} kg",
                        f"{delta:+.1f} kg ({delta_pct:+.1f}%)",
                        delta_color="inverse",
                        help=f"{baseline_name} 대비"
                    )

                # 감축률
                reduction_pct = summary['total_reduction_pct']
                if idx == 0:
                    st.metric(
                        "감축률",
                        f"{reduction_pct:.1f}%",
                        help="기준 대비 감축률"
                    )
                else:
                    delta_pct = reduction_pct - baseline_summary['total_reduction_pct']
                    st.metric(
                        "감축률",
                        f"{reduction_pct:.1f}%",
                        f"{delta_pct:+.1f}%p",
                        help=f"{baseline_name} 대비"
                    )

                # 평균 Tier1 RE
                avg_tier1_re = summary['avg_tier1_re']
                st.metric(
                    "평균 Tier1 RE",
                    f"{avg_tier1_re:.1f}%"
                )

                # 평균 재활용 비율
                avg_recycle = summary['avg_recycle_ratio']
                st.metric(
                    "평균 재활용",
                    f"{avg_recycle:.1f}%"
                )

    def _render_comparison_table(self, selected_scenarios: List[str]) -> None:
        """상세 비교 테이블"""
        st.subheader("📋 자재별 상세 비교")

        # 모든 자재 목록 수집
        all_materials = set()
        for scenario_name in selected_scenarios:
            result_df = self.scenarios[scenario_name]['result_df']
            all_materials.update(result_df['자재명'].tolist())

        all_materials = sorted(list(all_materials))

        # 비교 메트릭 선택
        metric_options = {
            '최적_배출량(kgCO2eq)': '최적 배출량',
            '감축률(%)': '감축률',
            'Tier1_RE(%)': 'Tier1 RE',
            '재활용_비율(%)': '재활용 비율',
            '저탄소_비율(%)': '저탄소 비율'
        }

        selected_metric = st.selectbox(
            "비교 메트릭 선택",
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )

        # 비교 데이터 생성
        comparison_data = {'자재명': all_materials}

        for scenario_name in selected_scenarios:
            result_df = self.scenarios[scenario_name]['result_df']

            # 각 자재에 대한 값 매핑
            material_values = {}
            for _, row in result_df.iterrows():
                material_values[row['자재명']] = row.get(selected_metric, 0)

            comparison_data[scenario_name] = [
                material_values.get(mat, 0) for mat in all_materials
            ]

        comparison_df = pd.DataFrame(comparison_data)

        # 델타 계산 (첫 번째 시나리오 대비)
        if len(selected_scenarios) > 1:
            baseline_col = selected_scenarios[0]
            for scenario_name in selected_scenarios[1:]:
                delta_col = f"Δ {scenario_name}"
                comparison_df[delta_col] = (
                    comparison_df[scenario_name] - comparison_df[baseline_col]
                )

        # 테이블 표시
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                selected_metric: st.column_config.NumberColumn(
                    format="%.2f"
                )
            }
        )

        # CSV 다운로드
        csv = comparison_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 비교 결과 다운로드 (CSV)",
            data=csv,
            file_name=f"scenario_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    def _render_comparison_charts(self, selected_scenarios: List[str]) -> None:
        """비교 차트"""
        st.subheader("📊 비교 시각화")

        tab1, tab2, tab3 = st.tabs([
            "배출량 비교",
            "감축 성과 비교",
            "자재 구성 비교"
        ])

        with tab1:
            self._render_emission_comparison(selected_scenarios)

        with tab2:
            self._render_reduction_comparison(selected_scenarios)

        with tab3:
            self._render_composition_comparison(selected_scenarios)

    def _render_emission_comparison(self, selected_scenarios: List[str]) -> None:
        """배출량 비교 차트"""
        st.markdown("### 📉 총 배출량 비교")

        # 데이터 준비
        scenario_names = []
        baseline_emissions = []
        optimized_emissions = []

        for scenario_name in selected_scenarios:
            summary = self.scenarios[scenario_name]['summary']
            scenario_names.append(scenario_name)
            baseline_emissions.append(summary['baseline_total_emission'])
            optimized_emissions.append(summary['optimized_total_emission'])

        # 그룹 막대 차트
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='기준',
            x=scenario_names,
            y=baseline_emissions,
            marker_color='#e74c3c',
            text=[f"{val:,.0f}" for val in baseline_emissions],
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='최적화',
            x=scenario_names,
            y=optimized_emissions,
            marker_color='#2ecc71',
            text=[f"{val:,.0f}" for val in optimized_emissions],
            textposition='outside'
        ))

        fig.update_layout(
            barmode='group',
            title='시나리오별 총 배출량 비교',
            xaxis_title='시나리오',
            yaxis_title='배출량 (kgCO2eq)',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # 감축량 비교
        st.markdown("### 📊 절대 감축량 비교")

        reductions = [
            summary['total_reduction']
            for summary in [self.scenarios[name]['summary'] for name in selected_scenarios]
        ]

        fig = go.Figure(go.Bar(
            x=scenario_names,
            y=reductions,
            marker_color='#3498db',
            text=[f"{val:,.0f}" for val in reductions],
            textposition='outside'
        ))

        fig.update_layout(
            title='시나리오별 절대 감축량',
            xaxis_title='시나리오',
            yaxis_title='감축량 (kgCO2eq)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_reduction_comparison(self, selected_scenarios: List[str]) -> None:
        """감축 성과 비교 차트"""
        st.markdown("### 📈 감축률 비교")

        # 데이터 준비
        scenario_names = []
        reduction_pcts = []
        avg_tier1_res = []
        avg_recycle_ratios = []
        avg_low_carbon_ratios = []

        for scenario_name in selected_scenarios:
            summary = self.scenarios[scenario_name]['summary']
            scenario_names.append(scenario_name)
            reduction_pcts.append(summary['total_reduction_pct'])
            avg_tier1_res.append(summary['avg_tier1_re'])
            avg_recycle_ratios.append(summary['avg_recycle_ratio'])
            avg_low_carbon_ratios.append(summary['avg_low_carbon_ratio'])

        # 감축률 비교
        fig = go.Figure(go.Bar(
            x=scenario_names,
            y=reduction_pcts,
            marker_color='#9b59b6',
            text=[f"{val:.1f}%" for val in reduction_pcts],
            textposition='outside'
        ))

        fig.update_layout(
            title='시나리오별 총 감축률',
            xaxis_title='시나리오',
            yaxis_title='감축률 (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # RE 및 자재 구성 비교
        st.markdown("### ⚡ RE 및 자재 구성 비교")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='평균 Tier1 RE',
            x=scenario_names,
            y=avg_tier1_res,
            marker_color='#f39c12'
        ))

        fig.add_trace(go.Bar(
            name='평균 재활용 비율',
            x=scenario_names,
            y=avg_recycle_ratios,
            marker_color='#2ecc71'
        ))

        fig.add_trace(go.Bar(
            name='평균 저탄소 비율',
            x=scenario_names,
            y=avg_low_carbon_ratios,
            marker_color='#3498db'
        ))

        fig.update_layout(
            barmode='group',
            title='시나리오별 평균 구성 비율',
            xaxis_title='시나리오',
            yaxis_title='비율 (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_composition_comparison(self, selected_scenarios: List[str]) -> None:
        """자재 구성 비교 차트"""
        st.markdown("### 🔄 자재 구성 비교")

        # 각 시나리오별 파이 차트
        num_scenarios = len(selected_scenarios)
        cols = st.columns(min(num_scenarios, 3))

        for idx, scenario_name in enumerate(selected_scenarios):
            result_df = self.scenarios[scenario_name]['result_df']

            avg_composition = {
                '재활용': result_df['재활용_비율(%)'].mean(),
                '저탄소': result_df['저탄소_비율(%)'].mean(),
                '버진': result_df['버진_비율(%)'].mean()
            }

            col_idx = idx % 3
            with cols[col_idx]:
                fig = go.Figure(data=[go.Pie(
                    labels=list(avg_composition.keys()),
                    values=list(avg_composition.values()),
                    hole=0.3,
                    marker=dict(colors=['#2ecc71', '#3498db', '#95a5a6'])
                )])

                fig.update_layout(
                    title=scenario_name,
                    height=300,
                    showlegend=True,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)

    def _render_single_scenario_summary(self, scenario_name: str) -> None:
        """단일 시나리오 요약 (비교 불가 시)"""
        st.info(f"현재 시나리오: **{scenario_name}**")

        scenario = self.scenarios[scenario_name]
        summary = scenario['summary']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "총 배출량",
                f"{summary['optimized_total_emission']:,.1f} kg"
            )

        with col2:
            st.metric(
                "감축률",
                f"{summary['total_reduction_pct']:.1f}%"
            )

        with col3:
            st.metric(
                "평균 Tier1 RE",
                f"{summary['avg_tier1_re']:.1f}%"
            )

        with col4:
            st.metric(
                "평균 재활용",
                f"{summary['avg_recycle_ratio']:.1f}%"
            )

    def clear_scenarios(self) -> None:
        """모든 시나리오 삭제"""
        self.scenarios = {}

    def remove_scenario(self, name: str) -> bool:
        """
        특정 시나리오 삭제

        Args:
            name: 삭제할 시나리오 이름

        Returns:
            bool: 삭제 성공 여부
        """
        if name in self.scenarios:
            del self.scenarios[name]
            return True
        return False

    def get_scenario_names(self) -> List[str]:
        """시나리오 이름 목록 반환"""
        return list(self.scenarios.keys())

    def export_all_scenarios(self) -> pd.DataFrame:
        """
        모든 시나리오를 하나의 DataFrame으로 내보내기

        Returns:
            pd.DataFrame: 통합 비교 데이터
        """
        if not self.scenarios:
            return pd.DataFrame()

        all_data = []

        for scenario_name, scenario in self.scenarios.items():
            result_df = scenario['result_df'].copy()
            result_df['시나리오'] = scenario_name
            result_df['생성시간'] = scenario['timestamp']
            all_data.append(result_df)

        return pd.concat(all_data, ignore_index=True)

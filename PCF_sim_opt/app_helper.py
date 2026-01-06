import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.utils.file_operations import FileOperations, FileLoadError, FileSaveError

def create_basic_scenarios_visualizations(basic_df: pd.DataFrame) -> Dict[str, go.Figure]:
    # 컬럼명 매핑 추가
    rename_dict = {
        '총_배출량_kg_CO2e': 'PCF (kgCO2eq)',
        '감축량_kg_CO2e': '감축량 (kgCO2eq)',
        '감축률_퍼센트': '감축률 (%)',
        '원재료_기여도_퍼센트': '원재료 기여도 (%)',
        '재활용_비율_퍼센트': '재활용 비율 (%)',
        'Energy_Tier1_전력_기여도_퍼센트': 'Energy_Tier1_기여도 (%)',
        'Energy_Tier2_전력_기여도_퍼센트': 'Energy_Tier2_기여도 (%)',
    }
    basic_df = basic_df.rename(columns=rename_dict)
    visualizations = {}
    
    if basic_df.empty:
        st.warning("시각화할 데이터가 없습니다.")
        return visualizations
    
    # 1. 시나리오별 탄소배출량 비교 (원재료/에너지 기여도 색상)
    fig1 = scenario_emission_bar_with_contribution(basic_df)
    visualizations['scenario_emission_bar'] = fig1
    
    # 2. 감축량(bar) + 감축률(line) 복합 plot
    fig2 = reduction_bar_line_plot(basic_df)
    visualizations['reduction_bar_line'] = fig2
    
    # 3. 효율성 분석 (감축량 vs 감축률 산점도)
    fig3 = efficiency_scatter_plot(basic_df)
    visualizations['efficiency_scatter'] = fig3
    
    
    return visualizations


def scenario_emission_bar_with_contribution(df: pd.DataFrame):
    """
    시나리오별 탄소배출량(bar), bar 색상은 원재료/에너지 기여도(스택)로 표시
    """
    import plotly.graph_objects as go
    scenarios = df['시나리오']
    total = df['PCF (kgCO2eq)']
    raw = df['원재료 기여도 (%)'] / 100
    energy = 1 - raw
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scenarios,
        y=total * raw,
        name='원재료 기여도',
        marker_color='#1f77b4',
        hovertemplate='원재료: %{y:.2f} kgCO2eq<br>'
    ))
    fig.add_trace(go.Bar(
        x=scenarios,
        y=total * energy,
        name='에너지 기여도',
        marker_color='#ff7f0e',
        hovertemplate='에너지: %{y:.2f} kgCO2eq<br>'
    ))
    fig.update_layout(
        barmode='stack',
        title={'text': '시나리오별 탄소배출량 비교 (원재료/에너지 기여도)', 'x':0.5, 'font': {'size': 18, 'color': '#fff'}},
        xaxis_title='시나리오',
        yaxis_title='탄소배출량 (kgCO2eq)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fff'}
    )
    return fig

def reduction_bar_line_plot(df: pd.DataFrame):
    """
    감축량(bar) + 감축률(line) 복합 plot
    """
    import plotly.graph_objects as go
    scenarios = df['시나리오']
    reduction = df['감축량 (kgCO2eq)']
    rate = df['감축률 (%)']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scenarios,
        y=reduction,
        name='감축량',
        marker_color='#2ca02c',
        yaxis='y1',
        hovertemplate='감축량: %{y:.2f} kgCO2eq<br>'
    ))
    fig.add_trace(go.Scatter(
        x=scenarios,
        y=rate,
        name='감축률',
        mode='lines+markers+text',
        marker_color='#d62728',
        yaxis='y2',
        text=[f'{v:.1f}%' for v in rate],
        textposition='top center',
        textfont=dict(color='black', size=12),
        hovertemplate='감축률: %{y:.2f}%<br>'
    ))
    fig.update_layout(
        title={'text': '시나리오별 감축량/감축률', 'x':0.5, 'font': {'size': 18, 'color': '#fff'}},
        xaxis_title='시나리오',
        yaxis=dict(title='감축량 (kgCO2eq)', showgrid=True, gridcolor='#333'),
        yaxis2=dict(title='감축률 (%)', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fff'}
    )
    return fig

def efficiency_scatter_plot(df: pd.DataFrame):
    """
    감축량(x) vs 감축률(y) 산점도
    """
    import plotly.graph_objects as go
    non_baseline = df[df['시나리오'] != 'Baseline']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=non_baseline['감축량 (kgCO2eq)'],
        y=non_baseline['감축률 (%)'],
        mode='markers+text',
        text=non_baseline['시나리오'],
        textposition='top center',
        textfont=dict(color='black'),  # 텍스트 색상을 검은색으로 변경
        marker=dict(size=14, color='#9467bd'),
        name='효율성',
        hovertemplate='감축량: %{x:.2f} kgCO2eq<br>감축률: %{y:.2f}%<br>'
    ))
    fig.update_layout(
        title={'text': '시나리오별 효율성 분석 (감축량 vs 감축률)', 'x':0.5, 'font': {'size': 18, 'color': '#fff'}},
        xaxis_title='감축량 (kgCO2eq)',
        yaxis_title='감축률 (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fff'},
        showlegend=False
    )
    return fig

def separation_analysis_plot(df: pd.DataFrame):
    """
    분리 시나리오 분석 (재활용 only vs 저탄소메탈 only vs 동시적용) 비교 plot
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # 재활용 only, 저탄소메탈 only, 동시적용 시나리오 필터링
    recycling_only = df[df['시나리오'].str.contains('^재활용 적용$', na=False)]
    
    low_carb_only = df[df['시나리오'].str.contains('^저탄소메탈 적용$', na=False)]
    
    combined = df[df['시나리오'].str.contains('동시.*적용', na=False)]
    
    # 데이터가 충분하지 않으면 None 반환
    if recycling_only.empty and low_carb_only.empty and combined.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('감축량 비교', '감축률 비교', '배출량 비교', '기여도 비교'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    scenarios = []
    reductions = []
    reduction_rates = []
    emissions = []
    material_contributions = []
    
    # 재활용 only 데이터
    if not recycling_only.empty:
        row = recycling_only.iloc[0]
        scenarios.append('재활용 Only')
        reductions.append(row.get('감축량_kg_CO2e', 0))
        reduction_rates.append(row.get('감축률_퍼센트', 0))
        emissions.append(row.get('총_배출량_kg_CO2e', 0))
        material_contributions.append(row.get('원재료_기여도_퍼센트', 0))
    
    # 저탄소메탈 only 데이터
    if not low_carb_only.empty:
        row = low_carb_only.iloc[0]
        scenarios.append('저탄소메탈 Only')
        reductions.append(row.get('감축량_kg_CO2e', 0))
        reduction_rates.append(row.get('감축률_퍼센트', 0))
        emissions.append(row.get('총_배출량_kg_CO2e', 0))
        material_contributions.append(row.get('원재료_기여도_퍼센트', 0))
    
    # 동시적용 데이터
    if not combined.empty:
        row = combined.iloc[0]
        scenarios.append('동시 적용')
        reductions.append(row.get('감축량_kg_CO2e', 0))
        reduction_rates.append(row.get('감축률_퍼센트', 0))
        emissions.append(row.get('총_배출량_kg_CO2e', 0))
        material_contributions.append(row.get('원재료_기여도_퍼센트', 0))
    
    if not scenarios:
        return None
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(scenarios)]
    
    # 1. 감축량 비교 (좌상)
    fig.add_trace(
        go.Bar(x=scenarios, y=reductions, name='감축량', marker_color=colors,
               hovertemplate='%{y:.3f} kgCO2eq<br>'),
        row=1, col=1
    )
    
    # 2. 감축률 비교 (우상)
    fig.add_trace(
        go.Bar(x=scenarios, y=reduction_rates, name='감축률', marker_color=colors,
               hovertemplate='%{y:.2f}%<br>'),
        row=1, col=2
    )
    
    # 3. 총 배출량 비교 (좌하)
    fig.add_trace(
        go.Bar(x=scenarios, y=emissions, name='총 배출량', marker_color=colors,
               hovertemplate='%{y:.3f} kgCO2eq<br>'),
        row=2, col=1
    )
    
    # 4. 원재료 기여도 비교 (우하)
    fig.add_trace(
        go.Bar(x=scenarios, y=material_contributions, name='원재료 기여도', marker_color=colors,
               hovertemplate='%{y:.2f}%<br>'),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': '분리 시나리오 효과 비교 분석 (재활용 vs 저탄소메탈 vs 동시적용)', 
            'x': 0.5, 
            'font': {'size': 18, 'color': '#fff'}
        },
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fff'},
        height=600
    )
    
    # y축 레이블 설정
    fig.update_yaxes(title_text="감축량 (kgCO2eq)", row=1, col=1)
    fig.update_yaxes(title_text="감축률 (%)", row=1, col=2)
    fig.update_yaxes(title_text="총 배출량 (kgCO2eq)", row=2, col=1)
    fig.update_yaxes(title_text="원재료 기여도 (%)", row=2, col=2)
    
    return fig

def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 시나리오 데이터에 대한 요약 통계를 생성합니다.
    
    Args:
        df: 기본 시나리오 데이터프레임
        
    Returns:
        pd.DataFrame: 요약 통계 데이터프레임
    """
    if df.empty:
        return pd.DataFrame()
    
    summary_data = []
    
    # 시나리오별 통계
    if '시나리오' in df.columns:
        for scenario in df['시나리오'].unique():
            scenario_data = df[df['시나리오'] == scenario]
            
            summary_row = {'시나리오': scenario}
            
            # PCF 통계
            if 'PCF (kgCO2eq)' in df.columns:
                pcf_values = scenario_data['PCF (kgCO2eq)'].dropna()
                if len(pcf_values) > 0:
                    summary_row.update({
                        'PCF_평균': pcf_values.mean(),
                        'PCF_최대': pcf_values.max(),
                        'PCF_최소': pcf_values.min(),
                        'PCF_표준편차': pcf_values.std()
                    })
            
            # 감축률 통계
            if '감축률 (%)' in df.columns:
                reduction_values = scenario_data['감축률 (%)'].dropna()
                if len(reduction_values) > 0:
                    summary_row.update({
                        '감축률_평균': reduction_values.mean(),
                        '감축률_최대': reduction_values.max(),
                        '감축률_최소': reduction_values.min()
                    })
            
            # 감축량 통계
            if '감축량 (kgCO2eq)' in df.columns:
                reduction_amount_values = scenario_data['감축량 (kgCO2eq)'].dropna()
                if len(reduction_amount_values) > 0:
                    summary_row.update({
                        '감축량_평균': reduction_amount_values.mean(),
                        '감축량_최대': reduction_amount_values.max(),
                        '감축량_최소': reduction_amount_values.min()
                    })
            
            summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)

# RuleBasedSimulationHelper의 Streamlit용 출력 함수들
def display_simulation_overview(result_df: pd.DataFrame, scenario: str):
    """
    시뮬레이션 결과 전체 개요를 Streamlit에 표시
    """
    # PCF 관련 열들 확인
    pcf_columns = [col for col in result_df.columns if col.startswith('PCF_')]
    has_reference = 'PCF_reference' in result_df.columns
    
    # 매칭 관련 열들 확인
    has_formula_matching = 'formula_matched' in result_df.columns
    has_proportions_matching = 'proportions_matched' in result_df.columns
    
    # 저감활동 관련 열 확인
    has_reduction_activity = '저감활동_적용여부' in result_df.columns
    
    # 수정된 배출계수 열들 확인
    modified_coeff_columns = [col for col in result_df.columns if col.startswith('modified_coeff_case')]
    
    st.markdown("### 🚀 RuleBasedSimulation 결과 분석")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📋 시나리오", scenario.upper())
        st.metric("📊 총 데이터 행 수", f"{len(result_df):,}개")
        st.metric("📈 PCF 관련 열", f"{len(pcf_columns)}개")
    
    with col2:
        st.metric("🔍 매칭 분석 가능", "✅" if has_formula_matching and has_proportions_matching else "❌")
        st.metric("⚡ 저감활동 분석 가능", "✅" if has_reduction_activity else "❌")
        st.metric("📊 수정된 배출계수 열", f"{len(modified_coeff_columns)}개")

def display_pcf_analysis(result_df: pd.DataFrame):
    """
    PCF 탄소배출량 분석 결과를 Streamlit에 표시
    """
    st.markdown("### 📊 PCF 탄소배출량 분석")
    st.markdown("---")
    
    # PCF 관련 열들 확인
    pcf_columns = [col for col in result_df.columns if col.startswith('PCF_')]
    has_reference = 'PCF_reference' in result_df.columns
    
    if not pcf_columns:
        st.warning("❌ PCF 관련 열이 없습니다.")
        return
    
    # PCF 합계 계산
    pcf_sums = {}
    for col in pcf_columns:
        pcf_sums[col] = result_df[col].sum()
    
    # Reference 값 먼저 출력
    if has_reference:
        reference_total = pcf_sums['PCF_reference']
        st.metric("📈 PCF_reference", f"{reference_total:.3f} kgCO2eq")
        st.markdown("---")
        
        # Case별 감소율 계산 및 출력
        for col in pcf_columns:
            if col != 'PCF_reference':
                case_total = pcf_sums[col]
                reduction = reference_total - case_total
                reduction_rate = (reduction / reference_total) * 100 if reference_total > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"📉 {col}", f"{case_total:.3f} kgCO2eq")
                with col2:
                    st.metric("절대 감소량", f"{reduction:.3f} kgCO2eq")
                with col3:
                    st.metric("감소율", f"{reduction_rate:.2f}%")
                st.markdown("---")
    else:
        # Reference가 없는 경우 단순 합계만 출력
        for col, total in pcf_sums.items():
            st.metric(f"📊 {col}", f"{total:.3f} kgCO2eq")

def display_matching_analysis(result_df: pd.DataFrame):
    """
    매칭 결과 분석을 Streamlit에 표시
    """
    # 매칭 관련 열들 확인
    has_formula_matching = 'formula_matched' in result_df.columns
    has_proportions_matching = 'proportions_matched' in result_df.columns
    
    if not (has_formula_matching and has_proportions_matching):
        st.warning("❌ 매칭 분석을 위한 열이 없습니다.")
        return
    
    st.markdown("### 🔍 매칭 결과 분석")
    st.markdown("---")
    
    # 매칭 통계 계산
    formula_matched_count = len(result_df[result_df['formula_matched'] == True])
    proportions_matched_count = len(result_df[result_df['proportions_matched'] == True])
    both_matched_count = len(result_df[
        (result_df['formula_matched'] == True) & 
        (result_df['proportions_matched'] == True)
    ])
    unmatched_count = len(result_df[
        (result_df['formula_matched'] == False) & 
        (result_df['proportions_matched'] == False)
    ])
    
    total_rows = len(result_df)
    
    # 매칭 통계 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 총 데이터", f"{total_rows:,}개")
    with col2:
        st.metric("📊 공식 매칭", f"{formula_matched_count:,}개", f"{formula_matched_count/total_rows*100:.1f}%")
    with col3:
        st.metric("📊 비율 매칭", f"{proportions_matched_count:,}개", f"{proportions_matched_count/total_rows*100:.1f}%")
    with col4:
        st.metric("📊 매칭 실패", f"{unmatched_count:,}개", f"{unmatched_count/total_rows*100:.1f}%")
    
    # PCF 관점에서의 매칭 분석
    pcf_columns = [col for col in result_df.columns if col.startswith('PCF_')]
    has_reference = 'PCF_reference' in result_df.columns
    
    if has_reference:
        st.markdown("### 📈 PCF 관점에서의 매칭 분석")
        st.markdown("---")
        
        # 공식 매칭만 된 행들의 PCF 분석
        formula_only = result_df[
            (result_df['formula_matched'] == True) & 
            (result_df['proportions_matched'] == False)
        ]
        if len(formula_only) > 0:
            st.markdown("#### 🔍 공식 매칭만")
            ref_sum = formula_only['PCF_reference'].sum()
            st.metric("행 수", f"{len(formula_only)}개")
            st.metric("PCF_reference", f"{ref_sum:.3f} kgCO2eq")
            
            for col in pcf_columns:
                if col != 'PCF_reference':
                    case_sum = formula_only[col].sum()
                    reduction = ref_sum - case_sum
                    reduction_rate = (reduction / ref_sum * 100) if ref_sum > 0 else 0
                    st.metric(f"{col}", f"{case_sum:.3f} kgCO2eq", f"감소율: {reduction_rate:.2f}%")
        
        # 비율 매칭만 된 행들의 PCF 분석
        proportions_only = result_df[
            (result_df['formula_matched'] == False) & 
            (result_df['proportions_matched'] == True)
        ]
        if len(proportions_only) > 0:
            st.markdown("#### 🔍 비율 매칭만")
            ref_sum = proportions_only['PCF_reference'].sum()
            st.metric("행 수", f"{len(proportions_only)}개")
            st.metric("PCF_reference", f"{ref_sum:.3f} kgCO2eq")
            
            for col in pcf_columns:
                if col != 'PCF_reference':
                    case_sum = proportions_only[col].sum()
                    reduction = ref_sum - case_sum
                    reduction_rate = (reduction / ref_sum * 100) if ref_sum > 0 else 0
                    st.metric(f"{col}", f"{case_sum:.3f} kgCO2eq", f"감소율: {reduction_rate:.2f}%")

def display_reduction_activity_analysis(result_df: pd.DataFrame):
    """
    저감활동 적용 분석을 Streamlit에 표시
    """
    # 저감활동 관련 열 확인
    has_reduction_activity = '저감활동_적용여부' in result_df.columns
    
    if not has_reduction_activity:
        st.warning("❌ 저감활동 분석을 위한 열이 없습니다.")
        return
    
    st.markdown("### ⚡ 저감활동 적용 분석")
    st.markdown("---")
    
    # 저감활동 통계
    applicable_count = len(result_df[result_df['저감활동_적용여부'] == 1.0])
    non_applicable_count = len(result_df[result_df['저감활동_적용여부'] == 0.0])
    total_rows = len(result_df)
    
    # 저감활동 통계 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 총 데이터", f"{total_rows:,}개")
    with col2:
        st.metric("✅ 저감활동 적용", f"{applicable_count:,}개", f"{applicable_count/total_rows*100:.1f}%")
    with col3:
        st.metric("❌ 저감활동 미적용", f"{non_applicable_count:,}개", f"{non_applicable_count/total_rows*100:.1f}%")
    
    # PCF 관점에서의 저감활동 분석
    pcf_columns = [col for col in result_df.columns if col.startswith('PCF_')]
    has_reference = 'PCF_reference' in result_df.columns
    
    if has_reference:
        st.markdown("### 📈 PCF 관점에서의 저감활동 분석")
        st.markdown("---")
        
        # 저감활동 적용된 행들의 PCF 분석
        applicable_rows = result_df[result_df['저감활동_적용여부'] == 1.0]
        if len(applicable_rows) > 0:
            st.markdown("#### ✅ 저감활동 적용")
            ref_sum = applicable_rows['PCF_reference'].sum()
            st.metric("행 수", f"{len(applicable_rows)}개")
            st.metric("PCF_reference", f"{ref_sum:.3f} kgCO2eq")
            
            for col in pcf_columns:
                if col != 'PCF_reference':
                    case_sum = applicable_rows[col].sum()
                    reduction = ref_sum - case_sum
                    reduction_rate = (reduction / ref_sum * 100) if ref_sum > 0 else 0
                    st.metric(f"{col}", f"{case_sum:.3f} kgCO2eq", f"감소율: {reduction_rate:.2f}%")

def display_material_analysis(result_df: pd.DataFrame):
    """
    자재별 분석을 Streamlit에 표시
    """
    st.markdown("### 📋 자재별 분석")
    st.markdown("---")
    
    # 자재품목별 분석
    if '자재품목' in result_df.columns:
        st.markdown("#### 📊 자재품목별 분포")
        material_counts = result_df['자재품목'].value_counts()
        
        # 자재품목별 통계 표시
        for material, count in material_counts.items():
            material_data = result_df[result_df['자재품목'] == material]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"📦 {material}", f"{count}개")
            
            # PCF 분석 (PCF 열이 있는 경우)
            pcf_columns = [col for col in result_df.columns if col.startswith('PCF_')]
            if pcf_columns and 'PCF_reference' in pcf_columns:
                ref_sum = material_data['PCF_reference'].sum()
                with col2:
                    st.metric("PCF_reference", f"{ref_sum:.3f} kgCO2eq")
                
                # 가장 큰 감소율을 가진 case 찾기
                max_reduction_rate = 0
                best_case = None
                for col in pcf_columns:
                    if col != 'PCF_reference':
                        case_sum = material_data[col].sum()
                        reduction_rate = ((ref_sum - case_sum) / ref_sum * 100) if ref_sum > 0 else 0
                        if reduction_rate > max_reduction_rate:
                            max_reduction_rate = reduction_rate
                            best_case = col
                
                with col3:
                    if best_case:
                        st.metric("최대 감소율", f"{max_reduction_rate:.2f}%", f"({best_case})")
            
            st.markdown("---")
    
    # 자재명별 상위 기여자 분석
    if '자재명' in result_df.columns and 'PCF_reference' in result_df.columns:
        st.markdown("#### 🏆 자재별 PCF 기여도 (상위 10개)")
        
        # PCF_reference 기준으로 상위 10개 자재
        top_contributors = result_df.nlargest(10, 'PCF_reference')[['자재명', '자재품목', 'PCF_reference']]
        
        for idx, row in top_contributors.iterrows():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{row['자재명']}**")
            with col2:
                st.write(f"({row['자재품목']})")
            with col3:
                st.metric("PCF", f"{row['PCF_reference']:.3f} kgCO2eq")

def display_comprehensive_analysis(result_df: pd.DataFrame, scenario: str):
    """
    종합 분석을 Streamlit에 표시
    """
    st.markdown("### 📊 종합 분석")
    st.markdown("---")
    
    # 기본 통계
    st.markdown("#### 📈 기본 통계")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 총 데이터", f"{len(result_df):,}개")
    with col2:
        st.metric("📋 시나리오", scenario.upper())
    with col3:
        if 'PCF_reference' in result_df.columns:
            total_pcf = result_df['PCF_reference'].sum()
            st.metric("📈 총 PCF", f"{total_pcf:.3f} kgCO2eq")
    
    # PCF 요약 통계
    pcf_columns = [col for col in result_df.columns if col.startswith('PCF_')]
    if pcf_columns:
        st.markdown("#### 📊 PCF 요약 통계")
        
        pcf_sums = {}
        for col in pcf_columns:
            pcf_sums[col] = result_df[col].sum()
        
        # PCF 통계를 데이터프레임으로 표시
        pcf_stats = []
        for col, total in pcf_sums.items():
            pcf_stats.append({
                'PCF 열': col,
                '총합 (kgCO2eq)': f"{total:.3f}",
                '평균 (kgCO2eq)': f"{result_df[col].mean():.3f}",
                '최대 (kgCO2eq)': f"{result_df[col].max():.3f}",
                '최소 (kgCO2eq)': f"{result_df[col].min():.3f}"
            })
        
        pcf_stats_df = pd.DataFrame(pcf_stats)
        st.dataframe(pcf_stats_df, use_container_width=True)
    
    # 감소율 분석
    if 'PCF_reference' in result_df.columns:
        st.markdown("#### 📉 감소율 분석")
        
        reduction_stats = []
        reference_total = result_df['PCF_reference'].sum()
        
        for col in pcf_columns:
            if col != 'PCF_reference':
                case_total = result_df[col].sum()
                reduction = reference_total - case_total
                reduction_rate = (reduction / reference_total) * 100 if reference_total > 0 else 0
                
                reduction_stats.append({
                    'Case': col,
                    '총 PCF (kgCO2eq)': f"{case_total:.3f}",
                    '절대 감소량 (kgCO2eq)': f"{reduction:.3f}",
                    '감소율 (%)': f"{reduction_rate:.2f}%"
                })
        
        if reduction_stats:
            reduction_df = pd.DataFrame(reduction_stats)
            st.dataframe(reduction_df, use_container_width=True)
    
    # 시나리오별 특별 분석
    st.markdown("#### 🎯 시나리오별 특별 분석")
    
    if scenario in ['recycling', 'site_change', 'both']:
        st.info(f"📋 {scenario.upper()} 시나리오 특별 분석")
        
        # 적용된 자재 종류
        if '자재품목' in result_df.columns:
            material_categories = result_df['자재품목'].value_counts()
            st.write(f"📋 적용된 자재품목: {len(material_categories)}종")
            
            # 자재품목별 분포 차트
            fig = px.pie(
                values=material_categories.values,
                names=material_categories.index,
                title=f"{scenario.upper()} 시나리오 자재품목별 분포"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#fff'}
            )
            st.plotly_chart(fig, use_container_width=True)

def validate_brm_file(uploaded_file) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    BRM 파일(CSV 또는 Excel)의 유효성을 검사합니다.

    Args:
        uploaded_file: Streamlit의 uploaded file object

    Returns:
        Tuple[bool, List[str], pd.DataFrame]: (유효성 여부, 누락된 열 목록, 데이터프레임)
    """
    import pandas as pd

    # 필수 열 목록 (일부는 부분 문자열로 매칭)
    required_columns = [
        '자재명', '자재품목', '배출계수명',
        '배출계수', '배출량(kgCO2eq)', '지역', '자재코드'
    ]

    # 부분 문자열로 매칭해야 하는 열들
    partial_match_columns = ['제품총소요량']

    try:
        # 파일 확장자에 따라 적절한 방법으로 읽기
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            # Excel 파일의 경우 BytesIO를 사용
            uploaded_file.seek(0)  # 파일 포인터를 처음으로 이동
            df = pd.read_excel(uploaded_file)
        else:
            return False, ["지원하지 않는 파일 형식입니다."], pd.DataFrame()

        # 누락된 열 확인
        missing_columns = [col for col in required_columns if col not in df.columns]

        # 부분 문자열로 매칭하는 열들 확인
        for partial_col in partial_match_columns:
            if not any(partial_col in col for col in df.columns):
                missing_columns.append(partial_col)

        if missing_columns:
            return False, missing_columns, df
        else:
            return True, [], df

    except Exception as e:
        return False, [f"파일 읽기 오류: {str(e)}"], pd.DataFrame()

# 이전 함수명과의 호환성을 위한 별칭
def validate_brm_csv(uploaded_file) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    기존 함수명과의 호환성을 위한 래퍼 함수
    """
    return validate_brm_file(uploaded_file)

def load_default_brm_data(user_id: Optional[str] = None) -> pd.DataFrame:
    """
    기본 BRM 데이터를 로드합니다.
    사용자별 파일 -> updated 파일 -> sample 파일 순서로 시도합니다.
    
    Args:
        user_id: 사용자 ID (옵션)
        
    Returns:
        pd.DataFrame: 기본 BRM 데이터
    """
    updated_file_path = "data/pcf_original_table_updated.csv"
    sample_file_path = "data/pcf_original_table_sample.csv"
    
    try:
        # 1. 사용자 ID가 있으면 사용자별 파일 로드 시도
        if user_id:
            try:
                return FileOperations.load_csv(updated_file_path, user_id=user_id)
            except FileLoadError:
                # 사용자별 파일이 없으면 사용자별 sample 파일 생성/로드
                try:
                    # 기본 sample 파일을 사용자별로 복사
                    sample_df = FileOperations.load_csv(sample_file_path)
                    # 사용자별 sample 파일로 저장
                    FileOperations.save_csv(sample_df, sample_file_path, user_id=user_id)
                    return sample_df
                except FileLoadError:
                    pass  # sample 파일도 없으면 다음 단계로
        
        # 2. 일반 updated 파일 로드 시도
        try:
            return FileOperations.load_csv(updated_file_path)
        except FileLoadError:
            pass
        
        # 3. sample 파일 로드 시도
        try:
            return FileOperations.load_csv(sample_file_path)
        except FileLoadError:
            pass
            
    except Exception as e:
        st.error(f"기본 파일 로드 중 오류 발생: {e}")
        
    return pd.DataFrame()

def save_brm_data(df: pd.DataFrame, filename: str = "pcf_original_table_updated.csv", user_id: Optional[str] = None):
    """
    BRM 데이터를 CSV 파일로 저장합니다.
    
    Args:
        df: 저장할 데이터프레임
        filename: 저장할 파일명
        user_id: 사용자 ID (옵션)
    """
    file_path = FileOperations.get_file_path("data", filename)
    
    try:
        FileOperations.save_csv(df, file_path, user_id=user_id, encoding='utf-8-sig')
        return True
    except FileSaveError as e:
        st.error(f"파일 저장 중 오류 발생: {e}")
        return False

def save_simulation_config(config_data: dict, filename: str = "sim_config.json", user_id: Optional[str] = None):
    """
    시뮬레이션 설정을 JSON 파일로 저장합니다.
    
    Args:
        config_data: 저장할 설정 데이터
        filename: 저장할 파일명
        user_id: 사용자 ID (옵션)
    """
    file_path = FileOperations.get_file_path("input", filename)
    
    try:
        FileOperations.save_json(file_path, config_data, user_id=user_id)
        return True
    except FileSaveError as e:
        st.error(f"설정 파일 저장 중 오류 발생: {e}")
        return False

def load_simulation_config(filename: str = "sim_config.json", user_id: Optional[str] = None) -> dict:
    """
    시뮬레이션 설정을 JSON 파일에서 로드합니다.
    
    Args:
        filename: 로드할 파일명
        user_id: 사용자 ID (옵션)
        
    Returns:
        dict: 로드된 설정 데이터
    """
    file_path = FileOperations.get_file_path("input", filename)
    
    try:
        return FileOperations.load_json(file_path, default={}, user_id=user_id)
    except FileLoadError as e:
        st.error(f"설정 파일 로드 중 오류 발생: {e}")
        return {}

def get_default_simulation_config() -> dict:
    """
    기본 시뮬레이션 설정을 반환합니다.
    
    Returns:
        dict: 기본 설정 데이터
    """
    return {
        "max_case": 3,
        "num_tier": 2,
        "description": "기본 시뮬레이션 설정"
    }

def save_scenario_data(df: pd.DataFrame, filename: str = "pcf_scenario_saved.csv", user_id: Optional[str] = None):
    """
    시나리오 데이터를 CSV 파일로 저장합니다.
    
    Args:
        df: 저장할 데이터프레임
        filename: 저장할 파일명
        user_id: 사용자 ID (옵션)
    """
    file_path = FileOperations.get_file_path("data", filename)
    
    try:
        FileOperations.save_csv(df, file_path, encoding='utf-8-sig', user_id=user_id)
        return True
    except FileSaveError as e:
        st.error(f"시나리오 파일 저장 중 오류 발생: {e}")
        return False

def generate_unique_emission_factor_name(row):
    """
    자재별로 고유한 배출계수명을 생성합니다.

    규칙:
    - 배출계수명이 비어있지 않고 자재명이 있으면: "배출계수명 - 자재명"
    - 배출계수명이 비어있으면: "자재품목 - 자재명"
    - 자재명이 nan이면: 원본 배출계수명 유지
    """
    import pandas as pd

    material_name = row.get('자재명', '')
    emission_name = row.get('배출계수명', '')
    material_category = row.get('자재품목', '')

    # 자재명이 없거나 nan인 경우 원본 유지
    if pd.isna(material_name) or str(material_name).strip() == '':
        return emission_name if not pd.isna(emission_name) else material_category

    # 배출계수명이 있는 경우
    if not pd.isna(emission_name) and str(emission_name).strip() != '':
        # 배출계수명에 이미 자재명의 핵심 키워드가 포함되어 있는지 확인
        material_name_str = str(material_name).strip()
        emission_name_str = str(emission_name).strip()

        # 자재명의 핵심 키워드 추출 (공백으로 분리)
        material_keywords = set(material_name_str.lower().split())
        emission_keywords = set(emission_name_str.lower().split())

        # 자재명의 모든 키워드가 배출계수명에 포함되어 있으면 중복 방지
        if material_keywords.issubset(emission_keywords):
            return emission_name_str
        else:
            # 중복되지 않는 자재명의 고유 키워드만 추출
            unique_material_keywords = material_keywords - emission_keywords

            if unique_material_keywords:
                # 고유 키워드가 있으면 원래 순서대로 재구성
                unique_parts = [word for word in material_name_str.split() if word.lower() in unique_material_keywords]
                unique_suffix = ' '.join(unique_parts)
                return f"{emission_name_str} - {unique_suffix}"
            else:
                # 고유 키워드가 없으면 (이론적으로 발생하지 않아야 함) 배출계수 값으로 구분
                emission_factor = row.get('배출계수', 0)
                return f"{emission_name_str} - {material_name_str} ({emission_factor:.6f})"
    else:
        # 배출계수명이 없는 경우 자재품목과 자재명 조합
        return f"{material_category} - {material_name}"


def apply_unique_emission_factor_names_with_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임에 고유한 배출계수명을 적용하고, 중복되는 경우 인덱스를 추가합니다.

    Args:
        df: 배출계수명을 고유화할 데이터프레임

    Returns:
        pd.DataFrame: 고유한 배출계수명이 적용된 데이터프레임
    """
    import pandas as pd
    from collections import Counter

    if '배출계수명' not in df.columns:
        return df

    # 원본 배출계수명 보존
    if '배출계수명_원본' not in df.columns:
        df['배출계수명_원본'] = df['배출계수명'].copy()

    # 1단계: 기본 고유화 적용
    df['배출계수명_임시'] = df.apply(generate_unique_emission_factor_name, axis=1)

    # 2단계: 자재품목별로 중복 검사 및 인덱스 추가
    for category in df['자재품목'].unique():
        if pd.isna(category):
            continue

        category_mask = df['자재품목'] == category
        category_df = df[category_mask]

        # 이 카테고리 내에서 배출계수명별 중복 카운트
        name_counts = Counter(category_df['배출계수명_임시'].values)

        # 중복되는 이름들에 대해 인덱스 추가
        for name, count in name_counts.items():
            if count > 1:
                # 이 이름을 가진 행들 찾기
                name_mask = category_mask & (df['배출계수명_임시'] == name)
                indices = df[name_mask].index

                # 각 행에 순번 추가
                for i, idx in enumerate(indices, 1):
                    df.loc[idx, '배출계수명'] = f"{name} ({i})"
            else:
                # 중복이 없으면 그대로 사용
                name_mask = category_mask & (df['배출계수명_임시'] == name)
                df.loc[name_mask, '배출계수명'] = name

    # 임시 컬럼 제거
    df = df.drop(columns=['배출계수명_임시'])

    return df


def validate_emission_factor_uniqueness(df: pd.DataFrame) -> dict:
    """
    배출계수명의 고유성을 검증하고 중복 현황을 반환합니다.

    Args:
        df: 검증할 데이터프레임

    Returns:
        dict: 검증 결과 (is_valid, duplicates, statistics)
    """
    import pandas as pd

    validation_report = {
        'is_valid': True,
        'duplicates': [],
        'statistics': {
            'total_materials': len(df),
            'total_categories': df['자재품목'].nunique(),
            'unique_emission_names': df['배출계수명'].nunique()
        }
    }

    for category in df['자재품목'].unique():
        if pd.isna(category):
            continue

        category_df = df[df['자재품목'] == category]

        # 배출계수명으로 그룹핑
        grouped = category_df.groupby('배출계수명').agg({
            '자재명': lambda x: list(x.unique()),
            '배출계수': lambda x: list(x.unique())
        }).reset_index()

        for _, row in grouped.iterrows():
            material_names = row['자재명']
            emission_factors = row['배출계수']

            # 같은 배출계수명에 여러 배출계수가 있는 경우 (문제!)
            if len(emission_factors) > 1:
                validation_report['is_valid'] = False
                validation_report['duplicates'].append({
                    'category': category,
                    'emission_name': row['배출계수명'],
                    'materials': material_names,
                    'emission_factors': emission_factors,
                    'severity': 'high'  # 배출계수가 다르면 심각
                })
            # 같은 배출계수명에 여러 자재명이 있는 경우 (경고)
            elif len(material_names) > 1:
                validation_report['duplicates'].append({
                    'category': category,
                    'emission_name': row['배출계수명'],
                    'materials': material_names,
                    'emission_factors': emission_factors,
                    'severity': 'low'  # 배출계수가 같으면 경미
                })

    return validation_report


def generate_scenario_from_brm(brm_df: pd.DataFrame) -> pd.DataFrame:
    """
    BRM 원본 데이터에서 기본 시나리오를 자동 생성합니다.

    Args:
        brm_df: BRM 원본 데이터프레임

    Returns:
        pd.DataFrame: 생성된 시나리오 데이터프레임
    """
    if brm_df.empty:
        return pd.DataFrame()

    # BRM 데이터에서 시나리오 생성에 필요한 열들 추출
    required_columns = ['자재명', '자재품목', '배출계수명', '배출계수', '배출량(kgCO2eq)']

    # 제품총소요량 열 찾기 (다양한 형태로 명명될 수 있음)
    total_amount_col = None
    for col in brm_df.columns:
        if '제품총소요량' in col:
            total_amount_col = col
            break

    if total_amount_col:
        required_columns.append(total_amount_col)

    # 필요한 열이 없으면 빈 데이터프레임 반환
    missing_cols = [col for col in required_columns if col not in brm_df.columns]
    if missing_cols:
        st.warning(f"시나리오 자동 생성을 위한 필수 열이 누락되었습니다: {missing_cols}")
        return pd.DataFrame()

    # 기본 데이터 복사
    scenario_df = brm_df[required_columns].copy()

    # 열명 표준화 (시나리오 형식에 맞게)
    if total_amount_col and total_amount_col != '제품총소요량(kg)':
        scenario_df = scenario_df.rename(columns={total_amount_col: '제품총소요량(kg)'})

    # 저감활동 적용 여부 결정 (자재품목 기반)
    reduction_materials = ['양극재', '음극재', '전해액', 'Al Foil', 'Cu Foil']

    def determine_reduction_activity(material_type):
        return 1 if material_type in reduction_materials else 0

    scenario_df['저감활동_적용여부'] = scenario_df['자재품목'].apply(determine_reduction_activity)

    # 기본 재생에너지 비율 설정 (저감활동 적용 자재에 대해서만)
    def set_re_ratio(is_applicable, material_type):
        if not is_applicable:
            return 0

        # 자재 유형별 기본 RE 비율 설정
        if material_type == '양극재':
            return {'case1': '100%', 'case2': '50%', 'case3': '30%'}
        elif material_type == '음극재':
            return {'case1': '100%', 'case2': '50%', 'case3': '40%'}
        elif material_type == '전해액':
            return {'case1': '100%', 'case2': '50%', 'case3': '30%'}
        elif material_type in ['Al Foil', 'Cu Foil']:
            return {'case1': '100%', 'case2': '50%', 'case3': '20%'}
        else:
            return {'case1': '100%', 'case2': '50%', 'case3': '30%'}

    # 각 행에 대해 RE 비율 설정
    tier1_cases = {f'Tier1_RE_case{i}': [] for i in range(1, 4)}
    tier2_cases = {f'Tier2_RE_case{i}': [] for i in range(1, 4)}

    for _, row in scenario_df.iterrows():
        is_applicable = row['저감활동_적용여부']
        material_type = row['자재품목']

        if is_applicable:
            re_ratios = set_re_ratio(is_applicable, material_type)

            # Tier1과 Tier2에 동일한 비율 적용
            for i in range(1, 4):
                tier1_cases[f'Tier1_RE_case{i}'].append(re_ratios[f'case{i}'])
                tier2_cases[f'Tier2_RE_case{i}'].append(re_ratios[f'case{i}'])
        else:
            # 저감활동 미적용 자재는 0%
            for i in range(1, 4):
                tier1_cases[f'Tier1_RE_case{i}'].append('0')
                tier2_cases[f'Tier2_RE_case{i}'].append('0')

    # RE 비율 열들 추가
    for key, values in tier1_cases.items():
        scenario_df[key] = values
    for key, values in tier2_cases.items():
        scenario_df[key] = values

    # 배출계수명 고유화 (원본 보존) - 인덱스 기반 고유화 사용
    print(f"🔧 배출계수명 고유화 시작...")
    original_row_count = len(scenario_df)
    print(f"   - 고유화 전 총 행 수: {original_row_count}개")

    scenario_df = apply_unique_emission_factor_names_with_index(scenario_df)

    # 중복 행 제거 (고유화 후) - 더 많은 컬럼으로 정확한 중복 검사
    # 고유 ID 생성 (행 인덱스 + 자재명 + 자재품목 + 배출량)
    # 배출량을 포함하여 같은 자재가 다른 반제품에 사용될 때 구분
    scenario_df['_unique_row_id'] = (
        scenario_df.index.astype(str) + '_' +
        scenario_df['자재명'].astype(str) + '_' +
        scenario_df['자재품목'].astype(str) + '_' +
        scenario_df['배출량(kgCO2eq)'].astype(str)
    )

    # 확장된 중복 검사 키 (더 많은 컬럼 포함)
    key_columns = ['자재명', '자재품목', '배출계수명', '배출계수', '제품총소요량(kg)', '배출량(kgCO2eq)', '지역', '자재코드']
    available_key_columns = [col for col in key_columns if col in scenario_df.columns]

    if available_key_columns:
        print(f"🔍 중복 제거 기준 컬럼: {available_key_columns}")

        # 중복 행 확인
        duplicates = scenario_df[scenario_df.duplicated(subset=available_key_columns, keep=False)]
        if not duplicates.empty:
            print(f"⚠️ 확장된 기준으로 중복 행 발견: {len(duplicates)}개")

            # 자재품목별 중복 상세 정보
            for category in duplicates['자재품목'].unique():
                category_dups = duplicates[duplicates['자재품목'] == category]
                print(f"   - {category}: {len(category_dups)}개 중복")

                # 음극재의 경우 상세 정보 출력
                if category == '음극재':
                    print(f"   📋 음극재 중복 상세:")
                    for idx, dup_row in category_dups.iterrows():
                        print(f"      • {dup_row['자재명']}: 배출계수={dup_row.get('배출계수', 'N/A')}, "
                              f"소요량={dup_row.get('제품총소요량(kg)', 'N/A')}, "
                              f"배출량={dup_row.get('배출량(kgCO2eq)', 'N/A')}")

        # 중복 제거 (첫 번째 행만 유지) - 단, _unique_row_id도 함께 사용하여 완전 동일한 행만 제거
        combined_key_columns = available_key_columns + ['_unique_row_id']
        scenario_df = scenario_df.drop_duplicates(subset=combined_key_columns, keep='first')

        # _unique_row_id 컬럼 제거
        scenario_df = scenario_df.drop(columns=['_unique_row_id'])

        deduplicated_row_count = len(scenario_df)
        removed_count = original_row_count - deduplicated_row_count

        if removed_count > 0:
            print(f"✅ 중복 제거 완료: {removed_count}개 행 제거됨")
            print(f"   - 제거 후 총 행 수: {deduplicated_row_count}개")
        else:
            print(f"✅ 중복 없음: 모든 행이 고유함")

        # 음극재 개수 재확인
        anode_count_after = len(scenario_df[scenario_df['자재품목'] == '음극재'])
        print(f"📊 중복 제거 후 음극재 개수: {anode_count_after}개")

    # 배출계수명 고유화 검증 및 로깅
    validation_result = validate_emission_factor_uniqueness(scenario_df)

    if validation_result['duplicates']:
        print(f"⚠️ 배출계수명 중복 검출: {len(validation_result['duplicates'])}건")
        for dup in validation_result['duplicates']:
            severity_emoji = "🔴" if dup['severity'] == 'high' else "🟡"
            print(f"{severity_emoji} {dup['category']} - {dup['emission_name']}")
            print(f"   자재: {dup['materials']}")
            print(f"   배출계수: {dup['emission_factors']}")
    else:
        print(f"✅ 배출계수명 고유성 검증 완료")

    print(f"📊 통계: 전체 {validation_result['statistics']['total_materials']}개 자재, "
          f"{validation_result['statistics']['total_categories']}개 품목, "
          f"{validation_result['statistics']['unique_emission_names']}개 고유 배출계수명")

    return scenario_df

def load_default_scenario_data(user_id: Optional[str] = None) -> pd.DataFrame:
    """
    기본 시나리오 데이터를 로드합니다.
    사용자별 파일 -> saved 파일 -> sample 파일 순서로 시도합니다.

    Args:
        user_id: 사용자 ID (옵션)

    Returns:
        pd.DataFrame: 기본 시나리오 데이터
    """
    saved_file_path = "data/pcf_scenario_saved.csv"
    sample_file_path = "data/pcf_scenario_sample.csv"

    try:
        # 1. 사용자 ID가 있으면 사용자별 파일 로드 시도
        if user_id:
            try:
                return FileOperations.load_csv(saved_file_path, user_id=user_id)
            except FileLoadError:
                # 사용자별 파일이 없으면 사용자별 sample 파일 생성/로드
                try:
                    # 기본 sample 파일을 사용자별로 복사
                    sample_df = FileOperations.load_csv(sample_file_path)
                    # 사용자별 sample 파일로 저장
                    FileOperations.save_csv(sample_df, sample_file_path, user_id=user_id)
                    return sample_df
                except FileLoadError:
                    pass  # sample 파일도 없으면 다음 단계로

        # 2. 일반 saved 파일 로드 시도
        try:
            return FileOperations.load_csv(saved_file_path)
        except FileLoadError:
            pass

        # 3. sample 파일 로드 시도
        try:
            return FileOperations.load_csv(sample_file_path)
        except FileLoadError:
            pass

    except Exception as e:
        st.error(f"기본 시나리오 파일 로드 중 오류 발생: {e}")

    return pd.DataFrame()

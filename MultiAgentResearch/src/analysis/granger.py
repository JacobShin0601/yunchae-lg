import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from statsmodels.tsa.stattools import grangercausalitytests
# 상대 경로 임포트를 절대 경로 임포트로 변경
from correlation import get_top_lagged_correlations_by_category
from data_loader import extract_variables_by_prefix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_granger_causality_for_correlated_vars(
    df: pd.DataFrame,
    target_variable: str,
    independent_variable_prefixes: List[str],
    cross_corr_threshold: float = 0.0, 
    num_top_corr_vars: int = 5,      
    max_lag_cross_corr: int = 60,    
    max_lag_granger: int = 12,       
    granger_significance_level: float = 0.05, 
    apply_diff_to_granger: bool = True, 
    diff_periods_granger: int = 1       
) -> Dict[str, Dict[str, Any]]:
    """
    교차 상관관계가 높은 변수들을 식별하고, 이 변수들이 타겟 변수에 대해 그레인저 인과성이 있는지 분석합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        target_variable (str): 타겟(종속) 변수의 컬럼명.
        independent_variable_prefixes (List[str]): 독립변수 접두사 리스트.
        cross_corr_threshold (float, optional): 교차 상관 분석 시 사용할 상관계수 절대값 임계값. Defaults to 0.0.
        num_top_corr_vars (int, optional): 그레인저 인과 분석 대상으로 선택할 상위 교차 상관 변수의 수. Defaults to 5.
        max_lag_cross_corr (int, optional): 교차 상관 분석 시 고려할 최대 시차. Defaults to 60.
        max_lag_granger (int, optional): 그레인저 인과 분석 시 고려할 최대 시차. Defaults to 12.
        granger_significance_level (float, optional): 그레인저 인과 검정의 유의수준. Defaults to 0.05.
        apply_diff_to_granger (bool, optional): 그레인저 분석 전 시계열에 차분을 적용할지 여부. Defaults to True.
        diff_periods_granger (int, optional): 차분을 적용할 경우의 기간. Defaults to 1.

    Returns:
        Dict[str, Dict[str, Any]]: 각 독립변수명를 키로 하고, 
                                   교차 상관관계 정보와 그레인저 인과 분석 결과를 값으로 하는 딕셔너리.
    """
    if target_variable not in df.columns:
        raise ValueError(f"타겟 변수 '{target_variable}'가 데이터프레임에 없습니다.")
    if not all(isinstance(prefix, str) for prefix in independent_variable_prefixes):
        raise ValueError("independent_variable_prefixes는 문자열 리스트여야 합니다.")
    if max_lag_granger <= 0:
        raise ValueError("max_lag_granger는 0보다 큰 정수여야 합니다.")

    # 독립변수 추출
    independent_variables = extract_variables_by_prefix(df, independent_variable_prefixes)
    if not independent_variables:
        raise ValueError("주어진 접두사로 매칭되는 독립변수를 찾을 수 없습니다.")

    # 교차 상관관계 분석
    correlated_vars_info = []
    for var in independent_variables:
        if var == target_variable:
            continue
            
        correlations = []
        for lag in range(max_lag_cross_corr + 1):
            corr = df[target_variable].corr(df[var].shift(lag))
            if abs(corr) >= cross_corr_threshold:
                correlations.append({
                    'var_name': var,
                    'lag': lag,
                    'corr': corr
                })
        
        if correlations:
            best_corr = max(correlations, key=lambda x: abs(x['corr']))
            correlated_vars_info.append(best_corr)
    
    # 상관관계 기준으로 상위 변수 선택
    correlated_vars_info.sort(key=lambda x: abs(x['corr']), reverse=True)
    correlated_vars_info = correlated_vars_info[:num_top_corr_vars]

    final_analysis_results = {}

    if not correlated_vars_info:
        print("교차 상관 분석 결과 유의미한 변수를 찾지 못했습니다. 그레인저 인과 분석을 진행하지 않습니다.")
        return final_analysis_results

    for var_info in correlated_vars_info:
        var_name = var_info['var_name']
        if var_name == target_variable: 
            continue

        current_var_analysis = {
            'cross_correlation': {
                'lag': var_info['lag'],
                'value': var_info['corr']
            },
            'granger_causality_to_target': {}
        }

        data_for_granger = df[[target_variable, var_name]].copy()

        if apply_diff_to_granger:
            if diff_periods_granger <= 0:
                raise ValueError("diff_periods_granger는 0보다 큰 정수여야 합니다.")
            data_for_granger = data_for_granger.diff(periods=diff_periods_granger).dropna()
            current_var_analysis['granger_causality_to_target']['differencing_applied'] = True
            current_var_analysis['granger_causality_to_target']['diff_periods'] = diff_periods_granger
        else:
            current_var_analysis['granger_causality_to_target']['differencing_applied'] = False

        current_var_analysis['granger_causality_to_target']['max_lag_tested'] = max_lag_granger
        current_var_analysis['granger_causality_to_target']['significant_at_alpha'] = granger_significance_level
        current_var_analysis['granger_causality_to_target']['lag_results'] = {}
        current_var_analysis['granger_causality_to_target']['is_causal_overall'] = False

        if len(data_for_granger) < max_lag_granger + 5: 
            print(f"변수 '{var_name}'에 대해 그레인저 인과 분석을 위한 데이터가 충분하지 않습니다 (차분 후 길이: {len(data_for_granger)}). 건너뜁니다.")
            final_analysis_results[var_name] = current_var_analysis 
            continue
        
        try:
            granger_results_raw = grangercausalitytests(data_for_granger, maxlag=max_lag_granger, verbose=False)
            
            for lag_val in range(1, max_lag_granger + 1):
                if lag_val not in granger_results_raw:
                    continue
                
                f_test_result = granger_results_raw[lag_val][0]
                p_value = f_test_result['ssr_ftest'][1] # F-검정의 p-value
                f_statistic = f_test_result['ssr_ftest'][0] # F-검정의 F-statistic
                is_causal = p_value < granger_significance_level

                current_var_analysis['granger_causality_to_target']['lag_results'][lag_val] = {
                    'F_statistic': f_statistic,
                    'p_value': p_value,
                    'is_causal': is_causal
                }
                if is_causal:
                    current_var_analysis['granger_causality_to_target']['is_causal_overall'] = True
        
        except Exception as e:
            print(f"변수 '{var_name}'에 대한 그레인저 인과 분석 중 오류 발생: {e}")
            current_var_analysis['granger_causality_to_target']['error'] = str(e)

        final_analysis_results[var_name] = current_var_analysis

    return final_analysis_results

def filter_best_granger_results(granger_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    그레인저 인과성 분석 결과에서 각 변수별로 가장 낮은 p-value를 가진 결과만 필터링합니다.
    p-value가 0.05 이하인 결과만 포함됩니다.

    Args:
        granger_results (Dict[str, Dict]): 그레인저 인과성 분석 결과 딕셔너리

    Returns:
        Dict[str, Dict]: 필터링된 결과 딕셔너리
    """
    filtered_results = {}
    significance_level = 0.05
    
    for var_name, var_results in granger_results.items():
        if not isinstance(var_results, dict) or 'granger_causality_to_target' not in var_results:
            continue
            
        lag_results = var_results['granger_causality_to_target'].get('lag_results', {})
        if not lag_results:
            continue
            
        # p-value가 0.05 이하인 lag 결과만 필터링
        significant_lags = {lag: result for lag, result in lag_results.items() 
                          if result['p_value'] <= significance_level}
        
        if not significant_lags:
            continue
            
        # 가장 낮은 p-value를 가진 lag 결과 찾기
        best_lag = min(significant_lags.items(), key=lambda x: x[1]['p_value'])
        lag_val, best_result = best_lag
        
        # 필터링된 결과 생성
        filtered_var_results = {
            'granger_causality_to_target': {
                'differencing_applied': var_results['granger_causality_to_target']['differencing_applied'],
                'diff_periods': var_results['granger_causality_to_target'].get('diff_periods'),
                'max_lag_tested': var_results['granger_causality_to_target']['max_lag_tested'],
                'significant_at_alpha': var_results['granger_causality_to_target']['significant_at_alpha'],
                'is_causal_overall': var_results['granger_causality_to_target']['is_causal_overall'],
                'best_lag': {
                    'lag': lag_val,
                    'F_statistic': best_result['F_statistic'],
                    'p_value': best_result['p_value'],
                    'is_causal': best_result['is_causal']
                }
            }
        }
        
        filtered_results[var_name] = filtered_var_results
        
    return filtered_results

def visualize_granger_results(filtered_results: Dict[str, Dict], 
                            figsize: tuple = (10, 10),
                            title: str = "그레인저 인과성 분석 결과",
                            output_path: Optional[str] = None) -> None:
    """
    필터링된 그레인저 인과성 분석 결과를 시각화합니다.
    
    Args:
        filtered_results (Dict[str, Dict]): filter_best_granger_results 함수의 결과
        figsize (tuple, optional): 그래프 크기. Defaults to (15, 10).
        title (str, optional): 그래프 제목. Defaults to "그레인저 인과성 분석 결과".
        output_path (Optional[str], optional): 그래프 저장 경로. Defaults to None.
    """
    if not filtered_results:
        print("시각화할 결과가 없습니다.")
        return
        
    # 데이터 준비
    variables = []
    lags = []
    p_values = []
    f_stats = []
    
    for var_name, results in filtered_results.items():
        best_lag = results['granger_causality_to_target']['best_lag']
        variables.append(var_name)
        lags.append(best_lag['lag'])
        p_values.append(best_lag['p_value'])
        f_stats.append(best_lag['F_statistic'])
    
    # 데이터프레임 생성
    df_results = pd.DataFrame({
        'Variable': variables,
        'Lag': lags,
        'P-value': p_values,
        'F-statistic': f_stats
    })
    
    # 시각화
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # 1. P-value 히트맵
    ax1 = fig.add_subplot(gs[0, :])
    p_value_matrix = df_results.pivot_table(
        values='P-value',
        index='Variable',
        columns='Lag',
        fill_value=1.0
    )
    sns.heatmap(p_value_matrix, 
                cmap='RdYlGn_r',
                vmin=0, 
                vmax=0.05,
                ax=ax1,
                cbar_kws={'label': 'P-value'})
    ax1.set_title('P-value 히트맵 (0.05 이하가 유의미)')
    
    # 2. F-statistic 바 차트
    ax2 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df_results, 
                x='Variable', 
                y='F-statistic',
                ax=ax2)
    ax2.set_title('F-statistic 분포')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Lag 분포
    ax3 = fig.add_subplot(gs[1, 1])
    sns.barplot(data=df_results, 
                x='Variable', 
                y='Lag',
                ax=ax3)
    ax3.set_title('최적 Lag 분포')
    ax3.tick_params(axis='x', rotation=45)
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    
    # 결과 요약 출력
    print("\n=== 그레인저 인과성 분석 결과 요약 ===")
    print(f"총 분석 변수 수: {len(variables)}")
    print("\n상위 5개 변수의 결과:")
    top_5 = df_results.nsmallest(5, 'P-value')
    print(top_5[['Variable', 'P-value', 'F-statistic', 'Lag']].to_string(index=False))
    
    # 그래프 저장
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"그래프가 {output_path}에 저장되었습니다.")
    
    plt.show()
    return fig

def plot_granger_causality_network(filtered_results: Dict[str, Dict],
                                 figsize: tuple = (12, 8),
                                 title: str = "그레인저 인과성 네트워크",
                                 output_path: Optional[str] = None) -> None:
    """
    그레인저 인과성 분석 결과를 네트워크 그래프로 시각화합니다.
    
    Args:
        filtered_results (Dict[str, Dict]): filter_best_granger_results 함수의 결과
        figsize (tuple, optional): 그래프 크기. Defaults to (12, 8).
        title (str, optional): 그래프 제목. Defaults to "그레인저 인과성 네트워크".
        output_path (Optional[str], optional): 그래프 저장 경로. Defaults to None.
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx 패키지가 필요합니다. 'pip install networkx'로 설치해주세요.")
        return
        
    if not filtered_results:
        print("시각화할 결과가 없습니다.")
        return
    
    # 네트워크 그래프 생성
    G = nx.DiGraph()
    
    # 노드와 엣지 추가
    for var_name, results in filtered_results.items():
        best_lag = results['granger_causality_to_target']['best_lag']
        p_value = best_lag['p_value']
        f_stat = best_lag['F_statistic']
        lag = best_lag['lag']
        
        # 노드 추가
        G.add_node(var_name)
        
        # 엣지 추가 (p-value가 0.05 이하인 경우만)
        if p_value <= 0.05:
            G.add_edge(var_name, 'target',
                      weight=1/p_value,  # p-value가 낮을수록 엣지 두께 증가
                      lag=lag,
                      f_stat=f_stat)
    
    # 그래프 그리기
    plt.figure(figsize=figsize)
    
    # 레이아웃 최적화
    pos = nx.spring_layout(G, 
                          k=2.0,  # 노드 간 거리 증가
                          iterations=100,  # 반복 횟수 증가
                          seed=42)  # 재현성을 위한 시드 설정
    
    # 노드 크기 계산 (변수 수에 따라 동적 조정)
    n_nodes = len(G.nodes())
    node_size = min(3000, max(1000, 5000 / n_nodes))
    
    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=node_size,
                          alpha=0.7,
                          edgecolors='gray',  # 노드 테두리 추가
                          linewidths=2)  # 테두리 두께
    
    # 엣지 그리기
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    
    # 엣지 두께 계산 (p-value에 따라)
    edge_widths = [w/max_weight * 2 for w in weights]  # 최대 두께 조정
    
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          width=edge_widths,
                          arrows=True,
                          arrowsize=20,
                          connectionstyle='arc3,rad=0.2')  # 곡선 엣지로 변경
    
    # 레이블 그리기 (노드 크기에 맞게 조정)
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold')
    
    # 엣지 레이블 (lag 값)
    edge_labels = {(u, v): f"lag={G[u][v]['lag']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, 
                                edge_labels=edge_labels,
                                font_size=8,
                                bbox=dict(facecolor='white', 
                                        edgecolor='none', 
                                        alpha=0.7))
    
    plt.title(title, pad=20)  # 제목과 그래프 사이 간격 추가
    plt.axis('off')
    
    # 여백 추가
    plt.tight_layout()
    
    # 그래프 저장
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"그래프가 {output_path}에 저장되었습니다.")
    
    plt.show()
    return plt.gcf()

def run_process(
    df: pd.DataFrame,
    target_variable: str,
    independent_variable_prefixes: List[str],
    cross_corr_threshold: float = 0.0,
    num_top_corr_vars: int = 5,
    max_lag_cross_corr: int = 60,
    max_lag_granger: int = 12,
    granger_significance_level: float = 0.05,
    apply_diff_to_granger: bool = True,
    diff_periods_granger: int = 1,
    visualize: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Dict]:
    """
    전체 그레인저 인과성 분석 및 시각화 파이프라인을 실행합니다.
    (correlation.py의 run_process 스타일 참고)

    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        target_variable (str): 타겟 변수명
        independent_variable_prefixes (List[str]): 독립변수 접두사 리스트
        cross_corr_threshold (float, optional): 교차 상관 분석 임계값
        num_top_corr_vars (int, optional): 상위 교차 상관 변수 개수
        max_lag_cross_corr (int, optional): 교차 상관 최대 시차
        max_lag_granger (int, optional): 그레인저 분석 최대 시차
        granger_significance_level (float, optional): 그레인저 유의수준
        apply_diff_to_granger (bool, optional): 차분 적용 여부
        diff_periods_granger (int, optional): 차분 기간
        visualize (bool, optional): 시각화 실행 여부
        output_path (Optional[str], optional): 그래프 저장 경로. Defaults to None.

    Returns:
        Dict[str, Dict]: 필터링된 그레인저 인과성 분석 결과
    """
    print("[1/4] 그레인저 인과성 분석 수행...")
    granger_result_dict = analyze_granger_causality_for_correlated_vars(
        df=df,
        target_variable=target_variable,
        independent_variable_prefixes=independent_variable_prefixes,
        cross_corr_threshold=cross_corr_threshold,
        num_top_corr_vars=num_top_corr_vars,
        max_lag_cross_corr=max_lag_cross_corr,
        max_lag_granger=max_lag_granger,
        granger_significance_level=granger_significance_level,
        apply_diff_to_granger=apply_diff_to_granger,
        diff_periods_granger=diff_periods_granger
    )
    print(f"  - 분석된 변수 수: {len(granger_result_dict)}")

    print("[2/4] 유의미한 인과성 결과 필터링...")
    filtered_granger_dict = filter_best_granger_results(granger_result_dict)
    print(f"  - 유의미한 인과성 변수 수: {len(filtered_granger_dict)}")

    if visualize:
        from datetime import datetime
        import os
        import json
        
        # output_path가 None인 경우 기본 경로 설정
        if output_path is None:
            # output 디렉토리 생성
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.join(output_dir, f"granger_analysis_{target_variable}_{timestamp}")
            
            # 기본 시각화 저장
            print("[3/4] 인과성 분석 결과 시각화...")
            visualize_granger_results(filtered_granger_dict, output_path=f"{base_path}_1.png")
            
            # 네트워크 시각화 저장
            print("[4/4] 인과성 네트워크 시각화...")
            plot_granger_causality_network(filtered_granger_dict, output_path=f"{base_path}_2.png")
            
            # JSON 결과 저장
            json_path = f"{base_path}_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                # bool 타입을 int로 변환하여 저장
                json_results = {}
                for var_name, var_results in filtered_granger_dict.items():
                    json_results[var_name] = {
                        'granger_causality_to_target': {
                            'differencing_applied': int(var_results['granger_causality_to_target']['differencing_applied']),
                            'diff_periods': var_results['granger_causality_to_target'].get('diff_periods'),
                            'max_lag_tested': var_results['granger_causality_to_target']['max_lag_tested'],
                            'significant_at_alpha': var_results['granger_causality_to_target']['significant_at_alpha'],
                            'is_causal_overall': int(var_results['granger_causality_to_target']['is_causal_overall']),
                            'best_lag': {
                                'lag': var_results['granger_causality_to_target']['best_lag']['lag'],
                                'F_statistic': float(var_results['granger_causality_to_target']['best_lag']['F_statistic']),
                                'p_value': float(var_results['granger_causality_to_target']['best_lag']['p_value']),
                                'is_causal': int(var_results['granger_causality_to_target']['best_lag']['is_causal'])
                            }
                        }
                    }
                json.dump(json_results, f, ensure_ascii=False, indent=4)
            print(f"그레인저 분석 결과가 {json_path}에 저장되었습니다.")
        else:
            # 사용자가 지정한 경로가 있는 경우
            output_dir = os.path.dirname(output_path)
            if output_dir:  # 디렉토리가 지정된 경우
                os.makedirs(output_dir, exist_ok=True)
            
            base_path = output_path.rsplit('.', 1)[0]  # 확장자 제거
            
            # 기본 시각화 저장 (_1 접미사)
            print("[3/4] 인과성 분석 결과 시각화...")
            visualize_granger_results(filtered_granger_dict, output_path=f"{base_path}_1.png")
            
            # 네트워크 시각화 저장 (_2 접미사)
            print("[4/4] 인과성 네트워크 시각화...")
            plot_granger_causality_network(filtered_granger_dict, output_path=f"{base_path}_2.png")
            
            # JSON 결과 저장
            json_path = f"{base_path}_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                # bool 타입을 int로 변환하여 저장
                json_results = {}
                for var_name, var_results in filtered_granger_dict.items():
                    json_results[var_name] = {
                        'granger_causality_to_target': {
                            'differencing_applied': int(var_results['granger_causality_to_target']['differencing_applied']),
                            'diff_periods': var_results['granger_causality_to_target'].get('diff_periods'),
                            'max_lag_tested': var_results['granger_causality_to_target']['max_lag_tested'],
                            'significant_at_alpha': var_results['granger_causality_to_target']['significant_at_alpha'],
                            'is_causal_overall': int(var_results['granger_causality_to_target']['is_causal_overall']),
                            'best_lag': {
                                'lag': var_results['granger_causality_to_target']['best_lag']['lag'],
                                'F_statistic': float(var_results['granger_causality_to_target']['best_lag']['F_statistic']),
                                'p_value': float(var_results['granger_causality_to_target']['best_lag']['p_value']),
                                'is_causal': int(var_results['granger_causality_to_target']['best_lag']['is_causal'])
                            }
                        }
                    }
                json.dump(json_results, f, ensure_ascii=False, indent=4)
            print(f"그레인저 분석 결과가 {json_path}에 저장되었습니다.")

    print("\n[완료] 그레인저 인과성 분석 및 시각화가 종료되었습니다.")
    return filtered_granger_dict






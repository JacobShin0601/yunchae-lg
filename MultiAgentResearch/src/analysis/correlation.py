# 삽입될 코드 상단에 필요한 import 문 (실제 삽입 시에는 생략될 수 있음):
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Dict, Any # Union은 pd.Series | Dict 와 같은 경우에 사용하나, 여기서는 pd.Series로 통일
from statsmodels.tsa.stattools import ccf # 추가된 import
from typing import Optional, List, Dict, Any, Union
import os
import json

def get_significant_correlations_with_target(df: pd.DataFrame, target_variable: str, threshold: float, independent_variables: List[str]) -> Tuple[List[str], pd.Series]:
    """
    지정된 종속변수와 독립변수들 간의 피어슨 상관관계를 계산하고,
    상관계수의 절댓값이 특정 임계값 이상인 변수들을 선택합니다.
    선택된 변수 리스트와 해당 변수들과 종속변수 간의 상관계수 시리즈를 반환합니다.
    종속변수는 항상 리스트와 시리즈의 첫 번째 요소로 포함됩니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        target_variable (str): 종속변수의 컬럼명.
        threshold (float): 상관관계 절댓값의 임계값 (0과 1 사이).
        independent_variables (List[str]): 분석할 독립변수들의 접두사 리스트.
                                         이 리스트를 통해 선택된 독립변수들은 target_variable과 달라야 합니다.

    Returns:
        Tuple[List[str], pd.Series]:
            - significant_variables_list (List[str]): 
                선택된 유의미한 변수들의 이름 리스트. 종속변수가 첫 번째로 오고, 
                나머지는 종속변수와의 상관계수 절대값 기준 내림차순으로 정렬됩니다.
            - final_target_correlations (pd.Series): 
                종속변수와 significant_variables_list에 포함된 변수들 간의 실제 상관계수.
                인덱스는 변수명, 값은 상관계수이며, significant_variables_list와 순서가 동일합니다.
    """
    if target_variable not in df.columns:
        raise ValueError(f"종속변수 '{target_variable}'가 데이터프레임의 컬럼에 존재하지 않습니다.")
    if not (0 <= threshold <= 1):
        raise ValueError("상관관계 임계값은 0과 1 사이의 값이어야 합니다.")

    # 독립변수들 중에서 해당 접두사를 가진 컬럼들만 선택
    selected_independent_cols = []
    for prefix in independent_variables:
        matching_columns = [
            col for col in df.columns if col.startswith(prefix) and col != target_variable
        ]
        selected_independent_cols.extend(matching_columns)
    
    unique_independent_cols = sorted(list(set(selected_independent_cols)))
    cols_for_analysis = [target_variable] + unique_independent_cols
    analysis_df = df[cols_for_analysis]

    # 수치형 데이터만 선택
    numeric_df = analysis_df.select_dtypes(include=[np.number])
    
    if target_variable not in numeric_df.columns:
        raise ValueError(f"종속변수 '{target_variable}'는 수치형 데이터가 아닙니다.")

    # 피어슨 상관계수 행렬 계산
    corr_matrix = numeric_df.corr(method='pearson')
    # print("피어슨 상관계수 행렬:") # 사용자 요청이 아니므로 주석 처리 또는 로깅 레벨 조정
    # print(corr_matrix)
    
    all_target_correlations = corr_matrix[target_variable].copy()
    target_self_correlation_value = all_target_correlations[target_variable]
    other_vars_correlations = all_target_correlations.drop(target_variable)
    
    # print("\n종속변수와의 피어슨 상관계수:") # 사용자 요청이 아니므로 주석 처리
    # print(other_vars_correlations)
    
    if not other_vars_correlations.empty:
        significant_other_correlations = other_vars_correlations[abs(other_vars_correlations) >= threshold]
        # if not significant_other_correlations.empty:
            # print(f"\n임계값 {threshold} 이상의 피어슨 상관계수를 가진 변수들:") # 사용자 요청이 아니므로 주석 처리
            # print(significant_other_correlations)
        # else:
            # print(f"\n임계값 {threshold} 이상의 피어슨 상관계수를 가진 변수가 없습니다.") # 사용자 요청이 아니므로 주석 처리
    else:
        significant_other_correlations = pd.Series(dtype='float64')
        # print("\n임계값 이상의 피어슨 상관계수를 가진 변수가 없습니다.") # 사용자 요청이 아니므로 주석 처리

    sorted_significant_other_correlations = significant_other_correlations.reindex(
        significant_other_correlations.abs().sort_values(ascending=False).index
    )

    final_target_correlations = pd.concat([
        pd.Series({target_variable: target_self_correlation_value}), 
        sorted_significant_other_correlations
    ])

    significant_variables_list = final_target_correlations.index.tolist()
    
    return significant_variables_list, final_target_correlations

def get_cross_correlation_with_lag(series1: pd.Series, series2: pd.Series, max_lag: Union[int, None] = None) -> Tuple[int, float]:
    """
    두 시계열 데이터 간의 교차 상관관계를 계산하고,
    가장 높은 절대 상관계수를 가지는 시차(lag)와 해당 상관계수를 반환합니다.
    이 함수는 series1[t]와 series2[t-k] 간의 상관관계를 계산합니다 (k >= 0).

    Args:
        series1 (pd.Series): 기준 시계열 데이터 (x).
        series2 (pd.Series): 비교할 시계열 데이터 (y). 두 시리즈의 길이는 동일해야 합니다.
        max_lag (Union[int, None], optional): 
            고려할 최대 시차 (k의 최댓값). 0부터 max_lag까지의 시차를 고려합니다.
            None이면 statsmodels ccf의 기본 동작에 따라 가능한 모든 양의 시차를 고려합니다.
            (일반적으로 N-1, N은 시리즈 길이). 사용자가 0~60으로 제한하고 싶다면 이 값을 60으로 설정합니다.

    Returns:
        Tuple[int, float]:
            - best_lag (int): 가장 높은 절대 교차 상관계수를 나타내는 시차 (0 <= best_lag <= max_lag).
                              series2가 series1에 비해 best_lag만큼 과거 시점일 때 상관관계가 높음을 의미.
            - max_ccf (float): best_lag에서의 교차 상관계수 값.

    Raises:
        ValueError: series1과 series2의 길이가 다르거나, NaN 처리 후 유효 데이터가 부족할 경우.
    """
    if len(series1) != len(series2):
        # 이 검사는 실제로는 아래 common_index 이후에 더 의미있을 수 있으나, 기본적인 길이 불일치 방지
        raise ValueError("두 시계열의 초기 길이가 동일해야 합니다.")

    s1_clean = series1.dropna()
    s2_clean = series2.dropna()
    
    common_index = s1_clean.index.intersection(s2_clean.index)
    if len(common_index) < 2:
        # print(f"경고: NaN 값을 제거한 후 공통된 데이터 포인트가 너무 적어({len(common_index)}개) 교차 상관관계를 계산할 수 없습니다.")
        return 0, 0.0 

    s1_aligned = s1_clean.loc[common_index].reset_index(drop=True) # ccf는 인덱스 정렬을 가정하지 않으므로 reset_index
    s2_aligned = s2_clean.loc[common_index].reset_index(drop=True)

    if len(s1_aligned) < 2 : # 재확인
        # print(f"경고: NaN 값 정렬 후 두 시계열의 유효 길이가 2 미만({len(s1_aligned)}개)이 되어 교차 상관관계를 계산할 수 없습니다.")
        return 0, 0.0
    
    # ccf 함수는 nlags 파라미터를 통해 0부터 nlags까지의 lag를 계산.
    # max_lag가 N-1보다 크면 ccf 내부에서 N-1로 조정됨.
    # max_lag가 0이면 lag 0만 계산.
    nlags_param = max_lag
    if max_lag is not None and max_lag >= len(s1_aligned):
        nlags_param = len(s1_aligned) - 1
    
    if nlags_param is not None and nlags_param < 0: # max_lag가 음수로 들어올 경우 방지
        nlags_param = 0


    try:
        if nlags_param is None: # 사용자가 max_lag를 지정하지 않은 경우
             cross_corr_func = ccf(s1_aligned, s2_aligned, adjusted=False)
        else: # 사용자가 max_lag (즉, nlags_param)을 지정한 경우
             cross_corr_func = ccf(s1_aligned, s2_aligned, adjusted=False, nlags=nlags_param)
    except Exception as e:
        # print(f"오류: ccf 계산 중 예외 발생 - {e}")
        return 0, 0.0


    if cross_corr_func is None or len(cross_corr_func) == 0:
        # print("경고: 교차 상관 함수 결과가 비어있습니다.")
        return 0, 0.0

    # lag 0부터 nlags_param까지의 결과가 cross_corr_func에 저장됨.
    # 인덱스가 바로 lag 값.
    abs_ccf = np.abs(cross_corr_func)
    best_lag_index = np.argmax(abs_ccf) # 이것이 best_lag (0부터 시작)
    max_abs_ccf_value = cross_corr_func[best_lag_index]
    
    return int(best_lag_index), float(max_abs_ccf_value)

def get_top_lagged_correlations_by_category(
    df: pd.DataFrame, 
    target_variable: str, 
    independent_variable_prefixes: List[str], 
    threshold: float,
    top_n: int = 2,
    max_lag_cross_corr: int = 60
) -> List[dict]:
    """
    카테고리(독립변수 접두사)별로 종속변수와의 교차 상관관계를 계산하고,
    각 카테고리 내에서 상관계수 절댓값이 threshold 이상인 상위 top_n개의 변수를 선택합니다.
    최종 결과는 {'var_name': str, 'best_lag': int, 'best_corr': float} 형태의 딕셔너리 리스트로,
    var_name은 고유하며, 여러 카테고리에서 중복 선택 시 가장 높은 절대 상관계수 값을 가진 정보로 통합됩니다.
    best_lag는 종속변수(target_variable) 대비 독립변수의 시차를 나타냅니다.
    - best_lag > 0: 독립변수가 종속변수보다 'best_lag'만큼 과거일 때 최적 상관관계 (종속변수 선행)
    - best_lag < 0: 독립변수가 종속변수보다 'abs(best_lag)'만큼 미래일 때 최적 상관관계 (독립변수 선행)
    - best_lag = 0: 동시간대 최적 상관관계

    Args:
        df (pd.DataFrame): 분석할 데이터프레임.
        target_variable (str): 종속변수의 컬럼명.
        independent_variable_prefixes (List[str]): 분석할 독립변수들의 카테고리(접두사) 리스트.
        threshold (float): 교차 상관계수 절댓값의 임계값 (0과 1 사이).
        top_n (int, optional): 각 카테고리별로 선택할 상위 변수의 수. Defaults to 2.
        max_lag_cross_corr (int, optional): 
            교차 상관관계 계산 시 고려할 최대 시차 (양수, 음수 방향 모두). 
            0부터 이 값까지의 시차를 확인합니다. Defaults to 60.

    Returns:
        List[dict]: 선택된 변수들의 정보를 담은 딕셔너리 리스트.
                    각 딕셔너리는 'var_name', 'best_lag', 'best_corr' 키를 가집니다.
                    리스트는 'best_corr'의 절대값 기준으로 내림차순 정렬됩니다.
    
    Raises:
        ValueError: target_variable이 df에 없거나, threshold가 유효하지 않거나, target_variable이 수치형이 아닌 경우.
    """
    if target_variable not in df.columns:
        raise ValueError(f"종속변수 '{target_variable}'가 데이터프레임의 컬럼에 존재하지 않습니다.")
    if not (0 <= threshold <= 1):
        raise ValueError("상관관계 임계값은 0과 1 사이의 값이어야 합니다.")
    if target_variable not in df.select_dtypes(include=[np.number]).columns:
        raise ValueError(f"종속변수 '{target_variable}'는 수치형 데이터가 아닙니다.")
    if max_lag_cross_corr < 0:
        raise ValueError("max_lag_cross_corr는 0 이상의 정수여야 합니다.")

    target_series = df[target_variable].copy()
    all_selected_vars_info = {} 

    for prefix in independent_variable_prefixes:
        category_vars_info = []
        matching_columns = [
            col for col in df.columns 
            if col.startswith(prefix) and \
               col != target_variable and \
               col in df.select_dtypes(include=[np.number]).columns
        ]

        if not matching_columns:
            continue

        for col_name in matching_columns:
            independent_series = df[col_name].copy()
            
            try:
                # Case 1: target이 선행하는 경우 (independent가 지연)
                # target_series[t] vs independent_series[t - lag_target_lead]
                # 여기서 lag_target_lead는 independent_series의 '지연' 정도 (0 또는 양수)
                lag_target_lead, corr_target_lead = get_cross_correlation_with_lag(
                    target_series, independent_series, max_lag=max_lag_cross_corr
                )

                # Case 2: independent가 선행하는 경우 (target이 지연)
                # independent_series[t] vs target_series[t - lag_ind_lead]
                # 여기서 lag_ind_lead는 target_series의 '지연' 정도 (0 또는 양수)
                lag_ind_lead, corr_ind_lead = get_cross_correlation_with_lag(
                    independent_series, target_series, max_lag=max_lag_cross_corr
                )
                
                # 절댓값이 더 큰 상관관계 선택
                if abs(corr_target_lead) >= abs(corr_ind_lead):
                    current_best_lag = lag_target_lead  # 양수: independent가 target에 비해 lag_target_lead 만큼 과거 시점일 때 최적.
                    current_best_corr = corr_target_lead
                else:
                    # independent_series가 target_series를 lag_ind_lead 만큼 선행.
                    # 즉, target_series[t] vs independent_series[t + lag_ind_lead] 와 유사한 관계.
                    # 이를 표현하기 위해 lag_ind_lead에 음수를 취함.
                    current_best_lag = -lag_ind_lead  # 음수: independent가 target에 비해 lag_ind_lead 만큼 미래 시점일 때 최적.
                    current_best_corr = corr_ind_lead
                
                if abs(current_best_corr) >= threshold:
                    category_vars_info.append({
                        'var_name': col_name,
                        'best_lag': int(current_best_lag),
                        'best_corr': float(current_best_corr),
                        'abs_corr': abs(current_best_corr) 
                    })
            except (ValueError, Exception) as e:
                continue

        sorted_category_vars = sorted(category_vars_info, key=lambda x: x['abs_corr'], reverse=True)
        
        for var_info in sorted_category_vars[:top_n]:
            var_name = var_info['var_name']
            if var_name not in all_selected_vars_info or \
               var_info['abs_corr'] > all_selected_vars_info[var_name]['abs_corr']:
                all_selected_vars_info[var_name] = var_info

    final_list_of_dicts = [
        {'var_name': info['var_name'], 'best_lag': info['best_lag'], 'best_corr': info['best_corr']}
        for info in all_selected_vars_info.values()
    ]
    
    return sorted(final_list_of_dicts, key=lambda x: abs(x['best_corr']), reverse=True)

def plot_target_correlations_bar_chart(
    df: pd.DataFrame,
    target_variable: str,
    features_to_correlate: Union[List[str], List[Dict[str, Any]]],
    title: str = "타겟 변수와의 상관계수",
    sort_by_abs: bool = True,
    output_path: Optional[str] = None
) -> None:
    """
    지정된 타겟 변수와 다른 특성(feature) 변수들 간의 피어슨 상관계수를 계산하고,
    이를 막대 그래프로 시각화합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임. 수치형 데이터로 구성되어야 합니다.
        target_variable (str): 상관관계를 계산할 기준이 되는 타겟 변수의 컬럼명.
        features_to_correlate (Union[List[str], List[Dict[str, Any]]]):
            타겟 변수와 상관관계를 계산하고 시각화할 특성 변수들의 리스트.
            문자열 리스트이거나, 각 항목이 'var_name' 키를 가진 딕셔너리 리스트일 수 있습니다.
        title (str, optional): 그래프의 제목. Defaults to "타겟 변수와의 상관계수".
        sort_by_abs (bool, optional): 
            True이면 상관계수의 절대값 기준으로 내림차순 정렬, 
            False이면 상관계수 값 자체를 기준으로 내림차순 정렬합니다. Defaults to True.
        output_path (Optional[str], optional): 그래프를 저장할 경로. Defaults to None.
    """
    if target_variable not in df.columns:
        print(f"경고: 타겟 변수 '{target_variable}'가 데이터프레임에 없습니다.")
        return
    if not pd.api.types.is_numeric_dtype(df[target_variable]):
        print(f"경고: 타겟 변수 '{target_variable}'는 수치형 데이터가 아닙니다.")
        return

    if not features_to_correlate:
        print("정보: 상관관계를 분석할 특성 변수가 없습니다.")
        return

    actual_feature_names: List[str]
    if all(isinstance(item, str) for item in features_to_correlate):
        actual_feature_names = features_to_correlate
    elif all(isinstance(item, dict) and 'var_name' in item for item in features_to_correlate):
        try:
            actual_feature_names = [item['var_name'] for item in features_to_correlate]
        except KeyError:
            print("경고: 'features_to_correlate'의 일부 딕셔너리에 'var_name' 키가 없습니다.")
            return
        except Exception as e:
            print(f"경고: 'features_to_correlate' 처리 중 오류: {e}")
            return
    else:
        print("경고: 'features_to_correlate'는 문자열 리스트 또는 'var_name' 키를 가진 딕셔너리 리스트여야 합니다.")
        return

    valid_features = []
    for feature in actual_feature_names:
        if feature == target_variable:
            continue # 타겟 변수 자신과의 상관관계는 1이므로 제외하거나 필요시 포함
        if feature not in df.columns:
            print(f"경고: 특성 변수 '{feature}'가 데이터프레임에 없어 제외됩니다.")
            continue
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"경고: 특성 변수 '{feature}'는 수치형이 아니라 제외됩니다.")
            continue
        valid_features.append(feature)

    if not valid_features:
        print("정보: 타겟 변수와 상관관계를 계산할 유효한 특성 변수가 없습니다.")
        return

    correlations = {}
    target_series = df[target_variable]
    for feature_name in valid_features:
        feature_series = df[feature_name]
        correlations[feature_name] = target_series.corr(feature_series)
    
    if not correlations:
        print("정보: 상관계수를 계산할 수 있는 특성 변수가 없습니다.")
        return

    correlation_series = pd.Series(correlations)

    if sort_by_abs:
        sorted_correlations = correlation_series.reindex(correlation_series.abs().sort_values(ascending=False).index)
    else:
        sorted_correlations = correlation_series.sort_values(ascending=False)

    num_features = len(sorted_correlations)
    fig_height = max(8, num_features * 0.6)  # 높이 증가
    fig_width = max(10, num_features * 0.4 + 8)  # 너비 증가

    plt.figure(figsize=(fig_width, fig_height))
    bars = sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index, color='#404040', orient='h')  # 진한 회색으로 통일
    
    # 막대에 값 표시
    for bar in bars.patches:
        value = bar.get_width()
        x_pos = value + (0.02 * plt.xlim()[1] if value >= 0 else -0.02 * plt.xlim()[0] -0.08) # 위치 조정
        y_pos = bar.get_y() + bar.get_height() / 2
        bars.text(x_pos, y_pos, f"{value:.2f}", va='center', ha='left' if value >=0 else 'right', fontsize=9)

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel(f"'{target_variable}'와의 최적 상관계수", fontsize=12)
    plt.ylabel("거시경제 변수", fontsize=12)
    plt.axvline(0, color='grey', linewidth=0.8) # 0점선 추가
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"그래프가 {output_path}에 저장되었습니다.")
    
    plt.show()

def run_process(
    df: pd.DataFrame,
    target_variable: str,
    independent_variable_prefixes: List[str],
    corr_threshold: float = 0.2,
    visualize: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    상관분석 전체 파이프라인을 실행하고 주요 결과를 반환합니다.
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        target_variable (str): 타겟 변수명
        independent_variable_prefixes (List[str]): 독립변수 접두사 리스트
        corr_threshold (float, optional): 상관계수 임계값. Defaults to 0.2.
        visualize (bool, optional): 시각화 실행 여부. Defaults to True.
        output_path (Optional[str], optional): 그래프 저장 경로. Defaults to None.

    Returns:
        Dict[str, Any]: 분석 결과를 담은 딕셔너리
            - significant_variables: 유의미한 변수 리스트
            - correlation_series: 상관계수 시리즈 (딕셔너리로 변환)
            - cross_correlation_results: 교차 상관관계 분석 결과
    """
    print("[1/2] 피어슨 상관분석 수행...")
    sig_vars, corr_series = get_significant_correlations_with_target(
        df, target_variable, corr_threshold, independent_variable_prefixes
    )
    print(f"  - 임계값 이상 변수 수: {len(sig_vars)-1}")

    # 교차 상관관계 분석 수행
    print("[2/3] 교차 상관관계 분석 수행...")
    cross_corr_dict = get_top_lagged_correlations_by_category(
        df=df,
        target_variable=target_variable,
        independent_variable_prefixes=independent_variable_prefixes,
        threshold=corr_threshold,
        top_n=2,
        max_lag_cross_corr=60
    )
    print(f"  - 교차 상관관계 분석 완료: {len(cross_corr_dict)}개 변수")

    if visualize:
        print("[3/3] 상관계수 시각화...")
        from datetime import datetime
        import os
        import json
        
        # output_path가 None인 경우 기본 경로 설정
        if output_path is None:
            # output 디렉토리 생성
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.join(output_dir, f"correlation_{target_variable}_{timestamp}")
            
            # # 상관계수 막대 그래프 저장
            # plot_target_correlations_bar_chart(
            #     df, target_variable, sig_vars[1:],  # 첫 번째는 타겟 변수 자신이므로 제외
            #     title=f"{target_variable}와의 상관계수",
            #     output_path=f"{base_path}_1.png"
            # )
            
            # 교차 상관관계 막대 그래프 저장
            plot_target_correlations_bar_chart(
                df, target_variable, cross_corr_dict,
                title=f"{target_variable}와의 교차 상관계수",
                output_path=f"{base_path}.png"
            )
            
            # JSON 결과 저장
            results = {
                'significant_variables': sig_vars,
                'correlation_series': corr_series.to_dict(),
                'cross_correlation_results': cross_corr_dict
            }
            json_path = f"{base_path}_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"상관분석 결과가 {json_path}에 저장되었습니다.")
        else:
            # 사용자가 지정한 경로가 있는 경우
            output_dir = os.path.dirname(output_path)
            if output_dir:  # 디렉토리가 지정된 경우
                os.makedirs(output_dir, exist_ok=True)
            
            base_path = output_path.rsplit('.', 1)[0]  # 확장자 제거
            
            # # 상관계수 막대 그래프 저장 (_1 접미사)
            # plot_target_correlations_bar_chart(
            #     df, target_variable, sig_vars[1:],  # 첫 번째는 타겟 변수 자신이므로 제외
            #     title=f"{target_variable}와의 상관계수",
            #     output_path=f"{base_path}_1.png"
            # )
            
            # 교차 상관관계 막대 그래프 저장 (_2 접미사)
            plot_target_correlations_bar_chart(
                df, target_variable, cross_corr_dict,
                title=f"{target_variable}와의 교차 상관계수",
                output_path=f"{base_path}.png"
            )
            
            # JSON 결과 저장
            results = {
                'significant_variables': sig_vars,
                'correlation_series': corr_series.to_dict(),
                'cross_correlation_results': cross_corr_dict
            }
            json_path = f"{base_path}_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"상관분석 결과가 {json_path}에 저장되었습니다.")

    print("\n[완료] 상관분석 및 시각화가 종료되었습니다.")
    
    # 결과를 딕셔너리로 반환
    return sig_vars, corr_series.to_dict(), cross_corr_dict



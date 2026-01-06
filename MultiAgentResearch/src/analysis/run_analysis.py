import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import pickle
import json
import sys

# src 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from analysis.data_loader import run_process as data_loader_run_process
from analysis.correlation import run_process as correlation_run_process, plot_target_correlations_bar_chart
from analysis.granger import run_process as granger_run_process
from analysis.rf_model import run_process as rf_model_run_process

def convert_to_json_serializable(obj: Any) -> Any:
    """
    객체를 JSON 직렬화 가능한 형태로 변환합니다.
    
    Args:
        obj (Any): 변환할 객체
        
    Returns:
        Any: JSON 직렬화 가능한 객체
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    else:
        return obj

def run_full_analysis(
    data_path: str,
    target_variable: str,
    independent_variables: List[str],
    output_dir: str = "artifacts",
    corr_threshold: float = 0.2,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    전체 분석 파이프라인을 실행하고 결과를 저장합니다.

    Args:
        data_path (str): 데이터 파일 경로
        target_variable (str): 타겟 변수명
        independent_variables (List[str]): 독립변수 접두사 리스트
        output_dir (str, optional): 결과 저장 디렉토리. Defaults to "artifacts".
        corr_threshold (float, optional): 상관계수 임계값. Defaults to 0.2.
        visualize (bool, optional): 시각화 실행 여부. Defaults to True.

    Returns:
        Dict[str, Any]: 모든 분석 결과를 담은 딕셔너리
    """
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_path = os.path.join(output_dir, f"analysis_{target_variable}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 데이터 로딩 및 전처리 시작 ===")
    data, features, scaled_data, scaler, q_agg_data = data_loader_run_process(
        path=data_path,
        sales_translated_in_energy=target_variable,
        independent_variables=independent_variables
    )
    print("데이터 로딩 및 전처리 완료")

    # q_agg_data를 CSV로 저장
    q_agg_data_path = f"{base_output_path}_quarterly_data.csv"
    q_agg_data.to_csv(q_agg_data_path, index=True)
    print(f"분기별 집계 데이터가 {q_agg_data_path}에 저장되었습니다.")

    # scaler를 pickle로 저장
    scaler_path = f"{base_output_path}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"스케일러가 {scaler_path}에 저장되었습니다.")

    print("\n=== 상관관계 분석 시작 ===")
    # 상관관계 분석 실행 (시각화 없이)
    sig_vars, corr_series, cross_corr_dict = correlation_run_process(
        df=q_agg_data,
        target_variable=target_variable,
        independent_variable_prefixes=independent_variables,
        corr_threshold=corr_threshold,
        visualize=False  # 시각화는 별도로 수행
    )
    
    # 교차 상관관계 결과를 사용하여 시각화
    if visualize:
        correlation_output_path = f"{base_output_path}_correlation.png"
        plot_target_correlations_bar_chart(
            df=q_agg_data,
            target_variable=target_variable,
            features_to_correlate=cross_corr_dict,  # 교차 상관관계 결과 사용
            title=f"{target_variable}와의 상관계수",
            output_path=correlation_output_path
        )
    print("상관관계 분석 완료")

    print("\n=== 그레인저 인과성 분석 시작 ===")
    granger_results = granger_run_process(
        df=q_agg_data,
        target_variable=target_variable,
        independent_variable_prefixes=independent_variables,
        cross_corr_threshold=corr_threshold,
        num_top_corr_vars=5,
        max_lag_cross_corr=60,
        max_lag_granger=12,
        granger_significance_level=0.05,
        apply_diff_to_granger=True,
        diff_periods_granger=1,
        visualize=visualize,
        output_path=f"{base_output_path}_granger.png"
    )
    print("그레인저 인과성 분석 완료")

    print("\n=== 랜덤 포레스트 분석 시작 ===")
    rf_importance, rf_lag_results = rf_model_run_process(
        df=q_agg_data,
        target_variable=target_variable,
        cross_corr_dict=cross_corr_dict,
        visualize=visualize,
        output_path=f"{base_output_path}_rf.png"
    )
    print("랜덤 포레스트 분석 완료")

    # 분석 결과를 JSON으로 저장하기 위한 딕셔너리 구성
    results = {
        'correlation': {
            'significant_variables': sig_vars,
            'correlation_series': corr_series.to_dict(),
            'cross_correlation_dict': cross_corr_dict  # 리스트 형태 그대로 저장
        },
        'granger': granger_results,
        'random_forest': {
            'feature_importance': rf_importance.to_dict(),
            'lag_results': {str(k): v.to_dict() for k, v in rf_lag_results.items()}
        }
    }

    # JSON 직렬화 가능한 형태로 변환
    json_results = convert_to_json_serializable(results)

    # 결과를 JSON 파일로 저장
    results_path = f"{base_output_path}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)
    print(f"\n분석 결과가 {results_path}에 저장되었습니다.")

    return results

def main():
    # 분석 설정
    data_path = "../../data/merged_na_df.csv"
    target_variable = "quarterly_MWh"
    independent_variables = [
        "interest_rates", "gdp_income", "employment",
        "price_indices", "exchange_rates",
        "liquidity_and_reserves", "auto_loans"
    ]
    
    # 전체 분석 실행
    results = run_full_analysis(
        data_path=data_path,
        target_variable=target_variable,
        independent_variables=independent_variables,
        corr_threshold=0.2,
        visualize=True
    )

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json

def select_features_from_cross_correlation(
    cross_corr_dict: Dict[str, Dict[str, float]],
) -> List[str]:
    """
    Cross Correlation 결과에서 유의미한 특성을 선택합니다.

    Args:
        cross_corr_dict (Dict[str, Dict[str, float]]): Cross Correlation 결과
            - 딕셔너리 형태: {'feature_name': {'lag': correlation_value}}
            - 또는 리스트 형태: [{'var_name': 'feature_name', ...}]

    Returns:
        List[str]: 선택된 특성 변수명 리스트
    """
    selected_features = []
    
    # 딕셔너리 형태인 경우
    if isinstance(cross_corr_dict, dict):
        selected_features = list(cross_corr_dict.keys())
    # 리스트 형태인 경우
    elif isinstance(cross_corr_dict, list):
        for item in cross_corr_dict:
            if isinstance(item, dict) and 'var_name' in item:
                selected_features.append(item['var_name'])
            elif isinstance(item, str):
                selected_features.append(item)
    else:
        raise ValueError("cross_corr_dict는 딕셔너리 또는 리스트 형태여야 합니다.")
            
    return selected_features
    
def train_random_forest(
    df: pd.DataFrame,
    target_variable: str,
    feature_variables: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
) -> Tuple[RandomForestRegressor, Dict[str, float], Dict[str, float]]:
    """
    랜덤포레스트 모델을 학습하고 성능을 평가합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        target_variable (str): 타겟 변수명
        feature_variables (List[str]): 특성 변수명 리스트
        test_size (float, optional): 테스트 데이터 비율. Defaults to 0.2
        random_state (int, optional): 랜덤 시드. Defaults to 42
        n_estimators (int, optional): 트리 개수. Defaults to 100
        max_depth (Optional[int], optional): 최대 깊이. Defaults to None
        min_samples_split (int, optional): 분할에 필요한 최소 샘플 수. Defaults to 2
        min_samples_leaf (int, optional): 리프 노드에 필요한 최소 샘플 수. Defaults to 1

    Returns:
        Tuple[RandomForestRegressor, Dict[str, float], Dict[str, float]]:
            - 학습된 랜덤포레스트 모델
            - 학습 데이터 성능 지표 (MSE, R2)
            - 테스트 데이터 성능 지표 (MSE, R2)
    """
    # 데이터 준비
    X = df[feature_variables]
    y = df[target_variable]

    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 모델 학습
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    # 성능 평가
    train_pred = rf_model.predict(X_train)
    test_pred = rf_model.predict(X_test)

    train_metrics = {
        'mse': mean_squared_error(y_train, train_pred),
        'r2': r2_score(y_train, train_pred)
    }

    test_metrics = {
        'mse': mean_squared_error(y_test, test_pred),
        'r2': r2_score(y_test, test_pred)
    }

    return rf_model, train_metrics, test_metrics

def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: List[str],
    plot: bool = True,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    랜덤포레스트 모델의 특성 중요도를 계산하고 시각화합니다.

    Args:
        model (RandomForestRegressor): 학습된 랜덤포레스트 모델
        feature_names (List[str]): 특성 변수명 리스트
        plot (bool, optional): 중요도 시각화 여부. Defaults to True
        top_n (Optional[int], optional): 상위 N개 특성만 표시. Defaults to None

    Returns:
        pd.DataFrame: 특성 중요도 데이터프레임
    """
    # 특성 중요도 계산
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # 중요도 기준 내림차순 정렬
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # top_n이 지정된 경우 상위 N개만 선택
    if top_n is not None:
        feature_importance = feature_importance.head(top_n)

    # 시각화
    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    return feature_importance

def analyze_feature_importance_with_lags(
    df: pd.DataFrame,
    target_variable: str,
    feature_variables: List[str],
    max_lag: int = 12,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    plot: bool = True
) -> Dict[int, pd.DataFrame]:
    """
    다양한 시차에 대한 특성 중요도를 분석합니다.

    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        target_variable (str): 타겟 변수명
        feature_variables (List[str]): 특성 변수명 리스트
        max_lag (int, optional): 최대 시차. Defaults to 12
        test_size (float, optional): 테스트 데이터 비율. Defaults to 0.2
        random_state (int, optional): 랜덤 시드. Defaults to 42
        n_estimators (int, optional): 트리 개수. Defaults to 100
        plot (bool, optional): 중요도 시각화 여부. Defaults to True

    Returns:
        Dict[int, pd.DataFrame]: 시차별 특성 중요도 데이터프레임 딕셔너리
    """
    lag_importance_results = {}

    for lag in range(max_lag + 1):
        # 시차 데이터 준비
        lagged_df = df.copy()
        for feature in feature_variables:
            lagged_df[f'{feature}_lag_{lag}'] = lagged_df[feature].shift(lag)
        
        # NaN 제거
        lagged_df = lagged_df.dropna()
        
        # 시차 특성 변수명 생성
        lagged_features = [f'{feature}_lag_{lag}' for feature in feature_variables]
        
        # 모델 학습
        model, train_metrics, test_metrics = train_random_forest(
            df=lagged_df,
            target_variable=target_variable,
            feature_variables=lagged_features,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators
        )
        
        # 특성 중요도 계산 (plot=False로 설정하여 개별 시차별 plot 비활성화)
        importance_df = get_feature_importance(
            model=model,
            feature_names=lagged_features,
            plot=False
        )
        
        lag_importance_results[lag] = importance_df

    return lag_importance_results

def plot_lag_importance_heatmap(
    lag_importance_results: Dict[int, pd.DataFrame],
    feature_variables: List[str],
    top_n: int = 5,
    output_path: Optional[str] = None
) -> None:
    """
    시차별 특성 중요도를 히트맵으로 시각화합니다.

    Args:
        lag_importance_results (Dict[int, pd.DataFrame]): 시차별 특성 중요도 결과
        feature_variables (List[str]): 원본 특성 변수명 리스트
        top_n (int, optional): 상위 N개 특성만 표시. Defaults to 5
        output_path (Optional[str], optional): 그래프 저장 경로. Defaults to None
    """
    # 히트맵 데이터 준비
    heatmap_data = pd.DataFrame(index=feature_variables)
    
    for lag, importance_df in lag_importance_results.items():
        # 원본 특성명 추출
        importance_df['original_feature'] = importance_df['feature'].str.split('_lag_').str[0]
        
        # 각 원본 특성별 최대 중요도 추출
        max_importance = importance_df.groupby('original_feature')['importance'].max()
        heatmap_data[f'lag_{lag}'] = max_importance
    
    # 상위 N개 특성만 선택
    if top_n is not None:
        mean_importance = heatmap_data.mean(axis=1)
        top_features = mean_importance.nlargest(top_n).index
        heatmap_data = heatmap_data.loc[top_features]
    
    # 히트맵 시각화
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Feature Importance by Lag')
    plt.xlabel('Lag')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"그래프가 {output_path}에 저장되었습니다.")
    
    plt.show()

def run_process(
    df: pd.DataFrame, 
    target_variable: str, 
    cross_corr_dict: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    visualize: bool = True
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    랜덤 포레스트 모델 학습 및 특성 중요도 분석을 실행하는 함수

    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        target_variable (str): 타겟 변수명
        cross_corr_dict (Dict[str, pd.DataFrame]): 교차 상관관계 분석 결과
        output_path (Optional[str], optional): 그래프 저장 경로. Defaults to None
        visualize (bool, optional): 시각화 실행 여부. Defaults to True

    Returns:
        Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]: 
            - 특성 중요도 데이터프레임
            - 시차별 특성 중요도 결과 딕셔너리
    """
    # 교차 상관관계 기반 특성 선택
    features = select_features_from_cross_correlation(cross_corr_dict)

    # 기본 모델 학습 및 특성 중요도 분석
    model, train_metrics, test_metrics = train_random_forest(
        df=df,
        target_variable=target_variable,
        feature_variables=features
    )

    # 특성 중요도 확인
    importance_df = get_feature_importance(
        model=model,
        feature_names=features,
        top_n=5,
        plot=visualize
    )

    # 시차별 특성 중요도 분석
    lag_results = analyze_feature_importance_with_lags(
        df=df,
        target_variable=target_variable,
        feature_variables=features,
        max_lag=12
    )

    # 시각화가 활성화된 경우에만 그래프 생성 및 저장
    if visualize:
        from datetime import datetime
        import os
        
        # output_path가 None인 경우 기본 경로 설정
        if output_path is None:
            # output 디렉토리 생성
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.join(output_dir, f"rf_analysis_{target_variable}_{timestamp}")
            
            # 특성 중요도 그래프 저장
            importance_output_path = f"{base_path}_1.png"
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(importance_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"특성 중요도 그래프가 {importance_output_path}에 저장되었습니다.")
            
            # 시차별 중요도 히트맵 저장
            heatmap_output_path = f"{base_path}_2.png"
            plot_lag_importance_heatmap(
                lag_importance_results=lag_results,
                feature_variables=features,
                top_n=5,
                output_path=heatmap_output_path
            )
            
            # JSON 결과 저장
            results = {
                'feature_importance': importance_df.to_dict(),
                'lag_results': {str(k): v.to_dict() for k, v in lag_results.items()},
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            json_path = f"{base_path}_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"랜덤 포레스트 분석 결과가 {json_path}에 저장되었습니다.")
        else:
            # 사용자가 지정한 경로가 있는 경우
            output_dir = os.path.dirname(output_path)
            if output_dir:  # 디렉토리가 지정된 경우
                os.makedirs(output_dir, exist_ok=True)
            
            base_path = output_path.rsplit('.', 1)[0]  # 확장자 제거
            
            # 특성 중요도 그래프 저장 (_1 접미사)
            importance_output_path = f"{base_path}_1.png"
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(importance_output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"특성 중요도 그래프가 {importance_output_path}에 저장되었습니다.")
            
            # 시차별 중요도 히트맵 저장 (_2 접미사)
            heatmap_output_path = f"{base_path}_2.png"
            plot_lag_importance_heatmap(
                lag_importance_results=lag_results,
                feature_variables=features,
                top_n=5,
                output_path=heatmap_output_path
            )
            
            # JSON 결과 저장
            results = {
                'feature_importance': importance_df.to_dict(),
                'lag_results': {str(k): v.to_dict() for k, v in lag_results.items()},
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            json_path = f"{base_path}_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"랜덤 포레스트 분석 결과가 {json_path}에 저장되었습니다.")
    
    return importance_df, lag_results

def main():
    # 데이터 로드 및 전처리
    path = "../../data/merged_na_df.csv"
    sales_translated_in_energy = "quarterly_MWh"
    independent_variables = ["interest_rates", "gdp_income", "employment", 
                            "price_indices", "exchange_rates", 
                            "liquidity_and_reserves", "auto_loans"]
    
    # 데이터 처리 파이프라인 실행
    data, features, scaled_data, scaler, q_agg_data = run_process(
        path=path,
        sales_translated_in_energy=sales_translated_in_energy,
        independent_variables=independent_variables
    )
    
    # 교차 상관관계 분석
    cross_corr_dict = {}
    for feature in features:
        if feature != sales_translated_in_energy:
            cross_corr_dict[feature] = calculate_cross_correlation(
                data[q_agg_data], 
                sales_translated_in_energy, 
                feature
            )
    
    # 랜덤 포레스트 모델 실행
    run_process(
        df=q_agg_data,
        target_variable=sales_translated_in_energy,
        cross_corr_dict=cross_corr_dict
    )

if __name__ == "__main__":
    main()

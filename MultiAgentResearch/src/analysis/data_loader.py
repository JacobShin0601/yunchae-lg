import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import logging
import shutil

logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    데이터를 로드하되 스케일링은 수행하지 않는 함수
    
    Args:
        file_path (str): 데이터 파일 경로
        
    Returns:
        pd.DataFrame: 로드된 원본 데이터
    """
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    return df

def scale_data(df: pd.DataFrame, features: List[str]) -> tuple:
    """
    지정된 특성들에 대해서만 Min-Max 스케일링을 수행하는 함수
    
    Args:
        df (pd.DataFrame): 스케일링할 데이터프레임
        features (List[str]): 스케일링할 특성 변수명 리스트
        
    Returns:
        tuple: (스케일링된 데이터, 스케일러 객체)
    """
    # features가 None이거나 비어있는 경우 처리
    if features is None or len(features) == 0:
        raise ValueError("features 리스트가 비어있거나 None입니다.")
    
    # 스케일링하지 않을 변수들
    exclude_columns = ['original_index', 'year', 'quarter']
    
    # features가 실제로 데이터프레임에 존재하는지 확인
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        raise ValueError("주어진 features가 데이터프레임에 존재하지 않습니다.")
    
    # 수치형 데이터만 선택
    numeric_df = df[valid_features].select_dtypes(include=[np.number])
    
    # 스케일링하지 않을 변수들 제외
    numeric_df = numeric_df.drop(columns=[col for col in exclude_columns if col in numeric_df.columns])
    
    # 스케일링하지 않을 컬럼들 저장
    non_scaled_columns = [col for col in df.columns if col not in numeric_df.columns]
    non_scaled_data = df[non_scaled_columns].copy()
    
    # MinMaxScaler 초기화 및 스케일링
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # 스케일링된 데이터를 DataFrame으로 변환
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)
    
    # 스케일링하지 않은 데이터와 결합
    final_df = pd.concat([non_scaled_data, scaled_df], axis=1)
    
    # 원래 컬럼 순서 유지
    final_df = final_df[df.columns]
    
    return final_df, scaler

def extract_variables_by_prefix(df: pd.DataFrame, independent_variables: List[str]) -> List[str]:
    """
    주어진 접두사 리스트에 해당하는 변수들을 추출하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임
        independent_variables (List[str]): 변수 접두사 리스트
        
    Returns:
        List[str]: 추출된 변수명 리스트
    """
    extracted_variables = []
    
    for prefix in independent_variables:
        matching_columns = [col for col in df.columns if col.startswith(prefix)]
        extracted_variables.extend(matching_columns)
        
    return extracted_variables

def inverse_transform_data(scaled_data, scaler, original_columns):
    """
    스케일링된 데이터를 원래 스케일로 되돌리는 함수
    
    Args:
        scaled_data (numpy.ndarray): 스케일링된 데이터
        scaler (MinMaxScaler): 사용된 스케일러 객체
        original_columns (list): 원본 컬럼명 리스트
        
    Returns:
        pd.DataFrame: 원래 스케일로 변환된 데이터
    """
    # 역변환 수행
    original_data = scaler.inverse_transform(scaled_data)
    
    # DataFrame으로 변환
    original_df = pd.DataFrame(original_data, columns=original_columns)
    
    return original_df


def aggregate_data_by_quarter(df: pd.DataFrame, features: List[str], target_column: str = 'quarterly_MWh') -> pd.DataFrame:
    """
    데이터를 year_Q별로 집계하는 함수
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        features (List[str]): 집계할 특성 컬럼 리스트
        target_column (str): 타겟 컬럼명 (기본값: 'quarterly_MWh')
        
    Returns:
        pd.DataFrame: year_Q별로 집계된 데이터프레임
    """
    # year_Q 컬럼이 없다면 생성
    if 'year_Q' not in df.columns:
        df['year_Q'] = df['year'].astype(str) + '_Q' + df['quarter'].astype(str)
    
    # 집계할 컬럼들 준비
    agg_columns = [target_column] + features
    
    # 집계 함수 정의
    agg_dict = {
        target_column: 'sum'  # 타겟 변수는 합계
    }
    # 나머지 변수들은 평균
    for col in features:
        agg_dict[col] = 'mean'
    
    # year_Q별로 집계
    aggregated_df = df.groupby('year_Q').agg(agg_dict).reset_index()
    
    return aggregated_df


def run_process(path: str,
               sales_translated_in_energy: str,
               independent_variables: List[str],
               save_as_csv: bool = False) -> Tuple[pd.DataFrame, List[str], np.ndarray, MinMaxScaler, pd.DataFrame]:
    """
    데이터 처리 파이프라인을 실행하는 함수
    
    Args:
        path (str): 데이터 파일 경로
        sales_translated_in_energy (str): 에너지 판매량 컬럼명
        independent_variables (List[str]): 독립 변수 리스트
        save_as_csv (bool): 처리된 데이터를 CSV로 저장할지 여부 (기본값: False)
        
    Returns:
        Tuple[pd.DataFrame, List[str], np.ndarray, MinMaxScaler, pd.DataFrame]: 
            - 원본 데이터
            - 특성 컬럼 리스트
            - 스케일링된 데이터
            - 스케일러 객체
            - 분기별 집계 데이터
    """
    data = load_data(path)
    ind_vars = extract_variables_by_prefix(data, independent_variables)
    features = [sales_translated_in_energy] + ind_vars
    scaled_data, scaler = scale_data(data, features)
    q_agg_data = aggregate_data_by_quarter(data, features, target_column=sales_translated_in_energy)
    
    # CSV 저장 옵션이 활성화된 경우
    if save_as_csv:
        # 저장할 디렉토리 생성
        output_dir = "../../artifacts"
        os.makedirs(output_dir, exist_ok=True)
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 스케일링된 데이터 저장
        scaled_df = pd.DataFrame(scaled_data, columns=features)
        scaled_csv_path = os.path.join(output_dir, f"scaled_data_{timestamp}.csv")
        scaled_df.to_csv(scaled_csv_path, index=False)
        logger.info(f"스케일링된 데이터가 저장되었습니다: {scaled_csv_path}")
        
        # 분기별 집계 데이터 저장
        q_agg_csv_path = os.path.join(output_dir, f"quarterly_aggregated_{timestamp}.csv")
        q_agg_data.to_csv(q_agg_csv_path, index=False)
        logger.info(f"분기별 집계 데이터가 저장되었습니다: {q_agg_csv_path}")
    
    return data, features, scaled_data, scaler, q_agg_data

def _load_esd_data(path: str = "../../data/ESD_preprocessed_data_20250418.xlsx"):
    df = pd.read_excel(path)
    df.insert(0, 'original_index', df.index)
    return df

def _process_vehicle_volume_data(ihs_df: pd.DataFrame) -> pd.DataFrame:
    df = ihs_df.copy()
    
    yearly_volume_cols = [col for col in df.columns if 'Vehicle Volume' in col]
    if not yearly_volume_cols:
        raise ValueError("Vehicle Volume 컬럼이 없습니다.")
    
    id_vars = [col for col in df.columns if col not in yearly_volume_cols]
    yearly_volume_df = df.melt(id_vars=id_vars, value_vars=yearly_volume_cols, var_name="yearly_vehicle_volume", value_name="Vehicle_Volume_Value")
    
    # 컬럼명을 다시 "Vehicle Volume"으로 변경
    yearly_volume_df = yearly_volume_df.rename(columns={"Vehicle_Volume_Value": "Vehicle Volume"})
    
    yearly_volume_df['year'] = yearly_volume_df['yearly_vehicle_volume'].str.extract(r"(\d{4})").astype(int)
    yearly_volume_df = yearly_volume_df.drop(columns=['yearly_vehicle_volume'])

    result_df = yearly_volume_df.sort_values(by=['original_index', 'year'], ascending=[True, True]).reset_index(drop=True)
    return result_df

def _filter_vehicle_volume_data(ihs_df: pd.DataFrame) -> pd.DataFrame:
    df = ihs_df.copy()
    extracted_cols = ['year', 'Region(최종)', 'Type(최종)', 'OEM 그룹(최종)', '브랜드계열', '양극재', 'POS(MI)', 'Battery Pack Supplier', 'Battery Pack Supplier Country/Territory', 'Battery Pack Supplier Plant', 'Vehicle Volume']
    filtered_df = df[extracted_cols]
    return filtered_df

def get_prate_data(path: Optional[str] = "../../data/ESD_preprocessed_data_20250418.xlsx") -> pd.DataFrame:
    ihs_df = _load_esd_data(path)
    df = _process_vehicle_volume_data(ihs_df)
    df = _filter_vehicle_volume_data(df)

    ev_df = df[df['Type(최종)'].isin(['EV', 'PHEV'])]
    non_ev_df = df[df['Type(최종)'].isin(['ICE', 'FCEV', 'HEV', '48V', '12V'])]

    ev_yearly_df = ev_df.groupby('year').agg({'Vehicle Volume': 'sum'}).reset_index()
    non_ev_yearly_df = non_ev_df.groupby('year').agg({'Vehicle Volume': 'sum'}).reset_index()

    return df, ev_df, non_ev_df

def remove_artifact_folder(folder_path: str = "../../artifacts/") -> None:
    """
    ./artifact/ 폴더가 존재하면 삭제하는 함수
    
    Args:
        folder_path (str): 삭제할 폴더 경로
    """
    if os.path.exists(folder_path):
        logger.info(f"'{folder_path}' 폴더를 삭제합니다...")
        try:
            shutil.rmtree(folder_path)
            logger.info(f"'{folder_path}' 폴더가 성공적으로 삭제되었습니다.")
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            raise
    else:
        logger.info(f"'{folder_path}' 폴더가 존재하지 않습니다.")


def main():
    path = "../../data/merged_na_df.csv"
    sales_translated_in_energy = "quarterly_MWh"
    independent_variables = ["interest_rates", "gdp_income", "employment", 
                           "price_indices", "exchange_rates", 
                           "liquidity_and_reserves", "auto_loans"]
    
    data, features, scaled_data, scaler, q_agg_data = run_process(
        path=path,
        sales_translated_in_energy=sales_translated_in_energy,
        independent_variables=independent_variables
    )

if __name__ == "__main__":
    main()

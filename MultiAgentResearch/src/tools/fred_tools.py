import os
import json
import logging
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from src.tools.decorators import tool
from src.agents.agents import Colors

# 로거 설정
logger = logging.getLogger(__name__)
logger.propagate = False
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class FredDataCollector:
    def __init__(self, api_key: Optional[str] = None):
        """
        FRED API를 사용하여 거시경제 데이터를 수집하는 클래스
        
        Args:
            api_key (str, optional): FRED API 키. 환경 변수에서 가져오거나 직접 제공
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API 키가 필요합니다. 환경 변수 FRED_API_KEY를 설정하거나 직접 제공하세요.")
        
        self.fred = Fred(api_key=self.api_key)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_series_data(self, series_id: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        FRED에서 시계열 데이터를 가져옴
        
        Args:
            series_id (str): FRED 시리즈 ID
            start_date (str, optional): 시작 날짜 (YYYY-MM-DD)
            end_date (str, optional): 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 시계열 데이터
        """
        try:
            # 기본 날짜 설정
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
            
            # 데이터 가져오기
            data = self.fred.get_series(series_id, start_date=start_date, end_date=end_date)
            
            # DataFrame으로 변환
            df = pd.DataFrame(data)
            df.columns = [series_id]
            df.index.name = 'date'
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 가져오기 실패 (시리즈 ID: {series_id}): {str(e)}")
            raise
    
    def save_series_data(self, series_id: str, df: pd.DataFrame) -> None:
        """
        시계열 데이터를 CSV 파일로 저장
        
        Args:
            series_id (str): FRED 시리즈 ID
            df (pd.DataFrame): 저장할 데이터프레임
        """
        try:
            filename = f"fred_{series_id}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            logger.info(f"데이터 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"데이터 저장 실패 (시리즈 ID: {series_id}): {str(e)}")
            raise
    
    def collect_and_save_series(self, series_ids: List[str], 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        여러 시리즈의 데이터를 수집하고 저장
        
        Args:
            series_ids (List[str]): FRED 시리즈 ID 리스트
            start_date (str, optional): 시작 날짜
            end_date (str, optional): 종료 날짜
            
        Returns:
            Dict[str, pd.DataFrame]: 시리즈 ID를 키로 하는 데이터프레임 딕셔너리
        """
        collected_data = {}
        
        for series_id in series_ids:
            try:
                df = self.get_series_data(series_id, start_date, end_date)
                self.save_series_data(series_id, df)
                collected_data[series_id] = df
                logger.info(f"시리즈 {series_id} 데이터 수집 및 저장 완료")
                
            except Exception as e:
                logger.error(f"시리즈 {series_id} 처리 실패: {str(e)}")
                continue
        
        return collected_data

# FRED 도구 설정
fred_tool_config = {
    "name": "fred_data_collector",
    "description": "FRED API를 사용하여 거시경제 데이터를 수집하고 저장하는 도구",
    "parameters": {
        "type": "object",
        "properties": {
            "series_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "수집할 FRED 시리즈 ID 리스트"
            },
            "start_date": {
                "type": "string",
                "description": "데이터 수집 시작 날짜 (YYYY-MM-DD)"
            },
            "end_date": {
                "type": "string",
                "description": "데이터 수집 종료 날짜 (YYYY-MM-DD)"
            }
        },
        "required": ["series_ids"]
    }
}

@tool(fred_tool_config)
def process_fred_tool(tool_input: Dict) -> Dict:
    """
    FRED 도구 처리 함수
    
    Args:
        tool_input (Dict): 도구 입력 파라미터
        
    Returns:
        Dict: 처리 결과
    """
    try:
        collector = FredDataCollector()
        series_ids = tool_input.get('series_ids', [])
        start_date = tool_input.get('start_date')
        end_date = tool_input.get('end_date')
        
        collected_data = collector.collect_and_save_series(series_ids, start_date, end_date)
        
        return {
            "status": "success",
            "message": f"{len(collected_data)}개의 시리즈 데이터 수집 완료",
            "collected_series": list(collected_data.keys())
        }
        
    except Exception as e:
        logger.error(f"FRED 도구 처리 실패: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 
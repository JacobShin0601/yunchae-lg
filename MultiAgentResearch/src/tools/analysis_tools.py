from typing import Dict, List, Optional
import pandas as pd
from ..analysis.correlation import run_process as correlation_process
from ..analysis.granger import run_process as granger_process
from ..analysis.rf_model import run_process as rf_process

def correlation_analysis(
    df: pd.DataFrame,
    target_variable: str,
    independent_variables: List[str],
    threshold: float = 0.2
) -> Dict:
    """상관분석을 수행하는 도구"""
    return correlation_process(
        df=df,
        target_variable=target_variable,
        independent_variable_prefixes=independent_variables,
        corr_threshold=threshold
    )

def granger_causality_analysis(
    df: pd.DataFrame,
    target_variable: str,
    independent_variables: List[str],
    threshold: float = 0.05
) -> Dict:
    """그레인저 인과성 분석을 수행하는 도구"""
    return granger_process(
        df=df,
        target_variable=target_variable,
        independent_variable_prefixes=independent_variables,
        granger_significance_level=threshold
    )

def random_forest_analysis(
    df: pd.DataFrame,
    target_variable: str,
    cross_corr_dict: Dict
) -> Dict:
    """랜덤포레스트 분석을 수행하는 도구"""
    return rf_process(
        df=df,
        target_variable=target_variable,
        cross_corr_dict=cross_corr_dict
    )
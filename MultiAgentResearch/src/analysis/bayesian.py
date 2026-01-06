import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from .granger import analyze_granger_causality_for_correlated_vars

class BayesianStructuralTimeSeries:
    def __init__(
        self,
        df: pd.DataFrame,
        target_variable: str,
        independent_variables: List[str],
        granger_results: Optional[Dict] = None,
        seasonal_period: Optional[int] = None,
        trend_type: str = 'local linear trend',
        random_state: int = 42
    ):
        """
        베이지안 구조적 시계열 모델을 초기화합니다.

        Args:
            df (pd.DataFrame): 분석할 데이터프레임
            target_variable (str): 타겟 변수명
            independent_variables (List[str]): 독립 변수명 리스트
            granger_results (Optional[Dict]): 그레인저 인과성 분석 결과
            seasonal_period (Optional[int]): 계절성 주기 (예: 12 for 월별, 4 for 분기별)
            trend_type (str): 추세 유형 ('local linear trend' or 'random walk')
            random_state (int): 랜덤 시드
        """
        self.df = df.copy()
        self.target_variable = target_variable
        self.independent_variables = independent_variables
        self.seasonal_period = seasonal_period
        self.trend_type = trend_type
        self.random_state = random_state
        
        # 그레인저 결과가 없는 경우 분석 수행
        if granger_results is None:
            self.granger_results = self._analyze_granger_causality()
        else:
            self.granger_results = granger_results
            
        # 사전지식 설정
        self.prior_knowledge = self._set_prior_knowledge()
        
        # 모델 초기화
        self.model = None
        self.fitted_model = None
        
    def _analyze_granger_causality(self) -> Dict:
        """
        그레인저 인과성 분석을 수행합니다.
        """
        return analyze_granger_causality_for_correlated_vars(
            df=self.df,
            target_variable=self.target_variable,
            independent_variable_prefixes=self.independent_variables,
            cross_corr_threshold=0.0,
            num_top_corr_vars=5,
            max_lag_granger=12,
            granger_significance_level=0.05
        )
    
    def _set_prior_knowledge(self) -> Dict:
        """
        그레인저 인과성 분석 결과를 바탕으로 사전지식을 설정합니다.
        """
        prior_knowledge = {
            'causal_structure': {},
            'lag_effects': {},
            'variable_importance': {},
            'uncertainty': {
                'measurement_error': 0.05,
                'model_uncertainty': 0.1,
                'shock_probability': 0.01
            }
        }
        
        # 그레인저 결과를 바탕으로 사전지식 설정
        for var_name, var_results in self.granger_results.items():
            if 'granger_causality_to_target' in var_results:
                granger_info = var_results['granger_causality_to_target']
                
                # 인과관계가 있는 경우
                if granger_info.get('is_causal_overall', False):
                    prior_knowledge['causal_structure'][var_name] = {
                        'has_causality': True,
                        'direction': 'to_target'
                    }
                    
                    # 최적 시차 정보 추가
                    if 'best_lag' in granger_info:
                        best_lag = granger_info['best_lag']
                        prior_knowledge['lag_effects'][var_name] = {
                            'best_lag': best_lag['lag'],
                            'significance': 1 - best_lag['p_value']
                        }
                    
                    # 변수 중요도 설정
                    prior_knowledge['variable_importance'][var_name] = {
                        'importance': 1 - granger_info['best_lag']['p_value'],
                        'f_statistic': granger_info['best_lag']['F_statistic']
                    }
        
        return prior_knowledge
    
    def fit(
        self,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> 'BayesianStructuralTimeSeries':
        """
        베이지안 구조적 시계열 모델을 학습합니다.

        Args:
            test_size (float): 테스트 데이터 비율
            verbose (bool): 학습 과정 출력 여부

        Returns:
            BayesianStructuralTimeSeries: 학습된 모델
        """
        # 데이터 분할
        train_size = int(len(self.df) * (1 - test_size))
        train_df = self.df.iloc[:train_size]
        test_df = self.df.iloc[train_size:]
        
        # 모델 설정
        self.model = UnobservedComponents(
            endog=train_df[self.target_variable],
            level=self.trend_type,
            seasonal=self.seasonal_period,
            exog=train_df[self.independent_variables],
            # 사전지식 기반 파라미터 설정
            level_variance_prior=self.prior_knowledge['uncertainty']['model_uncertainty'],
            seasonal_variance_prior=self.prior_knowledge['uncertainty']['model_uncertainty'] * 0.5,
            exog_variance_prior=self.prior_knowledge['uncertainty']['model_uncertainty'] * 2
        )
        
        # 모델 학습
        self.fitted_model = self.model.fit(disp=verbose)
        
        return self
    
    def predict(
        self,
        steps: int,
        exog: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        미래 값을 예측합니다.

        Args:
            steps (int): 예측할 기간 수
            exog (Optional[pd.DataFrame]): 예측에 사용할 외생변수 데이터

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 예측값, 예측 구간 하한, 예측 구간 상한
        """
        if self.fitted_model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        # 예측 수행
        forecast = self.fitted_model.get_forecast(
            steps=steps,
            exog=exog
        )
        
        # 예측 결과
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        return mean_forecast, conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    
    def plot_components(self) -> None:
        """
        모델의 구성요소(추세, 계절성, 회귀효과 등)를 시각화합니다.
        """
        if self.fitted_model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        # 구성요소 시각화
        self.fitted_model.plot_components(figsize=(15, 10))
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(
        self,
        steps: int,
        exog: Optional[pd.DataFrame] = None,
        actual: Optional[pd.Series] = None
    ) -> None:
        """
        예측 결과를 시각화합니다.

        Args:
            steps (int): 예측할 기간 수
            exog (Optional[pd.DataFrame]): 예측에 사용할 외생변수 데이터
            actual (Optional[pd.Series]): 실제값 (있는 경우)
        """
        if self.fitted_model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        # 예측 수행
        mean_forecast, lower_ci, upper_ci = self.predict(steps, exog)
        
        # 시각화
        plt.figure(figsize=(12, 6))
        
        # 실제값 플롯
        if actual is not None:
            plt.plot(actual.index, actual.values, label='Actual', color='black')
        
        # 예측값 플롯
        forecast_index = pd.date_range(
            start=self.df.index[-1],
            periods=steps + 1,
            freq=self.df.index.freq
        )[1:]
        
        plt.plot(forecast_index, mean_forecast, label='Forecast', color='blue')
        plt.fill_between(
            forecast_index,
            lower_ci,
            upper_ci,
            color='blue',
            alpha=0.1,
            label='95% Confidence Interval'
        )
        
        plt.title(f'{self.target_variable} Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_variable_importance(self) -> pd.DataFrame:
        """
        변수 중요도를 계산하고 반환합니다.

        Returns:
            pd.DataFrame: 변수별 중요도 정보
        """
        if self.fitted_model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 실행하세요.")
        
        # 회귀계수 추출
        coefficients = self.fitted_model.params[self.independent_variables]
        
        # 표준오차 추출
        std_errors = self.fitted_model.bse[self.independent_variables]
        
        # t-통계량 계산
        t_stats = coefficients / std_errors
        
        # 중요도 계산 (t-통계량의 절대값)
        importance = pd.DataFrame({
            'variable': self.independent_variables,
            'coefficient': coefficients,
            'std_error': std_errors,
            't_statistic': t_stats,
            'importance': np.abs(t_stats)
        })
        
        # 중요도 기준 내림차순 정렬
        importance = importance.sort_values('importance', ascending=False)
        
        return importance
    
    def plot_variable_importance(self) -> None:
        """
        변수 중요도를 시각화합니다.
        """
        importance_df = self.get_variable_importance()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='importance',
            y='variable',
            data=importance_df,
            color='skyblue'
        )
        
        plt.title('Variable Importance')
        plt.xlabel('Importance (|t-statistic|)')
        plt.ylabel('Variable')
        plt.tight_layout()
        plt.show()

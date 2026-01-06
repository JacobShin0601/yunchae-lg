"""
Stochastic Risk Quantification - Monte Carlo Simulation

Monte Carlo 시뮬레이션을 통한 파라미터 불확실성 정량화 및 리스크 분석
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
from datetime import datetime
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings


@dataclass
class ParameterUncertainty:
    """
    파라미터 불확실성 정의

    Attributes:
        name: 파라미터 이름
        distribution: 확률 분포 타입 ('normal', 'uniform', 'triangular', 'lognormal')
        params: 분포 파라미터
            - normal: {'mean': float, 'std': float}
            - uniform: {'low': float, 'high': float}
            - triangular: {'low': float, 'mode': float, 'high': float}
            - lognormal: {'mean': float, 'sigma': float}
        bounds: (min, max) 샘플 값의 허용 범위 (optional)
        description: 설명
    """
    name: str
    distribution: str
    params: Dict[str, float]
    bounds: Optional[Tuple[float, float]] = None
    description: str = ""

    def __post_init__(self):
        """검증 및 초기화"""
        valid_distributions = ['normal', 'uniform', 'triangular', 'lognormal']
        if self.distribution not in valid_distributions:
            raise ValueError(f"distribution must be one of {valid_distributions}")

        # 분포별 필수 파라미터 검증
        if self.distribution == 'normal':
            if 'mean' not in self.params or 'std' not in self.params:
                raise ValueError("normal distribution requires 'mean' and 'std'")
        elif self.distribution == 'uniform':
            if 'low' not in self.params or 'high' not in self.params:
                raise ValueError("uniform distribution requires 'low' and 'high'")
        elif self.distribution == 'triangular':
            if not all(k in self.params for k in ['low', 'mode', 'high']):
                raise ValueError("triangular distribution requires 'low', 'mode', 'high'")
        elif self.distribution == 'lognormal':
            if 'mean' not in self.params or 'sigma' not in self.params:
                raise ValueError("lognormal distribution requires 'mean' and 'sigma'")

    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        분포로부터 샘플 생성

        Args:
            size: 샘플 개수
            random_state: 난수 시드

        Returns:
            샘플 배열
        """
        if random_state is not None:
            np.random.seed(random_state)

        if self.distribution == 'normal':
            samples = np.random.normal(
                self.params['mean'],
                self.params['std'],
                size=size
            )
        elif self.distribution == 'uniform':
            samples = np.random.uniform(
                self.params['low'],
                self.params['high'],
                size=size
            )
        elif self.distribution == 'triangular':
            samples = np.random.triangular(
                self.params['low'],
                self.params['mode'],
                self.params['high'],
                size=size
            )
        elif self.distribution == 'lognormal':
            samples = np.random.lognormal(
                self.params['mean'],
                self.params['sigma'],
                size=size
            )

        # Bounds 적용
        if self.bounds is not None:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])

        return samples

    def get_statistics(self) -> Dict[str, float]:
        """분포의 이론적 통계량 계산"""
        stats_dict = {}

        if self.distribution == 'normal':
            stats_dict['mean'] = self.params['mean']
            stats_dict['std'] = self.params['std']
            stats_dict['var'] = self.params['std'] ** 2
        elif self.distribution == 'uniform':
            low, high = self.params['low'], self.params['high']
            stats_dict['mean'] = (low + high) / 2
            stats_dict['std'] = (high - low) / np.sqrt(12)
            stats_dict['var'] = (high - low) ** 2 / 12
        elif self.distribution == 'triangular':
            low, mode, high = self.params['low'], self.params['mode'], self.params['high']
            stats_dict['mean'] = (low + mode + high) / 3
            stats_dict['var'] = (low**2 + mode**2 + high**2 - low*mode - low*high - mode*high) / 18
            stats_dict['std'] = np.sqrt(stats_dict['var'])
        elif self.distribution == 'lognormal':
            mean, sigma = self.params['mean'], self.params['sigma']
            stats_dict['mean'] = np.exp(mean + sigma**2 / 2)
            stats_dict['var'] = (np.exp(sigma**2) - 1) * np.exp(2*mean + sigma**2)
            stats_dict['std'] = np.sqrt(stats_dict['var'])

        return stats_dict


class StochasticAnalyzer:
    """
    Monte Carlo 시뮬레이션 기반 불확실성 정량화

    파라미터 불확실성을 고려하여 목적함수의 확률 분포를 추정하고
    리스크 메트릭(VaR, CVaR)을 계산합니다.
    """

    def __init__(
        self,
        engine,
        data_loader,
        n_samples: int = 1000,
        random_state: Optional[int] = None,
        parallel: bool = False,
        n_jobs: int = -1
    ):
        """
        Args:
            engine: OptimizationEngine 인스턴스
            data_loader: DataLoader 인스턴스
            n_samples: Monte Carlo 샘플 개수
            random_state: 재현성을 위한 난수 시드
            parallel: 병렬 실행 여부
            n_jobs: 병렬 작업 수 (-1: 모든 CPU)
        """
        self.engine = engine
        self.data_loader = data_loader
        self.n_samples = n_samples
        self.random_state = random_state
        self.parallel = parallel
        self.n_jobs = n_jobs

        # 불확실성 정의
        self.uncertainties: Dict[str, ParameterUncertainty] = {}

        # Monte Carlo 결과
        self.samples: Optional[pd.DataFrame] = None
        self.results: Optional[List[Dict]] = None
        self.objective_values: Optional[np.ndarray] = None

        # 통계 및 리스크 메트릭
        self.statistics: Dict[str, float] = {}
        self.risk_metrics: Dict[str, float] = {}

    def define_uncertainty(
        self,
        parameter_name: str,
        distribution: str,
        params: Dict[str, float],
        bounds: Optional[Tuple[float, float]] = None,
        description: str = ""
    ):
        """
        파라미터 불확실성 정의

        Args:
            parameter_name: 파라미터 이름 (예: 'emission_factor_Ni', 'recycle_cost')
            distribution: 확률 분포 ('normal', 'uniform', 'triangular', 'lognormal')
            params: 분포 파라미터
            bounds: 샘플 값의 허용 범위
            description: 설명
        """
        uncertainty = ParameterUncertainty(
            name=parameter_name,
            distribution=distribution,
            params=params,
            bounds=bounds,
            description=description
        )

        self.uncertainties[parameter_name] = uncertainty

    def generate_samples(self) -> pd.DataFrame:
        """
        Monte Carlo 샘플 생성

        Returns:
            샘플 DataFrame (n_samples x n_parameters)
        """
        if not self.uncertainties:
            raise ValueError("No uncertainties defined. Use define_uncertainty() first.")

        samples_dict = {}

        for param_name, uncertainty in self.uncertainties.items():
            samples = uncertainty.sample(
                size=self.n_samples,
                random_state=self.random_state
            )
            samples_dict[param_name] = samples

        self.samples = pd.DataFrame(samples_dict)

        return self.samples

    def _apply_sample_to_data(
        self,
        base_data: Dict[str, Any],
        sample: pd.Series
    ) -> Dict[str, Any]:
        """
        샘플을 base_data에 적용

        Args:
            base_data: 기준 최적화 데이터
            sample: 파라미터 샘플 (Series)

        Returns:
            수정된 최적화 데이터
        """
        # Deep copy to avoid modifying original
        import copy
        modified_data = copy.deepcopy(base_data)

        # scenario_df 수정
        if 'scenario_df' in modified_data and modified_data['scenario_df'] is not None:
            df = modified_data['scenario_df'].copy()

            for param_name, param_value in sample.items():
                # 파라미터 이름 파싱 (예: 'emission_factor_Ni' → material='Ni')
                if param_name.startswith('emission_factor_'):
                    material = param_name.replace('emission_factor_', '')
                    mask = df['자재명'].str.contains(material, na=False)
                    if mask.any():
                        # 배출계수에 비율 적용 (예: 1.1 = +10% 변동)
                        df.loc[mask, '배출계수'] = df.loc[mask, '배출계수'] * param_value

                elif param_name.startswith('cost_multiplier_'):
                    # 비용 변동 (미래 확장용)
                    pass

            modified_data['scenario_df'] = df

        return modified_data

    def _run_single_sample(
        self,
        sample_idx: int,
        sample: pd.Series,
        base_data: Dict[str, Any],
        objective_type: str
    ) -> Dict[str, Any]:
        """
        단일 샘플에 대한 최적화 실행

        Args:
            sample_idx: 샘플 인덱스
            sample: 파라미터 샘플
            base_data: 기준 데이터
            objective_type: 목적함수 타입

        Returns:
            최적화 결과 딕셔너리
        """
        try:
            # 샘플 적용
            modified_data = self._apply_sample_to_data(base_data, sample)

            # 최적화 실행
            model = self.engine.build_model(modified_data, objective_type=objective_type)
            solution = self.engine.solve()

            if solution:
                result = {
                    'sample_idx': sample_idx,
                    'success': True,
                    'objective': solution['summary']['total_carbon'],
                    'solution': solution,
                    'parameters': sample.to_dict()
                }
            else:
                result = {
                    'sample_idx': sample_idx,
                    'success': False,
                    'objective': np.nan,
                    'solution': None,
                    'parameters': sample.to_dict()
                }

        except Exception as e:
            warnings.warn(f"Sample {sample_idx} failed: {str(e)}")
            result = {
                'sample_idx': sample_idx,
                'success': False,
                'objective': np.nan,
                'solution': None,
                'parameters': sample.to_dict(),
                'error': str(e)
            }

        return result

    def run_monte_carlo(
        self,
        base_data: Dict[str, Any],
        objective_type: str = 'minimize_carbon'
    ) -> List[Dict]:
        """
        Monte Carlo 시뮬레이션 실행

        Args:
            base_data: 기준 최적화 데이터
            objective_type: 목적함수 타입

        Returns:
            결과 리스트 (각 샘플의 최적화 결과)
        """
        if self.samples is None:
            self.generate_samples()

        results = []

        if self.parallel:
            # 병렬 실행
            print(f"🚀 Starting parallel Monte Carlo with {self.n_jobs} workers...")
            results = self._run_monte_carlo_parallel(base_data, objective_type)
        else:
            # 순차 실행
            print(f"🔄 Starting sequential Monte Carlo with {self.n_samples} samples...")
            for idx, (_, sample) in enumerate(self.samples.iterrows()):
                result = self._run_single_sample(idx, sample, base_data, objective_type)
                results.append(result)

                # 진행 표시 (매 100개)
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{self.n_samples} samples...")

        self.results = results

        # 목적함수 값 추출
        self.objective_values = np.array([
            r['objective'] for r in results if r['success']
        ])

        # 통계 및 리스크 메트릭 계산
        self.calculate_statistics()
        self.calculate_risk_metrics()

        return results

    def _run_monte_carlo_parallel(
        self,
        base_data: Dict[str, Any],
        objective_type: str
    ) -> List[Dict]:
        """
        병렬 Monte Carlo 시뮬레이션 실행

        Args:
            base_data: 기준 최적화 데이터
            objective_type: 목적함수 타입

        Returns:
            결과 리스트
        """
        from functools import partial

        # _run_single_sample을 partial로 감싸서 고정 인자 전달
        run_sample_func = partial(
            self._run_single_sample_static,
            base_data=base_data,
            objective_type=objective_type,
            engine_class=type(self.engine),
            engine_solver=self.engine.solver_name,
            data_loader=self.data_loader,
            uncertainties=self.uncertainties
        )

        # ProcessPoolExecutor로 병렬 실행
        results = []
        completed_count = 0

        with ProcessPoolExecutor(max_workers=self.n_jobs if self.n_jobs > 0 else None) as executor:
            # 모든 샘플에 대해 future 생성
            future_to_idx = {}
            for idx, (_, sample) in enumerate(self.samples.iterrows()):
                future = executor.submit(run_sample_func, idx, sample.to_dict())
                future_to_idx[future] = idx

            # 완료된 순서대로 결과 수집
            for future in as_completed(future_to_idx):
                result = future.result()
                results.append(result)
                completed_count += 1

                # 진행 표시 (매 100개)
                if completed_count % 100 == 0:
                    print(f"Completed {completed_count}/{self.n_samples} samples...")

        # 샘플 인덱스 순서대로 정렬
        results.sort(key=lambda x: x['sample_idx'])

        return results

    @staticmethod
    def _run_single_sample_static(
        sample_idx: int,
        sample_dict: Dict[str, float],
        base_data: Dict[str, Any],
        objective_type: str,
        engine_class,
        engine_solver: str,
        data_loader,
        uncertainties: Dict[str, 'ParameterUncertainty']
    ) -> Dict[str, Any]:
        """
        단일 샘플 실행 (정적 메서드 - 병렬 실행용)

        Args:
            sample_idx: 샘플 인덱스
            sample_dict: 파라미터 샘플 딕셔너리
            base_data: 기준 데이터
            objective_type: 목적함수 타입
            engine_class: OptimizationEngine 클래스
            engine_solver: 솔버 이름
            data_loader: DataLoader 인스턴스
            uncertainties: 불확실성 정의 딕셔너리

        Returns:
            최적화 결과 딕셔너리
        """
        try:
            # 새 엔진 인스턴스 생성 (각 프로세스마다 독립적)
            engine = engine_class(solver_name=engine_solver)

            # 샘플 적용
            modified_data = StochasticAnalyzer._apply_sample_to_data_static(
                base_data, sample_dict, uncertainties
            )

            # 최적화 실행
            model = engine.build_model(modified_data, objective_type=objective_type)
            solution = engine.solve()

            if solution:
                result = {
                    'sample_idx': sample_idx,
                    'success': True,
                    'objective': solution['summary']['total_carbon'],
                    'solution': solution,
                    'parameters': sample_dict
                }
            else:
                result = {
                    'sample_idx': sample_idx,
                    'success': False,
                    'objective': np.nan,
                    'solution': None,
                    'parameters': sample_dict
                }

        except Exception as e:
            result = {
                'sample_idx': sample_idx,
                'success': False,
                'objective': np.nan,
                'solution': None,
                'parameters': sample_dict,
                'error': str(e)
            }

        return result

    @staticmethod
    def _apply_sample_to_data_static(
        base_data: Dict[str, Any],
        sample_dict: Dict[str, float],
        uncertainties: Dict[str, 'ParameterUncertainty']
    ) -> Dict[str, Any]:
        """
        샘플을 데이터에 적용 (정적 메서드 - 병렬 실행용)

        Args:
            base_data: 기준 데이터
            sample_dict: 파라미터 샘플 딕셔너리
            uncertainties: 불확실성 정의 딕셔너리

        Returns:
            수정된 데이터
        """
        import copy
        modified_data = copy.deepcopy(base_data)

        # 각 불확실성 파라미터에 대해 적용
        for param_name, multiplier in sample_dict.items():
            if param_name not in uncertainties:
                continue

            uncertainty = uncertainties[param_name]

            # 파라미터 타입에 따라 적용
            if 'emission_factor' in param_name:
                # 배출계수 변동
                material = param_name.replace('emission_factor_', '')

                if 'scenario_df' in modified_data:
                    df = modified_data['scenario_df']
                    mask = df['자재명'].str.contains(material, case=False, na=False)
                    if mask.any():
                        df.loc[mask, '배출계수'] *= multiplier

            elif 'cost' in param_name:
                # 비용 변동
                if 'baseline_cost' in modified_data:
                    modified_data['baseline_cost'] *= multiplier

            # 추가 파라미터 타입은 여기에 구현...

        return modified_data

    def calculate_statistics(self):
        """결과 통계 계산"""
        if self.objective_values is None or len(self.objective_values) == 0:
            raise ValueError("No objective values available. Run run_monte_carlo() first.")

        # 유효한 값만 사용 (NaN 제거)
        valid_values = self.objective_values[~np.isnan(self.objective_values)]

        if len(valid_values) == 0:
            raise ValueError("No valid objective values.")

        self.statistics = {
            'mean': float(np.mean(valid_values)),
            'median': float(np.median(valid_values)),
            'std': float(np.std(valid_values)),
            'var': float(np.var(valid_values)),
            'min': float(np.min(valid_values)),
            'max': float(np.max(valid_values)),
            'p5': float(np.percentile(valid_values, 5)),
            'p25': float(np.percentile(valid_values, 25)),
            'p75': float(np.percentile(valid_values, 75)),
            'p95': float(np.percentile(valid_values, 95)),
            'iqr': float(np.percentile(valid_values, 75) - np.percentile(valid_values, 25)),
            'cv': float(np.std(valid_values) / np.mean(valid_values)) if np.mean(valid_values) != 0 else 0,
            'n_samples': len(valid_values),
            'n_failed': self.n_samples - len(valid_values)
        }

    def calculate_risk_metrics(
        self,
        var_level: float = 0.95,
        cvar_level: float = 0.95
    ):
        """
        리스크 메트릭 계산

        Args:
            var_level: VaR 신뢰수준 (default: 0.95)
            cvar_level: CVaR 신뢰수준 (default: 0.95)
        """
        if self.objective_values is None or len(self.objective_values) == 0:
            raise ValueError("No objective values available.")

        valid_values = self.objective_values[~np.isnan(self.objective_values)]

        if len(valid_values) == 0:
            raise ValueError("No valid objective values.")

        # VaR (Value at Risk): 신뢰수준의 분위수
        var = np.percentile(valid_values, var_level * 100)

        # CVaR (Conditional VaR): VaR 초과 값들의 평균
        cvar_values = valid_values[valid_values >= var]
        cvar = np.mean(cvar_values) if len(cvar_values) > 0 else var

        # 확률 초과 (Probability of Exceedance) - 예: 기준값 초과 확률
        # 여기서는 중간값을 기준으로 설정
        threshold = self.statistics['median']
        prob_exceed = np.sum(valid_values > threshold) / len(valid_values)

        # 극단 리스크 (Tail Risk)
        tail_values = valid_values[valid_values >= self.statistics['p95']]
        tail_mean = np.mean(tail_values) if len(tail_values) > 0 else var

        self.risk_metrics = {
            f'var_{int(var_level*100)}': float(var),
            f'cvar_{int(cvar_level*100)}': float(cvar),
            'prob_exceed_median': float(prob_exceed),
            'tail_mean': float(tail_mean),
            'worst_case': float(np.max(valid_values)),
            'best_case': float(np.min(valid_values)),
            'downside_risk': float(np.std(valid_values[valid_values > self.statistics['mean']])) if np.any(valid_values > self.statistics['mean']) else 0.0
        }

    def get_probability_distribution(
        self,
        n_bins: int = 50
    ) -> Dict[str, Any]:
        """
        확률 분포 데이터 (히스토그램)

        Args:
            n_bins: 히스토그램 빈 개수

        Returns:
            히스토그램 데이터 딕셔너리
        """
        if self.objective_values is None:
            raise ValueError("No objective values available.")

        valid_values = self.objective_values[~np.isnan(self.objective_values)]

        # 히스토그램
        counts, bin_edges = np.histogram(valid_values, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # PDF (확률 밀도)
        pdf = counts / (np.sum(counts) * (bin_edges[1] - bin_edges[0]))

        # CDF (누적 분포)
        cdf = np.cumsum(counts) / np.sum(counts)

        return {
            'bin_centers': bin_centers.tolist(),
            'bin_edges': bin_edges.tolist(),
            'counts': counts.tolist(),
            'pdf': pdf.tolist(),
            'cdf': cdf.tolist()
        }

    def get_parameter_correlation(self) -> pd.DataFrame:
        """
        파라미터-목적함수 상관관계

        Returns:
            상관계수 DataFrame
        """
        if self.samples is None or self.objective_values is None:
            raise ValueError("Run Monte Carlo simulation first.")

        # 샘플과 목적함수를 결합
        df = self.samples.copy()
        df['objective'] = self.objective_values

        # 유효한 행만 선택
        df = df.dropna(subset=['objective'])

        # 상관계수 계산
        correlation = df.corr()['objective'].drop('objective')

        return correlation.to_frame('correlation').sort_values('correlation', ascending=False)

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        시각화용 데이터 준비

        Returns:
            시각화 데이터 딕셔너리
        """
        if self.objective_values is None:
            raise ValueError("Run Monte Carlo simulation first.")

        return {
            'statistics': self.statistics,
            'risk_metrics': self.risk_metrics,
            'distribution': self.get_probability_distribution(),
            'correlation': self.get_parameter_correlation().to_dict()['correlation'],
            'samples': self.samples.describe().to_dict() if self.samples is not None else {}
        }

    def export_results(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        결과 직렬화

        Args:
            filepath: JSON 저장 경로 (optional)

        Returns:
            결과 딕셔너리
        """
        export_data = {
            'meta': {
                'n_samples': self.n_samples,
                'random_state': self.random_state,
                'timestamp': datetime.now().isoformat()
            },
            'uncertainties': {
                name: {
                    'distribution': unc.distribution,
                    'params': unc.params,
                    'bounds': unc.bounds,
                    'description': unc.description,
                    'theoretical_stats': unc.get_statistics()
                }
                for name, unc in self.uncertainties.items()
            },
            'statistics': self.statistics,
            'risk_metrics': self.risk_metrics,
            'visualization_data': self.get_visualization_data()
        }

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_data

    def generate_report(self) -> str:
        """
        분석 보고서 생성 (텍스트)

        Returns:
            보고서 문자열
        """
        if self.statistics is None or self.risk_metrics is None:
            raise ValueError("Run Monte Carlo simulation first.")

        report = []
        report.append("=" * 60)
        report.append("Stochastic Risk Analysis Report")
        report.append("=" * 60)
        report.append("")

        # 파라미터 불확실성
        report.append("Parameter Uncertainties:")
        report.append("-" * 60)
        for name, unc in self.uncertainties.items():
            report.append(f"  {name}")
            report.append(f"    Distribution: {unc.distribution}")
            report.append(f"    Parameters: {unc.params}")
            if unc.bounds:
                report.append(f"    Bounds: {unc.bounds}")
            report.append("")

        # 통계
        report.append("Objective Function Statistics:")
        report.append("-" * 60)
        report.append(f"  Mean: {self.statistics['mean']:.4f}")
        report.append(f"  Median: {self.statistics['median']:.4f}")
        report.append(f"  Std Dev: {self.statistics['std']:.4f}")
        report.append(f"  Min: {self.statistics['min']:.4f}")
        report.append(f"  Max: {self.statistics['max']:.4f}")
        report.append(f"  CV (Coeff. of Variation): {self.statistics['cv']:.4f}")
        report.append("")

        # 리스크 메트릭
        report.append("Risk Metrics:")
        report.append("-" * 60)
        for metric, value in self.risk_metrics.items():
            report.append(f"  {metric}: {value:.4f}")
        report.append("")

        # 성공률
        success_rate = (self.statistics['n_samples'] / self.n_samples) * 100
        report.append(f"Success Rate: {success_rate:.1f}% ({self.statistics['n_samples']}/{self.n_samples})")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

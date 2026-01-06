"""
민감도 분석 (Sensitivity Analysis)

파라미터 변화가 최적화 결과에 미치는 영향을 정량화합니다.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import copy
from SALib.sample import saltelli
from SALib.analyze import sobol
from ..core.optimization_engine import OptimizationEngine
from ..utils.data_loader import DataLoader


class SensitivityAnalyzer:
    """
    민감도 분석기

    파라미터 변동이 탄소배출/비용 목적함수에 미치는 영향을 분석합니다.

    주요 기능:
    - OAT (One-At-a-Time) 민감도 분석
    - 탄성도(Elasticity) 계산
    - 파라미터 중요도 순위화
    """

    def __init__(
        self,
        optimization_engine: OptimizationEngine,
        data_loader: DataLoader
    ):
        """
        민감도 분석기 초기화

        Args:
            optimization_engine: 최적화 엔진 인스턴스
            data_loader: 데이터 로더 인스턴스
        """
        self.engine = optimization_engine
        self.data_loader = data_loader
        self.results = {}

    def run_oat_analysis(
        self,
        base_data: Dict[str, Any],
        parameter_specs: List[Dict[str, Any]],
        variation_range: Tuple[float, float] = (-20, 20),
        n_points: int = 5,
        objective_type: str = 'minimize_carbon'
    ) -> Dict[str, Any]:
        """
        OAT (One-At-a-Time) 민감도 분석 실행

        각 파라미터를 하나씩 변화시키며 목적함수 변화를 측정합니다.

        Args:
            base_data: 기준 최적화 데이터
            parameter_specs: 분석할 파라미터 스펙 리스트
                각 스펙 형식: {
                    'name': str,  # 파라미터 이름
                    'type': str,  # 'emission_factor' | 'cost' | 'constraint_bound'
                    'material': str (optional),  # 자재명
                    'path': str,  # 데이터 경로 (예: 'materials.NCM622.emission_factor')
                    'baseline_value': float  # 기준값
                }
            variation_range: 변동 범위 (%, 예: (-20, 20) = -20% ~ +20%)
            n_points: 각 파라미터당 측정 포인트 수
            objective_type: 목적함수 유형

        Returns:
            민감도 분석 결과 딕셔너리
        """
        print("\n" + "=" * 60)
        print("🔍 OAT 민감도 분석 시작")
        print("=" * 60)

        # 기준 케이스 최적화
        print("\n📊 기준 케이스 최적화 중...")
        base_model = self.engine.build_model(base_data, objective_type)
        base_solution = self.engine.solve()

        if not base_solution['success']:
            raise ValueError(f"기준 케이스 최적화 실패: {base_solution['message']}")

        base_objective = base_solution['objective_value']
        print(f"✅ 기준 목적함수 값: {base_objective:.4f}")

        # 각 파라미터에 대해 민감도 분석
        sensitivity_results = {}

        for param_spec in parameter_specs:
            param_name = param_spec['name']
            baseline_value = param_spec['baseline_value']

            print(f"\n📈 분석 중: {param_name} (기준값: {baseline_value:.4f})")

            # 변동률 생성
            variation_pcts = np.linspace(
                variation_range[0],
                variation_range[1],
                n_points
            )

            param_results = {
                'param_name': param_name,
                'param_spec': param_spec,
                'baseline_value': baseline_value,
                'base_objective': base_objective,
                'variation_pcts': [],
                'variation_values': [],
                'objective_values': [],
                'objective_changes': [],
                'solutions': []
            }

            # 각 변동률에 대해 최적화
            for var_pct in variation_pcts:
                try:
                    # 데이터 복사 및 파라미터 변경
                    varied_data = copy.deepcopy(base_data)
                    varied_value = baseline_value * (1 + var_pct / 100)

                    self._apply_parameter_variation(
                        varied_data,
                        param_spec,
                        varied_value
                    )

                    # 최적화 실행
                    model = self.engine.build_model(varied_data, objective_type)
                    solution = self.engine.solve()

                    if solution['success']:
                        obj_value = solution['objective_value']
                        obj_change = ((obj_value - base_objective) / base_objective) * 100

                        param_results['variation_pcts'].append(var_pct)
                        param_results['variation_values'].append(varied_value)
                        param_results['objective_values'].append(obj_value)
                        param_results['objective_changes'].append(obj_change)
                        param_results['solutions'].append(solution)

                        print(f"  {var_pct:+6.1f}% → 목적함수: {obj_value:.4f} ({obj_change:+.2f}%)")
                    else:
                        print(f"  {var_pct:+6.1f}% → 최적화 실패")

                except Exception as e:
                    print(f"  {var_pct:+6.1f}% → 오류: {str(e)}")
                    continue

            # 탄성도 계산
            elasticity = self._calculate_elasticity(param_results)
            param_results['elasticity'] = elasticity

            sensitivity_results[param_name] = param_results

        # 파라미터 중요도 순위화
        ranking = self._rank_parameters(sensitivity_results)

        analysis_result = {
            'base_objective': base_objective,
            'base_solution': base_solution,
            'objective_type': objective_type,
            'variation_range': variation_range,
            'n_points': n_points,
            'parameter_results': sensitivity_results,
            'parameter_ranking': ranking
        }

        self.results = analysis_result

        print("\n" + "=" * 60)
        print("✅ OAT 민감도 분석 완료")
        print("=" * 60)

        return analysis_result

    def _apply_parameter_variation(
        self,
        data: Dict[str, Any],
        param_spec: Dict[str, Any],
        new_value: float
    ):
        """
        데이터에 파라미터 변동 적용

        Args:
            data: 최적화 데이터
            param_spec: 파라미터 스펙
            new_value: 새로운 값
        """
        param_type = param_spec['type']

        if param_type == 'emission_factor':
            material = param_spec['material']
            if material in data['materials']:
                data['materials'][material]['emission_factor'] = new_value

        elif param_type == 'virgin_cost':
            material = param_spec['material']
            if material in data['materials']:
                data['materials'][material]['virgin_cost'] = new_value

        elif param_type == 'recycled_cost':
            material = param_spec['material']
            if material in data['materials']:
                data['materials'][material]['recycled_cost'] = new_value

        elif param_type == 'low_carbon_cost':
            material = param_spec['material']
            if material in data['materials']:
                data['materials'][material]['low_carbon_cost'] = new_value

        elif param_type == 'constraint_bound':
            # 제약조건 바운드 변경은 constraint_manager를 통해 처리
            constraint_name = param_spec.get('constraint_name')
            if constraint_name:
                # constraint_manager에 직접 접근하여 수정
                for constraint in self.engine.constraint_manager.constraints:
                    if constraint.name == constraint_name:
                        bound_attr = param_spec.get('bound_attribute')
                        if bound_attr and hasattr(constraint, bound_attr):
                            setattr(constraint, bound_attr, new_value)
                            break

    def _calculate_elasticity(
        self,
        param_results: Dict[str, Any]
    ) -> float:
        """
        탄성도 계산: (Δobjective/objective) / (Δparameter/parameter)

        선형회귀를 사용하여 평균 탄성도 추정

        Args:
            param_results: 파라미터 민감도 결과

        Returns:
            탄성도 값
        """
        if len(param_results['variation_pcts']) < 2:
            return 0.0

        # 변동률(%) vs 목적함수 변화율(%)
        x = np.array(param_results['variation_pcts'])
        y = np.array(param_results['objective_changes'])

        # 선형회귀로 기울기(탄성도) 추정
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        else:
            return 0.0

    def _rank_parameters(
        self,
        sensitivity_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        파라미터를 탄성도(영향도) 기준으로 순위화

        Args:
            sensitivity_results: 전체 민감도 분석 결과

        Returns:
            순위화된 데이터프레임
        """
        ranking_data = []

        for param_name, param_result in sensitivity_results.items():
            elasticity = param_result.get('elasticity', 0.0)
            baseline_value = param_result['baseline_value']

            # 최대/최소 목적함수 변화
            obj_changes = param_result['objective_changes']
            max_change = max(obj_changes) if obj_changes else 0.0
            min_change = min(obj_changes) if obj_changes else 0.0
            impact_range = max_change - min_change

            ranking_data.append({
                '파라미터': param_name,
                '기준값': baseline_value,
                '탄성도': abs(elasticity),
                '최대_변화(%)': max_change,
                '최소_변화(%)': min_change,
                '영향_범위(%)': impact_range
            })

        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('탄성도', ascending=False)
        ranking_df = ranking_df.reset_index(drop=True)

        return ranking_df

    def get_parameter_impact_summary(self) -> pd.DataFrame:
        """
        파라미터 영향도 요약 테이블 생성

        Returns:
            요약 데이터프레임
        """
        if not self.results:
            return pd.DataFrame()

        return self.results['parameter_ranking']

    def run_sobol_analysis(
        self,
        base_data: Dict[str, Any],
        parameter_specs: List[Dict[str, Any]],
        n_samples: int = 1000,
        objective_type: str = 'minimize_carbon'
    ) -> Dict[str, Any]:
        """
        Sobol 전역 민감도 분석 실행

        파라미터 간 상호작용을 포함한 전역 민감도 지수를 계산합니다.

        Args:
            base_data: 기준 최적화 데이터
            parameter_specs: 분석할 파라미터 스펙 리스트
                각 스펙 형식: {
                    'name': str,
                    'type': str,
                    'material': str (optional),
                    'baseline_value': float,
                    'bounds': [min, max]  # 절대값 범위
                }
            n_samples: 샘플 수 (실제 평가 횟수는 N * (2D + 2), D=파라미터 개수)
            objective_type: 목적함수 유형

        Returns:
            Sobol 민감도 분석 결과
        """
        print("\n" + "=" * 60)
        print("🌐 Sobol 전역 민감도 분석 시작")
        print("=" * 60)

        n_params = len(parameter_specs)
        print(f"📊 파라미터 개수: {n_params}")
        print(f"📊 기본 샘플 수: {n_samples}")
        print(f"📊 실제 평가 횟수: {n_samples * (2 * n_params + 2)}")

        # SALib 문제 정의
        problem = {
            'num_vars': n_params,
            'names': [spec['name'] for spec in parameter_specs],
            'bounds': [spec['bounds'] for spec in parameter_specs]
        }

        # Saltelli 샘플링 (Sobol 분석용 특수 샘플링)
        print("\n🎲 Saltelli 샘플링 생성 중...")
        param_samples = saltelli.sample(problem, n_samples)
        n_evaluations = len(param_samples)
        print(f"✅ {n_evaluations}개 샘플 생성 완료")

        # 각 샘플에 대해 최적화 실행
        print("\n🔧 최적화 평가 시작...")
        objective_values = []

        for idx, sample in enumerate(param_samples):
            if (idx + 1) % 100 == 0:
                print(f"  진행률: {idx + 1}/{n_evaluations} ({(idx+1)/n_evaluations*100:.1f}%)")

            try:
                # 샘플 파라미터로 데이터 수정
                varied_data = copy.deepcopy(base_data)

                for param_idx, param_value in enumerate(sample):
                    param_spec = parameter_specs[param_idx]
                    self._apply_parameter_variation(
                        varied_data,
                        param_spec,
                        param_value
                    )

                # 최적화 실행
                model = self.engine.build_model(varied_data, objective_type)
                solution = self.engine.solve()

                if solution['success']:
                    objective_values.append(solution['objective_value'])
                else:
                    # 실패 시 NaN 추가 (Sobol 분석에서 제외됨)
                    objective_values.append(np.nan)

            except Exception as e:
                objective_values.append(np.nan)
                continue

        objective_values = np.array(objective_values)

        # NaN 비율 체크
        nan_ratio = np.sum(np.isnan(objective_values)) / len(objective_values)
        print(f"\n⚠️  실패한 평가: {np.sum(np.isnan(objective_values))}/{len(objective_values)} ({nan_ratio*100:.1f}%)")

        if nan_ratio > 0.2:
            print(f"⚠️  경고: 20% 이상의 평가가 실패했습니다. 결과가 부정확할 수 있습니다.")

        # Sobol 지수 계산
        print("\n📈 Sobol 지수 계산 중...")
        Si = sobol.analyze(problem, objective_values, print_to_console=False)

        # 결과 정리
        sobol_results = {
            'problem': problem,
            'n_samples': n_samples,
            'n_evaluations': n_evaluations,
            'objective_type': objective_type,
            'nan_ratio': nan_ratio,
            'first_order': {},  # S1: 개별 파라미터 효과
            'total_order': {},  # ST: 상호작용 포함 전체 효과
            'second_order': {}  # S2: 2차 상호작용
        }

        for idx, param_name in enumerate(problem['names']):
            sobol_results['first_order'][param_name] = {
                'S1': Si['S1'][idx],
                'S1_conf': Si['S1_conf'][idx]
            }
            sobol_results['total_order'][param_name] = {
                'ST': Si['ST'][idx],
                'ST_conf': Si['ST_conf'][idx]
            }

        # 2차 상호작용 (선택적)
        if 'S2' in Si:
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    pair = f"{problem['names'][i]} × {problem['names'][j]}"
                    sobol_results['second_order'][pair] = {
                        'S2': Si['S2'][i, j],
                        'S2_conf': Si['S2_conf'][i, j]
                    }

        # 순위화
        ranking = self._rank_sobol_indices(sobol_results)
        sobol_results['ranking'] = ranking

        print("\n" + "=" * 60)
        print("✅ Sobol 전역 민감도 분석 완료")
        print("=" * 60)

        return sobol_results

    def _rank_sobol_indices(
        self,
        sobol_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Sobol 지수를 기준으로 파라미터 순위화

        Args:
            sobol_results: Sobol 분석 결과

        Returns:
            순위화된 데이터프레임
        """
        ranking_data = []

        for param_name in sobol_results['problem']['names']:
            S1 = sobol_results['first_order'][param_name]['S1']
            ST = sobol_results['total_order'][param_name]['ST']
            interaction = ST - S1  # 상호작용 효과

            ranking_data.append({
                '파라미터': param_name,
                'First-Order (S1)': S1,
                'Total (ST)': ST,
                '상호작용': interaction,
                'S1 신뢰구간': sobol_results['first_order'][param_name]['S1_conf'],
                'ST 신뢰구간': sobol_results['total_order'][param_name]['ST_conf']
            })

        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Total (ST)', ascending=False)
        ranking_df = ranking_df.reset_index(drop=True)

        return ranking_df

    def export_results_to_dict(self) -> Dict[str, Any]:
        """
        결과를 직렬화 가능한 딕셔너리로 변환

        Returns:
            결과 딕셔너리
        """
        if not self.results:
            return {}

        # 복잡한 객체(model, solution 등) 제외하고 숫자 데이터만 추출
        export_data = {
            'base_objective': self.results['base_objective'],
            'objective_type': self.results['objective_type'],
            'variation_range': self.results['variation_range'],
            'n_points': self.results['n_points'],
            'parameter_ranking': self.results['parameter_ranking'].to_dict('records'),
            'parameters': {}
        }

        for param_name, param_result in self.results['parameter_results'].items():
            export_data['parameters'][param_name] = {
                'baseline_value': param_result['baseline_value'],
                'elasticity': param_result['elasticity'],
                'variation_pcts': param_result['variation_pcts'],
                'variation_values': param_result['variation_values'],
                'objective_values': param_result['objective_values'],
                'objective_changes': param_result['objective_changes']
            }

        return export_data

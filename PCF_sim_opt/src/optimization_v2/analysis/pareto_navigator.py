"""
파레토 대화형 탐색기 (Interactive Pareto Navigator)

사전 계산된 파레토 프론티어를 실시간으로 탐색합니다.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, Rbf
from scipy.spatial import distance


class ParetoNavigator:
    """
    파레토 대화형 탐색기

    사전 계산된 파레토 결과를 보간하여 실시간 탐색을 지원합니다.

    주요 기능:
    - 가중치 슬라이더 기반 파레토 포인트 탐색
    - 스플라인/RBF 보간을 통한 부드러운 탐색
    - 제약조건 필터링
    - 가장 가까운 실제 솔루션 찾기
    """

    def __init__(
        self,
        pareto_results: List[Dict[str, Any]],
        interpolation_method: str = 'linear'
    ):
        """
        파레토 탐색기 초기화

        Args:
            pareto_results: 파레토 최적화 결과 리스트
                각 결과 형식: {
                    'carbon_weight': float,
                    'cost_weight': float,
                    'summary': {'total_carbon': float, 'total_cost': float, ...},
                    'decision_variables': {...},
                    ...
                }
            interpolation_method: 보간 방법
                - 'linear': 선형 보간 (빠름)
                - 'cubic': 3차 스플라인 (부드러움)
                - 'rbf': Radial Basis Function (비선형)
        """
        self.pareto_results = pareto_results
        self.interpolation_method = interpolation_method

        # 파레토 프론티어 데이터 추출
        self._extract_pareto_data()

        # 보간 함수 구축
        self._build_interpolators()

    def _extract_pareto_data(self):
        """파레토 결과에서 핵심 데이터 추출"""
        self.carbon_values = []
        self.cost_values = []
        self.carbon_weights = []
        self.cost_weights = []

        for result in self.pareto_results:
            summary = result.get('summary', {})
            self.carbon_values.append(summary.get('total_carbon', 0))
            self.cost_values.append(summary.get('total_cost', 0))
            self.carbon_weights.append(result.get('carbon_weight', 0.5))
            self.cost_weights.append(result.get('cost_weight', 0.5))

        self.carbon_values = np.array(self.carbon_values)
        self.cost_values = np.array(self.cost_values)
        self.carbon_weights = np.array(self.carbon_weights)
        self.cost_weights = np.array(self.cost_weights)

    def _build_interpolators(self):
        """보간 함수 구축"""
        if len(self.pareto_results) < 2:
            print("⚠️  경고: 파레토 포인트가 2개 미만입니다. 보간을 사용할 수 없습니다.")
            self.interpolator_carbon = None
            self.interpolator_cost = None
            return

        try:
            if self.interpolation_method == 'linear':
                # 선형 보간 (가중치 → 목적함수)
                # carbon_weight 기준으로 정렬
                sort_idx = np.argsort(self.carbon_weights)
                sorted_weights = self.carbon_weights[sort_idx]
                sorted_carbon = self.carbon_values[sort_idx]
                sorted_cost = self.cost_values[sort_idx]

                self.interpolator_carbon = interp1d(
                    sorted_weights,
                    sorted_carbon,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                self.interpolator_cost = interp1d(
                    sorted_weights,
                    sorted_cost,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )

            elif self.interpolation_method == 'cubic':
                # 3차 스플라인 보간
                sort_idx = np.argsort(self.carbon_weights)
                sorted_weights = self.carbon_weights[sort_idx]
                sorted_carbon = self.carbon_values[sort_idx]
                sorted_cost = self.cost_values[sort_idx]

                if len(sorted_weights) >= 4:
                    self.interpolator_carbon = interp1d(
                        sorted_weights,
                        sorted_carbon,
                        kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    self.interpolator_cost = interp1d(
                        sorted_weights,
                        sorted_cost,
                        kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                else:
                    # 포인트가 4개 미만이면 선형으로 대체
                    print("⚠️  포인트 부족으로 선형 보간 사용")
                    self.interpolation_method = 'linear'
                    self._build_interpolators()

            elif self.interpolation_method == 'rbf':
                # Radial Basis Function (비선형)
                self.interpolator_carbon = Rbf(
                    self.carbon_weights,
                    self.carbon_values,
                    function='multiquadric',
                    smooth=0.1
                )
                self.interpolator_cost = Rbf(
                    self.carbon_weights,
                    self.cost_values,
                    function='multiquadric',
                    smooth=0.1
                )

        except Exception as e:
            print(f"⚠️  보간 함수 구축 실패: {str(e)}")
            self.interpolator_carbon = None
            self.interpolator_cost = None

    def get_solution_at_weight(
        self,
        carbon_weight: float,
        return_nearest: bool = True
    ) -> Dict[str, Any]:
        """
        특정 가중치에서의 솔루션 추정

        Args:
            carbon_weight: 탄소 가중치 (0.0 ~ 1.0)
                - 0.0: 100% 비용 최소화
                - 1.0: 100% 탄소 최소화
            return_nearest: True면 가장 가까운 실제 솔루션 반환,
                          False면 보간된 값 반환

        Returns:
            솔루션 딕셔너리
        """
        carbon_weight = np.clip(carbon_weight, 0.0, 1.0)

        if return_nearest:
            # 가장 가까운 실제 파레토 포인트 찾기
            return self._find_nearest_solution(carbon_weight)
        else:
            # 보간된 값 반환
            return self._interpolate_solution(carbon_weight)

    def _find_nearest_solution(
        self,
        target_weight: float
    ) -> Dict[str, Any]:
        """
        목표 가중치와 가장 가까운 실제 솔루션 찾기

        Args:
            target_weight: 목표 탄소 가중치

        Returns:
            가장 가까운 실제 솔루션
        """
        distances = np.abs(self.carbon_weights - target_weight)
        nearest_idx = np.argmin(distances)

        nearest_solution = self.pareto_results[nearest_idx].copy()
        nearest_solution['interpolated'] = False
        nearest_solution['distance_from_target'] = distances[nearest_idx]

        return nearest_solution

    def _interpolate_solution(
        self,
        carbon_weight: float
    ) -> Dict[str, Any]:
        """
        보간을 통한 솔루션 추정

        Args:
            carbon_weight: 탄소 가중치

        Returns:
            보간된 솔루션 (근사값)
        """
        if self.interpolator_carbon is None:
            # 보간 불가능하면 가장 가까운 것 반환
            return self._find_nearest_solution(carbon_weight)

        try:
            estimated_carbon = float(self.interpolator_carbon(carbon_weight))
            estimated_cost = float(self.interpolator_cost(carbon_weight))

            # 가장 가까운 실제 솔루션의 decision_variables 사용
            nearest = self._find_nearest_solution(carbon_weight)

            interpolated_solution = {
                'carbon_weight': carbon_weight,
                'cost_weight': 1.0 - carbon_weight,
                'summary': {
                    'total_carbon': estimated_carbon,
                    'total_cost': estimated_cost,
                },
                'decision_variables': nearest['decision_variables'],  # 근사
                'interpolated': True,
                'nearest_actual_solution': nearest
            }

            return interpolated_solution

        except Exception as e:
            print(f"⚠️  보간 실패: {str(e)}, 가장 가까운 실제 솔루션 반환")
            return self._find_nearest_solution(carbon_weight)

    def filter_by_constraints(
        self,
        min_carbon: Optional[float] = None,
        max_carbon: Optional[float] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        custom_filter: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        제약조건으로 파레토 포인트 필터링

        Args:
            min_carbon: 최소 탄소 배출량
            max_carbon: 최대 탄소 배출량
            min_cost: 최소 비용
            max_cost: 최대 비용
            custom_filter: 커스텀 필터 함수 (result -> bool)

        Returns:
            필터링된 파레토 결과 리스트
        """
        filtered_results = []

        for result in self.pareto_results:
            summary = result.get('summary', {})
            carbon = summary.get('total_carbon', 0)
            cost = summary.get('total_cost', 0)

            # 범위 체크
            if min_carbon is not None and carbon < min_carbon:
                continue
            if max_carbon is not None and carbon > max_carbon:
                continue
            if min_cost is not None and cost < min_cost:
                continue
            if max_cost is not None and cost > max_cost:
                continue

            # 커스텀 필터
            if custom_filter is not None and not custom_filter(result):
                continue

            filtered_results.append(result)

        return filtered_results

    def get_pareto_summary(self) -> pd.DataFrame:
        """
        파레토 프론티어 요약 테이블 생성

        Returns:
            요약 데이터프레임
        """
        summary_data = []

        for idx, result in enumerate(self.pareto_results):
            summary = result.get('summary', {})

            summary_data.append({
                'Index': idx,
                'Carbon Weight': result.get('carbon_weight', 0),
                'Total Carbon': summary.get('total_carbon', 0),
                'Total Cost': summary.get('total_cost', 0),
                'Carbon Reduction (%)': summary.get('carbon_reduction_pct', 0),
                'Cost Premium (%)': summary.get('cost_premium_pct', 0)
            })

        return pd.DataFrame(summary_data)

    def find_pareto_optimal_subset(
        self,
        objective1_key: str = 'total_carbon',
        objective2_key: str = 'total_cost'
    ) -> List[int]:
        """
        파레토 최적 부분집합 찾기 (비지배 솔루션)

        Args:
            objective1_key: 첫 번째 목적함수 키
            objective2_key: 두 번째 목적함수 키

        Returns:
            파레토 최적 솔루션의 인덱스 리스트
        """
        objectives = []

        for result in self.pareto_results:
            summary = result.get('summary', {})
            obj1 = summary.get(objective1_key, 0)
            obj2 = summary.get(objective2_key, 0)
            objectives.append([obj1, obj2])

        objectives = np.array(objectives)

        # 파레토 프론티어 계산 (최소화 문제)
        pareto_indices = []

        for i in range(len(objectives)):
            dominated = False
            for j in range(len(objectives)):
                if i != j:
                    # i가 j에 의해 지배되는지 확인
                    if (objectives[j][0] <= objectives[i][0] and
                        objectives[j][1] <= objectives[i][1] and
                        (objectives[j][0] < objectives[i][0] or
                         objectives[j][1] < objectives[i][1])):
                        dominated = True
                        break

            if not dominated:
                pareto_indices.append(i)

        return pareto_indices

    def get_extreme_solutions(self) -> Dict[str, Dict[str, Any]]:
        """
        극단 솔루션 추출 (최소 탄소, 최소 비용)

        Returns:
            극단 솔루션 딕셔너리
        """
        min_carbon_idx = np.argmin(self.carbon_values)
        min_cost_idx = np.argmin(self.cost_values)

        return {
            'min_carbon': self.pareto_results[min_carbon_idx],
            'min_cost': self.pareto_results[min_cost_idx]
        }

    def get_balanced_solution(self) -> Dict[str, Any]:
        """
        균형 잡힌 솔루션 찾기 (탄소-비용 트레이드오프 중간)

        Returns:
            균형 솔루션
        """
        # 정규화
        carbon_norm = (self.carbon_values - self.carbon_values.min()) / (
            self.carbon_values.max() - self.carbon_values.min() + 1e-10
        )
        cost_norm = (self.cost_values - self.cost_values.min()) / (
            self.cost_values.max() - self.cost_values.min() + 1e-10
        )

        # 유클리드 거리 계산 (이상점 [0, 0]으로부터)
        distances = np.sqrt(carbon_norm**2 + cost_norm**2)

        # 가장 가까운 것이 균형 솔루션
        balanced_idx = np.argmin(distances)

        return self.pareto_results[balanced_idx]

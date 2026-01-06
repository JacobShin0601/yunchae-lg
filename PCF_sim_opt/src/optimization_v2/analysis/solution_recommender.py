"""
솔루션 추천기 (Solution Recommender)

파레토 솔루션을 사용자 기준에 따라 순위화하고 추천합니다.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class SolutionRecommender:
    """
    솔루션 추천기

    다양한 기준으로 파레토 솔루션을 평가하고 순위화합니다.

    추천 기준:
    - minimize_carbon: 탄소 배출 최소화 우선
    - minimize_cost: 비용 최소화 우선
    - balanced: 탄소-비용 균형
    - implementation_ease: 구현 용이성 (변화 최소)
    - risk_averse: 리스크 회피 (baseline과 유사)
    """

    def __init__(
        self,
        pareto_results: List[Dict[str, Any]],
        baseline_solution: Optional[Dict[str, Any]] = None
    ):
        """
        솔루션 추천기 초기화

        Args:
            pareto_results: 파레토 최적화 결과 리스트
            baseline_solution: 기준 솔루션 (변화량 계산용)
        """
        self.pareto_results = pareto_results
        self.baseline_solution = baseline_solution

        # 목적함수 값 추출
        self._extract_objective_values()

    def _extract_objective_values(self):
        """목적함수 값 추출 및 정규화"""
        self.carbon_values = []
        self.cost_values = []

        for result in self.pareto_results:
            summary = result.get('summary', {})
            self.carbon_values.append(summary.get('total_carbon', 0))
            self.cost_values.append(summary.get('total_cost', 0))

        self.carbon_values = np.array(self.carbon_values)
        self.cost_values = np.array(self.cost_values)

        # 정규화 (0-1 범위)
        carbon_range = self.carbon_values.max() - self.carbon_values.min()
        cost_range = self.cost_values.max() - self.cost_values.min()

        if carbon_range > 0:
            self.carbon_norm = (self.carbon_values - self.carbon_values.min()) / carbon_range
        else:
            self.carbon_norm = np.zeros_like(self.carbon_values)

        if cost_range > 0:
            self.cost_norm = (self.cost_values - self.cost_values.min()) / cost_range
        else:
            self.cost_norm = np.zeros_like(self.cost_values)

    def rank_solutions(
        self,
        criteria: str = 'balanced',
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        솔루션 순위화

        Args:
            criteria: 평가 기준
                - 'minimize_carbon': 탄소 최소화
                - 'minimize_cost': 비용 최소화
                - 'balanced': 균형 잡힌 솔루션
                - 'implementation_ease': 구현 용이성
                - 'risk_averse': 리스크 회피
            top_n: 상위 N개 반환

        Returns:
            순위화된 솔루션 데이터프레임
        """
        if criteria == 'minimize_carbon':
            scores = self._score_minimize_carbon()
        elif criteria == 'minimize_cost':
            scores = self._score_minimize_cost()
        elif criteria == 'balanced':
            scores = self._score_balanced()
        elif criteria == 'implementation_ease':
            scores = self._score_implementation_ease()
        elif criteria == 'risk_averse':
            scores = self._score_risk_averse()
        else:
            raise ValueError(f"알 수 없는 기준: {criteria}")

        # 순위화
        ranking_data = []

        for idx, (result, score) in enumerate(zip(self.pareto_results, scores)):
            summary = result.get('summary', {})

            ranking_data.append({
                'Rank': 0,  # 나중에 설정
                'Index': idx,
                'Score': score,
                'Carbon (kg)': summary.get('total_carbon', 0),
                'Cost ($)': summary.get('total_cost', 0),
                'Carbon Reduction (%)': summary.get('carbon_reduction_pct', 0),
                'Cost Premium (%)': summary.get('cost_premium_pct', 0),
                'Carbon Weight': result.get('carbon_weight', 0.5)
            })

        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Score', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df.reset_index(drop=True)

        return ranking_df.head(top_n)

    def _score_minimize_carbon(self) -> np.ndarray:
        """
        탄소 최소화 스코어 계산

        Returns:
            스코어 배열 (높을수록 좋음)
        """
        # 탄소가 낮을수록 높은 점수
        return 1.0 - self.carbon_norm

    def _score_minimize_cost(self) -> np.ndarray:
        """
        비용 최소화 스코어 계산

        Returns:
            스코어 배열
        """
        # 비용이 낮을수록 높은 점수
        return 1.0 - self.cost_norm

    def _score_balanced(self) -> np.ndarray:
        """
        균형 스코어 계산 (탄소-비용 트레이드오프)

        Returns:
            스코어 배열
        """
        # 이상점 (0, 0)으로부터의 거리가 가까울수록 높은 점수
        distances = np.sqrt(self.carbon_norm**2 + self.cost_norm**2)
        max_distance = np.sqrt(2)  # 최대 거리 (1, 1)까지

        return 1.0 - (distances / max_distance)

    def _score_implementation_ease(self) -> np.ndarray:
        """
        구현 용이성 스코어 계산

        baseline 대비 변화량이 적을수록 높은 점수

        Returns:
            스코어 배열
        """
        if self.baseline_solution is None:
            # baseline이 없으면 변화량이 작은 솔루션 선호
            # (탄소 감축과 비용 증가가 모두 작은 것)
            change_magnitude = self.carbon_norm + self.cost_norm
            return 1.0 - (change_magnitude / 2.0)

        # baseline 솔루션의 decision variables와 비교
        baseline_vars = self.baseline_solution.get('decision_variables', {})
        change_scores = []

        for result in self.pareto_results:
            solution_vars = result.get('decision_variables', {})

            # decision variable 변화량 계산
            total_change = 0
            var_count = 0

            for var_name, baseline_value in baseline_vars.items():
                if var_name in solution_vars:
                    solution_value = solution_vars[var_name]

                    # 비율 변수인 경우
                    if isinstance(baseline_value, dict):
                        # 딕셔너리 형태 (자재별 변수)
                        for material, base_val in baseline_value.items():
                            sol_val = solution_value.get(material, 0)
                            if base_val > 0:
                                change = abs((sol_val - base_val) / base_val)
                            else:
                                change = abs(sol_val)
                            total_change += change
                            var_count += 1
                    else:
                        # 단일 값
                        if baseline_value > 0:
                            change = abs((solution_value - baseline_value) / baseline_value)
                        else:
                            change = abs(solution_value)
                        total_change += change
                        var_count += 1

            avg_change = total_change / max(var_count, 1)
            change_scores.append(avg_change)

        change_scores = np.array(change_scores)

        # 정규화
        if change_scores.max() > 0:
            change_norm = change_scores / change_scores.max()
        else:
            change_norm = np.zeros_like(change_scores)

        # 변화가 작을수록 높은 점수
        return 1.0 - change_norm

    def _score_risk_averse(self) -> np.ndarray:
        """
        리스크 회피 스코어 계산

        baseline과 유사하면서 탄소는 감소시키는 솔루션 선호

        Returns:
            스코어 배열
        """
        # 구현 용이성 + 탄소 감축 절충
        ease_scores = self._score_implementation_ease()
        carbon_scores = self._score_minimize_carbon()

        # 7:3 비율 (안정성 우선, 탄소 감축은 부차적)
        return 0.7 * ease_scores + 0.3 * carbon_scores

    def get_recommended_solution(
        self,
        criteria: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        추천 솔루션 반환

        Args:
            criteria: 평가 기준

        Returns:
            최고 점수 솔루션
        """
        ranking = self.rank_solutions(criteria, top_n=1)

        if len(ranking) == 0:
            return None

        best_idx = ranking.iloc[0]['Index']
        return self.pareto_results[int(best_idx)]

    def compare_solutions(
        self,
        solution_indices: List[int]
    ) -> pd.DataFrame:
        """
        여러 솔루션 비교

        Args:
            solution_indices: 비교할 솔루션 인덱스 리스트

        Returns:
            비교 테이블
        """
        comparison_data = []

        for idx in solution_indices:
            if idx < 0 or idx >= len(self.pareto_results):
                continue

            result = self.pareto_results[idx]
            summary = result.get('summary', {})

            comparison_data.append({
                'Index': idx,
                'Total Carbon (kg)': summary.get('total_carbon', 0),
                'Total Cost ($)': summary.get('total_cost', 0),
                'Carbon Reduction (%)': summary.get('carbon_reduction_pct', 0),
                'Cost Premium (%)': summary.get('cost_premium_pct', 0),
                'Carbon Weight': result.get('carbon_weight', 0.5),
                'Cost Weight': result.get('cost_weight', 0.5)
            })

        return pd.DataFrame(comparison_data)

    def explain_recommendation(
        self,
        criteria: str = 'balanced'
    ) -> str:
        """
        추천 이유 설명

        Args:
            criteria: 평가 기준

        Returns:
            설명 텍스트
        """
        explanations = {
            'minimize_carbon': (
                "이 솔루션은 **탄소 배출을 최소화**하는 데 최적화되어 있습니다.\n\n"
                "✅ 환경 영향을 최대한 줄이고자 할 때 적합합니다.\n"
                "⚠️  비용 프리미엄이 가장 높을 수 있습니다."
            ),
            'minimize_cost': (
                "이 솔루션은 **비용을 최소화**하는 데 최적화되어 있습니다.\n\n"
                "✅ 예산 제약이 엄격할 때 적합합니다.\n"
                "⚠️  탄소 감축 효과가 제한적일 수 있습니다."
            ),
            'balanced': (
                "이 솔루션은 **탄소 감축과 비용 효율성의 균형**을 추구합니다.\n\n"
                "✅ 환경과 경제성을 모두 고려한 최적의 절충안입니다.\n"
                "✅ 대부분의 경우 가장 합리적인 선택입니다."
            ),
            'implementation_ease': (
                "이 솔루션은 **구현이 용이**합니다.\n\n"
                "✅ 현재 상태에서 최소한의 변화로 개선할 수 있습니다.\n"
                "✅ 실행 리스크가 낮고 단기간 내 적용 가능합니다."
            ),
            'risk_averse': (
                "이 솔루션은 **리스크를 최소화**합니다.\n\n"
                "✅ 현재 운영 방식을 크게 바꾸지 않으면서 탄소를 감축합니다.\n"
                "✅ 안정성과 점진적 개선을 선호하는 경우 적합합니다."
            )
        }

        return explanations.get(criteria, "추천 기준에 대한 설명이 없습니다.")

    def get_criteria_scores(
        self,
        solution_idx: int
    ) -> Dict[str, float]:
        """
        특정 솔루션의 모든 기준 점수 계산

        Args:
            solution_idx: 솔루션 인덱스

        Returns:
            기준별 점수 딕셔너리
        """
        scores = {}

        for criteria in ['minimize_carbon', 'minimize_cost', 'balanced',
                        'implementation_ease', 'risk_averse']:
            ranking = self.rank_solutions(criteria, top_n=len(self.pareto_results))
            solution_row = ranking[ranking['Index'] == solution_idx]

            if len(solution_row) > 0:
                scores[criteria] = solution_row.iloc[0]['Score']
            else:
                scores[criteria] = 0.0

        return scores

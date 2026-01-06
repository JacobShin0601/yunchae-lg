"""
적합도 점수 계산 시스템

시나리오의 적합도를 다차원적으로 평가합니다.
배출량, 비용, 실현가능성, 리스크 등을 고려한 종합 점수를 계산합니다.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class FitnessWeights:
    """적합도 가중치"""
    emission: float = 0.4  # 배출량 중요도 (0~1)
    cost: float = 0.3  # 비용 중요도 (0~1)
    feasibility: float = 0.2  # 실현가능성 중요도 (0~1)
    risk: float = 0.1  # 리스크 중요도 (0~1)

    def __post_init__(self):
        """가중치 합이 1인지 검증"""
        total = self.emission + self.cost + self.feasibility + self.risk
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"가중치 합이 1이 아닙니다: {total}")


@dataclass
class FitnessScore:
    """적합도 점수"""
    scenario_id: int
    emission_score: float  # 배출량 점수 (0~1, 높을수록 좋음)
    cost_score: float  # 비용 점수 (0~1, 높을수록 좋음)
    feasibility_score: float  # 실현가능성 점수 (0~1, 높을수록 좋음)
    risk_score: float  # 리스크 점수 (0~1, 높을수록 좋음)
    total_score: float  # 종합 점수 (0~1, 높을수록 좋음)
    rank: Optional[int] = None  # 순위


class FitnessCalculator:
    """
    적합도 점수 계산 클래스

    시나리오의 배출량, 비용, 실현가능성, 리스크를 평가하여
    종합 적합도 점수를 계산합니다.
    """

    def __init__(self, weights: Optional[FitnessWeights] = None):
        """
        적합도 계산기 초기화

        Args:
            weights: 가중치 설정 (기본값: FitnessWeights())
        """
        self.weights = weights or FitnessWeights()
        self.scores: List[FitnessScore] = []

    def calculate_fitness(
        self,
        scenario_results: List,
        baseline_emission: Optional[float] = None,
        baseline_cost: Optional[float] = None
    ) -> List[FitnessScore]:
        """
        시나리오들의 적합도 점수 계산

        Args:
            scenario_results: ScenarioResult 리스트
            baseline_emission: 기준 배출량 (정규화용)
            baseline_cost: 기준 비용 (정규화용)

        Returns:
            FitnessScore 리스트
        """
        print("\n" + "="*60)
        print("📊 적합도 점수 계산 시작")
        print("="*60)

        # 실현 가능한 시나리오만 필터링
        feasible_scenarios = [s for s in scenario_results if s.is_feasible]

        if not feasible_scenarios:
            print("⚠️  실현 가능한 시나리오가 없습니다.")
            return []

        print(f"실현 가능한 시나리오: {len(feasible_scenarios)}개")

        # 배출량 및 비용 범위 계산
        emissions = [s.total_emission for s in feasible_scenarios if s.total_emission is not None]
        costs = [s.total_cost for s in feasible_scenarios if s.total_cost is not None]

        if not emissions:
            print("⚠️  배출량 데이터가 없습니다.")
            return []

        min_emission = min(emissions)
        max_emission = max(emissions)
        min_cost = min(costs) if costs else 0
        max_cost = max(costs) if costs else 0

        # 기준값 설정
        if baseline_emission is None:
            baseline_emission = max_emission

        if baseline_cost is None:
            baseline_cost = max_cost if max_cost > 0 else 1.0

        print(f"\n배출량 범위: {min_emission:,.2f} ~ {max_emission:,.2f} kgCO2eq")
        print(f"비용 범위: ${min_cost:,.2f} ~ ${max_cost:,.2f}")
        print(f"기준 배출량: {baseline_emission:,.2f} kgCO2eq")
        print(f"기준 비용: ${baseline_cost:,.2f}")

        # 각 시나리오의 적합도 계산
        scores = []

        for scenario in feasible_scenarios:
            # 1. 배출량 점수 (낮을수록 좋음 → 높은 점수)
            if max_emission > min_emission:
                emission_score = 1.0 - (scenario.total_emission - min_emission) / (max_emission - min_emission)
            else:
                emission_score = 1.0

            # 2. 비용 점수 (낮을수록 좋음 → 높은 점수)
            if scenario.total_cost is not None and max_cost > min_cost:
                cost_score = 1.0 - (scenario.total_cost - min_cost) / (max_cost - min_cost)
            else:
                cost_score = 1.0

            # 3. 실현가능성 점수
            feasibility_score = self._calculate_feasibility_score(scenario)

            # 4. 리스크 점수
            risk_score = self._calculate_risk_score(scenario)

            # 5. 종합 점수 (가중 평균)
            total_score = (
                self.weights.emission * emission_score +
                self.weights.cost * cost_score +
                self.weights.feasibility * feasibility_score +
                self.weights.risk * risk_score
            )

            score = FitnessScore(
                scenario_id=scenario.scenario_id,
                emission_score=round(emission_score, 4),
                cost_score=round(cost_score, 4),
                feasibility_score=round(feasibility_score, 4),
                risk_score=round(risk_score, 4),
                total_score=round(total_score, 4)
            )

            scores.append(score)

        # 점수 순으로 정렬 및 순위 부여
        scores.sort(key=lambda x: x.total_score, reverse=True)

        for rank, score in enumerate(scores, start=1):
            score.rank = rank

        self.scores = scores

        print(f"\n✅ 적합도 계산 완료")
        print(f"  상위 3개 시나리오:")
        for i, score in enumerate(scores[:3], start=1):
            print(f"  {i}. 시나리오 #{score.scenario_id}: 총점 {score.total_score:.3f} "
                  f"(배출 {score.emission_score:.2f} | 비용 {score.cost_score:.2f} | "
                  f"실현 {score.feasibility_score:.2f} | 리스크 {score.risk_score:.2f})")

        print("="*60)

        return scores

    def _calculate_feasibility_score(self, scenario) -> float:
        """
        실현가능성 점수 계산

        Args:
            scenario: ScenarioResult

        Returns:
            점수 (0~1)
        """
        # 기본 점수: 실현 가능하면 1.0
        base_score = 1.0 if scenario.is_feasible else 0.0

        # 솔버 시간 패널티 (긴 시간 = 복잡도 높음 = 약간의 감점)
        time_penalty = min(scenario.solver_time / 60.0, 0.1)  # 최대 10% 감점

        # 파라미터 극단성 패널티 (극단적인 값 = 실현 어려움)
        params = scenario.parameters
        extremity_penalty = 0.0

        if 'recycle_ratio' in params:
            # 재활용 비율이 매우 높으면 감점 (50% 이상)
            if params['recycle_ratio'] > 0.5:
                extremity_penalty += (params['recycle_ratio'] - 0.5) * 0.2

        if 'low_carbon_ratio' in params:
            # 저탄소 비율이 매우 높으면 감점 (50% 이상)
            if params['low_carbon_ratio'] > 0.5:
                extremity_penalty += (params['low_carbon_ratio'] - 0.5) * 0.2

        feasibility_score = max(0.0, base_score - time_penalty - extremity_penalty)

        return feasibility_score

    def _calculate_risk_score(self, scenario) -> float:
        """
        리스크 점수 계산

        Args:
            scenario: ScenarioResult

        Returns:
            점수 (0~1, 높을수록 리스크 낮음)
        """
        params = scenario.parameters

        # 기본 점수
        score = 1.0

        # 리스크 요소 1: 공급망 집중도
        # 재활용재나 저탄소메탈에 과도하게 의존하면 리스크 증가
        recycle = params.get('recycle_ratio', 0)
        low_carbon = params.get('low_carbon_ratio', 0)

        # 단일 소스 의존도가 높으면 리스크 증가
        max_dependency = max(recycle, low_carbon, params.get('virgin_ratio', 1.0))
        if max_dependency > 0.7:
            score -= (max_dependency - 0.7) * 0.5  # 최대 15% 감점

        # 리스크 요소 2: 다변화 점수
        # 재활용, 저탄소, 버진이 적절히 분산되면 리스크 낮음
        ratios = [recycle, low_carbon, params.get('virgin_ratio', 1.0)]
        non_zero_ratios = [r for r in ratios if r > 0.05]  # 5% 이상만 카운트

        if len(non_zero_ratios) >= 2:
            score += 0.1  # 다변화 보너스

        # 리스크 요소 3: 파라미터 안정성
        # 극단적인 값은 불안정 (리스크 증가)
        if recycle > 0.6 or low_carbon > 0.6:
            score -= 0.1

        # 점수 범위 보정
        score = max(0.0, min(1.0, score))

        return score

    def get_top_scenarios(self, n: int = 10) -> List[FitnessScore]:
        """
        상위 N개 시나리오 반환

        Args:
            n: 반환할 시나리오 수

        Returns:
            FitnessScore 리스트
        """
        return self.scores[:n]

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        점수를 DataFrame으로 변환

        Returns:
            점수 DataFrame
        """
        rows = [asdict(score) for score in self.scores]
        return pd.DataFrame(rows)

    def set_weights(self, weights: FitnessWeights) -> None:
        """
        가중치 업데이트

        Args:
            weights: 새로운 가중치
        """
        self.weights = weights
        print(f"✅ 가중치 업데이트됨")
        print(f"  배출: {weights.emission:.2f} | 비용: {weights.cost:.2f} | "
              f"실현: {weights.feasibility:.2f} | 리스크: {weights.risk:.2f}")

    def get_pareto_frontier(self) -> List[FitnessScore]:
        """
        파레토 프론티어 추출

        배출량과 비용 간의 파레토 최적 해를 반환합니다.

        Returns:
            파레토 프론티어에 속하는 FitnessScore 리스트
        """
        if not self.scores:
            return []

        pareto_frontier = []

        # 배출량-비용 기준으로 정렬 (배출량 오름차순, 비용 오름차순)
        sorted_scores = sorted(
            self.scores,
            key=lambda x: (1.0 - x.emission_score, 1.0 - x.cost_score)
        )

        current_best_cost = float('inf')

        for score in sorted_scores:
            cost = 1.0 - score.cost_score  # 낮은 비용 = 높은 점수

            # 현재 배출량 수준에서 비용이 개선되면 파레토 프론티어에 추가
            if cost < current_best_cost:
                pareto_frontier.append(score)
                current_best_cost = cost

        print(f"📈 파레토 프론티어: {len(pareto_frontier)}개 시나리오")

        return pareto_frontier

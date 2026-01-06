"""
솔루션 평가기 (Solution Evaluator)

주어진 솔루션을 여러 시나리오에서 평가하고 성능 분포를 분석합니다.

주요 기능:
- 시나리오별 솔루션 평가
- 성능 통계 계산 (평균, 표준편차, 분위수)
- 리스크 메트릭 (VaR, CVaR, Regret)
- 시각화 데이터 준비
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..core.optimization_engine import OptimizationEngine
from ..utils.data_loader import DataLoader
from .scenario_manager import ScenarioManager, Scenario


class SolutionEvaluator:
    """
    솔루션 평가기

    주요 기능:
    - 시나리오별 성능 평가
    - 통계 분석 (평균, 분산, 분위수)
    - 리스크 메트릭 (VaR, CVaR, Regret)
    - 비교 분석 (여러 솔루션 비교)
    """

    def __init__(
        self,
        optimization_engine: OptimizationEngine,
        data_loader: DataLoader,
        scenario_manager: ScenarioManager
    ):
        """
        솔루션 평가기 초기화

        Args:
            optimization_engine: 최적화 엔진
            data_loader: 데이터 로더
            scenario_manager: 시나리오 관리자
        """
        self.engine = optimization_engine
        self.data_loader = data_loader
        self.scenario_manager = scenario_manager
        self.evaluation_results = {}

    def evaluate_solution(
        self,
        solution: Dict[str, Any],
        base_data: Dict[str, Any],
        objective_type: str = 'minimize_carbon',
        solution_name: str = 'Solution'
    ) -> Dict[str, Any]:
        """
        주어진 솔루션을 모든 시나리오에서 평가

        Args:
            solution: 평가할 솔루션
            base_data: 기준 데이터
            objective_type: 목적함수 유형
            solution_name: 솔루션 이름

        Returns:
            평가 결과 딕셔너리
        """
        print("\n" + "=" * 60)
        print(f"📊 솔루션 평가: {solution_name}")
        print("=" * 60)

        if not self.scenario_manager.scenarios:
            raise ValueError("시나리오가 정의되지 않았습니다.")

        # 각 시나리오에서 솔루션 평가
        scenario_performances = []
        scenario_details = []

        for scenario in self.scenario_manager.scenarios:
            print(f"\n  • 시나리오: {scenario.name} (확률: {scenario.probability:.2%})")

            scenario_data = self.scenario_manager.apply_scenario(base_data, scenario)

            try:
                obj_value = self._evaluate_solution_in_scenario(
                    solution,
                    scenario_data,
                    objective_type
                )

                scenario_performances.append(obj_value)
                scenario_details.append({
                    'scenario_name': scenario.name,
                    'probability': scenario.probability,
                    'objective_value': obj_value
                })

                print(f"    ✅ 목적함수: {obj_value:.4f}")
            except Exception as e:
                print(f"    ❌ 평가 실패: {e}")
                scenario_performances.append(None)
                scenario_details.append({
                    'scenario_name': scenario.name,
                    'probability': scenario.probability,
                    'objective_value': None
                })

        # 유효한 성능 값만 추출
        valid_performances = [p for p in scenario_performances if p is not None]
        valid_probabilities = [
            self.scenario_manager.scenarios[i].probability
            for i, p in enumerate(scenario_performances)
            if p is not None
        ]

        if not valid_performances:
            raise ValueError("모든 시나리오에서 평가 실패")

        # 통계 계산
        statistics = self._calculate_statistics(valid_performances, valid_probabilities)

        # 리스크 메트릭 계산
        risk_metrics = self._calculate_risk_metrics(valid_performances, valid_probabilities)

        # 결과 정리
        result = {
            'solution_name': solution_name,
            'scenario_details': scenario_details,
            'performances': valid_performances,
            'probabilities': valid_probabilities,
            'statistics': statistics,
            'risk_metrics': risk_metrics
        }

        self.evaluation_results[solution_name] = result

        # 요약 출력
        print(f"\n📈 평가 요약:")
        print(f"   • 평균: {statistics['mean']:.4f}")
        print(f"   • 표준편차: {statistics['std']:.4f}")
        print(f"   • 최선: {statistics['min']:.4f}")
        print(f"   • 최악: {statistics['max']:.4f}")
        print(f"   • VaR95: {risk_metrics['var_95']:.4f}")
        print(f"   • CVaR95: {risk_metrics['cvar_95']:.4f}")

        print("\n" + "=" * 60)

        return result

    def compare_solutions(
        self,
        solutions: Dict[str, Dict[str, Any]],
        base_data: Dict[str, Any],
        objective_type: str = 'minimize_carbon'
    ) -> pd.DataFrame:
        """
        여러 솔루션을 비교 평가

        Args:
            solutions: {솔루션_이름: 솔루션_딕셔너리} 형식
            base_data: 기준 데이터
            objective_type: 목적함수 유형

        Returns:
            비교 결과 데이터프레임
        """
        print("\n" + "=" * 60)
        print(f"🔄 솔루션 비교 평가 ({len(solutions)}개)")
        print("=" * 60)

        comparison_results = []

        for solution_name, solution in solutions.items():
            result = self.evaluate_solution(
                solution,
                base_data,
                objective_type,
                solution_name
            )

            comparison_results.append({
                '솔루션': solution_name,
                '평균': result['statistics']['mean'],
                '표준편차': result['statistics']['std'],
                '최선': result['statistics']['min'],
                '최악': result['statistics']['max'],
                'VaR95': result['risk_metrics']['var_95'],
                'CVaR95': result['risk_metrics']['cvar_95'],
                '중앙값': result['statistics']['median']
            })

        comparison_df = pd.DataFrame(comparison_results)

        # 평균 기준으로 정렬
        comparison_df = comparison_df.sort_values('평균')

        print("\n📊 비교 결과:")
        print(comparison_df.to_string(index=False))

        return comparison_df

    def _evaluate_solution_in_scenario(
        self,
        solution: Dict[str, Any],
        scenario_data: Dict[str, Any],
        objective_type: str
    ) -> float:
        """
        주어진 솔루션을 특정 시나리오에서 평가

        Args:
            solution: 평가할 솔루션
            scenario_data: 시나리오 데이터
            objective_type: 목적함수 유형

        Returns:
            목적함수 값
        """
        # 모델 구축
        model = self.engine.build_model(scenario_data, objective_type)

        # 솔루션의 결정 변수 값을 모델에 고정
        for material_name, material_result in solution['materials'].items():
            if material_name not in self.engine.model.materials:
                continue

            # Formula 자재 변수 고정
            if 'tier1_re' in material_result:
                self.engine.model.tier1_re[material_name].fix(material_result['tier1_re'])
            if 'tier2_re' in material_result:
                self.engine.model.tier2_re[material_name].fix(material_result['tier2_re'])

            # Ni/Co/Li 자재 변수 고정
            if 'recycle_ratio' in material_result:
                self.engine.model.recycle_ratio[material_name].fix(material_result['recycle_ratio'])
            if 'low_carbon_ratio' in material_result:
                self.engine.model.low_carbon_ratio[material_name].fix(material_result['low_carbon_ratio'])
            if 'virgin_ratio' in material_result:
                self.engine.model.virgin_ratio[material_name].fix(material_result['virgin_ratio'])

        # 양극재 원소별 변수 고정 (있는 경우)
        if 'cathode' in solution and 'elements' in solution['cathode']:
            if hasattr(self.engine.model, 'elements'):
                for element, ratios in solution['cathode']['elements'].items():
                    if element in self.engine.model.elements:
                        self.engine.model.element_virgin_ratio[element].fix(ratios['virgin_ratio'])
                        self.engine.model.element_recycle_ratio[element].fix(ratios['recycle_ratio'])
                        self.engine.model.element_low_carb_ratio[element].fix(ratios['low_carbon_ratio'])

        # 최적화 실행 (변수가 고정되어 있어 즉시 계산됨)
        results = self.engine.solve()

        import pyomo.environ as pyo
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            return pyo.value(self.engine.model.objective)
        elif results.solver.termination_condition == pyo.TerminationCondition.feasible:
            return pyo.value(self.engine.model.objective)
        else:
            raise ValueError(f"솔루션 평가 실패: {results.solver.termination_condition}")

    def _calculate_statistics(
        self,
        performances: List[float],
        probabilities: List[float]
    ) -> Dict[str, float]:
        """
        성능 통계 계산

        Args:
            performances: 성능 값 리스트
            probabilities: 확률 리스트

        Returns:
            통계 딕셔너리
        """
        performances_array = np.array(performances)
        probabilities_array = np.array(probabilities)

        # 확률 가중 평균
        mean = np.average(performances_array, weights=probabilities_array)

        # 확률 가중 분산
        variance = np.average((performances_array - mean) ** 2, weights=probabilities_array)
        std = np.sqrt(variance)

        # 분위수 (확률 가중)
        sorted_indices = np.argsort(performances_array)
        sorted_performances = performances_array[sorted_indices]
        sorted_probabilities = probabilities_array[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probabilities)

        # 중앙값 (50th percentile)
        median_idx = np.searchsorted(cumulative_probs, 0.5)
        median = sorted_performances[median_idx] if median_idx < len(sorted_performances) else sorted_performances[-1]

        # 25th, 75th percentile
        p25_idx = np.searchsorted(cumulative_probs, 0.25)
        p75_idx = np.searchsorted(cumulative_probs, 0.75)
        p25 = sorted_performances[p25_idx] if p25_idx < len(sorted_performances) else sorted_performances[-1]
        p75 = sorted_performances[p75_idx] if p75_idx < len(sorted_performances) else sorted_performances[-1]

        return {
            'mean': mean,
            'std': std,
            'min': float(np.min(performances_array)),
            'max': float(np.max(performances_array)),
            'median': median,
            'p25': p25,
            'p75': p75,
            'range': float(np.max(performances_array) - np.min(performances_array))
        }

    def _calculate_risk_metrics(
        self,
        performances: List[float],
        probabilities: List[float],
        alpha_var: float = 0.95,
        alpha_cvar: float = 0.95
    ) -> Dict[str, float]:
        """
        리스크 메트릭 계산

        Args:
            performances: 성능 값 리스트
            probabilities: 확률 리스트
            alpha_var: VaR 신뢰수준 (0.95 = 95%)
            alpha_cvar: CVaR 신뢰수준 (0.95 = 95%)

        Returns:
            리스크 메트릭 딕셔너리
        """
        performances_array = np.array(performances)
        probabilities_array = np.array(probabilities)

        # 정렬
        sorted_indices = np.argsort(performances_array)[::-1]  # 내림차순 (worst first)
        sorted_performances = performances_array[sorted_indices]
        sorted_probabilities = probabilities_array[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probabilities)

        # VaR (Value at Risk): α 백분위수
        var_threshold = 1 - alpha_var
        var_idx = np.searchsorted(cumulative_probs, var_threshold)
        var = sorted_performances[var_idx] if var_idx < len(sorted_performances) else sorted_performances[-1]

        # CVaR (Conditional Value at Risk): worst α% 시나리오의 평균
        cvar_mask = cumulative_probs <= var_threshold
        if np.any(cvar_mask):
            cvar_performances = sorted_performances[cvar_mask]
            cvar_probabilities = sorted_probabilities[cvar_mask]
            # 정규화
            cvar_probabilities_normalized = cvar_probabilities / np.sum(cvar_probabilities)
            cvar = np.average(cvar_performances, weights=cvar_probabilities_normalized)
        else:
            cvar = var

        return {
            'var_95': float(var),
            'cvar_95': float(cvar),
            'worst_case': float(np.max(performances_array)),
            'best_case': float(np.min(performances_array))
        }

    def get_visualization_data(
        self,
        solution_name: str
    ) -> Dict[str, Any]:
        """
        시각화용 데이터 준비

        Args:
            solution_name: 솔루션 이름

        Returns:
            시각화 데이터 딕셔너리
        """
        if solution_name not in self.evaluation_results:
            raise ValueError(f"솔루션 '{solution_name}'의 평가 결과가 없습니다.")

        result = self.evaluation_results[solution_name]

        # Violin plot / Box plot 데이터
        scenario_df = pd.DataFrame(result['scenario_details'])
        scenario_df = scenario_df[scenario_df['objective_value'].notna()]

        # 히스토그램 데이터 (확률 가중)
        performances = result['performances']
        probabilities = result['probabilities']

        return {
            'scenario_dataframe': scenario_df,
            'performances': performances,
            'probabilities': probabilities,
            'statistics': result['statistics'],
            'risk_metrics': result['risk_metrics']
        }

    def export_results(self, solution_name: str) -> Dict[str, Any]:
        """
        평가 결과 Export

        Args:
            solution_name: 솔루션 이름

        Returns:
            직렬화 가능한 결과 딕셔너리
        """
        if solution_name not in self.evaluation_results:
            raise ValueError(f"솔루션 '{solution_name}'의 평가 결과가 없습니다.")

        result = self.evaluation_results[solution_name]

        return {
            'solution_name': result['solution_name'],
            'scenario_details': result['scenario_details'],
            'statistics': result['statistics'],
            'risk_metrics': result['risk_metrics']
        }

    def __repr__(self) -> str:
        return f"<SolutionEvaluator(scenarios={len(self.scenario_manager.scenarios)}, evaluated={len(self.evaluation_results)})>"

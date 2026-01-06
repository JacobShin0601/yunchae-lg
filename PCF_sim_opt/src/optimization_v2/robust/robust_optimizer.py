"""
강건 최적화기 (Robust Optimizer)

불확실성을 고려한 강건한 최적화 솔루션을 찾습니다.

주요 기능:
- Minimax Regret 공식
- Expected Value + CVaR 공식
- Light Robust 공식
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import copy
from ..core.optimization_engine import OptimizationEngine
from ..utils.data_loader import DataLoader
from .scenario_manager import ScenarioManager, Scenario


class RobustOptimizer:
    """
    강건 최적화기

    세 가지 강건 최적화 공식 제공:
    1. Minimax Regret: 최악의 후회(regret)를 최소화
    2. Expected Value + CVaR: 기댓값과 조건부 위험가치(CVaR) 균형
    3. Light Robust: 모든 시나리오에서 실행가능, 기준 케이스에서 최적
    """

    def __init__(
        self,
        optimization_engine: OptimizationEngine,
        data_loader: DataLoader,
        scenario_manager: ScenarioManager
    ):
        """
        강건 최적화기 초기화

        Args:
            optimization_engine: 최적화 엔진
            data_loader: 데이터 로더
            scenario_manager: 시나리오 관리자
        """
        self.engine = optimization_engine
        self.data_loader = data_loader
        self.scenario_manager = scenario_manager
        self.robust_solutions = {}
        self.scenario_results = {}

    def optimize_minimax_regret(
        self,
        base_data: Dict[str, Any],
        objective_type: str = 'minimize_carbon',
        n_candidates: int = 10
    ) -> Dict[str, Any]:
        """
        Minimax Regret 강건 최적화

        각 시나리오에서 최적해를 찾은 후,
        모든 시나리오에서 후회(regret)의 최댓값이 가장 작은 솔루션을 선택합니다.

        Regret_s(x) = Obj_s(x) - Obj*_s
        여기서 Obj*_s는 시나리오 s에서의 최적 목적함수 값

        Args:
            base_data: 기준 최적화 데이터
            objective_type: 목적함수 유형
            n_candidates: 후보 솔루션 수

        Returns:
            Minimax regret 솔루션 및 분석 결과
        """
        print("\n" + "=" * 60)
        print("🛡️  Minimax Regret 강건 최적화")
        print("=" * 60)

        if not self.scenario_manager.scenarios:
            raise ValueError("시나리오가 정의되지 않았습니다. ScenarioManager에 시나리오를 추가하세요.")

        # 확률 검증
        if not self.scenario_manager.validate_probabilities():
            print("⚠️  확률 합이 1이 아닙니다. 정규화합니다...")
            self.scenario_manager.normalize_probabilities()

        # Step 1: 각 시나리오에서 최적해 찾기 (최적 목적함수 값 계산)
        print("\n📊 Step 1: 각 시나리오에서 최적 솔루션 찾기...")

        scenario_optimal_objectives = {}
        scenario_optimal_solutions = {}

        for scenario in self.scenario_manager.scenarios:
            print(f"\n  • 시나리오: {scenario.name} (확률: {scenario.probability:.2%})")

            # 시나리오 데이터 생성
            scenario_data = self.scenario_manager.apply_scenario(base_data, scenario)

            # 최적화 실행
            try:
                model = self.engine.build_model(scenario_data, objective_type)
                results = self.engine.solve()

                import pyomo.environ as pyo
                if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                    optimal_obj = pyo.value(self.engine.model.objective)
                    scenario_optimal_objectives[scenario.name] = optimal_obj
                    scenario_optimal_solutions[scenario.name] = self.engine.extract_solution()
                    print(f"    ✅ 최적 목적함수: {optimal_obj:.4f}")
                else:
                    print(f"    ❌ 최적화 실패: {results.solver.termination_condition}")
                    scenario_optimal_objectives[scenario.name] = float('inf')
                    scenario_optimal_solutions[scenario.name] = None
            except Exception as e:
                print(f"    ❌ 오류 발생: {e}")
                scenario_optimal_objectives[scenario.name] = float('inf')
                scenario_optimal_solutions[scenario.name] = None

        # Step 2: 후보 솔루션 생성 (각 시나리오 최적해 + 기준 케이스)
        print(f"\n📋 Step 2: 후보 솔루션 생성 ({len(scenario_optimal_solutions)}개 시나리오 최적해)...")

        candidate_solutions = {}

        # 각 시나리오의 최적해를 후보로 추가
        for scenario_name, solution in scenario_optimal_solutions.items():
            if solution is not None:
                candidate_solutions[f"opt_in_{scenario_name}"] = solution

        # 기준 케이스 솔루션도 후보로 추가
        print("\n  • 기준 케이스 솔루션 추가...")
        try:
            model = self.engine.build_model(base_data, objective_type)
            results = self.engine.solve()

            import pyomo.environ as pyo
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                baseline_solution = self.engine.extract_solution()
                candidate_solutions['baseline'] = baseline_solution
                print("    ✅ 기준 케이스 솔루션 추가 완료")
            else:
                print("    ⚠️  기준 케이스 최적화 실패")
        except Exception as e:
            print(f"    ⚠️  기준 케이스 오류: {e}")

        print(f"\n✅ 총 {len(candidate_solutions)}개 후보 솔루션 생성")

        # Step 3: 각 후보 솔루션의 Minimax Regret 계산
        print(f"\n🔍 Step 3: 후보 솔루션의 Minimax Regret 계산...")

        candidate_regrets = {}

        for candidate_name, candidate_solution in candidate_solutions.items():
            print(f"\n  • 후보: {candidate_name}")

            # 이 후보 솔루션을 각 시나리오에 적용했을 때의 regret 계산
            regrets = []

            for scenario in self.scenario_manager.scenarios:
                scenario_data = self.scenario_manager.apply_scenario(base_data, scenario)

                # 후보 솔루션의 결정 변수를 시나리오 데이터에 고정
                try:
                    fixed_obj = self._evaluate_solution_in_scenario(
                        candidate_solution,
                        scenario_data,
                        objective_type
                    )

                    optimal_obj = scenario_optimal_objectives[scenario.name]

                    if optimal_obj != float('inf') and fixed_obj != float('inf'):
                        regret = fixed_obj - optimal_obj
                        regrets.append(regret)
                        print(f"    - {scenario.name}: Obj={fixed_obj:.4f}, Optimal={optimal_obj:.4f}, Regret={regret:.4f}")
                    else:
                        regrets.append(float('inf'))
                        print(f"    - {scenario.name}: 평가 불가")
                except Exception as e:
                    regrets.append(float('inf'))
                    print(f"    - {scenario.name}: 오류 ({e})")

            # Minimax regret (최대 후회)
            max_regret = max(regrets) if regrets else float('inf')
            candidate_regrets[candidate_name] = {
                'max_regret': max_regret,
                'regrets': regrets,
                'solution': candidate_solution
            }

            print(f"    ➡️  Max Regret: {max_regret:.4f}")

        # Step 4: Minimax regret 최소인 솔루션 선택
        print(f"\n🏆 Step 4: 최선의 강건 솔루션 선택...")

        best_candidate_name = min(
            candidate_regrets.keys(),
            key=lambda k: candidate_regrets[k]['max_regret']
        )

        best_solution_info = candidate_regrets[best_candidate_name]

        print(f"\n✅ 선택된 솔루션: {best_candidate_name}")
        print(f"   • Max Regret: {best_solution_info['max_regret']:.4f}")

        # 결과 정리
        result = {
            'method': 'minimax_regret',
            'best_candidate': best_candidate_name,
            'robust_solution': best_solution_info['solution'],
            'max_regret': best_solution_info['max_regret'],
            'scenario_regrets': {
                scenario.name: best_solution_info['regrets'][i]
                for i, scenario in enumerate(self.scenario_manager.scenarios)
            },
            'scenario_optimal_objectives': scenario_optimal_objectives,
            'all_candidate_regrets': {
                name: info['max_regret']
                for name, info in candidate_regrets.items()
            }
        }

        self.robust_solutions['minimax_regret'] = result

        print("\n" + "=" * 60)
        print("✅ Minimax Regret 최적화 완료")
        print("=" * 60)

        return result

    def optimize_expected_cvar(
        self,
        base_data: Dict[str, Any],
        objective_type: str = 'minimize_carbon',
        lambda_risk: float = 0.3,
        beta: float = 0.95
    ) -> Dict[str, Any]:
        """
        Expected Value + CVaR 강건 최적화

        목적함수 = E[Obj] + λ * CVaR_β[Obj]

        CVaR_β는 worst β% 시나리오에서의 조건부 기댓값입니다.

        Args:
            base_data: 기준 최적화 데이터
            objective_type: 목적함수 유형
            lambda_risk: CVaR 가중치 (0~1, 높을수록 리스크 회피)
            beta: CVaR 임계값 (0.9~0.99, 높을수록 극단 시나리오 중시)

        Returns:
            Expected Value + CVaR 솔루션 및 분석 결과
        """
        print("\n" + "=" * 60)
        print(f"📊 Expected Value + CVaR 강건 최적화")
        print(f"   λ={lambda_risk:.2f}, β={beta:.2%}")
        print("=" * 60)

        if not self.scenario_manager.scenarios:
            raise ValueError("시나리오가 정의되지 않았습니다.")

        # 확률 검증
        if not self.scenario_manager.validate_probabilities():
            print("⚠️  확률 합이 1이 아닙니다. 정규화합니다...")
            self.scenario_manager.normalize_probabilities()

        # Step 1: 각 시나리오에서 최적해 찾기
        print("\n📊 Step 1: 각 시나리오에서 최적 솔루션 찾기...")

        scenario_solutions = {}

        for scenario in self.scenario_manager.scenarios:
            print(f"\n  • 시나리오: {scenario.name} (확률: {scenario.probability:.2%})")

            scenario_data = self.scenario_manager.apply_scenario(base_data, scenario)

            try:
                model = self.engine.build_model(scenario_data, objective_type)
                results = self.engine.solve()

                import pyomo.environ as pyo
                if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                    solution = self.engine.extract_solution()
                    scenario_solutions[scenario.name] = solution
                    print(f"    ✅ 목적함수: {solution['objective_value']:.4f}")
                else:
                    print(f"    ❌ 최적화 실패")
                    scenario_solutions[scenario.name] = None
            except Exception as e:
                print(f"    ❌ 오류: {e}")
                scenario_solutions[scenario.name] = None

        # Step 2: 후보 솔루션 생성 및 평가
        print(f"\n🔍 Step 2: 후보 솔루션 평가 (Expected Value + CVaR)...")

        candidate_scores = {}

        # 각 시나리오 최적해 + 기준 케이스를 후보로
        candidates = {}

        for scenario_name, solution in scenario_solutions.items():
            if solution is not None:
                candidates[f"opt_in_{scenario_name}"] = solution

        # 기준 케이스 솔루션
        try:
            model = self.engine.build_model(base_data, objective_type)
            results = self.engine.solve()

            import pyomo.environ as pyo
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                baseline_solution = self.engine.extract_solution()
                candidates['baseline'] = baseline_solution
        except:
            pass

        print(f"\n✅ 총 {len(candidates)}개 후보 솔루션 평가")

        for candidate_name, candidate_solution in candidates.items():
            print(f"\n  • 후보: {candidate_name}")

            # 각 시나리오에서 이 솔루션의 목적함수 값 계산
            objectives = []
            probabilities = []

            for scenario in self.scenario_manager.scenarios:
                scenario_data = self.scenario_manager.apply_scenario(base_data, scenario)

                try:
                    obj_value = self._evaluate_solution_in_scenario(
                        candidate_solution,
                        scenario_data,
                        objective_type
                    )

                    objectives.append(obj_value)
                    probabilities.append(scenario.probability)
                    print(f"    - {scenario.name}: {obj_value:.4f}")
                except Exception as e:
                    objectives.append(float('inf'))
                    probabilities.append(scenario.probability)
                    print(f"    - {scenario.name}: 평가 불가")

            # Expected Value 계산
            expected_value = np.average(objectives, weights=probabilities)

            # CVaR 계산 (worst β% 시나리오의 조건부 기댓값)
            sorted_indices = np.argsort(objectives)[::-1]  # 내림차순 (worst first)
            cumulative_prob = 0.0
            cvar_values = []
            cvar_probs = []

            for idx in sorted_indices:
                if cumulative_prob < (1 - beta):
                    cvar_values.append(objectives[idx])
                    cvar_probs.append(probabilities[idx])
                    cumulative_prob += probabilities[idx]

            if cvar_values:
                cvar = np.average(cvar_values, weights=cvar_probs)
            else:
                cvar = max(objectives)

            # 복합 목적함수
            composite_score = expected_value + lambda_risk * cvar

            candidate_scores[candidate_name] = {
                'expected_value': expected_value,
                'cvar': cvar,
                'composite_score': composite_score,
                'objectives': objectives,
                'solution': candidate_solution
            }

            print(f"    ➡️  E[Obj]={expected_value:.4f}, CVaR={cvar:.4f}, Score={composite_score:.4f}")

        # Step 3: 최선의 솔루션 선택 (composite_score 최소)
        print(f"\n🏆 Step 3: 최선의 강건 솔루션 선택...")

        best_candidate_name = min(
            candidate_scores.keys(),
            key=lambda k: candidate_scores[k]['composite_score']
        )

        best_solution_info = candidate_scores[best_candidate_name]

        print(f"\n✅ 선택된 솔루션: {best_candidate_name}")
        print(f"   • Expected Value: {best_solution_info['expected_value']:.4f}")
        print(f"   • CVaR_{beta:.0%}: {best_solution_info['cvar']:.4f}")
        print(f"   • Composite Score: {best_solution_info['composite_score']:.4f}")

        # 결과 정리
        result = {
            'method': 'expected_cvar',
            'best_candidate': best_candidate_name,
            'robust_solution': best_solution_info['solution'],
            'expected_value': best_solution_info['expected_value'],
            'cvar': best_solution_info['cvar'],
            'composite_score': best_solution_info['composite_score'],
            'lambda_risk': lambda_risk,
            'beta': beta,
            'scenario_objectives': {
                scenario.name: best_solution_info['objectives'][i]
                for i, scenario in enumerate(self.scenario_manager.scenarios)
            },
            'all_candidate_scores': {
                name: info['composite_score']
                for name, info in candidate_scores.items()
            }
        }

        self.robust_solutions['expected_cvar'] = result

        print("\n" + "=" * 60)
        print("✅ Expected Value + CVaR 최적화 완료")
        print("=" * 60)

        return result

    def optimize_light_robust(
        self,
        base_data: Dict[str, Any],
        objective_type: str = 'minimize_carbon'
    ) -> Dict[str, Any]:
        """
        Light Robust 강건 최적화

        모든 시나리오에서 제약조건을 만족하면서,
        기준 시나리오에서 목적함수를 최적화합니다.

        가장 보수적인 접근이지만, 모든 시나리오에서 실행가능성이 보장됩니다.

        Args:
            base_data: 기준 최적화 데이터
            objective_type: 목적함수 유형

        Returns:
            Light robust 솔루션 및 분석 결과
        """
        print("\n" + "=" * 60)
        print("🔒 Light Robust 강건 최적화")
        print("=" * 60)

        if not self.scenario_manager.scenarios:
            raise ValueError("시나리오가 정의되지 않았습니다.")

        print("\n⚠️  현재 구현: 단순화된 Light Robust")
        print("   • 기준 케이스에서 최적화")
        print("   • 모든 시나리오에서 실행가능성 검증")
        print("   • 완전한 구현은 Pyomo에서 모든 시나리오 제약을 동시에 추가해야 함")

        # Step 1: 기준 케이스에서 최적화
        print("\n📊 Step 1: 기준 케이스 최적화...")

        try:
            model = self.engine.build_model(base_data, objective_type)
            results = self.engine.solve()

            import pyomo.environ as pyo
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                raise ValueError("기준 케이스 최적화 실패")

            candidate_solution = self.engine.extract_solution()
            base_objective = candidate_solution['objective_value']

            print(f"✅ 기준 목적함수: {base_objective:.4f}")
        except Exception as e:
            raise ValueError(f"기준 케이스 최적화 실패: {e}")

        # Step 2: 모든 시나리오에서 실행가능성 검증
        print(f"\n🔍 Step 2: 모든 시나리오에서 실행가능성 검증...")

        scenario_feasibility = {}
        scenario_objectives = {}

        all_feasible = True

        for scenario in self.scenario_manager.scenarios:
            print(f"\n  • 시나리오: {scenario.name} (확률: {scenario.probability:.2%})")

            scenario_data = self.scenario_manager.apply_scenario(base_data, scenario)

            try:
                obj_value = self._evaluate_solution_in_scenario(
                    candidate_solution,
                    scenario_data,
                    objective_type
                )

                scenario_feasibility[scenario.name] = True
                scenario_objectives[scenario.name] = obj_value
                print(f"    ✅ 실행가능, Obj={obj_value:.4f}")
            except Exception as e:
                scenario_feasibility[scenario.name] = False
                scenario_objectives[scenario.name] = None
                all_feasible = False
                print(f"    ❌ 실행불가능: {e}")

        # Step 3: 결과 정리
        if all_feasible:
            print(f"\n✅ 모든 시나리오에서 실행가능!")
            status = 'feasible_in_all_scenarios'
        else:
            print(f"\n⚠️  일부 시나리오에서 실행불가능")
            status = 'infeasible_in_some_scenarios'

        result = {
            'method': 'light_robust',
            'status': status,
            'robust_solution': candidate_solution if all_feasible else None,
            'base_objective': base_objective,
            'scenario_feasibility': scenario_feasibility,
            'scenario_objectives': scenario_objectives,
            'all_feasible': all_feasible
        }

        self.robust_solutions['light_robust'] = result

        print("\n" + "=" * 60)
        print("✅ Light Robust 최적화 완료")
        print("=" * 60)

        return result

    def _evaluate_solution_in_scenario(
        self,
        solution: Dict[str, Any],
        scenario_data: Dict[str, Any],
        objective_type: str
    ) -> float:
        """
        주어진 솔루션을 특정 시나리오에서 평가

        솔루션의 결정 변수를 시나리오 데이터에 고정하고 목적함수 값 계산

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
        material_classification = scenario_data['material_classification']

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

    def compare_robust_methods(self) -> Dict[str, Any]:
        """
        구현된 모든 강건 최적화 방법 비교

        Returns:
            비교 결과 딕셔너리
        """
        if not self.robust_solutions:
            raise ValueError("먼저 강건 최적화를 실행하세요.")

        print("\n" + "=" * 60)
        print("📊 강건 최적화 방법 비교")
        print("=" * 60)

        comparison = {}

        for method_name, result in self.robust_solutions.items():
            print(f"\n🔹 {method_name.upper()}")

            if method_name == 'minimax_regret':
                print(f"   • Max Regret: {result['max_regret']:.4f}")
                print(f"   • Best Candidate: {result['best_candidate']}")
                comparison[method_name] = {
                    'metric': result['max_regret'],
                    'metric_name': 'Max Regret'
                }

            elif method_name == 'expected_cvar':
                print(f"   • Expected Value: {result['expected_value']:.4f}")
                print(f"   • CVaR: {result['cvar']:.4f}")
                print(f"   • Composite Score: {result['composite_score']:.4f}")
                comparison[method_name] = {
                    'metric': result['composite_score'],
                    'metric_name': 'Composite Score'
                }

            elif method_name == 'light_robust':
                print(f"   • Base Objective: {result['base_objective']:.4f}")
                print(f"   • All Feasible: {result['all_feasible']}")
                comparison[method_name] = {
                    'metric': result['base_objective'] if result['all_feasible'] else float('inf'),
                    'metric_name': 'Base Objective'
                }

        print("\n" + "=" * 60)

        return comparison

    def export_results(self) -> Dict[str, Any]:
        """
        강건 최적화 결과 Export

        Returns:
            직렬화 가능한 결과 딕셔너리
        """
        if not self.robust_solutions:
            return {}

        export_data = {}

        for method_name, result in self.robust_solutions.items():
            # 딥 카피하여 수정
            method_data = copy.deepcopy(result)

            # 솔루션 객체는 제외 (너무 큼)
            if 'robust_solution' in method_data:
                method_data['has_solution'] = method_data['robust_solution'] is not None
                method_data.pop('robust_solution', None)

            export_data[method_name] = method_data

        return export_data

    def __repr__(self) -> str:
        return f"<RobustOptimizer(scenarios={len(self.scenario_manager.scenarios)}, methods_run={len(self.robust_solutions)})>"

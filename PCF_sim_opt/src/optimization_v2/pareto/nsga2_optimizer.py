"""
NSGA-II 진화 알고리즘 기반 파레토 최적화
"""

import streamlit as st
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import random

from .config_loader import ParetoConfigLoader
from .base_pareto_optimizer import BaseParetoOptimizer
from ..core.optimization_engine import OptimizationEngine
from ..core.result_processor import ResultProcessor
from ..utils.total_cost_calculator import TotalCostCalculator


@dataclass
class Individual:
    """개체 (염색체)"""
    genes: Dict[str, Any]  # 의사결정 변수 값
    objectives: Tuple[float, float]  # (탄소, 비용)
    rank: int = 0  # 파레토 계층
    crowding_distance: float = 0.0  # 혼잡도 거리
    feasible: bool = True  # 실현가능성

    def dominates(self, other: 'Individual') -> bool:
        """지배 관계 확인"""
        # Infeasible solutions don't dominate anyone
        if not self.feasible:
            return False

        # Feasible solutions dominate infeasible ones
        if self.feasible and not other.feasible:
            return True

        # Check for invalid objectives (penalties)
        if self.objectives[0] >= 1e10 or self.objectives[1] >= 1e10:
            return False
        if other.objectives[0] >= 1e10 or other.objectives[1] >= 1e10:
            return True

        # self가 other를 지배하는가?
        carbon_better = self.objectives[0] <= other.objectives[0]
        cost_better = self.objectives[1] <= other.objectives[1]
        at_least_one_strictly_better = (
            self.objectives[0] < other.objectives[0] or
            self.objectives[1] < other.objectives[1]
        )

        return carbon_better and cost_better and at_least_one_strictly_better


class NSGA2Optimizer(BaseParetoOptimizer):
    """NSGA-II 최적화 실행기"""

    def __init__(self, user_id: str = None):
        super().__init__(user_id)  # Call base class init
        self.config = self.config_loader.config.get('nsga2', {})

        self.population = []
        self.pareto_history = []  # 세대별 파레토 프론티어 기록
        self.hypervolume_history = []  # 세대별 hypervolume 기록
        self.diversity_history = []  # 세대별 diversity 기록

    def run_optimization(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run Pareto optimization using NSGA-II evolutionary algorithm

        This implements the abstract method from BaseParetoOptimizer.

        Args:
            optimization_data: Optimization data
            cost_calculator: RE100PremiumCalculator instance
            **kwargs: Additional parameters for run_nsga2
                - baseline_case: str (default: 'case1')
                - constraint_preset: str (default: 'medium')
                - scenario_template: str (optional)

        Returns:
            List of Pareto points (results)
        """
        # Extract kwargs with defaults
        baseline_case = kwargs.get('baseline_case', 'case1')
        constraint_preset = kwargs.get('constraint_preset', 'medium')
        scenario_template = kwargs.get('scenario_template', None)

        # Call the standard NSGA-II method
        return self.run_nsga2(
            optimization_data=optimization_data,
            cost_calculator=cost_calculator,
            baseline_case=baseline_case,
            constraint_preset=constraint_preset,
            scenario_template=scenario_template
        )

    def run_nsga2(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str = 'case1',
        constraint_preset: str = 'medium',
        scenario_template: str = None
    ) -> List[Dict[str, Any]]:
        """
        NSGA-II 실행

        Args:
            optimization_data: 최적화 데이터
            cost_calculator: RE100PremiumCalculator 인스턴스
            baseline_case: 기준 케이스
            constraint_preset: 제약조건 프리셋
            scenario_template: 시나리오 템플릿 (optional)

        Returns:
            파레토 프론티어 해 리스트
        """
        # Setup cost calculator and calculate baseline using base class
        self.setup_cost_calculator(cost_calculator)
        baselines = self.calculate_baseline(optimization_data)
        # zero_premium_baseline is now stored in self.zero_premium_baseline by base class

        # Store additional references
        self.cost_calculator = cost_calculator
        self.optimization_data = optimization_data

        population_size = self.config.get('population_size', 50)
        max_generations = self.config.get('generations', 100)

        print(f"\n🧬 NSGA-II 최적화 시작")
        print(f"   Zero-Premium Baseline: ${self.zero_premium_baseline:,.2f}")
        print(f"   개체 수: {population_size}")
        print(f"   세대 수: {max_generations}")

        # 1. 초기 집단 생성
        self.population = self._initialize_population(
            population_size,
            optimization_data,
            cost_calculator,
            baseline_case,
            constraint_preset,
            scenario_template
        )

        print(f"   ✅ 초기 집단 생성 완료: {len(self.population)}개체")

        # Check if we have any feasible solutions
        feasible_count = sum(1 for ind in self.population if ind.feasible)
        print(f"   ℹ️  실현가능 개체: {feasible_count}/{len(self.population)}개")

        if feasible_count == 0:
            raise RuntimeError(
                "No feasible solutions found in initial population. "
                "Try relaxing constraints or adjusting the constraint preset."
            )

        # 2. 진화 루프
        for generation in range(max_generations):
            print(f"\n[세대 {generation+1}/{max_generations}]")

            # 2.1 비지배 정렬
            fronts = self._non_dominated_sort(self.population)

            # Safety check: ensure we have at least one front with solutions
            if not fronts or not fronts[0]:
                print(f"   ⚠️ Warning: No non-dominated solutions found in generation {generation+1}")
                print(f"   Hint: Try relaxing constraints or increasing population size")
                continue

            print(f"   파레토 계층: {len(fronts)}개")

            # 2.2 혼잡도 거리 계산
            for front in fronts:
                self._calculate_crowding_distance(front)

            # 2.3 파레토 프론티어 저장 및 메트릭 계산
            pareto_front = fronts[0]
            self.pareto_history.append([
                {
                    'generation': generation + 1,
                    'carbon': ind.objectives[0],
                    'cost': ind.objectives[1],
                    'genes': ind.genes
                }
                for ind in pareto_front
            ])

            # Calculate and store hypervolume
            objectives = [(ind.objectives[0], ind.objectives[1]) for ind in pareto_front]
            hv = self._calculate_hypervolume(objectives)
            self.hypervolume_history.append(hv)

            # Calculate and store diversity
            diversity = self._calculate_diversity(objectives)
            self.diversity_history.append(diversity)

            print(f"   파레토 프론티어 크기: {len(pareto_front)}")
            print(f"   Hypervolume: {hv:.2e}, Diversity: {diversity:.4f}")

            # 2.4 조기 종료 체크
            if self._check_convergence(generation):
                print(f"\n🏁 수렴 감지, 조기 종료 (세대 {generation+1})")
                break

            # 2.5 선택/교차/돌연변이로 자손 생성
            offspring = self._create_offspring(
                self.population,
                population_size,
                optimization_data,
                cost_calculator,
                baseline_case,
                constraint_preset,
                scenario_template
            )

            # 2.6 부모 + 자손 합병 후 선택
            combined = self.population + offspring
            self.population = self._environmental_selection(combined, population_size)

        # 3. 최종 파레토 프론티어 추출
        final_fronts = self._non_dominated_sort(self.population)

        # Safety check: ensure we have final solutions
        if not final_fronts or not final_fronts[0]:
            raise RuntimeError(
                "NSGA-II failed to find any non-dominated solutions. "
                "Try relaxing constraints, increasing population size, or adjusting parameters."
            )

        final_pareto = final_fronts[0]

        print(f"\n✅ NSGA-II 완료: 최종 파레토 프론티어 {len(final_pareto)}개")

        # 4. Individual → 표준 result 형식 변환
        results = self._convert_to_results(
            final_pareto,
            optimization_data,
            cost_calculator,
            baseline_case
        )

        # 5. 결과 저장
        self._save_results(results)

        return results

    def _initialize_population(
        self,
        size: int,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str,
        constraint_preset: str,
        scenario_template: str
    ) -> List[Individual]:
        """초기 집단 생성"""
        population = []

        for i in range(size):
            # 무작위 유전자 생성
            genes = self._generate_random_genes(optimization_data)

            # 적합도 평가 (목적함수 계산)
            objectives, feasible = self._evaluate_individual(
                genes,
                optimization_data,
                cost_calculator,
                baseline_case,
                constraint_preset,
                scenario_template
            )

            individual = Individual(
                genes=genes,
                objectives=objectives,
                feasible=feasible
            )

            population.append(individual)

        return population

    def _generate_random_genes(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """무작위 의사결정 변수 생성"""
        genes = {}

        material_classification = optimization_data['material_classification']

        for material_name, info in material_classification.items():
            material_type = info['type']

            if material_type == 'Formula':
                # Formula 자재: Tier1/Tier2 RE 비율
                # Ensure Tier1 >= Tier2 to satisfy constraint
                tier2_re = random.uniform(0.0, 1.0)
                tier1_re = random.uniform(tier2_re, 1.0)  # Tier1 always >= Tier2
                genes[f"{material_name}_tier1_re"] = tier1_re
                genes[f"{material_name}_tier2_re"] = tier2_re

            elif material_type == 'Ni-Co-Li':
                # Ni/Co/Li 자재: 재활용/저탄소/신재 비율
                element = info.get('element', '')
                recycle = random.uniform(0.0, 0.5)
                low_carbon = random.uniform(0.0, 0.3)
                virgin = 1.0 - recycle - low_carbon

                if virgin < 0:
                    # 조정
                    recycle = random.uniform(0.0, 0.4)
                    low_carbon = random.uniform(0.0, 1.0 - recycle)
                    virgin = 1.0 - recycle - low_carbon

                genes[f"{material_name}_recycle"] = recycle
                genes[f"{material_name}_low_carbon"] = low_carbon
                genes[f"{material_name}_virgin"] = virgin

        return genes

    def _evaluate_individual(
        self,
        genes: Dict[str, Any],
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str,
        constraint_preset: str,
        scenario_template: str
    ) -> Tuple[Tuple[float, float], bool]:
        """
        개체의 목적함수 계산

        Returns:
            ((탄소, 비용), 실현가능성)
        """
        # 실제로는 OptimizationEngine을 사용하여 genes에서 탄소/비용을 계산해야 함
        # 여기서는 간소화된 근사 계산을 사용
        try:
            carbon = self._calculate_carbon_from_genes(genes, optimization_data)
            cost = self._calculate_cost_from_genes(genes, optimization_data, cost_calculator, baseline_case)

            # 제약조건 체크 (간소화)
            feasible = self._check_feasibility(genes, constraint_preset)

            # Check for invalid values
            if carbon <= 0 or cost <= 0 or not np.isfinite(carbon) or not np.isfinite(cost):
                return (1e10, 1e10), False

            return (carbon, cost), feasible

        except Exception as e:
            # 평가 실패 시 큰 페널티
            print(f"      ⚠️ 평가 실패: {str(e)}")
            return (1e10, 1e10), False

    def _calculate_carbon_from_genes(self, genes: Dict, optimization_data: Dict) -> float:
        """유전자로부터 탄소 배출량 계산"""
        total_carbon = 0.0
        scenario_df = optimization_data['scenario_df']
        material_classification = optimization_data['material_classification']

        for material_name, info in material_classification.items():
            material_df = scenario_df[scenario_df['자재명'] == material_name]

            if material_df.empty:
                continue

            base_emission = material_df['배출계수'].iloc[0]
            quantity = material_df['제품총소요량(kg)'].iloc[0]
            material_type = info['type']

            # Calculate adjusted emission based on genes
            adjusted_emission = base_emission

            if material_type == 'Formula':
                # Formula materials: RE100 reduces energy-related emissions
                tier1_re = genes.get(f"{material_name}_tier1_re", 0)
                tier2_re = genes.get(f"{material_name}_tier2_re", 0)

                # Energy ratio from material classification
                tier1_energy_ratio = info.get('tier1_energy_ratio', 0.5)
                tier2_energy_ratio = info.get('tier2_energy_ratio', 0.5)

                # RE100 reduces energy-related emissions
                # Assume energy accounts for 50% of emissions (typical for manufacturing)
                energy_emission_ratio = 0.5

                # Weighted average RE100 application
                avg_re = (tier1_re * tier1_energy_ratio + tier2_re * tier2_energy_ratio)

                # RE100 reduces energy-related emissions
                emission_reduction = base_emission * energy_emission_ratio * avg_re
                adjusted_emission = base_emission - emission_reduction

            elif material_type == 'Ni-Co-Li':
                # Ni/Co/Li materials: recycling reduces emissions
                recycle_ratio = genes.get(f"{material_name}_recycle", 0)
                low_carbon_ratio = genes.get(f"{material_name}_low_carbon", 0)

                # Recycling typically reduces emissions by 60-80%
                recycle_reduction_factor = 0.7
                # Low-carbon materials reduce emissions by 30-50%
                low_carbon_reduction_factor = 0.4

                emission_reduction = (
                    base_emission * recycle_ratio * recycle_reduction_factor +
                    base_emission * low_carbon_ratio * low_carbon_reduction_factor
                )
                adjusted_emission = base_emission - emission_reduction

            # Calculate total carbon for this material
            material_carbon = adjusted_emission * quantity
            total_carbon += material_carbon

        return total_carbon

    def _calculate_cost_from_genes(
        self, genes: Dict, optimization_data: Dict, cost_calculator, baseline_case: str
    ) -> float:
        """
        유전자로부터 비용을 정확하게 계산 (TotalCostCalculator 로직 기반).
        Pyomo 모델 없이 분석적 근사로 계산 (속도 최적화).
        """
        scenario_df = optimization_data['scenario_df']
        material_classification = optimization_data['material_classification']

        total_premium = 0.0

        # 1. RE100 Premium
        for material_name, info in material_classification.items():
            if info['type'] != 'Formula':
                continue

            material_row = scenario_df[scenario_df['자재명'] == material_name]
            if material_row.empty:
                continue

            quantity = material_row['제품총소요량(kg)'].iloc[0]
            material_category = material_row['자재품목'].iloc[0]

            # Get RE values from genes
            tier1_re = genes.get(f"{material_name}_tier1_re", 0)
            tier2_re = genes.get(f"{material_name}_tier2_re", 0)

            # Calculate RE100 conversion prices
            opt_material = cost_calculator._map_material_category(material_category)
            country = "한국"  # Default

            tier1_conversion = cost_calculator.calculate_re100_conversion_price(
                opt_material, "Tier1", country
            )
            tier2_conversion = cost_calculator.calculate_re100_conversion_price(
                opt_material, "Tier2", country
            )

            re100_premium = quantity * (tier1_conversion * tier1_re + tier2_conversion * tier2_re)
            total_premium += re100_premium

        # 2. Recycling Premium (양극재 + Ni/Co/Li)
        for material_name, info in material_classification.items():
            material_row = scenario_df[scenario_df['자재명'] == material_name]
            if material_row.empty:
                continue

            quantity = material_row['제품총소요량(kg)'].iloc[0]
            material_category = material_row['자재품목'].iloc[0]

            # Get basic cost
            opt_material = cost_calculator._map_material_category(material_category)
            basic_cost = cost_calculator._get_basic_cost(opt_material, "Tier1")

            if info['type'] == 'Ni-Co-Li':
                # Ni/Co/Li 자재: Direct calculation
                recycle_ratio = genes.get(f"{material_name}_recycle", 0)

                # Get recycle premium %
                try:
                    recycle_premium_pct = cost_calculator.get_recycle_premium_pct(material_name) / 100
                except:
                    recycle_premium_pct = 0.1  # Default 10%

                recycling_premium = quantity * basic_cost * recycle_premium_pct * recycle_ratio
                total_premium += recycling_premium

            elif 'Cathode' in material_name or '양극재' in material_name or 'CAM' in material_name:
                # 양극재: Element-level calculation
                for element in ['Ni', 'Co', 'Li']:
                    element_ratio = info.get(f'{element}_ratio', 0)
                    if element_ratio == 0:
                        continue

                    # Get element recycle ratio from genes
                    # Note: genes structure may vary, try different keys
                    element_recycle = 0
                    possible_keys = [
                        f"{material_name}_{element}_recycle",
                        f"{material_name}_recycle"
                    ]
                    for key in possible_keys:
                        if key in genes:
                            element_recycle = genes[key]
                            break

                    if element_recycle > 0:
                        # Get element recycle premium %
                        try:
                            recycle_premium_pct = cost_calculator.get_element_recycle_premium_pct(element) / 100
                        except:
                            recycle_premium_pct = 0.15  # Default 15%

                        element_premium = quantity * element_ratio * basic_cost * recycle_premium_pct * element_recycle
                        total_premium += element_premium

        # 3. Low-Carbon Premium (similar structure to recycling)
        for material_name, info in material_classification.items():
            material_row = scenario_df[scenario_df['자재명'] == material_name]
            if material_row.empty:
                continue

            quantity = material_row['제품총소요량(kg)'].iloc[0]
            material_category = material_row['자재품목'].iloc[0]

            opt_material = cost_calculator._map_material_category(material_category)
            basic_cost = cost_calculator._get_basic_cost(opt_material, "Tier1")

            if info['type'] == 'Ni-Co-Li':
                low_carbon_ratio = genes.get(f"{material_name}_low_carbon", 0)

                try:
                    low_carbon_premium_pct = cost_calculator.get_low_carbon_premium_pct(material_name) / 100
                except:
                    low_carbon_premium_pct = 0.05  # Default 5%

                low_carbon_premium = quantity * basic_cost * low_carbon_premium_pct * low_carbon_ratio
                total_premium += low_carbon_premium

            elif 'Cathode' in material_name or '양극재' in material_name or 'CAM' in material_name:
                for element in ['Ni', 'Co', 'Li']:
                    element_ratio = info.get(f'{element}_ratio', 0)
                    if element_ratio == 0:
                        continue

                    # Get element low-carbon ratio from genes
                    element_low_carbon = 0
                    possible_keys = [
                        f"{material_name}_{element}_low_carbon",
                        f"{material_name}_low_carbon"
                    ]
                    for key in possible_keys:
                        if key in genes:
                            element_low_carbon = genes[key]
                            break

                    if element_low_carbon > 0:
                        try:
                            low_carbon_premium_pct = cost_calculator.get_element_low_carbon_premium_pct(element) / 100
                        except:
                            low_carbon_premium_pct = 0.08  # Default 8%

                        element_premium = quantity * element_ratio * basic_cost * low_carbon_premium_pct * element_low_carbon
                        total_premium += element_premium

        return self.zero_premium_baseline + total_premium

    def _check_feasibility(self, genes: Dict, constraint_preset: str) -> bool:
        """제약조건 만족 여부 체크 (간소화)"""
        # 비율 합이 1인지 체크
        material_names = set()
        for key in genes.keys():
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                material_names.add(parts[0])

        for material_name in material_names:
            recycle_key = f"{material_name}_recycle"
            low_carbon_key = f"{material_name}_low_carbon"
            virgin_key = f"{material_name}_virgin"

            if all(k in genes for k in [recycle_key, low_carbon_key, virgin_key]):
                total = genes[recycle_key] + genes[low_carbon_key] + genes[virgin_key]

                if abs(total - 1.0) > 1e-6:
                    return False

        return True

    def _non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """비지배 정렬"""
        # Handle empty population
        if not population:
            return [[]]

        fronts = [[]]

        domination_count = {}  # 각 개체를 지배하는 개체 수 (key: id(individual))
        dominated_solutions = {}  # 각 개체가 지배하는 개체 리스트 (key: id(individual))

        # 1. 지배 관계 계산
        for p in population:
            p_id = id(p)
            domination_count[p_id] = 0
            dominated_solutions[p_id] = []

            for q in population:
                if p is q:
                    continue

                if p.dominates(q):
                    dominated_solutions[p_id].append(q)
                elif q.dominates(p):
                    domination_count[p_id] += 1

            # Rank 0 (비지배)
            if domination_count[p_id] == 0:
                p.rank = 0
                fronts[0].append(p)

        # 2. 나머지 계층 계산
        i = 0
        while i < len(fronts) and fronts[i]:  # Fixed: check both i < len(fronts) and fronts[i] is not empty
            next_front = []

            for p in fronts[i]:
                p_id = id(p)
                for q in dominated_solutions[p_id]:
                    q_id = id(q)
                    domination_count[q_id] -= 1

                    if domination_count[q_id] == 0:
                        q.rank = i + 1
                        next_front.append(q)

            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _calculate_crowding_distance(self, front: List[Individual]):
        """혼잡도 거리 계산"""
        if len(front) == 0:
            return

        # 초기화
        for ind in front:
            ind.crowding_distance = 0.0

        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return

        # 각 목적함수에 대해
        for obj_idx in range(2):  # 탄소, 비용
            # 목적함수 값으로 정렬
            front.sort(key=lambda ind: ind.objectives[obj_idx])

            # 경계 개체는 무한대
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # 정규화
            obj_min = front[0].objectives[obj_idx]
            obj_max = front[-1].objectives[obj_idx]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # 중간 개체들
            for i in range(1, len(front) - 1):
                distance = (front[i+1].objectives[obj_idx] - front[i-1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance

    def _create_offspring(
        self,
        population: List[Individual],
        size: int,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str,
        constraint_preset: str,
        scenario_template: str
    ) -> List[Individual]:
        """자손 생성 (선택/교차/돌연변이)"""
        offspring = []

        crossover_prob = self.config.get('crossover_prob', 0.9)
        mutation_prob = self.config.get('mutation_prob', 0.1)

        for _ in range(size):
            # 1. 토너먼트 선택
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # 2. 교차
            if random.random() < crossover_prob:
                child_genes = self._crossover(parent1.genes, parent2.genes)
            else:
                child_genes = parent1.genes.copy()

            # 3. 돌연변이
            if random.random() < mutation_prob:
                child_genes = self._mutate(child_genes)

            # 4. Repair genes to maintain Tier1 >= Tier2 constraint
            child_genes = self._repair_tier_constraint(child_genes)

            # 4. 적합도 평가
            objectives, feasible = self._evaluate_individual(
                child_genes,
                optimization_data,
                cost_calculator,
                baseline_case,
                constraint_preset,
                scenario_template
            )

            child = Individual(
                genes=child_genes,
                objectives=objectives,
                feasible=feasible
            )

            offspring.append(child)

        return offspring

    def _tournament_selection(self, population: List[Individual], size: int = 3) -> Individual:
        """토너먼트 선택"""
        tournament = random.sample(population, min(size, len(population)))

        # Rank가 낮은 것 우선, 같으면 혼잡도 거리가 큰 것 우선
        tournament.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        return tournament[0]

    def _crossover(self, genes1: Dict, genes2: Dict) -> Dict:
        """SBX (Simulated Binary Crossover)"""
        child_genes = {}
        eta = self.config.get('crossover', {}).get('eta', 20)

        for key in genes1.keys():
            if random.random() < 0.5:
                # SBX 적용
                parent1_val = genes1[key]
                parent2_val = genes2[key]

                # SBX 공식
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child_val = 0.5 * ((1 + beta) * parent1_val + (1 - beta) * parent2_val)

                # 범위 제한
                child_genes[key] = np.clip(child_val, 0.0, 1.0)
            else:
                # 부모 중 하나 선택
                child_genes[key] = genes1[key] if random.random() < 0.5 else genes2[key]

        return child_genes

    def _mutate(self, genes: Dict) -> Dict:
        """다항식 돌연변이 (Polynomial Mutation)"""
        mutated_genes = genes.copy()
        eta = self.config.get('mutation', {}).get('eta', 20)

        for key in mutated_genes.keys():
            if random.random() < (1.0 / len(mutated_genes)):  # 1/n 확률
                val = mutated_genes[key]

                # 다항식 돌연변이
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                mutated_val = val + delta

                # 범위 제한
                mutated_genes[key] = np.clip(mutated_val, 0.0, 1.0)

        return mutated_genes

    def _repair_tier_constraint(self, genes: Dict) -> Dict:
        """
        Repair genes to maintain Tier1 >= Tier2 constraint

        If Tier1 < Tier2 for any material, swap them or adjust to midpoint.
        """
        repaired_genes = genes.copy()

        # Find all materials with tier1_re and tier2_re
        material_names = set()
        for key in genes.keys():
            if key.endswith('_tier1_re'):
                material_name = key[:-len('_tier1_re')]
                material_names.add(material_name)

        for material_name in material_names:
            tier1_key = f"{material_name}_tier1_re"
            tier2_key = f"{material_name}_tier2_re"

            if tier1_key in repaired_genes and tier2_key in repaired_genes:
                tier1_val = repaired_genes[tier1_key]
                tier2_val = repaired_genes[tier2_key]

                # If constraint violated, repair by setting both to their average
                # This maintains genetic diversity while satisfying constraint
                if tier1_val < tier2_val:
                    avg = (tier1_val + tier2_val) / 2.0
                    repaired_genes[tier1_key] = avg
                    repaired_genes[tier2_key] = avg

        return repaired_genes

    def _environmental_selection(self, population: List[Individual], size: int) -> List[Individual]:
        """환경 선택 (부모 + 자손 → 다음 세대)"""
        # Handle empty population
        if not population:
            print(f"      ⚠️ Warning: Empty population in environmental selection")
            return []

        # 비지배 정렬
        fronts = self._non_dominated_sort(population)

        # Check if we have any fronts with individuals
        if not fronts or not any(fronts):
            print(f"      ⚠️ Warning: No valid fronts in environmental selection")
            # Return best individuals from population based on objectives
            sorted_pop = sorted(population, key=lambda ind: (ind.objectives[0], ind.objectives[1]))
            return sorted_pop[:size]

        selected = []

        for front in fronts:
            if not front:  # Skip empty fronts
                continue

            if len(selected) + len(front) <= size:
                # 프론트 전체 추가
                selected.extend(front)
            else:
                # 혼잡도 거리로 정렬하여 나머지 추가
                self._calculate_crowding_distance(front)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)

                remaining = size - len(selected)
                selected.extend(front[:remaining])
                break

        # If we still don't have enough individuals, fill with remaining from population
        if len(selected) < size and len(population) > len(selected):
            print(f"      ℹ️  Only selected {len(selected)}/{size} individuals, filling remaining")
            remaining_pop = [ind for ind in population if ind not in selected]
            selected.extend(remaining_pop[:size - len(selected)])

        return selected

    def _calculate_hypervolume(self, objectives: List[Tuple[float, float]]) -> float:
        """
        Calculate hypervolume indicator

        Uses 2D slicing method for efficiency.

        Args:
            objectives: List of (carbon, cost) tuples

        Returns:
            Hypervolume value
        """
        if not objectives:
            return 0.0

        # Determine reference point (worst point + 10%)
        max_carbon = max(o[0] for o in objectives) * 1.1
        max_cost = max(o[1] for o in objectives) * 1.1
        ref_point = (max_carbon, max_cost)

        # Sort by first objective (carbon)
        sorted_objs = sorted(objectives, key=lambda x: x[0])

        # Calculate 2D hypervolume
        hypervolume = 0.0
        prev_carbon = 0.0

        for carbon, cost in sorted_objs:
            if carbon > ref_point[0] or cost > ref_point[1]:
                continue

            width = carbon - prev_carbon
            height = ref_point[1] - cost

            if width > 0 and height > 0:
                hypervolume += width * height

            prev_carbon = carbon

        return hypervolume

    def _calculate_diversity(self, objectives: List[Tuple[float, float]]) -> float:
        """
        Calculate diversity metric (spacing)

        Measures uniformity of distribution.

        Args:
            objectives: List of (carbon, cost) tuples

        Returns:
            Spacing value (lower = more uniform)
        """
        if len(objectives) < 2:
            return 0.0

        # Normalize objectives
        carbons = np.array([o[0] for o in objectives])
        costs = np.array([o[1] for o in objectives])

        carbon_range = carbons.max() - carbons.min()
        cost_range = costs.max() - costs.min()

        if carbon_range == 0 or cost_range == 0:
            return 0.0

        normalized = np.column_stack([
            (carbons - carbons.min()) / carbon_range,
            (costs - costs.min()) / cost_range
        ])

        # Calculate distances to nearest neighbors
        n = len(normalized)
        distances = []

        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(normalized[i] - normalized[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)

        # Spacing metric
        mean_dist = np.mean(distances)
        spacing = np.sqrt(np.sum((np.array(distances) - mean_dist) ** 2) / n)

        return spacing

    def _check_convergence(self, generation: int) -> bool:
        """
        수렴 여부 체크 (Hypervolume 기반)

        Uses hypervolume indicator for accurate convergence detection.
        """
        termination = self.config.get('termination', {})

        if not termination.get('early_stopping', True):
            return False

        patience = termination.get('patience', 20)
        threshold = termination.get('convergence_threshold', 0.01)  # 1% improvement

        # Need at least patience generations
        if generation < patience:
            return False

        # Check hypervolume improvement over patience window
        if len(self.hypervolume_history) < patience:
            return False

        recent_hvs = self.hypervolume_history[-patience:]

        # Calculate improvement rate
        if recent_hvs[0] == 0:
            return False  # Avoid division by zero

        improvement = (recent_hvs[-1] - recent_hvs[0]) / recent_hvs[0]

        # Converged if improvement < threshold
        if abs(improvement) < threshold:
            print(f"   ℹ️  Hypervolume improvement: {improvement*100:.2f}% (threshold: {threshold*100:.1f}%)")
            return True

        return False

    def _convert_to_results(
        self,
        pareto_front: List[Individual],
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_case: str
    ) -> List[Dict[str, Any]]:
        """Individual 객체를 표준 result 형식으로 변환"""
        results = []

        for ind in pareto_front:
            # Calculate reduction percentage
            baseline_carbon = self.baseline_carbon if self.baseline_carbon > 0 else ind.objectives[0]
            reduction_pct = ((baseline_carbon - ind.objectives[0]) / baseline_carbon * 100) if baseline_carbon > 0 else 0

            # 간소화된 summary 생성
            summary = {
                'total_carbon': ind.objectives[0],
                'total_cost': ind.objectives[1],
                'total_reduction_pct': reduction_pct
            }

            # Create simplified result_df from genes
            result_df_data = []
            scenario_df = optimization_data['scenario_df']
            material_classification = optimization_data['material_classification']

            for material_name, info in material_classification.items():
                material_type = info['type']

                row = {
                    '자재명': material_name,
                    '자재품목': scenario_df[scenario_df['자재명'] == material_name]['자재품목'].iloc[0] if not scenario_df[scenario_df['자재명'] == material_name].empty else '',
                }

                if material_type == 'Formula':
                    tier1_re = ind.genes.get(f"{material_name}_tier1_re", 0) * 100
                    tier2_re = ind.genes.get(f"{material_name}_tier2_re", 0) * 100
                    row['Tier1_RE(%)'] = f"{tier1_re:.1f}"
                    row['Tier2_RE(%)'] = f"{tier2_re:.1f}"
                elif material_type == 'Ni-Co-Li':
                    recycle = ind.genes.get(f"{material_name}_recycle", 0) * 100
                    low_carbon = ind.genes.get(f"{material_name}_low_carbon", 0) * 100
                    virgin = ind.genes.get(f"{material_name}_virgin", 0) * 100
                    row['재활용(%)'] = f"{recycle:.1f}"
                    row['저탄소(%)'] = f"{low_carbon:.1f}"
                    row['신재(%)'] = f"{virgin:.1f}"

                result_df_data.append(row)

            result_df = pd.DataFrame(result_df_data)

            # 메타데이터 추가
            result = {
                'genes': ind.genes,
                'summary': summary,
                'result_df': result_df,
                'objectives': {
                    'carbon': ind.objectives[0],
                    'cost': ind.objectives[1]
                },
                'rank': ind.rank,
                'crowding_distance': ind.crowding_distance,
                'timestamp': datetime.now().isoformat(),
                'method': 'nsga2',
                'baseline_cost': self.zero_premium_baseline,  # For legacy compatibility
                'zero_premium_baseline': self.zero_premium_baseline,
                'baseline_carbon': baseline_carbon
            }

            results.append(result)

        return results

    def _save_results(self, results: List[Dict[str, Any]]):
        """결과 저장"""
        # Use base class's standardized save_results method
        self.save_results('nsga2', results)

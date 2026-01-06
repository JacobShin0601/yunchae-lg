"""
Supplier Selector - Multi-Criteria Decision Analysis (MCDA)

다기준 의사결정 분석을 통한 최적 공급업체 선택 시스템입니다.
여러 평가 기준(배출량, 비용, 품질, 리드타임, 신뢰성)을 고려하여 공급업체를 추천합니다.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from .supplier_database import Supplier, SupplierDatabase


@dataclass
class SelectionCriteria:
    """
    공급업체 선택 기준

    Attributes:
        carbon_weight: 탄소배출 가중치 (0~1)
        cost_weight: 비용 가중치 (0~1)
        quality_weight: 품질 가중치 (0~1)
        lead_time_weight: 리드타임 가중치 (0~1)
        reliability_weight: 신뢰성 가중치 (0~1)

    Note: 모든 가중치의 합은 1.0이어야 합니다.
    """
    carbon_weight: float = 0.30
    cost_weight: float = 0.25
    quality_weight: float = 0.20
    lead_time_weight: float = 0.15
    reliability_weight: float = 0.10

    def validate(self) -> bool:
        """가중치 합이 1.0인지 검증"""
        total = (
            self.carbon_weight +
            self.cost_weight +
            self.quality_weight +
            self.lead_time_weight +
            self.reliability_weight
        )
        return abs(total - 1.0) < 0.001

    def normalize(self) -> 'SelectionCriteria':
        """가중치를 정규화하여 합이 1.0이 되도록 조정"""
        total = (
            self.carbon_weight +
            self.cost_weight +
            self.quality_weight +
            self.lead_time_weight +
            self.reliability_weight
        )

        if total == 0:
            # 모두 0이면 균등 배분
            return SelectionCriteria(0.2, 0.2, 0.2, 0.2, 0.2)

        return SelectionCriteria(
            carbon_weight=self.carbon_weight / total,
            cost_weight=self.cost_weight / total,
            quality_weight=self.quality_weight / total,
            lead_time_weight=self.lead_time_weight / total,
            reliability_weight=self.reliability_weight / total
        )


class SupplierSelector:
    """
    공급업체 선택 시스템 (MCDA 기반)

    Methods:
    - rank_suppliers: 다기준 점수로 공급업체 순위 매기기
    - find_pareto_optimal: 파레토 최적 공급업체 찾기
    - recommend_supplier: 최적 공급업체 추천
    - calculate_total_score: 총점 계산
    """

    def __init__(self, database: SupplierDatabase):
        """
        초기화

        Args:
            database: 공급업체 데이터베이스
        """
        self.database = database

    def calculate_total_score(
        self,
        supplier: Supplier,
        criteria: SelectionCriteria
    ) -> float:
        """
        공급업체의 총 점수 계산 (0~100)

        Args:
            supplier: 공급업체
            criteria: 선택 기준

        Returns:
            총 점수 (0~100)
        """
        # 각 기준별 점수 (0~100)
        carbon_score = supplier.get_score('carbon')
        cost_score = supplier.get_score('cost')
        quality_score = supplier.get_score('quality')
        lead_time_score = supplier.get_score('lead_time')
        reliability_score = supplier.get_score('reliability')

        # 가중 평균
        total_score = (
            carbon_score * criteria.carbon_weight +
            cost_score * criteria.cost_weight +
            quality_score * criteria.quality_weight +
            lead_time_score * criteria.lead_time_weight +
            reliability_score * criteria.reliability_weight
        )

        return total_score

    def rank_suppliers(
        self,
        suppliers: List[Supplier],
        criteria: SelectionCriteria
    ) -> List[Tuple[Supplier, float, Dict[str, float]]]:
        """
        공급업체 순위 매기기

        Args:
            suppliers: 공급업체 리스트
            criteria: 선택 기준

        Returns:
            [(supplier, total_score, score_breakdown), ...] 형식의 리스트
            (높은 점수 순으로 정렬)
        """
        # 가중치 정규화
        criteria = criteria.normalize()

        results = []
        for supplier in suppliers:
            total_score = self.calculate_total_score(supplier, criteria)

            # 세부 점수
            score_breakdown = {
                'carbon': supplier.get_score('carbon'),
                'cost': supplier.get_score('cost'),
                'quality': supplier.get_score('quality'),
                'lead_time': supplier.get_score('lead_time'),
                'reliability': supplier.get_score('reliability')
            }

            results.append((supplier, total_score, score_breakdown))

        # 총점 기준 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def find_pareto_optimal(
        self,
        suppliers: List[Supplier],
        objectives: List[str] = None
    ) -> List[Supplier]:
        """
        파레토 최적 공급업체 찾기

        두 목적함수 간에 트레이드오프가 있을 때, 어느 목적함수도
        희생하지 않고 개선할 수 없는 솔루션들을 찾습니다.

        Args:
            suppliers: 공급업체 리스트
            objectives: 목적함수 리스트 (기본: ['carbon', 'cost'])

        Returns:
            파레토 최적 공급업체 리스트
        """
        if objectives is None:
            objectives = ['carbon', 'cost']  # 기본: 탄소 vs 비용

        if not suppliers:
            return []

        # 점수 행렬 생성 (높을수록 좋음)
        scores = np.array([
            [supplier.get_score(obj) for obj in objectives]
            for supplier in suppliers
        ])

        # 파레토 최적 찾기
        pareto_indices = []
        for i, score_i in enumerate(scores):
            is_dominated = False

            for j, score_j in enumerate(scores):
                if i == j:
                    continue

                # score_j가 score_i를 지배하는지 확인
                # (모든 목적함수에서 >= 이고, 적어도 하나는 >)
                if all(score_j >= score_i) and any(score_j > score_i):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_indices.append(i)

        pareto_suppliers = [suppliers[i] for i in pareto_indices]

        return pareto_suppliers

    def recommend_supplier(
        self,
        material_type: str,
        criteria: SelectionCriteria,
        region: Optional[str] = None,
        re100_required: bool = False,
        top_n: int = 3
    ) -> List[Tuple[Supplier, float, str]]:
        """
        최적 공급업체 추천

        Args:
            material_type: 자재 타입
            criteria: 선택 기준
            region: 지역 필터 (선택사항)
            re100_required: RE100 지원 필수 여부
            top_n: 상위 N개 추천

        Returns:
            [(supplier, score, recommendation_reason), ...] 리스트
        """
        # 1. 공급업체 검색
        suppliers = self.database.search_suppliers(
            material_type=material_type,
            region=region,
            re100_required=re100_required
        )

        if not suppliers:
            print(f"❌ {material_type}에 대한 공급업체를 찾을 수 없습니다.")
            return []

        print(f"\n🔍 {material_type} 공급업체 검색: {len(suppliers)}개 발견")

        # 2. 순위 매기기
        ranked = self.rank_suppliers(suppliers, criteria)

        # 3. 상위 N개 선택
        top_suppliers = ranked[:top_n]

        # 4. 추천 이유 생성
        recommendations = []
        for supplier, score, breakdown in top_suppliers:
            # 가장 높은 점수 기준 찾기
            max_criterion = max(breakdown.items(), key=lambda x: x[1])
            reason = f"최고 {max_criterion[0]} 점수 ({max_criterion[1]:.1f})"

            recommendations.append((supplier, score, reason))

        return recommendations

    def compare_suppliers_detailed(
        self,
        supplier_ids: List[str],
        criteria: SelectionCriteria
    ) -> Dict[str, any]:
        """
        공급업체 상세 비교

        Args:
            supplier_ids: 비교할 공급업체 ID 리스트
            criteria: 선택 기준

        Returns:
            상세 비교 결과 딕셔너리
        """
        suppliers = [self.database.get_supplier(sid) for sid in supplier_ids]
        suppliers = [s for s in suppliers if s is not None]

        if not suppliers:
            return {}

        # 가중치 정규화
        criteria = criteria.normalize()

        # 각 공급업체 평가
        results = {
            'suppliers': [],
            'scores': {
                'total': [],
                'carbon': [],
                'cost': [],
                'quality': [],
                'lead_time': [],
                'reliability': []
            },
            'weights': {
                'carbon': criteria.carbon_weight,
                'cost': criteria.cost_weight,
                'quality': criteria.quality_weight,
                'lead_time': criteria.lead_time_weight,
                'reliability': criteria.reliability_weight
            },
            'raw_data': {
                'emission_factor': [],
                'cost_per_kg': [],
                'quality_score': [],
                'lead_time_days': [],
                'reliability_score': []
            }
        }

        for supplier in suppliers:
            results['suppliers'].append(supplier.name)

            # 총점
            total_score = self.calculate_total_score(supplier, criteria)
            results['scores']['total'].append(total_score)

            # 세부 점수
            results['scores']['carbon'].append(supplier.get_score('carbon'))
            results['scores']['cost'].append(supplier.get_score('cost'))
            results['scores']['quality'].append(supplier.get_score('quality'))
            results['scores']['lead_time'].append(supplier.get_score('lead_time'))
            results['scores']['reliability'].append(supplier.get_score('reliability'))

            # Raw 데이터
            results['raw_data']['emission_factor'].append(supplier.emission_factor)
            results['raw_data']['cost_per_kg'].append(supplier.cost_per_kg)
            results['raw_data']['quality_score'].append(supplier.quality_score)
            results['raw_data']['lead_time_days'].append(supplier.lead_time_days)
            results['raw_data']['reliability_score'].append(supplier.reliability_score)

        return results

    def sensitivity_analysis(
        self,
        suppliers: List[Supplier],
        base_criteria: SelectionCriteria,
        vary_criterion: str,
        vary_range: Tuple[float, float] = (0.0, 1.0),
        steps: int = 11
    ) -> Dict[str, any]:
        """
        민감도 분석: 특정 기준의 가중치 변화에 따른 순위 변화

        Args:
            suppliers: 공급업체 리스트
            base_criteria: 기본 선택 기준
            vary_criterion: 변화시킬 기준 (carbon, cost, quality, lead_time, reliability)
            vary_range: 가중치 변화 범위 (min, max)
            steps: 변화 단계 수

        Returns:
            민감도 분석 결과
        """
        weights = np.linspace(vary_range[0], vary_range[1], steps)
        results = {
            'weights': weights.tolist(),
            'rankings': [],  # [step][rank] = supplier_id
            'scores': []     # [step][supplier] = score
        }

        for weight in weights:
            # 기준 복사 및 수정
            criteria = SelectionCriteria(
                carbon_weight=base_criteria.carbon_weight,
                cost_weight=base_criteria.cost_weight,
                quality_weight=base_criteria.quality_weight,
                lead_time_weight=base_criteria.lead_time_weight,
                reliability_weight=base_criteria.reliability_weight
            )

            # 특정 기준의 가중치 변경
            if vary_criterion == 'carbon':
                criteria.carbon_weight = weight
            elif vary_criterion == 'cost':
                criteria.cost_weight = weight
            elif vary_criterion == 'quality':
                criteria.quality_weight = weight
            elif vary_criterion == 'lead_time':
                criteria.lead_time_weight = weight
            elif vary_criterion == 'reliability':
                criteria.reliability_weight = weight

            # 정규화
            criteria = criteria.normalize()

            # 순위 매기기
            ranked = self.rank_suppliers(suppliers, criteria)

            # 결과 저장
            step_ranking = [supplier.id for supplier, _, _ in ranked]
            step_scores = {supplier.id: score for supplier, score, _ in ranked}

            results['rankings'].append(step_ranking)
            results['scores'].append(step_scores)

        return results

    def get_recommendation_summary(
        self,
        material_type: str,
        criteria: SelectionCriteria,
        top_n: int = 3
    ) -> str:
        """
        추천 요약 문자열 생성

        Args:
            material_type: 자재 타입
            criteria: 선택 기준
            top_n: 상위 N개

        Returns:
            추천 요약 문자열
        """
        recommendations = self.recommend_supplier(
            material_type=material_type,
            criteria=criteria,
            top_n=top_n
        )

        if not recommendations:
            return f"❌ {material_type}에 대한 추천 공급업체가 없습니다."

        summary = f"🏆 {material_type} 추천 공급업체 Top {len(recommendations)}:\n\n"

        for rank, (supplier, score, reason) in enumerate(recommendations, 1):
            summary += f"{rank}. {supplier.name} (총점: {score:.1f}/100)\n"
            summary += f"   • 지역: {supplier.region}\n"
            summary += f"   • 배출계수: {supplier.emission_factor:.2f} kgCO2eq/kg\n"
            summary += f"   • 단가: ${supplier.cost_per_kg:.2f}/kg\n"
            summary += f"   • 품질: {supplier.quality_score:.0f}/100\n"
            summary += f"   • 리드타임: {supplier.lead_time_days}일\n"
            summary += f"   • 추천 이유: {reason}\n\n"

        return summary

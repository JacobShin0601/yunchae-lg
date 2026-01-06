"""
Supplier Database

공급업체 데이터베이스 및 관리 시스템입니다.
공급업체별 속성(지역, 비용, 배출계수, 품질, 리드타임 등)을 관리하고 조회할 수 있습니다.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os


@dataclass
class Supplier:
    """
    공급업체 정보 데이터 클래스

    Attributes:
        id: 공급업체 고유 ID
        name: 공급업체명
        region: 지역 (Korea, China, USA, Europe, Japan, etc.)
        material_type: 공급 자재 타입 (Cathode, Anode, Electrolyte, Separator, etc.)

        # Performance metrics
        emission_factor: 배출계수 (kgCO2eq/kg)
        cost_per_kg: 단가 ($/kg)
        quality_score: 품질 점수 (0~100)
        lead_time_days: 리드타임 (일)
        reliability_score: 신뢰성 점수 (0~100)

        # RE100 capabilities
        re100_tier1_available: Tier1 RE100 지원 여부
        re100_tier2_available: Tier2 RE100 지원 여부
        re100_tier1_premium: Tier1 RE100 프리미엄 ($/kWh)
        re100_tier2_premium: Tier2 RE100 프리미엄 ($/kWh)

        # Additional info
        certifications: 인증 목록 (ISO14001, ISO9001 등)
        capacity_kg_per_year: 연간 생산 용량 (kg/year)
        minimum_order_kg: 최소 주문량 (kg)
        notes: 기타 메모
    """
    id: str
    name: str
    region: str
    material_type: str

    # Performance metrics
    emission_factor: float  # kgCO2eq/kg
    cost_per_kg: float  # $/kg
    quality_score: float = 80.0  # 0~100
    lead_time_days: int = 30  # days
    reliability_score: float = 85.0  # 0~100

    # RE100 capabilities
    re100_tier1_available: bool = False
    re100_tier2_available: bool = False
    re100_tier1_premium: float = 0.15  # $/kWh
    re100_tier2_premium: float = 0.12  # $/kWh

    # Additional info
    certifications: List[str] = field(default_factory=list)
    capacity_kg_per_year: float = 1_000_000.0  # kg/year
    minimum_order_kg: float = 100.0  # kg
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """공급업체 정보를 딕셔너리로 변환"""
        return {
            'id': self.id,
            'name': self.name,
            'region': self.region,
            'material_type': self.material_type,
            'emission_factor': self.emission_factor,
            'cost_per_kg': self.cost_per_kg,
            'quality_score': self.quality_score,
            'lead_time_days': self.lead_time_days,
            'reliability_score': self.reliability_score,
            're100_tier1_available': self.re100_tier1_available,
            're100_tier2_available': self.re100_tier2_available,
            're100_tier1_premium': self.re100_tier1_premium,
            're100_tier2_premium': self.re100_tier2_premium,
            'certifications': self.certifications,
            'capacity_kg_per_year': self.capacity_kg_per_year,
            'minimum_order_kg': self.minimum_order_kg,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Supplier':
        """딕셔너리에서 공급업체 정보 생성"""
        return cls(**data)

    def get_score(self, criteria: str) -> float:
        """
        특정 기준에 대한 점수 반환

        Args:
            criteria: 평가 기준 (carbon, cost, quality, lead_time, reliability)

        Returns:
            정규화된 점수 (0~100)
        """
        if criteria == 'carbon':
            # 배출계수가 낮을수록 좋음 (역수 사용)
            # 가정: 최대 배출계수 20 kgCO2eq/kg
            return max(0, 100 * (1 - self.emission_factor / 20.0))
        elif criteria == 'cost':
            # 비용이 낮을수록 좋음
            # 가정: 최대 비용 $100/kg
            return max(0, 100 * (1 - self.cost_per_kg / 100.0))
        elif criteria == 'quality':
            return self.quality_score
        elif criteria == 'lead_time':
            # 리드타임이 짧을수록 좋음
            # 가정: 최대 리드타임 180일
            return max(0, 100 * (1 - self.lead_time_days / 180.0))
        elif criteria == 'reliability':
            return self.reliability_score
        else:
            return 0.0


class SupplierDatabase:
    """
    공급업체 데이터베이스 관리 클래스

    공급업체 등록, 조회, 검색, 비교 기능을 제공합니다.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        데이터베이스 초기화

        Args:
            db_path: 데이터베이스 JSON 파일 경로 (None이면 메모리만 사용)
        """
        self.suppliers: Dict[str, Supplier] = {}  # {supplier_id: Supplier}
        self.db_path = db_path

        if db_path and os.path.exists(db_path):
            self.load_from_file(db_path)

    def add_supplier(self, supplier: Supplier) -> None:
        """
        공급업체 추가

        Args:
            supplier: Supplier 객체
        """
        if supplier.id in self.suppliers:
            print(f"⚠️  공급업체 '{supplier.id}'가 이미 존재합니다. 덮어씁니다.")

        self.suppliers[supplier.id] = supplier
        print(f"✅ 공급업체 추가: {supplier.name} (ID: {supplier.id})")

    def get_supplier(self, supplier_id: str) -> Optional[Supplier]:
        """
        공급업체 조회

        Args:
            supplier_id: 공급업체 ID

        Returns:
            Supplier 객체 또는 None
        """
        return self.suppliers.get(supplier_id)

    def remove_supplier(self, supplier_id: str) -> bool:
        """
        공급업체 제거

        Args:
            supplier_id: 공급업체 ID

        Returns:
            제거 성공 여부
        """
        if supplier_id in self.suppliers:
            del self.suppliers[supplier_id]
            print(f"✅ 공급업체 제거: {supplier_id}")
            return True
        else:
            print(f"❌ 공급업체를 찾을 수 없습니다: {supplier_id}")
            return False

    def search_suppliers(
        self,
        material_type: Optional[str] = None,
        region: Optional[str] = None,
        max_emission: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_quality: Optional[float] = None,
        max_lead_time: Optional[int] = None,
        re100_required: bool = False
    ) -> List[Supplier]:
        """
        공급업체 검색

        Args:
            material_type: 자재 타입 필터
            region: 지역 필터
            max_emission: 최대 배출계수 (kgCO2eq/kg)
            max_cost: 최대 단가 ($/kg)
            min_quality: 최소 품질 점수
            max_lead_time: 최대 리드타임 (일)
            re100_required: RE100 지원 필수 여부

        Returns:
            조건을 만족하는 공급업체 리스트
        """
        results = []

        for supplier in self.suppliers.values():
            # 자재 타입 필터
            if material_type and supplier.material_type != material_type:
                continue

            # 지역 필터
            if region and supplier.region != region:
                continue

            # 배출계수 필터
            if max_emission and supplier.emission_factor > max_emission:
                continue

            # 비용 필터
            if max_cost and supplier.cost_per_kg > max_cost:
                continue

            # 품질 필터
            if min_quality and supplier.quality_score < min_quality:
                continue

            # 리드타임 필터
            if max_lead_time and supplier.lead_time_days > max_lead_time:
                continue

            # RE100 필터
            if re100_required and not (supplier.re100_tier1_available or supplier.re100_tier2_available):
                continue

            results.append(supplier)

        return results

    def get_suppliers_by_material(self, material_type: str) -> List[Supplier]:
        """
        특정 자재 타입의 모든 공급업체 조회

        Args:
            material_type: 자재 타입

        Returns:
            공급업체 리스트
        """
        return [s for s in self.suppliers.values() if s.material_type == material_type]

    def get_suppliers_by_region(self, region: str) -> List[Supplier]:
        """
        특정 지역의 모든 공급업체 조회

        Args:
            region: 지역명

        Returns:
            공급업체 리스트
        """
        return [s for s in self.suppliers.values() if s.region == region]

    def get_best_suppliers(
        self,
        material_type: str,
        criteria: str = 'carbon',
        top_n: int = 5
    ) -> List[Supplier]:
        """
        특정 기준으로 상위 N개 공급업체 조회

        Args:
            material_type: 자재 타입
            criteria: 평가 기준 (carbon, cost, quality, lead_time, reliability)
            top_n: 상위 개수

        Returns:
            상위 N개 공급업체 리스트
        """
        suppliers = self.get_suppliers_by_material(material_type)

        if not suppliers:
            return []

        # 기준에 따라 정렬
        sorted_suppliers = sorted(
            suppliers,
            key=lambda s: s.get_score(criteria),
            reverse=True  # 높은 점수가 좋음
        )

        return sorted_suppliers[:top_n]

    def compare_suppliers(self, supplier_ids: List[str]) -> Dict[str, List[Any]]:
        """
        여러 공급업체 비교

        Args:
            supplier_ids: 비교할 공급업체 ID 리스트

        Returns:
            비교 테이블 딕셔너리
        """
        suppliers = [self.get_supplier(sid) for sid in supplier_ids]
        suppliers = [s for s in suppliers if s is not None]

        if not suppliers:
            return {}

        comparison = {
            'name': [s.name for s in suppliers],
            'region': [s.region for s in suppliers],
            'emission_factor': [s.emission_factor for s in suppliers],
            'cost_per_kg': [s.cost_per_kg for s in suppliers],
            'quality_score': [s.quality_score for s in suppliers],
            'lead_time_days': [s.lead_time_days for s in suppliers],
            'reliability_score': [s.reliability_score for s in suppliers],
            're100_available': [
                s.re100_tier1_available or s.re100_tier2_available
                for s in suppliers
            ]
        }

        return comparison

    def save_to_file(self, file_path: str) -> None:
        """
        데이터베이스를 JSON 파일로 저장

        Args:
            file_path: 저장할 파일 경로
        """
        data = {
            'suppliers': [s.to_dict() for s in self.suppliers.values()]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 데이터베이스 저장: {file_path} ({len(self.suppliers)}개 공급업체)")

    def load_from_file(self, file_path: str) -> None:
        """
        JSON 파일에서 데이터베이스 로드

        Args:
            file_path: 로드할 파일 경로
        """
        if not os.path.exists(file_path):
            print(f"⚠️  파일이 없습니다: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.suppliers.clear()
        for supplier_data in data.get('suppliers', []):
            supplier = Supplier.from_dict(supplier_data)
            self.suppliers[supplier.id] = supplier

        print(f"✅ 데이터베이스 로드: {file_path} ({len(self.suppliers)}개 공급업체)")

    def get_summary(self) -> str:
        """
        데이터베이스 요약 정보

        Returns:
            요약 문자열
        """
        if not self.suppliers:
            return "📦 공급업체 데이터베이스: 0개"

        material_types = set(s.material_type for s in self.suppliers.values())
        regions = set(s.region for s in self.suppliers.values())

        summary = f"📦 공급업체 데이터베이스: {len(self.suppliers)}개\n"
        summary += f"   • 자재 타입: {len(material_types)}개 ({', '.join(sorted(material_types))})\n"
        summary += f"   • 지역: {len(regions)}개 ({', '.join(sorted(regions))})"

        return summary

    def create_sample_database(self) -> None:
        """
        샘플 공급업체 데이터 생성 (테스트용)
        """
        sample_suppliers = [
            # Cathode suppliers
            Supplier(
                id="cathode_kr_001", name="한국 양극재 A사", region="Korea",
                material_type="Cathode", emission_factor=8.5, cost_per_kg=45.0,
                quality_score=90, lead_time_days=21, reliability_score=92,
                re100_tier1_available=True, re100_tier2_available=True,
                re100_tier1_premium=0.15, re100_tier2_premium=0.12,
                certifications=["ISO14001", "ISO9001"],
                capacity_kg_per_year=500_000
            ),
            Supplier(
                id="cathode_cn_001", name="중국 양극재 B사", region="China",
                material_type="Cathode", emission_factor=12.0, cost_per_kg=38.0,
                quality_score=75, lead_time_days=28, reliability_score=78,
                re100_tier1_available=False, re100_tier2_available=True,
                re100_tier2_premium=0.18,
                certifications=["ISO9001"],
                capacity_kg_per_year=1_000_000
            ),
            Supplier(
                id="cathode_jp_001", name="일본 양극재 C사", region="Japan",
                material_type="Cathode", emission_factor=7.2, cost_per_kg=52.0,
                quality_score=95, lead_time_days=14, reliability_score=95,
                re100_tier1_available=True, re100_tier2_available=True,
                re100_tier1_premium=0.18, re100_tier2_premium=0.15,
                certifications=["ISO14001", "ISO9001", "IATF16949"],
                capacity_kg_per_year=300_000
            ),

            # Electrolyte suppliers
            Supplier(
                id="electrolyte_kr_001", name="한국 전해액 A사", region="Korea",
                material_type="Electrolyte", emission_factor=3.2, cost_per_kg=28.0,
                quality_score=88, lead_time_days=18, reliability_score=90,
                re100_tier1_available=True, re100_tier2_available=False,
                re100_tier1_premium=0.15,
                certifications=["ISO14001", "ISO9001"],
                capacity_kg_per_year=800_000
            ),
            Supplier(
                id="electrolyte_cn_001", name="중국 전해액 B사", region="China",
                material_type="Electrolyte", emission_factor=4.5, cost_per_kg=22.0,
                quality_score=70, lead_time_days=25, reliability_score=75,
                re100_tier1_available=False, re100_tier2_available=False,
                certifications=[],
                capacity_kg_per_year=1_500_000
            ),

            # Separator suppliers
            Supplier(
                id="separator_kr_001", name="한국 분리막 A사", region="Korea",
                material_type="Separator", emission_factor=2.7, cost_per_kg=15.0,
                quality_score=92, lead_time_days=20, reliability_score=93,
                re100_tier1_available=True, re100_tier2_available=True,
                re100_tier1_premium=0.15, re100_tier2_premium=0.12,
                certifications=["ISO14001", "ISO9001"],
                capacity_kg_per_year=600_000
            ),
            Supplier(
                id="separator_usa_001", name="미국 분리막 B사", region="USA",
                material_type="Separator", emission_factor=2.3, cost_per_kg=18.0,
                quality_score=90, lead_time_days=35, reliability_score=88,
                re100_tier1_available=True, re100_tier2_available=False,
                re100_tier1_premium=0.10,
                certifications=["ISO14001", "ISO9001"],
                capacity_kg_per_year=400_000
            ),
        ]

        for supplier in sample_suppliers:
            self.add_supplier(supplier)

        print(f"\n✅ 샘플 데이터베이스 생성 완료: {len(sample_suppliers)}개 공급업체")

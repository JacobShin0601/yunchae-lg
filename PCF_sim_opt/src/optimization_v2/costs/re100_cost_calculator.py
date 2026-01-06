"""
RE100 비용 계산 모듈 (optimization_v2)

이 모듈은 재생에너지(RE100) 적용 시 발생하는 비용 증가를 계산합니다.
지역별, Tier별 RE100 가격 프리미엄을 고려하여 상세한 비용 분석을 제공합니다.
"""

from typing import Dict, Any, Optional
import pandas as pd
import pyomo.environ as pyo


class RE100CostCalculator:
    """
    RE100 비용 계산 클래스

    주요 기능:
    1. 지역별 RE100 가격 프리미엄 관리
    2. 에너지 소비량 기반 비용 증가 계산
    3. Tier1/Tier2 구분 비용 계산
    4. Pyomo 최적화 모델과의 통합

    비용 계산 공식:
    RE100_cost = energy_consumption × RE100_rate × regional_premium

    where:
    - energy_consumption: 자재 생산 시 에너지 소비량 (kWh/kg)
    - RE100_rate: RE100 적용 비율 (0~1)
    - regional_premium: 지역별 RE100 가격 프리미엄 ($/kWh)
    """

    def __init__(
        self,
        regional_prices: Optional[Dict[str, Dict[str, float]]] = None,
        debug_mode: bool = False
    ):
        """
        RE100CostCalculator 초기화

        Args:
            regional_prices: 지역별 RE100 가격 프리미엄
                예시: {
                    "Korea": {"tier1": 0.15, "tier2": 0.12},
                    "China": {"tier1": 0.20, "tier2": 0.18}
                }
                단위: $/kWh (renewable premium over conventional electricity)
            debug_mode: 디버그 모드 (상세 로그 출력)
        """
        self.debug_mode = debug_mode

        # 기본 지역별 RE100 가격 프리미엄 ($/kWh)
        # Tier1: 자재 제조 단계 에너지
        # Tier2: 전구체/원료 제조 단계 에너지
        self.regional_prices = regional_prices or {
            "Korea": {
                "tier1": 0.15,  # 한국 전력 RE100 프리미엄 (예: 재생에너지 REC 가격)
                "tier2": 0.12   # Tier2는 다소 낮은 프리미엄
            },
            "China": {
                "tier1": 0.20,  # 중국은 재생에너지 인프라가 덜 발달하여 높은 프리미엄
                "tier2": 0.18
            },
            "USA": {
                "tier1": 0.10,  # 미국은 재생에너지 시장이 발달하여 낮은 프리미엄
                "tier2": 0.08
            },
            "Europe": {
                "tier1": 0.12,  # 유럽은 중간 수준
                "tier2": 0.10
            },
            "Japan": {
                "tier1": 0.18,
                "tier2": 0.15
            },
            "default": {
                "tier1": 0.15,  # 기타 지역 기본값
                "tier2": 0.12
            }
        }

        # 지역명 정규화 매핑 (다양한 표기를 표준 지역명으로 변환)
        self.region_mapping = {
            "한국": "Korea",
            "Korea": "Korea",
            "KR": "Korea",
            "중국": "China",
            "China": "China",
            "CN": "China",
            "미국": "USA",
            "USA": "USA",
            "US": "USA",
            "유럽": "Europe",
            "Europe": "Europe",
            "EU": "Europe",
            "일본": "Japan",
            "Japan": "Japan",
            "JP": "Japan"
        }

        if self.debug_mode:
            self._print_debug_info()

    def _print_debug_info(self) -> None:
        """디버깅 정보 출력"""
        print("\n" + "="*60)
        print("💰 RE100CostCalculator 초기화")
        print("="*60)
        print(f"지역별 RE100 프리미엄 ($/kWh):")
        for region, prices in self.regional_prices.items():
            print(f"  • {region:10s}: Tier1=${prices['tier1']:.3f}/kWh, Tier2=${prices['tier2']:.3f}/kWh")
        print("="*60)

    def get_regional_premium(
        self,
        region: str,
        tier: str = "tier1"
    ) -> float:
        """
        지역 및 Tier별 RE100 가격 프리미엄 조회

        Args:
            region: 지역명 (예: "Korea", "한국", "China")
            tier: Tier 유형 ("tier1" or "tier2")

        Returns:
            RE100 가격 프리미엄 ($/kWh)
        """
        # 지역명 정규화
        normalized_region = self.region_mapping.get(region, region)

        # 지역별 프리미엄 조회 (없으면 default 사용)
        region_prices = self.regional_prices.get(
            normalized_region,
            self.regional_prices["default"]
        )

        # Tier별 가격 반환 (tier1 기본값)
        return region_prices.get(tier.lower(), region_prices.get("tier1", 0.15))

    def calculate_material_re100_cost(
        self,
        energy_consumption_tier1: float,
        energy_consumption_tier2: float,
        re100_rate_tier1: float,
        re100_rate_tier2: float,
        region: str,
        quantity: float
    ) -> float:
        """
        자재별 RE100 비용 계산 (정적 계산 - baseline 용)

        Args:
            energy_consumption_tier1: Tier1 에너지 소비량 (kWh/kg)
            energy_consumption_tier2: Tier2 에너지 소비량 (kWh/kg)
            re100_rate_tier1: Tier1 RE100 적용 비율 (0~1)
            re100_rate_tier2: Tier2 RE100 적용 비율 (0~1)
            region: 생산 지역
            quantity: 자재 소요량 (kg)

        Returns:
            RE100 비용 (USD)
        """
        # Tier1 비용
        tier1_premium = self.get_regional_premium(region, "tier1")
        tier1_cost = (
            energy_consumption_tier1 *
            re100_rate_tier1 *
            tier1_premium *
            quantity
        )

        # Tier2 비용
        tier2_premium = self.get_regional_premium(region, "tier2")
        tier2_cost = (
            energy_consumption_tier2 *
            re100_rate_tier2 *
            tier2_premium *
            quantity
        )

        total_cost = tier1_cost + tier2_cost

        if self.debug_mode:
            print(f"\n[RE100 Cost] {region}")
            print(f"  Tier1: {energy_consumption_tier1:.2f} kWh/kg × {re100_rate_tier1:.2%} × ${tier1_premium:.3f}/kWh × {quantity:.2f} kg = ${tier1_cost:.2f}")
            print(f"  Tier2: {energy_consumption_tier2:.2f} kWh/kg × {re100_rate_tier2:.2%} × ${tier2_premium:.3f}/kWh × {quantity:.2f} kg = ${tier2_cost:.2f}")
            print(f"  Total: ${total_cost:.2f}")

        return total_cost

    def calculate_re100_premium_expression(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any]
    ) -> pyo.Expression:
        """
        Pyomo 모델을 위한 RE100 프리미엄 표현식 생성

        이 표현식은 최적화 과정에서 RE100 변수 (tier1_re, tier2_re)의
        값에 따라 동적으로 RE100 비용을 계산합니다.

        Args:
            model: Pyomo ConcreteModel
            data: 최적화 데이터
                - scenario_df: 시나리오 DataFrame
                - material_classification: 자재 분류 정보
                - original_df: 원본 DataFrame (지역 정보 포함)

        Returns:
            pyo.Expression: RE100 프리미엄 총합 표현식
        """
        scenario_df = data['scenario_df']
        material_classification = data['material_classification']
        original_df = data.get('original_df')

        re100_premium_expr = 0.0

        for material in model.materials:
            material_info = material_classification[material]

            # 에너지 비율 (0~1 범위)
            tier1_energy_ratio = material_info.get('tier1_energy_ratio', 0)
            tier2_energy_ratio = material_info.get('tier2_energy_ratio', 0)

            # RE100 적용 가능 여부 확인
            if tier1_energy_ratio == 0 and tier2_energy_ratio == 0:
                continue  # RE100 적용 불가 자재는 스킵

            # 자재 소요량
            quantity = material_info['quantity']

            # 원본 배출계수 (kgCO2eq/kg)
            original_emission = material_info['original_emission']

            # 에너지 소비량 추정 (간단한 모델: 배출계수 기반)
            # 실제로는 electricity_usage_per_material.json 에서 가져와야 하지만
            # 여기서는 배출계수를 기반으로 에너지 소비량을 추정
            # 가정: 1 kgCO2eq ≈ 2 kWh (전력 배출계수 ~0.5 kgCO2eq/kWh)
            EMISSION_TO_ENERGY_FACTOR = 2.0
            estimated_energy = original_emission * EMISSION_TO_ENERGY_FACTOR

            # Tier별 에너지 소비량
            tier1_energy = estimated_energy * tier1_energy_ratio
            tier2_energy = estimated_energy * tier2_energy_ratio

            # 지역 정보 추출
            region = self._get_material_region(material, original_df, scenario_df)

            # 지역별 프리미엄
            tier1_premium = self.get_regional_premium(region, "tier1")
            tier2_premium = self.get_regional_premium(region, "tier2")

            # RE100 비용 표현식
            # cost = energy × re100_rate × premium × quantity
            material_re100_cost = (
                tier1_energy * model.tier1_re[material] * tier1_premium * quantity +
                tier2_energy * model.tier2_re[material] * tier2_premium * quantity
            )

            re100_premium_expr += material_re100_cost

        return re100_premium_expr

    def _get_material_region(
        self,
        material_name: str,
        original_df: Optional[pd.DataFrame],
        scenario_df: pd.DataFrame
    ) -> str:
        """
        자재의 생산 지역 추출

        Args:
            material_name: 자재명
            original_df: 원본 DataFrame
            scenario_df: 시나리오 DataFrame

        Returns:
            지역명 (예: "Korea", "China")
        """
        # 1차 시도: original_df에서 '국가' 컬럼 조회
        if original_df is not None:
            material_rows = original_df[original_df['자재명'] == material_name]
            if not material_rows.empty:
                country = material_rows.iloc[0].get('국가', 'Korea')
                return self.region_mapping.get(country, country)

        # 2차 시도: scenario_df에서 조회
        material_rows = scenario_df[scenario_df['자재명'] == material_name]
        if not material_rows.empty:
            country = material_rows.iloc[0].get('국가', 'Korea')
            return self.region_mapping.get(country, country)

        # Fallback: Korea
        return "Korea"

    def calculate_zero_re100_baseline(
        self,
        data: Dict[str, Any]
    ) -> float:
        """
        RE100 미적용 시 기준 비용 (Zero RE100 Baseline)

        RE100을 전혀 적용하지 않았을 때의 기준 비용입니다.
        이 값은 항상 0입니다 (RE100 프리미엄이 없으므로).

        Args:
            data: 최적화 데이터

        Returns:
            0.0 (RE100 미적용 시 추가 비용 없음)
        """
        return 0.0

    def set_regional_premium(
        self,
        region: str,
        tier1_premium: float,
        tier2_premium: float
    ) -> None:
        """
        지역별 RE100 프리미엄 업데이트

        Args:
            region: 지역명 (예: "Korea", "China")
            tier1_premium: Tier1 프리미엄 ($/kWh)
            tier2_premium: Tier2 프리미엄 ($/kWh)
        """
        normalized_region = self.region_mapping.get(region, region)

        self.regional_prices[normalized_region] = {
            "tier1": tier1_premium,
            "tier2": tier2_premium
        }

        if self.debug_mode:
            print(f"✅ {normalized_region} RE100 프리미엄 업데이트:")
            print(f"   Tier1: ${tier1_premium:.3f}/kWh")
            print(f"   Tier2: ${tier2_premium:.3f}/kWh")

    def get_summary(self) -> Dict[str, Any]:
        """
        RE100CostCalculator 설정 요약

        Returns:
            설정 요약 딕셔너리
        """
        return {
            'regional_prices': self.regional_prices,
            'region_mapping': self.region_mapping,
            'debug_mode': self.debug_mode
        }

    def __repr__(self) -> str:
        """RE100CostCalculator 문자열 표현"""
        num_regions = len(self.regional_prices)
        return f"<RE100CostCalculator(regions={num_regions}, debug={self.debug_mode})>"

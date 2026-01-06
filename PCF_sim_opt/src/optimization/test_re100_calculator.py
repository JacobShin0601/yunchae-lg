"""
RE100PremiumCalculator 간단한 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimization.re100_premium_calculator import RE100PremiumCalculator

def test_calculator():
    """RE100PremiumCalculator 기본 동작 테스트"""

    print("=" * 60)
    print("RE100PremiumCalculator 테스트")
    print("=" * 60)

    # Calculator 초기화
    calculator = RE100PremiumCalculator(debug_mode=True)

    print("\n" + "=" * 60)
    print("테스트 1: RE100 전환가격 계산")
    print("=" * 60)

    # 양극재 Tier1, 한국
    print("\n[Test] 양극재 Tier1, 한국")
    conversion_price = calculator.calculate_re100_conversion_price("양극재", "Tier1", "한국")
    print(f"✅ 결과: ${conversion_price:.6f}/kg")
    print(f"   예상: $0.055440/kg (7.92 × 0.007)")

    # 양극재 Tier1, 중국
    print("\n[Test] 양극재 Tier1, 중국")
    conversion_price = calculator.calculate_re100_conversion_price("양극재", "Tier1", "중국")
    print(f"✅ 결과: ${conversion_price:.6f}/kg")
    print(f"   예상: $0.028433/kg (7.92 × 0.00359)")

    print("\n" + "=" * 60)
    print("테스트 2: 상승률 계산")
    print("=" * 60)

    # 양극재 Tier1, 한국
    print("\n[Test] 양극재 Tier1, 한국")
    rate = calculator.calculate_premium_rate("양극재", "Tier1", "한국")
    print(f"✅ 결과: {rate:.2f}%")
    print(f"   예상: 0.75% (0.055440 / 7.350 × 100)")

    # Cu-Foil Tier1, 한국
    print("\n[Test] Cu-Foil Tier1, 한국")
    rate = calculator.calculate_premium_rate("Cu-Foil", "Tier1", "한국")
    print(f"✅ 결과: {rate:.2f}%")
    print(f"   예상: 1.93% (0.102200 / 5.300 × 100)")

    print("\n" + "=" * 60)
    print("테스트 3: Case별 프리미엄 계산")
    print("=" * 60)

    # Cu Foil, 0.085396943 kg, 한국, Tier1_RE=100%
    print("\n[Test] Cu Foil, 한국, Tier1_RE=100%")
    premium = calculator.calculate_case_premium(
        material_name="Foil Cu General",
        material_category="Cu Foil",
        quantity=0.085396943,
        country="한국",
        case_config={"tier1_re": 1.0, "tier2_re": 0.0}
    )

    print(f"✅ Tier1 프리미엄: ${premium['tier1_premium']:.6f}")
    print(f"✅ Tier2 프리미엄: ${premium['tier2_premium']:.6f}")
    print(f"✅ 총 프리미엄: ${premium['total_premium']:.6f}")
    print(f"✅ 상승률: {premium['premium_rate']:.2f}%")

    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    test_calculator()

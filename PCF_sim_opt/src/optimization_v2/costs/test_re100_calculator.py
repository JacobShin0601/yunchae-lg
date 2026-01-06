"""
RE100CostCalculator 간단한 테스트 스크립트

Phase 2.3 구현 검증용
"""

import sys
import os

# 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)

from src.optimization_v2.costs.re100_cost_calculator import RE100CostCalculator


def test_basic_functionality():
    """기본 기능 테스트"""
    print("="*60)
    print("TEST 1: 기본 초기화 및 지역별 프리미엄 조회")
    print("="*60)

    # Calculator 초기화 (debug mode)
    calculator = RE100CostCalculator(debug_mode=True)

    # 지역별 프리미엄 조회
    korea_tier1 = calculator.get_regional_premium("Korea", "tier1")
    korea_tier2 = calculator.get_regional_premium("Korea", "tier2")
    china_tier1 = calculator.get_regional_premium("China", "tier1")
    usa_tier1 = calculator.get_regional_premium("USA", "tier1")

    print(f"\n✅ TEST 1 통과")
    print(f"   Korea Tier1: ${korea_tier1:.3f}/kWh")
    print(f"   Korea Tier2: ${korea_tier2:.3f}/kWh")
    print(f"   China Tier1: ${china_tier1:.3f}/kWh")
    print(f"   USA Tier1: ${usa_tier1:.3f}/kWh")

    assert korea_tier1 == 0.15, "Korea Tier1 가격이 예상과 다릅니다"
    assert korea_tier2 == 0.12, "Korea Tier2 가격이 예상과 다릅니다"
    assert china_tier1 == 0.20, "China Tier1 가격이 예상과 다릅니다"
    assert usa_tier1 == 0.10, "USA Tier1 가격이 예상과 다릅니다"


def test_region_normalization():
    """지역명 정규화 테스트"""
    print("\n" + "="*60)
    print("TEST 2: 지역명 정규화 (한글/영문/약자 모두 지원)")
    print("="*60)

    calculator = RE100CostCalculator(debug_mode=False)

    # 다양한 형태의 지역명 테스트
    korea_variants = ["Korea", "한국", "KR"]
    china_variants = ["China", "중국", "CN"]
    usa_variants = ["USA", "미국", "US"]

    for variant in korea_variants:
        premium = calculator.get_regional_premium(variant, "tier1")
        print(f"   {variant:10s} → Tier1: ${premium:.3f}/kWh")
        assert premium == 0.15, f"{variant} 정규화 실패"

    for variant in china_variants:
        premium = calculator.get_regional_premium(variant, "tier1")
        print(f"   {variant:10s} → Tier1: ${premium:.3f}/kWh")
        assert premium == 0.20, f"{variant} 정규화 실패"

    for variant in usa_variants:
        premium = calculator.get_regional_premium(variant, "tier1")
        print(f"   {variant:10s} → Tier1: ${premium:.3f}/kWh")
        assert premium == 0.10, f"{variant} 정규화 실패"

    print(f"\n✅ TEST 2 통과 (지역명 정규화 성공)")


def test_material_cost_calculation():
    """자재별 RE100 비용 계산 테스트"""
    print("\n" + "="*60)
    print("TEST 3: 자재별 RE100 비용 계산")
    print("="*60)

    calculator = RE100CostCalculator(debug_mode=True)

    # 예시 자재: Cathode Active Material
    # - Tier1 에너지 소비: 10 kWh/kg
    # - Tier2 에너지 소비: 5 kWh/kg
    # - RE100 적용률: Tier1=50%, Tier2=30%
    # - 지역: Korea
    # - 수량: 100 kg

    cost = calculator.calculate_material_re100_cost(
        energy_consumption_tier1=10.0,
        energy_consumption_tier2=5.0,
        re100_rate_tier1=0.5,
        re100_rate_tier2=0.3,
        region="Korea",
        quantity=100.0
    )

    # 예상 비용:
    # Tier1: 10 kWh/kg × 0.5 × 0.15 $/kWh × 100 kg = $75.0
    # Tier2: 5 kWh/kg × 0.3 × 0.12 $/kWh × 100 kg = $18.0
    # Total: $93.0

    expected_cost = 93.0
    print(f"\n✅ TEST 3 통과")
    print(f"   계산된 비용: ${cost:.2f}")
    print(f"   예상 비용: ${expected_cost:.2f}")

    assert abs(cost - expected_cost) < 0.01, f"비용 계산이 예상과 다릅니다 (차이: ${abs(cost - expected_cost):.2f})"


def test_dynamic_premium_update():
    """프리미엄 동적 업데이트 테스트"""
    print("\n" + "="*60)
    print("TEST 4: 프리미엄 동적 업데이트")
    print("="*60)

    calculator = RE100CostCalculator(debug_mode=True)

    # 초기 Korea Tier1 프리미엄
    initial_premium = calculator.get_regional_premium("Korea", "tier1")
    print(f"\n초기 Korea Tier1: ${initial_premium:.3f}/kWh")

    # 프리미엄 업데이트
    calculator.set_regional_premium("Korea", tier1_premium=0.20, tier2_premium=0.18)

    # 업데이트된 프리미엄 확인
    updated_premium = calculator.get_regional_premium("Korea", "tier1")
    print(f"업데이트 후 Korea Tier1: ${updated_premium:.3f}/kWh")

    print(f"\n✅ TEST 4 통과")
    assert updated_premium == 0.20, "프리미엄 업데이트 실패"


def test_zero_baseline():
    """Zero RE100 Baseline 테스트"""
    print("\n" + "="*60)
    print("TEST 5: Zero RE100 Baseline (항상 0)")
    print("="*60)

    calculator = RE100CostCalculator(debug_mode=False)

    # Zero RE100 Baseline은 항상 0 (RE100 미적용 시 추가 비용 없음)
    baseline = calculator.calculate_zero_re100_baseline({})

    print(f"   Zero RE100 Baseline: ${baseline:.2f}")

    print(f"\n✅ TEST 5 통과")
    assert baseline == 0.0, "Zero RE100 Baseline이 0이 아닙니다"


def test_summary():
    """설정 요약 테스트"""
    print("\n" + "="*60)
    print("TEST 6: Calculator 설정 요약")
    print("="*60)

    calculator = RE100CostCalculator(debug_mode=False)
    summary = calculator.get_summary()

    print(f"   지역 수: {len(summary['regional_prices'])}")
    print(f"   지역 매핑 수: {len(summary['region_mapping'])}")
    print(f"   Debug mode: {summary['debug_mode']}")

    print(f"\n✅ TEST 6 통과")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*80)
    print("🧪 RE100CostCalculator 테스트 시작")
    print("="*80)

    try:
        test_basic_functionality()
        test_region_normalization()
        test_material_cost_calculation()
        test_dynamic_premium_update()
        test_zero_baseline()
        test_summary()

        print("\n" + "="*80)
        print("✅ 모든 테스트 통과!")
        print("="*80)
        return True

    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
RE100-비용 관계 검증 도구

RE100 최적화 결과가 비용 제약을 준수하는지 검증하고,
RE100 비율과 실제 비용 프리미엄의 관계를 분석합니다.
"""

from typing import Dict, Any, List, Tuple, Optional
import pyomo.environ as pyo


class RE100CostValidator:
    """RE100 설정이 비용 제약을 준수하는지 검증"""

    def __init__(self):
        """검증기 초기화"""
        self.validation_results: Dict[str, Any] = {}

    def validate_solution(
        self,
        model: pyo.ConcreteModel,
        optimization_data: Dict[str, Any],
        cost_calculator,
        cost_baseline: float,
        cost_limit: float
    ) -> Dict[str, Any]:
        """
        최적화 결과의 RE100-비용 관계 검증

        Args:
            model: Pyomo 최적화 모델 (solved)
            optimization_data: 최적화 데이터
            cost_calculator: TotalCostCalculator 인스턴스
            cost_baseline: Zero-premium baseline 비용
            cost_limit: 최대 허용 비용

        Returns:
            {
                'is_valid': bool,
                'total_cost': float,
                'cost_breakdown': {
                    'baseline_cost': float,
                    're100_premium': float,
                    'material_premium': float
                },
                're100_stats': {
                    'avg_tier1': float,
                    'avg_tier2': float,
                    'materials_with_re100': int
                },
                'warnings': List[str],
                'recommendations': List[str]
            }
        """
        warnings = []
        recommendations = []

        print("\n" + "=" * 60)
        print("🔍 RE100-비용 관계 검증")
        print("=" * 60)

        # 1. RE100 통계 계산
        re100_stats = self._calculate_re100_stats(model, optimization_data)

        print(f"\n📊 RE100 통계:")
        print(f"   • 평균 Tier1 RE: {re100_stats['avg_tier1']*100:.2f}%")
        print(f"   • 평균 Tier2 RE: {re100_stats['avg_tier2']*100:.2f}%")
        print(f"   • RE100 적용 자재: {re100_stats['materials_with_re100']}개")
        print(f"   • 총 자재: {re100_stats['total_materials']}개")

        # 2. 실제 총 비용 계산
        total_cost = self._calculate_total_cost(model, optimization_data, cost_calculator)

        print(f"\n💰 비용 분석:")
        print(f"   • Zero-Premium Baseline: ${cost_baseline:,.2f}")
        print(f"   • 실제 총 비용: ${total_cost:,.2f}")
        print(f"   • 비용 한도: ${cost_limit:,.2f}")
        print(f"   • 프리미엄: ${total_cost - cost_baseline:,.2f} ({(total_cost/cost_baseline - 1)*100:.2f}%)")

        # 3. 비용 분해 (RE100 vs 자재 프리미엄)
        cost_breakdown = self._decompose_costs(
            model, optimization_data, cost_calculator, cost_baseline
        )

        print(f"\n📈 비용 분해:")
        print(f"   • Baseline 비용: ${cost_breakdown['baseline_cost']:,.2f}")
        print(f"   • RE100 프리미엄: ${cost_breakdown['re100_premium']:,.2f}")
        print(f"   • 재활용/저탄소 프리미엄: ${cost_breakdown['material_premium']:,.2f}")
        print(f"   • 총 프리미엄: ${cost_breakdown['total_premium']:,.2f}")

        # 4. 비용 한도 초과 여부
        cost_within_limit = total_cost <= cost_limit
        cost_margin = cost_limit - total_cost

        print(f"\n✅ 비용 제약 준수 여부:")
        if cost_within_limit:
            print(f"   ✅ 통과 - 한도 내 ${cost_margin:,.2f} 여유")
        else:
            print(f"   ❌ 실패 - 한도 초과 ${-cost_margin:,.2f}")

        # 5. 경고 및 권장사항 생성
        # 5.1 RE100이 매우 높은 경우
        if re100_stats['avg_tier1'] > 0.95 or re100_stats['avg_tier2'] > 0.95:
            warnings.append(
                f"⚠️  RE100이 매우 높음 (Tier1={re100_stats['avg_tier1']*100:.1f}%, "
                f"Tier2={re100_stats['avg_tier2']*100:.1f}%). "
                f"비용 제약이 실제로 작동하는지 확인하세요."
            )

            # 비용 한도가 매우 높은 경우
            if cost_limit > cost_baseline * 1.5:
                recommendations.append(
                    f"💡 비용 한도({(cost_limit/cost_baseline - 1)*100:.1f}%)가 높아 "
                    f"RE100을 최대한 사용할 수 있습니다. "
                    f"더 엄격한 비용 제약을 시도해보세요."
                )

        # 5.2 RE100 프리미엄이 너무 낮은 경우 (의심)
        if cost_breakdown['re100_premium'] < cost_baseline * 0.01:
            warnings.append(
                f"⚠️  RE100 프리미엄이 매우 낮음 (${cost_breakdown['re100_premium']:,.2f}). "
                f"RE100 비용 계산이 올바른지 확인하세요."
            )

        # 5.3 RE100과 비용의 관계 분석
        if re100_stats['avg_tier1'] > 0.9 and cost_within_limit:
            recommendations.append(
                f"✅ RE100 {re100_stats['avg_tier1']*100:.1f}%는 비용 제약 내에서 달성 가능합니다. "
                f"이는 정상적인 최적화 결과입니다."
            )

        # 5.4 비용 한도를 거의 사용한 경우
        if cost_margin < (cost_limit - cost_baseline) * 0.1:
            recommendations.append(
                f"ℹ️  비용 한도를 거의 사용 중 (여유: ${cost_margin:,.2f}). "
                f"RE100을 더 높이려면 비용 한도를 늘려야 합니다."
            )

        # 6. 경고 및 권장사항 출력
        if warnings:
            print(f"\n⚠️  경고:")
            for warning in warnings:
                print(f"   {warning}")

        if recommendations:
            print(f"\n💡 권장사항:")
            for rec in recommendations:
                print(f"   {rec}")

        print("\n" + "=" * 60)

        # 결과 반환
        return {
            'is_valid': cost_within_limit,
            'total_cost': total_cost,
            'cost_baseline': cost_baseline,
            'cost_limit': cost_limit,
            'cost_margin': cost_margin,
            'cost_breakdown': cost_breakdown,
            're100_stats': re100_stats,
            'warnings': warnings,
            'recommendations': recommendations
        }

    def _calculate_re100_stats(
        self,
        model: pyo.ConcreteModel,
        optimization_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        RE100 통계 계산

        Returns:
            {
                'avg_tier1': float,
                'avg_tier2': float,
                'materials_with_re100': int,
                'total_materials': int,
                'max_tier1': float,
                'max_tier2': float,
                'min_tier1': float,
                'min_tier2': float
            }
        """
        tier1_values = []
        tier2_values = []
        materials_with_re100 = 0

        material_classification = optimization_data['material_classification']

        for material in model.materials:
            material_info = material_classification[material]

            # Formula 자재만 RE100 적용
            if material_info['is_formula_applicable']:
                try:
                    tier1_re = pyo.value(model.tier1_re[material])
                    tier2_re = pyo.value(model.tier2_re[material])

                    tier1_values.append(tier1_re)
                    tier2_values.append(tier2_re)

                    if tier1_re > 0.01 or tier2_re > 0.01:
                        materials_with_re100 += 1
                except:
                    # 변수가 초기화되지 않은 경우 0으로 처리
                    tier1_values.append(0)
                    tier2_values.append(0)

        return {
            'avg_tier1': sum(tier1_values) / len(tier1_values) if tier1_values else 0,
            'avg_tier2': sum(tier2_values) / len(tier2_values) if tier2_values else 0,
            'max_tier1': max(tier1_values) if tier1_values else 0,
            'max_tier2': max(tier2_values) if tier2_values else 0,
            'min_tier1': min(tier1_values) if tier1_values else 0,
            'min_tier2': min(tier2_values) if tier2_values else 0,
            'materials_with_re100': materials_with_re100,
            'total_materials': len(model.materials)
        }

    def _calculate_total_cost(
        self,
        model: pyo.ConcreteModel,
        optimization_data: Dict[str, Any],
        cost_calculator
    ) -> float:
        """
        실제 총 비용 계산

        Args:
            model: Pyomo 모델
            optimization_data: 최적화 데이터
            cost_calculator: TotalCostCalculator 인스턴스

        Returns:
            총 비용 (float)
        """
        try:
            total_cost = pyo.value(
                cost_calculator.calculate_total_cost(model, optimization_data)
            )
            return float(total_cost)
        except Exception as e:
            print(f"   ⚠️  총 비용 계산 실패: {e}")
            return 0.0

    def _decompose_costs(
        self,
        model: pyo.ConcreteModel,
        optimization_data: Dict[str, Any],
        cost_calculator,
        baseline_cost: float
    ) -> Dict[str, float]:
        """
        비용을 구성 요소로 분해

        Returns:
            {
                'baseline_cost': float,
                're100_premium': float,
                'material_premium': float,
                'total_premium': float
            }
        """
        try:
            # TotalCostCalculator를 사용하여 상세 비용 계산
            # (실제 구현은 TotalCostCalculator의 메서드에 따라 달라질 수 있음)

            # 전체 비용
            total_cost = self._calculate_total_cost(model, optimization_data, cost_calculator)
            total_premium = total_cost - baseline_cost

            # RE100 프리미엄 계산
            re100_premium = self._calculate_re100_premium(
                model, optimization_data, cost_calculator
            )

            # 재활용/저탄소 프리미엄 계산
            material_premium = total_premium - re100_premium

            return {
                'baseline_cost': baseline_cost,
                're100_premium': re100_premium,
                'material_premium': material_premium,
                'total_premium': total_premium
            }

        except Exception as e:
            print(f"   ⚠️  비용 분해 실패: {e}")
            return {
                'baseline_cost': baseline_cost,
                're100_premium': 0.0,
                'material_premium': 0.0,
                'total_premium': 0.0
            }

    def _calculate_re100_premium(
        self,
        model: pyo.ConcreteModel,
        optimization_data: Dict[str, Any],
        cost_calculator
    ) -> float:
        """
        RE100 프리미엄만 계산

        Args:
            model: Pyomo 모델
            optimization_data: 최적화 데이터
            cost_calculator: TotalCostCalculator 인스턴스

        Returns:
            RE100 프리미엄 비용
        """
        try:
            scenario_df = optimization_data['scenario_df']
            material_classification = optimization_data['material_classification']

            re100_premium = 0.0

            for material in model.materials:
                material_info = material_classification[material]

                # Formula 자재만 RE100 적용
                if not material_info['is_formula_applicable']:
                    continue

                # 자재 정보 가져오기
                material_row = scenario_df[scenario_df['자재명'] == material]
                if material_row.empty:
                    continue

                quantity = material_row['제품총소요량(kg)'].iloc[0]
                material_category = material_row['자재품목'].iloc[0]

                # RE100 비율 추출
                try:
                    tier1_re = pyo.value(model.tier1_re[material])
                    tier2_re = pyo.value(model.tier2_re[material])
                except:
                    tier1_re = 0
                    tier2_re = 0

                # RE100 전환 가격 계산
                # (TotalCostCalculator의 cost_calculator 사용)
                opt_material = cost_calculator._map_material_category(material_category)
                country = "한국"  # 기본값

                tier1_conversion = cost_calculator.calculate_re100_conversion_price(
                    opt_material, "Tier1", country
                )
                tier2_conversion = cost_calculator.calculate_re100_conversion_price(
                    opt_material, "Tier2", country
                )

                # RE100 프리미엄 누적
                material_re100_premium = quantity * (
                    tier1_conversion * tier1_re + tier2_conversion * tier2_re
                )
                re100_premium += material_re100_premium

            return float(re100_premium)

        except Exception as e:
            print(f"   ⚠️  RE100 프리미엄 계산 실패: {e}")
            return 0.0

    def generate_report(self, validation_result: Dict[str, Any]) -> str:
        """
        검증 결과 리포트 생성

        Args:
            validation_result: validate_solution() 반환값

        Returns:
            마크다운 형식의 리포트
        """
        report = []
        report.append("# RE100-비용 검증 리포트\n")
        report.append(f"**검증 결과**: {'✅ 통과' if validation_result['is_valid'] else '❌ 실패'}\n")
        report.append("\n## 비용 분석\n")
        report.append(f"- Zero-Premium Baseline: ${validation_result['cost_baseline']:,.2f}")
        report.append(f"- 실제 총 비용: ${validation_result['total_cost']:,.2f}")
        report.append(f"- 비용 한도: ${validation_result['cost_limit']:,.2f}")
        report.append(f"- 비용 여유: ${validation_result['cost_margin']:,.2f}\n")

        report.append("\n## 비용 분해\n")
        breakdown = validation_result['cost_breakdown']
        report.append(f"- Baseline 비용: ${breakdown['baseline_cost']:,.2f}")
        report.append(f"- RE100 프리미엄: ${breakdown['re100_premium']:,.2f}")
        report.append(f"- 재활용/저탄소 프리미엄: ${breakdown['material_premium']:,.2f}")
        report.append(f"- 총 프리미엄: ${breakdown['total_premium']:,.2f}\n")

        report.append("\n## RE100 통계\n")
        stats = validation_result['re100_stats']
        report.append(f"- 평균 Tier1 RE: {stats['avg_tier1']*100:.2f}%")
        report.append(f"- 평균 Tier2 RE: {stats['avg_tier2']*100:.2f}%")
        report.append(f"- RE100 적용 자재: {stats['materials_with_re100']}개 / {stats['total_materials']}개\n")

        if validation_result['warnings']:
            report.append("\n## ⚠️  경고\n")
            for warning in validation_result['warnings']:
                report.append(f"- {warning}")
            report.append("")

        if validation_result['recommendations']:
            report.append("\n## 💡 권장사항\n")
            for rec in validation_result['recommendations']:
                report.append(f"- {rec}")
            report.append("")

        return "\n".join(report)

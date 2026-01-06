"""
결과 처리기

최적화 결과를 사용자 친화적인 형식으로 변환하고 시각화 데이터를 준비합니다.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd


class ResultProcessor:
    """
    최적화 결과 처리 클래스

    - 결과 DataFrame 생성
    - 요약 통계 계산
    - 비교 분석
    - 시각화 데이터 준비
    """

    def __init__(self):
        """결과 처리기 초기화"""
        self.solution: Optional[Dict[str, Any]] = None
        self.baseline_data: Optional[Dict[str, Any]] = None

    def _calculate_element_reduction(self, material_data: Dict[str, Any]) -> float:
        """
        Element-level 최적화 기여도 계산 (양극재 전용)

        Args:
            material_data: 자재 데이터

        Returns:
            Element-level 감축률 (%)
        """
        if not material_data.get('is_cathode'):
            return 0.0

        original = material_data.get('original_emission', 0)
        if original == 0:
            return 0.0

        # Element-level만 적용한 배출계수 (RE100 제외)
        # 재활용/저탄소/버진 비율만 고려한 배출계수를 추정
        # 실제 계산은 optimization_engine에서 수행되므로 근사값 사용
        virgin_ratio = material_data.get('virgin_ratio', 1.0)
        recycle_ratio = material_data.get('recycle_ratio', 0.0)
        low_carbon_ratio = material_data.get('low_carbon_ratio', 0.0)

        # 재활용재/저탄소 메탈의 배출계수 감축 효과 (대략적 추정)
        # 재활용재: ~30-40% 감축, 저탄소: ~50-60% 감축
        recycle_impact = 0.65  # 재활용재는 버진의 65% 배출
        low_carbon_impact = 0.45  # 저탄소는 버진의 45% 배출

        element_only_emission = original * (
            virgin_ratio * 1.0 +
            recycle_ratio * recycle_impact +
            low_carbon_ratio * low_carbon_impact
        )

        return (1 - element_only_emission / original) * 100

    def _calculate_re100_reduction(self, material_data: Dict[str, Any]) -> float:
        """
        RE100 적용 기여도 계산

        Args:
            material_data: 자재 데이터

        Returns:
            RE100 감축률 (%)
        """
        if not material_data.get('is_cathode'):
            # 양극재가 아닌 경우, 전체 감축률에서 element 기여 제외
            tier1_re = material_data.get('tier1_re', 0)
            tier2_re = material_data.get('tier2_re', 0)
            if tier1_re > 0 or tier2_re > 0:
                # RE100만 적용된 자재
                return material_data.get('reduction_pct', 0)
            return 0.0

        # 양극재의 경우
        element_reduction = self._calculate_element_reduction(material_data)
        total_reduction = material_data.get('reduction_pct', 0)

        # RE100 기여도 = 전체 감축률 - Element 기여도 (곱셈 관계 고려)
        # 1 - total = (1 - element) * (1 - re100)
        # → 1 - re100 = (1 - total) / (1 - element)
        # → re100 = 1 - (1 - total) / (1 - element)

        if element_reduction >= 99.9:  # Element만으로 거의 100% 감축
            return 0.0

        element_factor = 1 - element_reduction / 100
        total_factor = 1 - total_reduction / 100

        if element_factor > 0:
            re100_factor = 1 - (total_factor / element_factor)
            return re100_factor * 100

        return 0.0

    def process_solution(self, solution: Dict[str, Any]) -> pd.DataFrame:
        """
        최적화 결과를 DataFrame으로 변환

        Args:
            solution: extract_solution()에서 반환된 결과

        Returns:
            결과 DataFrame
        """
        self.solution = solution

        print("\n📋 결과 처리 중...")

        # 자재별 결과를 DataFrame으로 변환
        rows = []
        for material_name, material_data in solution['materials'].items():
            row = {
                '자재명': material_name,
                '자재_타입': material_data.get('type', 'General'),
                '제품총소요량(kg)': material_data['quantity'],
                '원본_배출계수': material_data['original_emission'],
                '최적_배출계수': material_data['modified_emission'],
                '감축률(%)': material_data.get('reduction_pct', 0),
            }

            # 자재 타입에 따라 해당 필드만 추가
            row['Tier1_RE(%)'] = material_data.get('tier1_re', 0) * 100
            row['Tier2_RE(%)'] = material_data.get('tier2_re', 0) * 100
            row['재활용_비율(%)'] = material_data.get('recycle_ratio', 0) * 100
            row['저탄소_비율(%)'] = material_data.get('low_carbon_ratio', 0) * 100
            row['버진_비율(%)'] = material_data.get('virgin_ratio', 0) * 100

            # 감축률 세분화 (양극재 전용)
            if material_data.get('is_cathode'):
                row['Element감축률(%)'] = self._calculate_element_reduction(material_data)
                row['RE100감축률(%)'] = self._calculate_re100_reduction(material_data)
            else:
                row['Element감축률(%)'] = 0.0
                row['RE100감축률(%)'] = self._calculate_re100_reduction(material_data)

            # DIAGNOSTIC POINT C: After display processing
            if '양극재' in material_name or '양극활물질' in material_name or 'Cathode' in material_name:
                print(f"   ✓ [POINT C - DISPLAY ROW] {material_name[:40]}")
                print(f"      Source material_data keys: {list(material_data.keys())}")
                print(f"      tier1_re from material_data: {material_data.get('tier1_re', 'MISSING')}")
                print(f"      tier2_re from material_data: {material_data.get('tier2_re', 'MISSING')}")
                print(f"      recycle_ratio from material_data: {material_data.get('recycle_ratio', 'MISSING')}")
                print(f"      Tier1_RE(%): {row['Tier1_RE(%)']}")
                print(f"      Tier2_RE(%): {row['Tier2_RE(%)']}")
                print(f"      재활용_비율(%): {row['재활용_비율(%)']}")
                print(f"      저탄소_비율(%): {row['저탄소_비율(%)']}")
                print(f"      버진_비율(%): {row['버진_비율(%)']}")
                print(f"      감축률(%): {row['감축률(%)']}")

            # 총 배출량 계산
            row['원본_배출량(kgCO2eq)'] = row['원본_배출계수'] * row['제품총소요량(kg)']
            row['최적_배출량(kgCO2eq)'] = row['최적_배출계수'] * row['제품총소요량(kg)']
            row['배출량_감축(kgCO2eq)'] = row['원본_배출량(kgCO2eq)'] - row['최적_배출량(kgCO2eq)']

            rows.append(row)

        df = pd.DataFrame(rows)

        print(f"   ✅ {len(df)}개 자재 처리 완료")

        return df

    def calculate_summary(self, result_df: pd.DataFrame) -> Dict[str, Any]:
        """
        요약 통계 계산

        Args:
            result_df: process_solution()에서 반환된 DataFrame

        Returns:
            요약 통계 딕셔너리
        """
        print("\n📊 요약 통계 계산 중...")

        optimized_emission = result_df['최적_배출량(kgCO2eq)'].sum()

        summary = {
            # 전체 통계
            'material_count': len(result_df),
            'total_quantity': result_df['제품총소요량(kg)'].sum(),

            # 배출량 통계
            'baseline_total_emission': result_df['원본_배출량(kgCO2eq)'].sum(),
            'optimized_total_emission': optimized_emission,
            'total_carbon': optimized_emission,  # 파레토 최적화 호환성을 위한 별칭
            'total_reduction': result_df['배출량_감축(kgCO2eq)'].sum(),

            # 평균 감축률
            'average_reduction_pct': result_df['감축률(%)'].mean(),
            'max_reduction_pct': result_df['감축률(%)'].max(),
            'min_reduction_pct': result_df['감축률(%)'].min(),

            # RE 적용 통계
            'avg_tier1_re': result_df['Tier1_RE(%)'].mean(),
            'avg_tier2_re': result_df['Tier2_RE(%)'].mean(),

            # 자재 구성 통계
            'avg_recycle_ratio': result_df['재활용_비율(%)'].mean(),
            'avg_low_carbon_ratio': result_df['저탄소_비율(%)'].mean(),
            'avg_virgin_ratio': result_df['버진_비율(%)'].mean(),
        }

        # 전체 감축률 계산
        if summary['baseline_total_emission'] > 0:
            summary['total_reduction_pct'] = (
                summary['total_reduction'] / summary['baseline_total_emission'] * 100
            )
        else:
            summary['total_reduction_pct'] = 0

        print(f"   ✅ 요약 통계 계산 완료")
        print(f"      • 총 배출량 감축: {summary['total_reduction']:.2f} kgCO2eq ({summary['total_reduction_pct']:.2f}%)")

        return summary

    def compare_with_baseline(
        self,
        result_df: pd.DataFrame,
        baseline_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        기준선(baseline)과 비교

        Args:
            result_df: 최적화 결과 DataFrame
            baseline_df: 기준선 DataFrame (원본 시뮬레이션 결과)

        Returns:
            비교 DataFrame
        """
        print("\n🔄 기준선과 비교 중...")

        # 자재명 기준으로 병합
        comparison = result_df.merge(
            baseline_df[['자재명', '배출계수', '배출량(kgCO2eq)']],
            on='자재명',
            how='left',
            suffixes=('', '_baseline')
        )

        # 비교 컬럼 추가
        comparison['배출계수_변화(%)'] = (
            (comparison['최적_배출계수'] - comparison['배출계수']) /
            comparison['배출계수'] * 100
        )

        comparison['배출량_변화(%)'] = (
            (comparison['최적_배출량(kgCO2eq)'] - comparison['배출량(kgCO2eq)']) /
            comparison['배출량(kgCO2eq)'] * 100
        )

        print(f"   ✅ {len(comparison)}개 자재 비교 완료")

        return comparison

    def get_top_contributors(
        self,
        result_df: pd.DataFrame,
        top_n: int = 10,
        metric: str = 'reduction'
    ) -> pd.DataFrame:
        """
        주요 기여 자재 추출

        Args:
            result_df: 결과 DataFrame
            top_n: 상위 N개
            metric: 기준 지표
                - 'reduction': 배출량 감축
                - 'emission': 최적 배출량
                - 'reduction_pct': 감축률

        Returns:
            상위 N개 자재 DataFrame
        """
        print(f"\n🏆 상위 {top_n}개 자재 추출 (기준: {metric})...")

        if metric == 'reduction':
            sorted_df = result_df.sort_values('배출량_감축(kgCO2eq)', ascending=False)
            print(f"   기준: 배출량 감축 (kgCO2eq)")

        elif metric == 'emission':
            sorted_df = result_df.sort_values('최적_배출량(kgCO2eq)', ascending=False)
            print(f"   기준: 최적 배출량 (kgCO2eq)")

        elif metric == 'reduction_pct':
            sorted_df = result_df.sort_values('감축률(%)', ascending=False)
            print(f"   기준: 감축률 (%)")

        else:
            raise ValueError(f"알 수 없는 metric: {metric}")

        top_materials = sorted_df.head(top_n)

        print(f"   ✅ 상위 {len(top_materials)}개 추출 완료")

        return top_materials

    def prepare_visualization_data(
        self,
        result_df: pd.DataFrame,
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        시각화용 데이터 준비

        Args:
            result_df: 결과 DataFrame
            summary: 요약 통계

        Returns:
            시각화 데이터 딕셔너리
        """
        print("\n📈 시각화 데이터 준비 중...")

        viz_data = {
            # 막대 차트: 자재별 감축률
            'reduction_by_material': {
                'materials': result_df['자재명'].tolist(),
                'reduction_pct': result_df['감축률(%)'].tolist(),
                'emission_reduction': result_df['배출량_감축(kgCO2eq)'].tolist()
            },

            # 파이 차트: 전체 배출량 구성
            'emission_composition': {
                'categories': ['기준', '최적화'],
                'values': [
                    summary['baseline_total_emission'],
                    summary['optimized_total_emission']
                ]
            },

            # 히트맵: RE 적용률
            're_application': {
                'materials': result_df['자재명'].tolist(),
                'tier1_re': result_df['Tier1_RE(%)'].tolist(),
                'tier2_re': result_df['Tier2_RE(%)'].tolist()
            },

            # 스택 바: 자재 구성
            'material_composition': {
                'materials': result_df['자재명'].tolist(),
                'recycle': result_df['재활용_비율(%)'].tolist(),
                'low_carbon': result_df['저탄소_비율(%)'].tolist(),
                'virgin': result_df['버진_비율(%)'].tolist()
            },

            # 워터폴: 배출량 변화
            'emission_waterfall': {
                'baseline': summary['baseline_total_emission'],
                'reduction': summary['total_reduction'],
                'optimized': summary['optimized_total_emission']
            }
        }

        print(f"   ✅ 시각화 데이터 준비 완료")

        return viz_data

    def export_to_dict(
        self,
        result_df: pd.DataFrame,
        summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        결과를 딕셔너리로 내보내기 (JSON 직렬화 가능)

        Args:
            result_df: 결과 DataFrame
            summary: 요약 통계

        Returns:
            결과 딕셔너리
        """
        return {
            'summary': summary,
            'materials': result_df.to_dict(orient='records'),
            'metadata': {
                'material_count': len(result_df),
                'status': 'success'
            }
        }

    def generate_report(
        self,
        result_df: pd.DataFrame,
        summary: Dict[str, Any]
    ) -> str:
        """
        텍스트 리포트 생성

        Args:
            result_df: 결과 DataFrame
            summary: 요약 통계

        Returns:
            리포트 문자열
        """
        report = []
        report.append("=" * 70)
        report.append("📊 최적화 결과 리포트")
        report.append("=" * 70)
        report.append("")

        # 요약
        report.append("[ 요약 ]")
        report.append(f"  • 대상 자재: {summary['material_count']}개")
        report.append(f"  • 총 물량: {summary['total_quantity']:.2f} kg")
        report.append("")

        # 배출량
        report.append("[ 배출량 ]")
        report.append(f"  • 기준 배출량: {summary['baseline_total_emission']:.2f} kgCO2eq")
        report.append(f"  • 최적 배출량: {summary['optimized_total_emission']:.2f} kgCO2eq")
        report.append(f"  • 총 감축량: {summary['total_reduction']:.2f} kgCO2eq")
        report.append(f"  • 감축률: {summary['total_reduction_pct']:.2f}%")
        report.append("")

        # 감축 활동
        report.append("[ 감축 활동 ]")
        report.append(f"  • 평균 Tier1 RE: {summary['avg_tier1_re']:.1f}%")
        report.append(f"  • 평균 Tier2 RE: {summary['avg_tier2_re']:.1f}%")
        report.append(f"  • 평균 재활용 비율: {summary['avg_recycle_ratio']:.1f}%")
        report.append(f"  • 평균 저탄소 비율: {summary['avg_low_carbon_ratio']:.1f}%")
        report.append("")

        # 상위 기여 자재
        report.append("[ 상위 5개 감축 자재 ]")
        top5 = self.get_top_contributors(result_df, top_n=5, metric='reduction')
        for idx, row in top5.iterrows():
            report.append(
                f"  {idx+1}. {row['자재명']}: "
                f"{row['배출량_감축(kgCO2eq)']:.2f} kgCO2eq (-{row['감축률(%)']:.1f}%)"
            )
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def __repr__(self) -> str:
        """ResultProcessor 문자열 표현"""
        has_solution = self.solution is not None
        return f"<ResultProcessor(solution_processed={has_solution})>"

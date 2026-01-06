"""
기본 워크플로우 테스트

DataLoader → OptimizationEngine → ResultProcessor의 전체 워크플로우를 테스트합니다.

실행 방법:
    python src/optimization_v2/tests/test_basic_workflow.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import pandas as pd
from src.optimization_v2.utils.data_loader import DataLoader
from src.optimization_v2.core.optimization_engine import OptimizationEngine
from src.optimization_v2.core.result_processor import ResultProcessor
from src.optimization_v2.constraints import (
    MaterialManagementConstraint,
    CostConstraint,
    LocationConstraint
)


def create_sample_data():
    """테스트용 샘플 데이터 생성"""
    print("\n📝 샘플 데이터 생성 중...")

    # 샘플 scenario_df
    scenario_df = pd.DataFrame([
        {
            '자재명': '양극재',
            '자재품목': 'Cathode',
            '제품총소요량(kg)': 100.0,
            '배출계수': 10.0,
            '배출량(kgCO2eq)': 1000.0,
            '저감활동_적용여부': 1,
            '지역': '한국'
        },
        {
            '자재명': 'Cu Foil',
            '자재품목': 'Foil',
            '제품총소요량(kg)': 50.0,
            '배출계수': 8.0,
            '배출량(kgCO2eq)': 400.0,
            '저감활동_적용여부': 1,
            '지역': '일본'
        },
        {
            '자재명': '분리막',
            '자재품목': 'Separator',
            '제품총소요량(kg)': 30.0,
            '배출계수': 12.0,
            '배출량(kgCO2eq)': 360.0,
            '저감활동_적용여부': 1,
            '지역': '중국'
        }
    ])

    # 샘플 ref_formula_df (Formula 적용 가능 자재)
    ref_formula_df = pd.DataFrame([
        {'자재명': '양극재'},
        {'자재명': '분리막'}
    ])

    # 샘플 ref_proportions_df (Ni/Co/Li 자재)
    ref_proportions_df = pd.DataFrame([
        {'자재명': 'Cu Foil'}
    ])

    print("   ✅ 샘플 데이터 생성 완료")

    return {
        'scenario_df': scenario_df,
        'original_df': scenario_df.copy(),
        'ref_formula_df': ref_formula_df,
        'ref_proportions_df': ref_proportions_df
    }


def test_data_loader():
    """DataLoader 테스트"""
    print("\n" + "=" * 70)
    print("TEST 1: DataLoader")
    print("=" * 70)

    # 샘플 데이터 생성
    sample_data = create_sample_data()

    # DataLoader 생성
    loader = DataLoader()
    loader.scenario_df = sample_data['scenario_df']
    loader.original_df = sample_data['original_df']
    loader.ref_formula_df = sample_data['ref_formula_df']
    loader.ref_proportions_df = sample_data['ref_proportions_df']

    # 데이터 검증
    is_valid, errors = loader.validate_data()

    if is_valid:
        print("✅ DataLoader 테스트 통과!")
    else:
        print(f"❌ DataLoader 테스트 실패: {errors}")
        return None

    # 최적화 데이터 반환
    return loader.get_optimization_data()


def test_optimization_engine(data):
    """OptimizationEngine 테스트"""
    print("\n" + "=" * 70)
    print("TEST 2: OptimizationEngine")
    print("=" * 70)

    try:
        # OptimizationEngine 생성
        engine = OptimizationEngine(solver_name='glpk')

        # 제약조건 추가 (선택적)
        material_constraint = MaterialManagementConstraint()
        material_constraint.add_rule(
            rule_type='exclude_low_carbon',
            material_name='양극재',
            params={}
        )
        engine.constraint_manager.add_constraint(material_constraint, priority=10)

        # 모델 구축
        model = engine.build_model(data, objective_type='minimize_carbon')

        # 최적화 실행
        results = engine.solve(time_limit=60, verbose=False)

        # 결과 추출
        solution = engine.extract_solution()

        print("✅ OptimizationEngine 테스트 통과!")
        return solution

    except Exception as e:
        print(f"❌ OptimizationEngine 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_result_processor(solution):
    """ResultProcessor 테스트"""
    print("\n" + "=" * 70)
    print("TEST 3: ResultProcessor")
    print("=" * 70)

    try:
        # ResultProcessor 생성
        processor = ResultProcessor()

        # 결과 처리
        result_df = processor.process_solution(solution)

        # 요약 통계 계산
        summary = processor.calculate_summary(result_df)

        # 리포트 생성
        report = processor.generate_report(result_df, summary)

        print("\n" + report)

        print("\n✅ ResultProcessor 테스트 통과!")
        return result_df, summary

    except Exception as e:
        print(f"❌ ResultProcessor 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 70)
    print("🧪 Optimization V2 - 기본 워크플로우 테스트")
    print("=" * 70)

    # Step 1: DataLoader 테스트
    data = test_data_loader()
    if data is None:
        print("\n❌ 테스트 중단: DataLoader 실패")
        return

    # Step 2: OptimizationEngine 테스트
    solution = test_optimization_engine(data)
    if solution is None:
        print("\n❌ 테스트 중단: OptimizationEngine 실패")
        return

    # Step 3: ResultProcessor 테스트
    result_df, summary = test_result_processor(solution)
    if result_df is None:
        print("\n❌ 테스트 중단: ResultProcessor 실패")
        return

    # 전체 성공
    print("\n" + "=" * 70)
    print("✅ 모든 테스트 통과!")
    print("=" * 70)
    print("\n🎉 Phase 2 완료: 최적화 엔진 통합 성공!")


if __name__ == "__main__":
    main()

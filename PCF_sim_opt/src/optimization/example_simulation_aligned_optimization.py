"""
시뮬레이션 정렬 최적화 사용 예제
rule_based.py와 정확히 동일한 로직으로 탄소발자국을 계산하는 최적화
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 현재 디렉토리 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from input import OptimizationInput
from carbon_minimization import CarbonMinimization
from simulation_aligned_objective import SimulationAlignedObjective


def load_sample_simulation_data() -> tuple:
    """
    샘플 시뮬레이션 데이터 로드
    실제 사용 시에는 RuleBasedSim에서 가져온 데이터 사용
    """
    # 샘플 scenario_df
    scenario_df = pd.DataFrame({
        '자재명': ['Cathode Active Material', 'Separator', 'Electrolyte', 'Anode Natural', 'Cu Foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        '배출계수': [12.5, 2.3, 4.1, 3.8, 15.2],
        '제품총소요량(kg)': [0.856, 0.025, 0.098, 0.103, 0.067],
        '저감활동_적용여부': [1.0, 1.0, 1.0, 1.0, 1.0],
        '배출량(kgCO2eq)': [10.7, 0.0575, 0.4018, 0.3914, 1.0184]
    })
    
    # 샘플 ref_formula_df
    ref_formula_df = pd.DataFrame({
        '자재명': ['Cathode Active Material', 'Separator', 'Electrolyte', 'Anode Natural', 'Cu Foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        '자재코드': ['CAM01', 'SEP01', 'ELY01', 'ANG01', 'CUF01'],
        '지역': ['중국', '중국', '중국', '중국', '중국'],
        'Tier1_RE100(kgCO2eq/kg)': [0.8, 0.15, 0.25, 0.18, 1.2],
        'Tier2_RE100(kgCO2eq/kg)': [0.6, 0.12, 0.18, 0.14, 0.9],
        'Tier3_RE100(kgCO2eq/kg)': [0.4, 0.08, 0.12, 0.09, 0.6]
    })
    
    # 샘플 ref_proportions_df
    ref_proportions_df = pd.DataFrame({
        '자재명(포함)': ['cathode active material', 'separator', 'electrolyte', 'natural', 'foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        'Tier1_RE100(%)': [15.0, 8.0, 12.0, 10.0, 20.0],
        'Tier2_RE100(%)': [25.0, 12.0, 18.0, 15.0, 30.0],
        'Tier3_RE100(%)': [35.0, 15.0, 25.0, 20.0, 40.0]
    })
    
    # 샘플 original_df
    original_df = pd.DataFrame({
        '자재명': ['Cathode Active Material', 'Separator', 'Electrolyte', 'Anode Natural', 'Cu Foil'],
        '자재품목': ['양극재', '분리막', '전해액', '음극재', '동박'],
        '자재코드': ['CAM01', 'SEP01', 'ELY01', 'ANG01', 'CUF01'],
        '지역': ['중국', '중국', '중국', '중국', '중국'],
        '배출계수': [12.5, 2.3, 4.1, 3.8, 15.2]
    })
    
    return scenario_df, ref_formula_df, ref_proportions_df, original_df


def create_simulation_aligned_optimization_config():
    """시뮬레이션 정렬 최적화 설정 생성"""
    config = {
        "metadata": {
            "name": "시뮬레이션 정렬 PCF 최적화",
            "description": "rule_based.py와 동일한 로직을 사용하는 탄소발자국 최적화",
            "version": "2.0"
        },
        "objective": "minimize_carbon",
        "decision_vars": {
            "reduction_rates": {
                "tier1_re_rate": {"min": 0, "max": 1.0, "default": 0.3},  # 30% RE 적용률
                "tier2_re_rate": {"min": 0, "max": 1.0, "default": 0.5},  # 50% RE 적용률
                "tier3_re_rate": {"min": 0, "max": 1.0, "default": 0.7}   # 70% RE 적용률
            },
            "cathode": {
                "type": "B",
                "type_B_config": {
                    "emission_fixed": 10.0,
                    "recycle_range": [0.1, 0.5],
                    "low_carbon_range": [0.05, 0.3]
                }
            },
            "use_binary_variables": False
        },
        "constraints": {
            "target_carbon": 10.0,  # 목표 탄소발자국 (kgCO2eq)
            "max_cost": 50000,
            "feasibility_threshold": 0.8
        }
    }
    return config


def run_simulation_aligned_optimization_example():
    """시뮬레이션 정렬 최적화 실행 예제"""
    print("=" * 60)
    print("시뮬레이션 정렬 PCF 최적화 예제")
    print("=" * 60)
    
    # 1. 시뮬레이션 데이터 로드
    print("\n1️⃣ 시뮬레이션 데이터 로드")
    scenario_df, ref_formula_df, ref_proportions_df, original_df = load_sample_simulation_data()
    
    print(f"   📊 시나리오 데이터: {len(scenario_df)}개 자재")
    print(f"   📊 참조 공식 데이터: {len(ref_formula_df)}개 자재")  
    print(f"   📊 참조 비율 데이터: {len(ref_proportions_df)}개 자재")
    print(f"   📊 원본 데이터: {len(original_df)}개 자재")
    
    # 2. 최적화 설정 생성
    print("\n2️⃣ 최적화 설정 생성")
    config = create_simulation_aligned_optimization_config()
    
    # 3. OptimizationInput 객체 생성 (시뮬레이션 데이터 포함)
    print("\n3️⃣ 최적화 입력 객체 생성")
    opt_input = OptimizationInput(
        scenario_df=scenario_df,
        ref_formula_df=ref_formula_df, 
        ref_proportions_df=ref_proportions_df,
        original_df=original_df
    )
    
    # 설정 업데이트
    opt_input.update_config(config)
    
    # 4. SimulationAlignedObjective로 매칭 정보 분석
    print("\n4️⃣ 시뮬레이션 정렬 목적함수 분석")
    sim_objective = SimulationAlignedObjective(
        scenario_df=scenario_df,
        ref_formula_df=ref_formula_df,
        ref_proportions_df=ref_proportions_df,
        original_df=original_df
    )
    
    # 매칭 정보 요약
    material_summary = sim_objective.get_material_summary()
    print("   📋 자재별 매칭 요약:")
    for _, row in material_summary.iterrows():
        print(f"      • {row['자재명']} ({row['자재품목']}): {row['매칭_방식']} 방식")
    
    # 매칭 완성도 검증
    validation = sim_objective.validate_matching_completeness()
    print(f"\n   📊 매칭 완성도: {validation['matching_rate_percent']:.1f}%")
    print(f"      - Formula 매칭: {validation['formula_matches']}개")
    print(f"      - Proportions 매칭: {validation['proportions_matches']}개")
    print(f"      - 미매칭: {validation['unmatched_materials']}개")
    
    # 5. 최적화 실행
    print("\n5️⃣ 탄소발자국 최적화 실행")
    carbon_optimizer = CarbonMinimization(opt_input)
    
    try:
        # 솔버 자동 선택 및 실행
        results = carbon_optimizer.solve()
        
        if results['status'] == 'optimal':
            print("   ✅ 최적화 성공!")
            print(f"   📊 최적 탄소발자국: {results['objective_value']:.4f} kgCO2eq")
            print(f"   ⏱️  해결 시간: {results.get('solver_time', 'N/A'):.2f}초")
            
            # 최적 변수값 출력
            print("\n   🔧 최적 RE 적용률:")
            variables = results.get('variables', {})
            for var_name, value in variables.items():
                if 're_rate' in var_name:
                    print(f"      • {var_name}: {value:.1%}")
            
            # 시뮬레이션과 비교 분석
            print("\n6️⃣ 시뮬레이션과 결과 비교")
            baseline_pcf = scenario_df['배출량(kgCO2eq)'].sum()
            optimized_pcf = results['objective_value']
            reduction_rate = (baseline_pcf - optimized_pcf) / baseline_pcf * 100
            
            print(f"   📊 기준 PCF (시뮬레이션): {baseline_pcf:.4f} kgCO2eq")
            print(f"   📊 최적화 PCF: {optimized_pcf:.4f} kgCO2eq")
            print(f"   📉 감축률: {reduction_rate:.2f}%")
            
        else:
            print(f"   ❌ 최적화 실패: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ 최적화 오류: {e}")
        import traceback
        traceback.print_exc()


def compare_objective_functions():
    """기존 단순화 방식과 시뮬레이션 정렬 방식 비교"""
    print("\n" + "=" * 60)
    print("목적함수 방식 비교 분석")
    print("=" * 60)
    
    # 샘플 데이터 로드
    scenario_df, ref_formula_df, ref_proportions_df, original_df = load_sample_simulation_data()
    
    print("\n📊 비교 대상:")
    print("   1. 기존 단순화 방식: total_emission = base - reduction_effect - cathode_effect - recycle_effect")
    print("   2. 시뮬레이션 정렬 방식: 자재별 배출계수 수정 → PCF 계산")
    
    # 기준 PCF 계산 (시뮬레이션 방식)
    baseline_pcf = scenario_df['배출량(kgCO2eq)'].sum()
    print(f"\n📈 기준 PCF (시뮬레이션): {baseline_pcf:.4f} kgCO2eq")
    
    # 자재별 기여도 분석
    print("\n📋 자재별 PCF 기여도:")
    for _, row in scenario_df.iterrows():
        contribution = row['배출량(kgCO2eq)'] / baseline_pcf * 100
        print(f"   • {row['자재명']} ({row['자재품목']}): {row['배출량(kgCO2eq)']:.4f} kgCO2eq ({contribution:.1f}%)")
    
    # 매칭 방식별 감축 잠재력 분석
    sim_objective = SimulationAlignedObjective(scenario_df, ref_formula_df, ref_proportions_df, original_df)
    
    print("\n🔍 매칭 방식별 감축 잠재력:")
    for material_key, info in sim_objective.material_matching_info.items():
        print(f"   • {info['material_row']['자재명']}:")
        print(f"     - 매칭 방식: {info['matching_type']}")
        print(f"     - 원본 배출계수: {info['original_coeff']:.4f} kgCO2eq/kg")
        print(f"     - 제품 소요량: {info['product_amount']:.4f} kg")
        
        if info['matching_type'] == 'formula' and info['formula_match']:
            max_reduction = sum(info['formula_match'].values()) * info['product_amount']
            print(f"     - 최대 감축 가능량: {max_reduction:.4f} kgCO2eq")
        elif info['matching_type'] == 'proportions' and info['proportions_match']:
            max_reduction_rate = sum(info['proportions_match'].values()) / 100
            max_reduction = info['original_coeff'] * info['product_amount'] * max_reduction_rate
            print(f"     - 최대 감축률: {max_reduction_rate:.1%}")
            print(f"     - 최대 감축 가능량: {max_reduction:.4f} kgCO2eq")


if __name__ == "__main__":
    # 시뮬레이션 정렬 최적화 예제 실행
    run_simulation_aligned_optimization_example()
    
    # 목적함수 방식 비교
    compare_objective_functions()
    
    print("\n" + "=" * 60)
    print("✅ 시뮬레이션 정렬 최적화 예제 완료")
    print("=" * 60)
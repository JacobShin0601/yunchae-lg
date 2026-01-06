"""
최적화 시나리오 활용 예제

이 스크립트는 구현된 다양한 최적화 시나리오의 사용 방법을 보여줍니다.
"""

import os
import sys
import time
from pathlib import Path

# 상위 디렉토리를 경로에 추가하여 src 모듈 접근
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.optimization.scenario_base import ScenarioFactory
from src.optimization.carbon_minimization import CarbonMinimization
from src.optimization.cost_minimization import CostMinimization
from src.optimization.multi_objective import MultiObjective
from src.optimization.implementation_ease import ImplementationEase
from src.optimization.regional_optimization import RegionalOptimization


def print_separator(title):
    """구분선과 제목 출력"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def run_carbon_minimization_scenario():
    """탄소발자국 최소화 시나리오 실행"""
    print_separator("탄소발자국 최소화 시나리오")
    
    # 1. 시나리오 생성
    scenario = CarbonMinimization()
    
    # 2. 시나리오 실행
    print("시나리오 실행 중...")
    start_time = time.time()
    results = scenario.run_scenario()
    elapsed = time.time() - start_time
    
    # 3. 결과 출력
    if results.get('status') == 'optimal':
        print(f"\n✅ 최적화 성공! (소요 시간: {elapsed:.2f}초)")
        print(f"🎯 목적함수: {results.get('objective', {}).get('유형')}")
        print(f"📊 탄소발자국: {results.get('carbon_footprint')}")
        
        # 양극재 구성 결과
        if 'cathode_composition' in results:
            print("\n📋 양극재 구성:")
            for key, value in results['cathode_composition'].items():
                print(f"  • {key}: {value}")
        
        # 감축비율 결과
        if 'reduction_ratios' in results:
            print("\n📉 감축비율:")
            for tier, items in results['reduction_ratios'].items():
                print(f"  • {tier}:")
                for item, value in items.items():
                    print(f"    - {item}: {value}")
    else:
        print(f"\n❌ 최적화 실패: {results.get('message', '알 수 없는 오류')}")
    
    # 4. 결과 내보내기
    report_path = scenario.create_report()
    print(f"\n📄 보고서 생성: {report_path}")
    
    return results


def run_cost_minimization_scenario():
    """비용 최소화 시나리오 실행"""
    print_separator("비용 최소화 시나리오")
    
    # 1. 시나리오 생성
    scenario = CostMinimization()
    
    # 2. 시나리오 실행
    print("시나리오 실행 중...")
    start_time = time.time()
    results = scenario.run_scenario()
    elapsed = time.time() - start_time
    
    # 3. 결과 출력
    if results.get('status') == 'optimal':
        print(f"\n✅ 최적화 성공! (소요 시간: {elapsed:.2f}초)")
        print(f"🎯 목적함수: {results.get('objective', {}).get('유형')}")
        
        # 비용 분석
        total_cost = scenario.get_total_cost()
        print(f"💰 총 비용: {total_cost:.2f} 원")
        
        # 비용 항목별 분석 
        cost_breakdown = scenario.get_cost_breakdown()
        if cost_breakdown.get('status') == 'success':
            print("\n📋 비용 항목별 분석:")
            for name, value in cost_breakdown.get('breakdown', {}).items():
                print(f"  • {name}: {value:.2f} 원")
        
        # 탄소발자국 (제약조건)
        print(f"📊 탄소발자국: {results.get('carbon_footprint')}")
    else:
        print(f"\n❌ 최적화 실패: {results.get('message', '알 수 없는 오류')}")
    
    return results


def run_multi_objective_scenario():
    """다목적 최적화 시나리오 실행"""
    print_separator("다목적 최적화 시나리오")
    
    # 1. 시나리오 생성 (가중치 설정: 탄소 60%, 비용 40%)
    scenario = MultiObjective(carbon_weight=0.6, cost_weight=0.4)
    
    # 2. 시나리오 실행
    print("시나리오 실행 중...")
    start_time = time.time()
    results = scenario.run_scenario()
    elapsed = time.time() - start_time
    
    # 3. 결과 출력
    if results.get('status') == 'optimal':
        print(f"\n✅ 최적화 성공! (소요 시간: {elapsed:.2f}초)")
        print(f"🎯 목적함수: {results.get('objective', {}).get('유형')}")
        
        # 다목적 분석 결과
        if 'multi_objective_analysis' in results:
            analysis = results['multi_objective_analysis']
            print(f"\n📋 다목적 최적화 분석:")
            print(f"  • 탄소발자국: {analysis.get('carbon_footprint'):.4f}")
            print(f"  • 비용: {analysis.get('cost'):.2f}")
            print(f"  • 가중치: 탄소={analysis.get('weights', {}).get('carbon', 0):.2f}, 비용={analysis.get('weights', {}).get('cost', 0):.2f}")
        
        # 파레토 프론트 생성 (5개 점)
        print("\n📈 파레토 프론트 생성 중...")
        pareto = scenario.generate_pareto_front(num_points=5)
        
        if pareto.get('status') == 'success':
            print("  • 효율적 파레토 점:")
            for point in pareto.get('efficient_points', []):
                print(f"    - 가중치(탄소={point['carbon_weight']:.2f}, 비용={point['cost_weight']:.2f}): 탄소발자국={point['carbon_footprint']:.4f}, 비용={point['cost']:.2f}")
    else:
        print(f"\n❌ 최적화 실패: {results.get('message', '알 수 없는 오류')}")
    
    return results


def run_implementation_ease_scenario():
    """구현 용이성 최적화 시나리오 실행"""
    print_separator("구현 용이성 최적화 시나리오")
    
    # 1. 시나리오 생성 (탄소발자국 목표 설정)
    scenario = ImplementationEase(carbon_target=48.0)
    
    # 2. 시나리오 실행
    print("시나리오 실행 중...")
    start_time = time.time()
    results = scenario.run_scenario()
    elapsed = time.time() - start_time
    
    # 3. 결과 출력
    if results.get('status') == 'optimal':
        print(f"\n✅ 최적화 성공! (소요 시간: {elapsed:.2f}초)")
        print(f"🎯 목적함수: {results.get('objective', {}).get('유형')}")
        
        # 구현 분석 결과
        if 'implementation_analysis' in results:
            analysis = results['implementation_analysis']
            print(f"\n📋 구현 용이성 분석:")
            print(f"  • 활성화된 활동 수: {analysis.get('num_activities')}")
            print(f"  • 탄소발자국 목표: {analysis.get('carbon_target'):.4f}")
            print(f"  • 달성된 탄소발자국: {analysis.get('carbon_footprint'):.4f}")
            print(f"  • 구현 용이성 점수: {analysis.get('implementation_ease'):.1f}/100")
            
            # 활성화된 활동 목록
            print("\n📋 활성화된 활동:")
            for activity in analysis.get('active_activities', []):
                contribution = analysis.get('activity_contributions', {}).get(activity, 0)
                print(f"  • {activity}: {contribution:.2f}% 감축")
        
        # 구현 단계
        if 'implementation_steps' in results:
            print("\n📋 구현 단계 추천:")
            for step in results['implementation_steps']:
                print(f"  {step['step']}. {step['description']}")
    else:
        print(f"\n❌ 최적화 실패: {results.get('message', '알 수 없는 오류')}")
    
    return results


def run_regional_optimization_scenario():
    """지역별 최적화 시나리오 실행"""
    print_separator("지역별 최적화 시나리오")
    
    # 1. 시나리오 생성 (고려할 지역 지정)
    regions = ["한국", "중국", "일본", "미국", "독일"]
    scenario = RegionalOptimization(target_regions=regions)
    
    # 2. 시나리오 실행
    print("시나리오 실행 중...")
    start_time = time.time()
    results = scenario.run_scenario()
    elapsed = time.time() - start_time
    
    # 3. 결과 출력
    if results.get('status') == 'optimal':
        print(f"\n✅ 최적화 성공! (소요 시간: {elapsed:.2f}초)")
        print(f"🎯 목적함수: {results.get('objective', {}).get('유형')}")
        
        # 지역 분석 결과
        if 'regional_analysis' in results:
            analysis = results['regional_analysis']
            locations = analysis.get('optimal_locations', {})
            
            print(f"\n📋 최적 생산 위치:")
            print(f"  • CAM 생산지: {locations.get('CAM')}")
            print(f"  • pCAM 생산지: {locations.get('pCAM')}")
            
            # 전력 배출계수
            factors = analysis.get('electricity_factors', {})
            print(f"\n📋 전력 배출계수:")
            for facility, factor in factors.items():
                print(f"  • {facility}: {factor}")
            
            # 탄소발자국
            print(f"\n📊 탄소발자국: {analysis.get('carbon_footprint'):.4f}")
            
            # 물류 정보
            logistics = analysis.get('logistics', {})
            print(f"\n📦 물류 정보:")
            print(f"  • 운송 거리: {logistics.get('transport_distance', 0)} km")
            print(f"  • 운송 비용: {logistics.get('transport_cost', 0):.2f}")
            print(f"  • 운송 탄소배출량: {logistics.get('carbon_emissions_from_transport', 0):.4f}")
        
        # 위치 추천
        location_recommendations = scenario.generate_location_recommendations()
        print(f"\n📋 위치 추천:")
        print(f"  • 주 추천: {location_recommendations.get('primary', {}).get('description')}")
        
        print(f"\n📋 대안 추천:")
        for i, alt in enumerate(location_recommendations.get('alternatives', [])):
            print(f"  {i+1}. {alt.get('description')}")
    else:
        print(f"\n❌ 최적화 실패: {results.get('message', '알 수 없는 오류')}")
    
    return results


def run_all_scenarios():
    """모든 시나리오 순차 실행"""
    results = {}
    
    # 1. 탄소발자국 최소화
    results['carbon'] = run_carbon_minimization_scenario()
    
    # 2. 비용 최소화
    results['cost'] = run_cost_minimization_scenario()
    
    # 3. 다목적 최적화
    results['multi'] = run_multi_objective_scenario()
    
    # 4. 구현 용이성 최적화
    results['ease'] = run_implementation_ease_scenario()
    
    # 5. 지역별 최적화
    results['regional'] = run_regional_optimization_scenario()
    
    print_separator("모든 시나리오 실행 완료")
    print(f"\n📊 시나리오 결과 요약:")
    
    for scenario, result in results.items():
        status = "✅ 성공" if result.get('status') == 'optimal' else "❌ 실패"
        print(f"  • {scenario}: {status}")
        
        if result.get('status') == 'optimal':
            # 탄소발자국 추출
            carbon_footprint = "N/A"
            if 'carbon_footprint' in result:
                carbon_footprint = result['carbon_footprint']
            print(f"    - 탄소발자국: {carbon_footprint}")
    
    return results


def compare_scenarios():
    """시나리오 결과 비교 분석"""
    print_separator("시나리오 비교 분석")
    
    # 팩토리를 통해 시나리오 생성
    scenarios = {
        'carbon': ScenarioFactory.create_scenario('carbon_minimization'),
        'cost': ScenarioFactory.create_scenario('cost_minimization'),
        'multi': ScenarioFactory.create_scenario('multi_objective'),
        'ease': ScenarioFactory.create_scenario('implementation_ease'),
        'regional': ScenarioFactory.create_scenario('regional_optimization')
    }
    
    # 각 시나리오 실행
    print("시나리오 실행 중...")
    results = {}
    solve_times = {}
    
    for name, scenario in scenarios.items():
        print(f"  • {name} 시나리오 실행 중...")
        start_time = time.time()
        result = scenario.run_scenario()
        elapsed = time.time() - start_time
        
        results[name] = result
        solve_times[name] = elapsed
    
    # 결과 비교 및 출력
    print("\n📊 시나리오 비교:")
    print(f"{'시나리오':<15} {'상태':<8} {'탄소발자국':<15} {'계산 시간':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        status = "✅" if result.get('status') == 'optimal' else "❌"
        
        # 탄소발자국 추출
        carbon_footprint = "N/A"
        if result.get('carbon_footprint'):
            carbon_footprint = result['carbon_footprint']
        elif 'carbon_footprint' in result:
            carbon_footprint = result['carbon_footprint']
        
        print(f"{name:<15} {status:<8} {carbon_footprint:<15} {solve_times.get(name, 0):.2f}초")
    
    # 비교 차트 (생략 - 실제 구현에서는 matplotlib을 사용하여 그래프 생성 가능)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="최적화 시나리오 예제 실행")
    parser.add_argument('--scenario', '-s', choices=['carbon', 'cost', 'multi', 'ease', 'regional', 'all', 'compare'], 
                        default='all', help='실행할 시나리오 선택')
    
    args = parser.parse_args()
    
    if args.scenario == 'carbon':
        run_carbon_minimization_scenario()
    elif args.scenario == 'cost':
        run_cost_minimization_scenario()
    elif args.scenario == 'multi':
        run_multi_objective_scenario()
    elif args.scenario == 'ease':
        run_implementation_ease_scenario()
    elif args.scenario == 'regional':
        run_regional_optimization_scenario()
    elif args.scenario == 'compare':
        compare_scenarios()
    else:  # 'all'
        run_all_scenarios()
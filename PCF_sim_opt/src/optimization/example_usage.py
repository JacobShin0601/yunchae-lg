"""
PCF 최적화 프레임워크 사용 예제

이 스크립트는 최적화 구조를 어떻게 사용하는지 보여줍니다.
다양한 시나리오에 대한 예제를 포함합니다.
"""

import os
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, Any

# 현재 디렉토리를 기준으로 모듈 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 최적화 모듈 임포트
from src.optimization import (
    # 핵심 컴포넌트
    OptimizationController, CarbonMinimization,
    OptimizationConstants, OptimizationVariables, OptimizationInput,
    
    # 모듈화된 컴포넌트
    ConstraintManager, ObjectiveManager, ResultsProcessor, Validator,
    OptimizationScenario, ScenarioFactory,
    
    # 솔버 인터페이스
    SolverFactory, get_available_solvers
)


def example_basic_usage():
    """가장 기본적인 사용 예제"""
    print("\n=== 기본 사용 예제 ===")
    
    # 1. 컨트롤러 생성
    controller = OptimizationController()
    
    # 2. 설정 로드
    config_path = "src/optimization/config_opt.yaml"
    controller.load_config(config_path)
    
    # 3. 시나리오 적용
    controller.apply_scenario("balanced")
    
    # 4. 모델 구축
    controller.build_model()
    
    # 5. 최적화 실행
    results = controller.solve()
    
    # 6. 결과 출력
    print(f"최적화 상태: {results['status']}")
    if results['status'] == 'optimal':
        print(f"목적함수 값: {results['objective_value']:.4f}")
        print(f"사용 솔버: {results['solver']}")
        print(f"계산 시간: {results.get('solver_time', 'N/A')}")
        
        # 변수값 요약
        print("\n주요 결과:")
        formatted = controller.get_formatted_results(results)
        print(f"탄소발자국: {formatted['carbon_footprint']}")
        
        # 양극재 구성
        print("\n양극재 구성:")
        for key, value in formatted.get('cathode_composition', {}).items():
            print(f"  {key}: {value}")
    
    return results


def example_carbon_minimization():
    """탄소배출 최소화 시나리오 예제"""
    print("\n=== 탄소배출 최소화 시나리오 예제 ===")
    
    # 1. 설정 파일 경로
    config_path = "src/optimization/config_opt.yaml"
    
    # 2. 탄소배출 최소화 객체 생성
    scenario = CarbonMinimization(config_path)
    
    # 3. 시나리오 실행
    results = scenario.run_scenario()
    
    # 4. 결과 출력
    print("\n탄소배출 최소화 시나리오 결과:")
    print(f"상태: {results['status']}")
    print(f"탄소발자국: {results['carbon_footprint']}")
    
    # 감축비율 결과
    print("\n감축비율 결과:")
    for tier, items in results.get('reduction_ratios', {}).items():
        for item, value in items.items():
            print(f"  {tier} {item}: {value}")
    
    # 양극재 구성 결과
    print("\n양극재 구성:")
    for key, value in results.get('cathode_composition', {}).items():
        print(f"  {key}: {value}")
    
    # 5. 결과 파일 내보내기
    output_path = "carbon_minimization_results.json"
    scenario.export_results(output_path)
    print(f"\n결과가 {output_path}에 저장되었습니다.")
    
    # 6. 기준 대비 개선율 확인
    comparison = scenario.compare_to_baseline()
    if comparison['status'] == 'success':
        print("\n기준 대비 개선율:")
        print(f"  기준 배출량: {comparison['baseline']:.4f} kg CO2eq/kWh")
        print(f"  최적화 배출량: {comparison['optimized']:.4f} kg CO2eq/kWh")
        print(f"  절감량: {comparison['reduction']:.4f} kg CO2eq/kWh")
        print(f"  절감율: {comparison['reduction_percentage']}")
    
    # 7. 결과 시각화
    visualization_path = scenario.visualize_results("carbon_minimization_visualization.png")
    print(f"\n결과 시각화가 {visualization_path}에 저장되었습니다.")
    
    # 8. 보고서 생성
    report_path = scenario.create_report("carbon_minimization_report.md")
    print(f"\n결과 보고서가 {report_path}에 저장되었습니다.")
    
    return results


def example_custom_config():
    """커스텀 설정 예제"""
    print("\n=== 커스텀 설정 예제 ===")
    
    # 1. 컨트롤러 생성
    controller = OptimizationController()
    
    # 2. 기본 설정 로드
    config_path = "src/optimization/config_opt.yaml"
    controller.load_config(config_path)
    
    # 3. 커스텀 설정 생성
    custom_config = controller.create_custom_config(
        objective="minimize_cost",
        constraints={
            "target_carbon": 45.0,
            "max_cost": 80000
        },
        decision_vars={
            "location": "중국",
            "cathode": {
                "type": "B",
                "type_B_config": {
                    "recycle_range": [0.2, 0.6]
                }
            }
        }
    )
    
    # 4. 커스텀 설정 저장 (선택 사항)
    custom_config_path = "custom_config.yaml"
    controller.save_config(custom_config_path)
    print(f"커스텀 설정이 {custom_config_path}에 저장되었습니다.")
    
    # 5. 최적화 실행
    results = controller.solve()
    
    # 6. 결과 출력
    print(f"최적화 상태: {results['status']}")
    if results['status'] == 'optimal':
        print(f"목적함수 값: {results['objective_value']:.4f}")
        print(f"사용 솔버: {results['solver']}")
        
        # 변수값 요약
        print("\n주요 결과:")
        formatted = controller.get_formatted_results(results)
        print(f"탄소발자국: {formatted['carbon_footprint']}")
        
        # 양극재 구성
        print("\n양극재 구성:")
        for key, value in formatted.get('cathode_composition', {}).items():
            print(f"  {key}: {value}")
    
    return results


def example_modular_components():
    """모듈화된 컴포넌트 사용 예제"""
    print("\n=== 모듈화된 컴포넌트 사용 예제 ===")
    
    # 1. 입력 객체 생성 및 설정 로드
    opt_input = OptimizationInput()
    config_path = "src/optimization/config_opt.yaml"
    opt_input.load_config(config_path)
    
    # 2. 상수 객체 생성
    constants = OptimizationConstants()
    print("\n상수값 샘플:")
    print(f"  기본 배출량: {constants.get('base_emission')} kg CO2eq/kWh")
    print(f"  한국 전력배출계수: {constants.get_location_factor('한국')}")
    print(f"  재활용재 환경영향 (Ni): {constants.get_recycle_impact('Ni')}")
    
    # 3. 변수 객체 생성
    variables = OptimizationVariables()
    
    # 4. 모델 생성
    from pyomo.environ import ConcreteModel
    model = ConcreteModel()
    
    # 5. 변수 정의
    variables.define_variables(model)
    
    # 6. 제약조건 관리자 생성 및 제약조건 정의
    constraint_manager = ConstraintManager(opt_input, model)
    constraint_manager.register_standard_constraints()
    
    print("\n사용 가능한 제약조건:")
    available_constraints = constraint_manager.get_available_constraints()
    for constraint in available_constraints:
        print(f"  - {constraint}")
    
    # 7. 목적함수 관리자 생성 및 목적함수 정의
    objective_manager = ObjectiveManager(opt_input, model)
    objective_manager.register_standard_objectives()
    
    print("\n사용 가능한 목적함수:")
    available_objectives = objective_manager.get_available_objectives()
    for objective in available_objectives:
        print(f"  - {objective}")
    
    # 8. 탄소발자국 최소화 목적함수 적용
    objective_manager.define_objective_from_name("minimize_carbon")
    print(f"\n현재 설정된 목적함수: {objective_manager.get_current_objective()}")
    
    # 9. 제약조건 적용
    constraint_manager.apply_all_constraints()
    
    # 10. 솔버 선택
    solver_name = opt_input.get_solver_recommendation()
    print(f"\n추천 솔버: {solver_name}")
    
    # 11. 솔버 생성 및 최적화 실행
    solver = SolverFactory.create_solver(solver_name, opt_input, model)
    results = solver.solve()
    
    # 12. 결과 처리
    results_processor = ResultsProcessor(opt_input)
    formatted_results = results_processor.process_results(results)
    
    # 13. 결과 출력
    print("\n최적화 결과:")
    print(f"  상태: {formatted_results['status']}")
    if formatted_results['status'] == 'optimal':
        print(f"  탄소발자국: {formatted_results['carbon_footprint']}")
    
    # 14. 결과 유효성 검증
    validator = Validator(opt_input)
    is_valid, errors = validator.validate_results(results)
    
    print(f"\n결과 유효성: {'유효함' if is_valid else '유효하지 않음'}")
    if not is_valid:
        for error in errors:
            print(f"  - {error}")
    
    return formatted_results


def example_multi_solver_comparison():
    """여러 솔버 비교 예제"""
    print("\n=== 여러 솔버 비교 예제 ===")
    
    # 1. 컨트롤러 생성
    controller = OptimizationController()
    
    # 2. 설정 로드
    config_path = "src/optimization/config_opt.yaml"
    controller.load_config(config_path)
    
    # 3. 여러 솔버로 최적화 실행
    comparison_results = controller.run_multi_solver_comparison()
    
    # 4. 결과 출력
    print("\n솔버별 결과 비교:")
    print(f"{'솔버':<10} {'상태':<10} {'목적함수값':<15} {'계산 시간':<10}")
    print("-" * 50)
    
    for solver, results in comparison_results.items():
        status = results.get('status', 'N/A')
        obj_value = f"{results.get('objective_value', 'N/A'):.4f}" if results.get('objective_value') else 'N/A'
        solve_time = f"{results.get('solver_time', 'N/A'):.2f}s" if results.get('solver_time') else 'N/A'
        
        print(f"{solver:<10} {status:<10} {obj_value:<15} {solve_time:<10}")
    
    # 5. 결과 비교 시각화
    results_processor = ResultsProcessor()
    
    # 각 솔버 결과를 처리
    processed_results = {}
    for solver, result in comparison_results.items():
        if result.get('status') == 'optimal':
            processed = results_processor.process_results(result)
            processed_results[solver] = processed
    
    # 결과 비교
    comparison = results_processor.compare_results(
        list(processed_results.values()),
        list(processed_results.keys())
    )
    
    # 비교 결과 시각화
    visualization_path = results_processor.visualize_comparison(
        comparison, "solver_comparison_visualization.png"
    )
    print(f"\n솔버 비교 시각화가 {visualization_path}에 저장되었습니다.")
    
    return comparison_results


def example_scenario_factory():
    """시나리오 팩토리 사용 예제"""
    print("\n=== 시나리오 팩토리 사용 예제 ===")
    
    # 1. 사용 가능한 시나리오 조회
    available_scenarios = ScenarioFactory.get_available_scenario_types()
    scenario_info = ScenarioFactory.get_scenario_info()
    
    print("\n사용 가능한 시나리오:")
    for scenario_type in available_scenarios:
        info = scenario_info.get(scenario_type, {})
        print(f"  - {scenario_type}: {info.get('description', '설명 없음')}")
    
    # 2. 시나리오 생성
    scenario = ScenarioFactory.create_scenario('carbon_minimization')
    
    # 3. 시나리오 구성 내보내기
    config_path = scenario.export_scenario_config("carbon_minimization_scenario_config.yaml")
    print(f"\n시나리오 설정이 {config_path}에 저장되었습니다.")
    
    # 4. 시나리오 실행
    results = scenario.run_scenario()
    
    # 5. 결과 출력
    print(f"\n시나리오 '{scenario.get_name()}' 실행 결과:")
    print(f"  상태: {results['status']}")
    if results['status'] == 'optimal':
        print(f"  탄소발자국: {results['carbon_footprint']}")
    
    return results


if __name__ == "__main__":
    # 모든 예제 실행
    print("PCF 최적화 프레임워크 사용 예제\n")
    
    # 예제 선택 (모두 실행하거나 특정 예제만 실행)
    run_all = False
    
    if run_all:
        example_basic_usage()
        example_carbon_minimization()
        example_custom_config()
        example_modular_components()
        example_multi_solver_comparison()
        example_scenario_factory()
    else:
        # 특정 예제만 실행
        example_modular_components()
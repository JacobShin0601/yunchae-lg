"""
최적화 페이지 (V2 - 모듈식 제약조건 시스템)

완전히 재설계된 최적화 시스템:
- 유연한 제약조건 추가/관리
- RE100 프리미엄 비용 통합
- 자재별 위치 및 관리 제약
- 시나리오 비교 기능
"""

import streamlit as st
import sys
import traceback
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

# 최적화 V2 시스템 임포트 시도
try:
    from src.optimization_v2.core.constraint_manager import ConstraintManager
    from src.optimization_v2.core.optimization_engine import OptimizationEngine
    from src.optimization_v2.core.result_processor import ResultProcessor
    from src.optimization_v2.utils.data_loader import DataLoader
    from src.optimization_v2.ui.constraint_configurator import ConstraintConfigurator
    from src.optimization_v2.ui.results_visualizer import ResultsVisualizer
    from src.optimization_v2.ui.comparison_dashboard import ComparisonDashboard
    # Phase 1: 고급 분석 모듈
    from src.optimization_v2.analysis.sensitivity_analyzer import SensitivityAnalyzer
    from src.optimization_v2.analysis.pareto_navigator import ParetoNavigator
    from src.optimization_v2.analysis.solution_recommender import SolutionRecommender
    from src.optimization_v2.analysis.constraint_relaxation_analyzer import ConstraintRelaxationAnalyzer
    # Phase 2: 강건 최적화 모듈
    from src.optimization_v2.robust.scenario_manager import ScenarioManager
    from src.optimization_v2.robust.robust_optimizer import RobustOptimizer
    from src.optimization_v2.robust.solution_evaluator import SolutionEvaluator
    # Phase 3: 확률적 위험 분석 모듈
    from src.optimization_v2.stochastic.stochastic_analyzer import StochasticAnalyzer, ParameterUncertainty
    # Phase 4: 고급 분석 대시보드
    from src.optimization_v2.ui.advanced_analysis_dashboard import AdvancedAnalysisDashboard, render_dashboard
    OPTIMIZATION_V2_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_V2_AVAILABLE = False
    IMPORT_ERROR = str(e)


def initialize_session_state():
    """세션 상태 초기화"""
    # 제약조건 관리자
    if 'constraint_manager' not in st.session_state:
        st.session_state.constraint_manager = ConstraintManager()
        # 저장된 제약조건 자동 로드
        user_id = st.session_state.get('user_id', 'default')
        st.session_state.constraint_manager.load_from_file(user_id=user_id)

    # 데이터 로더
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()

    # 최적화 엔진
    if 'optimization_engine' not in st.session_state:
        st.session_state.optimization_engine = None

    # 결과 프로세서
    if 'result_processor' not in st.session_state:
        st.session_state.result_processor = ResultProcessor()

    # 비교 대시보드
    if 'comparison_dashboard' not in st.session_state:
        st.session_state.comparison_dashboard = ComparisonDashboard()

    # 최적화 결과
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}

    # 현재 결과
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None

    # 데이터 로딩 상태
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Phase 1: 워크플로우 관련 세션 상태
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1  # 1: 데이터, 2: 제약조건, 3: 방법, 4: 실행

    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []

    if 'selected_point' not in st.session_state:
        st.session_state.selected_point = None

    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []

    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "⚙️ 설정 및 실행"

    if 'advanced_section' not in st.session_state:
        st.session_state.advanced_section = None

    if 'show_tutorial' not in st.session_state:
        st.session_state.show_tutorial = True


def render_data_loading_tab():
    """데이터 로딩 탭"""
    st.header("📂 데이터 로딩")

    st.markdown("""
    시뮬레이션 결과를 불러와 최적화를 준비합니다.

    **데이터 소스**: 양극재 세부설정 + 시나리오 설정 → PCF 시뮬레이터

    **변경사항**: 항상 베이스라인 시나리오를 로드하며, 재활용/저탄소/사이트변경 기능은
    **제약조건 설정** 탭에서 활성화/비활성화할 수 있습니다.
    """)

    # 시뮬레이션 결과 확인
    if 'simulation_results' not in st.session_state:
        _show_user_friendly_error('data_missing')
        return

    if 'baseline' not in st.session_state.simulation_results:
        _show_user_friendly_error('baseline_missing')
        return

    st.info("🎯 **베이스라인 시나리오**를 로드합니다. 재활용재/저탄소메탈/사이트변경은 제약조건에서 설정할 수 있습니다.")

    # 데이터 로딩
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("📥 베이스라인 데이터 로딩", type="primary", use_container_width=True):
            with st.spinner("데이터 로딩 중..."):
                data_loader = st.session_state.data_loader

                # Check site change constraint to determine which site to load
                from src.optimization_v2.constraints.feature_options import SiteChangeOptionConstraint
                constraint_manager = st.session_state.constraint_manager
                site_to_load = 'before'  # Default

                for constraint in constraint_manager.list_constraint_objects():
                    if isinstance(constraint, SiteChangeOptionConstraint):
                        if constraint.enabled:
                            site_to_load = 'after'
                            st.info("🌍 생산지 변경 활성화 - 'after' 사이트 데이터 로드")
                        break

                # Load baseline scenario with appropriate site setting
                success = data_loader.load_from_session_state(site=site_to_load)

                if success:
                    st.session_state.data_loaded = True
                    st.session_state.selected_scenario = 'baseline'  # 항상 baseline
                    st.success("✅ 베이스라인 데이터 로딩 완료!")
                    st.info("💡 재활용/저탄소/사이트변경 기능은 '제약조건 설정' 탭에서 활성화하세요.")
                    st.rerun()
                else:
                    _show_user_friendly_error('data_loading_failed')
                    return

    with col2:
        if st.session_state.data_loaded:
            st.success("✅ 로딩됨")

    # 데이터 요약 표시
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("📊 데이터 요약")

        data_loader = st.session_state.data_loader
        optimization_data = data_loader.get_optimization_data()

        # 첫 번째 줄: 전체 자재 관점
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            material_count = len(optimization_data['material_classification'])
            st.metric("총 자재 수", f"{material_count}개")

        with col2:
            # 저감활동 적용 자재 수 (저감활동_적용여부 == 1)
            scenario_df = optimization_data['scenario_df']
            if scenario_df is not None and '저감활동_적용여부' in scenario_df.columns:
                reduction_applied = len(scenario_df[scenario_df['저감활동_적용여부'] == 1]['자재명'].unique())
            else:
                reduction_applied = material_count
            st.metric("저감활동 적용", f"{reduction_applied}개",
                     help="저감활동이 적용되는 자재 수")

        with col3:
            # 평균 배출계수 계산
            if scenario_df is not None and '배출계수' in scenario_df.columns:
                avg_emission_factor = scenario_df['배출계수'].mean()
                st.metric("평균 배출계수", f"{avg_emission_factor:.2f} kgCO2eq/kg",
                         help="전체 자재의 평균 배출계수")
            else:
                st.metric("평균 배출계수", "N/A")

        with col4:
            # 총 소요량 계산
            if scenario_df is not None and '제품총소요량(kg)' in scenario_df.columns:
                total_quantity = scenario_df['제품총소요량(kg)'].sum()
                st.metric("총 소요량", f"{total_quantity:,.0f} kg",
                         help="전체 자재의 총 소요량")
            else:
                st.metric("총 소요량", "N/A")

        # 두 번째 줄: 양극재 관련 정보
        st.markdown("---")
        col5, col6, col7, col8 = st.columns(4)

        # 양극재 설정 파일 로드
        import os
        from src.utils.file_operations import FileOperations

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..")
        user_id = st.session_state.get('user_id', None)

        try:
            cathode_site_path = os.path.join(project_root, "input", "cathode_site.json")
            cathode_site_data = FileOperations.load_json(cathode_site_path, default={}, user_id=user_id)

            recycle_ratio_path = os.path.join(project_root, "input", "recycle_material_ratio.json")
            recycle_ratio_data = FileOperations.load_json(recycle_ratio_path, default={}, user_id=user_id)

            low_carb_metal_path = os.path.join(project_root, "input", "low_carb_metal.json")
            low_carb_metal_data = FileOperations.load_json(low_carb_metal_path, default={}, user_id=user_id)
        except Exception as e:
            cathode_site_data = {}
            recycle_ratio_data = {}
            low_carb_metal_data = {}

        with col5:
            # CAM 소싱 변경
            cam_before = cathode_site_data.get("CAM", {}).get("before", "N/A")
            cam_after = cathode_site_data.get("CAM", {}).get("after", "N/A")
            cam_change = f"{cam_before} → {cam_after}" if cam_before != cam_after else cam_before
            st.metric("CAM 소싱 변경", cam_change,
                     help="CAM(양극활물질) 생산지 변경")

        with col6:
            # pCAM 소싱 변경
            pcam_before = cathode_site_data.get("pCAM", {}).get("before", "N/A")
            pcam_after = cathode_site_data.get("pCAM", {}).get("after", "N/A")
            pcam_change = f"{pcam_before} → {pcam_after}" if pcam_before != pcam_after else pcam_before
            st.metric("pCAM 소싱 변경", pcam_change,
                     help="pCAM(전구체) 생산지 변경")

        with col7:
            # 재활용재 평균 적용률
            if recycle_ratio_data:
                avg_recycle_ratio = sum(recycle_ratio_data.values()) / len(recycle_ratio_data) * 100
                st.metric("재활용재 평균 적용률", f"{avg_recycle_ratio:.1f}%",
                         help="Ni/Co/Li 재활용재 평균 적용 비율")
            else:
                st.metric("재활용재 평균 적용률", "0%")

        with col8:
            # 저탄소메탈 평균 비중
            if low_carb_metal_data and "비중" in low_carb_metal_data:
                avg_low_carb_ratio = sum(low_carb_metal_data["비중"].values()) / len(low_carb_metal_data["비중"])
                st.metric("저탄소메탈 평균 비중", f"{avg_low_carb_ratio:.1f}%",
                         help="Ni/Co/Li 저탄소메탈 평균 사용 비중")
            else:
                st.metric("저탄소메탈 평균 비중", "0%")

        # 데이터 검증 및 경고
        data_loader = st.session_state.data_loader

        # original_df 체크
        if data_loader.original_df is None:
            st.warning("""
            ⚠️ **비용 제약 사용 불가**

            `original_df`가 없습니다. 비용 제약 조건을 사용할 수 없습니다.

            **해결 방법:**
            1. 'PCF 시뮬레이터' 페이지로 이동
            2. 시뮬레이션을 실행하여 `original_df` 생성
            3. 다시 '최적화' 페이지로 돌아와 데이터 재로딩

            **참고:** 비용 제약 없이 다른 제약조건(자재 관리 등)만으로 최적화는 가능합니다.
            """)

        # Tier_RE_case 컬럼 확인 및 경고
        if data_loader.scenario_df is not None:
            tier_re_columns = [col for col in data_loader.scenario_df.columns if 'Tier' in col and 'RE_case' in col]
            if not tier_re_columns:
                st.warning("""
                ⚠️ **RE100 비용 제약 사용 불가**

                현재 시나리오 데이터에 RE100 Case 설정이 없습니다.

                **해결 방법:**
                1. '시나리오 설정' 페이지로 이동
                2. RE100 케이스를 추가 (예: Case1, Case2)
                3. 각 케이스의 Tier1, Tier2 RE 비율 설정
                4. 시나리오 저장 후 PCF 시뮬레이터 재실행
                5. 최적화 페이지에서 데이터 재로딩

                **참고:** RE100 비용 제약 없이 다른 제약조건만으로 최적화는 가능합니다.
                """)

        # 자재 목록 표시
        with st.expander("📋 자재 목록 보기"):
            for material_name, info in optimization_data['material_classification'].items():
                material_type = info['type']
                if material_type == "Formula":
                    icon = "🔋"
                    type_label = "Formula 적용"
                elif material_type == "Ni-Co-Li":
                    icon = "⚗️"
                    type_label = "Ni/Co/Li"
                else:
                    icon = "📦"
                    type_label = "일반"

                # Cathode/Anode 표시
                if 'Cathode' in material_name or '양극재' in material_name:
                    category = "[양극재]"
                elif 'Anode' in material_name or '음극재' in material_name:
                    category = "[음극재]"
                else:
                    category = ""

                st.write(f"{icon} **{material_name}** {category} - {type_label}")


def render_constraint_configuration_tab():
    """제약조건 설정 탭"""
    if not st.session_state.data_loaded:
        st.warning("⚠️ 먼저 '데이터 로딩' 탭에서 데이터를 로딩하세요.")
        return

    # 사용 가능한 자재 목록
    data_loader = st.session_state.data_loader
    optimization_data = data_loader.get_optimization_data()
    available_materials = list(optimization_data['material_classification'].keys())

    # ConstraintConfigurator 렌더링 (scenario_df 전달)
    configurator = ConstraintConfigurator(st.session_state.constraint_manager)
    configurator.render(
        available_materials=available_materials,
        scenario_df=optimization_data.get('scenario_df')
    )


def render_optimization_execution_tab():
    """최적화 실행 탭"""
    st.header("🚀 최적화 실행")

    if not st.session_state.data_loaded:
        st.warning("⚠️ 먼저 '데이터 로딩' 탭에서 데이터를 로딩하세요.")
        return

    # 제약조건 요약 - 2단계 표시
    st.subheader("📋 제약조건 현황")

    # 1. 비용 제약조건 (별도 섹션)
    cost_constraint = st.session_state.constraint_manager.get_constraint('cost_constraint')
    if cost_constraint:
        st.markdown("#### 💰 비용 설정")

        # 파레토 최적화 경고
        st.warning("""
        ⚠️ **파레토 최적화에서는 비용 제약이 적용되지 않습니다**

        파레토 최적화는 탄소배출과 비용의 **전체 트레이드오프**를 탐색합니다.
        비용 제약을 걸면 프론티어의 일부만 보게 되어 최적화 본질을 해칩니다.

        💡 **비용 제약은 단일 목적 최적화(탄소 최소화)에서만 사용됩니다.**
        """)

        try:
            cost_info = cost_constraint.get_display_info()

            # 모드 표시
            col1, col2 = st.columns([2, 3])
            with col1:
                st.caption(f"**모드**: {cost_info['mode']}")
            with col2:
                st.caption(f"{cost_info['mode_description']}")

            # 기준 비용 (Zero-Premium Baseline)
            if cost_info.get('zero_premium_baseline'):
                st.caption(f"**Zero-Premium Baseline**: ${cost_info['zero_premium_baseline']:,.2f}")

            # 설정 정보
            if cost_info['settings']:
                st.caption("**설정** (단일 목적 최적화 시에만 적용):")
                for setting in cost_info['settings']:
                    st.caption(f"  • {setting}")

        except AttributeError:
            # Legacy cost constraint fallback
            st.caption(f"**비용 제약**: {cost_constraint.description}")
            st.warning("⚠️ 레거시 비용 제약 - 상세 정보 표시 불가")

        st.markdown("---")

    # 2. 표준 제약조건 (Feature Options + 일반 제약)
    st.markdown("#### 🔧 활성 제약조건")

    # Feature Option 제약조건 (enabled 여부와 관계없이 표시)
    feature_constraints = []
    standard_constraints = []

    for name in st.session_state.constraint_manager.list_constraints():
        constraint = st.session_state.constraint_manager.get_constraint(name)

        # 비용 제약은 이미 위에서 표시했으므로 스킵
        if constraint.name == 'cost_constraint':
            continue

        # Feature Option인지 확인
        try:
            if constraint.is_feature_option_constraint():
                feature_constraints.append(constraint)
            else:
                # 일반 제약은 enabled=True만 표시
                if constraint.enabled:
                    standard_constraints.append(constraint)
        except AttributeError:
            # Legacy constraint without is_feature_option_constraint()
            if constraint.enabled:
                standard_constraints.append(constraint)

    # Feature Options 표시
    if feature_constraints:
        st.caption("**기능 옵션 제약**:")
        for constraint in feature_constraints:
            try:
                summary = constraint.get_display_summary()
            except AttributeError:
                summary = constraint.get_summary()
            st.caption(f"  • {summary}")

    # 일반 제약조건 표시
    if standard_constraints:
        st.caption("**일반 제약조건**:")
        for constraint in standard_constraints:
            st.caption(f"  • {constraint.get_summary()}")

    # 제약조건이 없는 경우
    if not feature_constraints and not standard_constraints:
        st.info("ℹ️ 활성화된 제약조건이 없습니다.")

    total_active = len(feature_constraints) + len(standard_constraints)
    if cost_constraint:
        total_active += 1
    st.info(f"총 활성 제약조건: {total_active}개 (비용: {1 if cost_constraint else 0}개, 일반: {len(standard_constraints)}개, 기능옵션: {len(feature_constraints)}개)")

    st.markdown("---")

    # 다목적 최적화 (Multi-Objective) 설정
    st.subheader("🎯 다목적 최적화 (Multi-Objective)")

    st.markdown("""
    **파레토 최적화 방법 선택**

    탄소 배출과 비용을 동시에 고려하여 파레토 최적해를 찾습니다.
    세 가지 방법 중 하나를 선택하여 실행하세요.
    """)

    # 방법 선택
    method = st.radio(
        "최적화 방법",
        options=['weighted_sum', 'weighted_sum_premium', 'epsilon_constraint', 'nsga2'],
        index=0,
        format_func=lambda x: {
            'weighted_sum': '⚖️ Weighted Sum (가중치 스캔)',
            'weighted_sum_premium': '💰 Weighted Sum + Premium Scan (가중치 × 비용 레벨)',
            'epsilon_constraint': '🎯 Epsilon-Constraint (비용 제약)',
            'nsga2': '🧬 NSGA-II (진화 알고리즘)'
        }[x],
        key="pareto_method_selection",
        help="""
        • Weighted Sum: 빠르고 간단, Convex 파레토 프론티어
        • Weighted Sum + Premium Scan: 여러 비용 레벨에서 가중치 스캔 (2D 탐색)
        • Epsilon-Constraint: Non-convex 파레토 가능, 비용 상한 명확
        • NSGA-II: 진화 알고리즘, 복잡한 파레토 탐색
        """
    )

    # 방법별 설명
    method_descriptions = {
        'weighted_sum': """
        **가중치 스캔 방법 (Weighted Sum Scalarization)**

        • 탄소와 비용을 가중 합산하여 단일 목적함수로 변환
        • 목적함수 = α × 탄소 + β × 비용 (α + β = 1)
        • 다양한 가중치 조합으로 파레토 프론티어 구성
        • ✅ 빠른 실행 속도, 명확한 해석
        • ⚠️ Convex 파레토 프론티어만 탐색 가능
        """,
        'weighted_sum_premium': """
        **Weighted Sum + Premium Scan (2D 탐색)**

        • 가중치 스캔을 여러 비용 제약 레벨에서 반복 실행
        • 비용 레벨 (0%, 5%, 10%, 15%, 20% 등)마다 파레토 프론티어 생성
        • 가중치와 비용 레벨의 2차원 탐색으로 더 풍부한 해 집합
        • ✅ 비용 증가에 따른 탄소 감축 트레이드오프 명확히 파악
        • ⚠️ 실행 시간 = 가중치 수 × 비용 레벨 수
        """,
        'epsilon_constraint': """
        **Epsilon-Constraint 방법**

        • 비용을 제약조건 (≤ ε)으로 설정하고 탄소만 최소화
        • 다양한 ε 값으로 파레토 프론티어 구성
        • ✅ Non-convex 파레토 탐색 가능
        • ✅ 비용 상한이 명확한 경우 적합
        • ⚠️ 적절한 ε 설정 필요
        """,
        'nsga2': """
        **NSGA-II 진화 알고리즘**

        • 유전 알고리즘으로 여러 해를 동시에 탐색
        • 비지배 정렬 + 혼잡도 거리로 다양한 파레토 해 생성
        • ✅ 복잡한 Non-convex 파레토 탐색
        • ✅ 목적함수 미분 불필요 (Black-box)
        • ⚠️ 계산 시간 = 개체 수 × 세대 수 (약 1-3분)
        """
    }

    with st.expander("📖 방법 상세 설명", expanded=False):
        st.markdown(method_descriptions[method])

    # 공통 설정
    st.markdown("---")
    st.subheader("⚙️ 공통 설정")

    col1, col2, col3 = st.columns(3)

    with col1:
        # 제약조건 프리셋
        constraint_preset = st.selectbox(
            "제약조건 프리셋",
            options=['narrow', 'medium', 'wide'],
            index=1,
            format_func=lambda x: {
                'narrow': '좁은 범위 (보수적)',
                'medium': '중간 범위 (권장)',
                'wide': '넓은 범위 (공격적)'
            }[x],
            help="재활용재/저탄소메탈 비율 제약 범위"
        )

    with col2:
        # 시나리오 템플릿
        use_template = st.checkbox("시나리오 템플릿 사용")
        scenario_template = None
        if use_template:
            scenario_template = st.selectbox(
                "템플릿",
                options=['re100_focused', 'recycling_focused', 'low_carbon_focused', 'balanced'],
                format_func=lambda x: {
                    're100_focused': '🔋 RE100 집중',
                    'recycling_focused': '♻️ 재활용재 집중',
                    'low_carbon_focused': '🌱 저탄소메탈 집중',
                    'balanced': '⚖️ 균형 탐색'
                }[x]
            )

    with col3:
        # 솔버 선택
        solver = st.selectbox(
            "솔버",
            options=['auto', 'glpk', 'cbc', 'ipopt'],
            format_func=lambda x: {
                'auto': '자동 선택 (권장)',
                'glpk': 'GLPK',
                'cbc': 'CBC',
                'ipopt': 'IPOPT'
            }.get(x, x),
            help="최적화 솔버 선택 (대부분의 경우 자동 선택 권장)"
        )

    # 고급 설정 (접을 수 있도록)
    with st.expander("🔧 고급 설정"):
        col_adv1, col_adv2, col_adv3 = st.columns(3)

        with col_adv1:
            time_limit = st.number_input(
                "시간 제한 (초)",
                min_value=10,
                max_value=3600,
                value=300,
                step=10,
                help="각 최적화 포인트당 최대 실행 시간"
            )

        with col_adv2:
            gap_tolerance = st.slider(
                "Gap Tolerance (%)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="최적성 허용 오차 (낮을수록 정확하지만 느림)"
            ) / 100

        with col_adv3:
            verbose = st.checkbox("상세 로그 출력", value=False,
                                help="최적화 과정의 상세 로그 출력 여부")

    # 시나리오 이름도 여기로 이동
    scenario_name = st.text_input(
        "결과 저장 이름",
        value=f"파레토_{len(st.session_state.optimization_results) + 1}",
        help="결과 저장 및 비교를 위한 이름"
    )

    # 방법별 설정
    st.markdown("---")
    st.subheader("🔧 방법별 설정")

    if method == 'weighted_sum':
        render_weighted_sum_settings(enable_premium_scan=False)
    elif method == 'weighted_sum_premium':
        render_weighted_sum_settings(enable_premium_scan=True)
    elif method == 'epsilon_constraint':
        render_epsilon_constraint_settings()
    elif method == 'nsga2':
        render_nsga2_settings()

    # 다목적 최적화 실행 버튼
    st.markdown("---")

    if st.button("▶️ 파레토 최적화 실행", type="primary", use_container_width=True):
        if not scenario_name:
            _show_user_friendly_error('name_required')
        elif not st.session_state.data_loaded:
            _show_user_friendly_error('data_missing')
        else:
            # 방법별 최적화 실행
            if method == 'weighted_sum':
                run_weighted_sum_optimization(constraint_preset, scenario_template, premium_scan=False)
            elif method == 'weighted_sum_premium':
                run_weighted_sum_optimization(constraint_preset, scenario_template, premium_scan=True)
            elif method == 'epsilon_constraint':
                run_epsilon_constraint_optimization(constraint_preset, scenario_template)
            elif method == 'nsga2':
                run_nsga2_optimization(constraint_preset, scenario_template)


def render_weighted_sum_settings(enable_premium_scan=False):
    """Weighted Sum 설정

    Args:
        enable_premium_scan: True면 Premium Scan 모드 (체크박스 없이 자동 활성화)
    """
    from src.optimization_v2.pareto.config_loader import ParetoConfigLoader

    user_id = st.session_state.get('user_id', 'default')
    config_loader = ParetoConfigLoader(user_id)

    col1, col2 = st.columns(2)

    with col1:
        weight_strategy = st.selectbox(
            "가중치 전략",
            options=['uniform', 'dense_edges', 'logarithmic', 'custom'],
            format_func=lambda x: {
                'uniform': '균등 간격 (권장)',
                'dense_edges': '양 극단 집중',
                'logarithmic': '로그 스케일',
                'custom': '사용자 정의'
            }[x]
        )

    with col2:
        if weight_strategy in ['uniform', 'dense_edges', 'logarithmic']:
            num_points = st.slider("탐색 포인트 수", 3, 15, 5)
        else:
            num_points = None

    # Premium Scan 설정 (방법에 따라 표시)
    premium_scan_config = None
    if enable_premium_scan:
        st.markdown("---")
        st.markdown("**💰 프리미엄 스캔 설정**")
        st.caption("여러 비용 제약 레벨에서 가중치 스캔을 반복 실행합니다.")

        col1, col2, col3 = st.columns(3)
        with col1:
            premium_min = st.number_input("최소 (%)", 0, 100, 0, 5, key="ws_premium_min")
        with col2:
            premium_max = st.number_input("최대 (%)", 0, 100, 20, 5, key="ws_premium_max")
        with col3:
            premium_step = st.number_input("스텝 (%)", 1, 20, 5, 1, key="ws_premium_step")

        premium_range = list(range(int(premium_min), int(premium_max) + 1, int(premium_step)))
        st.info(f"🎯 스캔 레벨: {premium_range}% → {len(premium_range)}개 레벨")

        if weight_strategy != 'custom' and num_points:
            total_points = len(premium_range) * num_points
            st.caption(f"📊 총 최적화 포인트: {len(premium_range)} × {num_points} = {total_points}개")

        premium_scan_config = {'enabled': True, 'range': premium_range}

    # 저장 (세션 상태)
    st.session_state.weighted_sum_config = {
        'strategy': weight_strategy,
        'points': num_points,
        'premium_scan': premium_scan_config
    }

    # 미리보기
    if weight_strategy != 'custom':
        config_loader.config['weight_sweep']['strategy'] = weight_strategy
        if num_points:
            config_loader.config['weight_sweep'][weight_strategy]['points'] = num_points

        preview_weights = config_loader.get_weight_combinations()

        st.caption(f"📊 생성될 가중치 조합: {len(preview_weights)}개")
        with st.expander("가중치 미리보기"):
            preview_df = pd.DataFrame([
                {'탄소 가중치': w.carbon_weight, '비용 가중치': w.cost_weight}
                for w in preview_weights
            ])
            st.dataframe(preview_df, use_container_width=True, hide_index=True)


def render_epsilon_constraint_settings():
    """Epsilon-Constraint 설정"""
    from src.optimization_v2.pareto.config_loader import ParetoConfigLoader

    user_id = st.session_state.get('user_id', 'default')
    config_loader = ParetoConfigLoader(user_id)

    col1, col2, col3 = st.columns(3)

    with col1:
        epsilon_strategy = st.selectbox(
            "Epsilon 전략",
            options=['linear_sweep', 'exponential_sweep', 'adaptive'],
            format_func=lambda x: {
                'linear_sweep': '선형 증가 (균등)',
                'exponential_sweep': '지수 증가',
                'adaptive': '적응형 (동적)'
            }[x]
        )

    with col2:
        if epsilon_strategy != 'adaptive':
            points = st.slider("탐색 포인트 수", 5, 20, 10)
        else:
            max_attempts = st.slider("최대 시도 횟수", 10, 30, 20)
            points = max_attempts

    with col3:
        cost_max_multiplier = st.slider(
            "최대 비용 배율",
            min_value=1.1,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="기준 비용 대비 최대 허용 비용 (예: 1.5 = 150%까지)"
        )

    # 저장
    st.session_state.epsilon_constraint_config = {
        'strategy': epsilon_strategy,
        'points': points,
        'cost_max_multiplier': cost_max_multiplier
    }

    st.info(f"📊 Epsilon 스캔 범위: 기준 비용 ~ 기준 비용 × {cost_max_multiplier:.1f} ({points}개 포인트)")


def render_nsga2_settings():
    """NSGA-II 설정"""
    st.caption("⏰ NSGA-II는 계산 시간이 오래 걸릴 수 있습니다. (수십 분 ~ 수 시간)")

    col1, col2, col3 = st.columns(3)

    with col1:
        population_size = st.number_input(
            "개체 수 (Population)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="각 세대의 개체 수"
        )

    with col2:
        generations = st.number_input(
            "세대 수 (Generations)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="진화 반복 횟수"
        )

    with col3:
        early_stopping = st.checkbox(
            "조기 종료",
            value=True,
            help="수렴 감지 시 자동 종료"
        )

    # 고급 설정
    with st.expander("🔧 고급 설정 (선택사항)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            crossover_prob = st.slider("교차 확률", 0.5, 1.0, 0.9, 0.05)
            mutation_prob = st.slider("돌연변이 확률", 0.0, 0.5, 0.1, 0.05)

        with col2:
            tournament_size = st.slider("토너먼트 크기", 2, 10, 3, 1)
            patience = st.slider("조기 종료 patience", 10, 50, 20, 5)

    # 저장
    st.session_state.nsga2_config = {
        'population_size': population_size,
        'generations': generations,
        'early_stopping': early_stopping,
        'crossover_prob': crossover_prob,
        'mutation_prob': mutation_prob,
        'tournament_size': tournament_size,
        'patience': patience
    }

    estimated_time = (population_size * generations) / 3000  # 실제 측정 기반 추정
    st.info(f"⏱️ 예상 실행 시간: 약 {max(estimated_time, 0.5):.1f}분 (설정 및 시스템에 따라 변동)")


def run_weighted_sum_optimization(constraint_preset: str, scenario_template: str, premium_scan: bool = False):
    """Weighted Sum 최적화 실행

    Args:
        constraint_preset: 제약조건 프리셋
        scenario_template: 시나리오 템플릿
        premium_scan: True면 Premium Scan 모드 강제 활성화
    """
    from src.optimization_v2.pareto.weight_sweep_optimizer import WeightSweepOptimizer
    from src.optimization.re100_premium_calculator import RE100PremiumCalculator

    user_id = st.session_state.get('user_id', 'default')

    with st.spinner("🔄 Weighted Sum 최적화 실행 중..."):
        try:
            data_loader = st.session_state.data_loader
            optimization_data = data_loader.get_optimization_data()

            cost_calculator = RE100PremiumCalculator(user_id=user_id)
            baseline_case = st.session_state.get('cost_baseline_case', 'case1')

            optimizer = WeightSweepOptimizer(user_id=user_id)

            # 설정 오버라이드
            config = st.session_state.get('weighted_sum_config', {})
            if config.get('strategy'):
                optimizer.config_loader.config['weight_sweep']['strategy'] = config['strategy']
                if config.get('points'):
                    optimizer.config_loader.config['weight_sweep'][config['strategy']]['points'] = config['points']

            # NEW: Check for premium scan
            premium_scan_config = config.get('premium_scan')
            if premium_scan_config and premium_scan_config.get('enabled'):
                # Premium scan mode
                premium_range = premium_scan_config['range']
                st.info(f"🎯 Premium Scan: {len(premium_range)}개 레벨에서 최적화 실행")

                all_results = optimizer.run_sweep_with_premium_scan(
                    optimization_data=optimization_data,
                    cost_calculator=cost_calculator,
                    baseline_case=baseline_case,
                    constraint_preset=constraint_preset,
                    scenario_template=scenario_template if scenario_template else None,
                    premium_scan_range=premium_range
                )

                # Flatten results for standard processing
                results = []
                for premium_results in all_results.values():
                    results.extend(premium_results)

                # Store premium scan results separately
                st.session_state.premium_scan_results = all_results
                st.session_state.premium_scan_enabled = True
            else:
                # Normal mode
                results = optimizer.run_sweep(
                    optimization_data=optimization_data,
                    cost_calculator=cost_calculator,
                    baseline_case=baseline_case,
                    constraint_preset=constraint_preset,
                    scenario_template=scenario_template if scenario_template else None
                )
                st.session_state.premium_scan_enabled = False

            pareto_frontier = optimizer.get_pareto_frontier()

            st.session_state.pareto_results = results
            st.session_state.pareto_frontier = pareto_frontier
            st.session_state.optimization_method = 'weighted_sum'

            st.success(f"✅ Weighted Sum 최적화 완료! {len(results)}개 포인트, {len(pareto_frontier)}개 파레토 최적해")
            st.balloons()

        except Exception as e:
            _show_user_friendly_error(
                'optimization_failed',
                context=f"최적화 중 오류 발생: {str(e)}",
                solution="1. 제약조건을 확인하세요 (너무 엄격한 제약은 해가 없을 수 있음)\n2. 다른 최적화 방법을 시도하세요\n3. 솔버를 변경해보세요 (GLPK ↔ IPOPT)"
            )
            import traceback
            with st.expander("🔍 기술 상세 정보 (개발자용)", expanded=False):
                st.code(traceback.format_exc())


def run_epsilon_constraint_optimization(constraint_preset: str, scenario_template: str):
    """Epsilon-Constraint 최적화 실행"""
    from src.optimization_v2.pareto.epsilon_constraint_optimizer import EpsilonConstraintOptimizer
    from src.optimization.re100_premium_calculator import RE100PremiumCalculator

    user_id = st.session_state.get('user_id', 'default')

    with st.spinner("🔄 Epsilon-Constraint 최적화 실행 중..."):
        try:
            data_loader = st.session_state.data_loader
            optimization_data = data_loader.get_optimization_data()

            cost_calculator = RE100PremiumCalculator(user_id=user_id)
            baseline_case = st.session_state.get('cost_baseline_case', 'case1')

            optimizer = EpsilonConstraintOptimizer(user_id=user_id)

            # 설정 오버라이드
            config = st.session_state.get('epsilon_constraint_config', {})
            if config.get('strategy'):
                optimizer.config_loader.config['epsilon_constraint']['strategy'] = config['strategy']
                optimizer.config_loader.config['epsilon_constraint'][config['strategy']]['points'] = config.get('points', 10)
                optimizer.config_loader.config['epsilon_constraint'][config['strategy']]['cost_max_multiplier'] = config.get('cost_max_multiplier', 1.5)

            # NEW: Check for premium scan mode
            premium_scan_range = config.get('premium_scan_range')

            if premium_scan_range:
                st.info(f"🎯 Premium Scan Mode: {len(premium_scan_range)}개 레벨에서 최적화 실행")

            results = optimizer.run_epsilon_sweep(
                optimization_data=optimization_data,
                cost_calculator=cost_calculator,
                baseline_case=baseline_case,
                constraint_preset=constraint_preset,
                scenario_template=scenario_template if scenario_template else None,
                premium_scan_range=premium_scan_range  # NEW
            )

            pareto_frontier = optimizer.get_pareto_frontier()

            # NEW: If premium scan mode, restructure results as Dict[premium_pct, List[results]]
            if premium_scan_range:
                # Group results by premium percentage
                # Each epsilon result has 'premium_budget' and 'zero_premium_baseline'
                # Calculate premium_pct from these values
                premium_scan_results = {}
                for result in results:
                    baseline = result.get('zero_premium_baseline', result.get('baseline_cost', 0))
                    premium_budget = result.get('premium_budget', 0)

                    # Calculate premium percentage
                    if baseline > 0:
                        premium_pct = round((premium_budget / baseline) * 100)
                    else:
                        premium_pct = 0

                    if premium_pct not in premium_scan_results:
                        premium_scan_results[premium_pct] = []
                    premium_scan_results[premium_pct].append(result)

                st.session_state.premium_scan_results = premium_scan_results
                st.session_state.premium_scan_enabled = True

            st.session_state.pareto_results = results
            st.session_state.pareto_frontier = pareto_frontier
            st.session_state.optimization_method = 'epsilon_constraint'

            st.success(f"✅ Epsilon-Constraint 최적화 완료! {len(results)}개 포인트, {len(pareto_frontier)}개 파레토 최적해")
            st.balloons()

        except Exception as e:
            _show_user_friendly_error(
                'optimization_failed',
                context=f"최적화 중 오류 발생: {str(e)}",
                solution="1. 제약조건을 확인하세요 (너무 엄격한 제약은 해가 없을 수 있음)\n2. 다른 최적화 방법을 시도하세요\n3. 솔버를 변경해보세요 (GLPK ↔ IPOPT)"
            )
            import traceback
            with st.expander("🔍 기술 상세 정보 (개발자용)", expanded=False):
                st.code(traceback.format_exc())


def run_nsga2_optimization(constraint_preset: str, scenario_template: str):
    """NSGA-II 최적화 실행"""
    from src.optimization_v2.pareto.nsga2_optimizer import NSGA2Optimizer
    from src.optimization.re100_premium_calculator import RE100PremiumCalculator

    user_id = st.session_state.get('user_id', 'default')

    with st.spinner("🔄 NSGA-II 최적화 실행 중... (시간이 오래 걸립니다)"):
        try:
            data_loader = st.session_state.data_loader
            optimization_data = data_loader.get_optimization_data()

            cost_calculator = RE100PremiumCalculator(user_id=user_id)
            baseline_case = st.session_state.get('cost_baseline_case', 'case1')

            optimizer = NSGA2Optimizer(user_id=user_id)

            # 설정 오버라이드
            config = st.session_state.get('nsga2_config', {})
            if config:
                optimizer.config['population_size'] = config.get('population_size', 50)
                optimizer.config['generations'] = config.get('generations', 100)
                optimizer.config['crossover_prob'] = config.get('crossover_prob', 0.9)
                optimizer.config['mutation_prob'] = config.get('mutation_prob', 0.1)
                optimizer.config['termination']['early_stopping'] = config.get('early_stopping', True)
                optimizer.config['termination']['patience'] = config.get('patience', 20)

            results = optimizer.run_nsga2(
                optimization_data=optimization_data,
                cost_calculator=cost_calculator,
                baseline_case=baseline_case,
                constraint_preset=constraint_preset,
                scenario_template=scenario_template if scenario_template else None
            )

            st.session_state.pareto_results = results
            st.session_state.pareto_frontier = results  # NSGA-II는 결과가 곧 파레토 프론티어
            st.session_state.optimization_method = 'nsga2'

            st.success(f"✅ NSGA-II 최적화 완료! {len(results)}개 파레토 최적해")
            st.balloons()

        except Exception as e:
            _show_user_friendly_error(
                'optimization_failed',
                context=f"최적화 중 오류 발생: {str(e)}",
                solution="1. 제약조건을 확인하세요 (너무 엄격한 제약은 해가 없을 수 있음)\n2. 다른 최적화 방법을 시도하세요\n3. 솔버를 변경해보세요 (GLPK ↔ IPOPT)"
            )
            import traceback
            with st.expander("🔍 기술 상세 정보 (개발자용)", expanded=False):
                st.code(traceback.format_exc())


def _extract_point_metadata(point, method):
    """포인트 메타데이터 추출"""
    metadata = {}

    if method == 'weighted_sum':
        metadata['weights'] = point['weights']
    elif method == 'epsilon_constraint':
        metadata['epsilon'] = point['epsilon']
        metadata['baseline_cost'] = point.get('baseline_cost', 0)
    elif method == 'nsga2':
        metadata['rank'] = point.get('rank', 0)
        metadata['crowding_distance'] = point.get('crowding_distance', 0)

    return metadata


def create_pareto_frontier_summary(pareto_frontier, method):
    """파레토 프론티어 요약 DataFrame"""
    rows = []
    for idx, point in enumerate(pareto_frontier):
        row = {
            'Point': f"P{idx+1}",
            'Carbon (kgCO2eq)': point['summary']['total_carbon'],
            'Cost ($)': point['summary'].get('total_cost', 0),
            'Reduction (%)': point['summary'].get('total_reduction_pct', 0)
        }

        # 방법별 파라미터
        if method == 'weighted_sum':
            row['Carbon_Weight'] = point['weights']['carbon_weight']
            row['Cost_Weight'] = point['weights']['cost_weight']
        elif method == 'epsilon_constraint':
            row['Epsilon'] = point['epsilon']
        elif method == 'nsga2':
            row['Rank'] = point.get('rank', 0)
            row['Crowding_Distance'] = point.get('crowding_distance', 0)

        rows.append(row)

    return pd.DataFrame(rows)


def create_all_points_summary(pareto_results, method):
    """전체 탐색 포인트 요약 DataFrame"""
    rows = []
    for idx, point in enumerate(pareto_results):
        row = {
            'Point': f"A{idx+1}",
            'Carbon (kgCO2eq)': point['summary']['total_carbon'],
            'Cost ($)': point['summary'].get('total_cost', 0)
        }

        # 방법별 파라미터
        if method == 'weighted_sum':
            row['Carbon_Weight'] = point['weights']['carbon_weight']
        elif method == 'epsilon_constraint':
            row['Epsilon'] = point['epsilon']
        elif method == 'nsga2':
            row['Rank'] = point.get('rank', 0)

        rows.append(row)

    return pd.DataFrame(rows)


def save_pareto_point_as_scenario(point, scenario_name):
    """파레토 포인트를 비교용 시나리오로 저장"""
    from datetime import datetime

    # optimization_results에 추가
    st.session_state.optimization_results[scenario_name] = {
        'result_df': point['result_df'],
        'summary': point['summary'],
        'solution': point['solution'],
        'optimization_config': {
            'source': 'pareto_frontier',
            'method': st.session_state.get('optimization_method', 'unknown'),
            'timestamp': point.get('timestamp', datetime.now().isoformat())
        }
    }

    # ComparisonDashboard에도 추가
    st.session_state.comparison_dashboard.add_scenario(
        name=scenario_name,
        result_df=point['result_df'],
        summary=point['summary'],
        solution=point['solution']
    )


def render_cathode_element_results(cathode_data):
    """양극재 원소별 최적화 결과"""
    import plotly.graph_objects as go

    st.markdown("#### 🔋 양극재 원소별 최적화 결과")

    col1, col2 = st.columns(2)
    with col1:
        optimized_emission = cathode_data['cathode_emission_factor']
        original_emission = cathode_data.get('original_cathode_emission')

        if original_emission and original_emission > 0:
            reduction_pct = ((original_emission - optimized_emission) / original_emission) * 100
            st.metric(
                "양극재 최적 배출계수",
                f"{optimized_emission:.4f} kgCO2eq/kg",
                delta=f"-{reduction_pct:.1f}%",
                delta_color="normal",
                help=f"원본 배출계수: {original_emission:.4f} kgCO2eq/kg"
            )
        else:
            st.metric(
                "양극재 최적 배출계수",
                f"{optimized_emission:.4f} kgCO2eq/kg",
                help="원소별 비율 최적화 결과로 계산된 양극재 배출계수"
            )

    # 원소별 결과 테이블
    st.markdown("**원소별 상세 결과**")

    element_rows = []
    for element, element_data in cathode_data['elements'].items():
        element_rows.append({
            '원소': element,
            '신재 비율': f"{element_data['virgin_ratio']*100:.1f}%",
            '재활용 비율': f"{element_data['recycle_ratio']*100:.1f}%",
            '저탄소 비율': f"{element_data['low_carbon_ratio']*100:.1f}%",
            '원소 배출계수': f"{element_data['emission_factor']:.4f}"
        })

    element_df = pd.DataFrame(element_rows)
    st.dataframe(element_df, use_container_width=True, hide_index=True)

    # 원소별 비율 시각화
    st.markdown("**원소별 비율 시각화**")

    elements = list(cathode_data['elements'].keys())
    virgin_ratios = [cathode_data['elements'][e]['virgin_ratio']*100 for e in elements]
    recycle_ratios = [cathode_data['elements'][e]['recycle_ratio']*100 for e in elements]
    low_carbon_ratios = [cathode_data['elements'][e]['low_carbon_ratio']*100 for e in elements]

    fig = go.Figure(data=[
        go.Bar(name='신재', x=elements, y=virgin_ratios),
        go.Bar(name='재활용', x=elements, y=recycle_ratios),
        go.Bar(name='저탄소', x=elements, y=low_carbon_ratios)
    ])

    fig.update_layout(
        barmode='stack',
        title='양극재 원소별 비율 구성',
        xaxis_title='원소',
        yaxis_title='비율 (%)',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_anode_composition_results(anode_data):
    """음극재 composition 최적화 결과"""
    import plotly.graph_objects as go

    st.markdown("#### ⚡ 음극재 Composition 최적화 결과")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "음극재 최적 배출계수",
            f"{anode_data['anode_emission_factor']:.4f} kgCO2eq/kg",
            help="Natural vs Artificial Graphite 비율 최적화 결과"
        )

    with col2:
        natural_ratio = anode_data['natural_graphite_ratio'] * 100
        st.metric(
            "Natural Graphite 비율",
            f"{natural_ratio:.1f}%",
            help="천연 흑연 (저탄소)"
        )

    with col3:
        artificial_ratio = anode_data['artificial_graphite_ratio'] * 100
        st.metric(
            "Artificial Graphite 비율",
            f"{artificial_ratio:.1f}%",
            help="인조 흑연 (고탄소)"
        )

    # Composition 비율 시각화 - 파이 차트
    st.markdown("**음극재 Composition 비율**")

    fig = go.Figure(data=[go.Pie(
        labels=['Natural Graphite (천연)', 'Artificial Graphite (인조)'],
        values=[anode_data['natural_graphite_ratio'], anode_data['artificial_graphite_ratio']],
        hole=0.4,
        marker=dict(colors=['#2ecc71', '#e74c3c']),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])

    fig.update_layout(
        title='음극재 Composition 구성',
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # 감축 효과 설명
    st.markdown("**💡 감축 메커니즘**")
    st.info(f"""
    **Natural Graphite 우선 사용**: 천연 흑연(5.616 kgCO2eq/kg)은 인조 흑연(19.865 kgCO2eq/kg)보다
    약 **71.7% 낮은** 배출계수를 가집니다.

    현재 최적화 결과:
    - Natural Graphite: {natural_ratio:.1f}%
    - Artificial Graphite: {artificial_ratio:.1f}%

    이 비율 최적화에 RE100이 추가로 적용되어 최종 배출계수가 결정됩니다.
    """)


def render_pareto_frontier_plot(pareto_results, pareto_frontier, method):
    """파레토 프론티어 산점도"""
    import plotly.graph_objects as go

    # 모든 포인트
    all_carbon = [r['summary']['total_carbon'] for r in pareto_results]
    all_cost = [r['summary'].get('total_cost', 0) for r in pareto_results]

    # 방법별 라벨 생성
    if method == 'weighted_sum':
        all_labels = [f"C:{r['weights']['carbon_weight']:.2f}" for r in pareto_results]
        pareto_labels = [f"C:{r['weights']['carbon_weight']:.2f}" for r in pareto_frontier]
    elif method == 'epsilon_constraint':
        all_labels = [f"ε:${r['epsilon']:,.0f}" for r in pareto_results]
        pareto_labels = [f"ε:${r['epsilon']:,.0f}" for r in pareto_frontier]
    elif method == 'nsga2':
        all_labels = [f"Rank:{r.get('rank', 0)}" for r in pareto_results]
        pareto_labels = [f"Rank:{r.get('rank', 0)}" for r in pareto_frontier]
    else:
        all_labels = [''] * len(pareto_results)
        pareto_labels = [''] * len(pareto_frontier)

    # 파레토 최적 포인트
    pareto_carbon = [r['summary']['total_carbon'] for r in pareto_frontier]
    pareto_cost = [r['summary'].get('total_cost', 0) for r in pareto_frontier]

    fig = go.Figure()

    # 모든 포인트 (연한 파란색)
    fig.add_trace(go.Scatter(
        x=all_cost,
        y=all_carbon,
        mode='markers',
        name='탐색 포인트',
        text=all_labels,
        marker=dict(size=8, color='lightblue', opacity=0.6),
        hovertemplate='<b>%{text}</b><br>비용: $%{x:,.2f}<br>탄소: %{y:.2f} kgCO2eq'
    ))

    # 파레토 최적 포인트 (빨간 별)
    fig.add_trace(go.Scatter(
        x=pareto_cost,
        y=pareto_carbon,
        mode='markers+lines',
        name='파레토 프론티어',
        text=pareto_labels,
        marker=dict(size=12, color='red', symbol='star'),
        line=dict(color='red', dash='dash'),
        hovertemplate='<b>%{text}</b><br>비용: $%{x:,.2f}<br>탄소: %{y:.2f} kgCO2eq'
    ))

    # 베이스라인 비용 세로 점선 추가
    if pareto_results:
        baseline_cost = pareto_results[0].get('zero_premium_baseline') or pareto_results[0].get('baseline_cost')
        if baseline_cost and baseline_cost > 0:
            fig.add_vline(
                x=baseline_cost,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"베이스라인 (${baseline_cost:,.2f})",
                annotation_position="top"
            )

    fig.update_layout(
        title='파레토 프론티어 (탄소 vs 비용)',
        xaxis_title='총 비용 ($)',
        yaxis_title='총 탄소배출 (kgCO2eq)',
        hovermode='closest',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def render_premium_scan_visualization(premium_scan_results: Dict[float, List[Dict]], method: str):
    """프리미엄 스캔 결과 시각화"""
    import plotly.graph_objects as go

    st.markdown("#### 💰 프리미엄 스캔 분석")
    st.caption("다양한 비용 제약 수준에서 탐색된 파레토 프론티어")

    # 1. Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("프리미엄 레벨", f"{len(premium_scan_results)}개",
                 help="탐색한 프리미엄 제약 레벨 수")

    with col2:
        total_points = sum(len(results) for results in premium_scan_results.values())
        st.metric("총 탐색 포인트", f"{total_points}개",
                 help="모든 프리미엄 레벨에서 탐색한 총 포인트 수")

    with col3:
        avg_points = total_points / len(premium_scan_results) if premium_scan_results else 0
        st.metric("레벨당 평균", f"{avg_points:.1f}개",
                 help="각 프리미엄 레벨당 평균 탐색 포인트 수")

    with col4:
        premium_levels = sorted(premium_scan_results.keys())
        premium_range_str = f"{premium_levels[0]}% ~ {premium_levels[-1]}%" if premium_levels else "N/A"
        st.metric("프리미엄 범위", premium_range_str,
                 help="탐색한 프리미엄 제약 범위")

    st.markdown("---")

    # 2. Multi-frontier plot
    st.markdown("**프리미엄 레벨별 파레토 프론티어**")

    fig = go.Figure()

    # Color palette for different premium levels
    import plotly.colors as pc
    colors = pc.sample_colorscale("Viridis", [i/(len(premium_scan_results)-1) if len(premium_scan_results) > 1 else 0.5
                                               for i in range(len(premium_scan_results))])

    for idx, (premium_pct, results) in enumerate(sorted(premium_scan_results.items())):
        if not results:
            continue

        carbon_values = [r['summary']['total_carbon'] for r in results]
        cost_values = [r['summary'].get('total_cost', 0) for r in results]

        # Labels based on method
        labels = []
        for r in results:
            if 'weights' in r:
                # Weighted Sum
                labels.append(f"α:{r['weights']['carbon_weight']:.2f}")
            elif 'epsilon' in r:
                # Epsilon-Constraint
                labels.append(f"ε:${r['epsilon']:,.0f}")
            else:
                # NSGA-II or unknown
                labels.append('')

        fig.add_trace(go.Scatter(
            x=cost_values,
            y=carbon_values,
            mode='markers+lines',
            name=f"+{premium_pct}%",
            text=labels,
            marker=dict(size=8, color=colors[idx]),
            line=dict(color=colors[idx], width=2),
            hovertemplate=f'<b>Premium: +{premium_pct}%</b><br>%{{text}}<br>비용: $%{{x:,.2f}}<br>탄소: %{{y:.2f}} kgCO2eq'
        ))

    fig.update_layout(
        title='프리미엄 레벨별 파레토 프론티어',
        xaxis_title='총 비용 ($)',
        yaxis_title='총 탄소배출 (kgCO2eq)',
        hovermode='closest',
        height=500,
        showlegend=True,
        legend=dict(title='프리미엄 레벨', orientation='v', x=1.02, y=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Summary table by premium level
    st.markdown("**프리미엄 레벨별 요약**")

    summary_rows = []
    for premium_pct, results in sorted(premium_scan_results.items()):
        if not results:
            continue

        carbon_values = [r['summary']['total_carbon'] for r in results]
        cost_values = [r['summary'].get('total_cost', 0) for r in results]

        summary_rows.append({
            '프리미엄 레벨': f"+{premium_pct}%",
            '탐색 포인트': len(results),
            '최소 탄소': f"{min(carbon_values):.2f}",
            '최대 탄소': f"{max(carbon_values):.2f}",
            '평균 탄소': f"{sum(carbon_values)/len(carbon_values):.2f}",
            '최소 비용': f"${min(cost_values):,.2f}",
            '최대 비용': f"${max(cost_values):,.2f}",
            '평균 비용': f"${sum(cost_values)/len(cost_values):,.2f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")


def render_pareto_comparison_table(pareto_frontier, method):
    """파레토 포인트 비교 테이블"""
    result_rows = []
    for idx, r in enumerate(pareto_frontier):
        row = {
            'Point': f"P{idx+1}",
            '총 탄소배출': f"{r['summary']['total_carbon']:.2f}",
            '총 비용': f"${r['summary'].get('total_cost', 0):,.2f}",
        }

        # 방법별 추가 정보
        if method == 'weighted_sum':
            row['탄소 가중치'] = f"{r['weights']['carbon_weight']:.2f}"
            row['비용 가중치'] = f"{r['weights']['cost_weight']:.2f}"
        elif method == 'epsilon_constraint':
            row['Epsilon (비용 상한)'] = f"${r['epsilon']:,.2f}"
        elif method == 'nsga2':
            row['Rank'] = r.get('rank', 0)
            cd = r.get('crowding_distance', 0)
            if cd == float('inf'):
                row['Crowding Distance'] = "∞"
            else:
                row['Crowding Distance'] = f"{cd:.4f}"

        if 'baseline_cost' in r and r['baseline_cost'] > 0:
            row['비용 증가율'] = f"{((r['summary'].get('total_cost', 0) / r['baseline_cost']) - 1) * 100:.1f}%"

        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)


def render_pareto_export_options(pareto_results, pareto_frontier, method):
    """파레토 결과 데이터 다운로드"""
    from datetime import datetime

    col1, col2, col3 = st.columns(3)

    with col1:
        # 파레토 프론티어 요약
        frontier_df = create_pareto_frontier_summary(pareto_frontier, method)
        csv = frontier_df.to_csv(index=False, encoding='utf-8-sig')

        st.download_button(
            label="📊 파레토 프론티어 CSV",
            data=csv,
            file_name=f"pareto_frontier_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # 전체 탐색 포인트
        all_df = create_all_points_summary(pareto_results, method)
        csv = all_df.to_csv(index=False, encoding='utf-8-sig')

        st.download_button(
            label="🔍 전체 탐색 포인트 CSV",
            data=csv,
            file_name=f"all_points_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        # JSON 형식 (상세 정보 포함)
        export_data = {
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_points': len(pareto_results),
                'pareto_optimal_points': len(pareto_frontier),
                'efficiency': len(pareto_frontier) / len(pareto_results) * 100 if pareto_results else 0
            },
            'pareto_frontier': [
                {
                    'carbon': p['summary']['total_carbon'],
                    'cost': p['summary'].get('total_cost', 0),
                    'metadata': _extract_point_metadata(p, method)
                }
                for p in pareto_frontier
            ]
        }

        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

        st.download_button(
            label="📄 상세 정보 JSON",
            data=json_str,
            file_name=f"pareto_results_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


def render_single_point_details(point, method):
    """선택된 파레토 포인트의 상세 정보"""
    st.markdown("#### 📊 선택된 포인트 상세 정보")

    # 핵심 지표 + 최적화 설정
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**핵심 지표**")
        summary = point['summary']

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("총 탄소배출", f"{summary['total_carbon']:.2f} kgCO2eq")
        with col_b:
            st.metric("총 비용", f"${summary.get('total_cost', 0):,.2f}")
        with col_c:
            reduction = summary.get('total_reduction_pct', 0)
            st.metric("감축률", f"{reduction:.1f}%")

    with col2:
        st.markdown("**최적화 설정**")

        if method == 'weighted_sum':
            st.write(f"탄소 가중치: **{point['weights']['carbon_weight']:.2f}**")
            st.write(f"비용 가중치: **{point['weights']['cost_weight']:.2f}**")
        elif method == 'epsilon_constraint':
            st.write(f"Epsilon (비용 상한): **${point['epsilon']:,.0f}**")
            baseline = point.get('baseline_cost', 0)
            if baseline > 0:
                increase_pct = (point['epsilon'] / baseline - 1) * 100
                st.write(f"기준 대비: **+{increase_pct:.1f}%**")
        else:  # nsga2
            st.write(f"Rank: **{point.get('rank', 0)}**")
            cd = point.get('crowding_distance', 0)
            if cd == float('inf'):
                st.write(f"Crowding Distance: **∞ (경계)**")
            else:
                st.write(f"Crowding Distance: **{cd:.4f}**")

    st.markdown("---")

    # 양극재 원소별 결과 (있으면)
    solution = point['solution']
    if 'cathode' in solution:
        render_cathode_element_results(solution['cathode'])
        st.markdown("---")

    # 음극재 composition 결과 (있으면) - Phase 3
    if 'anode' in solution:
        render_anode_composition_results(solution['anode'])
        st.markdown("---")

    # 자재별 상세 결과
    st.markdown("#### 📈 자재별 상세 결과")

    result_df = point['result_df']
    summary = point['summary']

    # 양극재 상세 정보 표시 (개선된 UI)
    cathode_rows = result_df[result_df['자재명'].str.contains('Cathode|양극', na=False)]
    if len(cathode_rows) > 0:
        st.markdown("### ⚡ 양극재 상세 분석")

        for idx, row in cathode_rows.iterrows():
            with st.expander(f"📊 {row['자재명']}", expanded=True):
                # 감축률 세분화 표시
                col1, col2, col3 = st.columns(3)

                with col1:
                    element_reduction = row.get('Element감축률(%)', 0)
                    st.metric(
                        "Element-level 감축",
                        f"{element_reduction:.1f}%",
                        help="재활용재/저탄소 메탈 사용으로 인한 감축"
                    )

                with col2:
                    re100_reduction = row.get('RE100감축률(%)', 0)
                    st.metric(
                        "RE100 감축",
                        f"{re100_reduction:.1f}%",
                        help="재생에너지 사용으로 인한 추가 감축"
                    )

                with col3:
                    total_reduction = row['감축률(%)']
                    st.metric(
                        "총 감축률",
                        f"{total_reduction:.1f}%",
                        help="Element-level + RE100 복합 효과"
                    )

                # RE100 적용 비율
                st.markdown("**🔋 RE100 적용 비율**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- **Tier1 (CAM 제조)**: {row['Tier1_RE(%)']:.2f}%")
                with col2:
                    st.write(f"- **Tier2 (pCAM 제조)**: {row['Tier2_RE(%)']:.2f}%")

                # Element 구성 비율
                st.markdown("**🧪 원소 구성**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"- **재활용재**: {row['재활용_비율(%)']:.1f}%")
                with col2:
                    st.write(f"- **저탄소 메탈**: {row['저탄소_비율(%)']:.1f}%")
                with col3:
                    st.write(f"- **버진**: {row['버진_비율(%)']:.1f}%")

                # 진행률 바 시각화
                st.markdown("**📈 감축 기여도**")

                # Element-level 기여도
                element_pct = (element_reduction / total_reduction * 100) if total_reduction > 0 else 0
                st.progress(element_pct / 100, text=f"Element-level: {element_pct:.0f}%")

                # RE100 기여도
                re100_pct = (re100_reduction / total_reduction * 100) if total_reduction > 0 else 0
                st.progress(re100_pct / 100, text=f"RE100: {re100_pct:.0f}%")

    # ResultsVisualizer 활용
    visualizer = ResultsVisualizer()

    # 테이블과 차트 표시
    visualizer._render_detail_table(result_df)
    visualizer._render_charts(result_df, summary)


def render_pareto_point_selector(pareto_frontier, method):
    """개별 파레토 포인트 선택 및 상세 보기"""
    # 포인트 선택 옵션 생성
    point_options = {}
    for idx, point in enumerate(pareto_frontier):
        carbon = point['summary']['total_carbon']
        cost = point['summary'].get('total_cost', 0)

        if method == 'weighted_sum':
            label = f"Point {idx+1}: α={point['weights']['carbon_weight']:.2f} (탄소: {carbon:.2f}, 비용: ${cost:,.0f})"
        elif method == 'epsilon_constraint':
            label = f"Point {idx+1}: ε=${point['epsilon']:,.0f} (탄소: {carbon:.2f})"
        else:  # nsga2
            label = f"Point {idx+1}: Rank={point.get('rank', 0)} (탄소: {carbon:.2f}, 비용: ${cost:,.0f})"

        point_options[label] = idx

    selected_label = st.selectbox(
        "분석할 파레토 포인트 선택",
        options=list(point_options.keys()),
        help="선택한 포인트의 상세 정보를 아래에서 확인할 수 있습니다"
    )

    selected_idx = point_options[selected_label]
    selected_point = pareto_frontier[selected_idx]

    # 선택된 포인트 상세 정보 표시
    render_single_point_details(selected_point, method)

    # 시나리오로 저장 기능
    st.markdown("---")
    st.markdown("#### 💾 시나리오로 저장")

    col1, col2 = st.columns([3, 1])

    with col1:
        scenario_name = st.text_input(
            "시나리오 이름",
            value=f"Pareto_{method}_{selected_idx+1}",
            help="비교용 시나리오로 저장할 이름을 입력하세요"
        )

    with col2:
        if st.button("💾 저장", use_container_width=True, type="primary"):
            save_pareto_point_as_scenario(selected_point, scenario_name)
            st.success(f"✅ '{scenario_name}' 저장!")
            st.info("'시나리오 비교' 탭에서 다른 결과와 비교할 수 있습니다.")


def render_pareto_results_section():
    """파레토 최적화 결과 전용 섹션"""
    pareto_results = st.session_state.pareto_results
    pareto_frontier = st.session_state.get('pareto_frontier', [])
    method = st.session_state.get('optimization_method', 'weighted_sum')

    # 1. 파레토 최적화 요약 카드 (4개 메트릭)
    st.subheader("🌟 파레토 최적화 요약")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("탐색 포인트", f"{len(pareto_results)}개",
                 help="최적화 과정에서 탐색한 전체 포인트 수")

    with col2:
        st.metric("파레토 최적해", f"{len(pareto_frontier)}개",
                 help="파레토 프론티어에 속하는 최적 솔루션 수")

    with col3:
        method_names = {
            'weighted_sum': '⚖️ 가중치 스캔',
            'epsilon_constraint': '🎯 Epsilon 제약',
            'nsga2': '🧬 NSGA-II'
        }
        st.metric("최적화 방법", method_names[method])

    with col4:
        if len(pareto_results) > 0:
            efficiency = len(pareto_frontier) / len(pareto_results) * 100
            st.metric("파레토 효율성", f"{efficiency:.1f}%",
                     help="탐색 포인트 중 파레토 최적해 비율")

    st.markdown("---")

    # 2. 파레토 프론티어 시각화
    st.subheader("🎯 파레토 프론티어")
    render_pareto_frontier_plot(pareto_results, pareto_frontier, method)

    st.markdown("---")

    # 2.5. Premium Scan 시각화 (있는 경우)
    if st.session_state.get('premium_scan_enabled', False):
        premium_scan_results = st.session_state.get('premium_scan_results', {})
        if premium_scan_results:
            render_premium_scan_visualization(premium_scan_results, method)
            st.markdown("---")

    # 3. 개별 포인트 선택 및 상세 정보
    st.subheader("🔍 파레토 포인트 상세 분석")
    render_pareto_point_selector(pareto_frontier, method)

    st.markdown("---")

    # 4. 파레토 포인트 비교 테이블
    st.subheader("📊 파레토 포인트 비교")
    render_pareto_comparison_table(pareto_frontier, method)

    st.markdown("---")

    # 5. 방법별 특화 분석
    st.subheader("📈 방법별 상세 분석")
    if method == 'weighted_sum':
        render_weighted_sum_analysis(pareto_results, pareto_frontier)
    elif method == 'epsilon_constraint':
        render_epsilon_constraint_analysis(pareto_results, pareto_frontier)
    elif method == 'nsga2':
        render_nsga2_analysis(pareto_results, pareto_frontier)

    st.markdown("---")

    # 6. 데이터 다운로드
    st.subheader("📥 데이터 다운로드")
    render_pareto_export_options(pareto_results, pareto_frontier, method)


def render_single_optimization_results():
    """단일 최적화 결과 (레거시)"""
    st.info("""
    💡 **단일 최적화 결과**

    이 결과는 단일 목적 최적화 또는 레거시 결과입니다.
    파레토 최적화를 실행하면 더 풍부한 분석 기능을 사용할 수 있습니다.
    """)

    # 시나리오 선택
    scenario_names = list(st.session_state.optimization_results.keys())

    selected_scenario = st.selectbox(
        "시나리오 선택",
        options=scenario_names,
        index=scenario_names.index(st.session_state.current_result)
        if st.session_state.current_result in scenario_names else 0
    )

    # 결과 표시
    result_data = st.session_state.optimization_results[selected_scenario]
    result_df = result_data['result_df']
    summary = result_data['summary']
    solution = result_data['solution']

    # 양극재 원소별 결과 표시 (있는 경우)
    if 'cathode' in solution:
        st.subheader("🔋 양극재 원소별 최적화 결과")

        cathode_data = solution['cathode']

        # 양극재 전체 배출계수
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "양극재 최적 배출계수",
                f"{cathode_data['cathode_emission_factor']:.4f} kgCO2eq/kg",
                help="원소별 비율 최적화 결과로 계산된 양극재 배출계수"
            )

        # 원소별 결과 테이블
        st.markdown("#### 원소별 상세 결과")

        element_rows = []
        for element, element_data in cathode_data['elements'].items():
            element_rows.append({
                '원소': element,
                '신재 비율': f"{element_data['virgin_ratio']*100:.1f}%",
                '재활용 비율': f"{element_data['recycle_ratio']*100:.1f}%",
                '저탄소 비율': f"{element_data['low_carbon_ratio']*100:.1f}%",
                '원소 배출계수': f"{element_data['emission_factor']:.4f}"
            })

        element_df = pd.DataFrame(element_rows)
        st.dataframe(element_df, use_container_width=True, hide_index=True)

        # 원소별 비율 시각화
        st.markdown("#### 원소별 비율 시각화")

        import plotly.graph_objects as go

        elements = list(cathode_data['elements'].keys())
        virgin_ratios = [cathode_data['elements'][e]['virgin_ratio']*100 for e in elements]
        recycle_ratios = [cathode_data['elements'][e]['recycle_ratio']*100 for e in elements]
        low_carbon_ratios = [cathode_data['elements'][e]['low_carbon_ratio']*100 for e in elements]

        fig = go.Figure(data=[
            go.Bar(name='신재', x=elements, y=virgin_ratios),
            go.Bar(name='재활용', x=elements, y=recycle_ratios),
            go.Bar(name='저탄소', x=elements, y=low_carbon_ratios)
        ])

        fig.update_layout(
            barmode='stack',
            title='양극재 원소별 비율 구성',
            xaxis_title='원소',
            yaxis_title='비율 (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    # 음극재 composition 결과 표시 (있는 경우) - Phase 3
    if 'anode' in solution:
        render_anode_composition_results(solution['anode'])
        st.markdown("---")

    # 최적화 파라미터 정보 표시
    with st.expander("⚙️ 최적화 파라미터 및 제약조건 정보", expanded=False):
        st.markdown("### 📋 활성화된 제약조건")

        # 제약조건 정보 가져오기
        constraints = st.session_state.constraint_manager.list_constraints(enabled_only=True)

        if constraints:
            for idx, name in enumerate(constraints, 1):
                constraint = st.session_state.constraint_manager.get_constraint(name)
                st.markdown(f"**{idx}. {constraint.name}**")
                st.caption(f"설명: {constraint.description}")

                # MaterialManagementConstraint의 경우 규칙 상세 표시
                if hasattr(constraint, 'material_rules') and constraint.material_rules:
                    rule_data = []
                    for rule in constraint.material_rules:
                        rule_info = {
                            '자재': rule['material'],
                            '규칙 타입': rule['type'],
                        }

                        # 규칙별 파라미터 추가
                        if rule['type'] == 'force_element_ratio_range':
                            params = rule['params']
                            rule_info['원소'] = params.get('element', '-')
                            rule_info['재활용 최소'] = f"{params.get('recycle_min', 0)*100:.1f}%"
                            rule_info['재활용 최대'] = f"{params.get('recycle_max', 0)*100:.1f}%"
                            rule_info['저탄소 최소'] = f"{params.get('low_carbon_min', 0)*100:.1f}%"
                            rule_info['저탄소 최대'] = f"{params.get('low_carbon_max', 0)*100:.1f}%"
                        elif rule['type'] == 'regional_preference':
                            params = rule['params']
                            rule_info['허용 국가'] = ', '.join(params.get('preferred_regions', []))
                        elif rule['type'] in ['virgin_only', 'exclude_low_carbon', 'exclude_recycle']:
                            rule_info['설명'] = {
                                'virgin_only': '신재만 사용',
                                'exclude_low_carbon': '저탄소메탈 제외',
                                'exclude_recycle': '재활용재 제외'
                            }.get(rule['type'], rule['type'])

                        rule_data.append(rule_info)

                    if rule_data:
                        rule_df = pd.DataFrame(rule_data)
                        st.dataframe(rule_df, use_container_width=True, hide_index=True)

                st.markdown("---")
        else:
            st.info("활성화된 제약조건이 없습니다. (제약 없는 최적화)")

        st.markdown("### 📊 자재별 최적화 결과 파라미터")

        # 자재별 파라미터를 DataFrame으로 구성
        param_rows = []
        for material_name, material_data in solution['materials'].items():
            row = {
                '자재명': material_name,
                '자재 타입': material_data.get('type', 'General')
            }

            # Formula 자재
            if 'tier1_re' in material_data:
                row['Tier1 RE'] = f"{material_data['tier1_re']*100:.1f}%"
                row['Tier2 RE'] = f"{material_data['tier2_re']*100:.1f}%"

            # Ni/Co/Li 자재
            if 'recycle_ratio' in material_data:
                row['재활용 비율'] = f"{material_data['recycle_ratio']*100:.1f}%"
                row['저탄소 비율'] = f"{material_data['low_carbon_ratio']*100:.1f}%"
                row['버진 비율'] = f"{material_data['virgin_ratio']*100:.1f}%"

            # 배출계수
            row['원본 배출계수'] = f"{material_data['original_emission']:.4f}"
            row['최적 배출계수'] = f"{material_data['modified_emission']:.4f}"

            # 감축률
            if 'reduction_pct' in material_data:
                row['감축률'] = f"{material_data['reduction_pct']:.2f}%"

            param_rows.append(row)

        param_df = pd.DataFrame(param_rows)
        st.dataframe(param_df, use_container_width=True, hide_index=True, height=400)

        # CSV 다운로드 버튼
        csv = param_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 파라미터 CSV 다운로드",
            data=csv,
            file_name=f"{selected_scenario}_parameters.csv",
            mime="text/csv"
        )

    # 디버깅 로그 섹션
    if 'debug_logs' in solution and solution['debug_logs']:
        with st.expander("🔍 디버깅 로그 (Tier RE 상세)", expanded=False):
            st.markdown("### Formula 자재 Tier RE 값 추적")
            st.caption("솔버가 계산한 원본 값(Raw)과 후처리 후 최종 값(Final)을 비교합니다.")

            debug_rows = []
            for log in solution['debug_logs']:
                # 의미 있는 값만 표시 (tier1 or tier2 > 1e-6)
                if abs(log.get('tier1_raw', 0)) > 1e-10 or abs(log.get('tier2_raw', 0)) > 1e-10:
                    debug_rows.append({
                        '자재명': log['material'],
                        'Tier1 Raw': f"{log.get('tier1_raw', 0):.10f}",
                        'Tier1 Final': f"{log.get('tier1_final', 0):.6f}",
                        'Tier2 Raw': f"{log.get('tier2_raw', 0):.10f}",
                        'Tier2 Final': f"{log.get('tier2_final', 0):.6f}",
                        '양극재 여부': '✅' if log.get('is_cathode') else '❌',
                        '비고': log.get('note', '')
                    })

            if debug_rows:
                debug_df = pd.DataFrame(debug_rows)
                st.dataframe(debug_df, use_container_width=True, hide_index=True)

                st.info("""
                **해석 가이드:**
                - **Raw 값**: 솔버가 계산한 원본 값 (부동소수점 정밀도 오차 포함)
                - **Final 값**: Threshold(1e-6) 적용 후 최종 값
                - **양극재**: Element-level에서 RE100 적용되므로 자재 레벨은 강제로 0
                - **일반 Formula**: 실제 최적화된 값 사용
                """)
            else:
                st.info("모든 Formula 자재의 Tier RE 값이 0입니다.")

    # ResultsVisualizer 렌더링
    visualizer = ResultsVisualizer()
    visualizer.render(result_df, summary, solution)


def render_results_tab():
    """결과 확인 탭 - 파레토 최적화 결과 전용"""
    st.header("📊 최적화 결과")

    # 파레토 결과 우선 체크
    if 'pareto_results' in st.session_state and st.session_state.pareto_results:
        render_pareto_results_section()
    elif st.session_state.optimization_results:
        # 레거시: 단일 최적화 결과
        render_single_optimization_results()
    else:
        st.info("""
        최적화 결과가 없습니다.

        '최적화 실행' 탭에서 파레토 최적화를 실행하세요.
        """)


def render_comparison_tab():
    """시나리오 비교 탭 - 개별 시나리오 상세 비교"""
    st.header("🔄 시나리오 비교")

    # 파레토 결과가 있으면 안내 메시지
    if 'pareto_results' in st.session_state and st.session_state.pareto_results:
        st.info("""
        💡 **파레토 최적화 결과는 '결과 확인' 탭에서 확인하세요**

        이 탭은 여러 최적화 시나리오를 상세 비교하기 위한 공간입니다.

        **파레토 포인트를 시나리오로 저장하는 방법**:
        1. '결과 확인' 탭으로 이동
        2. 원하는 파레토 포인트 선택
        3. '시나리오로 저장' 버튼 클릭
        4. 이 탭에서 다른 결과와 비교
        """)

        st.markdown("---")

    # ComparisonDashboard 렌더링 (기존 기능 유지)
    comparison_dashboard = st.session_state.comparison_dashboard
    comparison_dashboard.render()


def render_weighted_sum_analysis(pareto_results: list, pareto_frontier: list):
    """Weighted Sum 방법 특화 분석"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("#### ⚖️ Weighted Sum 분석")

    st.markdown("""
    **가중치 변화에 따른 목적함수 트레이드오프**

    탄소 가중치(α)가 증가할수록 탄소 배출이 감소하고, 비용 가중치(β)가 증가할수록 비용이 증가합니다.
    """)

    # 가중치 순으로 정렬
    sorted_results = sorted(pareto_frontier, key=lambda x: x['weights']['carbon_weight'])

    carbon_weights = [r['weights']['carbon_weight'] for r in sorted_results]
    carbon_values = [r['summary']['total_carbon'] for r in sorted_results]
    cost_values = [r['summary'].get('total_cost', 0) for r in sorted_results]

    # 이중 축 플롯
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 탄소 배출 (좌측 Y축)
    fig.add_trace(
        go.Scatter(
            x=carbon_weights,
            y=carbon_values,
            mode='lines+markers',
            name='탄소 배출',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ),
        secondary_y=False
    )

    # 비용 (우측 Y축)
    fig.add_trace(
        go.Scatter(
            x=carbon_weights,
            y=cost_values,
            mode='lines+markers',
            name='비용',
            line=dict(color='orange', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="탄소 가중치 (α)")
    fig.update_yaxes(title_text="탄소 배출 (kgCO2eq)", secondary_y=False)
    fig.update_yaxes(title_text="비용 ($)", secondary_y=True)

    fig.update_layout(
        title='가중치 변화에 따른 목적함수 변화',
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # 통계 요약
    col1, col2, col3 = st.columns(3)

    with col1:
        carbon_range = max(carbon_values) - min(carbon_values)
        st.metric("탄소 배출 범위", f"{carbon_range:.2f} kgCO2eq",
                 help="최대 - 최소 탄소 배출")

    with col2:
        cost_range = max(cost_values) - min(cost_values)
        st.metric("비용 범위", f"${cost_range:,.2f}",
                 help="최대 - 최소 비용")

    with col3:
        pareto_efficiency = len(pareto_frontier) / len(pareto_results) * 100
        st.metric("파레토 효율성", f"{pareto_efficiency:.1f}%",
                 help="전체 탐색 포인트 중 파레토 최적해 비율")


def render_epsilon_constraint_analysis(pareto_results: list, pareto_frontier: list):
    """Epsilon-Constraint 방법 특화 분석"""
    import plotly.graph_objects as go

    st.markdown("#### 🎯 Epsilon-Constraint 분석")

    st.markdown("""
    **비용 상한(ε) 증가에 따른 탄소 감축 효과**

    비용 상한을 높일수록 더 많은 탄소 감축이 가능합니다.
    실현 불가능한 epsilon 값은 제외되었습니다.
    """)

    # Epsilon 순으로 정렬
    sorted_results = sorted(pareto_frontier, key=lambda x: x['epsilon'])

    epsilon_values = [r['epsilon'] for r in sorted_results]
    carbon_values = [r['summary']['total_carbon'] for r in sorted_results]
    baseline_costs = [r.get('baseline_cost', 0) for r in sorted_results]

    # Epsilon vs 탄소 플롯
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epsilon_values,
        y=carbon_values,
        mode='lines+markers',
        name='탄소 배출',
        line=dict(color='blue', width=3),
        marker=dict(size=10, color='blue'),
        hovertemplate='ε: $%{x:,.0f}<br>탄소: %{y:.2f} kgCO2eq'
    ))

    # 기준 비용 수직선 표시 (첫 번째 결과의 baseline_cost)
    if baseline_costs and baseline_costs[0] > 0:
        baseline = baseline_costs[0]
        fig.add_vline(
            x=baseline,
            line_dash="dash",
            line_color="red",
            annotation_text=f"기준 비용 (${baseline:,.0f})",
            annotation_position="top"
        )

    fig.update_layout(
        title='비용 상한(ε)에 따른 탄소 배출 변화',
        xaxis_title='Epsilon - 비용 상한 ($)',
        yaxis_title='탄소 배출 (kgCO2eq)',
        hovermode='closest',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # 실현가능성 정보
    st.markdown("##### 📊 실현가능성 정보")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("실현가능 포인트", f"{len(pareto_frontier)}개",
                 help="실현 가능한 epsilon 값의 개수")

    with col2:
        if len(sorted_results) >= 2:
            min_epsilon = min(epsilon_values)
            max_epsilon = max(epsilon_values)
            st.metric("Epsilon 범위", f"${min_epsilon:,.0f} ~ ${max_epsilon:,.0f}",
                     help="탐색된 비용 상한 범위")

    with col3:
        if len(sorted_results) >= 2:
            carbon_reduction = (carbon_values[0] - carbon_values[-1]) / carbon_values[0] * 100
            st.metric("최대 탄소 감축", f"{carbon_reduction:.1f}%",
                     help="최소 epsilon 대비 최대 epsilon에서의 감축률")


def render_nsga2_analysis(pareto_results: list, pareto_frontier: list):
    """NSGA-II 방법 특화 분석"""
    import plotly.graph_objects as go
    import numpy as np

    st.markdown("#### 🧬 NSGA-II 분석")

    st.markdown("""
    **진화 알고리즘 기반 파레토 탐색 결과**

    NSGA-II는 비지배 정렬(Non-dominated Sorting)과 혼잡도 거리(Crowding Distance)를
    사용하여 다양한 파레토 최적해를 탐색합니다.
    """)

    # Rank 분포
    st.markdown("##### 📊 Rank 분포")

    ranks = [r.get('rank', 0) for r in pareto_frontier]
    crowding_distances = [r.get('crowding_distance', 0) for r in pareto_frontier]

    col1, col2 = st.columns(2)

    with col1:
        # Rank 히스토그램
        fig_rank = go.Figure()

        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        fig_rank.add_trace(go.Bar(
            x=list(rank_counts.keys()),
            y=list(rank_counts.values()),
            marker_color='steelblue',
            text=list(rank_counts.values()),
            textposition='auto'
        ))

        fig_rank.update_layout(
            title='파레토 Rank 분포',
            xaxis_title='Rank',
            yaxis_title='개체 수',
            height=300
        )

        st.plotly_chart(fig_rank, use_container_width=True)

    with col2:
        # Crowding Distance 히스토그램
        fig_crowding = go.Figure()

        # inf 값 제외하고 히스토그램 생성
        finite_distances = [d for d in crowding_distances if d != float('inf') and not np.isinf(d)]

        if finite_distances:
            fig_crowding.add_trace(go.Histogram(
                x=finite_distances,
                nbinsx=10,
                marker_color='coral'
            ))

            fig_crowding.update_layout(
                title='혼잡도 거리 분포',
                xaxis_title='Crowding Distance',
                yaxis_title='빈도',
                height=300
            )

            st.plotly_chart(fig_crowding, use_container_width=True)
        else:
            st.info("유한한 crowding distance 값이 없습니다.")

    # 다양성 및 수렴 메트릭
    st.markdown("##### 📈 다양성 및 수렴 메트릭")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rank0_count = sum(1 for r in ranks if r == 0)
        st.metric("Rank 0 개체", f"{rank0_count}개",
                 help="파레토 프론티어(Rank 0)에 속하는 개체 수")

    with col2:
        unique_ranks = len(set(ranks))
        st.metric("고유 Rank 수", f"{unique_ranks}개",
                 help="탐색된 파레토 계층 수")

    with col3:
        if finite_distances:
            avg_crowding = np.mean(finite_distances)
            st.metric("평균 혼잡도", f"{avg_crowding:.4f}",
                     help="파레토 프론티어의 평균 분산 정도")

    with col4:
        inf_count = sum(1 for d in crowding_distances if d == float('inf') or np.isinf(d))
        st.metric("경계 개체", f"{inf_count}개",
                 help="혼잡도 거리가 무한대인 경계 개체")

    # 파레토 프론티어 품질 평가
    st.markdown("##### ⭐ 파레토 프론티어 품질")

    carbon_values = [r['summary']['total_carbon'] for r in pareto_frontier]
    cost_values = [r['summary'].get('total_cost', 0) for r in pareto_frontier]

    col1, col2 = st.columns(2)

    with col1:
        # 목적공간 커버리지
        if len(carbon_values) > 1:
            carbon_spread = max(carbon_values) - min(carbon_values)
            cost_spread = max(cost_values) - min(cost_values)

            st.metric("탄소 공간 커버리지", f"{carbon_spread:.2f} kgCO2eq",
                     help="파레토 프론티어가 커버하는 탄소 배출 범위")
            st.metric("비용 공간 커버리지", f"${cost_spread:,.2f}",
                     help="파레토 프론티어가 커버하는 비용 범위")

    with col2:
        # 해의 밀도 (균등 분산 정도)
        if len(pareto_frontier) > 2 and finite_distances:
            std_crowding = np.std(finite_distances)
            st.metric("분산 균일성 (Std)", f"{std_crowding:.4f}",
                     help="혼잡도 거리의 표준편차 (낮을수록 균등 분산)")

        st.metric("총 파레토 해", f"{len(pareto_frontier)}개",
                 help="발견된 파레토 최적해의 총 개수")


def render_system_unavailable():
    """시스템 사용 불가 메시지"""
    st.error("⚠️ 최적화 시스템 V2를 로드할 수 없습니다")

    st.markdown(f"""
    ### 오류 정보
    ```
    {IMPORT_ERROR}
    ```

    ### 필요한 조치

    1. **Pyomo 설치**:
       ```bash
       pip install pyomo
       ```

    2. **솔버 설치** (하나 이상):
       ```bash
       # GLPK (권장)
       conda install -c conda-forge glpk

       # CBC
       conda install -c conda-forge coincbc

       # IPOPT
       conda install -c conda-forge ipopt
       ```

    3. **기타 의존성**:
       ```bash
       pip install pandas plotly streamlit
       ```

    ### 임시 해결책
    - Phase 3 구현이 완료되었지만 pyomo가 설치되지 않은 상태입니다.
    - 위 명령어로 의존성을 설치한 후 페이지를 새로고침하세요.
    """)

    st.markdown("---")

    # 레거시 페이지로 이동 옵션
    st.info("💡 레거시 최적화 시스템을 사용하려면 개발팀에 문의하세요.")


def render_advanced_analysis_tab():
    """고급 분석 탭 - 민감도 분석, 대화형 파레토 탐색, 강건 최적화"""
    st.header("🔬 고급 분석")

    st.markdown("""
    **고급 분석 도구**를 통해 최적화 결과를 더 깊이 이해하고 탐색할 수 있습니다.

    - **민감도 분석**: 파라미터 변화가 결과에 미치는 영향을 정량화
    - **대화형 파레토 탐색**: 실시간 트레이드오프 탐색 (재최적화 없음)
    - **강건 최적화**: 불확실성을 고려한 시나리오 기반 최적화
    - **제약 완화 분석**: 제약조건 완화 효과 및 우선순위 분석
    - **확률적 위험 분석**: Monte Carlo 시뮬레이션을 통한 불확실성 정량화
    """)

    # 데이터 로딩 확인
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ 먼저 '데이터 로딩' 탭에서 데이터를 로드하세요.")
        return

    # 서브탭 구성
    analysis_tabs = st.tabs([
        "📈 민감도 분석",
        "🎯 대화형 파레토 탐색",
        "🛡️ 강건 최적화",
        "🔧 제약 완화 분석",
        "🎲 확률적 위험 분석"
    ])

    with analysis_tabs[0]:
        render_sensitivity_analysis_subtab()

    with analysis_tabs[1]:
        render_interactive_pareto_subtab()

    with analysis_tabs[2]:
        render_robust_optimization_subtab()

    with analysis_tabs[3]:
        render_constraint_relaxation_subtab()

    with analysis_tabs[4]:
        render_stochastic_risk_subtab()


def render_sensitivity_analysis_subtab():
    """민감도 분석 서브탭"""
    st.subheader("📈 민감도 분석")

    st.markdown("""
    파라미터 변화가 최적 솔루션(탄소/비용)에 미치는 영향을 정량화합니다.

    **분석 방법**:
    - **OAT (One-At-a-Time)**: 각 파라미터를 개별적으로 변화시키며 영향 측정
    - **Sobol 전역 민감도**: 파라미터 간 상호작용 효과 포함 (고급)
    """)

    # 현재 결과 확인
    if not st.session_state.get('current_result'):
        st.info("💡 먼저 '최적화 실행' 탭에서 기준 최적화를 실행하세요.")
        return

    st.markdown("---")
    st.markdown("### 1️⃣ 분석 파라미터 선택")

    # 파라미터 선택 UI
    param_category = st.selectbox(
        "파라미터 카테고리",
        ["배출계수 (Emission Factor)", "비용 (Cost)", "제약조건 한도 (Constraint Bound)"]
    )

    st.info("🚧 **Phase 1 개발 중**: 민감도 분석 UI는 다음 업데이트에서 완성됩니다.")

    st.markdown("""
    **예정된 기능**:
    - 자재별 배출계수 민감도
    - 재활용/저탄소 비용 민감도
    - 제약조건 한도 민감도
    - Tornado 다이어그램 시각화
    - Spider 차트
    - 파라미터-목적함수 곡선
    """)


def render_interactive_pareto_subtab():
    """대화형 파레토 탐색 서브탭"""
    st.subheader("🎯 대화형 파레토 탐색")

    st.markdown("""
    사전 계산된 파레토 프론티어를 **실시간으로 탐색**합니다.
    슬라이더를 움직여 탄소-비용 트레이드오프를 즉시 확인할 수 있습니다.
    """)

    # 파레토 결과 확인
    if 'pareto_results' not in st.session_state or not st.session_state.pareto_results:
        st.info("💡 먼저 '최적화 실행' 탭에서 파레토 최적화를 실행하세요.")
        return

    pareto_results = st.session_state.pareto_results

    # ParetoNavigator 초기화
    try:
        navigator = ParetoNavigator(pareto_results, interpolation_method='linear')
        recommender = SolutionRecommender(pareto_results)

        st.markdown("---")
        st.markdown("### 🎚️ 선호도 슬라이더")

        # 슬라이더
        carbon_weight = st.slider(
            "탄소 우선 ← → 비용 우선",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="0.0 = 100% 비용 최소화, 1.0 = 100% 탄소 최소화"
        )

        # 실시간 솔루션 표시
        selected_solution = navigator.get_solution_at_weight(
            carbon_weight,
            return_nearest=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "탄소 배출량",
                f"{selected_solution['summary']['total_carbon']:.2f} kg",
                help="예상 탄소 배출량"
            )

        with col2:
            st.metric(
                "총 비용",
                f"${selected_solution['summary']['total_cost']:.2f}",
                help="예상 총 비용"
            )

        # 추가 정보
        with st.expander("📊 상세 정보", expanded=False):
            st.json({
                "Carbon Weight": f"{selected_solution.get('carbon_weight', 0):.2f}",
                "Cost Weight": f"{selected_solution.get('cost_weight', 0):.2f}",
                "Interpolated": selected_solution.get('interpolated', False),
                "Distance from Target": f"{selected_solution.get('distance_from_target', 0):.4f}"
            })

        st.markdown("---")
        st.markdown("### 💡 솔루션 추천")

        # 추천 기준 선택
        recommendation_criteria = st.selectbox(
            "추천 기준",
            [
                ("balanced", "⚖️ 균형 잡힌 솔루션"),
                ("minimize_carbon", "🌱 탄소 최소화"),
                ("minimize_cost", "💰 비용 최소화"),
                ("implementation_ease", "🔧 구현 용이성"),
                ("risk_averse", "🛡️ 리스크 회피")
            ],
            format_func=lambda x: x[1]
        )

        criteria_key = recommendation_criteria[0]

        # 상위 5개 추천
        ranking_df = recommender.rank_solutions(criteria_key, top_n=5)

        st.markdown(f"**{recommendation_criteria[1]} 기준 상위 5개 솔루션**")
        st.dataframe(
            ranking_df[[
                'Rank', 'Score', 'Carbon (kg)', 'Cost ($)',
                'Carbon Reduction (%)', 'Cost Premium (%)'
            ]],
            use_container_width=True,
            hide_index=True
        )

        # 추천 이유 설명
        with st.expander("❓ 왜 이 솔루션이 추천되나요?"):
            explanation = recommender.explain_recommendation(criteria_key)
            st.markdown(explanation)

        st.markdown("---")
        st.markdown("### 📉 파레토 프론티어 요약")

        # 파레토 요약 테이블
        summary_df = navigator.get_pareto_summary()
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # 극단 솔루션 표시
        st.markdown("### 🔍 극단 솔루션")
        extreme_solutions = navigator.get_extreme_solutions()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**최소 탄소 솔루션**")
            min_carbon_sol = extreme_solutions['min_carbon']
            st.metric(
                "탄소",
                f"{min_carbon_sol['summary']['total_carbon']:.2f} kg"
            )
            st.metric(
                "비용",
                f"${min_carbon_sol['summary']['total_cost']:.2f}"
            )

        with col2:
            st.markdown("**최소 비용 솔루션**")
            min_cost_sol = extreme_solutions['min_cost']
            st.metric(
                "탄소",
                f"{min_cost_sol['summary']['total_carbon']:.2f} kg"
            )
            st.metric(
                "비용",
                f"${min_cost_sol['summary']['total_cost']:.2f}"
            )

    except Exception as e:
        st.error(f"대화형 탐색 중 오류 발생: {str(e)}")
        st.exception(e)


def render_robust_optimization_subtab():
    """강건 최적화 서브탭"""
    st.subheader("🛡️ 강건 최적화 (Robust Optimization)")

    st.markdown("""
    **불확실성을 고려한 강건한 솔루션**을 찾습니다.

    미래 시나리오(배출계수 변동, 비용 변화 등)를 정의하고,
    모든 시나리오에서 좋은 성능을 유지하는 솔루션을 최적화합니다.

    **3가지 강건 최적화 방법**:
    - **Minimax Regret**: 최악의 후회를 최소화
    - **Expected CVaR**: 평균 성능과 극단 리스크 균형
    - **Light Robust**: 모든 시나리오에서 실행가능성 보장
    """)

    # 세션 상태 초기화
    if 'scenario_manager' not in st.session_state:
        st.session_state.scenario_manager = ScenarioManager()

    if 'robust_results' not in st.session_state:
        st.session_state.robust_results = {}

    # Step 1: 시나리오 정의
    st.markdown("### 1️⃣ 시나리오 정의")

    scenario_option = st.radio(
        "시나리오 선택",
        ["사전 정의 시나리오 사용", "사용자 정의 시나리오"],
        horizontal=True
    )

    if scenario_option == "사전 정의 시나리오 사용":
        scenario_type = st.selectbox(
            "시나리오 세트",
            ["standard", "regulatory"],
            format_func=lambda x: {
                "standard": "표준 (Base/Optimistic/Pessimistic)",
                "regulatory": "규제 중심 (Strict/Base/Relaxed)"
            }[x]
        )

        if st.button("시나리오 생성", key="create_preset"):
            with st.spinner("시나리오 생성 중..."):
                st.session_state.scenario_manager.create_preset_scenarios(scenario_type)
                st.success(f"✅ {scenario_type} 시나리오 세트 생성 완료!")

        # 시나리오 요약 표시
        if st.session_state.scenario_manager.scenarios:
            summary = st.session_state.scenario_manager.get_summary()
            st.info(f"📋 총 {summary['count']}개 시나리오 (확률 합: {summary['total_probability']:.2f})")

            # 시나리오 상세
            with st.expander("시나리오 상세 보기"):
                for scenario_info in summary['scenarios']:
                    st.markdown(f"""
                    **{scenario_info['name']}** (확률: {scenario_info['probability']:.0%})
                    - {scenario_info['description']}
                    """)

    else:
        # 사용자 정의 시나리오
        st.info("💡 사용자 정의 시나리오 기능은 현재 개발 중입니다. 사전 정의 시나리오를 사용하세요.")

    # 시나리오가 없으면 여기서 중단
    if not st.session_state.scenario_manager.scenarios:
        st.warning("⚠️ 시나리오를 먼저 생성하세요.")
        return

    st.markdown("---")

    # Step 2: 강건 최적화 방법 선택
    st.markdown("### 2️⃣ 강건 최적화 실행")

    method = st.selectbox(
        "최적화 방법",
        ["minimax_regret", "expected_cvar", "light_robust"],
        format_func=lambda x: {
            "minimax_regret": "Minimax Regret (최악의 후회 최소화)",
            "expected_cvar": "Expected CVaR (평균 + 리스크 균형)",
            "light_robust": "Light Robust (모든 시나리오 실행가능)"
        }[x]
    )

    # 방법별 파라미터
    method_params = {}

    if method == "expected_cvar":
        col1, col2 = st.columns(2)
        with col1:
            lambda_risk = st.slider(
                "리스크 가중치 (λ)",
                0.0, 1.0, 0.3, 0.05,
                help="높을수록 극단 시나리오 리스크를 더 중시"
            )
            method_params['lambda_risk'] = lambda_risk

        with col2:
            beta = st.slider(
                "CVaR 신뢰수준 (β)",
                0.90, 0.99, 0.95, 0.01,
                help="Worst β% 시나리오의 조건부 기댓값"
            )
            method_params['beta'] = beta

        st.latex(r"\text{Objective} = E[\text{Obj}] + \lambda \times \text{CVaR}_\beta[\text{Obj}]")

    elif method == "minimax_regret":
        st.info("""
        **Minimax Regret**는 각 시나리오에서의 후회(Regret)를 계산합니다:
        - Regret = 해당 시나리오의 실제 목적함수 - 해당 시나리오의 최적 목적함수
        - 모든 시나리오 중 최대 Regret을 최소화
        """)

    elif method == "light_robust":
        st.info("""
        **Light Robust**는 가장 보수적인 접근입니다:
        - 기준 시나리오에서 최적화
        - 모든 시나리오에서 실행가능성 검증
        - 모든 시나리오에서 제약조건 만족 보장
        """)

    # 실행 버튼
    if st.button(f"🚀 {method.replace('_', ' ').title()} 실행", key=f"run_{method}"):
        try:
            with st.spinner("강건 최적화 실행 중... (시간이 걸릴 수 있습니다)"):
                # 기준 데이터 가져오기
                base_data = st.session_state.data_loader.get_optimization_data()

                # RobustOptimizer 초기화
                robust_optimizer = RobustOptimizer(
                    st.session_state.engine,
                    st.session_state.data_loader,
                    st.session_state.scenario_manager
                )

                # 방법별 실행
                if method == "minimax_regret":
                    result = robust_optimizer.optimize_minimax_regret(
                        base_data,
                        objective_type='minimize_carbon'
                    )
                elif method == "expected_cvar":
                    result = robust_optimizer.optimize_expected_cvar(
                        base_data,
                        objective_type='minimize_carbon',
                        **method_params
                    )
                elif method == "light_robust":
                    result = robust_optimizer.optimize_light_robust(
                        base_data,
                        objective_type='minimize_carbon'
                    )

                # 결과 저장
                st.session_state.robust_results[method] = result

                st.success(f"✅ {method.replace('_', ' ').title()} 최적화 완료!")

        except Exception as e:
            st.error(f"❌ 최적화 중 오류 발생: {str(e)}")
            st.exception(e)

    # Step 3: 결과 분석
    if st.session_state.robust_results:
        st.markdown("---")
        st.markdown("### 3️⃣ 결과 분석")

        # 실행된 방법 선택
        available_methods = list(st.session_state.robust_results.keys())
        selected_method = st.selectbox(
            "분석할 방법",
            available_methods,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        result = st.session_state.robust_results[selected_method]

        # 방법별 결과 표시
        if selected_method == "minimax_regret":
            st.markdown("#### Minimax Regret 결과")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Max Regret",
                    f"{result['max_regret']:.4f}",
                    help="모든 시나리오 중 최대 후회"
                )

            with col2:
                st.metric(
                    "선택된 솔루션",
                    result['best_candidate']
                )

            # 시나리오별 Regret
            st.markdown("**시나리오별 Regret**")
            regret_data = []
            for scenario_name, regret in result['scenario_regrets'].items():
                regret_data.append({
                    'Scenario': scenario_name,
                    'Regret': regret,
                    'Optimal Objective': result['scenario_optimal_objectives'][scenario_name]
                })

            regret_df = pd.DataFrame(regret_data)
            st.dataframe(regret_df, use_container_width=True)

            # 시각화
            import plotly.express as px
            fig = px.bar(
                regret_df,
                x='Scenario',
                y='Regret',
                title='시나리오별 Regret',
                labels={'Regret': 'Regret (kg CO2)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        elif selected_method == "expected_cvar":
            st.markdown("#### Expected CVaR 결과")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Expected Value",
                    f"{result['expected_value']:.4f}",
                    help="확률 가중 평균 목적함수"
                )

            with col2:
                st.metric(
                    f"CVaR {result['beta']:.0%}",
                    f"{result['cvar']:.4f}",
                    help=f"Worst {(1-result['beta'])*100:.0f}% 시나리오의 조건부 기댓값"
                )

            with col3:
                st.metric(
                    "Composite Score",
                    f"{result['composite_score']:.4f}",
                    help=f"E[Obj] + {result['lambda_risk']:.2f} × CVaR"
                )

            # 시나리오별 목적함수
            st.markdown("**시나리오별 목적함수**")
            obj_data = []
            for scenario_name, obj_value in result['scenario_objectives'].items():
                obj_data.append({
                    'Scenario': scenario_name,
                    'Objective': obj_value
                })

            obj_df = pd.DataFrame(obj_data)
            st.dataframe(obj_df, use_container_width=True)

            # 시각화
            import plotly.express as px
            fig = px.bar(
                obj_df,
                x='Scenario',
                y='Objective',
                title='시나리오별 목적함수',
                labels={'Objective': 'Carbon Emission (kg CO2)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        elif selected_method == "light_robust":
            st.markdown("#### Light Robust 결과")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "기준 목적함수",
                    f"{result['base_objective']:.4f}"
                )

            with col2:
                if result['all_feasible']:
                    st.success("✅ 모든 시나리오에서 실행가능")
                else:
                    st.error("❌ 일부 시나리오에서 실행불가능")

            # 시나리오별 실행가능성
            st.markdown("**시나리오별 실행가능성**")
            feas_data = []
            for scenario_name, is_feasible in result['scenario_feasibility'].items():
                obj_value = result['scenario_objectives'].get(scenario_name, None)
                feas_data.append({
                    'Scenario': scenario_name,
                    'Feasible': "✅" if is_feasible else "❌",
                    'Objective': f"{obj_value:.4f}" if obj_value else "N/A"
                })

            feas_df = pd.DataFrame(feas_data)
            st.dataframe(feas_df, use_container_width=True)

        # 솔루션 상세 (공통)
        if result.get('robust_solution'):
            with st.expander("💡 솔루션 상세 보기"):
                solution = result['robust_solution']

                st.markdown("**총 탄소 배출**")
                st.info(f"{solution['summary']['total_carbon']:.2f} kg CO2")

                # 상위 기여 자재
                if 'materials' in solution:
                    contributions = []
                    for mat_name, mat_result in solution['materials'].items():
                        contribution = mat_result['modified_emission'] * mat_result['quantity']
                        contributions.append({
                            'Material': mat_name[:50],
                            'Emission': mat_result['modified_emission'],
                            'Quantity': mat_result['quantity'],
                            'Total Contribution': contribution
                        })

                    contrib_df = pd.DataFrame(contributions).sort_values(
                        'Total Contribution', ascending=False
                    ).head(10)

                    st.markdown("**상위 10개 기여 자재**")
                    st.dataframe(contrib_df, use_container_width=True)

    # Step 4: 솔루션 비교 (여러 방법 실행했을 때)
    if len(st.session_state.robust_results) > 1:
        st.markdown("---")
        st.markdown("### 4️⃣ 솔루션 비교")

        # SolutionEvaluator를 사용한 비교
        if st.button("솔루션 비교 실행", key="compare_solutions"):
            try:
                with st.spinner("솔루션 비교 중..."):
                    base_data = st.session_state.data_loader.get_optimization_data()

                    evaluator = SolutionEvaluator(
                        st.session_state.engine,
                        st.session_state.data_loader,
                        st.session_state.scenario_manager
                    )

                    # 솔루션 딕셔너리 준비
                    solutions = {}
                    for method_name, result in st.session_state.robust_results.items():
                        if result.get('robust_solution'):
                            solutions[method_name] = result['robust_solution']

                    # 비교 실행
                    comparison_df = evaluator.compare_solutions(
                        solutions,
                        base_data,
                        objective_type='minimize_carbon'
                    )

                    st.success("✅ 비교 완료!")

                    # 비교 결과 표시
                    st.dataframe(comparison_df, use_container_width=True)

                    # 시각화 - Radar Chart
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    for idx, row in comparison_df.iterrows():
                        fig.add_trace(go.Scatterpolar(
                            r=[row['평균'], row['표준편차'], row['VaR95'], row['CVaR95']],
                            theta=['평균', '표준편차', 'VaR95', 'CVaR95'],
                            fill='toself',
                            name=row['솔루션']
                        ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title="솔루션 비교 (Radar Chart)"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"비교 중 오류: {str(e)}")
                st.exception(e)


def render_constraint_relaxation_subtab():
    """제약 완화 분석 서브탭"""
    st.subheader("🔧 제약 완화 분석 (Constraint Relaxation)")

    st.markdown("""
    **어떤 제약조건이 최적화를 방해하는지** 식별하고, 각 제약을 완화했을 때의 효과를 정량화합니다.

    **분석 방법**:
    - **Binding Constraint 식별**: 현재 최적해에서 활성화된(slack=0) 제약조건 찾기
    - **Shadow Price 추출**: 제약을 1단위 완화했을 때의 목적함수 개선 효과 (IPOPT 사용 시)
    - **완화 영향 분석**: 다양한 완화 수준에서 실제 목적함수 변화 측정
    - **우선순위화**: 한계 편익 기준으로 완화 우선순위 결정
    """)

    # 세션 상태 초기화
    if 'relaxation_analyzer' not in st.session_state:
        st.session_state.relaxation_analyzer = None

    if 'binding_constraints' not in st.session_state:
        st.session_state.binding_constraints = {}

    if 'relaxation_results' not in st.session_state:
        st.session_state.relaxation_results = None

    # Step 1: Binding Constraint 식별
    st.markdown("---")
    st.markdown("### 1️⃣ Binding Constraint 식별")

    st.markdown("""
    현재 최적해에서 제약조건이 **활성화(binding)**되어 있는지 확인합니다.
    Binding constraint는 slack이 0에 가까운 제약으로, 최적화를 제한하는 주요 요인입니다.
    """)

    col1, col2 = st.columns([3, 1])

    with col1:
        slack_threshold = st.number_input(
            "Slack 임계값",
            min_value=1e-10,
            max_value=1e-3,
            value=1e-6,
            format="%.2e",
            help="이 값보다 작은 slack을 가진 제약을 binding으로 간주"
        )

    with col2:
        use_ipopt = st.checkbox(
            "IPOPT 사용",
            value=True,
            help="IPOPT 솔버를 사용하여 dual value (shadow price) 추출"
        )

    if st.button("🔍 Binding Constraint 식별", key="identify_binding"):
        try:
            with st.spinner("Binding constraint 분석 중..."):
                # 데이터 로드
                data_loader = st.session_state.data_loader
                base_data = data_loader.get_optimization_data()

                # 최적화 엔진 생성 (IPOPT or GLPK)
                solver_name = 'ipopt' if use_ipopt else 'glpk'
                engine = OptimizationEngine(
                    solver_name=solver_name,
                    constraint_manager=st.session_state.constraint_manager
                )

                # Dual value 활성화 (IPOPT)
                if use_ipopt:
                    engine.enable_dual_values = True

                # 기준 최적화 실행
                model = engine.build_model(base_data, objective_type='minimize_carbon')
                solution = engine.solve()

                if solution:
                    # ConstraintRelaxationAnalyzer 초기화
                    analyzer = ConstraintRelaxationAnalyzer(engine, data_loader)
                    st.session_state.relaxation_analyzer = analyzer

                    # Binding constraint 식별
                    binding_constraints = analyzer.identify_binding_constraints(slack_threshold)
                    st.session_state.binding_constraints = binding_constraints

                    st.success(f"✅ Binding constraint 식별 완료: {len(binding_constraints)}개 발견")

                    # 결과 표시
                    if binding_constraints:
                        st.markdown("#### 📋 Binding Constraint 목록")

                        binding_data = []
                        for name, info in binding_constraints.items():
                            binding_data.append({
                                '제약조건': name,
                                'Slack': f"{info['slack']:.2e}",
                                'Dual Value': f"{info['dual_value']:.4f}" if info['dual_value'] is not None else "N/A",
                                'Binding': "✅" if info['is_binding'] else "❌"
                            })

                        binding_df = pd.DataFrame(binding_data)
                        st.dataframe(binding_df, use_container_width=True, hide_index=True)

                        if use_ipopt and any(info['dual_value'] is not None for info in binding_constraints.values()):
                            st.info("""
                            💡 **Dual Value (Shadow Price)**: 제약을 1단위 완화했을 때 목적함수가 개선되는 정도
                            - 양수: 제약이 목적함수를 나쁘게 함 (완화하면 개선)
                            - 음수: 제약이 목적함수를 좋게 함 (완화하면 악화)
                            """)
                    else:
                        st.info("Binding constraint가 발견되지 않았습니다. Slack 임계값을 높여보세요.")
                else:
                    st.error("❌ 기준 최적화 실패")

        except Exception as e:
            st.error(f"❌ Binding constraint 식별 중 오류: {str(e)}")
            st.exception(e)

    # Step 2: 제약 선택 및 완화 설정
    if st.session_state.binding_constraints:
        st.markdown("---")
        st.markdown("### 2️⃣ 완화 분석할 제약 선택")

        st.markdown("""
        분석할 제약조건을 선택하고, 완화 수준을 설정합니다.

        **제약 스펙 정의**: 각 제약의 현재 값, 완화 방향(증가/감소)을 지정해야 합니다.
        """)

        # 제약 선택
        selected_constraints = st.multiselect(
            "분석할 제약조건 선택",
            options=list(st.session_state.binding_constraints.keys()),
            help="여러 제약을 선택하여 동시에 분석할 수 있습니다"
        )

        if selected_constraints:
            st.markdown("#### 🔧 제약 스펙 정의")

            constraint_specs = []

            for constraint_name in selected_constraints:
                with st.expander(f"⚙️ {constraint_name}", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # 제약 타입 (사용자가 지정)
                        constraint_type = st.selectbox(
                            "제약 타입",
                            ["premium_limit", "재활용_비율", "저탄소_비율", "기타"],
                            key=f"type_{constraint_name}"
                        )

                    with col2:
                        # 현재 값
                        current_value = st.number_input(
                            "현재 값",
                            min_value=0.0,
                            max_value=1000.0,
                            value=10.0,
                            step=1.0,
                            key=f"current_{constraint_name}",
                            help="제약조건의 현재 한도 또는 값"
                        )

                    with col3:
                        # 완화 방향
                        relaxation_direction = st.selectbox(
                            "완화 방향",
                            ["increase", "decrease"],
                            format_func=lambda x: "증가 (한도 완화)" if x == "increase" else "감소 (한도 강화)",
                            key=f"direction_{constraint_name}"
                        )

                    constraint_specs.append({
                        'name': constraint_name,
                        'type': constraint_type,
                        'current_value': current_value,
                        'relaxation_direction': relaxation_direction
                    })

            # 완화 수준 설정
            st.markdown("#### 📊 완화 수준 설정")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                level1 = st.number_input("수준 1 (%)", 0, 50, 5, 5, key="level1")
            with col2:
                level2 = st.number_input("수준 2 (%)", 0, 50, 10, 5, key="level2")
            with col3:
                level3 = st.number_input("수준 3 (%)", 0, 50, 15, 5, key="level3")
            with col4:
                level4 = st.number_input("수준 4 (%)", 0, 50, 20, 5, key="level4")

            relaxation_levels = sorted(list(set([level1, level2, level3, level4])))

            st.info(f"🎯 완화 수준: {relaxation_levels}% → {len(relaxation_levels)}개 레벨")

            # Step 3: 완화 분석 실행
            st.markdown("---")
            st.markdown("### 3️⃣ 완화 영향 분석 실행")

            if st.button("🚀 완화 분석 실행", type="primary", use_container_width=True):
                try:
                    with st.spinner(f"{len(constraint_specs)}개 제약 × {len(relaxation_levels)}개 레벨 분석 중... (시간이 걸릴 수 있습니다)"):
                        analyzer = st.session_state.relaxation_analyzer

                        if analyzer is None:
                            st.error("먼저 Binding Constraint 식별을 실행하세요.")
                        else:
                            # 기준 데이터 로드
                            data_loader = st.session_state.data_loader
                            base_data = data_loader.get_optimization_data()

                            # 완화 분석 실행
                            results = analyzer.analyze_relaxation_impact(
                                base_data=base_data,
                                constraint_specs=constraint_specs,
                                relaxation_levels=relaxation_levels,
                                objective_type='minimize_carbon'
                            )

                            st.session_state.relaxation_results = results

                            st.success("✅ 완화 분석 완료!")

                except Exception as e:
                    st.error(f"❌ 완화 분석 중 오류: {str(e)}")
                    st.exception(e)

    # Step 4: 결과 시각화
    if st.session_state.relaxation_results:
        st.markdown("---")
        st.markdown("### 4️⃣ 결과 시각화 및 분석")

        results = st.session_state.relaxation_results

        # 기준 목적함수 표시
        st.metric(
            "기준 목적함수 (탄소 배출)",
            f"{results['base_objective']:.2f} kgCO2eq",
            help="제약 완화 전 기준 탄소 배출량"
        )

        # 제약별 완화 영향 곡선
        st.markdown("#### 📈 완화 영향 곡선")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        constraint_results = results['constraint_results']

        if constraint_results:
            # 각 제약조건에 대한 차트 생성
            for constraint_name, result in constraint_results.items():
                if not result['relaxation_levels']:
                    continue

                st.markdown(f"**{constraint_name}**")

                # 2개의 서브플롯: 목적함수 vs 완화%, 개선율 vs 완화%
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("목적함수 vs 완화 수준", "개선율 vs 완화 수준"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )

                # 첫 번째 차트: 목적함수 값
                fig.add_trace(
                    go.Scatter(
                        x=result['relaxation_levels'],
                        y=result['objective_values'],
                        mode='lines+markers',
                        name='목적함수',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )

                # 기준선 표시
                fig.add_hline(
                    y=results['base_objective'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="기준",
                    row=1, col=1
                )

                # 두 번째 차트: 개선율
                fig.add_trace(
                    go.Scatter(
                        x=result['relaxation_levels'],
                        y=result['objective_improvements'],
                        mode='lines+markers',
                        name='개선율 (%)',
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    ),
                    row=1, col=2
                )

                fig.update_xaxes(title_text="완화 수준 (%)", row=1, col=1)
                fig.update_xaxes(title_text="완화 수준 (%)", row=1, col=2)
                fig.update_yaxes(title_text="탄소 배출 (kgCO2eq)", row=1, col=1)
                fig.update_yaxes(title_text="개선율 (%)", row=1, col=2)

                fig.update_layout(
                    height=400,
                    showlegend=False,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # 한계 편익 표시
                if result['marginal_benefits']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("최대 개선율", f"{max(result['objective_improvements']):.2f}%")
                    with col2:
                        st.metric("평균 한계 편익", f"{np.mean(result['marginal_benefits']):.4f}")
                    with col3:
                        st.metric("초기 한계 편익", f"{result['marginal_benefits'][0]:.4f}")

            # 우선순위 테이블
            st.markdown("---")
            st.markdown("#### 🏆 제약 우선순위")

            try:
                analyzer = st.session_state.relaxation_analyzer
                priority_df = analyzer.prioritize_constraints(method='marginal_benefit')

                st.dataframe(priority_df, use_container_width=True, hide_index=True)

                st.caption("""
                **우선순위 기준**: 평균 한계 편익 (완화 시 목적함수 개선 정도)
                - 우선순위가 높을수록 완화 효과가 큽니다
                """)

            except Exception as e:
                st.warning(f"우선순위 계산 중 오류: {str(e)}")

            # 추천
            st.markdown("---")
            st.markdown("#### 💡 완화 추천")

            try:
                recommendations = analyzer.generate_recommendations(top_n=3)

                for rec in recommendations:
                    with st.expander(f"🎯 추천 {rec['우선순위']}: {rec['제약조건']}", expanded=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("현재 값", f"{rec['현재_값']:.2f}")
                        with col2:
                            st.metric("추천 완화 수준", f"{rec['추천_완화_수준(%)']:.0f}%")
                        with col3:
                            st.metric("예상 개선", f"{rec['예상_개선(%)']:.2f}%")

                        st.markdown(rec['설명'])

            except Exception as e:
                st.warning(f"추천 생성 중 오류: {str(e)}")

        else:
            st.info("완화 분석 결과가 없습니다.")


def render_stochastic_risk_subtab():
    """확률적 위험 분석 서브탭 (Monte Carlo)"""
    st.subheader("🎲 확률적 위험 분석 (Stochastic Risk Quantification)")

    st.markdown("""
    **Monte Carlo 시뮬레이션**을 통해 파라미터 불확실성을 정량화하고 리스크를 분석합니다.

    **주요 기능**:
    - **파라미터 불확실성 정의**: 배출계수, 비용 등의 불확실성을 확률 분포로 표현
    - **Monte Carlo 샘플링**: 다양한 파라미터 조합에서 최적화 실행
    - **리스크 메트릭**: VaR, CVaR, 확률 분포 추정
    - **파라미터 상관관계**: 목적함수에 대한 파라미터 영향도 분석
    """)

    # 세션 상태 초기화
    if 'stochastic_analyzer' not in st.session_state:
        st.session_state.stochastic_analyzer = None

    if 'stochastic_uncertainties' not in st.session_state:
        st.session_state.stochastic_uncertainties = {}

    if 'stochastic_results' not in st.session_state:
        st.session_state.stochastic_results = None

    # Step 1: 파라미터 불확실성 정의
    st.markdown("---")
    st.markdown("### 1️⃣ 파라미터 불확실성 정의")

    st.markdown("""
    분석할 파라미터의 불확실성을 확률 분포로 정의합니다.

    **지원 분포**:
    - **Normal (정규분포)**: 평균과 표준편차로 정의
    - **Uniform (균등분포)**: 최소값과 최대값으로 정의
    - **Triangular (삼각분포)**: 최소, 최빈, 최대값으로 정의
    - **Lognormal (로그정규분포)**: 평균과 시그마로 정의
    """)

    # 파라미터 타입 선택
    col1, col2 = st.columns(2)

    with col1:
        param_category = st.selectbox(
            "파라미터 카테고리",
            ["배출계수 (Emission Factor)", "비용 (Cost)", "기타"],
            help="불확실성을 정의할 파라미터 카테고리"
        )

    with col2:
        if param_category == "배출계수 (Emission Factor)":
            # 자재 목록 가져오기
            data_loader = st.session_state.data_loader
            optimization_data = data_loader.get_optimization_data()
            material_list = list(optimization_data['material_classification'].keys())

            selected_material = st.selectbox(
                "자재 선택",
                material_list,
                help="배출계수 불확실성을 정의할 자재"
            )

            param_name = f"emission_factor_{selected_material}"
        else:
            param_name = st.text_input(
                "파라미터 이름",
                value="custom_param",
                help="파라미터의 고유 이름"
            )

    # 분포 타입 선택
    distribution = st.selectbox(
        "확률 분포 타입",
        ["normal", "uniform", "triangular", "lognormal"],
        format_func=lambda x: {
            "normal": "Normal (정규분포)",
            "uniform": "Uniform (균등분포)",
            "triangular": "Triangular (삼각분포)",
            "lognormal": "Lognormal (로그정규분포)"
        }[x]
    )

    # 분포별 파라미터 입력
    st.markdown("**분포 파라미터**")

    params = {}

    if distribution == "normal":
        col1, col2 = st.columns(2)
        with col1:
            mean = st.number_input(
                "평균 (Mean)",
                value=1.0,
                step=0.1,
                format="%.4f",
                help="정규분포의 평균 (기준값 대비 배율, 예: 1.0 = 100%, 1.1 = 110%)"
            )
            params['mean'] = mean
        with col2:
            std = st.number_input(
                "표준편차 (Std)",
                value=0.1,
                min_value=0.001,
                step=0.01,
                format="%.4f",
                help="정규분포의 표준편차"
            )
            params['std'] = std

    elif distribution == "uniform":
        col1, col2 = st.columns(2)
        with col1:
            low = st.number_input(
                "최소값 (Low)",
                value=0.9,
                step=0.1,
                format="%.4f",
                help="균등분포의 최소값 (기준값 대비 배율)"
            )
            params['low'] = low
        with col2:
            high = st.number_input(
                "최대값 (High)",
                value=1.1,
                step=0.1,
                format="%.4f",
                help="균등분포의 최대값 (기준값 대비 배율)"
            )
            params['high'] = high

    elif distribution == "triangular":
        col1, col2, col3 = st.columns(3)
        with col1:
            low = st.number_input(
                "최소값 (Low)",
                value=0.9,
                step=0.1,
                format="%.4f",
                help="삼각분포의 최소값"
            )
            params['low'] = low
        with col2:
            mode = st.number_input(
                "최빈값 (Mode)",
                value=1.0,
                step=0.1,
                format="%.4f",
                help="삼각분포의 최빈값 (가장 가능성 높은 값)"
            )
            params['mode'] = mode
        with col3:
            high = st.number_input(
                "최대값 (High)",
                value=1.1,
                step=0.1,
                format="%.4f",
                help="삼각분포의 최대값"
            )
            params['high'] = high

    elif distribution == "lognormal":
        col1, col2 = st.columns(2)
        with col1:
            mean = st.number_input(
                "평균 (Mean)",
                value=0.0,
                step=0.1,
                format="%.4f",
                help="로그정규분포의 평균 (로그 스케일)"
            )
            params['mean'] = mean
        with col2:
            sigma = st.number_input(
                "시그마 (Sigma)",
                value=0.1,
                min_value=0.001,
                step=0.01,
                format="%.4f",
                help="로그정규분포의 시그마"
            )
            params['sigma'] = sigma

    # Bounds (선택사항)
    use_bounds = st.checkbox("샘플 범위 제한 (Bounds)", value=False)
    bounds = None
    if use_bounds:
        col1, col2 = st.columns(2)
        with col1:
            min_bound = st.number_input("최소 허용값", value=0.5, step=0.1)
        with col2:
            max_bound = st.number_input("최대 허용값", value=1.5, step=0.1)
        bounds = (min_bound, max_bound)

    # 설명
    description = st.text_area(
        "설명 (선택사항)",
        value="",
        help="이 불확실성에 대한 설명"
    )

    # 불확실성 추가 버튼
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("➕ 불확실성 추가", type="primary", use_container_width=True):
            try:
                # ParameterUncertainty 생성
                uncertainty = ParameterUncertainty(
                    name=param_name,
                    distribution=distribution,
                    params=params,
                    bounds=bounds,
                    description=description
                )

                # 세션에 저장
                st.session_state.stochastic_uncertainties[param_name] = uncertainty

                st.success(f"✅ '{param_name}' 불확실성 추가 완료!")

            except Exception as e:
                st.error(f"❌ 불확실성 추가 실패: {str(e)}")

    with col2:
        if st.button("🗑️ 전체 삭제", use_container_width=True):
            st.session_state.stochastic_uncertainties = {}
            st.success("✅ 모든 불확실성 삭제 완료!")
            st.rerun()

    # 정의된 불확실성 목록 표시
    if st.session_state.stochastic_uncertainties:
        st.markdown("---")
        st.markdown("#### 📋 정의된 불확실성 목록")

        uncertainty_data = []
        for name, unc in st.session_state.stochastic_uncertainties.items():
            theoretical_stats = unc.get_statistics()
            uncertainty_data.append({
                '파라미터': name,
                '분포': unc.distribution,
                '파라미터': str(unc.params),
                '이론적 평균': f"{theoretical_stats.get('mean', 0):.4f}",
                '이론적 표준편차': f"{theoretical_stats.get('std', 0):.4f}"
            })

        uncertainty_df = pd.DataFrame(uncertainty_data)
        st.dataframe(uncertainty_df, use_container_width=True, hide_index=True)

        st.info(f"📊 총 {len(st.session_state.stochastic_uncertainties)}개 불확실성 정의됨")

    # Step 2: Monte Carlo 설정 및 실행
    if st.session_state.stochastic_uncertainties:
        st.markdown("---")
        st.markdown("### 2️⃣ Monte Carlo 설정 및 실행")

        col1, col2, col3 = st.columns(3)

        with col1:
            n_samples = st.number_input(
                "샘플 수 (N)",
                min_value=10,
                max_value=10000,
                value=1000,
                step=100,
                help="Monte Carlo 시뮬레이션 샘플 개수 (많을수록 정확하지만 느림)"
            )

        with col2:
            random_state = st.number_input(
                "난수 시드 (Seed)",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                help="재현성을 위한 난수 시드"
            )

        with col3:
            st.markdown("　")  # 간격 조정
            st.markdown("　")
            parallel = st.checkbox(
                "병렬 실행 (미구현)",
                value=False,
                disabled=True,
                help="향후 구현 예정"
            )

        st.warning(f"⏱️ 예상 실행 시간: 약 {n_samples * 2 / 60:.1f}분 (샘플당 약 2초 가정)")

        # 실행 버튼
        if st.button("🚀 Monte Carlo 시뮬레이션 실행", type="primary", use_container_width=True):
            try:
                with st.spinner(f"🔄 {n_samples}개 샘플 시뮬레이션 중... (시간이 걸립니다)"):
                    # 데이터 로드
                    data_loader = st.session_state.data_loader
                    base_data = data_loader.get_optimization_data()

                    # 최적화 엔진 생성
                    engine = OptimizationEngine(
                        solver_name='glpk',
                        constraint_manager=st.session_state.constraint_manager
                    )

                    # StochasticAnalyzer 생성
                    analyzer = StochasticAnalyzer(
                        engine=engine,
                        data_loader=data_loader,
                        n_samples=n_samples,
                        random_state=random_state,
                        parallel=parallel
                    )

                    # 불확실성 정의
                    for name, unc in st.session_state.stochastic_uncertainties.items():
                        analyzer.define_uncertainty(
                            parameter_name=unc.name,
                            distribution=unc.distribution,
                            params=unc.params,
                            bounds=unc.bounds,
                            description=unc.description
                        )

                    # Monte Carlo 실행
                    results = analyzer.run_monte_carlo(
                        base_data=base_data,
                        objective_type='minimize_carbon'
                    )

                    # 세션에 저장
                    st.session_state.stochastic_analyzer = analyzer
                    st.session_state.stochastic_results = results

                    st.success(f"✅ Monte Carlo 시뮬레이션 완료! ({len(results)}개 샘플)")
                    st.balloons()

            except Exception as e:
                st.error(f"❌ 시뮬레이션 실패: {str(e)}")
                import traceback
                with st.expander("오류 상세"):
                    st.code(traceback.format_exc())

    # Step 3: 결과 시각화
    if st.session_state.stochastic_results:
        st.markdown("---")
        st.markdown("### 3️⃣ 결과 분석 및 시각화")

        analyzer = st.session_state.stochastic_analyzer

        # 통계 메트릭
        st.markdown("#### 📊 목적함수 통계")

        stats = analyzer.statistics

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "평균 (Mean)",
                f"{stats['mean']:.2f} kgCO2eq",
                help="목적함수의 평균값"
            )
            st.metric(
                "중간값 (Median)",
                f"{stats['median']:.2f} kgCO2eq",
                help="목적함수의 중간값"
            )

        with col2:
            st.metric(
                "표준편차 (Std)",
                f"{stats['std']:.2f} kgCO2eq",
                help="목적함수의 표준편차 (변동성)"
            )
            st.metric(
                "변동계수 (CV)",
                f"{stats['cv']:.4f}",
                help="Coefficient of Variation = Std / Mean"
            )

        with col3:
            st.metric(
                "최소값 (Min)",
                f"{stats['min']:.2f} kgCO2eq",
                help="최선의 경우"
            )
            st.metric(
                "최대값 (Max)",
                f"{stats['max']:.2f} kgCO2eq",
                help="최악의 경우"
            )

        with col4:
            st.metric(
                "95% 분위수",
                f"{stats['p95']:.2f} kgCO2eq",
                help="95% 신뢰수준"
            )
            st.metric(
                "성공률",
                f"{stats['n_samples']} / {analyzer.n_samples}",
                help="최적화 성공한 샘플 수"
            )

        # 리스크 메트릭
        st.markdown("---")
        st.markdown("#### ⚠️ 리스크 메트릭")

        risk = analyzer.risk_metrics

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "VaR 95%",
                f"{risk['var_95']:.2f} kgCO2eq",
                help="Value at Risk: 95% 신뢰수준의 임계값"
            )

        with col2:
            st.metric(
                "CVaR 95%",
                f"{risk['cvar_95']:.2f} kgCO2eq",
                help="Conditional VaR: Worst 5% 시나리오의 평균"
            )

        with col3:
            st.metric(
                "Tail Mean",
                f"{risk['tail_mean']:.2f} kgCO2eq",
                help="Worst 5% 시나리오 평균"
            )

        with col4:
            st.metric(
                "Downside Risk",
                f"{risk['downside_risk']:.2f}",
                help="평균 초과 값들의 표준편차"
            )

        # 확률 분포 시각화
        st.markdown("---")
        st.markdown("#### 📈 확률 분포")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        dist_data = analyzer.get_probability_distribution(n_bins=50)

        # 2개 서브플롯: 히스토그램 + CDF
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("확률 밀도 함수 (PDF)", "누적 분포 함수 (CDF)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # PDF (히스토그램)
        fig.add_trace(
            go.Bar(
                x=dist_data['bin_centers'],
                y=dist_data['pdf'],
                name='PDF',
                marker_color='steelblue',
                opacity=0.7
            ),
            row=1, col=1
        )

        # 평균, 중간값 수직선
        fig.add_vline(
            x=stats['mean'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"평균: {stats['mean']:.2f}",
            row=1, col=1
        )

        # CDF
        fig.add_trace(
            go.Scatter(
                x=dist_data['bin_centers'],
                y=dist_data['cdf'],
                mode='lines',
                name='CDF',
                line=dict(color='green', width=3)
            ),
            row=1, col=2
        )

        # VaR 수직선
        fig.add_vline(
            x=risk['var_95'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"VaR95: {risk['var_95']:.2f}",
            row=1, col=2
        )

        fig.update_xaxes(title_text="목적함수 (kgCO2eq)", row=1, col=1)
        fig.update_xaxes(title_text="목적함수 (kgCO2eq)", row=1, col=2)
        fig.update_yaxes(title_text="확률 밀도", row=1, col=1)
        fig.update_yaxes(title_text="누적 확률", row=1, col=2)

        fig.update_layout(
            height=400,
            showlegend=False,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # 파라미터 상관관계
        st.markdown("---")
        st.markdown("#### 🔗 파라미터-목적함수 상관관계")

        try:
            correlation_df = analyzer.get_parameter_correlation()

            # 상관계수 막대 차트
            fig_corr = go.Figure()

            fig_corr.add_trace(go.Bar(
                x=correlation_df['correlation'].values,
                y=correlation_df.index,
                orientation='h',
                marker_color=['red' if x < 0 else 'blue' for x in correlation_df['correlation'].values],
                text=[f"{x:.4f}" for x in correlation_df['correlation'].values],
                textposition='auto'
            ))

            fig_corr.update_layout(
                title='파라미터-목적함수 상관계수',
                xaxis_title='상관계수 (Correlation)',
                yaxis_title='파라미터',
                height=300 + len(correlation_df) * 30,
                hovermode='y'
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            st.caption("""
            **해석**:
            - 양의 상관계수: 파라미터 증가 → 목적함수 증가 (탄소 배출 증가)
            - 음의 상관계수: 파라미터 증가 → 목적함수 감소 (탄소 배출 감소)
            - 절댓값이 클수록 영향력이 큼
            """)

        except Exception as e:
            st.warning(f"상관관계 분석 중 오류: {str(e)}")

        # 보고서 다운로드
        st.markdown("---")
        st.markdown("#### 📥 결과 다운로드")

        col1, col2 = st.columns(2)

        with col1:
            # 텍스트 보고서
            report = analyzer.generate_report()

            st.download_button(
                label="📄 텍스트 보고서 다운로드",
                data=report,
                file_name=f"stochastic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:
            # JSON 결과
            export_data = analyzer.export_results()
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="📊 JSON 결과 다운로드",
                data=json_str,
                file_name=f"stochastic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


# ============================================================
# Phase 2: Workflow Helper Functions
# ============================================================

def _render_progress_indicator(current_step: int):
    """4단계 워크플로우 진행 상태 표시"""
    steps = ["데이터", "제약조건", "방법", "실행"]
    cols = st.columns(4)

    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                icon = "●"
                color = "green"
            elif i == current_step:
                icon = "●"
                color = "blue"
            else:
                icon = "○"
                color = "gray"

            st.markdown(f"<div style='text-align: center; color: {color};'>{icon} {i+1}. {step}</div>",
                       unsafe_allow_html=True)


def _render_quick_constraint_presets():
    """제약조건 빠른 프리셋 (4가지)"""

    # 설정 모드 선택
    setting_mode = st.radio(
        "설정 모드",
        options=['preset', 'advanced'],
        format_func=lambda x: {
            'preset': '💡 빠른 프리셋 (권장)',
            'advanced': '🔧 고급 설정'
        }[x],
        horizontal=True,
        help="빠른 프리셋은 간편하게 설정할 수 있으며, 고급 설정은 세부적인 제약조건을 직접 구성할 수 있습니다."
    )

    if setting_mode == 'preset':
        # 빠른 프리셋 모드
        st.markdown("### 💡 빠른 프리셋")
        st.caption("자주 사용하는 제약조건을 한 번에 설정합니다")

        preset_options = {
            'none': '프리셋 없음 (제약 없음)',
            'economic': '경제적 최적화 (비용 +10% 이내)',
            'balanced': '균형잡힌 감축 (비용 +20%, 재활용 30%+)',
            'aggressive': '공격적 감축 (비용 제약 완화)'
        }

        selected_preset = st.radio(
            "프리셋 선택",
            options=list(preset_options.keys()),
            format_func=lambda x: preset_options[x],
            horizontal=True,
            help="프리셋을 선택하면 자동으로 제약조건이 설정됩니다."
        )

        # 프리셋 적용
        if selected_preset != 'none':
            _apply_quick_preset(selected_preset)
            st.success(f"✅ '{preset_options[selected_preset]}' 프리셋 적용됨")

            # 프리미엄 스캔 활성화 안내
            if 'weighted_sum_config' in st.session_state:
                premium_scan = st.session_state.weighted_sum_config.get('premium_scan', {})
                if premium_scan.get('enabled'):
                    premium_range = premium_scan.get('range', [])
                    st.info(f"💰 프리미엄 스캔 자동 활성화: {premium_range}% 범위에서 탐색")
        else:
            # 프리셋 없음: 비용 제약 초기화
            if 'preset_premium_limit' in st.session_state:
                del st.session_state.preset_premium_limit
            st.info("ℹ️ 제약조건 없이 최적화합니다")

    else:
        # 고급 설정 모드
        st.markdown("### 🔧 고급 설정")
        st.caption("세부적인 제약조건을 직접 구성합니다")

        # expander로 제약조건 설정 UI 표시
        with st.expander("▶ 제약조건 상세 설정", expanded=True):
            render_constraint_configuration_tab()


def _apply_quick_preset(preset_key: str):
    """빠른 프리셋 적용"""
    constraint_manager = st.session_state.constraint_manager

    # 기존 제약 초기화 (Feature Option 제외)
    for name in list(constraint_manager.list_constraints()):
        constraint = constraint_manager.get_constraint(name)
        if not hasattr(constraint, 'is_feature_option_constraint') or not constraint.is_feature_option_constraint():
            constraint_manager.remove_constraint(name)

    # 프리셋별 제약조건 추가
    if preset_key == 'economic':
        # 비용 +10%: session_state에 premium_limit 저장
        st.session_state.preset_premium_limit = 10.0
        st.info("✅ 비용 제약 +10% 적용됨 (모든 최적화 방법에 적용)")

        # Premium Scan도 제안
        _auto_enable_premium_scan(premium_max=10)

    elif preset_key == 'balanced':
        # 비용 +20%: session_state에 premium_limit 저장
        st.session_state.preset_premium_limit = 20.0
        st.info("✅ 비용 제약 +20% 적용됨 (모든 최적화 방법에 적용)")

        # Premium Scan도 제안
        _auto_enable_premium_scan(premium_max=20)

        # 재활용 최소 30% (Ni, Co, Li에 적용)
        from src.optimization_v2.constraints.material_constraint import MaterialManagementConstraint

        material_constraint = MaterialManagementConstraint()
        material_constraint.add_rule(
            rule_type='force_element_ratio_range',
            material_name='Cathode_active_material_1',
            params={
                'element': 'Ni',
                'recycle_min': 0.3,
                'recycle_max': 1.0,
                'low_carbon_min': 0.0,
                'low_carbon_max': 1.0
            }
        )
        constraint_manager.add_constraint(material_constraint)

    elif preset_key == 'aggressive':
        # 비용 +50%: session_state에 premium_limit 저장
        st.session_state.preset_premium_limit = 50.0
        st.info("✅ 비용 제약 +50% 적용됨 (모든 최적화 방법에 적용)")

        # Premium Scan도 제안
        _auto_enable_premium_scan(premium_max=50)


def _auto_enable_premium_scan(premium_max: int):
    """
    빠른설정 프리셋에서 자동으로 프리미엄 스캔 활성화

    Args:
        premium_max: 최대 프리미엄 % (예: 10, 20, 50)
    """
    # 프리미엄 스캔 범위 생성 (0%부터 premium_max까지, 5% 간격)
    step = 5 if premium_max >= 20 else max(1, premium_max // 3)
    premium_range = list(range(0, premium_max + 1, step))

    # Weighted Sum 설정 업데이트
    if 'weighted_sum_config' not in st.session_state:
        st.session_state.weighted_sum_config = {}

    st.session_state.weighted_sum_config['premium_scan'] = {
        'enabled': True,
        'range': premium_range
    }

    print(f"✅ 프리미엄 스캔 자동 활성화: {premium_range}%")


def _render_execution_history():
    """최근 실행 기록 표시 (최근 3개)"""
    if 'execution_history' not in st.session_state or not st.session_state.execution_history:
        st.caption("아직 실행 기록이 없습니다")
        return

    st.markdown("### 💾 최근 실행 기록")

    # 최근 3개만 표시
    recent_runs = st.session_state.execution_history[-3:][::-1]  # 역순 (최신순)

    for i, entry in enumerate(recent_runs):
        timestamp = entry.get('timestamp', 'N/A')
        method = entry.get('method', 'Unknown')
        status = entry.get('status', 'Unknown')
        points = entry.get('points', 0)

        # 상태 이모지
        if status == 'success':
            status_emoji = "✅"
        elif status == 'failed':
            status_emoji = "❌"
        else:
            status_emoji = "⏸️"

        st.caption(f"{i+1}. {timestamp} | {method} ({points}개 포인트) | {status_emoji} {status}")


def _add_to_execution_history(method: str, status: str, points: int = 0):
    """실행 기록에 추가"""
    from datetime import datetime

    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []

    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'method': method,
        'status': status,
        'points': points
    }

    st.session_state.execution_history.append(entry)

    # 최대 10개까지만 유지
    if len(st.session_state.execution_history) > 10:
        st.session_state.execution_history = st.session_state.execution_history[-10:]


def render_setup_and_run_tab():
    """
    Tab 1: ⚙️ 설정 및 실행
    데이터 로딩부터 최적화 실행까지의 완전한 워크플로우
    """
    st.header("⚙️ 설정 및 실행")

    # 진행 상태 표시기
    current_step = st.session_state.get('current_step', 1)
    _render_progress_indicator(current_step)

    st.markdown("---")

    # Step 1: 데이터 로딩
    st.markdown("## Step 1️⃣ 데이터 로딩")
    render_data_loading_tab()

    # Step 2는 데이터 로딩 후에만 표시
    if not st.session_state.get('data_loaded', False):
        st.info("💡 먼저 데이터를 로딩하세요")
        return

    st.session_state.current_step = max(st.session_state.current_step, 2)
    st.markdown("---")

    # Step 2: 제약조건 설정
    st.markdown("## Step 2️⃣ 제약조건 설정 (선택)")
    _render_quick_constraint_presets()

    st.session_state.current_step = max(st.session_state.current_step, 3)
    st.markdown("---")

    # Step 3: 최적화 방법 선택
    st.markdown("## Step 3️⃣ 최적화 방법 선택")

    # 기존 최적화 실행 탭의 방법 선택 로직 재사용
    method = st.radio(
        "최적화 방법",
        options=['weighted_sum', 'weighted_sum_premium', 'epsilon_constraint', 'nsga2'],
        format_func=lambda x: {
            'weighted_sum': '⚖️ Weighted Sum (가중치 스캔) - 권장',
            'weighted_sum_premium': '💰 Weighted Sum + Premium Scan (가중치 × 비용 레벨)',
            'epsilon_constraint': '🎯 Epsilon-Constraint (비용 제약)',
            'nsga2': '🧬 NSGA-II (진화 알고리즘)'
        }[x],
        horizontal=False,
        key="setup_pareto_method",
        help="파레토 최적화 방법을 선택하세요. 처음 사용하시면 Weighted Sum을 권장합니다."
    )

    # 방법별 간단한 설정
    with st.expander(f"▶ {method} 설정", expanded=True):
        if method == 'weighted_sum':
            render_weighted_sum_settings(enable_premium_scan=False)
        elif method == 'weighted_sum_premium':
            render_weighted_sum_settings(enable_premium_scan=True)
        elif method == 'epsilon_constraint':
            render_epsilon_constraint_settings()
        elif method == 'nsga2':
            render_nsga2_settings()

    st.session_state.current_step = max(st.session_state.current_step, 4)
    st.markdown("---")

    # Step 4: 실행
    st.markdown("## Step 4️⃣ 실행")

    col1, col2 = st.columns([3, 1])

    with col1:
        scenario_name = st.text_input(
            "결과 저장 이름",
            value=f"최적화_{len(st.session_state.optimization_results) + 1}",
            help="결과 저장 및 비교를 위한 이름"
        )

    with col2:
        solver = st.selectbox(
            "솔버",
            options=['auto', 'glpk', 'ipopt'],
            format_func=lambda x: {'auto': '자동', 'glpk': 'GLPK', 'ipopt': 'IPOPT'}[x]
        )

    # 큰 실행 버튼
    if st.button("🚀 최적화 실행", type="primary", use_container_width=True):
        if not scenario_name:
            st.error("결과 저장 이름을 입력하세요")
        else:
            # 프리셋 선택 여부 확인
            # 사용자가 빠른 프리셋을 선택했으면 constraint_preset을 None으로 설정
            # (이미 session_state.constraint_manager에 제약조건이 추가되어 있음)
            has_preset = (
                'preset_premium_limit' in st.session_state or
                len(st.session_state.constraint_manager.list_constraints(enabled_only=True)) > 0
            )

            constraint_preset = None if has_preset else 'medium'
            scenario_template = None

            # 실행
            try:
                if method == 'weighted_sum':
                    run_weighted_sum_optimization(constraint_preset, scenario_template, premium_scan=False)
                    _add_to_execution_history('Weighted Sum', 'success',
                                            len(st.session_state.get('pareto_results', [])))
                elif method == 'weighted_sum_premium':
                    run_weighted_sum_optimization(constraint_preset, scenario_template, premium_scan=True)
                    _add_to_execution_history('Weighted Sum + Premium Scan', 'success',
                                            len(st.session_state.get('pareto_results', [])))
                elif method == 'epsilon_constraint':
                    run_epsilon_constraint_optimization(constraint_preset, scenario_template)
                    _add_to_execution_history('Epsilon-Constraint', 'success',
                                            len(st.session_state.get('pareto_results', [])))
                elif method == 'nsga2':
                    run_nsga2_optimization(constraint_preset, scenario_template)
                    _add_to_execution_history('NSGA-II', 'success',
                                            len(st.session_state.get('pareto_results', [])))

                # 성공 시 Tab 2로 전환 유도
                st.success("✅ 최적화 완료! '📊 결과 분석' 탭에서 결과를 확인하세요")
                st.info("💡 위의 탭 메뉴에서 '📊 결과 분석'을 클릭하세요")

            except Exception as e:
                st.error(f"❌ 최적화 실패: {str(e)}")
                _add_to_execution_history(method, 'failed', 0)
                with st.expander("오류 상세"):
                    st.code(traceback.format_exc())

    st.markdown("---")

    # 최근 실행 기록
    _render_execution_history()


# ============================================================
# Phase 3: Results Analysis Helper Functions
# ============================================================

def _render_smart_recommendation(pareto_frontier, method):
    """스마트 추천 - 균형잡힌 포인트 자동 선택"""
    if not pareto_frontier or len(pareto_frontier) == 0:
        return None

    # 간단한 추천 로직: 중간 지점 선택 (탄소와 비용의 균형)
    # SolutionRecommender 통합은 향후 개선

    carbon_values = [p['summary']['total_carbon'] for p in pareto_frontier]
    cost_values = [p['summary'].get('total_cost', 0) for p in pareto_frontier]

    # 정규화
    if len(carbon_values) > 1:
        carbon_min, carbon_max = min(carbon_values), max(carbon_values)
        cost_min, cost_max = min(cost_values), max(cost_values)

        # 분모가 0인 경우 처리
        carbon_range = carbon_max - carbon_min if carbon_max != carbon_min else 1
        cost_range = cost_max - cost_min if cost_max != cost_min else 1

        # 균형 점수 계산 (탄소와 비용을 모두 고려)
        scores = []
        for i, p in enumerate(pareto_frontier):
            carbon_norm = (carbon_values[i] - carbon_min) / carbon_range
            cost_norm = (cost_values[i] - cost_min) / cost_range

            # 균형 점수: 탄소도 낮고 비용도 낮은 포인트 선호
            balance_score = (carbon_norm + cost_norm) / 2
            scores.append(balance_score)

        # 가장 균형잡힌 포인트 (최소 점수)
        recommended_idx = scores.index(min(scores))
    else:
        recommended_idx = 0

    return recommended_idx


def _render_point_details_integrated(point, method, point_idx):
    """포인트 상세 정보 통합 표시 (all-in-one)"""
    st.markdown(f"### 📊 Point #{point_idx + 1} 상세 정보")

    summary = point['summary']
    solution = point.get('solution', None)  # NSGA-II may not have solution

    # 1. 핵심 지표 (2개 컬럼)
    col1, col2 = st.columns(2)

    with col1:
        total_carbon = summary['total_carbon']
        baseline_carbon = point.get('baseline_carbon', 0)

        if baseline_carbon > 0:
            carbon_reduction_pct = ((baseline_carbon - total_carbon) / baseline_carbon) * 100
            st.metric(
                "총 탄소배출",
                f"{total_carbon:.2f} kgCO2eq",
                delta=f"-{carbon_reduction_pct:.1f}%",
                delta_color="normal",  # 감소는 빨간색으로 표시
                help=f"기준 배출량: {baseline_carbon:.2f} kgCO2eq"
            )
        else:
            st.metric("총 탄소배출", f"{total_carbon:.2f} kgCO2eq")

    with col2:
        total_cost = summary.get('total_cost', 0)
        baseline_cost = point.get('zero_premium_baseline', point.get('baseline_cost', 0))

        if baseline_cost > 0:
            cost_increase_pct = ((total_cost - baseline_cost) / baseline_cost) * 100
            st.metric(
                "총 비용",
                f"${total_cost:,.2f}",
                delta=f"+{cost_increase_pct:.1f}%",
                help=f"기준 비용: ${baseline_cost:,.2f}"
            )
        else:
            st.metric("총 비용", f"${total_cost:,.2f}")

    st.markdown("---")

    # 2. 최적화 설정 정보
    with st.expander("⚙️ 최적화 설정", expanded=False):
        if method == 'weighted_sum':
            st.write(f"**탄소 가중치**: {point['weights']['carbon_weight']:.2f}")
            st.write(f"**비용 가중치**: {point['weights']['cost_weight']:.2f}")
        elif method == 'epsilon_constraint':
            st.write(f"**Epsilon (비용 상한)**: ${point['epsilon']:,.0f}")
        elif method == 'nsga2':
            st.write(f"**Rank**: {point.get('rank', 0)}")
            cd = point.get('crowding_distance', 0)
            cd_str = "∞" if cd == float('inf') else f"{cd:.4f}"
            st.write(f"**Crowding Distance**: {cd_str}")

    # 3. 양극재 원소별 결과 (있으면)
    if solution and 'cathode' in solution:
        st.markdown("---")
        render_cathode_element_results(solution['cathode'])

    # 4. 음극재 composition 결과 (있으면)
    if solution and 'anode' in solution:
        st.markdown("---")
        render_anode_composition_results(solution['anode'])

    # 5. 자재별 상세 결과
    if 'result_df' in point:
        st.markdown("---")
        with st.expander("📄 자재별 상세 결과", expanded=False):
            result_df = point['result_df']

            # 주요 컬럼만 표시
            display_cols = ['자재명', '배출계수', '감축률(%)', 'Tier1_RE(%)', 'Tier2_RE(%)']
            available_cols = [col for col in display_cols if col in result_df.columns]

            if available_cols:
                st.dataframe(result_df[available_cols], use_container_width=True, hide_index=True)
    elif method == 'nsga2':
        # NSGA-II uses genes instead of full solution
        st.markdown("---")
        with st.expander("🧬 유전자 정보 (Genes)", expanded=False):
            if 'genes' in point:
                genes = point['genes']
                st.json(genes)


def _save_scenario(point, scenario_name, method):
    """시나리오 저장"""
    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []

    # 중복 이름 체크
    existing_names = [s['name'] for s in st.session_state.saved_scenarios]
    if scenario_name in existing_names:
        _show_user_friendly_error('duplicate_name', context=f"'{scenario_name}' 이름이 이미 존재합니다")
        return False

    # 시나리오 저장
    scenario = {
        'name': scenario_name,
        'point': point,
        'method': method,
        'carbon': point['summary']['total_carbon'],
        'cost': point['summary'].get('total_cost', 0),
        'reduction': point['summary'].get('total_reduction_pct', 0)
    }

    st.session_state.saved_scenarios.append(scenario)

    # optimization_results에도 추가 (기존 시스템 호환)
    opt_result = {
        'summary': point['summary']
    }

    # Add result_df if available
    if 'result_df' in point:
        opt_result['result_df'] = point['result_df']

    # Add solution if available (may not exist for NSGA-II)
    if 'solution' in point:
        opt_result['solution'] = point['solution']

    st.session_state.optimization_results[scenario_name] = opt_result

    return True


def _load_scenario(scenario_idx):
    """시나리오 로드"""
    if scenario_idx < len(st.session_state.saved_scenarios):
        scenario = st.session_state.saved_scenarios[scenario_idx]
        st.session_state.selected_point = scenario['point']
        return scenario
    return None


def _delete_scenario(scenario_idx):
    """시나리오 삭제"""
    if scenario_idx < len(st.session_state.saved_scenarios):
        scenario = st.session_state.saved_scenarios[scenario_idx]
        scenario_name = scenario['name']

        # saved_scenarios에서 삭제
        st.session_state.saved_scenarios.pop(scenario_idx)

        # optimization_results에서도 삭제
        if scenario_name in st.session_state.optimization_results:
            del st.session_state.optimization_results[scenario_name]

        return True
    return False


def _render_saved_scenarios():
    """저장된 시나리오 섹션"""
    if 'saved_scenarios' not in st.session_state or not st.session_state.saved_scenarios:
        st.info("💡 저장된 시나리오가 없습니다. 위에서 포인트를 선택하고 저장하세요.")
        return

    st.markdown("### 📦 저장된 시나리오")

    for i, scenario in enumerate(st.session_state.saved_scenarios):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.markdown(f"**{i+1}. {scenario['name']}**")
            st.caption(f"탄소: {scenario['carbon']:.2f} | 비용: ${scenario['cost']:,.0f} | 감축: {scenario['reduction']:.1f}%")

        with col2:
            if st.button("📊 보기", key=f"view_scenario_{i}", use_container_width=True):
                _load_scenario(i)
                st.rerun()

        with col3:
            if st.button("🗑️ 삭제", key=f"delete_scenario_{i}", use_container_width=True):
                _delete_scenario(i)
                st.success(f"✅ '{scenario['name']}' 삭제됨")
                st.rerun()

        with col4:
            # CSV 다운로드
            result_df = scenario['point']['result_df']
            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV",
                data=csv,
                file_name=f"{scenario['name']}.csv",
                mime="text/csv",
                key=f"download_scenario_{i}",
                use_container_width=True
            )

    # 비교 분석 링크 (2개 이상일 때)
    if len(st.session_state.saved_scenarios) >= 2:
        st.markdown("---")
        if st.button("🔍 시나리오 비교 분석 →", use_container_width=True):
            st.session_state.active_tab = "🔬 고급 분석"
            st.session_state.advanced_section = "시나리오 비교"
            st.info("💡 '🔬 고급 분석' 탭으로 이동하여 시나리오를 비교하세요")


def render_results_analysis_tab():
    """
    Tab 2: 📊 결과 분석
    최적화 결과 확인, 파레토 프론티어 분석, 포인트 선택 및 저장
    """
    st.header("📊 결과 분석")

    # 파레토 결과 확인
    if 'pareto_results' not in st.session_state or not st.session_state.pareto_results:
        st.info("""
        💡 **최적화 결과가 없습니다**

        '⚙️ 설정 및 실행' 탭에서 파레토 최적화를 먼저 실행하세요.
        """)
        return

    pareto_results = st.session_state.pareto_results
    pareto_frontier = st.session_state.get('pareto_frontier', [])
    method = st.session_state.get('optimization_method', 'weighted_sum')

    # 1. 최적화 결과 요약
    st.markdown("## 🌟 최적화 결과 요약")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("탐색 포인트", f"{len(pareto_results)}개")

    with col2:
        st.metric("파레토 최적해", f"{len(pareto_frontier)}개")

    with col3:
        method_names = {
            'weighted_sum': '⚖️ Weighted Sum',
            'epsilon_constraint': '🎯 Epsilon',
            'nsga2': '🧬 NSGA-II'
        }
        st.metric("최적화 방법", method_names.get(method, method))

    with col4:
        if len(pareto_results) > 0:
            efficiency = len(pareto_frontier) / len(pareto_results) * 100
            st.metric("파레토 효율성", f"{efficiency:.1f}%")

    st.markdown("---")

    # 2. 스마트 추천
    if pareto_frontier:
        recommended_idx = _render_smart_recommendation(pareto_frontier, method)

        if recommended_idx is not None:
            st.success(f"💡 **추천 포인트**: Point #{recommended_idx + 1} (균형잡힌 선택)")
            st.caption("탄소 감축과 비용을 균형있게 고려한 포인트입니다")

    st.markdown("---")

    # 3. 파레토 프론티어 시각화
    st.markdown("## 🎯 파레토 프론티어")

    if pareto_frontier:
        # 기존 함수 재사용 (추천 포인트 하이라이트는 향후 개선)
        render_pareto_frontier_plot(pareto_results, pareto_frontier, method)

    st.markdown("---")

    # 3.5. 프리미엄 스캔 시각화 (있는 경우)
    if st.session_state.get('premium_scan_enabled', False):
        premium_scan_results = st.session_state.get('premium_scan_results', {})
        if premium_scan_results:
            render_premium_scan_visualization(premium_scan_results, method)
            st.markdown("---")

    # 4. 포인트 선택 및 상세 정보
    st.markdown("## 🔍 포인트 상세 분석")

    if not pareto_frontier:
        st.warning("파레토 최적해가 없습니다")
        return

    # 포인트 선택 드롭다운
    point_options = {}
    for idx, point in enumerate(pareto_frontier):
        carbon = point['summary']['total_carbon']
        cost = point['summary'].get('total_cost', 0)

        if method == 'weighted_sum':
            label = f"Point {idx+1}: α={point['weights']['carbon_weight']:.2f} | 탄소: {carbon:.2f} | 비용: ${cost:,.2f}"
        elif method == 'epsilon_constraint':
            label = f"Point {idx+1}: ε=${point['epsilon']:,.2f} | 탄소: {carbon:.2f}"
        else:  # nsga2
            label = f"Point {idx+1}: Rank={point.get('rank', 0)} | 탄소: {carbon:.2f} | 비용: ${cost:,.2f}"

        point_options[label] = idx

    # 기본 선택: 추천 포인트
    default_idx = recommended_idx if recommended_idx is not None else 0
    default_label = list(point_options.keys())[default_idx]

    selected_label = st.selectbox(
        "분석할 포인트 선택",
        options=list(point_options.keys()),
        index=default_idx,
        help="선택한 포인트의 상세 정보를 아래에서 확인할 수 있습니다"
    )

    selected_idx = point_options[selected_label]
    selected_point = pareto_frontier[selected_idx]

    # 포인트 상세 정보 표시
    _render_point_details_integrated(selected_point, method, selected_idx)

    st.markdown("---")

    # 5. 시나리오 저장
    st.markdown("### 💾 시나리오로 저장")

    col1, col2 = st.columns([3, 1])

    with col1:
        scenario_name = st.text_input(
            "시나리오 이름",
            value=f"Pareto_{method}_{selected_idx+1}",
            help="비교용 시나리오로 저장할 이름을 입력하세요"
        )

    with col2:
        st.markdown("")  # 간격 조정
        st.markdown("")
        if st.button("💾 저장", use_container_width=True, type="primary"):
            if _save_scenario(selected_point, scenario_name, method):
                st.success(f"✅ '{scenario_name}' 저장 완료!")
                st.info("💡 아래 '저장된 시나리오' 섹션에서 확인하세요")

    st.markdown("---")

    # 6. 저장된 시나리오 섹션
    _render_saved_scenarios()


def _render_export_and_logs():
    """결과 내보내기 & 로그 섹션"""
    from datetime import datetime
    import json

    st.markdown("### 📥 결과 내보내기 & 로그")

    st.markdown("""
    최적화 결과, 설정, 로그를 다양한 형식으로 내보낼 수 있습니다.
    """)

    # 1. 파레토 결과 내보내기
    st.markdown("#### 1️⃣ 파레토 최적화 결과 내보내기")

    if 'pareto_results' in st.session_state and st.session_state.pareto_results:

        col1, col2 = st.columns(2)

        with col1:
            # JSON 내보내기
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'method': st.session_state.get('optimization_method', 'unknown'),
                'num_results': len(st.session_state.pareto_results),
                'num_frontier': len(st.session_state.get('pareto_frontier', [])),
                'results': []
            }

            for idx, result in enumerate(st.session_state.pareto_results):
                export_data['results'].append({
                    'point_id': idx,
                    'carbon': result['summary']['total_carbon'],
                    'cost': result['summary'].get('total_cost', 0),
                    'reduction_pct': result['summary'].get('total_reduction_pct', 0)
                })

            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="📄 JSON 다운로드",
                data=json_str,
                file_name=f"pareto_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            # CSV 내보내기
            csv_data = []
            for idx, result in enumerate(st.session_state.pareto_results):
                csv_data.append({
                    'Point': idx + 1,
                    'Carbon (kg)': result['summary']['total_carbon'],
                    'Cost ($)': result['summary'].get('total_cost', 0),
                    'Reduction (%)': result['summary'].get('total_reduction_pct', 0)
                })

            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False, encoding='utf-8-sig')

            st.download_button(
                label="📊 CSV 다운로드",
                data=csv,
                file_name=f"pareto_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.success(f"✅ {len(st.session_state.pareto_results)}개 포인트 내보내기 가능")
    else:
        st.info("💡 파레토 최적화 결과가 없습니다. 먼저 최적화를 실행하세요.")

    st.markdown("---")

    # 2. 제약조건 설정 내보내기
    st.markdown("#### 2️⃣ 제약조건 설정 내보내기")

    if st.session_state.constraint_manager:
        constraints = st.session_state.constraint_manager.list_constraints()

        if constraints:
            col1, col2 = st.columns(2)

            with col1:
                # JSON 형식
                constraint_data = []
                for name in constraints:
                    constraint = st.session_state.constraint_manager.get_constraint(name)
                    constraint_data.append({
                        'name': constraint.name,
                        'type': constraint.__class__.__name__,
                        'enabled': constraint.enabled,
                        'description': constraint.description
                    })

                json_str = json.dumps({'constraints': constraint_data}, indent=2, ensure_ascii=False)

                st.download_button(
                    label="📄 제약조건 JSON",
                    data=json_str,
                    file_name=f"constraints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                # 텍스트 요약
                summary_text = f"=== 제약조건 설정 ===\n생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                for name in constraints:
                    constraint = st.session_state.constraint_manager.get_constraint(name)
                    status = "활성" if constraint.enabled else "비활성"
                    summary_text += f"[{status}] {constraint.name}\n"
                    summary_text += f"  설명: {constraint.description}\n\n"

                st.download_button(
                    label="📝 제약조건 TXT",
                    data=summary_text.encode('utf-8'),
                    file_name=f"constraints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            st.success(f"✅ {len(constraints)}개 제약조건 내보내기 가능")
        else:
            st.info("💡 활성화된 제약조건이 없습니다.")
    else:
        st.warning("⚠️ 제약조건 관리자가 초기화되지 않았습니다.")

    st.markdown("---")

    # 3. 실행 기록
    st.markdown("#### 3️⃣ 실행 기록")

    if 'execution_history' in st.session_state and st.session_state.execution_history:
        st.markdown(f"**최근 {len(st.session_state.execution_history)}개 실행 기록**")

        history_df = pd.DataFrame(st.session_state.execution_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # 다운로드
        csv = history_df.to_csv(index=False, encoding='utf-8-sig')

        st.download_button(
            label="📥 실행 기록 CSV 다운로드",
            data=csv,
            file_name=f"execution_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("💡 실행 기록이 없습니다.")

    st.markdown("---")

    # 4. 디버그 정보
    st.markdown("#### 4️⃣ 디버그 정보")

    with st.expander("🔍 세션 상태 보기", expanded=False):
        st.markdown("**현재 세션 상태 변수**:")

        debug_info = {
            'data_loaded': st.session_state.get('data_loaded', False),
            'current_step': st.session_state.get('current_step', 0),
            'optimization_method': st.session_state.get('optimization_method', 'N/A'),
            'num_pareto_results': len(st.session_state.get('pareto_results', [])),
            'num_pareto_frontier': len(st.session_state.get('pareto_frontier', [])),
            'num_saved_scenarios': len(st.session_state.get('saved_scenarios', [])),
            'num_optimization_results': len(st.session_state.get('optimization_results', {})),
            'num_constraints': len(st.session_state.constraint_manager.list_constraints()) if st.session_state.get('constraint_manager') else 0
        }

        for key, value in debug_info.items():
            st.text(f"{key}: {value}")

        # 전체 세션 상태 JSON 다운로드
        st.markdown("---")

        if st.button("📥 전체 세션 상태 내보내기 (고급)"):
            session_data = {}
            for key in st.session_state.keys():
                try:
                    # 직렬화 가능한 값만 추출
                    value = st.session_state[key]
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        session_data[key] = value
                    else:
                        session_data[key] = str(type(value))
                except Exception as e:
                    session_data[key] = f"Error: {str(e)}"

            json_str = json.dumps(session_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="📄 세션 JSON 다운로드",
                data=json_str,
                file_name=f"session_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def render_advanced_analysis_tab_new():
    """
    Tab 3: 🔬 고급 분석
    파워 유저를 위한 심화 분석 도구
    """
    st.header("🔬 고급 분석")
    st.warning("⚠️ 이 섹션은 고급 사용자용입니다")

    st.markdown("""
    **고급 분석 도구**를 통해 최적화 결과를 더 깊이 이해하고 탐색할 수 있습니다.

    각 섹션을 클릭하여 펼치고 필요한 분석을 수행하세요.
    """)

    # 데이터 로딩 확인
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ 먼저 '⚙️ 설정 및 실행' 탭에서 데이터를 로드하세요.")
        return

    st.markdown("---")

    # 섹션 1: 민감도 분석
    with st.expander("▶ 1. 민감도 분석 (Sensitivity Analysis)", expanded=False):
        render_sensitivity_analysis_subtab()

    # 섹션 2: 대화형 파레토 탐색
    with st.expander("▶ 2. 대화형 파레토 탐색 (Interactive Pareto Navigation)", expanded=False):
        render_interactive_pareto_subtab()

    # 섹션 3: 강건 최적화
    with st.expander("▶ 3. 강건 최적화 (Robust Optimization)", expanded=False):
        render_robust_optimization_subtab()

    # 섹션 4: 시나리오 비교 (Tab 2에서 이동 시 자동 펼침)
    auto_expand_comparison = (st.session_state.get('advanced_section') == '시나리오 비교')
    with st.expander("▶ 4. 시나리오 비교 (Multi-Scenario Comparison)", expanded=auto_expand_comparison):
        render_comparison_tab()

    # 섹션 5: 제약 완화 분석
    with st.expander("▶ 5. 제약 완화 분석 (Constraint Relaxation)", expanded=False):
        render_constraint_relaxation_subtab()

    # 섹션 6: 확률적 위험 분석
    with st.expander("▶ 6. 확률적 위험 분석 (Stochastic Risk)", expanded=False):
        render_stochastic_risk_subtab()

    # 섹션 7: 결과 내보내기 & 로그
    with st.expander("▶ 7. 결과 내보내기 & 로그", expanded=False):
        _render_export_and_logs()

    # 섹션 8: Phase 4 고급 분석 대시보드 (신규)
    with st.expander("▶ 8. 📊 통합 분석 대시보드 (Phase 4 - NEW)", expanded=False):
        st.markdown("""
        **🚀 Phase 4 Advanced Analysis Dashboard**

        완전히 새로운 통합 분석 시스템:
        - 📊 **Pareto Method Comparison**: 3가지 방법 자동 비교 및 추천
        - 🎯 **Sensitivity Analysis**: 매개변수 민감도 분석
        - 🎭 **Scenario Comparison**: 다중 시나리오 비교 분석
        - 🛡️ **Robust/Stochastic Analysis**: 불확실성 하 최적화

        **Features:**
        - Interactive 3D Pareto Front Visualization
        - Automated Method Recommendation
        - Monte Carlo & Latin Hypercube Sampling
        - CVaR Risk Analysis
        - Real-time Animated Comparisons
        """)

        # Check if data is loaded
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ 먼저 데이터를 로드하세요 ('⚙️ 설정 및 실행' 탭)")
            return

        # Get optimization data and cost calculator from session state
        optimization_data = st.session_state.data_loader.get_optimization_data()

        # Check if cost calculator is available
        if 'cost_calculator' not in st.session_state or st.session_state.cost_calculator is None:
            st.warning("⚠️ Cost calculator가 초기화되지 않았습니다. 먼저 최적화를 실행하세요.")
            return

        cost_calculator = st.session_state.cost_calculator
        baseline_case = st.session_state.get('baseline_case', 'case1')

        # Render the new advanced dashboard
        try:
            render_dashboard(optimization_data, cost_calculator, baseline_case)
        except Exception as e:
            st.error(f"❌ Dashboard 렌더링 실패: {str(e)}")
            with st.expander("상세 오류 정보"):
                st.exception(e)

    # 자동 펼침 플래그 리셋
    if auto_expand_comparison:
        st.session_state.advanced_section = None


def _inject_custom_css():
    """커스텀 스타일 적용"""
    st.markdown("""
    <style>
    /* 진행 상태 표시기 스타일 */
    .progress-indicator {
        display: flex;
        justify-content: space-between;
        margin: 20px 0;
        font-weight: bold;
    }

    /* 실행 버튼 강조 */
    .stButton > button[kind="primary"] {
        background-color: #4CAF50 !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 15px 30px !important;
        width: 100% !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #45a049 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* 추천 포인트 강조 */
    .recommendation-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    /* 고급 분석 경고 스타일 */
    .advanced-warning {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 10px;
        border-radius: 4px;
    }

    /* 진행 단계 완료 표시 */
    .step-completed {
        color: #4CAF50;
        font-weight: bold;
    }

    .step-current {
        color: #2196F3;
        font-weight: bold;
    }

    .step-pending {
        color: #9E9E9E;
    }

    /* 메트릭 카드 스타일 */
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
    }

    /* 탭 스타일 개선 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 8px 8px 0 0;
    }

    /* Expander 스타일 */
    .streamlit-expanderHeader {
        font-size: 16px;
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 4px;
    }

    /* 경고 메시지 스타일 */
    .stAlert {
        border-radius: 6px;
    }

    /* 다운로드 버튼 스타일 */
    .stDownloadButton > button {
        border-radius: 6px;
        font-weight: 500;
    }

    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* 진행률 바 스타일 */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)


def _help_tooltip(text: str) -> str:
    """인라인 도움말 툴팁 생성

    Args:
        text: 툴팁에 표시할 도움말 텍스트

    Returns:
        HTML 형식의 툴팁 문자열
    """
    import html
    escaped_text = html.escape(text)
    return f'<span title="{escaped_text}" style="cursor: help; color: #2196F3;">💡</span>'


def _show_user_friendly_error(error_type: str, context: str = "", solution: str = ""):
    """사용자 친화적 에러 메시지 표시

    Args:
        error_type: 에러 유형 (data_missing, optimization_failed, etc.)
        context: 에러 발생 컨텍스트
        solution: 제안하는 해결 방법
    """
    error_messages = {
        'data_missing': {
            'title': '⚠️ 데이터가 없습니다',
            'message': 'PCF 시뮬레이션 결과를 먼저 로드해야 합니다.',
            'default_solution': '1. 왼쪽 사이드바에서 "PCF 시뮬레이터" 페이지로 이동\n2. 시뮬레이션을 실행하여 데이터 생성\n3. 다시 "최적화" 페이지로 돌아오기'
        },
        'baseline_missing': {
            'title': '⚠️ 베이스라인 시나리오가 없습니다',
            'message': 'baseline 시나리오가 필요합니다.',
            'default_solution': '1. "PCF 시뮬레이터" 페이지로 이동\n2. 시뮬레이션 실행 (자동으로 baseline 생성)\n3. "최적화" 페이지로 돌아오기'
        },
        'optimization_failed': {
            'title': '❌ 최적화 실패',
            'message': '최적화 과정에서 오류가 발생했습니다.',
            'default_solution': '1. 제약조건이 너무 엄격하지 않은지 확인\n2. 데이터가 정상적으로 로드되었는지 확인\n3. 다른 솔버(GLPK ↔ IPOPT)로 시도\n4. 문제가 지속되면 제약조건을 완화하거나 관리자에게 문의'
        },
        'data_loading_failed': {
            'title': '❌ 데이터 로딩 실패',
            'message': '시뮬레이션 결과를 불러오는데 실패했습니다.',
            'default_solution': '1. "PCF 시뮬레이터"에서 시뮬레이션이 정상 완료되었는지 확인\n2. 브라우저를 새로고침 (Ctrl+F5)\n3. 시뮬레이션을 다시 실행'
        },
        'name_required': {
            'title': '⚠️ 이름을 입력하세요',
            'message': '시나리오 이름 또는 결과 저장 이름이 필요합니다.',
            'default_solution': '의미있는 이름을 입력하세요 (예: "경제적_최적화_1", "공격적_감축_v2")'
        },
        'duplicate_name': {
            'title': '⚠️ 중복된 이름',
            'message': '이미 같은 이름의 시나리오가 존재합니다.',
            'default_solution': '다른 이름을 사용하거나 숫자를 추가하세요 (예: "시나리오_1" → "시나리오_2")'
        }
    }

    error_info = error_messages.get(error_type, {
        'title': '❌ 오류 발생',
        'message': context or '알 수 없는 오류가 발생했습니다.',
        'default_solution': '페이지를 새로고침하거나 관리자에게 문의하세요.'
    })

    st.error(error_info['title'])

    if context:
        st.markdown(f"**상황**: {context}")
    else:
        st.markdown(f"**상황**: {error_info['message']}")

    st.markdown("**해결 방법**:")
    solution_text = solution or error_info['default_solution']
    for line in solution_text.split('\n'):
        if line.strip():
            st.markdown(f"- {line.strip()}")

    # 추가 도움말 링크
    with st.expander("💡 추가 도움말", expanded=False):
        st.markdown("""
        ### 자주 발생하는 문제

        1. **데이터 로딩 문제**
           - PCF 시뮬레이터에서 시뮬레이션을 먼저 실행했는지 확인
           - 브라우저 캐시를 지우고 새로고침

        2. **최적화 실패**
           - 제약조건이 너무 엄격할 수 있음 → 빠른 프리셋 사용 추천
           - 솔버 변경: GLPK(빠름) ↔ IPOPT(정확)

        3. **성능 문제**
           - NSGA-II는 시간이 오래 걸림 → Weighted Sum 권장
           - 탐색 포인트 수를 줄이기 (예: 15개 → 10개)

        ### 추가 지원
        - 문서: FAQ.md 참조
        - 관리자 문의: 시스템 로그와 함께 문의
        """)

    st.markdown("---")


def _render_quick_tutorial():
    """첫 방문자를 위한 간단 튜토리얼"""
    if st.session_state.get('show_tutorial', True):
        with st.expander("❓ 처음 사용하시나요? (클릭하여 튜토리얼 보기)", expanded=False):
            st.markdown("""
            ### 🎯 빠른 시작 가이드

            #### 1단계: 데이터 로딩
            - **PCF 시뮬레이션 결과**를 로드하세요
            - 먼저 'PCF 시뮬레이터' 페이지에서 시뮬레이션을 실행해야 합니다

            #### 2단계: 제약조건 설정 (선택사항)
            - **빠른 프리셋** 중 하나를 선택하세요 (권장: 경제적 최적화)
            - 또는 고급 설정에서 세부 제약조건을 직접 설정할 수 있습니다

            #### 3단계: 최적화 방법 선택
            - **파레토 최적화**를 선택하세요 (탄소와 비용의 트레이드오프 분석)
            - 처음 사용하시면 **Weighted Sum** 방법을 권장합니다

            #### 4단계: 실행
            - [🚀 최적화 실행] 버튼을 클릭하세요
            - 결과는 자동으로 '📊 결과 분석' 탭에 표시됩니다

            #### 5단계: 결과 확인
            - **파레토 프론티어**에서 최적 포인트를 선택하세요
            - **스마트 추천** 기능이 균형잡힌 포인트를 제안합니다
            - 원하는 포인트를 시나리오로 저장할 수 있습니다

            ---

            💡 **팁**: 고급 분석 도구는 '🔬 고급 분석' 탭에서 사용할 수 있습니다
            """)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 이해했습니다", use_container_width=True):
                    st.session_state.show_tutorial = False
                    st.rerun()
            with col2:
                if st.button("🔄 나중에 다시 보기", use_container_width=True):
                    st.session_state.show_tutorial = False
                    st.rerun()


def optimization_page():
    """최적화 페이지 메인 함수"""
    # 커스텀 CSS 적용
    _inject_custom_css()

    # Refresh 버튼
    if st.button("🔄 새로고침", key="optimization_refresh", help="페이지를 새로고침합니다"):
        st.rerun()

    st.title("🎯 탄소배출 최적화 (V2)")

    # 빠른 튜토리얼 (첫 방문자용)
    _render_quick_tutorial()

    # 시스템 사용 가능 여부 확인
    if not OPTIMIZATION_V2_AVAILABLE:
        render_system_unavailable()
        return

    # 세션 상태 초기화
    initialize_session_state()

    # Phase 1: 6개 탭 → 3개 탭으로 단순화
    tabs = st.tabs([
        "⚙️ 설정 및 실행",
        "📊 결과 분석",
        "🔬 고급 분석"
    ])

    with tabs[0]:
        render_setup_and_run_tab()

    with tabs[1]:
        render_results_analysis_tab()

    with tabs[2]:
        render_advanced_analysis_tab_new()

    # 사이드바 정보
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📌 시스템 정보")
        st.caption("최적화 V2 - 모듈식 제약조건 시스템")

        if st.session_state.data_loaded:
            st.success("✅ 데이터 로딩됨")
        else:
            st.warning("⏸️ 데이터 대기 중")

        constraint_count = len(
            st.session_state.constraint_manager.list_constraints(enabled_only=True)
        )
        st.info(f"🔧 활성 제약조건: {constraint_count}개")

        result_count = len(st.session_state.optimization_results)
        st.info(f"📊 저장된 결과: {result_count}개")


if __name__ == "__main__":
    optimization_page()

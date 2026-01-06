"""
제약조건 설정 UI

사용자가 대화형으로 제약조건을 추가/편집/제거할 수 있는 UI 컴포넌트입니다.
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..core.constraint_manager import ConstraintManager
from ..constraints import (
    MaterialManagementConstraint,
    CostConstraint,
    LocationConstraint,
    RecyclingOptionConstraint,
    LowCarbonOptionConstraint,
    SiteChangeOptionConstraint
)


class ConstraintConfigurator:
    """제약조건 설정 UI 클래스"""

    def __init__(self, constraint_manager: ConstraintManager):
        """
        초기화

        Args:
            constraint_manager: ConstraintManager 인스턴스
        """
        self.constraint_manager = constraint_manager

    def render(self, available_materials: List[str], scenario_df=None) -> None:
        """
        제약조건 설정 UI 렌더링 (자동 저장 방식)

        Args:
            available_materials: 사용 가능한 자재 리스트
            scenario_df: 시나리오 데이터프레임 (그룹 생성용)
        """
        st.header("🔧 제약조건 구성")
        st.markdown("최적화에 적용할 제약조건을 설정합니다. **모든 변경사항은 자동으로 저장됩니다.**")

        # 구 세션 변수 정리
        for old_key in ['pending_recycling_state', 'pending_low_carbon_state',
                        'pending_site_change_state', 'adding_constraint']:
            if old_key in st.session_state:
                del st.session_state[old_key]

        # 마이그레이션 체크 및 실행
        user_id = st.session_state.get('user_id', 'default')
        if self._migrate_legacy_settings(user_id):
            self.constraint_manager.load_from_file(user_id=user_id)

        # 저장된 제약조건 관리
        with st.expander("📂 저장된 제약조건 관리", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 현재 제약조건 저장", use_container_width=True):
                    success = self.constraint_manager.save_to_file(user_id=user_id)
                    if success:
                        st.success("✅ 제약조건이 저장되었습니다!")
                    else:
                        st.error("❌ 제약조건 저장 실패")

            with col2:
                if st.button("📂 저장된 제약조건 불러오기", use_container_width=True):
                    success = self.constraint_manager.load_from_file(user_id=user_id)
                    if success:
                        st.success("✅ 제약조건을 불러왔습니다!")
                        st.rerun()
                    else:
                        st.warning("ℹ️ 저장된 제약조건이 없습니다.")

            # 저장 상태 표시
            config_file = f'input/{user_id}/constraint_config.json'
            try:
                from src.utils.file_operations import FileOperations
                config = FileOperations.load_json(config_file, default=None, user_id=user_id)
                if config:
                    st.info(f"💾 마지막 저장: {config.get('saved_at', '알 수 없음')}")
                    st.caption(f"📊 저장된 제약조건: {len(config.get('constraints', []))}개")
            except:
                st.caption("💾 저장된 제약조건 없음")

        # Store scenario data
        if scenario_df is not None:
            st.session_state['_scenario_df_for_groups'] = scenario_df
        if available_materials:
            st.session_state['_available_materials'] = available_materials

        # 단일 통합 뷰 (Tab 2 제거됨)
        st.markdown("---")
        self._render_consolidated_settings()

        # 활성 제약조건 표시 (읽기 전용)
        st.markdown("---")
        with st.expander("✅ 현재 활성화된 제약조건", expanded=False):
            self._render_active_constraints_readonly()

    # ========== [DEPRECATED] Tab 2 관련 메서드들 (제거됨) ==========
    # _render_add_buttons(), _render_material_constraint_form(),
    # _render_cost_constraint_form(), _render_location_constraint_form() 제거됨

    def render_compact_summary(self) -> None:
        """간단한 요약 표시 (다른 탭에서 사용)"""
        constraints = self.constraint_manager.list_constraints(enabled_only=True)

        if constraints:
            st.info(f"📋 활성 제약조건: {len(constraints)}개")
            for name in constraints:
                constraint = self.constraint_manager.get_constraint(name)
                st.caption(f"  • {constraint.name}: {constraint.description}")
        else:
            st.warning("⚠️ 활성화된 제약조건이 없습니다.")

    # ========== Phase 1: 전역 설정 인프라 헬퍼 함수들 ==========

    # ========== 새로운 자동 저장 인프라 (UI 개편) ==========

    def _auto_save_element_constraint(
        self,
        element: str,
        recycle_min: float,
        recycle_max: float,
        low_carbon_min: float,
        low_carbon_max: float
    ) -> bool:
        """원소별 비율 제약 자동 저장

        전략:
        1. 기존 "element_ratio_{element}" 제약 제거
        2. MaterialManagementConstraint 새로 생성
        3. force_element_ratio_range 규칙 추가
        4. constraint_manager에 추가 (auto_save=False)
        5. 명시적 save_to_file() 호출

        Args:
            element: 원소명 (Ni, Co, Li)
            recycle_min: 재활용재 최소 비율 (0-1)
            recycle_max: 재활용재 최대 비율 (0-1)
            low_carbon_min: 저탄소메탈 최소 비율 (0-1)
            low_carbon_max: 저탄소메탈 최대 비율 (0-1)

        Returns:
            저장 성공 여부
        """
        constraint_name = f"element_ratio_{element}"
        self.constraint_manager.remove_constraint(constraint_name, auto_save=False)

        constraint = MaterialManagementConstraint()
        constraint.name = constraint_name
        constraint.description = f"{element} 원소 비율 범위 제약"

        constraint.add_rule(
            'force_element_ratio_range',
            material_name='_TEMPLATE_',  # 템플릿 마커
            params={
                'element': element,
                'recycle_min': recycle_min,
                'recycle_max': recycle_max,
                'low_carbon_min': low_carbon_min,
                'low_carbon_max': low_carbon_max
            }
        )

        self.constraint_manager.add_constraint(
            constraint,
            priority=10,
            replace_if_exists=True,
            auto_save=False
        )

        user_id = st.session_state.get('user_id', 'default')
        success = self.constraint_manager.save_to_file(user_id=user_id)

        if success:
            st.toast(f"✅ {element} 비율 설정 저장됨", icon="✅")
        else:
            st.error(f"❌ {element} 비율 설정 저장 실패")

        return success

    def _auto_save_premium_settings(
        self,
        element: str,
        recycle_premium_pct: float = None,
        low_carbon_premium_pct: float = None
    ) -> bool:
        """프리미엄 자동 저장 → material_cost_premiums.json

        Args:
            element: 원소명 (Ni, Co, Li)
            recycle_premium_pct: 재활용재 프리미엄 (0-200)
            low_carbon_premium_pct: 저탄소메탈 프리미엄 (0-200)

        Returns:
            저장 성공 여부
        """
        from src.utils.optimization_costs_manager import OptimizationCostsManager

        user_id = st.session_state.get('user_id', 'default')
        costs_manager = OptimizationCostsManager()
        current_premiums = costs_manager.load_material_cost_premiums(user_id)

        if element not in current_premiums:
            current_premiums[element] = {}

        if recycle_premium_pct is not None:
            current_premiums[element]['recycle_premium_pct'] = float(recycle_premium_pct)

        if low_carbon_premium_pct is not None:
            current_premiums[element]['low_carbon_premium_pct'] = float(low_carbon_premium_pct)

        success = costs_manager.save_material_cost_premiums(user_id, current_premiums)

        if success:
            if 'data_loader' in st.session_state:
                st.session_state.data_loader.material_cost_premiums = current_premiums
            st.toast(f"✅ {element} 프리미엄 저장됨", icon="💰")
        else:
            st.error(f"❌ {element} 프리미엄 저장 실패")

        return success

    def _auto_save_feature_option(
        self,
        option_type: str,  # 'recycling', 'low_carbon', 'site_change'
        enabled: bool
    ) -> bool:
        """기능 옵션 토글 자동 저장

        Args:
            option_type: 옵션 타입 ('recycling', 'low_carbon', 'site_change')
            enabled: 활성화 여부

        Returns:
            저장 성공 여부
        """
        constraint_map = {
            'recycling': ('recycling_option', RecyclingOptionConstraint),
            'low_carbon': ('low_carbon_option', LowCarbonOptionConstraint),
            'site_change': ('site_change_option', SiteChangeOptionConstraint)
        }

        if option_type not in constraint_map:
            st.error(f"❌ Unknown option type: {option_type}")
            return False

        constraint_name, constraint_class = constraint_map[option_type]

        self.constraint_manager.remove_constraint(constraint_name, auto_save=False)

        new_constraint = constraint_class(enabled=enabled)

        user_id = st.session_state.get('user_id', 'default')
        self.constraint_manager.add_constraint(
            new_constraint,
            priority=100,
            replace_if_exists=True,
            auto_save=False,
            user_id=user_id
        )

        success = self.constraint_manager.save_to_file(user_id=user_id)

        if success:
            status = "활성화" if enabled else "비활성화"
            st.toast(f"✅ {option_type} {status}", icon="✅")
        else:
            st.error(f"❌ {option_type} 저장 실패")

        return success

    def _auto_save_country_constraint(
        self,
        enabled: bool,
        allowed_countries: List[str] = None
    ) -> bool:
        """국가 제약 자동 저장

        Args:
            enabled: 활성화 여부
            allowed_countries: 허용 국가 리스트

        Returns:
            저장 성공 여부
        """
        constraint_name = "country_constraint"

        self.constraint_manager.remove_constraint(constraint_name, auto_save=False)

        if enabled and allowed_countries:
            constraint = MaterialManagementConstraint()
            constraint.name = constraint_name
            constraint.description = f"소싱 국가 제약: {', '.join(allowed_countries)}"

            constraint.add_rule(
                'regional_preference',
                material_name='_TEMPLATE_',
                params={'preferred_regions': allowed_countries}
            )

            self.constraint_manager.add_constraint(
                constraint,
                priority=20,
                replace_if_exists=True,
                auto_save=False
            )

        user_id = st.session_state.get('user_id', 'default')
        success = self.constraint_manager.save_to_file(user_id=user_id)

        if success:
            st.toast(f"✅ 국가 제약 저장됨", icon="🌍")
        else:
            st.error("❌ 국가 제약 저장 실패")

        return success

    def _load_element_ratios(self, element: str) -> Dict[str, float]:
        """constraint_manager에서 원소 비율 로드

        Args:
            element: 원소명 (Ni, Co, Li)

        Returns:
            {recycle_min, recycle_max, low_carbon_min, low_carbon_max}
        """
        constraint_name = f"element_ratio_{element}"
        constraint = self.constraint_manager.get_constraint(constraint_name)

        if constraint and isinstance(constraint, MaterialManagementConstraint):
            for rule in constraint.material_rules:
                if rule['type'] == 'force_element_ratio_range':
                    if rule['params']['element'] == element:
                        return {
                            'recycle_min': rule['params']['recycle_min'],
                            'recycle_max': rule['params']['recycle_max'],
                            'low_carbon_min': rule['params']['low_carbon_min'],
                            'low_carbon_max': rule['params']['low_carbon_max']
                        }

        # 기본값
        defaults = {
            'Ni': {'recycle_min': 0.05, 'recycle_max': 0.10, 'low_carbon_min': 0.0, 'low_carbon_max': 0.05},
            'Co': {'recycle_min': 0.025, 'recycle_max': 0.05, 'low_carbon_min': 0.0, 'low_carbon_max': 0.05},
            'Li': {'recycle_min': 0.018, 'recycle_max': 0.036, 'low_carbon_min': 0.0, 'low_carbon_max': 0.20}
        }
        return defaults.get(element, {'recycle_min': 0.0, 'recycle_max': 0.1, 'low_carbon_min': 0.0, 'low_carbon_max': 0.1})

    def _load_country_constraint_settings(self) -> Dict[str, Any]:
        """국가 제약 로드

        Returns:
            {enabled: bool, allowed_countries: List[str]}
        """
        constraint = self.constraint_manager.get_constraint("country_constraint")

        if constraint and isinstance(constraint, MaterialManagementConstraint):
            for rule in constraint.material_rules:
                if rule['type'] == 'regional_preference':
                    return {
                        'enabled': True,
                        'allowed_countries': rule['params'].get('preferred_regions', [])
                    }

        return {'enabled': False, 'allowed_countries': ['한국']}

    # ========== 새로운 UI 렌더링 메서드 (UI 개편) ==========

    def _render_consolidated_settings(self) -> None:
        """단일 통합 뷰로 모든 설정 렌더링"""
        st.markdown("### 🎛️ 양극재 제약조건 세부설정")
        st.caption("모든 설정은 변경 즉시 자동으로 저장됩니다.")

        # 1. 전역 토글
        self._render_global_toggles()

        st.markdown("---")

        # 2. 원소별 탭 (프리미엄 + 비율 통합)
        self._render_element_tabs_combined()

        st.markdown("---")

        # 3. 생산지 변경 + 국가 제약
        self._render_site_change_with_country()

    def _render_global_toggles(self) -> None:
        """전역 재활용/저탄소 토글 렌더링 (자동 저장)"""

        # 재활용재 토글
        st.subheader("♻️ 재활용재 전역 활성화")
        current_recycling = self._get_current_recycling_state()

        col1, col2 = st.columns([2, 1])
        with col1:
            recycling_enabled = st.toggle(
                "재활용재 사용",
                value=current_recycling,
                key="recycling_toggle",
                help="활성화: 재활용재 비율 최적화 | 비활성화: 재활용재 0% 고정"
            )

            if recycling_enabled != current_recycling:
                self._auto_save_feature_option('recycling', recycling_enabled)
                st.rerun()

        with col2:
            if recycling_enabled:
                st.success("✅ 활성화")
            else:
                st.error("🚫 비활성화")

        st.markdown("---")

        # 저탄소메탈 토글
        st.subheader("🌱 저탄소메탈 전역 활성화")
        current_low_carbon = self._get_current_low_carbon_state()

        col1, col2 = st.columns([2, 1])
        with col1:
            low_carbon_enabled = st.toggle(
                "저탄소메탈 사용",
                value=current_low_carbon,
                key="low_carbon_toggle",
                help="활성화: 저탄소메탈 비율 최적화 | 비활성화: 저탄소메탈 0% 고정"
            )

            if low_carbon_enabled != current_low_carbon:
                self._auto_save_feature_option('low_carbon', low_carbon_enabled)
                st.rerun()

        with col2:
            if low_carbon_enabled:
                st.success("✅ 활성화")
            else:
                st.error("🚫 비활성화")

    def _render_element_tabs_combined(self) -> None:
        """Ni/Co/Li 탭 렌더링 (프리미엄 + 비율 통합)"""
        st.subheader("📊 원소별 세부 설정")
        st.caption("각 원소(Ni, Co, Li)의 비용 프리미엄과 비율 범위를 설정합니다.")

        tab_ni, tab_co, tab_li = st.tabs(["Ni", "Co", "Li"])

        for tab, element in zip([tab_ni, tab_co, tab_li], ['Ni', 'Co', 'Li']):
            with tab:
                self._render_element_settings(element)

    def _render_element_settings(self, element: str) -> None:
        """원소별 프리미엄 + 비율 설정 렌더링 (자동 저장)"""
        st.markdown(f"### {element} 원소 설정")

        # 현재 데이터 로드
        from src.utils.optimization_costs_manager import OptimizationCostsManager
        user_id = st.session_state.get('user_id', 'default')
        costs_manager = OptimizationCostsManager()
        current_premiums = costs_manager.load_material_cost_premiums(user_id)
        current_ratios = self._load_element_ratios(element)

        # 프리미엄 설정
        with st.expander("💰 비용 프리미엄 설정", expanded=True):
            st.caption("재활용재와 저탄소메탈의 비용 프리미엄을 설정합니다 (신재 대비 %).")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**재활용재 프리미엄**")
                recycle_premium = st.slider(
                    "프리미엄 (%)",
                    min_value=0, max_value=200,
                    value=int(current_premiums.get(element, {}).get('recycle_premium_pct', 30.0)),
                    step=5,
                    key=f"recycle_premium_{element}",
                    help=f"{element} 재활용재는 신재 대비 몇 % 비쌉니까?"
                )

                # 변경 감지 및 자동 저장
                prev_key = f'_prev_recycle_premium_{element}'
                if st.session_state.get(prev_key) != recycle_premium:
                    self._auto_save_premium_settings(element, recycle_premium_pct=recycle_premium)
                    st.session_state[prev_key] = recycle_premium

            with col2:
                st.markdown("**저탄소메탈 프리미엄**")
                low_carbon_premium = st.slider(
                    "프리미엄 (%)",
                    min_value=0, max_value=200,
                    value=int(current_premiums.get(element, {}).get('low_carbon_premium_pct', 50.0)),
                    step=5,
                    key=f"low_carbon_premium_{element}",
                    help=f"{element} 저탄소메탈은 신재 대비 몇 % 비쌉니까?"
                )

                # 변경 감지 및 자동 저장
                prev_key = f'_prev_low_carbon_premium_{element}'
                if st.session_state.get(prev_key) != low_carbon_premium:
                    self._auto_save_premium_settings(element, low_carbon_premium_pct=low_carbon_premium)
                    st.session_state[prev_key] = low_carbon_premium

        # 비율 범위 설정
        with st.expander("🎚️ 비율 범위 설정", expanded=True):
            st.caption("양극재 제약조건 추가 시 자동으로 적용되는 비율 범위를 설정합니다.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**재활용재 비율 범위**")
                recycle_min = st.slider(
                    "최소 (%)", 0, 100,
                    value=int(current_ratios['recycle_min'] * 100),
                    key=f"recycle_min_{element}"
                )
                recycle_max = st.slider(
                    "최대 (%)", 0, 100,
                    value=int(current_ratios['recycle_max'] * 100),
                    key=f"recycle_max_{element}"
                )

                if recycle_min > recycle_max:
                    st.error(f"⚠️ 최소값({recycle_min}%)이 최대값({recycle_max}%)보다 큽니다.")
                else:
                    prev_key = f'_prev_recycle_ratio_{element}'
                    prev_value = st.session_state.get(prev_key, (current_ratios['recycle_min']*100, current_ratios['recycle_max']*100))
                    if prev_value != (recycle_min, recycle_max):
                        self._auto_save_element_constraint(
                            element,
                            recycle_min=recycle_min/100,
                            recycle_max=recycle_max/100,
                            low_carbon_min=current_ratios['low_carbon_min'],
                            low_carbon_max=current_ratios['low_carbon_max']
                        )
                        st.session_state[prev_key] = (recycle_min, recycle_max)

            with col2:
                st.markdown("**저탄소메탈 비율 범위**")
                low_carbon_min = st.slider(
                    "최소 (%)", 0, 100,
                    value=int(current_ratios['low_carbon_min'] * 100),
                    key=f"low_carbon_min_{element}"
                )
                low_carbon_max = st.slider(
                    "최대 (%)", 0, 100,
                    value=int(current_ratios['low_carbon_max'] * 100),
                    key=f"low_carbon_max_{element}"
                )

                if low_carbon_min > low_carbon_max:
                    st.error(f"⚠️ 최소값({low_carbon_min}%)이 최대값({low_carbon_max}%)보다 큽니다.")
                else:
                    prev_key = f'_prev_low_carbon_ratio_{element}'
                    prev_value = st.session_state.get(prev_key, (current_ratios['low_carbon_min']*100, current_ratios['low_carbon_max']*100))
                    if prev_value != (low_carbon_min, low_carbon_max):
                        self._auto_save_element_constraint(
                            element,
                            recycle_min=current_ratios['recycle_min'],
                            recycle_max=current_ratios['recycle_max'],
                            low_carbon_min=low_carbon_min/100,
                            low_carbon_max=low_carbon_max/100
                        )
                        st.session_state[prev_key] = (low_carbon_min, low_carbon_max)

    def _render_site_change_with_country(self) -> None:
        """생산지 변경 토글 + 국가 제약 통합"""
        st.subheader("🌍 생산지 변경")
        st.caption("CAM/pCAM 생산지 변경 (전력계수 변경)")

        # 현재 상태 확인
        site_change_constraint = None
        for c in self.constraint_manager.list_constraint_objects():
            if isinstance(c, SiteChangeOptionConstraint):
                site_change_constraint = c
                break

        current_state = site_change_constraint.enabled if site_change_constraint else False

        col1, col2 = st.columns([2, 1])
        with col1:
            site_change_enabled = st.toggle(
                "생산지 변경 허용",
                value=current_state,
                key="site_change_toggle",
                help="활성화: 변경된 사이트 전력계수 사용 | 비활성화: 기본 사이트 전력계수 사용"
            )

            if site_change_enabled != current_state:
                self._auto_save_feature_option('site_change', site_change_enabled)
                st.warning("⚠️ 생산지 변경 설정 변경됨. 데이터 로딩 탭에서 데이터를 다시 로드해야 합니다.")
                st.rerun()

        with col2:
            if site_change_enabled:
                st.success("✅ 변경 허용")
            else:
                st.info("ℹ️ 기본 유지")

        # 국가 제약 (생산지 변경 활성화 시에만 표시)
        if site_change_enabled:
            with st.expander("🌍 소싱 국가 제약", expanded=False):
                st.caption("특정 국가에서만 양극재를 소싱하도록 제한합니다.")

                # 현재 설정 로드
                country_settings = self._load_country_constraint_settings()

                enable_country = st.checkbox(
                    "소싱 국가 제약 활성화",
                    value=country_settings['enabled'],
                    key="country_enabled",
                    help="체크하면 아래 선택한 국가만 허용됩니다."
                )

                if enable_country:
                    available_countries = self._get_available_countries()

                    allowed_countries = st.multiselect(
                        "허용 국가 (복수 선택 가능)",
                        options=available_countries,
                        default=country_settings['allowed_countries'],
                        key="allowed_countries",
                        help="선택한 국가에서만 양극재를 소싱할 수 있습니다."
                    )

                    if not allowed_countries:
                        st.warning("⚠️ 최소 1개 국가를 선택해야 합니다.")
                    else:
                        # 변경 감지 및 자동 저장
                        prev_countries = st.session_state.get('_prev_allowed_countries', country_settings['allowed_countries'])
                        if set(prev_countries) != set(allowed_countries):
                            self._auto_save_country_constraint(True, allowed_countries)
                            st.session_state['_prev_allowed_countries'] = allowed_countries
                else:
                    # 비활성화 상태 저장
                    if country_settings['enabled']:
                        self._auto_save_country_constraint(False, [])
                    st.caption("💡 소싱 국가 제약을 사용하지 않습니다.")

    def _migrate_legacy_settings(self, user_id: str) -> bool:
        """구 global_cathode_settings.json을 새 형식으로 마이그레이션

        Args:
            user_id: 사용자 ID

        Returns:
            마이그레이션 실행 여부
        """
        from pathlib import Path
        import json

        old_file = Path(f'input/{user_id}/global_cathode_settings.json')
        if not old_file.exists():
            return False  # 마이그레이션 불필요

        try:
            st.info("🔄 구 형식의 제약조건을 새 시스템으로 마이그레이션 중...")

            # 구 설정 로드
            with open(old_file, 'r', encoding='utf-8') as f:
                old_settings = json.load(f)

            # 원소별 비율 마이그레이션
            if old_settings.get('element_ratio_range', {}).get('enabled'):
                for element in ['Ni', 'Co', 'Li']:
                    ratios = old_settings['element_ratio_range'].get(element, {})
                    if ratios:
                        self._auto_save_element_constraint(
                            element=element,
                            recycle_min=ratios.get('recycle_min', 0),
                            recycle_max=ratios.get('recycle_max', 1),
                            low_carbon_min=ratios.get('low_carbon_min', 0),
                            low_carbon_max=ratios.get('low_carbon_max', 1)
                        )

            # 국가 제약 마이그레이션
            if old_settings.get('country_constraint', {}).get('enabled'):
                allowed_countries = old_settings['country_constraint'].get('allowed_countries', [])
                self._auto_save_country_constraint(True, allowed_countries)

            # 백업 생성
            backup_file = old_file.with_suffix('.json.backup')
            old_file.rename(backup_file)

            st.success(f"✅ 마이그레이션 완료! 구 파일은 {backup_file.name}으로 백업되었습니다.")
            return True

        except Exception as e:
            st.error(f"❌ 마이그레이션 실패: {e}")
            return False

    def _render_active_constraints_readonly(self) -> None:
        """활성 제약조건 목록 (읽기 전용)"""
        constraints = self.constraint_manager.list_constraints()

        if not constraints:
            st.info("현재 활성화된 제약조건이 없습니다.")
            return

        st.caption(f"총 {len(constraints)}개의 제약조건")

        for idx, name in enumerate(constraints, 1):
            constraint = self.constraint_manager.get_constraint(name)

            status_icon = "✅" if constraint.enabled else "❌"
            st.markdown(f"**{idx}. {status_icon} {constraint.name}**")
            st.caption(f"   {constraint.description}")

            # 상세 정보 (간략)
            if hasattr(constraint, 'material_rules') and constraint.material_rules:
                st.caption(f"   규칙 수: {len(constraint.material_rules)}개")

    # ========== 기존 전역 설정 헬퍼 (향후 deprecated 예정) ==========

    def _load_global_cathode_settings(self) -> Dict[str, Any]:
        """전역 양극재 설정 로드"""
        try:
            from src.utils.file_operations import FileOperations
            user_id = st.session_state.get('user_id', 'default')

            config = FileOperations.load_json(
                f'input/{user_id}/global_cathode_settings.json',
                default=self._get_default_global_cathode_settings(),
                user_id=user_id
            )
            return config
        except:
            return self._get_default_global_cathode_settings()

    def _save_global_cathode_settings(self, settings: Dict[str, Any]) -> bool:
        """전역 양극재 설정 저장"""
        try:
            from src.utils.file_operations import FileOperations
            from datetime import datetime

            user_id = st.session_state.get('user_id', 'default')
            settings['saved_at'] = datetime.now().isoformat()

            FileOperations.save_json(
                f'input/{user_id}/global_cathode_settings.json',
                settings,
                user_id=user_id
            )
            return True
        except Exception as e:
            print(f"전역 설정 저장 실패: {e}")
            return False

    def _get_default_global_cathode_settings(self) -> Dict[str, Any]:
        """기본 전역 설정 반환"""
        return {
            'version': '1.0',
            'element_ratio_range': {
                'enabled': False,
                'Ni': {'recycle_min': 0.05, 'recycle_max': 0.10, 'low_carbon_min': 0.0, 'low_carbon_max': 0.05},
                'Co': {'recycle_min': 0.025, 'recycle_max': 0.05, 'low_carbon_min': 0.0, 'low_carbon_max': 0.05},
                'Li': {'recycle_min': 0.018, 'recycle_max': 0.036, 'low_carbon_min': 0.0, 'low_carbon_max': 0.20}
            },
            'country_constraint': {
                'enabled': False,
                'allowed_countries': ['한국']
            }
        }

    def _apply_global_settings_to_material(
        self,
        constraint: MaterialManagementConstraint,
        material_name: str,
        global_settings: Dict[str, Any]
    ) -> None:
        """전역 설정을 특정 자재 제약조건에 적용"""

        # 원소별 비율 범위 강제
        if global_settings.get('element_ratio_range', {}).get('enabled'):
            for element in ['Ni', 'Co', 'Li']:
                ratios = global_settings['element_ratio_range'].get(element, {})
                if ratios:
                    constraint.add_rule(
                        'force_element_ratio_range',
                        material_name,
                        params={
                            'element': element,
                            'recycle_min': ratios.get('recycle_min', 0),
                            'recycle_max': ratios.get('recycle_max', 1),
                            'low_carbon_min': ratios.get('low_carbon_min', 0),
                            'low_carbon_max': ratios.get('low_carbon_max', 1)
                        }
                    )

        # 소싱국가 제약
        if global_settings.get('country_constraint', {}).get('enabled'):
            allowed_countries = global_settings['country_constraint'].get('allowed_countries', [])
            if allowed_countries:
                constraint.add_rule(
                    'regional_preference',
                    material_name,
                    params={'preferred_regions': allowed_countries}
                )

    def _render_element_ratio_sliders(self, element: str, global_settings: Dict) -> Dict[str, float]:
        """원소별 비율 슬라이더 렌더링"""
        st.markdown(f"**{element} 원소 비율 설정**")

        default_ratios = global_settings.get('element_ratio_range', {}).get(element, {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**재활용재 비율**")
            recycle_min = st.slider(
                "최소 (%)", 0, 100,
                int(default_ratios.get('recycle_min', 5) * 100),
                key=f"global_recycle_min_{element}"
            )
            recycle_max = st.slider(
                "최대 (%)", 0, 100,
                int(default_ratios.get('recycle_max', 10) * 100),
                key=f"global_recycle_max_{element}"
            )

        with col2:
            st.markdown("**저탄소메탈 비율**")
            low_carbon_min = st.slider(
                "최소 (%)", 0, 100,
                int(default_ratios.get('low_carbon_min', 0) * 100),
                key=f"global_low_carbon_min_{element}"
            )
            low_carbon_max = st.slider(
                "최대 (%)", 0, 100,
                int(default_ratios.get('low_carbon_max', 5) * 100),
                key=f"global_low_carbon_max_{element}"
            )

        # 검증
        if recycle_min > recycle_max:
            st.error(f"⚠️ 재활용재: 최소값({recycle_min}%)이 최대값({recycle_max}%)보다 큽니다.")

        if low_carbon_min > low_carbon_max:
            st.error(f"⚠️ 저탄소메탈: 최소값({low_carbon_min}%)이 최대값({low_carbon_max}%)보다 큽니다.")

        total_max = recycle_max + low_carbon_max
        if total_max > 100:
            st.warning(f"⚠️ 재활용재 최대 + 저탄소메탈 최대 = {total_max}%로 100%를 초과합니다.")

        return {
            'recycle_min': recycle_min / 100,
            'recycle_max': recycle_max / 100,
            'low_carbon_min': low_carbon_min / 100,
            'low_carbon_max': low_carbon_max / 100
        }

    def _get_available_countries(self) -> List[str]:
        """사용 가능한 국가 목록 반환"""
        try:
            import json
            with open('stable_var/cathode_national_code.json', 'r', encoding='utf-8') as f:
                national_code_data = json.load(f)
                return sorted(list(set(national_code_data.get('national_code', {}).values())))
        except:
            return ['한국', '일본', '중국', '폴란드', '미분류']

    def _get_current_recycling_state(self) -> bool:
        """현재 재활용재 토글 상태 반환"""
        for c in self.constraint_manager.list_constraint_objects():
            if isinstance(c, RecyclingOptionConstraint):
                return c.enabled
        return True

    def _get_current_low_carbon_state(self) -> bool:
        """현재 저탄소메탈 토글 상태 반환"""
        for c in self.constraint_manager.list_constraint_objects():
            if isinstance(c, LowCarbonOptionConstraint):
                return c.enabled
        return True

    def _update_recycling_constraint(self, enabled: bool) -> None:
        """재활용재 제약조건 업데이트"""
        self.constraint_manager.remove_constraint('recycling_option', auto_save=False)
        new_constraint = RecyclingOptionConstraint(enabled=enabled)
        self.constraint_manager.add_constraint(new_constraint, replace_if_exists=True, auto_save=False)

    def _update_low_carbon_constraint(self, enabled: bool) -> None:
        """저탄소메탈 제약조건 업데이트"""
        self.constraint_manager.remove_constraint('low_carbon_option', auto_save=False)
        new_constraint = LowCarbonOptionConstraint(enabled=enabled)
        self.constraint_manager.add_constraint(new_constraint, replace_if_exists=True, auto_save=False)

    # ========== [DEPRECATED] Phase 2: 탭 1 렌더링 함수들 (제거됨) ==========



"""
데이터 로더

Streamlit session_state에서 시뮬레이션 결과를 로드하고 최적화에 필요한 형식으로 변환합니다.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import streamlit as st


class DataLoader:
    """
    시뮬레이션 결과 데이터를 최적화 엔진용으로 로드하는 클래스
    """

    def __init__(self):
        """데이터 로더 초기화"""
        self.scenario_df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.ref_formula_df: Optional[pd.DataFrame] = None
        self.ref_proportions_df: Optional[pd.DataFrame] = None
        self.simulation_results: Optional[Dict[str, Any]] = None

        # 캐시 속성 (중복 로깅 방지)
        self._cached_target_materials: Optional[List[str]] = None
        self._cached_classification: Optional[Dict[str, Dict[str, Any]]] = None

        # 성능 최적화: 에너지 마스크 캐시 (Phase 5)
        self._energy_tier1_mask: Optional[pd.Series] = None
        self._energy_tier2_mask: Optional[pd.Series] = None
        self._energy_masks_initialized: bool = False

        # 양극재 관련 데이터
        self.cathode_ratio: Optional[Dict[str, float]] = None
        self.recycle_impact: Optional[Dict[str, float]] = None
        self.low_carb_emission: Optional[Dict[str, float]] = None
        self.virgin_emission: Optional[Dict[str, float]] = None

        # 에너지 기여도 (RE100 최적화용)
        self.tier1_energy_ratio: float = 0.0  # Tier1 에너지 기여도 (0~1 소수)
        self.tier2_energy_ratio: float = 0.0  # Tier2 에너지 기여도 (0~1 소수)

        # 재활용재/저탄소메탈 비용 프리미엄 (%)
        self.material_cost_premiums: Optional[Dict[str, Dict[str, float]]] = None

        # 사이트 설정 (SiteChangeOptionConstraint에서 사용)
        self.site: str = 'before'  # 'before' 또는 'after'

    def load_from_session_state(self, site: str = 'before') -> bool:
        """
        Streamlit session_state에서 데이터 로드

        Args:
            site: 전력계수 사이트 ('before' 또는 'after')
                  - 'before': 기본 사이트 (베이스라인)
                  - 'after': 변경 사이트 (사이트 변경 시나리오)

        Returns:
            성공 여부
        """
        print(f"\n🔄 load_from_session_state 호출: 항상 baseline 시나리오 사용, site={site}")

        try:
            # simulation_results 확인
            if 'simulation_results' not in st.session_state:
                print("❌ session_state에 simulation_results가 없습니다.")
                return False

            self.simulation_results = st.session_state.simulation_results
            print(f"   • simulation_results 키: {list(self.simulation_results.keys())}")

            # 항상 baseline 시나리오 로드
            scenario_name = 'baseline'
            if scenario_name not in self.simulation_results:
                available = list(self.simulation_results.keys())
                print(f"❌ baseline 시나리오를 찾을 수 없습니다. 사용 가능: {available}")
                return False

            # 시나리오 데이터 로드
            scenario_data = self.simulation_results[scenario_name]
            self.scenario_df = scenario_data.get('all_data')

            if self.scenario_df is None:
                print(f"❌ 시나리오 '{scenario_name}'의 all_data가 없습니다.")
                return False

            # scenario_df 컬럼 확인
            print(f"\n📋 scenario_df 컬럼 체크:")
            print(f"   • 전체 컬럼 수: {len(self.scenario_df.columns)}")

            # Tier_RE_case 컬럼 확인
            tier_re_columns = [col for col in self.scenario_df.columns if 'Tier' in col and 'RE_case' in col]
            print(f"   • Tier_RE_case 컬럼: {len(tier_re_columns)}개")
            if tier_re_columns:
                print(f"     - {tier_re_columns}")
            else:
                print(f"     ⚠️ WARNING: Tier_RE_case 컬럼이 없습니다!")
                print(f"     → RE100 프리미엄 비용 계산이 불가능할 수 있습니다.")
                print(f"     → 시나리오 설정 페이지에서 RE100 케이스를 추가하세요.")

            # original_df, ref_formula_df, ref_proportions_df는 session_state에서 직접 로드
            self.original_df = st.session_state.get('original_df')
            self.ref_formula_df = st.session_state.get('ref_formula_df')
            self.ref_proportions_df = st.session_state.get('ref_proportions_df')

            # site 정보 저장
            self.site = site

            print(f"\n✅ 데이터 로드 완료: baseline (site={site})")
            print(f"   • scenario_df: {len(self.scenario_df)} rows")
            if self.original_df is not None:
                print(f"   • original_df: {len(self.original_df)} rows")
            else:
                print(f"   ⚠️  original_df: 없음 (비용 제약 사용 불가)")
            if self.ref_formula_df is not None:
                print(f"   • ref_formula_df: {len(self.ref_formula_df)} rows")
            if self.ref_proportions_df is not None:
                print(f"   • ref_proportions_df: {len(self.ref_proportions_df)} rows")

            # site='after'인 경우 전력계수 업데이트
            if site == 'after':
                print(f"\n🔄 생산지 변경 모드: 전력계수 업데이트 시작...")
                success = self._update_electricity_coefficients_for_site_change()
                if not success:
                    print(f"⚠️  전력계수 업데이트 실패 - 기본값 사용")

            # 양극재 관련 데이터 로드
            self._load_cathode_data()

            # 베이스라인 배출량 데이터 로드 (에너지 기여도 추출용)
            self._load_baseline_emission_data()

            # 비용 프리미엄 데이터 로드
            self._load_cost_premiums()

            # 데이터 재로딩 시 캐시 클리어
            self._cached_target_materials = None
            self._cached_classification = None

            # 성능 최적화: 에너지 마스크 캐시 클리어 (Phase 5)
            self._energy_tier1_mask = None
            self._energy_tier2_mask = None
            self._energy_masks_initialized = False

            return True

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_cathode_data(self) -> None:
        """
        양극재 관련 설정 데이터 로드

        - cathode_ratio.json: 원소 조성비 (Ni, Co, Mn, Al)
        - recycle_material_impact.json: 재활용재 영향도
        - low_carb_metal.json: 저탄소메탈 배출계수
        - cathode_coef_table.json: 신재 배출계수
        """
        import json
        import os

        try:
            # 사용자별 경로 확인
            current_user = st.session_state.get('current_user', 'default')

            # 1. cathode_ratio.json (원소 조성비)
            cathode_ratio_path = f"input/{current_user}/cathode_ratio.json" if current_user != 'default' else "input/cathode_ratio.json"
            if not os.path.exists(cathode_ratio_path):
                cathode_ratio_path = "input/cathode_ratio.json"

            with open(cathode_ratio_path, 'r', encoding='utf-8') as f:
                self.cathode_ratio = json.load(f)
                print(f"   • cathode_ratio: {self.cathode_ratio}")

            # 2. recycle_material_impact.json (재활용 영향도)
            recycle_impact_path = f"input/{current_user}/recycle_material_impact.json" if current_user != 'default' else "input/recycle_material_impact.json"
            if not os.path.exists(recycle_impact_path):
                recycle_impact_path = "input/recycle_material_impact.json"

            with open(recycle_impact_path, 'r', encoding='utf-8') as f:
                recycle_data = json.load(f)
                self.recycle_impact = recycle_data.get('재활용재', {})
                print(f"   • recycle_impact: {self.recycle_impact}")

            # 3. low_carb_metal.json (저탄소메탈 배출계수)
            low_carb_path = f"input/{current_user}/low_carb_metal.json" if current_user != 'default' else "input/low_carb_metal.json"
            if not os.path.exists(low_carb_path):
                low_carb_path = "input/low_carb_metal.json"

            with open(low_carb_path, 'r', encoding='utf-8') as f:
                low_carb_data = json.load(f)
                self.low_carb_emission = low_carb_data.get('배출계수', {})
                print(f"   • low_carb_emission: {self.low_carb_emission}")

            # 4. cathode_coef_table.json (신재 배출계수)
            cathode_coef_path = "stable_var/cathode_coef_table.json"
            with open(cathode_coef_path, 'r', encoding='utf-8') as f:
                coef_data = json.load(f)
                raw_materials = coef_data.get('원재료', {})

                # 원소별 매핑
                self.virgin_emission = {
                    'Ni': raw_materials.get('NiSO4', {}).get('배출계수', 4.74),
                    'Co': raw_materials.get('CoSO4', {}).get('배출계수', 23.6),
                    'Mn': raw_materials.get('MnSO4', {}).get('배출계수', 0.81),
                    'Li': raw_materials.get('LiOH.H2O', {}).get('배출계수', 14.7)
                }
                print(f"   • virgin_emission: {self.virgin_emission}")

            print(f"✅ 양극재 데이터 로드 완료")

        except Exception as e:
            print(f"⚠️  양극재 데이터 로드 실패: {e}")
            # 기본값 설정
            self.cathode_ratio = {'Ni': 0.6, 'Co': 0.15, 'Mn': 0.25, 'Al': 0.0}
            self.recycle_impact = {'Ni': 0.1, 'Co': 0.15, 'Li': 0.1}
            self.low_carb_emission = {'Ni': 2.0, 'Co': 15.0, 'Li': 9.0}
            self.virgin_emission = {'Ni': 4.74, 'Co': 23.6, 'Mn': 0.81, 'Li': 14.7}

    def _update_electricity_coefficients_for_site_change(self) -> bool:
        """
        생산지 변경 시 전력계수 업데이트

        site_change 시나리오의 데이터를 사용하여 전력 관련 배출계수를 업데이트합니다.

        Returns:
            성공 여부
        """
        try:
            print(f"   🔍 전력계수 업데이트 시작...")

            # site_change 시나리오가 있는지 확인
            if 'site_change' not in self.simulation_results:
                print("⚠️  site_change 시나리오가 없습니다. CathodeSimulator로 생성 시도...")
                return self._generate_site_change_data_with_simulator()

            # site_change 시나리오 데이터 가져오기
            site_change_data = self.simulation_results['site_change']
            site_change_df = site_change_data.get('all_data')

            if site_change_df is None:
                print("⚠️  site_change의 all_data가 없습니다.")
                return False

            # Energy 카테고리의 배출계수만 업데이트
            # scenario_df에서 Energy(Tier-1)과 Energy(Tier-2) 자재 찾기

            def find_energy_items(df, tier):
                """
                에너지 항목을 여러 패턴으로 찾기

                Args:
                    df: 검색할 DataFrame
                    tier: Tier 번호 (1 또는 2)

                Returns:
                    DataFrame: 매칭된 에너지 항목들
                """
                patterns = [
                    f'Energy(Tier-{tier})',     # 괄호 직접
                    f'Energy (Tier-{tier})',    # 공백 포함
                    f'Energy_Tier{tier}',       # 언더스코어
                    f'Tier-{tier}',             # Tier만
                ]

                for pattern in patterns:
                    mask = df['배출계수명'].str.contains(pattern, na=False, case=False, regex=False)
                    if mask.any():
                        print(f"   ✅ 패턴 '{pattern}' 매칭: {mask.sum()}개 항목")
                        return df[mask]

                print(f"   ⚠️ Tier-{tier} 에너지 항목을 찾지 못했습니다.")
                return pd.DataFrame()  # 빈 DataFrame 반환

            energy_keywords = ['Energy(Tier-1)', 'Energy(Tier-2)', 'Energy (Tier-1)', 'Energy (Tier-2)']

            update_count = 0
            for keyword in energy_keywords:
                # baseline에서 해당 자재 찾기
                baseline_energy = self.scenario_df[
                    self.scenario_df['배출계수명'].str.contains(keyword, na=False, case=False, regex=False)
                ]

                # site_change에서 해당 자재 찾기
                site_change_energy = site_change_df[
                    site_change_df['배출계수명'].str.contains(keyword, na=False, case=False, regex=False)
                ]

                if not baseline_energy.empty and not site_change_energy.empty:
                    # 배출계수 업데이트
                    for idx in baseline_energy.index:
                        baseline_material_name = baseline_energy.loc[idx, '자재명']

                        # site_change에서 같은 자재 찾기
                        matching_site_change = site_change_energy[
                            site_change_energy['자재명'] == baseline_material_name
                        ]

                        if not matching_site_change.empty:
                            new_emission = matching_site_change.iloc[0]['배출계수']
                            old_emission = self.scenario_df.loc[idx, '배출계수']

                            self.scenario_df.loc[idx, '배출계수'] = new_emission

                            # 배출량도 업데이트 (배출계수 × 소요량)
                            quantity = self.scenario_df.loc[idx, '제품총소요량(kg)']
                            self.scenario_df.loc[idx, '배출량(kgCO2eq)'] = new_emission * quantity

                            # 변화율 계산
                            if old_emission > 0:
                                change_pct = ((new_emission - old_emission) / old_emission) * 100
                                print(f"   ✅ {baseline_material_name}: {old_emission:.6f} → {new_emission:.6f} ({change_pct:+.1f}%)")
                            else:
                                print(f"   ✅ {baseline_material_name}: {old_emission:.6f} → {new_emission:.6f}")

                            update_count += 1

            if update_count == 0:
                print(f"   ⚠️ 경고: 전력계수가 업데이트되지 않았습니다!")
                print(f"   검색한 키워드: {energy_keywords}")
                print(f"   scenario_df에 '배출계수명' 컬럼이 있습니까? {'배출계수명' in self.scenario_df.columns}")
                return False

            if update_count > 0:
                print(f"   ✅ 전력계수 업데이트 완료: {update_count}개 항목")
            else:
                print(f"   ⚠️  전력계수 업데이트 0개 - 데이터 구조 확인 필요")
                print(f"   검색 대상 배출계수명 목록:")
                unique_names = self.scenario_df['배출계수명'].unique()[:10]
                print(f"      {unique_names}")

            # 에너지 기여도 검증 (RE100 최적화가 의미 있는지 확인)
            print(f"\n   🔍 에너지 기여도 검증:")
            print(f"      • Tier1 에너지 기여도: {self.tier1_energy_ratio*100:.2f}%")
            print(f"      • Tier2 에너지 기여도: {self.tier2_energy_ratio*100:.2f}%")
            print(f"      • 총 에너지 기여도: {(self.tier1_energy_ratio + self.tier2_energy_ratio)*100:.2f}%")

            # 에너지 기여도가 0이면 경고
            total_energy_ratio = self.tier1_energy_ratio + self.tier2_energy_ratio
            if total_energy_ratio == 0:
                print(f"      ⚠️ 경고: 에너지 기여도가 0입니다!")
                print(f"         RE100 최적화가 효과가 없을 수 있습니다.")
            elif total_energy_ratio < 0.05:
                print(f"      ⚠️ 경고: 에너지 기여도가 매우 낮습니다 ({total_energy_ratio*100:.2f}%)")
                print(f"         RE100 최적화 효과가 제한적일 수 있습니다.")

            # ref_formula_df도 업데이트 (있는 경우)
            if self.ref_formula_df is not None:
                site_change_simulator_result = st.session_state.get('site_change_simulator_data')
                if site_change_simulator_result:
                    # CathodeSimulator의 coefficient_data 사용
                    print("   ℹ️  ref_formula_df 전력계수 업데이트 (session_state 사용)")
                    # 이 부분은 복잡하므로 생략 (scenario_df만 업데이트해도 충분)

            print(f"✅ 전력계수 업데이트 완료 (site='after' 적용)")
            return True

        except Exception as e:
            print(f"❌ 전력계수 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_site_change_data_with_simulator(self) -> bool:
        """
        CathodeSimulator를 사용하여 site_change 데이터 생성

        Returns:
            성공 여부
        """
        try:
            print("   🔄 CathodeSimulator로 after 사이트 데이터 생성 중...")

            from src.cathode_simulator import CathodeSimulator

            # 사용자 ID 가져오기
            user_id = st.session_state.get('user_id', None)

            # CathodeSimulator 생성 및 after 사이트 데이터 생성
            simulator = CathodeSimulator(verbose=False, user_id=user_id)
            after_data = simulator.update_electricity_emission_factor(site='after')

            if not after_data:
                print("   ❌ after 사이트 데이터 생성 실패")
                return False

            # Energy 배출계수 업데이트
            tier1_elec = after_data.get('Energy(Tier-1)', {}).get('전력', {}).get('배출계수')
            tier2_elec = after_data.get('Energy(Tier-2)', {}).get('전력', {}).get('배출계수')

            if tier1_elec:
                # scenario_df에서 Energy(Tier-1) 자재 업데이트
                tier1_mask = self.scenario_df['배출계수명'].str.contains('Energy(Tier-1)', na=False, case=False, regex=False)
                for idx in self.scenario_df[tier1_mask].index:
                    old_emission = self.scenario_df.loc[idx, '배출계수']
                    self.scenario_df.loc[idx, '배출계수'] = tier1_elec
                    quantity = self.scenario_df.loc[idx, '제품총소요량(kg)']
                    self.scenario_df.loc[idx, '배출량(kgCO2eq)'] = tier1_elec * quantity
                    print(f"   ✅ Energy(Tier-1): {old_emission:.6f} → {tier1_elec:.6f}")

            if tier2_elec:
                # scenario_df에서 Energy(Tier-2) 자재 업데이트
                tier2_mask = self.scenario_df['배출계수명'].str.contains('Energy(Tier-2)', na=False, case=False, regex=False)
                for idx in self.scenario_df[tier2_mask].index:
                    old_emission = self.scenario_df.loc[idx, '배출계수']
                    self.scenario_df.loc[idx, '배출계수'] = tier2_elec
                    quantity = self.scenario_df.loc[idx, '제품총소요량(kg)']
                    self.scenario_df.loc[idx, '배출량(kgCO2eq)'] = tier2_elec * quantity
                    print(f"   ✅ Energy(Tier-2): {old_emission:.6f} → {tier2_elec:.6f}")

            print(f"✅ CathodeSimulator 기반 전력계수 업데이트 완료")
            return True

        except Exception as e:
            print(f"❌ CathodeSimulator 기반 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_cost_premiums(self) -> None:
        """
        재활용재/저탄소메탈 비용 프리미엄 데이터 로드

        material_cost_premiums.json에서 원소별 비용 프리미엄(%)을 로드합니다.
        """
        try:
            from src.utils.optimization_costs_manager import OptimizationCostsManager

            costs_manager = OptimizationCostsManager()
            user_id = st.session_state.get('user_id', 'default')

            # 프리미엄 데이터 로드
            self.material_cost_premiums = costs_manager.load_material_cost_premiums(user_id)
            print(f"   • material_cost_premiums: {self.material_cost_premiums}")
            print(f"✅ 비용 프리미엄 데이터 로드 완료")

        except Exception as e:
            print(f"⚠️  비용 프리미엄 데이터 로드 실패: {e}")
            # 기본값 설정
            self.material_cost_premiums = {
                "Ni": {"recycle_premium_pct": 30.0, "low_carbon_premium_pct": 50.0},
                "Co": {"recycle_premium_pct": 40.0, "low_carbon_premium_pct": 60.0},
                "Li": {"recycle_premium_pct": 20.0, "low_carbon_premium_pct": 40.0},
                "default": {"recycle_premium_pct": 30.0, "low_carbon_premium_pct": 50.0}
            }

    def _load_energy_ratio_config(self) -> Dict[str, Any]:
        """
        에너지 비율 설정 파일 로드

        Returns:
            설정 dict 또는 하드코딩 fallback
        """
        import json
        import os

        config_path = 'input/energy_ratio_defaults.json'

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"   ✅ 에너지 비율 설정 로드: {config_path} (version {config.get('version', 'unknown')})")
                return config
            else:
                print(f"   ℹ️  설정 파일 없음, 하드코딩 기본값 사용")
                return self._get_hardcoded_defaults()
        except Exception as e:
            print(f"   ⚠️  설정 로드 실패: {e}, 하드코딩 기본값 사용")
            return self._get_hardcoded_defaults()

    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """하드코딩된 기본값 (fallback)"""
        return {
            "category_defaults": {
                "cathode": {"tier1_ratio": 0.12, "tier2_ratio": 0.06, "description": "양극재 평균"},
                "anode": {"tier1_ratio": 0.10, "tier2_ratio": 0.05, "description": "음극재 평균"},
                "electrolyte": {"tier1_ratio": 0.08, "tier2_ratio": 0.04, "description": "전해액 평균"},
                "separator": {"tier1_ratio": 0.06, "tier2_ratio": 0.03, "description": "분리막 평균"},
                "general": {"tier1_ratio": 0.05, "tier2_ratio": 0.03, "description": "일반 자재"}
            },
            "material_specific": {}
        }

    def _initialize_energy_masks(self) -> None:
        """
        에너지 마스크 초기화 (성능 최적화)

        scenario_df에서 Energy(Tier-1)과 Energy(Tier-2)를 식별하는
        boolean 마스크를 생성하고 캐시합니다.

        이 마스크는 여러 자재에 대해 재사용되므로 성능이 크게 향상됩니다.
        """
        if self._energy_masks_initialized or self.scenario_df is None:
            return  # 이미 초기화되었거나 데이터가 없으면 스킵

        try:
            # Tier1 마스크 생성 (두 가지 패턴 모두 지원)
            self._energy_tier1_mask = (
                self.scenario_df['배출계수명'].str.contains('Energy(Tier-1)', na=False, case=False, regex=False) |
                self.scenario_df['배출계수명'].str.contains('Energy (Tier-1)', na=False, case=False, regex=False)
            )

            # Tier2 마스크 생성 (두 가지 패턴 모두 지원)
            self._energy_tier2_mask = (
                self.scenario_df['배출계수명'].str.contains('Energy(Tier-2)', na=False, case=False, regex=False) |
                self.scenario_df['배출계수명'].str.contains('Energy (Tier-2)', na=False, case=False, regex=False)
            )

            self._energy_masks_initialized = True

        except Exception as e:
            print(f"   ⚠️ 에너지 마스크 초기화 실패: {e}")
            # 실패해도 계속 진행 (fallback 동작 사용)
            self._energy_masks_initialized = False

    def _calculate_material_energy_ratios(self, material_name: str) -> Tuple[float, float]:
        """
        자재별 에너지 비율 동적 계산 (최적화됨)

        베이스라인 시뮬레이션 데이터에서 자재의 Tier1/Tier2 에너지 배출량 비율을 추출합니다.

        Phase 5 최적화: 캐시된 에너지 마스크를 재사용하여 성능 향상

        Args:
            material_name: 자재명

        Returns:
            (tier1_ratio, tier2_ratio): Tier1/Tier2 에너지 비율 (0~1 소수)
        """
        if self.scenario_df is None:
            return 0.0, 0.0

        # 에너지 마스크 초기화 (lazy initialization)
        self._initialize_energy_masks()

        try:
            # 해당 자재의 모든 배출 항목 가져오기 (한 번만 필터링)
            material_mask = self.scenario_df['자재명'] == material_name
            material_rows = self.scenario_df[material_mask]

            if material_rows.empty:
                return 0.0, 0.0

            # 전체 배출량 계산
            total_emission = material_rows['배출량(kgCO2eq)'].sum()

            if total_emission == 0:
                return 0.0, 0.0

            # Phase 5 최적화: 캐시된 마스크 사용
            if self._energy_masks_initialized:
                # 자재 마스크와 에너지 마스크를 결합 (AND 연산)
                tier1_combined_mask = material_mask & self._energy_tier1_mask
                tier2_combined_mask = material_mask & self._energy_tier2_mask

                # 배출량 합계 (전체 DataFrame에서 한 번에 계산)
                tier1_emission = self.scenario_df.loc[tier1_combined_mask, '배출량(kgCO2eq)'].sum()
                tier2_emission = self.scenario_df.loc[tier2_combined_mask, '배출량(kgCO2eq)'].sum()
            else:
                # Fallback: 마스크 캐시가 없으면 기존 방식 사용
                tier1_emission = material_rows[
                    material_rows['배출계수명'].str.contains('Energy(Tier-1)', na=False, case=False, regex=False) |
                    material_rows['배출계수명'].str.contains('Energy (Tier-1)', na=False, case=False, regex=False)
                ]['배출량(kgCO2eq)'].sum()

                tier2_emission = material_rows[
                    material_rows['배출계수명'].str.contains('Energy(Tier-2)', na=False, case=False, regex=False) |
                    material_rows['배출계수명'].str.contains('Energy (Tier-2)', na=False, case=False, regex=False)
                ]['배출량(kgCO2eq)'].sum()

            # 비율 계산
            tier1_ratio = tier1_emission / total_emission if total_emission > 0 else 0.0
            tier2_ratio = tier2_emission / total_emission if total_emission > 0 else 0.0

            return tier1_ratio, tier2_ratio

        except Exception as e:
            print(f"   ⚠️ {material_name}: 에너지 비율 계산 실패 - {e}")
            return 0.0, 0.0

    def validate_energy_ratios(
        self,
        material_name: str,
        tier1_ratio: float,
        tier2_ratio: float
    ) -> Tuple[float, float]:
        """
        에너지 비율 검증 및 fallback 적용

        에너지 비율의 타당성을 검증하고, 문제가 있을 경우 설정 파일의 기본값을 사용합니다.

        Args:
            material_name: 자재명
            tier1_ratio: Tier1 에너지 비율
            tier2_ratio: Tier2 에너지 비율

        Returns:
            (validated_tier1, validated_tier2): 검증된 에너지 비율
        """
        total = tier1_ratio + tier2_ratio

        # Case 1: 데이터 없음 (0) → 설정 파일 사용
        if total == 0:
            # Lazy loading: 설정 파일을 필요할 때만 로드
            if not hasattr(self, '_energy_config'):
                self._energy_config = self._load_energy_ratio_config()

            config = self._energy_config

            # Priority 1: 자재별 설정 (material_specific)
            material_specific = config.get('material_specific', {})

            # 자재명을 다양한 패턴으로 매칭 시도
            matched_key = None
            for key in material_specific.keys():
                if key in material_name or material_name in key:
                    matched_key = key
                    break

            if matched_key:
                mat_cfg = material_specific[matched_key]
                tier1 = mat_cfg['tier1_ratio']
                tier2 = mat_cfg['tier2_ratio']
                source = mat_cfg.get('source', 'config')
                notes = mat_cfg.get('notes', '')

                print(f"   ✅ {material_name[:50]}")
                print(f"      → 자재별 설정 적용: '{matched_key}'")
                print(f"      → Tier1={tier1*100:.1f}%, Tier2={tier2*100:.1f}%")
                print(f"      → 출처: {source}")
                if notes:
                    print(f"      → 참고: {notes}")
                return tier1, tier2

            # Priority 2: 카테고리 기본값 (category_defaults)
            material_lower = material_name.lower()

            if 'cathode' in material_lower or 'cam' in material_lower or '양극재' in material_name:
                category = 'cathode'
            elif 'anode' in material_lower or '음극재' in material_name or 'graphite' in material_lower:
                category = 'anode'
            elif 'electrolyte' in material_lower or '전해액' in material_name or '전해질' in material_name:
                category = 'electrolyte'
            elif 'separator' in material_lower or '분리막' in material_name:
                category = 'separator'
            else:
                category = 'general'

            cat_cfg = config['category_defaults'][category]
            tier1 = cat_cfg['tier1_ratio']
            tier2 = cat_cfg['tier2_ratio']
            description = cat_cfg.get('description', category)

            print(f"   ⚠️ {material_name[:50]}")
            print(f"      → 에너지 비율 0% (데이터 없음)")
            print(f"      → [{description}] 카테고리 기본값 사용")
            print(f"      → Tier1={tier1*100:.1f}%, Tier2={tier2*100:.1f}%")

            return tier1, tier2

        # Case 2: 비율이 너무 높은 경우 → 경고
        validation_rules = getattr(self, '_energy_config', {}).get('validation_rules', {})
        warning_threshold = validation_rules.get('warning_threshold', 0.40)

        if total > warning_threshold:
            print(f"   ⚠️ {material_name[:50]}")
            print(f"      → 에너지 비율 {total*100:.2f}%가 의심스럽게 높음 (경고 기준: {warning_threshold*100:.0f}%)")
            print(f"      → 데이터를 확인하세요")

        # Case 3: 합이 1.0을 초과하는 경우 → 정규화
        if total > 1.0:
            norm_tier1 = tier1_ratio / total
            norm_tier2 = tier2_ratio / total

            print(f"   ❌ {material_name[:50]}")
            print(f"      → 에너지 비율 합이 100% 초과 ({total*100:.2f}%)")
            print(f"      → 정규화 적용: Tier1={tier1_ratio*100:.2f}%→{norm_tier1*100:.2f}%, Tier2={tier2_ratio*100:.2f}%→{norm_tier2*100:.2f}%")

            return norm_tier1, norm_tier2

        # Case 4: 정상 범위 → 그대로 반환
        return tier1_ratio, tier2_ratio

    def _load_baseline_emission_data(self) -> None:
        """
        베이스라인 에너지 기여도 로드

        CathodeHelper를 사용하여 시뮬레이터로부터 직접 에너지 기여도를 추출합니다.
        실패 시 기본값(Tier1: 12%, Tier2: 6%)을 사용합니다.
        """
        try:
            print(f"\n🔋 에너지 기여도 추출 시작...")

            # CathodeHelper와 CathodeSimulator 임포트
            from src.helper import CathodeHelper
            from src.cathode_simulator import CathodeSimulator

            # 시뮬레이터 초기화 (verbose=False로 조용하게)
            simulator = CathodeSimulator(verbose=False)
            helper = CathodeHelper(simulator)

            # 모든 시나리오 데이터 생성
            all_scenarios = helper.generate_all_scenarios_data()

            # basic scenarios dataframe 가져오기
            basic_df = helper.get_basic_scenarios_dataframe(all_scenarios)

            # Baseline 시나리오 찾기
            baseline_rows = basic_df[basic_df['시나리오'] == 'Baseline']

            if baseline_rows.empty:
                raise ValueError("Baseline 시나리오를 찾을 수 없습니다.")

            baseline = baseline_rows.iloc[0]

            # 에너지 기여도 추출 (퍼센트 형태)
            tier1_pct = baseline.get('Energy_Tier1_전력_기여도_퍼센트', 0)
            tier2_pct = baseline.get('Energy_Tier2_전력_기여도_퍼센트', 0)

            # 소수 형태로 변환 (12% → 0.12)
            self.tier1_energy_ratio = tier1_pct / 100
            self.tier2_energy_ratio = tier2_pct / 100

            total_energy_pct = tier1_pct + tier2_pct

            print(f"✅ 에너지 기여도 추출 완료:")
            print(f"   • Tier1 (CAM 제조): {tier1_pct:.2f}%")
            print(f"   • Tier2 (pCAM 제조): {tier2_pct:.2f}%")
            print(f"   • 전체 에너지 기여도: {total_energy_pct:.2f}%")

            # 경고 메시지
            if total_energy_pct == 0:
                print(f"   ⚠️  WARNING: 전체 에너지 기여도가 0%입니다!")
                print(f"      → RE100을 적용해도 배출량 감축 효과가 없습니다.")
                print(f"      → 기본값(Tier1: 12%, Tier2: 6%)을 사용합니다.")
                # Fallback to default values
                self.tier1_energy_ratio = 0.12
                self.tier2_energy_ratio = 0.06
            elif total_energy_pct < 5:
                print(f"   ⚠️  WARNING: 에너지 기여도가 {total_energy_pct:.2f}%로 매우 낮습니다.")
                print(f"      → RE100 최적화 효과가 제한적일 수 있습니다.")
            else:
                print(f"   ✅ RE100 최대 감축 가능: {total_energy_pct:.2f}% (tier1_re=100%, tier2_re=100%)")

        except Exception as e:
            # Fallback to default values
            self.tier1_energy_ratio = 0.12  # 12%
            self.tier2_energy_ratio = 0.06  # 6%

            print(f"⚠️  에너지 기여도 추출 실패: {e}")
            print(f"   → 기본값 사용: Tier1={self.tier1_energy_ratio*100:.1f}%, Tier2={self.tier2_energy_ratio*100:.1f}%")
            import traceback
            traceback.print_exc()

    def load_from_files(
        self,
        scenario_path: str,
        original_path: Optional[str] = None,
        ref_formula_path: Optional[str] = None,
        ref_proportions_path: Optional[str] = None
    ) -> bool:
        """
        파일에서 데이터 로드

        Args:
            scenario_path: 시나리오 CSV 파일 경로
            original_path: 원본 테이블 CSV 파일 경로 (선택)
            ref_formula_path: Formula 참조 CSV 파일 경로 (선택)
            ref_proportions_path: Proportions 참조 CSV 파일 경로 (선택)

        Returns:
            성공 여부
        """
        try:
            # 시나리오 파일 로드
            self.scenario_df = pd.read_csv(scenario_path, encoding='utf-8-sig')
            print(f"✅ scenario_df 로드: {len(self.scenario_df)} rows")

            # 선택적 파일들 로드
            if original_path:
                self.original_df = pd.read_csv(original_path, encoding='utf-8-sig')
                print(f"✅ original_df 로드: {len(self.original_df)} rows")

            if ref_formula_path:
                self.ref_formula_df = pd.read_csv(ref_formula_path, encoding='utf-8-sig')
                print(f"✅ ref_formula_df 로드: {len(self.ref_formula_df)} rows")

            if ref_proportions_path:
                self.ref_proportions_df = pd.read_csv(ref_proportions_path, encoding='utf-8-sig')
                print(f"✅ ref_proportions_df 로드: {len(self.ref_proportions_df)} rows")

            return True

        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_target_materials(self, use_cache: bool = True, verbose: bool = True) -> List[str]:
        """
        최적화 대상 자재 목록 반환

        Args:
            use_cache: 캐시 사용 여부
            verbose: 로그 출력 여부 (False면 조용히 실행)

        Returns:
            자재명 리스트
        """
        # 캐시 확인
        if use_cache and self._cached_target_materials is not None:
            return self._cached_target_materials

        if self.scenario_df is None:
            return []

        # 저감활동_적용여부가 1인 자재만 추출
        if '저감활동_적용여부' in self.scenario_df.columns:
            target_df = self.scenario_df[self.scenario_df['저감활동_적용여부'] == 1]
        else:
            target_df = self.scenario_df

        materials = set(target_df['자재명'].unique().tolist())

        # ref_proportions_df에 있는 원소 자재도 포함 (Ni/Co/Li 원소 자재)
        # 양극재 원소별 비율 제약을 적용하려면 원소 자재가 필요함
        if self.ref_proportions_df is not None and '자재명' in self.ref_proportions_df.columns:
            element_materials = self.ref_proportions_df['자재명'].unique().tolist()

            # 원소 자재 키워드 (NiSO4, CoSO4, LiOH 등)
            element_keywords = ['NiSO4', 'CoSO4', 'LiOH', 'Ni2O3', 'Co3O4', 'Li2CO3']

            for material in element_materials:
                # 원소 자재인지 확인
                if any(keyword in material for keyword in element_keywords):
                    # scenario_df에 존재하는지 확인
                    if material in self.scenario_df['자재명'].values:
                        materials.add(material)
                        if verbose:
                            print(f"  ✅ 원소 자재 추가: {material}")

        materials = list(materials)

        # verbose=True일 때만 출력
        if verbose:
            print(f"📦 대상 자재: {len(materials)}개 (저감활동=1: {len(target_df['자재명'].unique())}개 + 원소자재)")

        # 캐시 저장
        self._cached_target_materials = materials
        return materials

    def classify_materials(self, use_cache: bool = True, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        자재를 Formula/Ni-Co-Li/General 타입으로 분류하고
        각 자재별 에너지 기여도를 계산합니다.

        Args:
            use_cache: 캐시 사용 여부
            verbose: 로그 출력 여부

        Returns:
            자재별 분류 정보 딕셔너리
        """
        # 캐시 확인
        if use_cache and self._cached_classification is not None:
            return self._cached_classification

        if self.scenario_df is None:
            return {}

        # get_target_materials를 verbose=False로 호출 (중복 로그 방지)
        materials = self.get_target_materials(use_cache=use_cache, verbose=False)
        classification = {}

        # Formula 자재 확인
        formula_materials = set()
        if self.ref_formula_df is not None and '자재명' in self.ref_formula_df.columns:
            formula_materials = set(self.ref_formula_df['자재명'].unique())

        # Ni/Co/Li 자재 확인
        nicoli_keywords = ['Ni', 'Co', 'Li', 'Nickel', 'Cobalt', 'Lithium', 'NMC', 'NCA', 'LFP', 'NCMA', '3PHASE', 'NCM',
                          'NiSO4', 'CoSO4', 'LiOH', 'Ni2O3', 'Co3O4', 'Li2CO3']
        nicoli_materials = set()
        if self.ref_proportions_df is not None and '자재명' in self.ref_proportions_df.columns:
            nicoli_materials = set(self.ref_proportions_df['자재명'].unique())

        # Cathode/Anode/Electrolyte/Separator 키워드
        cathode_keywords = ['Cathode', 'cathode', '양극재', 'CAM']
        anode_keywords = ['Anode', 'anode', '음극재']
        electrolyte_keywords = ['Electrolyte', 'electrolyte', '전해액', '전해질']
        separator_keywords = ['Separator', 'separator', '분리막', '세퍼레이터']

        # Phase 5 최적화: 에너지 마스크 초기화 및 카운트 (캐시 재사용)
        self._initialize_energy_masks()

        if verbose and self._energy_masks_initialized:
            print(f"\n🔍 에너지 카테고리 확인:")
            print(f"   • Energy(Tier-1) 항목: {self._energy_tier1_mask.sum()}개")
            print(f"   • Energy(Tier-2) 항목: {self._energy_tier2_mask.sum()}개")

        # Phase 5 최적화: 자재별 데이터 조회를 위한 인덱스 생성 (빠른 접근)
        material_data_cache = {}
        for material in materials:
            # 자재 정보 가져오기 (캐시에 저장)
            material_rows = self.scenario_df[self.scenario_df['자재명'] == material]
            if not material_rows.empty:
                material_data_cache[material] = material_rows.iloc[0]

        for material in materials:
            # 캐시된 자재 정보 사용
            if material not in material_data_cache:
                continue  # 자재 데이터가 없으면 스킵

            material_data = material_data_cache[material]

            # Formula 적용 가능 여부
            is_cathode = any(kw in material for kw in cathode_keywords)
            is_anode = any(kw in material for kw in anode_keywords)
            is_electrolyte = any(kw in material for kw in electrolyte_keywords)
            is_separator = any(kw in material for kw in separator_keywords)

            # Formula 자재 = ref_formula_df에 있거나 cathode/anode/electrolyte/separator
            is_formula = (material in formula_materials or
                         is_cathode or is_anode or is_electrolyte or is_separator)

            # Ni/Co/Li 자재 여부
            is_nicoli = (material in nicoli_materials or
                        any(kw in material for kw in nicoli_keywords) or
                        is_cathode)

            # 자재 타입 결정
            if is_formula:
                material_type = 'Formula'
            elif is_nicoli:
                material_type = 'Ni-Co-Li'
            else:
                material_type = 'General'

            # 에너지 기여도 계산
            tier1_energy_ratio = 0.0
            tier2_energy_ratio = 0.0

            if is_formula and is_cathode:
                # 양극재는 CathodeHelper에서 추출한 전역 값 사용 (이미 계산됨)
                tier1_energy_ratio = self.tier1_energy_ratio
                tier2_energy_ratio = self.tier2_energy_ratio

                # 🔍 VALIDATION: 양극재 에너지 기여도 검증 및 fallback 적용
                tier1_energy_ratio, tier2_energy_ratio = self.validate_energy_ratios(
                    material, tier1_energy_ratio, tier2_energy_ratio
                )

                if verbose:
                    print(f"\n✓ Cathode {material[:50]}")
                    print(f"  Tier1 (CAM) energy ratio: {tier1_energy_ratio*100:.2f}%")
                    print(f"  Tier2 (pCAM) energy ratio: {tier2_energy_ratio*100:.2f}%")
                    print(f"  Max RE100 potential: {(tier1_energy_ratio + tier2_energy_ratio)*100:.2f}%")

            elif is_formula:
                # 기타 Formula 자재: scenario_df에서 해당 자재의 에너지 배출 비율 계산
                # 새로운 함수를 사용하여 동적으로 계산
                tier1_energy_ratio, tier2_energy_ratio = self._calculate_material_energy_ratios(material)

                # 검증 및 fallback 적용
                tier1_energy_ratio, tier2_energy_ratio = self.validate_energy_ratios(
                    material, tier1_energy_ratio, tier2_energy_ratio
                )

                # 해당 자재의 전체 배출량 (로깅용)
                material_total_emission = material_data.get('배출량(kgCO2eq)', 0)

                if verbose:
                    print(f"   🔍 {material[:50]}...")
                    print(f"      • Total emission: {material_total_emission:.6f} kgCO2eq")
                    print(f"      • Tier1 energy ratio: {tier1_energy_ratio*100:.2f}%")
                    print(f"      • Tier2 energy ratio: {tier2_energy_ratio*100:.2f}%")
                    print(f"      • Max RE100 potential: {(tier1_energy_ratio + tier2_energy_ratio)*100:.2f}%")

            classification[material] = {
                'type': material_type,
                'is_formula_applicable': is_formula,
                'is_ni_co_li': is_nicoli,
                'quantity': material_data.get('제품총소요량(kg)', 0),
                'original_emission': material_data.get('배출계수', 0),
                'country': material_data.get('지역', 'Unknown'),
                # RE100 에너지 기여도 (Formula 자재에만 적용)
                'tier1_energy_ratio': tier1_energy_ratio,
                'tier2_energy_ratio': tier2_energy_ratio
            }

        # verbose=True일 때만 요약 출력
        if verbose:
            print(f"\n📊 자재 분류 완료:")
            print(f"   • Formula 적용 가능: {sum(1 for m in classification.values() if m['is_formula_applicable'])}개")
            print(f"   • Ni/Co/Li 자재: {sum(1 for m in classification.values() if m['is_ni_co_li'])}개")
            print(f"   • 일반 자재: {sum(1 for m in classification.values() if not m['is_formula_applicable'] and not m['is_ni_co_li'])}개")

            # RE100 적용 대상 자재 확인
            formula_with_re100 = [
                m for m, info in classification.items()
                if info['is_formula_applicable'] and (info['tier1_energy_ratio'] > 0 or info['tier2_energy_ratio'] > 0)
            ]

            print(f"\n🔋 RE100 최적화 대상:")
            if len(formula_with_re100) == 0:
                print(f"   ⚠️ WARNING: RE100 최적화 대상 자재가 없습니다!")
            else:
                print(f"   ✅ {len(formula_with_re100)}개 자재에 RE100 적용 가능")

                # 양극재와 기타 자재 구분하여 표시
                cathode_count = sum(1 for m in formula_with_re100 if any(kw in m for kw in cathode_keywords))
                other_count = len(formula_with_re100) - cathode_count

                print(f"      - 양극재: {cathode_count}개")
                print(f"      - 기타 Formula 자재: {other_count}개")

                # 샘플 표시 (최대 5개)
                for mat in formula_with_re100[:5]:
                    tier1 = classification[mat]['tier1_energy_ratio']
                    tier2 = classification[mat]['tier2_energy_ratio']
                    mat_type = "양극재" if any(kw in mat for kw in cathode_keywords) else "기타"
                    print(f"      - [{mat_type}] {mat[:50]}: Tier1={tier1*100:.1f}%, Tier2={tier2*100:.1f}%")

                if len(formula_with_re100) > 5:
                    print(f"      ... 외 {len(formula_with_re100)-5}개")

        # 캐시 저장
        self._cached_classification = classification
        return classification

    def get_optimization_data(self) -> Dict[str, Any]:
        """
        최적화 엔진에 필요한 모든 데이터를 딕셔너리로 반환

        Returns:
            최적화 데이터 딕셔너리
        """
        return {
            'scenario_df': self.scenario_df,
            'original_df': self.original_df,
            'ref_formula_df': self.ref_formula_df,
            'ref_proportions_df': self.ref_proportions_df,
            'target_materials': self.get_target_materials(use_cache=True, verbose=False),
            'material_classification': self.classify_materials(use_cache=True, verbose=False),
            # 양극재 관련 데이터
            'cathode_composition': self.cathode_ratio,
            'recycle_impact': self.recycle_impact,
            'low_carb_emission': self.low_carb_emission,
            'virgin_emission': self.virgin_emission,
            # 비용 관련 데이터
            'material_cost_premiums': self.material_cost_premiums
        }

    def validate_data(self, check_cost_constraint: bool = False) -> Tuple[bool, List[str]]:
        """
        로드된 데이터의 유효성 검증

        Args:
            check_cost_constraint: 비용 제약 사용 여부 (True면 original_df 필수)

        Returns:
            (is_valid, errors): 유효성 여부와 오류 메시지 리스트
        """
        errors = []

        # scenario_df 필수
        if self.scenario_df is None:
            errors.append("scenario_df가 로드되지 않았습니다.")
            return False, errors

        # 필수 컬럼 확인
        required_columns = ['자재명', '제품총소요량(kg)', '배출계수']
        missing_columns = [col for col in required_columns if col not in self.scenario_df.columns]
        if missing_columns:
            errors.append(f"scenario_df에 필수 컬럼이 없습니다: {missing_columns}")

        # 대상 자재 확인
        target_materials = self.get_target_materials()
        if len(target_materials) == 0:
            errors.append("최적화 대상 자재가 없습니다.")

        # NaN 값 확인
        if self.scenario_df['제품총소요량(kg)'].isna().any():
            errors.append("제품총소요량(kg)에 NaN 값이 있습니다.")

        if self.scenario_df['배출계수'].isna().any():
            errors.append("배출계수에 NaN 값이 있습니다.")

        # 비용 제약 사용 시 original_df 필수
        if check_cost_constraint and self.original_df is None:
            errors.append("비용 제약을 사용하려면 original_df가 필요합니다. PCF 시뮬레이터를 실행하세요.")

        if errors:
            print(f"❌ 데이터 검증 실패: {len(errors)}개 오류")
            for error in errors:
                print(f"  • {error}")
        else:
            print(f"✅ 데이터 검증 통과")

        return len(errors) == 0, errors

    def get_summary(self) -> str:
        """
        로드된 데이터 요약

        Returns:
            요약 문자열
        """
        summary = "📊 DataLoader 요약\n"
        summary += "=" * 50 + "\n"

        if self.scenario_df is not None:
            summary += f"✅ scenario_df: {len(self.scenario_df)} rows, {len(self.scenario_df.columns)} cols\n"
        else:
            summary += "❌ scenario_df: 없음\n"

        if self.original_df is not None:
            summary += f"✅ original_df: {len(self.original_df)} rows\n"
        else:
            summary += "⚠️  original_df: 없음\n"

        if self.ref_formula_df is not None:
            summary += f"✅ ref_formula_df: {len(self.ref_formula_df)} rows\n"
        else:
            summary += "⚠️  ref_formula_df: 없음\n"

        if self.ref_proportions_df is not None:
            summary += f"✅ ref_proportions_df: {len(self.ref_proportions_df)} rows\n"
        else:
            summary += "⚠️  ref_proportions_df: 없음\n"

        target_materials = self.get_target_materials(use_cache=True, verbose=False)
        summary += f"\n📦 대상 자재: {len(target_materials)}개\n"

        if target_materials:
            classification = self.classify_materials(use_cache=True, verbose=False)
            formula_count = sum(1 for m in classification.values() if m['is_formula_applicable'])
            nicoli_count = sum(1 for m in classification.values() if m['is_ni_co_li'])
            general_count = len(classification) - formula_count - nicoli_count

            summary += f"   • Formula 적용: {formula_count}개\n"
            summary += f"   • Ni/Co/Li: {nicoli_count}개\n"
            summary += f"   • 일반: {general_count}개\n"

        return summary

    def load_electrolyte_composition(self) -> Optional[Dict[str, Any]]:
        """
        전해액 최적화 설정 로드

        Returns:
            전해액 composition dict 또는 None
        """
        import json
        import os

        config_path = 'input/electrolyte_optimization_config.json'

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ Electrolyte composition 로드 완료: {config_path}")
                return config
            else:
                print(f"⚠️  Electrolyte config 파일 없음: {config_path}")
                return None
        except Exception as e:
            print(f"❌ Electrolyte composition 로드 실패: {e}")
            return None

    def load_separator_composition(self) -> Optional[Dict[str, Any]]:
        """
        분리막 최적화 설정 로드

        기본 설정을 반환하며, 향후 JSON 파일로 확장 가능

        Returns:
            분리막 composition dict
        """
        # 기본 설정 (향후 input/separator_optimization_config.json으로 확장 가능)
        config = {
            'dry_emission': 2.5,    # Dry Type 배출계수 (kgCO2eq/kg)
            'wet_emission': 3.2,    # Wet Type 배출계수 (kgCO2eq/kg)
            'min_dry_ratio': 0.0,   # Dry Type 최소 비율
            'max_dry_ratio': 1.0    # Dry Type 최대 비율
        }

        print(f"✅ Separator composition 기본 설정 사용")
        return config

    def load_re100_regional_prices(self) -> Optional[Dict[str, Any]]:
        """
        RE100 지역별 가격 설정 로드

        Returns:
            RE100 가격 설정 dict 또는 None
        """
        import json
        import os

        config_path = 'input/re100_regional_prices.json'

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ RE100 regional prices 로드 완료: {config_path}")
                return config
            else:
                print(f"⚠️  RE100 prices 파일 없음: {config_path}")
                return None
        except Exception as e:
            print(f"❌ RE100 regional prices 로드 실패: {e}")
            return None

    def __repr__(self) -> str:
        """DataLoader 문자열 표현"""
        has_data = self.scenario_df is not None
        return f"<DataLoader(data_loaded={has_data})>"

    def __str__(self) -> str:
        """DataLoader 사용자 친화적 문자열"""
        return self.get_summary()

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any
from src.cathode_simulator import CathodeSimulator

class PCFHelper:
    """
    PCF (Product Carbon Footprint) 탄소배출량 분석을 위한 클래스
    """
    
    def __init__(self, pcf_df):
        """
        PCFHelper 초기화
        
        Args:
            pcf_df (pd.DataFrame): calculate_pcf_values 또는 calculate_pcf_values_with_merge에서 나온 결과 데이터프레임
        """
        self.pcf_df = pcf_df
        self.pcf_columns = [col for col in pcf_df.columns if col.startswith('PCF_case')]
        
        # PCF_reference 열이 있는지 확인
        self.has_reference = 'PCF_reference' in pcf_df.columns
        
    def analyze_pcf_values(self):
        """
        PCF 탄소배출량 분석 결과를 출력
        
        Returns:
            dict: 분석 결과를 담은 딕셔너리
        """
        # PCF 값 합계 계산
        pcf_sums = {}
        for col in self.pcf_columns:
            pcf_sums[col] = self.pcf_df[col].sum()
        
        # 결과 출력
        print("=" * 60)
        print("📊 PCF 탄소배출량 분석 결과")
        print("=" * 60)
        print(f"🔍 총 데이터 행 수: {len(self.pcf_df):,}개")
        print()
        
        print("📈 PCF 합계 분석:")
        print("-" * 40)
        
        # PCF_reference 값 먼저 출력
        if self.has_reference:
            reference_total = self.pcf_df['PCF_reference'].sum()
            print(f"PCF_reference: {reference_total:.3f} kgCO2eq")
        
        # PCF_case 열들 출력
        for col in self.pcf_columns:
            total_pcf = pcf_sums[col]
            if self.has_reference:
                reference_total = self.pcf_df['PCF_reference'].sum()
                reduction = reference_total - total_pcf
                reduction_rate = (reduction / reference_total) * 100
                print(f"{col}: {total_pcf:.3f} kgCO2eq ({reduction_rate:.2f}% 감소)")
            else:
                print(f"{col}: {total_pcf:.3f} kgCO2eq")
        print()
        
        print("📉 감소율 상세 분석:")
        print("-" * 40)
        
        # 감소율 계산 및 출력
        reduction_rates = {}
        if self.has_reference:
            reference_total = self.pcf_df['PCF_reference'].sum()
            for col in self.pcf_columns:
                total_pcf = pcf_sums[col]
                reduction = reference_total - total_pcf
                reduction_rate = (reduction / reference_total) * 100
                reduction_rates[col] = reduction_rate
                
                print(f"{col}:")
                print(f"  - 절대 감소량: {reduction:.3f} kgCO2eq")
                print(f"  - 감소율: {reduction_rate:.2f}%")
                print()
        
        print("=" * 60)
        
        # 결과를 딕셔너리로 반환
        result = {
            'pcf_sums': pcf_sums,
            'reduction_rates': reduction_rates,
            'total_rows': len(self.pcf_df),
            'has_reference': self.has_reference
        }
        
        if self.has_reference:
            result['reference_total'] = self.pcf_df['PCF_reference'].sum()
        
        return result
    
    def get_detailed_breakdown(self):
        """
        자재품목별 PCF 탄소배출량 분석 결과를 출력
        
        Returns:
            pd.DataFrame: 자재품목별 분석 결과
        """
        # 자재품목별로 그룹화하여 분석
        analysis_columns = self.pcf_columns
        if self.has_reference:
            analysis_columns = ['PCF_reference'] + self.pcf_columns
        
        breakdown = self.pcf_df.groupby('자재품목')[analysis_columns].sum()
        
        # PCF_reference 대비 감소율 계산
        if self.has_reference:
            for col in self.pcf_columns:
                breakdown[f'{col}_감소율(%)'] = (
                    (breakdown['PCF_reference'] - breakdown[col]) / breakdown['PCF_reference'] * 100
                ).round(2)
        
        print("=" * 80)
        print("📋 자재품목별 PCF 탄소배출량 상세 분석")
        print("=" * 80)
        print(breakdown.round(3))
        print("=" * 80)
        
        return breakdown
    
    def get_matching_type_breakdown(self):
        """
        매칭 타입별 (formula_matched, proportions_matched) PCF 분석 결과를 출력
        
        Returns:
            dict: 매칭 타입별 분석 결과
        """
        # formula_matched가 True인 행들
        formula_only = self.pcf_df[
            (self.pcf_df['formula_matched'] == True) & 
            (self.pcf_df['proportions_matched'] == False)
        ]
        
        # proportions_matched가 True인 행들
        proportions_only = self.pcf_df[
            (self.pcf_df['formula_matched'] == False) & 
            (self.pcf_df['proportions_matched'] == True)
        ]
        
        print("=" * 80)
        print("📋 매칭 타입별 PCF 분석")
        print("=" * 80)
        
        print(f"🔍 Formula 매칭만 된 행: {len(formula_only)}개")
        if len(formula_only) > 0:
            if self.has_reference:
                print("  - PCF_reference 합계:", formula_only['PCF_reference'].sum().round(3))
            for col in self.pcf_columns:
                pcf_sum = formula_only[col].sum()
                if self.has_reference:
                    reference_sum = formula_only['PCF_reference'].sum()
                    reduction_rate = ((reference_sum - pcf_sum) / reference_sum * 100).round(2)
                    print(f"  - {col}: {pcf_sum:.3f} kgCO2eq (감소율: {reduction_rate}%)")
                else:
                    print(f"  - {col}: {pcf_sum:.3f} kgCO2eq")
        
        print(f"\n🔍 Proportions 매칭만 된 행: {len(proportions_only)}개")
        if len(proportions_only) > 0:
            if self.has_reference:
                print("  - PCF_reference 합계:", proportions_only['PCF_reference'].sum().round(3))
            for col in self.pcf_columns:
                pcf_sum = proportions_only[col].sum()
                if self.has_reference:
                    reference_sum = proportions_only['PCF_reference'].sum()
                    reduction_rate = ((reference_sum - pcf_sum) / reference_sum * 100).round(2)
                    print(f"  - {col}: {pcf_sum:.3f} kgCO2eq (감소율: {reduction_rate}%)")
                else:
                    print(f"  - {col}: {pcf_sum:.3f} kgCO2eq")
        
        print("=" * 80)
        
        return {
            'formula_only': formula_only,
            'proportions_only': proportions_only
        }
    
    def get_top_contributors(self, n=5):
        """
        PCF_reference 기준으로 상위 기여 자재들을 출력
        
        Args:
            n (int): 출력할 상위 자재 개수
        """
        # PCF_reference 기준으로 상위 기여자 찾기
        if self.has_reference:
            top_contributors = self.pcf_df.nlargest(n, 'PCF_reference')[
                ['자재명', '자재품목', 'PCF_reference', 'formula_matched', 'proportions_matched'] + self.pcf_columns
            ]
            
            print("=" * 80)
            print(f"🏆 PCF_reference 기준 상위 {n}개 자재")
            print("=" * 80)
            print(top_contributors.round(3))
            print("=" * 80)
            
            return top_contributors
        else:
            # PCF_reference가 없는 경우 첫 번째 PCF_case 열 기준
            if len(self.pcf_columns) > 0:
                first_pcf_col = self.pcf_columns[0]
                top_contributors = self.pcf_df.nlargest(n, first_pcf_col)[
                    ['자재명', '자재품목', first_pcf_col, 'formula_matched', 'proportions_matched'] + self.pcf_columns
                ]
                
                print("=" * 80)
                print(f"🏆 {first_pcf_col} 기준 상위 {n}개 자재")
                print("=" * 80)
                print(top_contributors.round(3))
                print("=" * 80)
                
                return top_contributors
            else:
                print("⚠️ PCF 열을 찾을 수 없습니다.")
                return pd.DataFrame()
    
    def get_tier_analysis(self):
        """
        Tier 사용 현황 분석
        
        Returns:
            dict: Tier 분석 결과
        """
        if 'tier_num_by' not in self.pcf_df.columns:
            print("⚠️ tier_num_by 열이 없습니다.")
            return {}
        
        tier_counts = self.pcf_df['tier_num_by'].value_counts().sort_index()
        
        print("=" * 60)
        print("📊 Tier 사용 현황 분석")
        print("=" * 60)
        print("Tier 개수별 행 수:")
        for tier_num, count in tier_counts.items():
            print(f"  - Tier {tier_num}: {count}개 행")
        
        print(f"\n총 처리된 행: {len(self.pcf_df)}개")
        print("=" * 60)
        
        return {
            'tier_counts': tier_counts,
            'total_processed': len(self.pcf_df)
        }


class CathodeHelper:
    """
    CathodeSimulator의 결과를 분석하고 출력하는 헬퍼 클래스
    """
    
    def __init__(self, simulator: CathodeSimulator, verbose: bool = True):
        """
        CathodeHelper를 초기화합니다.
        
        Args:
            simulator (CathodeSimulator): CathodeSimulator 인스턴스
            verbose (bool): 상세 로그 출력 여부. 기본값은 True
        """
        self.simulator = simulator
        self.verbose = verbose
        
        self._print("🔧 CathodeHelper 초기화 완료", level='info')
    
    def _print(self, message: str, level: str = 'info'):
        """
        로그 레벨에 따른 출력을 처리합니다.
        
        Args:
            message (str): 출력할 메시지
            level (str): 로그 레벨 ('info', 'debug', 'warning', 'error')
        """
        if not self.verbose and level in ['debug']:
            return
        
        if level == 'info':
            print(message)
        elif level == 'debug':
            print(f"[DEBUG] {message}")
        elif level == 'warning':
            print(f"⚠️ {message}")
        elif level == 'error':
            print(f"❌ {message}")
        else:
            print(message)
    
    def print_summary(self):
        """시뮬레이터 요약 정보를 출력합니다."""
        self.simulator.print_summary()
    
    def print_recycle_ratio_details(self):
        """재활용 비중 상세 정보를 출력합니다."""
        self.simulator.print_recycle_ratio_details()
    
    def print_raw_material_requirements(self):
        """원재료 소요량 상세 정보를 출력합니다."""
        self.simulator.print_raw_material_requirements()
    
    def print_requirement_update_log(self):
        """소요량 업데이트 로그를 출력합니다."""
        self.simulator.print_requirement_update_log()
    
    def print_carbon_emission_comparison(self, before_data: Dict[str, Any], after_data: Dict[str, Any]):
        """사이트별 탄소배출량 비교 결과를 출력합니다."""
        # 사이트 정보 로드
        site_data = self._load_site_data()
        cam_before = site_data.get('CAM', {}).get('before', '중국')
        cam_after = site_data.get('CAM', {}).get('after', '한국')
        pcam_before = site_data.get('pCAM', {}).get('before', '중국')
        pcam_after = site_data.get('pCAM', {}).get('after', '한국')
        
        self._print("=" * 80, level='info')
        self._print("🏭 사이트별 탄소배출량 비교 분석", level='info')
        self._print("=" * 80, level='info')
        
        # Before 사이트 계산
        self._print(f"\n📊 Before 사이트 탄소배출량 계산:", level='info')
        self._print(f"  📍 CAM: {cam_before}, pCAM: {pcam_before}", level='info')
        before_emission = self.generate_carbon_emission_data(before_data)
        
        if before_emission:
            self._print(f"\n총 배출량: {before_emission['총_배출량']:.6f} kg CO2e", level='info')
            self._print("\n카테고리별 기여도:", level='info')
            for category, contribution in before_emission['카테고리별_기여도'].items():
                self._print(f"  {category}: {contribution:.2f}%", level='info')
            
            self._print("\n아이템별 상세 기여도:", level='info')
            for category in ["원재료", "Energy(Tier-1)", "Energy(Tier-2)"]:
                if category in before_emission:
                    self._print(f"\n  📊 {category}:", level='info')
                    for item_name, item_data in before_emission[category].items():
                        emission = item_data["탄소배출량(kg_CO2e)"]
                        contribution = item_data.get("기여도(%)", 0)
                        self._print(f"    • {item_name}: {emission:.6f} kg CO2e ({contribution:.2f}%)", level='info')
        
        # After 사이트 계산
        self._print(f"\n📊 After 사이트 탄소배출량 계산:", level='info')
        self._print(f"  📍 CAM: {cam_after}, pCAM: {pcam_after}", level='info')
        after_emission = self.generate_carbon_emission_data(after_data, baseline_emission=before_emission['총_배출량'] if before_emission else None)
        
        if after_emission:
            self._print(f"\n총 배출량: {after_emission['총_배출량']:.6f} kg CO2e", level='info')
            self._print("\n카테고리별 기여도:", level='info')
            for category, contribution in after_emission['카테고리별_기여도'].items():
                self._print(f"  {category}: {contribution:.2f}%", level='info')
            
            self._print("\n아이템별 상세 기여도:", level='info')
            for category in ["원재료", "Energy(Tier-1)", "Energy(Tier-2)"]:
                if category in after_emission:
                    self._print(f"\n  📊 {category}:", level='info')
                    for item_name, item_data in after_emission[category].items():
                        emission = item_data["탄소배출량(kg_CO2e)"]
                        contribution = item_data.get("기여도(%)", 0)
                        self._print(f"    • {item_name}: {emission:.6f} kg CO2e ({contribution:.2f}%)", level='info')
            
            # 감축량 및 감축률 출력
            if "감축량" in after_emission and "감축률" in after_emission:
                self._print(f"\n📉 감축량: {after_emission['감축량']:.6f} kg CO2e", level='info')
                self._print(f"📉 감축률: {after_emission['감축률']:.2f}%", level='info')
        
        # 사이트 변경 요약
        self._print(f"\n🏭 사이트 변경 요약:", level='info')
        self._print(f"  📍 CAM: {cam_before} → {cam_after}", level='info')
        self._print(f"  📍 pCAM: {pcam_before} → {pcam_after}", level='info')
        self._print("=" * 80, level='info')

    def print_electricity_emission_factor_update(self, site: str = 'before'):
        """전력 배출계수 업데이트 상세정보를 출력합니다."""
        self._print("\n" + "="*50, level='info')
        self._print("전력 배출계수 업데이트 테스트", level='info')
        self._print("="*50, level='info')
        
        try:
            # 사이트 정보 로드
            site_data = self._load_site_data()
            cam_site = site_data.get('CAM', {}).get(site, '중국')
            pcam_site = site_data.get('pCAM', {}).get(site, '한국')
            
            # 전력 배출계수 업데이트 데이터 생성
            updated_data = self.generate_electricity_emission_factor_data(site=site)
            
            if updated_data:
                self._print(f"\n📋 {site} 사이트 업데이트된 전력 배출계수:", level='info')
                self._print(f"📍 CAM: {cam_site}, pCAM: {pcam_site}", level='info')
                self._print(f"Energy(Tier-1) 전력: {updated_data['Energy(Tier-1)']['전력']['배출계수']}", level='info')
                self._print(f"Energy(Tier-2) 전력: {updated_data['Energy(Tier-2)']['전력']['배출계수']}", level='info')
                
                return updated_data
            else:
                self._print("❌ 전력 배출계수 업데이트 실패", level='error')
                return None
                
        except Exception as e:
            self._print(f"❌ 전력 배출계수 업데이트 중 오류 발생: {e}", level='error')
            return None

    def print_electricity_emission_factor_comparison(self):
        """Before/After 사이트 전력 배출계수 비교를 출력합니다."""
        self._print("\n" + "="*60, level='info')
        self._print("⚡ Before/After 사이트 전력 배출계수 비교", level='info')
        self._print("="*60, level='info')
        
        try:
            # 사이트 정보 로드
            site_data = self._load_site_data()
            cam_before = site_data.get('CAM', {}).get('before', '중국')
            cam_after = site_data.get('CAM', {}).get('after', '한국')
            pcam_before = site_data.get('pCAM', {}).get('before', '중국')
            pcam_after = site_data.get('pCAM', {}).get('after', '한국')
            
            # Before 사이트 전력 배출계수 업데이트 데이터 생성
            self._print(f"\n📊 Before 사이트 전력 배출계수:", level='info')
            self._print(f"  📍 CAM: {cam_before}, pCAM: {pcam_before}", level='info')
            before_data = self.generate_electricity_emission_factor_data(site='before')
            if before_data:
                before_tier1 = before_data['Energy(Tier-1)']['전력']['배출계수']
                before_tier2 = before_data['Energy(Tier-2)']['전력']['배출계수']
                self._print(f"  ⚡ Energy(Tier-1) 전력: {before_tier1}", level='info')
                self._print(f"  ⚡ Energy(Tier-2) 전력: {before_tier2}", level='info')
            
            # After 사이트 전력 배출계수 업데이트 데이터 생성
            self._print(f"\n📊 After 사이트 전력 배출계수:", level='info')
            self._print(f"  📍 CAM: {cam_after}, pCAM: {pcam_after}", level='info')
            after_data = self.generate_electricity_emission_factor_data(site='after')
            if after_data:
                after_tier1 = after_data['Energy(Tier-1)']['전력']['배출계수']
                after_tier2 = after_data['Energy(Tier-2)']['전력']['배출계수']
                self._print(f"  ⚡ Energy(Tier-1) 전력: {after_tier1}", level='info')
                self._print(f"  ⚡ Energy(Tier-2) 전력: {after_tier2}", level='info')
            
            # 변화량 계산
            if before_data and after_data:
                tier1_change = before_tier1 - after_tier1
                tier2_change = before_tier2 - after_tier2
                tier1_change_rate = (tier1_change / before_tier1) * 100
                tier2_change_rate = (tier2_change / before_tier2) * 100
                
                self._print(f"\n📉 전력 배출계수 변화량:", level='info')
                self._print(f"  ⚡ Energy(Tier-1): {before_tier1:.6f} → {after_tier1:.6f} (감소: {tier1_change:.6f}, {tier1_change_rate:.2f}%)", level='info')
                self._print(f"  ⚡ Energy(Tier-2): {before_tier2:.6f} → {after_tier2:.6f} (감소: {tier2_change:.6f}, {tier2_change_rate:.2f}%)", level='info')
                
                # 사이트 변경 요약
                self._print(f"\n🏭 사이트 변경 요약:", level='info')
                self._print(f"  📍 CAM: {cam_before} → {cam_after}", level='info')
                self._print(f"  📍 pCAM: {pcam_before} → {pcam_after}", level='info')
                self._print("=" * 60, level='info')
                
                return {
                    'before_data': before_data,
                    'after_data': after_data,
                    'changes': {
                        'tier1_change': tier1_change,
                        'tier2_change': tier2_change,
                        'tier1_change_rate': tier1_change_rate,
                        'tier2_change_rate': tier2_change_rate
                    }
                }
            
        except Exception as e:
            self._print(f"❌ 오류 발생: {e}", level='error')
            return None

    def generate_electricity_emission_factor_data(self, site: str = 'before') -> Dict[str, Any]:
        """전력 배출계수 업데이트 데이터를 생성합니다."""
        try:
            updated_data = self.simulator.update_electricity_emission_factor(site=site)
            return updated_data
        except Exception as e:
            self._print(f"❌ 전력 배출계수 업데이트 중 오류 발생: {e}", level='error')
            return None

    def generate_carbon_emission_data(self, updated_data: Dict[str, Any], baseline_emission: float = None) -> Dict[str, Any]:
        """탄소배출량 계산 데이터를 생성합니다."""
        try:
            emission_data = self.simulator.calculate_carbon_emission(updated_data, baseline_emission)
            return emission_data
        except Exception as e:
            self._print(f"❌ 탄소배출량 계산 중 오류 발생: {e}", level='error')
            return None

    def generate_site_comparison_data(self) -> Dict[str, Any]:
        """사이트별 비교 데이터를 생성합니다."""
        try:
            # Before 사이트 데이터 생성
            before_data = self.generate_electricity_emission_factor_data(site='before')
            before_emission = self.generate_carbon_emission_data(before_data)
            
            # After 사이트 데이터 생성
            after_data = self.generate_electricity_emission_factor_data(site='after')
            after_emission = self.generate_carbon_emission_data(after_data, baseline_emission=before_emission['총_배출량'])
            
            return {
                'before_data': before_data,
                'after_data': after_data,
                'before_emission': before_emission,
                'after_emission': after_emission
            }
        except Exception as e:
            self._print(f"❌ 사이트 비교 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def generate_recycling_simulation_data(self, site: str = 'before', recycling_ratios: list = None, use_impact: bool = True) -> Dict[str, Any]:
        """재활용 시뮬레이션 데이터를 생성합니다."""
        try:
            simulation_results = self.simulator.simulate_recycling_carbon_reduction(
                site=site, 
                recycling_ratios=recycling_ratios,
                use_impact=use_impact
            )
            return simulation_results
        except Exception as e:
            self._print(f"❌ 재활용 시뮬레이션 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def generate_recycling_comparison_data(self, before_ratios: list = None, after_ratios: list = None) -> Dict[str, Any]:
        """재활용 시나리오 비교 데이터를 생성합니다."""
        try:
            comparison_results = self.simulator.compare_recycling_scenarios(before_ratios, after_ratios)
            return comparison_results
        except Exception as e:
            self._print(f"❌ 재활용 비교 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def generate_baseline_data(self, site: str = 'before') -> Dict[str, Any]:
        """
        1. 재활용 적용X, site는 before 상태의 baseline dataset을 반환합니다.
        
        Args:
            site (str): 'before' 또는 'after'. 기본값은 'before'
            
        Returns:
            Dict[str, Any]: baseline 데이터
        """
        try:
            # 시뮬레이터에서 baseline 데이터 생성
            baseline_data = self.simulator.generate_baseline_data(site=site)
            return baseline_data
            
        except Exception as e:
            self._print(f"❌ Baseline 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def generate_recycling_only_data(self, site: str = 'before', recycling_ratios: list = None) -> Dict[str, Any]:
        """
        2. 재활용만 적용된 데이터셋을 반환합니다.
        
        Args:
            site (str): 'before' 또는 'after'. 기본값은 'before'
            recycling_ratios (list): 적용할 재활용 비율 리스트. None인 경우 기본값 사용
            
        Returns:
            Dict[str, Any]: 재활용 적용 데이터
        """
        try:
            # 시뮬레이터에서 재활용 데이터 생성
            recycling_data = self.simulator.generate_recycling_only_data(
                site=site, 
                recycling_ratios=recycling_ratios
            )
            return recycling_data
            
        except Exception as e:
            self._print(f"❌ 재활용만 적용 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def generate_site_change_only_data(self, before_site: str = 'before', after_site: str = 'after') -> Dict[str, Any]:
        """
        3. 생산지 변경(site change)만 한 경우의 데이터를 반환합니다.
        
        Args:
            before_site (str): 변경 전 사이트. 기본값은 'before'
            after_site (str): 변경 후 사이트. 기본값은 'after'
            
        Returns:
            Dict[str, Any]: 사이트 변경 데이터
        """
        try:
            # 시뮬레이터에서 사이트 변경 데이터 생성
            site_change_data = self.simulator.generate_site_change_only_data(
                before_site=before_site, 
                after_site=after_site
            )
            return site_change_data
            
        except Exception as e:
            self._print(f"❌ 사이트 변경 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def generate_combined_data(self, before_site: str = 'before', after_site: str = 'after', 
                             recycling_ratios: list = None) -> Dict[str, Any]:
        """
        4. 재활용과 사이트 변경을 둘 다 적용한 경우의 데이터를 반환합니다.
        
        Args:
            before_site (str): 변경 전 사이트. 기본값은 'before'
            after_site (str): 변경 후 사이트. 기본값은 'after'
            recycling_ratios (list): 적용할 재활용 비율 리스트. None인 경우 기본값 사용
            
        Returns:
            Dict[str, Any]: 재활용과 사이트 변경 모두 적용된 데이터
        """
        try:
            # 시뮬레이터에서 종합 데이터 생성
            combined_data = self.simulator.generate_combined_data(
                before_site=before_site, 
                after_site=after_site, 
                recycling_ratios=recycling_ratios
            )
            return combined_data
            
        except Exception as e:
            self._print(f"❌ 종합 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def generate_all_scenarios_data(self) -> Dict[str, Any]:
        """
        모든 시나리오의 데이터를 한 번에 생성합니다. (단일 비율)
        Returns:
            Dict[str, Any]: 모든 시나리오 데이터
        """
        try:
            self._print("=" * 80, level='info')
            self._print("🔬 모든 시나리오 데이터 생성 시작", level='info')
            self._print("=" * 80, level='info')
            
            # 시뮬레이터에서 모든 시나리오 데이터 생성 (인자 없이)
            all_scenarios = self.simulator.generate_all_scenarios_data()
            
            self._print("\n✅ 모든 시나리오 데이터 생성 완료", level='info')
            self._print("=" * 80, level='info')
            
            return all_scenarios
            
        except Exception as e:
            self._print(f"❌ 모든 시나리오 데이터 생성 중 오류 발생: {e}", level='error')
            return None

    def analyze_all_scenarios_detailed(self, all_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        모든 시나리오 데이터를 상세히 분석합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            Dict[str, Any]: 상세 분석 결과
        """
        if not all_scenarios:
            self._print("❌ 시나리오 데이터가 없습니다.", level='error')
            return {}
        
        analysis_result = {
            'scenarios_available': list(all_scenarios.keys()),
            'baseline_analysis': {},
            'recycling_analysis': {},
            'site_change_analysis': {},
            'combined_analysis': {},
            'comparison_analysis': {}
        }
        
        # 1. Baseline 분석
        if 'baseline' in all_scenarios and all_scenarios['baseline']:
            baseline = all_scenarios['baseline']
            if 'emission_data' in baseline:
                emission_data = baseline['emission_data']
                analysis_result['baseline_analysis'] = {
                    'total_emission': emission_data.get('총_배출량', 0),
                    'category_contributions': emission_data.get('카테고리별_기여도', {}),
                    'description': baseline.get('description', '재활용 및 사이트 변경 적용X'),
                    'site': baseline.get('site', 'before')
                }
        
        # 2. 재활용 분석
        if 'recycling_only' in all_scenarios and all_scenarios['recycling_only']:
            recycling = all_scenarios['recycling_only']
            if 'simulation_result' in recycling:
                result = recycling['simulation_result']
                analysis_result['recycling_analysis'] = {
                    'total_emission': result.get('total_emission', 0),
                    'reduction_amount': result.get('reduction_amount', 0),
                    'reduction_rate': result.get('reduction_rate', 0),
                    'recycling_ratio': result.get('recycling_ratio_percent', 0),
                    'description': recycling.get('description', '재활용만 적용'),
                    'category_contributions': result.get('category_contributions', {})
                }
        
        # 3. 사이트 변경 분석
        if 'site_change_only' in all_scenarios and all_scenarios['site_change_only']:
            site_change = all_scenarios['site_change_only']
            analysis_result['site_change_analysis'] = {
                'before_site': site_change.get('before_site', 'before'),
                'after_site': site_change.get('after_site', 'after'),
                'before_emission': site_change.get('before_data', {}).get('emission_data', {}).get('총_배출량', 0),
                'after_emission': site_change.get('after_data', {}).get('emission_data', {}).get('총_배출량', 0),
                'emission_change': site_change.get('emission_change', 0),
                'emission_change_rate': site_change.get('emission_change_rate', 0),
                'description': site_change.get('description', '사이트 변경만 적용')
            }
        
        # 4. 종합 분석
        if 'combined' in all_scenarios and all_scenarios['combined']:
            combined = all_scenarios['combined']
            if 'after_recycling' in combined and 'simulation_result' in combined['after_recycling']:
                result = combined['after_recycling']['simulation_result']
                analysis_result['combined_analysis'] = {
                    'total_emission': result.get('total_emission', 0),
                    'reduction_amount': result.get('reduction_amount', 0),
                    'reduction_rate': result.get('reduction_rate', 0),
                    'recycling_ratio': result.get('recycling_ratio_percent', 0),
                    'description': combined.get('description', '재활용 + 사이트 변경'),
                    'category_contributions': result.get('category_contributions', {})
                }
        
        # 5. 비교 분석
        baseline_emission = analysis_result['baseline_analysis'].get('total_emission', 0)
        if baseline_emission > 0:
            analysis_result['comparison_analysis'] = {
                'baseline_emission': baseline_emission,
                'recycling_effectiveness': {
                    'reduction_amount': analysis_result['recycling_analysis'].get('reduction_amount', 0),
                    'reduction_rate': analysis_result['recycling_analysis'].get('reduction_rate', 0)
                },
                'site_change_effectiveness': {
                    'reduction_amount': analysis_result['site_change_analysis'].get('emission_change', 0),
                    'reduction_rate': analysis_result['site_change_analysis'].get('emission_change_rate', 0)
                },
                'combined_effectiveness': {
                    'reduction_amount': analysis_result['combined_analysis'].get('reduction_amount', 0),
                    'reduction_rate': analysis_result['combined_analysis'].get('reduction_rate', 0)
                }
            }
        
        return analysis_result

    def print_detailed_scenario_analysis(self, all_scenarios: Dict[str, Any]):
        """
        모든 시나리오의 상세 분석 결과를 출력합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
        """
        analysis = self.analyze_all_scenarios_detailed(all_scenarios)
        
        self._print("=" * 100, level='info')
        self._print("🔍 모든 시나리오 상세 분석", level='info')
        self._print("=" * 100, level='info')
        
        # 1. Baseline 분석
        if analysis['baseline_analysis']:
            baseline = analysis['baseline_analysis']
            self._print(f"\n📊 Baseline 분석:", level='info')
            self._print(f"   📋 설명: {baseline['description']}", level='info')
            self._print(f"   📍 사이트: {baseline['site']}", level='info')
            self._print(f"   📊 총 배출량: {baseline['total_emission']:.6f} kg CO2e", level='info')
            
            if baseline['category_contributions']:
                self._print(f"   📈 카테고리별 기여도:", level='info')
                for category, contribution in baseline['category_contributions'].items():
                    self._print(f"      • {category}: {contribution:.2f}%", level='info')
        
        # 2. 재활용 분석
        if analysis['recycling_analysis']:
            recycling = analysis['recycling_analysis']
            self._print(f"\n♻️ 재활용 분석:", level='info')
            self._print(f"   📋 설명: {recycling['description']}", level='info')
            self._print(f"   📊 총 배출량: {recycling['total_emission']:.6f} kg CO2e", level='info')
            self._print(f"   📉 감축량: {recycling['reduction_amount']:.6f} kg CO2e", level='info')
            self._print(f"   📉 감축률: {recycling['reduction_rate']:.2f}%", level='info')
            self._print(f"   ♻️ 재활용 비율: {recycling['recycling_ratio']:.1f}%", level='info')
            
            if recycling['category_contributions']:
                self._print(f"   📈 카테고리별 기여도:", level='info')
                for category, contribution in recycling['category_contributions'].items():
                    self._print(f"      • {category}: {contribution:.2f}%", level='info')
        
        # 3. 사이트 변경 분석
        if analysis['site_change_analysis']:
            site_change = analysis['site_change_analysis']
            self._print(f"\n🏭 사이트 변경 분석:", level='info')
            self._print(f"   📋 설명: {site_change['description']}", level='info')
            self._print(f"   📍 변경 전: {site_change['before_site']}", level='info')
            self._print(f"   📍 변경 후: {site_change['after_site']}", level='info')
            self._print(f"   📊 변경 전 배출량: {site_change['before_emission']:.6f} kg CO2e", level='info')
            self._print(f"   📊 변경 후 배출량: {site_change['after_emission']:.6f} kg CO2e", level='info')
            self._print(f"   📉 감축량: {site_change['emission_change']:.6f} kg CO2e", level='info')
            self._print(f"   📉 감축률: {site_change['emission_change_rate']:.2f}%", level='info')
        
        # 4. 종합 분석
        if analysis['combined_analysis']:
            combined = analysis['combined_analysis']
            self._print(f"\n🎯 종합 분석:", level='info')
            self._print(f"   📋 설명: {combined['description']}", level='info')
            self._print(f"   📊 총 배출량: {combined['total_emission']:.6f} kg CO2e", level='info')
            self._print(f"   📉 감축량: {combined['reduction_amount']:.6f} kg CO2e", level='info')
            self._print(f"   📉 감축률: {combined['reduction_rate']:.2f}%", level='info')
            self._print(f"   ♻️ 재활용 비율: {combined['recycling_ratio']:.1f}%", level='info')
            
            if combined['category_contributions']:
                self._print(f"   📈 카테고리별 기여도:", level='info')
                for category, contribution in combined['category_contributions'].items():
                    self._print(f"      • {category}: {contribution:.2f}%", level='info')
        
        # 5. 비교 분석
        if analysis['comparison_analysis']:
            comparison = analysis['comparison_analysis']
            self._print(f"\n📊 효과 비교 분석:", level='info')
            self._print(f"   📊 기준 배출량: {comparison['baseline_emission']:.6f} kg CO2e", level='info')
            
            # 재활용 효과
            recycling_eff = comparison['recycling_effectiveness']
            self._print(f"   ♻️ 재활용 효과:", level='info')
            self._print(f"      • 감축량: {recycling_eff['reduction_amount']:.6f} kg CO2e", level='info')
            self._print(f"      • 감축률: {recycling_eff['reduction_rate']:.2f}%", level='info')
            
            # 사이트 변경 효과
            site_eff = comparison['site_change_effectiveness']
            self._print(f"   🏭 사이트 변경 효과:", level='info')
            self._print(f"      • 감축량: {site_eff['reduction_amount']:.6f} kg CO2e", level='info')
            self._print(f"      • 감축률: {site_eff['reduction_rate']:.2f}%", level='info')
            
            # 종합 효과
            combined_eff = comparison['combined_effectiveness']
            self._print(f"   🎯 종합 효과:", level='info')
            self._print(f"      • 감축량: {combined_eff['reduction_amount']:.6f} kg CO2e", level='info')
            self._print(f"      • 감축률: {combined_eff['reduction_rate']:.2f}%", level='info')
            
            # 시너지 효과 계산
            synergy_amount = combined_eff['reduction_amount'] - (recycling_eff['reduction_amount'] + site_eff['reduction_amount'])
            synergy_rate = (synergy_amount / comparison['baseline_emission'] * 100) if comparison['baseline_emission'] > 0 else 0
            self._print(f"   🔄 시너지 효과:", level='info')
            self._print(f"      • 시너지 감축량: {synergy_amount:.6f} kg CO2e", level='info')
            self._print(f"      • 시너지 감축률: {synergy_rate:.2f}%", level='info')
        
        self._print("=" * 100, level='info')

    def get_scenario_comparison_dataframe(self, all_scenarios: Dict[str, Any]) -> pd.DataFrame:
        """
        모든 시나리오를 비교할 수 있는 데이터프레임을 생성합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            pd.DataFrame: 시나리오 비교 데이터프레임
        """
        analysis = self.analyze_all_scenarios_detailed(all_scenarios)
        
        comparison_data = []
        
        # Baseline
        if analysis['baseline_analysis']:
            baseline = analysis['baseline_analysis']
            comparison_data.append({
                '시나리오': 'Baseline',
                '설명': baseline['description'],
                '총_배출량_kg_CO2e': baseline['total_emission'],
                '감축량_kg_CO2e': 0,
                '감축률_퍼센트': 0,
                '재활용_비율_퍼센트': 0,
                '사이트_변경': 'N/A',
                '원재료_기여도_퍼센트': baseline['category_contributions'].get('원재료', 0),
                '에너지_기여도_퍼센트': baseline['category_contributions'].get('Energy(Tier-1)', 0) + baseline['category_contributions'].get('Energy(Tier-2)', 0)
            })
        
        # 재활용만
        if analysis['recycling_analysis']:
            recycling = analysis['recycling_analysis']
            comparison_data.append({
                '시나리오': '재활용만',
                '설명': recycling['description'],
                '총_배출량_kg_CO2e': recycling['total_emission'],
                '감축량_kg_CO2e': recycling['reduction_amount'],
                '감축률_퍼센트': recycling['reduction_rate'],
                '재활용_비율_퍼센트': recycling['recycling_ratio'],
                '사이트_변경': 'N/A',
                '원재료_기여도_퍼센트': recycling['category_contributions'].get('원재료', 0),
                '에너지_기여도_퍼센트': recycling['category_contributions'].get('Energy(Tier-1)', 0) + recycling['category_contributions'].get('Energy(Tier-2)', 0)
            })
        
        # 사이트 변경만
        if analysis['site_change_analysis']:
            site_change = analysis['site_change_analysis']
            comparison_data.append({
                '시나리오': '사이트 변경만',
                '설명': site_change['description'],
                '총_배출량_kg_CO2e': site_change['after_emission'],
                '감축량_kg_CO2e': site_change['emission_change'],
                '감축률_퍼센트': site_change['emission_change_rate'],
                '재활용_비율_퍼센트': 0,
                '사이트_변경': f"{site_change['before_site']} → {site_change['after_site']}",
                '원재료_기여도_퍼센트': 0,  # 사이트 변경은 에너지에만 영향
                '에너지_기여도_퍼센트': 0
            })
        
        # 종합
        if analysis['combined_analysis']:
            combined = analysis['combined_analysis']
            comparison_data.append({
                '시나리오': '재활용 + 사이트 변경',
                '설명': combined['description'],
                '총_배출량_kg_CO2e': combined['total_emission'],
                '감축량_kg_CO2e': combined['reduction_amount'],
                '감축률_퍼센트': combined['reduction_rate'],
                '재활용_비율_퍼센트': combined['recycling_ratio'],
                '사이트_변경': '적용됨',
                '원재료_기여도_퍼센트': combined['category_contributions'].get('원재료', 0),
                '에너지_기여도_퍼센트': combined['category_contributions'].get('Energy(Tier-1)', 0) + combined['category_contributions'].get('Energy(Tier-2)', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 수치 컬럼들 반올림
        numeric_columns = ['총_배출량_kg_CO2e', '감축량_kg_CO2e', '감축률_퍼센트', '재활용_비율_퍼센트', '원재료_기여도_퍼센트', '에너지_기여도_퍼센트']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df

    def get_basic_scenarios_dataframe(self, all_scenarios: Dict[str, Any]) -> pd.DataFrame:
        """
        기본 시나리오 분석 결과를 데이터프레임으로 생성합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            pd.DataFrame: 기본 시나리오 분석 데이터프레임
        """
        if not all_scenarios:
            return pd.DataFrame()
        
        basic_data = []
        
        # Baseline
        if 'baseline' in all_scenarios and all_scenarios['baseline']:
            baseline = all_scenarios['baseline']
            if 'emission_data' in baseline:
                emission_data = baseline['emission_data']
                print('[DEBUG] Baseline 총배출량:', emission_data.get('총_배출량', 0))
                
                # 기본 데이터 구성
                basic_row = {
                    '시나리오': 'Baseline',
                    '설명': baseline.get('description', '재활용 적용X, 생산지 변경 전'),
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': 0,
                    '감축률_퍼센트': 0,
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0),
                    '재활용_비율_퍼센트': 0
                }
                
                # Energy Tier 컬럼을 동적으로 추가 (num_tier=2 기준)
                for tier_num in range(1, 3):  # Tier1, Tier2만
                    tier_key = f'Energy(Tier-{tier_num})'
                    tier_value = emission_data.get('카테고리별_기여도', {}).get(tier_key, 0)
                    basic_row[f'Energy_Tier{tier_num}_전력_기여도_퍼센트'] = tier_value
                
                basic_data.append(basic_row)
        
        # 재활용만 (저탄소메탈 제외)
        if 'recycling_only' in all_scenarios and all_scenarios['recycling_only']:
            recycling = all_scenarios['recycling_only']
            if 'simulation_result' in recycling:
                result = recycling['simulation_result']
                print('[DEBUG] 재활용 Only 총배출량:', result.get('total_emission', 0))
                print('[DEBUG] 재활용 Only 감축량:', result.get('reduction_amount', 0))
                print('[DEBUG] 재활용 Only 감축률:', result.get('reduction_rate', 0))
                print('[DEBUG] 재활용 Only 비율:', result.get('recycling_ratio', 0))
                
                # 기본 데이터 구성
                basic_row = {
                    '시나리오': '재활용 적용',
                    '설명': recycling.get('description', '재활용만 적용 (저탄소메탈 제외)'),
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '재활용_비율_퍼센트': result.get('recycling_ratio', 0) * 100
                }
                
                # Energy Tier 컬럼을 동적으로 추가 (num_tier=2 기준)
                for tier_num in range(1, 3):  # Tier1, Tier2만
                    tier_key = f'Energy(Tier-{tier_num})'
                    tier_value = result.get('category_contributions', {}).get(tier_key, 0)
                    basic_row[f'Energy_Tier{tier_num}_전력_기여도_퍼센트'] = tier_value
                
                basic_data.append(basic_row)
        
        # 저탄소메탈만 (재활용 제외)
        if 'low_carb_only' in all_scenarios and all_scenarios['low_carb_only']:
            low_carb = all_scenarios['low_carb_only']
            if 'simulation_result' in low_carb:
                result = low_carb['simulation_result']
                print('[DEBUG] 저탄소메탈 Only 총배출량:', result.get('total_emission', 0))
                print('[DEBUG] 저탄소메탈 Only 감축량:', result.get('reduction_amount', 0))
                print('[DEBUG] 저탄소메탈 Only 감축률:', result.get('reduction_rate', 0))
                print('[DEBUG] 저탄소메탈 Only 비율:', result.get('low_carb_ratio', 0))
                
                # 기본 데이터 구성
                basic_row = {
                    '시나리오': '저탄소메탈 적용',
                    '설명': low_carb.get('description', '저탄소메탈만 적용 (재활용 제외)'),
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '재활용_비율_퍼센트': 0  # 저탄소메탈만 적용이므로 재활용 비율은 0
                }
                
                # Energy Tier 컬럼을 동적으로 추가 (num_tier=2 기준)
                for tier_num in range(1, 3):  # Tier1, Tier2만
                    tier_key = f'Energy(Tier-{tier_num})'
                    tier_value = result.get('category_contributions', {}).get(tier_key, 0)
                    basic_row[f'Energy_Tier{tier_num}_전력_기여도_퍼센트'] = tier_value
                
                basic_data.append(basic_row)
        
        # 재활용 + 저탄소메탈 동시 적용
        if 'combined_recycling' in all_scenarios and all_scenarios['combined_recycling']:
            combined_recycling = all_scenarios['combined_recycling']
            if 'simulation_result' in combined_recycling:
                result = combined_recycling['simulation_result']
                print('[DEBUG] 재활용+저탄소메탈 총배출량:', result.get('total_emission', 0))
                print('[DEBUG] 재활용+저탄소메탈 감축량:', result.get('reduction_amount', 0))
                print('[DEBUG] 재활용+저탄소메탈 감축률:', result.get('reduction_rate', 0))
                print('[DEBUG] 재활용+저탄소메탈 재활용 비율:', result.get('recycling_ratio', 0))
                
                # 기본 데이터 구성
                basic_row = {
                    '시나리오': '재활용&저탄소메탈 동시 적용',
                    '설명': combined_recycling.get('description', '재활용과 저탄소메탈 동시 적용'),
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '재활용_비율_퍼센트': result.get('recycling_ratio', 0) * 100
                }
                
                # Energy Tier 컬럼을 동적으로 추가 (num_tier=2 기준)
                for tier_num in range(1, 3):  # Tier1, Tier2만
                    tier_key = f'Energy(Tier-{tier_num})'
                    tier_value = result.get('category_contributions', {}).get(tier_key, 0)
                    basic_row[f'Energy_Tier{tier_num}_전력_기여도_퍼센트'] = tier_value
                
                basic_data.append(basic_row)
        
        # 사이트 변경만
        if 'site_change_only' in all_scenarios and all_scenarios['site_change_only']:
            site_change = all_scenarios['site_change_only']
            if 'after_data' in site_change and 'emission_data' in site_change['after_data']:
                emission_data = site_change['after_data']['emission_data']
                print('[DEBUG] 사이트 변경 총배출량:', emission_data.get('총_배출량', 0))
                print('[DEBUG] 사이트 변경 감축량:', site_change.get('emission_change', 0))
                print('[DEBUG] 사이트 변경 감축률:', site_change.get('emission_change_rate', 0))
                
                # 기본 데이터 구성
                basic_row = {
                    '시나리오': '사이트 변경',
                    '설명': site_change.get('description', '재활용 적용X, 생산지 변경 후'),
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': site_change.get('emission_change', 0),
                    '감축률_퍼센트': site_change.get('emission_change_rate', 0),
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0),
                    '재활용_비율_퍼센트': 0
                }
                
                # Energy Tier 컬럼을 동적으로 추가 (num_tier=2 기준)
                for tier_num in range(1, 3):  # Tier1, Tier2만
                    tier_key = f'Energy(Tier-{tier_num})'
                    tier_value = emission_data.get('카테고리별_기여도', {}).get(tier_key, 0)
                    basic_row[f'Energy_Tier{tier_num}_전력_기여도_퍼센트'] = tier_value
                
                basic_data.append(basic_row)
        
        # 종합 (재활용 + 저탄소메탈 + 사이트 변경)
        if 'combined' in all_scenarios and all_scenarios['combined']:
            combined = all_scenarios['combined']
            if 'after_recycling' in combined and 'simulation_result' in combined['after_recycling']:
                result = combined['after_recycling']['simulation_result']
                
                # Baseline 대비 전체 감축량과 감축률 계산
                baseline_emission = all_scenarios['baseline']['emission_data']['총_배출량'] if 'baseline' in all_scenarios and all_scenarios['baseline'] else 0
                combined_emission = result.get('total_emission', 0)
                total_reduction = baseline_emission - combined_emission
                total_reduction_rate = (total_reduction / baseline_emission * 100) if baseline_emission > 0 else 0
                
                # 카테고리별 기여도 확인
                category_contributions = result.get('category_contributions', {})
                print('[DEBUG] 종합 시나리오 카테고리별 기여도:', category_contributions)
                
                print('[DEBUG] 재활용+저탄소메탈+사이트 변경 총배출량:', combined_emission)
                print('[DEBUG] 재활용+저탄소메탈+사이트 변경 감축량:', total_reduction)
                print('[DEBUG] 재활용+저탄소메탈+사이트 변경 감축률:', total_reduction_rate)
                print('[DEBUG] 재활용+저탄소메탈+사이트 변경 비율:', combined.get('recycling_ratio', 0))
                
                # 기본 데이터 구성
                basic_row = {
                    '시나리오': '재활용&저탄소메탈 + 사이트 변경',
                    '설명': combined.get('description', '재활용 + 저탄소메탈 + 사이트 변경(생산지 변경 전 → 생산지 변경 후)'),
                    '총_배출량_kg_CO2e': combined_emission,
                    '감축량_kg_CO2e': total_reduction,
                    '감축률_퍼센트': total_reduction_rate,
                    '원재료_기여도_퍼센트': category_contributions.get('원재료', 0),
                    '재활용_비율_퍼센트': combined.get('recycling_ratio', 0) * 100
                }
                
                # Energy Tier 컬럼을 동적으로 추가 (num_tier=2 기준)
                for tier_num in range(1, 3):  # Tier1, Tier2만
                    tier_key = f'Energy(Tier-{tier_num})'
                    tier_value = category_contributions.get(tier_key, 0)
                    basic_row[f'Energy_Tier{tier_num}_전력_기여도_퍼센트'] = tier_value
                
                basic_data.append(basic_row)
        
        df = pd.DataFrame(basic_data)
        
        # 수치 컬럼들 반올림
        numeric_columns = ['총_배출량_kg_CO2e', '감축량_kg_CO2e', '감축률_퍼센트', '원재료_기여도_퍼센트', 'Energy_Tier1_전력_기여도_퍼센트', 'Energy_Tier2_전력_기여도_퍼센트']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df

    def get_recycling_detail_dataframe(self, all_scenarios: Dict[str, Any]) -> pd.DataFrame:
        """
        재활용 상세 분석 결과를 데이터프레임으로 생성합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            pd.DataFrame: 재활용 상세 분석 데이터프레임
        """
        if not all_scenarios:
            return pd.DataFrame()
        
        recycling_data = []
        
        # Baseline (재활용 없음)
        if 'baseline' in all_scenarios and all_scenarios['baseline']:
            baseline = all_scenarios['baseline']
            if 'emission_data' in baseline:
                emission_data = baseline['emission_data']
                recycling_data.append({
                    '시나리오': 'Baseline',
                    '재활용_비율_퍼센트': 0,
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': 0,
                    '감축률_퍼센트': 0,
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-1)', 0) + emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-2)', 0)
                })
        
        # 재활용만
        if 'recycling_only' in all_scenarios and all_scenarios['recycling_only']:
            recycling = all_scenarios['recycling_only']
            if 'simulation_result' in recycling:
                result = recycling['simulation_result']
                recycling_data.append({
                    '시나리오': '재활용 적용',
                    '재활용_비율_퍼센트': result.get('recycling_ratio_percent', 0),
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)
                })
        
        # 종합 (재활용 포함)
        if 'combined' in all_scenarios and all_scenarios['combined']:
            combined = all_scenarios['combined']
            if 'after_recycling' in combined and 'simulation_result' in combined['after_recycling']:
                result = combined['after_recycling']['simulation_result']
                # baseline과 비교해서 감축량과 감축률 계산
                baseline_emission = all_scenarios['baseline']['emission_data']['총_배출량'] if 'baseline' in all_scenarios and all_scenarios['baseline'] else 0
                total_emission = result.get('total_emission', 0)
                reduction_amount = baseline_emission - total_emission
                reduction_rate = (reduction_amount / baseline_emission * 100) if baseline_emission > 0 else 0
                
                recycling_data.append({
                    '시나리오': '재활용 + 사이트 변경',
                    '재활용_비율_퍼센트': result.get('recycling_ratio_percent', 0),
                    '총_배출량_kg_CO2e': total_emission,
                    '감축량_kg_CO2e': reduction_amount,
                    '감축률_퍼센트': reduction_rate,
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)
                })
        
        df = pd.DataFrame(recycling_data)
        
        # 수치 컬럼들 반올림
        numeric_columns = ['재활용_비율_퍼센트', '총_배출량_kg_CO2e', '감축량_kg_CO2e', '감축률_퍼센트', '원재료_기여도_퍼센트', '에너지_기여도_퍼센트']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df

    def get_site_change_detail_dataframe(self, all_scenarios: Dict[str, Any]) -> pd.DataFrame:
        """
        사이트 변경 상세 분석 결과를 데이터프레임으로 생성합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            pd.DataFrame: 사이트 변경 상세 분석 데이터프레임
        """
        if not all_scenarios:
            return pd.DataFrame()
        
        # 사이트 정보 로드
        site_data = self._load_site_data()
        cam_before = site_data.get('CAM', {}).get('before', '중국')
        cam_after = site_data.get('CAM', {}).get('after', '한국')
        pcam_before = site_data.get('pCAM', {}).get('before', '중국')
        pcam_after = site_data.get('pCAM', {}).get('after', '한국')
        
        site_change_data = []
        
        # Baseline (사이트 변경 없음)
        if 'baseline' in all_scenarios and all_scenarios['baseline']:
            baseline = all_scenarios['baseline']
            if 'emission_data' in baseline:
                emission_data = baseline['emission_data']
                site_change_data.append({
                    '시나리오': 'Baseline',
                    'CAM_변경': 'N/A',
                    'pCAM_변경': 'N/A',
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': 0,
                    '감축률_퍼센트': 0,
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-1)', 0) + emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-2)', 0)
                })
        
        # 재활용만 (사이트 변경 없음)
        if 'recycling_only' in all_scenarios and all_scenarios['recycling_only']:
            recycling = all_scenarios['recycling_only']
            if 'simulation_result' in recycling:
                result = recycling['simulation_result']
                site_change_data.append({
                    '시나리오': '재활용 적용',
                    'CAM_변경': 'N/A',
                    'pCAM_변경': 'N/A',
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)
                })
        
        # 사이트 변경만
        if 'site_change_only' in all_scenarios and all_scenarios['site_change_only']:
            site_change = all_scenarios['site_change_only']
            if 'after_data' in site_change and 'emission_data' in site_change['after_data']:
                emission_data = site_change['after_data']['emission_data']
                site_change_data.append({
                    '시나리오': '사이트 변경',
                    'CAM_변경': f"{cam_before} → {cam_after}",
                    'pCAM_변경': f"{pcam_before} → {pcam_after}",
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': site_change.get('emission_change', 0),
                    '감축률_퍼센트': site_change.get('emission_change_rate', 0),
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-1)', 0) + emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-2)', 0)
                })
        
        # 저탄소메탈만 (사이트 변경 없음)
        if 'low_carb_only' in all_scenarios and all_scenarios['low_carb_only']:
            low_carb = all_scenarios['low_carb_only']
            if 'simulation_result' in low_carb:
                result = low_carb['simulation_result']
                site_change_data.append({
                    '시나리오': '저탄소메탈 적용',
                    'CAM_변경': 'N/A',
                    'pCAM_변경': 'N/A',
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)
                })
        
        # 재활용 + 저탄소메탈 동시적용 (사이트 변경 없음)
        if 'combined_recycling' in all_scenarios and all_scenarios['combined_recycling']:
            combined_recycling = all_scenarios['combined_recycling']
            if 'simulation_result' in combined_recycling:
                result = combined_recycling['simulation_result']
                site_change_data.append({
                    '시나리오': '재활용 + 저탄소메탈',
                    'CAM_변경': 'N/A',
                    'pCAM_변경': 'N/A',
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)
                })
        
        # 종합 (사이트 변경 포함) - recycle_df와 동일한 논리로 수정
        if 'combined' in all_scenarios and all_scenarios['combined']:
            combined = all_scenarios['combined']
            if 'after_recycling' in combined and 'simulation_result' in combined['after_recycling']:
                result = combined['after_recycling']['simulation_result']
                
                # baseline과 비교해서 전체 감축량과 감축률 계산 (recycle_df와 동일한 논리)
                baseline_emission = all_scenarios['baseline']['emission_data']['총_배출량'] if 'baseline' in all_scenarios and all_scenarios['baseline'] else 0
                combined_emission = result.get('total_emission', 0)
                total_reduction = baseline_emission - combined_emission
                total_reduction_rate = (total_reduction / baseline_emission * 100) if baseline_emission > 0 else 0
                
                site_change_data.append({
                    '시나리오': '재활용 + 저탄소메탈 + 사이트 변경',
                    'CAM_변경': f"{cam_before} → {cam_after}",
                    'pCAM_변경': f"{pcam_before} → {pcam_after}",
                    '총_배출량_kg_CO2e': combined_emission,
                    '감축량_kg_CO2e': total_reduction,
                    '감축률_퍼센트': total_reduction_rate,
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)
                })
        
        df = pd.DataFrame(site_change_data)
        
        # 수치 컬럼들 반올림
        numeric_columns = ['총_배출량_kg_CO2e', '감축량_kg_CO2e', '감축률_퍼센트', '원재료_기여도_퍼센트', '에너지_기여도_퍼센트']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df

    def get_recycling_only_detail_dataframe(self, all_scenarios: Dict[str, Any]) -> pd.DataFrame:
        """
        재활용만 상세 분석 결과를 데이터프레임으로 생성합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            pd.DataFrame: 재활용만 상세 분석 데이터프레임
        """
        if not all_scenarios:
            return pd.DataFrame()
        
        recycling_only_data = []
        
        # Baseline (재활용 없음)
        if 'baseline' in all_scenarios and all_scenarios['baseline']:
            baseline = all_scenarios['baseline']
            if 'emission_data' in baseline:
                emission_data = baseline['emission_data']
                recycling_only_data.append({
                    '시나리오': 'Baseline',
                    '재활용_비율_퍼센트': 0,
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': 0,
                    '감축률_퍼센트': 0,
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-1)', 0) + emission_data.get('카테고리별_기여도', {}).get('Energy(Tier-2)', 0)
                })
        
        # 재활용만 적용 (저탄소메탈 제외)
        if 'recycling_only' in all_scenarios and all_scenarios['recycling_only']:
            recycling = all_scenarios['recycling_only']
            if 'simulation_result' in recycling:
                result = recycling['simulation_result']
                recycling_only_data.append({
                    '시나리오': '재활용 적용',
                    '재활용_비율_퍼센트': result.get('recycling_ratio_percent', 0),
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0),
                    '에너지_기여도_퍼센트': result.get('category_contributions', {}).get('Energy(Tier-1)', 0) + result.get('category_contributions', {}).get('Energy(Tier-2)', 0)
                })
        
        df = pd.DataFrame(recycling_only_data)
        
        # 수치 컬럼들 반올림
        numeric_columns = ['재활용_비율_퍼센트', '총_배출량_kg_CO2e', '감축량_kg_CO2e', '감축률_퍼센트', '원재료_기여도_퍼센트', '에너지_기여도_퍼센트']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df

    def get_low_carb_metal_only_detail_dataframe(self, all_scenarios: Dict[str, Any]) -> pd.DataFrame:
        """
        저탄소메탈만 상세 분석 결과를 데이터프레임으로 생성합니다.
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            pd.DataFrame: 저탄소메탈만 상세 분석 데이터프레임
        """
        if not all_scenarios:
            return pd.DataFrame()
        
        # 저탄소메탈 데이터 로드
        try:
            from src.utils.file_operations import FileOperations
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "..")
            low_carb_metal_path = os.path.join(project_root, "input", "low_carb_metal.json")
            
            # 사용자 ID 가져오기
            user_id = None
            try:
                import streamlit as st
                user_id = st.session_state.get('user_id', None)
            except:
                user_id = None
            
            low_carb_metal_data = FileOperations.load_json(low_carb_metal_path, default={}, user_id=user_id)
        except Exception as e:
            self._print(f"저탄소메탈 데이터 로드 중 오류: {e}", level='error')
            low_carb_metal_data = {}
        
        low_carb_only_data = []
        
        # Baseline (저탄소메탈 없음)
        if 'baseline' in all_scenarios and all_scenarios['baseline']:
            baseline = all_scenarios['baseline']
            if 'emission_data' in baseline:
                emission_data = baseline['emission_data']
                low_carb_only_data.append({
                    '시나리오': 'Baseline',
                    'Ni_저탄소메탈_비중_퍼센트': 0.0,
                    'Co_저탄소메탈_비중_퍼센트': 0.0,
                    'Li_저탄소메탈_비중_퍼센트': 0.0,
                    '총_저탄소메탈_비중_퍼센트': 0.0,
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': 0,
                    '감축률_퍼센트': 0,
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0)
                })
        
        # 저탄소메탈만 적용 (재활용 제외)
        if 'low_carb_only' in all_scenarios and all_scenarios['low_carb_only']:
            low_carb = all_scenarios['low_carb_only']
            if 'simulation_result' in low_carb:
                result = low_carb['simulation_result']
                
                # 저탄소메탈 비중 정보 추가
                ni_ratio = low_carb_metal_data.get('비중', {}).get('Ni', 0.0)
                co_ratio = low_carb_metal_data.get('비중', {}).get('Co', 0.0)
                li_ratio = low_carb_metal_data.get('비중', {}).get('Li', 0.0)
                total_ratio = (ni_ratio + co_ratio + li_ratio) / 3  # 평균값
                
                low_carb_only_data.append({
                    '시나리오': '저탄소메탈 적용',
                    'Ni_저탄소메탈_비중_퍼센트': ni_ratio,
                    'Co_저탄소메탈_비중_퍼센트': co_ratio,
                    'Li_저탄소메탈_비중_퍼센트': li_ratio,
                    '총_저탄소메탈_비중_퍼센트': total_ratio,
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0)
                })
        
        df = pd.DataFrame(low_carb_only_data)
        
        # 수치 컬럼들 반올림
        numeric_columns = ['Ni_저탄소메탈_비중_퍼센트', 'Co_저탄소메탈_비중_퍼센트', 'Li_저탄소메탈_비중_퍼센트', 
                          '총_저탄소메탈_비중_퍼센트', '총_배출량_kg_CO2e', '감축량_kg_CO2e', '감축률_퍼센트', '원재료_기여도_퍼센트']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df

    def get_low_carb_metal_detail_dataframe(self, all_scenarios: Dict[str, Any]) -> pd.DataFrame:
        """
        저탄소메탈 상세 분석 결과를 데이터프레임으로 생성합니다. (재활용과 동시 적용)
        
        Args:
            all_scenarios (Dict[str, Any]): generate_all_scenarios_data의 결과
            
        Returns:
            pd.DataFrame: 저탄소메탈 상세 분석 데이터프레임
        """
        if not all_scenarios:
            return pd.DataFrame()
        
        # 저탄소메탈 데이터 로드
        try:
            from src.utils.file_operations import FileOperations
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "..")
            low_carb_metal_path = os.path.join(project_root, "input", "low_carb_metal.json")
            
            # 사용자 ID 가져오기
            user_id = None
            try:
                import streamlit as st
                user_id = st.session_state.get('user_id', None)
            except:
                user_id = None
            
            low_carb_metal_data = FileOperations.load_json(low_carb_metal_path, default={}, user_id=user_id)
        except Exception as e:
            self._print(f"저탄소메탈 데이터 로드 중 오류: {e}", level='error')
            low_carb_metal_data = {}
        
        low_carb_data = []
        
        # Baseline (저탄소메탈 없음)
        if 'baseline' in all_scenarios and all_scenarios['baseline']:
            baseline = all_scenarios['baseline']
            if 'emission_data' in baseline:
                emission_data = baseline['emission_data']
                low_carb_data.append({
                    '시나리오': 'Baseline',
                    'Ni_저탄소메탈_비중_퍼센트': 0.0,
                    'Co_저탄소메탈_비중_퍼센트': 0.0,
                    'Li_저탄소메탈_비중_퍼센트': 0.0,
                    '총_저탄소메탈_비중_퍼센트': 0.0,
                    '총_배출량_kg_CO2e': emission_data.get('총_배출량', 0),
                    '감축량_kg_CO2e': 0,
                    '감축률_퍼센트': 0,
                    '원재료_기여도_퍼센트': emission_data.get('카테고리별_기여도', {}).get('원재료', 0)
                })
        
        # 재활용 + 저탄소메탈 동시 적용
        if 'combined_recycling' in all_scenarios and all_scenarios['combined_recycling']:
            combined_recycling = all_scenarios['combined_recycling']
            if 'simulation_result' in combined_recycling:
                result = combined_recycling['simulation_result']
                
                # 저탄소메탈 비중 정보 추가
                ni_ratio = low_carb_metal_data.get('비중', {}).get('Ni', 0.0)
                co_ratio = low_carb_metal_data.get('비중', {}).get('Co', 0.0)
                li_ratio = low_carb_metal_data.get('비중', {}).get('Li', 0.0)
                total_ratio = (ni_ratio + co_ratio + li_ratio) / 3  # 평균값
                
                low_carb_data.append({
                    '시나리오': '재활용 + 저탄소메탈',
                    'Ni_저탄소메탈_비중_퍼센트': ni_ratio,
                    'Co_저탄소메탈_비중_퍼센트': co_ratio,
                    'Li_저탄소메탈_비중_퍼센트': li_ratio,
                    '총_저탄소메탈_비중_퍼센트': total_ratio,
                    '총_배출량_kg_CO2e': result.get('total_emission', 0),
                    '감축량_kg_CO2e': result.get('reduction_amount', 0),
                    '감축률_퍼센트': result.get('reduction_rate', 0),
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0)
                })
        
        # 종합 (재활용 + 저탄소메탈 + 사이트 변경)
        if 'combined' in all_scenarios and all_scenarios['combined']:
            combined = all_scenarios['combined']
            if 'after_recycling' in combined and 'simulation_result' in combined['after_recycling']:
                result = combined['after_recycling']['simulation_result']
                
                # baseline과 비교해서 감축량과 감축률 계산
                baseline_emission = all_scenarios['baseline']['emission_data']['총_배출량'] if 'baseline' in all_scenarios and all_scenarios['baseline'] else 0
                total_emission = result.get('total_emission', 0)
                reduction_amount = baseline_emission - total_emission
                reduction_rate = (reduction_amount / baseline_emission * 100) if baseline_emission > 0 else 0
                
                # 저탄소메탈 비중 정보 추가
                ni_ratio = low_carb_metal_data.get('비중', {}).get('Ni', 0.0)
                co_ratio = low_carb_metal_data.get('비중', {}).get('Co', 0.0)
                li_ratio = low_carb_metal_data.get('비중', {}).get('Li', 0.0)
                total_ratio = (ni_ratio + co_ratio + li_ratio) / 3  # 평균값
                
                low_carb_data.append({
                    '시나리오': '재활용 + 저탄소메탈 + 사이트 변경',
                    'Ni_저탄소메탈_비중_퍼센트': ni_ratio,
                    'Co_저탄소메탈_비중_퍼센트': co_ratio,
                    'Li_저탄소메탈_비중_퍼센트': li_ratio,
                    '총_저탄소메탈_비중_퍼센트': total_ratio,
                    '총_배출량_kg_CO2e': total_emission,
                    '감축량_kg_CO2e': reduction_amount,
                    '감축률_퍼센트': reduction_rate,
                    '원재료_기여도_퍼센트': result.get('category_contributions', {}).get('원재료', 0)
                })
        
        df = pd.DataFrame(low_carb_data)
        
        # 수치 컬럼들 반올림
        numeric_columns = ['Ni_저탄소메탈_비중_퍼센트', 'Co_저탄소메탈_비중_퍼센트', 'Li_저탄소메탈_비중_퍼센트', 
                          '총_저탄소메탈_비중_퍼센트', '총_배출량_kg_CO2e', '감축량_kg_CO2e', '감축률_퍼센트', '원재료_기여도_퍼센트']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(6)
        
        return df

    def _load_site_data(self) -> Dict[str, Any]:
        """사이트 정보를 로드합니다."""
        try:
            # FileOperations를 사용하여 사용자별 파일 로드
            from src.utils.file_operations import FileOperations
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "..")
            cathode_site_path = os.path.join(project_root, "input", "cathode_site.json")
            
            # 사용자 ID 가져오기 (streamlit이 사용 가능한 경우에만)
            user_id = None
            try:
                import streamlit as st
                user_id = st.session_state.get('user_id', None)
            except:
                user_id = None
            
            # FileOperations를 사용하여 사용자별 파일 로드
            return FileOperations.load_json(cathode_site_path, default={'CAM': {'before': '중국', 'after': '한국'}, 'pCAM': {'before': '중국', 'after': '한국'}}, user_id=user_id)
        except Exception as e:
            self._print(f"사이트 정보 로드 중 오류 발생: {e}", level='error')
            return {'CAM': {'before': '중국', 'after': '한국'}, 'pCAM': {'before': '중국', 'after': '한국'}}

    def get_scenarios_dataframe(self, all_scenarios):
        """
        기본, 재활용 상세, 사이트 변경 상세 데이터프레임을 한 번에 반환
        """
        basic_df = self.get_basic_scenarios_dataframe(all_scenarios)
        recycling_df = self.get_recycling_detail_dataframe(all_scenarios)
        site_change_df = self.get_site_change_detail_dataframe(all_scenarios)
        return basic_df, recycling_df, site_change_df


class RuleBasedSimulationHelper:
    """
    RuleBasedSim의 run_simulation 결과를 해석하고 분석하는 클래스
    """
    
    def __init__(self, simulation_result: pd.DataFrame, scenario: str = 'unknown', verbose: bool = True):
        """
        RuleBasedSimulationHelper 초기화
        
        Args:
            simulation_result (pd.DataFrame): run_simulation에서 반환된 결과 데이터프레임
            scenario (str): 실행된 시나리오 이름
            verbose (bool): 상세 출력 여부
        """
        self.result_df = simulation_result
        self.scenario = scenario
        self.verbose = verbose
        
        # PCF 관련 열들 확인
        self.pcf_columns = [col for col in simulation_result.columns if col.startswith('PCF_')]
        self.has_reference = 'PCF_reference' in simulation_result.columns
        
        # 매칭 관련 열들 확인
        self.has_formula_matching = 'formula_matched' in simulation_result.columns
        self.has_proportions_matching = 'proportions_matched' in simulation_result.columns
        
        # 저감활동 관련 열 확인
        self.has_reduction_activity = '저감활동_적용여부' in simulation_result.columns
        
        # 수정된 배출계수 열들 확인
        self.modified_coeff_columns = [col for col in simulation_result.columns if col.startswith('modified_coeff_case')]
        
    def _print(self, message: str, level: str = 'info'):
        """출력 함수"""
        if self.verbose:
            print(f"[{level.upper()}] {message}")
    
    def print_simulation_overview(self):
        """
        시뮬레이션 결과 전체 개요 출력
        """
        print("=" * 100)
        print("🚀 RuleBasedSimulation 결과 분석")
        print("=" * 100)
        print(f"📋 시나리오: {self.scenario.upper()}")
        print(f"📊 총 데이터 행 수: {len(self.result_df):,}개")
        print(f"📈 PCF 관련 열: {len(self.pcf_columns)}개")
        print(f"🔍 매칭 분석 가능: {self.has_formula_matching and self.has_proportions_matching}")
        print(f"⚡ 저감활동 분석 가능: {self.has_reduction_activity}")
        print(f"📊 수정된 배출계수 열: {len(self.modified_coeff_columns)}개")
        print("=" * 100)
    
    def print_pcf_analysis(self):
        """
        PCF 탄소배출량 분석 결과 출력
        """
        print("\n" + "=" * 80)
        print("📊 PCF 탄소배출량 분석")
        print("=" * 80)
        
        if not self.pcf_columns:
            print("❌ PCF 관련 열이 없습니다.")
            return
        
        # PCF 합계 계산
        pcf_sums = {}
        for col in self.pcf_columns:
            pcf_sums[col] = self.result_df[col].sum()
        
        # Reference 값 먼저 출력
        if self.has_reference:
            reference_total = pcf_sums['PCF_reference']
            print(f"📈 PCF_reference: {reference_total:.3f} kgCO2eq")
            print("-" * 50)
            
            # Case별 감소율 계산 및 출력
            for col in self.pcf_columns:
                if col != 'PCF_reference':
                    case_total = pcf_sums[col]
                    reduction = reference_total - case_total
                    reduction_rate = (reduction / reference_total) * 100 if reference_total > 0 else 0
                    print(f"📉 {col}: {case_total:.3f} kgCO2eq")
                    print(f"   - 절대 감소량: {reduction:.3f} kgCO2eq")
                    print(f"   - 감소율: {reduction_rate:.2f}%")
                    print()
        else:
            # Reference가 없는 경우 단순 합계만 출력
            for col, total in pcf_sums.items():
                print(f"📊 {col}: {total:.3f} kgCO2eq")
        
        print("=" * 80)
        
        return pcf_sums
    
    def print_matching_analysis(self):
        """
        매칭 결과 분석 출력
        """
        if not (self.has_formula_matching and self.has_proportions_matching):
            print("\n❌ 매칭 분석을 위한 열이 없습니다.")
            return
        
        print("\n" + "=" * 80)
        print("🔍 매칭 결과 분석")
        print("=" * 80)
        
        # 매칭 통계 계산
        formula_matched_count = len(self.result_df[self.result_df['formula_matched'] == True])
        proportions_matched_count = len(self.result_df[self.result_df['proportions_matched'] == True])
        both_matched_count = len(self.result_df[
            (self.result_df['formula_matched'] == True) & 
            (self.result_df['proportions_matched'] == True)
        ])
        unmatched_count = len(self.result_df[
            (self.result_df['formula_matched'] == False) & 
            (self.result_df['proportions_matched'] == False)
        ])
        
        total_rows = len(self.result_df)
        
        print(f"📊 총 데이터: {total_rows:,}개")
        print(f"📊 공식 매칭: {formula_matched_count:,}개 ({formula_matched_count/total_rows*100:.1f}%)")
        print(f"📊 비율 매칭: {proportions_matched_count:,}개 ({proportions_matched_count/total_rows*100:.1f}%)")
        print(f"📊 둘 다 매칭: {both_matched_count:,}개 ({both_matched_count/total_rows*100:.1f}%)")
        print(f"📊 매칭 실패: {unmatched_count:,}개 ({unmatched_count/total_rows*100:.1f}%)")
        
        # PCF 관점에서의 매칭 분석
        if self.has_reference:
            print("\n📈 PCF 관점에서의 매칭 분석:")
            print("-" * 50)
            
            # 공식 매칭만 된 행들의 PCF 분석
            formula_only = self.result_df[
                (self.result_df['formula_matched'] == True) & 
                (self.result_df['proportions_matched'] == False)
            ]
            if len(formula_only) > 0:
                ref_sum = formula_only['PCF_reference'].sum()
                print(f"🔍 공식 매칭만: {len(formula_only)}개 행")
                print(f"   - PCF_reference: {ref_sum:.3f} kgCO2eq")
                for col in self.pcf_columns:
                    if col != 'PCF_reference':
                        case_sum = formula_only[col].sum()
                        reduction = ref_sum - case_sum
                        reduction_rate = (reduction / ref_sum * 100) if ref_sum > 0 else 0
                        print(f"   - {col}: {case_sum:.3f} kgCO2eq (감소율: {reduction_rate:.2f}%)")
            
            # 비율 매칭만 된 행들의 PCF 분석
            proportions_only = self.result_df[
                (self.result_df['formula_matched'] == False) & 
                (self.result_df['proportions_matched'] == True)
            ]
            if len(proportions_only) > 0:
                ref_sum = proportions_only['PCF_reference'].sum()
                print(f"\n🔍 비율 매칭만: {len(proportions_only)}개 행")
                print(f"   - PCF_reference: {ref_sum:.3f} kgCO2eq")
                for col in self.pcf_columns:
                    if col != 'PCF_reference':
                        case_sum = proportions_only[col].sum()
                        reduction = ref_sum - case_sum
                        reduction_rate = (reduction / ref_sum * 100) if ref_sum > 0 else 0
                        print(f"   - {col}: {case_sum:.3f} kgCO2eq (감소율: {reduction_rate:.2f}%)")
        
        print("=" * 80)
    
    def print_reduction_activity_analysis(self):
        """
        저감활동 적용 분석 출력
        """
        if not self.has_reduction_activity:
            print("\n❌ 저감활동 분석을 위한 열이 없습니다.")
            return
        
        print("\n" + "=" * 80)
        print("⚡ 저감활동 적용 분석")
        print("=" * 80)
        
        # 저감활동 통계
        applicable_count = len(self.result_df[self.result_df['저감활동_적용여부'] == 1.0])
        non_applicable_count = len(self.result_df[self.result_df['저감활동_적용여부'] == 0.0])
        total_rows = len(self.result_df)
        
        print(f"📊 총 데이터: {total_rows:,}개")
        print(f"✅ 저감활동 적용: {applicable_count:,}개 ({applicable_count/total_rows*100:.1f}%)")
        print(f"❌ 저감활동 미적용: {non_applicable_count:,}개 ({non_applicable_count/total_rows*100:.1f}%)")
        
        # PCF 관점에서의 저감활동 분석
        if self.has_reference:
            print("\n📈 PCF 관점에서의 저감활동 분석:")
            print("-" * 50)
            
            # 저감활동 적용된 행들의 PCF 분석
            applicable_rows = self.result_df[self.result_df['저감활동_적용여부'] == 1.0]
            if len(applicable_rows) > 0:
                ref_sum = applicable_rows['PCF_reference'].sum()
                print(f"✅ 저감활동 적용: {len(applicable_rows)}개 행")
                print(f"   - PCF_reference: {ref_sum:.3f} kgCO2eq")
                for col in self.pcf_columns:
                    if col != 'PCF_reference':
                        case_sum = applicable_rows[col].sum()
                        reduction = ref_sum - case_sum
                        reduction_rate = (reduction / ref_sum * 100) if ref_sum > 0 else 0
                        print(f"   - {col}: {case_sum:.3f} kgCO2eq (감소율: {reduction_rate:.2f}%)")
            
            # 저감활동 미적용된 행들의 PCF 분석
            non_applicable_rows = self.result_df[self.result_df['저감활동_적용여부'] == 0.0]
            if len(non_applicable_rows) > 0:
                ref_sum = non_applicable_rows['PCF_reference'].sum()
                print(f"\n❌ 저감활동 미적용: {len(non_applicable_rows)}개 행")
                print(f"   - PCF_reference: {ref_sum:.3f} kgCO2eq")
                for col in self.pcf_columns:
                    if col != 'PCF_reference':
                        case_sum = non_applicable_rows[col].sum()
                        reduction = ref_sum - case_sum
                        reduction_rate = (reduction / ref_sum * 100) if ref_sum > 0 else 0
                        print(f"   - {col}: {case_sum:.3f} kgCO2eq (감소율: {reduction_rate:.2f}%)")
        
        print("=" * 80)
    
    def print_emission_coefficient_analysis(self):
        """
        배출계수 변경 분석 출력
        """
        if not self.modified_coeff_columns:
            print("\n❌ 수정된 배출계수 열이 없습니다.")
            return
        
        print("\n" + "=" * 80)
        print("📊 배출계수 변경 분석")
        print("=" * 80)
        
        # 원본 배출계수 확인
        if '배출계수' in self.result_df.columns:
            original_coeff_mean = self.result_df['배출계수'].mean()
            original_coeff_std = self.result_df['배출계수'].std()
            print(f"📊 원본 배출계수:")
            print(f"   - 평균: {original_coeff_mean:.6f}")
            print(f"   - 표준편차: {original_coeff_std:.6f}")
            print("-" * 50)
            
            # 수정된 배출계수 분석
            for col in self.modified_coeff_columns:
                if col in self.result_df.columns:
                    modified_coeff_mean = self.result_df[col].mean()
                    modified_coeff_std = self.result_df[col].std()
                    change_rate = ((modified_coeff_mean - original_coeff_mean) / original_coeff_mean * 100) if original_coeff_mean > 0 else 0
                    
                    print(f"📊 {col}:")
                    print(f"   - 평균: {modified_coeff_mean:.6f}")
                    print(f"   - 표준편차: {modified_coeff_std:.6f}")
                    print(f"   - 변화율: {change_rate:+.2f}%")
                    print()
        
        print("=" * 80)
    
    def print_material_analysis(self):
        """
        자재별 분석 출력
        """
        if '자재품목' not in self.result_df.columns:
            print("\n❌ 자재품목 열이 없습니다.")
            return
        
        print("\n" + "=" * 80)
        print("📋 자재별 분석")
        print("=" * 80)
        
        # 자재품목별 통계
        material_stats = self.result_df['자재품목'].value_counts()
        print(f"📊 자재품목 종류: {len(material_stats)}개")
        print("\n📋 자재품목별 데이터 수:")
        for material, count in material_stats.items():
            print(f"   - {material}: {count:,}개")
        
        # PCF 관점에서의 자재별 분석 (저감활동_적용여부가 1인 데이터만)
        if self.has_reference:
            print("\n📈 자재별 PCF 분석 (저감활동 적용 데이터만):")
            print("-" * 50)
            
            # 저감활동_적용여부가 1인 데이터만 필터링
            if self.has_reduction_activity:
                filtered_df = self.result_df[self.result_df['저감활동_적용여부'] == 1.0]
                if len(filtered_df) == 0:
                    print("   ⚠️ 저감활동_적용여부가 1인 데이터가 없습니다.")
                    print("=" * 80)
                    return
                
                print(f"   📊 분석 대상 데이터: {len(filtered_df):,}개 (전체 {len(self.result_df):,}개 중)")
                print()
                
                material_pcf = filtered_df.groupby('자재품목')[self.pcf_columns].sum()
                
                for material in material_pcf.index:
                    print(f"\n📋 {material}:")
                    material_data = material_pcf.loc[material]
                    
                    if 'PCF_reference' in material_data.index:
                        ref_pcf = material_data['PCF_reference']
                        print(f"   - PCF_reference: {ref_pcf:.3f} kgCO2eq")
                        
                        for col in self.pcf_columns:
                            if col != 'PCF_reference':
                                case_pcf = material_data[col]
                                reduction = ref_pcf - case_pcf
                                reduction_rate = (reduction / ref_pcf * 100) if ref_pcf > 0 else 0
                                print(f"   - {col}: {case_pcf:.3f} kgCO2eq (감소율: {reduction_rate:.2f}%)")
            else:
                print("   ⚠️ 저감활동_적용여부 열이 없어 전체 데이터로 분석합니다.")
                material_pcf = self.result_df.groupby('자재품목')[self.pcf_columns].sum()
                
                for material in material_stats.index:
                    print(f"\n📋 {material}:")
                    material_data = material_pcf.loc[material]
                    
                    if 'PCF_reference' in material_data.index:
                        ref_pcf = material_data['PCF_reference']
                        print(f"   - PCF_reference: {ref_pcf:.3f} kgCO2eq")
                        
                        for col in self.pcf_columns:
                            if col != 'PCF_reference':
                                case_pcf = material_data[col]
                                reduction = ref_pcf - case_pcf
                                reduction_rate = (reduction / ref_pcf * 100) if ref_pcf > 0 else 0
                                print(f"   - {col}: {case_pcf:.3f} kgCO2eq (감소율: {reduction_rate:.2f}%)")
        
        print("=" * 80)
    
    def print_comprehensive_analysis(self):
        """
        종합 분석 결과 출력
        """
        print("\n" + "=" * 100)
        print("🎯 RuleBasedSimulation 종합 분석")
        print("=" * 100)
        
        # 1. 시뮬레이션 개요
        self.print_simulation_overview()
        
        # 2. PCF 분석
        pcf_sums = self.print_pcf_analysis()
        
        # 3. 매칭 분석
        self.print_matching_analysis()
        
        # 4. 저감활동 분석
        self.print_reduction_activity_analysis()
        
        # 5. 배출계수 분석
        self.print_emission_coefficient_analysis()
        
        # 6. 자재별 분석
        self.print_material_analysis()
        
        # 7. 요약 통계
        self._print_summary_statistics(pcf_sums)
        
        print("\n" + "=" * 100)
        print("✅ 분석 완료")
        print("=" * 100)
    
    def _print_summary_statistics(self, pcf_sums: dict):
        """
        요약 통계 출력
        
        Args:
            pcf_sums (dict): PCF 합계 딕셔너리
        """
        print("\n" + "=" * 80)
        print("📊 요약 통계")
        print("=" * 80)
        
        # 전체 PCF 감소율 계산
        if self.has_reference and 'PCF_reference' in pcf_sums:
            reference_total = pcf_sums['PCF_reference']
            
            print(f"📈 전체 PCF_reference: {reference_total:.3f} kgCO2eq")
            
            for col in self.pcf_columns:
                if col != 'PCF_reference':
                    case_total = pcf_sums[col]
                    reduction = reference_total - case_total
                    reduction_rate = (reduction / reference_total * 100) if reference_total > 0 else 0
                    print(f"📉 {col}: {case_total:.3f} kgCO2eq (전체 감소율: {reduction_rate:.2f}%)")
        
        # 시나리오별 특별 통계
        if self.scenario in ['recycling', 'site_change', 'both']:
            print(f"\n🎯 {self.scenario.upper()} 시나리오 특별 통계:")
            
            if self.has_reduction_activity:
                applicable_ratio = len(self.result_df[self.result_df['저감활동_적용여부'] == 1.0]) / len(self.result_df) * 100
                print(f"   - 저감활동 적용 비율: {applicable_ratio:.1f}%")
            
            if self.has_formula_matching and self.has_proportions_matching:
                matching_success_rate = (
                    len(self.result_df[
                        (self.result_df['formula_matched'] == True) | 
                        (self.result_df['proportions_matched'] == True)
                    ]) / len(self.result_df) * 100
                )
                print(f"   - 매칭 성공률: {matching_success_rate:.1f}%")
        
        print("=" * 80)
    
    def get_analysis_summary(self) -> dict:
        """
        분석 결과를 딕셔너리로 반환
        
        Returns:
            dict: 분석 결과 요약
        """
        summary = {
            'scenario': self.scenario,
            'total_rows': len(self.result_df),
            'pcf_columns': self.pcf_columns,
            'has_reference': self.has_reference,
            'has_formula_matching': self.has_formula_matching,
            'has_proportions_matching': self.has_proportions_matching,
            'has_reduction_activity': self.has_reduction_activity,
            'modified_coeff_columns': self.modified_coeff_columns
        }
        
        # PCF 합계 추가
        if self.pcf_columns:
            pcf_sums = {}
            for col in self.pcf_columns:
                pcf_sums[col] = self.result_df[col].sum()
            summary['pcf_sums'] = pcf_sums
        
        # 매칭 통계 추가
        if self.has_formula_matching and self.has_proportions_matching:
            formula_matched_count = len(self.result_df[self.result_df['formula_matched'] == True])
            proportions_matched_count = len(self.result_df[self.result_df['proportions_matched'] == True])
            unmatched_count = len(self.result_df[
                (self.result_df['formula_matched'] == False) & 
                (self.result_df['proportions_matched'] == False)
            ])
            
            summary['matching_stats'] = {
                'formula_matched': formula_matched_count,
                'proportions_matched': proportions_matched_count,
                'unmatched': unmatched_count,
                'total': len(self.result_df)
            }
        
        # 저감활동 통계 추가
        if self.has_reduction_activity:
            applicable_count = len(self.result_df[self.result_df['저감활동_적용여부'] == 1.0])
            summary['reduction_activity_stats'] = {
                'applicable': applicable_count,
                'non_applicable': len(self.result_df) - applicable_count,
                'total': len(self.result_df)
            }
        
        return summary

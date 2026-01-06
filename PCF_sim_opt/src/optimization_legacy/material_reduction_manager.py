"""
자재별 감축 활동 관리 모듈

이 모듈은 자재의 특성에 따라 다양한 감축 활동을 적용하고 관리합니다.
각 자재 유형(양극재 Formula, 양극재 Proportion, 일반 자재)에 맞는 
감축 활동을 정의하고 배출계수를 계산합니다.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pathlib import Path
import os
import json


class MaterialReductionManager:
    """
    자재별 감축 활동 관리 클래스
    
    자재의 유형과 특성에 맞는 감축 활동을 관리하고 
    배출계수 수정 로직을 구현합니다.
    """
    
    def __init__(self, 
                 simulation_data: Dict[str, pd.DataFrame] = None,
                 stable_var_dir: str = "stable_var",
                 user_id: Optional[str] = None,
                 debug_mode: bool = True):
        """
        MaterialReductionManager 초기화
        
        Args:
            simulation_data: 시뮬레이션 데이터 (시나리오 및 참조 데이터프레임)
            stable_var_dir: stable_var 디렉토리 경로
            user_id: 사용자 ID (사용자별 데이터 사용시)
            debug_mode: 디버그 모드 사용 여부
        """
        self.debug_mode = debug_mode
        self.user_id = user_id
        self.stable_var_dir = Path(stable_var_dir)
        
        # 시뮬레이션 데이터 설정
        self.simulation_data = simulation_data or {}
        self.scenario_df = self.simulation_data.get('scenario_df', pd.DataFrame())
        self.ref_formula_df = self.simulation_data.get('ref_formula_df', pd.DataFrame())
        self.ref_proportions_df = self.simulation_data.get('ref_proportions_df', pd.DataFrame())
        self.original_df = self.simulation_data.get('original_df', pd.DataFrame())
        
        # 저감활동 적용 자재 추출
        self.target_materials = self._extract_target_materials()
        
        # 자재 유형 분류
        self.material_types = self._classify_materials()
        
        # 추가 데이터 로드
        self.recycle_impact_data = self._load_recycle_impact_data()
        self.electricity_coef_data = self._load_electricity_coef_data()
        
        # 디버그 정보 출력
        if self.debug_mode:
            self._print_debug_info()
    
    def _extract_target_materials(self) -> pd.DataFrame:
        """저감활동 적용 자재 추출"""
        if len(self.scenario_df) == 0 or '저감활동_적용여부' not in self.scenario_df.columns:
            return pd.DataFrame()
        
        return self.scenario_df[self.scenario_df['저감활동_적용여부'] == 1].copy()
    
    def _classify_materials(self) -> Dict[str, Dict[str, Any]]:
        """자재 유형 분류"""
        material_types = {}
        
        if len(self.target_materials) == 0:
            return material_types
        
        for idx, row in self.target_materials.iterrows():
            material_name = row['자재명']
            material_category = row.get('자재품목', '')
            
            # 양극재 여부 확인
            is_cathode = (material_category == '양극재' or 
                         '양극재' in material_name.lower() or 
                         'cathode' in material_name.lower())
            
            # Formula 적용 여부 확인
            is_formula_applicable = self._check_formula_applicable(material_name)
            
            # Proportion 적용 여부 확인
            is_proportion_applicable = self._check_proportion_applicable(material_name)
            
            # Ni, Co, Li 자재 여부 확인
            is_ni_co_li = self._check_is_ni_co_li(material_name)
            
            # 자재 유형 설정
            material_types[material_name] = {
                'material_name': material_name,
                'material_category': material_category,
                'is_cathode': is_cathode,
                'is_formula_applicable': is_formula_applicable,
                'is_proportion_applicable': is_proportion_applicable,
                'is_ni_co_li': is_ni_co_li,
                'original_emission': row.get('배출계수', 0),
                'quantity': row.get('제품총소요량(kg)', 0)
            }
        
        return material_types
    
    def _check_formula_applicable(self, material_name: str) -> bool:
        """Formula 적용 가능한 자재인지 확인"""
        if len(self.ref_formula_df) == 0:
            return False
        
        # 자재명 완전 일치하는 경우
        exact_match = material_name in self.ref_formula_df['자재명'].values
        if exact_match:
            return True
        
        # 부분 일치하는 경우 (대소문자 무시)
        for formula_material in self.ref_formula_df['자재명'].values:
            if formula_material.lower() in material_name.lower() or material_name.lower() in formula_material.lower():
                return True
        
        return False
    
    def _check_proportion_applicable(self, material_name: str) -> bool:
        """Proportion 적용 가능한 자재인지 확인"""
        if len(self.ref_proportions_df) == 0:
            return False
        
        # 자재명(포함) 컬럼에 해당 자재가 포함되는지 확인
        for proportion_material in self.ref_proportions_df['자재명(포함)'].values:
            if isinstance(proportion_material, str) and (
                proportion_material.lower() in material_name.lower() or 
                material_name.lower() in proportion_material.lower()):
                return True
        
        return False
    
    def _check_is_ni_co_li(self, material_name: str) -> bool:
        """Ni, Co, Li 관련 자재인지 확인"""
        # 자재명에 Ni, Co, Li 관련 키워드가 포함되어 있는지 확인
        keywords = ['ni', 'co', 'li', 'nickel', 'cobalt', 'lithium', '니켈', '코발트', '리튬']
        for keyword in keywords:
            if keyword in material_name.lower():
                return True
        return False
    
    def _load_recycle_impact_data(self) -> Dict[str, Any]:
        """재활용 환경영향 데이터 로드"""
        try:
            file_path = self.stable_var_dir / "recycle_material_impact.json"
            if not os.path.exists(file_path):
                # 사용자 경로 확인
                if self.user_id:
                    file_path = self.stable_var_dir / self.user_id / "recycle_material_impact.json"
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 기본값
            return {
                "Ni": {"impact_ratio": 0.4},  # 재활용 Ni 배출계수는 신재의 40%
                "Co": {"impact_ratio": 0.4},  # 재활용 Co 배출계수는 신재의 40%
                "Li": {"impact_ratio": 0.5}   # 재활용 Li 배출계수는 신재의 50%
            }
        except Exception as e:
            print(f"Error loading recycle_material_impact.json: {e}")
            # 기본값
            return {
                "Ni": {"impact_ratio": 0.4},
                "Co": {"impact_ratio": 0.4},
                "Li": {"impact_ratio": 0.5}
            }
    
    def _load_electricity_coef_data(self) -> Dict[str, float]:
        """국가별 전력 배출계수 데이터 로드"""
        try:
            file_path = self.stable_var_dir / "electricity_coef_by_country.json"
            if not os.path.exists(file_path):
                # 사용자 경로 확인
                if self.user_id:
                    file_path = self.stable_var_dir / self.user_id / "electricity_coef_by_country.json"
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # 기본값
            return {
                "한국": 0.4526,
                "중국": 0.5448,
                "일본": 0.4239,
                "미국": 0.3861,
                "독일": 0.3384,
                "폴란드": 0.7283
            }
        except Exception as e:
            print(f"Error loading electricity_coef_by_country.json: {e}")
            # 기본값
            return {
                "한국": 0.4526,
                "중국": 0.5448,
                "일본": 0.4239,
                "미국": 0.3861,
                "독일": 0.3384,
                "폴란드": 0.7283
            }
    
    def _print_debug_info(self) -> None:
        """디버그 정보 출력"""
        print("===== MaterialReductionManager 초기화 정보 =====")
        print(f"• 총 자재 수: {len(self.scenario_df)}개")
        print(f"• 저감활동 적용 자재: {len(self.target_materials)}개")
        
        # 자재 유형별 개수
        cathode_count = sum(1 for info in self.material_types.values() if info['is_cathode'])
        formula_count = sum(1 for info in self.material_types.values() if info['is_formula_applicable'])
        proportion_count = sum(1 for info in self.material_types.values() if info['is_proportion_applicable'])
        ni_co_li_count = sum(1 for info in self.material_types.values() if info['is_ni_co_li'])
        
        print(f"• 양극재 자재: {cathode_count}개")
        print(f"• Formula 적용 자재: {formula_count}개")
        print(f"• Proportion 적용 자재: {proportion_count}개")
        print(f"• Ni, Co, Li 자재: {ni_co_li_count}개")
        
        # 자재별 상세 정보
        print("\n===== 자재별 유형 정보 =====")
        for material_name, info in self.material_types.items():
            type_desc = []
            if info['is_cathode']:
                type_desc.append("양극재")
            if info['is_formula_applicable']:
                type_desc.append("Formula적용")
            if info['is_proportion_applicable']:
                type_desc.append("Proportion적용")
            if info['is_ni_co_li']:
                type_desc.append("Ni/Co/Li")
            
            print(f"• {material_name}")
            print(f"  - 유형: {', '.join(type_desc)}")
            print(f"  - 자재품목: {info['material_category']}")
            print(f"  - 배출계수: {info['original_emission']:.4f}")
            print(f"  - 소요량: {info['quantity']:.4f} kg")
    
    def apply_reductions_to_model(self, model: pyo.ConcreteModel) -> None:
        """최적화 모델에 자재별 감축 활동 적용"""
        # 자재별 감축 로직 적용
        self._apply_formula_reductions(model)
        self._apply_proportion_reductions(model)
        self._apply_general_reductions(model)
    
    def _apply_formula_reductions(self, model: pyo.ConcreteModel) -> None:
        """Formula 적용 자재 감축 로직"""
        # Formula 적용 자재 필터링
        formula_materials = [m for m, info in self.material_types.items() if info['is_formula_applicable']]
        
        if not formula_materials:
            return
        
        # Formula 감축 계수 준비
        formula_coefs = self._prepare_formula_coefficients()
        
        # Formula 자재 배출계수 계산 제약조건 추가
        def formula_emission_rule(model, m):
            if m not in formula_materials:
                return pyo.Constraint.Skip
            
            # 티어별 RE 적용률 계산
            tier1_reduction = model.tier1_re[m] * formula_coefs.get(m, {}).get('tier1', 0.3)
            tier2_reduction = model.tier2_re[m] * formula_coefs.get(m, {}).get('tier2', 0.5)
            tier3_reduction = model.tier3_re[m] * formula_coefs.get(m, {}).get('tier3', 0.2)
            
            # 배출계수 계산 공식: 원래 배출계수 × (1 - 티어별 감축률 합계)
            return model.modified_emission[m] == model.original_emission[m] * (1 - tier1_reduction - tier2_reduction - tier3_reduction)
        
        # 제약조건 추가
        model.formula_emission_constraint = pyo.Constraint(model.materials, rule=formula_emission_rule)
    
    def _prepare_formula_coefficients(self) -> Dict[str, Dict[str, float]]:
        """Formula 자재별 감축 계수 준비"""
        coefs = {}
        
        # ref_formula_df에서 티어별 RE100 적용시 감축 계수 추출
        if len(self.ref_formula_df) > 0:
            for idx, row in self.ref_formula_df.iterrows():
                material_name = row['자재명']
                
                # 해당 material_name과 일치하는 자재 찾기
                found_material = None
                for m in self.material_types.keys():
                    if material_name.lower() in m.lower() or m.lower() in material_name.lower():
                        found_material = m
                        break
                
                if found_material:
                    tier1_coef = 0.3  # 기본값
                    tier2_coef = 0.5  # 기본값
                    tier3_coef = 0.2  # 기본값
                    
                    # Tier별 배출계수 비율 계산 (있는 경우)
                    if 'Tier1_RE100(kgCO2eq/kg)' in row and row['배출계수'] > 0:
                        tier1_coef = row['Tier1_RE100(kgCO2eq/kg)'] / row['배출계수']
                    
                    if 'Tier2_RE100(kgCO2eq/kg)' in row and row['배출계수'] > 0:
                        tier2_coef = row['Tier2_RE100(kgCO2eq/kg)'] / row['배출계수']
                    
                    if 'Tier3_RE100(kgCO2eq/kg)' in row and row['배출계수'] > 0:
                        tier3_coef = row['Tier3_RE100(kgCO2eq/kg)'] / row['배출계수']
                    
                    coefs[found_material] = {
                        'tier1': tier1_coef,
                        'tier2': tier2_coef,
                        'tier3': tier3_coef
                    }
        
        return coefs
    
    def _apply_proportion_reductions(self, model: pyo.ConcreteModel) -> None:
        """Proportion 적용 자재(Ni, Co, Li) 감축 로직"""
        # Ni, Co, Li 자재 필터링
        ni_co_li_materials = [m for m, info in self.material_types.items() if info['is_ni_co_li']]
        
        if not ni_co_li_materials:
            return
        
        # 재활용 환경영향 데이터 준비
        recycle_coefs = {}
        for material in ni_co_li_materials:
            material_lower = material.lower()
            if 'ni' in material_lower or 'nickel' in material_lower:
                recycle_coefs[material] = self.recycle_impact_data.get('Ni', {}).get('impact_ratio', 0.4)
            elif 'co' in material_lower or 'cobalt' in material_lower:
                recycle_coefs[material] = self.recycle_impact_data.get('Co', {}).get('impact_ratio', 0.4)
            elif 'li' in material_lower or 'lithium' in material_lower:
                recycle_coefs[material] = self.recycle_impact_data.get('Li', {}).get('impact_ratio', 0.5)
            else:
                recycle_coefs[material] = 0.4  # 기본값
        
        # 저탄소메탈 환경영향 계수 (기본값: 원래 배출계수의 70%)
        low_carbon_impact_ratio = 0.7
        
        # Ni, Co, Li 자재 배출계수 계산 제약조건 추가
        def ni_co_li_emission_rule(model, m):
            if m not in ni_co_li_materials:
                return pyo.Constraint.Skip
            
            # 자재별 배출계수 계산
            virgin_emission = model.original_emission[m]  # 신재 배출계수
            recycle_emission = virgin_emission * recycle_coefs.get(m, 0.4)  # 재활용 배출계수
            low_carbon_emission = virgin_emission * low_carbon_impact_ratio  # 저탄소메탈 배출계수
            
            # 최종 배출계수 계산: 각 비율에 따른 가중평균
            return model.modified_emission[m] == (
                model.virgin_ratio[m] * virgin_emission +
                model.recycle_ratio[m] * recycle_emission +
                model.low_carbon_ratio[m] * low_carbon_emission
            )
        
        # 제약조건 추가
        model.ni_co_li_emission_constraint = pyo.Constraint(model.materials, rule=ni_co_li_emission_rule)
    
    def _apply_general_reductions(self, model: pyo.ConcreteModel) -> None:
        """일반 자재 감축 로직 (Formula, Ni/Co/Li 아닌 자재)"""
        # Formula와 Ni/Co/Li가 아닌 일반 자재 필터링
        formula_materials = [m for m, info in self.material_types.items() if info['is_formula_applicable']]
        ni_co_li_materials = [m for m, info in self.material_types.items() if info['is_ni_co_li']]
        general_materials = [m for m in self.material_types.keys() 
                           if m not in formula_materials and m not in ni_co_li_materials]
        
        if not general_materials:
            return
        
        # 일반 자재 배출계수 계산 제약조건 추가
        def general_emission_rule(model, m):
            if m not in general_materials:
                return pyo.Constraint.Skip
            
            # ref_proportions_df에서 해당 자재의 티어별 비율 찾기
            tier_ratios = self._get_tier_ratios_for_material(m)
            
            # 티어별 RE 적용률 계산
            tier1_reduction = model.tier1_re[m] * tier_ratios.get('tier1', 0.3)
            tier2_reduction = model.tier2_re[m] * tier_ratios.get('tier2', 0.5)
            tier3_reduction = model.tier3_re[m] * tier_ratios.get('tier3', 0.2)
            
            # 배출계수 계산 공식: 원래 배출계수 × (1 - 티어별 감축률 합계)
            return model.modified_emission[m] == model.original_emission[m] * (1 - tier1_reduction - tier2_reduction - tier3_reduction)
        
        # 제약조건 추가
        model.general_emission_constraint = pyo.Constraint(model.materials, rule=general_emission_rule)
    
    def _get_tier_ratios_for_material(self, material_name: str) -> Dict[str, float]:
        """자재의 티어별 비율 계산"""
        # ref_proportions_df에서 해당 자재와 관련된 행 찾기
        if len(self.ref_proportions_df) == 0:
            return {'tier1': 0.3, 'tier2': 0.5, 'tier3': 0.2}  # 기본값
        
        # 자재명과 일치하는 행 찾기
        material_rows = []
        for idx, row in self.ref_proportions_df.iterrows():
            if '자재명(포함)' in row and isinstance(row['자재명(포함)'], str):
                if (row['자재명(포함)'].lower() in material_name.lower() or 
                    material_name.lower() in row['자재명(포함)'].lower()):
                    material_rows.append(row)
        
        if not material_rows:
            return {'tier1': 0.3, 'tier2': 0.5, 'tier3': 0.2}  # 기본값
        
        # 첫 번째 일치하는 행 사용
        row = material_rows[0]
        tier_ratios = {}
        
        # Tier1 RE100 비율
        if 'Tier1_RE100(%)' in row:
            tier1_ratio = row['Tier1_RE100(%)'] / 100 if row['Tier1_RE100(%)'] > 0 else 0.3
            tier_ratios['tier1'] = tier1_ratio
        else:
            tier_ratios['tier1'] = 0.3
        
        # Tier2 RE100 비율
        if 'Tier2_RE100(%)' in row:
            tier2_ratio = row['Tier2_RE100(%)'] / 100 if row['Tier2_RE100(%)'] > 0 else 0.5
            tier_ratios['tier2'] = tier2_ratio
        else:
            tier_ratios['tier2'] = 0.5
        
        # Tier3 RE100 비율
        if 'Tier3_RE100(%)' in row:
            tier3_ratio = row['Tier3_RE100(%)'] / 100 if row['Tier3_RE100(%)'] > 0 else 0.2
            tier_ratios['tier3'] = tier3_ratio
        else:
            tier_ratios['tier3'] = 0.2
        
        return tier_ratios
    
    def add_location_constraints(self, model: pyo.ConcreteModel, country_config: Dict[str, Any]) -> None:
        """국가 변경에 따른 제약조건 추가"""
        if not country_config or not country_config.get('enabled', False):
            return
        
        # 국가 변경 가능한 자재 목록
        location_materials = country_config.get('materials', [])
        if not location_materials:
            return
        
        # 국가별 전력 배출계수
        electricity_coefs = self.electricity_coef_data
        
        # 위치 변수 정의
        model.locations = pyo.Set(initialize=list(electricity_coefs.keys()))
        model.location_selected = pyo.Var(model.materials, model.locations, domain=pyo.Binary, initialize=0)
        
        # 각 자재는 하나의 국가만 선택 가능
        def single_location_rule(model, m):
            if m not in location_materials:
                return pyo.Constraint.Skip
            return sum(model.location_selected[m, loc] for loc in model.locations) == 1
        
        model.single_location_constraint = pyo.Constraint(model.materials, rule=single_location_rule)
        
        # 국가별 기본 설정 (현재 위치)
        for m in location_materials:
            # 해당 자재의 현재 국가 설정
            current_country = self._get_current_country_for_material(m)
            
            # 현재 국가가 위치 목록에 없으면 기본값 '중국' 설정
            if current_country not in electricity_coefs:
                current_country = '중국'
            
            # 현재 국가 선택 변수 설정
            model.location_selected[m, current_country] = 1
    
    def _get_current_country_for_material(self, material_name: str) -> str:
        """자재의 현재 국가 확인"""
        # 원본 데이터프레임에서 국가 정보 확인
        if self.original_df is not None and '자재명' in self.original_df.columns and '지역' in self.original_df.columns:
            for idx, row in self.original_df.iterrows():
                if row['자재명'] == material_name and pd.notna(row['지역']):
                    return row['지역']
        
        # 시나리오 데이터프레임에서 확인
        if '자재명' in self.scenario_df.columns and '지역' in self.scenario_df.columns:
            for idx, row in self.scenario_df.iterrows():
                if row['자재명'] == material_name and pd.notna(row['지역']):
                    return row['지역']
        
        # 기본값
        return '중국'
    
    def get_optimal_settings(self, model: pyo.ConcreteModel) -> Dict[str, Dict[str, Any]]:
        """최적 설정값 추출"""
        if not model:
            return {}
        
        optimal_settings = {}
        
        for material_name in self.material_types.keys():
            material_settings = {
                'material_name': material_name,
                'material_category': self.material_types[material_name]['material_category'],
                'original_emission': pyo.value(model.original_emission[material_name]),
                'modified_emission': pyo.value(model.modified_emission[material_name]),
                'reduction_percentage': ((pyo.value(model.original_emission[material_name]) - 
                                        pyo.value(model.modified_emission[material_name])) / 
                                       pyo.value(model.original_emission[material_name]) * 100),
                'tier1_re': pyo.value(model.tier1_re[material_name]),
                'tier2_re': pyo.value(model.tier2_re[material_name]),
                'tier3_re': pyo.value(model.tier3_re[material_name])
            }
            
            # Ni, Co, Li 자재인 경우 추가 정보
            if self.material_types[material_name]['is_ni_co_li']:
                material_settings['recycle_ratio'] = pyo.value(model.recycle_ratio[material_name])
                material_settings['low_carbon_ratio'] = pyo.value(model.low_carbon_ratio[material_name])
                material_settings['virgin_ratio'] = pyo.value(model.virgin_ratio[material_name])
            
            optimal_settings[material_name] = material_settings
        
        return optimal_settings
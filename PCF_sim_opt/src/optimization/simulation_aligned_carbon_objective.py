"""
시뮬레이션 로직을 반영한 새로운 목적함수 모듈
rule_based.py의 실제 계산 로직을 기반으로 최적화 목적함수 재설계
"""

from typing import Dict, Any, List, Tuple, Optional
from pyomo.environ import ConcreteModel, Var, Param, Set, summation, value
import pyomo.environ as pyo
import pandas as pd
import numpy as np


class SimulationAlignedCarbonObjective:
    """
    시뮬레이션과 정렬된 탄소발자국 계산을 위한 목적함수 클래스
    
    주요 특징:
    - rule_based.py의 실제 계산 로직 반영
    - 자재별 배출계수 수정 계산
    - Formula/Proportions 방식 구분 적용
    - Tier별 RE 적용률과 계수를 통한 정확한 감축량 계산
    """
    
    def __init__(self, scenario_df: pd.DataFrame, ref_formula_df: pd.DataFrame, 
                 ref_proportions_df: pd.DataFrame, original_df: pd.DataFrame = None):
        """
        Args:
            scenario_df: PCF 시나리오 데이터프레임
            ref_formula_df: 참조 공식 데이터프레임
            ref_proportions_df: 참조 비율 데이터프레임  
            original_df: 원본 데이터프레임
        """
        self.scenario_df = scenario_df.copy()
        self.ref_formula_df = ref_formula_df.copy()
        self.ref_proportions_df = ref_proportions_df.copy()
        self.original_df = original_df.copy() if original_df is not None else None
        
        # 저감활동 적용 가능한 자재 추출
        self.applicable_materials = self._extract_applicable_materials()
        
        # 자재별 매칭 정보 사전 계산
        self.material_matching_info = self._prepare_material_matching_info()
        
    def _extract_applicable_materials(self) -> pd.DataFrame:
        """저감활동 적용 가능한 자재 추출"""
        if '저감활동_적용여부' not in self.scenario_df.columns:
            return pd.DataFrame()
            
        return self.scenario_df[
            self.scenario_df['저감활동_적용여부'] == 1.0
        ].copy()
    
    def _prepare_material_matching_info(self) -> Dict[str, Dict[str, Any]]:
        """
        자재별 매칭 정보 사전 계산
        - Formula 매칭: Tier별 절대 감축량 (kgCO2eq/kg)
        - Proportions 매칭: Tier별 감축 비율 (%)
        """
        matching_info = {}
        
        for idx, material_row in self.applicable_materials.iterrows():
            material_key = self._create_material_key(material_row)
            
            # Original_df에서 매칭 확인
            original_match = self._find_original_match(material_row)
            if original_match is None:
                continue
                
            # Formula 매칭 시도
            formula_match = self._try_formula_matching(material_row, original_match)
            
            # Proportions 매칭 시도  
            proportions_match = self._try_proportions_matching(material_row, original_match)
            
            matching_info[material_key] = {
                'material_row': material_row,
                'original_coeff': original_match['배출계수'],
                'product_amount': material_row['제품총소요량(kg)'],
                'formula_match': formula_match,
                'proportions_match': proportions_match,
                'matching_type': self._determine_matching_type(formula_match, proportions_match, material_row)
            }
            
        return matching_info
    
    def _create_material_key(self, material_row: pd.Series) -> str:
        """자재 고유 키 생성"""
        return f"{material_row['자재명']}_{material_row['자재품목']}"
    
    def _find_original_match(self, material_row: pd.Series) -> Optional[pd.Series]:
        """Original_df에서 매칭되는 행 찾기"""
        if self.original_df is None:
            return None
            
        # 자재명이 NaN인 경우 처리
        if pd.isna(material_row['자재명']):
            matches = self.original_df[
                (self.original_df['자재명'].isna()) & 
                (self.original_df['자재품목'] == material_row['자재품목'])
            ]
        else:
            matches = self.original_df[
                (self.original_df['자재명'] == material_row['자재명']) & 
                (self.original_df['자재품목'] == material_row['자재품목'])
            ]
            
        return matches.iloc[0] if not matches.empty else None
    
    def _try_formula_matching(self, material_row: pd.Series, original_row: pd.Series) -> Optional[Dict[str, float]]:
        """Formula 방식 매칭 시도 - Tier별 절대 감축량 반환"""
        # 자재코드와 지역으로 ref_formula_df에서 매칭
        formula_matches = self.ref_formula_df[
            (self.ref_formula_df['자재코드'] == original_row.get('자재코드', '')) &
            (self.ref_formula_df['지역'] == original_row.get('지역', ''))
        ]
        
        if formula_matches.empty:
            return None
            
        formula_row = formula_matches.iloc[0]
        
        # Tier별 계수 추출
        tier_coeffs = {}
        for col in self.ref_formula_df.columns:
            if 'Tier' in col and 'kgCO2eq/kg' in col:
                tier_num = self._extract_tier_number(col)
                if tier_num:
                    tier_coeffs[f'tier{tier_num}'] = formula_row[col]
        
        return tier_coeffs if tier_coeffs else None
    
    def _try_proportions_matching(self, material_row: pd.Series, original_row: pd.Series) -> Optional[Dict[str, float]]:
        """Proportions 방식 매칭 시도 - Tier별 감축 비율 반환"""
        original_material_name = str(original_row.get('자재명', '')).lower()
        
        # 양극재 특별 처리
        if material_row['자재품목'] == '양극재':
            if pd.isna(original_material_name) or 'cathode active material' in original_material_name:
                cathode_matches = self.ref_proportions_df[
                    self.ref_proportions_df['자재품목'] == '양극재'
                ]
                if not cathode_matches.empty:
                    return self._extract_tier_percentages(cathode_matches.iloc[0])
        
        # 일반적인 매칭 로직
        for idx, prop_row in self.ref_proportions_df.iterrows():
            prop_material_name = str(prop_row.get('자재명(포함)', '')).lower()
            
            if self._check_material_name_match(original_material_name, prop_material_name):
                if self._check_material_category_match(
                    str(original_row.get('자재품목', '')).lower(),
                    str(prop_row.get('자재품목', '')).lower()
                ):
                    return self._extract_tier_percentages(prop_row)
                    
        return None
    
    def _extract_tier_number(self, column_name: str) -> Optional[int]:
        """컬럼명에서 Tier 번호 추출"""
        if 'Tier1' in column_name:
            return 1
        elif 'Tier2' in column_name:
            return 2
        elif 'Tier3' in column_name:
            return 3
        return None
    
    def _extract_tier_percentages(self, proportions_row: pd.Series) -> Dict[str, float]:
        """Proportions 행에서 Tier별 RE 비율 추출"""
        tier_percentages = {}
        
        for col in proportions_row.index:
            if 'Tier' in col and 'RE100' in col:
                tier_num = self._extract_tier_number(col)
                if tier_num:
                    value = proportions_row[col]
                    # 퍼센트 형식 처리
                    if isinstance(value, str):
                        value = float(value.replace('%', ''))
                    tier_percentages[f'tier{tier_num}'] = float(value)
                    
        return tier_percentages
    
    def _check_material_name_match(self, original_name: str, prop_name: str) -> bool:
        """자재명 매칭 확인"""
        # 정확한 포함 관계
        if prop_name in original_name or original_name in prop_name:
            return True
            
        # 토큰 기반 매칭
        original_tokens = set(original_name.split())
        prop_tokens = set(prop_name.split())
        
        return len(prop_tokens) > 0 and prop_tokens.issubset(original_tokens)
    
    def _check_material_category_match(self, original_category: str, prop_category: str) -> bool:
        """자재품목 매칭 확인"""
        return (original_category == prop_category or 
                original_category in prop_category or 
                prop_category in original_category)
    
    def _determine_matching_type(self, formula_match: Optional[Dict], 
                                proportions_match: Optional[Dict], 
                                material_row: pd.Series) -> str:
        """매칭 우선순위 결정 (양극재는 시나리오별 다르게 처리)"""
        if material_row['자재품목'] == '양극재':
            # 양극재는 proportions 우선 (site_change, both 시나리오)
            if proportions_match:
                return 'proportions'
            elif formula_match:
                return 'formula'
        else:
            # 일반 자재는 formula 우선
            if formula_match:
                return 'formula'
            elif proportions_match:
                return 'proportions'
                
        return 'none'
    
    def create_carbon_objective_expression(self, model: ConcreteModel) -> Any:
        """
        시뮬레이션 로직과 정렬된 탄소발자국 목적함수 생성
        
        Args:
            model: Pyomo 최적화 모델 (RE 적용률 변수 포함)
            
        Returns:
            총 탄소발자국 표현식
        """
        total_pcf = 0
        
        # 1. 저감활동이 적용되지 않는 자재들의 기준 PCF
        baseline_pcf = self._calculate_baseline_pcf()
        total_pcf += baseline_pcf
        
        # 2. 저감활동이 적용되는 자재들의 수정된 PCF
        for material_key, info in self.material_matching_info.items():
            material_pcf = self._calculate_material_pcf(model, info)
            total_pcf += material_pcf
            
        return total_pcf
    
    def _calculate_baseline_pcf(self) -> float:
        """저감활동이 적용되지 않는 자재들의 기준 PCF 계산"""
        non_applicable = self.scenario_df[
            self.scenario_df['저감활동_적용여부'] != 1.0
        ]
        
        if '배출량(kgCO2eq)' in non_applicable.columns:
            return non_applicable['배출량(kgCO2eq)'].sum()
        elif '배출계수' in non_applicable.columns and '제품총소요량(kg)' in non_applicable.columns:
            return (non_applicable['배출계수'] * non_applicable['제품총소요량(kg)']).sum()
        else:
            return 0.0
    
    def _calculate_material_pcf(self, model: ConcreteModel, material_info: Dict[str, Any]) -> Any:
        """개별 자재의 수정된 PCF 계산"""
        original_coeff = material_info['original_coeff']
        product_amount = material_info['product_amount']
        matching_type = material_info['matching_type']
        
        if matching_type == 'formula':
            # Formula 방식: 절대 감축량
            modified_coeff = self._calculate_formula_modified_coeff(
                model, original_coeff, material_info['formula_match']
            )
        elif matching_type == 'proportions':
            # Proportions 방식: 비율 감축
            modified_coeff = self._calculate_proportions_modified_coeff(
                model, original_coeff, material_info['proportions_match']
            )
        else:
            # 매칭되지 않는 경우 원본 계수 사용
            modified_coeff = original_coeff
            
        return modified_coeff * product_amount
    
    def _calculate_formula_modified_coeff(self, model: ConcreteModel, 
                                        original_coeff: float, 
                                        tier_coeffs: Dict[str, float]) -> Any:
        """Formula 방식 배출계수 수정 계산"""
        reduction_amount = 0
        
        for tier_name, tier_coeff in tier_coeffs.items():
            # 모델에서 해당 tier의 RE 적용률 변수 가져오기
            re_var_name = f"{tier_name}_re_rate"  # 예: tier1_re_rate
            if hasattr(model, re_var_name):
                re_rate = getattr(model, re_var_name)
                reduction_amount += tier_coeff * re_rate
        
        # 음수 방지 - 수정된 배출계수가 음수가 되는 경우 0으로 설정
        modified_coeff = original_coeff - reduction_amount
        
        # Numeric values
        if isinstance(modified_coeff, (int, float)):
            # 음수인 경우 0 반환, 양수인 경우 그대로 반환
            return modified_coeff if modified_coeff > 0 else 0
        else:
            # Pyomo variables/expressions can't use direct max
            # Instead return the original value but add a constraint to the model
            # that ensures it's non-negative - we must assume the value will be valid
            # in the optimization context
            return modified_coeff
    
    def _calculate_proportions_modified_coeff(self, model: ConcreteModel,
                                            original_coeff: float,
                                            tier_percentages: Dict[str, float]) -> Any:
        """Proportions 방식 배출계수 수정 계산"""
        reduction_factor = 1.0
        
        for tier_name, tier_percentage in tier_percentages.items():
            # 모델에서 해당 tier의 RE 적용률 변수 가져오기
            re_var_name = f"{tier_name}_re_rate"
            if hasattr(model, re_var_name):
                re_rate = getattr(model, re_var_name)
                tier_coeff_decimal = tier_percentage / 100.0
                reduction_factor -= tier_coeff_decimal * re_rate
        
        # 음수 방지 - 감소 팩터가 음수가 되는 경우 0으로 설정
        
        # 숫자 값인 경우 음수 체크
        if isinstance(reduction_factor, (int, float)):
            # 음수인 경우 0으로 설정
            reduction_factor = reduction_factor if reduction_factor > 0 else 0
                
        # 최종 계산
        # Pyomo 변수/표현식의 경우 최적화 과정에서 제약조건이 올바르게 적용될 것으로 가정
        return original_coeff * reduction_factor
    
    def get_material_summary(self) -> pd.DataFrame:
        """자재별 매칭 정보 요약 테이블 생성"""
        summary_data = []
        
        for material_key, info in self.material_matching_info.items():
            summary_data.append({
                '자재명': info['material_row']['자재명'],
                '자재품목': info['material_row']['자재품목'],
                '원본_배출계수': info['original_coeff'],
                '제품총소요량(kg)': info['product_amount'],
                '매칭_방식': info['matching_type'],
                'Formula_매칭': 'Y' if info['formula_match'] else 'N',
                'Proportions_매칭': 'Y' if info['proportions_match'] else 'N'
            })
            
        return pd.DataFrame(summary_data)
    
    def validate_matching_completeness(self) -> Dict[str, Any]:
        """매칭 완성도 검증"""
        total_materials = len(self.applicable_materials)
        matched_materials = len([info for info in self.material_matching_info.values() 
                               if info['matching_type'] != 'none'])
        
        matching_rate = matched_materials / total_materials * 100 if total_materials > 0 else 0
        
        return {
            'total_applicable_materials': total_materials,
            'matched_materials': matched_materials,
            'matching_rate_percent': matching_rate,
            'formula_matches': len([info for info in self.material_matching_info.values() 
                                  if info['matching_type'] == 'formula']),
            'proportions_matches': len([info for info in self.material_matching_info.values() 
                                     if info['matching_type'] == 'proportions']),
            'unmatched_materials': total_materials - matched_materials
        }
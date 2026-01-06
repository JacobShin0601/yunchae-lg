import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from .utils.logging import PrintCompatibleLogger

# FileOperations 사용을 위해 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from src.utils.file_operations import FileOperations, FileLoadError

class RuleBasedSim:
    """
    규칙 기반 PCF 시뮬레이션을 위한 클래스
    """
    
    def __init__(self, scenario_df: pd.DataFrame, ref_formula_df: pd.DataFrame, ref_proportions_df: pd.DataFrame, original_df: pd.DataFrame = None, verbose: bool = True, user_id: str = None):
        """
        RuleBasedSim 초기화

        Args:
            scenario_df: PCF 시나리오 데이터프레임
            ref_formula_df: 참조 공식 데이터프레임
            ref_proportions_df: 참조 비율 데이터프레임
            original_df: 원본 데이터프레임 (선택사항)
            verbose: 상세 로그 출력 여부 (기본값: True)
            user_id: 사용자 ID. 사용자별 작업공간 사용시 필요
        """
        self.scenario_df = scenario_df.copy()
        self.ref_formula_df = ref_formula_df.copy()
        self.ref_proportions_df = ref_proportions_df.copy()
        self.original_df = original_df.copy() if original_df is not None else None
        self.verbose = verbose
        self.user_id = user_id

        # Create logger instance
        self._logger = PrintCompatibleLogger(module_name="RuleBasedSim")

        # 디버그 로그 수집을 위한 리스트 (Streamlit 표시용)
        self.debug_logs = []

        # CathodeSimulator 인스턴스 생성 (반드시 한 번만 초기화)
        from .cathode_simulator import CathodeSimulator
        self.cathode_simulator = None  # 최초에는 None으로 초기화
        # 실제 인스턴스 생성은 필요할 때 지연 생성 방식 사용
    
    def _print(self, *args, level="info", **kwargs):
        """
        로그 레벨에 따라 출력 제어
        verbose == "debug": 모든 로그 출력
        verbose == True: info, warning, error 레벨만 출력
        verbose == False: warning, error 레벨만 출력 (요약 정보만)
        """
        # Collect logs for Streamlit display
        message = ' '.join(str(arg) for arg in args)
        self.debug_logs.append({
            'level': level,
            'message': message
        })

        # Use the new logger with verbose control
        if self.verbose == "debug":
            self._logger._print(*args, level=level, **kwargs)
        elif self.verbose and level in ["info", "warning", "error"]:
            self._logger._print(*args, level=level, **kwargs)
        elif level in ["warning", "error"]:
            self._logger._print(*args, level=level, **kwargs)
    
    def update_ref_proportions_df(self, updated_ref_proportions_df: pd.DataFrame):
        """
        ref_proportions_df를 업데이트합니다.
        
        Args:
            updated_ref_proportions_df: 업데이트된 참조 비율 데이터프레임
        """
        self.ref_proportions_df = updated_ref_proportions_df.copy()
        self._print("✅ ref_proportions_df가 업데이트되었습니다.", level="info")
        
    def extract_reduction_applicable_rows(self) -> pd.DataFrame:
        """
        저감활동 적용 가능한 행들을 추출합니다.
        
        Returns:
            pd.DataFrame: 저감활동 적용 가능한 행들
        """
        self._print("🔍 저감활동 적용 가능한 행 추출", level="info")
        
        if '저감활동_적용여부' not in self.scenario_df.columns:
            self._print("❌ '저감활동_적용여부' 열이 없습니다.", level="info")
            return pd.DataFrame()
        
        # 저감활동_적용여부가 1인 행들 추출
        applicable_rows = self.scenario_df[self.scenario_df['저감활동_적용여부'] == 1.0].copy()
        
        self._print(f"📊 총 데이터 행 수: {len(self.scenario_df)}개", level="info")
        self._print(f"📊 저감활동_적용여부=1인 행 수: {len(applicable_rows)}개", level="info")
        
        # 디버그: 양극재와 음극재 소요량 확인
        cathode_in_scenario = self.scenario_df[self.scenario_df['자재품목'] == '양극재']
        anode_in_scenario = self.scenario_df[self.scenario_df['자재품목'] == '음극재']
        cathode_in_applicable = applicable_rows[applicable_rows['자재품목'] == '양극재']
        anode_in_applicable = applicable_rows[applicable_rows['자재품목'] == '음극재']
        
        self._print("=== RuleBasedSim 양극재/음극재 디버그 ===", level="info")
        self._print(f"scenario_df 중 양극재: {len(cathode_in_scenario)}개", level="info")
        self._print(f"scenario_df 중 음극재: {len(anode_in_scenario)}개", level="info")
        self._print(f"applicable_rows 중 양극재: {len(cathode_in_applicable)}개", level="info")
        self._print(f"applicable_rows 중 음극재: {len(anode_in_applicable)}개", level="info")
        
        # 양극재/음극재 상세 정보
        if not cathode_in_scenario.empty:
            self._print("scenario_df 양극재 상세:", level="info")
            for idx, row in cathode_in_scenario.iterrows():
                quantity = row.get('제품총소요량(kg)', 0)
                reduction_applied = row.get('저감활동_적용여부', 0)
                self._print(f"  • {row.get('자재명', 'N/A')}: 소요량={quantity:.6f}kg, 저감활동={reduction_applied}", level="info")
        
        if not anode_in_scenario.empty:
            self._print("scenario_df 음극재 상세:", level="info")
            for idx, row in anode_in_scenario.iterrows():
                quantity = row.get('제품총소요량(kg)', 0)
                reduction_applied = row.get('저감활동_적용여부', 0)
                self._print(f"  • {row.get('자재명', 'N/A')}: 소요량={quantity:.6f}kg, 저감활동={reduction_applied}", level="info")
        
        self._print("=======================================", level="info")
        
        # 자재품목별 분포 출력
        if '자재품목' in applicable_rows.columns:
            material_distribution = applicable_rows['자재품목'].value_counts()
            self._print("📊 자재품목별 분포:", level="info")
            for category, count in material_distribution.items():
                self._print(f"  • {category}: {count}개", level="info")
        
        # 자재명별 상세 정보 출력 (디버깅용)
        if self.verbose == "debug":
            self._print("🔍 저감활동 적용 자재 상세 정보:", level="info")
            for idx, row in applicable_rows.iterrows():
                material_name = row.get('자재명', 'N/A')
                material_category = row.get('자재품목', 'N/A')
                self._print(f"  • 행 {idx}: {material_name} ({material_category})", level="info")
        
        return applicable_rows
    
    def match_with_formula_data(self, applicable_rows: pd.DataFrame, scenario: str = 'baseline') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        요청된 로직에 따라 매칭 수행:
        1. scenario_df의 '자재명'과 '자재품목'이 일치하는 original_df의 행 추출
        2. 추출된 original_df의 행에서 '자재코드' 및 '지역'을 기억하고 
           ref_formula_df의 '자재명', '자재품목', '자재코드', '지역'이 일치하는 경우,
           ref_formula_df의 Tier1, Tier2 관련 열의 숫자를 변수에 저장
        
        Args:
            applicable_rows: 저감활동 적용 가능한 행들
            scenario: 현재 시나리오 (baseline, site_change, recycling, both)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (formula 매칭된 행들, 매칭되지 않은 행들)
        """
        self._print("🔍 2단계: 공식 데이터 매칭", level="info")
        
        if self.original_df is None:
            self._print("❌ original_df가 제공되지 않았습니다. 매칭을 수행할 수 없습니다.", level="info")
            return pd.DataFrame(), applicable_rows
        
        matched_rows = []
        unmatched_rows = []
        
        for idx, scenario_row in applicable_rows.iterrows():
            if self.verbose == "debug":
                self._print(f"🔍 처리 중: {scenario_row['자재명']} ({scenario_row['자재품목']})")
            
            # original_df에서 매칭 시도
            matching_original_rows = self._find_matching_original_rows(scenario_row)
            
            if len(matching_original_rows) == 0:
                if self.verbose == "debug":
                    self._print(f"❌ Original_df에서 매칭되는 행을 찾을 수 없음")
                
                # 양극재/음극재는 original_df 매칭 없이도 formula/proportions 매칭 시도
                if scenario_row['자재품목'] in ['양극재', '음극재']:
                    if self.verbose == "debug":
                        self._print(f"  🔄 {scenario_row['자재품목']} 감지: original_df 매칭 없이도 formula/proportions 매칭 시도")
                    
                    # 가상의 original_row 생성하여 매칭 시도
                    dummy_original_row = pd.Series({
                        '자재명': scenario_row.get('자재명', ''),
                        '자재품목': scenario_row['자재품목'],
                        '자재코드': '',
                        '지역': 'KR'  # 기본값
                    })
                    dummy_df = pd.DataFrame([dummy_original_row])
                    
                    matched_row = self._try_formula_and_proportions_matching(
                        scenario_row, 
                        dummy_df, 
                        scenario
                    )
                    if matched_row is not None:
                        if self.verbose == "debug":
                            self._print(f"  ✅ {scenario_row['자재품목']} dummy 매칭 성공!")
                        matched_rows.append(matched_row)
                        continue
                
                # 매칭되지 않은 행에 기본값 설정
                unmatched_row = scenario_row.copy()
                unmatched_row['formula_matched'] = False
                unmatched_row['proportions_matched'] = False
                unmatched_row['tier1_values'] = None
                unmatched_row['tier2_values'] = None
                unmatched_row['tier_num_by'] = None
                
                # PCF_reference 열이 없는 경우 계산하여 추가
                if 'PCF_reference' not in unmatched_row:
                    if '배출계수' in unmatched_row and '제품총소요량(kg)' in unmatched_row:
                        unmatched_row['PCF_reference'] = unmatched_row['배출계수'] * unmatched_row['제품총소요량(kg)']
                
                unmatched_rows.append(unmatched_row)
                continue
            
            # formula_df 또는 proportions_df에서 매칭 시도
            matched_row = self._try_formula_and_proportions_matching(scenario_row, matching_original_rows, scenario)
            
            if matched_row is not None:
                matched_rows.append(matched_row)
            else:
                if self.verbose == "debug":
                    self._print(f"❌ 최종 매칭 실패: {scenario_row['자재명']} ({scenario_row['자재품목']})")
                # 매칭되지 않은 행에 기본값 설정
                unmatched_row = scenario_row.copy()
                unmatched_row['formula_matched'] = False
                unmatched_row['proportions_matched'] = False
                unmatched_row['tier1_values'] = None
                unmatched_row['tier2_values'] = None
                unmatched_row['tier_num_by'] = None
                
                # PCF_reference 열이 없는 경우 계산하여 추가
                if 'PCF_reference' not in unmatched_row:
                    if '배출계수' in unmatched_row and '제품총소요량(kg)' in unmatched_row:
                        unmatched_row['PCF_reference'] = unmatched_row['배출계수'] * unmatched_row['제품총소요량(kg)']
                
                unmatched_rows.append(unmatched_row)
        
        matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame()
        unmatched_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame()
        
        self._print(f"📊 최종 매칭 결과: 공식 데이터 매칭 {len(matched_df)}개, 매칭 실패 {len(unmatched_df)}개", level="info")
        
        return matched_df, unmatched_df
    
    def _find_matching_original_rows(self, scenario_row: pd.Series) -> pd.DataFrame:
        """
        scenario_df의 '자재명'과 '자재품목'이 일치하는 original_df의 행 추출
        고유한 배출계수명을 우선적으로 사용하여 정확한 1:1 매칭 보장

        Args:
            scenario_row: scenario_df의 행

        Returns:
            pd.DataFrame: 매칭된 original_df 행들
        """
        # 1차: 고유한 배출계수명으로 정확한 매칭 시도
        if '배출계수명' in scenario_row.index and '배출계수명' in self.original_df.columns:
            scenario_emission_name = scenario_row.get('배출계수명', '')
            if not pd.isna(scenario_emission_name) and str(scenario_emission_name).strip() != '':
                if self.verbose == "debug":
                    self._print(f"    🔍 1차 매칭: 고유 배출계수명 '{scenario_emission_name}'으로 매칭 시도")

                # 고유한 배출계수명과 자재품목으로 매칭
                matching_original_rows = self.original_df[
                    (self.original_df['배출계수명'] == scenario_emission_name) &
                    (self.original_df['자재품목'] == scenario_row['자재품목'])
                ]

                # 추가로 제품총소요량과 배출량으로 정확한 행 찾기 (동일한 배출계수명이라도 소요량/배출량이 다르면 다른 행)
                if len(matching_original_rows) > 1:
                    # 1순위: 제품총소요량과 배출량 둘 다 일치
                    if '제품총소요량(kg)' in scenario_row.index and '배출량(kgCO2eq)' in scenario_row.index:
                        scenario_quantity = scenario_row['제품총소요량(kg)']
                        scenario_emission = scenario_row['배출량(kgCO2eq)']
                        exact_match = matching_original_rows[
                            (matching_original_rows['제품총소요량(kg)'] == scenario_quantity) &
                            (matching_original_rows['배출량(kgCO2eq)'] == scenario_emission)
                        ]
                        if not exact_match.empty:
                            matching_original_rows = exact_match
                            if self.verbose == "debug":
                                self._print(f"    ✓ 소요량 {scenario_quantity}kg + 배출량 {scenario_emission}kgCO2eq으로 정확한 행 매칭")
                        # 2순위: 제품총소요량만 일치
                        elif '제품총소요량(kg)' in scenario_row.index:
                            quantity_match = matching_original_rows[
                                matching_original_rows['제품총소요량(kg)'] == scenario_quantity
                            ]
                            if not quantity_match.empty:
                                matching_original_rows = quantity_match
                                if self.verbose == "debug":
                                    self._print(f"    ✓ 소요량 {scenario_quantity}kg으로 정확한 행 매칭")
                    # 제품총소요량만 있는 경우
                    elif '제품총소요량(kg)' in scenario_row.index:
                        scenario_quantity = scenario_row['제품총소요량(kg)']
                        quantity_match = matching_original_rows[
                            matching_original_rows['제품총소요량(kg)'] == scenario_quantity
                        ]
                        if not quantity_match.empty:
                            matching_original_rows = quantity_match
                            if self.verbose == "debug":
                                self._print(f"    ✓ 소요량 {scenario_quantity}kg으로 정확한 행 매칭")

                if not matching_original_rows.empty:
                    if self.verbose == "debug":
                        self._print(f"    ✅ 고유 배출계수명으로 {len(matching_original_rows)}개 행 매칭 성공")
                    return matching_original_rows

        # 2차: 기존 방식 (자재명 + 자재품목으로 매칭) - fallback
        if self.verbose == "debug":
            self._print(f"    🔍 2차 매칭: 자재명 + 자재품목으로 매칭 시도")

        # 자재명이 nan인 경우 처리
        if pd.isna(scenario_row['자재명']) or str(scenario_row['자재명']).strip() == '':
            if self.verbose == "debug":
                self._print(f"    🔍 자재명이 nan인 경우: original_df에서 자재명이 N/A이고 자재품목이 일치하는 행 검색")
            matching_original_rows = self.original_df[
                ((self.original_df['자재명'].isna()) |
                 (self.original_df['자재명'] == 'N/A') |
                 (self.original_df['자재명'] == 'NaN') |
                 (self.original_df['자재명'] == '')) &
                (self.original_df['자재품목'] == scenario_row['자재품목'])
            ]
        else:
            # 음극재 관련 특별 케이스 처리
            material_name = str(scenario_row['자재명']).lower()
            material_category = str(scenario_row['자재품목']).lower()

            # 자재품목이 "음극재"인 경우 자재명을 보고 판단
            if material_category == '음극재':
                if 'artificial' in material_name:
                    if self.verbose == "debug":
                        self._print(f"    🔍 음극재 + Artificial 감지: 음극재로 매칭 시도")
                    matching_original_rows = self.original_df[
                        (self.original_df['자재명'] == scenario_row['자재명']) &
                        (self.original_df['자재품목'] == '음극재')
                    ]
                elif 'natural' in material_name:
                    if self.verbose == "debug":
                        self._print(f"    🔍 음극재 + Natural 감지: 음극재로 매칭 시도")
                    matching_original_rows = self.original_df[
                        (self.original_df['자재명'] == scenario_row['자재명']) &
                        (self.original_df['자재품목'] == '음극재')
                    ]
                else:
                    # artificial/natural이 없는 경우 일반적인 음극재로 매칭
                    if self.verbose == "debug":
                        self._print(f"    🔍 일반 음극재: 기본 매칭 시도")
                    matching_original_rows = self.original_df[
                        (self.original_df['자재명'] == scenario_row['자재명']) &
                        (self.original_df['자재품목'] == scenario_row['자재품목'])
                    ]
            else:
                # 일반적인 매칭
                matching_original_rows = self.original_df[
                    (self.original_df['자재명'] == scenario_row['자재명']) &
                    (self.original_df['자재품목'] == scenario_row['자재품목'])
                ]

        # 여러 행이 매칭된 경우, 제품총소요량과 배출량으로 정확한 행 찾기
        if len(matching_original_rows) > 1:
            # 1순위: 제품총소요량과 배출량 둘 다 일치
            if '제품총소요량(kg)' in scenario_row.index and '배출량(kgCO2eq)' in scenario_row.index:
                scenario_quantity = scenario_row['제품총소요량(kg)']
                scenario_emission = scenario_row['배출량(kgCO2eq)']
                exact_match = matching_original_rows[
                    (matching_original_rows['제품총소요량(kg)'] == scenario_quantity) &
                    (matching_original_rows['배출량(kgCO2eq)'] == scenario_emission)
                ]
                if not exact_match.empty:
                    matching_original_rows = exact_match
                    if self.verbose == "debug":
                        self._print(f"    ✓ 소요량 {scenario_quantity}kg + 배출량 {scenario_emission}kgCO2eq으로 정확한 행 찾기")
                # 2순위: 제품총소요량만 일치
                elif '제품총소요량(kg)' in scenario_row.index:
                    quantity_match = matching_original_rows[
                        matching_original_rows['제품총소요량(kg)'] == scenario_quantity
                    ]
                    if not quantity_match.empty:
                        matching_original_rows = quantity_match
                        if self.verbose == "debug":
                            self._print(f"    ✓ 소요량 {scenario_quantity}kg으로 정확한 행 찾기")
            # 제품총소요량만 있는 경우
            elif '제품총소요량(kg)' in scenario_row.index:
                scenario_quantity = scenario_row['제품총소요량(kg)']
                quantity_match = matching_original_rows[
                    matching_original_rows['제품총소요량(kg)'] == scenario_quantity
                ]
                if not quantity_match.empty:
                    matching_original_rows = quantity_match
                    if self.verbose == "debug":
                        self._print(f"    ✓ 소요량 {scenario_quantity}kg으로 정확한 행 찾기")
        
        if self.verbose == "debug":
            self._print(f"📋 Original_df에서 매칭된 행 수: {len(matching_original_rows)}개")
            
            # original_df에서 추출이 되었는지 확인하고 비율 표시
            if len(matching_original_rows) > 0:
                total_original_rows = len(self.original_df)
                match_ratio = len(matching_original_rows) / total_original_rows * 100
                self._print(f"📊 Original_df 매칭 비율: {match_ratio:.2f}% ({len(matching_original_rows)}/{total_original_rows})")
        
        return matching_original_rows
    
    def _try_formula_and_proportions_matching(self, scenario_row: pd.Series, matching_original_rows: pd.DataFrame, scenario: str = 'baseline') -> Optional[pd.Series]:
        """
        각 매칭된 original_df 행에 대해 ref_formula_df 또는 ref_proportions_df에서 매칭 시도
        
        Args:
            scenario_row: scenario_df의 행
            matching_original_rows: 매칭된 original_df 행들
            scenario: 현재 시나리오 (baseline, site_change, recycling, both)
            
        Returns:
            Optional[pd.Series]: 매칭된 행 (매칭되지 않으면 None)
        """
        for orig_idx, original_row in matching_original_rows.iterrows():
            if self.verbose == "debug":
                self._print(f"  🔍 Original 행 처리: 자재코드={original_row.get('자재코드', 'N/A')}, 지역={original_row.get('지역', 'N/A')}")
            
            # 양극재의 경우 시나리오에 따라 매칭 우선순위 결정
            if scenario_row['자재품목'] == '양극재':
                if scenario in ['site_change', 'both']:
                    # site_change/both 시나리오: proportions 매칭을 우선 시도
                    if self.verbose == "debug":
                        self._print(f"  🔍 양극재 + {scenario} 시나리오: proportions 매칭을 우선 시도")
                    
                    # 1단계: ref_proportions_df에서 매칭 시도
                    matched_row = self._try_proportions_matching(scenario_row, original_row)
                    if matched_row is not None:
                        return matched_row
                    
                    # 2단계: ref_formula_df에서 매칭 시도
                    matched_row = self._try_formula_matching(scenario_row, original_row)
                    if matched_row is not None:
                        return matched_row
                else:
                    # baseline/recycling 시나리오: formula 매칭을 우선 시도
                    if self.verbose == "debug":
                        self._print(f"  🔍 양극재 + {scenario} 시나리오: formula 매칭을 우선 시도")
                    
                    # 1단계: ref_formula_df에서 매칭 시도
                    matched_row = self._try_formula_matching(scenario_row, original_row)
                    if matched_row is not None:
                        return matched_row
                    
                    # 2단계: ref_proportions_df에서 매칭 시도
                    matched_row = self._try_proportions_matching(scenario_row, original_row)
                    if matched_row is not None:
                        return matched_row
            else:
                # 일반적인 경우: formula 매칭을 우선 시도
                # 1단계: ref_formula_df에서 매칭 시도
                matched_row = self._try_formula_matching(scenario_row, original_row)
                if matched_row is not None:
                    return matched_row
                
                # 2단계: ref_proportions_df에서 매칭 시도
                matched_row = self._try_proportions_matching(scenario_row, original_row)
                if matched_row is not None:
                    return matched_row
        
        return None
    
    def _try_formula_matching(self, scenario_row: pd.Series, original_row: pd.Series) -> Optional[pd.Series]:
        """
        ref_formula_df에서 4개 조건으로 매칭 시도
        음극재 관련 특별 케이스 처리 포함
        
        Args:
            scenario_row: scenario_df의 행
            original_row: original_df의 행
            
        Returns:
            Optional[pd.Series]: 매칭된 행 (매칭되지 않으면 None)
        """
        # 자재명이 nan인 경우 처리
        if pd.isna(scenario_row['자재명']) or str(scenario_row['자재명']).strip() == '':
            if self.verbose == "debug":
                self._print(f"    🔍 자재명이 nan인 경우: formula_df에서 자재코드와 지역만으로 매칭 시도")
            formula_matches = self.ref_formula_df[
                (self.ref_formula_df['자재코드'] == original_row.get('자재코드', '')) &
                (self.ref_formula_df['지역'] == original_row.get('지역', ''))
            ]
        else:
            # 음극재 관련 특별 케이스 처리
            material_name = str(scenario_row['자재명']).lower()
            material_category = str(scenario_row['자재품목']).lower()
            
            # 자재품목이 "음극재"인 경우 자재명을 보고 판단
            if material_category == '음극재':
                if 'artificial' in material_name:
                    if self.verbose == "debug":
                        self._print(f"    🔍 음극재 + Artificial 감지: formula_df에서 자재코드와 지역만으로 매칭 시도")
                    # 먼저 정확한 지역 매칭 시도
                    formula_matches = self.ref_formula_df[
                        (self.ref_formula_df['자재코드'] == original_row.get('자재코드', '')) &
                        (self.ref_formula_df['지역'] == original_row.get('지역', ''))
                    ]
                    # 매칭이 실패하면 지역 무관하게 자재코드만으로 매칭 시도
                    if len(formula_matches) == 0:
                        if self.verbose == "debug":
                            self._print(f"    🔍 음극재 + Artificial: 지역 매칭 실패, 자재코드만으로 재시도")
                        formula_matches = self.ref_formula_df[
                            (self.ref_formula_df['자재코드'] == original_row.get('자재코드', ''))
                        ]
                    # 여전히 매칭이 실패하면 자재명 기반 매칭 시도
                    if len(formula_matches) == 0:
                        if self.verbose == "debug":
                            self._print(f"    🔍 음극재 + Artificial: 자재코드 매칭 실패, 자재명 기반 매칭 시도")
                        scenario_material_name = str(scenario_row['자재명']).lower()
                        formula_matches = self.ref_formula_df[
                            (self.ref_formula_df['자재명(포함)'].str.lower().str.contains('artificial', na=False)) &
                            (self.ref_formula_df['지역'] == original_row.get('지역', ''))
                        ]
                        if len(formula_matches) == 0:
                            formula_matches = self.ref_formula_df[
                                (self.ref_formula_df['자재명(포함)'].str.lower().str.contains('artificial', na=False))
                            ]
                elif 'natural' in material_name:
                    if self.verbose == "debug":
                        self._print(f"    🔍 음극재 + Natural 감지: formula_df에서 자재코드와 지역만으로 매칭 시도")
                    # 먼저 정확한 지역 매칭 시도
                    formula_matches = self.ref_formula_df[
                        (self.ref_formula_df['자재코드'] == original_row.get('자재코드', '')) &
                        (self.ref_formula_df['지역'] == original_row.get('지역', ''))
                    ]
                    # 매칭이 실패하면 지역 무관하게 자재코드만으로 매칭 시도
                    if len(formula_matches) == 0:
                        if self.verbose == "debug":
                            self._print(f"    🔍 음극재 + Natural: 지역 매칭 실패, 자재코드만으로 재시도")
                        formula_matches = self.ref_formula_df[
                            (self.ref_formula_df['자재코드'] == original_row.get('자재코드', ''))
                        ]
                    # 여전히 매칭이 실패하면 자재명 기반 매칭 시도
                    if len(formula_matches) == 0:
                        if self.verbose == "debug":
                            self._print(f"    🔍 음극재 + Natural: 자재코드 매칭 실패, 자재명 기반 매칭 시도")
                        scenario_material_name = str(scenario_row['자재명']).lower()
                        formula_matches = self.ref_formula_df[
                            (self.ref_formula_df['자재명(포함)'].str.lower().str.contains('natural', na=False)) &
                            (self.ref_formula_df['지역'] == original_row.get('지역', ''))
                        ]
                        if len(formula_matches) == 0:
                            formula_matches = self.ref_formula_df[
                                (self.ref_formula_df['자재명(포함)'].str.lower().str.contains('natural', na=False))
                            ]
                else:
                    # artificial/natural이 없는 경우 일반적인 매칭
                    if self.verbose == "debug":
                        self._print(f"    🔍 일반 음극재: formula_df에서 자재코드와 지역만으로 매칭 시도")
                    # 먼저 정확한 지역 매칭 시도
                    formula_matches = self.ref_formula_df[
                        (self.ref_formula_df['자재코드'] == original_row.get('자재코드', '')) &
                        (self.ref_formula_df['지역'] == original_row.get('지역', ''))
                    ]
                    # 매칭이 실패하면 지역 무관하게 자재코드만으로 매칭 시도
                    if len(formula_matches) == 0:
                        if self.verbose == "debug":
                            self._print(f"    🔍 일반 음극재: 지역 매칭 실패, 자재코드만으로 재시도")
                        formula_matches = self.ref_formula_df[
                            (self.ref_formula_df['자재코드'] == original_row.get('자재코드', ''))
                        ]
                    # 여전히 매칭이 실패하면 자재명 기반 매칭 시도
                    if len(formula_matches) == 0:
                        if self.verbose == "debug":
                            self._print(f"    🔍 일반 음극재: 자재코드 매칭 실패, 자재명 기반 매칭 시도")
                        formula_matches = self.ref_formula_df[
                            (self.ref_formula_df['지역'] == original_row.get('지역', ''))
                        ]
                        if len(formula_matches) == 0:
                            formula_matches = self.ref_formula_df
            else:
                # 일반적인 매칭 (자재코드와 지역만으로 매칭)
                formula_matches = self.ref_formula_df[
                    (self.ref_formula_df['자재코드'] == original_row.get('자재코드', '')) &
                    (self.ref_formula_df['지역'] == original_row.get('지역', ''))
                ]
        
        if self.verbose == "debug":
            self._print(f"    📊 Formula_df에서 매칭된 행 수: {len(formula_matches)}개")
        
        if len(formula_matches) > 0:
            formula_row = formula_matches.iloc[0]
            self._print(f"    ✅ Formula 매칭 성공!", level="info")
            
            # Tier1, Tier2 관련 열 찾기
            tier1_columns = [col for col in self.ref_formula_df.columns if 'Tier1' in col]
            tier2_columns = [col for col in self.ref_formula_df.columns if 'Tier2' in col]
            
            if self.verbose == "debug":
                self._print(f"    📋 Tier1 관련 열: {tier1_columns}")
                self._print(f"    📋 Tier2 관련 열: {tier2_columns}")
            
            # Tier1, Tier2 값 저장 및 출력
            tier1_values = {}
            tier2_values = {}
            
            for col in tier1_columns:
                value = formula_row[col]
                tier1_values[col] = value
                self._print(f"    📊 {col}: {value}", level="info")
            
            for col in tier2_columns:
                value = formula_row[col]
                tier2_values[col] = value
                self._print(f"    📊 {col}: {value}", level="info")
            
            # 매칭된 행에 Tier 정보 추가
            matched_row = scenario_row.copy()
            matched_row['tier1_values'] = tier1_values
            matched_row['tier2_values'] = tier2_values
            matched_row['formula_matched'] = True
            matched_row['proportions_matched'] = False
            
            return matched_row
        else:
            if self.verbose == "debug":
                self._print(f"    ❌ Formula_df에서 매칭되는 행을 찾을 수 없음")
            return None
    
    def _try_proportions_matching(self, scenario_row: pd.Series, original_row: pd.Series) -> Optional[pd.Series]:
        """
        ref_proportions_df에서 매칭 시도
        
        Args:
            scenario_row: scenario_df의 행
            original_row: original_df의 행
            
        Returns:
            Optional[pd.Series]: 매칭된 행 (매칭되지 않으면 None)
        """
        if self.verbose == "debug":
            self._print(f"    🔍 Proportions_df에서 매칭 시도...")
        tier1_value, tier2_value = self._match_with_proportions_data(scenario_row, original_row)
        
        if tier1_value is not None and tier2_value is not None:
            self._print(f"    ✅ Proportions_df 매칭 성공!", level="info")
            self._print(f"    📊 Tier1 값: {tier1_value}", level="info")
            self._print(f"    📊 Tier2 값: {tier2_value}", level="info")
            
            # 매칭된 행에 Tier 정보 추가
            matched_row = scenario_row.copy()
            matched_row['tier1_values'] = {'Tier1_RE100(%)': tier1_value}
            matched_row['tier2_values'] = {'Tier2_RE100(%)': tier2_value}
            matched_row['formula_matched'] = False
            matched_row['proportions_matched'] = True
            
            return matched_row
        else:
            if self.verbose == "debug":
                self._print(f"    ❌ Proportions_df에서도 매칭되는 행을 찾을 수 없음")
            return None
    
    def _match_with_proportions_data(self, scenario_row: pd.Series, original_row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """
        ref_proportions_df의 '자재명(포함)'을 Loop로 돌며 해당 항목이 original_df의 '자재명'에 포함되는지 확인
        이 때, 전부 소문자로 바꿔서 검사하고, 토큰 기반 매칭과 자재품목 일치 확인을 수행
        음극재 관련 특별 케이스 처리 포함
        
        Args:
            scenario_row: scenario_df의 행
            original_row: original_df의 행
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (Tier1 값, Tier2 값) - 매칭되지 않으면 (None, None)
        """
        # NaN 체크를 문자열 변환 전에 수행
        original_material_name_raw = original_row.get('자재명', '')
        scenario_material_name_raw = scenario_row.get('자재명', '')
        is_original_name_nan = pd.isna(original_material_name_raw)
        is_scenario_name_nan = pd.isna(scenario_material_name_raw)
        
        original_material_name = str(original_material_name_raw).lower()
        original_material_category = str(original_row.get('자재품목', '')).lower()
        scenario_material_name = str(scenario_material_name_raw).lower()
        scenario_material_category = str(scenario_row.get('자재품목', '')).lower()
        
        if self.verbose == "debug":
            self._print(f"      🔍 Original 자재명 (소문자): '{original_material_name}'")
            self._print(f"      🔍 Original 자재품목 (소문자): '{original_material_category}'")
            self._print(f"      🔍 Scenario 자재명 (소문자): '{scenario_material_name}'")
            self._print(f"      🔍 Scenario 자재품목 (소문자): '{scenario_material_category}'")
        
        # 음극재 관련 특별 케이스 처리
        anode_artificial_detected = ((scenario_material_category == '음극재' and 'artificial' in scenario_material_name) or
                                   (original_material_category == '음극재' and 'artificial' in original_material_name))
        anode_natural_detected = ((scenario_material_category == '음극재' and 'natural' in scenario_material_name) or
                                 (original_material_category == '음극재' and 'natural' in original_material_name))
        
        # 양극재 관련 특별 케이스 처리
        cathode_detected = scenario_material_category == '양극재' or original_material_category == '양극재'
        
        for idx, proportion_row in self.ref_proportions_df.iterrows():
            proportion_material_name = str(proportion_row.get('자재명(포함)', '')).lower()
            proportion_material_category = str(proportion_row.get('자재품목', '')).lower()
            
            if self.verbose == "debug":
                self._print(f"      🔍 비교 중: '{proportion_material_name}' (자재품목: '{proportion_material_category}')")
            
            # 양극재 매칭 - scenario 또는 original이 양극재인 경우
            if proportion_material_category == '양극재':
                # scenario가 양극재이거나 original이 양극재인 경우
                if (scenario_material_category == '양극재' or original_material_category == '양극재'):
                    if self.verbose == "debug":
                        self._print(f"      🔍 양극재 감지: 양극재로 매칭 시도")
                    # 자재명이 빈값/NaN이거나 cathode 관련 이름인 경우
                    if (is_original_name_nan or original_material_name in ['', 'nan', 'n/a'] or
                        is_scenario_name_nan or scenario_material_name in ['', 'nan', 'n/a'] or
                        'cathode' in original_material_name or 'cathode' in scenario_material_name):
                        self._print(f"      ✅ 양극재 매칭 성공! (통합 로직)", level="info")
                        return self._extract_tier_values(proportion_row)
                continue
            
            # 음극재 매칭 - Natural/Artificial 구분
            if proportion_material_category == '음극재':
                # Natural 케이스
                if ('natural' in original_material_name or 'natural' in scenario_material_name):
                    if 'natural' in proportion_material_name:
                        if self.verbose == "debug":
                            self._print(f"      🔍 음극재 Natural 감지: 매칭 시도")
                        self._print(f"      ✅ 음극재 Natural 매칭 성공!", level="info")
                        return self._extract_tier_values(proportion_row)
                # Artificial 케이스  
                elif ('artificial' in original_material_name or 'artificial' in scenario_material_name):
                    if 'artificial' in proportion_material_name:
                        if self.verbose == "debug":
                            self._print(f"      🔍 음극재 Artificial 감지: 매칭 시도")
                        self._print(f"      ✅ 음극재 Artificial 매칭 성공!", level="info")
                        return self._extract_tier_values(proportion_row)
                continue
            
            # 일반적인 매칭 로직
            # 1단계: 정확한 포함 관계 확인
            if proportion_material_name in original_material_name or original_material_name in proportion_material_name:
                self._print(f"      ✅ 정확한 포함 관계 매칭 성공! '{proportion_material_name}' <-> '{original_material_name}'", level="info")
                # 자재품목 일치 확인
                if self._check_material_category_match(original_material_category, proportion_material_category):
                    return self._extract_tier_values(proportion_row)
                else:
                    if self.verbose == "debug":
                        self._print(f"      ❌ 자재품목 불일치: '{original_material_category}' != '{proportion_material_category}'")
                    continue
            
            # 2단계: 토큰 기반 매칭 시도
            if self._check_token_based_match(original_material_name, proportion_material_name):
                self._print(f"      ✅ 토큰 기반 매칭 성공! '{proportion_material_name}' <-> '{original_material_name}'", level="info")
                # 자재품목 일치 확인
                if self._check_material_category_match(original_material_category, proportion_material_category):
                    return self._extract_tier_values(proportion_row)
                else:
                    if self.verbose == "debug":
                        self._print(f"      ❌ 자재품목 불일치: '{original_material_category}' != '{proportion_material_category}'")
                    continue
        
        if self.verbose == "debug":
            self._print(f"      ❌ Proportions_df에서 매칭되는 항목을 찾을 수 없음")
        return None, None
    
    def _check_token_based_match(self, original_material_name: str, proportion_material_name: str) -> bool:
        """
        토큰 기반 매칭: proportion_material_name의 모든 토큰이 original_material_name에 포함되는지 확인
        
        Args:
            original_material_name: original_df의 자재명 (소문자)
            proportion_material_name: ref_proportions_df의 자재명(포함) (소문자)
            
        Returns:
            bool: 모든 토큰이 포함되면 True, 아니면 False
        """
        original_tokens = set(original_material_name.split())
        proportion_tokens = set(proportion_material_name.split())
        
        # proportion_material_name의 모든 토큰이 original_material_name에 포함되는지 확인
        if len(proportion_tokens) > 0 and proportion_tokens.issubset(original_tokens):
            common_tokens = proportion_tokens.intersection(original_tokens)
            if self.verbose == "debug":
                self._print(f"      📋 토큰 매칭: {common_tokens} (모든 토큰 포함)")
            return True
        
        return False
    
    def _check_material_category_match(self, original_category: str, proportion_category: str) -> bool:
        """
        자재품목 일치 확인
        
        Args:
            original_category: original_df의 자재품목 (소문자)
            proportion_category: ref_proportions_df의 자재품목 (소문자)
            
        Returns:
            bool: 일치하면 True, 아니면 False
        """
        # 정확한 일치 확인
        if original_category == proportion_category:
            return True
        
        # 부분 일치 확인 (예: "al foil" vs "foil")
        if original_category in proportion_category or proportion_category in original_category:
            return True
        
        return False
    
    def _extract_tier_values(self, proportion_row: pd.Series) -> Tuple[float, float]:
        """
        proportion_row에서 Tier1, Tier2 값을 추출
        
        Args:
            proportion_row: ref_proportions_df의 행
            
        Returns:
            Tuple[float, float]: (Tier1 값, Tier2 값)
        """
        # Tier1, Tier2 값 추출
        tier1_value_raw = proportion_row.get('Tier1_RE100(%)', 0.0)
        tier2_value_raw = proportion_row.get('Tier2_RE100(%)', 0.0)
        
        # 값이 이미 숫자인지 확인하고 변환
        try:
            # 이미 숫자인 경우
            if isinstance(tier1_value_raw, (int, float)):
                tier1_value = float(tier1_value_raw)
            else:
                # 문자열인 경우 % 제거 후 변환
                tier1_value = float(str(tier1_value_raw).replace('%', ''))
            
            if isinstance(tier2_value_raw, (int, float)):
                tier2_value = float(tier2_value_raw)
            else:
                # 문자열인 경우 % 제거 후 변환
                tier2_value = float(str(tier2_value_raw).replace('%', ''))
            
            self._print(f"      📊 Tier1_RE100(%): {tier1_value}%", level="info")
            self._print(f"      📊 Tier2_RE100(%): {tier2_value}%", level="info")
            
            return tier1_value, tier2_value
        except (ValueError, AttributeError) as e:
            self._print(f"      ❌ 값 변환 오류: {e}", level="error")
            return 0.0, 0.0
    

    
    def calculate_final_pcf(self, matched_df: pd.DataFrame, proportion_df: pd.DataFrame) -> pd.DataFrame:
        """
        최종 PCF 계산 및 결과 통합
        
        Args:
            matched_df: 공식 데이터와 매칭된 행들
            proportion_df: 비율 기반 저감이 적용된 행들
            
        Returns:
            pd.DataFrame: 최종 계산된 PCF 결과
        """
        self._print("🔍 4단계: 최종 PCF 계산", level="info")
        
        # 모든 결과 통합
        all_results = []
        
        if len(matched_df) > 0:
            all_results.append(matched_df)
            self._print(f"📋 공식 매칭 결과: {len(matched_df)}개 행", level="info")
        
        if len(proportion_df) > 0:
            all_results.append(proportion_df)
            self._print(f"📋 비율 기반 결과: {len(proportion_df)}개 행", level="info")
        
        if not all_results:
            self._print("❌ 처리할 데이터가 없습니다.", level="info")
            return pd.DataFrame()
        
        # 결과 통합
        final_result = pd.concat(all_results, ignore_index=True)
        
        # PCF 합계 계산
        pcf_sums = {
            'PCF_reference': final_result['PCF_reference'].sum(),
            'PCF_case1': final_result['PCF_case1'].sum(),
            'PCF_case2': final_result['PCF_case2'].sum(),
            'PCF_case3': final_result['PCF_case3'].sum()
        }
        
        self._print("📈 최종 PCF 합계:")
        self._print("-" * 30)
        for case, total in pcf_sums.items():
            self._print(f"{case}: {total:.3f} kgCO2eq", level="info")
        
        # 감소율 계산
        reference = pcf_sums['PCF_reference']
        self._print("📉 Reference 대비 감소율:")
        self._print("-" * 30)
        for case in ['PCF_case1', 'PCF_case2', 'PCF_case3']:
            reduction = reference - pcf_sums[case]
            reduction_rate = (reduction / reference) * 100
            self._print(f"{case}: {reduction_rate:.2f}% 감소 ({reduction:.3f} kgCO2eq)", level="info")

        # 디버그 모드일 때 상세 계산 데이터 저장
        if self.verbose == "debug":
            try:
                debug_export_cols = [
                    '자재명', '자재품목', '배출계수명', '배출계수',
                    'modified_coeff_case1', 'modified_coeff_case2', 'modified_coeff_case3',
                    'PCF_reference', 'PCF_case1', 'PCF_case2', 'PCF_case3',
                    'Tier1_RE_case1', 'Tier1_RE_case2', 'Tier1_RE_case3',
                    'Tier2_RE_case1', 'Tier2_RE_case2', 'Tier2_RE_case3',
                    'formula_matched', 'proportions_matched', '제품총소요량(kg)'
                ]

                # 존재하는 컬럼만 선택
                available_cols = [col for col in debug_export_cols if col in final_result.columns]
                debug_df = final_result[available_cols].copy()

                # 감소율 추가 계산
                debug_df['reduction_rate_case1_%'] = ((debug_df['PCF_reference'] - debug_df['PCF_case1']) / debug_df['PCF_reference'] * 100).round(2)
                debug_df['reduction_rate_case2_%'] = ((debug_df['PCF_reference'] - debug_df['PCF_case2']) / debug_df['PCF_reference'] * 100).round(2)
                debug_df['reduction_rate_case3_%'] = ((debug_df['PCF_reference'] - debug_df['PCF_case3']) / debug_df['PCF_reference'] * 100).round(2)

                # 비정상 플래그 추가
                debug_df['anomaly_case2_better_than_case1'] = (debug_df['PCF_case2'] < debug_df['PCF_case1'])

                # 파일 저장
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.join(current_dir, "..")
                debug_file = os.path.join(project_root, "debug_calculation_details.csv")
                debug_df.to_csv(debug_file, index=False, encoding='utf-8-sig')
                self._print(f"📊 디버그 계산 상세 데이터 저장됨: {debug_file}", level="info")
            except Exception as e:
                self._print(f"⚠️ 디버그 데이터 저장 실패: {e}", level="warning")

        return final_result
    
    def calculate_modified_coefficients(self, matched_df: pd.DataFrame, max_case: int = 3, num_tier: int = 2, scenario: str = 'baseline', basic_df: pd.DataFrame = None, skip_updates: bool = False) -> pd.DataFrame:
        """
        배출계수 수정 계산 (formula_matched와 proportions_matched에 따라 분기)
        
        Args:
            matched_df: 매칭된 데이터프레임
            max_case: 계산할 최대 case 번호 (기본값: 3)
            num_tier: 계산할 최대 tier 번호 (기본값: 3)
            scenario: 현재 실행 중인 시나리오 (기본값: 'baseline')
            basic_df: CathodeSimulator의 기본 분석 결과 (시나리오별 업데이트용)
            
        Returns:
            pd.DataFrame: 수정된 배출계수가 추가된 데이터프레임
        """
        self._print(f"🔍 배출계수 수정 계산 (Case 1 ~ {max_case}, Tier 1 ~ {num_tier}, 시나리오: {scenario})", level="info")
        
        # 전력 배출계수가 변경된 시나리오인지 확인
        power_emission_factor_updated = False
        tier1_power_coeff = None
        tier2_power_coeff = None
        
        if hasattr(self, 'cathode_simulator') and self.cathode_simulator and hasattr(self.cathode_simulator, 'coefficient_data'):
            coefficient_data = self.cathode_simulator.coefficient_data
            if coefficient_data:
                # Energy(Tier-1) 전력 배출계수 확인
                if 'Energy(Tier-1)' in coefficient_data and '전력' in coefficient_data['Energy(Tier-1)']:
                    tier1_power_coeff = coefficient_data['Energy(Tier-1)']['전력']['배출계수']
                    power_emission_factor_updated = True
                    self._print(f"📊 Energy(Tier-1) 전력 배출계수 감지: {tier1_power_coeff}", level="info")
                
                # Energy(Tier-2) 전력 배출계수 확인
                if 'Energy(Tier-2)' in coefficient_data and '전력' in coefficient_data['Energy(Tier-2)']:
                    tier2_power_coeff = coefficient_data['Energy(Tier-2)']['전력']['배출계수']
                    power_emission_factor_updated = True
                    self._print(f"📊 Energy(Tier-2) 전력 배출계수 감지: {tier2_power_coeff}", level="info")
        
        if power_emission_factor_updated:
            self._print(f"🔧 {scenario} 시나리오: 전력 배출계수가 변경된 데이터 사용", level="info")
        else:
            self._print(f"🔧 {scenario} 시나리오: 기본 전력 배출계수 사용", level="info")
        
        if len(matched_df) == 0:
            self._print("❌ 처리할 데이터가 없습니다.", level="info")
            return matched_df
        
        # 시나리오별 데이터 업데이트 수행 (skip_updates가 False인 경우에만)
        if not skip_updates:
            self._print(f"🔄 {scenario} 시나리오별 데이터 업데이트 시작", level="info")
            
            if scenario == 'recycling':
                # recycling 시나리오: original_df의 양극재 배출계수 업데이트
                self._print("📋 Recycling 시나리오: original_df 양극재 배출계수 업데이트", level="info")
                
                # basic_df가 없으면 기본값 생성
                if basic_df is None:
                    self._print("📋 basic_df가 없어 재활용 시나리오용 기본 basic_df 생성", level="info")
                    basic_df = pd.DataFrame({
                        '시나리오': ['재활용&저탄소메탈 적용'],
                        '총_배출량': [35.0],  # 기본값
                        '감축률_퍼센트': [10.0],  # 기본값
                        '재활용_비율_퍼센트': [20.0]  # 기본값
                    })
                
                self.update_original_cathode_coefficients(basic_df, scenario='recycling')
                
            elif scenario == 'site_change' and basic_df is not None:
                # site_change 시나리오: 양극재 배출계수 + ref_proportions_df 업데이트
                # 생산지 변경으로 전력 배출계수가 변경되면 양극재 제조 과정의 배출량도 변경됨
                self._print("📋 Site Change 시나리오: 양극재 배출계수 + ref_proportions_df 업데이트", level="info")

                # 양극재 배출계수 업데이트 (전력 배출계수 변경 반영)
                self.update_original_cathode_coefficients(basic_df, scenario='site_change')

                # ref_proportions_df 업데이트
                self.update_ref_proportions_by_scenario('site_change', basic_df)
                
            elif scenario == 'both':
                # both 시나리오: 양극재 배출계수 + ref_proportions_df 모두 업데이트
                self._print("📋 Both 시나리오: 양극재 배출계수 + ref_proportions_df 업데이트", level="info")
                
                # basic_df가 없으면 기본값 생성
                if basic_df is None:
                    self._print("📋 basic_df가 없어 종합 시나리오용 기본 basic_df 생성", level="info")
                    basic_df = pd.DataFrame({
                        '시나리오': ['재활용&저탄소메탈 + 사이트 변경'],
                        '총_배출량': [30.0],  # 기본값
                        '감축률_퍼센트': [20.0],  # 기본값
                        '재활용_비율_퍼센트': [25.0]  # 기본값
                    })
                
                self.update_original_cathode_coefficients(basic_df, scenario='both')
                self.update_ref_proportions_by_scenario('both', basic_df)
                
            elif scenario == 'baseline':
                # baseline 시나리오: 업데이트 없음
                self._print("📋 Baseline 시나리오: 데이터 업데이트 없음", level="info")
                
            else:
                self._print(f"⚠️ {scenario} 시나리오에 대한 처리가 정의되지 않았습니다.", level="warning")
        else:
            self._print(f"⏭️ {scenario} 시나리오별 데이터 업데이트 건너뛰기 (skip_updates=True)", level="info")
        
        # 시나리오별 데이터 상태 로깅
        self._print(f"📊 {scenario} 시나리오 배출계수 수정 계산 시작", level="info")
        if self.original_df is not None and '배출계수' in self.original_df.columns:
            cathode_rows = self.original_df[self.original_df['자재품목'] == '양극재']
            if not cathode_rows.empty:
                self._print(f"  • 양극재 배출계수 상태:", level="debug")
                for idx, row in cathode_rows.iterrows():
                    self._print(f"    - {row['자재명']}: {row['배출계수']:.6f}", level="debug")
        
        # 매칭 결과 상세 분석
        self._print("📊 매칭 결과 분석:", level="info")
        formula_matched_count = len(matched_df[matched_df['formula_matched'] == True])
        proportions_matched_count = len(matched_df[matched_df['proportions_matched'] == True])
        both_matched_count = len(matched_df[(matched_df['formula_matched'] == True) & (matched_df['proportions_matched'] == True)])
        no_matched_count = len(matched_df[(matched_df['formula_matched'] == False) & (matched_df['proportions_matched'] == False)])
        
        self._print(f"  • Formula 매칭: {formula_matched_count}개", level="info")
        self._print(f"  • Proportions 매칭: {proportions_matched_count}개", level="info")
        self._print(f"  • 둘 다 매칭: {both_matched_count}개", level="info")
        self._print(f"  • 매칭 없음: {no_matched_count}개", level="info")
        
        # 결과를 저장할 새로운 데이터프레임 생성
        modified_df = matched_df.copy()
        
        # formula_matched가 True이고 proportions_matched가 False인 행들 처리
        formula_only_rows = modified_df[
            (modified_df['formula_matched'] == True) & 
            (modified_df['proportions_matched'] == False)
        ]
        
        self._print(f"📊 Formula 매칭만 된 행 수: {len(formula_only_rows)}개", level="info")
        modified_df = self._calculate_formula_modified_coefficients(modified_df, formula_only_rows, max_case, num_tier, scenario)
        
        # proportions_matched가 True이고 formula_matched가 False인 행들 처리
        proportions_only_rows = modified_df[
            (modified_df['formula_matched'] == False) & 
            (modified_df['proportions_matched'] == True)
        ]
        
        self._print(f"📊 Proportions 매칭만 된 행 수: {len(proportions_only_rows)}개", level="info")
        modified_df = self._calculate_proportions_modified_coefficients(modified_df, proportions_only_rows, max_case, num_tier, scenario)
        
        return modified_df
    
    def _calculate_formula_modified_coefficients(self, result_df: pd.DataFrame, formula_only_rows: pd.DataFrame, max_case: int, num_tier: int, scenario: str) -> pd.DataFrame:
        """
        formula_matched가 True인 경우의 배출계수 수정 계산
        원래 배출계수에서 (Tier1 계수 * Tier1_RE_case% + Tier2 계수 * Tier2_RE_case% + ...)를 뺌

        Args:
            result_df: 결과 데이터프레임
            formula_only_rows: formula 매칭만 된 행들
            max_case: 계산할 최대 case 번호
            num_tier: 계산할 최대 tier 번호
            scenario: 현재 실행 중인 시나리오

        Returns:
            pd.DataFrame: 수정된 데이터프레임
        """
        self._print("=" * 100, level="info")
        self._print("🔍 [배출계수 매칭 디버그] formula_only_rows 분석", level="info")
        self._print(f"   - formula_only_rows 컬럼: {list(formula_only_rows.columns)}", level="info")
        self._print(f"   - '배출계수명' in formula_only_rows: {'배출계수명' in formula_only_rows.columns}", level="info")
        if self.original_df is not None:
            self._print(f"   - original_df 컬럼: {list(self.original_df.columns)}", level="info")
            self._print(f"   - '배출계수명' in original_df: {'배출계수명' in self.original_df.columns}", level="info")
            # 음극재 행 확인
            anode_in_original = self.original_df[self.original_df['자재품목'] == '음극재']
            self._print(f"   - original_df 중 음극재 행 수: {len(anode_in_original)}개", level="info")
            if not anode_in_original.empty:
                self._print(f"   - original_df 음극재 배출계수명 목록:", level="info")
                for idx, row in anode_in_original.iterrows():
                    self._print(f"      • {row.get('자재명', 'N/A')} | 배출계수명: {row.get('배출계수명', 'N/A')} | 배출계수: {row.get('배출계수', 0):.6f}", level="info")
        self._print("=" * 100, level="info")

        # PCF_reference를 위한 base 배출계수 저장용 열 추가 (Tier-case 적용 전)
        if '배출계수_reference_base' not in result_df.columns:
            result_df['배출계수_reference_base'] = result_df['배출계수']
            self._print("📝 '배출계수_reference_base' 열 생성 (PCF_reference 계산용)", level="info")
        for idx, row in formula_only_rows.iterrows():
            self._print(f"🔍 Formula 처리 중: 자재명={row.get('자재명', 'N/A')}, 자재품목={row.get('자재품목', 'N/A')}, 배출계수명={row.get('배출계수명', 'N/A')}", level="info")

            # 시나리오별로 업데이트된 배출계수 사용
            if self.original_df is not None:
                # original_df에서 해당 자재의 현재 배출계수 찾기
                # 배출계수명을 우선 매칭 조건으로 사용 (고유화된 배출계수명 활용)
                if pd.isna(row['자재명']):
                    # 자재명이 NaN인 경우 자재품목만으로 매칭
                    matching_original = self.original_df[
                        (self.original_df['자재명'].isna()) &
                        (self.original_df['자재품목'] == row['자재품목'])
                    ]
                else:
                    # 1차 시도: 배출계수명 포함하여 매칭 (가장 정확)
                    if '배출계수명' in row.index and not pd.isna(row['배출계수명']) and '배출계수명' in self.original_df.columns:
                        self._print(f"   🔍 1차 매칭 시도: 배출계수명 포함 매칭", level="info")
                        self._print(f"      - 조건: 자재명={row['자재명']}, 자재품목={row['자재품목']}, 배출계수명={row['배출계수명']}", level="info")
                        matching_original = self.original_df[
                            (self.original_df['자재명'] == row['자재명']) &
                            (self.original_df['자재품목'] == row['자재품목']) &
                            (self.original_df['배출계수명'] == row['배출계수명'])
                        ]
                        self._print(f"      - 매칭된 행 수: {len(matching_original)}개", level="info")
                        if matching_original.empty:
                            # 2차 시도: 배출계수명 없이 자재명+자재품목만으로 매칭
                            self._print(f"   🔍 2차 매칭 시도: 배출계수명 제외, 자재명+자재품목만", level="info")
                            matching_original = self.original_df[
                                (self.original_df['자재명'] == row['자재명']) &
                                (self.original_df['자재품목'] == row['자재품목'])
                            ]
                            self._print(f"      - 매칭된 행 수: {len(matching_original)}개", level="info")
                    else:
                        # 배출계수명이 없는 경우 자재명+자재품목으로 매칭
                        self._print(f"   🔍 배출계수명 없음: 자재명+자재품목만으로 매칭", level="info")
                        self._print(f"      - 배출계수명 in row: {'배출계수명' in row.index}", level="info")
                        self._print(f"      - 배출계수명 값: {row.get('배출계수명', 'N/A')}", level="info")
                        self._print(f"      - 배출계수명 in original_df: {'배출계수명' in self.original_df.columns}", level="info")
                        matching_original = self.original_df[
                            (self.original_df['자재명'] == row['자재명']) &
                            (self.original_df['자재품목'] == row['자재품목'])
                        ]
                        self._print(f"      - 매칭된 행 수: {len(matching_original)}개", level="info")

                if not matching_original.empty:
                    # 여러 행이 매칭된 경우, 배출계수명으로 추가 필터링하여 정확한 행 선택
                    if len(matching_original) > 1 and '배출계수명' in row.index and not pd.isna(row['배출계수명']):
                        self._print(f"   🔍 여러 행 매칭됨 ({len(matching_original)}개), 배출계수명으로 정확한 필터링 시도...", level="info")
                        exact_match = matching_original[matching_original['배출계수명'] == row['배출계수명']]
                        if not exact_match.empty:
                            matching_original = exact_match
                            self._print(f"   🎯 배출계수명으로 정확한 매칭 찾음! ({len(matching_original)}개 행)", level="info")
                        else:
                            self._print(f"   ⚠️ 배출계수명 정확 매칭 실패, 첫 번째 행 사용", level="warning")

                    # 여전히 여러 행이면 추가 필터링 수행
                    if len(matching_original) > 1:
                        self._print(f"   🔍 배출계수명 필터링 후에도 {len(matching_original)}개 행 존재, 추가 필터링 수행", level="info")
                        matching_original = self._select_best_matching_row(matching_original, row)

                    original_coeff = matching_original.iloc[0]['배출계수']
                    matched_emission_name = matching_original.iloc[0].get('배출계수명', 'N/A')
                    matched_material_name = matching_original.iloc[0].get('자재명', 'N/A')
                    matched_quantity = matching_original.iloc[0].get('제품총소요량(kg)', 'N/A')
                    matched_emission = matching_original.iloc[0].get('배출량(kgCO2eq)', 'N/A')
                    self._print(f"✅ {scenario} 매칭 성공: 배출계수={original_coeff:.6f}, 매칭된_배출계수명={matched_emission_name}, 매칭된_자재명={matched_material_name}", level="info")
                    self._print(f"   - 소요량={matched_quantity}, 배출량={matched_emission}", level="info")
                else:
                    original_coeff = row['배출계수']
                    self._print(f"❌ original_df에서 찾을 수 없어 row의 기본 배출계수 사용: {original_coeff:.6f}", level="warning")
                    self._print(f"   - 요청 자재명: {row.get('자재명', 'N/A')}", level="warning")
                    self._print(f"   - 요청 자재품목: {row.get('자재품목', 'N/A')}", level="warning")
                    self._print(f"   - 요청 배출계수명: {row.get('배출계수명', 'N/A')}", level="warning")
            else:
                original_coeff = row['배출계수']
                self._print(f"❌ original_df가 없어 row의 기본 배출계수 사용: {original_coeff:.6f}", level="warning")

            self._print(f"📊 최종 사용 배출계수: {original_coeff:.6f}", level="info")
            self._print("=" * 80, level="info")

            # 디버깅: tier_values 열들의 실제 값 확인 (디버그 모드에서만)
            if self.verbose == "debug":
                self._print(f"🔍 Tier 값들 확인:", level="debug")
                for tier_num in range(1, num_tier + 1):
                    tier_values_col = f'tier{tier_num}_values'
                    tier_values = row.get(tier_values_col, None)
                    self._print(f"  • {tier_values_col}: {tier_values} (타입: {type(tier_values)})", level="debug")
            
            # Tier 값들을 동적으로 추출
            tier_coeffs, actual_tier_count = self._extract_tier_coefficients(row, num_tier)

            # tier_num_by 열에 실제 사용된 tier 개수 기록
            result_df.loc[idx, 'tier_num_by'] = actual_tier_count
            self._print(f"📊 실제 사용된 Tier 개수: {actual_tier_count}", level="info")

            # 1단계: 모든 case를 먼저 검사해서 fallback 필요 여부 확인
            need_fallback = False
            fallback_trigger_case = None

            for case_num in range(1, max_case + 1):
                tier_re_values = self._extract_tier_re_values(row, case_num, num_tier)
                reduction_amount = 0
                for tier_num in range(1, num_tier + 1):
                    tier_coeff = tier_coeffs.get(f'tier{tier_num}', 0)
                    tier_re_value = tier_re_values.get(f'tier{tier_num}', 0)
                    reduction_amount += tier_coeff * tier_re_value

                modified_coeff = original_coeff - reduction_amount

                # 음수가 되는 case가 하나라도 있으면 전체 fallback 필요
                if modified_coeff < 0:
                    need_fallback = True
                    fallback_trigger_case = case_num
                    self._print(f"    ⚠️ Case {case_num}에서 음수 배출계수 감지 ({modified_coeff:.6f})", level="warning")
                    break

            # Fallback 필요 시 경고 메시지
            if need_fallback:
                material_name = row.get('자재명', 'N/A')
                material_category = row.get('자재품목', 'N/A')
                self._print(f"🔄 ⚠️ FALLBACK TRIGGERED: {material_name} ({material_category})", level="warning")
                self._print(f"    - Case {fallback_trigger_case}에서 음수 발생으로 전체 케이스에 Proportion 로직 적용", level="warning")
                self._print(f"    - 원본 배출계수: {original_coeff:.6f}", level="warning")

                # 세션 상태에 fallback 경고 저장 (Streamlit에서 표시용)
                if not hasattr(self, 'fallback_warnings'):
                    self.fallback_warnings = []
                self.fallback_warnings.append({
                    '자재명': material_name,
                    '자재품목': material_category,
                    '원본_배출계수': original_coeff,
                    'trigger_case': fallback_trigger_case,
                    '배출계수명': row.get('배출계수명', 'N/A')
                })

            # 2단계: 각 case별로 계산 (fallback 여부에 따라 로직 분기)
            for case_num in range(1, max_case + 1):
                if self.verbose == "debug":
                    self._print(f"  📋 Case {case_num} 계산:")

                # Tier RE 값들을 동적으로 추출
                tier_re_values = self._extract_tier_re_values(row, case_num, num_tier)

                if need_fallback:
                    self._print(f"    ⚠️ 수정된 배출계수가 음수({modified_coeff:.6f})가 되어 proportion 로직으로 fallback", level="warning")
                    self._print(f"    🔄 Proportion 로직 적용: 배출계수 × (1 - Σ(tier_value × case%))", level="info")

                    # Proportion 로직 적용: 배출계수 * (1 - tier1_values의 value% * case% - tier2_values의 value% * case% - ...)
                    total_reduction_rate = 0
                    for tier_num in range(1, num_tier + 1):
                        tier_values_col = f'tier{tier_num}_values'
                        tier_re_col = f'Tier{tier_num}_RE_case{case_num}'

                        # tier_values에서 값 추출 및 안전하게 변환
                        tier_values = row.get(tier_values_col, None)
                        if tier_values is not None and isinstance(tier_values, dict):
                            try:
                                # Proportion 매칭 시 저장된 키 이름: 'Tier1_RE100(%)', 'Tier2_RE100(%)'
                                tier_key = f'Tier{tier_num}_RE100(%)'
                                tier_value = tier_values.get(tier_key)

                                # 키가 없으면 딕셔너리의 첫 번째 값 사용 (일반적인 경우)
                                if tier_value is None:
                                    if len(tier_values) > 0:
                                        tier_value = list(tier_values.values())[0]
                                    else:
                                        tier_value = 0

                                if isinstance(tier_value, str):
                                    tier_value = float(tier_value.replace('%', '').strip())
                                else:
                                    tier_value = float(tier_value)
                                tier_value_percent = tier_value / 100  # 퍼센트를 소수로 변환

                                if self.verbose == "debug":
                                    self._print(f"      🔍 Tier{tier_num}_values: {tier_values} → tier_value: {tier_value} → {tier_value_percent:.4f}", level="debug")
                            except (ValueError, AttributeError, TypeError) as e:
                                tier_value_percent = 0
                                self._print(f"      ⚠️ Tier{tier_num} 값 변환 실패: {e}, tier_values={tier_values}", level="warning")
                        else:
                            tier_value_percent = 0
                            if self.verbose == "debug":
                                self._print(f"      🔍 Tier{tier_num}_values: None 또는 dict 아님", level="debug")

                        # tier_re_value 추출 및 숫자로 변환
                        tier_re_value = row.get(tier_re_col, 0)

                        # 문자열인 경우 숫자로 변환 (안전하게)
                        try:
                            if isinstance(tier_re_value, str):
                                # '%' 기호 제거하고 소수로 변환
                                tier_re_value = float(tier_re_value.replace('%', '').strip())
                                # 퍼센트 형태면 100으로 나누기
                                if tier_re_value > 1:
                                    tier_re_value = tier_re_value / 100
                            else:
                                tier_re_value = float(tier_re_value)
                        except (ValueError, AttributeError):
                            tier_re_value = 0

                        # 누적 감소율 계산
                        tier_reduction = tier_value_percent * tier_re_value
                        total_reduction_rate += tier_reduction

                        if self.verbose == "debug":
                            self._print(f"      📊 Tier{tier_num} 감소율: {tier_value_percent:.4f} × {tier_re_value:.4f} = {tier_reduction:.4f}", level="debug")

                    # Proportion 방식으로 수정된 배출계수 계산
                    modified_coeff = original_coeff * (1 - total_reduction_rate)

                    self._print(f"    📊 총 감소율: {total_reduction_rate:.4f} ({total_reduction_rate * 100:.2f}%)", level="info")
                    self._print(f"    📊 Proportion으로 계산된 배출계수: {modified_coeff:.4f}", level="info")

                    # 여전히 음수인 경우 최소값으로 보정 (안전장치)
                    if modified_coeff < 0:
                        min_threshold = original_coeff * 0.01  # 원본의 1%
                        modified_coeff = max(min_threshold, 0.01)  # 최소 0.01로 설정
                        self._print(f"    ⚠️ Proportion 적용 후에도 음수가 되어 최소값({modified_coeff:.6f})으로 보정", level="warning")
                else:
                    # Formula 로직: 원래 배출계수에서 감소량을 뺌
                    reduction_amount = 0
                    for tier_num in range(1, num_tier + 1):
                        tier_coeff = tier_coeffs.get(f'tier{tier_num}', 0)
                        tier_re_value = tier_re_values.get(f'tier{tier_num}', 0)
                        reduction_amount += tier_coeff * tier_re_value

                        if self.verbose == "debug":
                            self._print(f"      📊 Tier{tier_num}: {tier_coeff:.4f} × {tier_re_value:.4f} = {tier_coeff * tier_re_value:.4f}", level="debug")

                    modified_coeff = original_coeff - reduction_amount
                    self._print(f"    📊 감소량: {reduction_amount:.4f}", level="info")
                    self._print(f"    📊 수정된 배출계수: {modified_coeff:.4f}", level="info")

                # 결과 데이터프레임에 수정된 배출계수 추가
                result_df.loc[idx, f'modified_coeff_case{case_num}'] = modified_coeff

            # 모든 case 계산 완료 후 비교 검증
            if max_case >= 2:
                case1_coeff = result_df.loc[idx, 'modified_coeff_case1']
                case2_coeff = result_df.loc[idx, 'modified_coeff_case2']

                # Case 2가 Case 1보다 더 큰 감소를 보이는지 확인 (배출계수가 더 작음)
                if case2_coeff < case1_coeff:
                    material_name = row.get('자재명', 'N/A')
                    material_category = row.get('자재품목', 'N/A')
                    quantity = row.get('제품총소요량(kg)', 0)

                    # PCF 계산
                    pcf_case1 = case1_coeff * quantity
                    pcf_case2 = case2_coeff * quantity
                    pcf_diff = pcf_case1 - pcf_case2

                    self._print(f"⚠️ ANOMALY DETECTED: Case 2 배출계수가 Case 1보다 작음", level="warning")
                    self._print(f"   자재명: {material_name}, 자재품목: {material_category}", level="warning")
                    self._print(f"   원본 배출계수: {original_coeff:.6f}", level="warning")
                    self._print(f"   Case 1 배출계수: {case1_coeff:.6f} → PCF: {pcf_case1:.6f}", level="warning")
                    self._print(f"   Case 2 배출계수: {case2_coeff:.6f} → PCF: {pcf_case2:.6f}", level="warning")
                    self._print(f"   PCF 차이: {pcf_diff:.6f} kgCO2eq (Case 2가 {pcf_diff:.6f} 더 낮음)", level="warning")

                    # RE 값 확인
                    for tier_num in range(1, num_tier + 1):
                        tier_re_col1 = f'Tier{tier_num}_RE_case1'
                        tier_re_col2 = f'Tier{tier_num}_RE_case2'
                        re1_raw = row.get(tier_re_col1, 'N/A')
                        re2_raw = row.get(tier_re_col2, 'N/A')
                        self._print(f"   Tier{tier_num}: Case1={re1_raw}, Case2={re2_raw}", level="warning")

        self._print(f"✅ Formula 배출계수 수정 완료: {len(formula_only_rows)}개 행 처리", level="info")
        return result_df
    
    def _update_ref_formula_with_site_data(self, site_coefficient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        사이트별 전력 배출계수 데이터를 사용하여 ref_formula_df를 업데이트합니다.
        
        Args:
            site_coefficient_data: 사이트별 전력 배출계수 데이터
            
        Returns:
            pd.DataFrame: 업데이트된 ref_formula_df
        """
        self._print("🔧 ref_formula_df를 사이트별 전력 배출계수로 업데이트 중...", level="info")
        
        # ref_formula_df 복사
        updated_ref_formula_df = self.ref_formula_df.copy()
        
        # Energy(Tier-1) 전력 배출계수 업데이트
        if 'Energy(Tier-1)' in site_coefficient_data and '전력' in site_coefficient_data['Energy(Tier-1)']:
            tier1_electricity_factor = site_coefficient_data['Energy(Tier-1)']['전력']['배출계수']
            
            # ref_formula_df에서 전력 관련 행 찾기 - 자재품목이 '전력'인 행 찾기
            tier1_mask = updated_ref_formula_df['자재품목'].str.contains('전력', case=False, na=False)
            if tier1_mask.any():
                updated_ref_formula_df.loc[tier1_mask, 'Tier1_RE100(kgCO2eq/kg)'] = tier1_electricity_factor
                self._print(f"📊 Energy(Tier-1) 전력 배출계수 업데이트: {tier1_electricity_factor}", level="info")
        
        # Energy(Tier-2) 전력 배출계수 업데이트
        if 'Energy(Tier-2)' in site_coefficient_data and '전력' in site_coefficient_data['Energy(Tier-2)']:
            tier2_electricity_factor = site_coefficient_data['Energy(Tier-2)']['전력']['배출계수']
            
            # ref_formula_df에서 전력 관련 행 찾기 - 자재품목이 '전력'인 행 찾기
            tier2_mask = updated_ref_formula_df['자재품목'].str.contains('전력', case=False, na=False)
            if tier2_mask.any():
                updated_ref_formula_df.loc[tier2_mask, 'Tier2_RE100(kgCO2eq/kg)'] = tier2_electricity_factor
                self._print(f"📊 Energy(Tier-2) 전력 배출계수 업데이트: {tier2_electricity_factor}", level="info")
        
        self._print("✅ ref_formula_df 사이트별 전력 배출계수 업데이트 완료", level="info")
        return updated_ref_formula_df

    def _update_original_df_with_site_data(self, original_df: pd.DataFrame, site_coefficient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        사이트별 전력 배출계수 데이터를 사용하여 original_df를 업데이트합니다.
        Energy(Tier-1)과 Energy(Tier-2) 카테고리의 전력 배출계수를 업데이트합니다.

        Args:
            original_df: 업데이트할 original_df
            site_coefficient_data: 사이트별 전력 배출계수 데이터

        Returns:
            pd.DataFrame: 업데이트된 original_df
        """
        self._print("🔧 original_df를 사이트별 전력 배출계수로 업데이트 중...", level="info")

        if original_df is None or original_df.empty:
            self._print("⚠️ original_df가 비어있어 업데이트를 건너뜁니다.", level="warning")
            return original_df

        # original_df 복사
        updated_original_df = original_df.copy()

        # Energy(Tier-1) 전력 배출계수 업데이트
        if 'Energy(Tier-1)' in site_coefficient_data and '전력' in site_coefficient_data['Energy(Tier-1)']:
            tier1_electricity_factor = site_coefficient_data['Energy(Tier-1)']['전력']['배출계수']

            # original_df에서 Energy(Tier-1) 전력 관련 행 찾기
            # 자재품목에 'Energy' 포함하고 'Tier-1' 또는 'Tier1' 포함하는 행
            tier1_energy_mask = (
                updated_original_df['자재품목'].str.contains('Energy', case=False, na=False) &
                (updated_original_df['자재품목'].str.contains('Tier-1', case=False, na=False) |
                 updated_original_df['자재품목'].str.contains('Tier1', case=False, na=False))
            )

            if tier1_energy_mask.any():
                before_values = updated_original_df.loc[tier1_energy_mask, '배출계수'].values
                updated_original_df.loc[tier1_energy_mask, '배출계수'] = tier1_electricity_factor

                # 배출량도 업데이트 (배출계수 × 제품총소요량)
                if '제품총소요량(kg)' in updated_original_df.columns:
                    updated_original_df.loc[tier1_energy_mask, '배출량(kgCO2eq)'] = (
                        updated_original_df.loc[tier1_energy_mask, '배출계수'] *
                        updated_original_df.loc[tier1_energy_mask, '제품총소요량(kg)']
                    )

                count = tier1_energy_mask.sum()
                self._print(f"📊 Energy(Tier-1) 전력 배출계수 업데이트: {count}개 행", level="info")
                self._print(f"   - 이전 평균: {before_values.mean():.6f}", level="info")
                self._print(f"   - 업데이트 값: {tier1_electricity_factor:.6f}", level="info")

        # Energy(Tier-2) 전력 배출계수 업데이트
        if 'Energy(Tier-2)' in site_coefficient_data and '전력' in site_coefficient_data['Energy(Tier-2)']:
            tier2_electricity_factor = site_coefficient_data['Energy(Tier-2)']['전력']['배출계수']

            # original_df에서 Energy(Tier-2) 전력 관련 행 찾기
            tier2_energy_mask = (
                updated_original_df['자재품목'].str.contains('Energy', case=False, na=False) &
                (updated_original_df['자재품목'].str.contains('Tier-2', case=False, na=False) |
                 updated_original_df['자재품목'].str.contains('Tier2', case=False, na=False))
            )

            if tier2_energy_mask.any():
                before_values = updated_original_df.loc[tier2_energy_mask, '배출계수'].values
                updated_original_df.loc[tier2_energy_mask, '배출계수'] = tier2_electricity_factor

                # 배출량도 업데이트 (배출계수 × 제품총소요량)
                if '제품총소요량(kg)' in updated_original_df.columns:
                    updated_original_df.loc[tier2_energy_mask, '배출량(kgCO2eq)'] = (
                        updated_original_df.loc[tier2_energy_mask, '배출계수'] *
                        updated_original_df.loc[tier2_energy_mask, '제품총소요량(kg)']
                    )

                count = tier2_energy_mask.sum()
                self._print(f"📊 Energy(Tier-2) 전력 배출계수 업데이트: {count}개 행", level="info")
                self._print(f"   - 이전 평균: {before_values.mean():.6f}", level="info")
                self._print(f"   - 업데이트 값: {tier2_electricity_factor:.6f}", level="info")

        self._print("✅ original_df 사이트별 전력 배출계수 업데이트 완료", level="info")
        return updated_original_df

    def _calculate_proportions_modified_coefficients(self, result_df: pd.DataFrame, proportions_only_rows: pd.DataFrame, max_case: int, num_tier: int, scenario: str) -> pd.DataFrame:
        """
        proportions_matched가 True인 경우의 배출계수 수정 계산
        배출계수 * (1 - tier1_values의 value% * case% - tier2_values의 value% * case% - ...)
        
        Args:
            result_df: 결과 데이터프레임
            proportions_only_rows: proportions 매칭만 된 행들
            max_case: 계산할 최대 case 번호
            num_tier: 계산할 최대 tier 번호
            scenario: 현재 실행 중인 시나리오
            
        Returns:
            pd.DataFrame: 수정된 데이터프레임
        """
        for idx, row in proportions_only_rows.iterrows():
            self._print(f"🔍 Proportions 처리 중: 자재명={row.get('자재명', 'N/A')}, 자재품목={row.get('자재품목', 'N/A')}, 배출계수명={row.get('배출계수명', 'N/A')}", level="info")

            # 시나리오별로 업데이트된 배출계수 사용
            if self.original_df is not None:
                # original_df에서 해당 자재의 현재 배출계수 찾기
                # 배출계수명을 우선 매칭 조건으로 사용 (고유화된 배출계수명 활용)
                if pd.isna(row['자재명']):
                    # 자재명이 NaN인 경우 자재품목만으로 매칭
                    matching_original = self.original_df[
                        (self.original_df['자재명'].isna()) &
                        (self.original_df['자재품목'] == row['자재품목'])
                    ]
                else:
                    # 1차 시도: 배출계수명 포함하여 매칭 (가장 정확)
                    if '배출계수명' in row.index and not pd.isna(row['배출계수명']) and '배출계수명' in self.original_df.columns:
                        self._print(f"   🔍 1차 매칭 시도: 배출계수명 포함 매칭", level="info")
                        self._print(f"      - 조건: 자재명={row['자재명']}, 자재품목={row['자재품목']}, 배출계수명={row['배출계수명']}", level="info")
                        matching_original = self.original_df[
                            (self.original_df['자재명'] == row['자재명']) &
                            (self.original_df['자재품목'] == row['자재품목']) &
                            (self.original_df['배출계수명'] == row['배출계수명'])
                        ]
                        self._print(f"      - 매칭된 행 수: {len(matching_original)}개", level="info")
                        if matching_original.empty:
                            # 2차 시도: 배출계수명 없이 자재명+자재품목만으로 매칭
                            self._print(f"   🔍 2차 매칭 시도: 배출계수명 제외, 자재명+자재품목만", level="info")
                            matching_original = self.original_df[
                                (self.original_df['자재명'] == row['자재명']) &
                                (self.original_df['자재품목'] == row['자재품목'])
                            ]
                            self._print(f"      - 매칭된 행 수: {len(matching_original)}개", level="info")
                    else:
                        # 배출계수명이 없는 경우 자재명+자재품목으로 매칭
                        self._print(f"   🔍 배출계수명 없음: 자재명+자재품목만으로 매칭", level="info")
                        self._print(f"      - 배출계수명 in row: {'배출계수명' in row.index}", level="info")
                        self._print(f"      - 배출계수명 값: {row.get('배출계수명', 'N/A')}", level="info")
                        self._print(f"      - 배출계수명 in original_df: {'배출계수명' in self.original_df.columns}", level="info")
                        matching_original = self.original_df[
                            (self.original_df['자재명'] == row['자재명']) &
                            (self.original_df['자재품목'] == row['자재품목'])
                        ]
                        self._print(f"      - 매칭된 행 수: {len(matching_original)}개", level="info")

                if not matching_original.empty:
                    # 여러 행이 매칭된 경우, 배출계수명으로 추가 필터링하여 정확한 행 선택
                    if len(matching_original) > 1 and '배출계수명' in row.index and not pd.isna(row['배출계수명']):
                        self._print(f"   🔍 여러 행 매칭됨 ({len(matching_original)}개), 배출계수명으로 정확한 필터링 시도...", level="info")
                        exact_match = matching_original[matching_original['배출계수명'] == row['배출계수명']]
                        if not exact_match.empty:
                            matching_original = exact_match
                            self._print(f"   🎯 배출계수명으로 정확한 매칭 찾음! ({len(matching_original)}개 행)", level="info")
                        else:
                            self._print(f"   ⚠️ 배출계수명 정확 매칭 실패, 첫 번째 행 사용", level="warning")

                    # 여전히 여러 행이면 추가 필터링 수행
                    if len(matching_original) > 1:
                        self._print(f"   🔍 배출계수명 필터링 후에도 {len(matching_original)}개 행 존재, 추가 필터링 수행", level="info")
                        matching_original = self._select_best_matching_row(matching_original, row)

                    original_coeff = matching_original.iloc[0]['배출계수']
                    matched_emission_name = matching_original.iloc[0].get('배출계수명', 'N/A')
                    matched_material_name = matching_original.iloc[0].get('자재명', 'N/A')
                    matched_quantity = matching_original.iloc[0].get('제품총소요량(kg)', 'N/A')
                    matched_emission = matching_original.iloc[0].get('배출량(kgCO2eq)', 'N/A')
                    self._print(f"✅ {scenario} 매칭 성공: 배출계수={original_coeff:.6f}, 매칭된_배출계수명={matched_emission_name}, 매칭된_자재명={matched_material_name}", level="info")
                    self._print(f"   - 소요량={matched_quantity}, 배출량={matched_emission}", level="info")
                else:
                    original_coeff = row['배출계수']
                    self._print(f"❌ original_df에서 찾을 수 없어 row의 기본 배출계수 사용: {original_coeff:.6f}", level="warning")
                    self._print(f"   - 요청 자재명: {row.get('자재명', 'N/A')}", level="warning")
                    self._print(f"   - 요청 자재품목: {row.get('자재품목', 'N/A')}", level="warning")
                    self._print(f"   - 요청 배출계수명: {row.get('배출계수명', 'N/A')}", level="warning")
            else:
                original_coeff = row['배출계수']
                self._print(f"❌ original_df가 없어 row의 기본 배출계수 사용: {original_coeff:.6f}", level="warning")

            self._print(f"📊 최종 사용 배출계수: {original_coeff:.6f}", level="info")
            self._print("=" * 80, level="info")

            # Tier 값들을 동적으로 추출
            tier_coeffs, actual_tier_count = self._extract_tier_coefficients(row, num_tier)
            
            # tier_num_by 열에 실제 사용된 tier 개수 기록
            result_df.loc[idx, 'tier_num_by'] = actual_tier_count
            self._print(f"📊 실제 사용된 Tier 개수: {actual_tier_count}", level="info")
            
            # 각 case별로 계산
            for case_num in range(1, max_case + 1):
                if self.verbose == "debug":
                    self._print(f"  📋 Case {case_num} 계산:")
                
                # Tier RE 값들을 동적으로 추출
                tier_re_values = self._extract_tier_re_values(row, case_num, num_tier)
                
                # 수정된 배출계수 계산
                # 배출계수 * (1 - tier1_values의 value% * case% - tier2_values의 value% * case% - ...)
                reduction_factor = 1.0
                total_reduction = 0
                for tier_num in range(1, num_tier + 1):
                    tier_coeff = tier_coeffs.get(f'tier{tier_num}', 0)
                    tier_re_value = tier_re_values.get(f'tier{tier_num}', 0)
                    # tier_coeff는 이미 % 단위이므로 100으로 나누어 소수로 변환
                    tier_coeff_decimal = tier_coeff / 100
                    reduction_amount = tier_coeff_decimal * tier_re_value
                    reduction_factor -= reduction_amount
                    total_reduction += reduction_amount
                    self._print(f"    📊 Tier{tier_num} 감소량: {tier_coeff_decimal:.4f} * {tier_re_value:.4f} = {reduction_amount:.4f}", level="info")

                # 음수 처리 개선: 감소 계수가 100%를 초과하는 경우 경고
                if total_reduction > 1.0:
                    reduction_rate = total_reduction * 100
                    self._print(f"    ⚠️ 총 감소율({reduction_rate:.1f}%)이 100%를 초과", level="warning")
                    self._print(f"       자재명: {row.get('자재명', 'N/A')}, 자재품목: {row.get('자재품목', 'N/A')}", level="warning")

                # 음수가 되지 않도록 보정: 최소 감소 계수를 1%로 설정
                if reduction_factor < 0:
                    min_factor = 0.01  # 최소 1%로 설정
                    reduction_factor = min_factor
                    self._print(f"    ⚠️ 감소 계수가 음수가 되어 최소값({min_factor:.4f})으로 보정", level="warning")

                modified_coeff = original_coeff * reduction_factor
                
                self._print(f"    📊 최종 감소 계수: {reduction_factor:.4f}", level="info")
                self._print(f"    📊 수정된 배출계수: {modified_coeff:.4f}", level="info")
                
                # 결과 데이터프레임에 수정된 배출계수 추가
                result_df.loc[idx, f'modified_coeff_case{case_num}'] = modified_coeff
        
        self._print(f"✅ Proportions 배출계수 수정 완료: {len(proportions_only_rows)}개 행 처리", level="info")
        return result_df
    
    def _select_best_matching_row(self, matching_rows: pd.DataFrame, target_row: pd.Series) -> pd.DataFrame:
        """
        여러 행이 매칭된 경우 가장 적합한 행을 선택하는 헬퍼 함수

        Args:
            matching_rows: 매칭된 original_df 행들
            target_row: scenario_df의 목표 행

        Returns:
            pd.DataFrame: 선택된 단일 행 (여전히 여러 개면 첫 번째)
        """
        if len(matching_rows) <= 1:
            return matching_rows

        self._print(f"   🔍 {len(matching_rows)}개 행 중 최적 행 선택 시작...", level="info")
        candidates = matching_rows.copy()

        # 1단계: 제품총소요량(kg) 일치 필터링
        if '제품총소요량(kg)' in target_row.index and '제품총소요량(kg)' in candidates.columns:
            target_quantity = target_row['제품총소요량(kg)']
            if not pd.isna(target_quantity):
                quantity_match = candidates[candidates['제품총소요량(kg)'] == target_quantity]
                if not quantity_match.empty:
                    self._print(f"   ✓ 제품총소요량 일치: {len(quantity_match)}개 행으로 축소", level="info")
                    candidates = quantity_match
                    if len(candidates) == 1:
                        self._print(f"   🎯 제품총소요량으로 단일 행 선택 완료", level="info")
                        return candidates

        # 2단계: 배출량(kgCO2eq) 일치 필터링
        if '배출량(kgCO2eq)' in target_row.index and '배출량(kgCO2eq)' in candidates.columns:
            target_emission = target_row['배출량(kgCO2eq)']
            if not pd.isna(target_emission):
                emission_match = candidates[candidates['배출량(kgCO2eq)'] == target_emission]
                if not emission_match.empty:
                    self._print(f"   ✓ 배출량({target_emission:.6f} kgCO2eq) 일치: {len(emission_match)}개 행으로 축소", level="info")
                    candidates = emission_match
                    if len(candidates) == 1:
                        self._print(f"   🎯 배출량으로 단일 행 선택 완료", level="info")
                        return candidates

        # 3단계: 지역 일치 필터링
        if '지역' in target_row.index and '지역' in candidates.columns:
            target_region = target_row['지역']
            if not pd.isna(target_region):
                region_match = candidates[candidates['지역'] == target_region]
                if not region_match.empty:
                    self._print(f"   ✓ 지역 일치: {len(region_match)}개 행으로 축소", level="info")
                    candidates = region_match
                    if len(candidates) == 1:
                        self._print(f"   🎯 지역으로 단일 행 선택 완료", level="info")
                        return candidates

        # 4단계: 배출계수 정확도 필터링 (가장 가까운 값)
        if '배출계수' in target_row.index and '배출계수' in candidates.columns:
            target_coeff = target_row['배출계수']
            if not pd.isna(target_coeff):
                # 배출계수 차이가 가장 작은 행 찾기
                candidates['_coeff_diff'] = abs(candidates['배출계수'] - target_coeff)
                min_diff = candidates['_coeff_diff'].min()
                exact_match = candidates[candidates['_coeff_diff'] == min_diff]
                candidates = exact_match.drop(columns=['_coeff_diff'])

                if min_diff == 0:
                    self._print(f"   ✓ 배출계수 정확히 일치: {len(candidates)}개 행", level="info")
                else:
                    self._print(f"   ✓ 배출계수 근사 일치 (차이: {min_diff:.6f}): {len(candidates)}개 행", level="info")

                if len(candidates) == 1:
                    self._print(f"   🎯 배출계수 정확도로 단일 행 선택 완료", level="info")
                    return candidates

        # 여전히 여러 개면 첫 번째 사용 (경고)
        if len(candidates) > 1:
            self._print(f"   ⚠️ 모든 필터링 후에도 {len(candidates)}개 행 남음, 첫 번째 행 사용", level="warning")
            self._print(f"   📋 남은 행들의 주요 정보:", level="warning")
            for i, (idx, row) in enumerate(candidates.iterrows()):
                self._print(f"     [{i}] 소요량={row.get('제품총소요량(kg)', 'N/A')}, "
                          f"배출량={row.get('배출량(kgCO2eq)', 'N/A')}, "
                          f"지역={row.get('지역', 'N/A')}, "
                          f"배출계수={row.get('배출계수', 0):.6f}", level="warning")

        return candidates.iloc[[0]]

    def _extract_tier_coefficients(self, row: pd.Series, num_tier: int) -> Tuple[Dict[str, float], int]:
        """
        Tier 계수들을 추출하는 헬퍼 함수

        Args:
            row: 데이터프레임 행
            num_tier: 계산할 최대 tier 번호

        Returns:
            Tuple[Dict[str, float], int]: (tier 계수들, 실제 사용된 tier 개수)
        """
        tier_coeffs = {}
        actual_tier_count = 0

        for tier_num in range(1, num_tier + 1):
            tier_values_col = f'tier{tier_num}_values'
            tier_values = row.get(tier_values_col, {})
            tier_coeff = 0

            if tier_values:
                if isinstance(tier_values, dict) and len(tier_values) > 0:
                    # 딕셔너리의 첫 번째 값 추출
                    tier_coeff = list(tier_values.values())[0]
                elif isinstance(tier_values, (list, tuple)) and len(tier_values) > 0:
                    # 리스트나 튜플의 첫 번째 값 추출
                    tier_coeff = tier_values[0]
                elif isinstance(tier_values, (int, float)):
                    # 숫자 값 그대로 사용
                    tier_coeff = float(tier_values)
                else:
                    self._print(f"  • ⚠️ Tier{tier_num} 지원하지 않는 형식 = {type(tier_values)}", level="warning")

                tier_coeffs[f'tier{tier_num}'] = tier_coeff
                actual_tier_count = tier_num
            else:
                tier_coeffs[f'tier{tier_num}'] = 0

        # 요약 정보만 출력
        if actual_tier_count > 0:
            coeff_summary = ", ".join([f"Tier{i}={tier_coeffs[f'tier{i}']:.4f}" for i in range(1, actual_tier_count + 1)])
            self._print(f"📊 Tier 계수 추출: {coeff_summary}", level="info")
        else:
            self._print(f"📊 Tier 계수: 값 없음", level="info")

        return tier_coeffs, actual_tier_count
    
    def _extract_tier_re_values(self, row: pd.Series, case_num: int, num_tier: int) -> Dict[str, float]:
        """
        Tier RE 값들을 추출하는 헬퍼 함수

        Args:
            row: 데이터프레임 행
            case_num: case 번호
            num_tier: 계산할 최대 tier 번호

        Returns:
            Dict[str, float]: tier RE 값들
        """
        tier_re_values = {}
        warnings = []

        for tier_num in range(1, num_tier + 1):
            tier_re_col_name = f'Tier{tier_num}_RE_case{case_num}'
            tier_re_raw = row.get(tier_re_col_name, '0%')

            # 다양한 형식 처리
            if pd.isna(tier_re_raw):
                tier_re_str = '0%'
                warnings.append(f"Tier{tier_num}: NaN→0%")
            elif isinstance(tier_re_raw, (int, float)):
                tier_re_str = f"{tier_re_raw}%"
            else:
                tier_re_str = str(tier_re_raw)

            # % 기호 제거 및 숫자 변환
            try:
                tier_re_value = float(tier_re_str.replace('%', '')) / 100
            except ValueError:
                tier_re_value = 0.0
                warnings.append(f"Tier{tier_num}: 변환실패→0.0")

            tier_re_values[f'tier{tier_num}'] = tier_re_value

        # 요약 정보 출력
        re_summary = ", ".join([f"Tier{i}={tier_re_values[f'tier{i}']*100:.1f}%" for i in range(1, num_tier + 1)])
        self._print(f"📊 Case{case_num} RE 값: {re_summary}", level="info")

        # 경고가 있으면 출력
        if warnings:
            self._print(f"  ⚠️ {', '.join(warnings)}", level="warning")

        return tier_re_values
    
    def calculate_pcf_values(self, modified_df: pd.DataFrame, max_case: int = 3, num_tier: int = 3) -> pd.DataFrame:
        """
        수정된 배출계수와 제품총소요량을 곱해서 PCF 값을 계산
        
        Args:
            modified_df: calculate_modified_coefficients 함수의 결과 데이터프레임
            max_case: 계산할 최대 case 번호 (기본값: 3)
            num_tier: 계산할 최대 tier 번호 (기본값: 3)
            
        Returns:
            pd.DataFrame: PCF 값이 추가된 데이터프레임
        """
        self._print(f"🔍 PCF 값 계산 (Case 1 ~ {max_case}, Tier 1 ~ {num_tier})", level="info")
        
        if len(modified_df) == 0:
            self._print("❌ 처리할 데이터가 없습니다.", level="info")
            return modified_df
        
        # 디버그: PCF 계산 전 양극재/음극재 데이터 확인
        cathode_rows = modified_df[modified_df['자재품목'] == '양극재']
        anode_rows = modified_df[modified_df['자재품목'] == '음극재']
        
        self._print("=== PCF 계산 양극재/음극재 디버그 ===", level="info")
        self._print(f"modified_df 중 양극재: {len(cathode_rows)}개", level="info")
        self._print(f"modified_df 중 음극재: {len(anode_rows)}개", level="info")
        
        if not cathode_rows.empty:
            self._print("양극재 PCF 계산 데이터:", level="info")
            for idx, row in cathode_rows.iterrows():
                quantity = row.get('제품총소요량(kg)', 0)
                emission_coef = row.get('배출계수', 0)
                self._print(f"  • {row.get('자재명', 'N/A')}: 소요량={quantity:.6f}kg, 배출계수={emission_coef:.6f}", level="info")
        
        if not anode_rows.empty:
            self._print("음극재 PCF 계산 데이터:", level="info")
            for idx, row in anode_rows.iterrows():
                quantity = row.get('제품총소요량(kg)', 0)
                emission_coef = row.get('배출계수', 0)
                self._print(f"  • {row.get('자재명', 'N/A')}: 소요량={quantity:.6f}kg, 배출계수={emission_coef:.6f}", level="info")
        
        self._print("=====================================", level="info")
        
        # 결과를 저장할 새로운 데이터프레임 생성
        pcf_result_df = modified_df.copy()
        
        # modified_coeff_case 열들을 자동으로 감지
        modified_coeff_columns = [col for col in modified_df.columns if col.startswith('modified_coeff_case')]
        
        if len(modified_coeff_columns) == 0:
            self._print("❌ modified_coeff_case 열을 찾을 수 없습니다.", level="info")
            return modified_df
        
        self._print(f"📊 발견된 modified_coeff_case 열: {modified_coeff_columns}", level="info")
        
        # 제품총소요량(kg) 열 확인
        if '제품총소요량(kg)' not in modified_df.columns:
            self._print("❌ '제품총소요량(kg)' 열을 찾을 수 없습니다.", level="info")
            return modified_df

        # 각 case별로 PCF 값 계산
        for case_num in range(1, max_case + 1):
            modified_coeff_col = f'modified_coeff_case{case_num}'
            
            # 해당 case의 modified_coeff 열이 존재하는지 확인
            if modified_coeff_col not in modified_df.columns:
                self._print(f"⚠️ {modified_coeff_col} 열이 없어서 건너뜁니다.", level="info")
                continue
            
            pcf_col_name = f'PCF_case{case_num}'
            
            self._print(f"📋 {pcf_col_name} 계산:")
            
            # PCF 값 계산: 제품총소요량(kg) * modified_coeff_case
            pcf_values = modified_df['제품총소요량(kg)'] * modified_df[modified_coeff_col]
            
            # 결과 데이터프레임에 추가
            pcf_result_df[pcf_col_name] = pcf_values
            
            # 통계 정보 출력
            total_pcf = pcf_values.sum()
            self._print(f"  📊 총 PCF 값: {total_pcf:.3f} kgCO2eq", level="info")
            self._print(f"  📊 평균 PCF 값: {pcf_values.mean():.3f} kgCO2eq", level="info")
            self._print(f"  📊 최대 PCF 값: {pcf_values.max():.3f} kgCO2eq", level="info")
            self._print(f"  📊 최소 PCF 값: {pcf_values.min():.3f} kgCO2eq", level="info")
        
        # 전체 PCF 합계 계산 및 출력
        pcf_columns = [col for col in pcf_result_df.columns if col.startswith('PCF_case')]
        self._print(f"📈 전체 PCF 합계(적용항목만 포함):")
        self._print("-" * 30)
        
        # PCF_reference 값 먼저 출력
        if 'PCF_reference' in pcf_result_df.columns:
            reference_total = pcf_result_df['PCF_reference'].sum()
            self._print(f"PCF_reference: {reference_total:.3f} kgCO2eq", level="info")
        
        # 그 다음 PCF_case 열들 출력
        for pcf_col in pcf_columns:
            total_pcf = pcf_result_df[pcf_col].sum()
            if 'PCF_reference' in pcf_result_df.columns:
                reference_total = pcf_result_df['PCF_reference'].sum()
                reduction = reference_total - total_pcf
                reduction_rate = (reduction / reference_total) * 100
                self._print(f"{pcf_col}: {total_pcf:.3f} kgCO2eq ({reduction_rate:.2f}% 감소)", level="info")
            else:
                self._print(f"{pcf_col}: {total_pcf:.3f} kgCO2eq", level="info")
        
        return pcf_result_df
    
    def calculate_pcf_values_with_merge(self, scenario_df: pd.DataFrame, pcf_result_df: pd.DataFrame, max_case: int = 3, num_tier: int = 2) -> tuple:
        """
        전체 scenario_df에 대해 PCF_caseN 열을 병합하여 반환 (저감활동_적용여부 NaN 포함)
        - all_result_df: 전체 scenario_df + PCF_caseN 열
        - modified_only_df: 실제로 PCF_caseN 값이 존재하는 행만 필터링

        Args:
            scenario_df: 시나리오 데이터프레임
            pcf_result_df: 수정된 배출계수 데이터프레임
            max_case: 계산할 최대 case 번호 (기본값: 3)
            num_tier: 계산할 최대 tier 번호 (기본값: 3)

        Returns:
            (all_result_df, modified_only_df)
        """
        self._print(f"🔍 PCF 값 계산 및 전체 데이터 병합 (Case 1 ~ {max_case}, Tier 1 ~ {num_tier})", level="info")
        
        # 1단계: modified_df에 PCF 값 계산
        modified_df_with_pcf = pcf_result_df.copy()
        
        # modified_coeff_case 열들을 자동으로 감지
        modified_coeff_columns = [col for col in pcf_result_df.columns if col.startswith('modified_coeff_case')]
        
        # PCF_case 열들이 이미 존재하는지 확인 (proportion_df에서 생성된 경우)
        existing_pcf_columns = [col for col in pcf_result_df.columns if col.startswith('PCF_case')]
        
        if len(modified_coeff_columns) == 0 and len(existing_pcf_columns) == 0:
            self._print("❌ modified_coeff_case 열 또는 PCF_case 열을 찾을 수 없습니다.", level="info")
            return scenario_df, pd.DataFrame()
        
        if len(modified_coeff_columns) > 0:
            self._print(f"📊 발견된 modified_coeff_case 열: {modified_coeff_columns}", level="info")
        
        if len(existing_pcf_columns) > 0:
            self._print(f"📊 발견된 기존 PCF_case 열: {existing_pcf_columns}", level="info")
        
        # 제품총소요량(kg) 열 확인
        if '제품총소요량(kg)' not in pcf_result_df.columns:
            self._print("❌ '제품총소요량(kg)' 열을 찾을 수 없습니다.", level="info")
            return scenario_df, pd.DataFrame()
        
        # 각 case별로 PCF 값 계산
        for case_num in range(1, max_case + 1):
            modified_coeff_col = f'modified_coeff_case{case_num}'
            pcf_col_name = f'PCF_case{case_num}'
            
            # 이미 PCF_case 열이 존재하는지 확인
            if pcf_col_name in pcf_result_df.columns:
                self._print(f"📋 {pcf_col_name} 이미 존재함", level="info")
                modified_df_with_pcf[pcf_col_name] = pcf_result_df[pcf_col_name]
                continue
            
            # 해당 case의 modified_coeff 열이 존재하는지 확인
            if modified_coeff_col not in pcf_result_df.columns:
                self._print(f"⚠️ {modified_coeff_col} 열이 없어서 건너뜁니다.", level="info")
                continue
            
            self._print(f"📋 {pcf_col_name} 계산:")
            
            # PCF 값 계산: 제품총소요량(kg) * modified_coeff_case
            pcf_values = pcf_result_df['제품총소요량(kg)'] * pcf_result_df[modified_coeff_col]
            
            # 결과 데이터프레임에 추가
            modified_df_with_pcf[pcf_col_name] = pcf_values
            
            # 통계 정보 출력
            total_pcf = pcf_values.sum()
            self._print(f"  📊 총 PCF 값: {total_pcf:.3f} kgCO2eq", level="info")
            self._print(f"  📊 평균 PCF 값: {pcf_values.mean():.3f} kgCO2eq", level="info")
            self._print(f"  📊 최대 PCF 값: {pcf_values.max():.3f} kgCO2eq", level="info")
            self._print(f"  📊 최소 PCF 값: {pcf_values.min():.3f} kgCO2eq", level="info")
            
            # 디버깅을 위한 상세 정보 출력
            if self.verbose == "debug":
                self._print(f"  🔍 {pcf_col_name} 상세 정보:")
                for idx, (pcf_val, coeff_val, amount_val) in enumerate(zip(pcf_values, modified_df[modified_coeff_col], modified_df['제품총소요량(kg)'])):
                    if idx < 5:  # 처음 5개만 출력
                        self._print(f"    행 {idx}: PCF={pcf_val:.3f} = {amount_val:.3f} × {coeff_val:.4f}")
                    else:
                        break
        
        # 2단계: scenario_df와 병합
        pcf_case_cols = [col for col in modified_df_with_pcf.columns if col.startswith('PCF_case')]
        modified_coeff_case_cols = [col for col in modified_df_with_pcf.columns if col.startswith('modified_coeff_case')]

        # 고유 병합 ID 추가 (행 번호 기반) - 1:1 매칭을 보장
        scenario_df = scenario_df.copy()
        modified_df_with_pcf = modified_df_with_pcf.copy()

        # 각 데이터프레임에 고유 병합 ID 추가
        scenario_df['_merge_id'] = range(len(scenario_df))
        modified_df_with_pcf['_merge_id'] = modified_df_with_pcf.index

        self._print(f"✅ 고유 병합 ID 추가: scenario_df {len(scenario_df)}개, modified_df {len(modified_df_with_pcf)}개", level="info")

        # 병합 기준 키 컬럼 찾기 - _merge_id를 우선 사용
        common_cols = list(set(scenario_df.columns) & set(modified_df_with_pcf.columns))

        # _merge_id를 최우선 키로 사용
        if '_merge_id' in common_cols:
            key_cols = ['_merge_id']
            self._print(f"✅ 고유 병합 ID를 사용한 1:1 매칭", level="info")
        else:
            # 가능한 키 컬럼 후보들 (고유 식별자 역할을 할 수 있는 컬럼들)
            # 배출계수명 추가: 고유화된 배출계수명을 병합 키로 활용하여 정확한 매칭 보장
            potential_key_cols = ['자재명', '자재품목', '배출계수명', '자재코드', '지역', '제품총소요량(kg)', 'PCF_reference']

            # 실제로 두 데이터프레임에 모두 존재하는 키 컬럼들만 선택
            key_cols = [col for col in potential_key_cols if col in common_cols]

            # 키 컬럼이 없으면 기본값 사용
            if not key_cols:
                key_cols = ['자재명', '자재품목']
                self._print(f"⚠️ 기본 키 컬럼 사용: {key_cols}", level="info")
            else:
                self._print(f"✅ 발견된 키 컬럼: {key_cols}", level="info")
        
        # 병합할 추가 열들 정의
        additional_cols = ['tier1_values', 'tier2_values', 'formula_matched', 'proportions_matched', 'tier_num_by']
        
        if self.verbose == "debug":
            self._print(f"📋 병합 기준 열: {key_cols}")
            self._print(f"📋 병합할 PCF 열: {pcf_case_cols}")
            self._print(f"📋 병합할 계수 열: {modified_coeff_case_cols}")
            self._print(f"📋 병합할 추가 열: {additional_cols}")
        
        # 데이터프레임 크기 정보 출력
        self._print(f"📊 scenario_df 크기: {scenario_df.shape}", level="info")
        self._print(f"📊 modified_df_with_pcf 크기: {modified_df_with_pcf.shape}", level="info")
        
        # modified_df의 모든 열을 가져오되, scenario_df와 공통된 열은 scenario_df 기준으로 유지
        # modified_df에서 병합할 열들만 추출
        cols_to_merge = key_cols + additional_cols + modified_coeff_case_cols + pcf_case_cols
        available_cols = [col for col in cols_to_merge if col in modified_df_with_pcf.columns]
        
        # modified_df에서 병합할 데이터 추출
        modified_data_to_merge = modified_df_with_pcf[available_cols].copy()
        
        # 키 컬럼 기반 병합 시도
        try:
            all_result_df = scenario_df.merge(
                modified_data_to_merge,
                on=key_cols,
                how='left'
            )
            self._print(f"✅ 키 컬럼 기반 병합 성공: {all_result_df.shape}", level="info")

            # _merge_id 컬럼 제거 (임시 컬럼이므로)
            if '_merge_id' in all_result_df.columns:
                all_result_df = all_result_df.drop(columns=['_merge_id'])
                self._print(f"✅ 임시 병합 ID 제거 완료", level="info")
        except Exception as e:
            self._print(f"❌ 키 컬럼 기반 병합 실패: {e}", level="error")
            self._print("🔄 인덱스 기반 병합으로 전환...", level="warning")
            
            # 인덱스 기반 병합으로 전환
            # modified_df의 인덱스가 scenario_df의 인덱스와 일치하는지 확인
            if len(modified_df_with_pcf) <= len(scenario_df):
                # modified_df의 인덱스를 scenario_df의 인덱스로 재설정
                modified_data_to_merge = modified_data_to_merge.reset_index(drop=True)
                
                # scenario_df에 modified_df의 열들을 추가 (인덱스 기반)
                all_result_df = scenario_df.copy()
                for col in available_cols:
                    if col in modified_data_to_merge.columns:
                        # modified_df의 행 수만큼만 값 할당
                        all_result_df.loc[:len(modified_data_to_merge)-1, col] = modified_data_to_merge[col].values
                        if self.verbose == "debug":
                            self._print(f"✅ {col} 열 추가 완료")
                
                self._print(f"✅ 인덱스 기반 병합 완료: {all_result_df.shape}", level="info")

                # _merge_id 컬럼 제거 (인덱스 기반 병합에서도)
                if '_merge_id' in all_result_df.columns:
                    all_result_df = all_result_df.drop(columns=['_merge_id'])
                    self._print(f"✅ 임시 병합 ID 제거 완료", level="info")
            else:
                self._print("❌ modified_df가 scenario_df보다 큽니다. 병합을 중단합니다.", level="warning")
                # _merge_id 컬럼 제거 후 반환
                if '_merge_id' in scenario_df.columns:
                    scenario_df = scenario_df.drop(columns=['_merge_id'])
                return scenario_df, pd.DataFrame()
        
        # modified_only_df: 실제로 PCF_caseN 값이 존재하는 행만
        # modified_df의 행 수와 일치하도록 필터링
        if len(modified_df_with_pcf) <= len(all_result_df):
            # modified_df의 행 수만큼만 선택
            modified_only_df = all_result_df.iloc[:len(modified_df_with_pcf)].copy()
            self._print(f"✅ modified_only_df 생성: {modified_only_df.shape} (modified_df와 동일한 행 수)", level="info")
        else:
            # PCF_case 열이 존재하는 행만 필터링
            modified_only_df = all_result_df[all_result_df[pcf_case_cols].notnull().any(axis=1)].copy()
            self._print(f"⚠️ PCF_case 열 기반 필터링: {modified_only_df.shape}", level="info")
        
        # PCF 합계 계산 (디버깅용)
        pcf_columns = [col for col in all_result_df.columns if col.startswith('PCF_case')]
        self._print(f"📊 중간 PCF 합계 (PCF_reference 생성 전):", level="info")
        for pcf_col in pcf_columns:
            total_pcf = all_result_df[pcf_col].sum()
            self._print(f"  {pcf_col}: {total_pcf:.3f} kgCO2eq", level="info")
        
        return all_result_df, modified_only_df
    
    def fill_pcf_case_with_reference(self, pcf_result_df: pd.DataFrame, scenario: str = 'baseline') -> pd.DataFrame:
        """
        시나리오별로 적절한 배출계수를 사용하여 PCF_reference 생성
        그런 다음 PCF_case 열을 PCF_reference 값으로 채우는 메서드 (_fill_nan_pcf_cases_with_reference)

        Args:
            pcf_result_df: 결과 데이터프레임
            scenario: 시나리오 타입 ('baseline', 'recycling', 'site_change', 'both')
                - baseline: Before 사이트, 재활용 전 배출계수 사용
                - recycling: Before 사이트, 재활용 전 배출계수 사용 (Case에서 재활용 적용)
                - site_change: Before 사이트 배출계수 사용 (Case에서 After 사이트 적용)
                - both: Before 사이트, 재활용 전 배출계수 사용 (Case에서 둘 다 적용)
        """
        self._print(f"🔧 PCF_reference 열 생성 및 NaN 처리 시작 (시나리오: {scenario})", level="info")
        self._print(f"🔍 현재 데이터프레임 열들: {list(pcf_result_df.columns)}", level="info")
        
        # 디버그: PCF_reference 생성 전 양극재/음극재 데이터 확인
        cathode_rows = pcf_result_df[pcf_result_df['자재품목'] == '양극재']
        anode_rows = pcf_result_df[pcf_result_df['자재품목'] == '음극재']
        
        self._print("=== PCF_reference 생성 전 양극재/음극재 디버그 ===", level="info")
        self._print(f"pcf_result_df 중 양극재: {len(cathode_rows)}개", level="info")
        self._print(f"pcf_result_df 중 음극재: {len(anode_rows)}개", level="info")
        
        if not cathode_rows.empty:
            self._print("양극재 PCF_reference 생성 전 데이터:", level="info")
            for idx, row in cathode_rows.iterrows():
                quantity = row.get('제품총소요량(kg)', 0)
                emission_coef = row.get('배출계수', 0)
                emission_amount = row.get('배출량(kgCO2eq)', 0)
                self._print(f"  • {row.get('자재명', 'N/A')}: 소요량={quantity:.6f}kg, 배출계수={emission_coef:.6f}, 배출량={emission_amount:.6f}", level="info")
        
        if not anode_rows.empty:
            self._print("음극재 PCF_reference 생성 전 데이터:", level="info")
            for idx, row in anode_rows.iterrows():
                quantity = row.get('제품총소요량(kg)', 0)
                emission_coef = row.get('배출계수', 0)
                emission_amount = row.get('배출량(kgCO2eq)', 0)
                self._print(f"  • {row.get('자재명', 'N/A')}: 소요량={quantity:.6f}kg, 배출계수={emission_coef:.6f}, 배출량={emission_amount:.6f}", level="info")
        
        self._print("===========================================", level="info")
        
        # PCF_baseline과 PCF_reference를 분리하여 계산
        # PCF_baseline: 모든 시나리오에서 동일한 원본 배출계수 기반 (시나리오 적용X, Tier-case 적용X)
        # PCF_reference: 시나리오 적용된 배출계수 기반 (시나리오 적용O, Tier-case 적용X)

        # PCF_reference 열 생성 - 모든 시나리오에서 동일하게 현재 배출량 사용
        if 'PCF_reference' not in pcf_result_df.columns:
            self._print(f"📝 PCF_reference 열 생성 (시나리오: {scenario})", level="info")

            # 모든 시나리오에서 PCF_reference = 배출량(kgCO2eq)
            # 이는 시나리오가 적용되었지만 Tier-case RE% 감축 전의 배출량
            pcf_result_df['PCF_reference'] = pcf_result_df['배출량(kgCO2eq)']

            total_reference = pcf_result_df['PCF_reference'].sum()
            self._print(f"📊 생성된 PCF_reference 총합: {total_reference:.3f} kgCO2eq", level="info")
        else:
            self._print("📝 PCF_reference 열이 이미 존재합니다.", level="info")
        
        return self._fill_nan_pcf_cases_with_reference(pcf_result_df)

    def _fill_nan_pcf_cases_with_reference(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        PCF_case1, PCF_case2, PCF_case3이 모두 NaN인 경우 PCF_reference 값으로 채우는 메서드
        
        Args:
            result_df: 시뮬레이션 결과 데이터프레임
            
        Returns:
            pd.DataFrame: NaN이 채워진 데이터프레임
        """
        self._print("🔧 PCF_case 열 NaN 값 처리", level="info")
        
        # PCF 관련 열들 확인
        pcf_case_columns = [col for col in result_df.columns if col.startswith('PCF_case')]
        pcf_reference_col = 'PCF_reference'
        
        self._print(f"🔍 발견된 PCF_case 열: {pcf_case_columns}", level="info")
        self._print(f"🔍 PCF_reference 열 존재: {pcf_reference_col in result_df.columns}", level="info")
        
        if not pcf_case_columns:
            self._print("⚠️ PCF_case 열을 찾을 수 없습니다.", level="info")
            return result_df
        
        if pcf_reference_col not in result_df.columns:
            self._print("⚠️ PCF_reference 열을 찾을 수 없습니다.", level="info")
            return result_df
        
        # 결과를 저장할 새로운 데이터프레임 생성
        filled_df = result_df.copy()
        
        # NaN 처리 통계
        total_rows = len(filled_df)
        nan_filled_count = 0
        
        # 각 행에 대해 PCF_case 열들이 모두 NaN인지 확인
        for idx, row in filled_df.iterrows():
            # PCF_case 열들의 NaN 여부 확인
            pcf_case_values = [row.get(col, np.nan) for col in pcf_case_columns]
            all_nan = all(pd.isna(val) for val in pcf_case_values)
            
            # 디버깅을 위한 상세 정보 (처음 5개 행만)
            if idx < 5 and self.verbose == "debug":
                material_name = row.get('자재명', 'Unknown')
                pcf_case_status = [f"{col}: {'NaN' if pd.isna(val) else f'{val:.3f}'}" for col, val in zip(pcf_case_columns, pcf_case_values)]
                reference_value = row.get(pcf_reference_col, 0.0)
                self._print(f"  🔍 행 {idx} ({material_name}): {pcf_case_status}, PCF_reference: {reference_value:.3f}, 모두 NaN: {all_nan}")
            
            if all_nan:
                # PCF_reference 값 가져오기
                reference_value = row.get(pcf_reference_col, 0.0)
                
                # 모든 PCF_case 열을 PCF_reference 값으로 채우기
                for col in pcf_case_columns:
                    filled_df.at[idx, col] = reference_value
                
                nan_filled_count += 1
                
                if self.verbose == "debug":
                    self._print(f"  📋 행 {idx}: {row.get('자재명', 'Unknown')} - PCF_case 열들을 PCF_reference 값({reference_value:.3f})으로 채움")
        
        # 처리 결과 출력
        self._print(f"📊 NaN 처리 결과:", level="info")
        self._print(f"  • 총 데이터 행 수: {total_rows:,}개", level="info")
        self._print(f"  • NaN 채워진 행 수: {nan_filled_count:,}개", level="info")
        self._print(f"  • 처리 비율: {nan_filled_count/total_rows*100:.1f}%", level="info")
        
        if nan_filled_count > 0:
            self._print(f"✅ PCF_case 열 NaN 값 처리 완료", level="info")
        else:
            self._print(f"📝 처리할 NaN 값이 없습니다.", level="info")
        
        return filled_df

    def update_ref_proportions_by_scenario(self, scenario: str, basic_df: pd.DataFrame):
        """
        시나리오별로 ref_proportions_df를 업데이트합니다. (기존 방식 - 하위 호환성 유지)
        
        Args:
            scenario: 시나리오 타입
            basic_df: CathodeSimulator의 기본 분석 결과
        """
        self._print(f"🔧 {scenario} 시나리오에 맞는 ref_proportions_df 업데이트 중...", level="info")
        
        # 업데이트 전 ref_proportions_df 상태 로깅
        self._print("📊 업데이트 전 ref_proportions_df 상태:", level="debug")
        if not self.ref_proportions_df.empty:
            self._print(f"  • 총 행 수: {len(self.ref_proportions_df)}개", level="debug")
            if 'Tier1_전력_기여도' in self.ref_proportions_df.columns:
                tier1_avg = self.ref_proportions_df['Tier1_전력_기여도'].mean()
                tier2_avg = self.ref_proportions_df['Tier2_전력_기여도'].mean() if 'Tier2_전력_기여도' in self.ref_proportions_df.columns else 0
                self._print(f"  • Tier1_전력_기여도 평균: {tier1_avg:.3f}", level="debug")
                self._print(f"  • Tier2_전력_기여도 평균: {tier2_avg:.3f}", level="debug")
        
        # 1단계: 시나리오별 ref_proportions_df 업데이트
        self._print("1️⃣ 시나리오별 ref_proportions_df 업데이트", level="info")
        
        if scenario == 'baseline':
            # baseline: CathodeSimulator의 update_ref_proportions_with_tier_contributions 사용
            self._print("📋 Baseline 시나리오: ref_proportions_df 업데이트", level="info")
            self.ref_proportions_df = self.cathode_simulator.update_ref_proportions_with_tier_contributions(
                basic_df=basic_df, 
                ref_proportions_df=self.ref_proportions_df,
                scenario='baseline'
            )
        
        elif scenario == 'recycling':
            # recycling: 재활용 기준으로 ref_proportions_df 업데이트
            self._print("📋 Recycling 시나리오: ref_proportions_df 업데이트", level="info")
            self.ref_proportions_df = self.cathode_simulator.update_ref_proportions_with_tier_contributions(
                basic_df=basic_df, 
                ref_proportions_df=self.ref_proportions_df,
                scenario='recycling'
            )
        
        elif scenario == 'site_change':
            # site_change: 사이트 변경 기준으로 ref_proportions_df 업데이트
            self._print("📋 Site Change 시나리오: ref_proportions_df 업데이트", level="info")
            self.ref_proportions_df = self.cathode_simulator.update_ref_proportions_with_tier_contributions(
                basic_df=basic_df, 
                ref_proportions_df=self.ref_proportions_df,
                scenario='site_change'
            )
        
        elif scenario == 'both':
            # both: 재활용 + 사이트 변경 모두 적용
            self._print("📋 Both 시나리오: ref_proportions_df 업데이트", level="info")
            self.ref_proportions_df = self.cathode_simulator.update_ref_proportions_with_tier_contributions(
                basic_df=basic_df, 
                ref_proportions_df=self.ref_proportions_df,
                scenario='both'
            )
        
        else:
            self._print(f"⚠️ 알 수 없는 시나리오: {scenario}", level="warning")
        
        # 업데이트 후 ref_proportions_df 상태 로깅
        self._print("📊 업데이트 후 ref_proportions_df 상태:", level="debug")
        if not self.ref_proportions_df.empty:
            self._print(f"  • 총 행 수: {len(self.ref_proportions_df)}개", level="debug")
            if 'Tier1_전력_기여도' in self.ref_proportions_df.columns:
                tier1_avg = self.ref_proportions_df['Tier1_전력_기여도'].mean()
                tier2_avg = self.ref_proportions_df['Tier2_전력_기여도'].mean() if 'Tier2_전력_기여도' in self.ref_proportions_df.columns else 0
                self._print(f"  • Tier1_전력_기여도 평균: {tier1_avg:.3f}", level="debug")
                self._print(f"  • Tier2_전력_기여도 평균: {tier2_avg:.3f}", level="debug")

    def update_original_cathode_coefficients(self, basic_df: pd.DataFrame, scenario: str = 'recycling'):
        """
        original_df의 양극재 배출계수를 업데이트합니다.
        
        Args:
            basic_df: CathodeSimulator의 기본 분석 결과
            scenario: 시나리오 타입 (기본값: 'recycling')
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (summary_df, updated_original_df)
        """
        self._print(f"🔧 {scenario} 시나리오로 original_df 양극재 배출계수 업데이트 중...", level="info")
        
        if self.original_df is None:
            self._print("⚠️ original_df가 없어 배출계수 업데이트를 건너뜁니다.", level="warning")
            return pd.DataFrame(), pd.DataFrame()
        
        # 업데이트 전 original_df 상태 로깅
        self._print("📊 업데이트 전 original_df 상태:", level="debug")
        if '배출계수' in self.original_df.columns:
            cathode_rows = self.original_df[self.original_df['자재품목'] == '양극재']
            if not cathode_rows.empty:
                for idx, row in cathode_rows.iterrows():
                    self._print(f"  • {row['자재명']}: 배출계수 = {row['배출계수']:.6f}", level="debug")
        
        # CathodeSimulator를 통해 original_df 업데이트
        summary_df, updated_original_df = self.cathode_simulator.update_original_cathode_coef(
            basic_df, self.original_df, scenario=scenario
        )
        
        # 업데이트된 original_df 저장
        self.original_df = updated_original_df
        
        # 업데이트 후 original_df 상태 로깅
        self._print("📊 업데이트 후 original_df 상태:", level="debug")
        if '배출계수' in self.original_df.columns:
            cathode_rows = self.original_df[self.original_df['자재품목'] == '양극재']
            if not cathode_rows.empty:
                for idx, row in cathode_rows.iterrows():
                    self._print(f"  • {row['자재명']}: 배출계수 = {row['배출계수']:.6f}", level="debug")
        
        if not summary_df.empty:
            self._print("✅ original_df 양극재 배출계수 업데이트 완료", level="info")
            self._print(f"📊 업데이트된 행 수: {len(summary_df)}개", level="info")
            
            # 업데이트 요약 정보 출력
            for _, row in summary_df.iterrows():
                self._print(f"  • {row['자재명']}: {row['원본_배출계수']:.6f} → {row['업데이트_배출계수']:.6f} (감소율: {row['감소율']:.2f}%)", level="info")
        else:
            self._print("⚠️ 업데이트할 양극재가 없습니다.", level="info")
        
        return summary_df, updated_original_df

    def _run_rule_based_simulation(self, max_case: int = 3, num_tier: int = 2, scenario: str = 'baseline', basic_df: pd.DataFrame = None, skip_updates: bool = False):
        """
        기본 규칙 기반 시뮬레이션 프로세스를 실행합니다.
        
        Args:
            max_case: 계산할 최대 case 번호 (기본값: 3)
            num_tier: 계산할 최대 tier 번호 (기본값: 2)
            scenario: 현재 실행 중인 시나리오 (기본값: 'baseline')
            basic_df: CathodeSimulator의 기본 분석 결과 (시나리오별 업데이트용)
            
        Returns:
            pd.DataFrame: 최종 시뮬레이션 결과
        """
        self._print("🚀 기본 규칙 기반 시뮬레이션 시작", level="info")
        self._print(f"🔍 _run_rule_based_simulation 메서드가 호출되었습니다! (시나리오: {scenario})", level="info")
        
        # 시나리오별 데이터 상태 로깅 (강제 출력)
        self._print(f"📊 {scenario} 시나리오 데이터 상태:", level="info")
        if self.original_df is not None:
            self._print(f"  • original_df 크기: {self.original_df.shape}", level="debug")
            if '배출계수' in self.original_df.columns:
                cathode_rows = self.original_df[self.original_df['자재품목'] == '양극재']
                if not cathode_rows.empty:
                    self._print(f"    - 양극재 행 수: {len(cathode_rows)}개", level="debug")
                    for idx, row in cathode_rows.iterrows():
                        self._print(f"      • {row['자재명']}: 배출계수 = {row['배출계수']:.6f}", level="debug")
        
        self._print(f"  • ref_proportions_df 크기: {self.ref_proportions_df.shape}", level="debug")
        if 'Tier1_RE100(%)' in self.ref_proportions_df.columns:
            try:
                # 퍼센트 기호 제거 후 숫자로 변환
                tier1_values = self.ref_proportions_df['Tier1_RE100(%)'].astype(str).str.replace('%', '').astype(float)
                tier1_avg = tier1_values.mean()
                self._print(f"    - Tier1_RE100(%) 평균: {tier1_avg:.3f}", level="debug")
            except Exception as e:
                self._print(f"    - Tier1_RE100(%) 평균 계산 실패: {e}", level="debug")
                tier1_avg = 0
        
        if 'Tier2_RE100(%)' in self.ref_proportions_df.columns:
            try:
                # 퍼센트 기호 제거 후 숫자로 변환
                tier2_values = self.ref_proportions_df['Tier2_RE100(%)'].astype(str).str.replace('%', '').astype(float)
                tier2_avg = tier2_values.mean()
                self._print(f"    - Tier2_RE100(%) 평균: {tier2_avg:.3f}", level="debug")
            except Exception as e:
                self._print(f"    - Tier2_RE100(%) 평균 계산 실패: {e}", level="debug")
                tier2_avg = 0
        
        # 1단계: 저감활동 적용 가능한 행 추출
        self._print("1️⃣ 저감활동 적용 가능한 행 추출", level="info")
        applicable_rows = self.extract_reduction_applicable_rows()
        
        # 2단계: 공식 데이터 매칭
        self._print("2️⃣ 공식 데이터 매칭", level="info")
        matched_rows, _ = self.match_with_formula_data(applicable_rows, scenario)
        
        # 3단계: 배출계수 수정 계산
        self._print("3️⃣ 배출계수 수정 계산", level="info")
        modified_df = self.calculate_modified_coefficients(matched_rows, max_case=max_case, num_tier=num_tier, scenario=scenario, basic_df=basic_df, skip_updates=skip_updates)
        
        # 4단계: PCF 값 계산
        self._print("4️⃣ PCF 값 계산", level="info")
        pcf_result_df = self.calculate_pcf_values(modified_df, max_case=max_case, num_tier=num_tier)
        
        # 5단계: 전체 데이터 병합
        self._print("5️⃣ 전체 데이터 병합", level="info")
        
        # 시나리오별로 업데이트된 데이터를 반영한 scenario_df 생성
        updated_scenario_df = self.scenario_df.copy()
        
        # 모든 시나리오에서 original_df의 배출계수를 scenario_df에 동기화
        for idx, row in updated_scenario_df.iterrows():
            # 1차 시도: 배출계수명을 포함한 정확한 매칭
            if '배출계수명' in row.index and '배출계수명' in self.original_df.columns:
                match = self.original_df[
                    (self.original_df['자재명'] == row['자재명']) &
                    (self.original_df['자재품목'] == row['자재품목']) &
                    (self.original_df['배출계수명'] == row['배출계수명'])
                ]
            else:
                match = pd.DataFrame()  # 빈 DataFrame

            # 2차 시도: 배출계수명 매칭 실패 시 기존 방식 (자재명 + 자재품목)
            if match.empty:
                match = self.original_df[
                    (self.original_df['자재명'] == row['자재명']) &
                    (self.original_df['자재품목'] == row['자재품목'])
                ]

                # 여러 개 매칭된 경우, 정확한 행 선택을 위한 다단계 필터링
                if len(match) > 1:
                    # 1순위: 배출량(kgCO2eq) 정확히 일치하는 행 찾기
                    if '배출량(kgCO2eq)' in row.index and '배출량(kgCO2eq)' in match.columns:
                        row_emission = row['배출량(kgCO2eq)']
                        emission_match = match[match['배출량(kgCO2eq)'] == row_emission]
                        if not emission_match.empty:
                            match = emission_match

                    # 2순위: 제품총소요량(kg) 정확히 일치하는 행 찾기
                    if len(match) > 1 and '제품총소요량(kg)' in row.index and '제품총소요량(kg)' in match.columns:
                        row_quantity = row['제품총소요량(kg)']
                        quantity_match = match[match['제품총소요량(kg)'] == row_quantity]
                        if not quantity_match.empty:
                            match = quantity_match

                    # 3순위: 제품총소요량이 가장 가까운 것 선택 (정확히 일치하는 것이 없을 때만)
                    if len(match) > 1 and '제품총소요량(kg)' in row.index and '제품총소요량(kg)' in match.columns:
                        row_quantity = row['제품총소요량(kg)']
                        match['_diff'] = abs(match['제품총소요량(kg)'] - row_quantity)
                        match = match.nsmallest(1, '_diff')
                        if '_diff' in match.columns:
                            match = match.drop(columns=['_diff'])

            if not match.empty:
                updated_coeff = match.iloc[0]['배출계수']
                updated_emission = updated_coeff * row['제품총소요량(kg)']
                updated_scenario_df.at[idx, '배출계수'] = updated_coeff
                updated_scenario_df.at[idx, '배출량(kgCO2eq)'] = updated_emission
        
        all_result_df, _ = self.calculate_pcf_values_with_merge(updated_scenario_df, pcf_result_df, max_case=max_case, num_tier=num_tier)
        
        # 6단계: PCF_case 열 NaN 값 처리
        self._print("6️⃣ PCF_case 열 NaN 값 처리", level="info")
        all_result_filled_df = self.fill_pcf_case_with_reference(all_result_df, scenario=scenario)
        
        # 7단계: 최종 PCF 합계 출력 (PCF_reference 포함)
        self._print("7️⃣ 최종 PCF 합계 출력", level="info")
        self._print(f"📈 {scenario} 시나리오 최종 PCF 합계:")
        self._print("-" * 30)
        
        # PCF_reference 값 먼저 출력
        if 'PCF_reference' in all_result_filled_df.columns:
            reference_total = all_result_filled_df['PCF_reference'].sum()
            self._print(f"PCF_reference: {reference_total:.3f} kgCO2eq", level="info")
        
        # PCF_case 열들 출력
        pcf_case_columns = [col for col in all_result_filled_df.columns if col.startswith('PCF_case')]
        for pcf_col in pcf_case_columns:
            total_pcf = all_result_filled_df[pcf_col].sum()
            if 'PCF_reference' in all_result_filled_df.columns:
                reference_total = all_result_filled_df['PCF_reference'].sum()
                reduction = reference_total - total_pcf
                reduction_rate = (reduction / reference_total) * 100
                self._print(f"{pcf_col}: {total_pcf:.3f} kgCO2eq ({reduction_rate:.2f}% 감소)", level="info")
            else:
                self._print(f"{pcf_col}: {total_pcf:.3f} kgCO2eq", level="info")
        
        self._print(f"✅ {scenario} 시나리오 기본 규칙 기반 시뮬레이션 완료", level="info")
        return all_result_filled_df

    def run_simulation(self, scenario: str = 'baseline', basic_df: pd.DataFrame = None, max_case: int = 3, num_tier: int = 2, verbose: bool = None, skip_updates: bool = False):
        """
        시나리오에 따른 시뮬레이션을 실행합니다.
        
        Args:
            scenario: 시나리오 타입 ('baseline', 'recycling', 'site_change', 'both')
            basic_df: CathodeSimulator의 기본 분석 결과 (선택사항)
            max_case: 계산할 최대 case 번호 (기본값: 3)
            num_tier: 계산할 최대 tier 번호 (기본값: 2)
            verbose: 상세 로그 출력 여부 (기본값: None, self.verbose 사용)
            skip_updates: 시나리오별 업데이트를 건너뛸지 여부 (기본값: False)
            
        Returns:
            pd.DataFrame: 시뮬레이션 결과
        """
        # verbose 설정 (None이면 self.verbose 사용)
        if verbose is not None:
            original_verbose = self.verbose
            self.verbose = verbose
        
        self._print(f"🚀 {scenario.upper()} 시나리오 시뮬레이션 시작", level="info")
        
        try:
            if scenario == 'baseline':
                return self._run_simulation_baseline(max_case, num_tier)
            elif scenario == 'recycling':
                return self._run_simulation_recycling(basic_df, max_case, num_tier, skip_updates)
            elif scenario == 'site_change':
                return self._run_simulation_site_change(basic_df, max_case, num_tier, skip_updates)
            elif scenario == 'both':
                return self._run_simulation_both(basic_df, max_case, num_tier, skip_updates)
            else:
                self._print(f"❌ 지원하지 않는 시나리오: {scenario}", level="error")
                self._print("✅ 지원되는 시나리오: baseline, recycling, site_change, both", level="info")
                return pd.DataFrame()
        finally:
            # verbose 복원
            if verbose is not None:
                self.verbose = original_verbose

    def _run_simulation_baseline(self, max_case: int = 3, num_tier: int = 2):
        """
        기본 시나리오 실행 (변화 없음)
        
        Args:
            max_case: 계산할 최대 case 번호
            num_tier: 계산할 최대 tier 번호
            
        Returns:
            pd.DataFrame: 시뮬레이션 결과
        """
        self._print("📋 기본 시나리오 실행 (변화 없음)", level="info")
        return self._run_rule_based_simulation(max_case, num_tier, scenario='baseline', basic_df=None, skip_updates=False)

    def _run_simulation_recycling(self, basic_df: pd.DataFrame = None, max_case: int = 3, num_tier: int = 2, skip_updates: bool = False):
        """
        재활용 시나리오 실행 (양극재 배출계수 수정)
        
        Args:
            basic_df: CathodeSimulator의 기본 분석 결과
            max_case: 계산할 최대 case 번호
            num_tier: 계산할 최대 tier 번호
            skip_updates: 시나리오별 업데이트를 건너뛸지 여부
            
        Returns:
            pd.DataFrame: 시뮬레이션 결과
        """
        self._print("📋 재활용 시나리오 실행 (양극재 배출계수 수정)", level="info")
        
        # 시나리오별 업데이트는 calculate_modified_coefficients에서 수행
        return self._run_rule_based_simulation(max_case, num_tier, scenario='recycling', basic_df=basic_df, skip_updates=skip_updates)

    def _run_simulation_site_change(self, basic_df: pd.DataFrame = None, max_case: int = 3, num_tier: int = 2, skip_updates: bool = False):
        """
        사이트 변경 시나리오 실행 (ref_proportions_df 업데이트)
        
        Args:
            basic_df: CathodeSimulator의 기본 분석 결과
            max_case: 계산할 최대 case 번호
            num_tier: 계산할 최대 tier 번호
            skip_updates: 시나리오별 업데이트를 건너뛸지 여부
            
        Returns:
            pd.DataFrame: 시뮬레이션 결과
        """
        self._print("📋 사이트 변경 시나리오 실행 (ref_proportions_df 업데이트)", level="info")
        
        # 시나리오별 업데이트는 calculate_modified_coefficients에서 수행
        return self._run_rule_based_simulation(max_case, num_tier, scenario='site_change', basic_df=basic_df, skip_updates=skip_updates)

    def _run_simulation_both(self, basic_df: pd.DataFrame = None, max_case: int = 3, num_tier: int = 2, skip_updates: bool = False):
        """
        종합 시나리오 실행 (양극재 배출계수 수정 + ref_proportions_df 업데이트)
        
        Args:
            basic_df: CathodeSimulator의 기본 분석 결과
            max_case: 계산할 최대 case 번호
            num_tier: 계산할 최대 tier 번호
            skip_updates: 시나리오별 업데이트를 건너뛸지 여부
            
        Returns:
            pd.DataFrame: 시뮬레이션 결과
        """
        self._print("📋 종합 시나리오 실행 (양극재 배출계수 수정 + ref_proportions_df 업데이트)", level="info")
        
        # 시나리오별 업데이트는 calculate_modified_coefficients에서 수행
        return self._run_rule_based_simulation(max_case, num_tier, scenario='both', basic_df=basic_df, skip_updates=skip_updates)

    def analyze_simulation_results(self, sim_data: pd.DataFrame, baseline_reference: float = None) -> Dict[str, pd.DataFrame]:
        """
        시뮬레이션 결과를 다양한 형태로 분석하여 반환합니다.

        Args:
            sim_data: run_simulation의 결과 데이터프레임
            baseline_reference: 모든 시나리오의 기준값 (baseline 시나리오의 PCF_reference)

        Returns:
            Dict[str, pd.DataFrame]: 분석 결과들
                - all_data: 원본 결과 데이터프레임
                - modified_data: 저감활동_적용여부가 1인 데이터
                - material_data: 자재품목별 요약 데이터
                - pcf_summary: PCF 값들 비교 요약
                - matching_summary: 저감활동 적용/미적용 비교
                - coefficient_summary: 배출계수 변경 분석
        """
        self._print("🔍 시뮬레이션 결과 분석 시작", level="info")

        # 1. all_data (원본 결과)
        all_data = sim_data.copy()
        self._print(f"📊 all_data: {len(all_data)}개 행", level="info")

        # 2. modified_data (저감활동 적용된 데이터)
        if '저감활동_적용여부' in sim_data.columns:
            modified_data = sim_data[sim_data['저감활동_적용여부'] == 1.0].copy()
            self._print(f"📊 modified_data: {len(modified_data)}개 행", level="info")
        else:
            modified_data = pd.DataFrame()
            self._print("⚠️ '저감활동_적용여부' 열이 없어 modified_data를 생성할 수 없습니다.", level="warning")

        # 3. material_data (자재품목별 요약)
        material_data = self._create_material_summary(sim_data)

        # 4. pcf_summary (PCF 값들 비교) - baseline_reference 전달
        pcf_summary = self._create_pcf_summary(sim_data, baseline_reference=baseline_reference)

        # 5. matching_summary (저감활동 적용/미적용 비교)
        matching_summary = self._create_matching_summary(sim_data)

        # 6. coefficient_summary (배출계수 변경 분석)
        coefficient_summary = self._create_coefficient_summary(sim_data)

        self._print("✅ 시뮬레이션 결과 분석 완료", level="info")

        return {
            'all_data': all_data,
            'modified_data': modified_data,
            'material_data': material_data,
            'pcf_summary': pcf_summary,
            'matching_summary': matching_summary,
            'coefficient_summary': coefficient_summary
        }
    
    def _create_material_summary(self, sim_data: pd.DataFrame) -> pd.DataFrame:
        """자재품목별 요약 데이터 생성"""
        if '자재품목' not in sim_data.columns:
            self._print("⚠️ '자재품목' 열이 없어 material_data를 생성할 수 없습니다.", level="warning")
            return pd.DataFrame()
        
        # PCF 관련 열들 찾기
        pcf_columns = [col for col in sim_data.columns if col.startswith('PCF_case')]
        pcf_reference_col = 'PCF_reference' if 'PCF_reference' in sim_data.columns else None
        
        # 자재품목별 그룹화
        material_summary = []
        
        for material_category in sim_data['자재품목'].unique():
            if pd.isna(material_category):
                continue
                
            material_df = sim_data[sim_data['자재품목'] == material_category]
            
            summary_row = {
                '자재품목': material_category,
                '자재개수': len(material_df),
                '총_소요량(kg)': material_df['제품총소요량(kg)'].sum() if '제품총소요량(kg)' in material_df.columns else 0,
                '평균_배출계수': material_df['배출계수'].mean() if '배출계수' in material_df.columns else 0
            }
            
            # PCF_reference 관련
            if pcf_reference_col:
                summary_row['PCF_reference_합계'] = material_df[pcf_reference_col].sum()
                summary_row['PCF_reference_평균'] = material_df[pcf_reference_col].mean()
            
            # PCF_case 열들 처리
            for pcf_col in pcf_columns:
                case_num = pcf_col.replace('PCF_case', '')
                summary_row[f'PCF_case{case_num}_합계'] = material_df[pcf_col].sum()
                summary_row[f'PCF_case{case_num}_평균'] = material_df[pcf_col].mean()
                
                # 감소율 계산
                if pcf_reference_col:
                    reference_sum = material_df[pcf_reference_col].sum()
                    case_sum = material_df[pcf_col].sum()
                    if reference_sum > 0:
                        reduction_rate = ((reference_sum - case_sum) / reference_sum) * 100
                        summary_row[f'PCF_case{case_num}_감소율(%)'] = reduction_rate
                    else:
                        summary_row[f'PCF_case{case_num}_감소율(%)'] = 0
                else:
                    summary_row[f'PCF_case{case_num}_감소율(%)'] = 0
            
            material_summary.append(summary_row)
        
        return pd.DataFrame(material_summary)
    
    def _create_pcf_summary(self, sim_data: pd.DataFrame, baseline_reference: float = None) -> pd.DataFrame:
        """PCF 값들 비교 요약 생성 - 간단하고 직관적인 형태

        Args:
            sim_data: 시뮬레이션 데이터
            baseline_reference: 모든 시나리오의 기준값 (baseline 시나리오의 PCF_reference)
        """
        pcf_columns = [col for col in sim_data.columns if col.startswith('PCF_case')]
        pcf_reference_col = 'PCF_reference' if 'PCF_reference' in sim_data.columns else None

        if not pcf_columns:
            self._print("⚠️ PCF_case 열이 없어 pcf_summary를 생성할 수 없습니다.", level="warning")
            return pd.DataFrame()

        pcf_summary = []

        # 기준값 결정: baseline_reference가 제공되면 사용, 없으면 현재 시나리오의 PCF_reference 사용
        if baseline_reference is not None:
            reference_total = baseline_reference
            self._print(f"📊 기준값으로 baseline PCF_reference 사용: {reference_total:.3f} kgCO2eq", level="info")
        else:
            reference_total = sim_data[pcf_reference_col].sum() if pcf_reference_col else 0
            self._print(f"📊 기준값으로 현재 시나리오 PCF_reference 사용: {reference_total:.3f} kgCO2eq", level="info")

        reference_row = {
            'Case': 'PCF Reference',
            'PCF_총합(kgCO2eq)': round(reference_total, 3),
            '자재_개수': len(sim_data[sim_data[pcf_reference_col].notna()]) if pcf_reference_col else len(sim_data),
            '기준값(kgCO2eq)': round(reference_total, 3),
            '감축량(kgCO2eq)': 0.0,
            '감축률(%)': 0.0,
            '효과': '기준'
        }
        pcf_summary.append(reference_row)

        # 각 Case별 요약
        for pcf_col in pcf_columns:
            case_num = pcf_col.replace('PCF_case', '')
            case_total = sim_data[pcf_col].sum()

            summary_row = {
                'Case': f'Case {case_num}',
                'PCF_총합(kgCO2eq)': round(case_total, 3),
                '자재_개수': len(sim_data[sim_data[pcf_col].notna()])
            }

            # 기준값 대비 감축량과 감축률 계산
            if reference_total > 0:
                reduction_amount = reference_total - case_total
                reduction_rate = (reduction_amount / reference_total) * 100

                summary_row['기준값(kgCO2eq)'] = round(reference_total, 3)
                summary_row['감축량(kgCO2eq)'] = round(reduction_amount, 3)
                summary_row['감축률(%)'] = round(reduction_rate, 2)

                # 감축 효과 평가
                if reduction_rate > 10:
                    summary_row['효과'] = '높음'
                elif reduction_rate > 5:
                    summary_row['효과'] = '보통'
                elif reduction_rate > 0:
                    summary_row['효과'] = '낮음'
                else:
                    summary_row['효과'] = '없음'
            else:
                summary_row['기준값(kgCO2eq)'] = '-'
                summary_row['감축량(kgCO2eq)'] = '-'
                summary_row['감축률(%)'] = '-'
                summary_row['효과'] = '-'

            pcf_summary.append(summary_row)
        
        
        # Case 열을 기준으로 정렬
        df = pd.DataFrame(pcf_summary)
        
        # 정렬 순서 정의
        sort_order = {
            'PCF Reference': 0,
            'Case 1': 1,
            'Case 2': 2,
            'Case 3': 3
        }
        
        # Case 열에 정렬 순서 추가
        df['sort_order'] = df['Case'].map(sort_order)
        df = df.sort_values('sort_order').drop('sort_order', axis=1)
        
        return df
    
    def _create_matching_summary(self, sim_data: pd.DataFrame) -> pd.DataFrame:
        """저감활동 적용/미적용 비교 요약 생성"""
        if '저감활동_적용여부' not in sim_data.columns:
            self._print("⚠️ '저감활동_적용여부' 열이 없어 matching_summary를 생성할 수 없습니다.", level="warning")
            return pd.DataFrame()
        
        # 저감활동 적용/미적용 데이터 분리
        applied_df = sim_data[sim_data['저감활동_적용여부'] == 1.0]
        not_applied_df = sim_data[sim_data['저감활동_적용여부'] != 1.0]
        
        matching_summary = []
        
        # 기술통계 비교
        for category, df in [('저감활동_적용', applied_df), ('저감활동_미적용', not_applied_df)]:
            if len(df) == 0:
                continue
                
            row = {'구분': category, '데이터_개수': len(df)}
            
            # 기본 통계
            if '제품총소요량(kg)' in df.columns:
                row['총소요량_합계'] = df['제품총소요량(kg)'].sum()
                row['총소요량_평균'] = df['제품총소요량(kg)'].mean()
                row['총소요량_표준편차'] = df['제품총소요량(kg)'].std()
            
            if '배출계수' in df.columns:
                row['배출계수_평균'] = df['배출계수'].mean()
                row['배출계수_표준편차'] = df['배출계수'].std()
                row['배출계수_최대'] = df['배출계수'].max()
                row['배출계수_최소'] = df['배출계수'].min()
            
            # PCF 관련 통계
            pcf_columns = [col for col in df.columns if col.startswith('PCF_case')]
            for pcf_col in pcf_columns:
                case_num = pcf_col.replace('PCF_case', '')
                row[f'PCF_case{case_num}_합계'] = df[pcf_col].sum()
                row[f'PCF_case{case_num}_평균'] = df[pcf_col].mean()
                row[f'PCF_case{case_num}_표준편차'] = df[pcf_col].std()
            
            if 'PCF_reference' in df.columns:
                row['PCF_reference_합계'] = df['PCF_reference'].sum()
                row['PCF_reference_평균'] = df['PCF_reference'].mean()
                row['PCF_reference_표준편차'] = df['PCF_reference'].std()
            
            matching_summary.append(row)
        
        return pd.DataFrame(matching_summary)
    
    def _create_coefficient_summary(self, sim_data: pd.DataFrame) -> pd.DataFrame:
        """배출계수 변경 분석 요약 생성"""
        coefficient_summary = []
        
        # modified_coeff_case 열들 찾기
        modified_coeff_columns = [col for col in sim_data.columns if col.startswith('modified_coeff_case')]
        
        if not modified_coeff_columns:
            self._print("⚠️ modified_coeff_case 열이 없어 coefficient_summary를 생성할 수 없습니다.", level="warning")
            return pd.DataFrame()
        
        if '배출계수' not in sim_data.columns:
            self._print("⚠️ '배출계수' 열이 없어 coefficient_summary를 생성할 수 없습니다.", level="warning")
            return pd.DataFrame()
        
        # 각 case별 배출계수 변경 분석
        for modified_col in modified_coeff_columns:
            case_num = modified_col.replace('modified_coeff_case', '')
            
            # NaN이 아닌 행들만 선택
            valid_df = sim_data[sim_data[modified_col].notna()]
            
            if len(valid_df) == 0:
                continue
            
            row = {'Case': f'Case{case_num}', '분석_대상_개수': len(valid_df)}
            
            # 원본 배출계수 통계
            original_coeffs = valid_df['배출계수']
            row['원본_배출계수_평균'] = original_coeffs.mean()
            row['원본_배출계수_표준편차'] = original_coeffs.std()
            row['원본_배출계수_최대'] = original_coeffs.max()
            row['원본_배출계수_최소'] = original_coeffs.min()
            
            # 수정된 배출계수 통계
            modified_coeffs = valid_df[modified_col]
            row['수정_배출계수_평균'] = modified_coeffs.mean()
            row['수정_배출계수_표준편차'] = modified_coeffs.std()
            row['수정_배출계수_최대'] = modified_coeffs.max()
            row['수정_배출계수_최소'] = modified_coeffs.min()
            
            # 변경량 통계
            changes = original_coeffs - modified_coeffs
            row['변경량_평균'] = changes.mean()
            row['변경량_표준편차'] = changes.std()
            row['변경량_최대'] = changes.max()
            row['변경량_최소'] = changes.min()
            
            # 감소율 통계
            reduction_rates = (changes / original_coeffs) * 100
            row['감소율_평균(%)'] = reduction_rates.mean()
            row['감소율_표준편차(%)'] = reduction_rates.std()
            row['감소율_최대(%)'] = reduction_rates.max()
            row['감소율_최소(%)'] = reduction_rates.min()
            
            coefficient_summary.append(row)
        
        return pd.DataFrame(coefficient_summary)

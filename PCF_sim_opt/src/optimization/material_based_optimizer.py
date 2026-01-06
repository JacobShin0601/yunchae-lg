"""
자재별 배출계수와 소요량 기반 최적화 모듈

이 모듈은 자재별 배출계수와 소요량을 기반으로 탄소배출을 최소화하는 최적화 엔진을 구현합니다.
시뮬레이션에서 설정한 저감활동_적용자재만을 최적화 대상으로 하며,
자재 유형에 따라 다른 최적화 접근법을 적용합니다.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pathlib import Path
import os
import json
import copy
from datetime import datetime

from .scenario_base import OptimizationScenario
from .input import OptimizationInput
from .constraints import ConstraintManager
from .objective import ObjectiveManager
from .results_processor import ResultsProcessor
from .validator import Validator
from .reduction_constraints import ReductionConstraintManager
from .premium_cost_calculator import MaterialPremiumCostCalculator

# 시뮬레이션 로직 통합을 위한 추가 import
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.append(project_root)

try:
    from src.cathode_simulator import CathodeSimulator
    from src.utils.file_operations import FileOperations
except ImportError:
    # Fallback for import issues
    CathodeSimulator = None
    FileOperations = None


class MaterialBasedOptimizer:
    """
    자재별 배출계수 및 소요량 기반 최적화 모델
    
    이 클래스는 자재별 특성에 맞는 최적화 모델을 구성하고,
    감축 목표를 달성하면서 탄소 배출을 최소화하는 최적 솔루션을 찾습니다.
    """
    
    def __init__(self, 
                 simulation_data: Dict[str, pd.DataFrame] = None,
                 config: Dict[str, Any] = None,
                 ui_params: Dict[str, Any] = None,
                 stable_var_dir: str = "stable_var",
                 user_id: Optional[str] = None,
                 scenario: str = "baseline",
                 debug_mode: bool = True,
                 streamlit_container=None):
        """
        MaterialBasedOptimizer 초기화 - 시뮬레이션 로직 완전 통합
        
        Args:
            simulation_data: 시뮬레이션 데이터 (시나리오 및 참조 데이터프레임)
            config: 최적화 설정 (없으면 기본값 사용)
            ui_params: UI에서 받은 파라미터 설정 (최우선 적용)
            stable_var_dir: stable_var 디렉토리 경로
            user_id: 사용자 ID (사용자별 데이터 사용시)
            scenario: 최적화 시나리오 ('baseline', 'recycling', 'site_change', 'both')
            debug_mode: 디버그 모드 사용 여부
            streamlit_container: Streamlit 컨테이너 (디버그 로그 표시용)
        """
        self.debug_mode = debug_mode
        self.user_id = user_id
        self.stable_var_dir = stable_var_dir
        self.scenario = scenario
        self.ui_params = ui_params or {}  # UI 파라미터 저장
        self.input_config = config  # 생성자에서 받은 원본 config 저장
        self.model = None
        self.results = None
        self.original_pcf = 0.0  # 기준 PCF
        self.premium_cost_calculator = None  # 프리미엄 비용 계산기
        self.debug_logs = []  # 디버그 로그 저장용
        self.streamlit_container = streamlit_container  # Streamlit 컨테이너
        self.ui_debug_container = streamlit_container  # UI 디버그 출력용 컨테이너
        
        # 🚨 수정: material_specific_targets 속성 초기화
        self.material_specific_targets = self.ui_params.get('material_specific_targets', {})
        
        # 시뮬레이션 데이터 설정
        self.simulation_data = simulation_data or {}
        self.scenario_df = self.simulation_data.get('scenario_df', pd.DataFrame())
        self.ref_formula_df = self.simulation_data.get('ref_formula_df', pd.DataFrame())
        self.ref_proportions_df = self.simulation_data.get('ref_proportions_df', pd.DataFrame())
        self.original_df = self.simulation_data.get('original_df', pd.DataFrame())
        
        # 시뮬레이션 로직 통합: CathodeSimulator 초기화
        self.cathode_simulator = None
        self.cathode_config = {}
        self.electricity_coef = {}
        self._initialize_simulation_components()
        
        # 최적화 설정 (UI 파라미터 우선 적용)
        self.config = self._get_effective_config()
        
        # 🚨 수정: config 업데이트 후 material_specific_targets 동기화
        if 'material_specific_targets' in self.config:
            self.material_specific_targets = self.config['material_specific_targets']
        
        if self.debug_mode:
            self._log_debug(f"🔍 DEBUG - MaterialBasedOptimizer 초기화:")
            self._log_debug(f"  - 시나리오: {self.scenario}")
            self._log_debug(f"  - 사용자 ID: {self.user_id}")
            self._log_debug(f"  - 시뮬레이션 데이터 키: {list(self.simulation_data.keys())}")
            self._log_debug(f"  - 시나리오 데이터 행 수: {len(self.scenario_df)}")
            self._log_debug(f"  - ref_proportions_df 행 수: {len(self.ref_proportions_df)}")
            if len(self.ref_proportions_df) > 0:
                self._log_debug(f"  - ref_proportions_df 컬럼: {list(self.ref_proportions_df.columns)}")
                self._log_debug(f"  - 자재명(포함) 샘플: {self.ref_proportions_df['자재명(포함)'].head(3).tolist() if '자재명(포함)' in self.ref_proportions_df.columns else 'N/A'}")
            else:
                self._log_debug(f"  ⚠️ ref_proportions_df가 비어있음!")
            self._log_debug(f"  - 설정 키: {list(self.config.keys())}")
        
        # 동적 tier 수 파악
        self.num_tiers = self._determine_num_tiers()
        
        if self.debug_mode:
            self._log_debug(f"  - 결정된 tier 수: {self.num_tiers}")
        
        # 유효성 검증
        self._validate_inputs()
        
        if self.debug_mode:
            self._log_debug(f"✅ DEBUG - 유효성 검증 통과")
        
        # 기준 PCF 계산
        self._calculate_original_pcf()
        
        if self.debug_mode:
            self._log_debug(f"  - 기준 PCF: {self.original_pcf:.4f}")
        
        # 최적화 대상 자재 추출 및 분류 (시뮬레이션 로직 기반)
        self._prepare_materials_with_simulation_logic()
        
        # 디버그 정보 출력
        if self.debug_mode:
            self._print_debug_info()
            
        # 매핑 개선사항 검증 (디버그 모드에서만)
        if self.debug_mode:
            validation_results = self._validate_mapping_improvements()
            self.mapping_validation_results = validation_results
        
        # 초기화 후 검증
        self._validate_post_initialization()
        
        if self.debug_mode:
            self._log_debug(f"✅ 초기화 후 검증 통과 - {len(self.target_materials)}개 자재 준비 완료")
    
    def _log_debug(self, message: str, level: str = "INFO") -> None:
        """
        디버그 메시지를 Streamlit/콘솔과 내부 로그에 동시 출력
        
        Args:
            message: 로그 메시지
            level: 로그 레벨 (INFO, WARNING, ERROR)
        """
        if self.debug_mode:
            # Streamlit 컨테이너가 있으면 Streamlit에 출력
            if self.streamlit_container:
                try:
                    # 레벨별 색상 적용
                    if level == "ERROR":
                        self.streamlit_container.error(f"🔴 {message}")
                    elif level == "WARNING":
                        self.streamlit_container.warning(f"🟡 {message}")
                    else:
                        self.streamlit_container.info(f"🔵 {message}")
                except Exception:
                    # Streamlit 출력 실패 시 콘솔로 대체
                    print(message)
            else:
                print(message)  # 콘솔 출력
            
        # 내부 로그에 저장 (debug_mode와 관계없이 항상 저장)
        self.debug_logs.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        })
    
    def _print_debug(self, message: str) -> None:
        """
        디버그 print() 호출을 Streamlit 또는 콘솔로 출력하는 헬퍼 메소드
        
        Args:
            message: 출력할 메시지
        """
        if self.debug_mode:
            # Streamlit 컨테이너가 있으면 Streamlit에 출력
            if self.streamlit_container:
                try:
                    # 메시지에 따라 적절한 Streamlit 함수 선택
                    if any(prefix in message for prefix in ["❌", "⚠️", "ERROR"]):
                        self.streamlit_container.error(message)
                    elif any(prefix in message for prefix in ["🟡", "WARNING"]):
                        self.streamlit_container.warning(message)
                    elif any(prefix in message for prefix in ["✅", "SUCCESS"]):
                        self.streamlit_container.success(message)
                    else:
                        self.streamlit_container.info(message)
                except Exception:
                    # Streamlit 출력 실패 시 콘솔로 대체
                    print(message)
            else:
                print(message)  # 콘솔 출력
    
    def _initialize_simulation_components(self) -> None:
        """시뮬레이션 컴포넌트 초기화 - cathode_configuration.py 로직 통합"""
        if CathodeSimulator is None or FileOperations is None:
            if self.debug_mode:
                self._log_debug("Warning: CathodeSimulator 또는 FileOperations 모듈을 가져올 수 없습니다.", "WARNING")
            return
        
        try:
            # CathodeSimulator 초기화 (사용자별 설정 사용)
            self.cathode_simulator = CathodeSimulator(verbose=False, user_id=self.user_id)
            
            # cathode_configuration.py 설정 로드
            project_root = os.path.join(os.path.dirname(__file__), "..", "..")
            
            # 양극재 관련 설정 파일들
            cathode_ratio_path = os.path.join(project_root, "input", "cathode_ratio.json")
            cathode_site_path = os.path.join(project_root, "input", "cathode_site.json") 
            recycle_ratio_path = os.path.join(project_root, "input", "recycle_material_ratio.json")
            recycle_impact_path = os.path.join(project_root, "stable_var", "recycle_material_impact.json")
            low_carb_metal_path = os.path.join(project_root, "input", "low_carb_metal.json")
            electricity_coef_path = os.path.join(project_root, "stable_var", "electricity_coef_by_country.json")
            
            # 설정 파일들 로드 (사용자별)
            self.cathode_config = {
                'cathode_ratio': FileOperations.load_json(cathode_ratio_path, default={}, user_id=self.user_id),
                'cathode_site': FileOperations.load_json(cathode_site_path, default={}, user_id=self.user_id),
                'recycle_ratio': FileOperations.load_json(recycle_ratio_path, default={}, user_id=self.user_id),
                'recycle_impact': FileOperations.load_json(recycle_impact_path, default={}, user_id=self.user_id),
                'low_carb_metal': FileOperations.load_json(low_carb_metal_path, default={}, user_id=self.user_id),
            }
            
            # 전력배출계수 로드
            self.electricity_coef = FileOperations.load_json(electricity_coef_path, default={}, user_id=self.user_id)
            
            # 시나리오별 전력배출계수 조정
            self._adjust_electricity_coefficients_for_scenario()
            
            if self.debug_mode:
                self._log_debug(f"✅ 시뮬레이션 컴포넌트 초기화 완료 (시나리오: {self.scenario})")
                
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ 시뮬레이션 컴포넌트 초기화 실패: {e}", "ERROR")
            # Fallback: 기본값으로 설정
            self.cathode_config = {}
            self.electricity_coef = {}
    
    def _adjust_electricity_coefficients_for_scenario(self) -> None:
        """시나리오별 전력배출계수 조정"""
        if self.scenario in ['site_change', 'both']:
            # site_change 시나리오: after 사이트의 전력배출계수 사용
            if self.cathode_simulator:
                try:
                    # after 사이트 데이터 생성
                    after_site_data = self.cathode_simulator.generate_baseline_data(site='after')
                    if after_site_data and 'updated_data' in after_site_data:
                        # 전력배출계수를 after 사이트 기준으로 업데이트
                        site_config = self.cathode_config.get('cathode_site', {})
                        
                        # CAM과 pCAM의 after 사이트 기준으로 전력배출계수 설정
                        cam_after = site_config.get('CAM', {}).get('after', '한국')
                        pcam_after = site_config.get('pCAM', {}).get('after', '한국')
                        
                        # 전력배출계수 업데이트 (after 사이트 기준)
                        if cam_after in self.electricity_coef:
                            self.current_electricity_coef = {
                                'CAM': self.electricity_coef[cam_after],
                                'pCAM': self.electricity_coef[pcam_after] if pcam_after in self.electricity_coef else self.electricity_coef.get('한국', 0.4644)
                            }
                        else:
                            self.current_electricity_coef = {'CAM': 0.4644, 'pCAM': 0.4644}  # 기본값
                            
                        if self.debug_mode:
                            self._log_debug(f"🔄 시나리오 {self.scenario}: after 사이트 전력배출계수 적용")
                            self._log_debug(f"   - CAM ({cam_after}): {self.current_electricity_coef['CAM']}")
                            self._log_debug(f"   - pCAM ({pcam_after}): {self.current_electricity_coef['pCAM']}")
                            
                except Exception as e:
                    if self.debug_mode:
                        self._log_debug(f"⚠️ after 사이트 전력배출계수 로드 실패: {e}", "WARNING")
                    self.current_electricity_coef = {'CAM': 0.4644, 'pCAM': 0.4644}
        else:
            # baseline, recycling 시나리오: before 사이트 전력배출계수 사용
            site_config = self.cathode_config.get('cathode_site', {})
            cam_before = site_config.get('CAM', {}).get('before', '중국')
            pcam_before = site_config.get('pCAM', {}).get('before', '한국')
            
            self.current_electricity_coef = {
                'CAM': self.electricity_coef.get(cam_before, 0.7035),  # 중국 기본값
                'pCAM': self.electricity_coef.get(pcam_before, 0.4644)  # 한국 기본값
            }
            
            if self.debug_mode:
                self._log_debug(f"🔄 시나리오 {self.scenario}: before 사이트 전력배출계수 적용")
                self._log_debug(f"   - CAM ({cam_before}): {self.current_electricity_coef['CAM']}")
                self._log_debug(f"   - pCAM ({pcam_before}): {self.current_electricity_coef['pCAM']}")
    
    def _prepare_materials_with_simulation_logic(self) -> None:
        """시뮬레이션 로직 기반 자재 분류 및 준비"""
        self._print_debug(f"🟢 CALLED: _prepare_materials_with_simulation_logic()")
        
        # 저감활동 적용 자재만 추출
        target_materials_df = self.scenario_df[self.scenario_df['저감활동_적용여부'] == 1].copy()
        
        self._print_debug(f"📊 DATA: scenario_df shape: {self.scenario_df.shape}")
        self._print_debug(f"📊 DATA: 저감활동 적용 자재 개수: {len(target_materials_df)}")
        
        if not target_materials_df.empty and '자재명' in target_materials_df.columns:
            all_material_names = target_materials_df['자재명'].tolist()
            self._print_debug(f"📊 DATA: 추출된 모든 자재명: {all_material_names}")
        
        # 유효한 자재명만 필터링하여 리스트 생성
        valid_materials = []
        for material_name in target_materials_df['자재명'].tolist():
            # 유효성 검사
            if material_name is None or (isinstance(material_name, float) and pd.isna(material_name)):
                continue
            if not isinstance(material_name, str):
                try:
                    material_name = str(material_name)
                except:
                    continue
            if material_name.strip():
                valid_materials.append(material_name)
        
        self.target_materials = valid_materials  # 유효한 자재명만 포함
        
        self._print_debug(f"📊 DATA: target_materials 설정 완료: {self.target_materials}")
        self._print_debug(f"📊 DATA: 유효한 자재 개수: {len(valid_materials)}")
        
        # 시뮬레이션 로직 기반 자재 유형 분류
        self.material_types = {}
        
        for idx, row in target_materials_df.iterrows():
            material_name = row['자재명']
            material_category = row['자재품목']
            
            # material_name 유효성 검사 및 건너뛰기
            if material_name is None or (isinstance(material_name, float) and pd.isna(material_name)):
                if self.debug_mode:
                    self._print_debug(f"⚠️ 자재명이 None 또는 NaN이므로 건너뜀 - 행 {idx}")
                continue
            
            # material_name을 문자열로 안전하게 변환
            if not isinstance(material_name, str):
                try:
                    material_name = str(material_name)
                except Exception as e:
                    if self.debug_mode:
                        self._print_debug(f"⚠️ 자재명을 문자열로 변환 실패 - 행 {idx}: {material_name} - {e}")
                    continue
            
            # 빈 문자열 검사
            if not material_name.strip():
                if self.debug_mode:
                    self._print_debug(f"⚠️ 자재명이 빈 문자열이므로 건너뜀 - 행 {idx}: '{material_name}'")
                continue
            
            # 🚨 특별 추적: 문제 자재들 처리
            problem_materials = ['Al Foil', 'Cu Foil', '양극재', '음극재', '전해액']
            if material_name in problem_materials:
                self._print_debug(f"📊 DATA SPECIAL: 문제 자재 '{material_name}' material_types 등록 중")
                self._print_debug(f"  - 자재품목: {material_category}")
                self._print_debug(f"  - 행 정보: {dict(row)}")
            
            # 시뮬레이션 로직에 따른 자재 분류
            material_type = self._classify_material_by_simulation_logic(material_name, material_category, row)
            self.material_types[material_name] = material_type
            
            if material_name in problem_materials:
                self._print_debug(f"📊 DATA SPECIAL: '{material_name}' material_type 분류 완료")
                self._print_debug(f"  - original_emission: {material_type.get('original_emission', 'N/A')}")
                self._print_debug(f"  - quantity: {material_type.get('quantity', 'N/A')}")
                self._print_debug(f"  - is_proportion_applicable: {material_type.get('is_proportion_applicable', 'N/A')}")
        
        self._print_debug(f"📊 DATA: material_types 구성 완료 - 총 {len(self.material_types)}개 자재")
        self._print_debug(f"📊 DATA: material_types keys: {list(self.material_types.keys())}")
    
    def _classify_material_by_simulation_logic(self, material_name: str, material_category: str, row: pd.Series) -> Dict[str, Any]:
        """시뮬레이션 로직에 따른 자재 분류"""
        
        # 1. 양극재 여부 확인 (cathode_configuration.py 대상)
        is_cathode = material_category == '양극재' or '양극재' in material_name.lower()
        
        # 2. Formula 적용 여부 (ref_formula_df 기반)
        is_formula_applicable = self._check_formula_applicable(material_name)
        
        # 3. Proportion 적용 여부 (ref_proportions_df 기반)
        is_proportion_applicable = self._check_proportion_applicable(material_name)
        
        # 4. Ni/Co/Li 자재 여부 (재활용 및 저탄소메탈 적용 대상)
        is_ni_co_li = self._check_is_ni_co_li(material_name)
        
        # 5. Energy(Tier) 자재 여부 (전력배출계수 적용 대상)
        is_energy_tier = 'Energy' in material_category and 'Tier' in material_category
        
        # 시뮬레이션 로직에 따른 처리 우선순위 결정
        processing_priority = self._determine_processing_priority(
            is_cathode, is_formula_applicable, is_proportion_applicable, 
            is_ni_co_li, is_energy_tier
        )
        
        return {
            'is_cathode': is_cathode,
            'is_formula_applicable': is_formula_applicable,
            'is_proportion_applicable': is_proportion_applicable,
            'is_ni_co_li': is_ni_co_li,
            'is_energy_tier': is_energy_tier,
            'processing_priority': processing_priority,
            'original_emission': row['배출계수'],
            'quantity': row['제품총소요량(kg)'],
            'category': material_category
        }
    
    def _determine_processing_priority(self, is_cathode: bool, is_formula: bool, 
                                     is_proportion: bool, is_ni_co_li: bool, 
                                     is_energy_tier: bool) -> str:
        """시뮬레이션 로직에 따른 처리 우선순위 결정"""
        
        # 시뮬레이션 계층구조에 따른 우선순위
        if is_cathode:
            # 양극재: cathode_configuration.py 로직 적용
            if is_ni_co_li:
                return 'cathode_ni_co_li'  # 양극재 중 Ni/Co/Li - 가장 복잡한 로직
            else:
                return 'cathode_general'   # 양극재 중 일般 원소
        elif is_energy_tier:
            # Energy(Tier) 자재: 전력배출계수 및 시나리오별 처리
            return 'energy_tier'
        elif is_proportion:
            # Proportion 적용 자재 (대부분의 자재가 여기에 해당)
            return 'proportion'
        elif is_formula:
            # Formula 적용 자재 (양극재/음극재 중 특수한 경우만)
            return 'formula'
        elif is_ni_co_li:
            # 일반 Ni/Co/Li 자재 (양극재 아님)
            return 'general_ni_co_li'
        else:
            # 기타 일반 자재: tier-RE만 적용
            return 'general'
    
    def _validate_inputs(self) -> None:
        """입력 데이터 유효성 검증"""
        # scenario_df 필수 컬럼 검증
        required_columns = ['자재명', '자재품목', '배출계수', '제품총소요량(kg)', '저감활동_적용여부']
        for col in required_columns:
            if col not in self.scenario_df.columns:
                raise ValueError(f"시나리오 데이터에 필수 컬럼이 없습니다: {col}")
        
        # 저감활동 적용 자재 존재 여부 검증
        reduction_materials = self.scenario_df[self.scenario_df['저감활동_적용여부'] == 1]
        if len(reduction_materials) == 0:
            raise ValueError("저감활동이 적용된 자재가 없습니다. 시뮬레이션에서 최소 1개 이상의 자재에 저감활동을 적용하세요.")
        
        # 감축 목표 설정 검증 (양수 값으로 처리: 10% = 10% 감축)
        if 'reduction_target' in self.config:
            min_reduction = self.config['reduction_target'].get('min', 0)  # 최소 허용 감축률 (더 적은 감축)
            max_reduction = self.config['reduction_target'].get('max', 0)  # 최대 허용 감축률 (더 많은 감축)
            
            if not (0 <= min_reduction < 100):
                raise ValueError(f"최소 감축률은 0% 이상 100% 미만이어야 합니다. 현재값: {min_reduction}%")
            
            if not (0 <= max_reduction < 100):
                raise ValueError(f"최대 감축률은 0% 이상 100% 미만이어야 합니다. 현재값: {max_reduction}%")
            
            # 양수에서는 min이 max보다 작아야 함 (예: min=5%, max=10% → 5%~10% 감축 범위)
            if min_reduction > max_reduction:
                raise ValueError(f"감축률 범위 오류: 최소 감축률({min_reduction}%)이 최대 감축률({max_reduction}%)보다 큽니다. 범위: {min_reduction}%~{max_reduction}% 감축")
    
    def _validate_post_initialization(self) -> None:
        """초기화 후 필수 속성들이 제대로 설정되었는지 검증"""
        # target_materials 검증
        if not hasattr(self, 'target_materials') or not isinstance(self.target_materials, list):
            raise ValueError("target_materials 속성이 올바르게 설정되지 않았습니다")
        
        if len(self.target_materials) == 0:
            raise ValueError("최적화 대상 자재가 없습니다. 시뮬레이션에서 저감활동 적용 자재를 확인하세요")
        
        # material_types 검증
        if not hasattr(self, 'material_types') or not isinstance(self.material_types, dict):
            raise ValueError("material_types 속성이 올바르게 설정되지 않았습니다")
        
        # target_materials와 material_types 일관성 검증
        for material_name in self.target_materials:
            if material_name not in self.material_types:
                raise ValueError(f"자재 '{material_name}'이 material_types에 없습니다")
            
            # 필수 필드 검증
            material_info = self.material_types[material_name]
            required_fields = ['original_emission', 'quantity']
            for field in required_fields:
                if field not in material_info:
                    raise ValueError(f"자재 '{material_name}'에 필수 필드 '{field}'가 없습니다")
                
                if not isinstance(material_info[field], (int, float)) or material_info[field] <= 0:
                    raise ValueError(f"자재 '{material_name}'의 '{field}' 값이 유효하지 않습니다: {material_info[field]}")
        
        # num_tiers 검증
        if not hasattr(self, 'num_tiers') or not isinstance(self.num_tiers, int) or self.num_tiers <= 0:
            self.num_tiers = 2  # 기본값 설정
            if self.debug_mode:
                self._log_debug(f"⚠️ num_tiers가 유효하지 않아 기본값 {self.num_tiers}로 설정", "WARNING")
    
    def _get_effective_config(self) -> Dict[str, Any]:
        """UI 설정 > 사용자 config > 기본값 순서로 설정 적용"""
        base_config = self._get_default_config()
        
        # 사용자 config 적용 (구 시스템 설정 필터링)
        if self.input_config:
            filtered_config = self._filter_legacy_config(self.input_config)
            base_config = self._merge_configs(base_config, filtered_config)
        
        # UI 파라미터 적용 (최우선)
        if self.ui_params:
            base_config = self._apply_ui_params(base_config, self.ui_params)
        
        return base_config
    
    def _filter_legacy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """구 시스템 설정을 필터링하여 MaterialBasedOptimizer에 맞는 설정만 유지"""
        import copy
        filtered = copy.deepcopy(config)
        
        # MaterialBasedOptimizer가 사용하는 설정만 유지
        allowed_keys = {
            'reduction_target',
            're_rates', 
            'material_ratios',
            'constraints',
            'cathode',
            'optimization_scenario',
            'material_specific_targets'
        }
        
        # 구 시스템 설정 제거
        legacy_keys = {
            'objective',      # 구 시스템 목적함수
            'decision_vars',  # 구 시스템 의사결정변수
        }
        
        # 허용된 키만 유지하고 레거시 키는 제거
        result = {}
        for key, value in filtered.items():
            if key in allowed_keys and key not in legacy_keys:
                result[key] = value
        
        if self.debug_mode:
            removed_keys = set(config.keys()) - set(result.keys())
            if removed_keys:
                self._log_debug(f"🔧 구 시스템 설정 제거: {removed_keys}")
        
        return result
    
    def _merge_configs(self, base: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """두 설정을 병합 (깊은 복사)"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key].update(value)
            else:
                result[key] = value
        
        return result
    
    def _apply_ui_params(self, base_config: Dict, ui_params: Dict) -> Dict:
        """UI에서 받은 파라미터를 optimizer 설정으로 매핑"""
        
        # 감축 목표
        if 'reduction_target' in ui_params:
            base_config['reduction_target'] = ui_params['reduction_target']
        
        # RE 적용률 범위 - 두 가지 형식 지원
        if 're_rates' in ui_params:
            # 중첩된 구조 ('re_rates' 딕셔너리)
            base_config['re_rates'] = ui_params['re_rates']
        else:
            # 플랫 구조 ('tier1_re_min', 'tier1_re_max' 등)를 중첩 구조로 변환
            re_rates = {}
            tier_keys = set()
            
            # UI 파라미터에서 tier 관련 키들을 찾아서 tier 번호 추출
            for key in ui_params.keys():
                if key.startswith('tier') and ('_re_min' in key or '_re_max' in key):
                    tier_num = key.split('_')[0]  # 'tier1', 'tier2', etc.
                    tier_keys.add(tier_num)
            
            # 각 tier별로 min/max 값을 추출하여 re_rates 구성
            for tier_key in tier_keys:
                min_key = f'{tier_key}_re_min'
                max_key = f'{tier_key}_re_max'
                
                if min_key in ui_params and max_key in ui_params:
                    re_rates[tier_key] = {
                        'min': ui_params[min_key],
                        'max': ui_params[max_key]
                    }
            
            # 변환된 re_rates가 있으면 적용
            if re_rates:
                base_config['re_rates'] = re_rates
                if self.debug_mode:
                    self._log_debug(f"🔧 플랫 구조 RE 파라미터를 중첩 구조로 변환: {list(re_rates.keys())}")
        
        # 자재 비율
        if 'material_ratios' in ui_params:
            base_config['material_ratios'] = ui_params['material_ratios']
        
        # 추가 제약조건들
        if 'constraints' in ui_params:
            if 'constraints' not in base_config:
                base_config['constraints'] = {}
            base_config['constraints'].update(ui_params['constraints'])
        
        # material_constraints (재활용/저탄소메탈 제약)
        if 'material_constraints' in ui_params:
            material_constraints = ui_params['material_constraints']
            if material_constraints.get('enabled', False):
                # 재활용재 비율
                if material_constraints.get('recycle_ratio', {}).get('enabled', False):
                    recycle_config = material_constraints['recycle_ratio']
                    base_config['material_ratios']['recycle'] = {
                        'min': recycle_config.get('min', 0.05),
                        'max': recycle_config.get('max', 0.5)
                    }
                
                # 저탄소메탈 비율
                if material_constraints.get('low_carbon_ratio', {}).get('enabled', False):
                    low_carbon_config = material_constraints['low_carbon_ratio']
                    base_config['material_ratios']['low_carbon'] = {
                        'min': low_carbon_config.get('min', 0.05),
                        'max': low_carbon_config.get('max', 0.3)
                    }
        
        # 양극재 설정 (cathode)
        if 'cathode' in ui_params:
            base_config['cathode'] = ui_params['cathode']
        
        # 자재별 감축 목표 (material_specific_targets)
        if 'material_specific_targets' in ui_params:
            base_config['material_specific_targets'] = ui_params['material_specific_targets']
            if self.debug_mode:
                material_count = len(ui_params['material_specific_targets'])
                self._log_debug(f"🎯 자재별 감축 목표 설정: {material_count}개 자재")
        
        if self.debug_mode:
            self._log_debug(f"🔧 UI 파라미터 적용 완료:")
            self._log_debug(f"  - 감축목표: {base_config.get('reduction_target', {})}")
            self._log_debug(f"  - 자재비율: {base_config.get('material_ratios', {})}")
            self._log_debug(f"  - RE 비율: {list(base_config.get('re_rates', {}).keys())}")
            self._log_debug(f"  - 자재별 목표: {len(base_config.get('material_specific_targets', {}))}개")
        
        return base_config
    
    def _determine_num_tiers_for_config(self) -> int:
        """설정 생성 시 tier 개수 결정 (초기화 중에 안전하게 사용 가능)"""
        # 시뮬레이션 데이터에서 tier 개수 추출 시도
        if hasattr(self, 'scenario_df') and len(self.scenario_df) > 0:
            # 컬럼명에서 Tier 개수 파악
            tier_columns = [col for col in self.scenario_df.columns if 'Tier' in col and 'RE_case' in col]
            if tier_columns:
                # Tier1_RE_case1에서 Tier 번호 추출
                tier_numbers = set()
                for col in tier_columns:
                    try:
                        tier_num = int(col.split('_')[0].replace('Tier', ''))
                        tier_numbers.add(tier_num)
                    except (ValueError, IndexError):
                        continue
                
                if tier_numbers:
                    max_tier = max(tier_numbers)
                    return max_tier
        
        # 기본값: 2
        return 2
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 최적화 설정 반환"""
        # 동적 tier 설정 생성 - num_tiers가 아직 초기화되지 않은 경우 처리
        if hasattr(self, 'num_tiers'):
            num_tiers = self.num_tiers
        else:
            # num_tiers가 아직 초기화되지 않은 경우, 임시로 결정
            num_tiers = self._determine_num_tiers_for_config()
        
        re_rates = {}
        for tier in range(1, num_tiers + 1):
            re_rates[f'tier{tier}'] = {'min': 0.1, 'max': 0.9}
        
        return {
            'reduction_target': {
                'min': 5,   # 최소 감축률 (%) - 더 작은 감축
                'max': 10,  # 최대 감축률 (%) - 더 큰 감축  
            },
            're_rates': re_rates,
            'material_ratios': {
                'recycle': {'min': 0.05, 'max': 0.5},      # 재활용 비율 범위
                'low_carbon': {'min': 0.05, 'max': 0.3},  # 저탄소메탈 비율 범위
            },
            'constraints': {
                'apply_formula_first': True,  # Formula 로직 우선 적용
                'min_pcf_ratio': 0.05,  # 최소 PCF 유지 비율 (5% 대신 하드코딩 제거)
                'max_tier_reduction': 0.95,  # 최대 tier 감축률 (95%)
                'proportion_max_reduction': 0.8,  # proportion 자재 최대 감축률 (80%)
            }
        }
    
    def _calculate_original_pcf(self) -> None:
        """기준 PCF 계산 (원본 배출계수 × 제품총소요량)"""
        if len(self.scenario_df) == 0:
            self.original_pcf = 0.0
            return
        
        if 'PCF_reference' in self.scenario_df.columns:
            # PCF_reference 컬럼이 있는 경우 합계 사용
            self.original_pcf = self.scenario_df['PCF_reference'].sum()
        elif '배출량(kgCO2eq)' in self.scenario_df.columns:
            # 배출량 컬럼이 있는 경우
            self.original_pcf = self.scenario_df['배출량(kgCO2eq)'].sum()
        elif '배출계수' in self.scenario_df.columns and '제품총소요량(kg)' in self.scenario_df.columns:
            # 배출계수 × 제품총소요량으로 계산
            self.original_pcf = (self.scenario_df['배출계수'] * self.scenario_df['제품총소요량(kg)']).sum()
        else:
            self.original_pcf = 0.0
    
    def _prepare_materials(self) -> None:
        """최적화 대상 자재 추출 및 분류"""
        # 저감활동 적용 자재만 추출
        target_materials_df = self.scenario_df[self.scenario_df['저감활동_적용여부'] == 1].copy()
        self.target_materials = target_materials_df['자재명'].tolist()  # 자재명 리스트로 변환
        
        # 자재 유형 분류
        self.material_types = {}
        
        for idx, row in target_materials_df.iterrows():
            material_name = row['자재명']
            material_category = row['자재품목']
            
            # 양극재 여부 확인
            is_cathode = material_category == '양극재' or '양극재' in material_name.lower()
            
            # Formula 적용 여부 확인
            is_formula_applicable = self._check_formula_applicable(material_name)
            
            # Proportion 적용 대상 (Ni, Co, Li) 여부 확인
            is_proportion_applicable = self._check_proportion_applicable(material_name)
            
            # 자재 유형 설정
            material_type = {
                'is_cathode': is_cathode,
                'is_formula_applicable': is_formula_applicable,
                'is_proportion_applicable': is_proportion_applicable,
                'is_ni_co_li': self._check_is_ni_co_li(material_name),
                'original_emission': row['배출계수'],
                'quantity': row['제품총소요량(kg)']
            }
            
            self.material_types[material_name] = material_type
    
    def _check_formula_applicable(self, material_name: str) -> bool:
        """Formula 적용 가능한 자재인지 확인 (양극재/음극재만 해당)"""
        # material_name 유효성 검사 및 문자열 변환
        if material_name is None or (isinstance(material_name, float) and pd.isna(material_name)):
            if self.debug_mode:
                self._print_debug(f"⚠️ material_name이 None 또는 NaN: {material_name}")
            return False
        
        # material_name을 문자열로 안전하게 변환
        if not isinstance(material_name, str):
            try:
                material_name = str(material_name)
            except Exception as e:
                if self.debug_mode:
                    self._print_debug(f"⚠️ material_name을 문자열로 변환 실패: {material_name} - {e}")
                return False
        
        # 빈 문자열 검사
        if not material_name.strip():
            if self.debug_mode:
                self._print_debug(f"⚠️ material_name이 빈 문자열: '{material_name}'")
            return False
        
        # ref_formula_df에 해당 자재가 있는지 확인
        if len(self.ref_formula_df) == 0:
            if self.debug_mode:
                self._print_debug(f"⚠️ ref_formula_df가 비어있음 - {material_name}")
            return False
        
        # formula는 양극재/음극재만 적용 - 자재명(포함) 컬럼 기준
        formula_materials = self.ref_formula_df['자재명(포함)'].tolist() if '자재명(포함)' in self.ref_formula_df.columns else []
        
        # 양극재/음극재 키워드로 확인
        cathode_anode_keywords = [
            'Cathode Active Material', 'Anode Active Material', 
            '양극재', '음극재'
        ]
        
        # material_name이 양극재/음극재에 해당하는지 확인
        for keyword in cathode_anode_keywords:
            if keyword in material_name:
                if self.debug_mode:
                    self._print_debug(f"✅ Formula 적용 대상 (키워드 매칭): {material_name} (키워드: {keyword})")
                return True
                
        # formula_materials에서도 확인
        for name in formula_materials:
            if isinstance(name, str) and material_name in name:
                if self.debug_mode:
                    self._print_debug(f"✅ Formula 적용 대상 (formula_materials 매칭): {material_name} (매칭: {name})")
                return True
        
        return False
    
    def _check_proportion_applicable(self, material_name: str) -> bool:
        """Proportion 적용 가능한 자재인지 확인"""
        try:
            # material_name 유효성 검사 및 문자열 변환
            if material_name is None or (isinstance(material_name, float) and pd.isna(material_name)):
                if self.debug_mode:
                    self._print_debug(f"⚠️ material_name이 None 또는 NaN: {material_name}")
                return False
            
            # material_name을 문자열로 안전하게 변환
            if not isinstance(material_name, str):
                try:
                    material_name = str(material_name)
                except Exception as e:
                    if self.debug_mode:
                        self._print_debug(f"⚠️ material_name을 문자열로 변환 실패: {material_name} - {e}")
                    return False
            
            # 빈 문자열 검사
            if not material_name.strip():
                if self.debug_mode:
                    print(f"⚠️ material_name이 빈 문자열: '{material_name}'")
                return False
            
            # ref_proportions_df에 해당 자재가 포함되는지 확인
            if len(self.ref_proportions_df) == 0:
                if self.debug_mode:
                    print(f"⚠️ ref_proportions_df가 비어있음 - {material_name}")
                return False
            
            # 컬럼 존재 여부 확인
            if '자재명(포함)' not in self.ref_proportions_df.columns:
                if self.debug_mode:
                    print(f"⚠️ '자재명(포함)' 컬럼이 없음 - 사용 가능한 컬럼: {list(self.ref_proportions_df.columns)}")
                return False
            
            # 자재명이 ref_proportions_df의 자재명(포함)에 포함되는지 확인
            proportion_materials = self.ref_proportions_df['자재명(포함)'].tolist()
            
            # 디버그: 원본 데이터 출력
            if self.debug_mode:
                self._print_debug(f"🔍 proportion_materials 원본: {proportion_materials[:5]}...")  # 처음 5개만
                data_types = [type(mat).__name__ for mat in proportion_materials[:5]]
                self._print_debug(f"🔍 데이터 타입들: {data_types}")
            
            # NaN이나 float 값을 필터링하고 문자열만 사용 (더 강력한 필터링)
            valid_materials = []
            for mat in proportion_materials:
                if mat is None:
                    continue
                if pd.isna(mat):
                    continue
                if not isinstance(mat, str):
                    # 숫자나 다른 타입을 문자열로 변환 시도
                    try:
                        mat_str = str(mat)
                        if mat_str.lower() not in ['nan', 'none', '']:
                            valid_materials.append(mat_str)
                    except:
                        continue
                else:
                    # 이미 문자열인 경우
                    if mat.strip():  # 빈 문자열 제외
                        valid_materials.append(mat)
            
            if self.debug_mode:
                self._print_debug(f"🔍 유효한 materials: {valid_materials}")
            
            # 향상된 매칭 로직 사용 (rule_based.py에서 추출)
            material_category = self.material_types.get(material_name, {}).get('category', '')
            
            # ref_proportions_df의 각 행에 대해 enhanced_material_matching 확인
            for idx, row in self.ref_proportions_df.iterrows():
                try:
                    # enhanced_material_matching을 사용하여 매칭 확인
                    match_found = self._enhanced_material_matching(
                        material_name, material_category, row
                    )
                    
                    if match_found:
                        if self.debug_mode:
                            proportion_name = str(row.get('자재명(포함)', ''))
                            self._print_debug(f"✅ Proportion 적용 대상 (enhanced matching): {material_name} (매칭: {proportion_name})")
                        return True
                        
                except Exception as e:
                    if self.debug_mode:
                        self._print_debug(f"⚠️ Enhanced matching 오류: {e}")
                    continue
            
            return False
            
        except Exception as e:
            if self.debug_mode:
                self._print_debug(f"❌ _check_proportion_applicable 오류 ({material_name}): {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def _check_is_ni_co_li(self, material_name: str) -> bool:
        """Ni, Co, Li 자재인지 확인"""
        # material_name이 문자열이 아닌 경우 처리
        if not isinstance(material_name, str):
            if material_name is None or (isinstance(material_name, float) and pd.isna(material_name)):
                return False
            material_name = str(material_name)
        
        # 자재명에 Ni, Co, Li가 포함되어 있는지 확인
        ni_co_li = ['ni', 'co', 'li', '니켈', '코발트', '리튬']
        return any(element in material_name.lower() for element in ni_co_li)
    
    def _determine_num_tiers(self) -> int:
        """시뮬레이션 설정에서 tier 개수 결정"""
        # 시뮬레이션 데이터에서 tier 개수 추출 시도
        if hasattr(self, 'scenario_df') and len(self.scenario_df) > 0:
            # 컬럼명에서 Tier 개수 파악
            tier_columns = [col for col in self.scenario_df.columns if 'Tier' in col and 'RE_case' in col]
            if tier_columns:
                # Tier1_RE_case1에서 Tier 번호 추출
                tier_numbers = set()
                for col in tier_columns:
                    try:
                        tier_num = int(col.split('_')[0].replace('Tier', ''))
                        tier_numbers.add(tier_num)
                    except (ValueError, IndexError):
                        continue
                
                if tier_numbers:
                    max_tier = max(tier_numbers)
                    if self.debug_mode:
                        self._print_debug(f"📊 시나리오 데이터에서 파악된 최대 Tier: {max_tier}")
                    return max_tier
        
        # 설정 파일에서 tier 개수 추출 시도
        if self.config and 're_rates' in self.config:
            tier_keys = [key for key in self.config['re_rates'].keys() if key.startswith('tier')]
            if tier_keys:
                tier_numbers = []
                for key in tier_keys:
                    try:
                        tier_num = int(key.replace('tier', ''))
                        tier_numbers.append(tier_num)
                    except ValueError:
                        continue
                
                if tier_numbers:
                    max_tier = max(tier_numbers)
                    if self.debug_mode:
                        self._print_debug(f"⚙️ 설정에서 파악된 최대 Tier: {max_tier}")
                    return max_tier
        
        # 기본값: 2
        if self.debug_mode:
            self._print_debug(f"🔧 기본값 사용: Tier 개수 = 2")
        return 2
    
    def _print_material_matching_table(self) -> None:
        """자재별 매칭 정보를 테이블 형태로 출력"""
        print("\n===== 자재별 매칭 정보 테이블 =====")
        
        # 테이블 헤더
        print(f"{'자재명':<40} {'자재품목':<15} {'분류':<15} {'Formula 매칭':<20} {'Proportion 매칭':<25}")
        print("-" * 120)
        
        for material_name in self.target_materials:
            # 안전성 검사: material_name이 유효한지 확인
            if material_name not in self.material_types:
                print(f"⚠️ 경고: '{material_name}'이 material_types에 없음 (건너뜀)")
                continue
                
            # 컬럼명이 잘못 들어온 경우 건너뜀
            if material_name in ['자재명', '자재품목', '배출계수', '제품총소요량(kg)', '배출량(kgCO2eq)']:
                print(f"⚠️ 경고: 컬럼명 '{material_name}'이 자재명으로 잘못 인식됨 (건너뜀)")
                continue
            
            info = self.material_types[material_name]
            category = info['category']
            priority = info['processing_priority']
            
            # Formula 매칭 정보
            formula_match = "❌ 없음"
            if info['is_formula_applicable']:
                matched_formula = self._get_matched_formula_info(material_name)
                if matched_formula:
                    formula_match = f"✅ {matched_formula[:15]}..."
                else:
                    formula_match = "✅ 매칭됨"
            
            # Proportion 매칭 정보  
            proportion_match = "❌ 없음"
            if info['is_proportion_applicable']:
                matched_proportion = self._get_matched_proportion_info(material_name)
                if matched_proportion:
                    proportion_match = f"✅ {matched_proportion[:20]}..."
                else:
                    proportion_match = "✅ 매칭됨"
            
            # 분류 결과
            classification_map = {
                'cathode_ni_co_li': '양극재(복잡)',
                'cathode_general': '양극재(일반)', 
                'energy_tier': 'Energy(Tier)',
                'formula': 'Formula',
                'proportion': 'Proportion',
                'general_ni_co_li': '일반(Ni/Co/Li)',
                'general': '기타'
            }
            classification = classification_map.get(priority, priority)
            
            # 자재명이 너무 길면 줄임
            display_name = material_name[:35] + "..." if len(material_name) > 35 else material_name
            
            print(f"{display_name:<40} {category:<15} {classification:<15} {formula_match:<20} {proportion_match:<25}")
        
        print("-" * 120)
        print(f"총 {len(self.target_materials)}개 자재")
        
        # 분류별 개수 요약
        classification_counts = {}
        for info in self.material_types.values():
            priority = info['processing_priority']
            classification_counts[priority] = classification_counts.get(priority, 0) + 1
        
        print("\n분류별 개수:")
        for priority, count in classification_counts.items():
            classification = classification_map.get(priority, priority)
            print(f"  • {classification}: {count}개")
    
    def _get_matched_formula_info(self, material_name: str) -> str:
        """Formula 매칭 정보 반환"""
        try:
            if len(self.ref_formula_df) == 0:
                return ""
            
            # 양극재/음극재 키워드로 먼저 확인
            cathode_anode_keywords = [
                'Cathode Active Material', 'Anode Active Material', 
                '양극재', '음극재'
            ]
            
            for keyword in cathode_anode_keywords:
                if keyword in material_name:
                    # 해당하는 첫 번째 formula 항목 반환
                    formula_name = self.ref_formula_df.get('자재명(포함)', pd.Series()).iloc[0] if not self.ref_formula_df.empty else ""
                    return str(formula_name) if pd.notna(formula_name) else ""
            
            # formula_materials에서 직접 매칭 확인
            if '자재명(포함)' in self.ref_formula_df.columns:
                formula_materials = self.ref_formula_df['자재명(포함)'].tolist()
                for name in formula_materials:
                    if isinstance(name, str) and material_name in name:
                        return name
            
            return ""
        except:
            return ""
    
    def _get_matched_proportion_info(self, material_name: str) -> str:
        """Proportion 매칭 정보 반환"""
        try:
            if len(self.ref_proportions_df) == 0:
                return ""
            
            proportion_materials = self.ref_proportions_df['자재명(포함)'].tolist()
            material_lower = material_name.lower()
            
            for mat in proportion_materials:
                if pd.isna(mat) or not isinstance(mat, str):
                    continue
                
                mat_lower = mat.lower()
                
                # 양방향 substring 매칭
                if mat_lower in material_lower or material_lower in mat_lower:
                    return mat
                
                # 토큰 기반 매칭
                mat_tokens = set(mat_lower.split())
                material_tokens = set(material_lower.split())
                if len(mat_tokens) > 0 and mat_tokens.issubset(material_tokens):
                    return mat
            
            return ""
        except:
            return ""
    
    def _print_debug_info(self) -> None:
        """시뮬레이션 로직 통합된 디버그 정보 출력"""
        print("===== 시뮬레이션 로직 기반 MaterialBasedOptimizer =====")
        print(f"• 최적화 시나리오: {self.scenario}")
        print(f"• 총 자재 수: {len(self.scenario_df)}개")
        print(f"• 저감활동 적용 자재: {len(self.target_materials)}개")
        print(f"• 기준 PCF: {self.original_pcf:.4f} kgCO2eq")
        print(f"• 설정된 Tier 개수: {self.num_tiers}")
        
        # 자재 매칭 정보 테이블 출력
        self._print_material_matching_table()
        
        # 시나리오별 전력배출계수 정보
        if hasattr(self, 'current_electricity_coef'):
            print(f"• 전력배출계수 (CAM): {self.current_electricity_coef.get('CAM', 'N/A')}")
            print(f"• 전력배출계수 (pCAM): {self.current_electricity_coef.get('pCAM', 'N/A')}")
        
        # 감축 목표 PCF 계산 (양수 감축률을 음수로 변환하여 계산)
        min_reduction = self.config['reduction_target'].get('min', 0) / 100  # 비율로 변환
        max_reduction = self.config['reduction_target'].get('max', 0) / 100  # 비율로 변환
        target_pcf_max = self.original_pcf * (1 - min_reduction)  # 최소 감축률 적용
        target_pcf_min = self.original_pcf * (1 - max_reduction)  # 최대 감축률 적용
        
        print(f"• 최소 감축률: {min_reduction*100:.1f}% → 목표 PCF 최대: {target_pcf_max:.4f} kgCO2eq")
        print(f"• 최대 감축률: {max_reduction*100:.1f}% → 목표 PCF 최소: {target_pcf_min:.4f} kgCO2eq")
        
        # 시뮬레이션 로직 기반 자재 유형별 개수
        priority_counts = {}
        for info in self.material_types.values():
            priority = info['processing_priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        print("\n===== 시뮬레이션 로직 기반 자재 분류 =====")
        priority_names = {
            'cathode_ni_co_li': '양극재 Ni/Co/Li (복잡)',
            'cathode_general': '양극재 일반 원소',
            'energy_tier': 'Energy(Tier) 전력자재',
            'formula': 'Formula 적용 자재',
            'proportion': 'Proportion 적용 자재',
            'general_ni_co_li': '일반 Ni/Co/Li',
            'general': '기타 일반 자재'
        }
        
        for priority, count in priority_counts.items():
            priority_name = priority_names.get(priority, priority)
            print(f"• {priority_name}: {count}개")
        
        # cathode_configuration.py 설정 정보
        if self.cathode_config:
            print("\n===== cathode_configuration.py 설정 정보 =====")
            recycle_ratios = self.cathode_config.get('recycle_ratio', {})
            if recycle_ratios:
                print("• 재활용 비율:", {k: f"{v*100:.1f}%" for k, v in recycle_ratios.items()})
            
            low_carb_config = self.cathode_config.get('low_carb_metal', {})
            if low_carb_config.get('비중'):
                print("• 저탄소메탈 비중:", {k: f"{v:.1f}%" for k, v in low_carb_config['비중'].items()})
        
        # 자재별 상세 정보 (처리 우선순위별로 그룹화)
        print("\n===== 최적화 대상 자재 상세 정보 =====")
        for priority in ['cathode_ni_co_li', 'cathode_general', 'energy_tier', 'formula', 'proportion', 'general_ni_co_li', 'general']:
            materials_in_priority = [name for name, info in self.material_types.items() if info['processing_priority'] == priority]
            
            if materials_in_priority:
                priority_name = priority_names.get(priority, priority)
                print(f"\n📁 {priority_name}:")
                
                for material_name in materials_in_priority:
                    info = self.material_types[material_name]
                    type_desc = []
                    if info['is_cathode']:
                        type_desc.append("양극재")
                    if info['is_energy_tier']:
                        type_desc.append("Energy(Tier)")
                    if info['is_formula_applicable']:
                        type_desc.append("Formula")
                    if info['is_proportion_applicable']:
                        type_desc.append("Proportion")
                    if info['is_ni_co_li']:
                        type_desc.append("Ni/Co/Li")
                    
                    print(f"  • {material_name}")
                    print(f"    - 카테고리: {info['category']}")
                    print(f"    - 유형: {', '.join(type_desc)}")
                    print(f"    - 배출계수: {info['original_emission']:.4f}")
                    print(f"    - 소요량: {info['quantity']:.4f} kg")
                    print(f"    - 배출량: {info['original_emission'] * info['quantity']:.4f} kgCO2eq")
    
    def build_optimization_model(self) -> pyo.ConcreteModel:
        """
        새로운 시나리오별 최적화 모델 구성
        
        - 기존 복잡한 제약조건 시스템 대신 단일 자재 중심의 단순한 최적화
        - 시나리오별 특화된 최적화 방식 적용
        """
        # 🟢 함수 호출 로그
        self._print_debug(f"🟢 CALLED: build_optimization_model()")
        self._print_debug(f"📍 Called from UI - material_based scenario")
        
        if self.debug_mode:
            self._print_debug(f"🏗️ 새로운 시나리오별 최적화 모델 시작:")
            self._print_debug(f"  - 시나리오: {self.scenario}")
            self._print_debug(f"  - 기준 PCF: {self.original_pcf:.4f}")
            self._print_debug(f"  - 대상 자재 수: {len(self.material_types)}")
            self._print_debug(f"📊 DEBUG: material_types 목록: {list(self.material_types.keys()) if hasattr(self, 'material_types') else 'None'}")
        
        # 시나리오별 최적화 결과 저장용
        self.scenario_results = {
            'scenario': self.scenario,
            'materials': {},
            'summary': {}
        }
        
        if self.scenario == 'baseline':
            self._print_debug(f"🔀 BRANCH: 시나리오 'baseline' - _build_baseline_scenario_model() 호출")
            return self._build_baseline_scenario_model()
        elif self.scenario == 'recycling':
            self._print_debug(f"🔀 BRANCH: 시나리오 'recycling' - _build_recycling_scenario_model() 호출")
            return self._build_recycling_scenario_model()
        elif self.scenario == 'site_change':
            self._print_debug(f"🔀 BRANCH: 시나리오 'site_change' - _build_site_change_scenario_model() 호출")
            return self._build_site_change_scenario_model()
        elif self.scenario == 'both':
            self._print_debug(f"🔀 BRANCH: 시나리오 'both' - _build_both_scenario_model() 호출")
            return self._build_both_scenario_model()
        else:
            self._print_debug(f"🔀 BRANCH: 알 수 없는 시나리오 '{self.scenario}' - 기본 처리")
            if self.debug_mode:
                self._print_debug(f"⚠️ 알 수 없는 시나리오: {self.scenario}, baseline으로 처리")
            return self._build_baseline_scenario_model()
    
    def _build_baseline_scenario_model(self) -> pyo.ConcreteModel:
        """기본 시나리오: 단일 자재 RE 최적화"""
        self._print_debug(f"🟢 CALLED: _build_baseline_scenario_model()")
        
        if self.debug_mode:
            self._print_debug(f"📋 기본 시나리오 모델 구성:")
            self._print_debug(f"📊 DEBUG: target_materials = {getattr(self, 'target_materials', 'Not found')}")
        
        # 각 자재별로 개별 최적화 수행
        for material_name in self.target_materials:
            if self.debug_mode:
                self._print_debug(f"  🔧 {material_name} 최적화 중...")
            
            try:
                # 🚨 여기서 optimize_single_material_re() 호출됨!
                self._print_debug(f"🟢 CALLING: optimize_single_material_re('{material_name}') - 실제 호출!")
                result = self.optimize_single_material_re(material_name)
                self._print_debug(f"✅ RESULT: optimize_single_material_re('{material_name}') 완료 - status: {result.get('status', 'N/A')}")
                
                # 결과 검증 - 필수 필드가 있는지 확인
                if not isinstance(result, dict):
                    result = {
                        'status': 'error',
                        'message': f'잘못된 결과 형식: {type(result)}',
                        'material_name': material_name
                    }
                elif 'status' not in result:
                    result['status'] = 'error'
                    result['message'] = '상태 정보 누락'
                    result['material_name'] = material_name
                
                self.scenario_results['materials'][material_name] = result
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                
                self._print_debug(f"❌ EXCEPTION: {material_name} 최적화 중 예외 발생: {str(e)}")
                if self.debug_mode:
                    self._print_debug(f"❌ 상세 오류:\n{error_details}")
                
                # 예외 발생 시 오류 결과 생성
                error_result = {
                    'status': 'error',
                    'message': f'최적화 중 예외 발생: {str(e)}',
                    'material_name': material_name,
                    'error_type': type(e).__name__,
                    'error_details': error_details
                }
                
                self.scenario_results['materials'][material_name] = error_result
        
        # 더미 모델 생성 (기존 인터페이스 호환성 유지)
        model = pyo.ConcreteModel()
        model.scenario_type = 'baseline'
        model.results = self.scenario_results
        
        if self.debug_mode:
            successful_optimizations = sum(1 for r in self.scenario_results['materials'].values() 
                                        if r.get('status') == 'optimal')
            self._print_debug(f"  ✅ 기본 시나리오 완료: {successful_optimizations}/{len(self.target_materials)} 성공")
        
        self.model = model
        return model
    
    def _build_recycling_scenario_model(self) -> pyo.ConcreteModel:
        """재활용&저탄소 시나리오: 양극재 비율 + RE 최적화"""
        if self.debug_mode:
            self._print_debug(f"♻️ 재활용&저탄소 시나리오 모델 구성:")
        
        # 각 자재별로 개별 최적화 수행
        for material_name in self.target_materials:
            if self.debug_mode:
                self._print_debug(f"  🔧 {material_name} 최적화 중...")
            
            # 양극재인 경우 재활용 최적화, 아닌 경우 기본 RE 최적화
            material_info = self.material_types.get(material_name, {})
            
            # 🚨 Streamlit UI에 라우팅 정보 표시
            if hasattr(self, 'ui_debug_container') and self.ui_debug_container:
                with self.ui_debug_container:
                    import streamlit as st
                    st.write(f"🔄 **{material_name} 최적화 라우팅**")
                    st.write(f"📋 자재 정보: {material_info}")
                    
                    is_cathode = material_info.get('is_cathode', False)
                    if is_cathode:
                        st.info(f"🔥 {material_name}은 양극재 → 재활용 최적화 사용")
                    else:
                        st.info(f"🔍 {material_name}은 일반 자재 → 기본 RE 최적화 사용")
            
            print(f"\n🔍 [SCENARIO_DEBUG] 재활용 시나리오 - {material_name} 처리")
            print(f"🔍 [SCENARIO_DEBUG] material_info: {material_info}")
            print(f"🔍 [SCENARIO_DEBUG] is_cathode: {material_info.get('is_cathode', False)}")
            
            if material_info.get('is_cathode', False):
                print(f"🔥 [SCENARIO_DEBUG] {material_name}은 양극재 - optimize_cathode_with_recycling() 호출")
                result = self.optimize_cathode_with_recycling(material_name)
                print(f"🔥 [SCENARIO_DEBUG] optimize_cathode_with_recycling() 결과: {result.get('status', 'None')}")
                if self.debug_mode:
                    self._print_debug(f"    ✅ 양극재 재활용 최적화 완료")
            else:
                print(f"🔍 [SCENARIO_DEBUG] {material_name}은 일반 자재 - optimize_single_material_re() 호출")
                result = self.optimize_single_material_re(material_name)
                print(f"🔍 [SCENARIO_DEBUG] optimize_single_material_re() 결과: {result.get('status', 'None')}")
                if self.debug_mode:
                    self._print_debug(f"    ✅ 일반 자재 RE 최적화 완료")
            
            self.scenario_results['materials'][material_name] = result
        
        # 더미 모델 생성 (기존 인터페이스 호환성 유지)
        model = pyo.ConcreteModel()
        model.scenario_type = 'recycling'
        model.results = self.scenario_results
        
        if self.debug_mode:
            successful_optimizations = sum(1 for r in self.scenario_results['materials'].values() 
                                        if r.get('status') == 'optimal')
            cathode_materials = sum(1 for name in self.target_materials 
                                  if self.material_types.get(name, {}).get('is_cathode', False))
            self._print_debug(f"  ✅ 재활용 시나리오 완료: {successful_optimizations}/{len(self.target_materials)} 성공")
            self._print_debug(f"    - 양극재: {cathode_materials}개 (재활용 비율 최적화)")
            self._print_debug(f"    - 일반자재: {len(self.target_materials) - cathode_materials}개 (RE 최적화)")
        
        self.model = model
        return model
    
    def _build_site_change_scenario_model(self) -> pyo.ConcreteModel:
        """사이트 변경 시나리오: 다국가 비교 + 최적 사이트 선택"""
        if self.debug_mode:
            print(f"🌍 사이트 변경 시나리오 모델 구성:")
        
        # 비교할 국가 목록 (기본값)
        countries = ['한국', '중국', '미국', '독일', '일본']
        
        # 각 자재별로 다국가 사이트 선택 최적화 수행
        for material_name in self.target_materials:
            if self.debug_mode:
                print(f"  🔧 {material_name} 다국가 최적화 중...")
            
            result = self.optimize_site_selection(material_name, countries)
            self.scenario_results['materials'][material_name] = result
        
        # 더미 모델 생성 (기존 인터페이스 호환성 유지)
        model = pyo.ConcreteModel()
        model.scenario_type = 'site_change'
        model.results = self.scenario_results
        
        if self.debug_mode:
            successful_optimizations = sum(1 for r in self.scenario_results['materials'].values() 
                                        if r.get('status') == 'optimal')
            print(f"  ✅ 사이트 변경 시나리오 완료: {successful_optimizations}/{len(self.target_materials)} 성공")
            
            # 선택된 국가별 통계
            selected_countries = {}
            for material_name, result in self.scenario_results['materials'].items():
                if result.get('status') == 'optimal' and 'best_country' in result:
                    country = result['best_country']
                    selected_countries[country] = selected_countries.get(country, 0) + 1
            
            print(f"    - 선택된 국가별 자재 수:")
            for country, count in selected_countries.items():
                print(f"      • {country}: {count}개")
        
        self.model = model
        return model
    
    def _build_both_scenario_model(self) -> pyo.ConcreteModel:
        """통합 시나리오: 재활용 + 사이트 변경"""
        if self.debug_mode:
            print(f"🔄 통합 시나리오 모델 구성 (재활용 + 사이트 변경):")
        
        # 비교할 국가 목록
        countries = ['한국', '중국', '미국', '독일', '일본']
        
        # 각 자재별로 최적화 수행
        for material_name in self.target_materials:
            if self.debug_mode:
                print(f"  🔧 {material_name} 통합 최적화 중...")
            
            material_info = self.material_types.get(material_name, {})
            
            if material_info.get('is_cathode', False):
                # 양극재: 각 국가에서 재활용 최적화 수행 후 최적 국가 선택
                country_results = {}
                best_country = None
                best_pcf = float('inf')
                
                for country in countries:
                    try:
                        # 해당 국가에서 재활용 최적화
                        country_result = self.optimize_cathode_with_recycling(material_name, country)
                        country_results[country] = country_result
                        
                        if (country_result.get('status') == 'optimal' and 
                            country_result.get('optimized_pcf', float('inf')) < best_pcf):
                            best_pcf = country_result['optimized_pcf']
                            best_country = country
                            
                    except Exception as e:
                        if self.debug_mode:
                            print(f"    ⚠️ {country} 최적화 실패: {e}")
                        country_results[country] = {'status': 'error', 'message': str(e)}
                
                # 최적 결과 저장
                result = {
                    'status': 'optimal' if best_country else 'error',
                    'best_country': best_country,
                    'best_pcf': best_pcf,
                    'country_results': country_results,
                    'material_type': 'cathode_both'
                }
                
                if best_country:
                    result.update(country_results[best_country])
                    if self.debug_mode:
                        print(f"    ✅ 양극재 통합 최적화: 최적 국가 = {best_country}")
                
            else:
                # 일반 자재: 다국가 RE 최적화
                result = self.optimize_site_selection(material_name, countries)
                result['material_type'] = 'general_both'
                if self.debug_mode and result.get('status') == 'optimal':
                    print(f"    ✅ 일반 자재 사이트 선택: 최적 국가 = {result.get('best_country', 'N/A')}")
            
            self.scenario_results['materials'][material_name] = result
        
        # 더미 모델 생성 (기존 인터페이스 호환성 유지)
        model = pyo.ConcreteModel()
        model.scenario_type = 'both'
        model.results = self.scenario_results
        
        if self.debug_mode:
            successful_optimizations = sum(1 for r in self.scenario_results['materials'].values() 
                                        if r.get('status') == 'optimal')
            cathode_materials = sum(1 for name in self.target_materials 
                                  if self.material_types.get(name, {}).get('is_cathode', False))
            print(f"  ✅ 통합 시나리오 완료: {successful_optimizations}/{len(self.target_materials)} 성공")
            print(f"    - 양극재: {cathode_materials}개 (재활용 + 사이트 최적화)")
            print(f"    - 일반자재: {len(self.target_materials) - cathode_materials}개 (사이트 최적화)")
        
        self.model = model
        return model
    
    def _set_material_params(self, model: pyo.ConcreteModel) -> None:
        """시뮬레이션 로직 기반 자재별 파라미터 설정"""
        # 자재 목록 인덱스 설정
        material_names = list(self.material_types.keys())
        model.materials = pyo.Set(initialize=material_names)
        
        if self.debug_mode:
            print(f"  📋 자재 목록 설정: {len(material_names)}개")
            for i, name in enumerate(material_names[:3]):  # 처음 3개만 출력
                print(f"    • {name}")
            if len(material_names) > 3:
                print(f"    ... 외 {len(material_names)-3}개")
        
        # 자재별 기본 정보
        emission_init = {name: info['original_emission'] for name, info in self.material_types.items()}
        quantity_init = {name: info['quantity'] for name, info in self.material_types.items()}
        
        model.original_emission = pyo.Param(model.materials, initialize=emission_init)
        model.quantity = pyo.Param(model.materials, initialize=quantity_init)
        
        if self.debug_mode:
            print(f"  📊 기본 정보 파라미터 설정:")
            print(f"    • 배출계수 범위: {min(emission_init.values()):.4f} ~ {max(emission_init.values()):.4f}")
            print(f"    • 소요량 범위: {min(quantity_init.values()):.4f} ~ {max(quantity_init.values()):.4f} kg")
        
        # 시뮬레이션 로직 기반 자재별 유형 정보
        cathode_init = {name: 1 if info['is_cathode'] else 0 for name, info in self.material_types.items()}
        formula_init = {name: 1 if info['is_formula_applicable'] else 0 for name, info in self.material_types.items()}
        proportion_init = {name: 1 if info['is_proportion_applicable'] else 0 for name, info in self.material_types.items()}
        ni_co_li_init = {name: 1 if info['is_ni_co_li'] else 0 for name, info in self.material_types.items()}
        energy_tier_init = {name: 1 if info['is_energy_tier'] else 0 for name, info in self.material_types.items()}
        
        model.is_cathode = pyo.Param(model.materials, initialize=cathode_init)
        model.is_formula = pyo.Param(model.materials, initialize=formula_init)
        model.is_proportion = pyo.Param(model.materials, initialize=proportion_init)
        model.is_ni_co_li = pyo.Param(model.materials, initialize=ni_co_li_init)
        model.is_energy_tier = pyo.Param(model.materials, initialize=energy_tier_init)
        
        if self.debug_mode:
            print(f"  🏷️ 자재 유형별 개수:")
            print(f"    • 양극재: {sum(cathode_init.values())}개")
            print(f"    • Formula 적용: {sum(formula_init.values())}개")
            print(f"    • Proportion 적용: {sum(proportion_init.values())}개")
            print(f"    • Ni/Co/Li: {sum(ni_co_li_init.values())}개")
            print(f"    • Energy(Tier): {sum(energy_tier_init.values())}개")
        
        # 시뮬레이션 로직 기반 처리 우선순위
        processing_priority_map = {priority: idx for idx, priority in enumerate([
            'cathode_ni_co_li', 'cathode_general', 'energy_tier', 'formula', 
            'proportion', 'general_ni_co_li', 'general'
        ])}
        
        priority_init = {name: info['processing_priority'] for name, info in self.material_types.items()}
        model.processing_priority = pyo.Param(model.materials, initialize=priority_init, within=pyo.Any)
        
        if self.debug_mode:
            print(f"  🎯 처리 우선순위 설정 완료: {len(set(priority_init.values()))}가지 유형")
    
    def _define_decision_variables(self, model: pyo.ConcreteModel) -> None:
        """의사결정 변수 정의"""
        if self.debug_mode:
            print(f"  🎛️ 의사결정 변수 정의 시작:")
        
        # Tier별 RE 적용률 (동적 생성)
        self.tier_vars = {}
        for tier in range(1, self.num_tiers + 1):
            var_name = f'tier{tier}_re'
            
            # 설정에서 범위 가져오기
            tier_key = f'tier{tier}'
            if tier_key in self.config['re_rates']:
                tier_min = self.config['re_rates'][tier_key]['min']
                tier_max = self.config['re_rates'][tier_key]['max']
            else:
                tier_min, tier_max = 0.0, 1.0
            
            # 🔧 버그 수정: 범위의 중간값으로 초기화 (기존 고정값 방식 대신)
            initial_value = (tier_min + tier_max) / 2
            
            # 초기값이 범위를 벗어나지 않도록 안전 검증
            initial_value = max(tier_min, min(tier_max, initial_value))
            
            tier_var = pyo.Var(model.materials, bounds=(tier_min, tier_max), initialize=initial_value)
            setattr(model, var_name, tier_var)
            self.tier_vars[tier] = tier_var
            
            if self.debug_mode:
                self._log_debug(f"    • {var_name}: 초기값={initial_value:.3f} (범위 중간값), 범위=[{tier_min:.3f}, {tier_max:.3f}]")
        
        # Ni, Co, Li 자재에 대한 추가 변수 (해당 자재만 생성)
        recycle_config = self.config.get('material_ratios', {}).get('recycle', {})
        low_carbon_config = self.config.get('material_ratios', {}).get('low_carbon', {})
        
        recycle_min = recycle_config.get('min', 0.0)
        recycle_max = recycle_config.get('max', 1.0)
        low_carbon_min = low_carbon_config.get('min', 0.0)
        low_carbon_max = low_carbon_config.get('max', 1.0)
        
        # 자재별 비율 변수 bounds 함수 정의
        def recycle_bounds_rule(model, m):
            if self.material_types[m]['is_ni_co_li']:
                return (recycle_min, recycle_max)
            else:
                return (0.0, 0.0)  # 일반 자재는 재활용 비율 0으로 고정
        
        def low_carbon_bounds_rule(model, m):
            if self.material_types[m]['is_ni_co_li']:
                return (low_carbon_min, low_carbon_max)
            else:
                return (0.0, 0.0)  # 일반 자재는 저탄소메탈 비율 0으로 고정
        
        def virgin_bounds_rule(model, m):
            if self.material_types[m]['is_ni_co_li']:
                return (0.0, 1.0)
            else:
                return (1.0, 1.0)  # 일반 자재는 신재 비율 1로 고정
        
        # 자재별 초기값 함수 정의
        def recycle_init_rule(model, m):
            if self.material_types[m]['is_ni_co_li']:
                return 0.1
            else:
                return 0.0  # 일반 자재는 0
        
        def low_carbon_init_rule(model, m):
            if self.material_types[m]['is_ni_co_li']:
                return 0.05
            else:
                return 0.0  # 일반 자재는 0
        
        def virgin_init_rule(model, m):
            if self.material_types[m]['is_ni_co_li']:
                return 0.85
            else:
                return 1.0  # 일반 자재는 1
        
        # 재활용 비율 (Ni/Co/Li 자재만 변동, 일반 자재는 0 고정)
        model.recycle_ratio = pyo.Var(model.materials, bounds=recycle_bounds_rule, initialize=recycle_init_rule)
        # 저탄소 메탈 비율 (Ni/Co/Li 자재만 변동, 일반 자재는 0 고정)
        model.low_carbon_ratio = pyo.Var(model.materials, bounds=low_carbon_bounds_rule, initialize=low_carbon_init_rule)
        # 신재 비율 (Ni/Co/Li 자재: 1 - 재활용 - 저탄소, 일반 자재: 1 고정)
        model.virgin_ratio = pyo.Var(model.materials, bounds=virgin_bounds_rule, initialize=virgin_init_rule)
        
        if self.debug_mode:
            # Ni/Co/Li 자재 개수 확인
            ni_co_li_count = sum(1 for info in self.material_types.values() if info['is_ni_co_li'])
            general_count = len(model.materials) - ni_co_li_count
            
            print(f"    • recycle_ratio: Ni/Co/Li 자재용 범위=[{recycle_min:.3f}, {recycle_max:.3f}], 일반자재=0 고정")
            print(f"    • low_carbon_ratio: Ni/Co/Li 자재용 범위=[{low_carbon_min:.3f}, {low_carbon_max:.3f}], 일반자재=0 고정")
            print(f"    • virgin_ratio: Ni/Co/Li 자재용 범위=[0.000, 1.000], 일반자재=1 고정")
            
            # 자재별 비율 변수 상태 상세 정보
            print(f"  📋 자재별 비율 변수 설정 상세:")
            
            # 모든 자재에 대해 변수 bounds 확인
            for name, info in self.material_types.items():
                material_type_desc = "Ni/Co/Li" if info['is_ni_co_li'] else "일반"
                processing_priority = info['processing_priority']
                
                # 실제 변수 bounds 확인
                recycle_bounds = recycle_bounds_rule(model, name)
                low_carbon_bounds = low_carbon_bounds_rule(model, name)
                virgin_bounds = virgin_bounds_rule(model, name)
                
                # 실제 초기값 확인
                recycle_init = recycle_init_rule(model, name)
                low_carbon_init = low_carbon_init_rule(model, name)
                virgin_init = virgin_init_rule(model, name)
                
                print(f"    • {name} ({material_type_desc}, {processing_priority}):")
                print(f"      - recycle_ratio: bounds={recycle_bounds}, init={recycle_init:.3f}")
                print(f"      - low_carbon_ratio: bounds={low_carbon_bounds}, init={low_carbon_init:.3f}")
                print(f"      - virgin_ratio: bounds={virgin_bounds}, init={virgin_init:.3f}")
                print(f"      - 원본 배출계수: {info['original_emission']:.4f}")
                print(f"      - 소요량: {info['quantity']:.4f} kg")
                
                # 변수 일관성 검증
                if info['is_ni_co_li']:
                    # Ni/Co/Li 자재: 비율 합계가 1이 되는지 확인
                    total_init = recycle_init + low_carbon_init + virgin_init
                    if abs(total_init - 1.0) > 0.001:
                        print(f"      ⚠️ 비율 합계 불일치: {total_init:.3f} ≠ 1.0")
                else:
                    # 일반 자재: 고정값이 올바른지 확인
                    if recycle_init != 0.0 or low_carbon_init != 0.0 or virgin_init != 1.0:
                        print(f"      ⚠️ 일반 자재 비율 오류: 재활용={recycle_init}, 저탄소메탈={low_carbon_init}, 신재={virgin_init}")
            
            # 요약 정보
            print(f"  📊 변수 설정 요약:")
            print(f"    • 총 자재: {len(self.material_types)}개")
            print(f"    • Ni/Co/Li 자재: {ni_co_li_count}개 (비율 최적화)")
            print(f"    • 일반 자재: {general_count}개 (비율 고정)")
            
            # 자재 유형별 그룹화
            materials_by_priority = {}
            for name, info in self.material_types.items():
                priority = info['processing_priority']
                if priority not in materials_by_priority:
                    materials_by_priority[priority] = []
                materials_by_priority[priority].append(name)
            
            print(f"    • 처리 우선순위별:")
            for priority, materials in materials_by_priority.items():
                print(f"      - {priority}: {len(materials)}개 ({', '.join(materials[:2])}{'...' if len(materials) > 2 else ''})")
        
        # 자재별 최종 배출계수 (계산용)
        model.modified_emission = pyo.Var(model.materials, bounds=(0, None), initialize=0.0)
        
        if self.debug_mode:
            print(f"    • modified_emission: 범위=[0.000, ∞]")
            print(f"  📊 변수 요약:")
            print(f"    • Tier RE 변수: {self.num_tiers}개 × {len(model.materials)}자재 = {self.num_tiers * len(model.materials)}개")
            print(f"    • 자재 비율 변수: 3개 × {len(model.materials)}자재 = {3 * len(model.materials)}개")
            print(f"      - Ni/Co/Li 자재: {ni_co_li_count}개 (비율 최적화)")
            print(f"      - 일반 자재: {general_count}개 (비율 고정)")
            print(f"    • 배출계수 변수: {len(model.materials)}개")
            total_vars = (self.num_tiers + 3 + 1) * len(model.materials)
            print(f"    • 총 변수 개수: {total_vars}개")
    
    def _set_objective_function(self, model: pyo.ConcreteModel) -> None:
        """목적함수: 총 탄소배출량 최소화 (배출계수 × 소요량의 합)"""
        def total_carbon_emission(model):
            return sum(model.modified_emission[m] * model.quantity[m] for m in model.materials)
        
        model.objective = pyo.Objective(rule=total_carbon_emission, sense=pyo.minimize)
        
        if self.debug_mode:
            # 목적함수 구성 요소 확인
            original_total = sum(self.material_types[m]['original_emission'] * self.material_types[m]['quantity'] 
                               for m in model.materials)
            print(f"  🎯 목적함수 설정:")
            print(f"    • 유형: 총 탄소배출량 최소화 (minimize)")
            print(f"    • 구성: Σ(modified_emission[m] × quantity[m]) for all materials")
            print(f"    • 기준 총배출량: {original_total:.4f} kgCO2eq")
            print(f"    • 자재 개수: {len(model.materials)}개")
    
    def _set_constraints(self, model: pyo.ConcreteModel) -> None:
        """제약조건 설정"""
        if self.debug_mode:
            print(f"  ⚖️ 제약조건 설정 시작:")
        
        # 배출계수 계산 제약조건
        if self.debug_mode:
            print(f"    🧮 배출계수 계산 제약조건 설정 중...")
        self._set_emission_calculation_constraints(model)
        
        # RE 적용률 범위 제약
        if self.debug_mode:
            print(f"    📊 RE 적용률 범위 제약조건 설정 중...")
        self._set_re_rate_constraints(model)
        
        # Ni, Co, Li 자재 비율 제약
        if self.debug_mode:
            print(f"    🔄 자재 비율 제약조건 설정 중...")
        self._set_material_ratio_constraints(model)
        
        # 총 PCF 감축 목표 제약
        if self.debug_mode:
            print(f"    🎯 PCF 감축 목표 제약조건 설정 중...")
        self._set_pcf_reduction_target_constraints(model)
        
        # 프리미엄 비용 제약은 단순화를 위해 제거 (필요시 나중에 추가)
        
        if self.debug_mode:
            # 제약조건 개수 요약
            constraint_count = len(list(model.component_objects(pyo.Constraint)))
            print(f"  ✅ 단순화된 제약조건 설정 완료: 총 {constraint_count}개")
    
    def _set_emission_calculation_constraints(self, model: pyo.ConcreteModel) -> None:
        """단순화된 배출계수 계산 제약조건 - 시나리오별 최적화"""
        
        if self.debug_mode:
            print(f"  🧮 단순화된 배출계수 계산 제약조건 설정:")
        
        # 시나리오별 분기 처리
        if self.scenario == 'baseline':
            self._set_basic_re_constraints(model)
        elif self.scenario == 'recycling':
            self._set_recycling_constraints(model)
        elif self.scenario == 'both':
            self._set_recycling_constraints(model)
        elif self.scenario == 'site_change':
            self._set_basic_re_constraints(model)  # RE 최적화는 동일
        else:
            self._set_basic_re_constraints(model)  # 기본값
    
    def _set_basic_re_constraints(self, model: pyo.ConcreteModel) -> None:
        """기본 RE 최적화 제약조건 (단일 자재 중심)"""
        
        def basic_emission_rule(model, m):
            # 단순한 tier RE 적용 로직
            tier_reduction = 0
            
            # proportion 자재인 경우 proportion 값 고려
            info = self.material_types.get(m, {})
            if info.get('is_proportion_applicable', False):
                for tier in range(1, self.num_tiers + 1):
                    tier_var = getattr(model, f'tier{tier}_re')
                    proportion_tier_value = self._get_proportion_tier_value(m, tier)
                    tier_reduction += tier_var[m] * proportion_tier_value
            else:
                # 일반 자재는 직접 합계
                for tier in range(1, self.num_tiers + 1):
                    tier_var = getattr(model, f'tier{tier}_re')
                    tier_reduction += tier_var[m]
            
            return model.modified_emission[m] == model.original_emission[m] * (1 - tier_reduction)
        
        model.basic_emission_constraint = pyo.Constraint(model.materials, rule=basic_emission_rule)
        
        # tier 감축률 상한 제약
        def tier_limit_rule(model, m):
            tier_sum = 0
            info = self.material_types.get(m, {})
            
            for tier in range(1, self.num_tiers + 1):
                tier_var = getattr(model, f'tier{tier}_re')
                if info.get('is_proportion_applicable', False):
                    proportion_tier_value = self._get_proportion_tier_value(m, tier)
                    tier_sum += tier_var[m] * proportion_tier_value
                else:
                    tier_sum += tier_var[m]
            
            max_reduction = self.config.get('constraints', {}).get('max_tier_reduction', 0.8)
            return tier_sum <= max_reduction  # 설정값 기반 최대 감축 제한
        
        model.tier_limit_constraint = pyo.Constraint(model.materials, rule=tier_limit_rule)
        
        if self.debug_mode:
            print(f"    ✅ 기본 RE 제약조건 설정 완료: {len(self.material_types)}개 자재")
    
    def _set_recycling_constraints(self, model: pyo.ConcreteModel) -> None:
        """재활용&저탄소 시나리오 제약조건"""
        
        def recycling_emission_rule(model, m):
            info = self.material_types.get(m, {})
            
            # 양극재인 경우 재활용/저탄소 비율 적용
            if info.get('is_cathode', False):
                # 재활용재와 저탄소재 배출계수 (설정값 기반)
                cathode_config = self.config.get('cathode', {})
                recycle_reduction = cathode_config.get('recycle_emission_reduction', 0.6)  # 기본 60% 감축
                low_carbon_reduction = cathode_config.get('low_carbon_emission_reduction', 0.3)  # 기본 30% 감축
                
                recycle_emission = model.original_emission[m] * (1 - recycle_reduction)
                low_carbon_emission = model.original_emission[m] * (1 - low_carbon_reduction)
                
                # tier RE 적용
                tier_reduction = 0
                for tier in range(1, self.num_tiers + 1):
                    tier_var = getattr(model, f'tier{tier}_re')
                    tier_reduction += tier_var[m]
                
                # 혼합 배출계수 계산
                mixed_emission = (model.virgin_ratio[m] * model.original_emission[m] + 
                                model.recycle_ratio[m] * recycle_emission + 
                                model.low_carbon_ratio[m] * low_carbon_emission)
                
                return model.modified_emission[m] == mixed_emission * (1 - tier_reduction)
            else:
                # 비양극재는 기본 RE 로직
                tier_reduction = 0
                for tier in range(1, self.num_tiers + 1):
                    tier_var = getattr(model, f'tier{tier}_re')
                    tier_reduction += tier_var[m]
                
                return model.modified_emission[m] == model.original_emission[m] * (1 - tier_reduction)
        
        model.recycling_emission_constraint = pyo.Constraint(model.materials, rule=recycling_emission_rule)
        
        if self.debug_mode:
            print(f"    ✅ 재활용 제약조건 설정 완료: {len(self.material_types)}개 자재")
    
    def _extract_element_name(self, material_name: str) -> Optional[str]:
        """자재명에서 원소명 추출 (Ni, Co, Li)"""
        material_lower = material_name.lower()
        
        if 'ni' in material_lower or '니켈' in material_lower:
            return 'Ni'
        elif 'co' in material_lower or '코발트' in material_lower:
            return 'Co'
        elif 'li' in material_lower or '리튬' in material_lower:
            return 'Li'
        else:
            return None
    
    def _calculate_tier_reduction(self, model: pyo.ConcreteModel, material_var: str):
        """UI에서 설정한 RE 비율을 직접 사용하는 감축률 계산"""
        tier_reduction = 0
        info = self.material_types.get(material_var, {})
        
        if info.get('is_proportion_applicable', False):
            # proportion 자재는 UI RE 비율과 ref_proportions_df 값을 곱해서 사용
            for tier in range(1, self.num_tiers + 1):
                tier_var = getattr(model, f'tier{tier}_re')
                ui_re_value = tier_var[material_var]  # UI에서 설정한 RE 비율
                
                # ref_proportions_df에서 해당 자재의 tier 비율 가져오기
                proportion_tier_value = self._get_proportion_tier_value(material_var, tier)
                
                # 두 값을 곱해서 실제 감축률 계산
                tier_reduction += ui_re_value * proportion_tier_value
                
                if self.debug_mode and proportion_tier_value > 0:
                    ui_re_val = pyo.value(ui_re_value) if hasattr(ui_re_value, 'value') else float(ui_re_value)
                    tier_product = ui_re_val * proportion_tier_value
                    print(f"    🔍 {material_var} Tier{tier}: UI({ui_re_val:.3f}) × Prop({proportion_tier_value:.3f}) = {tier_product:.3f}")
        else:
            # 기존 로직 (formula 등)
            for tier in range(1, self.num_tiers + 1):
                tier_var = getattr(model, f'tier{tier}_re')
                tier_reduction += tier_var[material_var]
        
        # 최대 감축률 제한은 constraint rule에서 직접 처리
        # Pyomo 표현식에서는 min() 사용 불가
        return tier_reduction
    
    def _get_proportion_tier_value(self, material_name: str, tier: int) -> float:
        """proportion 자재의 ref_proportions_df에서 tier 값 가져오기 - 개선된 버전"""
        try:
            if len(self.ref_proportions_df) == 0:
                if self.debug_mode:
                    self._log_debug(f"⚠️ ref_proportions_df가 비어있음 - {material_name} Tier{tier}", "WARNING")
                # 기본값 반환 (0이 아닌 작은 값으로 설정)
                return 0.1  # 10%로 기본값 설정
            
            # 해당 자재와 매칭되는 proportion 항목 찾기
            material_lower = material_name.lower()
            
            if self.debug_mode:
                self._log_debug(f"    🔍 [ENHANCED MAPPING] Proportion 매칭 시도: '{material_name}' Tier{tier}")
                self._log_debug(f"      - ref_proportions_df 행 수: {len(self.ref_proportions_df)}")
                self._log_debug(f"      - 사용 중인 매칭 로직: Enhanced Material Matching (from rule_based.py)")
            
            matched_values = []  # 매칭된 모든 값 수집
            
            for idx, row in self.ref_proportions_df.iterrows():
                # 향상된 매칭 로직 사용 (rule_based.py에서 추출)
                material_category = self.material_types.get(material_name, {}).get('category', '')
                
                # enhanced_material_matching을 사용하여 매칭 확인
                match_found = self._enhanced_material_matching(
                    material_name, material_category, row
                )
                
                if match_found:
                    match_type = "enhanced_matching"
                    proportion_name = str(row.get('자재명(포함)', ''))
                    if self.debug_mode:
                        self._log_debug(f"      ✅ 매칭 성공 ({match_type}): '{proportion_name}' (행 {idx})")
                    
                    tier_col = f'Tier{tier}_RE100(%)'
                    if tier_col in row:
                        tier_value_raw = row[tier_col]
                        
                        if self.debug_mode:
                            self._log_debug(f"        - {tier_col} 원본값: {tier_value_raw}")
                        
                        # % 제거 후 숫자로 변환
                        tier_value = self._parse_tier_value(tier_value_raw)
                        if tier_value > 0:
                            matched_values.append(tier_value)
                            if self.debug_mode:
                                self._log_debug(f"        - 변환 결과: {tier_value:.3f}")
            
            # 매칭된 값들 처리
            if matched_values:
                # 여러 값이 있으면 평균값 사용
                result = sum(matched_values) / len(matched_values)
                if self.debug_mode:
                    if len(matched_values) > 1:
                        self._log_debug(f"      📊 다중 매칭 ({len(matched_values)}개): 평균값 {result:.3f} 사용")
                    self._log_debug(f"      ✅ 최종 proportion 값: {result:.3f}")
                return result
            else:
                # 매칭 실패 시 기본값 처리
                default_value = self._get_default_tier_value(tier)
                if self.debug_mode:
                    self._log_debug(f"      ❌ 매칭 실패: '{material_name}' Tier{tier}")
                    self._log_debug(f"      🔧 기본값 적용: {default_value:.3f}")
                return default_value
                
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ _get_proportion_tier_value 오류 ({material_name} Tier{tier}): {e}", "ERROR")
            # 오류 발생 시에도 기본값 반환 (0이 아닌 값)
            return self._get_default_tier_value(tier)
    
    def _parse_tier_value(self, tier_value_raw) -> float:
        """Tier 값을 안전하게 파싱"""
        try:
            if pd.isna(tier_value_raw):
                return 0.0
            
            if isinstance(tier_value_raw, (int, float)):
                return max(0.0, float(tier_value_raw) / 100)  # 비율로 변환
            else:
                tier_str = str(tier_value_raw).replace('%', '').strip()
                if tier_str:
                    return max(0.0, float(tier_str) / 100)
                else:
                    return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _get_default_tier_value(self, tier: int) -> float:
        """Tier별 기본값 반환 (매칭 실패 시 사용)"""
        # tier별 기본값 설정 (실제 시스템에서 사용되는 일반적인 값)
        tier_defaults = {
            1: 0.2,  # 20%
            2: 0.15, # 15%  
            3: 0.1,  # 10%
            4: 0.05  # 5%
        }
        return tier_defaults.get(tier, 0.1)  # 기본 10%
    
    def _check_keyword_match(self, material_name: str, proportion_name: str) -> bool:
        """키워드 기반 매칭 개선"""
        # 공통 키워드들
        keywords = ['nickel', 'ni', 'cobalt', 'co', 'lithium', 'li', 'manganese', 'mn', 
                   '니켈', '코발트', '리튬', '망간', 'sulfate', 'hydroxide', 'carbonate']
        
        material_keywords = [kw for kw in keywords if kw in material_name]
        proportion_keywords = [kw for kw in keywords if kw in proportion_name]
        
        # 공통 키워드가 있으면 매칭
        return len(set(material_keywords) & set(proportion_keywords)) > 0
    
    def _check_token_match_enhanced(self, material_name: str, proportion_name: str) -> bool:
        """
        향상된 토큰 기반 매칭 (rule_based.py에서 추출)
        proportion_name의 모든 토큰이 material_name에 포함되는지 확인
        """
        if not material_name or not proportion_name:
            return False
        
        material_tokens = set(material_name.lower().split())
        proportion_tokens = set(proportion_name.lower().split())
        
        # proportion_name의 모든 토큰이 material_name에 포함되는지 확인
        return len(proportion_tokens) > 0 and proportion_tokens.issubset(material_tokens)
    
    def _check_token_match_simple(self, material_name: str, proportion_name: str) -> bool:
        """간단한 토큰 기반 매칭 (호환성 유지)"""
        return self._check_token_match_enhanced(material_name, proportion_name)
    
    def _validate_mapping_improvements(self) -> Dict[str, Any]:
        """
        매핑 개선사항 검증 - 기존 로직 vs 새 로직 비교
        
        Returns:
            Dict: 개선사항 검증 결과
        """
        validation_results = {
            'total_materials': len(self.target_materials),
            'old_logic_matches': 0,
            'new_logic_matches': 0,
            'improvement_cases': [],
            'regression_cases': [],
            'detailed_comparison': {}
        }
        
        if self.debug_mode:
            self._log_debug("🔍 [VALIDATION] 매핑 로직 개선사항 검증 시작")
        
        for material_name in self.target_materials:
            material_category = self.material_types.get(material_name, {}).get('category', '')
            
            # 기존 로직 결과 (간단한 토큰 매칭)
            old_match_found = False
            new_match_found = False
            
            for idx, row in self.ref_proportions_df.iterrows():
                proportion_name = str(row.get('자재명(포함)', '')).lower()
                if pd.isna(row.get('자재명(포함)')) or not proportion_name:
                    continue
                
                # 기존 로직: 간단한 토큰 매칭
                material_tokens = set(material_name.lower().split())
                proportion_tokens = set(proportion_name.split())
                if len(proportion_tokens) > 0 and proportion_tokens.issubset(material_tokens):
                    old_match_found = True
                    break
            
            # 새 로직 결과 (enhanced matching)
            new_match_found = self._check_proportion_applicable(material_name)
            
            # 결과 집계
            if old_match_found:
                validation_results['old_logic_matches'] += 1
            if new_match_found:
                validation_results['new_logic_matches'] += 1
            
            # 개선/회귀 케이스 식별
            if new_match_found and not old_match_found:
                validation_results['improvement_cases'].append({
                    'material': material_name,
                    'category': material_category,
                    'reason': 'Enhanced matching found new match'
                })
            elif old_match_found and not new_match_found:
                validation_results['regression_cases'].append({
                    'material': material_name,
                    'category': material_category,
                    'reason': 'Enhanced matching lost previous match'
                })
            
            validation_results['detailed_comparison'][material_name] = {
                'old_match': old_match_found,
                'new_match': new_match_found,
                'status': 'improved' if (new_match_found and not old_match_found) else
                          'regressed' if (old_match_found and not new_match_found) else
                          'unchanged'
            }
        
        # 개선 통계 계산
        validation_results['improvement_count'] = len(validation_results['improvement_cases'])
        validation_results['regression_count'] = len(validation_results['regression_cases'])
        validation_results['improvement_rate'] = (validation_results['new_logic_matches'] / 
                                                  validation_results['total_materials'] * 100) if validation_results['total_materials'] > 0 else 0
        
        if self.debug_mode:
            self._log_debug(f"✅ [VALIDATION] 매핑 검증 완료:")
            self._log_debug(f"    - 총 자재: {validation_results['total_materials']}")
            self._log_debug(f"    - 기존 로직 매칭: {validation_results['old_logic_matches']}")
            self._log_debug(f"    - 새 로직 매칭: {validation_results['new_logic_matches']}")
            self._log_debug(f"    - 개선된 케이스: {validation_results['improvement_count']}")
            self._log_debug(f"    - 회귀된 케이스: {validation_results['regression_count']}")
            self._log_debug(f"    - 새 로직 매칭률: {validation_results['improvement_rate']:.1f}%")
        
        return validation_results
    
    def _check_material_category_match_enhanced(self, category1: str, category2: str) -> bool:
        """
        자재품목 일치 확인 (부분 일치 포함) - rule_based.py에서 추출
        """
        if not category1 or not category2:
            return False
        
        category1 = category1.lower()
        category2 = category2.lower()
        
        # 정확한 일치 확인
        if category1 == category2:
            return True
        
        # 부분 일치 확인 (예: "al foil" vs "foil")
        if category1 in category2 or category2 in category1:
            return True
        
        return False
    
    def _enhanced_material_matching(self, material_name: str, material_category: str, proportion_row: pd.Series) -> bool:
        """
        향상된 자재 매칭 로직 (rule_based.py 방식 적용)
        """
        import pandas as pd
        
        # NaN 값 처리
        if pd.isna(material_name):
            material_name = ''
        if pd.isna(material_category):
            material_category = ''
        
        material_name_lower = str(material_name).lower()
        material_category_lower = str(material_category).lower()
        
        proportion_name = str(proportion_row.get('자재명(포함)', '')).lower()
        proportion_category = str(proportion_row.get('자재품목', '')).lower()
        
        if self.debug_mode:
            self._log_debug(f"      🔍 [ENHANCED RULE_BASED MATCHING] 자재명: '{material_name}' vs 비례명: '{proportion_name}'")
            self._log_debug(f"        - 자재품목: '{material_category}' vs '{proportion_category}'")
        
        # 특별 케이스 처리 - 음극재 (artificial/natural 구분)
        if material_category_lower == '음극재':
            if 'artificial' in material_name_lower and 'artificial' in proportion_name:
                if self.debug_mode:
                    self._log_debug(f"        ✅ 음극재(인조) 특별 케이스 매칭")
                return True
            elif 'natural' in material_name_lower and 'natural' in proportion_name:
                if self.debug_mode:
                    self._log_debug(f"        ✅ 음극재(천연) 특별 케이스 매칭")
                return True
        
        # 특별 케이스 처리 - 양극재
        if material_category_lower == '양극재' and proportion_category == '양극재':
            if (material_name_lower in ['', 'nan', 'n/a'] or 'cathode' in material_name_lower):
                if self.debug_mode:
                    self._log_debug(f"        ✅ 양극재 특별 케이스 매칭")
                return True
        
        # 1단계: 정확한 포함 관계 확인
        if (proportion_name in material_name_lower or 
            material_name_lower in proportion_name):
            if self.debug_mode:
                self._log_debug(f"        ✅ 포함 관계 매칭 성공")
            return True
        
        # 2단계: 토큰 기반 매칭 (향상된 버전)
        if self._check_token_match_enhanced(material_name_lower, proportion_name):
            if self.debug_mode:
                self._log_debug(f"        ✅ 토큰 기반 매칭 성공")
            return True
        
        # 3단계: 자재품목 기반 매칭
        if self._check_material_category_match_enhanced(material_category_lower, proportion_category):
            if self.debug_mode:
                self._log_debug(f"        ✅ 자재품목 매칭 성공")
            return True
        
        return False
    
    def _set_re_rate_constraints(self, model: pyo.ConcreteModel) -> None:
        """RE 적용률 범위 제약 (동적 tier 지원)"""
        # 각 tier별로 동적으로 제약조건 생성
        for tier in range(1, self.num_tiers + 1):
            tier_key = f'tier{tier}'
            
            # 설정에서 min/max 값 가져오기
            if tier_key in self.config['re_rates']:
                tier_min = self.config['re_rates'][tier_key]['min']
                tier_max = self.config['re_rates'][tier_key]['max']
            else:
                # 기본값 사용
                tier_min = 0.1
                tier_max = 0.9
                if self.debug_mode:
                    print(f"⚠️ {tier_key} 설정이 없어 기본값 사용: min={tier_min}, max={tier_max}")
            
            # tier 변수 참조
            tier_var = getattr(model, f'{tier_key}_re')
            
            # 최소값 제약조건
            def tier_min_rule(model, m, tier_min=tier_min, tier_var=tier_var):
                return tier_var[m] >= tier_min
            
            # 최대값 제약조건
            def tier_max_rule(model, m, tier_max=tier_max, tier_var=tier_var):
                return tier_var[m] <= tier_max
            
            # 제약조건을 모델에 추가
            setattr(model, f'{tier_key}_min_constraint', 
                   pyo.Constraint(model.materials, rule=tier_min_rule))
            setattr(model, f'{tier_key}_max_constraint', 
                   pyo.Constraint(model.materials, rule=tier_max_rule))
            
            if self.debug_mode:
                print(f"🔧 {tier_key} 제약조건 생성: {tier_min:.1f} ≤ RE ≤ {tier_max:.1f}")
    
    def _set_material_ratio_constraints(self, model: pyo.ConcreteModel) -> None:
        """자재 비율 제약 (Ni/Co/Li 자재만 적용, 일반 자재는 bounds에서 고정)"""
        
        if self.debug_mode:
            ni_co_li_count = sum(1 for info in self.material_types.values() if info['is_ni_co_li'])
            general_count = len(model.materials) - ni_co_li_count
            print(f"🔧 자재 비율 제약조건 설정:")
            print(f"  - 대상: Ni/Co/Li 자재 {ni_co_li_count}개")
            print(f"  - 제외: 일반 자재 {general_count}개 (bounds에서 고정)")
            print(f"  - 재활용 비율: {self.config['material_ratios']['recycle']['min']} ~ {self.config['material_ratios']['recycle']['max']}")
            print(f"  - 저탄소메탈 비율: {self.config['material_ratios']['low_carbon']['min']} ~ {self.config['material_ratios']['low_carbon']['max']}")
        
        # 비율 합계 제약 (재활용 + 저탄소메탈 + 신재 = 1) - Ni/Co/Li 자재만
        def ratio_sum_rule(model, m):
            if model.is_ni_co_li[m] == 0:
                return pyo.Constraint.Skip  # 일반 자재는 bounds에서 이미 고정됨
            return model.recycle_ratio[m] + model.low_carbon_ratio[m] + model.virgin_ratio[m] == 1.0
        
        model.ratio_sum_constraint = pyo.Constraint(model.materials, rule=ratio_sum_rule)
        
        # 신재 비율 0 이상 제약 (Ni/Co/Li 자재만, 일반 자재는 bounds에서 1로 고정)
        def virgin_min_rule(model, m):
            if model.is_ni_co_li[m] == 0:
                return pyo.Constraint.Skip  # 일반 자재는 bounds에서 이미 1로 고정됨
            return model.virgin_ratio[m] >= 0
        
        model.virgin_min_constraint = pyo.Constraint(model.materials, rule=virgin_min_rule)
        
        if self.debug_mode:
            # 실제로 적용될 제약조건 개수 확인
            applicable_materials = [name for name, info in self.material_types.items() if info['is_ni_co_li']]
            if applicable_materials:
                print(f"  ✅ 제약조건 적용 자재: {', '.join(applicable_materials[:3])}")
                if len(applicable_materials) > 3:
                    print(f"    ... 외 {len(applicable_materials) - 3}개")
                print(f"  📊 생성된 제약조건: ratio_sum_constraint ({len(applicable_materials)}개), virgin_min_constraint ({len(applicable_materials)}개)")
            else:
                print(f"  ⚠️ 제약조건 적용 대상 자재 없음 (모든 자재가 일반 자재)")
        
        # 재활용 시나리오에서는 비율 합계 제약 추가
        if self.scenario in ['recycling', 'both']:
            def ratio_sum_rule(model, m):
                # 양극재만 비율 합계 제약 적용
                info = self.material_types.get(m, {})
                if info.get('is_cathode', False):
                    return model.virgin_ratio[m] + model.recycle_ratio[m] + model.low_carbon_ratio[m] == 1.0
                else:
                    return pyo.Constraint.Skip
            
            model.ratio_sum_constraint = pyo.Constraint(model.materials, rule=ratio_sum_rule)
            
            if self.debug_mode:
                print(f"    ✅ 비율 합계 제약조건 설정 완료 (재활용 시나리오)")
    
    def _set_pcf_reduction_target_constraints(self, model: pyo.ConcreteModel) -> None:
        """설정값 기반 PCF 감축 목표 제약 - 단일 자재 최적화 중심"""
        # UI에서 설정한 제약조건 값들 사용
        constraints_config = self.config.get('constraints', {})
        min_pcf_ratio = constraints_config.get('min_pcf_ratio', 0.05)  # 기본값 5%
        max_tier_reduction = constraints_config.get('max_tier_reduction', 0.95)  # 기본값 95%
        
        def emission_lower_bound_rule(model, m):
            # 배출계수가 원본의 설정된 최소 비율 이상 유지
            return model.modified_emission[m] >= model.original_emission[m] * min_pcf_ratio
        
        model.emission_lower_bound_constraint = pyo.Constraint(model.materials, rule=emission_lower_bound_rule)
        
        if self.debug_mode:
            print(f"      🎯 설정 기반 PCF 제약조건 설정:")
            print(f"        • 배출계수 하한: 원본의 {min_pcf_ratio*100:.1f}% 이상 유지")
            print(f"        • 최대 감축률: {(1-min_pcf_ratio)*100:.1f}%")
    
    def solve(self, solver_name: str = 'glpk') -> Dict[str, Any]:
        """
        새로운 시나리오별 최적화 문제 해결
        
        Args:
            solver_name: 사용할 솔버 이름 (기본값: 'glpk')
            
        Returns:
            Dict: 시나리오별 최적화 결과
        """
        # 🟢 함수 호출 로그
        self._print_debug(f"🟢 CALLED: solve(solver_name='{solver_name}')")
        self._print_debug(f"📍 Called from UI after build_optimization_model()")
        
        if self.model is None:
            self._print_debug("🔀 BRANCH: model is None - calling build_optimization_model()")
            self.build_optimization_model()
        else:
            self._print_debug("🔀 BRANCH: model exists - proceeding with solve")
        
        try:
            if self.debug_mode:
                print(f"📊 시나리오별 최적화 결과 처리:")
                print(f"  - 시나리오: {self.scenario}")
                print(f"  - 솔버: {solver_name}")
                print(f"  - 모델 상태: {'존재함' if self.model else '없음'}")
            
            # 시나리오별 결과가 이미 model.results에 저장되어 있음
            if hasattr(self.model, 'results') and self.model.results:
                scenario_results = self.model.results
                
                # 통계 계산
                total_materials = len(scenario_results['materials'])
                successful_materials = sum(1 for r in scenario_results['materials'].values() 
                                        if r.get('status') == 'optimal')
                
                # 전체 PCF 계산
                total_original_pcf = 0.0
                total_optimized_pcf = 0.0
                
                for material_name, result in scenario_results['materials'].items():
                    if result.get('status') == 'optimal':
                        original_pcf = result.get('original_pcf', 0.0)
                        optimized_pcf = result.get('optimized_pcf', 0.0)
                        
                        total_original_pcf += original_pcf
                        total_optimized_pcf += optimized_pcf
                
                # 전체 감축률 계산
                if total_original_pcf > 0:
                    total_reduction_percentage = ((total_original_pcf - total_optimized_pcf) / total_original_pcf) * 100
                else:
                    total_reduction_percentage = 0.0
                
                # 실패한 자재들의 오류 메시지 수집
                failed_materials = []
                error_summary = []
                
                for material_name, result in scenario_results['materials'].items():
                    if result.get('status') != 'optimal':
                        failed_materials.append(material_name)
                        error_msg = result.get('message', '알 수 없는 오류')
                        error_summary.append(f"{material_name}: {error_msg}")
                
                # 상태 및 메시지 결정
                if successful_materials > 0:
                    status = 'optimal'
                    if failed_materials:
                        message = f"일부 자재 최적화 성공 ({successful_materials}/{total_materials}). 실패: {', '.join(failed_materials)}"
                    else:
                        message = f"모든 자재 최적화 성공 ({successful_materials}/{total_materials})"
                else:
                    status = 'error'
                    if error_summary:
                        message = f"모든 자재 최적화 실패. 주요 오류: {'; '.join(error_summary[:3])}{'...' if len(error_summary) > 3 else ''}"
                    else:
                        message = f"모든 자재 최적화 실패 ({total_materials}개 자재)"
                
                # 최종 결과 구성
                self.results = {
                    'status': status,
                    'message': message,
                    'scenario': self.scenario,
                    'solver': solver_name,
                    'total_materials': total_materials,
                    'successful_materials': successful_materials,
                    'failed_materials': len(failed_materials),
                    'failed_material_names': failed_materials,
                    'success_rate': (successful_materials / total_materials * 100) if total_materials > 0 else 0,
                    'original_pcf': total_original_pcf,
                    'optimized_pcf': total_optimized_pcf,
                    'reduction_amount': total_original_pcf - total_optimized_pcf,
                    'reduction_percentage': total_reduction_percentage,
                    'materials': scenario_results['materials'],
                    'error_summary': error_summary,
                    'debug_logs': self.debug_logs
                }
                
                # 시나리오별 추가 정보
                if self.scenario == 'site_change' or self.scenario == 'both':
                    # 선택된 국가별 통계
                    country_stats = {}
                    for material_name, result in scenario_results['materials'].items():
                        if result.get('status') == 'optimal' and 'best_country' in result:
                            country = result['best_country']
                            if country not in country_stats:
                                country_stats[country] = {'count': 0, 'materials': []}
                            country_stats[country]['count'] += 1
                            country_stats[country]['materials'].append(material_name)
                    
                    self.results['country_selection'] = country_stats
                
                if self.scenario == 'recycling' or self.scenario == 'both':
                    # 양극재 재활용 통계
                    cathode_stats = {
                        'total_cathode': 0,
                        'optimized_cathode': 0,
                        'avg_recycle_ratio': 0.0,
                        'avg_low_carbon_ratio': 0.0
                    }
                    
                    total_recycle = 0.0
                    total_low_carbon = 0.0
                    cathode_count = 0
                    
                    for material_name, result in scenario_results['materials'].items():
                        material_info = self.material_types.get(material_name, {})
                        if material_info.get('is_cathode', False):
                            cathode_stats['total_cathode'] += 1
                            if result.get('status') == 'optimal':
                                cathode_stats['optimized_cathode'] += 1
                                if 'optimal_recycle_ratio' in result:
                                    total_recycle += result['optimal_recycle_ratio']
                                    cathode_count += 1
                                if 'optimal_low_carbon_ratio' in result:
                                    total_low_carbon += result['optimal_low_carbon_ratio']
                    
                    if cathode_count > 0:
                        cathode_stats['avg_recycle_ratio'] = total_recycle / cathode_count
                        cathode_stats['avg_low_carbon_ratio'] = total_low_carbon / cathode_count
                    
                    self.results['cathode_recycling'] = cathode_stats
                
                if self.debug_mode:
                    print(f"  ✅ 시나리오별 결과 처리 완료:")
                    print(f"    - 성공률: {successful_materials}/{total_materials} ({self.results['success_rate']:.1f}%)")
                    print(f"    - 총 감축률: {total_reduction_percentage:.2f}%")
                    
                    if 'country_selection' in self.results:
                        print(f"    - 국가별 선택: {list(self.results['country_selection'].keys())}")
                    
                    if 'cathode_recycling' in self.results:
                        cathode_info = self.results['cathode_recycling']
                        print(f"    - 양극재 재활용: {cathode_info['optimized_cathode']}/{cathode_info['total_cathode']}개")
                
                return self.results
            
            else:
                # 시나리오별 결과가 없는 경우
                return {
                    'status': 'error',
                    'message': '시나리오별 최적화 결과를 찾을 수 없습니다.',
                    'scenario': self.scenario,
                    'debug_logs': self.debug_logs
                }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            self._log_debug(f"❌ 시나리오별 최적화 중 예외 발생: {str(e)}", "ERROR")
            self._log_debug(f"스택트레이스:\n{traceback.format_exc()}", "ERROR")
            
            return {
                'status': 'error',
                'message': f"시나리오별 최적화 중 오류 발생: {str(e)}",
                'scenario': self.scenario,
                'debug_logs': self.debug_logs
            }
    
    def _diagnose_infeasibility(self) -> Dict[str, Any]:
        """실행 불가능성 원인 상세 진단"""
        diagnostic = {
            'materials_count': len(self.target_materials),
            'target_materials': list(self.material_types.keys()),
            'config_summary': {},
            'potential_issues': [],
            'feasibility_analysis': {},
            'constraint_analysis': {},
            'recommended_fixes': []
        }
        
        try:
            if self.debug_mode:
                print("\n===== 📊 상세 실행불가능성 진단 =====")
            
            # 설정 요약
            diagnostic['config_summary'] = {
                'reduction_target': self.config.get('reduction_target', {}),
                're_rates': {k: v for k, v in self.config.get('re_rates', {}).items()},
                'material_ratios': self.config.get('material_ratios', {})
            }
            
            # 기본 파라미터 추출
            min_reduction = self.config['reduction_target'].get('min', 0) / 100
            max_reduction = self.config['reduction_target'].get('max', 0) / 100
            
            # 자재별 기준 배출량 계산
            material_emissions = {}
            total_original_emission = 0
            for name, info in self.material_types.items():
                emission = info['original_emission'] * info['quantity']
                material_emissions[name] = emission
                total_original_emission += emission
            
            diagnostic['material_emissions'] = material_emissions
            diagnostic['total_original_emission'] = total_original_emission
            
            if self.debug_mode:
                print(f"🔢 기준 배출량 정보:")
                print(f"  • 총 기준 배출량: {total_original_emission:.4f} kgCO2eq")
                print(f"  • 감축 목표: {min_reduction*100:.1f}% ~ {max_reduction*100:.1f}%")
            
            # === 1. 이론적 최대 감축 가능량 계산 ===
            max_possible_reduction = self._calculate_theoretical_max_reduction()
            target_pcf_max = total_original_emission * (1 - min_reduction)
            target_pcf_min = total_original_emission * (1 - max_reduction)
            required_reduction_for_target = max_reduction
            
            diagnostic['feasibility_analysis'] = {
                'max_possible_reduction_pct': max_possible_reduction * 100,
                'required_reduction_pct': required_reduction_for_target * 100,
                'target_pcf_range': [target_pcf_min, target_pcf_max],
                'is_theoretically_feasible': max_possible_reduction >= required_reduction_for_target,
                'reduction_gap_pct': (required_reduction_for_target - max_possible_reduction) * 100
            }
            
            if self.debug_mode:
                print(f"🎯 실행가능성 분석:")
                print(f"  • 이론적 최대 감축: {max_possible_reduction*100:.2f}%")
                print(f"  • 목표 감축 범위: {min_reduction*100:.1f}% ~ {max_reduction*100:.1f}%")
                print(f"  • 목표 PCF 범위: {target_pcf_min:.4f} ~ {target_pcf_max:.4f} kgCO2eq")
                
                if max_possible_reduction < required_reduction_for_target:
                    gap = (required_reduction_for_target - max_possible_reduction) * 100
                    print(f"  ❌ 이론적 불가능: {gap:.2f}% 부족")
                    diagnostic['potential_issues'].append(f"이론적 최대 감축량({max_possible_reduction*100:.1f}%)이 목표({required_reduction_for_target*100:.1f}%)에 미달")
                else:
                    print(f"  ✅ 이론적 가능: {(max_possible_reduction - required_reduction_for_target)*100:.2f}% 여유")
            
            # === 2. 제약조건별 상세 분석 ===
            constraint_analysis = self._analyze_constraints_detailed()
            diagnostic['constraint_analysis'] = constraint_analysis
            
            if self.debug_mode:
                print(f"\n⚖️ 제약조건 상세 분석:")
                for constraint_name, analysis in constraint_analysis.items():
                    print(f"  📋 {constraint_name}:")
                    if 'violation' in analysis:
                        print(f"    • 위반 여부: {analysis['violation']}")
                        if analysis['violation']:
                            print(f"    • 위반 정도: {analysis.get('violation_amount', 'N/A')}")
                    if 'recommendation' in analysis:
                        print(f"    • 권장사항: {analysis['recommendation']}")
            
            # === 3. 자재 비율 제약 검증 ===
            ratio_analysis = self._analyze_material_ratios()
            diagnostic['material_ratio_analysis'] = ratio_analysis
            
            if self.debug_mode and ratio_analysis.get('issues'):
                print(f"\n🧩 자재 비율 제약 문제:")
                for issue in ratio_analysis['issues']:
                    print(f"  • {issue}")
            
            # === 4. 구체적 해결방안 제안 ===
            recommendations = self._generate_fix_recommendations(
                max_possible_reduction, required_reduction_for_target,
                constraint_analysis, ratio_analysis
            )
            diagnostic['recommended_fixes'] = recommendations
            
            if self.debug_mode and recommendations:
                print(f"\n🔧 권장 해결방안:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            
            # 기존 간단한 검사들
            if min_reduction < 0.01 and max_reduction < 0.01:
                diagnostic['potential_issues'].append("감축 목표가 너무 작음 (1% 미만)")
            
            if max_reduction > 0.5:
                diagnostic['potential_issues'].append("최대 감축률이 너무 큼 (50% 이상)")
            
            if total_original_emission <= 0:
                diagnostic['potential_issues'].append("총 기준 배출량이 0 이하")
            
            # 재활용/저탄소 메탈 비율 제약 확인
            recycle_config = self.config.get('material_ratios', {}).get('recycle', {})
            low_carbon_config = self.config.get('material_ratios', {}).get('low_carbon', {})
            
            recycle_min = recycle_config.get('min', 0)
            recycle_max = recycle_config.get('max', 1)
            low_carbon_min = low_carbon_config.get('min', 0)
            low_carbon_max = low_carbon_config.get('max', 1)
            
            if recycle_min + low_carbon_min > 1.0:
                diagnostic['potential_issues'].append(f"재활용 최소비율({recycle_min}) + 저탄소메탈 최소비율({low_carbon_min}) > 1.0")
            
            if recycle_max + low_carbon_max < recycle_min + low_carbon_min:
                diagnostic['potential_issues'].append("비율 제약 범위 충돌")
            
            if self.debug_mode:
                print(f"\n📋 진단 완료 - 총 {len(diagnostic['potential_issues'])}개 문제 발견")
                print("="*50)
                
        except Exception as e:
            diagnostic['diagnostic_error'] = str(e)
            if self.debug_mode:
                print(f"❌ 진단 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        
        return diagnostic
    
    def _calculate_theoretical_max_reduction(self) -> float:
        """이론적 최대 감축 가능량 계산 (UI 파라미터 직접 사용)"""
        try:
            # Tier별 최대 RE 적용률을 직접 감축률로 사용 (하드코딩 제거)
            max_reduction = 0
            
            for tier in range(1, self.num_tiers + 1):
                tier_key = f'tier{tier}'
                if tier_key in self.config['re_rates']:
                    max_re = self.config['re_rates'][tier_key]['max']
                else:
                    max_re = 0.9  # 기본 최대값
                
                # UI RE 비율을 직접 감축률로 사용
                max_reduction += max_re
            
            # 재활용 및 저탄소메탈의 추가 감축 효과 고려 (시나리오에 따라)
            if self.scenario in ['recycling', 'both']:
                # 재활용재: 60% 감축 효과 (1 - 0.4 = 0.6)
                # 저탄소메탈: 30% 감축 효과 (1 - 0.7 = 0.3)
                recycle_max = self.config.get('material_ratios', {}).get('recycle', {}).get('max', 0.5)
                low_carbon_max = self.config.get('material_ratios', {}).get('low_carbon', {}).get('max', 0.3)
                
                additional_reduction = (recycle_max * 0.6 + low_carbon_max * 0.3) * 0.5  # 보수적 추정
                max_reduction = min(max_reduction + additional_reduction, 0.95)  # 최대 95% 제한
            
            return min(max_reduction, 0.95)  # 안전 상한선
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ 이론적 최대 감축량 계산 오류: {e}")
            return 0.5  # 기본값
    
    def _analyze_constraints_detailed(self) -> Dict[str, Dict[str, Any]]:
        """제약조건별 상세 분석"""
        analysis = {}
        
        try:
            # 1. PCF 감축 목표 제약 분석
            min_reduction = self.config['reduction_target'].get('min', 0) / 100
            max_reduction = self.config['reduction_target'].get('max', 0) / 100
            max_possible = self._calculate_theoretical_max_reduction()
            
            analysis['pcf_reduction_target'] = {
                'type': 'PCF 감축 목표',
                'min_required': min_reduction,
                'max_required': max_reduction,
                'max_possible': max_possible,
                'violation': max_reduction > max_possible,
                'violation_amount': max(0, max_reduction - max_possible),
                'recommendation': f"목표 감축률을 {max_possible*100:.1f}% 이하로 조정" if max_reduction > max_possible else "적절함"
            }
            
            # 2. RE 적용률 범위 제약 분석
            for tier in range(1, self.num_tiers + 1):
                tier_key = f'tier{tier}'
                if tier_key in self.config['re_rates']:
                    tier_min = self.config['re_rates'][tier_key]['min']
                    tier_max = self.config['re_rates'][tier_key]['max']
                    
                    analysis[f're_rate_{tier_key}'] = {
                        'type': f'{tier_key.upper()} RE 적용률',
                        'range': [tier_min, tier_max],
                        'violation': tier_min > tier_max or tier_min < 0 or tier_max > 1,
                        'recommendation': f"범위를 0.0-1.0 내에서 min ≤ max 로 설정" if tier_min > tier_max else "적절함"
                    }
            
            # 3. 자재 비율 제약 분석
            recycle_config = self.config.get('material_ratios', {}).get('recycle', {})
            low_carbon_config = self.config.get('material_ratios', {}).get('low_carbon', {})
            
            recycle_min = recycle_config.get('min', 0)
            recycle_max = recycle_config.get('max', 1)
            low_carbon_min = low_carbon_config.get('min', 0)
            low_carbon_max = low_carbon_config.get('max', 1)
            
            total_min = recycle_min + low_carbon_min
            total_max = recycle_max + low_carbon_max
            
            analysis['material_ratios'] = {
                'type': '자재 비율 제약',
                'recycle_range': [recycle_min, recycle_max],
                'low_carbon_range': [low_carbon_min, low_carbon_max],
                'total_min': total_min,
                'violation': total_min > 1.0,
                'violation_amount': max(0, total_min - 1.0),
                'recommendation': f"최소 비율 합계를 {1.0 - 0.1:.1f} 이하로 조정" if total_min > 1.0 else "적절함"
            }
            
        except Exception as e:
            analysis['error'] = {'message': str(e)}
        
        return analysis
    
    def _analyze_material_ratios(self) -> Dict[str, Any]:
        """자재 비율 제약 상세 분석"""
        analysis = {'issues': [], 'recommendations': []}
        
        try:
            recycle_config = self.config.get('material_ratios', {}).get('recycle', {})
            low_carbon_config = self.config.get('material_ratios', {}).get('low_carbon', {})
            
            recycle_min = recycle_config.get('min', 0)
            recycle_max = recycle_config.get('max', 1)
            low_carbon_min = low_carbon_config.get('min', 0)
            low_carbon_max = low_carbon_config.get('max', 1)
            
            # 비율 합계 검증
            if recycle_min + low_carbon_min > 1.0:
                excess = recycle_min + low_carbon_min - 1.0
                analysis['issues'].append(f"최소 비율 합계 초과: {excess:.3f}")
                analysis['recommendations'].append(f"재활용 또는 저탄소메탈 최소비율을 {excess:.3f} 이상 감소")
            
            if recycle_min + low_carbon_min > 0.9:
                analysis['issues'].append("최소 비율 합계가 너무 높음 (90% 초과) - 신재 여유도 부족")
                analysis['recommendations'].append("신재 비율을 위해 10% 이상 여유 두기")
            
            # 개별 비율 검증
            if recycle_min > recycle_max:
                analysis['issues'].append(f"재활용 비율 범위 오류: min({recycle_min}) > max({recycle_max})")
            
            if low_carbon_min > low_carbon_max:
                analysis['issues'].append(f"저탄소메탈 비율 범위 오류: min({low_carbon_min}) > max({low_carbon_max})")
            
            # 현실성 검증
            if recycle_max > 0.8:
                analysis['issues'].append("재활용 비율이 너무 높음 (80% 초과) - 현실적으로 어려울 수 있음")
            
            if low_carbon_max > 0.5:
                analysis['issues'].append("저탄소메탈 비율이 너무 높음 (50% 초과) - 비용 및 공급 제약")
                
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _generate_fix_recommendations(self, max_possible_reduction: float, required_reduction: float, 
                                   constraint_analysis: Dict, ratio_analysis: Dict) -> List[str]:
        """구체적 해결방안 생성"""
        recommendations = []
        
        try:
            # 1. 감축 목표 조정
            if required_reduction > max_possible_reduction:
                safe_target = max_possible_reduction * 0.9  # 10% 마진
                recommendations.append(f"감축 목표를 {safe_target*100:.1f}% 이하로 낮추기")
            
            # 2. RE 적용률 범위 조정
            for tier in range(1, self.num_tiers + 1):
                tier_key = f'tier{tier}'
                if tier_key in self.config['re_rates']:
                    tier_min = self.config['re_rates'][tier_key]['min']
                    tier_max = self.config['re_rates'][tier_key]['max']
                    
                    if tier_min > tier_max:
                        recommendations.append(f"{tier_key.upper()} RE 적용률: min값({tier_min:.2f})을 max값({tier_max:.2f}) 이하로 조정")
                    
                    if tier_max > 0.9:
                        recommendations.append(f"{tier_key.upper()} RE 적용률: max값을 0.9 이하로 조정 (현재: {tier_max:.2f})")
            
            # 3. 자재 비율 제약 조정
            if 'material_ratios' in constraint_analysis:
                mat_analysis = constraint_analysis['material_ratios']
                if mat_analysis.get('violation', False):
                    violation_amount = mat_analysis.get('violation_amount', 0)
                    recommendations.append(f"재활용+저탄소메탈 최소 비율을 {violation_amount:.3f} 이상 감소시키기")
            
            # 4. 비율별 구체적 조정안
            if ratio_analysis.get('issues'):
                recycle_config = self.config.get('material_ratios', {}).get('recycle', {})
                low_carbon_config = self.config.get('material_ratios', {}).get('low_carbon', {})
                
                recycle_min = recycle_config.get('min', 0)
                low_carbon_min = low_carbon_config.get('min', 0)
                
                if recycle_min + low_carbon_min > 1.0:
                    # 비율을 안전한 수준으로 조정
                    safe_recycle = min(recycle_min, 0.4)
                    safe_low_carbon = min(low_carbon_min, 0.3)
                    recommendations.append(f"안전 비율 설정: 재활용 ≤ {safe_recycle:.2f}, 저탄소메탈 ≤ {safe_low_carbon:.2f}")
            
            # 5. 시나리오별 추천
            if self.scenario == 'baseline':
                recommendations.append("저감 효과가 제한적인 baseline 시나리오 - 'recycling' 또는 'both' 시나리오 고려")
            
            # 6. 자재별 개별 처리 권장 (복잡한 경우)
            if len(self.target_materials) > 1:
                recommendations.append("자재별 개별 최적화 후 결과 통합 방식 고려")
                
        except Exception as e:
            recommendations.append(f"권장사항 생성 중 오류: {e}")
        
        return recommendations
    
    def _check_feasibility_before_modeling(self) -> Dict[str, Any]:
        """모델링 전 실행가능성 사전 검증"""
        result = {
            'is_feasible': True,
            'issues': [],
            'recommended_fixes': [],
            'auto_adjustable': True,
            'severity': 'none'  # none, warning, error, critical
        }
        
        try:
            if self.debug_mode:
                print(f"\n🔍 사전 실행가능성 검증 시작...")
            
            # 1. 기본 데이터 유효성 검증
            if self.original_pcf <= 0:
                result['issues'].append("기준 PCF가 0 이하")
                result['recommended_fixes'].append("시뮬레이션 데이터 재생성 필요")
                result['severity'] = 'critical'
                result['auto_adjustable'] = False
            
            if len(self.material_types) == 0:
                result['issues'].append("최적화 대상 자재가 없음")
                result['recommended_fixes'].append("저감활동 적용 자재를 1개 이상 설정")
                result['severity'] = 'critical'
                result['auto_adjustable'] = False
            
            # 2. 이론적 최대 감축량 vs 목표 감축량 검증
            max_possible_reduction = self._calculate_theoretical_max_reduction()
            min_reduction = self.config['reduction_target'].get('min', 0) / 100
            max_reduction = self.config['reduction_target'].get('max', 0) / 100
            
            if self.debug_mode:
                print(f"  • 이론적 최대 감축: {max_possible_reduction*100:.2f}%")
                print(f"  • 목표 감축 범위: {min_reduction*100:.1f}% ~ {max_reduction*100:.1f}%")
            
            if max_reduction > max_possible_reduction:
                gap = max_reduction - max_possible_reduction
                result['issues'].append(f"목표 감축률({max_reduction*100:.1f}%)이 이론적 최대치({max_possible_reduction*100:.1f}%)를 초과")
                result['recommended_fixes'].append(f"목표 감축률을 {max_possible_reduction*0.9*100:.1f}% 이하로 조정")
                result['severity'] = 'error'
                
                # 10% 이상 차이나면 자동 조정 불가
                if gap > 0.1:
                    result['auto_adjustable'] = False
            
            # 3. 자재 비율 제약 검증
            recycle_config = self.config.get('material_ratios', {}).get('recycle', {})
            low_carbon_config = self.config.get('material_ratios', {}).get('low_carbon', {})
            
            recycle_min = recycle_config.get('min', 0)
            low_carbon_min = low_carbon_config.get('min', 0)
            total_min = recycle_min + low_carbon_min
            
            if total_min > 1.0:
                result['issues'].append(f"자재 비율 최소값 합계가 1.0 초과 ({total_min:.3f})")
                result['recommended_fixes'].append("재활용/저탄소메탈 최소비율을 각각 0.02로 조정")
                result['severity'] = 'error'
            elif total_min > 0.95:
                result['issues'].append(f"자재 비율 최소값 합계가 너무 높음 ({total_min:.3f}) - 신재 여유도 부족")
                result['recommended_fixes'].append("자재 비율 최소값 합계를 0.9 이하로 유지")
                result['severity'] = 'warning'
            
            # 4. RE 적용률 범위 검증
            for tier in range(1, self.num_tiers + 1):
                tier_key = f'tier{tier}'
                if tier_key in self.config['re_rates']:
                    tier_min = self.config['re_rates'][tier_key]['min']
                    tier_max = self.config['re_rates'][tier_key]['max']
                    
                    if tier_min > tier_max:
                        result['issues'].append(f"{tier_key.upper()} RE 범위 오류: min({tier_min}) > max({tier_max})")
                        result['recommended_fixes'].append(f"{tier_key.upper()} 범위를 올바르게 설정 (min ≤ max)")
                        result['severity'] = 'error'
            
            # 5. 실행가능성 여유도 검증
            if max_possible_reduction > 0 and max_reduction > 0:
                feasibility_margin = (max_possible_reduction - max_reduction) / max_reduction
                if feasibility_margin < 0.1:  # 10% 미만 여유도
                    result['issues'].append(f"실행가능성 여유도 부족 ({feasibility_margin*100:.1f}%) - 수렴 어려울 수 있음")
                    result['recommended_fixes'].append("목표 감축률을 5-10% 낮추거나 RE 범위 확대")
                    if result['severity'] == 'none':
                        result['severity'] = 'warning'
            
            # 6. 최종 실행가능성 판정
            if result['severity'] in ['error', 'critical']:
                result['is_feasible'] = False
            
            if self.debug_mode:
                if result['is_feasible']:
                    print(f"  ✅ 사전 검증 통과 (심각도: {result['severity']})")
                else:
                    print(f"  ❌ 사전 검증 실패 (심각도: {result['severity']})")
                    print(f"  📋 발견된 문제: {len(result['issues'])}개")
            
        except Exception as e:
            result['issues'].append(f"사전 검증 중 오류: {str(e)}")
            result['severity'] = 'critical'
            result['is_feasible'] = False
            result['auto_adjustable'] = False
            if self.debug_mode:
                print(f"❌ 사전 검증 중 예외 발생: {e}")
        
        return result
    
    def _auto_adjust_parameters(self, feasibility_check: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """실행가능성 문제에 대한 자동 파라미터 조정"""
        if not feasibility_check.get('auto_adjustable', False):
            return None
        
        try:
            new_config = copy.deepcopy(self.config)
            changes = []
            
            # 1. 감축 목표 조정
            max_possible_reduction = self._calculate_theoretical_max_reduction()
            max_reduction = self.config['reduction_target'].get('max', 0) / 100
            
            if max_reduction > max_possible_reduction:
                # 안전 마진 10%를 두고 조정
                safe_target = max_possible_reduction * 0.9
                new_config['reduction_target']['max'] = int(safe_target * 100)
                
                # min도 max를 초과하지 않도록 조정
                if new_config['reduction_target']['min'] > new_config['reduction_target']['max']:
                    new_config['reduction_target']['min'] = max(1, new_config['reduction_target']['max'] - 2)
                
                changes.append(f"감축 목표를 {safe_target*100:.1f}%로 조정")
            
            # 2. 자재 비율 조정
            recycle_min = self.config.get('material_ratios', {}).get('recycle', {}).get('min', 0)
            low_carbon_min = self.config.get('material_ratios', {}).get('low_carbon', {}).get('min', 0)
            
            if recycle_min + low_carbon_min > 1.0:
                # 비례적으로 감소
                scale_factor = 0.8 / (recycle_min + low_carbon_min)
                new_recycle_min = recycle_min * scale_factor
                new_low_carbon_min = low_carbon_min * scale_factor
                
                new_config['material_ratios']['recycle']['min'] = round(new_recycle_min, 3)
                new_config['material_ratios']['low_carbon']['min'] = round(new_low_carbon_min, 3)
                
                changes.append(f"재활용 최소비율: {recycle_min:.3f} → {new_recycle_min:.3f}")
                changes.append(f"저탄소메탈 최소비율: {low_carbon_min:.3f} → {new_low_carbon_min:.3f}")
            
            # 3. RE 적용률 범위 조정
            for tier in range(1, self.num_tiers + 1):
                tier_key = f'tier{tier}'
                if tier_key in self.config['re_rates']:
                    tier_min = self.config['re_rates'][tier_key]['min']
                    tier_max = self.config['re_rates'][tier_key]['max']
                    
                    if tier_min > tier_max:
                        # min과 max를 바꿔서 수정
                        new_config['re_rates'][tier_key]['min'] = tier_max
                        new_config['re_rates'][tier_key]['max'] = tier_min
                        changes.append(f"{tier_key.upper()} RE 범위 수정: [{tier_min}, {tier_max}] → [{tier_max}, {tier_min}]")
            
            if changes:
                return {
                    'new_config': new_config,
                    'changes': changes
                }
            else:
                return None
                
        except Exception as e:
            if self.debug_mode:
                print(f"❌ 자동 파라미터 조정 중 오류: {e}")
            return None
    
    def _extract_optimal_values(self) -> Dict[str, Any]:
        """최적해 변수값 추출"""
        if not self.model:
            return {}
            
        result = {}
        
        # 자재별 최적값 추출
        for m in self.model.materials:
            material_name = m
            
            # Tier별 RE 적용률 (동적 tier 지원)
            for tier in range(1, self.num_tiers + 1):
                tier_var = getattr(self.model, f'tier{tier}_re')
                result[f'tier{tier}_re_{material_name}'] = pyo.value(tier_var[m])
            
            # Ni, Co, Li 자재인 경우 추가 변수
            if self.model.is_ni_co_li[m] == 1:
                result[f'recycle_ratio_{material_name}'] = pyo.value(self.model.recycle_ratio[m])
                result[f'low_carbon_ratio_{material_name}'] = pyo.value(self.model.low_carbon_ratio[m])
                result[f'virgin_ratio_{material_name}'] = pyo.value(self.model.virgin_ratio[m])
            
            # 최종 배출계수
            result[f'modified_emission_{material_name}'] = pyo.value(self.model.modified_emission[m])
            
            # 기존 배출계수
            result[f'original_emission_{material_name}'] = pyo.value(self.model.original_emission[m])
            
            # 자재 소요량
            result[f'quantity_{material_name}'] = pyo.value(self.model.quantity[m])
            
            # 자재별 탄소배출량
            result[f'carbon_emission_{material_name}'] = pyo.value(self.model.modified_emission[m] * self.model.quantity[m])
        
        return result
    
    def _set_premium_cost_constraints(self, model: pyo.ConcreteModel) -> None:
        """프리미엄 비용 제약 설정 (설정된 경우)"""
        # 프리미엄 비용 제약 설정이 있는지 확인
        if 'premium_cost' not in self.config or not self.config['premium_cost'].get('enabled', False):
            return
        
        # 프리미엄 비용 계산기 초기화
        cost_calculator = None
        baseline_premium_cost = 0.0
        
        # 자동 계산 설정이 있고 시뮬레이션 데이터가 있으면 계산
        if self.config['premium_cost'].get('auto_calculate', True) and self.simulation_data:
            try:
                cost_calculator = MaterialPremiumCostCalculator(
                    simulation_data=self.simulation_data,
                    stable_var_dir=self.stable_var_dir,
                    user_id=self.user_id,
                    debug_mode=self.debug_mode
                )
                # 기준 프리미엄 비용 계산
                baseline_costs = cost_calculator.calculate_baseline_premium_costs()
                baseline_premium_cost = baseline_costs.get('total', 0.0)
                if self.debug_mode:
                    print(f"자동 계산된 기준 프리미엄 비용: ${baseline_premium_cost:.2f}")
            except Exception as e:
                if self.debug_mode:
                    print(f"프리미엄 비용 자동 계산 오류: {e}")
                # 오류 발생시 사용자 설정값 사용
                baseline_premium_cost = self.config['premium_cost'].get('baseline_cost', 0.0)
        else:
            # 사용자가 설정한 기준 비용 사용
            baseline_premium_cost = self.config['premium_cost'].get('baseline_cost', 0.0)
        
        # ReductionConstraintManager를 통해 제약 설정
        reduction_manager = ReductionConstraintManager(
            original_pcf=self.original_pcf,
            config=self.config,
            stable_var_dir=self.stable_var_dir,
            user_id=self.user_id,
            debug_mode=self.debug_mode
        )
        
        # 프리미엄 비용 제약 설정
        reduction_target = self.config['premium_cost'].get('reduction_target', 0.0)
        reduction_manager.set_premium_cost_constraint(
            enabled=True,
            baseline_cost=baseline_premium_cost,
            reduction_target=reduction_target
        )
        
        # 모델에 프리미엄 비용 제약조건 추가
        reduction_manager.add_premium_cost_constraints(model)
        
        # 프리미엄 비용 계산기 저장 (결과 검증에 사용)
        self.premium_cost_calculator = cost_calculator
    
    def _check_constraints(self) -> Dict[str, Any]:
        """제약조건 만족 여부 확인"""
        if not self.model:
            return {}
            
        constraints = {}
        
        # 목표 PCF 범위 (양수 감축률을 음수로 변환)
        min_reduction = self.config['reduction_target'].get('min', 0) / 100
        max_reduction = self.config['reduction_target'].get('max', 0) / 100
        target_pcf_max = self.original_pcf * (1 - min_reduction)  # 양수를 음수로 변환
        target_pcf_min = self.original_pcf * (1 - max_reduction)  # 양수를 음수로 변환
        
        optimized_pcf = pyo.value(self.model.objective)
        constraints['target_pcf_max'] = target_pcf_max
        constraints['target_pcf_min'] = target_pcf_min
        constraints['optimized_pcf'] = optimized_pcf
        
        # 감축률 계산
        reduction_percentage = ((self.original_pcf - optimized_pcf) / self.original_pcf) * 100
        constraints['reduction_percentage'] = reduction_percentage
        
        # 프리미엄 비용 제약 검증
        if 'premium_cost' in self.config and self.config['premium_cost'].get('enabled', False):
            # 프리미엄 비용 계산기가 있으면 검증
            if hasattr(self, 'premium_cost_calculator') and self.premium_cost_calculator:
                try:
                    # 최적화 결과를 이용하여 프리미엄 비용 계산
                    optimized_costs = self.premium_cost_calculator.calculate_optimized_premium_costs(self.results)
                    optimized_premium_cost = optimized_costs.get('total', 0.0)
                    
                    # 기준 비용 (자동 계산 또는 사용자 설정)
                    baseline_premium_cost = self.config['premium_cost'].get('baseline_cost', 0.0)
                    if self.config['premium_cost'].get('auto_calculate', True):
                        baseline_costs = self.premium_cost_calculator.calculate_baseline_premium_costs()
                        baseline_premium_cost = baseline_costs.get('total', 0.0)
                    
                    # 감축률 계산
                    if baseline_premium_cost > 0:
                        premium_cost_reduction = ((baseline_premium_cost - optimized_premium_cost) / baseline_premium_cost) * 100
                    else:
                        premium_cost_reduction = 0.0
                    
                    # 결과에 추가
                    constraints['baseline_premium_cost'] = baseline_premium_cost
                    constraints['optimized_premium_cost'] = optimized_premium_cost
                    constraints['premium_cost_reduction'] = premium_cost_reduction
                    
                    # 목표 감축률과 비교
                    target_reduction = self.config['premium_cost'].get('reduction_target', 0.0)
                    constraints['premium_cost_target_met'] = premium_cost_reduction >= target_reduction
                except Exception as e:
                    if self.debug_mode:
                        print(f"프리미엄 비용 검증 오류: {e}")
        
        return constraints
    
    def _find_optimal_re_rate_in_range(self, material_name: str, tier: int, tier_min: float, tier_max: float, material_info: Dict[str, Any]) -> float:
        """
        UI 제약조건 범위 내에서 PCF 최소화하는 최적 RE 적용률 계산
        
        Args:
            material_name: 자재명
            tier: tier 번호 (1, 2, 3, ...)
            tier_min: UI에서 설정한 최소 RE 적용률
            tier_max: UI에서 설정한 최대 RE 적용률
            material_info: 자재 정보 딕셔너리
            
        Returns:
            float: 제약조건 범위 내 최적 RE 적용률
        """
        try:
            # 1. 기본 전략: 감축 효과 최대화 (일반적으로 최대값 선호)
            # 하지만 실제 제약조건과 목적함수를 고려
            
            # 2. proportion 자재인 경우 proportion 값 고려
            if material_info.get('is_proportion_applicable', False):
                proportion_value = self._get_proportion_tier_value(material_name, tier)
                
                # proportion이 작으면 RE 적용률을 높여야 효과적
                if proportion_value < 0.1:  # proportion이 매우 작은 경우
                    # 더 높은 RE 적용률이 필요 → 최대값 선호
                    return tier_max
                elif proportion_value > 0.5:  # proportion이 큰 경우
                    # 적당한 RE 적용률로도 충분 → 중간값 또는 더 보수적 값
                    return min(tier_max, tier_min + (tier_max - tier_min) * 0.7)
                else:
                    # 일반적인 경우 → 75% 지점
                    return tier_min + (tier_max - tier_min) * 0.75
            
            # 3. 일반 자재의 경우
            else:
                # 자재 유형에 따른 전략
                if material_info.get('is_cathode', False):
                    # 양극재: 보수적 접근 (다른 저감 수단이 많음)
                    return tier_min + (tier_max - tier_min) * 0.6
                elif material_info.get('is_energy_tier', False):
                    # Energy(Tier): 적극적 접근 (에너지 효율이 중요)
                    return tier_max
                else:
                    # 기타 일반 자재: 균형적 접근
                    return tier_min + (tier_max - tier_min) * 0.8
                    
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ _find_optimal_re_rate_in_range 오류: {e}", "ERROR")
            # 오류 발생 시 안전한 기본값 (중간값)
            return (tier_min + tier_max) / 2

    def _validate_constraints_compliance(self, optimal_re_rates: Dict[str, float], re_config_summary: Dict[str, Dict]) -> Dict[str, Any]:
        """
        최적화 결과가 UI 제약조건을 준수하는지 검증
        
        Args:
            optimal_re_rates: 최적화된 RE 적용률
            re_config_summary: RE 설정 요약
            
        Returns:
            Dict: 제약조건 준수 검증 결과
        """
        validation_result = {
            'all_constraints_satisfied': True,
            'violations': [],
            'compliance_details': {}
        }
        
        try:
            for tier in range(1, self.num_tiers + 1):
                tier_key = f'tier{tier}'
                optimal_key = f'optimal_tier{tier}_re'
                
                if tier_key in re_config_summary and optimal_key in optimal_re_rates:
                    config = re_config_summary[tier_key]
                    optimal_value = optimal_re_rates[optimal_key]
                    
                    tier_min = config['min']
                    tier_max = config['max']
                    
                    # 제약조건 준수 여부 확인
                    is_compliant = tier_min <= optimal_value <= tier_max
                    
                    validation_result['compliance_details'][tier_key] = {
                        'min_constraint': tier_min,
                        'max_constraint': tier_max,
                        'actual_value': optimal_value,
                        'is_compliant': is_compliant,
                        'constraint_applied': config.get('constraint_applied', False)
                    }
                    
                    if not is_compliant:
                        validation_result['all_constraints_satisfied'] = False
                        violation = {
                            'tier': tier_key,
                            'constraint_range': [tier_min, tier_max],
                            'actual_value': optimal_value,
                            'violation_type': 'below_min' if optimal_value < tier_min else 'above_max'
                        }
                        validation_result['violations'].append(violation)
                        
                        if self.debug_mode:
                            self._log_debug(f"⚠️ 제약조건 위반: {tier_key} = {optimal_value:.3f} (범위: [{tier_min:.3f}, {tier_max:.3f}])", "WARNING")
            
            if validation_result['all_constraints_satisfied'] and self.debug_mode:
                self._log_debug(f"✅ 모든 UI 제약조건 준수 확인됨")
                
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ 제약조건 검증 중 오류: {e}", "ERROR")
            validation_result['validation_error'] = str(e)
        
        return validation_result

    def _get_material_reduction_targets(self, material_name: str) -> Dict[str, float]:
        """
        자재별 감축 목표를 가져오기 (자재별 설정이 있으면 사용, 없으면 전역 설정 사용)
        
        Args:
            material_name: 자재명
            
        Returns:
            Dict: {'min': float, 'max': float} 감축 목표 (% 단위)
        """
        # 🚨 Streamlit UI에 디버그 정보 표시
        if hasattr(self, 'ui_debug_container') and self.ui_debug_container:
            with self.ui_debug_container:
                import streamlit as st
                st.write(f"🎯 {material_name} 감축 목표 확인")
                st.write(f"📊 Config 키들: {list(self.config.keys())}")
        
        print(f"\n🔍 [TARGET_DEBUG] _get_material_reduction_targets 호출")
        print(f"🔍 [TARGET_DEBUG] material_name: {material_name}")
        print(f"🔍 [TARGET_DEBUG] self.config 키들: {list(self.config.keys())}")
        
        # 1. 자재별 설정이 있는지 확인
        material_specific_targets = self.config.get('material_specific_targets', {})
        print(f"🔍 [TARGET_DEBUG] config에서 가져온 material_specific_targets: {material_specific_targets}")
        print(f"🔍 [TARGET_DEBUG] self.material_specific_targets: {getattr(self, 'material_specific_targets', {})}")
        
        # 🚨 UI에 material_specific_targets 상태 표시
        if hasattr(self, 'ui_debug_container') and self.ui_debug_container:
            with self.ui_debug_container:
                import streamlit as st
                st.write("📋 자재별 목표 설정 상태:")
                st.write(f"  • config에서: {material_specific_targets}")
                st.write(f"  • self에서: {getattr(self, 'material_specific_targets', {})}")
        
        if material_name in material_specific_targets:
            material_target = material_specific_targets[material_name]
            print(f"✅ [TARGET_DEBUG] 자재별 목표 발견: {material_target}")
            if self.debug_mode:
                self._log_debug(f"  🎯 {material_name}: 자재별 감축 목표 사용 - min: {material_target.get('min', 0)}%, max: {material_target.get('max', 0)}%")
            result = {
                'min': material_target.get('min', 0),
                'max': material_target.get('max', 0)
            }
            print(f"🔍 [TARGET_DEBUG] 반환할 자재별 목표: {result}")
            return result
        
        # self.material_specific_targets에서도 확인
        if material_name in self.material_specific_targets:
            material_target = self.material_specific_targets[material_name]
            print(f"✅ [TARGET_DEBUG] self.material_specific_targets에서 자재별 목표 발견: {material_target}")
            if self.debug_mode:
                self._log_debug(f"  🎯 {material_name}: self.material_specific_targets 자재별 감축 목표 사용 - min: {material_target.get('min', 0)}%, max: {material_target.get('max', 0)}%")
            result = {
                'min': material_target.get('min', 0),
                'max': material_target.get('max', 0)
            }
            print(f"🔍 [TARGET_DEBUG] 반환할 자재별 목표: {result}")
            return result
        
        # 2. 자재별 설정이 없으면 전역 설정 사용
        global_target = self.config.get('reduction_target', {})
        print(f"⚠️ [TARGET_DEBUG] 자재별 목표 없음, 전역 목표 사용: {global_target}")
        if self.debug_mode:
            self._log_debug(f"  🎯 {material_name}: 전역 감축 목표 사용 - min: {global_target.get('min', 5)}%, max: {global_target.get('max', 10)}%")
        result = {
            'min': global_target.get('min', 5),
            'max': global_target.get('max', 10)
        }
        print(f"🔍 [TARGET_DEBUG] 반환할 전역 목표: {result}")
        return result

    def optimize_single_material_re(self, material_name: str) -> Dict[str, Any]:
        """
        단일 자재에 대한 RE(Renewable Energy) 최적화 수행
        
        Args:
            material_name: 최적화할 자재명
            
        Returns:
            Dict: 최적화 결과
        """
        try:
            # 🚨 특별 추적 대상 자재들
            problem_materials = ['Al Foil', 'Cu Foil', '양극재', '음극재', '전해액']
            is_problem_material = material_name in problem_materials
            
            self._print_debug(f"🟢 CALLED: optimize_single_material_re('{material_name}')")
            if is_problem_material:
                self._print_debug(f"⚠️ SPECIAL TRACKING: 문제 자재 '{material_name}' 처리 시작!")
            
            if self.debug_mode:
                self._log_debug(f"🔧 단일 자재 RE 최적화 시작: {material_name}")
            
            # Config 검증 - 필수 키들 확인
            required_config_keys = ['re_rates', 'constraints']
            missing_keys = [key for key in required_config_keys if key not in self.config]
            if missing_keys:
                error_msg = f"필수 설정이 누락되었습니다: {missing_keys}"
                if self.debug_mode:
                    self._log_debug(f"❌ Config 검증 실패: {error_msg}", "ERROR")
                return {
                    'status': 'error',
                    'message': error_msg,
                    'material_name': material_name,
                    'missing_config_keys': missing_keys
                }
            
            # RE rates 설정 검증
            if not isinstance(self.config.get('re_rates'), dict):
                error_msg = "re_rates 설정이 유효하지 않습니다"
                if self.debug_mode:
                    self._log_debug(f"❌ RE rates 검증 실패: {error_msg}", "ERROR")
                return {
                    'status': 'error',
                    'message': error_msg,
                    'material_name': material_name
                }
            
            # Tier 설정 확인
            if not hasattr(self, 'num_tiers') or self.num_tiers <= 0:
                self.num_tiers = 2  # 기본값
                if self.debug_mode:
                    self._log_debug(f"⚠️ num_tiers가 설정되지 않음 - 기본값 {self.num_tiers} 사용", "WARNING")
            
            # 자재 정보 확인
            if material_name not in self.material_types:
                error_result = {
                    'status': 'error',
                    'message': f'자재 정보를 찾을 수 없습니다: {material_name}',
                    'material_name': material_name
                }
                if is_problem_material:
                    self._print_debug(f"❌ SPECIAL: {material_name} - material_types에 없음!")
                    self._print_debug(f"📊 SPECIAL: 현재 material_types keys: {list(self.material_types.keys()) if hasattr(self, 'material_types') else 'No material_types'}")
                return error_result
            
            material_info = self.material_types[material_name]
            original_emission = material_info['original_emission']
            quantity = material_info['quantity']
            original_pcf = original_emission * quantity
            
            # 🔍 데이터 검증 추가
            if self.debug_mode:
                self._log_debug(f"  📊 자재 데이터 확인:")
                self._log_debug(f"    - 원본 배출계수: {original_emission}")
                self._log_debug(f"    - 소요량: {quantity} kg")
                self._log_debug(f"    - 원본 PCF: {original_pcf} kgCO2eq")
                self._log_debug(f"    - 자재 유형: {material_info.get('processing_priority', 'N/A')}")
                self._log_debug(f"    - Proportion 적용: {material_info.get('is_proportion_applicable', False)}")
            
            # 기본 데이터 검증
            if original_emission <= 0:
                if self.debug_mode:
                    self._log_debug(f"  ⚠️ 원본 배출계수가 0 이하: {original_emission}", "WARNING")
                return {
                    'status': 'error',
                    'message': f'원본 배출계수가 유효하지 않습니다: {original_emission}',
                    'material_name': material_name,
                    'original_emission': original_emission,
                    'quantity': quantity
                }
            
            if quantity <= 0:
                if self.debug_mode:
                    self._log_debug(f"  ⚠️ 소요량이 0 이하: {quantity}", "WARNING")
                return {
                    'status': 'error',
                    'message': f'소요량이 유효하지 않습니다: {quantity}',
                    'material_name': material_name,
                    'original_emission': original_emission,
                    'quantity': quantity
                }
            
            # RE 적용률 최적화 (Tier별)
            optimal_re_rates = {}
            re_config_summary = {}
            
            if self.debug_mode:
                self._log_debug(f"  🔧 Tier별 RE 적용률 설정:")
            
            # PCF 최소화를 고려한 제약조건 기반 최적화
            for tier in range(1, self.num_tiers + 1):
                tier_key = f'tier{tier}'
                
                # 설정에서 범위 가져오기 (UI cap-floor 제약조건)
                if tier_key in self.config['re_rates']:
                    tier_min = self.config['re_rates'][tier_key]['min']
                    tier_max = self.config['re_rates'][tier_key]['max']
                else:
                    tier_min, tier_max = 0.1, 0.9
                    if self.debug_mode:
                        self._log_debug(f"    ⚠️ {tier_key} 설정 없음 - 기본값 사용: [{tier_min}, {tier_max}]", "WARNING")
                
                # 범위 유효성 검증
                if tier_max <= tier_min:
                    tier_max = tier_min + 0.1  # 최소 10% 차이 보장
                    if self.debug_mode:
                        self._log_debug(f"    🔧 {tier_key} 범위 수정: max를 {tier_max}로 조정", "WARNING")
                
                # 🎯 개선된 최적화: UI 제약조건 범위 내에서 PCF 최소화하는 최적값 계산
                optimal_rate = self._find_optimal_re_rate_in_range(
                    material_name, tier, tier_min, tier_max, material_info
                )
                
                optimal_re_rates[f'optimal_tier{tier}_re'] = optimal_rate
                re_config_summary[f'tier{tier}'] = {
                    'min': tier_min, 
                    'max': tier_max, 
                    'selected': optimal_rate,
                    'constraint_applied': True
                }
                
                if self.debug_mode:
                    self._log_debug(f"    • {tier_key}: 제약조건 [{tier_min:.3f}, {tier_max:.3f}] → 최적값: {optimal_rate:.3f}")
            
            # 자재별 감축 목표 가져오기
            material_targets = self._get_material_reduction_targets(material_name)
            
            if self.debug_mode:
                self._log_debug(f"  🧮 감축률 계산 시작:")
            
            # Proportion 자재인 경우 proportion 값 고려 - 상세 로그 추가
            total_reduction = 0
            tier_contributions = {}
            
            for tier in range(1, self.num_tiers + 1):
                re_rate = optimal_re_rates[f'optimal_tier{tier}_re']
                
                if material_info.get('is_proportion_applicable', False):
                    proportion_value = self._get_proportion_tier_value(material_name, tier)
                    tier_contribution = re_rate * proportion_value
                    total_reduction += tier_contribution
                    tier_contributions[f'tier{tier}'] = {
                        're_rate': re_rate,
                        'proportion': proportion_value, 
                        'contribution': tier_contribution
                    }
                    
                    if self.debug_mode:
                        self._log_debug(f"    • Tier{tier}: RE({re_rate:.3f}) × Proportion({proportion_value:.3f}) = {tier_contribution:.3f}")
                else:
                    tier_contribution = re_rate
                    total_reduction += tier_contribution
                    tier_contributions[f'tier{tier}'] = {
                        're_rate': re_rate,
                        'proportion': 1.0,  # 일반 자재는 1.0
                        'contribution': tier_contribution
                    }
                    
                    if self.debug_mode:
                        self._log_debug(f"    • Tier{tier}: RE({re_rate:.3f}) × 1.0 = {tier_contribution:.3f}")
            
            if self.debug_mode:
                self._log_debug(f"    📊 총 감축률 (제한 전): {total_reduction:.3f}")
            
            # 최대 감축률 제한 (설정값 기반)
            max_reduction = self.config.get('constraints', {}).get('max_tier_reduction', 0.8)
            original_total_reduction = total_reduction
            total_reduction = min(total_reduction, max_reduction)
            
            # 자재별 감축 목표 기반 제약 검증
            min_target_pct = material_targets.get('min', 0) / 100  # % → 비율 변환
            max_target_pct = material_targets.get('max', 0) / 100  # % → 비율 변환
            is_suboptimal = False
            is_exceeding_max = False
            target_compliance_status = "optimal"
            
            # PCF 기준 실제 감축률 계산 (최종 검증용)
            if original_pcf > 0:
                optimized_emission_temp = original_emission * (1 - total_reduction)
                optimized_pcf_temp = optimized_emission_temp * quantity
                actual_pcf_reduction = (original_pcf - optimized_pcf_temp) / original_pcf
                
                # 자재별 목표 달성 여부 확인
                if min_target_pct > 0 and actual_pcf_reduction < min_target_pct:
                    is_suboptimal = True
                    target_compliance_status = "suboptimal"
                    if self.debug_mode:
                        self._log_debug(f"    ⚠️ 자재별 최소 감축 목표 미달: {actual_pcf_reduction:.3f} ({actual_pcf_reduction*100:.1f}%) < {min_target_pct:.3f} ({min_target_pct*100:.1f}%)", "WARNING")
                
                if max_target_pct > 0 and actual_pcf_reduction > max_target_pct:
                    is_exceeding_max = True
                    if self.debug_mode:
                        self._log_debug(f"    ⚠️ 자재별 최대 감축 목표 초과: {actual_pcf_reduction:.3f} ({actual_pcf_reduction*100:.1f}%) > {max_target_pct:.3f} ({max_target_pct*100:.1f}%)", "WARNING")
                
                if self.debug_mode:
                    self._log_debug(f"    📊 자재별 목표 검증:")
                    self._log_debug(f"      - 실제 PCF 감축률: {actual_pcf_reduction:.3f} ({actual_pcf_reduction*100:.1f}%)")
                    self._log_debug(f"      - 목표 범위: {min_target_pct:.3f}~{max_target_pct:.3f} ({min_target_pct*100:.1f}%~{max_target_pct*100:.1f}%)")
                    self._log_debug(f"      - 달성 상태: {'✅ 달성' if not is_suboptimal and not is_exceeding_max else '❌ 미달성'}")
            
            if self.debug_mode:
                if original_total_reduction != total_reduction and not is_suboptimal:
                    self._log_debug(f"    🚫 최대 감축률 제한 적용: {original_total_reduction:.3f} → {total_reduction:.3f}")
                    self._log_debug(f"      - 최대 허용: {max_reduction:.3f}")
                
                self._log_debug(f"    ✅ 최종 감축률: {total_reduction:.3f} ({total_reduction*100:.1f}%)")
            
            # 🚨 특별 추적: 감축률이 0인지 확인
            if is_problem_material and total_reduction == 0:
                self._print_debug(f"⚠️ SPECIAL: {material_name} - 최종 감축률이 0!")
                self._print_debug(f"📊 SPECIAL: tier_contributions = {tier_contributions}")
                self._print_debug(f"📊 SPECIAL: re_config_summary = {re_config_summary}")
            
            # 최적화된 배출계수 및 PCF 계산
            optimized_emission = original_emission * (1 - total_reduction)
            optimized_pcf = optimized_emission * quantity
            
            reduction_amount = original_pcf - optimized_pcf
            reduction_percentage = (reduction_amount / original_pcf * 100) if original_pcf > 0 else 0
            
            # 🚨 특별 추적: 최종 결과가 0인지 확인
            if is_problem_material:
                self._print_debug(f"📊 SPECIAL: {material_name} 최종 결과")
                self._print_debug(f"  - original_emission: {original_emission:.6f}")
                self._print_debug(f"  - optimized_emission: {optimized_emission:.6f}")
                self._print_debug(f"  - reduction_amount: {reduction_amount:.6f}")
                self._print_debug(f"  - reduction_percentage: {reduction_percentage:.3f}%")
                self._print_debug(f"  - total_reduction: {total_reduction:.6f}")
                
                if reduction_amount == 0:
                    self._print_debug(f"⚠️ SPECIAL: {material_name} - 감축량이 정확히 0!")
                    self._print_debug(f"  - 원인: total_reduction = {total_reduction}")
                    self._print_debug(f"  - optimized_emission = original_emission * (1 - {total_reduction}) = {original_emission} * {1 - total_reduction} = {optimized_emission}")
                    self._print_debug(f"  - reduction_amount = {original_pcf} - {optimized_pcf} = {reduction_amount}")
            
            if self.debug_mode:
                self._log_debug(f"  🎯 최적화 결과 계산:")
                self._log_debug(f"    • 원본 배출계수: {original_emission:.6f} kgCO2eq/kg")
                self._log_debug(f"    • 최적화된 배출계수: {optimized_emission:.6f} kgCO2eq/kg")
                self._log_debug(f"    • 배출계수 감축: {(original_emission - optimized_emission):.6f} kgCO2eq/kg")
                self._log_debug(f"    • 원본 PCF: {original_pcf:.6f} kgCO2eq")
                self._log_debug(f"    • 최적화된 PCF: {optimized_pcf:.6f} kgCO2eq")
                self._log_debug(f"    • PCF 감축량: {reduction_amount:.6f} kgCO2eq")
                self._log_debug(f"    • PCF 감축률: {reduction_percentage:.3f}%")
            
            # 결과 검증 (음수 감축만 에러로 처리, 0% 감축은 허용)
            if reduction_amount < 0 or reduction_percentage < 0:
                error_msg = f"감축량이 음수입니다 - 감축량: {reduction_amount:.6f}, 감축률: {reduction_percentage:.3f}%"
                if self.debug_mode:
                    self._log_debug(f"  ❌ 결과 검증 실패: {error_msg}", "ERROR")
                    self._log_debug(f"    • 디버그 정보:")
                    self._log_debug(f"      - total_reduction: {total_reduction}")
                    self._log_debug(f"      - 1 - total_reduction: {1 - total_reduction}")
                    self._log_debug(f"      - original_emission * (1 - total_reduction): {original_emission * (1 - total_reduction)}")
                
                return {
                    'status': 'error',
                    'message': error_msg,
                    'material_name': material_name,
                    'debug_info': {
                        'original_emission': original_emission,
                        'quantity': quantity,
                        'total_reduction': total_reduction,
                        'optimized_emission': optimized_emission,
                        'tier_contributions': tier_contributions,
                        're_config_summary': re_config_summary
                    }
                }
            
            # 상태 결정: 자재별 목표 기준
            if is_suboptimal:
                status = 'suboptimal'
                warning_msg = f'자재별 최소 감축 목표({material_targets["min"]:.1f}%) 미달'
            elif is_exceeding_max:
                status = 'exceeding_target'
                warning_msg = f'자재별 최대 감축 목표({material_targets["max"]:.1f}%) 초과'
            else:
                status = 'optimal'
                warning_msg = None
            
            # 제약조건 준수 검증
            constraint_validation = self._validate_constraints_compliance(optimal_re_rates, re_config_summary)
            
            result = {
                'status': status,
                'material_name': material_name,
                'material_type': 'general_re',
                'original_emission': original_emission,
                'optimized_emission': optimized_emission,
                'original_pcf': original_pcf,
                'optimized_pcf': optimized_pcf,
                'reduction_amount': reduction_amount,
                'reduction_percentage': reduction_percentage,
                'total_reduction_rate': total_reduction,
                'tier_contributions': tier_contributions,
                're_config_summary': re_config_summary,
                'constraint_validation': constraint_validation,  # 🎯 제약조건 준수 여부
                'material_target_compliance': {  # 🎯 자재별 목표 준수 여부
                    'target_used': 'material_specific' if material_name in self.config.get('material_specific_targets', {}) else 'global',
                    'min_target_pct': material_targets['min'],
                    'max_target_pct': material_targets['max'],
                    'actual_reduction_pct': reduction_percentage,
                    'compliance_status': target_compliance_status,
                    'meets_min_target': not is_suboptimal,
                    'within_max_target': not is_exceeding_max
                },
                **optimal_re_rates
            }
            
            # 목표 미달성 시 추가 정보
            if is_suboptimal or is_exceeding_max:
                # actual_pcf_reduction이 정의된 경우에만 사용, 아니면 reduction_percentage 사용
                actual_pcf_reduction_pct = (actual_pcf_reduction * 100) if 'actual_pcf_reduction' in locals() and actual_pcf_reduction is not None else reduction_percentage
                
                result.update({
                    'warning': warning_msg,
                    'target_compliance_details': {
                        'actual_pcf_reduction_pct': actual_pcf_reduction_pct,
                        'min_target_pct': material_targets['min'],
                        'max_target_pct': material_targets['max'],
                        'shortfall_pct': max(0, material_targets['min'] - reduction_percentage) if is_suboptimal else 0,
                        'excess_pct': max(0, reduction_percentage - material_targets['max']) if is_exceeding_max else 0
                    }
                })
            
            if self.debug_mode:
                self._log_debug(f"  ✅ {material_name} RE 최적화 완료: {reduction_percentage:.2f}% 감축")
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # 더 구체적인 오류 메시지 생성
            error_type = type(e).__name__
            error_location = "Unknown"
            
            try:
                # 스택 트레이스에서 오류 발생 위치 파악
                tb_lines = error_details.split('\n')
                for line in tb_lines:
                    if 'line' in line and 'optimize_single_material_re' in line:
                        error_location = line.strip()
                        break
            except:
                pass
            
            enhanced_error_msg = f"{error_type}: {str(e)}"
            if error_location != "Unknown":
                enhanced_error_msg += f" (위치: {error_location})"
            
            if self.debug_mode:
                self._log_debug(f"❌ {material_name} RE 최적화 실패: {enhanced_error_msg}", "ERROR")
                self._log_debug(f"❌ 상세 오류:\n{error_details}", "ERROR")
                
                # 추가 디버그 정보
                self._log_debug(f"❌ 디버그 정보:")
                self._log_debug(f"   - 자재명: {material_name}")
                self._log_debug(f"   - config 상태: {type(self.config)} (키: {list(self.config.keys()) if hasattr(self, 'config') and self.config else 'None'})")
                self._log_debug(f"   - material_types 상태: {len(self.material_types) if hasattr(self, 'material_types') else 'None'}")
                self._log_debug(f"   - num_tiers: {getattr(self, 'num_tiers', 'None')}")
            
            # 🚨 Streamlit UI에도 향상된 오류 표시
            if hasattr(self, 'ui_debug_container') and self.ui_debug_container:
                try:
                    with self.ui_debug_container:
                        import streamlit as st
                        st.error(f"❌ {material_name} 최적화 실패")
                        st.write(f"**오류 유형**: {error_type}")
                        st.write(f"**오류 메시지**: {str(e)}")
                        if error_location != "Unknown":
                            st.write(f"**오류 위치**: {error_location}")
                        
                        with st.expander("상세 오류 정보"):
                            st.code(error_details)
                except:
                    # UI 오류 표시 실패 시 무시
                    pass
            
            return {
                'status': 'error',
                'material_name': material_name,
                'message': enhanced_error_msg,
                'error_type': error_type,
                'error_location': error_location,
                'error_details': error_details,
                'debug_info': {
                    'config_keys': list(self.config.keys()) if hasattr(self, 'config') and self.config else [],
                    'material_types_count': len(self.material_types) if hasattr(self, 'material_types') else 0,
                    'num_tiers': getattr(self, 'num_tiers', None)
                }
            }
    
    def optimize_cathode_with_recycling(self, material_name: str, country: str = None) -> Dict[str, Any]:
        """
        양극재 자재에 대한 재활용 + RE 최적화 수행 (자재별 감축 목표 고려)
        
        Args:
            material_name: 최적화할 양극재 자재명
            country: 대상 국가 (site_change 시나리오용)
            
        Returns:
            Dict: 최적화 결과
        """
        try:
            # 🚨 Streamlit UI에 디버그 정보 표시
            if hasattr(self, 'ui_debug_container') and self.ui_debug_container:
                with self.ui_debug_container:
                    import streamlit as st
                    st.info(f"🔍 양극재 최적화 시작: {material_name}")
                    if country:
                        st.write(f"📍 대상 국가: {country}")
                    
                    # material_specific_targets 표시
                    if hasattr(self, 'material_specific_targets'):
                        st.write("🎯 자재별 감축 목표:")
                        for mat_name, targets in self.material_specific_targets.items():
                            st.write(f"  • {mat_name}: {targets}")
            
            print(f"\n🔍 [CATHODE_DEBUG] optimize_cathode_with_recycling 함수 호출됨")
            print(f"🔍 [CATHODE_DEBUG] material_name: {material_name}")
            print(f"🔍 [CATHODE_DEBUG] country: {country}")
            
            if self.debug_mode:
                country_info = f" (국가: {country})" if country else ""
                self._log_debug(f"♻️ 양극재 재활용 최적화 시작 (목표 고려): {material_name}{country_info}")
            
            # material_specific_targets 전체 내용 확인
            print(f"🔍 [CATHODE_DEBUG] self.material_specific_targets 전체 내용:")
            for mat_name, targets in self.material_specific_targets.items():
                print(f"  - {mat_name}: {targets}")
            
            # 자재 정보 확인
            if material_name not in self.material_types:
                print(f"❌ [CATHODE_DEBUG] 자재 정보를 찾을 수 없음: {material_name}")
                print(f"🔍 [CATHODE_DEBUG] 사용 가능한 자재들: {list(self.material_types.keys())}")
                return {
                    'status': 'error',
                    'message': f'자재 정보를 찾을 수 없습니다: {material_name}',
                    'material_name': material_name
                }
            
            material_info = self.material_types[material_name]
            original_emission = material_info['original_emission']
            quantity = material_info['quantity']
            original_pcf = original_emission * quantity
            
            print(f"🔍 [CATHODE_DEBUG] material_info: {material_info}")
            print(f"🔍 [CATHODE_DEBUG] original_emission: {original_emission}, quantity: {quantity}, original_pcf: {original_pcf}")
            
            # 자재별 감축 목표 가져오기
            material_targets = self._get_material_reduction_targets(material_name)
            min_target_pct = material_targets['min'] / 100  # % → 비율 변환
            max_target_pct = material_targets['max'] / 100  # % → 비율 변환
            target_center_pct = (min_target_pct + max_target_pct) / 2  # 목표 중심값
            
            print(f"🔍 [CATHODE_DEBUG] _get_material_reduction_targets 결과: {material_targets}")
            print(f"🔍 [CATHODE_DEBUG] 목표 범위: {min_target_pct*100:.1f}%~{max_target_pct*100:.1f}% (중심: {target_center_pct*100:.1f}%)")
            
            if self.debug_mode:
                self._log_debug(f"  🎯 자재별 감축 목표: {material_targets['min']:.1f}%~{material_targets['max']:.1f}% (중심: {target_center_pct*100:.1f}%)")
            
            # 기본 데이터 검증
            if original_emission <= 0 or quantity <= 0 or original_pcf <= 0:
                return {
                    'status': 'error',
                    'message': f'유효하지 않은 자재 데이터: emission={original_emission}, quantity={quantity}',
                    'material_name': material_name
                }
            
            # 설정값 불러오기
            recycle_config = self.config.get('material_ratios', {}).get('recycle', {})
            low_carbon_config = self.config.get('material_ratios', {}).get('low_carbon', {})
            cathode_config = self.config.get('cathode', {})
            
            recycle_min = recycle_config.get('min', 0.05)
            recycle_max = recycle_config.get('max', 0.5)
            low_carbon_min = low_carbon_config.get('min', 0.05)
            low_carbon_max = low_carbon_config.get('max', 0.3)
            
            recycle_reduction = cathode_config.get('recycle_emission_reduction', 0.6)  # 재활용재 60% 감축
            low_carbon_reduction = cathode_config.get('low_carbon_emission_reduction', 0.3)  # 저탄소메탈 30% 감축
            
            # 🎯 목표 달성을 위한 최적화 알고리즘
            best_result = self._find_optimal_cathode_parameters(
                material_name=material_name,
                original_emission=original_emission,
                quantity=quantity,
                original_pcf=original_pcf,
                target_center_pct=target_center_pct,
                min_target_pct=min_target_pct,
                max_target_pct=max_target_pct,
                recycle_bounds=(recycle_min, recycle_max),
                low_carbon_bounds=(low_carbon_min, low_carbon_max),
                recycle_reduction=recycle_reduction,
                low_carbon_reduction=low_carbon_reduction,
                material_info=material_info
            )
            
            # 목표 달성 상태 평가
            actual_reduction_pct = best_result['reduction_percentage'] / 100
            is_suboptimal = min_target_pct > 0 and actual_reduction_pct < min_target_pct
            is_exceeding_max = max_target_pct > 0 and actual_reduction_pct > max_target_pct
            
            if is_suboptimal:
                status = 'suboptimal'
                warning_msg = f'자재별 최소 감축 목표({material_targets["min"]:.1f}%) 미달'
            elif is_exceeding_max:
                status = 'exceeding_target'
                warning_msg = f'자재별 최대 감축 목표({material_targets["max"]:.1f}%) 초과'
            else:
                status = 'optimal'
                warning_msg = None
            
            # 결과 구성
            result = {
                'status': status,
                'material_name': material_name,
                'material_type': 'cathode',
                'country': country,
                'original_emission': original_emission,
                'optimized_emission': best_result['optimized_emission'],
                'original_pcf': original_pcf,
                'optimized_pcf': best_result['optimized_pcf'],
                'reduction_amount': best_result['reduction_amount'],
                'reduction_percentage': best_result['reduction_percentage'],
                'optimal_recycle_ratio': best_result['optimal_recycle_ratio'],
                'optimal_low_carbon_ratio': best_result['optimal_low_carbon_ratio'],
                'optimal_virgin_ratio': best_result['optimal_virgin_ratio'],
                'mixed_emission': best_result['mixed_emission'],
                'total_re_reduction': best_result['total_re_reduction'],
                'material_target_compliance': {
                    'target_used': 'material_specific' if material_name in self.config.get('material_specific_targets', {}) else 'global',
                    'min_target_pct': material_targets['min'],
                    'max_target_pct': material_targets['max'],
                    'actual_reduction_pct': best_result['reduction_percentage'],
                    'compliance_status': 'optimal' if (not is_suboptimal and not is_exceeding_max) else ('suboptimal' if is_suboptimal else 'exceeding_target'),
                    'meets_min_target': not is_suboptimal,
                    'within_max_target': not is_exceeding_max
                },
                **best_result['re_rates']
            }
            
            # 목표 미달성 시 추가 정보
            if is_suboptimal or is_exceeding_max:
                result.update({
                    'warning': warning_msg,
                    'target_compliance_details': {
                        'actual_pcf_reduction_pct': actual_reduction_pct * 100,
                        'min_target_pct': material_targets['min'],
                        'max_target_pct': material_targets['max'],
                        'shortfall_pct': max(0, material_targets['min'] - best_result['reduction_percentage']) if is_suboptimal else 0,
                        'excess_pct': max(0, best_result['reduction_percentage'] - material_targets['max']) if is_exceeding_max else 0
                    }
                })
            
            if self.debug_mode:
                self._log_debug(f"  ✅ {material_name} 재활용 최적화 완료: {best_result['reduction_percentage']:.2f}% 감축 (상태: {status})")
                self._log_debug(f"    - 재활용: {best_result['optimal_recycle_ratio']*100:.1f}%, 저탄소: {best_result['optimal_low_carbon_ratio']*100:.1f}%, 신재: {best_result['optimal_virgin_ratio']*100:.1f}%")
                self._log_debug(f"    - 목표: {material_targets['min']:.1f}%~{material_targets['max']:.1f}%, 달성: {'✅' if not is_suboptimal and not is_exceeding_max else '❌'}")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ {material_name} 재활용 최적화 실패: {e}", "ERROR")
            
            return {
                'status': 'error',
                'material_name': material_name,
                'country': country,
                'message': str(e)
            }
    
    def _find_optimal_cathode_parameters(self, material_name: str, original_emission: float, quantity: float, 
                                       original_pcf: float, target_center_pct: float, min_target_pct: float, 
                                       max_target_pct: float, recycle_bounds: tuple, low_carbon_bounds: tuple,
                                       recycle_reduction: float, low_carbon_reduction: float, 
                                       material_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        목표 감축률 달성을 위한 양극재 최적 파라미터 찾기
        
        Args:
            material_name: 자재명
            original_emission: 원본 배출계수
            quantity: 소요량
            original_pcf: 원본 PCF
            target_center_pct: 목표 중심값 (비율)
            min_target_pct: 최소 목표 (비율) 
            max_target_pct: 최대 목표 (비율)
            recycle_bounds: 재활용 비율 범위 (min, max)
            low_carbon_bounds: 저탄소메탈 비율 범위 (min, max)
            recycle_reduction: 재활용재 배출계수 감축 효과
            low_carbon_reduction: 저탄소메탈 배출계수 감축 효과
            material_info: 자재 정보
            
        Returns:
            Dict: 최적 파라미터와 결과
        """
        try:
            # 🚨 Streamlit UI에 최적화 과정 표시
            if hasattr(self, 'ui_debug_container') and self.ui_debug_container:
                with self.ui_debug_container:
                    import streamlit as st
                    st.success(f"🔧 {material_name} 양극재 파라미터 최적화")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("📊 **기본 정보:**")
                        st.write(f"  • 원본 배출계수: {original_emission:.6f}")
                        st.write(f"  • 소요량: {quantity:.3f} kg")
                        st.write(f"  • 원본 PCF: {original_pcf:.3f}")
                    
                    with col2:
                        st.write("🎯 **감축 목표:**")
                        st.write(f"  • 목표 중심: {target_center_pct*100:.1f}%")
                        st.write(f"  • 범위: {min_target_pct*100:.1f}% ~ {max_target_pct*100:.1f}%")
                    
                    st.write("⚙️ **최적화 파라미터:**")
                    st.write(f"  • 재활용 비율 범위: {recycle_bounds[0]*100:.1f}% ~ {recycle_bounds[1]*100:.1f}%")
                    st.write(f"  • 저탄소 비율 범위: {low_carbon_bounds[0]*100:.1f}% ~ {low_carbon_bounds[1]*100:.1f}%")
                    st.write(f"  • 재활용 감축 효과: {recycle_reduction*100:.1f}%")
                    st.write(f"  • 저탄소 감축 효과: {low_carbon_reduction*100:.1f}%")
            
            print(f"\n🔍 [OPTIMIZE_DEBUG] _find_optimal_cathode_parameters 시작")
            print(f"🔍 [OPTIMIZE_DEBUG] material_name: {material_name}")
            print(f"🔍 [OPTIMIZE_DEBUG] original_emission: {original_emission}")
            print(f"🔍 [OPTIMIZE_DEBUG] quantity: {quantity}")
            print(f"🔍 [OPTIMIZE_DEBUG] original_pcf: {original_pcf}")
            print(f"🔍 [OPTIMIZE_DEBUG] target_center_pct: {target_center_pct*100:.1f}%")
            print(f"🔍 [OPTIMIZE_DEBUG] min_target_pct: {min_target_pct*100:.1f}%")
            print(f"🔍 [OPTIMIZE_DEBUG] max_target_pct: {max_target_pct*100:.1f}%")
            print(f"🔍 [OPTIMIZE_DEBUG] recycle_bounds: {recycle_bounds}")
            print(f"🔍 [OPTIMIZE_DEBUG] low_carbon_bounds: {low_carbon_bounds}")
            print(f"🔍 [OPTIMIZE_DEBUG] recycle_reduction: {recycle_reduction*100:.1f}%")
            print(f"🔍 [OPTIMIZE_DEBUG] low_carbon_reduction: {low_carbon_reduction*100:.1f}%")
            
            if self.debug_mode:
                self._log_debug(f"  🔍 목표 달성 최적화 시작: {material_name}")
                self._log_debug(f"    - 목표 중심: {target_center_pct*100:.1f}%, 범위: {min_target_pct*100:.1f}%~{max_target_pct*100:.1f}%")
            
            # 각 배출계수 계산
            virgin_emission = original_emission
            recycle_emission = original_emission * (1 - recycle_reduction)
            low_carbon_emission = original_emission * (1 - low_carbon_reduction)
            
            # 후보 시나리오들 생성 및 평가
            scenarios = []
            
            # 1. 보수적 접근: 최소값부터 시작
            scenarios.append(('conservative', recycle_bounds[0], low_carbon_bounds[0]))
            
            # 2. 중간 접근: 중간값 사용
            recycle_mid = (recycle_bounds[0] + recycle_bounds[1]) / 2
            low_carbon_mid = (low_carbon_bounds[0] + low_carbon_bounds[1]) / 2
            scenarios.append(('moderate', recycle_mid, low_carbon_mid))
            
            # 3. 적극적 접근: 최대값 사용
            scenarios.append(('aggressive', recycle_bounds[1], low_carbon_bounds[1]))
            
            # 4. 목표 중심 접근: 목표 달성에 최적화된 비율 계산
            if target_center_pct > 0:
                target_recycle, target_low_carbon = self._calculate_target_optimized_ratios(
                    target_center_pct, original_emission, recycle_emission, low_carbon_emission,
                    recycle_bounds, low_carbon_bounds
                )
                scenarios.append(('target_optimized', target_recycle, target_low_carbon))
            
            best_scenario = None
            best_distance = float('inf')
            
            if self.debug_mode:
                self._log_debug(f"    🧪 시나리오 평가 시작 ({len(scenarios)}개)")
            
            for scenario_name, recycle_ratio, low_carbon_ratio in scenarios:
                # 비율 합계 검증 및 조정
                if recycle_ratio + low_carbon_ratio > 0.95:  # 신재 5% 최소 보장
                    scale = 0.95 / (recycle_ratio + low_carbon_ratio)
                    recycle_ratio *= scale
                    low_carbon_ratio *= scale
                
                virgin_ratio = 1.0 - recycle_ratio - low_carbon_ratio
                
                # 혼합 배출계수 계산
                mixed_emission = (
                    virgin_ratio * virgin_emission +
                    recycle_ratio * recycle_emission +
                    low_carbon_ratio * low_carbon_emission
                )
                
                # RE 최적화 (UI 제약조건 고려)
                re_result = self._optimize_re_rates_for_target(
                    material_name, mixed_emission, quantity, original_pcf,
                    target_center_pct, material_info
                )
                
                final_emission = mixed_emission * (1 - re_result['total_re_reduction'])
                final_pcf = final_emission * quantity
                reduction_amount = original_pcf - final_pcf
                reduction_percentage = (reduction_amount / original_pcf * 100) if original_pcf > 0 else 0
                reduction_ratio = reduction_percentage / 100
                
                # 목표와의 거리 계산 (목표 중심값 기준)
                distance = abs(reduction_ratio - target_center_pct)
                
                scenario_result = {
                    'scenario_name': scenario_name,
                    'optimal_recycle_ratio': recycle_ratio,
                    'optimal_low_carbon_ratio': low_carbon_ratio,
                    'optimal_virgin_ratio': virgin_ratio,
                    'mixed_emission': mixed_emission,
                    'optimized_emission': final_emission,
                    'optimized_pcf': final_pcf,
                    'reduction_amount': reduction_amount,
                    'reduction_percentage': reduction_percentage,
                    'total_re_reduction': re_result['total_re_reduction'],
                    're_rates': re_result['re_rates'],
                    'target_distance': distance
                }
                
                if self.debug_mode:
                    self._log_debug(f"      • {scenario_name}: 감축률 {reduction_percentage:.1f}%, 목표와 거리 {distance*100:.1f}%p")
                    self._log_debug(f"        - 비율: 재활용({recycle_ratio*100:.1f}%), 저탄소({low_carbon_ratio*100:.1f}%), 신재({virgin_ratio*100:.1f}%)")
                
                # 🚨 수정: 목표 범위 엄격 준수 - 범위 내 시나리오 우선 선택
                is_in_range = (min_target_pct <= reduction_ratio <= max_target_pct) if max_target_pct > 0 else (reduction_ratio >= min_target_pct)
                
                if is_in_range:
                    # 목표 범위 내 시나리오 우선 선택
                    if best_scenario is None or not best_scenario.get('is_in_range', False) or distance < best_distance:
                        best_distance = distance
                        best_scenario = scenario_result
                        best_scenario['is_in_range'] = True
                elif best_scenario is None or not best_scenario.get('is_in_range', False):
                    # 범위 내 시나리오가 없으면 가장 가까운 것 선택 (범위 밖에서)
                    if best_distance == float('inf') or distance < best_distance:
                        best_distance = distance
                        best_scenario = scenario_result
                        best_scenario['is_in_range'] = False
            
            if self.debug_mode and best_scenario:
                self._log_debug(f"    ✅ 선택된 시나리오: {best_scenario['scenario_name']}")
                self._log_debug(f"      - 최종 감축률: {best_scenario['reduction_percentage']:.2f}%")
                self._log_debug(f"      - 목표와의 거리: {best_scenario['target_distance']*100:.1f}%p")
            
            # 🚨 Streamlit UI에 최적화 결과 표시
            if hasattr(self, 'ui_debug_container') and self.ui_debug_container:
                with self.ui_debug_container:
                    import streamlit as st
                    if best_scenario:
                        st.success(f"✅ {material_name} 최적화 완료!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("최종 감축률", f"{best_scenario['reduction_percentage']:.1f}%")
                            st.metric("선택된 시나리오", best_scenario['scenario_name'])
                        
                        with col2:
                            st.write("📊 **재료 비율:**")
                            st.write(f"  • 재활용: {best_scenario['optimal_recycle_ratio']*100:.1f}%")
                            st.write(f"  • 저탄소: {best_scenario['optimal_low_carbon_ratio']*100:.1f}%")
                            st.write(f"  • 신재: {best_scenario['optimal_virgin_ratio']*100:.1f}%")
                        
                        with col3:
                            st.write("🎯 **성과:**")
                            st.write(f"  • 목표 범위: {min_target_pct*100:.1f}%~{max_target_pct*100:.1f}%")
                            st.write(f"  • 달성: {best_scenario['reduction_percentage']:.1f}%")
                            
                            # 목표 달성 여부 표시
                            is_in_range = min_target_pct*100 <= best_scenario['reduction_percentage'] <= max_target_pct*100
                            if is_in_range:
                                st.success("🎯 목표 범위 내 달성!")
                            else:
                                st.warning("⚠️ 목표 범위 벗어남")
                    else:
                        st.error(f"❌ {material_name} 최적화 실패")
            
            print(f"\n🔍 [OPTIMIZE_RESULT] 최적화 결과:")
            if best_scenario:
                print(f"🔍 [OPTIMIZE_RESULT] 선택된 시나리오: {best_scenario['scenario_name']}")
                print(f"🔍 [OPTIMIZE_RESULT] 최종 감축률: {best_scenario['reduction_percentage']:.2f}%")
                print(f"🔍 [OPTIMIZE_RESULT] 재활용 비율: {best_scenario['optimal_recycle_ratio']*100:.1f}%")
                print(f"🔍 [OPTIMIZE_RESULT] 저탄소 비율: {best_scenario['optimal_low_carbon_ratio']*100:.1f}%")
                print(f"🔍 [OPTIMIZE_RESULT] 신재 비율: {best_scenario['optimal_virgin_ratio']*100:.1f}%")
                print(f"🔍 [OPTIMIZE_RESULT] 배출계수: {best_scenario['optimized_emission']:.6f}")
                print(f"🔍 [OPTIMIZE_RESULT] 최적화 PCF: {best_scenario['optimized_pcf']:.3f}")
                print(f"🔍 [OPTIMIZE_RESULT] 감축 목표 범위: {min_target_pct*100:.1f}%~{max_target_pct*100:.1f}%")
            else:
                print(f"❌ [OPTIMIZE_RESULT] 최적화 실패 - fallback 사용")
            
            return best_scenario or {
                'scenario_name': 'fallback',
                'optimal_recycle_ratio': recycle_bounds[0],
                'optimal_low_carbon_ratio': low_carbon_bounds[0],
                'optimal_virgin_ratio': 1.0 - recycle_bounds[0] - low_carbon_bounds[0],
                'mixed_emission': original_emission,
                'optimized_emission': original_emission,
                'optimized_pcf': original_pcf,
                'reduction_amount': 0,
                'reduction_percentage': 0,
                'total_re_reduction': 0,
                're_rates': {},
                'target_distance': float('inf')
            }
            
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ 목표 최적화 실패: {e}", "ERROR")
            
            # 오류 발생 시 안전한 기본값 반환
            return {
                'scenario_name': 'error_fallback',
                'optimal_recycle_ratio': recycle_bounds[0],
                'optimal_low_carbon_ratio': low_carbon_bounds[0],
                'optimal_virgin_ratio': 1.0 - recycle_bounds[0] - low_carbon_bounds[0],
                'mixed_emission': original_emission,
                'optimized_emission': original_emission,
                'optimized_pcf': original_pcf,
                'reduction_amount': 0,
                'reduction_percentage': 0,
                'total_re_reduction': 0,
                're_rates': {},
                'target_distance': float('inf'),
                'error': str(e)
            }
    
    def _calculate_target_optimized_ratios(self, target_pct: float, original_emission: float,
                                         recycle_emission: float, low_carbon_emission: float,
                                         recycle_bounds: tuple, low_carbon_bounds: tuple) -> tuple:
        """
        목표 감축률에 최적화된 재활용/저탄소메탈 비율 계산
        
        Args:
            target_pct: 목표 감축률 (비율)
            original_emission: 원본 배출계수
            recycle_emission: 재활용재 배출계수
            low_carbon_emission: 저탄소메탈 배출계수
            recycle_bounds: 재활용 비율 범위
            low_carbon_bounds: 저탄소메탈 비율 범위
            
        Returns:
            tuple: (최적 재활용 비율, 최적 저탄소메탈 비율)
        """
        try:
            # 목표 배출계수 계산 (RE 적용 전 기준)
            # 보수적으로 RE로 추가 20-30% 감축 가능하다고 가정
            estimated_re_reduction = 0.25  # 25% 추가 감축 예상
            target_mixed_emission = original_emission * (1 - target_pct) / (1 - estimated_re_reduction)
            
            # 재활용재와 저탄소메탈의 감축 효과를 고려한 최적 비율 계산
            # 목표 배출계수 = virgin_ratio * original + recycle_ratio * recycle + low_carbon_ratio * low_carbon
            # virgin_ratio = 1 - recycle_ratio - low_carbon_ratio
            
            # 재활용재 우선 전략 (더 큰 감축 효과)
            if recycle_emission < low_carbon_emission:
                # 재활용재를 최대한 사용하고 나머지를 저탄소메탈로
                max_recycle = min(recycle_bounds[1], 0.6)  # 최대 60%로 제한
                remaining_reduction_needed = target_mixed_emission - (original_emission * (1 - max_recycle) + recycle_emission * max_recycle)
                
                if remaining_reduction_needed < 0:  # 재활용재만으로 충분
                    # 필요한 재활용 비율 계산
                    needed_recycle = (original_emission - target_mixed_emission) / (original_emission - recycle_emission)
                    recycle_ratio = max(recycle_bounds[0], min(recycle_bounds[1], needed_recycle))
                    low_carbon_ratio = low_carbon_bounds[0]  # 최소값 사용
                else:
                    # 재활용재 + 저탄소메탈 조합 필요
                    recycle_ratio = max_recycle
                    
                    # 남은 감축량을 저탄소메탈로 달성
                    virgin_with_recycle = 1 - max_recycle
                    current_emission = virgin_with_recycle * original_emission + max_recycle * recycle_emission
                    additional_reduction_needed = current_emission - target_mixed_emission
                    
                    if additional_reduction_needed > 0 and (original_emission - low_carbon_emission) > 0:
                        needed_low_carbon = additional_reduction_needed / (original_emission - low_carbon_emission)
                        low_carbon_ratio = max(low_carbon_bounds[0], min(low_carbon_bounds[1], needed_low_carbon))
                    else:
                        low_carbon_ratio = low_carbon_bounds[0]
            else:
                # 저탄소메탈 우선 전략
                max_low_carbon = min(low_carbon_bounds[1], 0.4)  # 최대 40%로 제한
                remaining_reduction_needed = target_mixed_emission - (original_emission * (1 - max_low_carbon) + low_carbon_emission * max_low_carbon)
                
                if remaining_reduction_needed < 0:  # 저탄소메탈만으로 충분
                    needed_low_carbon = (original_emission - target_mixed_emission) / (original_emission - low_carbon_emission)
                    low_carbon_ratio = max(low_carbon_bounds[0], min(low_carbon_bounds[1], needed_low_carbon))
                    recycle_ratio = recycle_bounds[0]
                else:
                    # 저탄소메탈 + 재활용재 조합
                    low_carbon_ratio = max_low_carbon
                    
                    virgin_with_low_carbon = 1 - max_low_carbon
                    current_emission = virgin_with_low_carbon * original_emission + max_low_carbon * low_carbon_emission
                    additional_reduction_needed = current_emission - target_mixed_emission
                    
                    if additional_reduction_needed > 0 and (original_emission - recycle_emission) > 0:
                        needed_recycle = additional_reduction_needed / (original_emission - recycle_emission)
                        recycle_ratio = max(recycle_bounds[0], min(recycle_bounds[1], needed_recycle))
                    else:
                        recycle_ratio = recycle_bounds[0]
            
            # 비율 합계 검증
            if recycle_ratio + low_carbon_ratio > 0.95:
                scale = 0.95 / (recycle_ratio + low_carbon_ratio)
                recycle_ratio *= scale
                low_carbon_ratio *= scale
            
            return (
                max(recycle_bounds[0], min(recycle_bounds[1], recycle_ratio)),
                max(low_carbon_bounds[0], min(low_carbon_bounds[1], low_carbon_ratio))
            )
            
        except Exception:
            # 계산 실패 시 중간값 반환
            return (
                (recycle_bounds[0] + recycle_bounds[1]) / 2,
                (low_carbon_bounds[0] + low_carbon_bounds[1]) / 2
            )
    
    def _optimize_re_rates_for_target(self, material_name: str, mixed_emission: float, quantity: float,
                                    original_pcf: float, target_pct: float, material_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        목표 달성을 위한 RE 적용률 최적화 (UI 제약조건 고려)
        
        Args:
            material_name: 자재명
            mixed_emission: 혼합 배출계수 (재활용+저탄소메탈 적용 후)
            quantity: 소요량
            original_pcf: 원본 PCF
            target_pct: 목표 감축률 (비율)
            material_info: 자재 정보
            
        Returns:
            Dict: RE 최적화 결과
        """
        try:
            total_re_reduction = 0
            re_rates = {}
            
            # 목표 달성에 필요한 추가 감축률 계산
            current_pcf_after_mixing = mixed_emission * quantity
            current_reduction = (original_pcf - current_pcf_after_mixing) / original_pcf
            remaining_reduction_needed = target_pct - current_reduction
            
            if self.debug_mode:
                self._log_debug(f"      🔧 RE 최적화: 현재 감축률 {current_reduction*100:.1f}%, 목표 {target_pct*100:.1f}%, 추가 필요 {remaining_reduction_needed*100:.1f}%")
            
            if remaining_reduction_needed <= 0:
                # 이미 목표 달성 - 최소 RE 사용
                for tier in range(1, self.num_tiers + 1):
                    tier_key = f'tier{tier}'
                    if tier_key in self.config['re_rates']:
                        tier_min = self.config['re_rates'][tier_key]['min']
                        re_rates[f'optimal_tier{tier}_re'] = tier_min
                        total_re_reduction += tier_min
                    else:
                        re_rates[f'optimal_tier{tier}_re'] = 0.1
                        total_re_reduction += 0.1
            else:
                # 목표 달성을 위한 RE 최적화 - 🚨 수정: 목표 감축률 고려한 보수적 접근
                # remaining_reduction_needed가 클 수록 적절히 분산하여 적용
                
                # 최대 총 RE 감축률을 제한하여 과도한 감축 방지
                max_total_re_reduction = min(0.4, remaining_reduction_needed * 1.2)  # 최대 40% 또는 필요량의 120%
                available_reduction = max_total_re_reduction
                
                if self.debug_mode:
                    self._log_debug(f"      🎯 최대 총 RE 감축률 제한: {max_total_re_reduction*100:.1f}%")
                
                for tier in range(1, self.num_tiers + 1):
                    tier_key = f'tier{tier}'
                    
                    if tier_key in self.config['re_rates']:
                        tier_min = self.config['re_rates'][tier_key]['min']
                        tier_max = self.config['re_rates'][tier_key]['max']
                    else:
                        tier_min, tier_max = 0.1, 0.9
                    
                    # proportion 자재면 proportion 값 고려
                    if material_info.get('is_proportion_applicable', False):
                        proportion_value = self._get_proportion_tier_value(material_name, tier)
                        
                        # 필요한 RE 적용률 = 남은 감축량 / (proportion 값 * tier 개수)  
                        if proportion_value > 0 and available_reduction > 0:
                            # 균등 분산하되 제한된 범위 내에서 적용
                            target_re_per_tier = available_reduction / max(1, self.num_tiers - tier + 1)
                            needed_re = target_re_per_tier / proportion_value
                            optimal_re = max(tier_min, min(tier_max, needed_re))
                        else:
                            optimal_re = tier_min
                        
                        tier_contribution = optimal_re * proportion_value
                    else:
                        # 일반 자재 - 균등 분산 적용
                        if available_reduction > 0:
                            target_re_per_tier = available_reduction / max(1, self.num_tiers - tier + 1)
                            optimal_re = max(tier_min, min(tier_max, target_re_per_tier))
                        else:
                            optimal_re = tier_min
                        
                        tier_contribution = optimal_re
                    
                    re_rates[f'optimal_tier{tier}_re'] = optimal_re
                    total_re_reduction += tier_contribution
                    
                    # 남은 가능 감축량 업데이트
                    available_reduction = max(0, available_reduction - tier_contribution)
                    
                    if self.debug_mode:
                        self._log_debug(f"        - Tier{tier}: RE적용률 {optimal_re*100:.1f}%, 기여도 {tier_contribution*100:.1f}%, 남은가용량 {available_reduction*100:.1f}%")
            
            # 최대 감축률 제한
            max_reduction = self.config.get('constraints', {}).get('max_tier_reduction', 0.8)
            total_re_reduction = min(total_re_reduction, max_reduction)
            
            return {
                'total_re_reduction': total_re_reduction,
                're_rates': re_rates
            }
            
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ RE 최적화 실패: {e}", "ERROR")
            
            # 오류 시 기본값
            re_rates = {}
            for tier in range(1, self.num_tiers + 1):
                re_rates[f'optimal_tier{tier}_re'] = 0.1
            
            return {
                'total_re_reduction': 0.1 * self.num_tiers,
                're_rates': re_rates
            }

    def optimize_site_selection(self, material_name: str, countries: List[str]) -> Dict[str, Any]:
        """
        자재별 다국가 사이트 선택 최적화
        
        Args:
            material_name: 최적화할 자재명
            countries: 비교할 국가 목록
            
        Returns:
            Dict: 최적화 결과
        """
        try:
            if self.debug_mode:
                self._log_debug(f"🌍 사이트 선택 최적화 시작: {material_name} (국가: {countries})")
            
            # 자재 정보 확인
            if material_name not in self.material_types:
                return {
                    'status': 'error',
                    'message': f'자재 정보를 찾을 수 없습니다: {material_name}',
                    'material_name': material_name
                }
            
            material_info = self.material_types[material_name]
            original_emission = material_info['original_emission']
            quantity = material_info['quantity']
            original_pcf = original_emission * quantity
            
            # 각 국가별 최적화 수행
            country_results = {}
            best_country = None
            best_pcf = float('inf')
            
            for country in countries:
                try:
                    # 국가별 전력배출계수 적용
                    country_electricity_coef = self.electricity_coef.get(country, 0.5)  # 기본값
                    
                    # Energy(Tier) 자재인 경우 전력배출계수 적용
                    if material_info.get('is_energy_tier', False):
                        # 전력배출계수 비율로 배출계수 조정
                        base_coef = self.electricity_coef.get('한국', 0.4644)  # 기준값
                        adjustment_ratio = country_electricity_coef / base_coef
                        country_emission = original_emission * adjustment_ratio
                    else:
                        # 일반 자재는 원본 배출계수 사용
                        country_emission = original_emission
                    
                    # RE 최적화 적용
                    total_re_reduction = 0
                    optimal_re_rates = {}
                    
                    for tier in range(1, self.num_tiers + 1):
                        tier_key = f'tier{tier}'
                        
                        if tier_key in self.config['re_rates']:
                            tier_max = self.config['re_rates'][tier_key]['max']
                        else:
                            tier_max = 0.9
                        
                        optimal_re_rates[f'optimal_tier{tier}_re'] = tier_max
                        
                        if material_info.get('is_proportion_applicable', False):
                            proportion_value = self._get_proportion_tier_value(material_name, tier)
                            total_re_reduction += tier_max * proportion_value
                        else:
                            total_re_reduction += tier_max
                    
                    max_reduction = self.config.get('constraints', {}).get('max_tier_reduction', 0.8)
                    total_re_reduction = min(total_re_reduction, max_reduction)
                    
                    # 최적화된 배출계수 및 PCF
                    optimized_emission = country_emission * (1 - total_re_reduction)
                    optimized_pcf = optimized_emission * quantity
                    
                    reduction_amount = original_pcf - optimized_pcf
                    reduction_percentage = (reduction_amount / original_pcf * 100) if original_pcf > 0 else 0
                    
                    country_result = {
                        'status': 'optimal',
                        'country': country,
                        'country_electricity_coef': country_electricity_coef,
                        'country_emission': country_emission,
                        'optimized_emission': optimized_emission,
                        'optimized_pcf': optimized_pcf,
                        'reduction_amount': reduction_amount,
                        'reduction_percentage': reduction_percentage,
                        'total_re_reduction': total_re_reduction,
                        **optimal_re_rates
                    }
                    
                    country_results[country] = country_result
                    
                    # 최적 국가 업데이트
                    if optimized_pcf < best_pcf:
                        best_pcf = optimized_pcf
                        best_country = country
                    
                except Exception as country_error:
                    if self.debug_mode:
                        self._log_debug(f"  ⚠️ {country} 최적화 실패: {country_error}")
                    
                    country_results[country] = {
                        'status': 'error',
                        'country': country,
                        'message': str(country_error)
                    }
            
            # 최종 결과 구성
            if best_country:
                best_result = country_results[best_country]
                
                result = {
                    'status': 'optimal',
                    'material_name': material_name,
                    'material_type': 'site_selection',
                    'best_country': best_country,
                    'best_pcf': best_pcf,
                    'original_emission': original_emission,
                    'original_pcf': original_pcf,
                    'country_results': country_results,
                    **{k: v for k, v in best_result.items() if k not in ['status', 'country']}
                }
                
                if self.debug_mode:
                    self._log_debug(f"  ✅ {material_name} 사이트 선택 완료: 최적 국가 = {best_country}")
                
                return result
            else:
                return {
                    'status': 'error',
                    'material_name': material_name,
                    'message': '모든 국가에서 최적화 실패',
                    'country_results': country_results
                }
            
        except Exception as e:
            if self.debug_mode:
                self._log_debug(f"❌ {material_name} 사이트 선택 최적화 실패: {e}", "ERROR")
            
            return {
                'status': 'error',
                'material_name': material_name,
                'message': str(e)
            }

    def get_formatted_results(self) -> Dict[str, Any]:
        """새로운 시나리오별 결과 포맷팅"""
        if not self.results or self.results.get('status') != 'optimal':
            return {
                'status': 'error', 
                'message': '최적화 결과가 없거나 실패했습니다.',
                'scenario': self.results.get('scenario', 'unknown') if self.results else 'unknown'
            }
        
        # 기본 결과 정보
        formatted = {
            'status': 'optimal',
            'scenario': self.results['scenario'],
            'summary': {
                'original_pcf': f"{self.results['original_pcf']:.4f} kgCO2eq",
                'optimized_pcf': f"{self.results['optimized_pcf']:.4f} kgCO2eq",
                'reduction_amount': f"{self.results['reduction_amount']:.4f} kgCO2eq",
                'reduction_percentage': f"{self.results['reduction_percentage']:.2f}%",
                'success_rate': f"{self.results['success_rate']:.1f}%",
                'total_materials': self.results['total_materials'],
                'successful_materials': self.results['successful_materials']
            }
        }
        
        # 자재별 최적화 결과
        materials_data = []
        for material_name, result in self.results['materials'].items():
            if result.get('status') != 'optimal':
                # 최적화 실패한 자재
                material_data = {
                    'name': material_name,
                    'status': 'failed',
                    'message': result.get('message', '최적화 실패'),
                    'category': self._get_material_category(material_name)
                }
            else:
                # 성공한 자재
                material_data = {
                    'name': material_name,
                    'status': 'optimal',
                    'category': self._get_material_category(material_name),
                    'original_pcf': f"{result.get('original_pcf', 0):.4f} kgCO2eq",
                    'optimized_pcf': f"{result.get('optimized_pcf', 0):.4f} kgCO2eq",
                    'reduction_amount': f"{result.get('original_pcf', 0) - result.get('optimized_pcf', 0):.4f} kgCO2eq",
                    'reduction_percentage': f"{result.get('reduction_percentage', 0):.2f}%"
                }
                
                # 시나리오별 추가 정보
                if self.scenario == 'baseline' or 'tier' in result:
                    # RE 최적화 결과
                    tier_results = {}
                    for tier in range(1, self.num_tiers + 1):
                        if f'optimal_tier{tier}_re' in result:
                            tier_results[f'tier{tier}_re'] = f"{result[f'optimal_tier{tier}_re'] * 100:.1f}%"
                    material_data['tier_optimization'] = tier_results
                
                if (self.scenario == 'recycling' or self.scenario == 'both') and result.get('material_type') in ['cathode', 'cathode_both']:
                    # 양극재 재활용 비율 결과
                    material_data['recycling_optimization'] = {
                        'recycle_ratio': f"{result.get('optimal_recycle_ratio', 0) * 100:.1f}%",
                        'low_carbon_ratio': f"{result.get('optimal_low_carbon_ratio', 0) * 100:.1f}%",
                        'virgin_ratio': f"{result.get('optimal_virgin_ratio', 0) * 100:.1f}%"
                    }
                
                if (self.scenario == 'site_change' or self.scenario == 'both') and 'best_country' in result:
                    # 사이트 선택 결과
                    material_data['site_optimization'] = {
                        'best_country': result['best_country'],
                        'country_pcf': f"{result.get('best_pcf', 0):.4f} kgCO2eq"
                    }
                    
                    # 다른 국가들의 결과도 포함 (상위 3개만)
                    if 'country_results' in result:
                        country_comparison = []
                        sorted_countries = sorted(
                            result['country_results'].items(),
                            key=lambda x: x[1].get('optimized_pcf', float('inf'))
                        )
                        for country, country_result in sorted_countries[:3]:
                            if country_result.get('status') == 'optimal':
                                country_comparison.append({
                                    'country': country,
                                    'pcf': f"{country_result.get('optimized_pcf', 0):.4f} kgCO2eq",
                                    'reduction': f"{country_result.get('reduction_percentage', 0):.2f}%"
                                })
                        material_data['country_comparison'] = country_comparison
            
            materials_data.append(material_data)
        
        formatted['materials'] = materials_data
        
        # 시나리오별 전체 통계
        if self.scenario == 'site_change' or self.scenario == 'both':
            if 'country_selection' in self.results:
                formatted['country_statistics'] = {
                    country: {
                        'selected_materials': stats['count'],
                        'material_names': stats['materials'][:3]  # 상위 3개만
                    }
                    for country, stats in self.results['country_selection'].items()
                }
        
        if self.scenario == 'recycling' or self.scenario == 'both':
            if 'cathode_recycling' in self.results:
                cathode_stats = self.results['cathode_recycling']
                formatted['recycling_statistics'] = {
                    'total_cathode_materials': cathode_stats['total_cathode'],
                    'optimized_cathode_materials': cathode_stats['optimized_cathode'],
                    'avg_recycle_ratio': f"{cathode_stats['avg_recycle_ratio'] * 100:.1f}%",
                    'avg_low_carbon_ratio': f"{cathode_stats['avg_low_carbon_ratio'] * 100:.1f}%"
                }
        
        return formatted
    
    def _get_material_category(self, material_name: str) -> str:
        """자재 카테고리 반환"""
        # 자재명으로 시나리오 데이터프레임에서 자재품목 찾기
        for idx, row in self.scenario_df.iterrows():
            if row['자재명'] == material_name and '자재품목' in row:
                return row['자재품목']
        return "기타"
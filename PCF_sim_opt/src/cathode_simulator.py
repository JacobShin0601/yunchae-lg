import json
import os
import sys

# FileOperations 사용을 위해 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from src.utils.file_operations import FileOperations, FileLoadError
import copy
from typing import Dict, Any
from .cathode_preprocessor import CathodePreprocessor
import pandas as pd


class CathodeSimulator:
    """
    Cathode 시뮬레이터 클래스
    cathode_coef_table.json 파일을 불러와서 계수 데이터를 관리합니다.
    """
    
    def __init__(self, json_file_path: str = None, recycle_ratio_path: str = None, verbose: bool = True, user_id: str = None):
        """
        CathodeSimulator를 초기화합니다.
        
        Args:
            json_file_path (str): 계수 데이터 JSON 파일 경로. None인 경우 기본 경로 사용
            recycle_ratio_path (str): 재활용 비중 데이터 JSON 파일 경로. None인 경우 기본 경로 사용
            verbose (bool): 상세 로그 출력 여부. 기본값은 True
            user_id (str): 사용자 ID. 사용자별 작업공간 사용시 필요
        """
        self.verbose = verbose
        self.user_id = user_id
        
        # 파일 경로 설정 - FileOperations를 통해 자동으로 사용자별 파일 생성
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if json_file_path is None:
            json_file_path = "stable_var/cathode_coef_table.json"
        
        if recycle_ratio_path is None:
            recycle_ratio_path = "input/recycle_material_ratio.json"
        
        self.json_file_path = json_file_path
        self.recycle_ratio_path = recycle_ratio_path
        
        # 데이터 로드
        self._print("🔧 1단계: CathodeSimulator 초기화", level="info")
        
        self.coefficient_data = self._load_coefficient_data()
        self.recycle_ratio_data = self._load_recycle_ratio_data()
        self.recycle_impact_data = self._load_recycle_impact_data()
        self.low_carb_metal_data = self._load_low_carb_metal_data()
        
        # 재활용 비율 설정 (recycle_material_ratio.json의 평균값)
        if self.recycle_ratio_data:
            self.recycling_ratio = sum(self.recycle_ratio_data.values()) / len(self.recycle_ratio_data)
            self._print(f"📊 재활용 비율 설정: {self.recycling_ratio:.3f} ({self.recycling_ratio*100:.1f}%)", level="info")
        else:
            self.recycling_ratio = 0.0
            self._print("⚠️ 재활용 비율 데이터가 없어 0%로 설정", level="warning")
        
        # 저탄소메탈 비율 설정 (low_carb_metal.json의 비중 평균값을 백분율로 변환)
        if self.low_carb_metal_data and '비중' in self.low_carb_metal_data:
            low_carb_weights = self.low_carb_metal_data['비중']
            self.low_carb_metal_ratio = sum(low_carb_weights.values()) / len(low_carb_weights) / 100.0  # 백분율을 비율로 변환
            self._print(f"📊 저탄소메탈 비율 설정: {self.low_carb_metal_ratio:.3f} ({self.low_carb_metal_ratio*100:.1f}%)", level="info")
        else:
            self.low_carb_metal_ratio = 0.0
            self._print("⚠️ 저탄소메탈 비율 데이터가 없어 0%로 설정", level="warning")
        
        # CathodePreprocessor 초기화 - user_id 전달
        self.preprocessor = CathodePreprocessor(verbose=self.verbose, user_id=self.user_id)
        
        # 데이터 업데이트
        self._print("🔧 2단계: 데이터 업데이트", level="info")
        self._update_recycle_ratios()
        self._update_raw_material_requirements()
        
        self._print("✅ CathodeSimulator 초기화 완료", level="info")
    
    def _print(self, *args, level="info", **kwargs):
        """
        로그 레벨에 따라 출력 제어 (info: 주요 단계/결과, debug: 상세/반복, False: 출력 없음)
        verbose == "debug"면 모두 출력, True면 info만, False면 출력 없음
        """
        if self.verbose == "debug":
            print(*args, **kwargs)
        elif self.verbose and level == "info":
            print(*args, **kwargs)
        elif level in ["warning", "error"]:
            print(*args, **kwargs)
    
    def update_electricity_emission_factor(self, site: str = 'before') -> Dict[str, Any]:
        """
        사이트별 전력 배출계수를 업데이트합니다.
        
        Args:
            site (str): 'before' 또는 'after'. 기본값은 'before'
            
        Returns:
            Dict[str, Any]: 업데이트된 데이터
        """
        try:
            self._print(f"🔧 3단계: 전력 배출계수 업데이트 (사이트: {site})", level="info")
            
            # 사이트 정보 로드 (사용자별 파일 사용)
            self._print(f"        📊 1단계: 사이트 정보 로드", level="debug")
            try:
                from src.utils.file_operations import FileOperations
                current_dir = os.path.dirname(os.path.abspath(__file__))
                site_file_path = os.path.join(current_dir, "..", "input", "cathode_site.json")
                site_data = FileOperations.load_json(site_file_path, default={'CAM': {'before': '중국', 'after': '한국'}, 'pCAM': {'before': '중국', 'after': '한국'}}, user_id=self.user_id)
                self._print(f"          - 사이트 정보 로드 성공: {site_data}", level="debug")
            except Exception as e:
                self._print(f"          ❌ 사이트 정보 로드 실패: {e}", level="error")
                return None
            
            # 계수 데이터 복사
            self._print(f"        📊 2단계: 계수 데이터 복사", level="debug")
            updated_data = copy.deepcopy(self.coefficient_data)
            self._print(f"          - 계수 데이터 복사 완료", level="debug")
            self._print(f"          - coefficient_data 키: {list(self.coefficient_data.keys())}", level="debug")
            
            # CAM/pCAM 사이트 정보 추출
            cam_site = site_data.get('CAM', {}).get(site, '중국')
            pcam_site = site_data.get('pCAM', {}).get(site, '한국')
            
            self._print(f"          - CAM 사이트: {cam_site}", level="debug")
            self._print(f"          - pCAM 사이트: {pcam_site}", level="debug")
            
            # 전력 배출계수 업데이트
            self._print(f"        📊 3단계: 전력 배출계수 업데이트", level="debug")
            
            # Energy(Tier-1) 전력 배출계수 업데이트
            if 'Energy(Tier-1)' in updated_data and '전력' in updated_data['Energy(Tier-1)']:
                tier1_electricity_factor = self._get_electricity_emission_factor(cam_site)
                updated_data['Energy(Tier-1)']['전력']['배출계수'] = tier1_electricity_factor
                self._print(f"          - Energy(Tier-1) 전력 배출계수: {tier1_electricity_factor}", level="info")
            else:
                self._print(f"          ❌ Energy(Tier-1) 또는 전력 데이터가 없습니다.", level="warning")
                if 'Energy(Tier-1)' in updated_data:
                    self._print(f"          - Energy(Tier-1) 키: {list(updated_data['Energy(Tier-1)'].keys())}", level="debug")
            
            # Energy(Tier-2) 전력 배출계수 업데이트
            if 'Energy(Tier-2)' in updated_data and '전력' in updated_data['Energy(Tier-2)']:
                tier2_electricity_factor = self._get_electricity_emission_factor(pcam_site)
                updated_data['Energy(Tier-2)']['전력']['배출계수'] = tier2_electricity_factor
                self._print(f"          - Energy(Tier-2) 전력 배출계수: {tier2_electricity_factor}", level="info")
            else:
                self._print(f"          ❌ Energy(Tier-2) 또는 전력 데이터가 없습니다.", level="warning")
                if 'Energy(Tier-2)' in updated_data:
                    self._print(f"          - Energy(Tier-2) 키: {list(updated_data['Energy(Tier-2)'].keys())}", level="debug")

            # Before/After 비교 로그 추가 (site='after'일 때만)
            if site == 'after':
                self._print(f"        📊 Before/After 전력 배출계수 비교", level="info")

                # Before 사이트 정보 로드
                cam_before_site = site_data.get('CAM', {}).get('before', '중국')
                pcam_before_site = site_data.get('pCAM', {}).get('before', '한국')

                # Before 배출계수 계산
                tier1_before_factor = self._get_electricity_emission_factor(cam_before_site)
                tier2_before_factor = self._get_electricity_emission_factor(pcam_before_site)

                # 변화율 계산
                tier1_change = ((tier1_before_factor - tier1_electricity_factor) / tier1_before_factor) * 100 if tier1_before_factor > 0 else 0
                tier2_change = ((tier2_before_factor - tier2_electricity_factor) / tier2_before_factor) * 100 if tier2_before_factor > 0 else 0
                tier1_diff = tier1_before_factor - tier1_electricity_factor
                tier2_diff = tier2_before_factor - tier2_electricity_factor

                self._print(f"        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", level="info")
                self._print(f"        🏭 CAM (Tier-1) 생산지 변경:", level="info")
                self._print(f"          • Before: {cam_before_site} (전력 배출계수: {tier1_before_factor:.4f} kgCO2eq/kWh)", level="info")
                self._print(f"          • After:  {cam_site} (전력 배출계수: {tier1_electricity_factor:.4f} kgCO2eq/kWh)", level="info")
                self._print(f"          • 변화:   {tier1_change:+.2f}% (↓ {tier1_diff:.4f} kgCO2eq/kWh)" if tier1_diff > 0 else f"          • 변화:   {tier1_change:+.2f}% (↑ {-tier1_diff:.4f} kgCO2eq/kWh)", level="info")
                self._print(f"        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", level="info")
                self._print(f"        🏭 pCAM (Tier-2) 생산지 변경:", level="info")
                self._print(f"          • Before: {pcam_before_site} (전력 배출계수: {tier2_before_factor:.4f} kgCO2eq/kWh)", level="info")
                self._print(f"          • After:  {pcam_site} (전력 배출계수: {tier2_electricity_factor:.4f} kgCO2eq/kWh)", level="info")
                self._print(f"          • 변화:   {tier2_change:+.2f}% (↓ {tier2_diff:.4f} kgCO2eq/kWh)" if tier2_diff > 0 else f"          • 변화:   {tier2_change:+.2f}% (↑ {-tier2_diff:.4f} kgCO2eq/kWh)", level="info")
                self._print(f"        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", level="info")

            self._print(f"        ✅ 전력 배출계수 업데이트 완료", level="info")

            return updated_data
            
        except Exception as e:
            self._print(f"❌ 전력 배출계수 업데이트 중 오류 발생: {e}", level="error")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_carbon_emission(self, updated_data: Dict[str, Any], baseline_emission: float = None) -> Dict[str, Any]:
        """
        탄소배출량을 계산합니다.
        
        Args:
            updated_data (Dict[str, Any]): 업데이트된 계수 데이터
            baseline_emission (float): 기준 배출량 (before 사이트의 총 배출량)
            
        Returns:
            Dict[str, Any]: 탄소배출량 계산 결과 (JSON 형태)
        """
        result = {
            "원재료": {},
            "Energy(Tier-1)": {},
            "Energy(Tier-2)": {},
            "총_배출량": 0,
            "카테고리별_기여도": {},
            "감축량": 0,
            "감축률": 0
        }
        
        total_emission = 0
        category_emissions = {}
        
        self._print("🧮 4단계: 탄소배출량 계산", level="info")
        
        # 각 카테고리별로 계산 (updated_data 사용)
        for category in ["원재료", "Energy(Tier-1)", "Energy(Tier-2)"]:
            if category in updated_data:
                category_data = updated_data[category]
                category_total = 0
                
                self._print(f"  📊 {category} 계산 중...", level="debug")
                
                # Energy Tier의 경우 전력과 LNG 모두 계산하되, 기여도는 전력만 사용
                if category in ["Energy(Tier-1)", "Energy(Tier-2)"]:
                    category_total = 0
                    electricity_emission = 0
                    
                    # 전력 계산
                    if '전력' in category_data:
                        item_data = category_data['전력']
                        requirement = item_data.get("소요량", 0)
                        emission_factor = item_data.get("배출계수", 0)
                        emission = requirement * emission_factor
                        electricity_emission = emission
                        
                        self._print(f"    📈 전력: {requirement:.6f} × {emission_factor:.6f} = {emission:.6f} kg CO2e", level="debug")
                        
                        # 결과에 추가
                        result[category]['전력'] = {
                            "소요량": requirement,
                            "배출계수": emission_factor,
                            "탄소배출량(kg_CO2e)": emission
                        }
                        
                        category_total += emission
                    
                    # LNG 계산 (총 배출량에 포함)
                    if 'LNG' in category_data:
                        item_data = category_data['LNG']
                        requirement = item_data.get("소요량", 0)
                        emission_factor = item_data.get("배출계수", 0)
                        emission = requirement * emission_factor
                        
                        self._print(f"    📈 LNG: {requirement:.6f} × {emission_factor:.6f} = {emission:.6f} kg CO2e", level="debug")
                        
                        # 결과에 추가
                        result[category]['LNG'] = {
                            "소요량": requirement,
                            "배출계수": emission_factor,
                            "탄소배출량(kg_CO2e)": emission
                        }
                        
                        category_total += emission
                    
                    self._print(f"  📊 {category} 총 배출량 (전력+LNG): {category_total:.6f} kg CO2e", level="info")
                    
                    # 전력 기여도 계산을 위해 별도 저장
                    result[category]['_전력_배출량'] = electricity_emission
                else:
                    # 원재료는 모든 항목 계산
                    # 배출량 상위 항목을 추적하기 위한 리스트
                    material_emissions = []
                    
                    for item_name, item_data in category_data.items():
                        requirement = item_data.get("소요량", 0)
                        emission_factor = item_data.get("배출계수", 0)
                        emission = requirement * emission_factor
                        
                        self._print(f"    📈 {item_name}: {requirement:.6f} × {emission_factor:.6f} = {emission:.6f} kg CO2e", level="debug")
                        
                        # 결과에 추가
                        result[category][item_name] = {
                            "소요량": requirement,
                            "배출계수": emission_factor,
                            "탄소배출량(kg_CO2e)": emission
                        }
                        
                        # 배출량 정보를 리스트에 추가
                        material_emissions.append((item_name, emission))
                        
                        category_total += emission
                    
                    # 배출량이 높은 순으로 정렬
                    material_emissions.sort(key=lambda x: x[1], reverse=True)
                    
                    # 상위 5개 재료의 배출량 정보를 로그로 출력
                    self._print(f"  📊 {category} 배출량 상위 항목:", level="info")
                    for i, (item_name, emission) in enumerate(material_emissions[:5]):
                        percentage = (emission / category_total * 100) if category_total > 0 else 0
                        self._print(f"    {i+1}. {item_name}: {emission:.6f} kg CO2e ({percentage:.2f}%)", level="info")
                    
                    self._print(f"  📊 {category} 총 배출량: {category_total:.6f} kg CO2e", level="info")
                
                category_emissions[category] = category_total
                total_emission += category_total
        
        result["총_배출량"] = total_emission
        
        # 카테고리별 기여도 계산
        for category, emission in category_emissions.items():
            contribution = (emission / total_emission * 100) if total_emission > 0 else 0
            result["카테고리별_기여도"][category] = contribution
            
            # 각 아이템의 기여도도 계산
            if category in result:
                for item_name, item_data in result[category].items():
                    # _전력_배출량은 특수 필드이므로 건너뛰기
                    if item_name == '_전력_배출량':
                        continue
                    
                    # item_data가 딕셔너리인지 확인
                    if isinstance(item_data, dict) and "탄소배출량(kg_CO2e)" in item_data:
                        item_emission = item_data["탄소배출량(kg_CO2e)"]
                        item_contribution = (item_emission / total_emission * 100) if total_emission > 0 else 0
                        item_data["기여도(%)"] = item_contribution
        
        # 감축량 계산 (baseline_emission이 제공된 경우)
        if baseline_emission is not None:
            reduction = baseline_emission - total_emission
            reduction_rate = (reduction / baseline_emission * 100) if baseline_emission > 0 else 0
            
            result["감축량"] = reduction
            result["감축률"] = reduction_rate
            
            self._print(f"📉 감축량: {reduction:.6f} kg CO2e ({reduction_rate:.2f}%)", level="debug")
        
        self._print(f"✅ 탄소배출량 계산 완료: 총 {total_emission:.6f} kg CO2e", level="info")
        
        return result
    
    def _load_coefficient_data(self) -> Dict[str, Any]:
        """
        JSON 파일에서 계수 데이터를 불러옵니다.
        
        Returns:
            Dict[str, Any]: 불러온 계수 데이터
            
        Raises:
            FileNotFoundError: JSON 파일을 찾을 수 없는 경우
            json.JSONDecodeError: JSON 파싱 오류가 발생한 경우
        """
        try:
            data = FileOperations.load_json(self.json_file_path, user_id=self.user_id)
            self._print(f"계수 데이터를 성공적으로 불러왔습니다: {self.json_file_path}", level="info")
            return data
        except FileLoadError as e:
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {e}")
    
    def _load_recycle_ratio_data(self) -> Dict[str, float]:
        """
        재활용 비중 JSON 파일을 불러옵니다.
        
        Returns:
            Dict[str, float]: 재활용 비중 데이터
            
        Raises:
            FileNotFoundError: JSON 파일을 찾을 수 없는 경우
            json.JSONDecodeError: JSON 파싱 오류가 발생한 경우
        """
        try:
            data = FileOperations.load_json(self.recycle_ratio_path, user_id=self.user_id)
            self._print(f"재활용 비중 데이터를 성공적으로 불러왔습니다: {self.recycle_ratio_path}", level="info")
            return data
        except FileLoadError:
            if self.verbose:
                print(f"⚠️ 재활용 비중 JSON 파일을 찾을 수 없습니다: {self.recycle_ratio_path}")
            return {}
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"⚠️ 재활용 비중 JSON 파싱 오류: {e}")
            return {}

    def _load_recycle_impact_data(self) -> Dict[str, Any]:
        """
        재활용 재료 영향도 JSON 파일을 불러옵니다.
        
        Returns:
            Dict[str, Any]: 재활용 재료 영향도 데이터
            
        Raises:
            FileNotFoundError: JSON 파일을 찾을 수 없는 경우
            json.JSONDecodeError: JSON 파싱 오류가 발생한 경우
        """
        try:
            # FileOperations를 사용하여 사용자별 파일 로드
            data = FileOperations.load_json("stable_var/recycle_material_impact.json", user_id=self.user_id)
            self._print(f"재활용 재료 영향도 데이터를 성공적으로 불러왔습니다: stable_var/recycle_material_impact.json", level="info")
            return data
        except FileLoadError:
            if self.verbose:
                print(f"⚠️ 재활용 재료 영향도 JSON 파일을 찾을 수 없습니다: stable_var/recycle_material_impact.json")
            return {"신재": 1, "재활용재": {"Ni": 0.05, "Co": 0.15, "Li": 0.2}}

    def _load_low_carb_metal_data(self) -> Dict[str, Any]:
        """
        저탄소메탈 JSON 파일을 불러옵니다.
        
        Returns:
            Dict[str, Any]: 저탄소메탈 데이터 (비중과 배출계수 포함)
            
        Raises:
            FileNotFoundError: JSON 파일을 찾을 수 없는 경우
            json.JSONDecodeError: JSON 파싱 오류가 발생한 경우
        """
        try:
            # FileOperations를 사용하여 사용자별 파일 로드
            data = FileOperations.load_json("input/low_carb_metal.json", user_id=self.user_id)
            self._print(f"저탄소메탈 데이터를 성공적으로 불러왔습니다: input/low_carb_metal.json", level="info")
            return data
        except FileLoadError:
            self._print(f"⚠️ 저탄소메탈 JSON 파일을 찾을 수 없습니다: input/low_carb_metal.json", level="warning")
            return {"비중": {"Ni": 5.0, "Co": 5.0, "Li": 20.0}, "배출계수": {"Ni": 2.0, "Co": 15.0, "Li": 9.0}}
        except json.JSONDecodeError as e:
            self._print(f"⚠️ 저탄소메탈 JSON 파싱 오류: {e}", level="warning")
            return {"비중": {"Ni": 5.0, "Co": 5.0, "Li": 20.0}, "배출계수": {"Ni": 2.0, "Co": 15.0, "Li": 9.0}}

    def _get_primary_cathode_site(self) -> str:
        """
        PCF 원본 테이블에서 양극재의 지역 정보를 가져와 국가명으로 변환합니다.
        
        Returns:
            str: 양극재 생산지 국가명 (예: "한국")
        """
        try:
            import pandas as pd
            
            # 현재 파일의 상대 경로로 파일들 위치 설정 (사용자별)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 사용자별 파일 경로 처리 - FileOperations 사용
            if self.user_id:
                pcf_table_path = os.path.join(current_dir, "..", "data", self.user_id, "pcf_original_table_updated.csv")
                
                # 사용자별 파일이 없으면 기본 파일 사용
                if not os.path.exists(pcf_table_path):
                    pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_updated.csv")
                    if not os.path.exists(pcf_table_path):
                        pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_sample.csv")
            else:
                pcf_table_path = os.path.join(current_dir, "..", "data", "pcf_original_table_updated.csv")
            
            # PCF 테이블 읽기
            df = pd.read_csv(pcf_table_path, encoding='utf-8-sig')
            
            # 양극재인 첫 번째 데이터의 지역 정보 추출
            cathode_data = df[df['자재품목'] == '양극재']
            if cathode_data.empty:
                self._print("⚠️ PCF 테이블에서 양극재 데이터를 찾을 수 없습니다.", level="warning")
                return "한국"  # 기본값
            
            # 첫 번째 양극재의 지역 코드
            region_code = cathode_data.iloc[0]['지역']
            self._print(f"📍 PCF 테이블에서 발견된 양극재 지역 코드: {region_code}", level="info")
            
            # 국가 코드 매핑 파일 읽기 - FileOperations 사용
            national_code_data = FileOperations.load_json("stable_var/cathode_national_code.json", user_id=self.user_id)
            
            # 지역 코드를 국가명으로 변환
            country_name = national_code_data.get("national_code", {}).get(region_code, "미분류")
            self._print(f"🌏 지역 코드 '{region_code}' → 국가명 '{country_name}'", level="info")

            # 매핑되지 않은 지역 코드에 대한 추가 로그
            if country_name == "미분류":
                self._print(f"⚠️ 지역 코드 '{region_code}'는 매핑 테이블에 없어 '미분류'로 설정됩니다.", level="warning")
            
            return country_name
            
        except FileNotFoundError as e:
            self._print(f"⚠️ 파일을 찾을 수 없습니다: {e}", level="warning")
            return "미분류"  # 기본값을 미분류로 변경
        except Exception as e:
            self._print(f"⚠️ 주 양극재 생산지 검색 중 오류: {e}", level="warning")
            return "미분류"  # 기본값을 미분류로 변경

    def get_default_cathode_site_config(self) -> Dict[str, Any]:
        """
        Primary logic을 사용하여 기본 양극재 생산지 설정을 생성합니다.
        
        Returns:
            Dict[str, Any]: 기본 양극재 생산지 설정
        """
        try:
            # Primary logic: PCF 테이블에서 before 사이트 감지
            primary_site = self._get_primary_cathode_site()
            
            # Fallback logic: 기존 cathode_site.json에서 after 사이트 읽기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            site_file_path = os.path.join(current_dir, "..", "input", "cathode_site.json")
            
            fallback_config = {
                "CAM": {"before": "중국", "after": "한국"},
                "pCAM": {"before": "한국", "after": "한국"}
            }
            
            try:
                from src.utils.file_operations import FileOperations
                existing_config = FileOperations.load_json(site_file_path, default=fallback_config, user_id=self.user_id)
                    
                # Primary logic으로 before 설정, 기존 파일에서 after 설정 가져오기
                default_config = {
                    "CAM": {
                        "before": primary_site,  # Primary logic
                        "after": existing_config.get("CAM", {}).get("after", fallback_config["CAM"]["after"])
                    },
                    "pCAM": {
                        "before": primary_site,  # Primary logic
                        "after": existing_config.get("pCAM", {}).get("after", fallback_config["pCAM"]["after"])
                    }
                }
                
                self._print(f"🎯 Primary logic 적용: before = '{primary_site}'", level="info")
                self._print(f"📁 Fallback logic 적용: after 값은 기존 설정 사용", level="info")
                
                return default_config
                
            except FileNotFoundError:
                # cathode_site.json이 없는 경우 새로 생성
                default_config = {
                    "CAM": {
                        "before": primary_site,  # Primary logic
                        "after": fallback_config["CAM"]["after"]
                    },
                    "pCAM": {
                        "before": primary_site,  # Primary logic
                        "after": fallback_config["pCAM"]["after"]
                    }
                }
                
                self._print(f"🎯 Primary logic 적용 (신규): before = '{primary_site}'", level="info")
                self._print(f"📁 Fallback 기본값 적용: after 값은 기본값 사용", level="info")
                
                return default_config
                
        except Exception as e:
            self._print(f"⚠️ 기본 양극재 생산지 설정 생성 중 오류: {e}", level="warning")
            return fallback_config

    def _update_recycle_ratios(self):
        """
        재활용 비중 데이터를 사용하여 원재료의 재활용_비중을 업데이트합니다.
        """
        if not self.recycle_ratio_data:
            if self.verbose:
                print("⚠️ 재활용 비중 데이터가 없어 업데이트를 건너뜁니다.")
            return
        
        self._print("🔧 2-1단계: 재활용 비중 업데이트", level="info")
        
        # 원재료 데이터 가져오기
        raw_materials = self.coefficient_data.get("원재료", {})
        
        # 업데이트 내역을 저장할 리스트
        self.recycle_update_log = []
        
        # 재활용 비중 데이터의 각 원소에 대해 매칭되는 원재료 찾기
        updated_count = 0
        for element, ratio in self.recycle_ratio_data.items():
            # 원재료에서 해당 원소가 포함된 항목 찾기
            matched_materials = []
            for material_name, material_data in raw_materials.items():
                if element in material_name:
                    matched_materials.append(material_name)
            
            # 매칭된 원재료들의 재활용_비중 업데이트
            for material_name in matched_materials:
                old_ratio = raw_materials[material_name].get("재활용_비중", 0)
                raw_materials[material_name]["재활용_비중"] = ratio
                
                # 업데이트 로그에 추가
                self.recycle_update_log.append({
                    "element": element,
                    "material": material_name,
                    "old_ratio": old_ratio,
                    "new_ratio": ratio
                })
                
                if self.verbose:
                    print(f"  📊 {material_name} ({element} 포함): {old_ratio} → {ratio}")
                updated_count += 1
        
        if updated_count > 0:
            self._print(f"✅ {updated_count}개 원재료의 재활용 비중이 업데이트되었습니다.", level="info")
        else:
            self._print("⚠️ 업데이트된 원재료가 없습니다.", level="warning")
    
    def _update_raw_material_requirements(self):
        """
        CathodePreprocessor에서 tier1과 tier2 데이터를 가져와서 원재료의 소요량을 업데이트합니다.
        """
        self._print("🔧 2-2단계: 원재료 소요량 업데이트", level="info")
        
        # 원재료 데이터 가져오기
        raw_materials = self.coefficient_data.get("원재료", {})
        
        # 업데이트 내역을 저장할 리스트
        update_log = []
        
        # CathodePreprocessor의 계산을 먼저 실행
        self._print("  🔄 CathodePreprocessor 계산 실행 중...", level="debug")
        
        # Tier1 데이터 계산 및 업데이트
        tier1_data = self.preprocessor.update_cathode_tier1_input(suppress_logs=True)
        if tier1_data:
            self._print("    ✅ Tier1 데이터 계산 완료", level="info")
        else:
            self._print("    ❌ Tier1 데이터 계산 실패", level="error")
        
        # Tier2 데이터 계산 및 업데이트
        tier2_data = self.preprocessor.update_cathode_tier2_input(suppress_logs=True)
        if tier2_data:
            self._print("    ✅ Tier2 데이터 계산 완료", level="info")
        else:
            self._print("    ❌ Tier2 데이터 계산 실패", level="error")
        
        # Tier2에서 얻은 질량들 (NiSO4.6H2O, CoSO4.7H2O, MnSO4.H2O, Al(OH)3, NaOH)
        if tier2_data:
            self._print("  📊 Tier2 데이터에서 질량 가져오기...", level="debug")
            
            # Tier2 매핑: 화합물명 → 원재료명
            tier2_mapping = {
                "NiSO4.6H2O": "NiSO4",
                "CoSO4.7H2O": "CoSO4", 
                "MnSO4.H2O": "MnSO4",
                "Al(OH)3": "Al(OH3)",
                "NaOH": "NaOH"
            }
            
            for compound_name, material_name in tier2_mapping.items():
                if compound_name in tier2_data and material_name in raw_materials:
                    mass = tier2_data[compound_name].get("질량(kg)", 0)
                    old_requirement = raw_materials[material_name].get("소요량", 0)
                    raw_materials[material_name]["소요량"] = mass
                    
                    # 업데이트 로그에 추가
                    update_log.append({
                        "source": "Tier2",
                        "compound": compound_name,
                        "material": material_name,
                        "old_value": old_requirement,
                        "new_value": mass,
                        "unit": "kg"
                    })
                    
                    self._print(f"    📊 {compound_name} → {material_name}: {old_requirement} → {mass:.6f} kg", level="debug")
        
        # Tier1에서 얻은 LiOH.H2O 질량
        if tier1_data and "LiOH.H2O" in tier1_data and "LiOH.H2O" in raw_materials:
            lioh_mass = tier1_data["LiOH.H2O"].get("질량(kg)", 0)
            old_requirement = raw_materials["LiOH.H2O"].get("소요량", 0)
            raw_materials["LiOH.H2O"]["소요량"] = lioh_mass
            
            # 업데이트 로그에 추가
            update_log.append({
                "source": "Tier1",
                "compound": "LiOH.H2O",
                "material": "LiOH.H2O",
                "old_value": old_requirement,
                "new_value": lioh_mass,
                "unit": "kg"
            })
            
            self._print(f"  📊 LiOH.H2O (Tier1): {old_requirement} → {lioh_mass:.6f} kg", level="debug")
        
        # Energy Tier 데이터도 업데이트 (cathode ratio 변경이 energy 소요량에도 영향을 줄 수 있음)
        if tier1_data:
            # Energy(Tier-1) 데이터 업데이트
            energy_tier1 = self.coefficient_data.get("Energy(Tier-1)", {})
            if energy_tier1:
                # CAM 질량이 변경되면 Tier1 energy 소요량도 비례적으로 변경
                if "CAM(=Li(NCM)O2)" in tier1_data:
                    cam_mass = tier1_data["CAM(=Li(NCM)O2)"].get("질량(kg)", 0)
                    original_cam_mass = 1.0  # 기준 질량 (1kg 기준으로 계수가 설정되어 있다고 가정)
                    
                    # 주석 처리: 전력과 LNG 소요량을 CAM 질량 비율에 따라 조정하지 않음
                    # 소요량은 원래 값 그대로 유지
                    if cam_mass > 0 and original_cam_mass > 0:
                        # mass_ratio 정보만 로그에 기록
                        mass_ratio = cam_mass / original_cam_mass
                        
                        for energy_type in ["전력", "LNG"]:
                            if energy_type in energy_tier1:
                                old_requirement = energy_tier1[energy_type].get("소요량", 0)
                                # energy_tier1[energy_type]["소요량"] = old_requirement * mass_ratio  # 이 부분 주석 처리
                                
                                update_log.append({
                                    "source": "Tier1_Energy",
                                    "compound": f"CAM_mass_ratio_{mass_ratio:.6f}",
                                    "material": f"Energy(Tier-1)_{energy_type}",
                                    "old_value": old_requirement,
                                    "new_value": old_requirement,  # 값 변경 없음
                                    "unit": "MWh" if energy_type == "전력" else "MJ"
                                })
                                
                                self._print(f"  📊 Energy(Tier-1) {energy_type}: {old_requirement:.6f} (소요량 그대로 유지)", level="debug")
        
        if tier2_data:
            # Energy(Tier-2) 데이터 업데이트
            energy_tier2 = self.coefficient_data.get("Energy(Tier-2)", {})
            if energy_tier2:
                # pCAM 질량이 변경되면 Tier2 energy 소요량도 비례적으로 변경
                if "pCAM(=M(OH)2)" in tier2_data:
                    pcam_mass = tier2_data["pCAM(=M(OH)2)"].get("질량(kg)", 0)
                    original_pcam_mass = 1.0  # 기준 질량 (1kg 기준으로 계수가 설정되어 있다고 가정)
                    
                    # 주석 처리: 전력과 LNG 소요량을 pCAM 질량 비율에 따라 조정하지 않음
                    # 소요량은 원래 값 그대로 유지
                    if pcam_mass > 0 and original_pcam_mass > 0:
                        # mass_ratio 정보만 로그에 기록
                        mass_ratio = pcam_mass / original_pcam_mass
                        
                        for energy_type in ["전력", "LNG"]:
                            if energy_type in energy_tier2:
                                old_requirement = energy_tier2[energy_type].get("소요량", 0)
                                # energy_tier2[energy_type]["소요량"] = old_requirement * mass_ratio  # 이 부분 주석 처리
                                
                                update_log.append({
                                    "source": "Tier2_Energy",
                                    "compound": f"pCAM_mass_ratio_{mass_ratio:.6f}",
                                    "material": f"Energy(Tier-2)_{energy_type}",
                                    "old_value": old_requirement,
                                    "new_value": old_requirement,  # 값 변경 없음
                                    "unit": "MWh" if energy_type == "전력" else "MJ"
                                })
                                
                                self._print(f"  📊 Energy(Tier-2) {energy_type}: {old_requirement:.6f} (소요량 그대로 유지)", level="debug")
        
        # 업데이트된 tier1/tier2 데이터를 파일에 저장 (cathode ratio 변경사항 반영)
        try:
            if tier1_data:
                # cathode_tier1_input.json 업데이트 및 저장
                FileOperations.save_json("stable_var/cathode_tier1_input.json", tier1_data, user_id=self.user_id)
                self._print(f"  📁 업데이트된 Tier1 데이터 저장됨", level="debug")
            
            if tier2_data:
                # cathode_tier2_input.json 업데이트 및 저장  
                FileOperations.save_json("stable_var/cathode_tier2_input.json", tier2_data, user_id=self.user_id)
                self._print(f"  📁 업데이트된 Tier2 데이터 저장됨", level="debug")
        except Exception as e:
            self._print(f"⚠️ Tier 데이터 저장 중 오류: {e}", level="warning")
        
        # 업데이트 로그를 인스턴스 변수에 저장
        self.requirement_update_log = update_log
        
        self._print(f"✅ 원재료 소요량 업데이트 완료 (총 {len(update_log)}개 항목 업데이트)", level="info")
        
        # 업데이트 로그 출력
        if update_log:
            self._print("\n📋 소요량 업데이트 내역:", level="debug")
            for log_entry in update_log:
                self._print(f"  • {log_entry['source']}: {log_entry['compound']} → {log_entry['material']}: "
                          f"{log_entry['old_value']:.6f} → {log_entry['new_value']:.6f} {log_entry['unit']}", level="debug")
        else:
            self._print("  ⚠️ 업데이트된 항목이 없습니다.", level="warning")
    
    def get_coefficient_data(self) -> Dict[str, Any]:
        """
        저장된 계수 데이터를 반환합니다.
        
        Returns:
            Dict[str, Any]: 계수 데이터
        """
        return self.coefficient_data
    
    def get_raw_materials(self) -> Dict[str, Any]:
        """
        원재료 데이터를 반환합니다.
        
        Returns:
            Dict[str, Any]: 원재료 데이터
        """
        return self.coefficient_data.get("원재료", {})
    
    def get_energy_tier1(self) -> Dict[str, Any]:
        """
        Energy(Tier-1) 데이터를 반환합니다.
        
        Returns:
            Dict[str, Any]: Energy(Tier-1) 데이터
        """
        return self.coefficient_data.get("Energy(Tier-1)", {})
    
    def get_energy_tier2(self) -> Dict[str, Any]:
        """
        Energy(Tier-2) 데이터를 반환합니다.
        
        Returns:
            Dict[str, Any]: Energy(Tier-2) 데이터
        """
        return self.coefficient_data.get("Energy(Tier-2)", {})
    
    def get_recycle_ratio_data(self) -> Dict[str, float]:
        """
        재활용 비중 데이터를 반환합니다.
        
        Returns:
            Dict[str, float]: 재활용 비중 데이터
        """
        return self.recycle_ratio_data
    
    def get_preprocessor(self) -> CathodePreprocessor:
        """
        CathodePreprocessor 인스턴스를 반환합니다.
        
        Returns:
            CathodePreprocessor: 전처리기 인스턴스
        """
        return self.preprocessor
    
    def get_requirement_update_log(self) -> list:
        """
        소요량 업데이트 로그를 반환합니다.
        
        Returns:
            list: 업데이트 로그 리스트
        """
        return getattr(self, 'requirement_update_log', [])
    
    def get_recycle_update_log(self) -> list:
        """
        재활용 비중 업데이트 로그를 반환합니다.
        
        Returns:
            list: 업데이트 로그 리스트
        """
        return getattr(self, 'recycle_update_log', [])
    

    
    def reload_data(self):
        """
        JSON 파일에서 데이터를 다시 불러옵니다.
        """
        self.coefficient_data = self._load_coefficient_data()
        self.recycle_ratio_data = self._load_recycle_ratio_data()
        self.recycle_impact_data = self._load_recycle_impact_data()
        self.low_carb_metal_data = self._load_low_carb_metal_data()
        
        # 저탄소메탈 비율 재설정
        if self.low_carb_metal_data and '비중' in self.low_carb_metal_data:
            low_carb_weights = self.low_carb_metal_data['비중']
            self.low_carb_metal_ratio = sum(low_carb_weights.values()) / len(low_carb_weights) / 100.0
            self._print(f"📊 저탄소메탈 비율 재설정: {self.low_carb_metal_ratio:.3f} ({self.low_carb_metal_ratio*100:.1f}%)", level="info")
        else:
            self.low_carb_metal_ratio = 0.0
            self._print("⚠️ 저탄소메탈 비율 데이터가 없어 0%로 설정", level="warning")
        
        self._update_recycle_ratios()
        self._update_raw_material_requirements()
        if self.verbose:
            print("계수 데이터, 재활용 비중, 저탄소메탈 데이터를 다시 불러왔습니다.")
    
    def print_summary(self):
        """
        불러온 데이터의 요약 정보를 출력합니다.
        """
        if self.verbose:
            print("=== CathodeSimulator 데이터 요약 ===")
        # if self.verbose:
        #     print(f"계수 JSON 파일 경로: {self.json_file_path}")
        # if self.verbose:
        #     print(f"재활용 비중 JSON 파일 경로: {self.recycle_ratio_path}")
        # if self.verbose:
        #     print(f"원재료 종류: {len(self.get_raw_materials())}개")
        # if self.verbose:
        #     print(f"Energy(Tier-1) 항목: {len(self.get_energy_tier1())}개")
        # if self.verbose:
        #     print(f"Energy(Tier-2) 항목: {len(self.get_energy_tier2())}개")
        if self.verbose:
            print(f"재활용 비중 원소: {len(self.recycle_ratio_data)}개")
        
        # 재활용 비중이 적용된 원재료 출력
        raw_materials = self.get_raw_materials()
        updated_materials = []
        for material_name, material_data in raw_materials.items():
            if material_data.get("재활용_비중", 0) > 0:
                updated_materials.append(f"{material_name}({material_data['재활용_비중']})")
        
        if updated_materials:
            self._print(f"재활용 비중 적용된 원재료: {', '.join(updated_materials)}", level="info")
        else:
            self._print("재활용 비중이 적용된 원재료가 없습니다.", level="info")
        
        # 소요량이 업데이트된 원재료 출력
        materials_with_requirements = []
        for material_name, material_data in raw_materials.items():
            if material_data.get("소요량", 0) > 0:
                materials_with_requirements.append(f"{material_name}({material_data['소요량']:.6f}kg)")
        
        if materials_with_requirements:
            self._print(f"소요량 업데이트된 원재료: {', '.join(materials_with_requirements)}", level="info")
        else:
            self._print("소요량이 업데이트된 원재료가 없습니다.", level="info")
        
        self._print("==================================", level="info")
    
    def print_recycle_ratio_details(self):
        """
        재활용 비중 적용 상세 정보를 출력합니다.
        """
        self._print("=== 재활용 비중 적용 상세 정보 ===", level="info")
        
        # 업데이트 로그가 있으면 실제 변화를 보여줌
        if hasattr(self, 'recycle_update_log') and self.recycle_update_log:
            self._print("\n📊 재활용 비중 업데이트 내역:", level="debug")
            for log_entry in self.recycle_update_log:
                element = log_entry['element']
                material = log_entry['material']
                old_ratio = log_entry['old_ratio']
                new_ratio = log_entry['new_ratio']
                
                self._print(f"🔍 {element} (재활용 비중: {new_ratio})", level="debug")
                self._print(f"  ✅ {material}: {old_ratio} → {new_ratio}", level="debug")
        else:
            # 기존 방식으로 출력 (업데이트 로그가 없는 경우)
            raw_materials = self.get_raw_materials()
            
            for element, ratio in self.recycle_ratio_data.items():
                self._print(f"\n🔍 {element} (재활용 비중: {ratio})", level="debug")
                matched_materials = []
                
                for material_name, material_data in raw_materials.items():
                    if element in material_name:
                        old_ratio = material_data.get("재활용_비중", 0)
                        matched_materials.append(f"{material_name}: {old_ratio} → {ratio}")
                
                if matched_materials:
                    for match in matched_materials:
                        self._print(f"  ✅ {match}", level="debug")
                else:
                    self._print(f"  ⚠️ {element}가 포함된 원재료를 찾을 수 없습니다.", level="warning")
        
        self._print("==================================", level="info")
    
    def print_raw_material_requirements(self):
        """
        원재료 소요량 상세 정보를 출력합니다.
        """
        self._print("=== 원재료 소요량 상세 정보 ===", level="info")
        
        raw_materials = self.get_raw_materials()
        
        self._print(f"{'원재료':<15} {'소요량(kg)':<12} {'재활용_비중':<12} {'배출계수':<10}", level="debug")
        self._print("-" * 50, level="debug")
        
        for material_name, material_data in raw_materials.items():
            requirement = material_data.get("소요량", 0)
            recycle_ratio = material_data.get("재활용_비중", 0)
            emission_factor = material_data.get("배출계수", 0)
            
            self._print(f"{material_name:<15} {requirement:<12.6f} {recycle_ratio:<12.3f} {emission_factor:<10.2f}", level="debug")
        
        self._print("=" * 50, level="debug")


    def _apply_recycling_ratio(self, base_data: Dict[str, Any], recycling_ratio: float) -> Dict[str, Any]:
        """
        재활용 비율을 적용하여 데이터를 수정합니다.
        
        Args:
            base_data (Dict[str, Any]): 기본 데이터
            recycling_ratio (float): 적용할 재활용 비율 (0~1)
            
        Returns:
            Dict[str, Any]: 재활용 비율이 적용된 데이터
        """
        recycled_data = copy.deepcopy(base_data)
        
        # 원재료 카테고리에만 재활용 비율 적용
        if '원재료' in recycled_data:
            for material_name, material_data in recycled_data['원재료'].items():
                # 재활용 가능한 원재료만 적용 (기존 재활용_비중이 있는 경우)
                if material_data.get('재활용_비중', 0) > 0:
                    # 재활용 비율을 업데이트
                    material_data['재활용_비중'] = recycling_ratio
                    
                    # 소요량 조정 (재활용 비율만큼 감소)
                    original_requirement = material_data.get('소요량', 0)
                    recycled_requirement = original_requirement * (1 - recycling_ratio)
                    material_data['소요량'] = recycled_requirement
                    
                    self._print(f"    ♻️ {material_name}: {original_requirement:.6f} → {recycled_requirement:.6f} kg (재활용 {recycling_ratio*100:.0f}%)", level="debug")
        
        return recycled_data

    def _apply_recycling_ratio_with_impact(self, base_data: Dict[str, Any], recycling_ratio: float) -> Dict[str, Any]:
        """
        재활용 비율과 재활용 재료 영향도를 적용하여 데이터를 수정합니다.
        
        Args:
            base_data (Dict[str, Any]): 기본 데이터
            recycling_ratio (float): 적용할 재활용 비율 (0~1)
            
        Returns:
            Dict[str, Any]: 재활용 비율과 영향도가 적용된 데이터
        """
        recycled_data = copy.deepcopy(base_data)
        
        # 원재료 카테고리에만 재활용 비율 적용
        if '원재료' in recycled_data:
            for material_name, material_data in recycled_data['원재료'].items():
                # 재활용 가능한 원재료만 적용 (기존 재활용_비중이 있는 경우)
                if material_data.get('재활용_비중', 0) > 0:
                    # 재활용 비율을 업데이트
                    material_data['재활용_비중'] = recycling_ratio
                    
                    # 원래 소요량과 배출계수
                    original_requirement = material_data.get('소요량', 0)
                    original_emission_factor = material_data.get('배출계수', 0)
                    
                    # 재활용 재료의 영향도 찾기
                    recycling_impact = self._get_recycling_impact_for_material(material_name)
                    
                    if recycling_impact is not None:
                        # 재활용 비율에 따른 소요량 조정
                        virgin_requirement = original_requirement * (1 - recycling_ratio)
                        recycled_requirement = original_requirement * recycling_ratio
                        
                        # 재활용 재료의 배출계수 조정 (영향도 적용)
                        recycled_emission_factor = original_emission_factor * recycling_impact
                        
                        # 총 탄소배출량 계산
                        virgin_emission = virgin_requirement * original_emission_factor
                        recycled_emission = recycled_requirement * recycled_emission_factor
                        total_emission = virgin_emission + recycled_emission
                        
                        # 새로운 배출계수 계산 (총 배출량 / 총 소요량)
                        new_emission_factor = total_emission / original_requirement if original_requirement > 0 else 0
                        
                        # 데이터 업데이트
                        material_data['소요량'] = original_requirement  # 총 소요량은 유지
                        material_data['배출계수'] = new_emission_factor
                        material_data['재활용_영향도'] = recycling_impact
                        
                        self._print(f"    ♻️ {material_name}: 재활용 {recycling_ratio*100:.0f}% 적용", level="debug")
                        self._print(f"      📊 신재: {virgin_requirement:.6f} kg × {original_emission_factor:.6f} = {virgin_emission:.6f} kg CO2e", level="debug")
                        self._print(f"      📊 재활용재: {recycled_requirement:.6f} kg × {recycled_emission_factor:.6f} = {recycled_emission:.6f} kg CO2e", level="debug")
                        self._print(f"      📊 총 배출량: {total_emission:.6f} kg CO2e", level="debug")
                        self._print(f"      📊 새로운 배출계수: {new_emission_factor:.6f}", level="debug")
                    else:
                        # 영향도 정보가 없는 경우 기존 방식 적용
                        recycled_requirement = original_requirement * (1 - recycling_ratio)
                        material_data['소요량'] = recycled_requirement
                        self._print(f"    ♻️ {material_name}: {original_requirement:.6f} → {recycled_requirement:.6f} kg (재활용 {recycling_ratio*100:.0f}%, 영향도 정보 없음)", level="debug")
        
        return recycled_data

    def _apply_three_way_material_split(self, base_data: Dict[str, Any], recycling_ratio: float, low_carb_ratio: float = None) -> Dict[str, Any]:
        """
        신재 + 재활용재 + 저탄소메탈의 세가지 구성으로 재료를 분할하여 데이터를 수정합니다.
        
        Args:
            base_data (Dict[str, Any]): 기본 데이터
            recycling_ratio (float): 적용할 재활용 비율 (0~1)
            low_carb_ratio (float): 적용할 저탄소메탈 비율 (0~1). None인 경우 재료별 개별 비율 사용
            
        Returns:
            Dict[str, Any]: 3가지 재료 구성이 적용된 데이터
        """
        processed_data = copy.deepcopy(base_data)
        
        # 원재료 카테고리에만 3원 분할 적용
        if '원재료' in processed_data:
            for material_name, material_data in processed_data['원재료'].items():
                # 재활용 가능한 원재료만 적용 (기존 재활용_비중이 있는 경우)
                if material_data.get('재활용_비중', 0) > 0:
                    # 원래 소요량과 배출계수
                    original_requirement = material_data.get('소요량', 0)
                    original_emission_factor = material_data.get('배출계수', 0)
                    
                    # 재활용 재료의 영향도 찾기
                    recycling_impact = self._get_recycling_impact_for_material(material_name)
                    
                    # 저탄소메탈 데이터 찾기
                    low_carb_data = self._get_low_carb_metal_data_for_material(material_name)
                    
                    # 저탄소메탈 비율 결정 (매개변수로 받은 값 우선, 없으면 재료별 개별 비율 사용)
                    material_low_carb_ratio = low_carb_ratio if low_carb_ratio is not None else (low_carb_data['ratio'] if low_carb_data else 0)
                    
                    # 비율 합계가 1을 넘지 않도록 조정
                    total_ratio = recycling_ratio + material_low_carb_ratio
                    if total_ratio > 1.0:
                        self._print(f"⚠️ {material_name}: 재활용({recycling_ratio*100:.1f}%) + 저탄소메탈({material_low_carb_ratio*100:.1f}%) = {total_ratio*100:.1f}% > 100%, 비율 조정", level="warning")
                        # 비례적으로 축소
                        recycling_ratio_adjusted = recycling_ratio / total_ratio
                        material_low_carb_ratio_adjusted = material_low_carb_ratio / total_ratio
                        virgin_ratio = 0
                    else:
                        recycling_ratio_adjusted = recycling_ratio
                        material_low_carb_ratio_adjusted = material_low_carb_ratio
                        virgin_ratio = 1 - total_ratio
                    
                    # 각 구성요소별 소요량 계산
                    virgin_requirement = original_requirement * virgin_ratio
                    recycled_requirement = original_requirement * recycling_ratio_adjusted
                    low_carb_requirement = original_requirement * material_low_carb_ratio_adjusted
                    
                    # 각 구성요소별 배출계수 계산
                    virgin_emission_factor = original_emission_factor
                    recycled_emission_factor = original_emission_factor * (recycling_impact if recycling_impact is not None else 1.0)
                    low_carb_emission_factor = low_carb_data['emission_factor'] if low_carb_data else original_emission_factor
                    
                    # 각 구성요소별 배출량 계산
                    virgin_emission = virgin_requirement * virgin_emission_factor
                    recycled_emission = recycled_requirement * recycled_emission_factor
                    low_carb_emission = low_carb_requirement * low_carb_emission_factor
                    total_emission = virgin_emission + recycled_emission + low_carb_emission
                    
                    # 새로운 전체 배출계수 계산 (가중평균)
                    new_emission_factor = total_emission / original_requirement if original_requirement > 0 else 0
                    
                    # 데이터 업데이트
                    material_data['소요량'] = original_requirement  # 총 소요량은 유지
                    material_data['배출계수'] = new_emission_factor
                    material_data['재활용_비중'] = recycling_ratio_adjusted
                    material_data['저탄소메탈_비중'] = material_low_carb_ratio_adjusted
                    material_data['신재_비중'] = virgin_ratio
                    
                    # 상세 정보 저장
                    if recycling_impact is not None:
                        material_data['재활용_영향도'] = recycling_impact
                    if low_carb_data:
                        material_data['저탄소메탈_배출계수'] = low_carb_emission_factor
                    
                    # 로그 출력
                    self._print(f"    🔄 {material_name}: 3원 분할 적용", level="debug")
                    self._print(f"      📊 신재: {virgin_requirement:.6f} kg × {virgin_emission_factor:.6f} = {virgin_emission:.6f} kg CO2e ({virgin_ratio*100:.1f}%)", level="debug")
                    self._print(f"      📊 재활용재: {recycled_requirement:.6f} kg × {recycled_emission_factor:.6f} = {recycled_emission:.6f} kg CO2e ({recycling_ratio_adjusted*100:.1f}%)", level="debug")
                    self._print(f"      📊 저탄소메탈: {low_carb_requirement:.6f} kg × {low_carb_emission_factor:.6f} = {low_carb_emission:.6f} kg CO2e ({material_low_carb_ratio_adjusted*100:.1f}%)", level="debug")
                    self._print(f"      📊 총 배출량: {total_emission:.6f} kg CO2e", level="debug")
                    self._print(f"      📊 새로운 배출계수: {new_emission_factor:.6f}", level="debug")
        
        return processed_data

    def _get_recycling_impact_for_material(self, material_name: str) -> float:
        """
        재료명에 해당하는 재활용 영향도를 반환합니다.
        
        Args:
            material_name (str): 재료명
            
        Returns:
            float: 재활용 영향도 (0~1). None인 경우 영향도 정보가 없음
        """
        if not self.recycle_impact_data or '재활용재' not in self.recycle_impact_data:
            return None
        
        recycling_impacts = self.recycle_impact_data['재활용재']
        
        # 재료명에서 원소 추출하여 매칭
        material_lower = material_name.lower()
        
        for element, impact in recycling_impacts.items():
            if element.lower() in material_lower:
                return impact
        
        return None

    def _get_low_carb_metal_data_for_material(self, material_name: str) -> Dict[str, float]:
        """
        재료명에 해당하는 저탄소메탈 데이터(비중, 배출계수)를 반환합니다.
        
        Args:
            material_name (str): 재료명
            
        Returns:
            Dict[str, float]: {'ratio': 비중(0~1), 'emission_factor': 배출계수}. 매칭되지 않으면 None
        """
        if not self.low_carb_metal_data or '비중' not in self.low_carb_metal_data or '배출계수' not in self.low_carb_metal_data:
            return None
        
        low_carb_ratios = self.low_carb_metal_data['비중']
        low_carb_emission_factors = self.low_carb_metal_data['배출계수']
        
        # 재료명에서 원소 추출하여 매칭
        material_lower = material_name.lower()
        
        for element in low_carb_ratios.keys():
            if element.lower() in material_lower:
                ratio = low_carb_ratios[element] / 100.0  # 백분율을 비율로 변환
                emission_factor = low_carb_emission_factors.get(element, 0)
                return {'ratio': ratio, 'emission_factor': emission_factor}
        
        return None

    def simulate_recycling_carbon_reduction(self, site: str = 'before', use_impact: bool = True) -> Dict[str, Any]:
        """
        재활용 비율을 적용한 탄소 저감 시뮬레이션을 실행합니다.
        Args:
            site (str): 'before' 또는 'after'. 기본값은 'before'
            use_impact (bool): 재활용 재료 영향도를 사용할지 여부. 기본값은 True
        Returns:
            Dict[str, Any]: 시뮬레이션 결과 (단일 비율)
        """
        # self.recycling_ratio 사용 (초기화 시 설정된 값)
        recycling_ratio = self.recycling_ratio

        self._print("=" * 80, level="info")
        self._print("♻️ 5단계: 재활용 비율 탄소 저감 시뮬레이션", level="info")
        self._print("=" * 80, level="info")
        self._print(f"📍 사이트: {site}", level="info")
        self._print(f"📊 적용할 재활용 비율: {recycling_ratio*100:.2f}% (from self.recycling_ratio)", level="info")
        self._print(f"📊 재활용 재료 영향도 사용: {'예' if use_impact else '아니오'}", level="info")
        self._print("=" * 80, level="info")
        
        try:
            # 기본 데이터 로드
            base_data = self.update_electricity_emission_factor(site=site)
            if not base_data:
                self._print("❌ 기본 데이터 로드 실패", level="error")
                return None
                
            base_emission = self.calculate_carbon_emission(base_data)
            if not base_emission:
                self._print("❌ 기준 탄소배출량 계산 실패", level="error")
                return None
                
            base_total = base_emission['총_배출량']
            
            self._print(f"\n📊 기준 탄소배출량 (재활용 0%): {base_total:.6f} kg CO2e", level="info")
            
            # 재활용 비율과 저탄소메탈 비율 적용된 데이터 생성
            if use_impact:
                # 저탄소메탈을 포함한 3원 분할 사용
                recycled_data = self._apply_three_way_material_split(base_data, recycling_ratio)
            else:
                # 기존 재활용만 적용
                recycled_data = self._apply_recycling_ratio(base_data, recycling_ratio)
            
            # 탄소배출량 계산
            emission_result = self.calculate_carbon_emission(recycled_data, baseline_emission=base_total)
            if not emission_result:
                self._print("❌ 재활용 적용 후 탄소배출량 계산 실패", level="error")
                return None
            
            # 결과 저장
            scenario_result = {
                'recycling_ratio': recycling_ratio,
                'recycling_ratio_percent': recycling_ratio * 100,
                'total_emission': emission_result['총_배출량'],
                'reduction_amount': emission_result['감축량'],
                'reduction_rate': emission_result['감축률'],
                'category_contributions': emission_result['카테고리별_기여도'],
                'detailed_emissions': {
                    'raw_materials': emission_result.get('원재료', {}),
                    'energy_tier1': emission_result.get('Energy(Tier-1)', {}),
                    'energy_tier2': emission_result.get('Energy(Tier-2)', {})
                }
            }
            
            self._print(f"  📉 총 배출량: {emission_result['총_배출량']:.6f} kg CO2e", level="info")
            self._print(f"  📉 감축량: {emission_result['감축량']:.6f} kg CO2e", level="info")
            self._print(f"  📉 감축률: {emission_result['감축률']:.2f}%", level="info")
            self._print("  📊 카테고리별 기여도:", level="info")
            for category, contribution in emission_result['카테고리별_기여도'].items():
                self._print(f"    • {category}: {contribution:.2f}%", level="info")
            self._print("=" * 80, level="info")
            
            return scenario_result
            
        except Exception as e:
            self._print(f"❌ 재활용 시뮬레이션 중 오류 발생: {e}", level="error")
            return None

    def simulate_recycling_only_carbon_reduction(self, site: str = 'before') -> Dict[str, Any]:
        """
        재활용 비율만 적용한 탄소 저감 시뮬레이션을 실행합니다 (저탄소메탈 제외).
        Args:
            site (str): 'before' 또는 'after'. 기본값은 'before'
        Returns:
            Dict[str, Any]: 재활용만 적용한 시뮬레이션 결과
        """
        recycling_ratio = self.recycling_ratio

        self._print("=" * 80, level="info")
        self._print("♻️ 재활용 Only 탄소 저감 시뮬레이션", level="info")
        self._print("=" * 80, level="info")
        self._print(f"📍 사이트: {site}", level="info")
        self._print(f"📊 적용할 재활용 비율: {recycling_ratio*100:.2f}%", level="info")
        self._print("📊 저탄소메탈: 미적용 (0%)", level="info")
        self._print("=" * 80, level="info")
        
        try:
            # 기본 데이터 로드
            base_data = self.update_electricity_emission_factor(site=site)
            if not base_data:
                self._print("❌ 기본 데이터 로드 실패", level="error")
                return None
                
            base_emission = self.calculate_carbon_emission(base_data)
            if not base_emission:
                self._print("❌ 기준 탄소배출량 계산 실패", level="error")
                return None
                
            base_total = base_emission['총_배출량']
            
            self._print(f"\n📊 기준 탄소배출량: {base_total:.6f} kg CO2e", level="info")
            
            # 재활용만 적용 (저탄소메탈 비율 0으로 설정)
            recycled_data = self._apply_three_way_material_split(base_data, recycling_ratio, low_carb_ratio=0.0)
            
            # 탄소배출량 계산
            emission_result = self.calculate_carbon_emission(recycled_data, baseline_emission=base_total)
            if not emission_result:
                self._print("❌ 재활용 적용 후 탄소배출량 계산 실패", level="error")
                return None
            
            # 결과 저장
            scenario_result = {
                'scenario_type': 'recycling_only',
                'recycling_ratio': recycling_ratio,
                'recycling_ratio_percent': recycling_ratio * 100,
                'low_carb_ratio': 0.0,
                'low_carb_ratio_percent': 0.0,
                'total_emission': emission_result['총_배출량'],
                'reduction_amount': emission_result['감축량'],
                'reduction_rate': emission_result['감축률'],
                'category_contributions': emission_result['카테고리별_기여도'],
                'detailed_emissions': {
                    'raw_materials': emission_result.get('원재료', {}),
                    'energy_tier1': emission_result.get('Energy(Tier-1)', {}),
                    'energy_tier2': emission_result.get('Energy(Tier-2)', {})
                }
            }
            
            self._print(f"  📉 총 배출량: {emission_result['총_배출량']:.6f} kg CO2e", level="info")
            self._print(f"  📉 감축량: {emission_result['감축량']:.6f} kg CO2e", level="info")
            self._print(f"  📉 감축률: {emission_result['감축률']:.2f}%", level="info")
            self._print("=" * 80, level="info")
            
            return scenario_result
            
        except Exception as e:
            self._print(f"❌ 재활용 Only 시뮬레이션 중 오류 발생: {e}", level="error")
            return None

    def simulate_low_carb_only_carbon_reduction(self, site: str = 'before') -> Dict[str, Any]:
        """
        저탄소메탈만 적용한 탄소 저감 시뮬레이션을 실행합니다 (재활용 제외).
        Args:
            site (str): 'before' 또는 'after'. 기본값은 'before'
        Returns:
            Dict[str, Any]: 저탄소메탈만 적용한 시뮬레이션 결과
        """
        low_carb_ratio = self.low_carb_metal_ratio

        self._print("=" * 80, level="info")
        self._print("🌱 저탄소메탈 Only 탄소 저감 시뮬레이션", level="info")
        self._print("=" * 80, level="info")
        self._print(f"📍 사이트: {site}", level="info")
        self._print("📊 재활용: 미적용 (0%)", level="info")
        self._print(f"📊 적용할 저탄소메탈 비율: {low_carb_ratio*100:.2f}%", level="info")
        self._print("=" * 80, level="info")
        
        try:
            # 기본 데이터 로드
            base_data = self.update_electricity_emission_factor(site=site)
            if not base_data:
                self._print("❌ 기본 데이터 로드 실패", level="error")
                return None
                
            base_emission = self.calculate_carbon_emission(base_data)
            if not base_emission:
                self._print("❌ 기준 탄소배출량 계산 실패", level="error")
                return None
                
            base_total = base_emission['총_배출량']
            
            self._print(f"\n📊 기준 탄소배출량: {base_total:.6f} kg CO2e", level="info")
            
            # 저탄소메탈만 적용 (재활용 비율 0으로 설정)
            low_carb_data = self._apply_three_way_material_split(base_data, recycling_ratio=0.0, low_carb_ratio=low_carb_ratio)
            
            # 탄소배출량 계산
            emission_result = self.calculate_carbon_emission(low_carb_data, baseline_emission=base_total)
            if not emission_result:
                self._print("❌ 저탄소메탈 적용 후 탄소배출량 계산 실패", level="error")
                return None
            
            # 결과 저장
            scenario_result = {
                'scenario_type': 'low_carb_only',
                'recycling_ratio': 0.0,
                'recycling_ratio_percent': 0.0,
                'low_carb_ratio': low_carb_ratio,
                'low_carb_ratio_percent': low_carb_ratio * 100,
                'total_emission': emission_result['총_배출량'],
                'reduction_amount': emission_result['감축량'],
                'reduction_rate': emission_result['감축률'],
                'category_contributions': emission_result['카테고리별_기여도'],
                'detailed_emissions': {
                    'raw_materials': emission_result.get('원재료', {}),
                    'energy_tier1': emission_result.get('Energy(Tier-1)', {}),
                    'energy_tier2': emission_result.get('Energy(Tier-2)', {})
                }
            }
            
            self._print(f"  📉 총 배출량: {emission_result['총_배출량']:.6f} kg CO2e", level="info")
            self._print(f"  📉 감축량: {emission_result['감축량']:.6f} kg CO2e", level="info")
            self._print(f"  📉 감축률: {emission_result['감축률']:.2f}%", level="info")
            self._print("=" * 80, level="info")
            
            return scenario_result
            
        except Exception as e:
            self._print(f"❌ 저탄소메탈 Only 시뮬레이션 중 오류 발생: {e}", level="error")
            return None

    def generate_recycling_only_data(self, site: str = 'before') -> Dict[str, Any]:
        """
        2. 재활용만 적용된 데이터셋을 반환합니다. (단일 비율)
        Args:
            site (str): 'before' 또는 'after'. 기본값은 'before'
        Returns:
            Dict[str, Any]: 재활용 적용 데이터
        """
        try:
            # 재활용 시뮬레이션 실행 (단일 비율)
            simulation_result = self.simulate_recycling_carbon_reduction(site=site, use_impact=True)
            if not simulation_result:
                return None
            
            # 사이트 설명 생성
            site_description = "생산지 변경 전" if site == 'before' else "생산지 변경 후"
            
            return {
                'site': site,
                'simulation_result': simulation_result,
                'scenario': 'recycling_only',
                'description': f'재활용만 적용, {site_description}',
                'recycling_ratio': simulation_result['recycling_ratio']
            }
        except Exception as e:
            self._print(f"❌ 재활용만 적용 데이터 생성 중 오류 발생: {e}", level="error")
            return None

    def generate_combined_data(self, before_site: str = 'before', after_site: str = 'after') -> Dict[str, Any]:
        """
        4. 재활용과 사이트 변경을 둘 다 적용한 경우의 데이터를 반환합니다. (단일 비율)
        Args:
            before_site (str): 변경 전 사이트. 기본값은 'before'
            after_site (str): 변경 후 사이트. 기본값은 'after'
        Returns:
            Dict[str, Any]: 재활용과 사이트 변경 모두 적용된 데이터
        """
        try:
            # 사이트 설정 파일에서 실제 사이트 정보 확인
            from src.utils.file_operations import FileOperations
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cathode_site_path = os.path.join(current_dir, "..", "input", "cathode_site.json")
            site_data = FileOperations.load_json(cathode_site_path, default={'CAM': {'before': '중국', 'after': '한국'}, 'pCAM': {'before': '중국', 'after': '한국'}}, user_id=self.user_id)
            
            cam_before = site_data.get('CAM', {}).get('before', '중국')
            cam_after = site_data.get('CAM', {}).get('after', '한국')
            pcam_before = site_data.get('pCAM', {}).get('before', '중국')
            pcam_after = site_data.get('pCAM', {}).get('after', '한국')
            
            # 실제 사이트 변경이 있는지 확인
            has_site_change = (cam_before != cam_after) or (pcam_before != pcam_after)
            
            # Before 사이트에서 재활용 시뮬레이션
            before_recycling = self.generate_recycling_only_data(site=before_site)
            
            if not has_site_change:
                # 사이트 변경이 없는 경우, 재활용만 적용된 결과를 반환
                self._print(f"💡 사이트 변경 없음 (CAM: {cam_before}→{cam_after}, pCAM: {pcam_before}→{pcam_after})", level="info")
                self._print("   재활용만 적용된 결과를 반환합니다.", level="info")
                
                if not before_recycling:
                    return None
                
                return {
                    'before_site': before_site,
                    'after_site': after_site,
                    'before_recycling': before_recycling,
                    'after_recycling': before_recycling,  # 사이트 변경이 없으므로 before와 동일
                    'emission_change': 0.0,  # 사이트 변경이 없으므로 추가 감축 없음
                    'emission_change_rate': 0.0,
                    'has_site_change': False,
                    'site_details': {
                        'CAM': {'before': cam_before, 'after': cam_after},
                        'pCAM': {'before': pcam_before, 'after': pcam_after}
                    },
                    'scenario': 'combined',
                    'description': f'재활용 + 사이트 변경(생산지 변경 전 → 생산지 변경 후)',
                    'recycling_ratio': before_recycling['recycling_ratio']
                }
            else:
                # 사이트 변경이 있는 경우, 기존 로직 실행
                after_recycling = self.generate_recycling_only_data(site=after_site)
                if not before_recycling or not after_recycling:
                    return None
                
                # 종합 비교 분석
                before_emission = before_recycling['simulation_result']['total_emission']
                after_emission = after_recycling['simulation_result']['total_emission']
                emission_change = before_emission - after_emission  # 감축량 = before - after
                emission_change_rate = (emission_change / before_emission * 100) if before_emission > 0 else 0
                
                self._print(f"💡 사이트 변경 확인 (CAM: {cam_before}→{cam_after}, pCAM: {pcam_before}→{pcam_after})", level="info")
                self._print(f"   감축량: {emission_change:.6f} kg CO2e, 감축률: {emission_change_rate:.2f}%", level="info")
                
                return {
                    'before_site': before_site,
                    'after_site': after_site,
                    'before_recycling': before_recycling,
                    'after_recycling': after_recycling,
                    'emission_change': emission_change,
                    'emission_change_rate': emission_change_rate,
                    'has_site_change': True,
                    'site_details': {
                        'CAM': {'before': cam_before, 'after': cam_after},
                        'pCAM': {'before': pcam_before, 'after': pcam_after}
                    },
                    'scenario': 'combined',
                    'description': f'재활용 + 사이트 변경(생산지 변경 전 → 생산지 변경 후)',
                    'recycling_ratio': before_recycling['recycling_ratio']
                }
        except Exception as e:
            self._print(f"❌ 종합 데이터 생성 중 오류 발생: {e}", level="error")
            return None

    def generate_all_scenarios_data(self) -> Dict[str, Any]:
        """
        모든 시나리오의 데이터를 한 번에 생성합니다. (단일 비율)
        Returns:
            Dict[str, Any]: 모든 시나리오 데이터
        """
        try:
            self._print("=" * 80, level="info")
            self._print("🔬 6단계: 모든 시나리오 데이터 생성", level="info")
            self._print("=" * 80, level="info")
            
            # 1. Baseline 데이터
            self._print("\n📊 6-1단계: Baseline 데이터 생성", level="info")
            baseline_data = self.generate_baseline_data(site='before')
            self._print(f"  - baseline_data 생성 결과: {baseline_data is not None}", level="debug")
            if baseline_data:
                self._print(f"  - baseline_data 키: {list(baseline_data.keys())}", level="debug")
                if 'emission_data' in baseline_data:
                    self._print(f"  - emission_data 키: {list(baseline_data['emission_data'].keys())}", level="debug")
                    if '총_배출량' in baseline_data['emission_data']:
                        self._print(f"  - 총_배출량: {baseline_data['emission_data']['총_배출량']}", level="debug")
            else:
                self._print(f"  ❌ baseline_data가 None입니다.", level="error")
            
            # 2. 재활용만 적용 데이터
            self._print("\n📊 6-2단계: 재활용 Only 데이터 생성", level="info")
            recycling_only_result = self.simulate_recycling_only_carbon_reduction(site='before')
            recycling_only_data = {
                'site': 'before',
                'simulation_result': recycling_only_result,
                'scenario': 'recycling_only',
                'description': '재활용만 적용 (저탄소메탈 제외)',
                'recycling_ratio': recycling_only_result['recycling_ratio'] if recycling_only_result else 0
            } if recycling_only_result else None
            
            self._print(f"  - recycling_only_data 생성 결과: {recycling_only_data is not None}", level="debug")
            
            # 3. 저탄소메탈만 적용 데이터
            self._print("\n📊 6-3단계: 저탄소메탈 Only 데이터 생성", level="info")
            low_carb_only_result = self.simulate_low_carb_only_carbon_reduction(site='before')
            low_carb_only_data = {
                'site': 'before',
                'simulation_result': low_carb_only_result,
                'scenario': 'low_carb_only',
                'description': '저탄소메탈만 적용 (재활용 제외)',
                'low_carb_ratio': low_carb_only_result['low_carb_ratio'] if low_carb_only_result else 0
            } if low_carb_only_result else None
            
            self._print(f"  - low_carb_only_data 생성 결과: {low_carb_only_data is not None}", level="debug")
            
            # 4. 재활용 + 저탄소메탈 동시 적용 데이터 (기존)
            self._print("\n📊 6-4단계: 재활용 + 저탄소메탈 동시 적용 데이터 생성", level="info")
            combined_recycling_data = self.generate_recycling_only_data(site='before')
            self._print(f"  - combined_recycling_data 생성 결과: {combined_recycling_data is not None}", level="debug")
            
            # 5. 사이트 변경만 데이터
            self._print("\n📊 6-5단계: 사이트 변경만 데이터 생성", level="info")
            site_change_only_data = self.generate_site_change_only_data(before_site='before', after_site='after')
            self._print(f"  - site_change_only_data 생성 결과: {site_change_only_data is not None}", level="debug")
            if site_change_only_data:
                self._print(f"  - site_change_only_data 키: {list(site_change_only_data.keys())}", level="debug")
                if 'after_data' in site_change_only_data:
                    self._print(f"  - after_data 키: {list(site_change_only_data['after_data'].keys())}", level="debug")
                else:
                    print(f"  ❌ site_change_only_data가 None입니다.")
            
            # 6. 종합 데이터 (사이트 변경 + 재활용 + 저탄소메탈)
            if self.verbose:
                print("\n📊 6-6단계: 종합 데이터 생성")
            combined_data = self.generate_combined_data(before_site='before', after_site='after')
            if self.verbose:
                print(f"  - combined_data 생성 결과: {combined_data is not None}")
                if combined_data:
                    print(f"  - combined_data 키: {list(combined_data.keys())}")
                    if 'after_recycling' in combined_data:
                        print(f"  - after_recycling 키: {list(combined_data['after_recycling'].keys())}")
                else:
                    print(f"  ❌ combined_data가 None입니다.")
            
            # 종합 결과
            all_scenarios = {
                'baseline': baseline_data,
                'recycling_only': recycling_only_data,
                'low_carb_only': low_carb_only_data,
                'combined_recycling': combined_recycling_data,  # 기존 재활용+저탄소메탈 동시 적용
                'site_change_only': site_change_only_data,
                'combined': combined_data,
                'summary': {
                    'total_scenarios': 6,
                    'recycling_ratio': recycling_only_data['recycling_ratio'] if recycling_only_data and 'recycling_ratio' in recycling_only_data else None,
                    'low_carb_ratio': low_carb_only_data['low_carb_ratio'] if low_carb_only_data and 'low_carb_ratio' in low_carb_only_data else None,
                    'generation_time': 'completed'
                }
            }
            
            if self.verbose:
                print("\n✅ 모든 시나리오 데이터 생성 완료")
                print("=" * 80)
                print(f"📊 최종 결과:")
                print(f"  - baseline: {baseline_data is not None}")
                print(f"  - recycling_only: {recycling_only_data is not None}")
                print(f"  - low_carb_only: {low_carb_only_data is not None}")
                print(f"  - combined_recycling: {combined_recycling_data is not None}")
                print(f"  - site_change_only: {site_change_only_data is not None}")
                print(f"  - combined: {combined_data is not None}")
            
            return all_scenarios
        except Exception as e:
            if self.verbose:
                print(f"❌ 모든 시나리오 데이터 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            return None

    def generate_baseline_data(self, site: str = 'before') -> Dict[str, Any]:
        """
        1. 재활용 적용X, 지정된 사이트의 baseline dataset을 반환합니다.
        """
        try:
            self._print(f"    🔍 generate_baseline_data 시작 (site: {site})", level="info")
            self._print(f"    📊 1단계: 전력 배출계수 업데이트", level="debug")
            updated_data = self.update_electricity_emission_factor(site=site)
            self._print(f"      - updated_data 생성 결과: {updated_data is not None}", level="debug")
            if updated_data:
                self._print(f"      - updated_data 키: {list(updated_data.keys())}", level="debug")
                if 'Energy(Tier-1)' in updated_data:
                    self._print(f"      - Energy(Tier-1) 키: {list(updated_data['Energy(Tier-1)'].keys())}", level="debug")
                    if '전력' in updated_data['Energy(Tier-1)']:
                        self._print(f"      - Energy(Tier-1) 전력 배출계수: {updated_data['Energy(Tier-1)']['전력']['배출계수']}", level="debug")
                if 'Energy(Tier-2)' in updated_data:
                    self._print(f"      - Energy(Tier-2) 키: {list(updated_data['Energy(Tier-2)'].keys())}", level="debug")
                    if '전력' in updated_data['Energy(Tier-2)']:
                        self._print(f"      - Energy(Tier-2) 전력 배출계수: {updated_data['Energy(Tier-2)']['전력']['배출계수']}", level="debug")
            else:
                self._print(f"      ❌ updated_data가 None입니다.", level="error")
                return None
            self._print(f"    📊 2단계: 탄소배출량 계산", level="debug")
            emission_data = self.calculate_carbon_emission(updated_data)
            self._print(f"      - emission_data 생성 결과: {emission_data is not None}", level="debug")
            if emission_data:
                self._print(f"      - emission_data 키: {list(emission_data.keys())}", level="debug")
                if '총_배출량' in emission_data:
                    self._print(f"      - 총_배출량: {emission_data['총_배출량']}", level="debug")
                if '카테고리별_기여도' in emission_data:
                    self._print(f"      - 카테고리별_기여도: {emission_data['카테고리별_기여도']}", level="debug")
            else:
                self._print(f"      ❌ emission_data가 None입니다.", level="error")
                return None
            site_description = "생산지 변경 전" if site == 'before' else "생산지 변경 후"
            result = {
                'site': site,
                'updated_data': updated_data,
                'emission_data': emission_data,
                'scenario': 'baseline',
                'description': f'재활용 적용X, {site_description}'
            }
            self._print(f"    ✅ generate_baseline_data 완료", level="info")
            self._print(f"      - result 키: {list(result.keys())}", level="debug")
            return result
        except Exception as e:
            self._print(f"❌ Baseline 데이터 생성 중 오류 발생: {e}", level="error")
            import traceback
            traceback.print_exc()
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
            # 사이트 설정 파일에서 실제 사이트 정보 확인
            from src.utils.file_operations import FileOperations
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cathode_site_path = os.path.join(current_dir, "..", "input", "cathode_site.json")
            site_data = FileOperations.load_json(cathode_site_path, default={'CAM': {'before': '중국', 'after': '한국'}, 'pCAM': {'before': '중국', 'after': '한국'}}, user_id=self.user_id)
            
            cam_before = site_data.get('CAM', {}).get('before', '중국')
            cam_after = site_data.get('CAM', {}).get('after', '한국')
            pcam_before = site_data.get('pCAM', {}).get('before', '중국')
            pcam_after = site_data.get('pCAM', {}).get('after', '한국')
            
            # 실제 사이트 변경이 있는지 확인
            has_site_change = (cam_before != cam_after) or (pcam_before != pcam_after)
            
            # Before 사이트 데이터
            before_data = self.generate_baseline_data(site=before_site)
            if not before_data:
                return None
            
            # After 사이트 데이터
            after_data = self.generate_baseline_data(site=after_site)
            if not after_data:
                return None
            
            # 사이트 변경 효과 계산
            before_emission = before_data['emission_data']['총_배출량']
            after_emission = after_data['emission_data']['총_배출량']
            
            # 실제 사이트 변경이 없는 경우 감축량과 감축률을 0으로 설정
            if not has_site_change:
                emission_change = 0.0
                emission_change_rate = 0.0
                if self.verbose:
                    print(f"💡 사이트 변경 없음 (CAM: {cam_before}→{cam_after}, pCAM: {pcam_before}→{pcam_after})")
                    print(f"   감축량: {emission_change:.6f} kg CO2e, 감축률: {emission_change_rate:.2f}%")
            else:
                emission_change = before_emission - after_emission  # 감축량 = before - after
                emission_change_rate = (emission_change / before_emission * 100) if before_emission > 0 else 0
                if self.verbose:
                    print(f"💡 사이트 변경 확인 (CAM: {cam_before}→{cam_after}, pCAM: {pcam_before}→{pcam_after})")
                    print(f"   감축량: {emission_change:.6f} kg CO2e, 감축률: {emission_change_rate:.2f}%")
            
            return {
                'before_site': before_site,
                'after_site': after_site,
                'before_data': before_data,
                'after_data': after_data,
                'emission_change': emission_change,
                'emission_change_rate': emission_change_rate,
                'has_site_change': has_site_change,
                'site_details': {
                    'CAM': {'before': cam_before, 'after': cam_after},
                    'pCAM': {'before': pcam_before, 'after': pcam_after}
                },
                'scenario': 'site_change_only',
                'description': f'사이트 변경(생산지 변경 전 → 생산지 변경 후)'
            }
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 사이트 변경 데이터 생성 중 오류 발생: {e}")
            return None

    def compare_recycling_scenarios(self, before_ratios: list = None, after_ratios: list = None) -> Dict[str, Any]:
        """
        Before/After 사이트에서 재활용 시나리오를 비교합니다.
        
        Args:
            before_ratios (list): Before 사이트 적용할 재활용 비율 리스트
            after_ratios (list): After 사이트 적용할 재활용 비율 리스트
            
        Returns:
            Dict[str, Any]: 비교 결과
        """
        if before_ratios is None:
            before_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        if after_ratios is None:
            after_ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        if self.verbose:
            print("=" * 80)
            print("🔄 Before/After 사이트 재활용 시나리오 비교")
            print("=" * 80)
        
        # Before 사이트 시뮬레이션
        if self.verbose:
            print("\n📊 Before 사이트 시뮬레이션 실행...")
        before_results = self.simulate_recycling_carbon_reduction(site='before', use_impact=True)
        
        # After 사이트 시뮬레이션
        if self.verbose:
            print("\n📊 After 사이트 시뮬레이션 실행...")
        after_results = self.simulate_recycling_carbon_reduction(site='after', use_impact=True)
        
        # 비교 분석
        comparison_results = self._analyze_scenario_comparison(before_results, after_results)
        
        # 비교 결과 출력
        self._print_comparison_results(comparison_results)
        
        return {
            'before_results': before_results,
            'after_results': after_results,
            'comparison': comparison_results
        }
    
    def _analyze_scenario_comparison(self, before_results: Dict[str, Any], after_results: Dict[str, Any]) -> Dict[str, Any]:
        """시나리오 비교 분석을 수행합니다."""
        comparison = {
            'site_difference': {},
            'recycling_effectiveness_comparison': {},
            'optimal_scenarios': {}
        }
        
        # 사이트 차이 분석
        before_base = before_results['total_emission']
        after_base = after_results['total_emission']
        site_difference = before_base - after_base
        site_difference_rate = (site_difference / before_base) * 100
        
        comparison['site_difference'] = {
            'before_base': before_base,
            'after_base': after_base,
            'difference': site_difference,
            'difference_rate': site_difference_rate
        }
        
        # 재활용 효과성 비교
        before_scenarios = [before_results]
        after_scenarios = [after_results]
        
        # 각 재활용 비율에서의 사이트별 차이 분석
        recycling_comparison = []
        for i, (before_scenario, after_scenario) in enumerate(zip(before_scenarios, after_scenarios)):
            if before_scenario['recycling_ratio'] == after_scenario['recycling_ratio']:
                ratio = before_scenario['recycling_ratio']
                before_emission = before_scenario['total_emission']
                after_emission = after_scenario['total_emission']
                difference = before_emission - after_emission
                difference_rate = (difference / before_emission) * 100
                
                recycling_comparison.append({
                    'recycling_ratio': ratio,
                    'before_emission': before_emission,
                    'after_emission': after_emission,
                    'difference': difference,
                    'difference_rate': difference_rate
                })
        
        comparison['recycling_effectiveness_comparison'] = recycling_comparison
        
        # 최적 시나리오 찾기
        before_optimal = min(before_scenarios, key=lambda x: x['total_emission'])
        after_optimal = min(after_scenarios, key=lambda x: x['total_emission'])
        
        comparison['optimal_scenarios'] = {
            'before_optimal': before_optimal,
            'after_optimal': after_optimal,
            'combined_optimal': {
                'before_emission': before_results['total_emission'],
                'after_emission': after_optimal['total_emission'],
                'total_reduction': before_results['total_emission'] - after_optimal['total_emission'],
                'total_reduction_rate': ((before_results['total_emission'] - after_optimal['total_emission']) / before_results['total_emission']) * 100
            }
        }
        
        return comparison
    
    def _print_comparison_results(self, comparison_results: Dict[str, Any]):
        """비교 결과를 출력합니다."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("📊 Before/After 사이트 재활용 시나리오 비교 결과")
            print("=" * 80)
            
            # 사이트 차이 분석
            site_diff = comparison_results['site_difference']
            print(f"\n🏭 사이트 차이 분석:")
            print(f"  • Before 기준 배출량: {site_diff['before_base']:.6f} kg CO2e")
            print(f"  • After 기준 배출량: {site_diff['after_base']:.6f} kg CO2e")
            print(f"  • 사이트 차이: {site_diff['difference']:.6f} kg CO2e")
            print(f"  • 사이트 차이율: {site_diff['difference_rate']:.2f}%")
            
            # 재활용 효과성 비교
            recycling_comparison = comparison_results['recycling_effectiveness_comparison']
            print(f"\n♻️ 재활용 비율별 사이트 차이:")
            print(f"{'재활용 비율':<12} {'Before':<15} {'After':<15} {'차이':<12} {'차이율':<10}")
            print("-" * 70)
            
            for comp in recycling_comparison:
                ratio = f"{comp['recycling_ratio']*100:.0f}%"
                before = f"{comp['before_emission']:.6f}"
                after = f"{comp['after_emission']:.6f}"
                diff = f"{comp['difference']:.6f}"
                diff_rate = f"{comp['difference_rate']:.2f}%"
                print(f"{ratio:<12} {before:<15} {after:<15} {diff:<12} {diff_rate:<10}")
            
            # 최적 시나리오
            optimal = comparison_results['optimal_scenarios']
            print(f"\n🏆 최적 시나리오:")
            print(f"  • Before 최적: 재활용 {optimal['before_optimal']['recycling_ratio_percent']:.0f}% → 감축 {optimal['before_optimal']['reduction_rate']:.2f}%")
            print(f"  • After 최적: 재활용 {optimal['after_optimal']['recycling_ratio_percent']:.0f}% → 감축 {optimal['after_optimal']['reduction_rate']:.2f}%")
            print(f"  • 통합 최적 감축량: {optimal['combined_optimal']['total_reduction']:.6f} kg CO2e")
            print(f"  • 통합 최적 감축률: {optimal['combined_optimal']['total_reduction_rate']:.2f}%")
            
            print("=" * 80)

    def update_ref_proportions_with_tier_contributions(self, basic_df: pd.DataFrame, ref_proportions_df: pd.DataFrame, scenario: str = 'baseline') -> pd.DataFrame:
        """
        basic_df의 Energy Tier 1과 Tier 2 기여도를 ref_proportions_df의 양극재/Cathode Active Material 행에 업데이트합니다.
        
        Args:
            basic_df (pd.DataFrame): 기본 시나리오 분석 결과 데이터프레임
            ref_proportions_df (pd.DataFrame): 참조 비율 데이터프레임
            scenario (str): 시나리오 타입 ('baseline', 'recycling', 'site_change', 'both')
                - baseline: 기본 시나리오 (변경 없음)
                - recycling: 재활용만 적용
                - site_change: 사이트 변경만 적용
                - both: 재활용 + 사이트 변경 모두 적용
            
        Returns:
            pd.DataFrame: 업데이트된 ref_proportions_df
        """
        try:
            if self.verbose:
                print(f"🔧 7단계: ref_proportions_df Energy Tier 기여도 업데이트 (시나리오: {scenario})")
            
            # ref_proportions_df 복사
            updated_df = ref_proportions_df.copy()
            
            # 양극재 + Cathode Active Material 조건으로 필터링
            # 자재명 컬럼이 있는지 확인하고, 없으면 자재명(포함) 컬럼 사용
            material_name_col = '자재명' if '자재명' in updated_df.columns else '자재명(포함)'
            
            cathode_materials = updated_df[
                (updated_df['자재품목'] == '양극재') & 
                (updated_df[material_name_col].str.contains('Cathode Active Material', case=False, na=False))
            ]
            
            if len(cathode_materials) == 0:
                if self.verbose:
                    print("⚠️ 양극재 + Cathode Active Material 조건에 맞는 행을 찾을 수 없습니다.")
                return updated_df
            
            if self.verbose:
                print(f"📊 업데이트 대상 행 수: {len(cathode_materials)}개")
            
            # 시나리오별 매핑
            scenario_mapping = {
                'baseline': 'Baseline',
                'recycling': '재활용 적용',
                'site_change': '사이트 변경',
                'both': '재활용 + 사이트 변경'
            }
            
            target_scenario = scenario_mapping.get(scenario, 'Baseline')
            
            # 해당 시나리오의 Energy Tier 기여도 가져오기
            scenario_row = basic_df[basic_df['시나리오'] == target_scenario]
            if len(scenario_row) == 0:
                if self.verbose:
                    print(f"⚠️ {target_scenario} 시나리오를 찾을 수 없습니다.")
                return updated_df
            
            tier1_contribution = scenario_row.iloc[0]['Energy_Tier1_전력_기여도_퍼센트']
            tier2_contribution = scenario_row.iloc[0]['Energy_Tier2_전력_기여도_퍼센트']
            
            if self.verbose:
                print(f"📊 {target_scenario} 시나리오 기여도:")
                print(f"  • Energy Tier 1 전력 기여도: {tier1_contribution:.6f}%")
                print(f"  • Energy Tier 2 전력 기여도: {tier2_contribution:.6f}%")
            
            # 업데이트 대상 행들의 인덱스
            update_indices = cathode_materials.index
            
            # Tier1_RE100(%)와 Tier2_RE100(%) 컬럼이 있는지 확인하고 업데이트
            if 'Tier1_RE100(%)' in updated_df.columns:
                updated_df.loc[update_indices, 'Tier1_RE100(%)'] = tier1_contribution
                if self.verbose:
                    print(f"✅ Tier1_RE100(%) 업데이트 완료: {tier1_contribution:.6f}%")
            else:
                if self.verbose:
                    print("⚠️ Tier1_RE100(%) 컬럼이 없습니다.")
            
            if 'Tier2_RE100(%)' in updated_df.columns:
                updated_df.loc[update_indices, 'Tier2_RE100(%)'] = tier2_contribution
                if self.verbose:
                    print(f"✅ Tier2_RE100(%) 업데이트 완료: {tier2_contribution:.6f}%")
            else:
                if self.verbose:
                    print("⚠️ Tier2_RE100(%) 컬럼이 없습니다.")
            
            # 재활용 관련 시나리오의 경우 재활용 비율도 업데이트
            if scenario in ['recycling', 'both']:
                recycling_ratio = scenario_row.iloc[0].get('재활용_비율_퍼센트', 0) / 100.0  # 퍼센트를 소수로 변환
                
                if '재활용_비율' in updated_df.columns:
                    updated_df.loc[update_indices, '재활용_비율'] = recycling_ratio
                    if self.verbose:
                        print(f"✅ 재활용_비율 업데이트 완료: {recycling_ratio:.3f}")
                else:
                    if self.verbose:
                        print("⚠️ 재활용_비율 컬럼이 없습니다.")
            
            # 업데이트된 행 정보 출력
            if self.verbose:
                print(f"\n📋 {target_scenario} 시나리오로 업데이트된 행 정보:")
                updated_rows = updated_df.loc[update_indices]
                for idx, row in updated_rows.iterrows():
                    material_name = row.get('자재명', row.get('자재명(포함)', 'Unknown'))
                    print(f"  • 행 {idx}: {material_name}")
                    if 'Tier1_RE100(%)' in updated_df.columns:
                        print(f"    - Tier1_RE100(%): {row['Tier1_RE100(%)']:.6f}%")
                    if 'Tier2_RE100(%)' in updated_df.columns:
                        print(f"    - Tier2_RE100(%): {row['Tier2_RE100(%)']:.6f}%")
                    if scenario in ['recycling', 'both'] and '재활용_비율' in updated_df.columns:
                        print(f"    - 재활용_비율: {row['재활용_비율']:.3f}")
            
            if self.verbose:
                print(f"✅ {target_scenario} 시나리오로 ref_proportions_df Energy Tier 기여도 업데이트 완료")
            
            return updated_df
            
        except Exception as e:
            if self.verbose:
                print(f"❌ ref_proportions_df 업데이트 중 오류 발생: {e}")
            return ref_proportions_df

    def _get_electricity_emission_factor(self, country: str) -> float:
        """
        국가별 전력 배출계수를 가져옵니다.
        
        Args:
            country (str): 국가명
            
        Returns:
            float: 전력 배출계수
        """
        try:
            if self.verbose:
                print(f"            🔍 _get_electricity_emission_factor 시작 (country: {country})")
            
            # FileOperations를 사용하여 사용자별 파일 로드
            electricity_data = FileOperations.load_json("stable_var/electricity_coef_by_country.json", user_id=self.user_id)
            
            if self.verbose:
                print(f"            - electricity_data 키: {list(electricity_data.keys())}")
            
            factor = electricity_data.get(country, 0)
            
            if self.verbose:
                print(f"            - {country} 전력 배출계수: {factor}")
            
            return factor
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 전력 배출계수 로드 실패 ({country}): {e}")
                import traceback
                traceback.print_exc()
            return 0

    def update_original_cathode_coef(self, basic_df: pd.DataFrame, original_df: pd.DataFrame, scenario: str = 'baseline') -> tuple:
        """
        basic_df의 감축률을 기반으로 original_df의 양극재 배출계수를 업데이트합니다.
        
        Args:
            basic_df (pd.DataFrame): 기본 시나리오 분석 결과 데이터프레임
            original_df (pd.DataFrame): 원본 데이터프레임
            scenario (str): 시나리오 타입 ('baseline', 'recycling', 'site_change', 'both')
                - baseline: 기본 시나리오 (변경 없음)
                - recycling: 재활용만 적용
                - site_change: 사이트 변경만 적용
                - both: 재활용 + 사이트 변경 모두 적용
            
        Returns:
            tuple: (적용결과 요약 df, 업데이트된 original_df)
        """
        try:
            if self.verbose:
                print(f"🔧 8단계: original_df 양극재 배출계수 업데이트 (시나리오: {scenario})")
            
            # original_df 깊은 복사
            updated_df = original_df.copy(deep=True)
            
            if self.verbose:
                print(f"📊 original_df 컬럼: {list(original_df.columns)}")
                print(f"📊 original_df shape: {original_df.shape}")
            
            # 양극재 조건으로 필터링
            cathode_materials = updated_df[updated_df['자재품목'] == '양극재']
            
            if len(cathode_materials) == 0:
                if self.verbose:
                    print("⚠️ 자재품목이 '양극재'인 행을 찾을 수 없습니다.")
                return pd.DataFrame(), updated_df
            
            if self.verbose:
                print(f"📊 업데이트 대상 행 수: {len(cathode_materials)}개")
                print(f"📊 업데이트 대상 행 인덱스: {list(cathode_materials.index)}")
                print(f"📊 업데이트 대상 행의 자재품목: {cathode_materials['자재품목'].tolist()}")
                if '배출계수' in cathode_materials.columns:
                    print(f"📊 업데이트 대상 행의 배출계수 타입: {cathode_materials['배출계수'].dtype}")
                    print(f"📊 업데이트 대상 행의 배출계수: {cathode_materials['배출계수'].tolist()}")
            
            # 시나리오별 매핑 - PCF simulation에서 생성되는 실제 시나리오명과 일치시킴
            scenario_mapping = {
                'baseline': 'Baseline',
                'recycling': '재활용&저탄소메탈 적용',
                'site_change': '사이트 변경',
                'both': '재활용&저탄소메탈 + 사이트 변경'
            }
            
            target_scenario = scenario_mapping.get(scenario, 'Baseline')

            # 시나리오별 양극재 배출계수 업데이트 로직
            if scenario == 'recycling':
                # 재활용 시나리오: "재활용&저탄소메탈 적용" 시나리오의 감축률 사용
                target_scenario = '재활용&저탄소메탈 적용'
                if self.verbose:
                    print(f"📋 양극재 배출계수 업데이트: {scenario} → {target_scenario} (재활용 적용)")
            elif scenario == 'site_change':
                # 사이트 변경 시나리오: "사이트 변경" 시나리오의 감축률 사용
                # 생산지 변경으로 전력 배출계수가 낮아지면 양극재 제조의 전력 배출량도 낮아짐
                target_scenario = '사이트 변경'
                if self.verbose:
                    print(f"📋 양극재 배출계수 업데이트: {scenario} → {target_scenario} (사이트 변경 적용)")
            elif scenario == 'both':
                # 종합 시나리오: "재활용&저탄소메탈 + 사이트 변경" 시나리오의 감축률 사용
                target_scenario = '재활용&저탄소메탈 + 사이트 변경'
                if self.verbose:
                    print(f"📋 양극재 배출계수 업데이트: {scenario} → {target_scenario} (재활용 + 사이트 변경 적용)")
            
            # 해당 시나리오의 감축률 가져오기
            scenario_row = basic_df[basic_df['시나리오'] == target_scenario]
            if len(scenario_row) == 0:
                if self.verbose:
                    print(f"⚠️ {target_scenario} 시나리오를 찾을 수 없습니다.")
                return pd.DataFrame(), updated_df
            
            reduction_rate = scenario_row.iloc[0]['감축률_퍼센트']
            
            if self.verbose:
                print(f"📊 {target_scenario} 시나리오 감축률: {reduction_rate:.6f}%")
            
            # 배출계수 감소율 계산 (1 - 감축률/100)
            reduction_factor = 1 - (reduction_rate / 100)
            
            if self.verbose:
                print(f"📊 배출계수 감소율: {reduction_factor:.6f} (원래 대비 {reduction_factor*100:.2f}%)")
            
            # 업데이트 대상 행들의 인덱스
            update_indices = cathode_materials.index
            
            # 배출계수 업데이트
            original_coef_values = updated_df.loc[update_indices, '배출계수'].copy()
            
            # 배출계수를 숫자형으로 변환 (문자열인 경우 대비)
            original_coef_values = pd.to_numeric(original_coef_values, errors='coerce')
            
            if self.verbose:
                print(f"📊 업데이트 전 배출계수:")
                for idx in update_indices:
                    print(f"  • 행 {idx}: {original_coef_values[idx]:.6f}")
            
            # 배출계수 업데이트
            new_coef_values = original_coef_values * reduction_factor
            
            if self.verbose:
                print(f"📊 계산된 새로운 배출계수:")
                for idx, new_coef in zip(update_indices, new_coef_values):
                    print(f"  • 행 {idx}: {new_coef:.6f}")
            
            # updated_df에 직접 할당
            for idx, new_coef in zip(update_indices, new_coef_values):
                updated_df.at[idx, '배출계수'] = new_coef
            
            if self.verbose:
                print(f"📊 업데이트 후 배출계수:")
                for idx in update_indices:
                    print(f"  • 행 {idx}: {updated_df.at[idx, '배출계수']:.6f}")
            
            # 적용결과 요약 데이터프레임 생성
            summary_data = []
            for idx in update_indices:
                original_coef = original_coef_values[idx]
                updated_coef = updated_df.loc[idx, '배출계수']
                material_name = updated_df.loc[idx, '자재명'] if '자재명' in updated_df.columns else f"행 {idx}"
                
                summary_data.append({
                    '행_인덱스': idx,
                    '자재명': material_name,
                    '원본_배출계수': original_coef,
                    '업데이트_배출계수': updated_coef,
                    '감소율': (1 - reduction_factor) * 100,
                    '시나리오': target_scenario
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            if self.verbose:
                print(f"\n📋 {target_scenario} 시나리오로 업데이트된 배출계수:")
                for _, row in summary_df.iterrows():
                    print(f"  • {row['자재명']}: {row['원본_배출계수']:.6f} → {row['업데이트_배출계수']:.6f} (감소율: {row['감소율']:.2f}%)")
            
            # 최종 확인: updated_df에서 실제로 업데이트되었는지 확인
            if self.verbose:
                print(f"\n🔍 최종 확인 - updated_df의 배출계수:")
                for idx in update_indices:
                    final_coef = updated_df.loc[idx, '배출계수']
                    print(f"  • 행 {idx}: {final_coef:.6f}")
            
            if self.verbose:
                print(f"✅ {target_scenario} 시나리오로 original_df 양극재 배출계수 업데이트 완료")
            
            return summary_df, updated_df
            
        except Exception as e:
            if self.verbose:
                print(f"❌ original_df 업데이트 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            return pd.DataFrame(), original_df


# 사용 예시
if __name__ == "__main__":
    # 기본 경로로 시뮬레이터 생성
    simulator = CathodeSimulator()
    simulator.print_summary()
    simulator.print_recycle_ratio_details()
    simulator.print_raw_material_requirements()
    simulator.print_requirement_update_log()
    
    # 원재료 데이터 확인
    raw_materials = simulator.get_raw_materials()
    if simulator.verbose:
        print(f"\n원재료 목록: {list(raw_materials.keys())}")
    
    # 특정 원재료 정보 확인
    if "NiSO4" in raw_materials:
        niso4_info = raw_materials["NiSO4"]
        if simulator.verbose:
            print(f"\nNiSO4 정보: {niso4_info}")
    
    # 전력 배출계수 업데이트 테스트
    if simulator.verbose:
        print("\n" + "="*50)
        print("전력 배출계수 업데이트 테스트")
        print("="*50)
    
    try:
        # before 사이트로 전력 배출계수 업데이트
        if simulator.verbose:
            print("\n🔧 Before 사이트 전력 배출계수 업데이트:")
        updated_data = simulator.update_electricity_emission_factor(site='before')
        
        if simulator.verbose:
            print("\n📋 업데이트된 전력 배출계수:")
        if "Energy(Tier-1)" in updated_data and "전력" in updated_data["Energy(Tier-1)"]:
            if simulator.verbose:
                print(f"Energy(Tier-1) 전력: {updated_data['Energy(Tier-1)']['전력']['배출계수']}")
        if "Energy(Tier-2)" in updated_data and "전력" in updated_data["Energy(Tier-2)"]:
            if simulator.verbose:
                print(f"Energy(Tier-2) 전력: {updated_data['Energy(Tier-2)']['전력']['배출계수']}")
        
        # after 사이트로도 테스트
        if simulator.verbose:
            print("\n" + "-"*30)
            print("🔧 After 사이트 전력 배출계수 업데이트:")
        updated_data_after = simulator.update_electricity_emission_factor(site='after')
        
        if simulator.verbose:
            print("\n📋 after 사이트 업데이트된 전력 배출계수:")
        if "Energy(Tier-1)" in updated_data_after and "전력" in updated_data_after["Energy(Tier-1)"]:
            if simulator.verbose:
                print(f"Energy(Tier-1) 전력: {updated_data_after['Energy(Tier-1)']['전력']['배출계수']}")
        if "Energy(Tier-2)" in updated_data_after and "전력" in updated_data_after["Energy(Tier-2)"]:
            if simulator.verbose:
                print(f"Energy(Tier-2) 전력: {updated_data_after['Energy(Tier-2)']['전력']['배출계수']}")
        
        # 전력 배출계수 비교
        if simulator.verbose:
            print("\n📊 전력 배출계수 비교:")
        if "Energy(Tier-1)" in updated_data and "전력" in updated_data["Energy(Tier-1)"]:
            if simulator.verbose:
                print(f"Before - Energy(Tier-1): {updated_data['Energy(Tier-1)']['전력']['배출계수']}")
        if "Energy(Tier-1)" in updated_data_after and "전력" in updated_data_after["Energy(Tier-1)"]:
            if simulator.verbose:
                print(f"After  - Energy(Tier-1): {updated_data_after['Energy(Tier-1)']['전력']['배출계수']}")
        if "Energy(Tier-2)" in updated_data and "전력" in updated_data["Energy(Tier-2)"]:
            if simulator.verbose:
                print(f"Before - Energy(Tier-2): {updated_data['Energy(Tier-2)']['전력']['배출계수']}")
        if "Energy(Tier-2)" in updated_data_after and "전력" in updated_data_after["Energy(Tier-2)"]:
            if simulator.verbose:
                print(f"After  - Energy(Tier-2): {updated_data_after['Energy(Tier-2)']['전력']['배출계수']}")
        
        # 탄소배출량 계산 테스트
        if simulator.verbose:
            print("\n" + "="*50)
            print("탄소배출량 계산 테스트")
            print("="*50)
        
        # 사이트 정보 가져오기 (사용자별 파일 사용)
        from src.utils.file_operations import FileOperations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        site_file_path = os.path.join(current_dir, "..", "input", "cathode_site.json")
        
        site_data = FileOperations.load_json(site_file_path, default={'CAM': {'before': '중국', 'after': '한국'}, 'pCAM': {'before': '중국', 'after': '한국'}}, user_id=None)
        
        before_country = site_data["CAM"]["before"]
        after_country = site_data["CAM"]["after"]
        
        # before 사이트 탄소배출량 계산
        if simulator.verbose:
            print(f"\n📊 {before_country}(before) 사이트 탄소배출량 계산:")
        before_emission = simulator.calculate_carbon_emission(updated_data)
        
        if simulator.verbose:
            print(f"\n총 배출량: {before_emission['총_배출량']} kg CO2e")
        if simulator.verbose:
            print("\n카테고리별 기여도:")
        for category, contribution in before_emission['카테고리별_기여도'].items():
            if simulator.verbose:
                print(f"  {category}: {contribution}%")
        
        if simulator.verbose:
            print("\n아이템별 상세 기여도:")
        for category in ["원재료", "Energy(Tier-1)", "Energy(Tier-2)"]:
            if category in before_emission:
                if simulator.verbose:
                    print(f"\n  📊 {category}:")
                for item_name, item_data in before_emission[category].items():
                    emission = item_data["탄소배출량(kg_CO2e)"]
                    contribution = item_data.get("기여도(%)", 0)
                    if simulator.verbose:
                        print(f"    • {item_name}: {emission:.6f} kg CO2e ({contribution:.2f}%)")
        
        # after 사이트 탄소배출량 계산 (before을 기준으로 감축량 계산)
        if simulator.verbose:
            print(f"\n📊 {after_country}(after) 사이트 탄소배출량 계산:")
        after_emission = simulator.calculate_carbon_emission(updated_data_after, baseline_emission=before_emission['총_배출량'])
        
        if simulator.verbose:
            print(f"\n총 배출량: {after_emission['총_배출량']} kg CO2e")
        if simulator.verbose:
            print(f"감축량: {after_emission['감축량']} kg CO2e")
        if simulator.verbose:
            print(f"감축률: {after_emission['감축률']}%")
        
        if simulator.verbose:
            print("\n카테고리별 기여도:")
        for category, contribution in after_emission['카테고리별_기여도'].items():
            if simulator.verbose:
                print(f"  {category}: {contribution}%")
        
        if simulator.verbose:
            print("\n아이템별 상세 기여도:")
        for category in ["원재료", "Energy(Tier-1)", "Energy(Tier-2)"]:
            if category in after_emission:
                if simulator.verbose:
                    print(f"\n  📊 {category}:")
                for item_name, item_data in after_emission[category].items():
                    emission = item_data["탄소배출량(kg_CO2e)"]
                    contribution = item_data.get("기여도(%)", 0)
                    if simulator.verbose:
                        print(f"    • {item_name}: {emission:.6f} kg CO2e ({contribution:.2f}%)")
        
        # 결과를 JSON 파일로 저장
        if simulator.verbose:
            print("\n" + "="*50)
            print("결과를 JSON 파일로 저장")
            print("="*50)
        
        import json
        from datetime import datetime
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Before 결과 저장
        before_filename = f"carbon_emission_before_{timestamp}.json"
        with open(before_filename, 'w', encoding='utf-8') as f:
            json.dump(before_emission, f, ensure_ascii=False, indent=2)
        if simulator.verbose:
            print(f"✅ Before 결과 저장: {before_filename}")
        
        # After 결과 저장
        after_filename = f"carbon_emission_after_{timestamp}.json"
        with open(after_filename, 'w', encoding='utf-8') as f:
            json.dump(after_emission, f, ensure_ascii=False, indent=2)
        if simulator.verbose:
            print(f"✅ After 결과 저장: {after_filename}")
        
    except Exception as e:
        if simulator.verbose:
            print(f"❌ 오류 발생: {e}")

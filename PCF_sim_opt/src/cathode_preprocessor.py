import json
import os
import sys
from pathlib import Path
from decimal import Decimal, getcontext

# FileOperations 사용을 위해 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from src.utils.file_operations import FileOperations, FileLoadError


class CathodePreprocessor:
    """
    양극재 전처리를 위한 클래스
    
    주요 기능:
    - cathode_ratio.json: 양극재 비율 데이터 로드
    - cathode_site.json: 양극재 생산 사이트 데이터 로드  
    - CAM.json: CAM(Cathode Active Material) 데이터 로드
    - pCAM.json: pCAM(precursor CAM) 데이터 로드
    """
    
    def __init__(self, base_path="input", verbose=False, user_id=None):
        """
        CathodePreprocessor 초기화
        
        Args:
            base_path (str): JSON 파일들이 위치한 기본 경로
            verbose (bool): 상세 로그 출력 여부. 기본값은 False
            user_id (str): 사용자 ID. 사용자별 작업공간 사용시 필요
        """
        self.base_path = Path(base_path)
        self.stable_var_path = Path("stable_var")
        self.verbose = verbose
        self.user_id = user_id
        
        # JSON 파일 경로 설정 (FileOperations가 사용자별 경로를 자동으로 처리)
        self.cathode_ratio_path = f"{base_path}/cathode_ratio.json"
        self.cathode_site_path = f"{base_path}/cathode_site.json"
        self.cam_path = "stable_var/CAM.json"
        self.pcam_path = "stable_var/pCAM.json"
        self.cathode_tier1_input_path = "stable_var/cathode_tier1_input.json"
        self.cathode_tier2_input_path = "stable_var/cathode_tier2_input.json"
        
        # 데이터 저장 변수들
        self.cathode_ratio_data = None
        self.cathode_site_data = None
        self.cam_data = None
        self.pcam_data = None
        self.cathode_tier1_input_data = None
        self.cathode_tier2_input_data = None
        
        # JSON 파일들 로드
        self._load_json_files()
    
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
    
    def _load_json_files(self):
        """모든 JSON 파일들을 로드하여 변수에 저장"""
        self._print("📁 1단계: JSON 파일 로드", level='info')
        
        try:
            loaded_files = []
            missing_files = []
            
            # cathode_ratio.json 로드
            try:
                self.cathode_ratio_data = FileOperations.load_json(self.cathode_ratio_path, user_id=self.user_id)
                loaded_files.append("cathode_ratio.json")
            except FileLoadError:
                missing_files.append("cathode_ratio.json")
                self.cathode_ratio_data = None
            
            # cathode_site.json 로드
            try:
                self.cathode_site_data = FileOperations.load_json(self.cathode_site_path, user_id=self.user_id)
                loaded_files.append("cathode_site.json")
            except FileLoadError:
                missing_files.append("cathode_site.json")
                self.cathode_site_data = None
            
            # CAM.json 로드 - stable_var 파일이므로 사용자별 처리 필요
            try:
                self.cam_data = FileOperations.load_json(self.cam_path, user_id=self.user_id)
                loaded_files.append("CAM.json")
            except FileLoadError:
                missing_files.append("CAM.json")
                self.cam_data = None
            
            # pCAM.json 로드 - stable_var 파일이므로 사용자별 처리 필요
            try:
                self.pcam_data = FileOperations.load_json(self.pcam_path, user_id=self.user_id)
                loaded_files.append("pCAM.json")
            except FileLoadError:
                missing_files.append("pCAM.json")
                self.pcam_data = None
            
            # cathode_tier1_input.json 로드 - stable_var 파일이므로 사용자별 처리 필요
            try:
                self.cathode_tier1_input_data = FileOperations.load_json(self.cathode_tier1_input_path, user_id=self.user_id)
                loaded_files.append("cathode_tier1_input.json")
            except FileLoadError:
                missing_files.append("cathode_tier1_input.json")
                self.cathode_tier1_input_data = None
            
            # cathode_tier2_input.json 로드 - stable_var 파일이므로 사용자별 처리 필요
            try:
                self.cathode_tier2_input_data = FileOperations.load_json(self.cathode_tier2_input_path, user_id=self.user_id)
                loaded_files.append("cathode_tier2_input.json")
            except FileLoadError:
                missing_files.append("cathode_tier2_input.json")
                self.cathode_tier2_input_data = None
            
            # 요약 로그 출력
            if loaded_files:
                self._print(f"✅ JSON 파일 로드 완료: {len(loaded_files)}개", level='info')
                if self.verbose == "debug":
                    for file in loaded_files:
                        self._print(f"  • {file}", level='debug')
            
            if missing_files:
                self._print(f"⚠️ 누락된 파일: {len(missing_files)}개", level='warning')
                if self.verbose == "debug":
                    for file in missing_files:
                        self._print(f"  • {file}", level='debug')
                
        except Exception as e:
            self._print(f"❌ JSON 파일 로드 중 오류 발생: {e}", level='error')
    
    def get_cathode_ratio_data(self):
        """양극재 비율 데이터 반환"""
        return self.cathode_ratio_data
    
    def get_cathode_site_data(self):
        """양극재 사이트 데이터 반환"""
        return self.cathode_site_data
    
    def get_cam_data(self):
        """CAM 데이터 반환"""
        return self.cam_data
    
    def get_pcam_data(self):
        """pCAM 데이터 반환"""
        return self.pcam_data
    
    def get_cathode_tier1_input_data(self):
        """cathode_tier1_input 데이터 반환"""
        return self.cathode_tier1_input_data
    
    def get_cathode_tier2_input_data(self):
        """cathode_tier2_input 데이터 반환"""
        return self.cathode_tier2_input_data
    
    def get_all_data(self):
        """모든 로드된 데이터를 딕셔너리로 반환"""
        return {
            'cathode_ratio': self.cathode_ratio_data,
            'cathode_site': self.cathode_site_data,
            'cam': self.cam_data,
            'pcam': self.pcam_data,
            'cathode_tier1_input': self.cathode_tier1_input_data,
            'cathode_tier2_input': self.cathode_tier2_input_data
        }
    
    def print_data_summary(self):
        """로드된 데이터 요약 정보 출력"""
        self._print("📊 CathodePreprocessor 데이터 요약", level='info')
        
        loaded_data = []
        missing_data = []
        
        if self.cathode_ratio_data:
            loaded_data.append(f"cathode_ratio ({len(self.cathode_ratio_data)}항목)")
        else:
            missing_data.append("cathode_ratio")
            
        if self.cathode_site_data:
            loaded_data.append(f"cathode_site ({len(self.cathode_site_data)}항목)")
        else:
            missing_data.append("cathode_site")
            
        if self.cam_data:
            loaded_data.append(f"cam ({len(self.cam_data)}항목)")
        else:
            missing_data.append("cam")
            
        if self.pcam_data:
            loaded_data.append(f"pcam ({len(self.pcam_data)}항목)")
        else:
            missing_data.append("pcam")
            
        if self.cathode_tier1_input_data:
            loaded_data.append(f"tier1_input ({len(self.cathode_tier1_input_data)}항목)")
        else:
            missing_data.append("tier1_input")
            
        if self.cathode_tier2_input_data:
            loaded_data.append(f"tier2_input ({len(self.cathode_tier2_input_data)}항목)")
        else:
            missing_data.append("tier2_input")
        
        if loaded_data:
            self._print(f"✅ 로드된 데이터: {', '.join(loaded_data)}", level='info')
        
        if missing_data:
            self._print(f"❌ 누락된 데이터: {', '.join(missing_data)}", level='error')
    
    def update_pcam_composition(self, suppress_logs=False):
        """
        cathode_ratio.json의 값을 pCAM.json의 조성비에 대응시켜 업데이트
        
        Args:
            suppress_logs (bool): 로그 출력을 억제할지 여부. 기본값은 False
            
        Returns:
            dict: 업데이트된 pCAM 데이터
        """
        if not self.cathode_ratio_data or not self.pcam_data:
            self._print("❌ cathode_ratio_data 또는 pcam_data가 로드되지 않았습니다.", level='error')
            return None
        
        # pCAM 데이터를 복사하여 수정
        updated_pcam = self.pcam_data.copy()
        
        if self.verbose and not suppress_logs:
            self._print("🧪 2단계: pCAM 조성비 업데이트", level='info')
        
        updated_count = 0
        missing_count = 0
        
        # cathode_ratio의 각 원소를 pCAM의 조성비에 대응
        for element, ratio in self.cathode_ratio_data.items():
            if element in updated_pcam:
                old_ratio = updated_pcam[element]["조성비"]
                updated_pcam[element]["조성비"] = ratio
                updated_count += 1
                if self.verbose == "debug" and not suppress_logs:
                    self._print(f"  📊 {element}: {old_ratio} → {ratio}", level='debug')
            else:
                missing_count += 1
                if self.verbose == "debug" and not suppress_logs:
                    self._print(f"  ⚠️ {element}: pCAM에 해당 원소가 없습니다.", level='warning')
        
        # 업데이트된 pCAM 데이터를 인스턴스 변수에 저장
        self.pcam_data = updated_pcam
        
        if self.verbose and not suppress_logs:
            self._print(f"✅ pCAM 조성비 업데이트 완료: {updated_count}개 업데이트, {missing_count}개 누락", level='info')
        return updated_pcam
    
    def get_updated_pcam_data(self):
        """
        업데이트된 pCAM 데이터 반환 (조성비가 cathode_ratio에 맞게 수정됨)
        
        Returns:
            dict: 업데이트된 pCAM 데이터
        """
        if not self.cathode_ratio_data or not self.pcam_data:
            self._print("❌ 데이터가 로드되지 않았습니다.", level='error')
            return None
        
        # 조성비 업데이트
        updated_pcam = self.update_pcam_composition()
        return updated_pcam
    
    def print_composition_comparison(self):
        """cathode_ratio와 pCAM 조성비 비교 출력"""
        if not self.cathode_ratio_data or not self.pcam_data:
            self._print("❌ 비교할 데이터가 없습니다.", level='error')
            return
        
        self._print("=" * 60, level='info')
        self._print("📊 조성비 비교 (cathode_ratio vs pCAM)", level='info')
        self._print("=" * 60, level='info')
        
        self._print(f"{'원소':<10} {'cathode_ratio':<15} {'pCAM_조성비':<15} {'일치여부':<10}", level='info')
        self._print("-" * 60, level='info')
        
        for element, ratio in self.cathode_ratio_data.items():
            if element in self.pcam_data:
                pcam_ratio = self.pcam_data[element]["조성비"]
                match_status = "✅" if ratio == pcam_ratio else "❌"
                self._print(f"{element:<10} {ratio:<15.3f} {pcam_ratio:<15.3f} {match_status:<10}", level='info')
            else:
                self._print(f"{element:<10} {ratio:<15.3f} {'N/A':<15} {'⚠️':<10}", level='info')
        
        self._print("=" * 60, level='info')
    
    def get_pcam_molar_mass(self):
        """
        pCAM 데이터에서 원소별 조성비와 분자량을 곱한 값들의 합을 계산
        
        Returns:
            float: pCAM의 몰 질량 (g/mol)
        """
        # 먼저 pCAM 조성비를 업데이트 (로그 억제)
        updated_pcam = self.update_pcam_composition(suppress_logs=True)
        
        if not updated_pcam:
            self._print("❌ pCAM 데이터를 가져올 수 없습니다.", level='error')
            return None
        
        if self.verbose:
            self._print("🧮 3단계: pCAM 몰 질량 계산", level='info')
        
        total_molar_mass = 0.0
        
        # 각 원소별로 조성비 × 분자량 계산
        for element, data in updated_pcam.items():
            composition_ratio = data["조성비"]
            molecular_weight = data["분자량"]
            contribution = composition_ratio * molecular_weight
            total_molar_mass += contribution
            
            if self.verbose == "debug":
                self._print(f"  📊 {element}: {composition_ratio} × {molecular_weight} = {contribution:.3f}", level='debug')
        
        if self.verbose:
            self._print(f"✅ pCAM 몰 질량: {total_molar_mass:.3f} g/mol", level='info')
        
        return total_molar_mass
    
    def get_pcam_molar_mass_details(self, suppress_logs=False):
        """
        pCAM 몰 질량 계산의 상세 정보를 반환
        
        Args:
            suppress_logs (bool): 로그 출력을 억제할지 여부. 기본값은 False
            
        Returns:
            dict: 계산 상세 정보
        """
        # 먼저 pCAM 조성비를 업데이트 (로그 억제)
        updated_pcam = self.update_pcam_composition(suppress_logs=True)
        
        if not updated_pcam:
            self._print("❌ pCAM 데이터를 가져올 수 없습니다.", level='error')
            return None
        
        calculation_details = []
        total_molar_mass = 0.0
        
        # 각 원소별로 조성비 × 분자량 계산
        for element, data in updated_pcam.items():
            composition_ratio = data["조성비"]
            molecular_weight = data["분자량"]
            contribution = composition_ratio * molecular_weight
            
            total_molar_mass += contribution
            calculation_details.append({
                "element": element,
                "composition_ratio": composition_ratio,
                "molecular_weight": molecular_weight,
                "contribution": contribution
            })
        
        return {
            "total_molar_mass": total_molar_mass,
            "details": calculation_details,
            "updated_pcam_data": updated_pcam
        }
    
    def update_cam_data(self, suppress_logs=False):
        """
        CAM 데이터를 업데이트:
        1. cathode_ratio에서 Ni, Co, Mn 값의 합을 CAM 데이터 내 NCM의 조성비에 넣기
        2. cathode_ratio에서 Al을 CAM 데이터 내 Al의 조성비로 넣기
        3. CAM 데이터 내의 NCM 분자량은 get_pcam_molar_mass_details의 details에서 
           element(Ni, Co, Mn) 마다 contribution을 가져와서 합산한 다음 넣기
        
        Args:
            suppress_logs (bool): 로그 출력을 억제할지 여부. 기본값은 False
            
        Returns:
            dict: 업데이트된 CAM 데이터
        """
        if not self.cathode_ratio_data or not self.cam_data:
            self._print("❌ cathode_ratio_data 또는 cam_data가 로드되지 않았습니다.", level='error')
            return None
        
        # pCAM 몰 질량 상세 정보 가져오기 (로그 억제)
        pcam_details = self.get_pcam_molar_mass_details(suppress_logs=True)
        if not pcam_details:
            self._print("❌ pCAM 몰 질량 상세 정보를 가져올 수 없습니다.", level='error')
            return None
        
        # CAM 데이터를 복사하여 수정
        updated_cam = self.cam_data.copy()
        
        if self.verbose and not suppress_logs:
            self._print("🔧 4단계: CAM 데이터 업데이트", level='info')
        
        # 1. cathode_ratio에서 Ni, Co, Mn 값의 합을 NCM의 조성비에 넣기
        ncm_elements = ['Ni', 'Co', 'Mn']
        ncm_composition_sum = 0.0
        
        for element in ncm_elements:
            if element in self.cathode_ratio_data:
                ncm_composition_sum += self.cathode_ratio_data[element]
                if self.verbose == "debug" and not suppress_logs:
                    self._print(f"  📊 {element}: {self.cathode_ratio_data[element]} (NCM 합계에 추가)", level='debug')
        
        if 'NCM' in updated_cam:
            updated_cam['NCM']['조성비'] = ncm_composition_sum
            if self.verbose == "debug" and not suppress_logs:
                self._print(f"  📊 NCM 조성비: {ncm_composition_sum}", level='debug')
        
        # 2. cathode_ratio에서 Al을 CAM 데이터 내 Al의 조성비로 넣기
        if 'Al' in self.cathode_ratio_data and 'Al' in updated_cam:
            updated_cam['Al']['조성비'] = self.cathode_ratio_data['Al']
            if self.verbose == "debug" and not suppress_logs:
                self._print(f"  📊 Al 조성비: {self.cathode_ratio_data['Al']}", level='debug')
        
        # 3. NCM 분자량 계산 (Ni, Co, Mn의 contribution 합계)
        ncm_molecular_weight = 0.0
        for detail in pcam_details['details']:
            if detail['element'] in ncm_elements:
                ncm_molecular_weight += detail['contribution']
                if self.verbose == "debug" and not suppress_logs:
                    self._print(f"  📊 {detail['element']} contribution: {detail['contribution']:.3f}", level='debug')
        
        if 'NCM' in updated_cam:
            updated_cam['NCM']['분자량'] = ncm_molecular_weight
            if self.verbose == "debug" and not suppress_logs:
                self._print(f"  📊 NCM 분자량: {ncm_molecular_weight:.3f}", level='debug')
        
        # 업데이트된 CAM 데이터를 인스턴스 변수에 저장
        self.cam_data = updated_cam
        
        if self.verbose and not suppress_logs:
            self._print("✅ CAM 데이터 업데이트 완료", level='info')
        return updated_cam
    
    def get_updated_cam_data(self):
        """
        업데이트된 CAM 데이터 반환
        
        Returns:
            dict: 업데이트된 CAM 데이터
        """
        if not self.cathode_ratio_data or not self.cam_data:
            print("❌ 데이터가 로드되지 않았습니다.")
            return None
        
        # CAM 데이터 업데이트
        updated_cam = self.update_cam_data()
        return updated_cam
    
    def get_cam_molar_mass(self):
        """
        CAM 데이터에서 원소별 조성비와 분자량을 곱한 값들의 합을 계산
        
        Returns:
            float: CAM의 몰 질량 (g/mol)
        """
        # 먼저 CAM 데이터를 업데이트 (로그 억제)
        updated_cam = self.update_cam_data(suppress_logs=True)
        
        if not updated_cam:
            self._print("❌ CAM 데이터를 가져올 수 없습니다.", level='error')
            return None
        
        if self.verbose:
            self._print("🧮 5단계: CAM 몰 질량 계산", level='info')
        
        total_molar_mass = 0.0
        
        # 각 원소별로 조성비 × 분자량 계산
        for element, data in updated_cam.items():
            composition_ratio = data["조성비"]
            molecular_weight = data["분자량"]
            contribution = composition_ratio * molecular_weight
            total_molar_mass += contribution
            
            if self.verbose == "debug":
                self._print(f"  📊 {element}: {composition_ratio} × {molecular_weight} = {contribution:.3f}", level='debug')
        
        if self.verbose:
            self._print(f"✅ CAM 몰 질량: {total_molar_mass:.3f} g/mol", level='info')
        
        return total_molar_mass
    
    def get_cam_molar_mass_details(self):
        """
        CAM 몰 질량 계산의 상세 정보를 반환
        
        Returns:
            dict: 계산 상세 정보
        """
        # 먼저 CAM 데이터를 업데이트 (로그 억제)
        updated_cam = self.update_cam_data(suppress_logs=True)
        
        if not updated_cam:
            print("❌ CAM 데이터를 가져올 수 없습니다.")
            return None
        
        calculation_details = []
        total_molar_mass = 0.0
        
        # 각 원소별로 조성비 × 분자량 계산
        for element, data in updated_cam.items():
            composition_ratio = data["조성비"]
            molecular_weight = data["분자량"]
            contribution = composition_ratio * molecular_weight
            
            total_molar_mass += contribution
            calculation_details.append({
                "element": element,
                "composition_ratio": composition_ratio,
                "molecular_weight": molecular_weight,
                "contribution": contribution
            })
        
        return {
            "total_molar_mass": total_molar_mass,
            "details": calculation_details,
            "updated_cam_data": updated_cam
        }
    
    def update_cathode_tier1_input(self, suppress_logs=False):
        """
        cathode_tier1_input.json 데이터를 수정:
        1. CAM의 분자량은 get_cam_molar_mass로 가져오기
        2. CAM의 질량을 분자량으로 나누어 몰량에 집어넣기
        3. pCAM의 분자량은 get_pcam_molar_mass로 가져오기
        4. pCAM의 몰비를 cathode_ratio에서 Ni, Co, Mn의 비율을 더해서 넣기
        5. pCAM의 몰량은 pCAM의 몰비와 CAM의 몰량을 곱해서 넣기
        6. pCAM의 질량은 pCAM의 분자량과 몰량을 곱해서 넣기
        7. LiOH.H2O의 몰량은 LiOH.H2O의 몰비와 CAM의 몰량을 곱해서 넣기
        8. LiOH.H2O의 질량은 LiOH.H2O의 분자량과 몰량을 곱해서 넣기
        9. Li2CO3의 몰비는 1-LiOH.H2O 몰비로 정의
        10. Li2CO3의 몰량과 질량은 위쪽 로직과 동일
        11. Al(OH)3의 몰비는 cathode_ratio에서 Al의 비율을 그대로 넣기
        12. Al(OH)3의 몰량과 질량은 위쪽 로직과 동일
        
        Args:
            suppress_logs (bool): 로그 출력을 억제할지 여부. 기본값은 False
            
        Returns:
            dict: 업데이트된 cathode_tier1_input 데이터
        """
        if not self.cathode_tier1_input_data or not self.cathode_ratio_data:
            self._print("❌ cathode_tier1_input_data 또는 cathode_ratio_data가 로드되지 않았습니다.", level='error')
            return None
        
        # 데이터를 복사하여 수정
        updated_tier1_input = self.cathode_tier1_input_data.copy()
        
        if self.verbose and not suppress_logs:
            self._print("🔧 6단계: Tier1 Input 데이터 업데이트", level='info')
        
        # 1. CAM의 분자량은 get_cam_molar_mass로 가져오기
        cam_molar_mass = self.get_cam_molar_mass()
        if cam_molar_mass is None:
            self._print("❌ CAM 몰 질량을 가져올 수 없습니다.", level='error')
            return None
        
        if "CAM(=Li(NCM)O2)" in updated_tier1_input:
            old_cam_mw = updated_tier1_input["CAM(=Li(NCM)O2)"]["분자량(g/mol)"]
            updated_tier1_input["CAM(=Li(NCM)O2)"]["분자량(g/mol)"] = cam_molar_mass
            if self.verbose and not suppress_logs:
                self._print(f"  📊 CAM 분자량: {old_cam_mw} → {cam_molar_mass:.3f}", level='info')
        
        # 2. CAM의 질량을 분자량으로 나누어 몰량에 집어넣기
        if "CAM(=Li(NCM)O2)" in updated_tier1_input:
            cam_mass = updated_tier1_input["CAM(=Li(NCM)O2)"]["질량(kg)"]
            cam_molar_amount = cam_mass / (cam_molar_mass / 1000)  # kg / (g/mol / 1000) = mol
            updated_tier1_input["CAM(=Li(NCM)O2)"]["몰량(mol)"] = cam_molar_amount
            if self.verbose and not suppress_logs:
                self._print(f"  📊 CAM 몰량: {cam_molar_amount:.6f} mol", level='info')
        
        # 3. pCAM의 분자량은 get_pcam_molar_mass로 가져오기
        pcam_molar_mass = self.get_pcam_molar_mass()
        if pcam_molar_mass is None:
            self._print("❌ pCAM 몰 질량을 가져올 수 없습니다.", level='error')
            return None
        
        if "pCAM(=M(OH)2)" in updated_tier1_input:
            old_pcam_mw = updated_tier1_input["pCAM(=M(OH)2)"]["분자량(g/mol)"]
            updated_tier1_input["pCAM(=M(OH)2)"]["분자량(g/mol)"] = pcam_molar_mass
            if self.verbose and not suppress_logs:
                self._print(f"  📊 pCAM 분자량: {old_pcam_mw} → {pcam_molar_mass:.3f}", level='info')
        
        # 4. pCAM의 몰비를 cathode_ratio에서 Ni, Co, Mn의 비율을 더해서 넣기
        ncm_elements = ['Ni', 'Co', 'Mn']
        pcam_molar_ratio = 0.0
        for element in ncm_elements:
            if element in self.cathode_ratio_data:
                pcam_molar_ratio += self.cathode_ratio_data[element]
                if self.verbose and not suppress_logs:
                    self._print(f"  📊 {element}: {self.cathode_ratio_data[element]} (pCAM 몰비에 추가)", level='info')
        
        if "pCAM(=M(OH)2)" in updated_tier1_input:
            old_pcam_ratio = updated_tier1_input["pCAM(=M(OH)2)"]["몰비"]
            updated_tier1_input["pCAM(=M(OH)2)"]["몰비"] = pcam_molar_ratio
            if self.verbose and not suppress_logs:
                self._print(f"  📊 pCAM 몰비: {old_pcam_ratio} → {pcam_molar_ratio}", level='info')
        
        # 5. pCAM의 몰량은 pCAM의 몰비와 CAM의 몰량을 곱해서 넣기
        if "pCAM(=M(OH)2)" in updated_tier1_input and "CAM(=Li(NCM)O2)" in updated_tier1_input:
            pcam_molar_amount = pcam_molar_ratio * cam_molar_amount
            updated_tier1_input["pCAM(=M(OH)2)"]["몰량(mol)"] = pcam_molar_amount
            if self.verbose and not suppress_logs:
                self._print(f"  📊 pCAM 몰량: {pcam_molar_amount:.6f} mol", level='info')
        
        # 6. pCAM의 질량은 pCAM의 분자량과 몰량을 곱해서 넣기
        if "pCAM(=M(OH)2)" in updated_tier1_input:
            pcam_mass = pcam_molar_mass * pcam_molar_amount / 1000  # g/mol × mol / 1000 = kg
            updated_tier1_input["pCAM(=M(OH)2)"]["질량(kg)"] = pcam_mass
            if self.verbose and not suppress_logs:
                self._print(f"  📊 pCAM 질량: {pcam_mass:.6f} kg", level='info')
        
        # 7. LiOH.H2O의 몰량은 LiOH.H2O의 몰비와 CAM의 몰량을 곱해서 넣기
        if "LiOH.H2O" in updated_tier1_input and "CAM(=Li(NCM)O2)" in updated_tier1_input:
            lioh_molar_ratio = updated_tier1_input["LiOH.H2O"]["몰비"]
            lioh_molar_amount = lioh_molar_ratio * cam_molar_amount
            updated_tier1_input["LiOH.H2O"]["몰량(mol)"] = lioh_molar_amount
            if self.verbose and not suppress_logs:
                self._print(f"  📊 LiOH.H2O 몰량: {lioh_molar_amount:.6f} mol", level='info')
        
        # 8. LiOH.H2O의 질량은 LiOH.H2O의 분자량과 몰량을 곱해서 넣기
        if "LiOH.H2O" in updated_tier1_input:
            lioh_mw = updated_tier1_input["LiOH.H2O"]["분자량(g/mol)"]
            lioh_mass = lioh_mw * lioh_molar_amount / 1000  # g/mol × mol / 1000 = kg
            updated_tier1_input["LiOH.H2O"]["질량(kg)"] = lioh_mass
            if self.verbose and not suppress_logs:
                self._print(f"  📊 LiOH.H2O 질량: {lioh_mass:.6f} kg", level='info')
        
        # 9. Li2CO3의 몰비는 1-LiOH.H2O 몰비로 정의
        if "Li2CO3" in updated_tier1_input and "LiOH.H2O" in updated_tier1_input:
            lioh_molar_ratio = updated_tier1_input["LiOH.H2O"]["몰비"]
            li2co3_molar_ratio = 1 - lioh_molar_ratio
            updated_tier1_input["Li2CO3"]["몰비"] = li2co3_molar_ratio
            if self.verbose and not suppress_logs:
                self._print(f"  📊 Li2CO3 몰비: {li2co3_molar_ratio}", level='info')
        
        # 10. Li2CO3의 몰량과 질량은 위쪽 로직과 동일
        if "Li2CO3" in updated_tier1_input and "CAM(=Li(NCM)O2)" in updated_tier1_input:
            li2co3_molar_amount = li2co3_molar_ratio * cam_molar_amount
            updated_tier1_input["Li2CO3"]["몰량(mol)"] = li2co3_molar_amount
            if self.verbose and not suppress_logs:
                self._print(f"  📊 Li2CO3 몰량: {li2co3_molar_amount:.6f} mol", level='info')
            
            li2co3_mw = updated_tier1_input["Li2CO3"]["분자량(g/mol)"]
            li2co3_mass = li2co3_mw * li2co3_molar_amount / 1000  # g/mol × mol / 1000 = kg
            updated_tier1_input["Li2CO3"]["질량(kg)"] = li2co3_mass
            if self.verbose and not suppress_logs:
                self._print(f"  📊 Li2CO3 질량: {li2co3_mass:.6f} kg", level='info')
        
        # 11. Al(OH)3의 몰비는 cathode_ratio에서 Al의 비율을 그대로 넣기
        if "Al" in self.cathode_ratio_data and "Al(OH)3" in updated_tier1_input:
            al_ratio = self.cathode_ratio_data["Al"]
            old_al_ratio = updated_tier1_input["Al(OH)3"]["몰비"]
            updated_tier1_input["Al(OH)3"]["몰비"] = al_ratio
            if self.verbose and not suppress_logs:
                self._print(f"  📊 Al(OH)3 몰비: {old_al_ratio} → {al_ratio}", level='info')
        
        # 12. Al(OH)3의 몰량과 질량은 위쪽 로직과 동일
        if "Al(OH)3" in updated_tier1_input and "CAM(=Li(NCM)O2)" in updated_tier1_input:
            al_molar_amount = al_ratio * cam_molar_amount
            updated_tier1_input["Al(OH)3"]["몰량(mol)"] = al_molar_amount
            if self.verbose and not suppress_logs:
                self._print(f"  📊 Al(OH)3 몰량: {al_molar_amount:.6f} mol", level='info')
            
            al_mw = updated_tier1_input["Al(OH)3"]["분자량(g/mol)"]
            al_mass = al_mw * al_molar_amount / 1000  # g/mol × mol / 1000 = kg
            updated_tier1_input["Al(OH)3"]["질량(kg)"] = al_mass
            if self.verbose and not suppress_logs:
                self._print(f"  📊 Al(OH)3 질량: {al_mass:.6f} kg", level='info')
        
        # 업데이트된 데이터를 인스턴스 변수에 저장
        self.cathode_tier1_input_data = updated_tier1_input
        
        if self.verbose and not suppress_logs:
            self._print("✅ Tier1 Input 데이터 업데이트 완료", level='info')
        return updated_tier1_input
    
    def get_updated_cathode_tier1_input_data(self):
        """
        업데이트된 cathode_tier1_input 데이터 반환
        
        Returns:
            dict: 업데이트된 cathode_tier1_input 데이터
        """
        if not self.cathode_tier1_input_data:
            print("❌ cathode_tier1_input_data가 로드되지 않았습니다.")
            return None
        
        # 데이터 업데이트
        updated_tier1_input = self.update_cathode_tier1_input()
        return updated_tier1_input
    
    def update_cathode_tier2_input(self, suppress_logs=False):
        """
        cathode_tier2_input.json 데이터를 수정:
        1. cathode_ratio에서 가져온 값을 Loop를 돌며 각 key값이 포함된 값을 찾아 몰비에 넣기
        2. pCAM의 분자량을 get_pcam_molar_mass로 만들어 넣기
        3. pCAM의 몰비는 1번 값에서 넣은 몰비의 합으로 넣기
        4. NaOH의 몰비는 pCAM 몰비에 곱하기 2를 해서 넣기
        5. 나머지 모든 행의 몰량과 질량은 cathode_tier1_input과 로직이 동일
        
        Args:
            suppress_logs (bool): 로그 출력을 억제할지 여부. 기본값은 False
            
        Returns:
            dict: 업데이트된 cathode_tier2_input 데이터
        """
        if not self.cathode_tier2_input_data or not self.cathode_ratio_data:
            print("❌ cathode_tier2_input_data 또는 cathode_ratio_data가 로드되지 않았습니다.")
            return None
        
        # 데이터를 복사하여 수정
        updated_tier2_input = self.cathode_tier2_input_data.copy()
        
        if self.verbose and not suppress_logs:
            self._print("🔧 7단계: Tier2 Input 데이터 업데이트", level='info')
        
        # 1. cathode_ratio에서 가져온 값을 Loop를 돌며 각 key값이 포함된 값을 찾아 몰비에 넣기
        pcam_molar_ratio_sum = 0.0
        
        for element, ratio in self.cathode_ratio_data.items():
            # 각 원소에 대응되는 화합물 찾기
            corresponding_compound = None
            for compound in updated_tier2_input.keys():
                if element in compound:
                    corresponding_compound = compound
                    break
            
            if corresponding_compound:
                old_ratio = updated_tier2_input[corresponding_compound]["몰비"]
                updated_tier2_input[corresponding_compound]["몰비"] = ratio
                pcam_molar_ratio_sum += ratio
                if self.verbose and not suppress_logs:
                    print(f"  📊 {element} → {corresponding_compound}: {old_ratio} → {ratio}")
            else:
                if self.verbose and not suppress_logs:
                    print(f"  ⚠️ {element}에 대응되는 화합물을 찾을 수 없습니다.")
        
        # 2. pCAM의 분자량을 get_pcam_molar_mass로 만들어 넣기
        pcam_molar_mass = self.get_pcam_molar_mass()
        if pcam_molar_mass is None:
            print("❌ pCAM 몰 질량을 가져올 수 없습니다.")
            return None
        
        if "pCAM(=M(OH)2)" in updated_tier2_input:
            old_pcam_mw = updated_tier2_input["pCAM(=M(OH)2)"]["분자량(g/mol)"]
            updated_tier2_input["pCAM(=M(OH)2)"]["분자량(g/mol)"] = pcam_molar_mass
            if self.verbose and not suppress_logs:
                print(f"  📊 pCAM 분자량: {old_pcam_mw} → {pcam_molar_mass:.3f}")
        
        # 3. pCAM의 몰비는 1번 값에서 넣은 몰비의 합으로 넣기
        if "pCAM(=M(OH)2)" in updated_tier2_input:
            old_pcam_ratio = updated_tier2_input["pCAM(=M(OH)2)"]["몰비"]
            updated_tier2_input["pCAM(=M(OH)2)"]["몰비"] = pcam_molar_ratio_sum
            if self.verbose and not suppress_logs:
                print(f"  📊 pCAM 몰비: {old_pcam_ratio} → {pcam_molar_ratio_sum}")
        
        # 4. NaOH의 몰비는 pCAM 몰비에 곱하기 2를 해서 넣기
        if "NaOH" in updated_tier2_input:
            naoh_molar_ratio = pcam_molar_ratio_sum * 2
            old_naoh_ratio = updated_tier2_input["NaOH"]["몰비"]
            updated_tier2_input["NaOH"]["몰비"] = naoh_molar_ratio
            if self.verbose and not suppress_logs:
                print(f"  📊 NaOH 몰비: {old_naoh_ratio} → {naoh_molar_ratio}")
        
        # 5. 나머지 모든 행의 몰량과 질량은 cathode_tier1_input과 로직이 동일
        # CAM의 몰량을 기준으로 계산 (cathode_tier1_input에서 가져옴)
        cam_molar_amount = None
        if self.cathode_tier1_input_data and "CAM(=Li(NCM)O2)" in self.cathode_tier1_input_data:
            cam_molar_amount = self.cathode_tier1_input_data["CAM(=Li(NCM)O2)"]["몰량(mol)"]
            if self.verbose and not suppress_logs:
                print(f"  📊 CAM 몰량 (Tier1에서 가져옴): {cam_molar_amount:.6f} mol")
        
        if cam_molar_amount is None:
            print("❌ CAM 몰량을 가져올 수 없습니다.")
            return None
        
        # 각 화합물의 몰량과 질량 계산
        for compound, data in updated_tier2_input.items():
            if compound == "pCAM(=M(OH)2)":
                # pCAM은 특별 처리 (이미 위에서 처리됨)
                continue
            
            molar_ratio = data["몰비"]
            molecular_weight = data["분자량(g/mol)"]
            
            # 몰량 계산
            molar_amount = molar_ratio * cam_molar_amount
            updated_tier2_input[compound]["몰량(mol)"] = molar_amount
            if self.verbose and not suppress_logs:
                print(f"  📊 {compound} 몰량: {molar_amount:.6f} mol")
            
            # 질량 계산
            mass = molecular_weight * molar_amount / 1000  # g/mol × mol / 1000 = kg
            updated_tier2_input[compound]["질량(kg)"] = mass
            if self.verbose and not suppress_logs:
                print(f"  📊 {compound} 질량: {mass:.6f} kg")
        
        # pCAM의 몰량과 질량 계산
        if "pCAM(=M(OH)2)" in updated_tier2_input:
            pcam_molar_amount = pcam_molar_ratio_sum * cam_molar_amount
            updated_tier2_input["pCAM(=M(OH)2)"]["몰량(mol)"] = pcam_molar_amount
            if self.verbose and not suppress_logs:
                print(f"  📊 pCAM 몰량: {pcam_molar_amount:.6f} mol")
            
            # pCAM 질량은 cathode_tier1_input에서 계산된 값을 가져오기
            tier1_updated_data = self.get_updated_cathode_tier1_input_data()
            if tier1_updated_data and "pCAM(=M(OH)2)" in tier1_updated_data:
                pcam_mass_from_tier1 = tier1_updated_data["pCAM(=M(OH)2)"]["질량(kg)"]
                updated_tier2_input["pCAM(=M(OH)2)"]["질량(kg)"] = pcam_mass_from_tier1
                if self.verbose and not suppress_logs:
                    print(f"  📊 pCAM 질량 (Tier1에서 가져옴): {pcam_mass_from_tier1:.6f} kg")
            else:
                # Tier1에서 가져올 수 없는 경우 직접 계산
                pcam_mass = pcam_molar_mass * pcam_molar_amount / 1000
                updated_tier2_input["pCAM(=M(OH)2)"]["질량(kg)"] = pcam_mass
                if self.verbose and not suppress_logs:
                    print(f"  📊 pCAM 질량 (직접 계산): {pcam_mass:.6f} kg")
        
        # 업데이트된 데이터를 인스턴스 변수에 저장
        self.cathode_tier2_input_data = updated_tier2_input
        
        if self.verbose and not suppress_logs:
            self._print("✅ Tier2 Input 데이터 업데이트 완료", level='info')
        return updated_tier2_input
    
    def get_updated_cathode_tier2_input_data(self):
        """
        업데이트된 cathode_tier2_input 데이터 반환
        
        Returns:
            dict: 업데이트된 cathode_tier2_input 데이터
        """
        if not self.cathode_tier2_input_data:
            print("❌ cathode_tier2_input_data가 로드되지 않았습니다.")
            return None
        
        # 데이터 업데이트
        updated_tier2_input = self.update_cathode_tier2_input()
        return updated_tier2_input

    def comprehensive_analysis(self, verbose_output=True):
        """
        CathodePreprocessor의 모든 기능을 종합적으로 활용하여 완전한 분석을 수행합니다.
        
        Args:
            verbose_output (bool): 상세한 출력을 표시할지 여부. 기본값은 True
            
        Returns:
            dict: 모든 분석 결과를 포함한 딕셔너리
        """
        if verbose_output:
            self._print("🔬 8단계: 종합 분석", level='info')
        
        # 1. 기본 데이터 요약
        self.print_data_summary()
        
        # 2. pCAM 분석
        if verbose_output:
            self._print("📊 pCAM 분석 중...", level='info')
        
        updated_pcam = self.update_pcam_composition()
        pcam_molar_mass = self.get_pcam_molar_mass()
        pcam_details = self.get_pcam_molar_mass_details()
        
        # 3. CAM 분석
        if verbose_output:
            self._print("📊 CAM 분석 중...", level='info')
        
        updated_cam = self.update_cam_data()
        cam_molar_mass = self.get_cam_molar_mass()
        cam_details = self.get_cam_molar_mass_details()
        
        # 4. Tier1 Input 분석
        if verbose_output:
            self._print("📊 Tier1 Input 분석 중...", level='info')
        
        updated_tier1 = self.get_updated_cathode_tier1_input_data()
        
        # 5. Tier2 Input 분석
        if verbose_output:
            self._print("📊 Tier2 Input 분석 중...", level='info')
        
        updated_tier2 = self.get_updated_cathode_tier2_input_data()
        
        # 6. 종합 요약
        total_tier1_mass = sum(data['질량(kg)'] for data in updated_tier1.values()) if updated_tier1 else 0
        total_tier2_mass = sum(data['질량(kg)'] for data in updated_tier2.values()) if updated_tier2 else 0
        
        if verbose_output:
            self._print("📈 분석 결과 요약:", level='info')
            self._print(f"  • pCAM 몰 질량: {pcam_molar_mass:.3f} g/mol", level='info')
            self._print(f"  • CAM 몰 질량: {cam_molar_mass:.3f} g/mol", level='info')
            self._print(f"  • Tier1 총 질량: {total_tier1_mass:.6f} kg", level='info')
            self._print(f"  • Tier2 총 질량: {total_tier2_mass:.6f} kg", level='info')
            self._print(f"  • 전체 총 질량: {total_tier1_mass + total_tier2_mass:.6f} kg", level='info')
        
        # 7. 결과 반환
        analysis_results = {
            'basic_data': {
                'cathode_ratio': self.get_cathode_ratio_data(),
                'cathode_site': self.get_cathode_site_data(),
                'cam_data': self.get_cam_data(),
                'pcam_data': self.get_pcam_data(),
                'tier1_input': self.get_cathode_tier1_input_data(),
                'tier2_input': self.get_cathode_tier2_input_data()
            },
            'updated_data': {
                'pcam': updated_pcam,
                'cam': updated_cam,
                'tier1_input': updated_tier1,
                'tier2_input': updated_tier2
            },
            'molar_mass_analysis': {
                'pcam_molar_mass': pcam_molar_mass,
                'pcam_details': pcam_details,
                'cam_molar_mass': cam_molar_mass,
                'cam_details': cam_details
            },
            'mass_summary': {
                'total_tier1_mass': total_tier1_mass,
                'total_tier2_mass': total_tier2_mass,
                'total_mass': total_tier1_mass + total_tier2_mass
            }
        }
        
        if verbose_output:
            self._print("✅ CathodePreprocessor 종합 분석 완료", level='info')
        
        return analysis_results
    
    def quick_analysis(self, verbose_output=False):
        """
        빠른 분석을 위한 간단한 함수 (로그 최소화)
        
        Args:
            verbose_output (bool): 상세한 출력을 표시할지 여부. 기본값은 False
            
        Returns:
            dict: 핵심 분석 결과만 포함한 딕셔너리
        """
        if verbose_output:
            self._print("⚡ 8단계: 빠른 분석", level='info')
        
        # 임시로 verbose 설정 변경
        original_verbose = self.verbose
        self.verbose = verbose_output
        
        try:
            # 핵심 데이터만 업데이트
            updated_pcam = self.update_pcam_composition(suppress_logs=True)
            updated_cam = self.update_cam_data(suppress_logs=True)
            updated_tier1 = self.get_updated_cathode_tier1_input_data()
            updated_tier2 = self.get_updated_cathode_tier2_input_data()
            
            # 핵심 지표 계산
            pcam_molar_mass = self.get_pcam_molar_mass()
            cam_molar_mass = self.get_cam_molar_mass()
            
            total_tier1_mass = sum(data['질량(kg)'] for data in updated_tier1.values()) if updated_tier1 else 0
            total_tier2_mass = sum(data['질량(kg)'] for data in updated_tier2.values()) if updated_tier2 else 0
            
            quick_results = {
                'molar_masses': {
                    'pcam': pcam_molar_mass,
                    'cam': cam_molar_mass
                },
                'total_masses': {
                    'tier1': total_tier1_mass,
                    'tier2': total_tier2_mass,
                    'total': total_tier1_mass + total_tier2_mass
                },
                'composition_ratios': self.get_cathode_ratio_data()
            }
            
            if verbose_output:
                self._print("📊 빠른 분석 결과:", level='info')
                self._print(f"  • pCAM 몰 질량: {pcam_molar_mass:.3f} g/mol", level='info')
                self._print(f"  • CAM 몰 질량: {cam_molar_mass:.3f} g/mol", level='info')
                self._print(f"  • 총 질량: {total_tier1_mass + total_tier2_mass:.6f} kg", level='info')
            
            return quick_results
            
        finally:
            # 원래 verbose 설정 복원
            self.verbose = original_verbose

"""
최적화 모델에서 사용되는 상수값을 관리하는 클래스
"""

from typing import Dict, Any, Optional, Union, List
import yaml
from pathlib import Path
import json


class OptimizationConstants:
    """
    최적화에 사용되는 모든 상수값을 관리하는 클래스
    
    주요 상수:
    - 기본 배출량
    - 국가별 전력배출계수
    - 재활용재 환경영향 계수
    - 감축활동 효과 계수
    - 감축활동별 비용
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로 (YAML 또는 JSON)
        """
        self.constants = {}
        self.load_default_constants()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_default_constants(self) -> None:
        """기본 상수값 설정"""
        self.constants = {
            "base_emission": 80.0,  # 기본 배출량 (kg CO2eq/kWh)
            
            "electricity_coef": {   # 국가별 전력배출계수 (kg CO2eq/kWh)
                "한국": 0.637420635,
                "중국": 0.8825,
                "일본": 0.667861719,
                "폴란드": 0.948984701,
                "미국": 0.522,
                "유럽": 0.432
            },
            
            "recycle_impact": {     # 재활용재 환경영향 계수 (신재 대비 비율)
                "신재": 1.0,
                "재활용재": {
                    "Ni": 0.1,
                    "Co": 0.15,
                    "Li": 0.1
                }
            },
            
            "reduction_effect_coefficient": 0.1,  # 감축활동 효과 계수 (감축비율 1%당 효과)
            
            "activity_costs": {     # 감축활동별 고정 비용 (원)
                "tier1_양극재": 10000,
                "tier1_분리막": 8000,
                "tier1_전해액": 7000,
                "tier2_양극재": 12000,
                "tier2_저탄소원료": 15000,
                "tier2_전구체": 9000,
                "tier3_니켈원료": 20000,
                "tier3_코발트": 25000
            },
            
            "variable_cost_per_percent": 50,  # 감축비율 1%당 가변 비용 (원)
            
            "material_costs": {     # 원료별 단위 비용 (원/단위)
                "recycle_material_cost": 500,
                "low_carbon_material_cost": 800
            }
        }
    
    def load_from_file(self, file_path: str) -> None:
        """
        외부 파일에서 상수 설정 로드
        
        Args:
            file_path: 설정 파일 경로 (YAML 또는 JSON)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"상수 설정 파일을 찾을 수 없습니다: {file_path}")
        
        # 파일 형식에 따라 로드
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                constants = yaml.safe_load(f)
                if constants.get('constants'):
                    self.update_constants(constants['constants'])
                else:
                    self.update_constants(constants)
        else:  # JSON으로 간주
            with open(path, 'r', encoding='utf-8') as f:
                constants = json.load(f)
                if constants.get('constants'):
                    self.update_constants(constants['constants'])
                else:
                    self.update_constants(constants)
    
    def update_constants(self, new_constants: Dict[str, Any]) -> None:
        """
        상수값 업데이트
        
        Args:
            new_constants: 새 상수값 딕셔너리
        """
        # 중첩 딕셔너리를 위한 재귀적 업데이트
        def update_dict_recursive(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_dict_recursive(target[key], value)
                else:
                    target[key] = value
        
        update_dict_recursive(self.constants, new_constants)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        상수값 조회
        
        Args:
            key: 조회할 상수 키
            default: 키가 없을 경우 반환할 기본값
            
        Returns:
            Any: 상수값
        """
        # 중첩 키 지원 (예: "electricity_coef.한국")
        if '.' in key:
            parts = key.split('.')
            value = self.constants
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        return self.constants.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        상수값 설정
        
        Args:
            key: 설정할 상수 키
            value: 설정할 값
        """
        # 중첩 키 지원
        if '.' in key:
            parts = key.split('.')
            target = self.constants
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            self.constants[key] = value
    
    def export_to_yaml(self, file_path: str) -> None:
        """
        상수 설정을 YAML 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump({'constants': self.constants}, f, default_flow_style=False, allow_unicode=True)
    
    def export_to_json(self, file_path: str) -> None:
        """
        상수 설정을 JSON 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'constants': self.constants}, f, ensure_ascii=False, indent=2)
    
    def get_location_factor(self, location: str) -> float:
        """
        생산지 전력배출계수 조회
        
        Args:
            location: 생산지 국가
            
        Returns:
            float: 해당 국가의 전력배출계수
        """
        electricity_coefs = self.get('electricity_coef', {})
        return electricity_coefs.get(location, 0.637)  # 기본값은 한국
    
    def get_recycle_impact(self, element: str = 'Ni') -> float:
        """
        재활용재 환경영향 계수 조회
        
        Args:
            element: 원소 (Ni, Co, Li)
            
        Returns:
            float: 환경영향 계수
        """
        recycle_impact = self.get('recycle_impact', {}).get('재활용재', {})
        return recycle_impact.get(element, 0.1)  # 기본값
    
    def get_activity_cost(self, activity_name: str) -> float:
        """
        감축활동 비용 조회
        
        Args:
            activity_name: 활동명
            
        Returns:
            float: 활동 비용
        """
        activity_costs = self.get('activity_costs', {})
        return activity_costs.get(activity_name, 10000)  # 기본값
        
    def get_tier_cost(self, tier: str, material: str, country: str = None) -> float:
        """
        특정 티어와 소재에 대한 비용 조회
        
        Args:
            tier: 티어 구분 (Tier1, Tier2 등)
            material: 소재 구분 (양극재, 음극재 등)
            country: 국가 (옵션)
        
        Returns:
            float: 해당 티어와 소재의 비용
        """
        # 1. stable_var에서 cost_by_tier 데이터 활용 (외부에서 주입)
        tier_data = self.get('cost_by_tier', {}).get('tier_data', [])
        
        # 해당 티어와 소재 검색
        cost = 0.0
        if country:
            # 국가까지 일치하는 경우 검색
            for item in tier_data:
                if item.get('tier') == tier and material in item.get('material', '') and item.get('country') == country:
                    cost = item.get('expected_cost', 0.0)
                    break
        else:
            # 국가 무관 첫 번째 매칭 항목
            for item in tier_data:
                if item.get('tier') == tier and material in item.get('material', ''):
                    cost = item.get('expected_cost', 0.0)
                    break
                    
        # 2. 데이터가 없으면 기존 activity_costs 참조
        if cost == 0.0:
            activity_key = f"{tier.lower()}_{material}"
            cost = self.get_activity_cost(activity_key)
            
        return cost
        
    def get_material_cost_by_tier(self, tier: str, material: str) -> Dict[str, Any]:
        """
        특정 티어와 소재에 대한 상세 비용 정보 조회
        
        Args:
            tier: 티어 구분 (Tier1, Tier2 등)
            material: 소재 구분 (양극재, 음극재 등)
        
        Returns:
            Dict: 해당 티어와 소재의 비용 정보
        """
        tier_data = self.get('cost_by_tier', {}).get('tier_data', [])
        
        # 해당 티어와 소재의 모든 국가별 비용 수집
        result = {
            'material': material,
            'tier': tier,
            'costs': []
        }
        
        for item in tier_data:
            if item.get('tier') == tier and material in item.get('material', ''):
                country_data = {
                    'country': item.get('country', ''),
                    'cost': item.get('expected_cost', 0.0),
                    'power_unit': item.get('power_unit', 0.0),
                    're_certification': item.get('re_certification', ''),
                    'cert_price': item.get('cert_price', 0.0),
                    'note': item.get('note', '')
                }
                result['costs'].append(country_data)
        
        return result
    
    def get_all_constants(self) -> Dict[str, Any]:
        """모든 상수 반환"""
        return self.constants
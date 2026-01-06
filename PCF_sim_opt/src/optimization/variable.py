"""
최적화 모델의 의사결정 변수를 관리하는 클래스
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import yaml
import json
from pyomo.environ import ConcreteModel, Var, Binary, Expression


class OptimizationVariables:
    """
    최적화 모델의 의사결정 변수 정의 및 관리
    
    주요 변수 유형:
    - 감축비율 변수 (Tier별)
    - 양극재 구성 변수 (비율 또는 원료구성)
    - 활성화 여부 변수 (이진 변수)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로 (YAML 또는 JSON)
        """
        self.variable_config = {}
        self.model = None
        self.load_default_config()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_default_config(self) -> None:
        """기본 변수 설정"""
        self.variable_config = {
            "reduction_rates": {
                "tier1_양극재": {
                    "min": 0,
                    "max": 100,
                    "default": 20,
                    "description": "Tier1 양극재 감축비율 (%)"
                },
                "tier1_분리막": {
                    "min": 0,
                    "max": 100,
                    "default": 15,
                    "description": "Tier1 분리막 감축비율 (%)"
                },
                "tier1_전해액": {
                    "min": 0,
                    "max": 100,
                    "default": 25,
                    "description": "Tier1 전해액 감축비율 (%)"
                },
                "tier2_양극재": {
                    "min": 0,
                    "max": 100,
                    "default": 30,
                    "description": "Tier2 양극재 감축비율 (%)"
                },
                "tier2_저탄소원료": {
                    "min": 0,
                    "max": 100,
                    "default": 35,
                    "description": "Tier2 저탄소원료 감축비율 (%)"
                },
                "tier2_전구체": {
                    "min": 0,
                    "max": 100,
                    "default": 20,
                    "description": "Tier2 전구체 감축비율 (%)"
                },
                "tier3_니켈원료": {
                    "min": 0,
                    "max": 100,
                    "default": 40,
                    "description": "Tier3 니켈원료 감축비율 (%)"
                },
                "tier3_코발트": {
                    "min": 0,
                    "max": 100,
                    "default": 35,
                    "description": "Tier3 코발트 감축비율 (%)"
                }
            },
            "cathode": {
                "type": "B",  # A: 원료구성 변수, B: 비율 변수
                "type_A_config": {
                    "recycle_ratio_fixed": 0.2,
                    "low_carbon_ratio_fixed": 0.1,
                    "emission_range": [5.0, 15.0],
                    "description": "Type A: 재활용재/저탄소원료 비율 고정, 배출계수 최적화"
                },
                "type_B_config": {
                    "emission_fixed": 10.0,
                    "recycle_range": [0.1, 0.5],
                    "low_carbon_range": [0.05, 0.3],
                    "description": "Type B: 배출계수 고정, 재활용재/저탄소원료 비율 최적화"
                }
            },
            "use_binary_variables": False  # 이진 변수 사용 여부 (CBC 솔버용)
        }
    
    def load_from_file(self, file_path: str) -> None:
        """
        외부 파일에서 변수 설정 로드
        
        Args:
            file_path: 설정 파일 경로 (YAML 또는 JSON)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"변수 설정 파일을 찾을 수 없습니다: {file_path}")
        
        # 파일 형식에 따라 로드
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config.get('decision_vars'):
                    self.update_config(config['decision_vars'])
                else:
                    self.update_config(config)
        else:  # JSON으로 간주
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if config.get('decision_vars'):
                    self.update_config(config['decision_vars'])
                else:
                    self.update_config(config)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        변수 설정 업데이트
        
        Args:
            new_config: 새 변수 설정 딕셔너리
        """
        # 중첩 딕셔너리를 위한 재귀적 업데이트
        def update_dict_recursive(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_dict_recursive(target[key], value)
                else:
                    target[key] = value
        
        update_dict_recursive(self.variable_config, new_config)
    
    def define_variables(self, model: ConcreteModel) -> ConcreteModel:
        """
        Pyomo 모델에 의사결정 변수 정의
        
        Args:
            model: Pyomo 구체적 모델
            
        Returns:
            ConcreteModel: 변수가 정의된 모델
        """
        self.model = model
        
        # 1. 감축비율 변수 정의
        self._define_reduction_variables()
        
        # 2. 양극재 구성 변수 정의
        self._define_cathode_variables()
        
        return model
    
    def _define_reduction_variables(self) -> None:
        """감축비율 변수 정의"""
        reduction_vars = self.variable_config.get('reduction_rates', {})
        use_binary = self.variable_config.get('use_binary_variables', False)
        
        # 최적화 대상 변수 확인
        optimize_reduction_rates = self.variable_config.get('optimize_reduction_rates', True)
        
        for var_name, config in reduction_vars.items():
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            
            # cap 값이 설정되어 있으면 max_val을 cap으로 제한
            if 'cap' in config:
                cap_val = config.get('cap')
                if cap_val is not None and cap_val < max_val:
                    max_val = cap_val
            
            if optimize_reduction_rates:
                # 연속 변수로 정의 (최적화 대상)
                setattr(self.model, var_name, Var(bounds=(min_val, max_val)))
            else:
                # 고정 값으로 설정 (기본값 사용)
                default_val = config.get('default', 0)
                if default_val > max_val:
                    default_val = max_val
                setattr(self.model, var_name, default_val)
            
            # 이진 변수 정의 (선택적)
            if use_binary:
                setattr(self.model, f"{var_name}_active", Var(domain=Binary))
                
                # 활성화 제약조건 추가
                @self.model.Constraint
                def activity_link_lower(m, var=var_name):
                    return getattr(m, var) >= 0
                
                @self.model.Constraint
                def activity_link_upper(m, var=var_name):
                    var_obj = getattr(m, var)
                    active_var = getattr(m, f"{var}_active")
                    # Var 객체인 경우에만 제약조건 적용
                    if isinstance(var_obj, Var):
                        return var_obj <= max_val * active_var
                    return Constraint.Skip
    
    def _define_cathode_variables(self) -> None:
        """양극재 구성 변수 정의"""
        cathode_config = self.variable_config.get('cathode', {})
        cathode_type = cathode_config.get('type', 'B')
        
        if cathode_type == 'A':
            # Type A: 원료구성이 변수, 비율은 고정
            type_a_config = cathode_config.get('type_A_config', {})
            emission_range = type_a_config.get('emission_range', [5.0, 15.0])
            
            # 배출계수 변수
            self.model.low_carbon_emission = Var(bounds=emission_range)
            
            # 비율은 고정값으로 설정
            recycle_ratio = type_a_config.get('recycle_ratio_fixed', 0.2)
            low_carbon_ratio = type_a_config.get('low_carbon_ratio_fixed', 0.1)
            
            print(f"Type A 양극재 구성 설정: recycle_ratio={recycle_ratio}, low_carbon_ratio={low_carbon_ratio}")
            
            # **중요**: 단순히 값을 직접 할당 (설정 값 그대로 사용)
            # 이것이 '호출' 문제가 없는 가장 단순한 방법
            self.model.recycle_ratio = float(recycle_ratio)
            self.model.low_carbon_ratio = float(low_carbon_ratio)
            self.model.new_material_ratio = 1.0 - float(recycle_ratio) - float(low_carbon_ratio)
            
            print(f"Type A 설정 완료: recycle_ratio={self.model.recycle_ratio}, low_carbon_ratio={self.model.low_carbon_ratio}")
            print(f"Type A 설정 완료: 유형 = recycle_ratio: {type(self.model.recycle_ratio)}, low_carbon_ratio: {type(self.model.low_carbon_ratio)}")
            
            # 모듈 사용은 중복 처리를 방지하기 위해 주석 처리
            # try:
            #    from .type_a_cathode_fix import fix_type_a_cathode_variables
            #    fix_type_a_cathode_variables(self.model, type_a_config)
            # except ImportError:
            #    # 수정 모듈을 가져올 수 없는 경우 기본 구현 사용 이미 위에서 처리했음
            
        else:  # Type B
            # Type B: 비율이 변수, 원료구성은 고정
            type_b_config = cathode_config.get('type_B_config', {})
            recycle_range = type_b_config.get('recycle_range', [0.1, 0.5])
            low_carbon_range = type_b_config.get('low_carbon_range', [0.05, 0.3])
            
            # 비율 변수
            self.model.recycle_ratio = Var(bounds=recycle_range)
            self.model.low_carbon_ratio = Var(bounds=low_carbon_range)
            
            # 배출계수는 고정값
            self.model.low_carbon_emission = type_b_config.get('emission_fixed', 10.0)
            
            # 신재 비율은 표현식으로 정의
            @self.model.Expression
            def new_material_ratio(m):
                return 1 - m.recycle_ratio - m.low_carbon_ratio
    
    def get_variable_by_name(self, var_name: str) -> Optional[Var]:
        """
        변수명으로 Pyomo 변수 객체 조회
        
        Args:
            var_name: 변수명
            
        Returns:
            Optional[Var]: Pyomo 변수 객체
        """
        if not self.model:
            raise ValueError("모델이 정의되지 않았습니다. define_variables()를 먼저 호출하세요.")
            
        return getattr(self.model, var_name, None)
    
    def get_reduction_variables(self) -> Dict[str, Var]:
        """
        감축비율 변수 딕셔너리 반환
        
        Returns:
            Dict[str, Var]: 변수명과 Pyomo 변수 객체의 매핑
        """
        if not self.model:
            return {}
            
        reduction_vars = self.variable_config.get('reduction_rates', {})
        result = {}
        
        for var_name in reduction_vars.keys():
            if hasattr(self.model, var_name):
                result[var_name] = getattr(self.model, var_name)
                
        return result
    
    def get_cathode_type(self) -> str:
        """
        양극재 프로젝트 유형 반환
        
        Returns:
            str: 'A' 또는 'B'
        """
        return self.variable_config.get('cathode', {}).get('type', 'B')
    
    def get_cathode_config(self) -> Dict[str, Any]:
        """
        양극재 설정 반환
        
        Returns:
            Dict: 양극재 설정 딕셔너리
        """
        return self.variable_config.get('cathode', {})
    
    def export_to_yaml(self, file_path: str) -> None:
        """
        변수 설정을 YAML 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump({'decision_vars': self.variable_config}, f, default_flow_style=False, allow_unicode=True)
    
    def export_to_json(self, file_path: str) -> None:
        """
        변수 설정을 JSON 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'decision_vars': self.variable_config}, f, ensure_ascii=False, indent=2)
    
    def get_all_config(self) -> Dict[str, Any]:
        """모든 변수 설정 반환"""
        return self.variable_config
        
    def set_reduction_rate_caps(self, caps: Dict[str, float]) -> None:
        """
        특정 tier별 감축 비율에 대한 cap 설정
        
        Args:
            caps: 변수명과 cap 값의 딕셔너리 (예: {'tier1_양극재': 30.0})
        """
        reduction_vars = self.variable_config.get('reduction_rates', {})
        
        for var_name, cap_value in caps.items():
            if var_name in reduction_vars:
                # 기존 설정에 cap 값 추가
                reduction_vars[var_name]['cap'] = cap_value
            else:
                # 새로운 변수 추가 (기본 설정 포함)
                reduction_vars[var_name] = {
                    'min': 0,
                    'max': 100,
                    'default': min(cap_value, 20),  # 기본값은 cap 이하로 설정
                    'cap': cap_value,
                    'description': f"{var_name} 감축비율 cap (%)"
                }
    
    def set_optimize_reduction_rates(self, optimize: bool) -> None:
        """
        감축 비율을 최적화 변수로 사용할지 여부 설정
        
        Args:
            optimize: True면 감축 비율을 최적화 변수로 사용, False면 고정값 사용
        """
        self.variable_config['optimize_reduction_rates'] = optimize
        
    def set_material_specific_caps(self, material_caps: Dict[str, Dict[str, float]]) -> None:
        """
        소재별로 다른 cap 값 설정
        
        Args:
            material_caps: 소재명과 tier별 cap 값의 딩셔너리
            (예: {'양극재': {'tier1': 30.0, 'tier2': 40.0}, '분리막': {...}})
        """
        for material, caps in material_caps.items():
            for tier, cap_value in caps.items():
                var_name = f"{tier}_{material}"
                # 이미 존재하는 변수인지 확인
                if var_name in self.variable_config.get('reduction_rates', {}):
                    self.variable_config['reduction_rates'][var_name]['cap'] = cap_value
                else:
                    # 새로운 변수 추가
                    if 'reduction_rates' not in self.variable_config:
                        self.variable_config['reduction_rates'] = {}
                    
                    self.variable_config['reduction_rates'][var_name] = {
                        'min': 0,
                        'max': 100,
                        'default': min(cap_value, 20),
                        'cap': cap_value,
                        'description': f"{tier.capitalize()} {material} 감축비율 cap (%)"
                    }
                    
    def apply_material_specific_configs(self, material_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        소재별 최적화 설정 적용
        
        Args:
            material_configs: 소재별 설정 딥셔너리
            (예: {'양극재': {'strategy': 'minimize_carbon', 'reduction_caps': {...}}})
        """
        for material, config in material_configs.items():
            # 감축량 cap 적용
            if 'reduction_caps' in config:
                material_caps = {material: config['reduction_caps']}
                self.set_material_specific_caps(material_caps)
                
            # 추가 구성 (예: 양극재의 경우 타입 설정)
            if material == '양극재' and 'type' in config:
                if 'cathode' not in self.variable_config:
                    self.variable_config['cathode'] = {}
                self.variable_config['cathode']['type'] = config['type']
                
    def get_material_specific_variables(self, material: str) -> Dict[str, Dict[str, Any]]:
        """
        특정 소재와 관련된 변수만 추출
        
        Args:
            material: 소재 이름 (예: '양극재', '분리막' 등)
            
        Returns:
            Dict: 해당 소재와 관련된 변수만 포함한 딥셔너리
        """
        reduction_vars = self.variable_config.get('reduction_rates', {})
        material_vars = {}
        
        for var_name, config in reduction_vars.items():
            # 변수명에서 소재 추출 (예: tier1_양극재 -> 양극재)
            if '_' in var_name:
                parts = var_name.split('_', 1)
                if len(parts) > 1 and parts[1] == material:
                    material_vars[var_name] = config
        
        return {'reduction_rates': material_vars}
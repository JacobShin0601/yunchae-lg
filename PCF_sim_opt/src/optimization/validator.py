"""
최적화 입력 및 모델의 유효성을 검증하는 모듈
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import yaml
import json
from .input import OptimizationInput


class Validator:
    """
    최적화 입력 데이터 및 모델의 유효성을 검증하는 클래스
    
    주요 기능:
    - 설정 파일 유효성 검증
    - 변수 및 제약조건 유효성 검증
    - 결과 유효성 검증
    """
    
    def __init__(self, opt_input: Optional[OptimizationInput] = None):
        """
        Args:
            opt_input: 최적화 입력 객체 (선택 사항)
        """
        self.opt_input = opt_input
        self.validation_rules = {}
        self.validation_errors = []
        
        # 기본 검증 규칙 로드
        self.load_default_validation_rules()
    
    def set_opt_input(self, opt_input: OptimizationInput) -> None:
        """
        최적화 입력 객체 설정
        
        Args:
            opt_input: 최적화 입력 객체
        """
        self.opt_input = opt_input
        
        # 입력 객체에서 검증 규칙 추출
        if self.opt_input:
            config = self.opt_input.get_config()
            if 'validation_rules' in config:
                self.update_validation_rules(config['validation_rules'])
    
    def load_default_validation_rules(self) -> None:
        """기본 검증 규칙 설정"""
        self.validation_rules = {
            "required_fields": [
                "objective",
                "decision_vars",
                "constraints"
            ],
            "valid_objectives": [
                "minimize_carbon",
                "minimize_cost",
                "maximize_ease",
                "multi_objective"
            ],
            "cathode_type_requirements": {
                "A": [
                    "recycle_ratio_fixed",
                    "low_carbon_ratio_fixed", 
                    "emission_range"
                ],
                "B": [
                    "emission_fixed",
                    "recycle_range",
                    "low_carbon_range"
                ]
            },
            "constraint_minimums": {
                "target_carbon": 10.0,
                "max_activities": 1,
                "max_cost": 1000
            },
            "value_ranges": {
                "decision_vars.cathode.type_B_config.recycle_range": [0, 1],
                "decision_vars.cathode.type_B_config.low_carbon_range": [0, 1],
                "decision_vars.cathode.type_A_config.recycle_ratio_fixed": [0, 1],
                "decision_vars.cathode.type_A_config.low_carbon_ratio_fixed": [0, 1]
            }
        }
    
    def load_validation_rules(self, file_path: str) -> None:
        """
        외부 파일에서 검증 규칙 로드
        
        Args:
            file_path: 검증 규칙 파일 경로 (YAML 또는 JSON)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"검증 규칙 파일을 찾을 수 없습니다: {file_path}")
        
        # 파일 형식에 따라 로드
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f)
        else:  # JSON으로 간주
            with open(path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
        
        self.update_validation_rules(rules)
    
    def update_validation_rules(self, new_rules: Dict[str, Any]) -> None:
        """
        검증 규칙 업데이트
        
        Args:
            new_rules: 새 검증 규칙
        """
        # 중첩 딕셔너리를 위한 재귀적 업데이트
        def update_dict_recursive(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_dict_recursive(target[key], value)
                else:
                    target[key] = value
        
        update_dict_recursive(self.validation_rules, new_rules)
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        설정 유효성 검증
        
        Args:
            config: 검증할 설정 딕셔너리 (None이면 opt_input에서 가져옴)
            
        Returns:
            Tuple[bool, List[str]]: (유효성 여부, 에러 메시지 목록)
        """
        # 설정이 제공되지 않은 경우 opt_input에서 가져옴
        if config is None:
            if self.opt_input is None:
                return False, ["최적화 입력 객체가 설정되지 않았습니다."]
            config = self.opt_input.get_config()
        
        self.validation_errors = []
        
        # 1. 필수 필드 검증
        self._validate_required_fields(config)
        
        # 2. 목적함수 유효성 검증
        if 'objective' in config:
            self._validate_objective(config['objective'])
        
        # 3. 양극재 설정 검증
        if 'decision_vars' in config and 'cathode' in config['decision_vars']:
            self._validate_cathode_config(config['decision_vars']['cathode'])
        
        # 4. 제약조건 최솟값 검증
        if 'constraints' in config:
            self._validate_constraints_minimums(config['constraints'])
        
        # 5. 값 범위 검증
        self._validate_value_ranges(config)
        
        # 6. 추가 검증 (확장 포인트)
        self._additional_validations(config)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> None:
        """
        필수 필드 유효성 검증
        
        Args:
            config: 검증할 설정
        """
        required_fields = self.validation_rules.get('required_fields', [])
        for field in required_fields:
            if field not in config:
                self.validation_errors.append(f"필수 필드 '{field}'가 누락되었습니다.")
    
    def _validate_objective(self, objective: str) -> None:
        """
        목적함수 유효성 검증
        
        Args:
            objective: 목적함수 유형
        """
        valid_objectives = self.validation_rules.get('valid_objectives', [])
        if objective not in valid_objectives:
            self.validation_errors.append(f"유효하지 않은 목적함수: {objective}. 사용 가능: {valid_objectives}")
    
    def _validate_cathode_config(self, cathode_config: Dict[str, Any]) -> None:
        """
        양극재 설정 유효성 검증
        
        Args:
            cathode_config: 양극재 설정
        """
        if 'type' not in cathode_config:
            self.validation_errors.append("양극재 'type' 설정이 누락되었습니다.")
            return
        
        cathode_type = cathode_config['type']
        if cathode_type not in ['A', 'B']:
            self.validation_errors.append(f"양극재 타입은 'A' 또는 'B'여야 합니다. 현재: {cathode_type}")
            return
        
        # Type별 필수 설정 확인
        type_requirements = self.validation_rules.get('cathode_type_requirements', {})
        required_keys = type_requirements.get(cathode_type, [])
        
        type_config_key = f'type_{cathode_type}_config'
        if type_config_key not in cathode_config:
            self.validation_errors.append(f"{cathode_type} 타입 설정 '{type_config_key}'가 누락되었습니다.")
            return
            
        type_config = cathode_config[type_config_key]
        for req_key in required_keys:
            if req_key not in type_config:
                self.validation_errors.append(f"Type {cathode_type} 필수 설정 '{req_key}'가 누락되었습니다.")
    
    def _validate_constraints_minimums(self, constraints: Dict[str, Any]) -> None:
        """
        제약조건 최솟값 검증
        
        Args:
            constraints: 제약조건 설정
        """
        constraint_minimums = self.validation_rules.get('constraint_minimums', {})
        
        for constraint_name, min_value in constraint_minimums.items():
            if constraint_name in constraints:
                if constraints[constraint_name] < min_value:
                    self.validation_errors.append(f"제약조건 '{constraint_name}'의 값({constraints[constraint_name]})이 최솟값 {min_value}보다 작습니다.")
    
    def _validate_value_ranges(self, config: Dict[str, Any]) -> None:
        """
        값 범위 검증
        
        Args:
            config: 검증할 설정
        """
        value_ranges = self.validation_rules.get('value_ranges', {})
        
        for path, range_values in value_ranges.items():
            if not isinstance(range_values, list) or len(range_values) != 2:
                continue
                
            min_value, max_value = range_values
            
            # 중첩 경로 처리
            parts = path.split('.')
            current = config
            found = True
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    found = False
                    break
            
            if found:
                # 값 범위 검증
                if isinstance(current, (int, float)):
                    if current < min_value or current > max_value:
                        self.validation_errors.append(f"설정값 '{path}'의 값({current})이 허용 범위 [{min_value}, {max_value}]를 벗어났습니다.")
                elif isinstance(current, list) and len(current) == 2:
                    # 범위 형식 값인 경우
                    range_min, range_max = current
                    if range_min < min_value:
                        self.validation_errors.append(f"설정값 '{path}'의 최솟값({range_min})이 허용된 최솟값 {min_value}보다 작습니다.")
                    if range_max > max_value:
                        self.validation_errors.append(f"설정값 '{path}'의 최댓값({range_max})이 허용된 최댓값 {max_value}보다 큽니다.")
    
    def _additional_validations(self, config: Dict[str, Any]) -> None:
        """
        추가 유효성 검증 (확장 포인트)
        
        Args:
            config: 검증할 설정
        """
        # 1. 비율 합 확인 (Type A)
        if ('decision_vars' in config and 'cathode' in config['decision_vars'] and
            config['decision_vars']['cathode'].get('type') == 'A'):
            
            type_a_config = config['decision_vars']['cathode'].get('type_A_config', {})
            recycle_ratio = type_a_config.get('recycle_ratio_fixed', 0)
            low_carbon_ratio = type_a_config.get('low_carbon_ratio_fixed', 0)
            
            if recycle_ratio + low_carbon_ratio > 1.0:
                self.validation_errors.append(f"Type A 설정의 비율 합(재활용재:{recycle_ratio}, 저탄소원료:{low_carbon_ratio})이 1을 초과합니다.")
        
        # 2. 목적함수와 제약조건의 일관성 확인
        if ('objective' in config and config['objective'] == 'minimize_carbon' and
            'constraints' in config and 'target_carbon' in config['constraints']):
            
            self.validation_errors.append("주의: 목적함수가 'minimize_carbon'인 경우 'target_carbon' 제약조건은 중복될 수 있습니다.")
    
    def validate_variables(self, variables: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        변수값 유효성 검증
        
        Args:
            variables: 검증할 변수값
            
        Returns:
            Tuple[bool, List[str]]: (유효성 여부, 에러 메시지 목록)
        """
        errors = []
        
        # 1. 감축비율 변수 검증
        for var_name, var_value in variables.items():
            if 'tier' in var_name and not var_name.endswith('_active'):
                if var_value < 0 or var_value > 100:
                    errors.append(f"감축비율 '{var_name}'의 값({var_value})이 유효 범위 [0, 100]를 벗어났습니다.")
        
        # 2. 양극재 비율 변수 검증
        recycle_ratio = variables.get('recycle_ratio', 0)
        low_carbon_ratio = variables.get('low_carbon_ratio', 0)
        
        if recycle_ratio + low_carbon_ratio > 1.0:
            errors.append(f"양극재 비율 합(재활용재:{recycle_ratio}, 저탄소원료:{low_carbon_ratio})이 1을 초과합니다.")
        
        if recycle_ratio < 0 or recycle_ratio > 1:
            errors.append(f"재활용재 비율({recycle_ratio})이 유효 범위 [0, 1]를 벗어났습니다.")
            
        if low_carbon_ratio < 0 or low_carbon_ratio > 1:
            errors.append(f"저탄소원료 비율({low_carbon_ratio})이 유효 범위 [0, 1]를 벗어났습니다.")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_results(self, results: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        결과 유효성 검증
        
        Args:
            results: 검증할 결과
            
        Returns:
            Tuple[bool, List[str]]: (유효성 여부, 에러 메시지 목록)
        """
        errors = []
        
        # 1. 기본 검증
        if results.get('status') != 'optimal':
            errors.append(f"최적화 상태가 'optimal'이 아닙니다: {results.get('status')}")
            return False, errors
        
        # 2. 목적함수 값 검증
        if 'objective_value' not in results:
            errors.append("목적함수 값이 누락되었습니다.")
        
        # 3. 변수값 검증
        if 'variables' in results:
            is_valid, var_errors = self.validate_variables(results['variables'])
            if not is_valid:
                errors.extend(var_errors)
        else:
            errors.append("변수 결과가 누락되었습니다.")
        
        # 4. 탄소발자국 검증
        if 'carbon_footprint' in results:
            carbon_footprint = results['carbon_footprint']
            
            # 음수 여부 확인
            try:
                if isinstance(carbon_footprint, str):
                    carbon_value = float(carbon_footprint.split()[0])
                else:
                    carbon_value = float(carbon_footprint)
                    
                if carbon_value < 0:
                    errors.append(f"탄소발자국({carbon_value})이 음수입니다.")
            except:
                errors.append(f"탄소발자국({carbon_footprint})의 형식이 유효하지 않습니다.")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def export_validation_rules(self, file_path: str, format: str = 'yaml') -> str:
        """
        검증 규칙을 파일로 내보내기
        
        Args:
            file_path: 저장할 파일 경로
            format: 파일 형식 ('yaml' 또는 'json')
            
        Returns:
            str: 저장된 파일 경로
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 형식에 따라 저장
        if format.lower() == 'yaml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump({'validation_rules': self.validation_rules}, f, default_flow_style=False, allow_unicode=True)
        else:  # json으로 저장
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'validation_rules': self.validation_rules}, f, ensure_ascii=False, indent=2)
        
        return str(path)
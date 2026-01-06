"""
최적화 설정 관리를 위한 ConfigManager 클래스
JSON 기반 설정 파일의 로드, 저장, 검증, 시나리오 관리를 담당합니다.
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class ConfigManager:
    """
    최적화 설정 파일을 관리하는 클래스
    
    주요 기능:
    - JSON 설정 파일 로드/저장
    - 설정 유효성 검증
    - 시나리오별 설정 관리
    - 솔버 자동 선택
    """
    
    def __init__(self, config_dir: str = "src/optimization"):
        """
        Args:
            config_dir: 설정 파일이 위치한 디렉토리 경로
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_config_path = self.config_dir / "config_opt.json"
        self._current_config = None
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        JSON 설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본 설정 사용)
            
        Returns:
            Dict: 로드된 설정 딕셔너리
            
        Raises:
            FileNotFoundError: 설정 파일이 존재하지 않음
            json.JSONDecodeError: JSON 형식 오류
        """
        if config_path is None:
            config_path = self.default_config_path
        else:
            config_path = Path(config_path)
            
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 설정 유효성 검증
            is_valid, error_msg = self.validate_config(config)
            if not is_valid:
                raise ValueError(f"설정 파일 유효성 검증 실패: {error_msg}")
                
            self._current_config = config
            return config
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON 형식 오류: {str(e)}", e.doc, e.pos)
            
    def save_config(self, config: Dict[str, Any], config_path: Optional[str] = None, 
                   update_metadata: bool = True) -> str:
        """
        설정을 JSON 파일로 저장
        
        Args:
            config: 저장할 설정 딕셔너리
            config_path: 저장할 파일 경로 (None이면 자동 생성)
            update_metadata: 메타데이터 자동 업데이트 여부
            
        Returns:
            str: 저장된 파일 경로
        """
        if config_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_path = self.config_dir / f"config_{timestamp}.json"
        else:
            config_path = Path(config_path)
            
        # 설정 유효성 검증
        is_valid, error_msg = self.validate_config(config)
        if not is_valid:
            raise ValueError(f"저장할 설정이 유효하지 않습니다: {error_msg}")
            
        # 메타데이터 업데이트
        if update_metadata:
            if 'metadata' not in config:
                config['metadata'] = {}
            config['metadata']['last_modified'] = datetime.now().strftime("%Y-%m-%d")
            
        # 파일 저장
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        return str(config_path)
        
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        설정 유효성 검증
        
        Args:
            config: 검증할 설정 딕셔너리
            
        Returns:
            Tuple[bool, Optional[str]]: (유효성 여부, 에러 메시지)
        """
        try:
            # 기본 구조 확인
            validation_rules = config.get('validation_rules', {})
            required_fields = validation_rules.get('required_fields', ['objective', 'decision_vars', 'constraints'])
            
            for field in required_fields:
                if field not in config:
                    return False, f"필수 필드 '{field}'가 누락되었습니다."
                    
            # 목적함수 유효성 확인
            objective = config.get('objective')
            valid_objectives = validation_rules.get('valid_objectives', 
                ['minimize_carbon', 'minimize_cost', 'maximize_ease', 'multi_objective'])
            
            if objective not in valid_objectives:
                return False, f"유효하지 않은 목적함수: {objective}. 사용 가능: {valid_objectives}"
                
            # 양극재 설정 확인
            decision_vars = config.get('decision_vars', {})
            cathode_config = decision_vars.get('cathode', {})
            cathode_type = cathode_config.get('type')
            
            if cathode_type not in ['A', 'B']:
                return False, f"양극재 타입은 'A' 또는 'B'여야 합니다. 현재: {cathode_type}"
                
            # Type별 필수 설정 확인
            type_requirements = validation_rules.get('cathode_type_requirements', {})
            required_keys = type_requirements.get(cathode_type, [])
            
            type_config_key = f'type_{cathode_type}_config'
            if type_config_key not in cathode_config:
                return False, f"{cathode_type} 타입 설정 '{type_config_key}'가 누락되었습니다."
                
            type_config = cathode_config[type_config_key]
            for req_key in required_keys:
                if req_key not in type_config:
                    return False, f"Type {cathode_type} 필수 설정 '{req_key}'가 누락되었습니다."
                    
            # 제약조건 최솟값 확인
            constraints = config.get('constraints', {})
            constraint_minimums = validation_rules.get('constraint_minimums', {})
            
            for constraint_name, min_value in constraint_minimums.items():
                if constraint_name in constraints:
                    if constraints[constraint_name] < min_value:
                        return False, f"제약조건 '{constraint_name}'의 값이 최솟값 {min_value}보다 작습니다."
                        
            # 감축비율 범위 확인
            reduction_rates = decision_vars.get('reduction_rates', {})
            for var_name, var_config in reduction_rates.items():
                if isinstance(var_config, dict):
                    min_val = var_config.get('min', 0)
                    max_val = var_config.get('max', 100)
                    default_val = var_config.get('default', 0)
                    
                    if min_val < 0 or max_val > 100 or min_val > max_val:
                        return False, f"감축비율 '{var_name}' 범위가 유효하지 않습니다."
                    
                    if default_val < min_val or default_val > max_val:
                        return False, f"감축비율 '{var_name}' 기본값이 범위를 벗어났습니다."
                        
            return True, None
            
        except Exception as e:
            return False, f"설정 검증 중 오류 발생: {str(e)}"
            
    def apply_scenario(self, config: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
        """
        시나리오 설정을 기본 설정에 적용
        
        Args:
            config: 기본 설정
            scenario_name: 적용할 시나리오 이름
            
        Returns:
            Dict: 시나리오가 적용된 설정
            
        Raises:
            ValueError: 존재하지 않는 시나리오
        """
        scenarios = config.get('scenarios', {})
        if scenario_name not in scenarios:
            available_scenarios = list(scenarios.keys())
            raise ValueError(f"시나리오 '{scenario_name}'를 찾을 수 없습니다. 사용 가능: {available_scenarios}")
            
        # 설정 복사 및 시나리오 적용
        new_config = copy.deepcopy(config)
        scenario = scenarios[scenario_name]
        
        # 제약조건 업데이트
        if 'target_carbon' in scenario:
            new_config['constraints']['target_carbon'] = scenario['target_carbon']
            
        if 'max_activities' in scenario:
            new_config['constraints']['max_activities'] = scenario['max_activities']
            
        # 감축비율 기본값 업데이트
        if 'default_reduction_rates' in scenario:
            default_rate = scenario['default_reduction_rates']
            reduction_rates = new_config.get('decision_vars', {}).get('reduction_rates', {})
            
            for var_name in reduction_rates:
                if isinstance(reduction_rates[var_name], dict):
                    reduction_rates[var_name]['default'] = default_rate
                    
        # 메타데이터 업데이트
        if 'metadata' not in new_config:
            new_config['metadata'] = {}
        new_config['metadata']['applied_scenario'] = scenario_name
        new_config['metadata']['scenario_description'] = scenario.get('description', '')
        
        return new_config
        
    def get_solver_recommendation(self, config: Dict[str, Any]) -> str:
        """
        설정에 기반하여 최적의 솔버 추천
        
        Args:
            config: 최적화 설정
            
        Returns:
            str: 추천 솔버 이름 ('ipopt', 'glpk', 'cbc')
        """
        objective = config.get('objective')
        cathode_config = config.get('decision_vars', {}).get('cathode', {})
        cathode_type = cathode_config.get('type', 'B')
        
        solver_settings = config.get('solver_settings', {})
        recommendations = solver_settings.get('recommended_solver_by_problem', {})
        
        # 목적함수가 maximize_ease인 경우
        if objective == 'maximize_ease':
            return recommendations.get('maximize_ease', 'cbc')
            
        # Type A (비선형 문제)
        if cathode_type == 'A':
            return recommendations.get('type_A', 'ipopt')
            
        # Type B
        # 정수 변수가 필요한 경우 (예: 최대 활동 수 제한)
        constraints = config.get('constraints', {})
        if 'max_activities' in constraints:
            return recommendations.get('type_B_integer', 'cbc')
        else:
            return recommendations.get('type_B_linear', 'glpk')
            
    def get_solver_options(self, config: Dict[str, Any], solver_name: str) -> Dict[str, Any]:
        """
        특정 솔버의 옵션 반환
        
        Args:
            config: 최적화 설정
            solver_name: 솔버 이름
            
        Returns:
            Dict: 솔버 옵션
        """
        solver_settings = config.get('solver_settings', {})
        return solver_settings.get(f'{solver_name}_options', {})
        
    def list_available_scenarios(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        사용 가능한 시나리오 목록 반환
        
        Args:
            config: 설정 (None이면 현재 로드된 설정 사용)
            
        Returns:
            List[str]: 시나리오 이름 목록
        """
        if config is None:
            config = self._current_config
            
        if config is None:
            return []
            
        return list(config.get('scenarios', {}).keys())
        
    def get_scenario_info(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        시나리오별 설명 정보 반환
        
        Args:
            config: 설정 (None이면 현재 로드된 설정 사용)
            
        Returns:
            Dict[str, str]: 시나리오 이름과 설명의 매핑
        """
        if config is None:
            config = self._current_config
            
        if config is None:
            return {}
            
        scenarios = config.get('scenarios', {})
        return {
            name: scenario.get('description', '설명 없음')
            for name, scenario in scenarios.items()
        }
        
    def create_custom_config(self, base_config: Optional[Dict[str, Any]] = None, 
                           **overrides) -> Dict[str, Any]:
        """
        커스텀 설정 생성
        
        Args:
            base_config: 기본 설정 (None이면 현재 설정 사용)
            **overrides: 덮어쓸 설정값들
            
        Returns:
            Dict: 커스텀 설정
        """
        if base_config is None:
            if self._current_config is None:
                raise ValueError("기본 설정이 로드되지 않았습니다. load_config()를 먼저 실행하세요.")
            base_config = self._current_config
            
        new_config = copy.deepcopy(base_config)
        
        # 중첩 딕셔너리 업데이트를 위한 재귀 함수
        def update_nested_dict(target_dict: Dict, updates: Dict, prefix: str = ""):
            for key, value in updates.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if '.' in key:
                    # 중첩 키 처리 (예: "constraints.target_carbon")
                    keys = key.split('.')
                    current = target_dict
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value
                else:
                    target_dict[key] = value
                    
        update_nested_dict(new_config, overrides)
        
        # 메타데이터 업데이트
        if 'metadata' not in new_config:
            new_config['metadata'] = {}
        new_config['metadata']['custom_config'] = True
        new_config['metadata']['last_modified'] = datetime.now().strftime("%Y-%m-%d")
        
        return new_config
        
    def export_config_template(self, output_path: Optional[str] = None) -> str:
        """
        새로운 설정 파일 생성을 위한 템플릿 내보내기
        
        Args:
            output_path: 출력 파일 경로
            
        Returns:
            str: 생성된 템플릿 파일 경로
        """
        if output_path is None:
            output_path = self.config_dir / "config_template.json"
        else:
            output_path = Path(output_path)
            
        if self._current_config is None:
            raise ValueError("기본 설정이 로드되지 않았습니다.")
            
        template = copy.deepcopy(self._current_config)
        
        # 템플릿용 메타데이터 설정
        template['metadata']['name'] = "새로운 PCF 최적화 설정"
        template['metadata']['description'] = "config_opt.json을 기반으로 생성된 템플릿"
        template['metadata']['is_template'] = True
        template['metadata']['created'] = datetime.now().strftime("%Y-%m-%d")
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
            
        return str(output_path)
        
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """현재 로드된 설정 반환"""
        return self._current_config
        
    def reset_to_defaults(self) -> Dict[str, Any]:
        """기본 설정으로 리셋"""
        return self.load_config()
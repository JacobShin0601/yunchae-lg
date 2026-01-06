"""
최적화 모델의 입력 데이터 및 설정을 관리하는 클래스
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import yaml
import json
import os
import pandas as pd
from .constant import OptimizationConstants
from .variable import OptimizationVariables


class OptimizationInput:
    """
    최적화 모델의 모든 입력 데이터를 관리하는 클래스
    
    주요 역할:
    - 설정 파일 로드 및 관리
    - 시나리오 적용
    - 제약조건 관리
    - stable_var 데이터 로드
    """
    
    def __init__(self, 
                config_path: Optional[str] = None,
                stable_var_dir: str = "stable_var",
                scenario_df: Optional['pd.DataFrame'] = None,
                ref_formula_df: Optional['pd.DataFrame'] = None,
                ref_proportions_df: Optional['pd.DataFrame'] = None,
                original_df: Optional['pd.DataFrame'] = None):
        """
        Args:
            config_path: 설정 파일 경로 (YAML 또는 JSON)
            stable_var_dir: stable_var 디렉토리 경로
            scenario_df: PCF 시나리오 데이터프레임 (시뮬레이션 정렬 모드용)
            ref_formula_df: 참조 공식 데이터프레임
            ref_proportions_df: 참조 비율 데이터프레임
            original_df: 원본 데이터프레임
        """
        self.config = {}
        self.stable_var_data = {}
        self.constants = OptimizationConstants()
        self.variables = OptimizationVariables()
        self.stable_var_dir = Path(stable_var_dir)
        
        # 시뮬레이션 데이터 저장 (시뮬레이션 정렬 모드용)
        self.scenario_df = scenario_df
        self.ref_formula_df = ref_formula_df
        self.ref_proportions_df = ref_proportions_df
        self.original_df = original_df
        
        # 기본 설정 로드
        self._load_default_config()
        
        # stable_var 데이터 로드
        self._load_stable_var_data()
        
        # 사용자 설정 로드 (있는 경우)
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> None:
        """기본 설정 로드"""
        self.config = {
            "metadata": {
                "name": "PCF 최적화 설정",
                "description": "배터리 소재 탄소발자국 최소화를 위한 최적화 모델 설정",
                "version": "1.0"
            },
            "objective": "minimize_carbon",
            "objective_options": {
                "available_types": [
                    "minimize_carbon",
                    "minimize_cost", 
                    "maximize_ease",
                    "multi_objective"
                ],
                "multi_objective_weights": {
                    "carbon": 0.7,
                    "cost": 0.3
                }
            },
            "decision_vars": self.variables.get_all_config(),
            "constraints": {
                "target_carbon": 50.0,
                "max_activities": 5,
                "max_cost": 100000,
                "ni_max": 300,
                "feasibility_threshold": 0.8
            },
            "case_constraints": {
                "enabled": False,
                "min_difference": 0.05,  # 5%
                "variables": {
                    "tier1": {
                        1: "tier1_case1",
                        2: "tier1_case2",
                        3: "tier1_case3"
                    },
                    "tier2": {
                        1: "tier2_case1",
                        2: "tier2_case2",
                        3: "tier2_case3"
                    },
                    "tier3": {
                        1: "tier3_case1",
                        2: "tier3_case2",
                        3: "tier3_case3"
                    }
                }
            },
            "location_constraints": {
                "enabled": False,
                "use_fixed_constraints": True,
                "variables": {
                    "양극재": "cathode_location",
                    "분리막": "separator_location",
                    "전해액": "electrolyte_location",
                    "음극재": "anode_location",
                    "동박": "cu_foil_location"
                },
                "material_locations": {
                    "양극재": ["한국", "중국", "일본"],
                    "분리막": ["한국", "중국", "폴란드"],
                    "전해액": ["한국", "중국", "일본"],
                    "음극재": ["중국", "일본"],
                    "동박": ["한국", "중국"]
                }
            },
            "material_constraints": {
                "enabled": False,
                "recycle_ratio": {
                    "enabled": True,
                    "min": 0.1,  # 10%
                    "max": 0.5   # 50%
                },
                "low_carbon_ratio": {
                    "enabled": True,
                    "min": 0.05,  # 5%
                    "max": 0.3    # 30%
                },
                "material_balance": {
                    "enabled": True,
                    "max_total": 0.7  # 재활용재 + 저탄소메탈 최대 70%
                },
                "proportions": {
                    "enabled": False,
                    "values": {
                        "recycle_to_low_carbon": 2.0  # 재활용재 비율이 저탄소 메탈의 2배
                    }
                }
            },
            "material_specific": {
                "enabled": False,
                "materials": {
                    "양극재": {
                        "strategy": "minimize_carbon",
                        "reduction_caps": {
                            "tier1": 30.0,
                            "tier2": 40.0
                        },
                        "constraints": {
                            "max_reduction": 60.0,
                            "max_cost": 1000.0
                        }
                    },
                    "분리막": {
                        "strategy": "minimize_cost",
                        "reduction_caps": {
                            "tier1": 25.0,
                            "tier2": 35.0
                        },
                        "constraints": {
                            "max_reduction": 40.0,
                            "target_carbon": 45.0
                        }
                    }
                }
            },
            "constants": self.constants.get_all_constants(),
            "solver_settings": {
                "recommended_solver_by_problem": {
                    "type_A": "ipopt",
                    "type_B_linear": "glpk", 
                    "type_B_integer": "cbc",
                    "maximize_ease": "cbc"
                },
                "ipopt_options": {
                    "max_iter": 3000,
                    "tol": 1e-6,
                    "print_level": 5
                },
                "glpk_options": {
                    "mipgap": 0.01,
                    "tmlim": 300
                },
                "cbc_options": {
                    "seconds": 300,
                    "ratio": 0.01,
                    "threads": 4
                }
            },
            "scenarios": {
                "conservative": {
                    "description": "보수적 시나리오 - 낮은 감축 목표",
                    "target_carbon": 60.0,
                    "max_activities": 3,
                    "default_reduction_rates": 15
                },
                "aggressive": {
                    "description": "적극적 시나리오 - 높은 감축 목표", 
                    "target_carbon": 35.0,
                    "max_activities": 8,
                    "default_reduction_rates": 40
                },
                "balanced": {
                    "description": "균형적 시나리오 - 중간 수준 목표",
                    "target_carbon": 50.0,
                    "max_activities": 5,
                    "default_reduction_rates": 25
                }
            }
        }
    
    def _load_stable_var_data(self) -> None:
        """stable_var 디렉토리에서 데이터 로드"""
        self.stable_var_data = {}
        
        # 디렉토리 존재 확인
        if not self.stable_var_dir.exists():
            return
        
        # 1. 국가별 전력배출계수 로드
        electricity_coef_path = self.stable_var_dir / "electricity_coef_by_country.json"
        if electricity_coef_path.exists():
            with open(electricity_coef_path, 'r', encoding='utf-8') as f:
                self.stable_var_data['electricity_coef'] = json.load(f)
        
        # 2. 재활용재 환경영향 로드
        recycle_impact_path = self.stable_var_dir / "recycle_material_impact.json"
        if recycle_impact_path.exists():
            with open(recycle_impact_path, 'r', encoding='utf-8') as f:
                self.stable_var_data['recycle_impact'] = json.load(f)
        
        # 3. 양극재 Tier1 입력 데이터 로드
        cathode_tier1_path = self.stable_var_dir / "cathode_tier1_input.json"
        if cathode_tier1_path.exists():
            with open(cathode_tier1_path, 'r', encoding='utf-8') as f:
                self.stable_var_data['cathode_tier1'] = json.load(f)
        
        # 4. 양극재 Tier2 입력 데이터 로드
        cathode_tier2_path = self.stable_var_dir / "cathode_tier2_input.json"
        if cathode_tier2_path.exists():
            with open(cathode_tier2_path, 'r', encoding='utf-8') as f:
                self.stable_var_data['cathode_tier2'] = json.load(f)
                
        # 5. Tier별 비용 데이터 로드
        cost_by_tier_path = self.stable_var_dir / "cost_by_tier.json"
        if cost_by_tier_path.exists():
            with open(cost_by_tier_path, 'r', encoding='utf-8') as f:
                cost_data = json.load(f)
                self.stable_var_data['cost_by_tier'] = cost_data
                # 상수에도 비용 데이터 추가
                self.constants.update_constants({'cost_by_tier': cost_data})
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        설정 파일 로드
        
        Args:
            config_path: 설정 파일 경로 (YAML 또는 JSON)
            
        Returns:
            Dict: 로드된 설정
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        
        # 파일 형식에 따라 로드
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
        else:  # JSON으로 간주
            with open(path, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
        
        # 설정 유효성 검증
        is_valid, error_msg = self.validate_config(new_config)
        if not is_valid:
            raise ValueError(f"설정 파일 유효성 검증 실패: {error_msg}")
        
        # 설정 업데이트
        self.update_config(new_config)
        
        # 상수와 변수 클래스 업데이트
        if 'constants' in new_config:
            self.constants.update_constants(new_config['constants'])
        if 'decision_vars' in new_config:
            self.variables.update_config(new_config['decision_vars'])
        
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        설정 업데이트
        
        Args:
            new_config: 새 설정 딕셔너리
        """
        # 중첩 딕셔너리를 위한 재귀적 업데이트
        def update_dict_recursive(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_dict_recursive(target[key], value)
                else:
                    target[key] = value
        
        update_dict_recursive(self.config, new_config)
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        설정 유효성 검증
        
        Args:
            config: 검증할 설정 딕셔너리
            
        Returns:
            Tuple[bool, Optional[str]]: (유효성 여부, 에러 메시지)
        """
        # 필수 키 확인
        required_keys = ['objective']
        for key in required_keys:
            if key not in config:
                return False, f"필수 설정 '{key}'가 누락되었습니다."
        
        # 목적함수 유효성 확인
        valid_objectives = ['minimize_carbon', 'minimize_cost', 'maximize_ease', 'multi_objective']
        if config.get('objective') not in valid_objectives:
            return False, f"유효하지 않은 목적함수: {config.get('objective')}"
        
        # 추가 검증 룰 적용 (있는 경우)
        validation_rules = config.get('validation_rules', self.config.get('validation_rules', {}))
        
        # 양극재 타입 검증
        cathode_config = config.get('decision_vars', {}).get('cathode', {})
        if cathode_config and 'type' in cathode_config:
            cathode_type = cathode_config['type']
            if cathode_type not in ['A', 'B']:
                return False, f"양극재 타입은 'A' 또는 'B'여야 합니다. 현재: {cathode_type}"
            
            # Type별 필수 설정 확인
            if cathode_type == 'A':
                type_a_config = cathode_config.get('type_A_config', {})
                for key in ['recycle_ratio_fixed', 'low_carbon_ratio_fixed', 'emission_range']:
                    if key not in type_a_config:
                        return False, f"Type A 필수 설정 '{key}'가 누락되었습니다."
            elif cathode_type == 'B':
                type_b_config = cathode_config.get('type_B_config', {})
                for key in ['emission_fixed', 'recycle_range', 'low_carbon_range']:
                    if key not in type_b_config:
                        return False, f"Type B 필수 설정 '{key}'가 누락되었습니다."
        
        return True, None
    
    def apply_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        시나리오 적용
        
        Args:
            scenario_name: 시나리오 이름
            
        Returns:
            Dict: 시나리오가 적용된 설정
        """
        scenarios = self.config.get('scenarios', {})
        if scenario_name not in scenarios:
            available_scenarios = list(scenarios.keys())
            raise ValueError(f"시나리오 '{scenario_name}'를 찾을 수 없습니다. 사용 가능: {available_scenarios}")
        
        scenario = scenarios[scenario_name]
        
        # 제약조건 업데이트
        if 'target_carbon' in scenario:
            self.config['constraints']['target_carbon'] = scenario['target_carbon']
        
        if 'max_activities' in scenario:
            self.config['constraints']['max_activities'] = scenario['max_activities']
        
        # 감축비율 기본값 업데이트
        if 'default_reduction_rates' in scenario:
            default_rate = scenario['default_reduction_rates']
            reduction_rates = self.config.get('decision_vars', {}).get('reduction_rates', {})
            
            for var_name in reduction_rates:
                if isinstance(reduction_rates[var_name], dict):
                    reduction_rates[var_name]['default'] = default_rate
        
        # 메타데이터 업데이트
        if 'metadata' not in self.config:
            self.config['metadata'] = {}
        self.config['metadata']['applied_scenario'] = scenario_name
        self.config['metadata']['scenario_description'] = scenario.get('description', '')
        
        # 변수 클래스 업데이트
        self.variables.update_config(self.config.get('decision_vars', {}))
        
        return self.config
    
    def get_objective(self) -> str:
        """
        목적함수 유형 조회
        
        Returns:
            str: 목적함수 유형
        """
        return self.config.get('objective', 'minimize_carbon')
    
    def get_constraints(self) -> Dict[str, Any]:
        """
        제약조건 조회
        
        Returns:
            Dict: 제약조건 딕셔너리
        """
        return self.config.get('constraints', {})
    
    def get_constraint(self, name: str, default: Any = None) -> Any:
        """
        특정 제약조건 값 조회
        
        Args:
            name: 제약조건 이름
            default: 기본값
            
        Returns:
            Any: 제약조건 값
        """
        return self.config.get('constraints', {}).get(name, default)
    
    def get_solver_recommendation(self) -> str:
        """
        설정에 기반한 솔버 추천
        
        Returns:
            str: 추천 솔버 이름
        """
        objective = self.get_objective()
        cathode_config = self.variables.get_cathode_config()
        cathode_type = cathode_config.get('type', 'B')
        
        solver_settings = self.config.get('solver_settings', {})
        recommendations = solver_settings.get('recommended_solver_by_problem', {})
        
        # 목적함수가 maximize_ease인 경우
        if objective == 'maximize_ease':
            return recommendations.get('maximize_ease', 'cbc')
        
        # Type A (비선형 문제)
        if cathode_type == 'A':
            return recommendations.get('type_A', 'ipopt')
        
        # Type B
        # 정수 변수가 필요한 경우 (예: 최대 활동 수 제한)
        constraints = self.get_constraints()
        if 'max_activities' in constraints:
            return recommendations.get('type_B_integer', 'cbc')
        else:
            return recommendations.get('type_B_linear', 'glpk')
    
    def get_solver_options(self, solver_name: str) -> Dict[str, Any]:
        """
        특정 솔버의 옵션 조회
        
        Args:
            solver_name: 솔버 이름
            
        Returns:
            Dict: 솔버 옵션
        """
        solver_settings = self.config.get('solver_settings', {})
        return solver_settings.get(f'{solver_name}_options', {})
    
    def get_available_scenarios(self) -> Dict[str, str]:
        """
        사용 가능한 시나리오 정보 조회
        
        Returns:
            Dict: 시나리오 이름과 설명의 매핑
        """
        scenarios = self.config.get('scenarios', {})
        return {
            name: scenario.get('description', '설명 없음')
            for name, scenario in scenarios.items()
        }
    
    def create_custom_config(self, **overrides) -> Dict[str, Any]:
        """
        커스텀 설정 생성
        
        Args:
            **overrides: 덮어쓸 설정값들
            
        Returns:
            Dict: 커스텀 설정
        """
        # 중첩 키 지원을 위한 함수
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
        
        update_nested_dict(self.config, overrides)
        
        # 메타데이터 업데이트
        if 'metadata' not in self.config:
            self.config['metadata'] = {}
        self.config['metadata']['custom_config'] = True
        
        # 상수와 변수 클래스 업데이트
        if 'constants' in overrides:
            self.constants.update_constants(overrides['constants'])
        if 'decision_vars' in overrides:
            self.variables.update_config(overrides['decision_vars'])
        
        return self.config
    
    def export_config(self, file_path: str, format: str = 'yaml') -> str:
        """
        현재 설정 내보내기
        
        Args:
            file_path: 저장할 파일 경로
            format: 파일 형식 ('yaml' 또는 'json')
            
        Returns:
            str: 저장된 파일 경로
        """
        path = Path(file_path)
        
        # 디렉토리 생성
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 형식에 따라 저장
        if format.lower() == 'yaml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        else:  # json으로 저장
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        return str(path)
    
    def get_config(self) -> Dict[str, Any]:
        """전체 설정 반환"""
        return self.config
    
    def get_stable_var_data(self) -> Dict[str, Any]:
        """stable_var 데이터 반환"""
        return self.stable_var_data
        
    def load_cost_by_tier(self, cost_by_tier_path: str) -> Dict[str, Any]:
        """
        Tier별 비용 데이터 로드 및 상수에 추가
        
        Args:
            cost_by_tier_path: cost_by_tier.json 파일 경로
            
        Returns:
            Dict: 로드된 비용 데이터
        """
        path = Path(cost_by_tier_path)
        if not path.exists():
            raise FileNotFoundError(f"비용 데이터 파일을 찾을 수 없습니다: {cost_by_tier_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            cost_data = json.load(f)
        
        # stable_var 데이터 업데이트
        self.stable_var_data['cost_by_tier'] = cost_data
        
        # 상수 업데이트
        self.constants.update_constants({'cost_by_tier': cost_data})
        
        return cost_data
    
    def get_constants(self) -> OptimizationConstants:
        """상수 클래스 반환"""
        return self.constants
    
    def get_variables(self) -> OptimizationVariables:
        """변수 클래스 반환"""
        return self.variables
        
    def is_material_specific_enabled(self) -> bool:
        """
        소재별 최적화 전략 사용 여부 확인
        
        Returns:
            bool: 소재별 최적화가 활성화되어 있으면 True, 아니면 False
        """
        return self.config.get('material_specific', {}).get('enabled', False)
    
    def get_material_config(self, material: str) -> Dict[str, Any]:
        """
        특정 소재에 대한 최적화 설정 가져오기
        
        Args:
            material: 소재 이름 (예: '양극재', '분리막' 등)
            
        Returns:
            Dict: 소재별 최적화 설정 (없으면 빈 딩셔너리)
        """
        if not self.is_material_specific_enabled():
            return {}
            
        materials = self.config.get('material_specific', {}).get('materials', {})
        return materials.get(material, {})
    
    def get_all_materials(self) -> List[str]:
        """
        최적화 설정에 정의된 모든 소재 목록 가져오기
        
        Returns:
            List[str]: 소재 이름 목록
        """
        materials = self.config.get('material_specific', {}).get('materials', {})
        return list(materials.keys())
    
    def set_material_specific_enabled(self, enabled: bool) -> None:
        """
        소재별 최적화 활성화/비활성화 설정
        
        Args:
            enabled: 활성화 여부
        """
        if 'material_specific' not in self.config:
            self.config['material_specific'] = {'enabled': enabled, 'materials': {}}
        else:
            self.config['material_specific']['enabled'] = enabled
    
    def set_material_config(self, material: str, strategy: str = None, 
                           reduction_caps: Dict[str, float] = None, 
                           constraints: Dict[str, Any] = None) -> None:
        """
        특정 소재에 대한 최적화 설정 추가/수정
        
        Args:
            material: 소재 이름 (예: '양극재', '분리막' 등)
            strategy: 최적화 전략 ('minimize_carbon', 'minimize_cost', 'maximize_ease' 등)
            reduction_caps: tier별 감축량 제한 (예: {'tier1': 30.0, 'tier2': 40.0})
            constraints: 추가 제약조건
        """
        # 소재별 최적화 활성화
        if not self.is_material_specific_enabled():
            self.set_material_specific_enabled(True)
        
        if 'materials' not in self.config['material_specific']:
            self.config['material_specific']['materials'] = {}
        
        # 현재 소재 설정 가져오기 (없으면 초기화)
        if material not in self.config['material_specific']['materials']:
            self.config['material_specific']['materials'][material] = {}
        
        material_config = self.config['material_specific']['materials'][material]
        
        # 전략 설정
        if strategy is not None:
            material_config['strategy'] = strategy
        
        # 감축량 제한 설정
        if reduction_caps is not None:
            material_config['reduction_caps'] = reduction_caps
        
        # 추가 제약조건 설정
        if constraints is not None:
            if 'constraints' not in material_config:
                material_config['constraints'] = {}
            material_config['constraints'].update(constraints)
"""
파레토 최적화 설정 로더
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class WeightCombination:
    """가중치 조합"""
    carbon_weight: float
    cost_weight: float

    def __repr__(self):
        return f"C:{self.carbon_weight:.2f}/Cost:{self.cost_weight:.2f}"


class ParetoConfigLoader:
    """파레토 최적화 설정 로더"""

    DEFAULT_CONFIG_PATH = "input/pareto_config.yaml"

    def __init__(self, user_id: str = None):
        self.user_id = user_id
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """설정 로드 (기본 + 사용자 오버라이드)"""
        # 1. 기본 설정 로드
        default_config = self._load_yaml(self.DEFAULT_CONFIG_PATH)

        # 2. 사용자 설정 오버라이드
        if self.user_id:
            user_config_path = f"input/{self.user_id}/pareto_config.yaml"
            if Path(user_config_path).exists():
                user_config = self._load_yaml(user_config_path)
                default_config = self._merge_configs(default_config, user_config)

        return default_config

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """YAML 파일 로드"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ 설정 파일 없음: {path}, 기본값 사용")
            return {}

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """설정 병합 (재귀적)"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def get_weight_combinations(self) -> List[WeightCombination]:
        """가중치 조합 생성"""
        strategy = self.config.get('weight_sweep', {}).get('strategy', 'uniform')

        if strategy == 'uniform':
            return self._generate_uniform_weights()
        elif strategy == 'dense_edges':
            return self._generate_dense_edge_weights()
        elif strategy == 'logarithmic':
            return self._generate_logarithmic_weights()
        elif strategy == 'custom':
            return self._load_custom_weights()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_uniform_weights(self) -> List[WeightCombination]:
        """균등 간격 가중치"""
        config = self.config['weight_sweep']['uniform']
        points = config.get('points', 5)

        combinations = []
        for i in range(points):
            carbon_weight = i / (points - 1)  # 0.0 ~ 1.0
            cost_weight = 1.0 - carbon_weight
            combinations.append(WeightCombination(carbon_weight, cost_weight))

        return combinations

    def _generate_dense_edge_weights(self) -> List[WeightCombination]:
        """양 극단에 더 많은 포인트"""
        config = self.config['weight_sweep']['dense_edges']
        points = config.get('points', 9)
        edge_density = config.get('edge_density', 0.7)

        # Beta 분포 사용 (양 끝에 더 많은 샘플)
        import numpy as np

        alpha, beta_param = 0.5, 0.5
        samples = np.random.beta(alpha, beta_param, points)
        samples = np.sort(samples)

        combinations = []
        for s in samples:
            combinations.append(WeightCombination(s, 1.0 - s))

        return combinations

    def _generate_logarithmic_weights(self) -> List[WeightCombination]:
        """로그 스케일 가중치"""
        config = self.config['weight_sweep']['logarithmic']
        points = config.get('points', 7)
        base = config.get('base', 10)

        import numpy as np

        # 로그 스케일: 작은 값에 더 많은 포인트
        log_space = np.logspace(-2, 0, points, base=base)  # 0.01 ~ 1.0

        combinations = []
        for val in log_space:
            combinations.append(WeightCombination(val, 1.0 - val))

        return combinations

    def _load_custom_weights(self) -> List[WeightCombination]:
        """사용자 정의 가중치"""
        custom_combinations = self.config['weight_sweep']['custom']['combinations']

        return [
            WeightCombination(c['carbon_weight'], c['cost_weight'])
            for c in custom_combinations
        ]

    def get_constraint_preset(self, preset_name: str = 'medium') -> Dict[str, Dict]:
        """제약조건 프리셋 가져오기"""
        presets = self.config.get('constraint_presets', {})
        return presets.get(preset_name, presets.get('medium', {}))

    def get_scenario_template(self, template_name: str) -> Dict[str, Any]:
        """시나리오 템플릿 가져오기"""
        templates = self.config.get('scenario_templates', {})
        return templates.get(template_name, {})

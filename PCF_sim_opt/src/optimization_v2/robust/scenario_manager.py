"""
시나리오 관리자 (Scenario Manager)

다중 시나리오 정의 및 관리
"""

from typing import Dict, Any, List, Optional
import copy
import json
from pathlib import Path


class Scenario:
    """
    단일 시나리오 클래스

    Attributes:
        name: 시나리오 이름
        probability: 발생 확률 (0~1)
        description: 시나리오 설명
        parameter_variations: 파라미터 변동 사항
    """

    def __init__(
        self,
        name: str,
        probability: float,
        parameter_variations: Dict[str, Any],
        description: str = ""
    ):
        """
        시나리오 초기화

        Args:
            name: 시나리오 이름
            probability: 발생 확률 (0~1)
            parameter_variations: 파라미터 변동 딕셔너리
                형식: {
                    'material_emission_factors': {
                        'NCM622': {'multiplier': 1.2},  # 20% 증가
                    },
                    'costs': {
                        'recycled_premium': {'multiplier': 0.8},  # 20% 감소
                    },
                    'constraint_bounds': {
                        'premium_limit_pct': {'value': 5.0},  # 절대값 설정
                    }
                }
            description: 시나리오 설명
        """
        self.name = name
        self.probability = probability
        self.parameter_variations = parameter_variations
        self.description = description

        # 검증
        self._validate()

    def _validate(self):
        """시나리오 검증"""
        if not 0 <= self.probability <= 1:
            raise ValueError(f"확률은 0~1 범위여야 합니다: {self.probability}")

        if not self.name:
            raise ValueError("시나리오 이름은 필수입니다")

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'name': self.name,
            'probability': self.probability,
            'parameter_variations': self.parameter_variations,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scenario':
        """딕셔너리에서 생성"""
        return cls(
            name=data['name'],
            probability=data['probability'],
            parameter_variations=data['parameter_variations'],
            description=data.get('description', '')
        )

    def __repr__(self) -> str:
        return f"<Scenario(name='{self.name}', probability={self.probability:.2f})>"


class ScenarioManager:
    """
    시나리오 관리자

    주요 기능:
    - 시나리오 정의 및 관리
    - 확률 검증 (합 = 1)
    - 기준 데이터에 시나리오 적용
    - 시나리오 저장/로드
    """

    def __init__(self):
        """시나리오 관리자 초기화"""
        self.scenarios: List[Scenario] = []

    def add_scenario(
        self,
        name: str,
        probability: float,
        parameter_variations: Dict[str, Any],
        description: str = ""
    ) -> Scenario:
        """
        시나리오 추가

        Args:
            name: 시나리오 이름
            probability: 발생 확률
            parameter_variations: 파라미터 변동
            description: 설명

        Returns:
            생성된 Scenario 객체
        """
        # 중복 이름 확인
        if any(s.name == name for s in self.scenarios):
            raise ValueError(f"시나리오 이름이 이미 존재합니다: {name}")

        scenario = Scenario(name, probability, parameter_variations, description)
        self.scenarios.append(scenario)

        print(f"✅ 시나리오 추가: {name} (확률: {probability:.2%})")

        return scenario

    def remove_scenario(self, name: str):
        """시나리오 제거"""
        self.scenarios = [s for s in self.scenarios if s.name != name]
        print(f"🗑️  시나리오 제거: {name}")

    def get_scenario(self, name: str) -> Optional[Scenario]:
        """이름으로 시나리오 가져오기"""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None

    def validate_probabilities(self) -> bool:
        """
        확률 합 검증 (합 = 1)

        Returns:
            검증 통과 여부
        """
        if not self.scenarios:
            return True

        total_prob = sum(s.probability for s in self.scenarios)

        if abs(total_prob - 1.0) > 1e-6:
            print(f"⚠️  확률 합이 1이 아닙니다: {total_prob:.4f}")
            return False

        return True

    def normalize_probabilities(self):
        """확률 정규화 (합 = 1로 조정)"""
        if not self.scenarios:
            return

        total_prob = sum(s.probability for s in self.scenarios)

        if total_prob == 0:
            # 균등 분포
            for scenario in self.scenarios:
                scenario.probability = 1.0 / len(self.scenarios)
        else:
            # 비율 유지하며 정규화
            for scenario in self.scenarios:
                scenario.probability /= total_prob

        print(f"✅ 확률 정규화 완료: {len(self.scenarios)}개 시나리오")

    def apply_scenario(
        self,
        base_data: Dict[str, Any],
        scenario: Scenario
    ) -> Dict[str, Any]:
        """
        기준 데이터에 시나리오 적용

        Args:
            base_data: 기준 최적화 데이터
            scenario: 적용할 시나리오

        Returns:
            시나리오가 적용된 데이터
        """
        # 깊은 복사
        scenario_data = copy.deepcopy(base_data)

        variations = scenario.parameter_variations

        # 1. 자재 배출계수 변동
        if 'material_emission_factors' in variations:
            for material_name, change in variations['material_emission_factors'].items():
                if material_name in scenario_data['materials']:
                    original = scenario_data['materials'][material_name]['emission_factor']

                    if 'multiplier' in change:
                        # 배수 적용
                        new_value = original * change['multiplier']
                    elif 'value' in change:
                        # 절대값 설정
                        new_value = change['value']
                    elif 'offset' in change:
                        # 절대값 더하기
                        new_value = original + change['offset']
                    else:
                        continue

                    scenario_data['materials'][material_name]['emission_factor'] = new_value

        # 2. 비용 변동
        if 'costs' in variations:
            cost_types = ['virgin_cost', 'recycled_cost', 'low_carbon_cost']

            for cost_type in cost_types:
                if cost_type in variations['costs']:
                    change = variations['costs'][cost_type]

                    for material_name in scenario_data['materials']:
                        if cost_type in scenario_data['materials'][material_name]:
                            original = scenario_data['materials'][material_name][cost_type]

                            if 'multiplier' in change:
                                new_value = original * change['multiplier']
                            elif 'value' in change:
                                new_value = change['value']
                            elif 'offset' in change:
                                new_value = original + change['offset']
                            else:
                                continue

                            scenario_data['materials'][material_name][cost_type] = new_value

        # 3. 제약조건 한도 변동
        if 'constraint_bounds' in variations:
            # ConstraintManager를 통해 제약 수정
            # 이 부분은 RobustOptimizer에서 처리됨
            pass

        return scenario_data

    def create_preset_scenarios(self, scenario_type: str = 'standard'):
        """
        사전 정의된 시나리오 세트 생성

        Args:
            scenario_type: 시나리오 유형
                - 'standard': 표준 (base, optimistic, pessimistic)
                - 'regulatory': 규제 중심
                - 'market': 시장 중심
        """
        self.scenarios.clear()

        if scenario_type == 'standard':
            # 기준 시나리오
            self.add_scenario(
                name='Base Case',
                probability=0.5,
                parameter_variations={},
                description='현재 상황 유지'
            )

            # 낙관적 시나리오
            self.add_scenario(
                name='Optimistic',
                probability=0.3,
                parameter_variations={
                    'costs': {
                        'recycled_cost': {'multiplier': 0.9},  # 재활용 비용 10% 감소
                        'low_carbon_cost': {'multiplier': 0.85}  # 저탄소 비용 15% 감소
                    }
                },
                description='재활용/저탄소 기술 발전으로 비용 감소'
            )

            # 비관적 시나리오
            self.add_scenario(
                name='Pessimistic',
                probability=0.2,
                parameter_variations={
                    'costs': {
                        'recycled_cost': {'multiplier': 1.2},  # 재활용 비용 20% 증가
                        'low_carbon_cost': {'multiplier': 1.3}  # 저탄소 비용 30% 증가
                    }
                },
                description='공급망 혼란으로 비용 증가'
            )

        elif scenario_type == 'regulatory':
            # 엄격한 규제
            self.add_scenario(
                name='Strict Regulation',
                probability=0.3,
                parameter_variations={
                    'constraint_bounds': {
                        'premium_limit_pct': {'value': 5.0}  # 5%로 제한
                    }
                },
                description='탄소 규제 강화, 낮은 프리미엄 허용'
            )

            # 기준
            self.add_scenario(
                name='Base Regulation',
                probability=0.5,
                parameter_variations={},
                description='현행 규제 유지'
            )

            # 완화된 규제
            self.add_scenario(
                name='Relaxed Regulation',
                probability=0.2,
                parameter_variations={
                    'constraint_bounds': {
                        'premium_limit_pct': {'value': 20.0}  # 20%로 완화
                    }
                },
                description='규제 완화, 높은 프리미엄 허용'
            )

        else:
            raise ValueError(f"알 수 없는 시나리오 유형: {scenario_type}")

        print(f"✅ {scenario_type} 시나리오 세트 생성 완료: {len(self.scenarios)}개")

    def save_to_file(self, filepath: str):
        """시나리오를 JSON 파일로 저장"""
        data = {
            'scenarios': [s.to_dict() for s in self.scenarios]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 시나리오 저장: {filepath}")

    def load_from_file(self, filepath: str):
        """JSON 파일에서 시나리오 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.scenarios = [Scenario.from_dict(s) for s in data['scenarios']]

        print(f"📂 시나리오 로드: {filepath} ({len(self.scenarios)}개)")

    def get_summary(self) -> Dict[str, Any]:
        """시나리오 요약 정보"""
        return {
            'count': len(self.scenarios),
            'total_probability': sum(s.probability for s in self.scenarios),
            'scenarios': [
                {
                    'name': s.name,
                    'probability': s.probability,
                    'description': s.description
                }
                for s in self.scenarios
            ]
        }

    def __repr__(self) -> str:
        return f"<ScenarioManager(scenarios={len(self.scenarios)})>"

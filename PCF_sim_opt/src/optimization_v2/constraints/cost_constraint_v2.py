"""
비용 제약조건 (V2 - Refactored)

단순화된 비용 제약조건:
- Zero-Premium Baseline 개념 도입
- 3가지 모드 → 2가지 파라미터로 통합
- TotalCostCalculator와 통합
"""

from typing import Dict, Any, Tuple, Optional
import warnings
import pandas as pd
import pyomo.environ as pyo
from ..core.constraint_base import ConstraintBase


class CostConstraint(ConstraintBase):
    """
    비용 제약조건 클래스 (V2)

    TotalCostCalculator를 사용하여:
    - Zero-Premium Baseline: 환경 개선 없는 상태의 기준 비용
    - Premium Limit (프리미엄 한도): baseline 대비 % 제약
    - Absolute Premium Budget (절대 프리미엄 예산): 추가 프리미엄 절대액 제약
    """

    def __init__(self, cost_calculator):
        """
        비용 제약조건 초기화

        Args:
            cost_calculator: TotalCostCalculator 인스턴스
        """
        super().__init__(
            name="cost_constraint",
            description="비용 제약 (Zero-Premium Baseline 기반)"
        )
        self.cost_calculator = cost_calculator
        self.zero_premium_baseline: Optional[float] = None

        # 2가지 제약 파라미터만 유지
        self.premium_limit_pct: Optional[float] = None  # 예: 10 (10%)
        self.absolute_premium_budget: Optional[float] = None  # 예: 50000 (USD)

        # 레거시 호환성을 위한 속성들 (deprecated, 경고만 출력)
        self._legacy_mode = None
        self._legacy_carbon_weight = None
        self._legacy_cost_weight = None
        self._legacy_epsilon_limit = None

    def calculate_zero_premium_baseline(
        self,
        scenario_df: pd.DataFrame,
        material_classification: Dict[str, Any],
        original_df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Zero-Premium Baseline 계산 및 저장

        환경 개선 활동 없는 상태의 기준 비용 (RE100=0%, 재활용=0%, 저탄소=0%)

        Args:
            scenario_df: 시나리오 DataFrame
            material_classification: 자재 분류 정보
            original_df: 원본 DataFrame (optional)

        Returns:
            float: Zero-Premium Baseline 비용 (USD)
        """
        self.zero_premium_baseline = self.cost_calculator.calculate_zero_premium_baseline(
            scenario_df,
            material_classification,
            original_df
        )

        print(f"✅ Zero-Premium Baseline 계산 완료: ${self.zero_premium_baseline:,.2f}")
        return self.zero_premium_baseline

    def set_premium_limit(self, pct: float) -> None:
        """
        프리미엄 한도 설정 (% 단위)

        Total Premium ≤ Zero-Premium Baseline × (pct / 100)

        Args:
            pct: 프리미엄 한도 (예: 10 = +10%)
        """
        if pct < 0:
            raise ValueError("프리미엄 한도는 0 이상이어야 합니다.")

        self.premium_limit_pct = pct
        print(f"✅ 프리미엄 한도 설정: +{pct}%")

        if self.zero_premium_baseline:
            max_premium = self.zero_premium_baseline * (pct / 100)
            max_total_cost = self.zero_premium_baseline + max_premium
            print(f"   기준: ${self.zero_premium_baseline:,.2f}")
            print(f"   최대 프리미엄: ${max_premium:,.2f}")
            print(f"   최대 총 비용: ${max_total_cost:,.2f}")

    def set_absolute_premium_budget(self, budget: float) -> None:
        """
        절대 프리미엄 예산 설정

        Total Premium ≤ budget

        Args:
            budget: 프리미엄 예산 한도 (USD)
        """
        if budget < 0:
            raise ValueError("프리미엄 예산은 0 이상이어야 합니다.")

        self.absolute_premium_budget = budget
        print(f"✅ 절대 프리미엄 예산 설정: ${budget:,.2f}")

        if self.zero_premium_baseline:
            max_total_cost = self.zero_premium_baseline + budget
            print(f"   Zero-Premium Baseline: ${self.zero_premium_baseline:,.2f}")
            print(f"   최대 총 비용: ${max_total_cost:,.2f}")

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        제약조건 설정 검증

        Args:
            config: 설정 딕셔너리

        Returns:
            (is_valid, message)
        """
        # Cost calculator 필수
        if self.cost_calculator is None:
            return False, "Cost calculator가 설정되지 않았습니다."

        # 최소한 하나의 제약 필요
        has_constraint = (
            self.premium_limit_pct is not None or
            self.absolute_premium_budget is not None
        )

        if not has_constraint:
            return False, "프리미엄 한도 또는 절대 프리미엄 예산 중 하나는 설정되어야 합니다."

        # Premium 한도 사용 시 zero_premium_baseline 필요
        if self.premium_limit_pct is not None and self.zero_premium_baseline is None:
            return False, "프리미엄 한도 사용 시 Zero-Premium Baseline이 필요합니다. calculate_zero_premium_baseline()을 먼저 호출하세요."

        return True, "검증 완료"

    def check_feasibility(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        실현 가능성 확인

        Args:
            data: 시뮬레이션 데이터

        Returns:
            (is_feasible, message)
        """
        scenario_df = data.get('scenario_df')
        material_classification = data.get('material_classification')

        if scenario_df is None or material_classification is None:
            return False, "scenario_df 또는 material_classification이 없습니다."

        # Cost calculator 확인
        if self.cost_calculator is None:
            return False, "Cost calculator가 설정되지 않았습니다."

        return True, "실현 가능"

    def apply_to_model(self, model: pyo.ConcreteModel, data: Dict[str, Any]) -> None:
        """
        Pyomo 모델에 제약조건 적용

        프리미엄 기반 제약조건:
        - premium_limit_pct: total_premium ≤ baseline × (pct/100)
        - absolute_premium_budget: total_premium ≤ budget

        Args:
            model: Pyomo 최적화 모델
            data: 시뮬레이션 데이터
        """
        # 진단 로그
        print(f"\n{'='*60}")
        print(f"💰 비용 제약조건 적용 (CostConstraint V2)")
        print(f"{'='*60}")
        print(f"   • Zero-Premium Baseline: ${self.zero_premium_baseline:,.2f}" if self.zero_premium_baseline else "   • Zero-Premium Baseline: 미설정")

        # 총 프리미엄 계산
        try:
            total_premium_expr = self.cost_calculator.calculate_total_premium(model, data)
        except Exception as e:
            print(f"    ⚠️ 프리미엄 표현식 계산 실패: {e}")
            print(f"    → 비용 제약조건을 적용하지 않습니다.")
            import traceback
            traceback.print_exc()
            return

        # 프리미엄 한도 제약
        if self.premium_limit_pct is not None and self.zero_premium_baseline is not None:
            max_premium = self.zero_premium_baseline * (self.premium_limit_pct / 100)

            model.add_component(
                'premium_limit_constraint',
                pyo.Constraint(expr=total_premium_expr <= max_premium)
            )

            print(f"    • 프리미엄 한도: +{self.premium_limit_pct}%")
            print(f"      → 최대 프리미엄: ${max_premium:,.2f}")
            print(f"      → 최대 총 비용: ${self.zero_premium_baseline + max_premium:,.2f}")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        # 절대 프리미엄 예산 제약
        if self.absolute_premium_budget is not None:
            model.add_component(
                'absolute_premium_budget_constraint',
                pyo.Constraint(expr=total_premium_expr <= self.absolute_premium_budget)
            )

            max_total_cost = (self.zero_premium_baseline or 0) + self.absolute_premium_budget
            print(f"    • 절대 프리미엄 예산: ${self.absolute_premium_budget:,.2f}")
            print(f"      → 최대 총 비용: ${max_total_cost:,.2f}")
            print(f"      ✅ Pyomo 제약조건 추가 완료")

        print(f"{'='*60}\n")

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 직렬화

        Returns:
            설정 딕셔너리
        """
        return {
            'type': 'cost_constraint_v2',
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'zero_premium_baseline': self.zero_premium_baseline,
            'premium_limit_pct': self.premium_limit_pct,
            'absolute_premium_budget': self.absolute_premium_budget
        }

    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        딕셔너리에서 설정 로드

        Args:
            config: 설정 딕셔너리
        """
        self.name = config.get('name', 'cost_constraint')
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.zero_premium_baseline = config.get('zero_premium_baseline')
        self.premium_limit_pct = config.get('premium_limit_pct')
        self.absolute_premium_budget = config.get('absolute_premium_budget')

    def get_summary(self) -> str:
        """
        제약조건 요약

        Returns:
            요약 문자열
        """
        base_summary = super().get_summary()
        details = []

        if self.zero_premium_baseline:
            details.append(f"Zero-Premium Baseline: ${self.zero_premium_baseline:,.2f}")

        if self.premium_limit_pct is not None:
            details.append(f"프리미엄 한도: +{self.premium_limit_pct}%")

        if self.absolute_premium_budget is not None:
            details.append(f"절대 예산: ${self.absolute_premium_budget:,.2f}")

        detail_str = " | ".join(details) if details else "설정 없음"
        return f"{base_summary}\n  💰 {detail_str}"

    def get_display_info(self) -> Dict[str, Any]:
        """UI 표시용 비용 제약조건 정보"""
        info = {
            'mode': 'premium_based_v2',
            'mode_description': '프리미엄 기반 제약 (Zero-Premium Baseline)',
            'zero_premium_baseline': self.zero_premium_baseline,
            'settings': [],
            'pareto_warning': ''
        }

        if self.premium_limit_pct is not None:
            info['settings'].append(f"프리미엄 한도: +{self.premium_limit_pct}%")
        if self.absolute_premium_budget is not None:
            info['settings'].append(f"절대 프리미엄 예산: ${self.absolute_premium_budget:,.2f}")

        return info

    # ========== 레거시 호환성 메서드 (Deprecated) ==========

    def set_multi_objective_mode(
        self,
        carbon_weight: float = 0.7,
        cost_weight: float = 0.3
    ) -> None:
        """
        [DEPRECATED] 복합 최적화 모드 설정

        이 메서드는 deprecated되었습니다.
        다목적 최적화는 Pareto optimizer에서 직접 처리합니다.

        Args:
            carbon_weight: 탄소 배출 가중치 (사용되지 않음)
            cost_weight: 비용 가중치 (사용되지 않음)
        """
        warnings.warn(
            "set_multi_objective_mode()는 deprecated되었습니다. "
            "다목적 최적화는 WeightSweepOptimizer에서 직접 처리합니다. "
            "이 호출은 무시됩니다.",
            DeprecationWarning,
            stacklevel=2
        )
        self._legacy_mode = 'multi_objective'
        self._legacy_carbon_weight = carbon_weight
        self._legacy_cost_weight = cost_weight
        print("⚠️ [DEPRECATED] set_multi_objective_mode() 호출 무시됨")

    def set_constraint_mode(self) -> None:
        """
        [DEPRECATED] 제약 모드로 전환

        이 메서드는 deprecated되었습니다.
        V2에서는 항상 제약 모드로 동작합니다.
        """
        warnings.warn(
            "set_constraint_mode()는 deprecated되었습니다. "
            "V2에서는 항상 제약 모드로 동작합니다.",
            DeprecationWarning,
            stacklevel=2
        )
        self._legacy_mode = 'constraint'
        print("⚠️ [DEPRECATED] set_constraint_mode() 호출 무시됨 (V2는 항상 제약 모드)")

    def set_epsilon_constraint_mode(self, epsilon_limit: float) -> None:
        """
        [DEPRECATED] Epsilon-Constraint 모드 설정

        이 메서드는 deprecated되었습니다.
        대신 set_absolute_premium_budget()를 사용하세요.

        Args:
            epsilon_limit: 비용 상한 ($)
        """
        warnings.warn(
            "set_epsilon_constraint_mode()는 deprecated되었습니다. "
            "대신 set_absolute_premium_budget()를 사용하세요. "
            "자동으로 변환하여 적용합니다.",
            DeprecationWarning,
            stacklevel=2
        )

        # Adapter: epsilon_limit을 프리미엄 예산으로 변환
        if self.zero_premium_baseline is not None:
            premium_budget = epsilon_limit - self.zero_premium_baseline
            if premium_budget < 0:
                print(f"⚠️ epsilon_limit (${epsilon_limit:,.2f})가 Zero-Premium Baseline (${self.zero_premium_baseline:,.2f})보다 작습니다.")
                print(f"   → 프리미엄 예산을 0으로 설정합니다.")
                premium_budget = 0
            self.set_absolute_premium_budget(premium_budget)
        else:
            print("⚠️ Zero-Premium Baseline이 설정되지 않아 epsilon_limit을 그대로 사용합니다.")
            self.absolute_premium_budget = epsilon_limit

        self._legacy_mode = 'epsilon_constraint'
        self._legacy_epsilon_limit = epsilon_limit

    def calculate_baseline_cost(
        self,
        scenario_df: pd.DataFrame,
        original_df: pd.DataFrame,
        case_name: str = 'case1'
    ) -> float:
        """
        [DEPRECATED] 기준 비용 계산

        이 메서드는 deprecated되었습니다.
        대신 calculate_zero_premium_baseline()를 사용하세요.

        레거시 호환성: 이 메서드는 원래 RE100 프리미엄 비용을 반환했지만,
        이제는 Zero-Premium Baseline을 반환합니다.

        Args:
            scenario_df: 시나리오 DataFrame
            original_df: 원본 DataFrame
            case_name: 케이스 이름 (무시됨)

        Returns:
            float: Zero-Premium Baseline 비용
        """
        warnings.warn(
            "calculate_baseline_cost()는 deprecated되었습니다. "
            "대신 calculate_zero_premium_baseline()를 사용하세요. "
            "주의: 반환값의 의미가 변경되었습니다 (RE100 프리미엄 → Zero-Premium Baseline).",
            DeprecationWarning,
            stacklevel=2
        )

        material_classification = {}
        for material in scenario_df['자재명'].unique():
            material_classification[material] = {'type': 'General'}

        return self.calculate_zero_premium_baseline(
            scenario_df,
            material_classification,
            original_df
        )

    def get_multi_objective_expression(
        self,
        model: pyo.ConcreteModel,
        data: Dict[str, Any],
        carbon_expr: pyo.Expression,
        baseline_carbon: Optional[float] = None
    ) -> pyo.Expression:
        """
        [DEPRECATED] 복합 최적화 목적함수 생성

        이 메서드는 deprecated되었습니다.
        다목적 최적화는 WeightSweepOptimizer에서 직접 처리합니다.

        Args:
            model: Pyomo 모델
            data: 데이터
            carbon_expr: 탄소 표현식
            baseline_carbon: 기준 탄소

        Returns:
            carbon_expr (그대로 반환, 경고 출력)
        """
        warnings.warn(
            "get_multi_objective_expression()는 deprecated되었습니다. "
            "다목적 최적화는 WeightSweepOptimizer에서 직접 처리합니다. "
            "탄소 표현식을 그대로 반환합니다.",
            DeprecationWarning,
            stacklevel=2
        )
        print("⚠️ [DEPRECATED] get_multi_objective_expression() 호출 무시됨")
        return carbon_expr

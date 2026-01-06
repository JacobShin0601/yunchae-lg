"""
감축 목표 제약조건 관리 모듈

이 모듈은 PCF 감축 목표와 관련된 제약조건을 관리합니다.
감축률 목표, RE 적용 상하한, 자재 비율 제약 등 다양한 제약조건을
모델에 추가하는 기능을 제공합니다.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import json
import os
from pathlib import Path


class ReductionConstraintManager:
    """
    감축 목표 제약조건 관리 클래스
    
    이 클래스는 최적화 모델의 감축 목표 및 다양한 제약조건을 관리합니다.
    감축률 목표, 자재별 RE 적용률 범위, 자재 비율 제약 등을 설정하고
    최적화 모델에 적용합니다.
    """
    
    def __init__(self, 
                 original_pcf: float = 0.0,
                 config: Dict[str, Any] = None,
                 stable_var_dir: str = "stable_var",
                 user_id: Optional[str] = None,
                 debug_mode: bool = False):
        """
        ReductionConstraintManager 초기화
        
        Args:
            original_pcf: 기준 PCF 값
            config: 제약조건 설정
            stable_var_dir: stable_var 디렉토리 경로
            user_id: 사용자 ID (사용자별 데이터 사용시)
            debug_mode: 디버그 모드 사용 여부
        """
        self.debug_mode = debug_mode
        self.user_id = user_id
        self.stable_var_dir = Path(stable_var_dir)
        self.original_pcf = original_pcf
        self.config = config or self._get_default_config()
        
        # 목표 PCF 계산
        self.target_pcf = self._calculate_target_pcf()
        
        # 프리미엄 비용 관련 변수 초기화
        self.premium_cost_enabled = False
        self.premium_cost_target = 0.0
        self.baseline_premium_cost = 0.0
        
        # 디버그 정보 출력
        if self.debug_mode:
            self._print_debug_info()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 제약조건 설정 반환"""
        return {
            'reduction_target': {
                'min': -5,   # 최소 감축률 (%)
                'max': -10,  # 최대 감축률 (%)
            },
            're_rates': {
                'tier1': {'min': 0.1, 'max': 0.9},  # Tier1 RE 적용률 범위
                'tier2': {'min': 0.1, 'max': 0.9},  # Tier2 RE 적용률 범위
                'tier3': {'min': 0.1, 'max': 0.9},  # Tier3 RE 적용률 범위
            },
            'material_ratios': {
                'recycle': {'min': 0.05, 'max': 0.5},      # 재활용 비율 범위
                'low_carbon': {'min': 0.05, 'max': 0.3},  # 저탄소메탈 비율 범위
            },
            'premium_cost': {
                'enabled': False,          # 프리미엄 비용 제약 활성화 여부
                'reduction_target': 0.0,   # 프리미엄 비용 감축 목표 (%)
            }
        }
    
    def _calculate_target_pcf(self) -> Dict[str, float]:
        """감축 목표에 따른 목표 PCF 범위 계산"""
        # 감축률을 비율로 변환 (저장된 값이 음수이므로 절댓값 사용)
        min_reduction_abs = abs(self.config['reduction_target'].get('min', 0)) / 100
        max_reduction_abs = abs(self.config['reduction_target'].get('max', 0)) / 100
        
        # 목표 PCF 계산 (감축 = 기준값에서 감축률만큼 빼기)
        target_pcf_max = self.original_pcf * (1 - min_reduction_abs)  # 최소 감축률에 해당하는 최대 PCF
        target_pcf_min = self.original_pcf * (1 - max_reduction_abs)  # 최대 감축률에 해당하는 최소 PCF
        
        return {
            'min': target_pcf_min,
            'max': target_pcf_max
        }
    
    def _print_debug_info(self) -> None:
        """디버그 정보 출력"""
        print("===== ReductionConstraintManager 초기화 정보 =====")
        print(f"• 기준 PCF: {self.original_pcf:.4f} kgCO2eq")
        
        # 감축 목표 정보
        min_reduction_stored = self.config['reduction_target'].get('min', 0)
        max_reduction_stored = self.config['reduction_target'].get('max', 0)
        min_reduction_actual = abs(min_reduction_stored)  # 실제 감축률 (절댓값)
        max_reduction_actual = abs(max_reduction_stored)  # 실제 감축률 (절댓값)
        target_pcf_max = self.target_pcf['max']
        target_pcf_min = self.target_pcf['min']
        
        print(f"• 감축 목표 설정:")
        print(f"  - 최소 감축률: {min_reduction_actual:.1f}% → 목표 PCF 상한: {target_pcf_max:.4f} kgCO2eq")
        print(f"  - 최대 감축률: {max_reduction_actual:.1f}% → 목표 PCF 하한: {target_pcf_min:.4f} kgCO2eq")
        
        # RE 적용률 범위
        print(f"• RE 적용률 범위:")
        print(f"  - Tier1: {self.config['re_rates']['tier1']['min']*100:.1f}% ~ {self.config['re_rates']['tier1']['max']*100:.1f}%")
        print(f"  - Tier2: {self.config['re_rates']['tier2']['min']*100:.1f}% ~ {self.config['re_rates']['tier2']['max']*100:.1f}%")
        print(f"  - Tier3: {self.config['re_rates']['tier3']['min']*100:.1f}% ~ {self.config['re_rates']['tier3']['max']*100:.1f}%")
        
        # 자재 비율 범위
        print(f"• 자재 비율 범위:")
        print(f"  - 재활용: {self.config['material_ratios']['recycle']['min']*100:.1f}% ~ {self.config['material_ratios']['recycle']['max']*100:.1f}%")
        print(f"  - 저탄소메탈: {self.config['material_ratios']['low_carbon']['min']*100:.1f}% ~ {self.config['material_ratios']['low_carbon']['max']*100:.1f}%")
        
        # 프리미엄 비용 정보
        if self.premium_cost_enabled:
            print(f"• 프리미엄 비용 제약 활성화")
            print(f"  - 기준 프리미엄 비용: ${self.baseline_premium_cost:.2f}")
            print(f"  - 프리미엄 비용 감축 목표: {self.config['premium_cost'].get('reduction_target', 0.0):.1f}%")
        else:
            print(f"• 프리미엄 비용 제약 비활성화")
    
    def apply_constraints(self, model: pyo.ConcreteModel, material_types: Dict[str, Dict[str, Any]] = None) -> None:
        """
        최적화 모델에 모든 제약조건 적용
        
        Args:
            model: Pyomo 최적화 모델
            material_types: 자재 유형 정보
        """
        # PCF 감축 목표 제약
        self.add_pcf_reduction_constraints(model)
        
        # RE 적용률 범위 제약
        self.add_re_rate_constraints(model)
        
        # 자재 비율 제약 (Ni, Co, Li 자재)
        if material_types:
            self.add_material_ratio_constraints(model, material_types)
            
        # 프리미엄 비용 제약 (활성화된 경우)
        if self.premium_cost_enabled:
            self.add_premium_cost_constraints(model)
    
    def add_pcf_reduction_constraints(self, model: pyo.ConcreteModel) -> None:
        """PCF 감축 목표 제약조건 추가"""
        # 모델에 목표 PCF 파라미터 추가
        model.target_pcf_min = pyo.Param(initialize=self.target_pcf['min'])
        model.target_pcf_max = pyo.Param(initialize=self.target_pcf['max'])
        
        # 상한값 제약 (최소 감축률)
        def pcf_max_rule(model):
            total_pcf = sum(model.modified_emission[m] * model.quantity[m] for m in model.materials)
            return total_pcf <= model.target_pcf_max
        
        model.pcf_max_constraint = pyo.Constraint(rule=pcf_max_rule)
        
        # 하한값 제약 (최대 감축률)
        def pcf_min_rule(model):
            total_pcf = sum(model.modified_emission[m] * model.quantity[m] for m in model.materials)
            return total_pcf >= model.target_pcf_min
        
        model.pcf_min_constraint = pyo.Constraint(rule=pcf_min_rule)
    
    def add_re_rate_constraints(self, model: pyo.ConcreteModel) -> None:
        """RE 적용률 범위 제약조건 추가"""
        # Tier1 RE 범위
        tier1_min = self.config['re_rates']['tier1']['min']
        tier1_max = self.config['re_rates']['tier1']['max']
        
        def tier1_min_rule(model, m):
            return model.tier1_re[m] >= tier1_min
        
        def tier1_max_rule(model, m):
            return model.tier1_re[m] <= tier1_max
        
        model.tier1_min_constraint = pyo.Constraint(model.materials, rule=tier1_min_rule)
        model.tier1_max_constraint = pyo.Constraint(model.materials, rule=tier1_max_rule)
        
        # Tier2 RE 범위
        tier2_min = self.config['re_rates']['tier2']['min']
        tier2_max = self.config['re_rates']['tier2']['max']
        
        def tier2_min_rule(model, m):
            return model.tier2_re[m] >= tier2_min
        
        def tier2_max_rule(model, m):
            return model.tier2_re[m] <= tier2_max
        
        model.tier2_min_constraint = pyo.Constraint(model.materials, rule=tier2_min_rule)
        model.tier2_max_constraint = pyo.Constraint(model.materials, rule=tier2_max_rule)
        
        # Tier3 RE 범위
        tier3_min = self.config['re_rates']['tier3']['min']
        tier3_max = self.config['re_rates']['tier3']['max']
        
        def tier3_min_rule(model, m):
            return model.tier3_re[m] >= tier3_min
        
        def tier3_max_rule(model, m):
            return model.tier3_re[m] <= tier3_max
        
        model.tier3_min_constraint = pyo.Constraint(model.materials, rule=tier3_min_rule)
        model.tier3_max_constraint = pyo.Constraint(model.materials, rule=tier3_max_rule)
    
    def add_material_ratio_constraints(self, model: pyo.ConcreteModel, material_types: Dict[str, Dict[str, Any]]) -> None:
        """자재 비율(재활용, 저탄소메탈) 제약조건 추가"""
        # Ni, Co, Li 자재 필터링
        ni_co_li_materials = [m for m, info in material_types.items() if info.get('is_ni_co_li', False)]
        
        if not ni_co_li_materials:
            return
        
        # 재활용 비율 범위
        recycle_min = self.config['material_ratios']['recycle']['min']
        recycle_max = self.config['material_ratios']['recycle']['max']
        
        def recycle_min_rule(model, m):
            if m not in ni_co_li_materials:
                return pyo.Constraint.Skip
            return model.recycle_ratio[m] >= recycle_min
        
        def recycle_max_rule(model, m):
            if m not in ni_co_li_materials:
                return pyo.Constraint.Skip
            return model.recycle_ratio[m] <= recycle_max
        
        model.recycle_min_constraint = pyo.Constraint(model.materials, rule=recycle_min_rule)
        model.recycle_max_constraint = pyo.Constraint(model.materials, rule=recycle_max_rule)
        
        # 저탄소 메탈 비율 범위
        low_carbon_min = self.config['material_ratios']['low_carbon']['min']
        low_carbon_max = self.config['material_ratios']['low_carbon']['max']
        
        def low_carbon_min_rule(model, m):
            if m not in ni_co_li_materials:
                return pyo.Constraint.Skip
            return model.low_carbon_ratio[m] >= low_carbon_min
        
        def low_carbon_max_rule(model, m):
            if m not in ni_co_li_materials:
                return pyo.Constraint.Skip
            return model.low_carbon_ratio[m] <= low_carbon_max
        
        model.low_carbon_min_constraint = pyo.Constraint(model.materials, rule=low_carbon_min_rule)
        model.low_carbon_max_constraint = pyo.Constraint(model.materials, rule=low_carbon_max_rule)
        
        # 비율 합계 제약 (재활용 + 저탄소메탈 + 신재 = 1)
        def ratio_sum_rule(model, m):
            if m not in ni_co_li_materials:
                return pyo.Constraint.Skip
            return model.recycle_ratio[m] + model.low_carbon_ratio[m] + model.virgin_ratio[m] == 1.0
        
        model.ratio_sum_constraint = pyo.Constraint(model.materials, rule=ratio_sum_rule)
    
    def update_reduction_targets(self, min_reduction: float, max_reduction: float) -> None:
        """감축 목표 업데이트"""
        # 감축률 설정 업데이트
        self.config['reduction_target']['min'] = min_reduction
        self.config['reduction_target']['max'] = max_reduction
        
        # 목표 PCF 재계산
        self.target_pcf = self._calculate_target_pcf()
        
        if self.debug_mode:
            print(f"✅ 감축 목표가 업데이트되었습니다: {min_reduction}% ~ {max_reduction}%")
            print(f"• 목표 PCF 범위: {self.target_pcf['min']:.4f} ~ {self.target_pcf['max']:.4f} kgCO2eq")
    
    def update_re_rate_limits(self, tier: int, min_rate: float, max_rate: float) -> None:
        """RE 적용률 범위 업데이트"""
        if tier < 1 or tier > 3:
            raise ValueError(f"유효하지 않은 Tier 번호: {tier}. 1-3 사이의 값이어야 합니다.")
        
        tier_key = f"tier{tier}"
        
        # 범위 검증
        if not (0 <= min_rate <= max_rate <= 1):
            raise ValueError(f"유효하지 않은 RE 적용률 범위: {min_rate} ~ {max_rate}. 0-1 사이의 값이어야 합니다.")
        
        # 범위 업데이트
        self.config['re_rates'][tier_key]['min'] = min_rate
        self.config['re_rates'][tier_key]['max'] = max_rate
        
        if self.debug_mode:
            print(f"✅ Tier{tier} RE 적용률 범위가 업데이트되었습니다: {min_rate*100:.1f}% ~ {max_rate*100:.1f}%")
    
    def update_material_ratio_limits(self, ratio_type: str, min_ratio: float, max_ratio: float) -> None:
        """자재 비율(재활용, 저탄소메탈) 범위 업데이트"""
        if ratio_type not in ['recycle', 'low_carbon']:
            raise ValueError(f"유효하지 않은 비율 유형: {ratio_type}. 'recycle' 또는 'low_carbon'이어야 합니다.")
        
        # 범위 검증
        if not (0 <= min_ratio <= max_ratio <= 1):
            raise ValueError(f"유효하지 않은 비율 범위: {min_ratio} ~ {max_ratio}. 0-1 사이의 값이어야 합니다.")
        
        # 범위 업데이트
        self.config['material_ratios'][ratio_type]['min'] = min_ratio
        self.config['material_ratios'][ratio_type]['max'] = max_ratio
        
        if self.debug_mode:
            print(f"✅ {ratio_type} 비율 범위가 업데이트되었습니다: {min_ratio*100:.1f}% ~ {max_ratio*100:.1f}%")
    
    def set_premium_cost_constraint(self, enabled: bool, baseline_cost: float = 0.0, reduction_target: float = 0.0) -> None:
        """프리미엄 비용 제약 설정
        
        Args:
            enabled: 프리미엄 비용 제약 활성화 여부
            baseline_cost: 기준 프리미엄 비용
            reduction_target: 프리미엄 비용 감축 목표 (%)
        """
        # 프리미엄 비용 제약 설정
        self.premium_cost_enabled = enabled
        self.baseline_premium_cost = baseline_cost
        
        # 설정 저장
        if 'premium_cost' not in self.config:
            self.config['premium_cost'] = {}
        
        self.config['premium_cost']['enabled'] = enabled
        self.config['premium_cost']['reduction_target'] = reduction_target
        
        if self.debug_mode:
            if enabled:
                print(f"✅ 프리미엄 비용 제약이 활성화되었습니다.")
                print(f"• 기준 프리미엄 비용: ${baseline_cost:.2f}")
                print(f"• 프리미엄 비용 감축 목표: {reduction_target:.1f}%")
            else:
                print(f"✅ 프리미엄 비용 제약이 비활성화되었습니다.")
    
    def add_premium_cost_constraints(self, model: pyo.ConcreteModel) -> None:
        """프리미엄 비용 제약조건 추가
        
        프리미엄 비용은 MaterialPremiumCostCalculator를 통해 계산되며,
        이 함수는 해당 비용이 목표 감축률을 달성하도록 제약조건을 추가합니다.
        
        Args:
            model: Pyomo 최적화 모델
        """
        if not self.premium_cost_enabled or self.baseline_premium_cost <= 0:
            return
        
        # 목표 감축률로 최대 허용 비용 계산
        reduction_target = self.config['premium_cost'].get('reduction_target', 0.0) / 100  # 비율로 변환
        max_allowed_cost = self.baseline_premium_cost * (1 - reduction_target)
        
        # 모델에 파라미터 추가
        model.baseline_premium_cost = pyo.Param(initialize=self.baseline_premium_cost)
        model.max_premium_cost = pyo.Param(initialize=max_allowed_cost)
        
        # 프리미엄 비용 변수 (최적화 과정에서 계산됨)
        model.total_premium_cost = pyo.Var(bounds=(0, None), initialize=self.baseline_premium_cost)
        
        # 자재별 프리미엄 비용 변수
        model.material_premium_costs = pyo.Var(model.materials, bounds=(0, None), initialize=0.0)
        
        # 자재별 프리미엄 비용 계산 제약조건
        # 이 부분은 MaterialPremiumCostCalculator의 로직을 간소화하여 모델에 적용
        def material_premium_cost_rule(model, m):
            # 가중 비용 = RE 적용률 * 해당 Tier 비용
            # 실제 구현에서는 자재별 특성에 따라 더 복잡한 계산이 필요할 수 있음
            weighted_cost = (model.tier1_re[m] * 0.5 + model.tier2_re[m] * 0.3) * model.original_emission[m] * 0.1
            return model.material_premium_costs[m] == weighted_cost * model.quantity[m]
        
        model.material_premium_cost_constraint = pyo.Constraint(model.materials, rule=material_premium_cost_rule)
        
        # 총 프리미엄 비용 계산 제약조건
        def total_premium_cost_rule(model):
            return model.total_premium_cost == sum(model.material_premium_costs[m] for m in model.materials)
        
        model.total_premium_cost_constraint = pyo.Constraint(rule=total_premium_cost_rule)
        
        # 프리미엄 비용 감축 목표 제약조건
        def premium_cost_target_rule(model):
            return model.total_premium_cost <= model.max_premium_cost
        
        model.premium_cost_target_constraint = pyo.Constraint(rule=premium_cost_target_rule)
    
    def validate_constraints(self, optimized_pcf: float, optimized_premium_cost: float = None) -> Dict[str, Any]:
        """최적해 제약조건 만족 여부 확인"""
        results = {
            'is_valid': True,
            'messages': [],
            'details': {
                'original_pcf': self.original_pcf,
                'optimized_pcf': optimized_pcf,
                'target_pcf_min': self.target_pcf['min'],
                'target_pcf_max': self.target_pcf['max'],
                'reduction_percentage': ((self.original_pcf - optimized_pcf) / self.original_pcf) * 100
            }
        }
        
        # 프리미엄 비용 정보 추가 (활성화된 경우)
        if self.premium_cost_enabled and optimized_premium_cost is not None:
            cost_reduction = ((self.baseline_premium_cost - optimized_premium_cost) / self.baseline_premium_cost) * 100
            results['details']['baseline_premium_cost'] = self.baseline_premium_cost
            results['details']['optimized_premium_cost'] = optimized_premium_cost
            results['details']['premium_cost_reduction_percentage'] = cost_reduction
        
        # PCF 제약 검증
        if optimized_pcf > self.target_pcf['max']:
            results['is_valid'] = False
            results['messages'].append(f"최적화된 PCF({optimized_pcf:.4f})가 목표 상한({self.target_pcf['max']:.4f})을 초과합니다.")
        
        if optimized_pcf < self.target_pcf['min']:
            results['is_valid'] = False
            results['messages'].append(f"최적화된 PCF({optimized_pcf:.4f})가 목표 하한({self.target_pcf['min']:.4f}) 미만입니다.")
        
        # 감축률 계산
        reduction = ((self.original_pcf - optimized_pcf) / self.original_pcf) * 100
        min_reduction = self.config['reduction_target']['min']
        max_reduction = self.config['reduction_target']['max']
        
        if reduction < min_reduction:
            results['is_valid'] = False
            results['messages'].append(f"감축률({reduction:.2f}%)이 최소 목표({min_reduction:.2f}%) 미만입니다.")
        
        if reduction > max_reduction:
            results['is_valid'] = False
            results['messages'].append(f"감축률({reduction:.2f}%)이 최대 목표({max_reduction:.2f}%) 초과입니다.")
        
        # 프리미엄 비용 제약 검증
        if self.premium_cost_enabled and optimized_premium_cost is not None:
            target_reduction = self.config['premium_cost'].get('reduction_target', 0.0)
            actual_reduction = ((self.baseline_premium_cost - optimized_premium_cost) / self.baseline_premium_cost) * 100
            
            if actual_reduction < target_reduction:
                results['is_valid'] = False
                results['messages'].append(f"프리미엄 비용 감축률({actual_reduction:.2f}%)이 목표({target_reduction:.2f}%) 미만입니다.")
        
        # 검증 결과 요약
        if results['is_valid']:
            summary = f"✅ 모든 제약조건이 만족됩니다. 감축률: {reduction:.2f}%"
            if self.premium_cost_enabled and optimized_premium_cost is not None:
                cost_reduction = ((self.baseline_premium_cost - optimized_premium_cost) / self.baseline_premium_cost) * 100
                summary += f", 프리미엄 비용 감축률: {cost_reduction:.2f}%"
            results['messages'].append(summary)
        
        return results
"""
자재별 프리미엄 비용 계산 모듈

이 모듈은 자재별 프리미엄 비용을 계산하는 기능을 제공합니다.
cost_by_tier.json의 tier, material, country별 expected_cost 데이터를 기반으로
각 자재의 프리미엄 비용을 계산하고, 총 비용을 집계합니다.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


class MaterialPremiumCostCalculator:
    """
    자재 프리미엄 비용 계산기
    
    각 자재의 tier, 소요량, 국가에 따라 프리미엄 비용을 계산합니다.
    cost_by_tier.json에서 자재 및 tier별 비용 정보를 로드하고,
    자재별 소요량과 매칭하여 프리미엄 비용을 산출합니다.
    """
    
    def __init__(self, 
                 simulation_data: Dict[str, pd.DataFrame] = None,
                 stable_var_dir: str = "stable_var",
                 user_id: Optional[str] = None,
                 debug_mode: bool = False):
        """
        MaterialPremiumCostCalculator 초기화
        
        Args:
            simulation_data: 시뮬레이션 데이터 (시나리오 및 참조 데이터프레임)
            stable_var_dir: stable_var 디렉토리 경로
            user_id: 사용자 ID (사용자별 데이터 사용시)
            debug_mode: 디버그 모드 사용 여부
        """
        self.debug_mode = debug_mode
        self.user_id = user_id
        self.stable_var_dir = Path(stable_var_dir)
        
        # 시뮬레이션 데이터 설정
        self.simulation_data = simulation_data or {}
        self.scenario_df = self.simulation_data.get('scenario_df', pd.DataFrame())
        self.original_df = self.simulation_data.get('original_df', pd.DataFrame())
        
        # 비용 데이터 로드
        self.cost_data = self._load_cost_data()
        
        # 자재별 기본 국가 매핑
        self.material_country_mapping = self._create_material_country_mapping()
        
        # 디버그 정보 출력
        if self.debug_mode:
            self._print_debug_info()
    
    def _load_cost_data(self) -> Dict[str, Any]:
        """비용 데이터 로드"""
        try:
            file_path = self.stable_var_dir / "cost_by_tier.json"
            if not os.path.exists(file_path):
                # 사용자 경로 확인
                if self.user_id:
                    file_path = self.stable_var_dir / self.user_id / "cost_by_tier.json"
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            if self.debug_mode:
                print(f"경고: cost_by_tier.json을 찾을 수 없습니다. 경로: {file_path}")
            
            # 기본값 반환
            return {"tier_data": [], "unit_info": {}}
        except Exception as e:
            if self.debug_mode:
                print(f"비용 데이터 로드 중 오류 발생: {e}")
            return {"tier_data": [], "unit_info": {}}
    
    def _create_material_country_mapping(self) -> Dict[str, str]:
        """자재별 기본 국가 매핑 생성"""
        mapping = {}
        
        # 원본 데이터프레임에서 국가 정보 추출
        if '자재명' in self.original_df.columns and '지역' in self.original_df.columns:
            for idx, row in self.original_df.iterrows():
                if pd.notna(row['자재명']) and pd.notna(row['지역']):
                    mapping[row['자재명']] = row['지역']
        
        # 시나리오 데이터프레임에서 보충
        if '자재명' in self.scenario_df.columns and '지역' in self.scenario_df.columns:
            for idx, row in self.scenario_df.iterrows():
                if pd.notna(row['자재명']) and pd.notna(row['지역']) and row['자재명'] not in mapping:
                    mapping[row['자재명']] = row['지역']
        
        return mapping
    
    def _print_debug_info(self) -> None:
        """디버그 정보 출력"""
        print("===== MaterialPremiumCostCalculator 초기화 정보 =====")
        
        # 비용 데이터 정보
        tier_data = self.cost_data.get('tier_data', [])
        print(f"• 비용 데이터 항목 수: {len(tier_data)}개")
        
        # 고유 자재 및 국가
        materials = set(item.get('material', '') for item in tier_data)
        countries = set(item.get('country', '') for item in tier_data)
        print(f"• 비용 데이터 포함 자재: {len(materials)}개")
        print(f"• 비용 데이터 포함 국가: {len(countries)}개")
        
        # 자재-국가 매핑 정보
        print(f"• 자재-국가 매핑: {len(self.material_country_mapping)}개")
        for material, country in self.material_country_mapping.items():
            print(f"  - {material}: {country}")
    
    def update_material_countries(self, material_countries: Dict[str, str]) -> None:
        """
        자재별 국가 정보 업데이트
        
        Args:
            material_countries: {자재명: 국가명} 형태의 딕셔너리
        """
        self.material_country_mapping.update(material_countries)
        
        if self.debug_mode:
            print(f"✅ 자재별 국가 정보 업데이트됨: {len(material_countries)}개 항목")
    
    def get_premium_cost_for_material(self, material_name: str, tier: str, quantity: float) -> float:
        """
        단일 자재의 프리미엄 비용 계산
        
        Args:
            material_name: 자재명
            tier: Tier 레벨 (Tier1, Tier2, Tier3)
            quantity: 소요량(kg)
            
        Returns:
            float: 프리미엄 비용
        """
        # 자재품목 추출 (자재명에서 제품/업체명 등을 제외한 핵심 품목만 사용)
        material_type = self._extract_material_type(material_name)
        
        # 국가 정보 확인
        country = self.material_country_mapping.get(material_name)
        if not country:
            if self.debug_mode:
                print(f"경고: '{material_name}' 자재의 국가 정보를 찾을 수 없습니다.")
            return 0.0
        
        # 일치하는 비용 데이터 항목 찾기
        matching_item = None
        for item in self.cost_data.get('tier_data', []):
            # 1. 자재품목, tier, 국가가 모두 일치하는 경우
            if (item.get('material') == material_type and 
                item.get('tier') == tier and 
                item.get('country') == country):
                matching_item = item
                break
            
            # 2. 자재품목과 tier가 일치하고 국가가 다른 경우 (임시 저장)
            if (not matching_item and
                item.get('material') == material_type and 
                item.get('tier') == tier):
                matching_item = item
        
        # 일치하는 데이터가 없는 경우
        if not matching_item:
            if self.debug_mode:
                print(f"경고: '{material_name}' (Tier: {tier}, 국가: {country})에 대한 비용 정보를 찾을 수 없습니다.")
            return 0.0
        
        # 예상 비용 계산: 소요량 × 단위당 비용
        expected_cost = matching_item.get('expected_cost', 0)
        premium_cost = quantity * expected_cost
        
        return premium_cost
    
    def _extract_material_type(self, material_name: str) -> str:
        """
        자재명에서 기본 자재 유형 추출
        예: "L&F 양극재" -> "양극재"
        """
        # 기본 자재 유형 목록
        material_types = [
            "양극재", "음극재", "분리막", "전해액", "동박", "양극재_전구체",
            "음극재(천연)", "음극재(인조)", "음극재(SiO)", "분리막_코팅", "분리막_원단"
        ]
        
        # 자재명에 포함된 자재 유형 찾기
        for material_type in material_types:
            if material_type in material_name:
                return material_type
        
        # 특별 케이스 처리
        if "CAM" in material_name:
            return "양극재"
        elif "Anode" in material_name:
            return "음극재(천연)"  # 기본값으로 천연 음극재 가정
        elif "Separator" in material_name:
            return "분리막"
        elif "Copper" in material_name or "Cu" in material_name:
            return "동박"
        
        # 자재품목을 시나리오 데이터에서 찾기
        if '자재명' in self.scenario_df.columns and '자재품목' in self.scenario_df.columns:
            for idx, row in self.scenario_df.iterrows():
                if row['자재명'] == material_name and pd.notna(row['자재품목']):
                    return row['자재품목']
        
        # 기본값
        return material_name
    
    def calculate_baseline_premium_costs(self) -> Dict[str, float]:
        """
        기준 시나리오의 프리미엄 비용 계산
        
        Returns:
            Dict: {자재명: 프리미엄 비용} 형태의 딕셔너리와 총합
        """
        if len(self.scenario_df) == 0:
            return {'total': 0.0}
        
        # 자재별 프리미엄 비용 계산
        premium_costs = {}
        total_cost = 0.0
        
        for idx, row in self.scenario_df.iterrows():
            material_name = row['자재명']
            
            # 소요량 확인
            if '제품총소요량(kg)' in row:
                quantity = row['제품총소요량(kg)']
            else:
                quantity = 0.0
                if self.debug_mode:
                    print(f"경고: '{material_name}' 자재의 소요량 정보가 없습니다.")
                continue
            
            # Tier1 비용 계산
            tier1_cost = self.get_premium_cost_for_material(material_name, "Tier1", quantity)
            
            # Tier2 비용 계산 (자재에 따라 다름)
            tier2_cost = self.get_premium_cost_for_material(material_name, "Tier2", quantity)
            
            # 총 비용 계산
            material_cost = tier1_cost + tier2_cost
            premium_costs[material_name] = material_cost
            total_cost += material_cost
        
        # 총합 포함
        premium_costs['total'] = total_cost
        
        return premium_costs
    
    def calculate_optimized_premium_costs(self, 
                                          optimization_results: Dict[str, Any], 
                                          material_countries: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        최적화 결과에 따른 프리미엄 비용 계산
        
        Args:
            optimization_results: 최적화 결과
            material_countries: 최적화 결과의 자재-국가 매핑 (None이면 기존 매핑 사용)
            
        Returns:
            Dict: {자재명: 프리미엄 비용} 형태의 딕셔너리와 총합
        """
        # 결과 유효성 검증
        if (not optimization_results or 
            optimization_results.get('status') != 'optimal' or 
            'variables' not in optimization_results):
            return {'total': 0.0}
        
        # 자재-국가 매핑 업데이트 (필요한 경우)
        if material_countries:
            self.update_material_countries(material_countries)
        
        # 자재별 프리미엄 비용 계산
        premium_costs = {}
        total_cost = 0.0
        
        # 변수에서 자재 정보 추출
        variables = optimization_results['variables']
        material_names = set()
        
        # 모든 자재 이름 추출
        for var_name in variables.keys():
            if var_name.startswith('tier1_re_') or var_name.startswith('tier2_re_'):
                parts = var_name.split('_', 2)
                if len(parts) >= 3:
                    material_name = parts[2]
                    material_names.add(material_name)
        
        # 자재별 프리미엄 비용 계산
        for material_name in material_names:
            # 소요량 확인
            quantity = 0.0
            quantity_var_name = f'quantity_{material_name}'
            if quantity_var_name in variables:
                quantity = variables[quantity_var_name]
            else:
                # 소요량 정보가 없으면 시나리오 데이터에서 찾기
                for idx, row in self.scenario_df.iterrows():
                    if row['자재명'] == material_name and '제품총소요량(kg)' in row:
                        quantity = row['제품총소요량(kg)']
                        break
            
            if quantity <= 0:
                if self.debug_mode:
                    print(f"경고: '{material_name}' 자재의 소요량이 0 또는 음수입니다.")
                continue
            
            # Tier1 RE 적용률과 비용 계산
            tier1_re = 0.0
            tier1_var_name = f'tier1_re_{material_name}'
            if tier1_var_name in variables:
                tier1_re = variables[tier1_var_name]
            
            # Tier1 비용: 적용률 × 소요량 × 비용
            tier1_cost = 0.0
            if tier1_re > 0:
                # RE가 적용된 부분에만 프리미엄 비용 발생
                tier1_premium = self.get_premium_cost_for_material(material_name, "Tier1", quantity)
                tier1_cost = tier1_re * tier1_premium
            
            # Tier2 RE 적용률과 비용 계산
            tier2_re = 0.0
            tier2_var_name = f'tier2_re_{material_name}'
            if tier2_var_name in variables:
                tier2_re = variables[tier2_var_name]
            
            # Tier2 비용: 적용률 × 소요량 × 비용
            tier2_cost = 0.0
            if tier2_re > 0:
                tier2_premium = self.get_premium_cost_for_material(material_name, "Tier2", quantity)
                tier2_cost = tier2_re * tier2_premium
            
            # 총 비용 계산
            material_cost = tier1_cost + tier2_cost
            premium_costs[material_name] = material_cost
            total_cost += material_cost
        
        # 총합 포함
        premium_costs['total'] = total_cost
        
        return premium_costs
    
    def calculate_cost_reduction(self, 
                                baseline_costs: Dict[str, float], 
                                optimized_costs: Dict[str, float]) -> Dict[str, Any]:
        """
        프리미엄 비용 감축률 계산
        
        Args:
            baseline_costs: 기준 시나리오 비용
            optimized_costs: 최적화 시나리오 비용
            
        Returns:
            Dict: 비용 감축 정보
        """
        baseline_total = baseline_costs.get('total', 0.0)
        optimized_total = optimized_costs.get('total', 0.0)
        
        # 감축액 및 감축률 계산
        reduction_amount = baseline_total - optimized_total
        
        # 0으로 나누기 방지
        if baseline_total > 0:
            reduction_percentage = (reduction_amount / baseline_total) * 100
        else:
            reduction_percentage = 0.0
        
        # 자재별 감축 정보
        materials = set(baseline_costs.keys()) | set(optimized_costs.keys())
        materials.discard('total')  # 총합은 제외
        
        material_reductions = {}
        for material in materials:
            baseline = baseline_costs.get(material, 0.0)
            optimized = optimized_costs.get(material, 0.0)
            amount = baseline - optimized
            
            # 0으로 나누기 방지
            if baseline > 0:
                percentage = (amount / baseline) * 100
            else:
                percentage = 0.0
            
            material_reductions[material] = {
                'baseline': baseline,
                'optimized': optimized,
                'reduction_amount': amount,
                'reduction_percentage': percentage
            }
        
        return {
            'baseline_total': baseline_total,
            'optimized_total': optimized_total,
            'reduction_amount': reduction_amount,
            'reduction_percentage': reduction_percentage,
            'material_reductions': material_reductions
        }
    
    def format_cost_results(self, cost_data: Dict[str, Any]) -> Dict[str, str]:
        """
        비용 결과 포맷팅
        
        Args:
            cost_data: 비용 계산 결과
            
        Returns:
            Dict: 포맷팅된 비용 정보
        """
        formatted = {}
        
        # 기본 정보
        formatted['baseline_total'] = f"${cost_data['baseline_total']:.2f}"
        formatted['optimized_total'] = f"${cost_data['optimized_total']:.2f}"
        formatted['reduction_amount'] = f"${cost_data['reduction_amount']:.2f}"
        formatted['reduction_percentage'] = f"{cost_data['reduction_percentage']:.2f}%"
        
        # 자재별 정보
        material_info = []
        for material, data in cost_data['material_reductions'].items():
            material_info.append({
                'material': material,
                'baseline': f"${data['baseline']:.2f}",
                'optimized': f"${data['optimized']:.2f}",
                'reduction': f"${data['reduction_amount']:.2f} ({data['reduction_percentage']:.1f}%)"
            })
        
        formatted['material_details'] = material_info
        
        return formatted
"""
최적 생산국가 추천 모듈
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json

class LocationRecommender:
    """
    여러 조건에 따라 최적의 생산국가를 추천하는 클래스
    
    주요 기능:
    - 국가별 전력 배출계수 기반 추천
    - 국가별 생산 비용 기반 추천
    - 여러 요소를 고려한 종합 점수 계산
    """
    
    def __init__(self, stable_var_dir: str = "stable_var", user_id: Optional[str] = None):
        """
        Args:
            stable_var_dir: stable_var 디렉토리 경로
            user_id: 사용자 ID (사용자별 작업공간 사용 시)
        """
        self.stable_var_dir = Path(stable_var_dir)
        if user_id:
            self.stable_var_dir = Path(stable_var_dir) / user_id
            
        self.electricity_coefficients = {}
        self.country_costs = {}
        self.transport_costs = {}
        self.country_scores = {}
        
        # 데이터 로드
        self._load_electricity_coefficients()
        self._load_country_costs()
        self._load_transport_costs()
    
    def _load_electricity_coefficients(self) -> None:
        """국가별 전력 배출계수 로드"""
        try:
            coef_path = self.stable_var_dir / "electricity_coef_by_country.json"
            if coef_path.exists():
                with open(coef_path, 'r', encoding='utf-8') as f:
                    self.electricity_coefficients = json.load(f)
        except Exception as e:
            print(f"전력 배출계수 로드 실패: {e}")
            # 기본값 설정
            self.electricity_coefficients = {
                "한국": 0.4567,
                "중국": 0.5839,
                "일본": 0.4689,
                "폴란드": 0.7814,
                "독일": 0.3467,
                "미국": 0.4112
            }
    
    def _load_country_costs(self) -> None:
        """국가별 생산 비용 로드"""
        try:
            cost_path = self.stable_var_dir / "country_costs.json"
            if cost_path.exists():
                with open(cost_path, 'r', encoding='utf-8') as f:
                    self.country_costs = json.load(f)
            else:
                # 기본값 설정
                self.country_costs = {
                    "한국": 1.0,     # 기준값
                    "중국": 0.85,    # 한국보다 15% 저렴
                    "일본": 1.2,     # 한국보다 20% 비쌈
                    "폴란드": 0.95,  # 한국보다 5% 저렴
                    "독일": 1.15,    # 한국보다 15% 비쌈
                    "미국": 1.1      # 한국보다 10% 비쌈
                }
        except Exception as e:
            print(f"국가별 비용 로드 실패: {e}")
            # 기본값 설정
            self.country_costs = {
                "한국": 1.0,
                "중국": 0.85,
                "일본": 1.2,
                "폴란드": 0.95,
                "독일": 1.15,
                "미국": 1.1
            }
    
    def _load_transport_costs(self) -> None:
        """국가별 운송 비용 로드"""
        try:
            transport_path = self.stable_var_dir / "transport_costs.json"
            if transport_path.exists():
                with open(transport_path, 'r', encoding='utf-8') as f:
                    self.transport_costs = json.load(f)
            else:
                # 기본값 설정 (국가간 운송 비용 매트릭스)
                self.transport_costs = {
                    "한국": {"한국": 0.0, "중국": 0.05, "일본": 0.08, "폴란드": 0.15},
                    "중국": {"한국": 0.05, "중국": 0.0, "일본": 0.07, "폴란드": 0.18},
                    "일본": {"한국": 0.08, "중국": 0.07, "일본": 0.0, "폴란드": 0.2},
                    "폴란드": {"한국": 0.15, "중국": 0.18, "일본": 0.2, "폴란드": 0.0}
                }
        except Exception as e:
            print(f"운송 비용 로드 실패: {e}")
            # 기본값 설정
            self.transport_costs = {
                "한국": {"한국": 0.0, "중국": 0.05, "일본": 0.08, "폴란드": 0.15},
                "중국": {"한국": 0.05, "중국": 0.0, "일본": 0.07, "폴란드": 0.18},
                "일본": {"한국": 0.08, "중국": 0.07, "일본": 0.0, "폴란드": 0.2},
                "폴란드": {"한국": 0.15, "중국": 0.18, "일본": 0.2, "폴란드": 0.0}
            }
    
    def get_available_countries(self) -> List[str]:
        """
        사용 가능한 국가 목록 반환
        
        Returns:
            List[str]: 국가 목록
        """
        # 전력 배출계수에 있는 국가들을 기준으로 함
        return list(self.electricity_coefficients.keys())
    
    def recommend_by_carbon(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        전력 배출계수가 낮은 순으로 국가 추천
        
        Args:
            top_n: 상위 추천 국가 수
            
        Returns:
            List[Tuple[str, float]]: (국가명, 배출계수) 튜플의 리스트
        """
        # 배출계수 기준 오름차순 정렬
        sorted_countries = sorted(
            self.electricity_coefficients.items(),
            key=lambda x: x[1]
        )
        
        return sorted_countries[:top_n]
    
    def recommend_by_cost(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        생산 비용이 낮은 순으로 국가 추천
        
        Args:
            top_n: 상위 추천 국가 수
            
        Returns:
            List[Tuple[str, float]]: (국가명, 비용계수) 튜플의 리스트
        """
        # 비용 기준 오름차순 정렬
        sorted_countries = sorted(
            self.country_costs.items(),
            key=lambda x: x[1]
        )
        
        return sorted_countries[:top_n]
    
    def recommend_by_logistics(self, destination: str = "한국", top_n: int = 3) -> List[Tuple[str, float]]:
        """
        특정 목적지까지의 운송 비용이 낮은 순으로 국가 추천
        
        Args:
            destination: 목적지 국가
            top_n: 상위 추천 국가 수
            
        Returns:
            List[Tuple[str, float]]: (국가명, 운송비용) 튜플의 리스트
        """
        if destination not in self.transport_costs:
            # 목적지가 transport_costs에 없는 경우
            print(f"목적지 {destination}에 대한 운송 비용 정보가 없습니다.")
            return []
        
        # 해당 목적지로의 운송 비용 추출
        destination_costs = {}
        for country, costs in self.transport_costs.items():
            if destination in costs:
                destination_costs[country] = costs[destination]
        
        # 운송 비용 기준 오름차순 정렬
        sorted_countries = sorted(
            destination_costs.items(),
            key=lambda x: x[1]
        )
        
        return sorted_countries[:top_n]
    
    def calculate_comprehensive_scores(self, 
                                     carbon_weight: float = 0.6,
                                     cost_weight: float = 0.3, 
                                     logistics_weight: float = 0.1,
                                     destination: str = "한국") -> Dict[str, float]:
        """
        다양한 요소를 고려한 종합 점수 계산
        
        Args:
            carbon_weight: 탄소 배출계수 가중치 (0~1)
            cost_weight: 생산 비용 가중치 (0~1)
            logistics_weight: 물류 비용 가중치 (0~1)
            destination: 목적지 국가
            
        Returns:
            Dict[str, float]: 국가별 종합 점수
        """
        # 가중치 합이 1이 되도록 조정
        total_weight = carbon_weight + cost_weight + logistics_weight
        carbon_weight = carbon_weight / total_weight
        cost_weight = cost_weight / total_weight
        logistics_weight = logistics_weight / total_weight
        
        # 국가 목록 (모든 데이터에 공통으로 존재하는 국가들)
        common_countries = set(self.electricity_coefficients.keys())
        common_countries = common_countries.intersection(set(self.country_costs.keys()))
        if logistics_weight > 0:
            # 물류 비용을 고려하는 경우, 해당 목적지에 대한 운송 비용이 있는 국가로 제한
            logistics_countries = set()
            for country, costs in self.transport_costs.items():
                if destination in costs:
                    logistics_countries.add(country)
            common_countries = common_countries.intersection(logistics_countries)
        
        # 각 데이터 정규화 (Min-Max Scaling)
        carbon_values = {country: self.electricity_coefficients.get(country, 0) for country in common_countries}
        cost_values = {country: self.country_costs.get(country, 0) for country in common_countries}
        logistics_values = {}
        
        if logistics_weight > 0:
            for country in common_countries:
                if country in self.transport_costs and destination in self.transport_costs[country]:
                    logistics_values[country] = self.transport_costs[country][destination]
                else:
                    # 정보가 없는 경우 최대값으로 설정
                    logistics_values[country] = 1.0
        
        # Min-Max 정규화 함수
        def normalize(values):
            min_val = min(values.values()) if values else 0
            max_val = max(values.values()) if values else 1
            if max_val == min_val:
                return {k: 0 for k in values}
            return {k: (v - min_val) / (max_val - min_val) if max_val != min_val else 0 for k, v in values.items()}
        
        # 정규화 적용
        norm_carbon = normalize(carbon_values)
        norm_cost = normalize(cost_values)
        norm_logistics = normalize(logistics_values) if logistics_weight > 0 else {}
        
        # 종합 점수 계산 (낮을수록 좋음)
        self.country_scores = {}
        for country in common_countries:
            score = carbon_weight * norm_carbon.get(country, 0)
            score += cost_weight * norm_cost.get(country, 0)
            if logistics_weight > 0:
                score += logistics_weight * norm_logistics.get(country, 0)
            self.country_scores[country] = score
        
        return self.country_scores
    
    def recommend_comprehensive(self, 
                              carbon_weight: float = 0.6,
                              cost_weight: float = 0.3, 
                              logistics_weight: float = 0.1,
                              destination: str = "한국",
                              top_n: int = 3) -> List[Tuple[str, float]]:
        """
        종합 점수 기반 최적 국가 추천
        
        Args:
            carbon_weight: 탄소 배출계수 가중치 (0~1)
            cost_weight: 생산 비용 가중치 (0~1)
            logistics_weight: 물류 비용 가중치 (0~1)
            destination: 목적지 국가
            top_n: 상위 추천 국가 수
            
        Returns:
            List[Tuple[str, float]]: (국가명, 종합점수) 튜플의 리스트 (점수가 낮을수록 좋음)
        """
        # 종합 점수 계산
        scores = self.calculate_comprehensive_scores(
            carbon_weight, cost_weight, logistics_weight, destination
        )
        
        # 점수 기준 오름차순 정렬 (낮은 점수가 더 좋음)
        sorted_countries = sorted(
            scores.items(),
            key=lambda x: x[1]
        )
        
        return sorted_countries[:top_n]
    
    def get_recommendation_details(self, countries: List[str]) -> pd.DataFrame:
        """
        지정된 국가들에 대한 상세 정보 반환
        
        Args:
            countries: 상세 정보를 확인할 국가 목록
            
        Returns:
            pd.DataFrame: 국가별 상세 정보 데이터프레임
        """
        data = []
        
        for country in countries:
            row = {
                "국가": country,
                "전력배출계수": self.electricity_coefficients.get(country, None),
                "생산비용지수": self.country_costs.get(country, None),
                "종합점수": self.country_scores.get(country, None)
            }
            
            # 운송비용 추가 (한국 기준)
            if country in self.transport_costs and "한국" in self.transport_costs[country]:
                row["한국까지운송비"] = self.transport_costs[country]["한국"]
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_recommendation_report(self, 
                                     carbon_weight: float = 0.6,
                                     cost_weight: float = 0.3, 
                                     logistics_weight: float = 0.1,
                                     destination: str = "한국",
                                     top_n: int = 3) -> Dict[str, Any]:
        """
        종합적인 국가 추천 보고서 생성
        
        Args:
            carbon_weight: 탄소 배출계수 가중치 (0~1)
            cost_weight: 생산 비용 가중치 (0~1)
            logistics_weight: 물류 비용 가중치 (0~1)
            destination: 목적지 국가
            top_n: 상위 추천 국가 수
            
        Returns:
            Dict[str, Any]: 추천 보고서 (추천 국가, 데이터프레임, 각 국가별 상세 정보 등)
        """
        # 종합 추천
        recommended = self.recommend_comprehensive(
            carbon_weight, cost_weight, logistics_weight, destination, top_n
        )
        recommended_countries = [country for country, _ in recommended]
        
        # 탄소 기준 추천
        carbon_recommended = self.recommend_by_carbon(top_n)
        carbon_countries = [country for country, _ in carbon_recommended]
        
        # 비용 기준 추천
        cost_recommended = self.recommend_by_cost(top_n)
        cost_countries = [country for country, _ in cost_recommended]
        
        # 물류 기준 추천 (목적지 기준)
        logistics_recommended = self.recommend_by_logistics(destination, top_n)
        logistics_countries = [country for country, _ in logistics_recommended]
        
        # 상세 데이터 생성
        all_countries = list(set(recommended_countries + carbon_countries + cost_countries + logistics_countries))
        details_df = self.get_recommendation_details(all_countries)
        
        report = {
            "최종추천국가": recommended_countries,
            "탄소기준추천": carbon_countries,
            "비용기준추천": cost_countries,
            "물류기준추천": logistics_countries,
            "가중치": {
                "탄소": carbon_weight,
                "비용": cost_weight,
                "물류": logistics_weight
            },
            "목적지": destination,
            "상세데이터": details_df.to_dict('records')
        }
        
        return report
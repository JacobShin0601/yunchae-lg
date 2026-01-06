import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from typing import Dict, Tuple, Optional
from data_loader import get_prate_data

class PRateAnalyzer:
    """
    P-rate analysis를 수행하고 시각화하는 클래스
    """
    
    def __init__(self, ihs_df: Optional[pd.DataFrame] = None):
        """
        PRateAnalyzer 초기화
        
        Args:
            ihs_df (pd.DataFrame, optional): IHS 데이터프레임
        """
        self.ihs_df = ihs_df
        self.original_df = None
        self.ev_df = None
        self.non_ev_df = None
        self.yearly_sales = None
        
    def load_data(self) -> None:
        """
        get_prate_data 함수를 사용하여 데이터 로드
        
        Args:
            ihs_df (pd.DataFrame): IHS 데이터프레임
        """
        self.original_df, self.ev_df, self.non_ev_df = get_prate_data()
        
    def calculate_yearly_sales(self, target_column: str = 'Vehicle Volume') -> Dict[str, pd.DataFrame]:
        """
        연도별 EV와 Non-EV 판매량 계산
        
        Args:
            target_column (str): 분석할 타겟 컬럼명
            
        Returns:
            Dict[str, pd.DataFrame]: 연도별 판매량 데이터
        """
        if self.ev_df is None or self.non_ev_df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data() 메서드를 먼저 호출하세요.")
        
        # 연도별 EV와 Non-EV 판매량 집계
        ev_yearly = self.ev_df.groupby('year')[target_column].sum().reset_index()
        non_ev_yearly = self.non_ev_df.groupby('year')[target_column].sum().reset_index()
        
        # 결과 데이터프레임 생성
        yearly_sales = pd.merge(ev_yearly, non_ev_yearly, on='year', suffixes=('_ev', '_non_ev'))
        
        self.yearly_sales = {
            'yearly_sales': yearly_sales,
            'ev_sales': ev_yearly,
            'non_ev_sales': non_ev_yearly
        }
        
        return self.yearly_sales
    
    def plot_ev_non_ev_sales(self, 
                           target_column: str = 'Vehicle Volume',
                           figsize: Tuple[int, int] = (12, 6),
                           save_plot: bool = False,
                           save_path: Optional[str] = None,
                           return_json: bool = False) -> Dict[str, pd.DataFrame]:
        """
        EV와 Non-EV의 판매량을 시각화하는 함수
        
        Args:
            target_column (str): 분석할 타겟 컬럼명
            figsize (Tuple[int, int]): 그래프 크기
            save_plot (bool): 그래프 저장 여부
            save_path (str, optional): 그래프 저장 디렉토리 경로
            return_json (bool): JSON 형식으로 결과 반환 여부
            
        Returns:
            Dict[str, pd.DataFrame]: 연도별 판매량 데이터
        """
        # 연도별 판매량 계산
        sales_data = self.calculate_yearly_sales(target_column)
        
        ev_yearly = sales_data['ev_sales']
        non_ev_yearly = sales_data['non_ev_sales']
        
        # 시각화
        plt.figure(figsize=figsize)
        plt.plot(ev_yearly['year'], ev_yearly[target_column], 
                marker='o', label='EV', linewidth=2, markersize=6)
        plt.plot(non_ev_yearly['year'], non_ev_yearly[target_column], 
                marker='s', label='Non-EV', linewidth=2, markersize=6)
        
        plt.xticks(ev_yearly['year'], rotation=45)
        plt.title('연도별 EV와 Non-EV 판매량 추이', fontsize=16, fontweight='bold')
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('판매량', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot and save_path:
            # 파일명 생성
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"prate_sales_{timestamp}.png"
            json_filename = f"prate_sales_{timestamp}.json"
            
            # 전체 경로 생성
            plot_path = os.path.join(save_path, plot_filename)
            json_path = os.path.join(save_path, json_filename)
            
            # 그래프 저장
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {plot_path}에 저장되었습니다.")
            plt.close()
            
            # JSON 데이터 저장
            json_data = {
                'yearly_sales': sales_data['yearly_sales'].to_dict(orient='records'),
                'ev_sales': sales_data['ev_sales'].to_dict(orient='records'),
                'non_ev_sales': sales_data['non_ev_sales'].to_dict(orient='records')
            }
            
            # JSON 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {json_path}에 저장되었습니다.")
        else:
            plt.show()
        
        if return_json:
            # DataFrame을 JSON 형식으로 변환하여 반환
            json_data = {
                'yearly_sales': sales_data['yearly_sales'].to_dict(orient='records'),
                'ev_sales': sales_data['ev_sales'].to_dict(orient='records'),
                'non_ev_sales': sales_data['non_ev_sales'].to_dict(orient='records')
            }
            return json_data
        
        return sales_data
    def analyze_ev_ratio(self, 
                        target_column: str = 'Vehicle Volume',
                        figsize: Tuple[int, int] = (12, 6),
                        save_plot: bool = False,
                        save_path: Optional[str] = None,
                        return_json: bool = False) -> Dict[str, pd.DataFrame]:
        """
        EV 비율을 분석하는 함수
        
        Args:
            target_column (str): 분석할 타겟 컬럼명
            figsize (Tuple[int, int]): 그래프 크기
            save_plot (bool): 그래프 저장 여부
            save_path (str, optional): 그래프 저장 디렉토리 경로
            return_json (bool): JSON 형식으로 결과 반환 여부
            
        Returns:
            Dict[str, pd.DataFrame]: 분석 결과 데이터
        """
        # 연도별 판매량 계산
        sales_data = self.calculate_yearly_sales(target_column)
        
        ev_yearly = sales_data['ev_sales']
        non_ev_yearly = sales_data['non_ev_sales']
        
        # 전체 판매량 계산
        total_yearly = pd.merge(ev_yearly, non_ev_yearly, on='year', suffixes=('_ev', '_non_ev'))
        total_yearly['total'] = total_yearly[f'{target_column}_ev'] + total_yearly[f'{target_column}_non_ev']
        
        # 비율 계산
        total_yearly['ev_ratio'] = (total_yearly[f'{target_column}_ev'] / total_yearly['total'] * 100).round(2)
        total_yearly['non_ev_ratio'] = (total_yearly[f'{target_column}_non_ev'] / total_yearly['total'] * 100).round(2)
        
        # 시각화
        plt.figure(figsize=figsize)
        plt.stackplot(total_yearly['year'], 
                     [total_yearly['ev_ratio'], total_yearly['non_ev_ratio']],
                     labels=['EV 비율', 'Non-EV 비율'],
                     colors=['#2ecc71', '#e74c3c'])
        
        plt.xticks(total_yearly['year'], rotation=45)
        plt.title('연도별 EV와 Non-EV 판매 비율 추이', fontsize=16, fontweight='bold')
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('비율 (%)', fontsize=12)
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        if save_plot and save_path:
            # 파일명 생성
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"prate_ratio_{timestamp}.png"
            json_filename = f"prate_ratio_{timestamp}.json"
            
            # 전체 경로 생성
            plot_path = os.path.join(save_path, plot_filename)
            json_path = os.path.join(save_path, json_filename)
            
            # 그래프 저장
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {plot_path}에 저장되었습니다.")
            plt.close()
            
            # JSON 데이터 저장
            json_data = {
                'yearly_data': total_yearly.to_dict(orient='records'),
                'ev_ratio': total_yearly['ev_ratio'].to_dict(),
                'non_ev_ratio': total_yearly['non_ev_ratio'].to_dict(),
                'summary': total_yearly[['year', 'ev_ratio', 'non_ev_ratio']].sort_values('year').to_dict('records')
            }
            
            # JSON 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {json_path}에 저장되었습니다.")
        else:
            plt.show()
        
        if return_json:
            # DataFrame을 JSON 형식으로 변환하여 반환
            json_data = {
                'yearly_data': total_yearly.to_dict(orient='records'),
                'ev_ratio': total_yearly['ev_ratio'].to_dict(),
                'non_ev_ratio': total_yearly['non_ev_ratio'].to_dict(),
                'summary': total_yearly[['year', 'ev_ratio', 'non_ev_ratio']].sort_values('year').to_dict('records')
            }
            return json_data
        
        result = {
            'yearly_data': total_yearly,
            'ev_ratio': total_yearly['ev_ratio'].to_dict(),
            'non_ev_ratio': total_yearly['non_ev_ratio'].to_dict(),
            'summary': total_yearly[['year', 'ev_ratio', 'non_ev_ratio']].sort_values('year').to_dict('records')
        }
        
        return result

    def plot_ratio_by_type(self, 
                          target_column: str = 'Vehicle Volume',
                          figsize: Tuple[int, int] = (12, 6),
                          save_plot: bool = False,
                          save_path: Optional[str] = None,
                          return_json: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Type(최종) 컬럼에 따른 100% stacked area chart를 그리는 함수
        
        Args:
            target_column (str): 집계할 타겟 컬럼명
            figsize (Tuple[int, int]): 그래프 크기
            save_plot (bool): 그래프 저장 여부
            save_path (str, optional): 그래프 저장 디렉토리 경로
            return_json (bool): JSON 형식으로 결과 반환 여부
            
        Returns:
            Dict[str, pd.DataFrame]: 분석 결과 딕셔너리
        """
        # 데이터가 로드되었는지 확인
        if self.original_df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data() 메서드를 먼저 호출하세요.")
        
        # Type(최종) 컬럼이 있는지 확인
        if 'Type(최종)' not in self.original_df.columns:
            raise KeyError("'Type(최종)' 컬럼이 데이터프레임에 존재하지 않습니다.")
        
        # 연도별, Type(최종)별 집계
        type_yearly = self.original_df.groupby(['year', 'Type(최종)'])[target_column].sum().reset_index()
        
        # 피벗 테이블 생성
        type_pivot = type_yearly.pivot(index='year', columns='Type(최종)', values=target_column).fillna(0)
        
        # 각 행의 합계 계산
        row_sums = type_pivot.sum(axis=1)
        
        # 비율 계산
        type_ratio = type_pivot.div(row_sums, axis=0) * 100
        
        # 시각화
        plt.figure(figsize=figsize)
        plt.stackplot(type_ratio.index, 
                     [type_ratio[col] for col in type_ratio.columns],
                     labels=type_ratio.columns,
                     colors=['#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c','#e74c3c'])
        
        plt.xticks(type_ratio.index, rotation=45)
        plt.title('연도별 차량 타입별 판매 비율 추이', fontsize=16, fontweight='bold')
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('비율 (%)', fontsize=12)
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        if save_plot and save_path:
            # 파일명 생성
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"prate_type_ratio_{timestamp}.png"
            json_filename = f"prate_type_ratio_{timestamp}.json"
            
            # 전체 경로 생성
            plot_path = os.path.join(save_path, plot_filename)
            json_path = os.path.join(save_path, json_filename)
            
            # 그래프 저장
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {plot_path}에 저장되었습니다.")
            plt.close()
            
            # JSON 데이터 저장
            json_data = {
                'yearly_data': type_yearly.to_dict(orient='records'),
                'type_ratio': type_ratio.to_dict(),
                'summary': type_ratio.reset_index().to_dict('records')
            }
            
            # JSON 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {json_path}에 저장되었습니다.")
        else:
            plt.show()
        
        if return_json:
            # DataFrame을 JSON 형식으로 변환하여 반환
            json_data = {
                'yearly_data': type_yearly.to_dict(orient='records'),
                'type_ratio': type_ratio.to_dict(),
                'summary': type_ratio.reset_index().to_dict('records')
            }
            return json_data
        
        result = {
            'yearly_data': type_yearly,
            'type_ratio': type_ratio,
            'summary': type_ratio.reset_index()
        }
        
        return result
    
    def plot_by_region(self, 
                      target_column: str = 'Vehicle Volume',
                      figsize: Tuple[int, int] = (12, 6),
                      save_plot: bool = False,
                      save_path: Optional[str] = None,
                      return_json: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Region(최종) 컬럼에 따른 100% stacked area chart를 그리는 함수
        
        Args:
            target_column (str): 집계할 타겟 컬럼명
            figsize (Tuple[int, int]): 그래프 크기
            save_plot (bool): 그래프 저장 여부
            save_path (str, optional): 그래프 저장 디렉토리 경로
            return_json (bool): JSON 형식으로 결과 반환 여부
            
        Returns:
            Dict[str, pd.DataFrame]: 분석 결과 딕셔너리
        """
        # 데이터가 로드되었는지 확인
        if self.original_df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data() 메서드를 먼저 호출하세요.")
        
        # Region(최종) 컬럼이 있는지 확인
        if 'Region(최종)' not in self.original_df.columns:
            raise KeyError("'Region(최종)' 컬럼이 데이터프레임에 존재하지 않습니다.")
        
        # 연도별, Region(최종)별 집계
        region_yearly = self.original_df.groupby(['year', 'Region(최종)'])[target_column].sum().reset_index()
        
        # 피벗 테이블 생성
        region_pivot = region_yearly.pivot(index='year', columns='Region(최종)', values=target_column).fillna(0)
        
        # 각 행의 합계 계산
        row_sums = region_pivot.sum(axis=1)
        
        # 비율 계산
        region_ratio = region_pivot.div(row_sums, axis=0) * 100
        
        # 시각화
        plt.figure(figsize=figsize)
        
        # 색상 팔레트 설정 (지역별로 구분하기 쉽게)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#f39c12']
        
        plt.stackplot(region_ratio.index, 
                     [region_ratio[col] for col in region_ratio.columns],
                     labels=region_ratio.columns,
                     colors=colors[:len(region_ratio.columns)])
        
        plt.xticks(region_ratio.index, rotation=45)
        plt.title('연도별 지역별 판매 비율 추이', fontsize=16, fontweight='bold')
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('비율 (%)', fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        if save_plot and save_path:
            # 파일명 생성
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"prate_region_ratio_{timestamp}.png"
            json_filename = f"prate_region_ratio_{timestamp}.json"
            
            # 전체 경로 생성
            plot_path = os.path.join(save_path, plot_filename)
            json_path = os.path.join(save_path, json_filename)
            
            # 그래프 저장
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {plot_path}에 저장되었습니다.")
            plt.close()
            
            # JSON 데이터 저장
            json_data = {
                'yearly_data': region_yearly.to_dict(orient='records'),
                'region_ratio': region_ratio.to_dict(),
                'summary': region_ratio.reset_index().to_dict('records')
            }
            
            # JSON 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {json_path}에 저장되었습니다.")
        else:
            plt.show()
        
        if return_json:
            # DataFrame을 JSON 형식으로 변환하여 반환
            json_data = {
                'yearly_data': region_yearly.to_dict(orient='records'),
                'region_ratio': region_ratio.to_dict(),
                'summary': region_ratio.reset_index().to_dict('records')
            }
            return json_data
        
        result = {
            'yearly_data': region_yearly,
            'region_ratio': region_ratio,
            'summary': region_ratio.reset_index()
        }
        
        return result
    
    def plot_by_cathode(self, 
                       target_column: str = 'Vehicle Volume',
                       figsize: Tuple[int, int] = (12, 6),
                       save_plot: bool = False,
                       save_path: Optional[str] = None,
                       return_json: bool = False) -> Dict[str, pd.DataFrame]:
        """
        양극재 컬럼에 따른 100% stacked area chart를 그리는 함수
        
        Args:
            target_column (str): 집계할 타겟 컬럼명
            figsize (Tuple[int, int]): 그래프 크기
            save_plot (bool): 그래프 저장 여부
            save_path (str, optional): 그래프 저장 디렉토리 경로
            return_json (bool): JSON 형식으로 결과 반환 여부
            
        Returns:
            Dict[str, pd.DataFrame]: 분석 결과 딕셔너리
        """
        # 데이터가 로드되었는지 확인
        if self.original_df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data() 메서드를 먼저 호출하세요.")
        
        # 양극재 컬럼이 있는지 확인
        if '양극재' not in self.original_df.columns:
            raise KeyError("'양극재' 컬럼이 데이터프레임에 존재하지 않습니다.")
        
        # 연도별, 양극재별 집계
        cathode_yearly = self.original_df.groupby(['year', '양극재'])[target_column].sum().reset_index()
        
        # 피벗 테이블 생성
        cathode_pivot = cathode_yearly.pivot(index='year', columns='양극재', values=target_column).fillna(0)
        
        # 각 행의 합계 계산
        row_sums = cathode_pivot.sum(axis=1)
        
        # 비율 계산
        cathode_ratio = cathode_pivot.div(row_sums, axis=0) * 100
        
        # 시각화
        plt.figure(figsize=figsize)
        
        # 양극재별 색상 팔레트 설정
        colors = ['#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e74c3c', '#e67e22', '#34495e', '#95a5a6', '#f39c12']
        
        plt.stackplot(cathode_ratio.index, 
                     [cathode_ratio[col] for col in cathode_ratio.columns],
                     labels=cathode_ratio.columns,
                     colors=colors[:len(cathode_ratio.columns)])
        
        plt.xticks(cathode_ratio.index, rotation=45)
        plt.title('연도별 양극재별 판매 비율 추이', fontsize=16, fontweight='bold')
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('비율 (%)', fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        if save_plot and save_path:
            # 파일명 생성
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"prate_cathode_ratio_{timestamp}.png"
            json_filename = f"prate_cathode_ratio_{timestamp}.json"
            
            # 전체 경로 생성
            plot_path = os.path.join(save_path, plot_filename)
            json_path = os.path.join(save_path, json_filename)
            
            # 그래프 저장
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {plot_path}에 저장되었습니다.")
            plt.close()
            
            # JSON 데이터 저장
            json_data = {
                'yearly_data': cathode_yearly.to_dict(orient='records'),
                'cathode_ratio': cathode_ratio.to_dict(),
                'summary': cathode_ratio.reset_index().to_dict('records')
            }
            
            # JSON 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {json_path}에 저장되었습니다.")
        else:
            plt.show()
        
        if return_json:
            # DataFrame을 JSON 형식으로 변환하여 반환
            json_data = {
                'yearly_data': cathode_yearly.to_dict(orient='records'),
                'cathode_ratio': cathode_ratio.to_dict(),
                'summary': cathode_ratio.reset_index().to_dict('records')
            }
            return json_data
        
        result = {
            'yearly_data': cathode_yearly,
            'cathode_ratio': cathode_ratio,
            'summary': cathode_ratio.reset_index()
        }
        
        return result
    
    def plot_by_position(self, 
                        target_column: str = 'Vehicle Volume',
                        figsize: Tuple[int, int] = (12, 6),
                        save_plot: bool = False,
                        save_path: Optional[str] = None,
                        return_json: bool = False) -> Dict[str, pd.DataFrame]:
        """
        POS(MI) 컬럼에 따른 100% stacked area chart를 그리는 함수
        
        Args:
            target_column (str): 집계할 타겟 컬럼명
            figsize (Tuple[int, int]): 그래프 크기
            save_plot (bool): 그래프 저장 여부
            save_path (str, optional): 그래프 저장 디렉토리 경로
            return_json (bool): JSON 형식으로 결과 반환 여부
            
        Returns:
            Dict[str, pd.DataFrame]: 분석 결과 딕셔너리
        """
        # 데이터가 로드되었는지 확인
        if self.original_df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data() 메서드를 먼저 호출하세요.")
        
        # POS(MI) 컬럼이 있는지 확인
        if 'POS(MI)' not in self.original_df.columns:
            raise KeyError("'POS(MI)' 컬럼이 데이터프레임에 존재하지 않습니다.")
        
        # 연도별, POS(MI)별 집계
        pos_yearly = self.original_df.groupby(['year', 'POS(MI)'])[target_column].sum().reset_index()
        
        # 피벗 테이블 생성
        pos_pivot = pos_yearly.pivot(index='year', columns='POS(MI)', values=target_column).fillna(0)
        
        # 각 행의 합계 계산
        row_sums = pos_pivot.sum(axis=1)
        
        # 비율 계산
        pos_ratio = pos_pivot.div(row_sums, axis=0) * 100
        
        # 시각화
        plt.figure(figsize=figsize)
        
        # POS(MI)별 색상 팔레트 설정
        colors = ['#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e74c3c', '#e67e22', '#34495e', '#95a5a6', '#f39c12']
        
        plt.stackplot(pos_ratio.index, 
                     [pos_ratio[col] for col in pos_ratio.columns],
                     labels=pos_ratio.columns,
                     colors=colors[:len(pos_ratio.columns)])
        
        plt.xticks(pos_ratio.index, rotation=45)
        plt.title('연도별 POS(MI)별 판매 비율 추이', fontsize=16, fontweight='bold')
        plt.xlabel('연도', fontsize=12)
        plt.ylabel('비율 (%)', fontsize=12)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        if save_plot and save_path:
            # 파일명 생성
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"prate_position_ratio_{timestamp}.png"
            json_filename = f"prate_position_ratio_{timestamp}.json"
            
            # 전체 경로 생성
            plot_path = os.path.join(save_path, plot_filename)
            json_path = os.path.join(save_path, json_filename)
            
            # 그래프 저장
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {plot_path}에 저장되었습니다.")
            plt.close()
            
            # JSON 데이터 저장
            json_data = {
                'yearly_data': pos_yearly.to_dict(orient='records'),
                'position_ratio': pos_ratio.to_dict(),
                'summary': pos_ratio.reset_index().to_dict('records')
            }
            
            # JSON 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {json_path}에 저장되었습니다.")
        else:
            plt.show()
        
        if return_json:
            # DataFrame을 JSON 형식으로 변환하여 반환
            json_data = {
                'yearly_data': pos_yearly.to_dict(orient='records'),
                'position_ratio': pos_ratio.to_dict(),
                'summary': pos_ratio.reset_index().to_dict('records')
            }
            return json_data
        
        result = {
            'yearly_data': pos_yearly,
            'position_ratio': pos_ratio,
            'summary': pos_ratio.reset_index()
        }
        
        return result
    
    def get_analysis_results(self, target_column: str = 'Vehicle Volume') -> Dict:
        """
        분석 결과를 반환하는 함수
        
        Args:
            target_column (str): 분석할 타겟 컬럼명
            
        Returns:
            Dict: 분석 결과 딕셔너리
        """
        if self.yearly_sales is None:
            self.calculate_yearly_sales(target_column)
        
        ev_sales = self.yearly_sales['ev_sales']
        non_ev_sales = self.yearly_sales['non_ev_sales']
        
        # 기본 통계 계산
        ev_total = ev_sales[target_column].sum()
        non_ev_total = non_ev_sales[target_column].sum()
        total_sales = ev_total + non_ev_total
        
        # 성장률 계산 (첫 해 대비 마지막 해)
        ev_growth_rate = ((ev_sales[target_column].iloc[-1] - ev_sales[target_column].iloc[0]) / 
                         ev_sales[target_column].iloc[0] * 100)
        non_ev_growth_rate = ((non_ev_sales[target_column].iloc[-1] - non_ev_sales[target_column].iloc[0]) / 
                             non_ev_sales[target_column].iloc[0] * 100)
        
        # 시장 점유율 계산 (마지막 해 기준)
        last_year_ev = ev_sales[target_column].iloc[-1]
        last_year_non_ev = non_ev_sales[target_column].iloc[-1]
        last_year_total = last_year_ev + last_year_non_ev
        
        ev_market_share = (last_year_ev / last_year_total) * 100
        non_ev_market_share = (last_year_non_ev / last_year_total) * 100
        
        return {
            'summary': {
                'total_ev_sales': ev_total,
                'total_non_ev_sales': non_ev_total,
                'total_sales': total_sales,
                'ev_market_share_last_year': ev_market_share,
                'non_ev_market_share_last_year': non_ev_market_share
            },
            'growth_rates': {
                'ev_growth_rate': ev_growth_rate,
                'non_ev_growth_rate': non_ev_growth_rate
            },
            'yearly_data': self.yearly_sales,
            'data_info': {
                'years_analyzed': len(ev_sales),
                'start_year': ev_sales['year'].min(),
                'end_year': ev_sales['year'].max()
            }
        }
    
    def print_summary(self, target_column: str = 'Vehicle Volume') -> None:
        """
        분석 결과 요약을 출력하는 함수
        
        Args:
            target_column (str): 분석할 타겟 컬럼명
        """
        results = self.get_analysis_results(target_column)
        
        print("=" * 50)
        print("P-Rate Analysis 결과 요약")
        print("=" * 50)
        print(f"분석 기간: {results['data_info']['start_year']} - {results['data_info']['end_year']}")
        print(f"분석 연도 수: {results['data_info']['years_analyzed']}년")
        print()
        
        print("전체 판매량:")
        print(f"  - EV 총 판매량: {results['summary']['total_ev_sales']:,}")
        print(f"  - Non-EV 총 판매량: {results['summary']['total_non_ev_sales']:,}")
        print(f"  - 전체 판매량: {results['summary']['total_sales']:,}")
        print()
        
        print(f"{results['data_info']['end_year']}년 시장 점유율:")
        print(f"  - EV: {results['summary']['ev_market_share_last_year']:.1f}%")
        print(f"  - Non-EV: {results['summary']['non_ev_market_share_last_year']:.1f}%")
        print()
        
        print("성장률 (첫 해 대비 마지막 해):")
        print(f"  - EV: {results['growth_rates']['ev_growth_rate']:+.1f}%")
        print(f"  - Non-EV: {results['growth_rates']['non_ev_growth_rate']:+.1f}%")
        print("=" * 50)

    def plot_static_ratio_by_pos_and_region(self, 
                                           target_year: int = 2025,
                                           target_column: str = 'Vehicle Volume',
                                           figsize: Tuple[int, int] = (15, 6),
                                           save_plot: bool = False,
                                           save_path: Optional[str] = None,
                                           return_json: bool = False) -> Dict:
        """
        특정 연도 기준으로 POS(MI)와 Region(최종)별 점유율을 분석하고 시각화하는 함수
        
        Args:
            target_year (int): 분석할 기준 연도
            target_column (str): 집계할 타겟 컬럼명
            figsize (Tuple[int, int]): 그래프 크기
            save_plot (bool): 그래프 저장 여부
            save_path (str, optional): 그래프 저장 디렉토리 경로
            return_json (bool): JSON 형식으로 결과 반환 여부
            
        Returns:
            Dict: 분석 결과 딕셔너리
        """
        # 데이터가 로드되었는지 확인
        if self.original_df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data() 메서드를 먼저 호출하세요.")
        
        # 필요한 컬럼들이 있는지 확인
        required_columns = ['POS(MI)', 'Region(최종)', 'year']
        for col in required_columns:
            if col not in self.original_df.columns:
                raise KeyError(f"'{col}' 컬럼이 데이터프레임에 존재하지 않습니다.")
        
        # 특정 연도 데이터 필터링
        year_data = self.original_df[self.original_df['year'] == target_year].copy()
        
        if year_data.empty:
            raise ValueError(f"{target_year}년 데이터가 존재하지 않습니다.")
        
        # POS(MI)별 집계
        pos_data = year_data.groupby('POS(MI)')[target_column].sum().reset_index()
        pos_total = pos_data[target_column].sum()
        pos_data['share_pct'] = (pos_data[target_column] / pos_total * 100).round(2)
        pos_data = pos_data.sort_values('share_pct', ascending=False)
        
        # Region(최종)별 집계
        region_data = year_data.groupby('Region(최종)')[target_column].sum().reset_index()
        region_total = region_data[target_column].sum()
        region_data['share_pct'] = (region_data[target_column] / region_total * 100).round(2)
        region_data = region_data.sort_values('share_pct', ascending=False)
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # POS(MI) 파이 차트
        colors_pos = ['#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c', '#e74c3c', '#e67e22', '#34495e', '#95a5a6', '#f39c12']
        wedges1, texts1, autotexts1 = ax1.pie(pos_data['share_pct'], 
                                             labels=pos_data['POS(MI)'], 
                                             autopct='%1.1f%%',
                                             colors=colors_pos[:len(pos_data)],
                                             startangle=90)
        ax1.set_title(f'{target_year}년 POS(MI)별 점유율', fontsize=14, fontweight='bold')
        
        # Region(최종) 파이 차트
        colors_region = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#f39c12']
        wedges2, texts2, autotexts2 = ax2.pie(region_data['share_pct'], 
                                             labels=region_data['Region(최종)'], 
                                             autopct='%1.1f%%',
                                             colors=colors_region[:len(region_data)],
                                             startangle=90)
        ax2.set_title(f'{target_year}년 Region(최종)별 점유율', fontsize=14, fontweight='bold')
        
        # 텍스트 크기 조정
        for autotext in autotexts1 + autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        plt.tight_layout()
        
        # 결과 딕셔너리 생성
        result = {
            'target_year': target_year,
            'pos_share': {
                'data': pos_data.to_dict('records'),
                'total_volume': int(pos_total),
                'top_3': pos_data.head(3)[['POS(MI)', 'share_pct']].to_dict('records')
            },
            'region_share': {
                'data': region_data.to_dict('records'),
                'total_volume': int(region_total),
                'top_3': region_data.head(3)[['Region(최종)', 'share_pct']].to_dict('records')
            },
            'summary': {
                'pos_count': len(pos_data),
                'region_count': len(region_data),
                'total_records': len(year_data)
            }
        }
        
        if save_plot and save_path:
            # 파일명 생성
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = f"prate_static_pos_region_{target_year}_{timestamp}.png"
            json_filename = f"prate_static_pos_region_{target_year}_{timestamp}.json"
            
            # 전체 경로 생성
            plot_path = os.path.join(save_path, plot_filename)
            json_path = os.path.join(save_path, json_filename)
            
            # 그래프 저장
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"그래프가 {plot_path}에 저장되었습니다.")
            plt.close()
            
            # JSON 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"JSON 데이터가 {json_path}에 저장되었습니다.")
        else:
            plt.show()
        
        if return_json:
            return result
        
        return result


# 사용 예시
if __name__ == "__main__":
    # 예시 사용법
    # analyzer = PRateAnalyzer()
    # analyzer.load_data(ihs_df)  # ihs_df는 실제 데이터프레임
    # analyzer.plot_ev_non_ev_sales()
    # analyzer.print_summary()
    pass

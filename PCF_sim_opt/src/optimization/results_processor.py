"""
최적화 결과를 처리하고 분석하는 모듈
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import json
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from .input import OptimizationInput


class ResultsProcessor:
    """
    최적화 결과를 처리하고 분석하는 클래스
    
    주요 기능:
    - 결과 포맷팅 및 요약
    - 파일 내보내기 (JSON, YAML, CSV)
    - 결과 시각화
    - 다중 시나리오/솔버 결과 비교
    """
    
    def __init__(self, opt_input: Optional[OptimizationInput] = None):
        """
        Args:
            opt_input: 최적화 입력 객체 (선택 사항)
        """
        self.opt_input = opt_input
        self.results = None
        self.variables = None
        self.formatted_results = None
    
    def set_opt_input(self, opt_input: OptimizationInput) -> None:
        """
        최적화 입력 객체 설정
        
        Args:
            opt_input: 최적화 입력 객체
        """
        self.opt_input = opt_input
    
    def process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        최적화 결과 처리
        
        Args:
            results: 최적화 결과 딕셔너리
            
        Returns:
            Dict: 포맷팅된 결과
        """
        self.results = results
        
        if results.get('status') != 'optimal':
            self.formatted_results = {
                'status': results.get('status', 'not_solved'), 
                'message': results.get('message', '최적화가 실행되지 않았거나 실패했습니다.')
            }
            return self.formatted_results
        
        # 변수값 추출
        self.variables = results.get('variables', {})
        
        # 결과 포맷팅
        formatted_results = self.format_results()
        
        # 저장
        self.formatted_results = formatted_results
        
        return formatted_results
    
    def format_results(self) -> Dict[str, Any]:
        """
        결과 포맷팅
        
        Returns:
            Dict: 포맷팅된 결과
        """
        if not self.results or self.results.get('status') != 'optimal':
            return {
                'status': self.results.get('status', 'not_solved'), 
                'message': self.results.get('message', '최적화가 실행되지 않았거나 실패했습니다.')
            }
        
        variables = self.variables
        
        # 감축비율 결과
        reduction_results = {}
        for var_name, var_value in variables.items():
            if 'tier' in var_name and not var_name.endswith('_active'):
                parts = var_name.split('_')
                tier = parts[0].upper()
                item = '_'.join(parts[1:])
                
                if tier not in reduction_results:
                    reduction_results[tier] = {}
                    
                reduction_results[tier][item] = f"{var_value:.2f}%"
        
        # 활성화된 활동 (이진 변수)
        active_activities = {}
        for var_name, var_value in variables.items():
            if var_name.endswith('_active') and var_value > 0.5:
                activity_name = var_name.replace('_active', '')
                active_activities[activity_name] = True
        
        # 양극재 구성 결과
        cathode_results = {}
        
        if 'recycle_ratio' in variables:
            cathode_results['재활용재_비율'] = f"{variables['recycle_ratio'] * 100:.2f}%"
            
        if 'low_carbon_ratio' in variables:
            cathode_results['저탄소원료_비율'] = f"{variables['low_carbon_ratio'] * 100:.2f}%"
            cathode_results['신재_비율'] = f"{(1 - variables.get('recycle_ratio', 0) - variables.get('low_carbon_ratio', 0)) * 100:.2f}%"
            
        if 'low_carbon_emission' in variables:
            cathode_results['저탄소원료_배출계수'] = f"{variables['low_carbon_emission']:.2f}"
        
        # 목적 관련 정보
        objective_info = {}
        objective_type = self.opt_input.get_objective() if self.opt_input else self.results.get('objective_type', 'unknown')
        objective_value = self.results.get('objective_value')
        
        objective_info['유형'] = self._get_objective_description(objective_type)
        objective_info['값'] = f"{objective_value:.4f}" if objective_value is not None else 'N/A'
        
        if objective_type == 'minimize_carbon':
            objective_info['탄소발자국'] = f"{self.results.get('carbon_footprint', objective_value):.4f} kg CO2eq/kWh"
        elif objective_type == 'minimize_cost':
            objective_info['총비용'] = f"{objective_value:.2f} 원"
        elif objective_type == 'maximize_ease':
            objective_info['활성화_활동_수'] = len(active_activities)
        
        # 최종 결과 구성
        formatted = {
            'status': 'optimal',
            'objective': objective_info,
            'carbon_footprint': f"{self.results.get('carbon_footprint', 0):.4f} kg CO2eq/kWh",
            'solver': self.results.get('solver'),
            'solver_time': f"{self.results.get('solver_time', 0):.2f} 초" if self.results.get('solver_time') else 'N/A',
            'reduction_ratios': reduction_results,
            'cathode_composition': cathode_results
        }
        
        if active_activities:
            formatted['active_activities'] = active_activities
        
        return formatted
    
    def _get_objective_description(self, objective_type: str) -> str:
        """
        목적함수 유형에 대한 설명
        
        Args:
            objective_type: 목적함수 유형
            
        Returns:
            str: 설명
        """
        descriptions = {
            'minimize_carbon': '탄소발자국 최소화',
            'minimize_cost': '총 비용 최소화',
            'maximize_ease': '구현 용이성 최대화',
            'multi_objective': '다목적 최적화'
        }
        
        return descriptions.get(objective_type, f'목적함수: {objective_type}')
    
    def export_to_json(self, file_path: Optional[str] = None) -> str:
        """
        결과를 JSON 파일로 내보내기
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.results:
            raise ValueError("내보낼 결과가 없습니다. process_results()를 먼저 호출하세요.")
        
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"optimization_results_{timestamp}.json"
        
        # 결과 데이터 구성
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.opt_input.get_config() if self.opt_input else {},
            'results': self.results,
            'formatted_results': self.formatted_results
        }
        
        # 파일 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return str(path)
    
    def export_to_yaml(self, file_path: Optional[str] = None) -> str:
        """
        결과를 YAML 파일로 내보내기
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.results:
            raise ValueError("내보낼 결과가 없습니다. process_results()를 먼저 호출하세요.")
        
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"optimization_results_{timestamp}.yaml"
        
        # 결과 데이터 구성
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.opt_input.get_config() if self.opt_input else {},
            'results': self.results,
            'formatted_results': self.formatted_results
        }
        
        # 파일 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
        
        return str(path)
    
    def export_to_csv(self, file_path: Optional[str] = None) -> str:
        """
        결과를 CSV 파일로 내보내기
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.variables:
            raise ValueError("내보낼 변수값이 없습니다. process_results()를 먼저 호출하세요.")
        
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"optimization_variables_{timestamp}.csv"
        
        # 결과 데이터 구성
        data = []
        
        for var_name, var_value in self.variables.items():
            if isinstance(var_value, dict):
                for idx, val in var_value.items():
                    data.append({
                        'variable': f"{var_name}[{idx}]",
                        'value': val
                    })
            else:
                data.append({
                    'variable': var_name,
                    'value': var_value
                })
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(data)
        
        # 파일 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(path, index=False)
        
        return str(path)
    
    def visualize_results(self, file_path: Optional[str] = None) -> str:
        """
        결과를 시각화하여 저장
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.formatted_results:
            raise ValueError("시각화할 결과가 없습니다. process_results()를 먼저 호출하세요.")
        
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"optimization_visualization_{timestamp}.png"
        
        # 감축 비율 시각화
        reduction_ratios = self.formatted_results.get('reduction_ratios', {})
        
        plt.figure(figsize=(12, 8))
        
        # 감축비율 그래프
        if reduction_ratios:
            plt.subplot(2, 2, 1)
            
            # 데이터 준비
            tiers = []
            items = []
            values = []
            
            for tier, tier_items in reduction_ratios.items():
                for item, value_str in tier_items.items():
                    tiers.append(tier)
                    items.append(item)
                    values.append(float(value_str.replace('%', '')))
            
            # 그래프 생성
            bars = plt.bar(items, values)
            
            # 막대 위에 값 표시
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', rotation=0)
            
            plt.title('Tier별 감축비율')
            plt.xticks(rotation=45)
            plt.ylabel('감축비율 (%)')
            plt.ylim(0, max(values) * 1.2)
        
        # 양극재 구성 그래프
        cathode_composition = self.formatted_results.get('cathode_composition', {})
        
        if cathode_composition:
            plt.subplot(2, 2, 2)
            
            # 비율 데이터만 추출
            ratio_data = {}
            for key, value_str in cathode_composition.items():
                if '비율' in key and '%' in value_str:
                    ratio_data[key] = float(value_str.replace('%', ''))
            
            if ratio_data:
                labels = list(ratio_data.keys())
                sizes = list(ratio_data.values())
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                plt.title('양극재 구성 비율')
        
        # 탄소발자국 목표 vs 결과
        plt.subplot(2, 2, 3)
        
        carbon_footprint = self.formatted_results.get('carbon_footprint', '0')
        # 문자열이 아닌 값은 그대로 사용
        if not isinstance(carbon_footprint, str):
            carbon_result = float(carbon_footprint)
        else:
            # 문자열인 경우 분리 후 처리
            carbon_result = float(carbon_footprint.split()[0])
        carbon_target = self.opt_input.get_constraint('target_carbon') if self.opt_input else 50.0
        
        plt.bar(['목표', '결과'], [carbon_target, carbon_result])
        plt.axhline(y=carbon_target, color='r', linestyle='-', alpha=0.3)
        plt.ylabel('탄소발자국 (kg CO2eq/kWh)')
        plt.title('탄소발자국 목표 vs 결과')
        
        # 목적함수 정보
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        info_text = f"목적함수: {self._get_objective_description(self.opt_input.get_objective() if self.opt_input else 'unknown')}\n"
        info_text += f"솔버: {self.results.get('solver', 'N/A')}\n"
        info_text += f"계산 시간: {self.results.get('solver_time', 'N/A')} 초\n"
        info_text += f"목적함수 값: {self.results.get('objective_value', 'N/A')}\n"
        info_text += f"탄소발자국: {carbon_result} kg CO2eq/kWh\n"
        
        plt.text(0.1, 0.5, info_text, fontsize=10)
        plt.title('최적화 정보')
        
        # 전체 제목
        plt.suptitle('PCF 최적화 결과 요약', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 파일 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def compare_results(self, results_list: List[Dict[str, Any]], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        여러 최적화 결과 비교
        
        Args:
            results_list: 최적화 결과 목록
            labels: 각 결과에 대한 레이블 (선택 사항)
            
        Returns:
            Dict: 비교 결과
        """
        if not results_list:
            return {}
        
        # 레이블이 없으면 자동 생성
        if labels is None:
            labels = [f"결과 {i+1}" for i in range(len(results_list))]
        elif len(labels) < len(results_list):
            labels.extend([f"결과 {i+1}" for i in range(len(labels), len(results_list))])
        
        # 결과를 처리하여 데이터 추출
        processed_results = []
        for result in results_list:
            if isinstance(result, dict) and 'status' in result:
                if result.get('status') == 'optimal':
                    # 이미 결과 딕셔너리인 경우
                    processed_results.append(result)
                else:
                    processed_results.append(None)
            else:
                # ResultsProcessor 객체로 결과 처리
                processor = ResultsProcessor()
                processed = processor.process_results(result)
                if processed.get('status') == 'optimal':
                    processed_results.append(processed)
                else:
                    processed_results.append(None)
        
        # 비교 데이터 구성
        comparison = {
            'summary': {},
            'carbon_footprint': {},
            'objective_values': {},
            'solver_times': {},
            'cathode_compositions': {}
        }
        
        # 요약 정보
        for i, (result, label) in enumerate(zip(processed_results, labels)):
            if result:
                comparison['summary'][label] = {
                    'status': result.get('status'),
                    'solver': result.get('solver'),
                    'carbon_footprint': result.get('carbon_footprint'),
                    'solver_time': result.get('solver_time')
                }
                
                # 탄소발자국
                carbon_footprint = result.get('carbon_footprint', '0')
                # 문자열이 아닌 값은 그대로 사용
                if not isinstance(carbon_footprint, str):
                    carbon_value = float(carbon_footprint)
                else:
                    # 문자열인 경우 분리 후 처리
                    carbon_value = float(carbon_footprint.split()[0])
                comparison['carbon_footprint'][label] = carbon_value
                
                # 목적함수 값
                if 'objective' in result:
                    if isinstance(result['objective'], dict) and '값' in result['objective']:
                        obj_value = result['objective']['값']
                        try:
                            obj_value = float(obj_value)
                        except:
                            pass
                    else:
                        obj_value = result.get('objective')
                    comparison['objective_values'][label] = obj_value
                
                # 솔버 시간
                solver_time = result.get('solver_time', 'N/A')
                if isinstance(solver_time, str) and '초' in solver_time:
                    solver_time = solver_time.replace('초', '').strip()
                try:
                    solver_time = float(solver_time)
                except:
                    pass
                comparison['solver_times'][label] = solver_time
                
                # 양극재 구성
                if 'cathode_composition' in result:
                    comparison['cathode_compositions'][label] = result['cathode_composition']
            else:
                comparison['summary'][label] = {'status': 'failed'}
        
        return comparison
    
    def visualize_comparison(self, comparison: Dict[str, Any], file_path: Optional[str] = None) -> str:
        """
        비교 결과 시각화
        
        Args:
            comparison: compare_results로 생성된 비교 결과
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not comparison or not comparison.get('summary'):
            raise ValueError("시각화할 비교 결과가 없습니다.")
        
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"optimization_comparison_{timestamp}.png"
        
        # 그래프 생성
        plt.figure(figsize=(12, 10))
        
        # 탄소발자국 비교
        if comparison.get('carbon_footprint'):
            plt.subplot(2, 2, 1)
            
            labels = list(comparison['carbon_footprint'].keys())
            values = list(comparison['carbon_footprint'].values())
            
            bars = plt.bar(labels, values)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.title('탄소발자국 비교')
            plt.xticks(rotation=45)
            plt.ylabel('탄소발자국 (kg CO2eq/kWh)')
        
        # 목적함수 값 비교
        if comparison.get('objective_values'):
            plt.subplot(2, 2, 2)
            
            labels = list(comparison['objective_values'].keys())
            values = list(comparison['objective_values'].values())
            
            # 값이 숫자인지 확인
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except:
                    numeric_values.append(0)
            
            bars = plt.bar(labels, numeric_values)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.title('목적함수 값 비교')
            plt.xticks(rotation=45)
            plt.ylabel('목적함수 값')
        
        # 솔버 시간 비교
        if comparison.get('solver_times'):
            plt.subplot(2, 2, 3)
            
            labels = list(comparison['solver_times'].keys())
            values = list(comparison['solver_times'].values())
            
            # 값이 숫자인지 확인
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except:
                    numeric_values.append(0)
            
            bars = plt.bar(labels, numeric_values)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s',
                        ha='center', va='bottom', rotation=0)
            
            plt.title('솔버 계산 시간 비교')
            plt.xticks(rotation=45)
            plt.ylabel('시간 (초)')
        
        # 요약 테이블
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # 테이블 데이터 준비
        table_data = []
        headers = ['', 'Status', 'Solver', 'Carbon']
        
        for label, info in comparison['summary'].items():
            row = [
                label,
                info.get('status', 'N/A'),
                info.get('solver', 'N/A'),
                info.get('carbon_footprint', 'N/A')
            ]
            table_data.append(row)
        
        plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        plt.title('결과 요약')
        
        # 전체 제목
        plt.suptitle('최적화 결과 비교', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 파일 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def create_report(self, file_path: Optional[str] = None) -> str:
        """
        종합 보고서 생성 및 저장
        
        Args:
            file_path: 저장할 파일 경로 (None이면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.formatted_results:
            raise ValueError("보고서 생성을 위한 결과가 없습니다. process_results()를 먼저 호출하세요.")
        
        # 파일명 자동 생성
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"optimization_report_{timestamp}.md"
        
        # 보고서 내용 생성
        report = []
        
        # 제목
        report.append("# PCF 최적화 결과 보고서")
        report.append(f"*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append("")
        
        # 요약
        report.append("## 최적화 요약")
        report.append(f"- **상태:** {self.formatted_results.get('status', 'N/A')}")
        report.append(f"- **솔버:** {self.formatted_results.get('solver', 'N/A')}")
        report.append(f"- **탄소발자국:** {self.formatted_results.get('carbon_footprint', 'N/A')}")
        report.append(f"- **계산 시간:** {self.formatted_results.get('solver_time', 'N/A')}")
        report.append("")
        
        # 목적함수
        objective_info = self.formatted_results.get('objective', {})
        if objective_info:
            report.append("## 목적함수 정보")
            for key, value in objective_info.items():
                report.append(f"- **{key}:** {value}")
            report.append("")
        
        # 감축비율
        reduction_ratios = self.formatted_results.get('reduction_ratios', {})
        if reduction_ratios:
            report.append("## Tier별 감축비율")
            for tier, items in reduction_ratios.items():
                report.append(f"### {tier}")
                for item, value in items.items():
                    report.append(f"- **{item}:** {value}")
                report.append("")
        
        # 양극재 구성
        cathode_composition = self.formatted_results.get('cathode_composition', {})
        if cathode_composition:
            report.append("## 양극재 구성")
            for key, value in cathode_composition.items():
                report.append(f"- **{key}:** {value}")
            report.append("")
        
        # 활성화된 활동
        active_activities = self.formatted_results.get('active_activities', {})
        if active_activities:
            report.append("## 활성화된 감축활동")
            for activity in active_activities.keys():
                report.append(f"- {activity}")
            report.append("")
        
        # 파일 저장
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        return str(path)
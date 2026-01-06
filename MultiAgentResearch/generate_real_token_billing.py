#!/usr/bin/env python3
"""
실제 텍스트 입력에서 토큰을 계산하여 정확한 Bedrock 비용을 산출하는 스크립트

Usage:
    python generate_real_token_billing.py [input_text_file]
    
Examples:
    python generate_real_token_billing.py                    # 대화형 입력
    python generate_real_token_billing.py input.txt          # 파일에서 입력
    echo "텍스트" | python generate_real_token_billing.py    # 파이프 입력
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTokenCalculator:
    """실제 토큰 계산 및 비용 산출"""
    
    def __init__(self):
        # Bedrock 모델별 가격표 (USD per 1K tokens)
        self.pricing = {
            "anthropic.claude-3-7-sonnet-20250219-v1:0": {
                "name": "Claude 3.7 Sonnet",
                "input": 0.003,
                "output": 0.015
            },
            "anthropic.claude-3-5-sonnet-20240620-v1:0": {
                "name": "Claude 3.5 Sonnet",
                "input": 0.003,
                "output": 0.015
            },
            "anthropic.claude-3-5-haiku-20241022-v1:0": {
                "name": "Claude 3.5 Haiku",
                "input": 0.0008,
                "output": 0.004
            },
            "us.amazon.nova-pro-v1:0": {
                "name": "Amazon Nova Pro",
                "input": 0.0008,
                "output": 0.0032
            },
            "us.amazon.nova-lite-v1:0": {
                "name": "Amazon Nova Lite",
                "input": 0.00006,
                "output": 0.00024
            },
            "us.amazon.nova-micro-v1:0": {
                "name": "Amazon Nova Micro",
                "input": 0.00004,
                "output": 0.00014
            }
        }
        
        # 에이전트별 모델 매핑 (실제 워크플로우와 동일)
        self.agent_workflow = {
            "coordinator": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "planner": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "supervisor": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "researcher": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "coder": "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "reporter": "anthropic.claude-3-7-sonnet-20250219-v1:0"
        }
    
    def count_tokens_approximation(self, text: str) -> int:
        """
        텍스트의 토큰 수를 근사 계산
        Claude 모델의 경우 대략 1 token = 4 characters (영어 기준)
        한국어의 경우 조금 더 효율적
        """
        # 한국어/중국어/일본어 문자 확인
        asian_chars = sum(1 for char in text if ord(char) > 0x4E00)
        total_chars = len(text)
        
        if asian_chars > total_chars * 0.3:  # 30% 이상이 아시아 문자
            # 한국어/중국어는 토큰 효율이 더 좋음
            return max(1, int(total_chars / 3))
        else:
            # 영어는 약 4글자당 1토큰
            return max(1, int(total_chars / 4))
    
    def count_tokens_tiktoken(self, text: str) -> int:
        """tiktoken을 사용한 토큰 계산 (더 정확함)"""
        try:
            import tiktoken
            
            # Claude는 정확한 tokenizer가 없으므로 GPT-4의 것을 근사치로 사용
            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
            
        except ImportError:
            logger.info("tiktoken이 설치되지 않았습니다. 근사 계산을 사용합니다.")
            return self.count_tokens_approximation(text)
        except Exception as e:
            logger.warning(f"tiktoken 사용 중 오류: {e}. 근사 계산을 사용합니다.")
            return self.count_tokens_approximation(text)
    
    def estimate_workflow_tokens(self, user_input: str) -> Dict:
        """워크플로우 전체의 토큰 사용량 추정"""
        
        input_tokens = self.count_tokens_tiktoken(user_input)
        
        # 각 에이전트별 예상 토큰 사용량 (실제 워크플로우 기반)
        workflow_estimates = {
            "planner": {
                "input_tokens": input_tokens + 1000,  # 시스템 프롬프트 + 사용자 입력
                "output_tokens": 500,  # 계획 수립
                "description": "분석 계획 수립 및 작업 분배"
            },
            "researcher": {
                "input_tokens": input_tokens + 2000,  # 검색 결과 포함
                "output_tokens": 2000,  # 연구 결과
                "description": "정보 수집 및 연구"
            },
            "coder": {
                "input_tokens": input_tokens + 3000,  # 연구 결과 + 코드 작성 프롬프트
                "output_tokens": 3000,  # 코드 + 분석 결과
                "description": "데이터 처리 및 시각화 코드 작성"
            },
            "reporter": {
                "input_tokens": input_tokens + 4000,  # 모든 결과 종합
                "output_tokens": 4000,  # 최종 보고서
                "description": "분석 결과 해석 및 보고서 작성"
            },
            "supervisor": {
                "input_tokens": input_tokens + 2000,  # 검증 대상
                "output_tokens": 800,  # 검증 결과
                "description": "코드 및 결과물 검증"
            },
            "coordinator": {
                "input_tokens": input_tokens + 1000,  # 전체 조정
                "output_tokens": 1000,  # 최종 응답
                "description": "전체 프로세스 조정 및 최종 응답 생성"
            }
        }
        
        total_cost = 0
        detailed_breakdown = {}
        
        for agent, estimates in workflow_estimates.items():
            model_id = self.agent_workflow[agent]
            model_info = self.pricing[model_id]
            
            input_cost = (estimates["input_tokens"] / 1000) * model_info["input"]
            output_cost = (estimates["output_tokens"] / 1000) * model_info["output"]
            agent_total_cost = input_cost + output_cost
            
            detailed_breakdown[agent] = {
                "model": model_info["name"],
                "model_id": model_id,
                "input_tokens": estimates["input_tokens"],
                "output_tokens": estimates["output_tokens"],
                "total_tokens": estimates["input_tokens"] + estimates["output_tokens"],
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": agent_total_cost,
                "description": estimates["description"]
            }
            
            total_cost += agent_total_cost
        
        return {
            "user_input_tokens": input_tokens,
            "total_estimated_cost": total_cost,
            "agent_breakdown": detailed_breakdown,
            "cost_by_model": self._group_by_model(detailed_breakdown)
        }
    
    def _group_by_model(self, breakdown: Dict) -> Dict:
        """모델별로 비용을 그룹화"""
        model_costs = {}
        
        for agent, data in breakdown.items():
            model_name = data["model"]
            if model_name not in model_costs:
                model_costs[model_name] = {
                    "total_cost": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "agents_using": []
                }
            
            model_costs[model_name]["total_cost"] += data["total_cost"]
            model_costs[model_name]["total_input_tokens"] += data["input_tokens"]
            model_costs[model_name]["total_output_tokens"] += data["output_tokens"]
            model_costs[model_name]["agents_using"].append(agent)
        
        return model_costs
    
    def generate_detailed_report(self, estimates: Dict) -> str:
        """상세한 비용 분석 보고서 생성"""
        
        current_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")
        
        report = f"""# Bedrock 실제 토큰 기반 비용 분석 보고서

생성일시: {current_time}
분석 방법: 실제 텍스트 토큰 계산 기반 추정
사용자 입력 토큰 수: {estimates['user_input_tokens']:,}개

## 📊 전체 비용 요약

**총 예상 비용**: ${estimates['total_estimated_cost']:.6f}
**총 예상 토큰**: {sum(agent['total_tokens'] for agent in estimates['agent_breakdown'].values()):,}개
**입력 토큰**: {sum(agent['input_tokens'] for agent in estimates['agent_breakdown'].values()):,}개
**출력 토큰**: {sum(agent['output_tokens'] for agent in estimates['agent_breakdown'].values()):,}개

## 💰 에이전트별 상세 분석

"""
        
        # 비용 순으로 정렬
        sorted_agents = sorted(
            estimates['agent_breakdown'].items(),
            key=lambda x: x[1]['total_cost'],
            reverse=True
        )
        
        for agent, data in sorted_agents:
            percentage = (data['total_cost'] / estimates['total_estimated_cost']) * 100
            
            report += f"""### {agent.upper()} 에이전트
- **역할**: {data['description']}
- **사용 모델**: {data['model']}
- **입력 토큰**: {data['input_tokens']:,}개 (${data['input_cost']:.6f})
- **출력 토큰**: {data['output_tokens']:,}개 (${data['output_cost']:.6f})
- **총 비용**: ${data['total_cost']:.6f} ({percentage:.1f}%)

"""
        
        report += "## 🎯 모델별 비용 분석\n\n"
        
        # 모델별 비용을 비용 순으로 정렬
        sorted_models = sorted(
            estimates['cost_by_model'].items(),
            key=lambda x: x[1]['total_cost'],
            reverse=True
        )
        
        for model, data in sorted_models:
            percentage = (data['total_cost'] / estimates['total_estimated_cost']) * 100
            
            report += f"""### {model}
- **사용 에이전트**: {', '.join(data['agents_using'])}
- **총 입력 토큰**: {data['total_input_tokens']:,}개
- **총 출력 토큰**: {data['total_output_tokens']:,}개
- **총 비용**: ${data['total_cost']:.6f} ({percentage:.1f}%)

"""
        
        report += f"""## 📈 비용 최적화 제안

### 현재 워크플로우 분석
1. **가장 비용이 높은 에이전트**: {sorted_agents[0][0]} (${sorted_agents[0][1]['total_cost']:.6f})
2. **가장 비용이 높은 모델**: {sorted_models[0][0]} (${sorted_models[0][1]['total_cost']:.6f})

### 최적화 방안
1. **모델 변경**: 비용이 높은 작업에 더 저렴한 모델 사용
   - Claude 3.7 Sonnet → Claude 3.5 Haiku로 변경 시 약 80% 비용 절감
   - 단순 작업에는 Nova Micro 사용 시 약 95% 비용 절감

2. **캐시 활용**: 반복되는 프롬프트에 캐시 사용으로 50-90% 절감

3. **토큰 최적화**: 프롬프트 길이 최적화로 10-30% 절감

## 📝 계산 기준
- 토큰 계산: tiktoken 라이브러리 기반 (Claude 근사치)
- 가격표: AWS Bedrock 공식 가격 (2024년 기준)
- 워크플로우: 실제 bedrock_manus 에이전트 구성 기반

**주의**: 이는 추정치이며, 실제 사용량은 쿼리 복잡도, 응답 길이, 도구 사용 등에 따라 달라질 수 있습니다.
"""
        
        return report

def get_input_text() -> str:
    """입력 텍스트 가져오기 (파일, 파이프, 또는 대화형)"""
    
    # 명령행 인수로 파일이 지정된 경우
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            sys.exit(1)
    
    # 파이프 입력 확인
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    
    # 대화형 입력
    print("분석할 텍스트를 입력하세요 (여러 줄 가능, 빈 줄 두 번으로 종료):")
    lines = []
    empty_lines = 0
    
    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_lines += 1
                if empty_lines >= 2:
                    break
            else:
                empty_lines = 0
            lines.append(line)
        except EOFError:
            break
    
    return "\n".join(lines).strip()

def main():
    """메인 함수"""
    try:
        print("🧮 실제 토큰 기반 Bedrock 비용 계산기")
        print("-" * 50)
        
        # 입력 텍스트 가져오기
        input_text = get_input_text()
        
        if not input_text:
            print("❌ 입력 텍스트가 없습니다.")
            sys.exit(1)
        
        print(f"📝 입력 텍스트 길이: {len(input_text):,} 문자")
        
        # 토큰 계산 및 비용 분석
        calculator = RealTokenCalculator()
        estimates = calculator.estimate_workflow_tokens(input_text)
        
        # 결과 출력
        print(f"🔢 예상 입력 토큰: {estimates['user_input_tokens']:,}개")
        print(f"💰 예상 총 비용: ${estimates['total_estimated_cost']:.6f}")
        
        # 상세 보고서 생성
        detailed_report = calculator.generate_detailed_report(estimates)
        
        # artifacts 폴더에 저장
        os.makedirs("./artifacts", exist_ok=True)
        
        with open("./artifacts/bedrock_billing.txt", "w", encoding="utf-8") as f:
            f.write(detailed_report)
        
        # JSON 데이터도 저장
        with open("./artifacts/token_analysis.json", "w", encoding="utf-8") as f:
            json.dump(estimates, f, indent=2, ensure_ascii=False)
        
        print("\n🎉 분석 완료!")
        print("📁 생성된 파일:")
        print("   - ./artifacts/bedrock_billing.txt (상세 보고서)")
        print("   - ./artifacts/token_analysis.json (원시 데이터)")
        
        # 비용이 높은 상위 3개 에이전트 표시
        sorted_agents = sorted(
            estimates['agent_breakdown'].items(),
            key=lambda x: x[1]['total_cost'],
            reverse=True
        )
        
        print(f"\n💡 비용 TOP 3:")
        for i, (agent, data) in enumerate(sorted_agents[:3], 1):
            percentage = (data['total_cost'] / estimates['total_estimated_cost']) * 100
            print(f"   {i}. {agent}: ${data['total_cost']:.6f} ({percentage:.1f}%)")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 도움말 요청 확인
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        sys.exit(0)
    
    main() 
#!/usr/bin/env python3
"""
Bedrock 비용 계산 및 billing 정보 생성 스크립트

Usage:
    python generate_billing.py [query_complexity]
    
Examples:
    python generate_billing.py simple     # 간단한 쿼리 복잡도
    python generate_billing.py medium     # 중간 쿼리 복잡도 (기본값)
    python generate_billing.py complex    # 복잡한 쿼리 복잡도
    python generate_billing.py very_complex # 매우 복잡한 쿼리 복잡도
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_billing_report(query_complexity: str = "medium") -> bool:
    """
    Bedrock billing 보고서 생성
    
    Args:
        query_complexity: 쿼리 복잡도 ("simple", "medium", "complex", "very_complex")
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 먼저 실제 토큰 추적 데이터로 시도
        try:
            from src.utils.token_tracker import token_tracker
            
            session_summary = token_tracker.get_session_summary()
            if "error" not in session_summary and session_summary.get('total_calls', 0) > 0:
                logger.info("실제 토큰 사용량 데이터가 있습니다. 실시간 billing 보고서를 생성합니다.")
                
                billing_result = token_tracker.save_billing_report(
                    filepath="./artifacts/bedrock_billing.txt"
                )
                
                print(f"✅ {billing_result}")
                print(f"💰 총 비용: ${session_summary['total_cost']:.6f}")
                print(f"📞 총 호출 횟수: {session_summary['total_calls']:,}회")
                return True
                
        except Exception as real_time_error:
            logger.debug(f"실시간 토큰 추적 실패: {real_time_error}")
        
        # fallback: 추정 billing 보고서 생성
        logger.info("실시간 데이터가 없습니다. 추정 billing 보고서를 생성합니다.")
        
        from src.tools.reporter_tools import handle_estimate_bedrock_costs
        
        billing_result = handle_estimate_bedrock_costs(
            query_complexity=query_complexity,
            execution_count=1,
            include_analysis=True
        )
        
        print(f"✅ {billing_result}")
        print(f"📊 복잡도: {query_complexity.upper()}")
        print(f"📄 상세 정보가 ./artifacts/bedrock_billing.txt에 저장되었습니다.")
        
        return True
        
    except Exception as e:
        logger.error(f"Billing 보고서 생성 실패: {e}")
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    try:
        # 명령행 인수 처리
        valid_complexities = ["simple", "medium", "complex", "very_complex"]
        
        if len(sys.argv) > 1:
            query_complexity = sys.argv[1].lower()
            if query_complexity not in valid_complexities:
                print(f"❌ 잘못된 복잡도: {query_complexity}")
                print(f"📋 사용 가능한 복잡도: {', '.join(valid_complexities)}")
                sys.exit(1)
        else:
            query_complexity = "medium"
        
        print("💰 Bedrock 비용 계산 시작")
        print(f"📊 쿼리 복잡도: {query_complexity.upper()}")
        print("📂 출력 위치: ./artifacts/bedrock_billing.txt")
        print("-" * 50)
        
        # artifacts 폴더 생성
        os.makedirs("./artifacts", exist_ok=True)
        
        # billing 보고서 생성
        success = generate_billing_report(query_complexity)
        
        if success:
            print("\n🎉 Bedrock billing 정보 생성이 완료되었습니다!")
            
            # 파일 확인
            billing_file = Path("./artifacts/bedrock_billing.txt")
            if billing_file.exists():
                file_size = billing_file.stat().st_size
                print(f"📁 파일 크기: {file_size:,} bytes")
                print(f"📄 파일 위치: {billing_file.absolute()}")
            
        else:
            print("\n❌ Billing 정보 생성에 실패했습니다.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)

def show_help():
    """도움말 출력"""
    help_text = """
💰 Bedrock 비용 계산 도구

이 스크립트는 Bedrock 사용량에 대한 비용 분석 보고서를 생성합니다:
1. 실시간 토큰 사용량이 있으면 정확한 비용 계산
2. 토큰 데이터가 없으면 쿼리 복잡도를 기반으로 추정 비용 계산
3. 상세한 분석 보고서를 ./artifacts/bedrock_billing.txt에 저장

사용법:
    python generate_billing.py [복잡도]

복잡도 옵션:
    simple       - 간단한 쿼리 (< 100자)
    medium       - 중간 복잡도 쿼리 (100-300자) [기본값]
    complex      - 복잡한 쿼리 (300-600자)
    very_complex - 매우 복잡한 쿼리 (> 600자)

예시:
    python generate_billing.py                # 기본 복잡도 (medium)
    python generate_billing.py simple         # 간단한 쿼리
    python generate_billing.py complex        # 복잡한 쿼리

출력:
    ./artifacts/bedrock_billing.txt          # 상세 비용 분석 보고서
"""
    print(help_text)

if __name__ == "__main__":
    # 도움말 요청 확인
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    main() 
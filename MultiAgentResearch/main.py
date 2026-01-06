"""
Entry point script for the LangGraph Demo.
"""
import os
import sys
import asyncio
import logging
from typing import Optional, Dict
from src.workflow import run_agent_workflow_async
from src.config.async_config import AsyncConfig
from src.utils.file_utils import remove_artifact_folder
from src.utils.report_finalizer import finalize_report

# 로깅 설정
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, config: Optional[AsyncConfig] = None):
        self.config = config or AsyncConfig()
        
    async def process_query(self, user_query: str) -> Dict:
        """사용자 쿼리 처리"""
        try:
            # 아티팩트 폴더 정리
            # remove_artifact_folder()
            
            # 워크플로우 실행
            result = await run_agent_workflow_async(
                user_input=user_query,
                debug=False,
                config=self.config
            )
            
            # 워크플로우 완료 후 보고서 마무리 처리
            self._finalize_report_if_completed(result)
            
            return result
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류 발생: {e}")
            raise
    
    def _finalize_report_if_completed(self, workflow_result: Dict) -> None:
        """
        워크플로우 결과를 확인하여 보고서가 완료되었으면 마무리 처리 수행
        
        Args:
            workflow_result: 워크플로우 실행 결과
        """
        try:
            # 워크플로우가 성공적으로 완료되었는지 확인
            if self._is_workflow_completed(workflow_result):
                logger.info("보고서 작성이 완료되었습니다. 마무리 처리를 시작합니다.")
                
                # HTML 수정 및 PDF 생성
                success = finalize_report(artifacts_dir="./artifacts")
                
                if success:
                    logger.info("✅ 보고서 마무리 처리가 완료되었습니다.")
                    print("\n🎉 보고서 작성 및 마무리 처리가 완료되었습니다!")
                    print("📁 생성된 파일:")
                    print("   - ./artifacts/progressive_report.html (수정 완료)")
                    print("   - ./artifacts/progressive_report.pdf (PDF 버전)")
                else:
                    logger.warning("⚠️ 보고서 마무리 처리 중 일부 오류가 발생했습니다.")
                    print("\n⚠️ 보고서는 완성되었지만 마무리 처리 중 문제가 발생했습니다.")
                    print("HTML 파일은 ./artifacts/progressive_report.html에서 확인할 수 있습니다.")
            else:
                logger.debug("워크플로우가 완전히 완료되지 않아 마무리 처리를 건너뜁니다.")
                
        except Exception as e:
            logger.error(f"보고서 마무리 처리 중 오류 발생: {e}")
            print(f"\n⚠️ 보고서 마무리 처리 중 오류가 발생했습니다: {e}")
    
    def _is_workflow_completed(self, workflow_result: Dict) -> bool:
        """
        워크플로우가 완료되었는지 확인
        
        Args:
            workflow_result: 워크플로우 실행 결과
            
        Returns:
            bool: 완료 여부
        """
        try:
            # 기본적인 완료 조건들을 확인
            conditions = [
                # 결과가 존재하고 None이 아님
                workflow_result is not None,
                
                # history가 존재함
                "history" in workflow_result,
                
                # history가 비어있지 않음
                len(workflow_result.get("history", [])) > 0,
                
                # progressive_report.html 파일이 존재함
                os.path.exists("./artifacts/progressive_report.html")
            ]
            
            # 모든 조건이 만족되면 완료로 간주
            is_completed = all(conditions)
            
            if is_completed:
                logger.info("워크플로우 완료 조건을 만족합니다.")
            else:
                logger.debug(f"워크플로우 완료 조건 확인: {conditions}")
            
            return is_completed
            
        except Exception as e:
            logger.error(f"워크플로우 완료 상태 확인 중 오류: {e}")
            return False

async def execute_query(user_query: str) -> Dict:
    """쿼리 실행 함수"""
    try:
        config = AsyncConfig()
        processor = QueryProcessor(config)
        result = await processor.process_query(user_query)
        return result
    except Exception as e:
        logger.error(f"쿼리 실행 중 오류 발생: {e}")
        raise

def execution(user_query: str) -> Dict:
    """
    app.py에서 사용하는 동기 실행 함수
    
    Args:
        user_query: 사용자 쿼리
        
    Returns:
        워크플로우 실행 결과
    """
    try:
        # 비동기 함수를 동기적으로 실행
        result = asyncio.run(execute_query(user_query))
        return result
    except Exception as e:
        logger.error(f"execution 함수 실행 중 오류 발생: {e}")
        raise

async def main():
    try:
        if len(sys.argv) > 1:
            user_query = " ".join(sys.argv[1:])
        else:
            user_query = input("Enter your query: ")
        
        result = await execute_query(user_query)
        
        print("\n=== Conversation History ===")
        print("result", result)
        if "history" in result:
            for history in result["history"]:
                print("===")
                print(f'agent: {history["agent"]}')
                print(f'message: {history["message"]}')
                
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
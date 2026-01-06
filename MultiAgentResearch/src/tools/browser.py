import asyncio
import nest_asyncio
import os
import logging
from pydantic import BaseModel, Field
from typing import ClassVar, Type, Optional
from langchain.tools import BaseTool
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from src.agents.llm import browser_llm
from src.tools.decorators import create_logged_tool
from src.config import CHROME_INSTANCE_PATH, BROWSER_HEADLESS

nest_asyncio.apply()
logger = logging.getLogger(__name__)

# 환경 변수로 SSL 검증 비활성화 (PostHog 연결 문제 해결)
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''

def get_browser_config() -> Optional[Browser]:
    """안전한 브라우저 설정을 반환합니다."""
    try:
        # BROWSER_HEADLESS 설정 확인 (기본값: True)
        headless = True
        if BROWSER_HEADLESS is not None:
            headless = str(BROWSER_HEADLESS).lower() in ('true', '1', 'yes')
        
        config = BrowserConfig(
            headless=headless,
            disable_security=True,  # SSL 문제 해결
            chrome_instance_path=CHROME_INSTANCE_PATH if CHROME_INSTANCE_PATH else None,
            additional_args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-web-security',
                '--ignore-certificate-errors',
                '--ignore-ssl-errors',
                '--allow-running-insecure-content',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        return Browser(config=config)
    except Exception as e:
        logger.warning(f"Failed to create browser config: {e}")
        return None

# 브라우저 설정
expected_browser = get_browser_config()

class BrowserUseInput(BaseModel):
    """BrowserTool 입력 스키마"""
    instruction: str = Field(..., description="브라우저 사용 지시사항")


class BrowserTool(BaseTool):
    name: ClassVar[str] = "browser"
    args_schema: Type[BaseModel] = BrowserUseInput
    description: ClassVar[str] = (
        "웹 브라우저와 상호작용하는 도구입니다. 입력은 'google.com에 접속해서 browser-use 검색' 또는 "
        "'Reddit에 방문해서 AI에 관한 인기 게시물 찾기'와 같은 자연어 설명이어야 합니다."
    )

    def _run(self, instruction: str) -> str:
        """브라우저 작업을 동기적으로 실행합니다."""
        if expected_browser is None:
            return "브라우저 설정에 문제가 있습니다. 대신 crawl_tool을 사용하여 특정 URL을 분석하거나 tavily_tool로 검색해보세요."
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_agent():
                try:
                    agent = BrowserAgent(
                        task=instruction,
                        llm=browser_llm,
                        browser=expected_browser
                    )
                    return await agent.run()
                except Exception as e:
                    logger.error(f"Browser agent execution failed: {e}")
                    return f"브라우저 실행 중 오류가 발생했습니다: {str(e)}\n\n대안으로 crawl_tool을 사용하여 특정 URL을 분석하거나 tavily_tool로 검색해보세요."
            
            result = loop.run_until_complete(run_agent())
            
            if isinstance(result, AgentHistoryList):
                return result.final_result()
            return str(result)
            
        except Exception as e:
            logger.error(f"Browser tool execution failed: {e}")
            return f"브라우저 도구 실행 중 오류가 발생했습니다: {str(e)}\n\n해결 방법:\n1. 모든 Chrome 브라우저를 종료하고 다시 시도\n2. crawl_tool을 사용하여 특정 URL 분석\n3. tavily_tool을 사용하여 웹 검색"
        finally:
            try:
                loop.close()
            except:
                pass
        
    async def _arun(self, instruction: str) -> str:
        """브라우저 작업을 비동기적으로 실행합니다."""
        if expected_browser is None:
            return "브라우저 설정에 문제가 있습니다. 대신 crawl_tool을 사용하여 특정 URL을 분석하거나 tavily_tool로 검색해보세요."
        
        try:
            agent = BrowserAgent(
                task=instruction,
                llm=browser_llm,
                browser=expected_browser
            )
            
            result = await agent.run()
            
            if isinstance(result, AgentHistoryList):
                return result.final_result()
            return str(result)
            
        except Exception as e:
            logger.error(f"Async browser execution failed: {e}")
            return f"브라우저 작업 실행 오류: {str(e)}\n\n대안으로 crawl_tool을 사용하여 특정 URL을 분석하거나 tavily_tool로 검색해보세요."
        

def handle_browser_tool(instruction: str) -> str:
    """
    주어진 지시사항에 따라 브라우저 도구 실행을 처리합니다.
    """
    logger.info(f"브라우저 도구 실행: {instruction}")
    
    try:
        # 로깅된 브라우저 도구 인스턴스 생성
        LoggedBrowserTool = create_logged_tool(BrowserTool)
        browser_tool = LoggedBrowserTool()
        
        # 동기적 _run 메서드 사용
        return browser_tool._run(instruction)
    except Exception as e:
        logger.error(f"Browser tool handler failed: {e}")
        return f"브라우저 도구 처리 중 오류가 발생했습니다: {str(e)}\n\n대안 방법:\n1. 모든 Chrome 인스턴스를 종료하고 재시도\n2. crawl_tool 사용: 특정 URL 제공\n3. tavily_tool 사용: 웹 검색"


# 사용 예시:
if __name__ == "__main__":
    # 동기적 사용 예시
    result = handle_browser_tool("google.com에 접속해서 query 검색")
    print("결과:", result)

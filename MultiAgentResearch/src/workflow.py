import logging
import asyncio
from textwrap import dedent
from src.config import TEAM_MEMBERS
from src.graph import build_graph
from src.utils.common_utils import get_message_from_string
from src.config.async_config import AsyncConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger(__name__).setLevel(logging.DEBUG)

# 로거 설정을 전역으로 한 번만 수행
logger = logging.getLogger(__name__)
logger.propagate = False
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Create the graph
graph = build_graph()

def get_graph():
    return graph

async def run_agent_workflow_async(user_input: str, debug: bool = False, config: AsyncConfig = None):
    """비동기로 에이전트 워크플로우를 실행합니다.

    Args:
        user_input: 사용자의 쿼리나 요청
        debug: 디버그 로깅 활성화 여부
        config: 비동기 설정

    Returns:
        워크플로우 완료 후의 최종 상태
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    if config is None:
        config = AsyncConfig()

    logger.info(f"{Colors.GREEN}===== Starting workflow ====={Colors.END}")
    logger.info(f"{Colors.GREEN}\nuser input: {user_input}{Colors.END}")
    
    # 사용자 프롬프트 준비
    user_prompts = dedent(
        '''
        Here is a user request: <user_request>{user_request}</user_request>
        '''
    )
    context = {"user_request": user_input}
    user_prompts = user_prompts.format(**context)
    messages = [get_message_from_string(role="user", string=user_prompts, imgs=[])]

    try:
        # 비동기로 그래프 실행
        result = await asyncio.to_thread(
            graph.invoke,
            input={
                "TEAM_MEMBERS": TEAM_MEMBERS,
                "messages": messages,
                "deep_thinking_mode": True,
                "search_before_planning": False,
                "request": user_input
            },
            config={
                "recursion_limit": 100
            }
        )
        
        logger.debug(f"{Colors.RED}Final workflow state: {result}{Colors.END}")
        logger.info(f"{Colors.GREEN}===== Workflow completed successfully ====={Colors.END}")
        
        # 워크플로우 완료 시 실제 토큰 사용량 기반으로 billing 정보 생성
        try:
            from src.utils.token_tracker import token_tracker
            logger.info(f"{Colors.BLUE}===== Generating Real-time Bedrock billing information ====={Colors.END}")
            
            # 실제 사용량 기반 billing 보고서 생성
            billing_result = token_tracker.save_billing_report(
                filepath="./artifacts/bedrock_billing.txt"
            )
            logger.info(f"{Colors.GREEN}{billing_result}{Colors.END}")
            
            # 세션 요약 정보 출력
            session_summary = token_tracker.get_session_summary()
            if "error" not in session_summary:
                total_cost = session_summary['total_cost']
                total_calls = session_summary['total_calls']
                logger.info(f"{Colors.YELLOW}💰 실시간 비용 요약: ${total_cost:.6f} (총 {total_calls}회 호출){Colors.END}")
            
        except Exception as billing_error:
            logger.warning(f"{Colors.YELLOW}Failed to generate real-time billing information: {str(billing_error)}{Colors.END}")
            # 대체 방법으로 기존 정적 추정 방식 사용
            try:
                from src.tools.reporter_tools import handle_estimate_bedrock_costs
                logger.info(f"{Colors.BLUE}===== Fallback: Generating estimated billing information ====={Colors.END}")
                
                # 쿼리 복잡도 자동 추정 (메시지 길이 기반)
                message_length = len(user_input)
                if message_length < 100:
                    complexity = "simple"
                elif message_length < 300:
                    complexity = "medium"
                elif message_length < 600:
                    complexity = "complex"
                else:
                    complexity = "very_complex"
                
                billing_result = handle_estimate_bedrock_costs(
                    query_complexity=complexity,
                    execution_count=1,
                    include_analysis=True
                )
                logger.info(f"{Colors.GREEN}Fallback billing estimation saved to artifacts/bedrock_billing.txt{Colors.END}")
                
            except Exception as fallback_error:
                logger.error(f"{Colors.RED}Both real-time and fallback billing failed: {str(fallback_error)}{Colors.END}")
        
        return result
        
    except Exception as e:
        logger.error(f"{Colors.RED}Workflow failed: {str(e)}{Colors.END}")
        raise

def run_agent_workflow(user_input: str, debug: bool = False):
    """기존 동기식 워크플로우 실행 함수 (하위 호환성 유지)"""
    return asyncio.run(run_agent_workflow_async(user_input, debug))

async def process_batch(batch: list, config: AsyncConfig):
    """배치 단위로 작업을 처리합니다."""
    tasks = []
    for item in batch:
        task = asyncio.create_task(
            run_agent_workflow_async(
                user_input=item,
                debug=False,
                config=config
            )
        )
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

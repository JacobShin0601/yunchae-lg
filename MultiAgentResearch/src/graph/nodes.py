import os
import json
import pprint
import logging
from copy import deepcopy
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import END

from src.agents.agents import create_react_agent
#from src.agents.llm import get_llm_by_type, llm_call
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP, AGENT_PROMPT_CACHE_MAP
from src.prompts.template import apply_prompt_template
from src.tools.search import tavily_tool
from .types import State, Router

from textwrap import dedent
from src.utils.common_utils import get_message_from_string
import re
from datetime import datetime

llm_module = os.environ.get('LLM_MODULE', 'src.agents.llm')
if llm_module == 'src.agents.llm_st': from src.agents.llm_st import get_llm_by_type, llm_call
else: from src.agents.llm import get_llm_by_type, llm_call

# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
#logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
FULL_PLAN_FORMAT = "Here is full plan :\n\n<full_plan>\n{}\n</full_plan>\n\n*Please consider this to select the next step.*"
CLUES_FORMAT = "Here is clues form {}:\n\n<clues>\n{}\n</clues>\n\n"

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def research_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the researcher agent that performs research tasks."""
    logger.info(f"{Colors.GREEN}===== Research agent starting task ====={Colors.END}")
    research_agent = create_react_agent(agent_name="researcher")
    result = research_agent.invoke(state=state)
    
    # 방어적으로 result 구조 확인
    if not result or "content" not in result or not result["content"]:
        logger.error(f"{Colors.RED}Research agent returned invalid result: {result}{Colors.END}")
        error_message = "Research agent failed to generate a proper response."
        history = state.get("history", [])
        history.append({"agent":"researcher", "message": error_message})
        return Command(
            update={
                "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("researcher", error_message), imgs=[])],
                "messages_name": "researcher",
                "clues": state.get("clues", ""),
                "history": history
            },
            goto="supervisor",
        )
    
    # 안전하게 텍스트 추출
    try:
        response_text = result["content"][-1]["text"]
    except (IndexError, KeyError, TypeError):
        logger.error(f"{Colors.RED}Failed to extract text from research result: {result}{Colors.END}")
        response_text = "Research agent response could not be processed properly."
    
    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("researcher", response_text)])
    
    # 연구 데이터 저장 로직 추가
    try:
        # artifacts 디렉토리 생성
        os.makedirs("./artifacts", exist_ok=True)
        
        # 연구 결과를 파일에 저장
        research_file = './artifacts/research_data.txt'
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 기존 파일이 있으면 추가, 없으면 새로 생성
        mode = 'a' if os.path.exists(research_file) else 'w'
        
        with open(research_file, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write("# 연구 데이터 수집 결과\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"## 연구 단계: {current_time}\n")
            f.write("=" * 80 + "\n\n")
            f.write("### 수집된 정보:\n")
            f.write(response_text)
            f.write("\n\n")
            f.write("-" * 80 + "\n\n")
        
        logger.info(f"{Colors.GREEN}Research data saved to: {research_file}{Colors.END}")
        
    except Exception as e:
        logger.error(f"{Colors.RED}Failed to save research data: {str(e)}{Colors.END}")

    logger.info("Research agent completed task")
    logger.debug(f"Research agent response: {response_text}")

    history = state.get("history", [])
    history.append({"agent":"researcher", "message": response_text})
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("researcher", response_text), imgs=[])],
            "messages_name": "researcher",
            "clues": clues,
            "history": history
        },
        goto="supervisor",
    )

def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for the coder agent that executes Python code."""
    logger.info(f"{Colors.GREEN}===== Code agent starting task ====={Colors.END}")
    coder_agent = create_react_agent(agent_name="coder")
    result = coder_agent.invoke(state=state)

    # 방어적으로 result 구조 확인
    if not result or "content" not in result or not result["content"]:
        logger.error(f"{Colors.RED}Coder agent returned invalid result: {result}{Colors.END}")
        error_message = "Coder agent failed to generate a proper response."
        history = state.get("history", [])
        history.append({"agent":"coder", "message": error_message})
        return Command(
            update={
                "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("coder", error_message), imgs=[])],
                "messages_name": "coder",
                "clues": state.get("clues", ""),
                "history": history,
                "artifacts": state.get("artifacts", [])
            },
            goto="supervisor",
        )
    
    # 안전하게 텍스트 추출
    try:
        response_text = result["content"][-1]["text"]
    except (IndexError, KeyError, TypeError):
        logger.error(f"{Colors.RED}Failed to extract text from coder result: {result}{Colors.END}")
        response_text = "Coder agent response could not be processed properly."

    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("coder", response_text)])

    logger.debug(f"\n{Colors.RED}Coder - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coder response:\n{pprint.pformat(response_text, indent=2, width=100)}{Colors.END}")

    # Extract code blocks from the response
    code_blocks = []
    if "```python" in response_text:
        code_blocks = re.findall(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    
    # Save code blocks to artifacts
    artifacts = state.get("artifacts", [])
    for i, code in enumerate(code_blocks):
        if "plt" in code or "plot" in code or "chart" in code or "graph" in code:
            artifact_name = f"chart_code_{i+1}.py"
            artifacts.append([artifact_name, code])
            
            # Save the code to file
            os.makedirs("./artifacts", exist_ok=True)
            with open(f"./artifacts/{artifact_name}", "w", encoding="utf-8") as f:
                f.write(code)

    history = state.get("history", [])
    history.append({"agent":"coder", "message": response_text})

    logger.info(f"{Colors.GREEN}===== Coder completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("coder", response_text), imgs=[])],
            "messages_name": "coder",
            "clues": clues,
            "history": history,
            "artifacts": artifacts
        },
        goto="supervisor",
    )

# def browser_node(state: State) -> Command[Literal["supervisor"]]:
#     """Node for the browser agent that performs web browsing tasks."""
#     logger.info("Browser agent starting task")
#     browser_agent = create_react_agent(agent_name="browser")
#     result = browser_agent.invoke(state=state)
#     
#     clues = state.get("clues", "")
#     clues = '\n\n'.join([clues, CLUES_FORMAT.format("browser", result["content"][-1]["text"])])
#     logger.info("Browser agent completed task")
#     logger.debug(f"Browser agent response: {result['content'][-1]["text"]}")
# 
#     history = state.get("history", [])
#     history.append({"agent":"browser", "message": result["content"][-1]["text"]})
#     return Command(
#         update={
#             "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("browser", result["content"][-1]["text"]), imgs=[])],
#             "messages_name": "browser",
#             "clues": clues,
#             "history": history
#         },
#         goto="supervisor"
#     )

def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
    """Supervisor node that decides which agent should act next."""
    logger.info(f"{Colors.GREEN}===== Supervisor evaluating next action ====={Colors.END}")
    
    # 레포트 완성 상태 확인 - 추가된 로직
    if state.get("report_completed", False):
        logger.info(f"{Colors.GREEN}===== Report already completed, ending workflow ====={Colors.END}")
        history = state.get("history", [])
        history.append({"agent":"supervisor", "message": "Report completed successfully. Workflow terminated."})
        return Command(
            goto="__end__",
            update={
                "next": "__end__",
                "history": history
            }
        )
    
    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["supervisor"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Supervisor - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Supervisor - Prompt Cache Disabled{Colors.END}")
    system_prompts, messages = apply_prompt_template("supervisor", state, prompt_cache=prompt_cache, cache_type=cache_type)    
    llm = get_llm_by_type(AGENT_LLM_MAP["supervisor"])
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
    
    clues, full_plan = state.get("clues", ""), state.get("full_plan", "")       
    messages[-1]["content"][-1]["text"] = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan), clues])
    
    response, ai_message = llm_caller.invoke(
        agent_name="supervisor",
        messages=messages,
        system_prompts=system_prompts,
        enable_reasoning=False,
        reasoning_budget_tokens=8192
    )
    full_response = response["text"]

    if full_response.startswith("```json"): full_response = full_response.removeprefix("```json")
    if full_response.endswith("```"): full_response = full_response.removesuffix("```")
    
    full_response = json.loads(full_response)   
    goto = full_response["next"]

    logger.debug(f"\n{Colors.RED}Supervisor - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Supervisor response:{full_response}{Colors.END}")

    # 최종 레포트 생성 완료 키워드 감지 - 추가된 로직
    if "final_files" in state and len(state.get("final_files", [])) >= 2:
        logger.info(f"{Colors.GREEN}===== Final report files detected, ending workflow ====={Colors.END}")
        goto = "__end__"
    elif "report generation completed" in str(full_response).lower() or "workflow finished" in str(full_response).lower():
        logger.info(f"{Colors.GREEN}===== Report completion detected in supervisor response, ending workflow ====={Colors.END}")
        goto = "__end__"
    elif goto == "FINISH":
        goto = "__end__"
        logger.info(f"\n{Colors.GREEN}===== Workflow completed ====={Colors.END}")
    else:
        # 만약 reporter에서 온 메시지이고 최종 파일들이 생성되었다면 종료
        if (state.get("messages_name") == "reporter" and 
            any("successfully generated" in msg.get("content", [{}])[0].get("text", "") 
                for msg in state.get("messages", []))):
            logger.info(f"{Colors.GREEN}===== Final report generated by reporter, ending workflow ====={Colors.END}")
            goto = "__end__"
        else:
            logger.info(f"{Colors.GREEN}Supervisor delegating to: {goto}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"supervisor", "message": full_response})
    logger.info(f"{Colors.GREEN}===== Supervisor completed task ====={Colors.END}")
    return Command(
        goto=goto,
        update={
            "next": goto,
            "history": history
        }
    )

def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Planner node that generate the full plan."""
    logger.info(f"{Colors.GREEN}===== Planner generating full plan ====={Colors.END}")
    logger.info(f"{Colors.BLUE}===== Planner - Deep thinking mode: {state.get("deep_thinking_mode")} ====={Colors.END}")
    logger.info(f"{Colors.BLUE}===== Planner - Search before planning: {state.get("search_before_planning")} ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Planner - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["planner"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Planner - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Planner - Prompt Cache Disabled{Colors.END}")
    system_prompts, messages = apply_prompt_template("planner", state, prompt_cache=prompt_cache, cache_type=cache_type)
    # whether to enable deep thinking mode
       
    full_plan = state.get("full_plan", "")
    messages[-1]["content"][-1]["text"] = '\n\n'.join([messages[-1]["content"][-1]["text"], FULL_PLAN_FORMAT.format(full_plan)])
    
    llm = get_llm_by_type(AGENT_LLM_MAP["planner"])    
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
        
    if state.get("deep_thinking_mode"): llm = get_llm_by_type("reasoning")
    if state.get("search_before_planning"):
        searched_content = tavily_tool.invoke({"query": state["request"]})
        messages = deepcopy(messages)
        messages[-1]["content"][-1]["text"] += f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    if AGENT_LLM_MAP["planner"] in ["reasoning"]: enable_reasoning = True
    else: enable_reasoning = False

    response, ai_message = llm_caller.invoke(
        agent_name="planner",
        messages=messages,
        system_prompts=system_prompts,
        enable_reasoning=enable_reasoning,
        reasoning_budget_tokens=8192
    )
    full_response = response["text"]
    logger.debug(f"\n{Colors.RED}Planner response:\n{pprint.pformat(full_response, indent=2, width=100)}{Colors.END}")

    goto = "supervisor"
        
    history = state.get("history", [])
    history.append({"agent":"planner", "message": full_response})
    logger.info(f"{Colors.GREEN}===== Planner completed task ====={Colors.END}")
    return Command(
        update={
            "messages": [get_message_from_string(role="user", string=full_response, imgs=[])],
            "messages_name": "planner",
            "full_plan": full_response,
            "history": history
        },
        goto=goto,
    )

def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info(f"{Colors.GREEN}===== Coordinator talking...... ====={Colors.END}")
    
    prompt_cache, cache_type = AGENT_PROMPT_CACHE_MAP["coordinator"]
    if prompt_cache: logger.debug(f"{Colors.GREEN}Coordinator - Prompt Cache Enabled{Colors.END}")
    else: logger.debug(f"{Colors.GREEN}Coordinator - Prompt Cache Disabled{Colors.END}")
    system_prompts, messages = apply_prompt_template("coordinator", state, prompt_cache=prompt_cache, cache_type=cache_type)
    llm = get_llm_by_type(AGENT_LLM_MAP["coordinator"])    
    llm.stream = True
    llm_caller = llm_call(llm=llm, verbose=False, tracking=False)
    if AGENT_LLM_MAP["coordinator"] in ["reasoning"]: enable_reasoning = True
    
    response, ai_message = llm_caller.invoke(
        agent_name="coordinator",
        messages=messages,
        system_prompts=system_prompts,
        enable_reasoning=False,
        reasoning_budget_tokens=8192
    )
    
    logger.debug(f"\n{Colors.RED}Current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Coordinator response:\n{pprint.pformat(response, indent=2, width=100)}{Colors.END}")

    goto = "__end__"
    if "handoff_to_planner" in response["text"]: goto = "planner"

    history = state.get("history", [])
    history.append({"agent":"coordinator", "message": response["text"]})

    logger.info(f"{Colors.GREEN}===== Coordinator completed task ====={Colors.END}")
    return Command(
        update={"history": history},
        goto=goto,
    )

def reporter_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    """Reporter node that write a final report."""
    logger.info(f"{Colors.GREEN}===== Reporter write final report ====={Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter11 - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")

    reporter_agent = create_react_agent(agent_name="reporter")
    result = reporter_agent.invoke(state=state)
    
    # 방어적으로 result 구조 확인
    if not result or "content" not in result or not result["content"]:
        logger.error(f"{Colors.RED}Reporter agent returned invalid result: {result}{Colors.END}")
        error_message = "Reporter agent failed to generate a proper response."
        history = state.get("history", [])
        history.append({"agent":"reporter", "message": error_message})
        return Command(
            update={
                "messages": [get_message_from_string(role="user", string=RESPONSE_FORMAT.format("reporter", error_message), imgs=[])],
                "messages_name": "reporter",
                "clues": state.get("clues", ""),
                "history": history
            },
            goto="supervisor",
        )
    
    # 안전하게 텍스트 추출 - 개선된 로직
    try:
        # content 배열에서 text가 있는 항목들을 모두 수집
        text_parts = []
        for content_item in result["content"]:
            if isinstance(content_item, dict) and "text" in content_item:
                text_parts.append(content_item["text"])
        
        if text_parts:
            full_response = "\n".join(text_parts)
        else:
            # text가 없으면 도구 사용 완료 메시지 생성
            tool_names = []
            for content_item in result["content"]:
                if isinstance(content_item, dict) and "toolUse" in content_item:
                    tool_names.append(content_item["toolUse"]["name"])
            
            if tool_names:
                full_response = f"Reporter agent completed successfully using tools: {', '.join(tool_names)}"
            else:
                full_response = "Reporter agent completed task successfully."
                
    except (IndexError, KeyError, TypeError) as e:
        logger.error(f"{Colors.RED}Failed to extract text from reporter result: {result}, Error: {e}{Colors.END}")
        full_response = "Reporter agent response could not be processed properly."

    clues = state.get("clues", "")
    clues = '\n\n'.join([clues, CLUES_FORMAT.format("reporter", full_response)])

    logger.debug(f"\n{Colors.RED}Reporter - current state messages:\n{pprint.pformat(state['messages'], indent=2, width=100)}{Colors.END}")
    logger.debug(f"\n{Colors.RED}Reporter response:\n{pprint.pformat(full_response, indent=2, width=100)}{Colors.END}")

    # 레포트 완성 여부 확인 - 새로운 로직
    report_completed = False
    final_files_generated = []
    
    # 보고서 생성 로직 개선
    try:
        # artifacts 디렉토리 생성
        os.makedirs("./artifacts", exist_ok=True)
        
        # all_results.txt 파일 확인 및 처리
        results_file = './artifacts/all_results.txt'
        analyses = []
        
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 분석 결과 블록 분리
            analysis_blocks = content.split("==================================================")
            
            for block in analysis_blocks:
                if not block.strip():
                    continue
                    
                # 분석 이름 추출
                import re
                analysis_name_match = re.search(r'## Analysis Stage: (.*?)$', block, re.MULTILINE)
                analysis_name = analysis_name_match.group(1) if analysis_name_match else "No analysis name"
                
                # 실행 시간 추출
                time_match = re.search(r'## Execution Time: (.*?)$', block, re.MULTILINE)
                execution_time = time_match.group(1) if time_match else "No time information"
                
                # 결과 설명 추출
                results_section = block.split("Result Description:", 1)
                results_text = results_section[1].split("--------------------------------------------------", 1)[0].strip() if len(results_section) > 1 else ""
                
                # 아티팩트 추출
                artifacts = []
                artifacts_section = block.split("Generated Files:", 1)
                if len(artifacts_section) > 1:
                    artifacts_text = artifacts_section[1]
                    artifact_lines = re.findall(r'- (.*?) : (.*?)$', artifacts_text, re.MULTILINE)
                    artifacts = artifact_lines
                    
                analyses.append({
                    "name": analysis_name,
                    "time": execution_time,
                    "results": results_text,
                    "artifacts": artifacts
                })
        
        # research_data.txt 파일 확인 및 처리
        research_file = './artifacts/research_data.txt'
        research_data = []
        
        if os.path.exists(research_file):
            with open(research_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 연구 데이터 블록 분리
            research_blocks = content.split("=" * 80)
            
            for block in research_blocks:
                if not block.strip():
                    continue
                    
                # 연구 주제 추출
                topic_match = re.search(r'## 연구 주제: (.*?)$', block, re.MULTILINE)
                topic = topic_match.group(1) if topic_match else "No topic"
                
                # 카테고리 추출
                category_match = re.search(r'## 카테고리: (.*?)$', block, re.MULTILINE)
                category = category_match.group(1) if category_match else "No category"
                
                # 수집 시간 추출
                time_match = re.search(r'## 수집 시간: (.*?)$', block, re.MULTILINE)
                collection_time = time_match.group(1) if time_match else "No time information"
                
                # 연구 내용 추출
                content_section = block.split("### 연구 내용:", 1)
                research_content = ""
                if len(content_section) > 1:
                    research_content = content_section[1].split("### 출처:", 1)[0].strip()
                
                # 출처 추출
                sources = []
                sources_section = block.split("### 출처:", 1)
                if len(sources_section) > 1:
                    sources_text = sources_section[1].split("-" * 80, 1)[0].strip()
                    source_lines = re.findall(r'\d+\. (.*?)$', sources_text, re.MULTILINE)
                    sources = source_lines
                    
                research_data.append({
                    "topic": topic,
                    "category": category,
                    "time": collection_time,
                    "content": research_content,
                    "sources": sources
                })
        
        # HTML 보고서 생성 (항상 생성)
        html_file_path = './report.html'
        
        # 한국어 컨텐츠 확인 함수
        def is_korean_content(content):
            korean_chars = sum(1 for char in content if '\uAC00' <= char <= '\uD7A3')
            return korean_chars > len(content) * 0.1
        
        # HTML 컨텐츠 생성
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>분석 보고서</title>
    <style>
        body {{
            font-family: 'Nanum Gothic', sans-serif;
            margin: 2cm;
            line-height: 1.5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 20px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 15px;
        }}
        .content {{
            margin-top: 20px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
        }}
        .image-caption {{
            text-align: center;
            font-style: italic;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .executive-summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 5px solid #3498db;
            margin: 20px 0;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .research-section {{
            background-color: #f0f8ff;
            padding: 15px;
            border-left: 5px solid #1e90ff;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>분석 보고서</h1>
    
    <div class="executive-summary">
        <h2>Executive Summary</h2>
        <p>{full_response[:500]}...</p>
    </div>
"""
        
        # 연구 데이터 섹션 추가
        if research_data:
            html_content += """
    <h2>연구 데이터 및 배경 정보</h2>
"""
            for research in research_data:
                html_content += f"""
    <div class="research-section">
        <h3>{research['topic']}</h3>
        <p><strong>카테고리:</strong> {research['category']}</p>
        <p><strong>수집 시간:</strong> {research['time']}</p>
        <div class="content">
            <pre>{research['content']}</pre>
        </div>
"""
                if research['sources']:
                    html_content += """
        <h4>출처:</h4>
        <ul>
"""
                    for source in research['sources']:
                        html_content += f"            <li>{source}</li>\n"
                    html_content += """
        </ul>
"""
                html_content += """
    </div>
"""
        
        # 각 분석 단계별 결과 추가
        html_content += """
    <h2>상세 분석 결과</h2>
"""
        for analysis in analyses:
            html_content += f"""
    <h3>{analysis['name']}</h3>
    <p><strong>실행 시간:</strong> {analysis['time']}</p>
    <div class="content">
        <pre>{analysis['results']}</pre>
    </div>
"""
            
            # 아티팩트 이미지 추가
            for artifact_path, artifact_desc in analysis['artifacts']:
                if artifact_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    html_content += f"""
    <div class="figure">
        <img src="{artifact_path}" alt="{artifact_desc}">
        <div class="image-caption">{artifact_desc}</div>
    </div>
"""
        
        html_content += """
    <h2>결론</h2>
    <div class="content">
        <p>본 분석을 통해 도출된 주요 결과와 시사점을 종합하면 다음과 같습니다.</p>
    </div>
</body>
</html>"""
        
        # HTML 파일 저장
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"{Colors.GREEN}HTML report successfully generated: {html_file_path}{Colors.END}")
        final_files_generated.append(html_file_path)
        
        # 마크다운 파일도 생성
        md_file_path = './final_report.md'
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write("# 분석 보고서\n\n")
            f.write("## Executive Summary\n\n")
            f.write(full_response + "\n\n")
            
            # 연구 데이터 추가
            if research_data:
                f.write("## 연구 데이터 및 배경 정보\n\n")
                for research in research_data:
                    f.write(f"### {research['topic']}\n\n")
                    f.write(f"**카테고리:** {research['category']}\n\n")
                    f.write(f"**수집 시간:** {research['time']}\n\n")
                    f.write(f"{research['content']}\n\n")
                    
                    if research['sources']:
                        f.write("**출처:**\n")
                        for i, source in enumerate(research['sources'], 1):
                            f.write(f"{i}. {source}\n")
                        f.write("\n")
            
            # 각 분석 결과 추가
            f.write("## 상세 분석 결과\n\n")
            for analysis in analyses:
                f.write(f"### {analysis['name']}\n\n")
                f.write(f"**실행 시간:** {analysis['time']}\n\n")
                f.write(f"{analysis['results']}\n\n")
                
                # 이미지 파일 포함
                for artifact_path, artifact_desc in analysis['artifacts']:
                    if artifact_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        f.write(f"![{artifact_desc}]({artifact_path})\n\n")
                        f.write(f"*{artifact_desc}*\n\n")
        
        logger.info(f"{Colors.GREEN}Markdown report successfully generated: {md_file_path}{Colors.END}")
        final_files_generated.append(md_file_path)
        
        # PDF 생성 시도 (선택적)
        try:
            pdf_file_path = './final_report.pdf'
            
            # pandoc 명령어 선택
            if is_korean_content(full_response):
                pandoc_cmd = f'pandoc {md_file_path} -o {pdf_file_path} --pdf-engine=xelatex -V mainfont="NanumGothic" -V geometry="margin=0.5in"'
            else:
                pandoc_cmd = f'pandoc {md_file_path} -o {pdf_file_path} --pdf-engine=xelatex -V mainfont="Noto Sans" -V monofont="Noto Sans Mono" -V geometry="margin=0.5in"'
            
            # pandoc 실행
            import subprocess
            result = subprocess.run(pandoc_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"{Colors.GREEN}PDF report successfully generated: {pdf_file_path}{Colors.END}")
            final_files_generated.append(pdf_file_path)
            
        except Exception as e:
            logger.warning(f"{Colors.YELLOW}PDF generation failed, but HTML and Markdown reports were created: {str(e)}{Colors.END}")
        
        # 레포트 완성 확인
        if len(final_files_generated) >= 2:  # HTML과 Markdown이 최소한 생성되었으면
            report_completed = True
            
    except Exception as e:
        logger.error(f"{Colors.RED}Report generation failed: {str(e)}{Colors.END}")

    history = state.get("history", [])
    history.append({"agent":"reporter", "message": full_response})
    
    # 레포트가 완성되었는지 확인하여 워크플로우 종료 여부 결정
    if report_completed:
        logger.info(f"{Colors.GREEN}===== Final report completed successfully! Generated files: {', '.join(final_files_generated)} ====={Colors.END}")
        logger.info(f"{Colors.GREEN}===== Workflow ending ====={Colors.END}")
        
        # 워크플로우 완료 메시지를 state에 추가
        final_completion_message = f"Final report generation completed successfully. Generated files: {', '.join(final_files_generated)}. Workflow finished."
        
        return Command(
            update={
                "messages": [get_message_from_string(role="user", string=final_completion_message, imgs=[])],
                "messages_name": "reporter",
                "history": history,
                "clues": clues,
                "report_completed": True,
                "final_files": final_files_generated
            },
            goto="__end__"  # 직접 워크플로우 종료
        )
    else:
        logger.info(f"{Colors.GREEN}===== Reporter completed task, continuing workflow ====={Colors.END}")
        return Command(
            update={
                "messages": [get_message_from_string(role="user", string=full_response, imgs=[])],
                "messages_name": "reporter",
                "history": history,
                "clues": clues
            },
            goto="supervisor"
        )
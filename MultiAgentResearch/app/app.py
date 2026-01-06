import sys, os
module_path = ".."
sys.path.append(os.path.abspath(module_path))
os.environ['LLM_MODULE'] = 'src.agents.llm_st'

import streamlit as st
from main import execution
from src.config.agents import AGENT_LLM_MAP

##################### Title ########################
st.set_page_config(page_title="AI Agent for Market Intelligence💬", page_icon="💬", layout="wide")
st.title("AI Agent for Market Intelligence 💬")
st.markdown('''- This is multi-agent based AI Agent for Market Intelligence''')
st.markdown('''
            - test 및 upgrade 진행 중
            ''')

####################### Initialization ###############################
# Store the initial value of widgets in session state
if "messages" not in st.session_state: st.session_state["messages"] = []
if "history_ask" not in st.session_state: st.session_state["history_ask"] = []
if "history_answer" not in st.session_state: st.session_state["history_answer"] = []
if "ai_results" not in st.session_state: st.session_state["ai_results"] = {"coordinator": {}, "planner": {}, "supervisor": {}, "coder": {}, "reporter": {}}
if "current_agent" not in st.session_state: st.session_state["current_agent"] = ""
    
####################### Application ###############################
#if len(st.session_state["history_ask"]) > 0: display_chat_history()

if user_input := st.chat_input(): # 사용자 입력 받기
    st.chat_message("user").write(user_input)
    st.session_state["recent_ask"] = user_input
    
    node_names = ["coordinator", "planner", "supervisor", "coder", "reporter"]
    tool_node_names = ["coder", "reporter"]
    node_descriptions = {
        "coordinator": "전체 프로세스 조정 및 최종 응답 생성",
        "planner": "분석 계획 수립 및 작업 분배",
        "supervisor": "코드 및 결과물 검증",
        "coder": "데이터 처리 및 시각화 코드 작성",
        "reporter": "분석 결과 해석 및 보고서 작성"
    }
    
    # 응답 프로세스 시작
    with st.chat_message("assistant"):
        # 초기 메시지
        main_response = st.empty()
        main_response.write("분석 작업을 시작합니다...")        
        # 각 단계별 진행 상황을 표시할 expander 생성
        if "process_containers" not in st.session_state:
            st.session_state["process_containers"] = {}
            st.session_state["tool_containers"] = {}
            st.session_state["reasoning_containers"] = {}
            
        for node_name in node_names:
            with st.expander(f"🔄 {node_name.upper()}: {node_descriptions[node_name]}", expanded=True):
                
                # Create two columns: left for Agent message, right for Reasoning and Tool
                left_col, right_col = st.columns([1, 1])
                
                # Left column - Agent message
                with left_col:
                    st.markdown(f"💬 Agent message:")
                    st.session_state["process_containers"][node_name] = st.empty()
                    st.session_state["process_containers"][node_name].info(f"Waiting...")
                
                # Right column - Reasoning and Tool
                with right_col:
                    if AGENT_LLM_MAP[node_name] == "reasoning" and node_name != "supervisor":
                        st.markdown(f"🧠 Reasoning:")
                        st.session_state["reasoning_containers"][node_name] = st.empty()
                        st.session_state["reasoning_containers"][node_name].info(f"Reasoning not used yet")
                        st.markdown("---")
                    
                    # 에이전트가 사용하는 툴 결과를 표시할 컨테이너
                    if node_name in tool_node_names:
                        st.markdown(f"🔧 Tool message:")
                        st.markdown(f"  - Input:")
                        st.session_state["tool_containers"][node_name] = {}
                        st.session_state["tool_containers"][node_name]["input"] = st.empty()
                        st.markdown(f"  - Output:")
                        st.session_state["tool_containers"][node_name]["output"] = st.empty()
                        st.session_state["tool_containers"][node_name]["input"].info(f"Tool not used yet")
                        st.session_state["tool_containers"][node_name]["output"].info(f"Tool not used yet")
                
                st.markdown("---")  # Divider between agent sections
        
        with st.spinner('분석 중...'):
            # 실행 및 결과 처리
            exe_results = execution(user_query=user_input)

            last_message = ""
            for history in exe_results["history"]:
                st.session_state["process_containers"][history["agent"]].write(history["message"])
                last_message = history["message"]
            
            main_response.write(last_message)
        
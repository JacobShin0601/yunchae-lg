import os
import json
import streamlit as st
from app import AgentHelper
from src.opensearch import OpenSearchManager
from pprint import pprint
from langchain.callbacks.streamlit import StreamlitCallbackHandler

def load_allowed_users():
    with open("allowed_users.json", "r") as file:
        return json.load(file)

def initialize_tab_state(tab_name):
    """특정 탭의 세션 상태를 초기화합니다."""
    if tab_name not in st.session_state:
        st.session_state[tab_name] = {"initialized": False, "messages": [], "user_input": ""}

def reset_session_state(keep_keys=None):
    """지정된 키를 제외한 모든 세션 상태를 초기화합니다."""
    if keep_keys is None:
        keep_keys = []

    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]

def reset_chat_state():
    """Reset chat-related session state when database changes."""
    st.session_state["tab2"]["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
    st.session_state["tab2"]["user_input"] = ""

def create_user_directories(user_name, databases):
    """사용자별 데이터베이스 디렉토리를 생성합니다."""
    updated_databases = [f"{user_name}__{db}" for db in databases]

    for db in updated_databases:
        os.makedirs(f"./data/{db}", exist_ok=True)
    for db in updated_databases:
        os.makedirs(f"./preprocessed_data/{db}", exist_ok=True)

    return updated_databases

def main():
    st.set_page_config(layout="wide")

    st.title("KOALA_ai 🐨✨")
    st.markdown("""
    <p style='font-size: 12px; color: gray;'>
    <sub>KOALA_ai: Knowledge Orchestration and Adaptive Learning Agent</sub>
    </p>
    """, unsafe_allow_html=True)
    # st.markdown("""
    # <p style='background-color: #f0f0f0; padding: 5px;'>
    # KOALA_ai: Knowledge Orchestration and Adaptive Learning Agent
    # </p>
    # """, unsafe_allow_html=True)    

    st.markdown("""
    <h2 style='margin-left: 28px;'> designed to streamline your research process,
    <br>saving you valuable time 🚀</h2>""", unsafe_allow_html=True)

    st.markdown(" - - - ")

    # 사용자 인증 입력
    st.sidebar.header("User Authentication")
    if "authenticated" not in st.session_state:
        user_name = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        st.sidebar.markdown(" - - - ")

        allowed_users = load_allowed_users()

        if st.sidebar.button("Login"):
            if user_name in allowed_users and password == allowed_users[user_name]:
                st.session_state["authenticated"] = True
                st.session_state["user_name"] = user_name
                st.sidebar.success(f"Welcome, {user_name}!")

                # 로그인 후 폴더 생성 로직 실행
                DATABASES = ["database1", "database2", "database3", "database4", "database5"]
                st.session_state["updated_databases"] = create_user_directories(user_name, DATABASES)

    else:
        st.sidebar.success(f"Welcome back, {st.session_state['user_name']}!")

    if st.session_state.get("authenticated"):
        # 주요 로직 실행
        st.sidebar.header("Upload Document")
        selected_db = st.sidebar.selectbox("Select Database", st.session_state["updated_databases"])
        uploaded_file = st.sidebar.file_uploader("Select file", type=["pdf"])

        if uploaded_file is not None:
            if uploaded_file.name in os.listdir(f"./data/{selected_db}"):
                st.sidebar.warning(f"{uploaded_file.name} already exists in the selected database")
            else:
                with st.spinner("Starting the process..."):
                    ah = AgentHelper(None, selected_db)
                    temp_file_path = ah.handle_file_upload(uploaded_file, selected_db)
                    ah.process_pdf(temp_file_path, uploaded_file, selected_db)

                    st.success(f"Finished processing 😊")

                    # 인증 정보는 유지하고 나머지 세션 상태 초기화
                    reset_session_state(keep_keys=["authenticated", "user_name", "updated_databases"])

                    st.warning("Uploaded file would be facilitated if you refresh the page")
                    st.stop()  # 새로 고침이 이루어질 때까지 실행 중지

        tab1, tab2 = st.tabs(["📄 Database Overview", " ✨ Chat with Agent"])

        with tab1:
            initialize_tab_state("tab1")
            
            st.header("Database Overview")
            selected_db_tab1 = st.selectbox(
                "Select Database to View Files", st.session_state["updated_databases"], key="tab1_db"
            )

            # 선택된 데이터베이스가 변경될 때, chat state 초기화
            if st.session_state.get("previous_selected_db") != selected_db_tab1:
                initialize_tab_state("tab2")  # Ensure tab2 is initialized before resetting
                reset_chat_state()
                st.session_state["previous_selected_db"] = selected_db_tab1

            st.markdown(f"Files in {selected_db_tab1}:")

            files_in_db = os.listdir(f"./data/{selected_db_tab1}")
            if files_in_db:
                for file in files_in_db:
                    st.write(f"- {file}")
            else:
                st.write("No files uploaded yet.")

        with tab2:
            # Initialize or reset tab2 state
            initialize_tab_state("tab2")

            # OpenSearchManager: index_name, user_name, region
            osm = OpenSearchManager(index_name=selected_db_tab1, region='us-east-1')
            osm.connect_to_opensearch()
            
            # AgentHelper: os_client, index_name
            ah = AgentHelper(os_client=osm.os_client, index_name=selected_db_tab1)

            # Display previous messages
            for msg in st.session_state["tab2"]["messages"]:
                if msg["role"] == "assistant_context":
                    with st.chat_message("assistant"):
                        st.write("Let me search...")
                else:
                    st.chat_message(msg["role"]).write(msg["content"])

            # User input at the bottom
            query = st.chat_input("Write down your question:")

            if query: 
                with st.spinner("Processing..."):
                    if query.strip() == "":
                        st.warning('Query is empty, please retry!')
                        st.stop()

                    # Save user input to session
                    st.session_state["tab2"]["messages"].append(
                        {"role": "user", "content": query}
                    )

                    # Display user input
                    with st.chat_message("user"):
                        st.write(query)

                    # Streamlit callback handler setup
                    st_cb = StreamlitCallbackHandler(
                        st.chat_message("assistant"), collapse_completed_thoughts=True
                    )

                    # Invoke function to process the query
                    response = ah.invoke(query=query, 
                                        streaming_callback=st_cb,
                                        alpha=0.7, 
                                        hyde=True, 
                                        reranker=True, 
                                        parent=True, 
                                        ragfusion=False)
                    
                    # Extract response details
                    answer = response[0]
                    contexts = response[1]
                    mid_answer = response[2] if len(response) > 2 else None

                    # Display answer of assistant
                    with st.chat_message("assistant"):
                        st.write(answer)
                        
                    # Display assistant answer
                    with st.chat_message("assistant"):
                        ah.show_similar_docs(contexts, answer)

                    # Save responses to session
                    st.session_state["tab2"]["messages"].append({"role": "assistant", "content": answer})
                    st.session_state["tab2"]["messages"].append(
                        {"role": "assistant_context", "content": contexts}
                    )

                    # Complete current thought
                    st_cb._complete_current_thought()


    else:
        st.warning("Please enter your username and password to proceed.")

if __name__ == "__main__":
    main()
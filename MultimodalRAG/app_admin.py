import os
import json
import boto3
import logging
import shutil
import streamlit as st
from src.local_utils.opensearch import opensearch_utils
from src.opensearch import OpenSearchManager
from app.AgentHelper import AgentHelper
from app.AccountManager import AccountManager

# 로깅 설정
logging.basicConfig(
    filename='app.log',  # 로그를 저장할 파일 이름
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='w'  # 파일 모드를 'w'로 설정하면 파일을 덮어쓰고, 'a'로 설정하면 추가 기록
)

# AWS Parameter Store에서 값을 가져오는 함수
def get_parameter(param_name, with_decryption=True):
    ssm = boto3.client('ssm', region_name='us-east-1')  # 적절한 리전으로 변경
    response = ssm.get_parameter(Name=param_name, WithDecryption=with_decryption)
    return response['Parameter']['Value']

def load_allowed_users():
    with open("allowed_users.json", "r") as file:
        return json.load(file)

def delete_database(user, database):
    logging.info(f"Starting deletion process for {user}__{database}")
    user_database = f"{user}__{database}"
    
    # 로컬 파일 시스템에서 데이터베이스 폴더 삭제
    data_path = f"./data/{user_database}"
    preprocessed_data_path = f"./preprocessed_data/{user_database}"

    # 데이터 폴더 내용 표시
    if os.path.exists(data_path):
        logging.info(f"Contents of {data_path}:")
        for root, dirs, files in os.walk(data_path):
            for name in files:
                logging.info(f"File: {os.path.join(root, name)}")
            for name in dirs:
                logging.info(f"Directory: {os.path.join(root, name)}")
    else:
        logging.warning(f"Data path '{data_path}' does not exist.")

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        logging.info(f"Deleted data path: {data_path}")
    else:
        raise ValueError(f"Data path '{data_path}' does not exist.")
    
    if os.path.exists(preprocessed_data_path):
        shutil.rmtree(preprocessed_data_path)
        logging.info(f"Deleted preprocessed data path: {preprocessed_data_path}")
    else:
        raise ValueError(f"Preprocessed data path '{preprocessed_data_path}' does not exist.")

    # OpenSearch 인덱스 삭제
    osm = OpenSearchManager(index_name=user_database, region='us-east-1')
    osm.connect_to_opensearch()
    os_client = osm.os_client

    if not os_client:
        raise ValueError("OpenSearch client is not connected. Call connect_to_opensearch() first.")

    ah = AgentHelper(os_client=os_client, index_name=user_database)

    index_exists = opensearch_utils.check_if_index_exists(os_client, user_database)

    if index_exists:
        opensearch_utils.delete_index(os_client, user_database)
        logging.info(f"Deleted OpenSearch index: {user_database}")
    else:
        logging.warning(f"Index '{user_database}' does not exist in OpenSearch.")


def main():
    st.set_page_config(layout="centered")
    st.title("KOALA_ai Admin Dashboard 🛠️")
    st.header("Manage Users and System Configuration")
    st.markdown("This dashboard is for administrators only. Please handle with care.")

    # 관리자 인증
    try:
        admin_user = "koala_admin"
        admin_password = get_parameter(admin_user)
    except Exception as e:
        st.error("Error fetching admin credentials from Parameter Store.")
        st.stop()

    # 세션 상태에서 로그인 상태 확인
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    with st.sidebar:
        if not st.session_state['logged_in']:
            st.sidebar.header("Admin Login")
            input_user = st.text_input("Admin Username")
            input_password = st.text_input("Admin Password", type="password")
            if st.button("Login"):
                if input_user == admin_user and input_password == admin_password:
                    st.session_state['logged_in'] = True
                    st.session_state['page'] = "Create New User"  # 기본 페이지 설정
                    st.success(f"Welcome, {admin_user}!")
                else:
                    st.error("Invalid admin credentials. Please try again.")
        else:
            st.sidebar.success(f"Logged in as {admin_user}")
            st.session_state['page'] = st.sidebar.selectbox(
                "Select a page:",
                ["Create New User", "Delete User", "View All Users", "Database Management"]
            )
            if st.sidebar.button("Logout"):
                st.session_state['logged_in'] = False
                st.experimental_rerun()

    if st.session_state['logged_in']:
        st.sidebar.success(f"Logged in as {admin_user}")
        st.session_state['page'] = st.sidebar.selectbox(
            "Select a function:",
            ["Create New User", "Delete User", "View All Users", "Database Management"]
        )

        osm = OpenSearchManager(index_name='test_user__database4', region='us-east-1')
        osm.connect_to_opensearch()
        os_client=osm.os_client

        acount_manager = AccountManager()

        page = st.session_state.get('page', "Create New User")  # 기본 페이지 설정

        if page == "Create New User":
            st.subheader("Create New User")
            new_user_name = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")

            if st.button("Create User"):
                # 빈 입력 필드 검사
                if not new_user_name or not new_password:
                    st.error("Both Username and Password fields must be filled out.")
                else:
                    try:
                        account_manager.create_user(new_user_name, new_password)
                        st.success(f"User {new_user_name} created successfully!")
                    except ValueError as e:
                        st.error(str(e))

        elif page == "Delete User":
            st.subheader("Delete User")
            delete_user_name = st.text_input("Username to Delete")
            if st.button("Delete User"):
                try:
                    account_manager.delete_user(delete_user_name)
                    st.success(f"User {delete_user_name} deleted successfully!")
                except ValueError as e:
                    st.error(str(e))

        elif page == "View All Users":
            st.subheader("View All Users")
            if st.button("Show All Users"):
                users = account_manager.load_allowed_users()
                if users:
                    st.write("### Registered Users")
                    for user in users.keys():
                        st.write(f"- {user}")
                else:
                    st.write("No users found.")

        elif page == "Database Management":
            st.subheader("Database Management")

            allowed_users = load_allowed_users()
            selected_user = st.selectbox("Select User", list(allowed_users.keys()))
            databases = ["database1", "database2", "database3", "database4", "database5"]
            selected_database = st.selectbox("Select Database", databases)

            user_database = f"{selected_user}__{selected_database}"
            st.write(f"Selected User Database: `{user_database}`")

            if st.button("Delete Database"):
                st.write(f"Attempting to delete database: {user_database}")
                delete_database(selected_user, selected_database)
                st.success(f"Database `{user_database}` has been deleted successfully!")
                # confirm = st.radio("Are you sure you want to delete this database?", options=("No", "Yes"), index=None)
                
                # if confirm is None:
                #     st.warning("Please select Yes or No to proceed.")
                # elif confirm == "Yes":
                #     st.write("select yes")
                #     try:
                #         st.write(f"Attempting to delete database: {user_database}")  # 직접 출력
                #         logging.info(f"Attempting to delete database: {user_database}")
                #         delete_database(selected_user, selected_database)
                #         st.success(f"Database `{user_database}` has been deleted successfully!")
                #     except ValueError as e:
                #         st.write(f"Error during deletion: {str(e)}")  # 직접 출력
                #         logging.error(f"Error during deletion: {str(e)}")
                #         st.error(str(e))
                # elif confirm == "No":
                #     st.write("Database deletion was canceled.")
                #     logging.info("Database deletion was canceled.")

        else:
            st.warning("Please log in with your admin credentials to access this dashboard.")

if __name__ == "__main__":
    main()
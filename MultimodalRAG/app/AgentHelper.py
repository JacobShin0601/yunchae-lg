import os
import sys
import json
import boto3
import botocore
import base64
import re
from glob import glob
import logging
import streamlit as st
import copy
from typing import List, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection
import numpy as np
from scipy.stats import norm
from langchain.callbacks import StreamlitCallbackHandler
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from src.local_utils.common_utils import print_html, retry, load_json
from src.local_utils.opensearch import opensearch_utils
from src.local_utils.ssm import parameter_store
from src.local_utils.rag_streamlit import (
    qa_chain,
    prompt_repo,
    OpenSearchHybridSearchRetriever,
)
from src.preprocessing import LLMLoader
from src.opensearch import OpenSearchManager
from app.ElementParser import ElementParser
from app.PipelineManager import PipelineManager

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 콘솔에 로그 메시지를 출력하도록 설정
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 로그 포맷 설정
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# 핸들러를 로거에 추가
logger.addHandler(ch)


class AgentHelper:
    def __init__(self, os_client, index_name):
        self.os_client = os_client
        self.index_name = index_name
        self.database = index_name

        self.llm_loader = LLMLoader(regions=["us-east-1"])
        self.llm_text = self.llm_loader._configure_llm_text()
        self.llm_emb = self.llm_loader._configure_llm_emb()
        self.region = "us-east-1"
        self.pm = parameter_store(self.region)

        # 각 데이터베이스에 맞는 경로 설정
        self.base_data_dir = "./data"
        self.base_output_dir = "./preprocessed_data"
        self.data_dir = os.path.join(self.base_data_dir, self.database)
        self.output_dir = os.path.join(
            self.base_output_dir, self.database
        )
        
        # 각 경로가 존재하지 않으면 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.all_input_files = glob(os.path.join(self.data_dir, "*"))
        self.all_preprocessed_files = glob(
            os.path.join(self.output_dir, "*")
        )

        self.ElementParser = ElementParser(database=self.database)
        self.all_elements_file = f"all_combined_json__{self.database}.json"
        
        # database 인자를 전달하여 save_combined_json 호출
        self.ElementParser.save_combined_json(output_file=self.all_elements_file)
        
        self.all_elements = load_json(os.path.join(self.output_dir, self.all_elements_file))
        self.matched_element = ""

        # 데이터 경로 설정
        self.allowed_users_file = "allowed_users.json"

        # 로컬 JSON 파일로부터 허가된 사용자 정보를 로드
        self.allowed_users = self.load_allowed_users()

        logging.info(f"AgentHelper initialized: {self.os_client}")
        logging.info(f"Index Name initialized: {self.index_name}")
    ############# 1. OpenSearch 관련 메서드들 ################
    def search_documents(self, query):
        """OpenSearch에서 주어진 쿼리에 해당하는 문서를 검색"""
        search_results = self.os_client.search(
            index=self.index_name, body={"query": {"match": {"text": query}}}
        )
        documents = [hit["_source"] for hit in search_results["hits"]["hits"]]
        logger.debug(f"Search results: {search_results}")
        logger.debug(f"Extracted documents: {documents}")
        return documents

    def get_opensearch_client(self):
        """OpenSearch 클라이언트를 생성하여 반환"""
        opensearch_domain_endpoint = self.pm.get_params(
            key="opensearch_domain_endpoint", enc=False
        )
        opensearch_user_id = self.pm.get_params(key="opensearch_user_id", enc=False)
        opensearch_user_password = self.pm.get_params(
            key="opensearch_user_password", enc=True
        )
        http_auth = (opensearch_user_id, opensearch_user_password)
        os_client = opensearch_utils.create_aws_opensearch_client(
            self.region, opensearch_domain_endpoint, http_auth
        )
        return os_client

    def get_retriever(
        self, streaming_callback, parent, reranker, hyde, ragfusion, alpha
    ):
        """OpenSearch와 연동된 Hybrid Search Retriever를 생성"""
        os_client = self.get_opensearch_client()
        llm_text = self.get_llm(streaming_callback)
        llm_emb = self.get_embedding_model()
        reranker_endpoint_name = self.pm.get_params(key="reranker_endpoint", enc=False)
        opensearch_hybrid_retriever = OpenSearchHybridSearchRetriever(
            os_client=os_client,
            index_name=self.index_name,
            llm_text=llm_text,
            llm_emb=llm_emb,
            minimum_should_match=0,
            filter=[],
            fusion_algorithm="RRF",
            complex_doc=True,
            ensemble_weights=[alpha, 1.0 - alpha],
            reranker=reranker,
            reranker_endpoint_name=reranker_endpoint_name,
            parent_document=parent,
            rag_fusion=ragfusion,
            rag_fusion_prompt=prompt_repo.get_rag_fusion(),
            hyde=hyde,
            hyde_query=["web_search"],
            query_augmentation_size=3,
            async_mode=True,
            k=5,
            verbose=True,
        )
        return opensearch_hybrid_retriever

    ############### 2. LLM 관련 메서드들 ################
    def get_llm(self, streaming_callback):
        """LLM 모델을 생성하여 반환"""
        boto3_bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region,
        )
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        model_kwargs = {
            "max_tokens": 199999,
            "stop_sequences": ["\n\nHuman"],
            "temperature": 0.01,
            "top_p": 0.9,
        }
        llm = ChatBedrock(
            client=boto3_bedrock,
            model_id=model_id,
            model_kwargs=model_kwargs,
            streaming=True,
            callbacks=[streaming_callback],
        )
        return llm

    def get_simple_bedrock(self, **kwargs):
        """간단한 Bedrock LLM 모델을 생성하여 반환"""
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region,
        )
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"  # Claude 3 Haiku
        model_kwargs = {"max_tokens": 100000, "temperature": 0.01, "top_p": 0.9}
        llm = ChatBedrock(
            client=bedrock_runtime,
            model_id=model_id,
            model_kwargs=model_kwargs,
        )
        return llm

    def get_embedding_model(self):
        """텍스트 임베딩 모델을 생성하여 반환"""
        llm_emb = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
        return llm_emb

    def generate_response(self, query, documents):
        """LLM을 사용하여 주어진 문서들로부터 응답을 생성"""
        logger.debug(f"Query: {query}")
        logger.debug(f"Documents: {documents}")

        if isinstance(documents, str):
            raise ValueError("Documents should be a list of objects, not a string.")

        synthesized_response = self.llm_text(
            f"Given the following documents: {documents}, answer the following query: {query}"
        )
        logger.debug(f"Synthesized Response: {synthesized_response}")
        return synthesized_response

    @retry(
        total_try_cnt=10,
        sleep_in_sec=20,
        retryable_exceptions=(
            botocore.exceptions.EventStreamError,
            botocore.exceptions.ReadTimeoutError,
        ),
    )
    def check_relevance(self, simple_llm, text: str, subject: str) -> str:
        """LLM을 사용하여 주어진 텍스트가 특정 주제와 관련이 있는지 확인"""
        template = """
        <paper>
        {text}
        </paper>
        
        [Task instructions]
        here is a paper above. 
        read the paper and determine whether the content is related to the {subject}. 
        If the content is related to the {subject}, please **just** say ‘yes’,
        IF the content is not related to the {subject}, please **just** say ‘no’. 
        You must follow this format. 
        
        result: ‘yes’ 
        result: ‘no’ 

        result: 
        """
        prompt = PromptTemplate(template=template, input_variables=["subject", "text"])
        chain = prompt | simple_llm
        intend = chain.invoke({"subject": subject, "text": text}).content
        return intend.lower().replace(" ", "").replace("result:", "")

    ################ 3. 문서 및 응답 처리 메서드들 ################
    def get_response(self, query):
        """검색 및 응답 생성을 결합하여 최종 응답을 반환"""
        documents = self.search_documents(query)
        response = self.generate_response(query, documents)
        return response

    def _find_matching_element(self, meta_dict, all_combined_json):
        """
        This function returns the dictionary from all_combined that matches the element_id in meta_dict.

        Parameters:
        meta_dict (dict): A dictionary containing the element_id under the 'element_id' key.
        all_combined (list): A list of dictionaries, each containing an 'element_id' key.

        Returns:
        dict: The dictionary from all_combined that matches the element_id in meta_dict. Returns None if no match is found.
        """
        # with open('test_meta_dict.json', 'w', encoding='utf-8') as f:
        #     json.dump(meta_dict, f, ensure_ascii=False, indent=4)
        # with open('test_comb.json', 'w', encoding='utf-8') as f:
        #     json.dump(all_combined_json, f, ensure_ascii=False, indent=4)
        # Extract element_id from meta_dict
        element_id = meta_dict["element_id"]

        if not element_id:
            raise ValueError("meta_dict does not contain 'element_id'.")

        # Search for the dictionary in all_combined that matches the element_id
        for item in all_combined_json:
            if item["element_id"] == element_id:
                return item
        # Return None if no matching dictionary is found
        return None

    def show_similar_docs(self, contexts, answer):
        """유사 문서와 그 관련성을 화면에 표시"""
        if not contexts:
            st.write("No documents found.")
            return

        contexts = contexts[0]
        scores = [context["score"] for context in contexts]
        if not scores:
            st.write("No scores available.")
            return

        mean_scores = np.mean(scores)
        std_scores = np.std(scores)
        z_scores = (scores - mean_scores) / std_scores
        scaled_scores = norm.cdf(z_scores) * 100

        st.write("**Reference** ⬇️")
        for idx, context in enumerate(contexts):
            self.matched_element = ""
            self.matched_element = self._find_matching_element(
                context["meta"], self.all_elements
            )
            # with open('matched_element.json', 'w', encoding='utf-8') as f:
            #     json.dump(self.matched_element, f, ensure_ascii=False, indent=4)
            filename = self.matched_element["metadata"]["filename"]

            if ".pdf" in filename:
                context_lines = "\n".join(context)
                result = self.check_relevance(
                    self.get_simple_bedrock(), context_lines, answer
                )
                # st.write(result)
                if result == "yes":
                    file_info = self.extract_file_info(self.matched_element)
                    with st.expander(f"{file_info}"):
                        st.markdown(
                            "##### `Relevance Score`: {:.2f}%".format(
                                scaled_scores[idx]
                            )
                        )
                        for line in context["lines"]:
                            st.write(line)
                        # st.write(context["meta"])
                        self.parse_metadata(
                            context["meta"], matched_element=self.matched_element
                        )
            else:
                continue
        st.markdown(" - - - ")

    def formatting_output(self, contexts):
        """LLM 응답 컨텍스트를 포맷팅"""
        formatted_contexts = []
        for doc, score in contexts:
            lines = doc.page_content.split("\n")
            metadata = doc.metadata
            formatted_contexts.append((score, lines))
        return formatted_contexts

    def invoke(
        self, query, streaming_callback, parent, reranker, hyde, ragfusion, alpha
    ):
        """질의를 처리하고 결과를 반환"""
        llm_text = self.get_llm(streaming_callback)
        opensearch_hybrid_retriever = self.get_retriever(
            streaming_callback, parent, reranker, hyde, ragfusion, alpha
        )
        system_prompt = prompt_repo.get_system_prompt()

        qa = qa_chain(
            llm_text=llm_text,
            retriever=opensearch_hybrid_retriever,
            system_prompt=system_prompt,
            return_context=False,
            verbose=False,
        )

        response, pretty_contexts, retrieval, augmentation = qa.invoke(
            query=query, complex_doc=True
        )

        return response, pretty_contexts, retrieval, augmentation

    def handle_query(self, query, st_cb, **kwargs):
        """사용자 질의를 처리하고 결과를 화면에 표시"""
        answer, contexts = self.invoke(query, st_cb, **kwargs)
        with st.chat_message("assistant"):
            self.show_similar_docs(contexts, answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.messages.append(
            {"role": "assistant_context", "content": contexts}
        )
        st_cb._complete_current_thought()

    ################# 4. 파일 및 메타데이터 처리 메서드들 ################
    def parse_image(self, metadata):
        """메타데이터에서 Base64 인코딩된 이미지를 디코딩하여 표시"""
        if "image_base64" in metadata:
            st.image(base64.b64decode(metadata["image_base64"]))

    def parse_html(self, metadata):
        """메타데이터에서 HTML 문자열을 추출하여 표시"""
        html = metadata.get("text_as_html", None)
        if html:
            st.markdown(html, unsafe_allow_html=True)

    def extract_file_info(self, matched_element):
        """메타데이터 경로에서 파일 정보를 추출"""
        filename = matched_element["metadata"]["filename"]
        if ".pdf" in filename:
            return self._extract_file_info(matched_element=matched_element)
        # elif ".xlsx" in filename:
        #     return self._extract_file_info_from_table(metapath) + ".xlsx"
        else:
            return "Unknown file info"

    def _extract_file_info(self, matched_element):
        """이미지 경로에서 파일 정보를 추출"""
        # match = re.match(r".*/(figure|table)-(.+)-p(\d+)-ele\d+\.jpg", metapath)
        # if match:
        #     file_type, filename, page_number = match.groups()
        #     filename = filename.replace("_", " ").replace("-", " ")
        file_type = matched_element["type"]
        page_number = matched_element["metadata"]["page_number"]
        filename = matched_element["metadata"]["filename"]
        return f"Type: {file_type} || Page: {page_number} || File name: {filename}"
        # return "Unknown file info"

    def parse_metadata(self, metadata, matched_element):
        """메타데이터를 파싱하여 적절한 컨텐츠를 표시"""
        if matched_element["type"] == "Image":
            self.parse_image(metadata)
        elif matched_element["type"] == "Table":
            try:
                self.parse_image(metadata)
            except:
                self.parse_html(matched_element)
        else:
            st.write("No images or table to display in this reference")

    ################ 5. Streamlit App 관련 함수들 ################
    def handle_file_upload(self, uploaded_file, selected_db):
        temp_file_path = f"data/{selected_db}/{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_file_path

    def process_pdf(self, temp_file_path, uploaded_file, database):
        if st.session_state.get(uploaded_file.name):
            st.info("This PDF has already been processed.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        pipeline_manager = PipelineManager(
            filename=uploaded_file.name, database=database
        )
        pipeline_manager.run_pipeline(
            steps=["preprocess"], progress_bar=progress_bar, status_text=status_text
        )
        pipeline_manager.run_pipeline(
            steps=["opensearch"], progress_bar=progress_bar, status_text=status_text
        )

        st.session_state[uploaded_file.name] = True
        st.success("PDF processing and indexing completed😊")

    def connect_opensearch(self, index_name, region="us-east-1"):
        try:
            osm = OpenSearchManager(
                index_name=index_name, region=region
            )
            osm.connect_to_opensearch()
            return osm
        except Exception as e:
            st.error(f"Failed to connect to OpenSearch: {str(e)}")
            return None

    def handle_user_query(self, query, st_cb):
        st.session_state["tab2"]["messages"].append({"role": "user", "content": query})

        st.chat_message("user").write(query)
        with st.chat_message("assistant"):
            st.markdown("...")  # Typing indicator

        response = self.invoke(
            query=query,
            streaming_callback=st_cb,
            alpha=0.7,
            hyde=True,
            reranker=True,
            parent=True,
            ragfusion=False,
        )

        answer = response[0]
        contexts = response[1]

        st.session_state["tab2"]["messages"].append(
            {"role": "assistant", "content": answer}
        )
        st.session_state["tab2"]["messages"].append(
            {"role": "assistant_context", "content": contexts}
        )
        st_cb._complete_current_thought()

    def reset_chat(self):
        st.session_state["tab2"]["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

############# 6. User 인증과정 관련 ##################

    def load_allowed_users(self):
        """허가된 사용자 정보를 로드합니다."""
        if os.path.exists(self.allowed_users_file):
            with open(self.allowed_users_file, "r") as f:
                return json.load(f)
        else:
            return {}

    # def save_allowed_users(self):
    #     """허가된 사용자 정보를 저장합니다."""
    #     with open(self.allowed_users_file, "w") as f:
    #         json.dump(self.allowed_users, f, indent=4)

    # def authenticate_user(self, user_name, password):
    #     """사용자 인증을 처리합니다."""
    #     return self.allowed_users.get(user_name) == password

    # def create_user(self, user_name, password):
    #     """새로운 사용자를 생성합니다."""
    #     if user_name in self.allowed_users:
    #         raise ValueError("User already exists.")
    #     self.allowed_users[user_name] = password
    #     self.save_allowed_users()

    # def delete_user(self, user_name):
    #     """사용자를 삭제합니다."""
    #     if user_name in self.allowed_users:
    #         del self.allowed_users[user_name]
    #         self.save_allowed_users()
    #     else:
    #         raise ValueError("User does not exist.")
    
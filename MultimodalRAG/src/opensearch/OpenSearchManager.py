import boto3
import os
import logging
from pprint import pprint
from glob import glob
from itertools import chain

from src.preprocessing import LLMLoader
from src.local_utils.ssm import parameter_store
from src.local_utils.common_utils import to_pickle, load_pickle, load_json, to_json
from src.local_utils.opensearch import opensearch_utils
from src.local_utils.chunk import parent_documents
from src.local_utils import bedrock
from src.local_utils.bedrock import bedrock_info
from src.opensearch.schema import get_index_body

from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class OpenSearchManager:
    def __init__(
        self,
        index_name="temp_index_name",
        region="us-east-1",
        dimension=1536,
    ):
        self.index_name = index_name
        self.region = region
        self.dimension = dimension

        # Initialize parameter store
        self.pm = parameter_store(self.region)
        self._setup_parameter_store()

        # Load OpenSearch credentials from parameter store
        self.opensearch_domain_endpoint = self.pm.get_params(
            key="opensearch_domain_endpoint", enc=False
        )
        self.opensearch_account = self.pm.get_params(
            key="opensearch_user_id", enc=False
        )
        self.opensearch_password = self.pm.get_params(
            key="opensearch_user_password", enc=True
        )
        self.http_auth = (self.opensearch_account, self.opensearch_password)

        # Initialize OpenSearch client
        self.os_client = None

        # Initialize Bedrock client and models
        self.llm_loader = LLMLoader(regions=["us-east-1"])
        self.llm_text = self.llm_loader._configure_llm_text()
        self.llm_emb = self.llm_loader._configure_llm_emb()

        # Log the initialization
        logging.info(
            f"OpenSearchManager initialized with index_name: {self.index_name}, "
            f"opensearch_domain_endpoint: {self.opensearch_domain_endpoint}, "
            f"region: {self.region}, dimension: {self.dimension}"
        )

    def _setup_parameter_store(self):
        # Create the parameter key using user_name
        param_key = f"{self.index_name}"

        # Check if the parameter already exists, if not, add it
        existing_params = self.pm.get_all_params()
        if param_key not in existing_params:
            self.pm.put_params(
                key=param_key, value=f"{self.index_name}", overwrite=True, enc=False
            )

### might be deleted ###
    def _setup_bedrock_client(self):
        # Setup Bedrock client
        self.boto3_bedrock = bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
            region=self.region,
        )

    def _setup_models(self):
        # Initialize Bedrock Embeddings model
        self.llm_emb = BedrockEmbeddings(
            client=self.boto3_bedrock,
            model_id=bedrock_info.get_model_id(model_name="Titan-Embeddings-G1"),
        )
        print("Bedrock Embeddings Model Loaded")

        # Initialize ChatBedrock model for text processing
        self.llm_text = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            client=self.boto3_bedrock,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_kwargs={
                "max_tokens": 199999,
                "stop_sequences": ["\n\nHuman"],
            },
        )
        print("Bedrock Multimoal Model Loaded")
####################################


    def connect_to_opensearch(self):
        http_auth = (self.opensearch_account, self.opensearch_password)
        self.os_client = opensearch_utils.create_aws_opensearch_client(
            self.region, self.opensearch_domain_endpoint, http_auth
        )
        self.http_auth = http_auth

    def manage_index(self):
        if not self.os_client:
            raise ValueError(
                "OpenSearch client is not connected. Call connect_to_opensearch() first."
            )

        # Check if the index exists
        index_exists = opensearch_utils.check_if_index_exists(
            self.os_client, self.index_name
        )

        # #If the index exists, optionally delete it
        # if index_exists:
        #     opensearch_utils.delete_index(self.os_client, self.index_name)

        if not index_exists:
            # Create the index using the schema's get_index_body function
            index_body = get_index_body(self.dimension)
            opensearch_utils.create_index(self.os_client, self.index_name, index_body)

            # Print index information
            index_info = self.os_client.indices.get(index=self.index_name)
            print("Index is created")
            pprint(index_info)

    def _delete_index(self):
        if not self.os_client:
            raise ValueError(
                "OpenSearch client is not connected. Call connect_to_opensearch() first."
            )

        # Check if the index exists
        index_exists = opensearch_utils.check_if_index_exists(
            self.os_client, self.index_name
        )

        # If it exists, delete it
        if index_exists:
            opensearch_utils.delete_index(self.os_client, self.index_name)
            print(f"Index '{self.index_name}' deleted.")
        else:
            print(f"Index '{self.index_name}' does not exist.")

    def create_vector_db(
        self, filename, doc_pickle_path="./preprocessed_data", image_path="./fig"
    ):

        doc_pickle_file = filename.split(".")[0] + "_docs.pkl"
        image_preprocessed_file = filename.split(".")[0] + "_image_preprocessed.pkl"
        table_preprocessed_file = filename.split(".")[0] + "_table_preprocessed.pkl"

        doc_path = os.path.join(doc_pickle_path, doc_pickle_file)
        image_preprocessed_full_path = os.path.join(doc_pickle_path, image_preprocessed_file)
        table_preprocessed_full_path = os.path.join(doc_pickle_path, table_preprocessed_file)

        # Load documents
        docs = load_pickle(doc_path)
        images_preprocessed = load_pickle(image_preprocessed_full_path)
        tables_preprocessed = load_pickle(table_preprocessed_full_path)

        # Filter texts (excluding tables)
        texts = [
            doc
            for doc in docs
            if doc.metadata.get("category") not in ["Table", "Image"]
        ]

        # Load images from directory
        images = glob(os.path.join(image_path, "*"))

        print(f" # texts: {len(texts)} \n # images & tables: {len(images)}")

        # Initialize OpenSearch Vector Search
        vector_db = OpenSearchVectorSearch(
            index_name=self.index_name,
            opensearch_url=self.opensearch_domain_endpoint,
            embedding_function=self.llm_emb,
            http_auth=self.http_auth,
            is_aoss=False,
            engine="faiss",
            space_type="l2",
            bulk_size=100000,
            timeout=60,
        )
        print("Vector database created.")

        # Create and add parent chunks to OpenSearch
        parent_chunk_docs = self._create_parent_chunks(texts)
        parent_ids = self._add_documents_to_vector_db(vector_db, parent_chunk_docs)

        # Add child chunks to OpenSearch
        child_chunk_docs = self._create_child_chunks(parent_chunk_docs, parent_ids)
        child_ids = self._add_documents_to_vector_db(vector_db, child_chunk_docs)

        # Process and add images
        for image in images_preprocessed:
            image.metadata["family_tree"] = "parent_image"
            image.metadata["parent_id"] = "NA"

        ## if table...
        for table in tables_preprocessed:
            table.metadata["family_tree"] = "parent_table"
            table.metadata["parent_id"] = "NA"

        # Combine child chunks and images
        docs_preprocessed = list(chain(child_chunk_docs, images_preprocessed, tables_preprocessed))

        # Add combined documents to OpenSearch
        combined_ids = vector_db.add_documents(
            documents=docs_preprocessed, vector_field="vector_field", bulk_size=1000000
        )

        print("Length of combined_ids: ", len(combined_ids))

        return vector_db

    def _create_parent_chunks(self, texts):
        parent_chunk_size = 4096
        parent_chunk_overlap = 0
        opensearch_parent_key_name = "parent_id"
        opensearch_family_tree_key_name = "family_tree"

        parent_chunk_docs = parent_documents.create_parent_chunk(
            docs=texts,
            parent_id_key=opensearch_parent_key_name,
            family_tree_id_key=opensearch_family_tree_key_name,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
        )
        print(f"Number of parent_chunk_docs= {len(parent_chunk_docs)}")
        return parent_chunk_docs

    def _add_documents_to_vector_db(self, vector_db, documents):
        parent_ids = vector_db.add_documents(
            documents=documents, vector_field="vector_field", bulk_size=1000000
        )
        total_count_docs = opensearch_utils.get_count(self.os_client, self.index_name)
        print("Total count docs: ", total_count_docs)
        return parent_ids

    def _create_child_chunks(self, parent_chunk_docs, parent_ids):
        child_chunk_size = 1024
        child_chunk_overlap = 256
        opensearch_parent_key_name = "parent_id"
        opensearch_family_tree_key_name = "family_tree"

        child_chunk_docs = parent_documents.create_child_chunk(
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            docs=parent_chunk_docs,
            parent_ids_value=parent_ids,
            parent_id_key=opensearch_parent_key_name,
            family_tree_id_key=opensearch_family_tree_key_name,
        )
        print(f"Number of child_chunk_docs= {len(child_chunk_docs)}")
        return child_chunk_docs

    def show_opensearch_doc_info(self, doc_id):
        response = opensearch_utils.get_document(
            self.os_client, doc_id=doc_id, index_name=self.index_name
        )
        print("OpenSearch document id:", response["_id"])
        print("Family tree:", response["_source"]["metadata"]["family_tree"])
        print("Parent document id:", response["_source"]["metadata"]["parent_id"])
        print("Parent document text: \n", response["_source"]["text"])

    def delete_index_and_params(self):
        """
        Delete the OpenSearch index and the related parameters from the parameter store.
        """
        if not self.os_client:
            raise ValueError(
                "OpenSearch client is not connected. Call connect_to_opensearch() first."
            )

        # Check if the index exists
        index_exists = opensearch_utils.check_if_index_exists(
            self.os_client, self.index_name
        )

        # If it exists, delete it
        if index_exists:
            opensearch_utils.delete_index(self.os_client, self.index_name)
            print(f"Index '{self.index_name}' deleted.")
        else:
            print(f"Index '{self.index_name}' does not exist.")

        # Delete the related parameter store entry
        param_key = f"{self.index_name}"
        existing_params = self.pm.get_all_params()

        if param_key in existing_params:
            self.pm.delete_params(key=param_key)
            print(f"Parameter '{param_key}' deleted from parameter store.")
        else:
            print(f"Parameter '{param_key}' does not exist in parameter store.")

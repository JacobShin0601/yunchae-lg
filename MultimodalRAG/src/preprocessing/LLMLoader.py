import os
import boto3
import botocore
import logging
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_aws import ChatBedrock
from langchain.embeddings import BedrockEmbeddings
from src.local_utils.common_utils import retry
from src.local_utils import bedrock
from src.local_utils.bedrock import bedrock_info


class LLMLoader:
    def __init__(
        self, 
        regions=None, 
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", 
        embedding_model_id="amazon.titan-embed-text-v1"
    ):
        """Initialize the LLMLoader with the specified regions and model."""
        if regions is None:
            regions = ["us-east-1", "us-west-2", "ap-northeast-1", "eu-central-1"]

        self.regions = regions
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.llm_clients = [self._configure_llm(region) for region in self.regions]

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()

    def _get_default_client(self, aws_region='us-east-1'):
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )
        return bedrock_runtime

    def _configure_llm(self, region_name):
        """Configure the multimodal LLM for a specific AWS region."""
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)
        model_kwargs = {
            "max_tokens": 199999,
            "stop_sequences": ["\n\nHuman"],
            "temperature": 0.01,
            "top_p": 0.9,
        }

        return ChatBedrock(
            client=bedrock_runtime,
            model_id=self.model_id,
            model_kwargs=model_kwargs,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

    def _configure_llm_text(self):
        llm_text = ChatBedrock(
            model_id=self.model_id,
            client=self._get_default_client(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_kwargs={
                "max_tokens": 199999,
                "stop_sequences": ["\n\nHuman"],
                "temperature": 0.01,
                # "top_k": 350,
                "top_p": 0.9
            },
        )
        return llm_text

    def _configure_llm_emb(self):
        """
        Configure the embedding LLM for a specific AWS region.
        """
        llm_emb = BedrockEmbeddings(
            client=self._get_default_client(),
            model_id=self.embedding_model_id
        )
        return llm_emb

    def generate_system_prompt(self):
        """Generate the system prompt for LLM."""
        return SystemMessagePromptTemplate.from_template(
            "You are an assistant tasked with describing table and image."
        )

    def generate_image_prompt(self):
        """Generate the human prompt template for image summarization."""
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + "{image_base64}",
                },
            },
            {
                "type": "text",
                "text": """
                        Given image, give a concise summary.
                        Don't insert any XML tag such as <text> and </text> when answering.
                        
                        If possible, please read the value of y-axis based on x-axis of the time-series chart. (unit of y-axis is mainly USD, Chinese Yuan, Tonne, Pound, etc.)
                        If yuan is used in the y-axis, please write it as CNY.
                        If dollar is used in the y-axis, please write it as USD.
                        Current year is 2024.
                        
                        Detailed information would be preferred.
                        Given document is related to ESG.
                        """,


                        # This article is written by professional research firm, so they are dealing with battery materials such as lithium, graphite, nickel, manganese, cobalt.
                        # BG stands for battery grade.
                        # EXW stands for ex-works in the context of pricing.
                        # CIF stands for cost, insurance, and freight, which is used in the context of pricing.
                        # Especially, please enumerate all the information such as price, supply and demand and forecast as much as you can if the given table is related to 'lithium carbonate or hydroxide'.            
                        
                        # Write in Korean.
                # """,
            },
        ]

    def generate_table_prompt(self):
        """Generate the human prompt template for table summarization."""
        return [
            {
                "type": "text",
                "text": """
                        Here is the table: <table>{table}</table>
                        Given table, give a summary regarding it.
                        Detailed information would be preferred.
                        This article is written by professional research firm related go ESG.
                        """,

                #         This article is written by professional research firm, so they are dealing with battery materials such as lithium, graphite, nikel, manganese, cobalt.
                #         BG stands for battery grade.
                #         Especially, please enumerate all the information such as price, supply and demand and forecast as much as you can if the given table is related to 'lithium carbonate or hydroxide'.
                #         Don't insert any XML tag such as <table> and </table> when answering.
                #         Please add '\n' in the last part of answer.
                #         Write in Korean.
                # """,
            },
        ]

    def create_summarize_chain(self, region_idx=0, for_table=False):
        """Create a chain for summarizing images or tables using the specified region's LLM."""
        if region_idx >= len(self.llm_clients):
            raise IndexError("Region index out of range for configured LLM clients.")

        system_message_template = self.generate_system_prompt()
        if for_table:
            human_prompt_table = self.generate_table_prompt()
            human_message_template_table = HumanMessagePromptTemplate.from_template(
                human_prompt_table
            )
            prompt = ChatPromptTemplate.from_messages(
                [system_message_template, human_message_template_table]
            )
        else:
            human_prompt_img = self.generate_image_prompt()
            human_message_template_img = HumanMessagePromptTemplate.from_template(
                human_prompt_img
            )
            prompt = ChatPromptTemplate.from_messages(
                [system_message_template, human_message_template_img]
            )

        return prompt | self.llm_clients[region_idx] | StrOutputParser()
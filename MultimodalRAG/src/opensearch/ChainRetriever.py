import logging
from langchain.callbacks.tracers import ConsoleCallbackHandler
from src.preprocessing import LLMLoader
from src.local_utils.rag_streamlit import qa_chain, prompt_repo, show_context_used
from src.local_utils.rag_streamlit import retriever_utils, OpenSearchHybridSearchRetriever

class ChainRetriever:
    def __init__(
        self, os_client, index_name, ensemble_weights=[0.7, 0.3]
    ):
        self.os_client = os_client
        self.index_name = index_name
        self.ensemble_weights = ensemble_weights

        self.llm_loader = LLMLoader(regions=['us-east-1'])
        self.llm_text = self.llm_loader._configure_llm_text()
        self.llm_emb = self.llm_loader._configure_llm_emb()
        
        self.retriever = self._initialize_retriever()
        logging.info(f"Opensearch Client initialized: {os_client}")
        logging.info(f"Index Name initialized: {index_name}")

    def _initialize_retriever(self):
        """
        Initialize the OpenSearchHybridSearchRetriever with the provided configurations.
        """
        return OpenSearchHybridSearchRetriever(
            os_client=self.os_client,
            index_name=self.index_name,
            llm_text=self.llm_text,
            llm_emb=self.llm_emb,
            minimum_should_match=0,
            filter=[],
            fusion_algorithm="RRF",
            ensemble_weights=self.ensemble_weights,
            reranker=False,
            parent_document=True,
            complex_doc=True,
            async_mode=True,
            k=7,
            verbose=False,
        )

    def retrieve(self, query, only_text=False):
        try:
            logging.info(f"Executing query: {query}")
            search_hybrid_result, tables, images = (
                self.retriever.get_relevant_documents(query)
            )

            if only_text:
                search_hybrid_result = [
                    doc
                    for doc in search_hybrid_result
                    if doc.metadata.get("category") not in ["Table", "Image"]
                ]
            return search_hybrid_result, tables, images
        except Exception as e:
            logging.error(f"Error occurred during retrieval: {str(e)}")
            raise e

    def visualize_context(self, search_hybrid_result):
        """
        Visualize the context used in the search result.
        """
        logging.info("Visualizing the context used in the search result.")
        show_context_used(search_hybrid_result)

import os
import logging
import requests
import shutil
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.staging.base import elements_to_json
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from src.local_utils.common_utils import to_pickle, load_pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


class FileProcessor:
    def __init__(
        self,
        base_data_path="./data",
        base_output_path="./preprocessed_data",
        base_image_path="./fig",
        database=None
    ):
        self.database = database
        self.base_data_path = base_data_path
        self.base_database = os.path.join(base_data_path, self.database) 
        self.base_output_path = base_output_path
        self.base_image_path = base_image_path

        self.file_path = None
        self.output_path = os.path.join(self.base_output_path, self.database)
        self.image_path = os.path.join(self.base_image_path, self.database)

    def _ensure_directories_exist(self):
        """
        Ensure the necessary directories exist, delete them if they already exist,
        and then recreate them.
        """
        # List of paths to ensure
        paths = [self.image_path, self.output_path]

        for path in paths:
            # If the directory exists, remove it
            if os.path.exists(path):
                shutil.rmtree(path)
            # Recreate the directory
            os.makedirs(path, exist_ok=True)
        logging.info("Directories necessary created.")

    def set_file_path(self, filename):
        """
        Set the file path based on the provided filename and the selected database.
        """
        self.file_path = os.path.join(self.base_database, filename)
        
        # Ensure the directories exist
        os.makedirs(self.base_output_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

        logging.info(f"File path set to: {self.file_path}")
        logging.info(f"Output path set to: {self.output_path}")
        logging.info(f"Image path set to: {self.image_path}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} does not exist.")

    def generate_metadata_from_pdf(self):
        """
        Generate a JSON file containing elements extracted from a PDF.
        """
        requests.packages.urllib3.disable_warnings(
            requests.packages.urllib3.exceptions.InsecureRequestWarning
        )

        attempts = 0
        max_attempts = 2
        infer_table_structure = True

        while attempts < max_attempts:
            try:
                elements = partition_pdf(
                    self.file_path,
                    strategy="hi_res",
                    hi_res_model_name="yolox",
                    infer_table_structure=infer_table_structure,
                )

                elements_filepath = (
                    os.path.basename(self.file_path).split(".")[0] + "_elements.json"
                )

                elements_to_json(
                    elements, filename=os.path.join(self.output_path, elements_filepath)
                )

                logging.info(f"JSON file created at {elements_filepath}")
                return elements

            except Exception as e:
                logging.error(
                    f"Error occurred during metadata generation with infer_table_structure={infer_table_structure}: {e}"
                )
                attempts += 1
                if attempts == max_attempts:
                    logging.warning(
                        "Max attempts reached with infer_table_structure=True. Retrying with infer_table_structure=False."
                    )
                    infer_table_structure = False
                    attempts = 0

        raise Exception(
            "Failed to generate metadata even after retrying with infer_table_structure=False."
        )

    def generate_pdf_docs_and_save(self):
        """
        Generate documents from PDF elements and save them as a pickle file.
        """
        if not self.file_path or not self.file_path.endswith(".pdf"):
            raise ValueError("A valid PDF file path must be provided.")

        loader = UnstructuredPDFLoader(
            file_path=self.file_path,
            chunking_strategy="by_title",
            mode="elements",
            strategy="hi_res",
            hi_res_model_name="yolox",
            max_characters=4096,
            new_after_n_chars=4000,
            combine_text_under_n_chars=2000,
            languages=["eng"],
            post_processors=[clean_bullets, clean_extra_whitespace],
        )

        try:
            docs = loader.load()
            logging.info("Documents loaded successfully.")
        except TypeError as e:
            logging.error(f"A TypeError occurred during document loading: {e}")
            raise e
        except Exception as e:
            logging.error(f"An unexpected error occurred during document loading: {e}")
            raise e

        elements_filepath = os.path.basename(self.file_path).split(".")[0] + "_docs.pkl"
        save_filepath = os.path.join(self.output_path, elements_filepath)
        self._save_docs_as_pickle(docs, save_filepath)

        return docs

    # def generate_metadata_from_xlsx(self):
    #     """
    #     Generate a JSON file containing elements extracted from an Excel file.
    #     """
    #     if not self.file_path or not self.file_path.endswith(".xlsx"):
    #         raise ValueError("A valid Excel file path must be provided.")

    #     requests.packages.urllib3.disable_warnings(
    #         requests.packages.urllib3.exceptions.InsecureRequestWarning
    #     )

    #     elements = partition_xlsx(self.file_path, image_path=self.image_path)

    #     elements_filepath = (
    #         os.path.basename(self.file_path).split(".")[0] + "_elements.json"
    #     )
    #     elements_to_json(
    #         elements, filename=os.path.join(self.output_path, elements_filepath)
    #     )
    #     logging.info(f"JSON file created at {elements_filepath}")

    #     return elements

    # def generate_xlsx_docs_and_save(self):
    #     """
    #     Generate documents from Excel elements and save them as a pickle file.
    #     """
    #     if not self.file_path or not self.file_path.endswith(".xlsx"):
    #         raise ValueError("A valid Excel file path must be provided.")

    #     loader = UnstructuredExcelLoader(
    #         file_path=self.file_path,
    #         chunking_strategy="by_title",
    #         mode="elements",
    #         extract_image_block_to_payload=False,
    #         max_characters=2048,
    #         new_after_n_chars=2000,
    #         combine_text_under_n_chars=1000,
    #         languages=["eng"],
    #         post_processors=[clean_extra_whitespace],
    #     )

    #     try:
    #         docs = loader.load()
    #         logging.info("Documents loaded successfully.")
    #     except TypeError as e:
    #         logging.error(f"A TypeError occurred during document loading: {e}")
    #         raise e
    #     except Exception as e:
    #         logging.error(f"An unexpected error occurred during document loading: {e}")
    #         raise e

    #     elements_filepath = os.path.basename(self.file_path).split(".")[0] + "_docs.pkl"
    #     save_filepath = os.path.join(self.output_path, elements_filepath)
    #     self._save_docs_as_pickle(docs, save_filepath)

    #     return docs

    def _save_docs_as_pickle(self, docs, save_filepath):
        """
        Save docs as a pickle file.
        """
        to_pickle(docs, save_filepath)
        logging.info(f"Saved documents to {save_filepath}")

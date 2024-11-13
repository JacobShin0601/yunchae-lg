import os
import re
import logging
import json
import uuid
from glob import glob
from io import BytesIO
from PIL import Image
import base64
import botocore
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import Document
from src.local_utils.common_utils import load_pickle, to_pickle, image_to_base64, retry
from src.preprocessing import LLMLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class MultimodalBabbler:
    def __init__(
        self,
        base_data_path="./data",
        base_output_path="./preprocessed_data",
        base_image_path="./fig",
        filename=None,
        database=None,
        num_regions=4,
    ):
        self.database = database
        self.base_data_path = base_data_path
        self.base_output_path = base_output_path
        self.base_image_path = base_image_path

        # 데이터베이스별 경로 설정
        self.data_path = os.path.join(self.base_data_path, self.database)
        self.output_path = os.path.join(self.base_output_path, self.database)
        self.image_path = os.path.join(self.base_image_path, self.database)
        self.filename = filename
        self.pickle_file = self.filename.split(".")[0] + "_docs.pkl"

        # elements 파일 경로 설정
        elements_path = os.path.join(
            self.output_path, self.filename.split(".")[0] + "_elements.json"
        )

        self.llm_loader = LLMLoader()

        logging.info(f"Pickle path set to: {self.output_path}")
        logging.info(f"Image path set to: {self.image_path}")
        logging.info(f"Elements file path: {elements_path}")

        with open(elements_path, "r") as json_file:
            elements = json.load(json_file)
        self.elements = elements

        self.table_as_image = True

    def categorize_documents(self, table_as_image=True):
        # Categorize documents into images, tables, and texts, and prepare image_base64 information.
        self.images = glob(os.path.join(self.image_path, "*"))
        self.img_info = {}
        self.table_info = {}

        # 이미지 파일을 self.img_info와 self.table_info에 저장
        for img_path in self.images:
            # 파일명이 figure- 또는 table-로 시작하는지 확인하고 각각의 정보에 저장
            if os.path.basename(img_path).startswith("figure-"):
                element_id = re.search(r"figure-.*-(.*)\.jpg", img_path).group(1)
                self.img_info[element_id] = img_path
                print(
                    f"[DEBUG] Figure image found: {img_path} with element_id: {element_id}"
                )
            elif os.path.basename(img_path).startswith("table-"):
                element_id = re.search(r"table-.*-(.*)\.jpg", img_path).group(1)
                self.table_info[element_id] = img_path
                print(
                    f"[DEBUG] Table image found: {img_path} with element_id: {element_id}"
                )

        # img_info와 table_info의 loop를 통해 doc.metadata["image_base64"]를 설정
        for doc in self.elements:
            element_id = doc["element_id"]
            if element_id in self.img_info:
                img_path = self.img_info[element_id]
                try:
                    doc["metadata"]["image_base64"] = image_to_base64(img_path)
                    print(
                        f"[DEBUG] Image base64 successfully created for figure with element_id: {element_id}"
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Failed to create image base64 for figure with element_id: {element_id}, error: {e}"
                    )
            elif element_id in self.table_info:
                img_path = self.table_info[element_id]
                try:
                    doc["metadata"]["image_base64"] = image_to_base64(img_path)
                    print(
                        f"[DEBUG] Image base64 successfully created for table with element_id: {element_id}"
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Failed to create image base64 for table with element_id: {element_id}, error: {e}"
                    )

        # 카테고리에 따라 self.images, self.tables, self.texts에 분류
        self.images = [
            Document(
                page_content=doc.get("text", None),  # text를 page_content에 저장
                metadata={
                    **doc["metadata"],
                    "element_id": doc["element_id"],
                    "type": doc["type"],
                    "text": doc.get("text", None),
                },  
            )
            for doc in self.elements
            if doc["metadata"].get("image_base64") and doc["type"] == "Image"
        ]

        self.tables = [
            Document(
                page_content=doc.get("text", None),  # text를 page_content에 저장
                metadata={
                    **doc["metadata"],
                    "element_id": doc["element_id"],
                    "type": doc["type"],
                    "text": doc.get("text", None),
                },  # metadata에 element_id 추가
            )
            for doc in self.elements
            if doc["metadata"].get("image_base64") and doc["type"] == "Table"
        ]

        self.texts = [
            Document(
                page_content=doc.get("text", None),  # text를 page_content에 저장
                metadata={
                    **doc["metadata"],
                    "element_id": doc["element_id"],
                    "type": doc["type"],
                },  # metadata에 element_id 추가
            )
            for doc in self.elements
            if doc["type"] not in ["Image", "Table"]
        ]

        print(
            f"[INFO] # of texts: {len(self.texts)} \n# of tables: {len(self.tables)} \n# of images: {len(self.images)}"
        )

    def save_texts(self):
        to_pickle(
            self.texts,
            os.path.join(
                self.output_path,
                self.filename.split(".")[0] + "_docs.pkl",
            ),
        )
        return self.texts

    def summarize_figures(self, verbose=False):
        img_info = {
            doc.metadata["element_id"]: doc.metadata["image_base64"]
            for doc in self.images
        }

        figure_summaries = {}
        batch_size = 5
        img_batches = [
            {k: img_info[k] for k in list(img_info)[i : i + batch_size]}
            for i in range(0, len(img_info), batch_size)
        ]

        max_workers = len(self.llm_loader.llm_clients)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_image_batch, batch): batch
                for batch in img_batches
            }
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    figure_summaries.update(batch_results)
                except Exception as e:
                    logger.error(f"Failed to process a batch: {e}")

        if verbose:
            for img_uuid, summary in figure_summaries.items():
                if summary:
                    print(f"\n== Image {img_uuid} Summary ==")
                    print(summary)

        return figure_summaries

    def process_image_batch(self, images_batch):
        """Process a batch of images for summarization."""
        results = {}
        for img_file, img_base64 in images_batch.items():
            try:
                chain_idx = hash(img_file) % len(self.llm_loader.llm_clients)
                summary = self._summary_img(img_base64, chain_idx)
                results[img_file] = summary
            except Exception as e:
                logger.error(f"Failed to summarize image: {e}")
                results[img_file] = None
        return results

    @retry(
        total_try_cnt=5,
        sleep_in_sec=10,
        retryable_exceptions=(
            botocore.exceptions.EventStreamError,
            botocore.exceptions.ReadTimeoutError,
        ),
    )
    def _summary_img(self, img_base64, region_idx=0):
        summarize_chain = self.llm_loader.create_summarize_chain(
            region_idx, for_table=False
        )
        logger.info(f"Invoking summarize chain for region: {region_idx}")
        estimated_tokens = len(img_base64) / 4
        logger.info(f"Estimated tokens for image summarization: {estimated_tokens}")
        summary = summarize_chain.invoke({"image_base64": img_base64})
        return summary

    def save_processed_figures(self, figure_summaries):
        figures_preprocessed = []

        # original_image_base64 딕셔너리를 생성 (element_id를 사용)
        original_image_base64 = {
            doc.metadata["element_id"]: doc.metadata["image_base64"]
            for doc in self.images
        }

        for element_id, summary in figure_summaries.items():
            if summary:
                metadata = {
                    "element_id": element_id,
                    "category": "Image",
                    "image_base64": original_image_base64.get(
                        element_id
                    ),  # 원본 이미지 데이터를 사용
                }
                doc = Document(page_content=summary, metadata=metadata)
                figures_preprocessed.append(doc)
            else:
                logger.warning(
                    f"Summary is None for image element_id {element_id}"
                )

        to_pickle(
            figures_preprocessed,
            os.path.join(
                self.output_path,
                self.filename.split(".")[0] + "_image_preprocessed.pkl",
            ),
        )

        return figures_preprocessed


    ####################### Table Processing #######################
    def summarize_tables(self, table_as_image=True, verbose=False):
        # table_as_image 상태를 저장
        self.table_as_image = table_as_image
        table_info = [
            {
                "element_id": doc.metadata["element_id"],
                "content_image": doc.metadata.get("image_base64"),
                "content_html": doc.metadata.get("text_as_html"),
                "content_text": doc.metadata.get("text"),
            }
            for doc in self.tables
        ]

        ####
        # # table_info를 JSON 파일로 저장
        # with open('table_info.json', 'w', encoding='utf-8') as f:
        #     json.dump(table_info, f, ensure_ascii=False, indent=4)
        ####

        table_summaries = {}
        batch_size = 5

        table_batches = [
            table_info[i: i + batch_size]
            for i in range(0, len(table_info), batch_size)
        ]

        max_workers = len(self.llm_loader.llm_clients)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_table_batch, batch): batch
                for batch in table_batches
            }
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    table_summaries.update(batch_results)
                except Exception as e:
                    logger.error(f"Failed to process a batch: {e}")

        if verbose:
            for element_id, summary in table_summaries.items():
                if summary:
                    print(f"\n== Table {element_id} Summary ==")
                    print(summary)

        return table_summaries

    def process_table_batch(self, table_batch):
        results = {}

        for table_data in table_batch:
            try:
                # with open('table_data.json', 'w', encoding='utf-8') as f:
                #     json.dump(table_data, f, ensure_ascii=False, indent=4)
                element_id = table_data["element_id"]
                content_image = table_data["content_image"]
                content_html = table_data["content_html"]
                content_text = table_data["content_text"]
                chain_idx = hash(element_id) % len(self.llm_loader.llm_clients)

                # A: 이미지 시도
                summary = self._try_summarize(content_image, "image", element_id, chain_idx)

                # B: 이미지가 실패한 경우 HTML 시도
                if summary is None:
                    summary = self._try_summarize(content_html, "HTML", element_id, chain_idx)

                # C: HTML도 실패한 경우 텍스트 시도
                if summary is None:
                    summary = self._try_summarize(content_text, "text", element_id, chain_idx)

                # 모든 시도가 실패한 경우
                if summary is None:
                    logger.error(f"Failed to summarize table {element_id} with image, HTML, and text.")
                    results[element_id] = None
                else:
                    results[element_id] = summary

            except Exception as e:
                logger.error(f"Failed to summarize table {element_id}: {e}")
                results[element_id] = None

        return results

    def _try_summarize(self, content, content_type, element_id, chain_idx):
        if content:
            try:
                summary = self._summary_table(content, chain_idx)
                if summary:
                    logger.info(f"Successfully summarized table {element_id} using {content_type} data.")
                else:
                    logger.warning(f"Summarization returned None for table {element_id} using {content_type} data.")
                return summary
            except Exception as e:
                logger.error(f"Error during processing table {element_id} with {content_type} data: {e}")
                return None
        else:
            logger.info(f"No {content_type} data provided for table {element_id}.")
        return None

    @retry(
        total_try_cnt=2,
        sleep_in_sec=10,
        retryable_exceptions=(
            botocore.exceptions.EventStreamError,
            botocore.exceptions.ReadTimeoutError,
        ),
    )
    def _summary_table(self, table_data, region_idx=0):
        summarize_chain = self.llm_loader.create_summarize_chain(region_idx, for_table=True)
        
        # 입력 데이터의 토큰 크기를 계산하여 로그에 기록
        token_count = len(table_data)/4  # 단순히 길이로 토큰을 계산할 수 있음
        logger.debug(f"Input token size for region {region_idx}: {token_count}")
        
        input_data = {"table": table_data}
        summary = summarize_chain.invoke(input_data)
        logger.debug(f"Summary result for region {region_idx}: {summary[:100]}")  # 로그에서 요약된 결과의 일부를 기록
        return summary


    def save_processed_tables(self, table_summaries):
        tables_preprocessed = []

        for idx, origin in enumerate(self.tables):
            element_id = origin.metadata["element_id"]
            summary = table_summaries.get(element_id, None)

            if summary:
                metadata = origin.metadata.copy()

                if "image_base64" in origin.metadata:
                    metadata["image_base64"] = origin.metadata["image_base64"]

                if "text_as_html" in origin.metadata:
                    metadata["text_as_html"] = origin.metadata["text_as_html"]
                    metadata["origin_table"] = origin.metadata["text_as_html"]
                else:
                    metadata["origin_table"] = origin.metadata["text"]

                doc = Document(page_content=summary, metadata=metadata)
                tables_preprocessed.append(doc)
            else:
                logger.warning(f"Summary is None for table {idx}")

        to_pickle(
            tables_preprocessed,
            os.path.join(
                self.output_path,
                self.filename.split(".")[0] + "_table_preprocessed.pkl",
            ),
        )

        return tables_preprocessed


    
    #####
    
    # def summarize_tables(self, table_as_image=True, verbose=False):
    #     # table_as_image 상태를 저장
    #     self.table_as_image = table_as_image
    #     table_info = [
    #         {
    #             "element_id": doc.metadata["element_id"],
    #             "content_image": doc.metadata.get("image_base64"),
    #             "content_html": doc.metadata.get("text_as_html"),
    #             "content_text": doc.metadata.get("text"),
    #         }
    #         for doc in self.tables
    #     ]

    #     table_summaries = {}
    #     batch_size = 5

    #     table_batches = [
    #         table_info[i: i + batch_size]
    #         for i in range(0, len(table_info), batch_size)
    #     ]

    #     max_workers = len(self.llm_loader.llm_clients)

    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         future_to_batch = {
    #             executor.submit(self.process_table_batch, batch): batch
    #             for batch in table_batches
    #         }
    #         for future in as_completed(future_to_batch):
    #             try:
    #                 batch_results = future.result()
    #                 table_summaries.update(batch_results)
    #             except Exception as e:
    #                 logger.error(f"Failed to process a batch: {e}")

    #     if verbose:
    #         for element_id, summary in table_summaries.items():
    #             if summary:
    #                 print(f"\n== Table {element_id} Summary ==")
    #                 print(summary)

    #     return table_summaries

    # def process_table_batch(self, table_batch):
    #     results = {}

    #     for table_data in table_batch:
    #         try:
    #             element_id = table_data["element_id"]
    #             content_image = table_data["content_image"]
    #             content_html = table_data.get("content_html")
    #             content_text = table_data.get("text")
    #             chain_idx = hash(element_id) % len(self.llm_loader.llm_clients)

    #             # 시도 순서 설정: 이미지 -> HTML -> 텍스트
    #             attempts = ['image', 'html', 'text']
    #             summary = None

    #             for attempt in attempts:
    #                 try:
    #                     if attempt == 'image' and content_image:
    #                         logger.info(f"Processing table {element_id} with image data.")
    #                         logger.debug(f"Image data length: {len(content_image) if content_image else 'N/A'}")
    #                         summary = self._summary_table(content_image, chain_idx, table_as_image=True)
    #                         if summary:
    #                             logger.info(f"Successfully summarized table {element_id} using image data.")
    #                             break
    #                         else:
    #                             logger.warning(f"Table {element_id}: Image processing failed or returned None.")

    #                     elif attempt == 'html' and content_html:
    #                         logger.info(f"Retrying table {element_id} with HTML content.")
    #                         logger.debug(f"HTML data length: {len(content_html) if content_html else 'N/A'}")
    #                         summary = self._summary_table(content_html, chain_idx, table_as_image=False)
    #                         if summary:
    #                             logger.info(f"Successfully summarized table {element_id} using HTML data.")
    #                             break
    #                         else:
    #                             logger.warning(f"Table {element_id}: HTML processing failed or returned None.")

    #                     elif attempt == 'text' and content_text:
    #                         logger.info(f"Retrying table {element_id} with plain text content.")
    #                         logger.debug(f"Text data length: {len(content_text) if content_text else 'N/A'}")
    #                         summary = self._summary_table(content_text, chain_idx, table_as_image=False)
    #                         if summary:
    #                             logger.info(f"Successfully summarized table {element_id} using text data.")
    #                             break
    #                         else:
    #                             logger.warning(f"Table {element_id}: Text processing failed or returned None.")

    #                 except Exception as e:
    #                     logger.error(f"Error during {attempt} processing for table {element_id}: {e}")
    #                     continue  # 예외 발생 시에도 다음 처리 방식으로 넘어감

    #             if summary is None:
    #                 logger.error(f"Failed to summarize table {element_id} with image, HTML, and text.")
    #                 results[element_id] = None
    #             else:
    #                 results[element_id] = summary

    #         except Exception as e:
    #             logger.error(f"Failed to summarize table {element_id}: {e}")
    #             results[element_id] = None

    #     return results

    # def _summary_table(self, table_data, region_idx=0, table_as_image=True):
    #     summarize_chain = self.llm_loader.create_summarize_chain(
    #         region_idx, for_table=True
    #     )

    #     try:
    #         logger.info(f"Processing table as {'image' if table_as_image else 'HTML or text'}.")
    #         input_data = {"table": table_data}
    #         summary = summarize_chain.invoke(input_data)
    #         logger.debug(f"Summary result for region {region_idx}: {summary[:100]}")  # 로그에서 요약된 결과의 일부를 기록
    #         return summary

    #     except Exception as e:
    #         logger.error(f"Error during summary for region {region_idx}: {e}")
    #         raise e  # 예외를 상위로 전달하여 상위 함수에서 처리하게 함


    # def save_processed_tables(self, table_summaries):
    #     tables_preprocessed = []

    #     for idx, origin in enumerate(self.tables):
    #         element_id = origin.metadata["element_id"]
    #         summary = table_summaries.get(element_id, None)

    #         if summary:
    #             metadata = origin.metadata.copy()

    #             if "image_base64" in origin.metadata:
    #                 metadata["image_base64"] = origin.metadata["image_base64"]

    #             if "text_as_html" in origin.metadata:
    #                 metadata["text_as_html"] = origin.metadata["text_as_html"]
    #                 metadata["origin_table"] = origin.metadata["text_as_html"]
    #             else:
    #                 metadata["origin_table"] = origin.metadata["text"]

    #             doc = Document(page_content=summary, metadata=metadata)
    #             tables_preprocessed.append(doc)
    #         else:
    #             logger.warning(f"Summary is None for table {idx}")

    #     to_pickle(
    #         tables_preprocessed,
    #         os.path.join(
    #             self.output_path,
    #             self.filename.split(".")[0] + "_table_preprocessed.pkl",
    #         ),
    #     )

    #     return tables_preprocessed
    ##
    
    
    # def summarize_tables(self, table_as_image=True, verbose=False):
    #     table_info = [
    #         {
    #             "element_id": doc.metadata["element_id"],
    #             "content_image": doc.metadata.get("image_base64"),
    #             "content_html": doc.metadata.get("text_as_html"),
    #             "content_text": doc.metadata.get("text"),
    #         }
    #         for doc in self.tables
    #     ]

    #     table_summaries = {}
    #     batch_size = 5

    #     table_batches = [
    #         table_info[i: i + batch_size]
    #         for i in range(0, len(table_info), batch_size)
    #     ]

    #     max_workers = len(self.llm_loader.llm_clients)

    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         future_to_batch = {
    #             executor.submit(self.process_table_batch, batch, table_as_image): batch
    #             for batch in table_batches
    #         }
    #         for future in as_completed(future_to_batch):
    #             try:
    #                 batch_results = future.result()
    #                 table_summaries.update(batch_results)
    #             except Exception as e:
    #                 logger.error(f"Failed to process a batch: {e}")

    #     if verbose:
    #         for element_id, summary in table_summaries.items():
    #             if summary:
    #                 print(f"\n== Table {element_id} Summary ==")
    #                 print(summary)

    #     return table_summaries

    # def process_table_batch(self, table_batch, table_as_image=True):
    #     results = {}

    #     for table_data in table_batch:
    #         try:
    #             element_id = table_data["element_id"]
    #             content_image = table_data["content_image"]
    #             content_html = table_data.get("content_html")
    #             content_text = table_data.get("text")
    #             chain_idx = hash(element_id) % len(self.llm_loader.llm_clients)

    #             logger.info(f"Processing table {element_id} with image data first.")
    #             logger.debug(f"Image data length: {len(content_image) if content_image else 'N/A'}")

    #             # First try with image_base64
    #             summary = self._summary_table(content_image, chain_idx, table_as_image=True)

    #             if summary is None:
    #                 logger.warning(f"Table {element_id}: Image processing failed or returned None.")
    #                 if content_html:
    #                     logger.info(f"Retrying table {element_id} with HTML content.")
    #                     logger.debug(f"HTML data length: {len(content_html) if content_html else 'N/A'}")
    #                     summary = self._summary_table(content_html, chain_idx, table_as_image=False)
    #                 else:
    #                     logger.warning(f"Table {element_id}: HTML content not available.")

    #                 if summary is None and content_text:
    #                     logger.info(f"Retrying table {element_id} with plain text content.")
    #                     logger.debug(f"Text data length: {len(content_text) if content_text else 'N/A'}")
    #                     summary = self._summary_table(content_text, chain_idx, table_as_image=False)

    #             if summary is None:
    #                 logger.error(f"Failed to summarize table {element_id} with image, HTML, and text.")
    #             else:
    #                 logger.info(f"Successfully summarized table {element_id}.")

    #             results[element_id] = summary
    #         except Exception as e:
    #             logger.error(f"Failed to summarize table {element_id}: {e}")
    #             results[element_id] = None

    #     return results

    # @retry(
    #     total_try_cnt=5,
    #     sleep_in_sec=10,
    #     retryable_exceptions=(
    #         botocore.exceptions.EventStreamError,
    #         botocore.exceptions.ReadTimeoutError,
    #     ),
    # )
    # def _summary_table(self, table_data, region_idx=0, table_as_image=True):
    #     summarize_chain = self.llm_loader.create_summarize_chain(
    #         region_idx, for_table=True
    #     )

    #     try:
    #         logger.info(f"Processing table as {'image' if table_as_image else 'HTML or text'}.")
    #         input_data = {"table": table_data}
    #         summary = summarize_chain.invoke(input_data)
    #         logger.debug(f"Summary result for region {region_idx}: {summary[:100]}")  # 로그에서 요약된 결과의 일부를 기록
    #         return summary

    #     except Exception as e:
    #         if 'Input is too long' in str(e) and table_as_image:
    #             logger.warning("Input too long for image processing, retrying with HTML or text.")
    #             return None  # Return None to trigger the HTML or text retry logic
    #         else:
    #             logger.error(f"Error during summary for region {region_idx}: {e}")
    #             raise e

        
    # def save_processed_tables(self, table_summaries):
    #     tables_preprocessed = []

    #     for idx, origin in enumerate(self.tables):
    #         element_id = origin.metadata["element_id"]
    #         summary = table_summaries.get(element_id, None)

    #         if summary:
    #             metadata = origin.metadata.copy()  # 원본 metadata를 복사하여 수정

    #             if "image_base64" in origin.metadata:
    #                 metadata["image_base64"] = origin.metadata["image_base64"]

    #             if "text_as_html" in origin.metadata:
    #                 metadata["text_as_html"] = origin.metadata["text_as_html"]
    #                 metadata["origin_table"] = origin.metadata["text_as_html"]
    #             else:
    #                 metadata["origin_table"] = origin.metadata["text"]

    #             doc = Document(page_content=summary, metadata=metadata)
    #             tables_preprocessed.append(doc)
    #         else:
    #             logger.warning(f"Summary is None for table {idx}")

    #     to_pickle(
    #         tables_preprocessed,
    #         os.path.join(
    #             self.output_path,
    #             self.filename.split(".")[0] + "_table_preprocessed.pkl",
    #         ),
    #     )

    #     return tables_preprocessed



import os
import time
import json
import streamlit as st
from tqdm import tqdm
from src.preprocessing import (
    ImageProcessor,
    FileProcessor,
    LLMLoader,
    MultimodalBabbler,
)
from src.opensearch import OpenSearchManager, ChainRetriever
from src.local_utils.ssm import parameter_store


class PipelineManager:
    def __init__(
        self,
        filename,
        database=None,
        base_data_dir="./data",
        base_output_dir="./preprocessed_data",
        base_image_dir="./fig",
    ):
        self.filename = filename
        self.database = database  # database 변수를 추가하여 경로에 반영
        self.base_data_dir = base_data_dir
        self.base_output_dir = base_output_dir
        self.base_image_dir = base_image_dir
        self.osm = None
        self.current_progress = 0.0

        # 데이터베이스에 따라 세부 폴더 경로 설정
        self.data_dir = os.path.join(self.base_data_dir, self.database)
        self.output_dir = os.path.join(
            self.base_output_dir, self.database
        )
        self.image_dir = os.path.join(self.base_image_dir, self.database)

        # 폴더가 존재하지 않으면 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.base_image_dir, exist_ok=True)

    def log_time_taken(self, start_time, task_name, status_text=None):
        elapsed_time = time.time() - start_time
        log_message = f"{task_name} took {elapsed_time:.2f} seconds"
        print(log_message)
        if status_text:
            status_text.text(log_message)

    def run_pipeline(self, steps=None, progress_bar=None, status_text=None):
        print("Starting the pipeline...\n")
        total_steps = 2  # We have two major steps: preprocess and opensearch
        step_increment = 1 / total_steps

        current_progress = 0

        if steps is None or "preprocess" in steps:
            # Run preprocessing
            start_time = time.time()
            if status_text:
                status_text.text("Starting Preprocessing...")
            self.run_preprocess(progress_bar, status_text)
            self.log_time_taken(
                start_time, "Preprocessing uploaded file...", status_text
            )
            current_progress += step_increment
            if progress_bar:
                progress_bar.progress(current_progress)

        if steps is None or "opensearch" in steps:
            # Run OpenSearch setup
            start_time = time.time()
            if status_text:
                status_text.text("Starting OpenSearch Setup...")
            self.run_opensearch(progress_bar, status_text)
            self.log_time_taken(
                start_time,
                "Setting up OpenSearch and uploading data to OpenSearch...",
                status_text,
            )
            current_progress += step_increment
            if progress_bar:
                progress_bar.progress(current_progress)

        print("Whole process of pipeline successfully completed😊")
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("Pipeline Completed Successfully!")

    ##############################################################
    # Preprocessing Functions
    def run_preprocess(self, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Running File Processing...")
        self._file_processing(progress_bar, status_text)
        self._image_table_processing(progress_bar, status_text)
        self._multimodal_processing(progress_bar, status_text)

    def _file_processing(self, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Running File Processing...")
        start_time = time.time()

        # base_data_path, base_output_path, database 인자를 한번만 전달하도록 수정
        fp = FileProcessor(
            database=self.database
        )
        fp.set_file_path(self.filename)  # 경로 설정에 database 사용
        fp.generate_metadata_from_pdf()

        self.log_time_taken(start_time, "File Processing", status_text)
        if progress_bar:
            self.current_progress += 0.2
            progress_bar.progress(self.current_progress)

    def _image_table_processing(self, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Processing Images and Tables...")
        start_time = time.time()

        ip = ImageProcessor(
            filename=self.filename,
            database=self.database,
        )

        full_path = os.path.join(
            ip.get_target_file(self.filename)
        )

        with open(full_path, "r") as json_file:
            element = json.load(json_file)

        ip.process_figures(elements=element)
        ip.process_tables(elements=element)

        self.log_time_taken(
            start_time, "Parsing document: image. and tables...", status_text
        )
        if progress_bar:
            self.current_progress += 0.2
            progress_bar.progress(self.current_progress)

    def _multimodal_processing(self, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Running Multimodal LLM Recognition...")
        start_time = time.time()

        mmbabbler = MultimodalBabbler(
            filename=self.filename,
            database=self.database,
        )
        mmbabbler.categorize_documents()
        figure_summaries = mmbabbler.summarize_figures()
        table_summaries = mmbabbler.summarize_tables()

        mmbabbler.save_texts()
        mmbabbler.save_processed_figures(figure_summaries)
        mmbabbler.save_processed_tables(table_summaries)

        self.log_time_taken(start_time, "Multimodal Babbler Processing", status_text)
        if progress_bar:
            self.current_progress += 0.2
            progress_bar.progress(self.current_progress)

    ##############################################################
    # OpenSearch Functions
    # OpenSearch Functions
    def run_opensearch(self, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Setting up database and uploading data...")
        self._initialize_opensearch(self.database, progress_bar, status_text)
        self._manage_opensearch_index(progress_bar, status_text)
        self._create_vector_db(progress_bar, status_text)

    def _initialize_opensearch(self, index_name, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Initializing OpenSearch...")
        start_time = time.time()

        # OpenSearchManager 초기화 시에 데이터베이스 이름(index_name)을 전달
        self.osm = OpenSearchManager(index_name=index_name)
        self.osm.connect_to_opensearch()

        self.log_time_taken(start_time, "OpenSearch Initialization", status_text)
        if progress_bar:
            self.current_progress += 0.1
            progress_bar.progress(self.current_progress)

    def _manage_opensearch_index(self, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Managing OpenSearch Index...")
        start_time = time.time()

        self.osm.manage_index()

        self.log_time_taken(start_time, "OpenSearch Index Management", status_text)
        if progress_bar:
            self.current_progress += 0.1
            progress_bar.progress(self.current_progress)

    def _create_vector_db(self, progress_bar=None, status_text=None):
        if status_text:
            status_text.text("Creating and connecting to Vector DB...")
        start_time = time.time()

        # filename의 전체 경로를 전달하여 벡터 DB 생성
        self.osm.create_vector_db(filename=self.filename, doc_pickle_path=self.output_dir, image_path=self.image_dir)

        self.log_time_taken(start_time, "Vector DB Creation", status_text)
        if progress_bar:
            self.current_progress += 0.1
            progress_bar.progress(self.current_progress)

import os
import shutil
import math
import cv2
import base64
import logging
from glob import glob
from pdf2image import convert_from_path


class ImageProcessor:
    def __init__(
        self,
        base_data_path="./data",
        image_path="./fig",
        output_path="./preprocessed_data",
        filename="default",
        margin_expand_ratio=(0.15, 0.2),
        database=None
    ):
        self.database = database
        self.base_data_path = os.path.join(base_data_path, self.database)
        self.image_path = os.path.join(image_path, self.database)
        self.output_path = os.path.join(output_path, self.database)
        self.filename = filename
        self.margin_expand_ratio = margin_expand_ratio
        self.image_tmp_path = os.path.join(self.image_path, self.database + "_tmp")

        # 폴더가 이미 존재하면 삭제
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)

        # 새로 폴더 생성
        os.makedirs(self.image_path, exist_ok=True)

        logging.info(f"Image processing path set to: {self.image_path}")
        logging.info(f"Temporary image path set to: {self.image_tmp_path}")
        logging.info(f"Output path set to: {self.output_path}")

    def get_target_file(self, filename):
        # output_path에 이미 database 경로가 포함되어 있다고 가정하고, filename에서 경로 정보를 제거하여 조합합니다.
        base_filename = os.path.splitext(os.path.basename(filename))[0] 
        full_path = os.path.join(self.output_path, base_filename + "_elements.json")
        return full_path

    def _prepare_temp_folder(self):
        if os.path.isdir(self.image_tmp_path):
            shutil.rmtree(self.image_tmp_path)
        os.makedirs(self.image_tmp_path, exist_ok=True)

    def _clean_temp_folder(self):
        if os.path.isdir(self.image_tmp_path):
            shutil.rmtree(self.image_tmp_path)

    def _convert_pdf_to_images(self):
        """Convert PDF pages to images and save them in the temporary directory."""
        self._prepare_temp_folder()
        
        # PDF 파일의 전체 경로 생성
        pdf_path = os.path.join(self.base_data_path, self.filename)
        pages = convert_from_path(pdf_path)
        
        for i, page in enumerate(pages):
            page_path = f"{self.image_tmp_path}/{str(i + 1)}.jpg"
            page.save(page_path, "JPEG")
            print(f"Saved PDF page {i} as image: {page_path}")

    def _calculate_expanded_margins(self, img_width, img_height):
        """Calculate expanded margins based on the image dimensions and margin expand ratio."""
        upper_expand = int(img_height * self.margin_expand_ratio[0])
        left_expand = int(img_width * self.margin_expand_ratio[1])
        return upper_expand, left_expand

    def _crop_image(self, img, points):
        img_height, img_width = img.shape[:2]
        upper_expand, left_expand = self._calculate_expanded_margins(
            img_width, img_height
        )

        # Calculate coordinates with the expanded margins
        y1 = max(math.ceil(points[0][1] - upper_expand), 0) if points[0][1] else 0
        y2 = min(math.ceil(points[1][1]), img_height)
        x1 = max(math.ceil(points[0][0] - left_expand), 0) if points[0][0] else 0
        x2 = min(math.ceil(points[3][0]), img_width)

        # Ensure coordinates are within image bounds
        if y1 < 0 or y2 > img_height or x1 < 0 or x2 > img_width:
            print("Invalid crop coordinates detected. Expanding to max available size.")
            y1 = max(math.ceil(points[0][1]), 0) if points[0][1] else 0
            y2 = min(math.ceil(points[1][1]), img_height)
            x1 = max(math.ceil(points[0][0]), 0) if points[0][0] else 0
            x2 = min(math.ceil(points[3][0]), img_width)

        crop_img = img[y1:y2, x1:x2]
        return crop_img

    def _save_image(self, img, image_type, filename, page_number, element_idx):
        """Save cropped image and return its path."""
        image_path = f"{self.image_path}/{image_type}-{filename}-p{page_number}-{element_idx}.jpg"
        cv2.imwrite(image_path, img)
        return image_path

    def _image_to_base64(self, image_path):
        """Convert an image file to a Base64 encoded string."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

    def process_figures(self, elements):
        self._convert_pdf_to_images()

        for idx, element in enumerate(elements):
            if element["type"] != "Image":
                continue

            filename = element["metadata"]["filename"].split(".")[0].replace("-", "_")
            points = element["metadata"]["coordinates"]["points"]
            page_number = element["metadata"]["page_number"]
            element_id = element["element_id"]

            img_path = f"{self.image_tmp_path}/{page_number}.jpg"
            img = cv2.imread(img_path)

            if img is None:
                print(f"Cannot load image: {img_path}")
                continue

            crop_img = self._crop_image(img, points)
            if crop_img.size == 0:
                print("Cropped image is empty.")
                continue

            width, height, _ = crop_img.shape
            image_token = width * height / 750

            # 만약 예상 토큰 수가 LLM의 max_tokens를 초과하는 경우 크기 조정
            max_tokens = 199999  # LLM의 최대 토큰 수
            if image_token > max_tokens:
                scale_factor = (max_tokens / image_token) ** 0.5
                crop_img = cv2.resize(
                    crop_img, (int(width * scale_factor), int(height * scale_factor))
                )
                print(f"Resized image to fit within token limits: {crop_img.shape}")

            figure_image_path = self._save_image(
                crop_img, "figure", filename, page_number, element_id
            )
            print(
                f"Image: {figure_image_path}, shape: {crop_img.shape}, image_token: {image_token}"
            )

        self._clean_temp_folder()

    def process_tables(self, elements):
        """Process table elements and save cropped images as well as encode them in Base64."""
        self._convert_pdf_to_images()

        for idx, element in enumerate(elements):
            if element["type"] != "Table":
                continue

            filename = element["metadata"]["filename"].split(".")[0].replace("-", "_")
            points = element["metadata"]["coordinates"]["points"]
            page_number = element["metadata"]["page_number"]
            element_id = element["element_id"]

            img_path = f"{self.image_tmp_path}/{page_number}.jpg"
            img = cv2.imread(img_path)

            if img is None:
                print(f"Cannot load image: {img_path}")
                continue

            crop_img = self._crop_image(img, points)
            if crop_img.size == 0:
                print("Cropped image is empty.")
                continue

            width, height, _ = crop_img.shape
            image_token = width * height / 750

            # 만약 예상 토큰 수가 LLM의 max_tokens를 초과하는 경우 크기 조정
            max_tokens = 199999  # LLM의 최대 토큰 수
            if image_token > max_tokens:
                scale_factor = (max_tokens / image_token) ** 0.5
                crop_img = cv2.resize(
                    crop_img, (int(width * scale_factor), int(height * scale_factor))
                )
                print(
                    f"Resized table image to fit within token limits: {crop_img.shape}"
                )

            table_image_path = self._save_image(
                crop_img, "table", filename, page_number, element_id
            )
            print(
                f"Image: {table_image_path}, shape: {crop_img.shape}, image_token: {image_token}"
            )

            img_base64 = self._image_to_base64(table_image_path)
            element["metadata"]["image_base64"] = img_base64

        self._clean_temp_folder()
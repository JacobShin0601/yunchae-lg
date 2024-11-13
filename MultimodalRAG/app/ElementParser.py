import os
import json


class ElementParser:
    def __init__(self, base_data_dir='./data', base_output_dir='./preprocessed_data', base_img_dir='./fig', database=None):
        self.database = database
        self.base_data_dir = base_data_dir
        self.base_output_dir = base_output_dir
        self.base_img_dir = base_img_dir
        
        self.data_dir = os.path.join(self.base_data_dir, self.database)
        self.output_dir = os.path.join(self.base_output_dir, self.database)
        self.img_dir = os.path.join(self.base_img_dir, self.database)

    def _get_json_files(self):
        # 데이터베이스별로 json 파일 경로 설정
        preprocessed_data_dir = self.output_dir
        
        # preprocessed_data_dir 내에서 `_elements.json`으로 끝나는 파일만 필터링하여 리스트에 담기
        all_files = os.listdir(preprocessed_data_dir)
        json_files = [file for file in all_files if file.endswith('_elements.json')]
        return json_files

    def _parse_and_combine_json(self):
        json_files = self._get_json_files()
        combined_data = []

        for json_file in json_files:
            file_path = os.path.join(self.output_dir, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)  # JSON 파일 로드
                combined_data.extend(data)  # 데이터를 합쳐서 하나의 리스트로 만듦

        return combined_data

    def save_combined_json(self, output_file):
        combined_data = self._parse_and_combine_json()
        output_path = os.path.join(self.output_dir, output_file)

        os.makedirs(self.output_dir, exist_ok=True)  # 저장 경로가 없을 경우 폴더 생성

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)

        print(f"Combined JSON saved to {output_path}")
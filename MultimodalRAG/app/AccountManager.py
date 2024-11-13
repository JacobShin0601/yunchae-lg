import os
import json

class AccountManager:
    def __init__(self, allowed_users_file='allowed_users.json'):
        # 데이터 경로 설정
        self.allowed_users_file = "allowed_users.json"

        # 로컬 JSON 파일로부터 허가된 사용자 정보를 로드
        self.allowed_users = self.load_allowed_users()

    def load_allowed_users(self):
        """허가된 사용자 정보를 로드합니다."""
        if os.path.exists(self.allowed_users_file):
            with open(self.allowed_users_file, "r") as f:
                return json.load(f)
        else:
            return {}

    def save_allowed_users(self):
        """허가된 사용자 정보를 저장합니다."""
        with open(self.allowed_users_file, "w") as f:
            json.dump(self.allowed_users, f, indent=4)

    def authenticate_user(self, user_name, password):
        """사용자 인증을 처리합니다."""
        return self.allowed_users.get(user_name) == password

    def create_user(self, user_name, password):
        """새로운 사용자를 생성합니다."""
        if user_name in self.allowed_users:
            raise ValueError("User already exists.")
        self.allowed_users[user_name] = password
        self.save_allowed_users()

    def delete_user(self, user_name):
        """사용자를 삭제합니다."""
        if user_name in self.allowed_users:
            del self.allowed_users[user_name]
            self.save_allowed_users()
        else:
            raise ValueError("User does not exist.")
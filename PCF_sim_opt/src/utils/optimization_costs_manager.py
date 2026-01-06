"""
사용자별 optimization_costs 파일 관리 모듈

이 모듈은 stable_var/optimization_costs/의 기본 템플릿 파일들을
사용자별 폴더(stable_var/{user_id}/optimization_costs/)로 복사하고 관리합니다.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List
import json


class OptimizationCostsManager:
    """
    사용자별 optimization_costs 파일 관리 클래스

    주요 기능:
    - 기본 템플릿 파일을 사용자별 폴더로 복사
    - 사용자별 경로 반환
    - 파일 존재 여부 확인
    """

    # 관리할 파일 목록
    COST_FILES = [
        "electricity_usage_per_material.json",
        "unit_cost_per_country.json",
        "basic_cost_per_material.json",
        "material_category_mapping.json",
        "material_cost_premiums.json"
    ]

    BASE_COSTS_DIR = "stable_var/optimization_costs"

    def __init__(self, base_dir: str = None):
        """
        OptimizationCostsManager 초기화

        Args:
            base_dir: 프로젝트 루트 디렉토리. None이면 현재 파일 기준 상대 경로 사용
        """
        if base_dir is None:
            # 현재 파일의 위치에서 프로젝트 루트 찾기
            current_file = Path(__file__).resolve()
            self.base_dir = current_file.parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.template_dir = self.base_dir / self.BASE_COSTS_DIR

    def initialize_user_costs(self, user_id: str, force: bool = False) -> bool:
        """
        사용자별 optimization_costs 폴더 초기화

        stable_var/optimization_costs/의 모든 파일을
        stable_var/{user_id}/optimization_costs/로 복사합니다.

        Args:
            user_id: 사용자 ID
            force: True면 기존 파일 덮어쓰기, False면 없는 파일만 복사

        Returns:
            bool: 성공 여부
        """
        if not user_id:
            print("⚠️ user_id가 제공되지 않았습니다.")
            return False

        # 사용자별 디렉토리 경로
        user_costs_dir = self.get_user_costs_dir(user_id)

        # 디렉토리 생성
        try:
            user_costs_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ 사용자 디렉토리 생성 실패: {e}")
            return False

        # 파일 복사
        copied_count = 0
        skipped_count = 0

        for filename in self.COST_FILES:
            template_file = self.template_dir / filename
            user_file = user_costs_dir / filename

            # 템플릿 파일 존재 확인
            if not template_file.exists():
                print(f"⚠️ 템플릿 파일이 없습니다: {template_file}")
                continue

            # 사용자 파일 존재 여부 확인
            if user_file.exists() and not force:
                skipped_count += 1
                continue

            # 파일 복사
            try:
                shutil.copy2(template_file, user_file)
                copied_count += 1
            except Exception as e:
                print(f"❌ 파일 복사 실패 ({filename}): {e}")
                return False

        if copied_count > 0:
            print(f"✅ {user_id} 사용자의 optimization_costs 파일 {copied_count}개 복사 완료")

        if skipped_count > 0:
            print(f"ℹ️ {skipped_count}개 파일은 이미 존재하여 스킵됨 (force=False)")

        return True

    def get_user_costs_dir(self, user_id: str) -> Path:
        """
        사용자별 optimization_costs 디렉토리 경로 반환

        Args:
            user_id: 사용자 ID

        Returns:
            Path: 사용자별 optimization_costs 디렉토리 경로
        """
        return self.base_dir / "stable_var" / user_id / "optimization_costs"

    def get_user_costs_path(self, user_id: str) -> str:
        """
        사용자별 optimization_costs 디렉토리 경로 반환 (문자열)

        Args:
            user_id: 사용자 ID

        Returns:
            str: 사용자별 optimization_costs 디렉토리 경로
        """
        return str(self.get_user_costs_dir(user_id))

    def check_user_costs_exist(self, user_id: str) -> bool:
        """
        사용자의 optimization_costs 파일이 모두 존재하는지 확인

        Args:
            user_id: 사용자 ID

        Returns:
            bool: 모든 필수 파일이 존재하면 True
        """
        user_costs_dir = self.get_user_costs_dir(user_id)

        if not user_costs_dir.exists():
            return False

        for filename in self.COST_FILES:
            user_file = user_costs_dir / filename
            if not user_file.exists():
                return False

        return True

    def get_missing_files(self, user_id: str) -> List[str]:
        """
        사용자의 누락된 optimization_costs 파일 목록 반환

        Args:
            user_id: 사용자 ID

        Returns:
            List[str]: 누락된 파일 이름 목록
        """
        user_costs_dir = self.get_user_costs_dir(user_id)
        missing = []

        for filename in self.COST_FILES:
            user_file = user_costs_dir / filename
            if not user_file.exists():
                missing.append(filename)

        return missing

    def update_user_file(
        self,
        user_id: str,
        filename: str,
        data: dict
    ) -> bool:
        """
        사용자의 특정 파일 업데이트

        Args:
            user_id: 사용자 ID
            filename: 파일명
            data: 저장할 JSON 데이터

        Returns:
            bool: 성공 여부
        """
        if filename not in self.COST_FILES:
            print(f"⚠️ 허용되지 않은 파일명: {filename}")
            return False

        user_costs_dir = self.get_user_costs_dir(user_id)
        user_file = user_costs_dir / filename

        try:
            # 디렉토리가 없으면 생성
            user_costs_dir.mkdir(parents=True, exist_ok=True)

            # JSON 파일 저장
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"✅ {filename} 업데이트 완료")
            return True

        except Exception as e:
            print(f"❌ 파일 업데이트 실패 ({filename}): {e}")
            return False

    def load_user_file(
        self,
        user_id: str,
        filename: str,
        fallback_to_template: bool = True
    ) -> Optional[dict]:
        """
        사용자의 특정 파일 로드

        Args:
            user_id: 사용자 ID
            filename: 파일명
            fallback_to_template: True면 사용자 파일이 없을 때 템플릿 파일 사용

        Returns:
            dict: JSON 데이터 또는 None
        """
        if filename not in self.COST_FILES:
            print(f"⚠️ 허용되지 않은 파일명: {filename}")
            return None

        user_costs_dir = self.get_user_costs_dir(user_id)
        user_file = user_costs_dir / filename

        # 사용자 파일 시도
        if user_file.exists():
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 사용자 파일 로드 실패 ({filename}): {e}")

        # 템플릿 파일 fallback
        if fallback_to_template:
            template_file = self.template_dir / filename
            if template_file.exists():
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"❌ 템플릿 파일 로드 실패 ({filename}): {e}")

        return None

    def load_material_cost_premiums(
        self,
        user_id: str,
        fallback_to_template: bool = True
    ) -> dict:
        """
        재활용재/저탄소메탈 비용 프리미엄 데이터 로드

        Args:
            user_id: 사용자 ID
            fallback_to_template: 사용자 파일이 없을 때 템플릿 사용 여부

        Returns:
            dict: {원소: {"recycle_premium_pct": float, "low_carbon_premium_pct": float}}
        """
        raw_data = self.load_user_file(user_id, "material_cost_premiums.json", fallback_to_template)

        if not raw_data:
            # 기본값 반환
            return {
                "Ni": {"recycle_premium_pct": 30.0, "low_carbon_premium_pct": 50.0},
                "Co": {"recycle_premium_pct": 40.0, "low_carbon_premium_pct": 60.0},
                "Li": {"recycle_premium_pct": 20.0, "low_carbon_premium_pct": 40.0},
                "default": {"recycle_premium_pct": 30.0, "low_carbon_premium_pct": 50.0}
            }

        # 배열 형태를 딕셔너리로 변환
        premiums = {}
        for item in raw_data:
            element = item.get("원소")
            premiums[element] = {
                "recycle_premium_pct": item.get("재활용재_프리미엄(%)", 30.0),
                "low_carbon_premium_pct": item.get("저탄소메탈_프리미엄(%)", 50.0)
            }

        return premiums

    def save_material_cost_premiums(
        self,
        user_id: str,
        premiums: dict
    ) -> bool:
        """
        재활용재/저탄소메탈 비용 프리미엄 데이터 저장

        Args:
            user_id: 사용자 ID
            premiums: {원소: {"recycle_premium_pct": float, "low_carbon_premium_pct": float}}

        Returns:
            bool: 성공 여부
        """
        # 딕셔너리를 배열 형태로 변환
        raw_data = []
        for element, values in premiums.items():
            raw_data.append({
                "원소": element,
                "재활용재_프리미엄(%)": values.get("recycle_premium_pct", 30.0),
                "저탄소메탈_프리미엄(%)": values.get("low_carbon_premium_pct", 50.0),
                "설명": f"{element} 원소의 재활용재/저탄소메탈 비용 프리미엄 (신재 대비)"
            })

        return self.update_user_file(user_id, "material_cost_premiums.json", raw_data)


# 편의 함수
def initialize_user_optimization_costs(user_id: str, force: bool = False) -> bool:
    """
    사용자별 optimization_costs 초기화 (편의 함수)

    Args:
        user_id: 사용자 ID
        force: True면 기존 파일 덮어쓰기

    Returns:
        bool: 성공 여부
    """
    manager = OptimizationCostsManager()
    return manager.initialize_user_costs(user_id, force=force)


def get_user_optimization_costs_path(user_id: str) -> str:
    """
    사용자별 optimization_costs 경로 반환 (편의 함수)

    Args:
        user_id: 사용자 ID

    Returns:
        str: 사용자별 optimization_costs 디렉토리 경로
    """
    manager = OptimizationCostsManager()
    return manager.get_user_costs_path(user_id)

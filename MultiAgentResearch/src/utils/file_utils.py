import os
import shutil
import logging

logger = logging.getLogger(__name__)

def remove_artifact_folder(folder_path: str = "./artifacts/") -> None:
    """
    ./artifact/ 폴더가 존재하면 삭제하는 함수
    
    Args:
        folder_path (str): 삭제할 폴더 경로
    """
    if os.path.exists(folder_path):
        logger.info(f"'{folder_path}' 폴더를 삭제합니다...")
        try:
            shutil.rmtree(folder_path)
            logger.info(f"'{folder_path}' 폴더가 성공적으로 삭제되었습니다.")
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            raise
    else:
        logger.info(f"'{folder_path}' 폴더가 존재하지 않습니다.")
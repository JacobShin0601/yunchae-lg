"""
Unified file operations module for PCF optimization project.
Consolidates all file I/O operations to avoid code duplication.
"""

import json
import pandas as pd
from typing import Any, Dict, Optional, Union
import os
from pathlib import Path


class FileOperationError(Exception):
    """Base exception for file operation errors"""
    pass


class FileLoadError(FileOperationError):
    """Exception raised when file loading fails"""
    pass


class FileSaveError(FileOperationError):
    """Exception raised when file saving fails"""
    pass


class FileOperations:
    """Centralized file operations handler"""
    
    @staticmethod
    def get_user_file_path(file_path: Union[str, Path], user_id: str) -> Path:
        """
        사용자별 파일 경로를 생성합니다.
        모든 폴더(stable_var, input, data)에 대해 사용자별 하위폴더를 생성합니다.
        
        Args:
            file_path: 기본 파일 경로
            user_id: 사용자 ID
            
        Returns:
            사용자별 파일 경로
        """
        path_obj = Path(file_path)
        
        # 모든 폴더에 대해 하위폴더 방식 적용: folder/user_id/filename.ext
        user_dir = path_obj.parent / user_id
        return user_dir / path_obj.name
    
    @staticmethod
    def load_json(file_path: Union[str, Path], default: Optional[Dict] = None, encoding: str = 'utf-8', user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load JSON file with error handling.
        사용자별 파일이 없을 경우 기본 파일을 복사해서 생성합니다.
        
        Args:
            file_path: Path to JSON file
            default: Default value to return if file doesn't exist
            encoding: File encoding (default: utf-8)
            user_id: 사용자 ID (있으면 사용자별 파일 경로로 변환)
            
        Returns:
            Loaded JSON data as dictionary
            
        Raises:
            FileLoadError: If file loading fails and no default is provided
        """
        if user_id:
            # 사용자별 파일 경로로 변환
            actual_path = FileOperations.get_user_file_path(file_path, user_id)
            
            # 사용자별 파일이 없으면 기본 파일을 복사
            if not os.path.exists(actual_path):
                try:
                    # 기본 파일 로드
                    with open(file_path, 'r', encoding=encoding) as f:
                        default_data = json.load(f)
                    
                    # 사용자별 디렉토리 생성
                    os.makedirs(os.path.dirname(actual_path), exist_ok=True)
                    
                    # 사용자별 파일로 저장
                    with open(actual_path, 'w', encoding=encoding) as f:
                        json.dump(default_data, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    # 기본 파일 복사에 실패하면 기본 파일 사용
                    actual_path = file_path
        else:
            actual_path = file_path
        
        try:
            with open(actual_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except FileNotFoundError:
            if default is not None:
                return default
            raise FileLoadError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise FileLoadError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            if default is not None:
                return default
            raise FileLoadError(f"Failed to load {file_path}: {e}")
    
    @staticmethod
    def save_json(file_path: Union[str, Path], data: Dict[str, Any], encoding: str = 'utf-8', indent: int = 2, user_id: Optional[str] = None) -> bool:
        """
        Save dictionary as JSON file.
        
        Args:
            file_path: Path to save JSON file
            data: Dictionary to save
            encoding: File encoding (default: utf-8)
            indent: JSON indentation (default: 2)
            user_id: 사용자 ID (있으면 사용자별 파일 경로로 변환)
            
        Returns:
            True if successful
            
        Raises:
            FileSaveError: If file saving fails
        """
        # 사용자 ID가 있으면 사용자별 파일 경로로 변환
        actual_path = FileOperations.get_user_file_path(file_path, user_id) if user_id else file_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(actual_path), exist_ok=True)
            
            with open(actual_path, 'w', encoding=encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            return True
        except Exception as e:
            raise FileSaveError(f"Failed to save {file_path}: {e}")
    
    @staticmethod
    def load_csv(file_path: Union[str, Path], user_id: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load CSV file as pandas DataFrame.
        사용자별 파일이 없을 경우 기본 파일을 복사해서 생성합니다.
        
        Args:
            file_path: Path to CSV file
            user_id: 사용자 ID (있으면 사용자별 파일 경로로 변환)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileLoadError: If file loading fails
        """
        if user_id:
            # 사용자별 파일 경로로 변환
            actual_path = FileOperations.get_user_file_path(file_path, user_id)
            
            # 사용자별 파일이 없으면 기본 파일을 복사
            if not os.path.exists(actual_path):
                try:
                    # 기본 파일 로드
                    default_df = pd.read_csv(file_path, **kwargs)
                    
                    # 사용자별 디렉토리 생성
                    os.makedirs(os.path.dirname(actual_path), exist_ok=True)
                    
                    # 사용자별 파일로 저장 (index=False 기본값)
                    save_kwargs = kwargs.copy()
                    if 'index' not in save_kwargs:
                        save_kwargs['index'] = False
                    default_df.to_csv(actual_path, **save_kwargs)
                    
                    return default_df
                        
                except Exception as e:
                    # 기본 파일 복사에 실패하면 기본 파일 사용
                    actual_path = file_path
        else:
            actual_path = file_path
        
        try:
            return pd.read_csv(actual_path, **kwargs)
        except FileNotFoundError:
            raise FileLoadError(f"CSV file not found: {file_path}")
        except Exception as e:
            raise FileLoadError(f"Failed to load CSV {file_path}: {e}")
    
    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: Union[str, Path], user_id: Optional[str] = None, **kwargs) -> bool:
        """
        Save DataFrame as CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save CSV file
            **kwargs: Additional arguments for df.to_csv
            
        Returns:
            True if successful
            
        Raises:
            FileSaveError: If file saving fails
        """
        # 사용자 ID가 있으면 사용자별 파일 경로로 변환
        actual_path = FileOperations.get_user_file_path(file_path, user_id) if user_id else file_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(actual_path), exist_ok=True)
            
            # Set default index=False if not specified
            if 'index' not in kwargs:
                kwargs['index'] = False
                
            df.to_csv(actual_path, **kwargs)
            return True
        except Exception as e:
            raise FileSaveError(f"Failed to save CSV {file_path}: {e}")
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """Check if file exists."""
        return os.path.exists(file_path)
    
    @staticmethod
    def create_directory(dir_path: Union[str, Path]) -> bool:
        """
        Create directory if it doesn't exist.
        
        Args:
            dir_path: Directory path to create
            
        Returns:
            True if directory was created or already exists
        """
        try:
            os.makedirs(dir_path, exist_ok=True)
            return True
        except Exception as e:
            raise FileOperationError(f"Failed to create directory {dir_path}: {e}")
    
    @staticmethod
    def get_file_path(base_dir: Union[str, Path], *path_components) -> Path:
        """
        Construct file path from base directory and components.
        
        Args:
            base_dir: Base directory
            *path_components: Path components to join
            
        Returns:
            Complete file path as Path object
        """
        return Path(base_dir).joinpath(*path_components)


# Convenience functions for backward compatibility
def load_json_file(file_path: Union[str, Path], default: Optional[Dict] = None) -> Dict[str, Any]:
    """Legacy function for loading JSON files."""
    return FileOperations.load_json(file_path, default)


def save_json_file(file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
    """Legacy function for saving JSON files."""
    return FileOperations.save_json(file_path, data)
from dataclasses import dataclass

@dataclass
class AsyncConfig:
    """비동기 처리 설정"""
    max_concurrent_tasks: int = 5
    batch_size: int = 10
    chunk_size: int = 1000
    timeout: int = 300
    retry_attempts: int = 3
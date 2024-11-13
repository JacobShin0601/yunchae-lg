# src/local_utils/__init__.py

import logging

# 패키지 초기화
logging.basicConfig(level=logging.INFO)

# 버전 정보
__version__ = '1.0.0'

# 하위 모듈 노출
__all__ = ['bedrock', 'chat', 'chunk', 'common_utils', 'opensearch', 'rag_streamlit', 'ssm']

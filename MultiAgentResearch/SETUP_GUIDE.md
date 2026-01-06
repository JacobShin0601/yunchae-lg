# Bedrock Manus 설정 가이드

## 환경 변수 설정

### 1. .env 파일 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# LLM Configuration
REASONING_MODEL=o1-mini
REASONING_API_KEY=your_openai_api_key_here

BASIC_MODEL=gpt-4o
BASIC_API_KEY=your_openai_api_key_here

VL_MODEL=gpt-4o
VL_API_KEY=your_openai_api_key_here

# Tavily Search API (필수)
TAVILY_API_KEY=your_tavily_api_key_here

# Browser Configuration
BROWSER_HEADLESS=true
PYTHONHTTPSVERIFY=0
CURL_CA_BUNDLE=
```

### 2. API 키 발급

#### OpenAI API 키
1. [OpenAI Platform](https://platform.openai.com/api-keys)에 접속
2. 새 API 키 생성
3. `.env` 파일의 `*_API_KEY` 항목에 입력

#### Tavily API 키 (검색 기능용)
1. [Tavily](https://tavily.com/)에 접속
2. 계정 생성 후 API 키 발급
3. `.env` 파일의 `TAVILY_API_KEY`에 입력

## Tavily Tool 문제 해결

### 문제 증상
```
Tavily search tool is not available. Please check TAVILY_API_KEY environment variable.
```

### 해결 방법

1. **API 키 확인**
   ```bash
   # 환경 변수가 설정되었는지 확인
   echo $TAVILY_API_KEY
   ```

2. **.env 파일 확인**
   - 파일이 프로젝트 루트에 있는지 확인
   - `TAVILY_API_KEY=` 뒤에 실제 키가 입력되었는지 확인
   - 키 앞뒤에 공백이나 따옴표가 없는지 확인

3. **Jupyter 노트북 재시작**
   - 환경 변수 변경 후 커널 재시작 필요

### 대안 방법

Tavily tool이 작동하지 않는 경우:

1. **직접 URL 제공**
   ```python
   # 연구하고 싶은 특정 URL을 제공
   user_query = "다음 URL의 내용을 분석해줘: https://example.com/article"
   ```

2. **Crawl Tool 사용**
   - researcher가 crawl_tool을 사용하여 특정 웹페이지 분석 가능

3. **수동 정보 제공**
   - 필요한 정보를 직접 텍스트로 제공하여 분석 요청

## 테스트 방법

### 1. 기본 테스트
```python
import asyncio
from main import execute_query

# 간단한 인사말 테스트 (coordinator 테스트)
result = await execute_query("안녕하세요")
```

### 2. 연구 기능 테스트
```python
# 검색이 필요한 쿼리 (researcher 테스트)
result = await execute_query("최신 AI 기술 동향에 대해 조사해줘")
```

### 3. 코딩 기능 테스트
```python
# 계산이나 코딩이 필요한 쿼리 (coder 테스트)
result = await execute_query("1부터 100까지의 합을 계산하고 그래프로 보여줘")
```

## 로그 확인

디버그 모드로 실행하여 상세한 로그 확인:

```python
from src.workflow import run_agent_workflow_async

result = await run_agent_workflow_async(
    user_input="테스트 쿼리",
    debug=True  # 디버그 모드 활성화
)
```

## 문제 해결 체크리스트

- [ ] `.env` 파일이 프로젝트 루트에 있는가?
- [ ] `TAVILY_API_KEY`가 올바르게 설정되었는가?
- [ ] OpenAI API 키가 유효한가?
- [ ] 필요한 Python 패키지가 모두 설치되었는가?
- [ ] Jupyter 커널을 재시작했는가?

## 지원

문제가 지속되면 다음 정보와 함께 문의하세요:
- 에러 메시지 전문
- 사용 중인 Python 버전
- 설치된 패키지 버전 (`pip list`)
- 환경 변수 설정 상태 (API 키 제외) 
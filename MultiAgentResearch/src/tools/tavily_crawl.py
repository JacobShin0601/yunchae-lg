import os
import requests
import json
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def handle_tavily_crawl_tool(
    url: str,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    instructions: Optional[str] = None,
    select_paths: Optional[List[str]] = None,
    select_domains: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    allow_external: bool = False,
    include_images: bool = False,
    categories: Optional[List[str]] = None,
    extract_depth: str = "basic"
) -> str:
    """
    Tavily Crawl API를 사용하여 웹사이트를 그래프처럼 탐색합니다.
    
    Args:
        url: 크롤링을 시작할 기본 URL
        max_depth: 최대 탐색 깊이 (기본값: 1)
        max_breadth: 각 레벨에서 탐색할 최대 페이지 수 (기본값: 20)
        limit: 반환할 최대 결과 수 (기본값: 50)
        instructions: 크롤링 지침 (예: "Python SDK")
        select_paths: 포함할 경로 패턴 리스트
        select_domains: 포함할 도메인 리스트
        exclude_paths: 제외할 경로 패턴 리스트
        exclude_domains: 제외할 도메인 리스트
        allow_external: 외부 도메인 허용 여부 (기본값: False)
        include_images: 이미지 포함 여부 (기본값: False)
        categories: 카테고리 필터
        extract_depth: 추출 깊이 ("basic" 또는 "advanced")
    
    Returns:
        크롤링 결과를 포함한 문자열
    """
    try:
        api_key = os.environ.get('TAVILY_API_KEY')
        if not api_key:
            return "❌ TAVILY_API_KEY 환경변수가 설정되지 않았습니다."
        
        # API 엔드포인트
        api_url = "https://api.tavily.com/crawl"
        
        # 요청 페이로드 구성
        payload = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "allow_external": allow_external,
            "include_images": include_images,
            "extract_depth": extract_depth
        }
        
        # 선택적 매개변수 추가
        if instructions:
            payload["instructions"] = instructions
        if select_paths:
            payload["select_paths"] = select_paths
        if select_domains:
            payload["select_domains"] = select_domains
        if exclude_paths:
            payload["exclude_paths"] = exclude_paths
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if categories:
            payload["categories"] = categories
        
        # API 요청 헤더
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Tavily Crawl 요청: {url} (깊이: {max_depth}, 너비: {max_breadth})")
        
        # API 호출
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return format_crawl_results(data, url)
        elif response.status_code == 401:
            return "❌ Tavily API 인증 실패. API 키를 확인해주세요."
        elif response.status_code == 429:
            return "❌ Tavily API 요청 한도 초과. 잠시 후 다시 시도해주세요."
        elif response.status_code == 432:
            return "❌ Tavily API 크레딧 부족. 크레딧을 충전해주세요."
        else:
            return f"❌ Tavily Crawl API 오류 (상태 코드: {response.status_code}): {response.text}"
            
    except requests.exceptions.Timeout:
        return "❌ Tavily Crawl API 요청 시간 초과"
    except requests.exceptions.RequestException as e:
        return f"❌ Tavily Crawl API 요청 오류: {str(e)}"
    except Exception as e:
        logger.error(f"Tavily Crawl 도구 오류: {str(e)}")
        return f"❌ Tavily Crawl 도구 실행 중 오류 발생: {str(e)}"

def format_crawl_results(data: Dict[str, Any], base_url: str) -> str:
    """크롤링 결과를 포맷팅합니다."""
    try:
        results = []
        results.append(f"🕷️ **Tavily Crawl 결과** - 기본 URL: {base_url}\n")
        
        if "base_url" in data:
            results.append(f"**기본 URL**: {data['base_url']}")
        
        if "response_time" in data:
            results.append(f"**응답 시간**: {data['response_time']}초")
        
        if "results" in data and data["results"]:
            results.append(f"**발견된 페이지 수**: {len(data['results'])}\n")
            
            for i, result in enumerate(data["results"], 1):
                results.append(f"### 📄 페이지 {i}: {result.get('url', 'URL 없음')}")
                
                if "raw_content" in result:
                    content = result["raw_content"]
                    # 내용이 너무 길면 요약
                    if len(content) > 2000:
                        content = content[:2000] + "...\n\n[내용이 길어 일부만 표시됨]"
                    results.append(f"**내용**:\n{content}\n")
                
                results.append("---\n")
        else:
            results.append("❌ 크롤링 결과가 없습니다.")
        
        return '\n'.join(results)
        
    except Exception as e:
        return f"❌ 크롤링 결과 포맷팅 오류: {str(e)}"

def get_crawl_suggestions(query: str, url: str) -> Dict[str, Any]:
    """쿼리와 URL을 기반으로 크롤링 매개변수 제안을 생성합니다."""
    suggestions = {
        "max_depth": 1,
        "max_breadth": 20,
        "limit": 50,
        "extract_depth": "basic"
    }
    
    # 쿼리 기반 최적화
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ["깊이", "상세", "전체", "모든"]):
        suggestions["max_depth"] = 2
        suggestions["max_breadth"] = 30
        suggestions["extract_depth"] = "advanced"
    
    if any(keyword in query_lower for keyword in ["빠른", "간단", "요약"]):
        suggestions["max_depth"] = 1
        suggestions["max_breadth"] = 10
        suggestions["limit"] = 20
    
    if any(keyword in query_lower for keyword in ["문서", "가이드", "튜토리얼"]):
        suggestions["instructions"] = "documentation and guides"
    
    if any(keyword in query_lower for keyword in ["API", "개발", "코드"]):
        suggestions["instructions"] = "API documentation and code examples"
    
    # URL 기반 최적화
    if "docs." in url or "documentation" in url:
        suggestions["instructions"] = "technical documentation"
        suggestions["select_paths"] = ["/docs/", "/documentation/", "/guide/"]
    
    if "blog" in url or "news" in url:
        suggestions["instructions"] = "latest articles and news"
        suggestions["max_depth"] = 1
    
    return suggestions 
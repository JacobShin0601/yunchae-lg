import json
import logging
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)

# Tavily API 키 확인
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Tavily search tool with logging
try:
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not found in environment variables")
        tavily_tool = None
    else:
        LoggedTavilySearch = create_logged_tool(TavilySearchResults)
        tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)
        logger.info("Tavily search tool initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Tavily search tool: {e}")
    tavily_tool = None


def handle_tavily_tool(query, include_domains=None):
    '''
    Use this tool to search the internet for real-time information, current events, or specific data. Provides relevant search results from Tavily's search engine API.
    
    Args:
        query (str): Search query
        include_domains (list): Optional list of domains to focus search on
    '''
    if tavily_tool is None:
        error_msg = "Tavily search tool is not available. Please check TAVILY_API_KEY environment variable."
        logger.error(error_msg)
        return f"\n\n# Search Error\n\n{error_msg}\n\n대안으로 다른 검색 방법을 사용하거나 직접 URL을 제공해주세요."
    
    try:
        # 도메인 제한이 있는 경우 쿼리 수정
        search_query = query
        if include_domains and len(include_domains) > 0:
            # site: 연산자를 사용하여 특정 도메인에서 검색
            domain_queries = []
            for domain in include_domains:
                # URL에서 도메인만 추출
                if domain.startswith('http'):
                    import re
                    domain_match = re.search(r'https?://([^/]+)', domain)
                    domain = domain_match.group(1) if domain_match else domain
                domain_queries.append(f"site:{domain}")
            
            # 여러 도메인이 있는 경우 OR 연산자로 연결
            if len(domain_queries) > 1:
                search_query = f"({' OR '.join(domain_queries)}) {query}"
            else:
                search_query = f"{domain_queries[0]} {query}"
            
            logger.info(f"Domain-focused search query: {search_query}")
        
        searched_content = tavily_tool.invoke({"query": search_query})
        
        # Debug: 결과 타입과 내용 확인
        logger.debug(f"Tavily search result type: {type(searched_content)}")
        logger.debug(f"Tavily search result: {searched_content}")
        
        # 결과가 None이거나 빈 경우
        if not searched_content:
            return f"\n\n# Search Results\n\n검색 결과가 없습니다. 다른 검색어를 시도해보세요."
        
        # 결과가 문자열인 경우 JSON으로 파싱 시도
        if isinstance(searched_content, str):
            try:
                searched_content = json.loads(searched_content)
            except json.JSONDecodeError:
                # JSON 파싱 실패시 원본 문자열 반환
                return f"\n\n# Search Results\n\n{searched_content}"
        
        # 결과가 리스트인 경우 처리
        if isinstance(searched_content, list):
            try:
                if not searched_content:
                    return f"\n\n# Search Results\n\n검색 결과가 없습니다."
                
                results = f"\n\n# Relative Search Results\n\n{json.dumps([{'title': elem['title'], 'url': elem['url'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
                print(f'Search Results: \n\n {[{"title": elem["title"], "url": elem["url"], "content": elem["content"]} for elem in searched_content]}')
                return results
            except (KeyError, TypeError) as e:
                logger.error(f"Error processing search results: {e}")
                return f"\n\n# Search Results\n\n{str(searched_content)}"
        
        # 기타 경우 문자열로 변환하여 반환
        return f"\n\n# Search Results\n\n{str(searched_content)}"
        
    except Exception as e:
        error_msg = f"Tavily search failed: {str(e)}"
        logger.error(error_msg)
        return f"\n\n# Search Error\n\n{error_msg}\n\n대안으로 crawl_tool을 사용하여 특정 URL의 내용을 분석하거나, 다른 검색 방법을 시도해보세요."
    
    
    

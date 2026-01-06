from typing import Dict, Any, List
# from src.tools.crawl import handle_crawl_tool
from src.tools.search import handle_tavily_tool
from src.tools.bash_tool import handle_bash_tool
from src.tools.python_repl import handle_python_repl_tool
from src.tools.tavily_crawl import handle_tavily_crawl_tool, get_crawl_suggestions
import re

def extract_sites_from_query(query: str) -> List[str]:
    """Extract website URLs and domain names from user query"""
    sites = []
    
    # URL 패턴 매칭
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, query)
    
    for url in urls:
        # URL에서 도메인 추출
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            # 한국어 문자나 특수문자가 포함된 경우 제거
            if not re.search(r'[가-힣]', domain) and domain not in sites:
                sites.append(domain)
    
    # 도메인 패턴 매칭 (www.example.com, example.com 형태)
    domain_pattern = r'(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})'
    domains = re.findall(domain_pattern, query)
    
    for domain in domains:
        # 한국어 문자가 포함되지 않고 중복이 아닌 경우만 추가
        if not re.search(r'[가-힣]', domain) and domain not in sites:
            sites.append(domain)
    
    # 특정 사이트 이름 매핑
    site_mappings = {
        'Energy Storage News': 'energy-storage.news',
        'PV Magazine': 'pv-magazine.com',
        'Utility Dive': 'utilitydive.com',
        'Electrek': 'electrek.co',
        'Renew Economy': 'reneweconomy.com.au',
        'SNE 리서치': 'sneresearch.com',
        'PV Tech News': 'pv-tech.org',
        'Cleantechnica': 'cleantechnica.com',
        '전기신문': 'electimes.com',
        '파이낸셜뉴스': 'fnnews.com',
        'BNEF News': 'bnef.com',
        'SolarQuotes': 'solarquotes.com.au',
        'One Step off the Grid': 'onestepoffthegrid.com.au',
        'ITP Renewable': 'batterytestcentre.com.au',
        'JEMA': 'jema-net.or.jp',
        'JPEA': 'jpea.gr.jp',
        'SII': 'sii.or.jp',
        'Tesla Powerwall': 'tesla.com',
        'BYD': 'bydbatterybox.com',
        '에너지경제연구원': 'keei.re.kr',
        '배터리다이브': 'batterydive.com'
    }
    
    for site_name, domain in site_mappings.items():
        if site_name.lower() in query.lower():
            if domain not in sites:
                sites.append(domain)
    
    return sites

tool_list = [
    # {
    #     "toolSpec": {
    #         "name": "crawl_tool",
    #         "description": "Use this to crawl a url and get a readable content in markdown format.",
    #         "inputSchema": {
    #             "json": {
    #                 "type": "object",
    #                 "properties": {
    #                     "url": {
    #                         "type": "string",
    #                         "description": "The url to crawl."
    #                     }
    #                 },
    #                 "required": ["url"]
    #             }
    #         }
    #     }
    # },
    {
        "toolSpec": {
            "name": "tavily_tool",
            "description": "Use this tool to search the internet for real-time information, current events, or specific data. This is the primary tool for information gathering and analysis. Provides relevant search results from Tavily's search engine API.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the internet. Make it specific and focused on information gathering."
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of specific domains to focus the search on (e.g., ['energy-storage.news', 'pv-magazine.com'])"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "site_focused_search",
            "description": "Search for information with focus on specific websites provided by the user. Use this when user provides specific site links or when you need to search within particular domains.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "target_sites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of specific websites/domains to search within"
                        },
                        "fallback_search": {
                            "type": "boolean",
                            "description": "Whether to perform general search if site-specific search yields insufficient results",
                            "default": True
                        }
                    },
                    "required": ["query", "target_sites"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "python_repl_tool",
            "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to execute to do further analysis or calculation."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "bash_tool",
            "description": "Use this to execute bash command and do necessary operations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The bash command to be executed."
                        }
                    },
                    "required": ["cmd"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "save_research_data",
            "description": "Save collected research data to a structured file for later use by Reporter Agent. Use this tool to store important findings, sources, and analysis results.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title or topic of the research data"
                        },
                        "content": {
                            "type": "string",
                            "description": "The research content to save"
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of sources/URLs for the research data"
                        },
                        "category": {
                            "type": "string",
                            "description": "Category of the research (e.g., 'market_analysis', 'policy_research', 'industry_trends')"
                        }
                    },
                    "required": ["title", "content"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "smart_search",
            "description": "Automatically detect target sites from user query and perform focused search. Use this as the primary search tool - it will automatically choose between site-focused search and general search based on user input.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "user_query": {
                            "type": "string",
                            "description": "The original user query to extract site preferences from"
                        }
                    },
                    "required": ["query", "user_query"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "latest_news_search",
            "description": "Search for the latest news and current information using multiple strategies. This tool prioritizes recent content and uses JINA for deep content analysis of found articles.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for latest news and information"
                        },
                        "time_filter": {
                            "type": "string",
                            "description": "Time filter for search results",
                            "enum": ["today", "week", "month", "all"],
                            "default": "week"
                        },
                        "max_articles": {
                            "type": "integer",
                            "description": "Maximum number of articles to analyze in detail",
                            "default": 5
                        },
                        "focus_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of news domains to prioritize (e.g., ['reuters.com', 'bloomberg.com'])"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "deep_article_analysis",
            "description": "Use JINA to perform deep analysis of specific news articles or web pages. This tool extracts full content and provides detailed analysis.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs to analyze in detail"
                        },
                        "analysis_focus": {
                            "type": "string",
                            "description": "Specific aspect to focus on during analysis (e.g., 'market_trends', 'policy_changes', 'technology_updates')"
                        }
                    },
                    "required": ["urls"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "tavily_crawl_tool",
            "description": "Use Tavily Crawl API to traverse a website like a graph starting from a base URL. This tool is perfect for comprehensive site exploration, documentation crawling, and gathering structured information from multiple related pages. It can intelligently navigate through site hierarchies and extract relevant content based on instructions.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The base URL to start crawling from (e.g., 'docs.example.com' or 'https://docs.example.com')"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth to crawl (1-3, default: 1). Higher values explore deeper into the site hierarchy.",
                            "default": 1,
                            "minimum": 1,
                            "maximum": 3
                        },
                        "max_breadth": {
                            "type": "integer",
                            "description": "Maximum number of pages to explore at each level (default: 20)",
                            "default": 20,
                            "minimum": 1,
                            "maximum": 50
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum total number of pages to return (default: 50)",
                            "default": 50,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Specific instructions for what to focus on during crawling (e.g., 'API documentation', 'Python SDK', 'latest news')"
                        },
                        "select_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of path patterns to include (e.g., ['/docs/', '/api/', '/guide/'])"
                        },
                        "exclude_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of path patterns to exclude (e.g., ['/admin/', '/private/'])"
                        },
                        "extract_depth": {
                            "type": "string",
                            "description": "Depth of content extraction",
                            "enum": ["basic", "advanced"],
                            "default": "basic"
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "smart_crawl_tool",
            "description": "Intelligently crawl a website with automatically optimized parameters based on the user query and URL. This tool analyzes the query context and URL type to suggest optimal crawling settings.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The base URL to start crawling from"
                        },
                        "query": {
                            "type": "string",
                            "description": "The user's search query to optimize crawling parameters"
                        },
                        "crawl_type": {
                            "type": "string",
                            "description": "Type of crawling needed",
                            "enum": ["quick", "comprehensive", "documentation", "news"],
                            "default": "quick"
                        }
                    },
                    "required": ["url", "query"]
                }
            }
        }
    }
]

def handle_site_focused_search(query: str, target_sites: List[str], fallback_search: bool = True) -> str:
    """Handle site-focused search with fallback to general search"""
    import re
    
    try:
        results = []
        
        # 각 타겟 사이트에 대해 site: 연산자를 사용한 검색 수행
        for site in target_sites:
            # URL에서 도메인 추출
            domain = site
            if site.startswith('http'):
                domain = re.search(r'https?://([^/]+)', site)
                domain = domain.group(1) if domain else site
            
            # site: 연산자를 사용한 검색 쿼리 생성
            site_query = f"site:{domain} {query}"
            
            try:
                site_results = handle_tavily_tool(query=site_query)
                if site_results and "결과를 찾을 수 없습니다" not in site_results:
                    results.append(f"=== {domain}에서 검색 결과 ===\n{site_results}\n")
            except Exception as e:
                results.append(f"=== {domain} 검색 중 오류 발생: {str(e)} ===\n")
        
        # 사이트별 검색 결과가 충분하지 않은 경우 일반 검색 수행
        if fallback_search and (not results or len(''.join(results)) < 500):
            try:
                general_results = handle_tavily_tool(query=query)
                if general_results:
                    results.append(f"=== 일반 검색 결과 ===\n{general_results}\n")
            except Exception as e:
                results.append(f"=== 일반 검색 중 오류 발생: {str(e)} ===\n")
        
        if results:
            return '\n'.join(results)
        else:
            return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."
            
    except Exception as e:
        return f"사이트 집중 검색 중 오류 발생: {str(e)}"

def handle_save_research_data(title: str, content: str, sources: List[str] = None, category: str = "general") -> str:
    """Save research data to structured file"""
    import os
    from datetime import datetime
    
    try:
        # artifacts 디렉토리 생성
        os.makedirs("./artifacts", exist_ok=True)
        
        # 연구 데이터 파일 경로
        research_file = './artifacts/research_data.txt'
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 기존 파일이 있으면 추가, 없으면 새로 생성
        mode = 'a' if os.path.exists(research_file) else 'w'
        
        with open(research_file, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write("# 연구 데이터 수집 결과\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"## 연구 주제: {title}\n")
            f.write(f"## 카테고리: {category}\n")
            f.write(f"## 수집 시간: {current_time}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("### 연구 내용:\n")
            f.write(content)
            f.write("\n\n")
            
            if sources:
                f.write("### 출처:\n")
                for i, source in enumerate(sources, 1):
                    f.write(f"{i}. {source}\n")
                f.write("\n")
            
            f.write("-" * 80 + "\n\n")
        
        return f"Research data successfully saved to {research_file}"
        
    except Exception as e:
        return f"Error saving research data: {str(e)}"

research_tool_config = {
    "tools": tool_list,
    # "toolChoice": {
    #    "tool": {
    #        "name": "summarize_email"
    #    }
    # }
}

def process_search_tool(tool) -> str:
    """Process a tool invocation
    
    Args:
        tool_name: Name of the tool to invoke
        tool_input: Input parameters for the tool
        
    Returns:
        Result of the tool invocation as a string
    """
    
    tool_name, tool_input = tool['name'], tool['input']
    
    if tool_name == "tavily_tool":
        # Create a new instance of the Tavily search tool
        results = handle_tavily_tool(query=tool_input["query"], include_domains=tool_input.get("include_domains", []))
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
        #return response
    # elif tool_name == "crawl_tool":
    #     results = handle_crawl_tool(tool_input)
    #     tool_result = {
    #         "toolUseId": tool['toolUseId'],
    #         "content": [{"json": {"text": results}}]
    #     }
    elif tool_name == "python_repl_tool":
        # Create a new instance of the Tavily search tool
        results = handle_python_repl_tool(code=tool_input["code"])
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
        #return response
    elif tool_name == "bash_tool":
        results = handle_bash_tool(cmd=tool_input["cmd"])
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "save_research_data":
        results = handle_save_research_data(
            title=tool_input["title"],
            content=tool_input["content"],
            sources=tool_input.get("sources", []),
            category=tool_input.get("category", "general")
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "site_focused_search":
        results = handle_site_focused_search(
            query=tool_input["query"],
            target_sites=tool_input["target_sites"],
            fallback_search=tool_input.get("fallback_search", True)
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "smart_search":
        results = handle_smart_search(
            query=tool_input["query"],
            user_query=tool_input["user_query"]
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "latest_news_search":
        results = handle_latest_news_search(
            query=tool_input["query"],
            time_filter=tool_input.get("time_filter", "week"),
            max_articles=tool_input.get("max_articles", 5),
            focus_domains=tool_input.get("focus_domains", None)
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "deep_article_analysis":
        results = handle_deep_article_analysis(
            urls=tool_input["urls"],
            analysis_focus=tool_input.get("analysis_focus", "general")
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "tavily_crawl_tool":
        results = handle_tavily_crawl_tool(
            url=tool_input["url"],
            max_depth=tool_input.get("max_depth", 1),
            max_breadth=tool_input.get("max_breadth", 20),
            limit=tool_input.get("limit", 50),
            instructions=tool_input.get("instructions", None),
            select_paths=tool_input.get("select_paths", None),
            exclude_paths=tool_input.get("exclude_paths", None),
            extract_depth=tool_input.get("extract_depth", "basic")
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "smart_crawl_tool":
        results = handle_smart_crawl_tool(
            url=tool_input["url"],
            query=tool_input["query"],
            crawl_type=tool_input.get("crawl_type", "quick")
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    else:
        print (f"Unknown tool: {tool_name}")
        
    results = {"role": "user","content": [{"toolResult": tool_result}]}
    
    return results

def handle_smart_search(query: str, user_query: str) -> str:
    """Smart search that automatically detects target sites and performs appropriate search"""
    try:
        # 사용자 쿼리에서 사이트 추출
        target_sites = extract_sites_from_query(user_query)
        
        if target_sites:
            # 특정 사이트가 감지된 경우 사이트 집중 검색 수행
            result = f"🎯 감지된 타겟 사이트: {', '.join(target_sites)}\n\n"
            result += handle_site_focused_search(query, target_sites, fallback_search=True)
            return result
        else:
            # 특정 사이트가 없는 경우 일반 검색 수행
            result = "🌐 일반 검색을 수행합니다.\n\n"
            result += handle_tavily_tool(query)
            return result
            
    except Exception as e:
        return f"스마트 검색 중 오류 발생: {str(e)}"

def handle_latest_news_search(query: str, time_filter: str = "week", max_articles: int = 5, focus_domains: List[str] = None) -> str:
    """Search for the latest news and current information using multiple strategies"""
    try:
        results = []
        
        # 1. 시간 필터를 포함한 검색 쿼리 생성
        time_queries = {
            "today": f"{query} (today OR 오늘 OR latest OR 최신)",
            "week": f"{query} (this week OR 이번주 OR recent OR 최근 OR 2024)",
            "month": f"{query} (this month OR 이번달 OR latest OR 최신 OR 2024)",
            "all": query
        }
        
        search_query = time_queries.get(time_filter, time_queries["week"])
        
        # 2. 뉴스 도메인 우선 검색
        news_domains = focus_domains or [
            "reuters.com", "bloomberg.com", "cnbc.com", "bbc.com", "cnn.com",
            "yonhapnews.co.kr", "chosun.com", "joongang.co.kr", "donga.com",
            "news.naver.com", "news.daum.net"
        ]
        
        # 3. 도메인별 검색 수행
        found_urls = []
        for domain in news_domains[:3]:  # 상위 3개 도메인만 우선 검색
            try:
                domain_query = f"site:{domain} {search_query}"
                domain_results = handle_tavily_tool(query=domain_query)
                
                if domain_results and "결과를 찾을 수 없습니다" not in domain_results:
                    results.append(f"=== {domain} 최신 뉴스 ===\n{domain_results}\n")
                    
                    # URL 추출 (간단한 패턴 매칭)
                    import re
                    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', domain_results)
                    found_urls.extend(urls[:2])  # 도메인당 최대 2개 URL
                    
            except Exception as e:
                results.append(f"=== {domain} 검색 중 오류: {str(e)} ===\n")
        
        # 4. 일반 최신 뉴스 검색
        try:
            general_results = handle_tavily_tool(query=search_query)
            if general_results:
                results.append(f"=== 일반 최신 뉴스 검색 ===\n{general_results}\n")
                
                # 추가 URL 추출
                import re
                urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', general_results)
                found_urls.extend(urls[:3])
                
        except Exception as e:
            results.append(f"=== 일반 검색 중 오류: {str(e)} ===\n")
        
        # 5. 발견된 URL들에 대해 JINA 크롤링 수행 (최대 max_articles개)
        if found_urls:
            unique_urls = list(set(found_urls))[:max_articles]
            results.append(f"\n=== 상세 기사 분석 (JINA 크롤링) ===\n")
            
            for i, url in enumerate(unique_urls, 1):
                try:
                    # JINA를 통한 크롤링
                    # crawled_content = handle_crawl_tool({"url": url})
                    
                    # if crawled_content and len(crawled_content) > 50:
                    if True:  # 임시로 항상 True로 설정
                        results.append(f"**기사 {i}: {url}** - 크롤링 기능이 일시적으로 비활성화됨\n\n")
                    
                    # 분석 초점에 따른 내용 필터링 (일시적으로 비활성화)
                    # if analysis_focus == "market_trends":
                    #     # 시장 동향 관련 키워드 강조
                    #     keywords = ["시장", "매출", "성장", "전망", "투자", "주가", "수익", "트렌드"]
                    #     filtered_content = extract_relevant_content(crawled_content, keywords)
                    #     results.append(f"**시장 동향 분석**:\n{filtered_content}\n")
                    # elif analysis_focus == "policy_changes":
                    #     # 정책 변화 관련 키워드 강조
                    #     keywords = ["정책", "법안", "규제", "정부", "법률", "제도", "개정", "시행"]
                    #     filtered_content = extract_relevant_content(crawled_content, keywords)
                    #     results.append(f"**정책 변화 분석**:\n{filtered_content}\n")
                    # elif analysis_focus == "technology_updates":
                    #     # 기술 업데이트 관련 키워드 강조
                    #     keywords = ["기술", "혁신", "개발", "AI", "인공지능", "자동화", "디지털", "플랫폼"]
                    #     filtered_content = extract_relevant_content(crawled_content, keywords)
                    #     results.append(f"**기술 업데이트 분석**:\n{filtered_content}\n")
                    # else:
                    #     # 일반 분석
                    #     results.append(f"**전체 내용**:\n{crawled_content[:2000]}{'...' if len(crawled_content) > 2000 else ''}\n")
                    
                    results.append("---\n")
                    
                except Exception as e:
                    results.append(f"**기사 {i}: {url}** - 크롤링 실패: {str(e)}\n\n")
        
        if results:
            final_result = '\n'.join(results)
            return f"🔍 **최신 뉴스 검색 결과** (필터: {time_filter})\n\n{final_result}"
        else:
            return f"최신 뉴스를 찾을 수 없습니다. 검색어를 다시 확인해주세요: {query}"
            
    except Exception as e:
        return f"최신 뉴스 검색 중 오류 발생: {str(e)}"

def handle_deep_article_analysis(urls: List[str], analysis_focus: str = "general") -> str:
    """Use JINA to perform deep analysis of specific news articles or web pages"""
    try:
        if not urls:
            return "분석할 URL이 제공되지 않았습니다."
        
        results = []
        results.append(f"🔬 **심층 기사 분석** (분석 초점: {analysis_focus})\n")
        
        for i, url in enumerate(urls, 1):
            try:
                # JINA를 통한 크롤링
                # crawled_content = handle_crawl_tool({"url": url})
                
                # if crawled_content and len(crawled_content) > 50:
                if True:  # 임시로 항상 True로 설정
                    results.append(f"\n**📰 기사 {i}: {url}**\n")
                    # results.append(f"**내용 길이**: {len(crawled_content)} 문자\n")
                    results.append(f"**내용**: 크롤링 기능이 일시적으로 비활성화됨\n")
                    
                    # 분석 초점에 따른 내용 필터링 (일시적으로 비활성화)
                    # if analysis_focus == "market_trends":
                    #     # 시장 동향 관련 키워드 강조
                    #     keywords = ["시장", "매출", "성장", "전망", "투자", "주가", "수익", "트렌드"]
                    #     filtered_content = extract_relevant_content(crawled_content, keywords)
                    #     results.append(f"**시장 동향 분석**:\n{filtered_content}\n")
                    # elif analysis_focus == "policy_changes":
                    #     # 정책 변화 관련 키워드 강조
                    #     keywords = ["정책", "법안", "규제", "정부", "법률", "제도", "개정", "시행"]
                    #     filtered_content = extract_relevant_content(crawled_content, keywords)
                    #     results.append(f"**정책 변화 분석**:\n{filtered_content}\n")
                    # elif analysis_focus == "technology_updates":
                    #     # 기술 업데이트 관련 키워드 강조
                    #     keywords = ["기술", "혁신", "개발", "AI", "인공지능", "자동화", "디지털", "플랫폼"]
                    #     filtered_content = extract_relevant_content(crawled_content, keywords)
                    #     results.append(f"**기술 업데이트 분석**:\n{filtered_content}\n")
                    # else:
                    #     # 일반 분석
                    #     results.append(f"**전체 내용**:\n{crawled_content[:2000]}{'...' if len(crawled_content) > 2000 else ''}\n")
                    
                    results.append("---\n")
                    
                else:
                    results.append(f"\n**❌ 기사 {i}: {url}**\n크롤링된 내용이 부족합니다.\n---\n")
                    
            except Exception as e:
                results.append(f"\n**❌ 기사 {i}: {url}**\n크롤링 실패: {str(e)}\n---\n")
        
        return '\n'.join(results)
        
    except Exception as e:
        return f"심층 기사 분석 중 오류 발생: {str(e)}"

def extract_relevant_content(content: str, keywords: List[str]) -> str:
    """키워드와 관련된 내용을 추출하는 헬퍼 함수"""
    try:
        sentences = content.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:10])  # 최대 10개 문장
        else:
            return content[:1000]  # 키워드가 없으면 처음 1000자
            
    except Exception:
        return content[:1000]

def handle_smart_crawl_tool(url: str, query: str, crawl_type: str = "quick") -> str:
    """스마트 크롤링 도구 - 쿼리와 URL을 분석하여 최적화된 크롤링 수행"""
    try:
        # 크롤링 매개변수 제안 생성
        suggestions = get_crawl_suggestions(query, url)
        
        # crawl_type에 따른 추가 최적화
        if crawl_type == "comprehensive":
            suggestions["max_depth"] = 2
            suggestions["max_breadth"] = 30
            suggestions["limit"] = 80
            suggestions["extract_depth"] = "advanced"
        elif crawl_type == "documentation":
            suggestions["max_depth"] = 2
            suggestions["instructions"] = "technical documentation and guides"
            suggestions["select_paths"] = ["/docs/", "/documentation/", "/guide/", "/api/"]
            suggestions["extract_depth"] = "advanced"
        elif crawl_type == "news":
            suggestions["max_depth"] = 1
            suggestions["max_breadth"] = 15
            suggestions["instructions"] = "latest news and articles"
            suggestions["select_paths"] = ["/news/", "/blog/", "/articles/"]
        elif crawl_type == "quick":
            suggestions["max_depth"] = 1
            suggestions["max_breadth"] = 10
            suggestions["limit"] = 20
        
        # Tavily Crawl 실행
        result = handle_tavily_crawl_tool(
            url=url,
            max_depth=suggestions.get("max_depth", 1),
            max_breadth=suggestions.get("max_breadth", 20),
            limit=suggestions.get("limit", 50),
            instructions=suggestions.get("instructions"),
            select_paths=suggestions.get("select_paths"),
            exclude_paths=suggestions.get("exclude_paths"),
            extract_depth=suggestions.get("extract_depth", "basic")
        )
        
        # 결과에 최적화 정보 추가
        optimization_info = f"""
🤖 **스마트 크롤링 최적화 정보**
- **크롤링 타입**: {crawl_type}
- **최적화된 깊이**: {suggestions.get("max_depth", 1)}
- **최적화된 너비**: {suggestions.get("max_breadth", 20)}
- **지침**: {suggestions.get("instructions", "없음")}
- **선택된 경로**: {suggestions.get("select_paths", "없음")}

---

{result}
"""
        
        return optimization_info
        
    except Exception as e:
        return f"스마트 크롤링 중 오류 발생: {str(e)}"
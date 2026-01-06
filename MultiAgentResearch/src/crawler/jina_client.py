import logging
import os

import requests

logger = logging.getLogger(__name__)


class JinaClient:
    def crawl(self, url: str, return_format: str = "html") -> str:
        headers = {
            "Content-Type": "application/json",
            "X-Return-Format": return_format,
        }
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        else:
            logger.warning(
                "Jina API key is not set. Provide your own key to access a higher rate limit. See https://jina.ai/reader for more information."
            )
        
        try:
            data = {"url": url}
            response = requests.post("https://r.jina.ai/", headers=headers, json=data, timeout=30)
            response.raise_for_status()  # HTTP 에러 체크
            
            content = response.text
            if not content or content.strip() == "":
                logger.warning(f"Empty content received from Jina for URL: {url}")
                return "<html><body><p>No content available</p></body></html>"
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while crawling {url}: {e}")
            return f"<html><body><p>Error: Failed to fetch content from {url}</p></body></html>"
        except Exception as e:
            logger.error(f"Unexpected error while crawling {url}: {e}")
            return f"<html><body><p>Error: {str(e)}</p></body></html>"

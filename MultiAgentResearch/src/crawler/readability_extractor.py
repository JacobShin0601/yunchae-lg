import logging
from readabilipy import simple_json_from_html_string

from .article import Article

logger = logging.getLogger(__name__)


class ReadabilityExtractor:
    def extract_article(self, html: str) -> Article:
        # HTML이 None이거나 빈 문자열인 경우 처리
        if not html or html.strip() == "":
            logger.warning("Empty or None HTML provided to ReadabilityExtractor")
            return Article(
                title="No Content Available",
                html_content="<p>Content could not be extracted from the provided URL.</p>",
            )
        
        try:
            article = simple_json_from_html_string(html, use_readability=True)
            return Article(
                title=article.get("title", "Unknown Title"),
                html_content=article.get("content", "<p>No content available</p>"),
            )
        except Exception as e:
            logger.error(f"Error extracting article with readabilipy: {e}")
            return Article(
                title="Extraction Error",
                html_content=f"<p>Error extracting content: {str(e)}</p>",
            )

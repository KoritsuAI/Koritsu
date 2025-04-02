"""
Web Page Content Fetcher Tool.

This tool uses Selenium with Chrome headless browser to fetch and render web pages,
allowing access to content that may be dynamically loaded with JavaScript.
It returns the processed content as a Document object.
"""

from langchain_core.tools import tool
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain_core.documents import Document
from typing import Optional

@tool
def fetch_web_page_content(url: str) -> Optional[Document]:
    """
    Fetch rendered content from a web page using a headless browser.
    
    This tool uses Selenium with Chrome to access web pages, including content
    that is dynamically loaded with JavaScript. It returns the page content
    as a Document object with metadata.
    
    Args:
        url: The URL of the web page to fetch
        
    Returns:
        A Document object containing the page content and metadata,
        or None if the page couldn't be fetched
        
    Example:
        >>> doc = fetch_web_page_content("https://example.com")
        >>> print(doc.page_content)
    """
    try:
        loader = SeleniumURLLoader(
            urls=[url],
            executable_path="/usr/bin/chromedriver",
            arguments=['--headless', '--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
        )
        pages = loader.load()
        
        if not pages:
            print(f"Failed to fetch content from {url} - no content returned")
            return None
            
        return pages[0]
        
    except Exception as e:
        print(f"Error fetching content from {url}: {str(e)}")
        return None
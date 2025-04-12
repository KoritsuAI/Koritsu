"""
Web Page Raw HTML Fetcher Tool.

This tool uses Selenium with Chrome headless browser to fetch the raw HTML of web pages.
Unlike other fetchers, this tool returns the complete unprocessed HTML, which is useful
for parsing or extracting specific elements using selectors.
"""

from langchain_core.tools import tool
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

@tool
def fetch_web_page_raw_html(url: str) -> Optional[str]:
    """
    Fetch the raw HTML of a web page using a headless browser.
    
    This tool uses Selenium with Chrome to render the web page including
    any JavaScript content, and returns the complete HTML of the rendered page.
    
    Args:
        url: The URL of the web page to fetch
        
    Returns:
        The raw HTML content of the webpage as a string, or None if an error occurs
        
    Example:
        >>> html = fetch_web_page_raw_html("https://example.com")
        >>> import re
        >>> title = re.search(r"<title>(.*?)</title>", html).group(1)
        >>> print(f"Page title: {title}")
    """
    driver = None
    
    try:
        # Configure Chrome options for headless operation
        options = Options()
        options.add_argument('--headless')
        options.add_argument("--disable-gpu")
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
            
        # Set up the Chrome service
        service = Service('/usr/bin/chromedriver')
        
        # Initialize the browser
        driver = webdriver.Chrome(options=options, service=service)
        
        # Set page load timeout to prevent hanging
        driver.set_page_load_timeout(30)
        
        # Navigate to the URL
        driver.get(url)
        
        # Extract the page HTML
        html = driver.execute_script("return document.documentElement.outerHTML;")
        
        return html
        
    except WebDriverException as e:
        print(f"Selenium error fetching {url}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching {url}: {str(e)}")
        return None
    finally:
        # Ensure the browser is properly closed
        if driver:
            driver.quit()
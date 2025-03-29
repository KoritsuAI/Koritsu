"""
DuckDuckGo Web Search Tool.

This tool provides a privacy-focused web search capability using the DuckDuckGo 
search engine. It returns relevant search results for a given query.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

@tool
def duck_duck_go_web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    
    This tool performs a web search using the DuckDuckGo search engine
    and returns the search results as text.
    
    Args:
        query: The search query string
        
    Returns:
        A string containing search results from DuckDuckGo
        
    Example:
        >>> duck_duck_go_web_search("latest AI developments")
    """
    search_tool = DuckDuckGoSearchResults()
    return search_tool.invoke(query)
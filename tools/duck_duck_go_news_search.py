"""
DuckDuckGo News Search Tool.

This tool provides access to news articles and current events by querying
the DuckDuckGo news search backend. It's useful for finding recent information
that may not be available in the model's training data.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from typing import Dict, List, Any

@tool
def duck_duck_go_news_search(query: str) -> List[Dict[str, Any]]:
    """
    Search for news articles using DuckDuckGo.
    
    This tool performs a news-specific search using DuckDuckGo's news backend
    and returns recent news articles related to the query.
    
    Args:
        query: The search query for news articles
        
    Returns:
        A list of news article results with titles, snippets, and links
        
    Example:
        >>> duck_duck_go_news_search("cryptocurrency market trends")
    """
    # Initialize the DuckDuckGo search with the news backend
    news_search = DuckDuckGoSearchResults(backend="news")
    
    # Execute the search and return results
    return news_search.invoke(query)
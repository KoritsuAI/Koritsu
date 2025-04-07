"""
Web Researcher Agent Module.

This module implements a ReAct agent capable of performing web-based research tasks.
It uses a combination of web search and content retrieval tools to gather information
from the internet based on user queries.

The agent follows a reasoning-acting loop:
1. It reasons about the information needed
2. It decides which tools to use (search or fetch content)
3. It processes the information retrieved
4. It formulates a comprehensive response
"""

from typing import Literal, Dict, List, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config

# System prompt defines the agent's personality and capabilities
system_prompt = """You are web_researcher, a ReAct agent that can use the web to research answers.

You have a tool to search the web, and a tool to fetch the content of a web page.

When researching, follow these steps:
1. Search for relevant information using duck_duck_go_web_search
2. Analyze search results to find the most promising pages
3. Fetch the content of those pages using fetch_web_page_content
4. Extract the most relevant information from the content
5. Synthesize a comprehensive, well-organized answer

Remember to cite your sources in your final response.
"""
    
# Import the tools the agent can use
from tools.duck_duck_go_web_search import duck_duck_go_web_search
from tools.fetch_web_page_content import fetch_web_page_content

# List of tools available to the agent
tools = [duck_duck_go_web_search, fetch_web_page_content]

def reasoning(state: MessagesState) -> Dict[str, List[Any]]:
    """
    The reasoning step of the agent's workflow.
    
    This function processes the current conversation state and generates
    the next response, which may include tool calls.
    
    Args:
        state: The current state containing conversation messages
        
    Returns:
        Updated state with the agent's response message added
    """
    print("web_researcher is thinking...")
    messages = state['messages']
    tooled_up_model = config.default_langchain_model.bind_tools(tools)
    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}

def check_for_tool_calls(state: MessagesState) -> Literal["tools", END]:
    """
    Checks if the latest message contains tool calls.
    
    This function decides whether to proceed to tool execution
    or end the agent's workflow.
    
    Args:
        state: The current state containing conversation messages
        
    Returns:
        "tools" if tool calls are present, otherwise END
    """
    messages = state['messages']
    last_message = messages[-1]
    
    if last_message.tool_calls:
        if not last_message.content.strip() == "":
            print("web_researcher thought this:")
            print(last_message.content)
        print()
        print("web_researcher is acting by invoking these tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"
    
    return END

# Tool execution node
acting = ToolNode(tools)

# Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("reasoning", reasoning)
workflow.add_node("tools", acting)
workflow.set_entry_point("reasoning")
workflow.add_conditional_edges(
    "reasoning",
    check_for_tool_calls,
)
workflow.add_edge("tools", 'reasoning')

# Compile the graph
graph = workflow.compile()


def web_researcher(task: str) -> Dict[str, List[Any]]:
    """
    Research a topic using web search and content retrieval.
    
    This function initializes the web researcher agent with a task
    and returns the final conversation state after the agent has
    completed its research.
    
    Args:
        task: The research query or task description
        
    Returns:
        The final state containing all conversation messages
        
    Example:
        >>> result = web_researcher("What are the latest developments in quantum computing?")
        >>> print(result["messages"][-1].content)
    """
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )
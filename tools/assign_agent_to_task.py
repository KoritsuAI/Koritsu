"""
Agent Task Assignment Tool.

This tool is the cornerstone of the agent orchestration system, enabling
collaboration between different specialized agents. It allows one agent
(typically Hermes) to delegate tasks to other agents and receive their responses.

This mechanism creates a multi-agent system where each agent can focus on
its specific expertise while collaborating on complex tasks.
"""

import sys
import traceback
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import utils

@tool
def assign_agent_to_task(agent_name: str, task: str) -> str:
    """
    Assign a task to a specified agent and return its response.
    
    This function dynamically loads the requested agent module, invokes
    the agent with the given task, and returns the final response from
    the agent's execution. It handles errors gracefully if the agent
    doesn't exist or encounters an error.
    
    Args:
        agent_name: The name of the agent to assign the task to (must match a Python file
                   in the agents/ directory without the .py extension)
        task: A description of the task for the agent to perform
        
    Returns:
        The agent's response as a string, or an error message if the assignment failed
        
    Example:
        >>> response = assign_agent_to_task("web_researcher", "Find information about climate change")
        Assigning agent web_researcher to task: Find information about climate change
        web_researcher responded:
        Climate change refers to long-term shifts in temperatures and weather patterns...
    """
    print(f"Assigning agent {agent_name} to task: {task}")
    
    # Check if the agent exists before trying to load it
    if agent_name not in utils.list_agents():
        error_msg = f"Agent '{agent_name}' not found. Available agents: {', '.join(utils.list_agents())}"
        print(error_msg)
        return error_msg
    
    # Handle the case where the call to the agent fails
    try:
        # Dynamically load the agent module
        agent_module = utils.load_module(f"agents/{agent_name}.py")
        agent_function = getattr(agent_module, agent_name)
        
        # Execute the agent with the given task
        result = agent_function(task=task)
        
        # Clean up by removing the module from sys.modules
        del sys.modules[agent_module.__name__]
        
        # Extract the agent's final response
        response = result["messages"][-1].content
        print(f"{agent_name} responded:")
        print(response)
        return response
        
    except AttributeError as e:
        error_msg = f"The agent module '{agent_name}' exists but doesn't have the expected function structure: {str(e)}"
        print(error_msg)
        return error_msg
        
    except Exception as e:
        exception_trace = traceback.format_exc()
        error = f"An error occurred while {agent_name} was working on task: {task}\n{str(e)}\n{exception_trace}"
        print(error)
        return error
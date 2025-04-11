"""
Utility Functions Module.

This module provides common utility functions for the application, including:
- Tool and agent discovery and loading
- Module loading utilities
- Checkpoint management for stateful operations

The utilities here are used across the application to dynamically discover
available tools and agents, handle module loading, and manage application state.
"""

import sqlite3
import importlib.util
import sys
import string
import secrets
import traceback

from langgraph.checkpoint.sqlite import SqliteSaver

# Initialize SQLite connection for checkpoints
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

def all_tool_functions():
    """
    Get all available tool functions from the tools directory.
    
    This function loads each tool module and extracts the tool function.
    It handles errors gracefully, logging when a tool cannot be loaded.
    
    Returns:
        List of callable tool functions that can be used by agents
    """
    tools = list_tools()
    tool_funcs = []
    
    for tool in tools:
        try:
            module = load_module(f"tools/{tool}.py")
            tool_func = getattr(module, tool)
            tool_funcs.append(tool_func)
        except Exception as e:
            print(f"WARN: Could not load tool \"{tool}\". {e.__class__.__name__}: {e}")
    
    return tool_funcs

def list_broken_tools():
    """
    Find and report tools that have errors when loaded.
    
    This function attempts to load each tool and captures any exceptions,
    helping identify which tools are broken and why.
    
    Returns:
        Dict mapping tool names to their errors and exception traces
    """
    tools = list_tools()
    broken_tools = {}
    
    for tool in tools:
        try:
            module = load_module(f"tools/{tool}.py")
            getattr(module, tool)
            del sys.modules[module.__name__]
        except Exception as e:
            exception_trace = traceback.format_exc()
            broken_tools[tool] = [e, exception_trace]
    
    return broken_tools

def list_tools():
    """
    List all tools available in the tools directory.
    
    This function scans the tools directory for Python files,
    each representing a tool that can be loaded.
    
    Returns:
        List of tool names (without the .py extension)
    """
    import os
    tools = []
    for file in os.listdir("tools"):
        if file.endswith(".py"):
            tools.append(file[:-3])

    return tools

def all_agents(exclude=["hermes"]):
    """
    Get all available agents with their descriptions.
    
    This function loads each agent module and extracts its docstring description.
    It handles errors gracefully, logging when an agent cannot be loaded.
    The 'hermes' agent is excluded by default as it's a system agent.
    
    Args:
        exclude: List of agent names to exclude from the results
        
    Returns:
        Dict mapping agent names to their docstring descriptions
    """
    agents = list_agents()
    agents = [agent for agent in agents if agent not in exclude]
    agent_funcs = {}
    
    for agent in agents:
        try:
            module = load_module(f"agents/{agent}.py")
            agent_func = getattr(module, agent)
            agent_funcs[agent] = agent_func.__doc__
            del sys.modules[module.__name__]
        except Exception as e:
            print(f"WARN: Could not load agent \"{agent}\". {e.__class__.__name__}: {e}")
    
    return agent_funcs

def list_broken_agents():
    """
    Find and report agents that have errors when loaded.
    
    This function attempts to load each agent and captures any exceptions,
    helping identify which agents are broken and why.
    
    Returns:
        Dict mapping agent names to their errors and exception traces
    """
    agents = list_agents()
    broken_agents = {}
    
    for agent in agents:
        try:
            module = load_module(f"agents/{agent}.py")
            getattr(module, agent)
            del sys.modules[module.__name__]
        except Exception as e:
            exception_trace = traceback.format_exc()
            broken_agents[agent] = [e, exception_trace]
    
    return broken_agents

def list_agents():
    """
    List all agents available in the agents directory.
    
    This function scans the agents directory for Python files,
    each representing an agent that can be loaded.
    It excludes __init__.py files.
    
    Returns:
        List of agent names (without the .py extension)
    """
    import os
    agents = []
    for file in os.listdir("agents"):
        if file.endswith(".py") and file != "__init__.py":
            agents.append(file[:-3])

    return agents

def gensym(length=32, prefix="gensym_"):
    """
    Generate a unique symbol for use as a module name.
    
    This function creates a random alphanumeric string that can be used
    as a unique module identifier when dynamically loading modules.
    
    Args:
        length: Length of the random part of the symbol
        prefix: Prefix to prepend to the random part
        
    Returns:
        A string containing the prefix followed by random characters
        
    Example:
        >>> gensym(8, "mod_")
        'mod_A7bC3d9F'
    """
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits
    symbol = "".join([secrets.choice(alphabet) for i in range(length)])

    return prefix + symbol

def load_module(source, module_name=None):
    """
    Dynamically load a Python file as a module.
    
    This function loads a Python file from disk and imports it as a module
    into the current Python process. It's used for dynamic loading of
    tools and agents.
    
    Args:
        source: Path to the Python file to load
        module_name: Name to give the module (generated if None)
        
    Returns:
        The loaded module object
        
    Raises:
        ImportError: If the module cannot be loaded
    """
    if module_name is None:
        module_name = gensym()

    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
"""
Software Engineer Agent Module.

This module implements a ReAct agent specialized in software development tasks.
It can create, modify, and delete code files, run shell commands, and collaborate
with other agents on complex programming tasks.

The agent follows a reasoning-acting loop pattern with access to file system
operations and command line tools, allowing it to function as a capable 
software development assistant.
"""

from typing import Literal, Dict, List, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import config

# System prompt defines the agent's personality and capabilities
system_prompt = """You are software_engineer, a ReAct agent that can create, modify, and delete code.

You have tools to manage files, run shell commands, and collaborate with other agents by assigning them tasks.

When working on code tasks, follow these best practices:
1. Break down complex problems into manageable steps
2. Research and understand requirements before coding
3. Use appropriate programming patterns and idioms
4. Comment your code and document your work
5. Test your implementations before considering them complete
"""

# Import the tools the agent can use
from tools.write_to_file import write_to_file
from tools.overwrite_file import overwrite_file
from tools.delete_file import delete_file
from tools.read_file import read_file
from tools.run_shell_command import run_shell_command
from tools.assign_agent_to_task import assign_agent_to_task
from tools.list_available_agents import list_available_agents

# List of tools available to the agent
tools = [
    write_to_file,
    overwrite_file,
    delete_file,
    read_file,
    run_shell_command,
    assign_agent_to_task,
    list_available_agents
]

def reasoning(state: MessagesState) -> Dict[str, List[Any]]:
    """
    The reasoning step of the agent's workflow.
    
    This function processes the current conversation state and generates
    the next response, which may include tool calls for coding operations.
    
    Args:
        state: The current state containing conversation messages
        
    Returns:
        Updated state with the agent's response message added
    """
    print("software_engineer is thinking...")
    messages = state['messages']
    tooled_up_model = config.default_langchain_model.bind_tools(tools)
    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}

def check_for_tool_calls(state: MessagesState) -> Literal["tools", END]:
    """
    Checks if the latest message contains tool calls.
    
    This function decides whether to proceed to tool execution
    (e.g., file operations, shell commands) or end the agent's workflow.
    
    Args:
        state: The current state containing conversation messages
        
    Returns:
        "tools" if tool calls are present, otherwise END
    """
    messages = state['messages']
    last_message = messages[-1]
    
    if last_message.tool_calls:
        if not last_message.content.strip() == "":
            print("software_engineer thought this:")
            print(last_message.content)
        print()
        print("software_engineer is acting by invoking these tools:")
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


def software_engineer(task: str) -> Dict[str, List[Any]]:
    """
    Perform software engineering tasks such as creating, modifying, and managing code.
    
    This function initializes the software engineer agent with a task description
    and returns the final conversation state after the agent has completed its work.
    
    The agent can:
    - Create new code files
    - Modify existing code
    - Delete files
    - Run shell commands
    - Collaborate with other agents
    
    Args:
        task: Description of the software engineering task to perform
        
    Returns:
        The final state containing all conversation messages
        
    Example:
        >>> result = software_engineer("Create a simple Flask API with two endpoints")
        >>> print(result["messages"][-1].content)
    """
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )
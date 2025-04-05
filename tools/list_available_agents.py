"""
Agent Listing Tool.

This tool provides information about all available agents in the system
along with descriptions of what tasks they're designed to handle.
"""

from langchain_core.tools import tool
import utils
from typing import Dict, Optional

@tool
def list_available_agents() -> Dict[str, Optional[str]]:
    """
    List all available agents and their purpose descriptions.
    
    This tool returns a dictionary mapping agent names to their docstring descriptions,
    which describe what type of tasks each agent is designed to handle.
    
    Returns:
        A dictionary where:
        - Keys are agent names (strings)
        - Values are the docstring descriptions (strings or None if no description)
        
    Example:
        >>> agents = list_available_agents()
        >>> for name, description in agents.items():
        ...     print(f"Agent: {name}")
        ...     print(f"Purpose: {description}")
    """
    # Fetch all available agents excluding certain system agents
    # This returns a dictionary of {agent_name: agent_description}
    return utils.all_agents()
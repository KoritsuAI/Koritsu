"""
Human Input Request Tool.

This tool enables the agent to request input from a human user during execution.
It displays a prompt and waits for user input via the command line.
"""

from langchain_core.tools import tool

@tool
def request_human_input(prompt: str) -> str:
    """
    Request input from a human user via the command line.
    
    This tool allows the agent to pause execution and ask for user input
    when it needs human guidance or data. It displays a prompt message
    and waits for the user to type a response.
    
    Args:
        prompt: The message to display to the human user
        
    Returns:
        The string input provided by the human user
        
    Example:
        >>> user_name = request_human_input("Please enter your name:")
        Please enter your name:
        > John
        # Returns: "John"
    """
    # Display the prompt message
    print(prompt)
    
    # Request input with a clear indicator and return the result
    return input("> ")

"""
File Reader Tool.

This tool provides an interface for reading file contents given a file path.
It returns the complete content of the file as a string.
"""

from langchain_core.tools import tool

@tool
def read_file(file_path: str) -> str:
    """
    Returns the content of the file at the given file path.
    
    Args:
        file_path: The path to the file to read
        
    Returns:
        The complete content of the file as a string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be accessed due to permissions
    """
    with open(file_path, 'r') as file:
        return file.read()

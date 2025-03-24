"""
File Overwrite Tool.

This tool provides an interface for overwriting an existing file's content or creating a new file.
Unlike write_to_file, this tool will overwrite existing files.
"""

import os
from langchain_core.tools import tool

@tool
def overwrite_file(file_path: str, content: str) -> str:
    """
    Replaces the file at the given path with the given content.
    
    This tool will create a new file if it doesn't exist, or overwrite
    the content if the file already exists.
    
    Args:
        file_path: The path of the file to overwrite or create
        content: The new content to write to the file
        
    Returns:
        A string message confirming success
        
    Raises:
        PermissionError: If the file can't be written due to permissions
        OSError: If other IO errors occur during file operations
    """
    # Make sure the directory exists
    directory = os.path.dirname(os.path.abspath(file_path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    with open(file_path, 'w') as file:
        file.write(content)
    
    return f"File at {file_path} has been successfully overwritten."
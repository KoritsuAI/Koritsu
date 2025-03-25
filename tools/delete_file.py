"""
File Deletion Tool.

This tool provides an interface for safely deleting files.
It handles errors gracefully and returns informative messages.
"""

import os
from langchain_core.tools import tool

@tool
def delete_file(file_path: str) -> str:
    """
    Deletes the file at the given path.
    
    Args:
        file_path: The path to the file that should be deleted
        
    Returns:
        A string confirming successful deletion or describing the error that occurred
        
    Possible errors:
        - FileNotFoundError: If the file doesn't exist
        - PermissionError: If the file can't be deleted due to permissions
        - IsADirectoryError: If the path points to a directory instead of a file
    """
    try:
        # Check if file exists before attempting deletion
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
            
        # Check if it's a file and not a directory
        if not os.path.isfile(file_path):
            return f"The path {file_path} is not a file"
            
        os.remove(file_path)
        return f"File at {file_path} has been deleted successfully."
    except PermissionError:
        return f"Permission denied: Unable to delete {file_path}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"

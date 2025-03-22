"""
File Writer Tool.

This tool provides an interface for writing content to a new file.
As a safety measure, it will not overwrite existing files.
"""

import os
from langchain_core.tools import tool

@tool
def write_to_file(file: str, file_contents: str) -> str:
    """
    Write the contents to a new file, will not overwrite an existing file.
    
    Args:
        file: The path where the new file should be created
        file_contents: The contents to write to the file
        
    Returns:
        A success message confirming the file was written
        
    Raises:
        FileExistsError: If the file already exists (prevents overwriting)
        PermissionError: If the path is not writable due to permissions
    """
    if os.path.exists(file):
        raise FileExistsError(f"File {file} already exists and will not be overwritten.")

    print(f"Writing to file: {file}")
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
    
    with open(file, 'w') as f:
        f.write(file_contents)

    return f"File {file} written successfully."
"""
Shell Command Execution Tool.

This tool provides an interface for executing shell commands and capturing their output.
It returns stdout, stderr, and the return code of the command.

WARNING: This tool executes commands with shell=True, which can be a security risk
if used with untrusted input. Use with caution.
"""

import subprocess
import shlex
from langchain_core.tools import tool
from typing import Dict, Union

@tool
def run_shell_command(command: str) -> Dict[str, Union[str, int]]:
    """
    Run a shell command and return the output.
    
    This tool executes the given command in a shell and captures
    its standard output, standard error, and return code.
    
    Args:
        command: The shell command to execute
        
    Returns:
        A dictionary containing:
        - stdout: The standard output from the command
        - stderr: The standard error output from the command
        - returncode: The exit code (0 typically means success)
        
    Raises:
        subprocess.SubprocessError: If the command fails to execute
    """
    print(f"Running shell command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60  # Add a timeout to prevent hanging
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Command timed out after 60 seconds",
            "returncode": -1
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "returncode": -1
        }
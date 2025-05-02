"""
Advanced logging utilities for Condor LLM server.

This module provides structured logging capabilities with configurable
outputs and formats to improve observability.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Define log levels with corresponding colors for console output
LOG_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",   # Green
    "WARNING": "\033[33m", # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m\033[37m", # White on Red background
    "RESET": "\033[0m"    # Reset color
}

# Define log directory
LOG_DIR = os.getenv("CONDOR_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""
    
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in LOG_COLORS:
            colored_levelname = f"{LOG_COLORS[levelname]}{levelname}{LOG_COLORS['RESET']}"
            record.levelname = colored_levelname
        return super().format(record)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add any extra fields
        if hasattr(record, "extra") and record.extra:
            log_data.update(record.extra)
        
        return json.dumps(log_data)

def setup_logger(
    name: str = "condor", 
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        json_format: Whether to format logs as JSON
        log_file: Custom log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler if enabled
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if json_format:
            formatter = JsonFormatter()
        else:
            formatter = ColoredFormatter(
                "%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Create file handler if enabled
    if file_output:
        if not log_file:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        if json_format:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class LogPerformance:
    """Context manager for logging performance metrics of code blocks."""
    
    def __init__(self, logger: logging.Logger, operation_name: str, extra_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance logger.
        
        Args:
            logger: Logger instance to use
            operation_name: Name of the operation being timed
            extra_info: Additional information to include in the log
        """
        self.logger = logger
        self.operation_name = operation_name
        self.extra_info = extra_info or {}
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log timing when exiting the context."""
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        log_data = {
            "operation": self.operation_name,
            "duration_ms": round(duration_ms, 2),
            **self.extra_info
        }
        
        if exc_type:
            log_data["status"] = "failed"
            log_data["exception"] = str(exc_val)
            self.logger.error(f"Operation {self.operation_name} failed after {duration_ms:.2f}ms", extra={"extra": log_data})
        else:
            log_data["status"] = "success"
            self.logger.info(f"Operation {self.operation_name} completed in {duration_ms:.2f}ms", extra={"extra": log_data})
            
def get_logger(name: str = "condor") -> logging.Logger:
    """
    Get a logger with the default configuration.
    
    This is a convenience function for getting a logger with the default settings.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Get log settings from environment variables
    log_level = os.getenv("CONDOR_LOG_LEVEL", "INFO")
    console_output = os.getenv("CONDOR_LOG_CONSOLE", "1") == "1"
    file_output = os.getenv("CONDOR_LOG_FILE", "1") == "1"
    json_format = os.getenv("CONDOR_LOG_JSON", "0") == "1"
    
    return setup_logger(
        name=name,
        log_level=log_level,
        console_output=console_output,
        file_output=file_output,
        json_format=json_format
    ) 
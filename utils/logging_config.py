"""
Logging Configuration for Aurora Chatbot Bias Testing System

This module provides a standardized logging configuration for all components
of the bias testing system, with support for both console and file logging.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Name of the logger
        log_file: Optional log file path (relative to logs directory)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_run_logger(run_id: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific run of the bias testing system.
    
    Args:
        run_id: Optional run identifier (default: timestamp)
        
    Returns:
        Configured logger instance
    """
    # Generate run_id if not provided
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log file name
    log_file = f"bias_test_run_{run_id}.log"
    
    # Set up and return logger
    return setup_logger(f"bias_test.run.{run_id}", log_file)

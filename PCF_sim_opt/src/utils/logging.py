"""
Unified logging module for PCF optimization project.
Consolidates all logging functionality to avoid code duplication.
"""

import logging
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path


class LoggerManager:
    """Centralized logger management for the PCF optimization project."""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.log_dir = Path("log")
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self._configure_root_logger()
    
    def _configure_root_logger(self):
        """Configure the root logger with default settings."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (usually module name)
            log_file: Optional log file name
            
        Returns:
            Logger instance
        """
        if name in self._loggers:
            return self._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Add file handler if log_file is specified
        if log_file is None:
            # Generate a log file name based on current timestamp - format: YYYYMMDD_HHMMSS_UUID.log
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            execution_id = str(uuid.uuid4())
            log_file = f"{timestamp}_{execution_id}.log"
        
        # Ensure file path exists
        file_path = self.log_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        self._loggers[name] = logger
        return logger


class UnifiedLogger:
    """
    Unified logger class that provides a consistent interface for logging
    across all modules in the PCF optimization project.
    """
    
    def __init__(self, module_name: str, log_file: Optional[str] = None):
        """
        Initialize a unified logger for a specific module.
        
        Args:
            module_name: Name of the module using this logger
            log_file: Optional log file name
        """
        self.module_name = module_name
        self.manager = LoggerManager()
        self.logger = self.manager.get_logger(module_name, log_file)
        self._execution_id = None
    
    def set_execution_id(self, execution_id: str):
        """Set execution ID for this logging session."""
        self._execution_id = execution_id
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format log message with additional context."""
        context = {}
        if self._execution_id:
            context['execution_id'] = self._execution_id
        if kwargs:
            context.update(kwargs)
        
        if context:
            context_str = json.dumps(context, ensure_ascii=False)
            return f"{message} | Context: {context_str}"
        return message
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def log_execution_time(self, operation: str, duration: float, **kwargs):
        """Log execution time for an operation."""
        self.info(f"Execution time for {operation}: {duration:.3f}s", 
                 operation=operation, duration=duration, **kwargs)
    
    def log_dataframe_info(self, df_name: str, shape: tuple, **kwargs):
        """Log DataFrame information."""
        self.info(f"DataFrame '{df_name}' shape: {shape}", 
                 df_name=df_name, rows=shape[0], columns=shape[1], **kwargs)
    
    def log_config_update(self, config_name: str, old_value: Any, new_value: Any, **kwargs):
        """Log configuration update."""
        self.info(f"Configuration '{config_name}' updated", 
                 config_name=config_name, old_value=old_value, new_value=new_value, **kwargs)
    
    def log_file_operation(self, operation: str, file_path: str, success: bool, **kwargs):
        """Log file operation."""
        status = "successful" if success else "failed"
        level = self.info if success else self.error
        level(f"File {operation} {status}: {file_path}", 
              operation=operation, file_path=file_path, success=success, **kwargs)
    
    def log_button_click(self, button_name: str, **kwargs):
        """Log button click event."""
        self.info(f"Button clicked: {button_name}", 
                 event_type="button_click", button_name=button_name, **kwargs)
    
    def log_input_change(self, input_name: str, old_value: Any, new_value: Any, **kwargs):
        """Log input change event."""
        self.info(f"Input changed: {input_name}", 
                 event_type="input_change", input_name=input_name, 
                 old_value=old_value, new_value=new_value, **kwargs)


# Legacy function compatibility - to be used during migration
def create_logger(module_name: str, log_file: Optional[str] = None) -> UnifiedLogger:
    """
    Create a unified logger instance.
    
    This is the main function to be used by other modules.
    
    Args:
        module_name: Name of the module creating the logger
        log_file: Optional log file name
        
    Returns:
        UnifiedLogger instance
    """
    return UnifiedLogger(module_name, log_file)


# Compatibility layer for existing _print methods
class PrintCompatibleLogger(UnifiedLogger):
    """Logger with _print method compatibility for existing code."""
    
    def _print(self, *args, level: str = "info", **kwargs):
        """
        Legacy _print method for compatibility.
        
        Args:
            *args: Messages to log (will be joined with space)
            level: Log level (info, warning, error, debug)
            **kwargs: Additional context
        """
        # Join all args with space to match original behavior
        message = " ".join(str(arg) for arg in args)
        
        # Normalize level to lowercase for compatibility
        level = level.lower()
        
        if level == "info":
            self.info(message, **kwargs)
        elif level == "warning":
            self.warning(message, **kwargs)
        elif level == "error":
            self.error(message, **kwargs)
        elif level == "debug":
            self.debug(message, **kwargs)
        else:
            self.info(message, **kwargs)
"""
Migration wrapper to help transition from old logging system to new unified logging.
This module provides compatibility functions for the existing logger.py interface.
"""

from typing import Any, Optional
from .logging import UnifiedLogger, create_logger
import streamlit as st


# Create a global logger instance for backward compatibility
_global_logger: Optional[UnifiedLogger] = None


def get_global_logger() -> UnifiedLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = create_logger("pcf_app")
        # Set execution ID from session state if available
        if hasattr(st, 'session_state') and 'session_id' in st.session_state:
            _global_logger.set_execution_id(st.session_state.session_id)
    return _global_logger


# Legacy function mappings
def log_info(message: str, **kwargs):
    """Legacy log_info function."""
    get_global_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Legacy log_warning function."""
    get_global_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Legacy log_error function."""
    get_global_logger().error(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Legacy log_debug function."""
    get_global_logger().debug(message, **kwargs)


def log_button_click(button_name: str, button_id: Optional[str] = None, **kwargs):
    """Legacy log_button_click function."""
    if button_id:
        kwargs['button_id'] = button_id
    get_global_logger().log_button_click(button_name, **kwargs)


def log_input_change(input_name: str, old_value: Any = None, new_value: Any = None, **kwargs):
    """Legacy log_input_change function."""
    get_global_logger().log_input_change(input_name, old_value, new_value, **kwargs)


def log_execution_time(operation: str, duration: float, **kwargs):
    """Legacy log_execution_time function."""
    get_global_logger().log_execution_time(operation, duration, **kwargs)


def log_file_upload(filename: str, file_size: int, file_type: str, **kwargs):
    """Legacy log_file_upload function."""
    get_global_logger().log_file_operation(
        "upload", filename, True, 
        file_size=file_size, file_type=file_type, **kwargs
    )


# Export all legacy functions for easy import
__all__ = [
    'log_info',
    'log_warning',
    'log_error',
    'log_debug',
    'log_button_click',
    'log_input_change',
    'log_execution_time',
    'log_file_upload',
    'get_global_logger'
]
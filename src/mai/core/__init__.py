"""
Mai Core Module

This module provides core functionality and utilities for Mai,
including configuration management, exception handling, and shared
utilities used across the application.
"""

# Import the real implementations instead of defining placeholders
from .exceptions import MaiError, ConfigurationError, ModelError
from .config import get_config

__all__ = ["MaiError", "ConfigurationError", "ModelError", "get_config"]

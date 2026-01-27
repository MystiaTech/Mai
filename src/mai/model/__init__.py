"""
Mai Model Interface Module

This module provides the core interface for interacting with various AI models,
with a focus on local Ollama models. It handles model discovery, capability
detection, and provides a unified interface for model switching and inference.

The model interface is designed to be extensible, allowing future support
for additional model providers while maintaining a consistent API.
"""

from .ollama_client import OllamaClient

__all__ = ["OllamaClient"]

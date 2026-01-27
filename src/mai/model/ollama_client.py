"""
Ollama Client Wrapper

Provides a robust wrapper around the Ollama Python client with model discovery,
capability detection, caching, and error handling.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import ollama
from src.mai.core import ModelError, ConfigurationError


logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Robust wrapper for Ollama API with model discovery and caching.

    This client handles connection management, model discovery, capability
    detection, and graceful error handling for Ollama operations.
    """

    def __init__(self, host: str = "http://localhost:11434", timeout: int = 30):
        """
        Initialize Ollama client with connection settings.

        Args:
            host: Ollama server URL
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.timeout = timeout
        self._client = None
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=30)

        # Initialize client (may fail if Ollama not running)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Ollama client with error handling."""
        try:
            self._client = ollama.Client(host=self.host, timeout=self.timeout)
            logger.info(f"Ollama client initialized for {self.host}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
            self._client = None

    def _check_client(self) -> None:
        """Check if client is initialized, attempt reconnection if needed."""
        if self._client is None:
            logger.info("Attempting to reconnect to Ollama...")
            self._initialize_client()
            if self._client is None:
                raise ModelError("Cannot connect to Ollama. Is it running?")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with basic metadata.

        Returns:
            List of models with name and basic info
        """
        try:
            self._check_client()
            if self._client is None:
                logger.warning("Ollama client not available")
                return []

            # Get raw model list from Ollama
            response = self._client.list()
            models = response.get("models", [])

            # Extract relevant information
            model_list = []
            for model in models:
                # Handle both dict and object responses from ollama
                if isinstance(model, dict):
                    model_name = model.get("name", "")
                    model_size = model.get("size", 0)
                    model_digest = model.get("digest", "")
                    model_modified = model.get("modified_at", "")
                else:
                    # Ollama returns model objects with 'model' attribute
                    model_name = getattr(model, "model", "")
                    model_size = getattr(model, "size", 0)
                    model_digest = getattr(model, "digest", "")
                    model_modified = getattr(model, "modified_at", "")

                model_info = {
                    "name": model_name,
                    "size": model_size,
                    "digest": model_digest,
                    "modified_at": model_modified,
                }
                model_list.append(model_info)

            logger.info(f"Found {len(model_list)} models")
            return model_list

        except ConnectionError as e:
            logger.error(f"Connection error listing models: {e}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model details
        """
        # Check cache first
        if model_name in self._model_cache:
            cache_entry = self._model_cache[model_name]
            if (
                self._cache_timestamp
                and datetime.now() - self._cache_timestamp < self._cache_duration
            ):
                logger.debug(f"Returning cached info for {model_name}")
                return cache_entry

        try:
            self._check_client()
            if self._client is None:
                raise ModelError("Cannot connect to Ollama")

            # Get model details from Ollama
            response = self._client.show(model_name)

            # Extract key information
            model_info = {
                "name": model_name,
                "parameter_size": response.get("details", {}).get("parameter_size", ""),
                "context_window": response.get("details", {}).get("context_length", 0),
                "model_family": response.get("details", {}).get("families", []),
                "model_format": response.get("details", {}).get("format", ""),
                "quantization": response.get("details", {}).get("quantization_level", ""),
                "size": response.get("details", {}).get("size", 0),
                "modelfile": response.get("modelfile", ""),
                "template": response.get("template", ""),
                "parameters": response.get("parameters", {}),
            }

            # Cache the result
            self._model_cache[model_name] = model_info
            self._cache_timestamp = datetime.now()

            logger.debug(f"Retrieved info for {model_name}: {model_info['parameter_size']} params")
            return model_info

        except Exception as e:
            error_msg = f"Error getting model info for {model_name}: {e}"
            logger.error(error_msg)
            raise ModelError(error_msg)

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available and can be queried.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model exists and is accessible
        """
        try:
            # First check if model exists in list
            models = self.list_models()
            model_names = [m["name"] for m in models]

            if model_name not in model_names:
                logger.debug(f"Model {model_name} not found in available models")
                return False

            # Try to get model info to verify accessibility
            self.get_model_info(model_name)
            return True

        except (ModelError, Exception) as e:
            logger.debug(f"Model {model_name} not accessible: {e}")
            return False

    def refresh_models(self) -> None:
        """
        Force refresh of model list and clear cache.

        This method clears all cached information and forces a fresh
        query to Ollama for all operations.
        """
        logger.info("Refreshing model information...")

        # Clear cache
        self._model_cache.clear()
        self._cache_timestamp = None

        # Reinitialize client if needed
        if self._client is None:
            self._initialize_client()

        logger.info("Model cache cleared")

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status and diagnostics.

        Returns:
            Dictionary with connection status information
        """
        status = {
            "connected": False,
            "host": self.host,
            "timeout": self.timeout,
            "models_count": 0,
            "cache_size": len(self._model_cache),
            "cache_valid": False,
            "error": None,
        }

        try:
            if self._client is None:
                status["error"] = "Client not initialized"
                return status

            # Try to list models to verify connection
            models = self.list_models()
            status["connected"] = True
            status["models_count"] = len(models)

            # Check cache validity
            if self._cache_timestamp:
                age = datetime.now() - self._cache_timestamp
                status["cache_valid"] = age < self._cache_duration
                status["cache_age_minutes"] = age.total_seconds() / 60

        except Exception as e:
            status["error"] = str(e)
            logger.debug(f"Connection status check failed: {e}")

        return status

    def generate_response(
        self, prompt: str, model: str, context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a response from the specified model.

        Args:
            prompt: User prompt/message
            model: Model name to use
            context: Optional conversation context

        Returns:
            Generated response text

        Raises:
            ModelError: If generation fails
        """
        try:
            self._check_client()
            if self._client is None:
                raise ModelError("Cannot connect to Ollama")

            if not model:
                raise ModelError("No model specified")

            # Build the full prompt with context if provided
            if context:
                messages = context + [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

            # Generate response using Ollama
            response = self._client.chat(model=model, messages=messages, stream=False)

            # Extract the response text
            result = response.get("message", {}).get("content", "")
            if not result:
                logger.warning(f"Empty response from {model}")
                return "I apologize, but I couldn't generate a response."

            logger.debug(f"Generated response from {model}")
            return result

        except ModelError:
            raise
        except Exception as e:
            error_msg = f"Error generating response from {model}: {e}"
            logger.error(error_msg)
            raise ModelError(error_msg)


# Convenience function for creating a client
def create_client(host: Optional[str] = None, timeout: int = 30) -> OllamaClient:
    """
    Create an OllamaClient with optional configuration.

    Args:
        host: Optional Ollama server URL
        timeout: Connection timeout in seconds

    Returns:
        Configured OllamaClient instance
    """
    return OllamaClient(host=host or "http://localhost:11434", timeout=timeout)

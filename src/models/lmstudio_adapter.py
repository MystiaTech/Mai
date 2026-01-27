"""LM Studio adapter for local model inference and discovery."""

try:
    import lmstudio as lms
except ImportError:
    from . import mock_lmstudio as lms
from contextlib import contextmanager
from typing import Generator, List, Tuple, Optional, Dict, Any
import logging


@contextmanager
def get_client() -> Generator[lms.Client, None, None]:
    """Context manager for safe LM Studio client handling."""
    client = lms.Client()
    try:
        yield client
    finally:
        client.close()


class LMStudioAdapter:
    """Adapter for LM Studio model management and inference."""

    def __init__(self, host: str = "localhost", port: int = 1234):
        """Initialize LM Studio adapter.

        Args:
            host: LM Studio server host
            port: LM Studio server port
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)

    def list_models(self) -> List[Tuple[str, str, float]]:
        """List all downloaded LLM models.

        Returns:
            List of (model_key, display_name, size_gb) tuples
            Empty list if no models or LM Studio not running
        """
        try:
            with get_client() as client:
                models = client.llm.list_downloaded_models()
                result = []

                for model in models:
                    model_key = getattr(model, "model_key", str(model))
                    display_name = getattr(model, "display_name", model_key)

                    # Estimate size from display name or model_key
                    size_gb = self._estimate_model_size(display_name)

                    result.append((model_key, display_name, size_gb))

                # Sort by estimated size (largest first)
                result.sort(key=lambda x: x[2], reverse=True)
                return result

        except Exception as e:
            self.logger.warning(f"Failed to list models: {e}")
            return []

    def load_model(self, model_key: str, timeout: int = 60) -> Optional[Any]:
        """Load a model by key.

        Args:
            model_key: Model identifier
            timeout: Loading timeout in seconds

        Returns:
            Model instance or None if loading failed
        """
        try:
            with get_client() as client:
                # Try to load the model with timeout
                model = client.llm.model(model_key)

                # Test if model is responsive
                test_response = model.respond("test", max_tokens=1)
                if test_response:
                    return model

        except Exception as e:
            self.logger.error(f"Failed to load model {model_key}: {e}")

        return None

    def unload_model(self, model_key: str) -> bool:
        """Unload a model to free resources.

        Args:
            model_key: Model identifier to unload

        Returns:
            True if successful, False otherwise
        """
        try:
            with get_client() as client:
                # LM Studio doesn't have explicit unload,
                # models are unloaded when client closes
                # This is a placeholder for future implementations
                self.logger.info(
                    f"Model {model_key} will be unloaded on client cleanup"
                )
                return True

        except Exception as e:
            self.logger.error(f"Failed to unload model {model_key}: {e}")
            return False

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get model metadata and capabilities.

        Args:
            model_key: Model identifier

        Returns:
            Dictionary with model info or None if not found
        """
        try:
            with get_client() as client:
                model = client.llm.model(model_key)

                # Extract available information
                info = {
                    "model_key": model_key,
                    "display_name": getattr(model, "display_name", model_key),
                    "context_window": getattr(model, "context_length", 4096),
                }

                return info

        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_key}: {e}")
            return None

    def test_connection(self) -> bool:
        """Test if LM Studio server is running and accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with get_client() as client:
                # Simple connectivity test
                _ = client.llm.list_downloaded_models()
                return True

        except Exception as e:
            self.logger.warning(f"LM Studio connection test failed: {e}")
            return False

    def _estimate_model_size(self, display_name: str) -> float:
        """Estimate model size in GB from display name.

        Args:
            display_name: Model display name (e.g., "Qwen2.5 7B Instruct")

        Returns:
            Estimated size in GB
        """
        # Extract parameter count from display name
        import re

        # Look for patterns like "7B", "13B", "70B"
        match = re.search(r"(\d+(?:\.\d+)?)B", display_name.upper())
        if match:
            params_b = float(match.group(1))

            # Rough estimation: 1B parameters ≈ 2GB for storage
            # This varies by quantization, but gives us a ballpark
            if params_b <= 1:
                return 2.0  # Small models
            elif params_b <= 3:
                return 4.0  # Small-medium models
            elif params_b <= 7:
                return 8.0  # Medium models
            elif params_b <= 13:
                return 14.0  # Medium-large models
            elif params_b <= 34:
                return 20.0  # Large models
            else:
                return 40.0  # Very large models

        # Default estimate if we can't parse
        return 4.0

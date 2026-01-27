"""Model manager for intelligent model selection and switching."""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
import yaml
from pathlib import Path

from .lmstudio_adapter import LMStudioAdapter
from .resource_monitor import ResourceMonitor
from .context_manager import ContextManager


class ModelManager:
    """
    Intelligent model selection and switching system.

    Coordinates between LM Studio adapter, resource monitoring, and context
    management to provide optimal model selection and seamless switching.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ModelManager with configuration.

        Args:
            config_path: Path to models configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config_path = (
            config_path
            or Path(__file__).parent.parent.parent / "config" / "models.yaml"
        )
        self.config = self._load_config()

        # Initialize subsystems
        self.lm_adapter = LMStudioAdapter()
        self.resource_monitor = ResourceMonitor()
        self.context_manager = ContextManager()

        # Current model state
        self.current_model_key: Optional[str] = None
        self.current_model_instance: Optional[Any] = None
        self.available_models: List[Dict[str, Any]] = []
        self.model_configurations: Dict[str, Dict[str, Any]] = {}

        # Switching state
        self._switching_lock = asyncio.Lock()
        self._failure_count = {}
        self._last_switch_time = 0

        # Load initial configuration
        self._load_model_configurations()
        self._refresh_available_models()

        self.logger.info("ModelManager initialized with intelligent switching enabled")

    def _load_config(self) -> Dict[str, Any]:
        """Load models configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            # Return minimal default config
            return {
                "models": [],
                "selection_rules": {
                    "resource_thresholds": {
                        "memory_available_gb": {"small": 2, "medium": 4, "large": 8}
                    },
                    "cpu_threshold_percent": 80,
                    "gpu_required_for_large": True,
                },
                "performance": {
                    "load_timeout_seconds": {"small": 30, "medium": 60, "large": 120},
                    "switching_triggers": {
                        "cpu_threshold": 85,
                        "memory_threshold": 85,
                        "response_time_threshold_ms": 5000,
                        "consecutive_failures": 3,
                    },
                },
            }

    def _load_model_configurations(self) -> None:
        """Load model configurations from config."""
        self.model_configurations = {}

        for model in self.config.get("models", []):
            self.model_configurations[model["key"]] = model

        self.logger.info(
            f"Loaded {len(self.model_configurations)} model configurations"
        )

    def _refresh_available_models(self) -> None:
        """Refresh list of available models from LM Studio."""
        try:
            model_list = self.lm_adapter.list_models()
            self.available_models = []

            for model_key, display_name, size_gb in model_list:
                if model_key in self.model_configurations:
                    model_info = self.model_configurations[model_key].copy()
                    model_info.update(
                        {
                            "display_name": display_name,
                            "estimated_size_gb": size_gb,
                            "available": True,
                        }
                    )
                    self.available_models.append(model_info)
                else:
                    # Create minimal config for unknown models
                    self.available_models.append(
                        {
                            "key": model_key,
                            "display_name": display_name,
                            "estimated_size_gb": size_gb,
                            "available": True,
                            "category": "unknown",
                        }
                    )

        except Exception as e:
            self.logger.error(f"Failed to refresh available models: {e}")
            self.available_models = []

    def select_best_model(
        self, conversation_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Select the best model based on current resources and context.

        Args:
            conversation_context: Optional context about the current conversation

        Returns:
            Selected model key or None if no suitable model found
        """
        try:
            # Get current resources
            resources = self.resource_monitor.get_current_resources()

            # Filter models that can fit current resources
            suitable_models = []

            for model in self.available_models:
                if not model.get("available", False):
                    continue

                # Check resource requirements
                required_memory = model.get("min_memory_gb", 2)
                required_vram = model.get("min_vram_gb", 1)

                available_memory = resources["available_memory_gb"]
                available_vram = resources.get("gpu_vram_gb", 0)

                # Check memory with safety margin
                if available_memory < required_memory * 1.5:
                    continue

                # Check VRAM if required for this model size
                if (
                    model.get("category") in ["large"]
                    and required_vram > available_vram
                ):
                    continue

                suitable_models.append(model)

            if not suitable_models:
                self.logger.warning("No models fit current resource constraints")
                return None

            # Sort by preference (large preferred if resources allow)
            selection_rules = self.config.get("selection_rules", {})

            # Apply preference scoring
            scored_models = []
            for model in suitable_models:
                score = 0.0

                # Category preference (large > medium > small)
                category = model.get("category", "unknown")
                if category == "large" and resources["available_memory_gb"] >= 8:
                    score += 100
                elif category == "medium" and resources["available_memory_gb"] >= 4:
                    score += 70
                elif category == "small":
                    score += 40

                # Preference rules from config
                preferred_when = model.get("preferred_when")
                if preferred_when:
                    if "memory" in preferred_when:
                        required_mem = int(
                            preferred_when.split("memory >= ")[1].split("GB")[0]
                        )
                        if resources["available_memory_gb"] >= required_mem:
                            score += 20

                # Factor in recent failures (penalize frequently failing models)
                failure_count = self._failure_count.get(model["key"], 0)
                score -= failure_count * 10

                # Factor in conversation complexity if provided
                if conversation_context:
                    task_type = conversation_context.get("task_type", "simple_chat")
                    model_capabilities = model.get("capabilities", [])

                    if task_type == "reasoning" and "reasoning" in model_capabilities:
                        score += 30
                    elif task_type == "analysis" and "analysis" in model_capabilities:
                        score += 30
                    elif (
                        task_type == "code_generation"
                        and "reasoning" in model_capabilities
                    ):
                        score += 20

                scored_models.append((score, model))

            # Sort by score and return best
            scored_models.sort(key=lambda x: x[0], reverse=True)

            if scored_models:
                best_model = scored_models[0][1]
                self.logger.info(
                    f"Selected model: {best_model['display_name']} (score: {scored_models[0][0]:.1f})"
                )
                return best_model["key"]

        except Exception as e:
            self.logger.error(f"Error in model selection: {e}")

        return None

    async def switch_model(self, target_model_key: str) -> bool:
        """Switch to a different model with proper resource cleanup.

        Args:
            target_model_key: Model key to switch to

        Returns:
            True if switch successful, False otherwise
        """
        async with self._switching_lock:
            try:
                if target_model_key == self.current_model_key:
                    self.logger.debug(f"Already using model {target_model_key}")
                    return True

                # Don't switch too frequently
                current_time = time.time()
                if current_time - self._last_switch_time < 30:  # 30 second cooldown
                    self.logger.warning(
                        "Model switch requested too frequently, ignoring"
                    )
                    return False

                self.logger.info(
                    f"Switching model: {self.current_model_key} -> {target_model_key}"
                )

                # Unload current model (silent - no user notification per CONTEXT.md)
                if self.current_model_instance and self.current_model_key:
                    try:
                        self.lm_adapter.unload_model(self.current_model_key)
                    except Exception as e:
                        self.logger.warning(f"Error unloading current model: {e}")

                # Load new model
                target_config = self.model_configurations.get(target_model_key)
                if not target_config:
                    target_config = {
                        "category": "unknown"
                    }  # Fallback for unknown models

                timeout = self.config.get("performance", {}).get(
                    "load_timeout_seconds", {}
                )
                timeout_seconds = timeout.get(
                    target_config.get("category", "medium"), 60
                )

                new_model = self.lm_adapter.load_model(
                    target_model_key, timeout_seconds
                )

                if new_model:
                    self.current_model_key = target_model_key
                    self.current_model_instance = new_model
                    self._last_switch_time = current_time

                    # Reset failure count for successful load
                    self._failure_count[target_model_key] = 0

                    self.logger.info(f"Successfully switched to {target_model_key}")
                    return True
                else:
                    # Increment failure count
                    self._failure_count[target_model_key] = (
                        self._failure_count.get(target_model_key, 0) + 1
                    )
                    self.logger.error(f"Failed to load model {target_model_key}")
                    return False

            except Exception as e:
                self.logger.error(f"Error during model switch: {e}")
                return False

    async def generate_response(
        self,
        message: str,
        conversation_id: str = "default",
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response with automatic model switching if needed.

        Args:
            message: User message to respond to
            conversation_id: Conversation ID for context
            conversation_context: Optional context for model selection

        Returns:
            Generated response text
        """
        try:
            # Ensure we have a model loaded
            if not self.current_model_instance:
                await self._ensure_model_loaded(conversation_context)

            if not self.current_model_instance:
                return "I'm sorry, I'm unable to load any models at the moment."

            # Get conversation context
            context_messages = self.context_manager.get_context_for_model(
                conversation_id
            )

            # Format messages for model (LM Studio uses OpenAI-like format)
            formatted_context = self._format_context_for_model(context_messages)

            # Attempt to generate response
            start_time = time.time()
            try:
                response = self.current_model_instance.respond(
                    f"{formatted_context}\n\nUser: {message}\n\nAssistant:",
                    max_tokens=1024,  # Reasonable default
                )

                response_time_ms = (time.time() - start_time) * 1000

                # Check if response is adequate
                if not response or len(response.strip()) < 10:
                    raise ValueError("Model returned empty or inadequate response")

                # Add messages to context
                from models.conversation import MessageRole

                self.context_manager.add_message(
                    conversation_id, MessageRole.USER, message
                )
                self.context_manager.add_message(
                    conversation_id, MessageRole.ASSISTANT, response
                )

                # Check if we should consider switching (slow response or struggling)
                if await self._should_consider_switching(response_time_ms, response):
                    await self._proactive_model_switch(conversation_context)

                return response

            except Exception as e:
                self.logger.warning(f"Model generation failed: {e}")

                # Try switching to a different model
                if await self._handle_model_failure(conversation_context):
                    # Retry with new model
                    return await self.generate_response(
                        message, conversation_id, conversation_context
                    )

                return "I'm experiencing difficulties generating a response. Please try again."

        except Exception as e:
            self.logger.error(f"Error in generate_response: {e}")
            return "An error occurred while processing your request."

    def get_current_model_status(self) -> Dict[str, Any]:
        """Get status of currently loaded model and resource usage.

        Returns:
            Dictionary with model status and resource information
        """
        status = {
            "current_model_key": self.current_model_key,
            "model_loaded": self.current_model_instance is not None,
            "resources": self.resource_monitor.get_current_resources(),
            "available_models": len(self.available_models),
            "recent_failures": dict(self._failure_count),
        }

        if (
            self.current_model_key
            and self.current_model_key in self.model_configurations
        ):
            config = self.model_configurations[self.current_model_key]
            status.update(
                {
                    "model_display_name": config.get(
                        "display_name", self.current_model_key
                    ),
                    "model_category": config.get("category", "unknown"),
                    "context_window": config.get("context_window", 4096),
                }
            )

        return status

    async def preload_model(self, model_key: str) -> bool:
        """Preload a model in background for faster switching.

        Args:
            model_key: Model to preload

        Returns:
            True if preload successful, False otherwise
        """
        try:
            if model_key not in self.model_configurations:
                self.logger.warning(f"Cannot preload unknown model: {model_key}")
                return False

            # Check if already loaded
            if model_key == self.current_model_key:
                return True

            self.logger.info(f"Preloading model: {model_key}")
            # For now, just attempt to load it
            # In a full implementation, this would use background loading
            model = self.lm_adapter.load_model(model_key, timeout=120)

            if model:
                self.logger.info(f"Successfully preloaded {model_key}")
                # Immediately unload to free resources
                self.lm_adapter.unload_model(model_key)
                return True
            else:
                self.logger.warning(f"Failed to preload {model_key}")
                return False

        except Exception as e:
            self.logger.error(f"Error preloading model {model_key}: {e}")
            return False

    async def _ensure_model_loaded(
        self, conversation_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Ensure we have a model loaded, selecting one if needed."""
        if not self.current_model_instance:
            best_model = self.select_best_model(conversation_context)
            if best_model:
                await self.switch_model(best_model)

    async def _should_consider_switching(
        self, response_time_ms: float, response: str
    ) -> bool:
        """Check if we should consider switching models based on performance.

        Args:
            response_time_ms: Response generation time in milliseconds
            response: Generated response content

        Returns:
            True if switching should be considered
        """
        triggers = self.config.get("performance", {}).get("switching_triggers", {})

        # Check response time threshold
        if response_time_ms > triggers.get("response_time_threshold_ms", 5000):
            return True

        # Check system resource thresholds
        resources = self.resource_monitor.get_current_resources()

        if resources["memory_percent"] > triggers.get("memory_threshold", 85):
            return True

        if resources["cpu_percent"] > triggers.get("cpu_threshold", 85):
            return True

        # Check for poor quality responses
        if len(response.strip()) < 20 or response.count("I don't know") > 0:
            return True

        return False

    async def _proactive_model_switch(
        self, conversation_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform proactive model switching without user notification (silent switching)."""
        try:
            best_model = self.select_best_model(conversation_context)
            if best_model and best_model != self.current_model_key:
                self.logger.info(
                    f"Proactively switching from {self.current_model_key} to {best_model}"
                )
                await self.switch_model(best_model)
        except Exception as e:
            self.logger.error(f"Error in proactive switch: {e}")

    async def _handle_model_failure(
        self, conversation_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle model failure by trying fallback models.

        Args:
            conversation_context: Context for selecting fallback model

        Returns:
            True if fallback was successful, False otherwise
        """
        if not self.current_model_key:
            return False

        # Increment failure count
        self._failure_count[self.current_model_key] = (
            self._failure_count.get(self.current_model_key, 0) + 1
        )

        # Get fallback chain from config
        fallback_chains = self.config.get("selection_rules", {}).get(
            "fallback_chains", {}
        )

        # Find appropriate fallback
        fallback_model = None
        current_config = self.model_configurations.get(self.current_model_key, {})
        current_category = current_config.get("category")

        if current_category == "large":
            for large_to_medium in fallback_chains.get("large_to_medium", []):
                if self.current_model_key in large_to_medium:
                    fallback_model = large_to_medium[self.current_model_key]
                    break
        elif current_category == "medium":
            for medium_to_small in fallback_chains.get("medium_to_small", []):
                if self.current_model_key in medium_to_small:
                    fallback_model = medium_to_small[self.current_model_key]
                    break

        if fallback_model:
            self.logger.info(
                f"Attempting fallback: {self.current_model_key} -> {fallback_model}"
            )
            return await self.switch_model(fallback_model)

        # If no specific fallback, try any smaller model
        smaller_models = [
            model["key"]
            for model in self.available_models
            if model.get("category") in ["small", "medium"]
            and model["key"] != self.current_model_key
        ]

        if smaller_models:
            self.logger.info(f"Falling back to smaller model: {smaller_models[0]}")
            return await self.switch_model(smaller_models[0])

        return False

    def _format_context_for_model(self, messages: List[Any]) -> str:
        """Format context messages for LM Studio model."""
        if not messages:
            return ""

        formatted_parts = []
        for msg in messages:
            role_str = getattr(msg, "role", "user")
            content_str = getattr(msg, "content", str(msg))

            if role_str == "user":
                formatted_parts.append(f"User: {content_str}")
            elif role_str == "assistant":
                formatted_parts.append(f"Assistant: {content_str}")
            elif role_str == "system":
                formatted_parts.append(f"System: {content_str}")

        return "\n".join(formatted_parts)

    def shutdown(self) -> None:
        """Clean up resources and unload models."""
        try:
            if self.current_model_instance and self.current_model_key:
                self.lm_adapter.unload_model(self.current_model_key)
                self.current_model_key = None
                self.current_model_instance = None

            self.logger.info("ModelManager shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

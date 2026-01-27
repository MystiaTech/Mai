"""Core Mai orchestration class."""

import asyncio
import logging
from typing import Dict, Any, Optional
import signal
import sys

from models.model_manager import ModelManager
from models.context_manager import ContextManager


class Mai:
    """
    Core Mai orchestration class.

    Coordinates between model management, context management, and other systems
    to provide a unified conversational interface.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize Mai and all subsystems.

        Args:
            config_path: Optional path to configuration files
        """
        self.logger = logging.getLogger(__name__)
        self.running = False

        # Initialize subsystems
        self.model_manager = ModelManager(config_path)
        self.context_manager = self.model_manager.context_manager

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        self.logger.info("Mai core initialized")

    def process_message(self, message: str, conversation_id: str = "default") -> str:
        """
        Process a user message and return response.

        Args:
            message: User input message
            conversation_id: Optional conversation identifier

        Returns:
            Generated response
        """
        try:
            # Simple synchronous wrapper for async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(
                    self.model_manager.generate_response(message, conversation_id)
                )
                return response
            finally:
                loop.close()

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error while processing your message."

    async def process_message_async(
        self, message: str, conversation_id: str = "default"
    ) -> str:
        """
        Asynchronous version of process_message.

        Args:
            message: User input message
            conversation_id: Optional conversation identifier

        Returns:
            Generated response
        """
        try:
            response = await self.model_manager.generate_response(
                message, conversation_id
            )
            return response
        except Exception as e:
            self.logger.error(f"Error processing async message: {e}")
            return "I'm sorry, I encountered an error while processing your message."

    def get_conversation_history(self, conversation_id: str = "default") -> list:
        """
        Retrieve conversation history.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of conversation messages
        """
        try:
            return self.context_manager.get_context_for_model(conversation_id)
        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {e}")
            return []

    def get_system_status(self) -> Dict[str, Any]:
        """
        Return current system status for monitoring.

        Returns:
            Dictionary with system state information
        """
        try:
            # Get model status
            model_status = self.model_manager.get_current_model_status()

            # Get conversation stats
            conversation_stats = {}
            for conv_id in ["default"]:  # Add more conv IDs as needed
                stats = self.context_manager.get_conversation_stats(conv_id)
                if stats:
                    conversation_stats[conv_id] = stats

            # Combine into comprehensive status
            status = {
                "mai_status": "running" if self.running else "stopped",
                "model": model_status,
                "conversations": conversation_stats,
                "system_resources": model_status.get("resources", {}),
            }

            return status

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"mai_status": "error", "error": str(e)}

    def start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        try:

            async def background_loop():
                while self.running:
                    try:
                        # Update resource monitoring
                        self.model_manager.resource_monitor.update_history()

                        # Check for resource-triggered model switches
                        if self.model_manager.current_model_instance:
                            resources = self.model_manager.resource_monitor.get_current_resources()

                            # Check if system is overloaded
                            if self.model_manager.resource_monitor.is_system_overloaded():
                                self.logger.warning(
                                    "System resources exceeded thresholds, considering model switch"
                                )
                                # This would trigger proactive switching in next generation

                        # Wait before next check (configurable interval)
                        await asyncio.sleep(5)  # 5 second interval

                    except Exception as e:
                        self.logger.error(f"Error in background loop: {e}")
                        await asyncio.sleep(10)  # Wait longer on error

            # Start background task
            asyncio.create_task(background_loop())
            self.logger.info("Background monitoring tasks started")

        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def shutdown(self) -> None:
        """Clean up resources and shutdown gracefully."""
        try:
            self.running = False
            self.logger.info("Shutting down Mai...")

            # Shutdown model manager
            if hasattr(self, "model_manager"):
                self.model_manager.shutdown()

            self.logger.info("Mai shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def list_available_models(self) -> list:
        """
        List all available models from ModelManager.

        Returns:
            List of available model information
        """
        try:
            return self.model_manager.available_models
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []

    async def switch_model(self, model_key: str) -> bool:
        """
        Manually switch to a specific model.

        Args:
            model_key: Model identifier to switch to

        Returns:
            True if switch successful, False otherwise
        """
        try:
            return await self.model_manager.switch_model(model_key)
        except Exception as e:
            self.logger.error(f"Error switching model: {e}")
            return False

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_key: Model identifier

        Returns:
            Model information dictionary or None if not found
        """
        try:
            return self.model_manager.model_configurations.get(model_key)
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return None

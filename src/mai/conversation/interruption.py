"""
Interruption Handling for Mai Conversations

Provides graceful interruption handling during conversation processing
with thread-safe operations and conversation restart capabilities.
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

# Import conversation state for integration
try:
    from .state import ConversationState
except ImportError:
    # Fallback for standalone usage
    ConversationState = None

logger = logging.getLogger(__name__)


class TurnType(Enum):
    """Types of conversation turns for different input sources."""

    USER_INPUT = "user_input"
    SELF_REFLECTION = "self_reflection"
    CODE_EXECUTION = "code_execution"
    SYSTEM_NOTIFICATION = "system_notification"


@dataclass
class InterruptionContext:
    """Context for conversation interruption and restart."""

    interruption_id: str
    original_message: str
    new_message: str
    conversation_id: str
    turn_type: TurnType
    timestamp: float
    processing_time: float
    reason: str = "user_input"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "interruption_id": self.interruption_id,
            "original_message": self.original_message,
            "new_message": self.new_message,
            "conversation_id": self.conversation_id,
            "turn_type": self.turn_type.value,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "reason": self.reason,
        }


class InterruptHandler:
    """
    Manages graceful conversation interruptions and restarts.

    Provides thread-safe interruption detection, context preservation,
    and timeout-based protection for long-running operations.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize interruption handler.

        Args:
            timeout_seconds: Maximum processing time before auto-interruption
        """
        self.timeout_seconds = timeout_seconds
        self.interrupt_flag = False
        self.processing_lock = threading.RLock()
        self.state_lock = threading.RLock()

        # Track active processing contexts
        self.active_contexts: Dict[str, Dict[str, Any]] = {}

        # Conversation state integration
        self.conversation_state: Optional[ConversationState] = None

        # Statistics
        self.interruption_count = 0
        self.timeout_count = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"InterruptHandler initialized with {timeout_seconds}s timeout")

    def set_conversation_state(self, conversation_state: ConversationState) -> None:
        """
        Set conversation state for integration.

        Args:
            conversation_state: ConversationState instance for context management
        """
        with self.state_lock:
            self.conversation_state = conversation_state
            self.logger.debug("Conversation state integrated")

    def start_processing(
        self,
        message: str,
        conversation_id: str,
        turn_type: TurnType = TurnType.USER_INPUT,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start processing a conversation turn.

        Args:
            message: Message being processed
            conversation_id: ID of conversation
            turn_type: Type of conversation turn
            context: Additional processing context

        Returns:
            Processing context ID for tracking
        """
        processing_id = str(uuid.uuid4())
        start_time = time.time()

        with self.processing_lock:
            self.active_contexts[processing_id] = {
                "message": message,
                "conversation_id": conversation_id,
                "turn_type": turn_type,
                "context": context or {},
                "start_time": start_time,
                "timeout_timer": None,
            }

            # Reset interruption flag for new processing
            self.interrupt_flag = False

        self.logger.debug(
            f"Started processing {processing_id}: {turn_type.value} for conversation {conversation_id}"
        )
        return processing_id

    def check_interruption(self, processing_id: Optional[str] = None) -> bool:
        """
        Check if interruption occurred during processing.

        Args:
            processing_id: Specific processing context to check (optional)

        Returns:
            True if interruption detected, False otherwise
        """
        with self.processing_lock:
            # Check global interruption flag
            was_interrupted = self.interrupt_flag

            # Check timeout for active contexts
            if processing_id and processing_id in self.active_contexts:
                context = self.active_contexts[processing_id]
                elapsed = time.time() - context["start_time"]

                if elapsed > self.timeout_seconds:
                    self.logger.info(f"Processing timeout for {processing_id} after {elapsed:.1f}s")
                    self.timeout_count += 1
                    was_interrupted = True

            # Reset flag after checking
            if was_interrupted:
                self.interrupt_flag = False
                self.interruption_count += 1

            return was_interrupted

    def interrupt_and_restart(
        self,
        new_message: str,
        conversation_id: str,
        turn_type: TurnType = TurnType.USER_INPUT,
        reason: str = "user_input",
    ) -> InterruptionContext:
        """
        Handle interruption and prepare for restart.

        Args:
            new_message: New message that triggered interruption
            conversation_id: ID of conversation
            turn_type: Type of new conversation turn
            reason: Reason for interruption

        Returns:
            InterruptionContext with restart information
        """
        interruption_id = str(uuid.uuid4())
        current_time = time.time()

        with self.processing_lock:
            # Find the active processing context for this conversation
            active_context = None
            original_message = ""
            processing_time = 0.0

            for proc_id, context in self.active_contexts.items():
                if context["conversation_id"] == conversation_id:
                    active_context = context
                    processing_time = current_time - context["start_time"]
                    original_message = context["message"]
                    break

            # Set interruption flag
            self.interrupt_flag = True

            # Clear pending response from conversation state
            if self.conversation_state:
                self.conversation_state.clear_pending_response(conversation_id)

            # Create interruption context
            interruption_context = InterruptionContext(
                interruption_id=interruption_id,
                original_message=original_message,
                new_message=new_message,
                conversation_id=conversation_id,
                turn_type=turn_type,
                timestamp=current_time,
                processing_time=processing_time,
                reason=reason,
            )

            self.logger.info(
                f"Interruption {interruption_id} for conversation {conversation_id}: {reason}"
            )

            return interruption_context

    def finish_processing(self, processing_id: str) -> None:
        """
        Mark processing as complete and cleanup context.

        Args:
            processing_id: Processing context ID to finish
        """
        with self.processing_lock:
            if processing_id in self.active_contexts:
                context = self.active_contexts[processing_id]
                elapsed = time.time() - context["start_time"]

                del self.active_contexts[processing_id]

                self.logger.debug(f"Finished processing {processing_id} in {elapsed:.2f}s")

    def get_active_processing(self, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get currently active processing contexts.

        Args:
            conversation_id: Filter by specific conversation (optional)

        Returns:
            List of active processing contexts
        """
        with self.processing_lock:
            active = []
            for proc_id, context in self.active_contexts.items():
                if conversation_id is None or context["conversation_id"] == conversation_id:
                    active_context = context.copy()
                    active_context["processing_id"] = proc_id
                    active_context["elapsed"] = time.time() - context["start_time"]
                    active.append(active_context)

            return active

    def cleanup_stale_processing(self, max_age_seconds: float = 300.0) -> int:
        """
        Clean up stale processing contexts.

        Args:
            max_age_seconds: Maximum age before cleanup

        Returns:
            Number of contexts cleaned up
        """
        current_time = time.time()
        stale_contexts = []

        with self.processing_lock:
            for proc_id, context in self.active_contexts.items():
                elapsed = current_time - context["start_time"]
                if elapsed > max_age_seconds:
                    stale_contexts.append(proc_id)

            for proc_id in stale_contexts:
                del self.active_contexts[proc_id]

        if stale_contexts:
            self.logger.info(f"Cleaned up {len(stale_contexts)} stale processing contexts")

        return len(stale_contexts)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get interruption handler statistics.

        Returns:
            Dictionary with performance and usage statistics
        """
        with self.processing_lock:
            return {
                "interruption_count": self.interruption_count,
                "timeout_count": self.timeout_count,
                "active_processing_count": len(self.active_contexts),
                "timeout_seconds": self.timeout_seconds,
                "last_activity": time.time(),
            }

    def configure_timeout(self, timeout_seconds: float) -> None:
        """
        Update timeout configuration.

        Args:
            timeout_seconds: New timeout value in seconds
        """
        with self.state_lock:
            self.timeout_seconds = max(5.0, timeout_seconds)  # Minimum 5 seconds
            self.logger.info(f"Timeout updated to {self.timeout_seconds}s")

    def reset_statistics(self) -> None:
        """Reset interruption handler statistics."""
        with self.state_lock:
            self.interruption_count = 0
            self.timeout_count = 0
            self.logger.info("Interruption statistics reset")

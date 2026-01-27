"""
Conversation State Management for Mai

Provides turn-by-turn conversation history with proper session isolation,
interruption handling, and context window management.
"""

import logging
import time
import threading
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# Import existing conversation models for consistency
try:
    from ..models.conversation import Message, Conversation
except ImportError:
    # Fallback if models not available yet
    Message = None
    Conversation = None

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn with comprehensive metadata."""

    conversation_id: str
    user_message: str
    ai_response: str
    timestamp: float
    model_used: str
    tokens_used: int
    response_time: float
    memory_context_applied: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "user_message": self.user_message,
            "ai_response": self.ai_response,
            "timestamp": self.timestamp,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "response_time": self.response_time,
            "memory_context_applied": self.memory_context_applied,
        }


class ConversationState:
    """
    Manages conversation state across multiple sessions with proper isolation.

    Provides turn-by-turn history tracking, automatic cleanup,
    thread-safe operations, and Ollama-compatible formatting.
    """

    def __init__(self, max_turns_per_conversation: int = 10):
        """
        Initialize conversation state manager.

        Args:
            max_turns_per_conversation: Maximum turns to keep per conversation
        """
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.max_turns = max_turns_per_conversation
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"ConversationState initialized with max {max_turns_per_conversation} turns per conversation"
        )

    def add_turn(self, turn: ConversationTurn) -> None:
        """
        Add a conversation turn with automatic timestamp and cleanup.

        Args:
            turn: ConversationTurn to add
        """
        with self._lock:
            conversation_id = turn.conversation_id

            # Initialize conversation if doesn't exist
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
                self.logger.debug(f"Created new conversation: {conversation_id}")

            # Add the turn
            self.conversations[conversation_id].append(turn)
            self.logger.debug(
                f"Added turn to conversation {conversation_id}: {turn.tokens_used} tokens, {turn.response_time:.2f}s"
            )

            # Automatic cleanup: maintain last N turns
            if len(self.conversations[conversation_id]) > self.max_turns:
                # Remove oldest turns to maintain limit
                excess_count = len(self.conversations[conversation_id]) - self.max_turns
                removed_turns = self.conversations[conversation_id][:excess_count]
                self.conversations[conversation_id] = self.conversations[conversation_id][
                    excess_count:
                ]

                self.logger.debug(
                    f"Cleaned up {excess_count} old turns from conversation {conversation_id}"
                )

                # Log removed turns for debugging
                for removed_turn in removed_turns:
                    self.logger.debug(
                        f"Removed turn: {removed_turn.timestamp} - {removed_turn.user_message[:50]}..."
                    )

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history in Ollama-compatible format.

        Args:
            conversation_id: ID of conversation to retrieve

        Returns:
            List of message dictionaries formatted for Ollama API
        """
        with self._lock:
            turns = self.conversations.get(conversation_id, [])

            # Convert to Ollama format: alternating user/assistant roles
            history = []
            for turn in turns:
                history.append({"role": "user", "content": turn.user_message})
                history.append({"role": "assistant", "content": turn.ai_response})

            self.logger.debug(
                f"Retrieved {len(history)} messages from conversation {conversation_id}"
            )
            return history

    def set_conversation_history(
        self, messages: List[Dict[str, str]], conversation_id: Optional[str] = None
    ) -> None:
        """
        Restore conversation history from session storage.

        Args:
            messages: List of message dictionaries in Ollama format [{"role": "user/assistant", "content": "..."}]
            conversation_id: Optional conversation ID to restore to (creates new if None)
        """
        with self._lock:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())

            # Clear existing conversation for this ID
            self.conversations[conversation_id] = []

            # Convert messages back to ConversationTurn objects
            # Messages should be in pairs: user, assistant, user, assistant, ...
            i = 0
            while i < len(messages):
                # Expect user message first
                if i >= len(messages) or messages[i].get("role") != "user":
                    self.logger.warning(f"Expected user message at index {i}, skipping")
                    i += 1
                    continue

                user_message = messages[i].get("content", "")
                i += 1

                # Expect assistant message next
                if i >= len(messages) or messages[i].get("role") != "assistant":
                    self.logger.warning(f"Expected assistant message at index {i}, skipping")
                    continue

                ai_response = messages[i].get("content", "")
                i += 1

                # Create ConversationTurn with estimated metadata
                turn = ConversationTurn(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    ai_response=ai_response,
                    timestamp=time.time(),  # Use current time as approximation
                    model_used="restored",  # Indicate this is from restoration
                    tokens_used=0,  # Token count not available from session
                    response_time=0.0,  # Response time not available from session
                    memory_context_applied=False,  # Memory context not tracked in session
                )

                self.conversations[conversation_id].append(turn)

            self.logger.info(
                f"Restored {len(self.conversations[conversation_id])} turns to conversation {conversation_id}"
            )

    def get_last_n_turns(self, conversation_id: str, n: int = 5) -> List[ConversationTurn]:
        """
        Get the last N turns from a conversation.

        Args:
            conversation_id: ID of conversation
            n: Number of recent turns to retrieve

        Returns:
            List of last N ConversationTurn objects
        """
        with self._lock:
            turns = self.conversations.get(conversation_id, [])
            return turns[-n:] if n > 0 else []

    def clear_pending_response(self, conversation_id: str) -> None:
        """
        Clear any pending response for interruption handling.

        Args:
            conversation_id: ID of conversation to clear
        """
        with self._lock:
            if conversation_id in self.conversations:
                # Find and remove incomplete turns (those without AI response)
                original_count = len(self.conversations[conversation_id])
                self.conversations[conversation_id] = [
                    turn
                    for turn in self.conversations[conversation_id]
                    if turn.ai_response.strip()  # Must have AI response
                ]

                removed_count = original_count - len(self.conversations[conversation_id])
                if removed_count > 0:
                    self.logger.info(
                        f"Cleared {removed_count} incomplete turns from conversation {conversation_id}"
                    )

    def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Start a new conversation or return existing ID.

        Args:
            conversation_id: Optional existing conversation ID

        Returns:
            Conversation ID (new or existing)
        """
        with self._lock:
            if conversation_id is None:
                conversation_id = str(uuid.uuid4())

            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
                self.logger.debug(f"Started new conversation: {conversation_id}")

            return conversation_id

    def is_processing(self, conversation_id: str) -> bool:
        """
        Check if conversation is currently being processed.

        Args:
            conversation_id: ID of conversation

        Returns:
            True if currently processing, False otherwise
        """
        with self._lock:
            return hasattr(self, "_processing_locks") and conversation_id in getattr(
                self, "_processing_locks", {}
            )

    def set_processing(self, conversation_id: str, processing: bool) -> None:
        """
        Set processing lock for conversation.

        Args:
            conversation_id: ID of conversation
            processing: Processing state
        """
        with self._lock:
            if not hasattr(self, "_processing_locks"):
                self._processing_locks = {}
            self._processing_locks[conversation_id] = processing

    def get_conversation_turns(self, conversation_id: str) -> List[ConversationTurn]:
        """
        Get all turns for a conversation.

        Args:
            conversation_id: ID of conversation

        Returns:
            List of ConversationTurn objects
        """
        with self._lock:
            return self.conversations.get(conversation_id, [])

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation completely.

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            True if conversation was deleted, False if not found
        """
        with self._lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                self.logger.info(f"Deleted conversation: {conversation_id}")
                return True
            return False

    def list_conversations(self) -> List[str]:
        """
        List all active conversation IDs.

        Returns:
            List of conversation IDs
        """
        with self._lock:
            return list(self.conversations.keys())

    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific conversation.

        Args:
            conversation_id: ID of conversation

        Returns:
            Dictionary with conversation statistics
        """
        with self._lock:
            turns = self.conversations.get(conversation_id, [])

            if not turns:
                return {
                    "turn_count": 0,
                    "total_tokens": 0,
                    "total_response_time": 0.0,
                    "average_response_time": 0.0,
                    "average_tokens": 0.0,
                }

            total_tokens = sum(turn.tokens_used for turn in turns)
            total_response_time = sum(turn.response_time for turn in turns)
            avg_response_time = total_response_time / len(turns)
            avg_tokens = total_tokens / len(turns)

            return {
                "turn_count": len(turns),
                "total_tokens": total_tokens,
                "total_response_time": total_response_time,
                "average_response_time": avg_response_time,
                "average_tokens": avg_tokens,
                "oldest_timestamp": min(turn.timestamp for turn in turns),
                "newest_timestamp": max(turn.timestamp for turn in turns),
            }

    def cleanup_old_conversations(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up conversations older than specified age.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of conversations cleaned up
        """
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)

            conversations_to_remove = []
            for conv_id, turns in self.conversations.items():
                if turns and turns[-1].timestamp < cutoff_time:
                    conversations_to_remove.append(conv_id)

            for conv_id in conversations_to_remove:
                del self.conversations[conv_id]

            if conversations_to_remove:
                self.logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")

            return len(conversations_to_remove)

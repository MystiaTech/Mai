"""
Context manager for conversation history and memory compression.

This module implements intelligent context window management with hybrid compression
strategies to maintain conversation continuity while respecting token limits.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import re

from .conversation import (
    Message,
    Conversation,
    ContextBudget,
    ContextWindow,
    MessageRole,
    MessageType,
    MessageMetadata,
    calculate_importance_score,
    estimate_token_count,
)


class CompressionStrategy:
    """Strategies for compressing conversation history."""

    @staticmethod
    def create_summary(messages: List[Message]) -> str:
        """
        Create a summary of compressed messages.

        This is a simple rule-based approach - in production, this could use
        an LLM to generate more sophisticated summaries.
        """
        if not messages:
            return ""

        # Extract key information
        user_instructions = []
        questions = []
        key_topics = []

        for msg in messages:
            if msg.role == MessageRole.USER:
                content_lower = msg.content.lower()
                if any(
                    word in content_lower
                    for word in ["please", "help", "create", "implement", "fix"]
                ):
                    user_instructions.append(
                        msg.content[:100] + "..."
                        if len(msg.content) > 100
                        else msg.content
                    )
                elif "?" in msg.content:
                    questions.append(
                        msg.content[:100] + "..."
                        if len(msg.content) > 100
                        else msg.content
                    )

            # Extract simple topic keywords
            words = re.findall(r"\b\w+\b", msg.content.lower())
            technical_terms = [w for w in words if len(w) > 6 and w.isalpha()]
            key_topics.extend(technical_terms[:3])

        # Build summary
        summary_parts = []

        if user_instructions:
            summary_parts.append(f"User requested: {'; '.join(user_instructions[:3])}")

        if questions:
            summary_parts.append(f"Key questions: {'; '.join(questions[:2])}")

        if key_topics:
            topic_counts = {}
            for topic in key_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            summary_parts.append(
                f"Topics discussed: {', '.join([topic for topic, _ in top_topics])}"
            )

        summary = " | ".join(summary_parts)
        return summary[:500] + "..." if len(summary) > 500 else summary

    @staticmethod
    def score_message_importance(message: Message, context: Dict[str, Any]) -> float:
        """
        Score message importance for retention during compression.
        """
        base_score = calculate_importance_score(message)

        # Factor in recency (more recent = slightly more important)
        if "current_time" in context:
            age_hours = (
                context["current_time"] - message.timestamp
            ).total_seconds() / 3600
            recency_factor = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
            base_score *= recency_factor

        # Boost for messages that started new topics
        if message.role == MessageRole.USER and len(message.content) > 50:
            # Likely a new topic or detailed request
            base_score *= 1.2

        # Boost for assistant responses that contain code or structured data
        if message.role == MessageRole.ASSISTANT:
            if (
                "```" in message.content
                or "def " in message.content
                or "class " in message.content
            ):
                base_score *= 1.3

        return min(1.0, base_score)


class ContextManager:
    """
    Manages conversation context with intelligent compression and token budgeting.
    """

    def __init__(
        self, default_context_size: int = 4096, compression_threshold: float = 0.7
    ):
        """
        Initialize context manager.

        Args:
            default_context_size: Default token limit for context windows
            compression_threshold: When to trigger compression (0.0-1.0)
        """
        self.default_context_size = default_context_size
        self.compression_threshold = compression_threshold
        self.conversations: Dict[str, Conversation] = {}
        self.context_windows: Dict[str, ContextWindow] = {}
        self.compression_strategy = CompressionStrategy()

    def create_conversation(
        self, conversation_id: str, model_context_size: Optional[int] = None
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            model_context_size: Specific model's context size (uses default if None)

        Returns:
            Created conversation object
        """
        context_size = model_context_size or self.default_context_size

        metadata = {"session_id": conversation_id, "context_window_size": context_size}

        conversation = Conversation(id=conversation_id, metadata=metadata)

        self.conversations[conversation_id] = conversation
        self.context_windows[conversation_id] = ContextWindow(
            budget=ContextBudget(
                max_tokens=context_size,
                compression_threshold=self.compression_threshold,
            )
        )

        return conversation

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            conversation_id: Target conversation ID
            role: Message role (user/assistant/system/tool)
            content: Message content
            metadata: Optional additional metadata

        Returns:
            Created message object
        """
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        # Create message
        message_id = hashlib.md5(
            f"{conversation_id}_{datetime.utcnow().isoformat()}_{len(self.conversations[conversation_id].messages)}".encode()
        ).hexdigest()[:12]

        msg_metadata = MessageMetadata()
        if metadata:
            for key, value in metadata.items():
                if hasattr(msg_metadata, key):
                    setattr(msg_metadata, key, value)

        # Determine message type and set priority
        if role == MessageRole.USER:
            if any(
                word in content.lower()
                for word in ["please", "help", "create", "implement", "fix"]
            ):
                msg_metadata.message_type = MessageType.INSTRUCTION
                msg_metadata.priority = 0.8
            elif "?" in content:
                msg_metadata.message_type = MessageType.QUESTION
                msg_metadata.priority = 0.6
            else:
                msg_metadata.message_type = MessageType.CONTEXT
                msg_metadata.priority = 0.4
        elif role == MessageRole.SYSTEM:
            msg_metadata.message_type = MessageType.SYSTEM
            msg_metadata.priority = 0.9
            msg_metadata.is_permanent = True
        elif role == MessageRole.ASSISTANT:
            msg_metadata.message_type = MessageType.RESPONSE
            msg_metadata.priority = 0.5

        message = Message(
            id=message_id,
            role=role,
            content=content,
            token_count=estimate_token_count(content),
            metadata=msg_metadata,
        )

        # Calculate importance score
        message.importance_score = self.compression_strategy.score_message_importance(
            message, {"current_time": datetime.utcnow()}
        )

        # Add to conversation
        conversation = self.conversations[conversation_id]
        conversation.add_message(message)

        # Add to context window and check compression
        context_window = self.context_windows[conversation_id]
        context_window.add_message(message)

        # Check if compression is needed
        if context_window.budget.should_compress:
            self.compress_conversation(conversation_id)

        return message

    def get_context_for_model(
        self, conversation_id: str, max_tokens: Optional[int] = None
    ) -> List[Message]:
        """
        Get context messages for a model, respecting token limits.

        Args:
            conversation_id: Conversation ID
            max_tokens: Maximum tokens (uses conversation default if None)

        Returns:
            List of messages in chronological order within token limit
        """
        if conversation_id not in self.context_windows:
            return []

        context_window = self.context_windows[conversation_id]
        effective_context = context_window.get_effective_context()

        # Apply token limit if specified
        if max_tokens is None:
            max_tokens = context_window.budget.max_tokens

        # If we're within limits, return as-is
        total_tokens = sum(msg.token_count for msg in effective_context)
        if total_tokens <= max_tokens:
            return effective_context

        # Otherwise, apply sliding window from most recent
        result = []
        current_tokens = 0

        # Iterate backwards (most recent first)
        for message in reversed(effective_context):
            if current_tokens + message.token_count <= max_tokens:
                result.insert(0, message)  # Insert at beginning to maintain order
                current_tokens += message.token_count
            else:
                break

        return result

    def compress_conversation(
        self, conversation_id: str, target_ratio: float = 0.5
    ) -> bool:
        """
        Compress conversation history using hybrid strategy.

        Args:
            conversation_id: Conversation to compress
            target_ratio: Target ratio of original size to keep

        Returns:
            True if compression was performed, False otherwise
        """
        if conversation_id not in self.conversations:
            return False

        conversation = self.conversations[conversation_id]
        context_window = self.context_windows[conversation_id]

        # Get all messages from context (excluding permanent ones)
        compressible_messages = [
            msg for msg in context_window.messages if not msg.metadata.is_permanent
        ]

        if len(compressible_messages) < 3:  # Need some messages to compress
            return False

        # Sort by importance (ascending - least important first)
        compressible_messages.sort(key=lambda m: m.importance_score)

        # Calculate target count
        target_count = max(2, int(len(compressible_messages) * target_ratio))
        messages_to_compress = compressible_messages[:-target_count]
        messages_to_keep = compressible_messages[-target_count:]

        if not messages_to_compress:
            return False

        # Create summary of compressed messages
        summary = self.compression_strategy.create_summary(messages_to_compress)

        # Update context window
        context_window.messages = [
            msg
            for msg in context_window.messages
            if msg.metadata.is_permanent or msg in messages_to_keep
        ]

        context_window.compressed_summary = summary

        # Recalculate token usage
        total_tokens = sum(msg.token_count for msg in context_window.messages)
        if summary:
            summary_tokens = estimate_token_count(summary)
            total_tokens += summary_tokens

        context_window.budget.used_tokens = total_tokens

        return True

    def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """
        Get a summary of the entire conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation summary or None if not available
        """
        if conversation_id not in self.context_windows:
            return None

        context_window = self.context_windows[conversation_id]
        if context_window.compressed_summary:
            # Combine current summary with remaining recent messages
            recent_content = " | ".join(
                [
                    f"{msg.role.value}: {msg.content[:100]}..."
                    for msg in context_window.messages[-3:]
                ]
            )
            return f"{context_window.compressed_summary} | Recent: {recent_content}"

        # Generate quick summary of recent messages
        if context_window.messages:
            recent_messages = context_window.messages[-5:]
            return " | ".join(
                [f"{msg.role.value}: {msg.content[:80]}..." for msg in recent_messages]
            )

        return None

    def clear_conversation(
        self, conversation_id: str, keep_system: bool = True
    ) -> None:
        """
        Clear a conversation's messages.

        Args:
            conversation_id: Conversation ID to clear
            keep_system: Whether to keep system messages
        """
        if conversation_id in self.conversations:
            self.conversations[conversation_id].clear_messages(keep_system)

        if conversation_id in self.context_windows:
            self.context_windows[conversation_id].clear()

    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics about a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Dictionary of conversation statistics
        """
        if conversation_id not in self.conversations:
            return {}

        conversation = self.conversations[conversation_id]
        context_window = self.context_windows.get(conversation_id)

        stats = {
            "conversation_id": conversation_id,
            "total_messages": len(conversation.messages),
            "total_tokens": conversation.metadata.total_tokens,
            "session_duration": (
                conversation.metadata.last_active - conversation.metadata.created_at
            ).total_seconds(),
            "messages_by_role": {},
        }

        # Count by role
        for role in MessageRole:
            count = len([msg for msg in conversation.messages if msg.role == role])
            if count > 0:
                stats["messages_by_role"][role.value] = count

        # Add context window stats if available
        if context_window:
            stats.update(
                {
                    "context_usage_percentage": context_window.budget.usage_percentage,
                    "context_should_compress": context_window.budget.should_compress,
                    "context_compressed": context_window.compressed_summary is not None,
                    "context_tokens_used": context_window.budget.used_tokens,
                    "context_tokens_max": context_window.budget.max_tokens,
                }
            )

        return stats

    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations with basic info.

        Returns:
            List of conversation summaries
        """
        return [
            {
                "id": conv_id,
                "message_count": len(conv.messages),
                "total_tokens": conv.metadata.total_tokens,
                "last_active": conv.metadata.last_active.isoformat(),
                "session_id": conv.metadata.session_id,
            }
            for conv_id, conv in self.conversations.items()
        ]

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation ID to delete

        Returns:
            True if deleted, False if not found
        """
        deleted = conversation_id in self.conversations
        if deleted:
            del self.conversations[conversation_id]
            del self.context_windows[conversation_id]
        return deleted

"""
Conversation data models and types for Mai.

This module defines the core data structures for managing conversations,
messages, and context windows. Provides type-safe models with validation
using Pydantic for serialization and data integrity.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """Message role types in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class MessageType(str, Enum):
    """Message type classifications for importance scoring."""

    INSTRUCTION = "instruction"  # User instructions, high priority
    QUESTION = "question"  # User questions, medium priority
    RESPONSE = "response"  # Assistant responses, medium priority
    SYSTEM = "system"  # System messages, high priority
    CONTEXT = "context"  # Context/background, low priority
    ERROR = "error"  # Error messages, variable priority


class MessageMetadata(BaseModel):
    """Metadata for messages including source and importance indicators."""

    source: str = Field(default="conversation", description="Source of the message")
    message_type: MessageType = Field(
        default=MessageType.CONTEXT, description="Type classification"
    )
    priority: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Priority score 0-1"
    )
    context_tags: List[str] = Field(
        default_factory=list, description="Context tags for retrieval"
    )
    is_permanent: bool = Field(default=False, description="Never compress this message")
    tool_name: Optional[str] = Field(
        default=None, description="Tool name for tool calls"
    )
    model_used: Optional[str] = Field(
        default=None, description="Model that generated this message"
    )


class Message(BaseModel):
    """Individual message in a conversation."""

    id: str = Field(description="Unique message identifier")
    role: MessageRole = Field(description="Message role (user/assistant/system/tool)")
    content: str = Field(description="Message content text")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Message creation time"
    )
    token_count: int = Field(default=0, description="Estimated token count")
    importance_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance for compression"
    )
    metadata: MessageMetadata = Field(
        default_factory=MessageMetadata, description="Additional metadata"
    )

    @validator("content")
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConversationMetadata(BaseModel):
    """Metadata for conversation sessions."""

    session_id: str = Field(description="Unique session identifier")
    title: Optional[str] = Field(default=None, description="Conversation title")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Session start time"
    )
    last_active: datetime = Field(
        default_factory=datetime.utcnow, description="Last activity time"
    )
    total_messages: int = Field(default=0, description="Total message count")
    total_tokens: int = Field(default=0, description="Total token count")
    model_history: List[str] = Field(
        default_factory=list, description="Models used in this session"
    )
    context_window_size: int = Field(
        default=4096, description="Context window size for this session"
    )


class Conversation(BaseModel):
    """Conversation manager for message sequences and metadata."""

    id: str = Field(description="Conversation identifier")
    messages: List[Message] = Field(
        default_factory=list, description="Messages in chronological order"
    )
    metadata: ConversationMetadata = Field(description="Conversation metadata")

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.metadata.total_messages = len(self.messages)
        self.metadata.total_tokens += message.token_count
        self.metadata.last_active = datetime.utcnow()

    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """Get all messages from a specific role."""
        return [msg for msg in self.messages if msg.role == role]

    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent N messages."""
        return self.messages[-count:] if count > 0 else []

    def get_message_range(self, start: int, end: Optional[int] = None) -> List[Message]:
        """Get messages in a range (start inclusive, end exclusive)."""
        if end is None:
            end = len(self.messages)
        return self.messages[start:end]

    def clear_messages(self, keep_system: bool = True) -> None:
        """Clear all messages, optionally keeping system messages."""
        if keep_system:
            self.messages = [
                msg for msg in self.messages if msg.role == MessageRole.SYSTEM
            ]
        else:
            self.messages.clear()
        self.metadata.total_messages = len(self.messages)
        self.metadata.total_tokens = sum(msg.token_count for msg in self.messages)


class ContextBudget(BaseModel):
    """Token budget tracker for context window management."""

    max_tokens: int = Field(description="Maximum tokens allowed")
    used_tokens: int = Field(default=0, description="Tokens currently used")
    compression_threshold: float = Field(
        default=0.7, description="Compression trigger ratio"
    )
    safety_margin: int = Field(default=100, description="Safety margin tokens")

    @property
    def available_tokens(self) -> int:
        """Calculate available tokens including safety margin."""
        return max(0, self.max_tokens - self.used_tokens - self.safety_margin)

    @property
    def usage_percentage(self) -> float:
        """Calculate current usage as percentage."""
        if self.max_tokens == 0:
            return 0.0
        return min(1.0, self.used_tokens / self.max_tokens)

    @property
    def should_compress(self) -> bool:
        """Check if compression should be triggered."""
        return self.usage_percentage >= self.compression_threshold

    def add_tokens(self, count: int) -> None:
        """Add tokens to the used count."""
        self.used_tokens += count
        self.used_tokens = max(0, self.used_tokens)  # Prevent negative

    def remove_tokens(self, count: int) -> None:
        """Remove tokens from the used count."""
        self.used_tokens -= count
        self.used_tokens = max(0, self.used_tokens)

    def reset(self) -> None:
        """Reset the token budget."""
        self.used_tokens = 0


class ContextWindow(BaseModel):
    """Context window representation with compression state."""

    messages: List[Message] = Field(
        default_factory=list, description="Current context messages"
    )
    budget: ContextBudget = Field(description="Token budget for this window")
    compressed_summary: Optional[str] = Field(
        default=None, description="Summary of compressed messages"
    )
    original_token_count: int = Field(
        default=0, description="Tokens before compression"
    )

    def add_message(self, message: Message) -> None:
        """Add a message to the context window."""
        self.messages.append(message)
        self.budget.add_tokens(message.token_count)
        self.original_token_count += message.token_count

    def get_effective_context(self) -> List[Message]:
        """Get the effective context including compressed summary if needed."""
        if self.compressed_summary:
            # Create a synthetic system message with the summary
            summary_msg = Message(
                id="compressed_summary",
                role=MessageRole.SYSTEM,
                content=f"[Previous conversation summary]\n{self.compressed_summary}",
                importance_score=0.8,  # High importance for summary
                metadata=MessageMetadata(
                    message_type=MessageType.SYSTEM,
                    is_permanent=True,
                    source="compression",
                ),
            )
            return [summary_msg] + self.messages
        return self.messages

    def clear(self) -> None:
        """Clear the context window."""
        self.messages.clear()
        self.budget.reset()
        self.compressed_summary = None
        self.original_token_count = 0


# Utility functions for message importance scoring
def calculate_importance_score(message: Message) -> float:
    """Calculate importance score for a message based on various factors."""
    score = message.metadata.priority

    # Boost for instructions and system messages
    if message.metadata.message_type in [MessageType.INSTRUCTION, MessageType.SYSTEM]:
        score = min(1.0, score + 0.3)

    # Boost for permanent messages
    if message.metadata.is_permanent:
        score = min(1.0, score + 0.4)

    # Boost for questions (user seeking information)
    if message.metadata.message_type == MessageType.QUESTION:
        score = min(1.0, score + 0.2)

    # Adjust based on length (longer messages might be more detailed)
    if message.token_count > 100:
        score = min(1.0, score + 0.1)

    return score


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.

    This is a rough approximation - actual tokenization depends on the model.
    As a heuristic: ~4 characters per token for English text.
    """
    if not text:
        return 0

    # Simple heuristic: ~4 characters per token, adjusted for structure
    base_count = len(text) // 4

    # Add extra for special characters, code blocks, etc.
    special_chars = len([c for c in text if not c.isalnum() and not c.isspace()])
    special_adjustment = special_chars // 10

    # Add for newlines (often indicate more tokens)
    newline_adjustment = text.count("\n") // 2

    return max(1, base_count + special_adjustment + newline_adjustment)

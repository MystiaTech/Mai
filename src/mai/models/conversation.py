"""
Conversation data models for Mai memory system.

Provides Pydantic models for conversations, messages, and related
data structures with proper validation and serialization.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import json


class Message(BaseModel):
    """Individual message within a conversation."""

    id: str = Field(..., description="Unique message identifier")
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content text")
    timestamp: str = Field(..., description="ISO timestamp of message")
    token_count: Optional[int] = Field(0, description="Token count for message")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional message metadata"
    )

    @validator("role")
    def validate_role(cls, v):
        """Validate that role is one of the allowed values."""
        allowed_roles = ["user", "assistant", "system"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {allowed_roles}")
        return v

    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Validate timestamp format and ensure it's ISO format."""
        try:
            # Try to parse the timestamp to ensure it's valid
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            # Return in standard ISO format
            return dt.isoformat()
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid timestamp format: {v}. Must be ISO format.") from e

    class Config:
        """Pydantic configuration for Message model."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class Conversation(BaseModel):
    """Complete conversation with messages and metadata."""

    id: str = Field(..., description="Unique conversation identifier")
    title: str = Field(..., description="Human-readable conversation title")
    created_at: str = Field(..., description="ISO timestamp when conversation was created")
    updated_at: str = Field(..., description="ISO timestamp when conversation was last updated")
    messages: List[Message] = Field(
        default_factory=list, description="List of messages in chronological order"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional conversation metadata"
    )
    message_count: Optional[int] = Field(0, description="Total number of messages")

    @validator("messages")
    def validate_message_order(cls, v):
        """Ensure messages are in chronological order."""
        if not v:
            return v

        # Sort by timestamp to ensure chronological order
        try:
            sorted_messages = sorted(
                v, key=lambda m: datetime.fromisoformat(m.timestamp.replace("Z", "+00:00"))
            )
            return sorted_messages
        except (ValueError, AttributeError) as e:
            raise ValueError("Messages have invalid timestamps") from e

    @validator("updated_at")
    def validate_updated_timestamp(cls, v, values):
        """Ensure updated_at is not earlier than created_at."""
        if "created_at" in values:
            try:
                created = datetime.fromisoformat(values["created_at"].replace("Z", "+00:00"))
                updated = datetime.fromisoformat(v.replace("Z", "+00:00"))

                if updated < created:
                    raise ValueError("updated_at cannot be earlier than created_at")
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid timestamp comparison: {e}") from e

        return v

    def add_message(self, message: Message) -> None:
        """
        Add a message to the conversation and update timestamps.

        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.message_count = len(self.messages)

        # Update the updated_at timestamp
        self.updated_at = datetime.now().isoformat()

    def get_message_count(self) -> int:
        """Get the actual message count."""
        return len(self.messages)

    def get_latest_message(self) -> Optional[Message]:
        """Get the most recent message in the conversation."""
        if not self.messages:
            return None

        # Return the message with the latest timestamp
        return max(
            self.messages, key=lambda m: datetime.fromisoformat(m.timestamp.replace("Z", "+00:00"))
        )

    class Config:
        """Pydantic configuration for Conversation model."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ConversationSummary(BaseModel):
    """Summary of a conversation for search results."""

    id: str = Field(..., description="Conversation identifier")
    title: str = Field(..., description="Conversation title")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Total messages in conversation")
    preview: Optional[str] = Field(None, description="Short preview of conversation content")
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Tags or keywords for conversation"
    )

    class Config:
        """Pydantic configuration for ConversationSummary model."""

        pass


class ConversationFilter(BaseModel):
    """Filter criteria for searching conversations."""

    role: Optional[str] = Field(None, description="Filter by message role")
    start_date: Optional[str] = Field(
        None, description="Filter messages after this date (ISO format)"
    )
    end_date: Optional[str] = Field(
        None, description="Filter messages before this date (ISO format)"
    )
    keywords: Optional[List[str]] = Field(None, description="Filter by keywords in message content")
    min_message_count: Optional[int] = Field(None, description="Minimum message count")
    max_message_count: Optional[int] = Field(None, description="Maximum message count")

    @validator("start_date", "end_date")
    def validate_date_filters(cls, v):
        """Validate date filter format."""
        if v is None:
            return v

        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid date format: {v}. Must be ISO format.") from e

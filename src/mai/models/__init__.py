"""
Mai data models package.

Exports all Pydantic models for conversations, memory, and related data structures.
"""

from .conversation import (
    Message,
    Conversation,
    ConversationSummary,
    ConversationFilter,
)

from .memory import (
    ConversationType,
    RelevanceType,
    SearchQuery,
    RetrievalResult,
    MemoryContext,
    ContextWeight,
    ConversationPattern,
    ContextPlacement,
)

__all__ = [
    # Conversation models
    "Message",
    "Conversation",
    "ConversationSummary",
    "ConversationFilter",
    # Memory models
    "ConversationType",
    "RelevanceType",
    "SearchQuery",
    "RetrievalResult",
    "MemoryContext",
    "ContextWeight",
    "ConversationPattern",
    "ContextPlacement",
]

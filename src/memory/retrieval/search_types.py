"""
Search result data structures for memory retrieval.

This module defines common data types for search results across
different search strategies including relevance scoring and metadata.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class SearchResult:
    """
    Represents a single search result from memory retrieval.

    Combines conversation data with relevance scoring and snippet
    generation for effective search result presentation.
    """

    conversation_id: str
    message_id: str
    content: str
    relevance_score: float
    snippet: str
    timestamp: datetime
    metadata: Dict[str, Any]
    search_type: str  # "semantic", "keyword", "context_aware", "timeline"

    def __post_init__(self):
        """Validate search result data."""
        if not self.conversation_id:
            raise ValueError("conversation_id is required")
        if not self.message_id:
            raise ValueError("message_id is required")
        if not self.content:
            raise ValueError("content is required")
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("relevance_score must be between 0.0 and 1.0")


@dataclass
class SearchQuery:
    """
    Represents a search query with optional filters and parameters.

    Encapsulates search intent, constraints, and ranking preferences
    for flexible search execution.
    """

    query: str
    limit: int = 5
    search_types: Optional[List[str]] = None  # None means all types
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    current_topic: Optional[str] = None
    min_relevance: float = 0.0

    def __post_init__(self):
        """Validate search query parameters."""
        if not self.query or not self.query.strip():
            raise ValueError("query is required and cannot be empty")
        if self.limit <= 0:
            raise ValueError("limit must be positive")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0.0 and 1.0")

        if self.search_types is None:
            self.search_types = ["semantic", "keyword", "context_aware", "timeline"]

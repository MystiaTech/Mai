"""
Memory system data models for Mai context retrieval.

Provides Pydantic models for memory context, search queries,
retrieval results, and related data structures.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

from .conversation import Conversation, Message


class ConversationType(str, Enum):
    """Enumeration of conversation types for adaptive weighting."""

    TECHNICAL = "technical"
    PERSONAL = "personal"
    PLANNING = "planning"
    GENERAL = "general"
    QUESTION = "question"
    CREATIVE = "creative"
    ANALYSIS = "analysis"


class RelevanceType(str, Enum):
    """Enumeration of relevance types for search results."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    RECENCY = "recency"
    PATTERN = "pattern"
    HYBRID = "hybrid"


class SearchQuery(BaseModel):
    """Query model for context search operations."""

    text: str = Field(..., description="Search query text")
    conversation_type: Optional[ConversationType] = Field(
        None, description="Detected conversation type"
    )
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    weights: Optional[Dict[str, float]] = Field(
        default_factory=dict, description="Search weight overrides"
    )
    limits: Optional[Dict[str, int]] = Field(default_factory=dict, description="Search limits")

    # Default limits
    max_results: int = Field(5, description="Maximum number of results to return")
    max_tokens: int = Field(2000, description="Maximum tokens in returned context")

    # Search facet controls
    include_semantic: bool = Field(True, description="Include semantic similarity search")
    include_keywords: bool = Field(True, description="Include keyword matching")
    include_recency: bool = Field(True, description="Include recency weighting")
    include_patterns: bool = Field(True, description="Include pattern matching")

    @validator("text")
    def validate_text(cls, v):
        """Validate search text is not empty."""
        if not v or not v.strip():
            raise ValueError("Search text cannot be empty")
        return v.strip()

    @validator("max_results")
    def validate_max_results(cls, v):
        """Validate max results is reasonable."""
        if v < 1:
            raise ValueError("max_results must be at least 1")
        if v > 20:
            raise ValueError("max_results cannot exceed 20")
        return v


class RetrievalResult(BaseModel):
    """Single result from context retrieval operation."""

    conversation_id: str = Field(..., description="ID of the conversation")
    title: str = Field(..., description="Title of the conversation")
    similarity_score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    relevance_type: RelevanceType = Field(..., description="Type of relevance")
    excerpt: str = Field(..., description="Relevant excerpt from conversation")
    context_type: Optional[ConversationType] = Field(None, description="Type of conversation")
    matched_message_id: Optional[str] = Field(None, description="ID of the best matching message")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional result metadata"
    )

    # Component scores for hybrid results
    semantic_score: Optional[float] = Field(None, description="Semantic similarity score")
    keyword_score: Optional[float] = Field(None, description="Keyword matching score")
    recency_score: Optional[float] = Field(None, description="Recency-based score")
    pattern_score: Optional[float] = Field(None, description="Pattern matching score")

    @validator("similarity_score")
    def validate_similarity_score(cls, v):
        """Validate similarity score is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("similarity_score must be between 0.0 and 1.0")
        return v

    @validator("excerpt")
    def validate_excerpt(cls, v):
        """Validate excerpt is not empty."""
        if not v or not v.strip():
            raise ValueError("excerpt cannot be empty")
        return v.strip()


class MemoryContext(BaseModel):
    """Complete memory context for current query."""

    current_query: SearchQuery = Field(..., description="The search query")
    relevant_conversations: List[RetrievalResult] = Field(
        default_factory=list, description="Retrieved conversations"
    )
    patterns: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Extracted patterns"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional context metadata"
    )

    # Context statistics
    total_conversations: int = Field(0, description="Total conversations found")
    total_tokens: int = Field(0, description="Total tokens in retrieved context")
    context_quality_score: Optional[float] = Field(
        None, description="Quality assessment of context"
    )

    # Weighting information
    applied_weights: Optional[Dict[str, float]] = Field(
        default_factory=dict, description="Weights applied to search"
    )
    conversation_type_detected: Optional[ConversationType] = Field(
        None, description="Detected conversation type"
    )

    def add_result(self, result: RetrievalResult) -> None:
        """Add a retrieval result to the context."""
        self.relevant_conversations.append(result)
        self.total_conversations = len(self.relevant_conversations)
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        self.total_tokens += len(result.excerpt) // 4

    def is_within_token_limit(self, max_tokens: Optional[int] = None) -> bool:
        """Check if context is within token limits."""
        limit = max_tokens or self.current_query.max_tokens
        return self.total_tokens <= limit

    def get_summary_text(self, max_chars: int = 500) -> str:
        """Get a summary of the retrieved context."""
        if not self.relevant_conversations:
            return "No relevant conversations found."

        summaries = []
        total_chars = 0

        for result in self.relevant_conversations[:3]:  # Top 3 results
            summary = f"{result.title}: {result.excerpt[:200]}..."
            if total_chars + len(summary) > max_chars:
                break
            summaries.append(summary)
            total_chars += len(summary)

        return " | ".join(summaries)

    class Config:
        """Pydantic configuration for MemoryContext model."""

        pass


class ContextWeight(BaseModel):
    """Weight configuration for different search facets."""

    semantic: float = Field(0.4, description="Weight for semantic similarity")
    keyword: float = Field(0.3, description="Weight for keyword matching")
    recency: float = Field(0.2, description="Weight for recency")
    pattern: float = Field(0.1, description="Weight for pattern matching")

    @validator("semantic", "keyword", "recency", "pattern")
    def validate_weights(cls, v):
        """Validate individual weights are non-negative."""
        if v < 0:
            raise ValueError("Weights cannot be negative")
        return v

    @validator("semantic", "keyword", "recency", "pattern")
    def validate_weight_range(cls, v):
        """Validate weights are reasonable."""
        if v > 2.0:
            raise ValueError("Individual weights cannot exceed 2.0")
        return v

    def normalize(self) -> "ContextWeight":
        """Normalize weights so they sum to 1.0."""
        total = self.semantic + self.keyword + self.recency + self.pattern
        if total == 0:
            return ContextWeight()

        return ContextWeight(
            semantic=self.semantic / total,
            keyword=self.keyword / total,
            recency=self.recency / total,
            pattern=self.pattern / total,
        )


class ConversationPattern(BaseModel):
    """Extracted pattern from conversations."""

    pattern_type: str = Field(..., description="Type of pattern (preference, topic, style, etc.)")
    pattern_value: str = Field(..., description="Pattern value or description")
    confidence: float = Field(..., description="Confidence score for pattern")
    frequency: int = Field(1, description="How often this pattern appears")
    conversation_ids: List[str] = Field(
        default_factory=list, description="Conversations where pattern appears"
    )
    last_seen: str = Field(..., description="ISO timestamp when pattern was last observed")

    @validator("confidence")
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    class Config:
        """Pydantic configuration for ConversationPattern model."""

        pass


class ContextPlacement(BaseModel):
    """Strategy for placing context to prevent 'lost in middle'."""

    strategy: str = Field(..., description="Placement strategy name")
    reasoning: str = Field(..., description="Why this strategy was chosen")
    high_priority_items: List[int] = Field(
        default_factory=list, description="Indices of high priority conversations"
    )
    distributed_items: List[int] = Field(
        default_factory=list, description="Indices of distributed conversations"
    )
    token_allocation: Dict[str, int] = Field(
        default_factory=dict, description="Token allocation per conversation"
    )

    class Config:
        """Pydantic configuration for ContextPlacement model."""

        pass

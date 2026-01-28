"""
Memory module for Mai conversation management.

This module provides persistent storage and retrieval of conversations,
messages, and associated vector embeddings for semantic search capabilities.
"""

from .storage.sqlite_manager import SQLiteManager
from .storage.vector_store import VectorStore
from .retrieval.semantic_search import SemanticSearch
from .retrieval.context_aware import ContextAwareSearch
from .retrieval.timeline_search import TimelineSearch

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import logging


class MemoryManager:
    """
    Enhanced memory manager with unified search interface.

    Provides comprehensive memory operations including semantic search,
    context-aware search, timeline filtering, and hybrid search strategies.
    """

    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize memory manager with SQLite database and search capabilities.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._sqlite_manager: Optional[SQLiteManager] = None
        self._vector_store: Optional[VectorStore] = None
        self._semantic_search: Optional[SemanticSearch] = None
        self._context_aware_search: Optional[ContextAwareSearch] = None
        self._timeline_search: Optional[TimelineSearch] = None
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """
        Initialize storage and search components.

        Creates database schema, vector tables, and search instances.
        """
        try:
            # Initialize storage components
            self._sqlite_manager = SQLiteManager(self.db_path)
            self._vector_store = VectorStore(self._sqlite_manager)

            # Initialize search components
            self._semantic_search = SemanticSearch(self._vector_store)
            self._context_aware_search = ContextAwareSearch(self._sqlite_manager)
            self._timeline_search = TimelineSearch(self._sqlite_manager)

            self.logger.info(
                f"Enhanced memory manager initialized with database: {self.db_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced memory manager: {e}")
            raise

    @property
    def sqlite_manager(self) -> SQLiteManager:
        """Get SQLite manager instance."""
        if self._sqlite_manager is None:
            raise RuntimeError(
                "Memory manager not initialized. Call initialize() first."
            )
        return self._sqlite_manager

    @property
    def vector_store(self) -> VectorStore:
        """Get vector store instance."""
        if self._vector_store is None:
            raise RuntimeError(
                "Memory manager not initialized. Call initialize() first."
            )
        return self._vector_store

    @property
    def semantic_search(self) -> SemanticSearch:
        """Get semantic search instance."""
        if self._semantic_search is None:
            raise RuntimeError(
                "Memory manager not initialized. Call initialize() first."
            )
        return self._semantic_search

    @property
    def context_aware_search(self) -> ContextAwareSearch:
        """Get context-aware search instance."""
        if self._context_aware_search is None:
            raise RuntimeError(
                "Memory manager not initialized. Call initialize() first."
            )
        return self._context_aware_search

    @property
    def timeline_search(self) -> TimelineSearch:
        """Get timeline search instance."""
        if self._timeline_search is None:
            raise RuntimeError(
                "Memory manager not initialized. Call initialize() first."
            )
        return self._timeline_search

    # Legacy methods for compatibility
    def close(self) -> None:
        """Close database connections."""
        if self._sqlite_manager:
            self._sqlite_manager.close()
        self.logger.info("Enhanced memory manager closed")

    # Unified search interface
    def search(
        self,
        query: str,
        search_type: str = "semantic",
        limit: int = 5,
        conversation_id: Optional[str] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
        current_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Unified search interface supporting multiple search strategies.

        Args:
            query: Search query text
            search_type: Type of search ("semantic", "keyword", "context_aware", "timeline", "hybrid")
            limit: Maximum number of results to return
            conversation_id: Current conversation ID for context-aware search
            date_start: Start date for timeline search
            date_end: End date for timeline search
            current_topic: Current topic for context-aware prioritization

        Returns:
            List of search results as dictionaries
        """
        if not self._is_initialized():
            raise RuntimeError("Memory manager not initialized")

        try:
            results = []

            if search_type == "semantic":
                results = self._semantic_search.search(query, limit)
            elif search_type == "keyword":
                results = self._semantic_search.keyword_search(query, limit)
            elif search_type == "context_aware":
                # Get base semantic results, then prioritize by topic
                base_results = self._semantic_search.search(query, limit * 2)
                results = self._context_aware_search.prioritize_by_topic(
                    base_results, current_topic, conversation_id
                )
            elif search_type == "timeline":
                if date_start and date_end:
                    results = self._timeline_search.search_by_date_range(
                        date_start, date_end, limit
                    )
                else:
                    # Default to recent search
                    results = self._timeline_search.search_recent(limit=limit)
            elif search_type == "hybrid":
                results = self._semantic_search.hybrid_search(query, limit)
            else:
                self.logger.warning(
                    f"Unknown search type: {search_type}, falling back to semantic"
                )
                results = self._semantic_search.search(query, limit)

            # Convert search results to dictionaries for external interface
            return [
                {
                    "conversation_id": result.conversation_id,
                    "message_id": result.message_id,
                    "content": result.content,
                    "relevance_score": result.relevance_score,
                    "snippet": result.snippet,
                    "timestamp": result.timestamp.isoformat()
                    if result.timestamp
                    else None,
                    "metadata": result.metadata,
                    "search_type": result.search_type,
                }
                for result in results
            ]

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def search_by_embedding(
        self, embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using pre-computed embedding vector.

        Args:
            embedding: Embedding vector as list of floats
            limit: Maximum number of results to return

        Returns:
            List of search results as dictionaries
        """
        if not self._is_initialized():
            raise RuntimeError("Memory manager not initialized")

        try:
            import numpy as np

            embedding_array = np.array(embedding)
            results = self._semantic_search.search_by_embedding(embedding_array, limit)

            # Convert to dictionaries
            return [
                {
                    "conversation_id": result.conversation_id,
                    "message_id": result.message_id,
                    "content": result.content,
                    "relevance_score": result.relevance_score,
                    "snippet": result.snippet,
                    "timestamp": result.timestamp.isoformat()
                    if result.timestamp
                    else None,
                    "metadata": result.metadata,
                    "search_type": result.search_type,
                }
                for result in results
            ]

        except Exception as e:
            self.logger.error(f"Embedding search failed: {e}")
            return []

    def get_topic_summary(
        self, conversation_id: str, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get topic analysis summary for a conversation.

        Args:
            conversation_id: ID of conversation to analyze
            limit: Number of messages to analyze

        Returns:
            Dictionary with topic analysis and statistics
        """
        if not self._is_initialized():
            raise RuntimeError("Memory manager not initialized")

        return self._context_aware_search.get_topic_summary(conversation_id, limit)

    def get_temporal_summary(
        self, conversation_id: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get temporal analysis summary of conversations.

        Args:
            conversation_id: Specific conversation to analyze (None for all)
            days: Number of recent days to analyze

        Returns:
            Dictionary with temporal statistics and patterns
        """
        if not self._is_initialized():
            raise RuntimeError("Memory manager not initialized")

        return self._timeline_search.get_temporal_summary(conversation_id, days)

    def suggest_related_topics(self, query: str, limit: int = 3) -> List[str]:
        """
        Suggest related topics based on query analysis.

        Args:
            query: Search query to analyze
            limit: Maximum number of suggestions

        Returns:
            List of suggested topic strings
        """
        if not self._is_initialized():
            raise RuntimeError("Memory manager not initialized")

        return self._context_aware_search.suggest_related_topics(query, limit)

    def index_conversation(
        self, conversation_id: str, messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Index conversation messages for semantic search.

        Args:
            conversation_id: ID of the conversation
            messages: List of message dictionaries

        Returns:
            True if indexing successful, False otherwise
        """
        if not self._is_initialized():
            raise RuntimeError("Memory manager not initialized")

        return self._semantic_search.index_conversation(conversation_id, messages)

    def _is_initialized(self) -> bool:
        """Check if all components are initialized."""
        return (
            self._sqlite_manager is not None
            and self._vector_store is not None
            and self._semantic_search is not None
            and self._context_aware_search is not None
            and self._timeline_search is not None
        )


# Export main classes for external import
__all__ = [
    "MemoryManager",
    "SQLiteManager",
    "VectorStore",
    "SemanticSearch",
    "ContextAwareSearch",
    "TimelineSearch",
]

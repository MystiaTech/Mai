"""
Semantic search implementation using sentence-transformers embeddings.

This module provides semantic search capabilities through embedding generation
and vector similarity search using the vector store.
"""

import sys
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import hashlib

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    np = None

from .search_types import SearchResult, SearchQuery
from ..storage.vector_store import VectorStore


class SemanticSearch:
    """
    Semantic search with embedding-based similarity.

    Provides semantic search capabilities through sentence-transformer embeddings
    combined with vector similarity search for efficient retrieval.
    """

    def __init__(self, vector_store: VectorStore, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search with vector store and embedding model.

        Args:
            vector_store: VectorStore instance for similarity search
            model_name: Name of sentence-transformer model to use
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self._model = None  # Lazy loading
        self.logger = logging.getLogger(__name__)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def model(self) -> Optional["SentenceTransformer"]:
        """
        Get embedding model (lazy loaded for performance).

        Returns:
            SentenceTransformer model instance
        """
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    def _generate_embedding(self, text: str) -> Optional["np.ndarray"]:
        """
        Generate embedding for text using sentence-transformers.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if model not available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            return None

        try:
            # Clean and normalize text
            text = text.strip()
            if not text:
                return None

            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None

    def _create_search_result(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        similarity: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> SearchResult:
        """
        Create search result with relevance scoring.

        Args:
            conversation_id: ID of the conversation
            message_id: ID of the message
            content: Message content
            similarity: Similarity score (0.0 to 1.0)
            timestamp: Message timestamp
            metadata: Additional metadata

        Returns:
            SearchResult with semantic search type
        """
        # Convert similarity to relevance score (higher = more relevant)
        relevance_score = float(similarity)

        # Generate snippet (first 200 characters)
        snippet = content[:200] + "..." if len(content) > 200 else content

        return SearchResult(
            conversation_id=conversation_id,
            message_id=message_id,
            content=content,
            relevance_score=relevance_score,
            snippet=snippet,
            timestamp=timestamp,
            metadata=metadata,
            search_type="semantic",
        )

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Perform semantic search for query.

        Args:
            query: Search query text
            limit: Maximum number of results to return

        Returns:
            List of search results ranked by relevance
        """
        if not query or not query.strip():
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            self.logger.warning(
                "Failed to generate query embedding, falling back to keyword search"
            )
            return self.keyword_search(query, limit)

        # Search vector store for similar embeddings
        try:
            vector_results = self.vector_store.search_similar(
                query_embedding, limit * 2
            )

            # Convert to search results
            results = []
            for result in vector_results:
                search_result = self._create_search_result(
                    conversation_id=result.get("conversation_id", ""),
                    message_id=result.get("message_id", ""),
                    content=result.get("content", ""),
                    similarity=result.get("similarity", 0.0),
                    timestamp=result.get("timestamp", datetime.utcnow()),
                    metadata=result.get("metadata", {}),
                )
                results.append(search_result)

            # Sort by relevance score and limit results
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []

    def search_by_embedding(
        self, embedding: "np.ndarray", limit: int = 5
    ) -> List[SearchResult]:
        """
        Search using pre-computed embedding.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of search results ranked by similarity
        """
        if embedding is None:
            return []

        try:
            vector_results = self.vector_store.search_similar(embedding, limit * 2)

            # Convert to search results
            results = []
            for result in vector_results:
                search_result = self._create_search_result(
                    conversation_id=result.get("conversation_id", ""),
                    message_id=result.get("message_id", ""),
                    content=result.get("content", ""),
                    similarity=result.get("similarity", 0.0),
                    timestamp=result.get("timestamp", datetime.utcnow()),
                    metadata=result.get("metadata", {}),
                )
                results.append(search_result)

            # Sort by relevance score and limit results
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]

        except Exception as e:
            self.logger.error(f"Embedding search failed: {e}")
            return []

    def keyword_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Fallback keyword-based search.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of search results with keyword search type
        """
        if not query or not query.strip():
            return []

        try:
            # Simple keyword search through vector store metadata
            # This is a basic implementation - could be enhanced with FTS
            results = self.vector_store.search_by_keyword(query, limit)

            # Convert to search results
            search_results = []
            for result in results:
                search_result = SearchResult(
                    conversation_id=result.get("conversation_id", ""),
                    message_id=result.get("message_id", ""),
                    content=result.get("content", ""),
                    relevance_score=result.get("relevance", 0.5),
                    snippet=result.get("snippet", ""),
                    timestamp=result.get("timestamp", datetime.utcnow()),
                    metadata=result.get("metadata", {}),
                    search_type="keyword",
                )
                search_results.append(search_result)

            # Sort by relevance and limit
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return search_results[:limit]

        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []

    def hybrid_search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword matching.

        Args:
            query: Search query text
            limit: Maximum number of results to return

        Returns:
            List of search results with hybrid scoring
        """
        if not query or not query.strip():
            return []

        # Get semantic results
        semantic_results = self.search(query, limit)

        # Get keyword results
        keyword_results = self.keyword_search(query, limit)

        # Combine and deduplicate results
        combined_results = {}

        # Add semantic results with higher weight
        for result in semantic_results:
            key = f"{result.conversation_id}_{result.message_id}"
            # Boost semantic results
            boosted_score = min(1.0, result.relevance_score * 1.2)
            result.relevance_score = boosted_score
            combined_results[key] = result

        # Add keyword results (only if not already present)
        for result in keyword_results:
            key = f"{result.conversation_id}_{result.message_id}"
            if key not in combined_results:
                # Lower weight for keyword results
                result.relevance_score = result.relevance_score * 0.8
                combined_results[key] = result
            else:
                # Merge scores if present in both
                existing = combined_results[key]
                existing.relevance_score = max(
                    existing.relevance_score, result.relevance_score * 0.8
                )

        # Convert to list and sort
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return final_results[:limit]

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
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            self.logger.warning("Cannot index: sentence-transformers not available")
            return False

        try:
            embeddings = []
            for message in messages:
                content = message.get("content", "")
                if content.strip():
                    embedding = self._generate_embedding(content)
                    if embedding is not None:
                        embeddings.append(
                            {
                                "conversation_id": conversation_id,
                                "message_id": message.get("id", ""),
                                "content": content,
                                "embedding": embedding,
                                "timestamp": message.get(
                                    "timestamp", datetime.utcnow()
                                ),
                                "metadata": message.get("metadata", {}),
                            }
                        )

            # Store embeddings in vector store
            if embeddings:
                self.vector_store.store_embeddings(embeddings)
                self.logger.info(
                    f"Indexed {len(embeddings)} messages for conversation {conversation_id}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to index conversation: {e}")
            return False

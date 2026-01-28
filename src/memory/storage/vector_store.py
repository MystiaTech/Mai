"""
Vector store implementation using sqlite-vec extension.

This module provides vector storage and retrieval capabilities for semantic search
using sqlite-vec virtual tables within SQLite database.
"""

import sqlite3
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging

try:
    import sqlite_vec  # sqlite-vec extension
except ImportError:
    sqlite_vec = None


class VectorStore:
    """
    Vector storage and retrieval using sqlite-vec extension.

    Provides semantic search capabilities through SQLite virtual tables
    for efficient embedding similarity search and storage.
    """

    def __init__(self, sqlite_manager):
        """
        Initialize vector store with SQLite manager.

        Args:
            sqlite_manager: SQLiteManager instance for database access
        """
        self.sqlite_manager = sqlite_manager
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        self.logger = logging.getLogger(__name__)
        self._initialize_vector_tables()

    def _initialize_vector_tables(self) -> None:
        """
        Initialize vector virtual tables for embedding storage.

        Creates vec0 virtual tables using sqlite-vec extension
        for efficient vector similarity search.
        """
        if sqlite_vec is None:
            raise ImportError(
                "sqlite-vec extension not installed. "
                "Install with: pip install sqlite-vec"
            )

        conn = self.sqlite_manager._get_connection()
        try:
            # Enable extension loading
            conn.enable_load_extension(True)

            # Load sqlite-vec extension
            try:
                conn.load_extension("vec0")
                self.logger.info("Loaded sqlite-vec extension")
            except sqlite3.OperationalError as e:
                self.logger.error(f"Failed to load sqlite-vec extension: {e}")
                raise ImportError(
                    "sqlite-vec extension not available. "
                    "Ensure sqlite-vec is installed and extension is accessible."
                )

            # Create virtual table for message embeddings
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_message_embeddings 
                USING vec0(
                    embedding float[{dimension}],
                    message_id TEXT,
                    content TEXT,
                    conversation_id TEXT,
                    timestamp TIMESTAMP,
                    model_version TEXT DEFAULT 'all-MiniLM-L6-v2'
                )
            """.format(dimension=self.embedding_dimension)
            )

            # Create virtual table for conversation embeddings
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_conversation_embeddings 
                USING vec0(
                    embedding float[{dimension}],
                    conversation_id TEXT,
                    title TEXT,
                    content_summary TEXT,
                    created_at TIMESTAMP,
                    model_version TEXT DEFAULT 'all-MiniLM-L6-v2'
                )
            """.format(dimension=self.embedding_dimension)
            )

            # Create indexes for efficient querying
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vec_message_id ON vec_message_embeddings(message_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vec_conversation_id ON vec_conversation_embeddings(conversation_id)"
            )

            conn.commit()
            self.logger.info("Vector tables initialized successfully")

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to initialize vector tables: {e}")
            raise
        finally:
            # Don't close connection here, sqlite_manager manages it
            pass

    def store_message_embedding(
        self,
        message_id: str,
        conversation_id: str,
        content: str,
        embedding: np.ndarray,
        model_version: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Store embedding for a message.

        Args:
            message_id: Unique message identifier
            conversation_id: Conversation ID
            content: Message content text
            embedding: Numpy array of embedding values
            model_version: Embedding model version
        """
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be numpy array")

        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        conn = self.sqlite_manager._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO vec_message_embeddings 
                (message_id, conversation_id, content, embedding, model_version)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    message_id,
                    conversation_id,
                    content,
                    embedding.tobytes(),
                    model_version,
                ),
            )
            conn.commit()
            self.logger.debug(f"Stored embedding for message {message_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to store message embedding: {e}")
            raise

    def store_conversation_embedding(
        self,
        conversation_id: str,
        title: str,
        content_summary: str,
        embedding: np.ndarray,
        model_version: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Store embedding for a conversation summary.

        Args:
            conversation_id: Conversation ID
            title: Conversation title
            content_summary: Summary of conversation content
            embedding: Numpy array of embedding values
            model_version: Embedding model version
        """
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be numpy array")

        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        conn = self.sqlite_manager._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO vec_conversation_embeddings 
                (conversation_id, title, content_summary, embedding, model_version)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    conversation_id,
                    title,
                    content_summary,
                    embedding.tobytes(),
                    model_version,
                ),
            )
            conn.commit()
            self.logger.debug(f"Stored embedding for conversation {conversation_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to store conversation embedding: {e}")
            raise

    def search_similar_messages(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        conversation_id: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar messages using vector similarity.

        Args:
            query_embedding: Query embedding numpy array
            limit: Maximum number of results
            conversation_id: Optional conversation filter
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of similar message results
        """
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("Query embedding must be numpy array")

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        conn = self.sqlite_manager._get_connection()
        try:
            query = """
                SELECT 
                    message_id,
                    conversation_id,
                    content,
                    distance,
                    (1.0 - distance) as similarity
                FROM vec_message_embeddings
                WHERE embedding MATCH ?
                {conversation_filter}
                ORDER BY distance
                LIMIT ?
            """

            params = [query_embedding.tobytes()]

            if conversation_id:
                query = query.format(conversation_filter="AND conversation_id = ?")
                params.append(conversation_id)
            else:
                query = query.format(conversation_filter="")

            params.append(limit)

            cursor = conn.execute(query, params)
            results = []
            for row in cursor:
                similarity = float(row["similarity"])
                if similarity >= min_similarity:
                    results.append(
                        {
                            "message_id": row["message_id"],
                            "conversation_id": row["conversation_id"],
                            "content": row["content"],
                            "similarity": similarity,
                            "distance": float(row["distance"]),
                        }
                    )

            return results
        except Exception as e:
            self.logger.error(f"Failed to search similar messages: {e}")
            raise

    def search_similar_conversations(
        self, query_embedding: np.ndarray, limit: int = 10, min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar conversations using vector similarity.

        Args:
            query_embedding: Query embedding numpy array
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of similar conversation results
        """
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("Query embedding must be numpy array")

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        conn = self.sqlite_manager._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT 
                    conversation_id,
                    title,
                    content_summary,
                    distance,
                    (1.0 - distance) as similarity
                FROM vec_conversation_embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """,
                (query_embedding.tobytes(), limit),
            )

            results = []
            for row in cursor:
                similarity = float(row["similarity"])
                if similarity >= min_similarity:
                    results.append(
                        {
                            "conversation_id": row["conversation_id"],
                            "title": row["title"],
                            "content_summary": row["content_summary"],
                            "similarity": similarity,
                            "distance": float(row["distance"]),
                        }
                    )

            return results
        except Exception as e:
            self.logger.error(f"Failed to search similar conversations: {e}")
            raise

    def get_message_embedding(self, message_id: str) -> Optional[np.ndarray]:
        """
        Get stored embedding for a specific message.

        Args:
            message_id: Message identifier

        Returns:
            Embedding numpy array or None if not found
        """
        conn = self.sqlite_manager._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT embedding FROM vec_message_embeddings 
                WHERE message_id = ?
            """,
                (message_id,),
            )

            row = cursor.fetchone()
            if row:
                embedding_bytes = row["embedding"]
                return np.frombuffer(embedding_bytes, dtype=np.float32)

            return None
        except Exception as e:
            self.logger.error(f"Failed to get message embedding {message_id}: {e}")
            raise

    def delete_message_embeddings(self, message_id: str) -> None:
        """
        Delete embedding for a specific message.

        Args:
            message_id: Message identifier
        """
        conn = self.sqlite_manager._get_connection()
        try:
            conn.execute(
                """
                DELETE FROM vec_message_embeddings 
                WHERE message_id = ?
            """,
                (message_id,),
            )
            conn.commit()
            self.logger.debug(f"Deleted embedding for message {message_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to delete message embedding: {e}")
            raise

    def delete_conversation_embeddings(self, conversation_id: str) -> None:
        """
        Delete all embeddings for a conversation.

        Args:
            conversation_id: Conversation identifier
        """
        conn = self.sqlite_manager._get_connection()
        try:
            # Delete message embeddings
            conn.execute(
                """
                DELETE FROM vec_message_embeddings 
                WHERE conversation_id = ?
            """,
                (conversation_id,),
            )

            # Delete conversation embedding
            conn.execute(
                """
                DELETE FROM vec_conversation_embeddings 
                WHERE conversation_id = ?
            """,
                (conversation_id,),
            )

            conn.commit()
            self.logger.debug(f"Deleted embeddings for conversation {conversation_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to delete conversation embeddings: {e}")
            raise

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        conn = self.sqlite_manager._get_connection()
        try:
            stats = {}

            # Message embedding stats
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM vec_message_embeddings"
            )
            stats["total_message_embeddings"] = cursor.fetchone()["count"]

            # Conversation embedding stats
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM vec_conversation_embeddings"
            )
            stats["total_conversation_embeddings"] = cursor.fetchone()["count"]

            # Model version distribution
            cursor = conn.execute("""
                SELECT model_version, COUNT(*) as count 
                FROM vec_message_embeddings 
                GROUP BY model_version
            """)
            stats["model_versions"] = {
                row["model_version"]: row["count"] for row in cursor
            }

            return stats
        except Exception as e:
            self.logger.error(f"Failed to get embedding stats: {e}")
            raise

    def set_embedding_dimension(self, dimension: int) -> None:
        """
        Set embedding dimension for new embeddings.

        Args:
            dimension: New embedding dimension
        """
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        self.embedding_dimension = dimension
        self.logger.info(f"Embedding dimension set to {dimension}")

    def validate_embedding_dimension(self, embedding: np.ndarray) -> bool:
        """
        Validate embedding dimension matches expected size.

        Args:
            embedding: Embedding to validate

        Returns:
            True if dimension matches, False otherwise
        """
        return len(embedding) == self.embedding_dimension

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
                if sqlite_vec is None:
                    raise ImportError("sqlite-vec not imported")
                extension_path = sqlite_vec.loadable_path()
                conn.load_extension(extension_path)
                self.logger.info(f"Loaded sqlite-vec extension from {extension_path}")
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
                        embedding float[{dimension}]
                    )
                """.format(dimension=self.embedding_dimension)
            )

            # Create metadata table for message embeddings
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS vec_message_metadata (
                        rowid INTEGER PRIMARY KEY,
                        message_id TEXT UNIQUE,
                        conversation_id TEXT,
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_version TEXT DEFAULT 'all-MiniLM-L6-v2'
                    )
                """
            )

            # Create virtual table for conversation embeddings
            conn.execute(
                """
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_conversation_embeddings 
                    USING vec0(
                        embedding float[{dimension}]
                    )
                """.format(dimension=self.embedding_dimension)
            )

            # Create metadata table for conversation embeddings
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS vec_conversation_metadata (
                        rowid INTEGER PRIMARY KEY,
                        conversation_id TEXT UNIQUE,
                        title TEXT,
                        content_summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_version TEXT DEFAULT 'all-MiniLM-L6-v2'
                    )
                """
            )

            # Create indexes for efficient querying
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_message_id ON vec_message_metadata(message_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_conversation_id ON vec_message_metadata(conversation_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_metadata_conversation_id ON vec_conversation_metadata(conversation_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metadata_timestamp ON vec_message_metadata(timestamp)"
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
            # Insert metadata first
            cursor = conn.execute(
                """
                    INSERT OR REPLACE INTO vec_message_metadata 
                    (message_id, conversation_id, content, model_version)
                    VALUES (?, ?, ?, ?)
            """,
                (
                    message_id,
                    conversation_id,
                    content,
                    model_version,
                ),
            )
            metadata_rowid = cursor.lastrowid

            # Insert embedding
            conn.execute(
                """
                    INSERT INTO vec_message_embeddings 
                    (rowid, embedding)
                    VALUES (?, ?)
            """,
                (metadata_rowid, embedding.tobytes()),
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
            # Insert metadata first
            cursor = conn.execute(
                """
                    INSERT OR REPLACE INTO vec_conversation_metadata 
                    (conversation_id, title, content_summary, model_version)
                    VALUES (?, ?, ?, ?)
            """,
                (
                    conversation_id,
                    title,
                    content_summary,
                    model_version,
                ),
            )
            metadata_rowid = cursor.lastrowid

            # Insert embedding
            conn.execute(
                """
                    INSERT INTO vec_conversation_embeddings 
                    (rowid, embedding)
                    VALUES (?, ?)
            """,
                (metadata_rowid, embedding.tobytes()),
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
                    vm.message_id,
                    vm.conversation_id,
                    vm.content,
                    vm.timestamp,
                    vme.distance,
                    (1.0 - vme.distance) as similarity
                FROM vec_message_embeddings vme
                JOIN vec_message_metadata vm ON vme.rowid = vm.rowid
                WHERE vme.embedding MATCH ?
                {conversation_filter}
                ORDER BY vme.distance
                LIMIT ?
            """

            params = [query_embedding.tobytes()]

            if conversation_id:
                query = query.format(conversation_filter="AND vm.conversation_id = ?")
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
                            "timestamp": row["timestamp"],
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
                        vcm.conversation_id,
                        vcm.title,
                        vcm.content_summary,
                        vcm.created_at,
                        vce.distance,
                        (1.0 - vce.distance) as similarity
                    FROM vec_conversation_embeddings vce
                    JOIN vec_conversation_metadata vcm ON vce.rowid = vcm.rowid
                    WHERE vce.embedding MATCH ?
                    ORDER BY vce.distance
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
                            "created_at": row["created_at"],
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
                    SELECT vme.embedding FROM vec_message_embeddings vme
                    JOIN vec_message_metadata vm ON vme.rowid = vm.rowid
                    WHERE vm.message_id = ?
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
            # Delete from both tables
            conn.execute(
                """
                    DELETE FROM vec_message_embeddings 
                    WHERE rowid IN (
                        SELECT rowid FROM vec_message_metadata WHERE message_id = ?
                    )
            """,
                (message_id,),
            )
            conn.execute(
                """
                    DELETE FROM vec_message_metadata 
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
                    WHERE rowid IN (
                        SELECT rowid FROM vec_message_metadata WHERE conversation_id = ?
                    )
            """,
                (conversation_id,),
            )
            conn.execute(
                """
                    DELETE FROM vec_message_metadata 
                    WHERE conversation_id = ?
            """,
                (conversation_id,),
            )

            # Delete conversation embedding
            conn.execute(
                """
                    DELETE FROM vec_conversation_embeddings 
                    WHERE rowid IN (
                        SELECT rowid FROM vec_conversation_metadata WHERE conversation_id = ?
                    )
            """,
                (conversation_id,),
            )
            conn.execute(
                """
                    DELETE FROM vec_conversation_metadata 
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
                FROM vec_message_metadata 
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

    def search_by_keyword(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for messages by keyword using FTS or LIKE queries.

        Args:
            query: Keyword search query
            limit: Maximum number of results

        Returns:
            List of message results with metadata
        """
        if not query or not query.strip():
            return []

        conn = self.sqlite_manager._get_connection()
        try:
            # Clean and prepare query
            keywords = query.strip().split()
            if not keywords:
                return []

            # Try FTS first if available
            fts_available = self._check_fts_available(conn)

            if fts_available:
                results = self._search_with_fts(conn, keywords, limit)
            else:
                results = self._search_with_like(conn, keywords, limit)

            return results

        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            return []

    def _check_fts_available(self, conn: sqlite3.Connection) -> bool:
        """
        Check if FTS virtual tables are available.

        Args:
            conn: SQLite connection

        Returns:
            True if FTS is available
        """
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'"
            )
            return cursor.fetchone() is not None
        except:
            return False

    def _search_with_fts(
        self, conn: sqlite3.Connection, keywords: List[str], limit: int
    ) -> List[Dict]:
        """
        Search using SQLite FTS (Full-Text Search).

        Args:
            conn: SQLite connection
            keywords: List of keywords to search
            limit: Maximum results

        Returns:
            List of search results
        """
        results = []

        # Build FTS query
        fts_query = " AND ".join([f'"{keyword}"' for keyword in keywords])

        try:
            # Search message metadata table content
            cursor = conn.execute(
                f"""
                SELECT 
                    message_id,
                    conversation_id,
                    content,
                    timestamp,
                    rank,
                    (rank * 1.0) as relevance
                FROM vec_message_metadata_fts
                WHERE vec_message_metadata_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """,
                (fts_query, limit),
            )

            for row in cursor:
                results.append(
                    {
                        "message_id": row["message_id"],
                        "conversation_id": row["conversation_id"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "relevance": float(row["relevance"]),
                        "score": float(row["relevance"]),  # For compatibility
                    }
                )

        except sqlite3.OperationalError:
            # FTS table doesn't exist, fall back to LIKE
            return self._search_with_like(conn, keywords, limit)

        return results

    def _search_with_like(
        self, conn: sqlite3.Connection, keywords: List[str], limit: int
    ) -> List[Dict]:
        """
        Search using LIKE queries when FTS is not available.

        Args:
            conn: SQLite connection
            keywords: List of keywords to search
            limit: Maximum results

        Returns:
            List of search results
        """
        results = []

        # Build WHERE clause for multiple keywords
        where_clauses = []
        params = []

        for keyword in keywords:
            where_clauses.append("content LIKE ?")
            params.extend([f"%{keyword}%"])

        where_clause = " AND ".join(where_clauses)
        params.append(limit)

        try:
            # Search message metadata table content
            base_params = [keywords[0].lower()] + params[
                :-1
            ]  # Exclude limit from base params
            cursor = conn.execute(
                f"""
                SELECT DISTINCT
                    vm.message_id,
                    vm.conversation_id,
                    vm.content,
                    vm.timestamp,
                    (LENGTH(vm.content) - LENGTH(REPLACE(LOWER(vm.content), ?, '')) * 10.0) as relevance
                FROM vec_message_metadata vm
                LEFT JOIN conversations c ON vm.conversation_id = c.id
                WHERE {where_clause}
                ORDER BY relevance DESC
                LIMIT ?
            """,
                base_params + [params[-1]],  # Add limit back
            )

            for row in cursor:
                results.append(
                    {
                        "message_id": row["message_id"],
                        "conversation_id": row["conversation_id"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "relevance": float(row["relevance"]),
                        "score": float(row["relevance"]),  # For compatibility
                    }
                )

        except Exception as e:
            self.logger.warning(f"LIKE search failed: {e}")
            # Final fallback - basic search
            try:
                cursor = conn.execute(
                    """
                    SELECT 
                        message_id,
                        conversation_id,
                        content,
                        timestamp,
                        0.5 as relevance
                    FROM vec_message_metadata
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (f"%{keywords[0]}%", limit),
                )

                for row in cursor:
                    results.append(
                        {
                            "message_id": row["message_id"],
                            "conversation_id": row["conversation_id"],
                            "content": row["content"],
                            "timestamp": row["timestamp"],
                            "relevance": float(row["relevance"]),
                            "score": float(row["relevance"]),
                        }
                    )

            except Exception as e2:
                self.logger.error(f"Fallback search failed: {e2}")

        return results

    def store_embeddings(self, embeddings: List[Dict]) -> bool:
        """
        Store multiple embeddings efficiently in batch.

        Args:
            embeddings: List of embedding dictionaries with message_id, embedding, etc.

        Returns:
            True if successful, False otherwise
        """
        if not embeddings:
            return True

        conn = self.sqlite_manager._get_connection()
        try:
            # Begin transaction
            conn.execute("BEGIN IMMEDIATE")

            stored_count = 0
            for embedding_data in embeddings:
                try:
                    # Extract required fields
                    message_id = embedding_data.get("message_id")
                    conversation_id = embedding_data.get("conversation_id")
                    content = embedding_data.get("content", "")
                    embedding = embedding_data.get("embedding")

                    if not message_id or not conversation_id or embedding is None:
                        self.logger.warning(
                            f"Skipping invalid embedding data: {embedding_data}"
                        )
                        continue

                    # Convert embedding to numpy array if needed
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)
                    else:
                        embedding = embedding.astype(np.float32)

                    # Validate dimension
                    if not self.validate_embedding_dimension(embedding):
                        self.logger.warning(
                            f"Invalid embedding dimension for {message_id}: {len(embedding)}"
                        )
                        continue

                    # Insert metadata first
                    cursor = conn.execute(
                        """
                        INSERT OR REPLACE INTO vec_message_metadata 
                        (message_id, conversation_id, content, model_version)
                        VALUES (?, ?, ?, ?)
                        """,
                        (message_id, conversation_id, content, "all-MiniLM-L6-v2"),
                    )
                    metadata_rowid = cursor.lastrowid

                    # Store the embedding
                    conn.execute(
                        """
                        INSERT INTO vec_message_embeddings 
                        (rowid, embedding)
                        VALUES (?, ?)
                        """,
                        (metadata_rowid, embedding.tobytes()),
                    )

                    stored_count += 1

                except Exception as e:
                    self.logger.error(
                        f"Failed to store embedding {embedding_data.get('message_id', 'unknown')}: {e}"
                    )
                    continue

            # Commit transaction
            conn.commit()
            self.logger.info(
                f"Successfully stored {stored_count}/{len(embeddings)} embeddings"
            )

            return stored_count > 0

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Batch embedding storage failed: {e}")
            return False

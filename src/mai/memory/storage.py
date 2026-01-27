"""
Memory Storage Implementation for Mai

Provides SQLite-based persistent storage with vector similarity search
for conversation retention and semantic retrieval.
"""

import os
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import dependencies
try:
    import sqlite_vec  # type: ignore
except ImportError:
    # Fallback if sqlite-vec not installed
    sqlite_vec = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Fallback if sentence-transformers not installed
    SentenceTransformer = None

# Import Mai components
try:
    from src.mai.core.exceptions import (
        MaiError,
        ContextError,
        create_error_context,
    )
    from src.mai.core.config import get_config
except ImportError:
    # Define fallbacks if modules not available
    class MaiError(Exception):
        pass

    class ContextError(MaiError):
        pass

    def create_error_context(component: str, operation: str, **data):
        return {"component": component, "operation": operation, "data": data}

    def get_config():
        return None


logger = logging.getLogger(__name__)


class MemoryStorageError(ContextError):
    """Memory storage specific errors."""

    def __init__(self, message: str, operation: str = None, **kwargs):
        context = create_error_context(
            component="memory_storage", operation=operation or "storage_operation", **kwargs
        )
        super().__init__(message, context=context)
        self.operation = operation


class VectorSearchError(MemoryStorageError):
    """Vector similarity search errors."""

    def __init__(self, query: str, error_details: str = None):
        message = f"Vector search failed for query: '{query}'"
        if error_details:
            message += f": {error_details}"

        super().__init__(
            message=message, operation="vector_search", query=query, error_details=error_details
        )


class DatabaseConnectionError(MemoryStorageError):
    """Database connection and operation errors."""

    def __init__(self, db_path: str, error_details: str = None):
        message = f"Database connection error: {db_path}"
        if error_details:
            message += f": {error_details}"

        super().__init__(
            message=message,
            operation="database_connection",
            db_path=db_path,
            error_details=error_details,
        )


class MemoryStorage:
    """
    SQLite-based memory storage with vector similarity search.

    Handles persistent storage of conversations, messages, and embeddings
    with semantic search capabilities using sqlite-vec extension.
    """

    def __init__(self, db_path: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize memory storage with database and embedding model.

        Args:
            db_path: Path to SQLite database file (default: ./data/mai_memory.db)
            embedding_model: Name of sentence-transformers model to use
        """
        # Set database path
        if db_path is None:
            # Default to ./data/mai_memory.db
            db_path = os.path.join(os.getcwd(), "data", "mai_memory.db")

        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._db: Optional[sqlite3.Connection] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
        self._config = get_config()

        # Initialize embedding model first (needed for database schema)
        self._initialize_embedding_model()
        # Then initialize database
        self._initialize_database()

        logger.info(f"MemoryStorage initialized with database: {self.db_path}")

    def _initialize_database(self) -> None:
        """Initialize SQLite database with schema and vector extension."""
        try:
            # Connect to database
            self._db = sqlite3.connect(str(self.db_path))
            self._db.row_factory = sqlite3.Row  # Enable dict-like row access

            # Enable foreign keys
            self._db.execute("PRAGMA foreign_keys = ON")

            # Load sqlite-vec extension if available
            if sqlite_vec is not None:
                try:
                    self._db.enable_load_extension(True)
                    # Try to load the full path to vec0.so
                    vec_path = sqlite_vec.__file__.replace("__init__.py", "vec0.so")
                    self._db.load_extension(vec_path)
                    logger.info("sqlite-vec extension loaded successfully")
                    self._vector_enabled = True
                except Exception as e:
                    logger.warning(f"Failed to load sqlite-vec extension: {e}")
                    # Try fallback with just extension name
                    try:
                        self._db.load_extension("vec0")
                        logger.info("sqlite-vec extension loaded successfully (fallback)")
                        self._vector_enabled = True
                    except Exception as e2:
                        logger.warning(f"Failed to load sqlite-vec extension (fallback): {e2}")
                        self._vector_enabled = False
            else:
                logger.warning("sqlite-vec not available - vector features disabled")
                self._vector_enabled = False

            # Create tables
            self._create_tables()

            # Verify schema
            self._verify_schema()

        except Exception as e:
            raise DatabaseConnectionError(db_path=str(self.db_path), error_details=str(e))

    def _initialize_embedding_model(self) -> None:
        """Initialize sentence-transformers embedding model."""
        try:
            if SentenceTransformer is not None:
                # Load embedding model (download if needed)
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)

                # Test embedding generation
                test_embedding = self._embedding_model.encode("test")
                self._embedding_dim = len(test_embedding)
                logger.info(
                    f"Embedding model loaded: {self.embedding_model_name} (dim: {self._embedding_dim})"
                )
            else:
                logger.warning("sentence-transformers not available - embeddings disabled")
                self._embedding_model = None
                self._embedding_dim = None

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self._embedding_model = None
            self._embedding_dim = None

    def _create_tables(self) -> None:
        """Create database schema for conversations, messages, and embeddings."""
        cursor = self._db.cursor()

        try:
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)

            # Vector embeddings table (if sqlite-vec available)
            if self._vector_enabled and self._embedding_dim:
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings 
                    USING vec0(
                        embedding float[{self._embedding_dim}]
                    )
                """)

                # Regular table for embedding metadata
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_metadata (
                        rowid INTEGER PRIMARY KEY,
                        message_id TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    )
                """)

            # Create indexes for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)"
            )

            # Commit schema changes
            self._db.commit()
            logger.info("Database schema created successfully")

        except Exception as e:
            self._db.rollback()
            raise MemoryStorageError(
                message=f"Failed to create database schema: {e}", operation="create_schema"
            )
        finally:
            cursor.close()

    def _verify_schema(self) -> None:
        """Verify that database schema is correct and up-to-date."""
        cursor = self._db.cursor()

        try:
            # Check if required tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('conversations', 'messages')
            """)
            required_tables = [row[0] for row in cursor.fetchall()]

            if len(required_tables) != 2:
                raise MemoryStorageError(
                    message="Required tables missing from database", operation="verify_schema"
                )

            # Check vector table if vector search is enabled
            if self._vector_enabled:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='message_embeddings'
                """)
                vector_tables = [row[0] for row in cursor.fetchall()]

                if not vector_tables:
                    logger.warning("Vector table not found - vector search disabled")
                    self._vector_enabled = False

            logger.info("Database schema verification passed")

        except Exception as e:
            raise MemoryStorageError(
                message=f"Schema verification failed: {e}", operation="verify_schema"
            )
        finally:
            cursor.close()

    def store_conversation(
        self,
        conversation_id: str,
        title: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store a complete conversation with all messages.

        Args:
            conversation_id: Unique identifier for the conversation
            title: Human-readable title for the conversation
            messages: List of messages with 'role', 'content', and optional 'timestamp'
            metadata: Additional metadata to store with conversation

        Returns:
            True if stored successfully

        Raises:
            MemoryStorageError: If storage operation fails
        """
        if self._db is None:
            raise DatabaseConnectionError(db_path=str(self.db_path))

        cursor = self._db.cursor()
        now = datetime.now().isoformat()

        try:
            # Insert conversation
            cursor.execute(
                """
                INSERT OR REPLACE INTO conversations 
                (id, title, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                [conversation_id, title, now, now, json.dumps(metadata or {})],
            )

            # Insert messages
            for i, message in enumerate(messages):
                message_id = f"{conversation_id}_{i}"
                role = message.get("role", "user")
                content = message.get("content", "")
                timestamp = message.get("timestamp", now)

                # Basic validation
                if role not in ["user", "assistant", "system"]:
                    role = "user"

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO messages 
                    (id, conversation_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    [message_id, conversation_id, role, content, timestamp],
                )

                # Generate and store embedding if available
                if self._embedding_model and self._vector_enabled:
                    try:
                        embedding = self._embedding_model.encode(content)

                        # Store embedding in vector table
                        cursor.execute(
                            """
                            INSERT INTO message_embeddings (rowid, embedding)
                            VALUES (?, ?)
                        """,
                            [len(content), embedding.tolist()],
                        )

                        # Store embedding metadata
                        vector_rowid = cursor.lastrowid
                        cursor.execute(
                            """
                            INSERT INTO embedding_metadata 
                            (rowid, message_id, conversation_id, created_at)
                            VALUES (?, ?, ?, ?)
                        """,
                            [vector_rowid, message_id, conversation_id, now],
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to generate embedding for message {message_id}: {e}"
                        )
                        # Continue without embedding - don't fail the whole operation

            self._db.commit()
            logger.info(f"Stored conversation '{conversation_id}' with {len(messages)} messages")
            return True

        except Exception as e:
            self._db.rollback()
            raise MemoryStorageError(
                message=f"Failed to store conversation: {e}",
                operation="store_conversation",
                conversation_id=conversation_id,
            )
        finally:
            cursor.close()

    def retrieve_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete conversation by ID.

        Args:
            conversation_id: ID of conversation to retrieve

        Returns:
            Dictionary with conversation data or None if not found

        Raises:
            MemoryStorageError: If retrieval operation fails
        """
        if self._db is None:
            raise DatabaseConnectionError(db_path=str(self.db_path))

        cursor = self._db.cursor()

        try:
            # Get conversation info
            cursor.execute(
                """
                SELECT id, title, created_at, updated_at, metadata
                FROM conversations 
                WHERE id = ?
            """,
                [conversation_id],
            )

            conversation_row = cursor.fetchone()
            if not conversation_row:
                return None

            # Get messages
            cursor.execute(
                """
                SELECT id, role, content, timestamp, token_count
                FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp
            """,
                [conversation_id],
            )

            message_rows = cursor.fetchall()

            # Build result
            conversation = {
                "id": conversation_row["id"],
                "title": conversation_row["title"],
                "created_at": conversation_row["created_at"],
                "updated_at": conversation_row["updated_at"],
                "metadata": json.loads(conversation_row["metadata"]),
                "messages": [
                    {
                        "id": msg["id"],
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"],
                        "token_count": msg["token_count"],
                    }
                    for msg in message_rows
                ],
            }

            logger.debug(
                f"Retrieved conversation '{conversation_id}' with {len(message_rows)} messages"
            )
            return conversation

        except Exception as e:
            raise MemoryStorageError(
                message=f"Failed to retrieve conversation: {e}",
                operation="retrieve_conversation",
                conversation_id=conversation_id,
            )
        finally:
            cursor.close()

    def search_conversations(
        self, query: str, limit: int = 5, include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search conversations using semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            include_content: Whether to include full message content in results

        Returns:
            List of matching conversations with similarity scores

        Raises:
            VectorSearchError: If search operation fails
        """
        if not self._vector_enabled or self._embedding_model is None:
            logger.warning("Vector search not available - falling back to text search")
            return self._text_search_fallback(query, limit, include_content)

        if self._db is None:
            raise DatabaseConnectionError(db_path=str(self.db_path))

        cursor = self._db.cursor()

        try:
            # For now, use text search as vector search needs sqlite-vec syntax fixes
            logger.info("Using text search fallback temporarily")
            return self._text_search_fallback(query, limit, include_content)

            # TODO: Fix sqlite-vec query syntax for proper vector search
            # Generate query embedding
            # query_embedding = self._embedding_model.encode(query)
            #
            # # Perform vector similarity search using sqlite-vec syntax
            # cursor.execute(
            #     """
            #         SELECT
            #             em.conversation_id,
            #             em.message_id,
            #             em.created_at,
            #             m.role,
            #             m.content,
            #             c.title,
            #             vec_distance_l2(e.embedding, ?) as distance
            #         FROM message_embeddings e
            #         JOIN embedding_metadata em ON e.rowid = em.rowid
            #         JOIN messages m ON em.message_id = m.id
            #         JOIN conversations c ON em.conversation_id = c.id
            #         WHERE e.embedding MATCH ?
            #         ORDER BY distance
            #         LIMIT ?
            #     """,
            #     [query_embedding.tolist(), query_embedding.tolist(), limit],
            # )

            results = []
            seen_conversations = set()

            for row in cursor.fetchall():
                conv_id = row["conversation_id"]
                if conv_id not in seen_conversations:
                    conversation = {
                        "conversation_id": conv_id,
                        "title": row["title"],
                        "similarity_score": 1.0 - row["distance"],  # Convert distance to similarity
                        "matched_message": {
                            "role": row["role"],
                            "content": row["content"]
                            if include_content
                            else row["content"][:200] + "..."
                            if len(row["content"]) > 200
                            else row["content"],
                            "timestamp": row["created_at"],
                        },
                    }
                    results.append(conversation)
                    seen_conversations.add(conv_id)

            logger.debug(f"Vector search found {len(results)} conversations for query: '{query}'")
            return results

        except Exception as e:
            raise VectorSearchError(query=query, error_details=str(e))
        finally:
            cursor.close()

    def _text_search_fallback(
        self, query: str, limit: int, include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fallback text search when vector search is unavailable.

        Args:
            query: Search query text
            limit: Maximum number of results
            include_content: Whether to include full message content

        Returns:
            List of matching conversations
        """
        cursor = self._db.cursor()

        try:
            # Simple text search in message content
            cursor.execute(
                """
                SELECT DISTINCT
                    c.id as conversation_id,
                    c.title,
                    m.role,
                    m.content,
                    m.timestamp
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                WHERE m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """,
                [f"%{query}%", limit],
            )

            results = []
            seen_conversations = set()

            for row in cursor.fetchall():
                conv_id = row["conversation_id"]
                if conv_id not in seen_conversations:
                    conversation = {
                        "conversation_id": conv_id,
                        "title": row["title"],
                        "similarity_score": 0.5,  # Default score for text search
                        "matched_message": {
                            "role": row["role"],
                            "content": row["content"]
                            if include_content
                            else row["content"][:200] + "..."
                            if len(row["content"]) > 200
                            else row["content"],
                            "timestamp": row["timestamp"],
                        },
                    }
                    results.append(conversation)
                    seen_conversations.add(conv_id)

            logger.debug(
                f"Text search fallback found {len(results)} conversations for query: '{query}'"
            )
            return results

        except Exception as e:
            logger.error(f"Text search fallback failed: {e}")
            return []
        finally:
            cursor.close()

    def get_conversation_list(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get a list of all conversations with basic info.

        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip

        Returns:
            List of conversation summaries

        Raises:
            MemoryStorageError: If operation fails
        """
        if self._db is None:
            raise DatabaseConnectionError(db_path=str(self.db_path))

        cursor = self._db.cursor()

        try:
            cursor.execute(
                """
                SELECT 
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    c.metadata,
                    COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
            """,
                [limit, offset],
            )

            conversations = []
            for row in cursor.fetchall():
                conversation = {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "metadata": json.loads(row["metadata"]),
                    "message_count": row["message_count"],
                }
                conversations.append(conversation)

            return conversations

        except Exception as e:
            raise MemoryStorageError(
                message=f"Failed to get conversation list: {e}", operation="get_conversation_list"
            )
        finally:
            cursor.close()

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            True if deleted successfully

        Raises:
            MemoryStorageError: If deletion fails
        """
        if self._db is None:
            raise DatabaseConnectionError(db_path=str(self.db_path))

        cursor = self._db.cursor()

        try:
            # Delete conversation (cascade will delete messages and embeddings)
            cursor.execute(
                """
                DELETE FROM conversations WHERE id = ?
            """,
                [conversation_id],
            )

            self._db.commit()
            deleted_count = cursor.rowcount

            if deleted_count > 0:
                logger.info(f"Deleted conversation '{conversation_id}'")
                return True
            else:
                logger.warning(f"Conversation '{conversation_id}' not found for deletion")
                return False

        except Exception as e:
            self._db.rollback()
            raise MemoryStorageError(
                message=f"Failed to delete conversation: {e}",
                operation="delete_conversation",
                conversation_id=conversation_id,
            )
        finally:
            cursor.close()

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics and health information.

        Returns:
            Dictionary with storage statistics

        Raises:
            MemoryStorageError: If operation fails
        """
        if self._db is None:
            raise DatabaseConnectionError(db_path=str(self.db_path))

        cursor = self._db.cursor()

        try:
            stats = {}

            # Count conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            stats["conversation_count"] = cursor.fetchone()["count"]

            # Count messages
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            stats["message_count"] = cursor.fetchone()["count"]

            # Database file size
            if self.db_path.exists():
                stats["database_size_bytes"] = self.db_path.stat().st_size
                stats["database_size_mb"] = stats["database_size_bytes"] / (1024 * 1024)
            else:
                stats["database_size_bytes"] = 0
                stats["database_size_mb"] = 0

            # Vector search capability
            stats["vector_search_enabled"] = self._vector_enabled
            stats["embedding_model"] = self.embedding_model_name
            stats["embedding_dim"] = self._embedding_dim

            # Database path
            stats["database_path"] = str(self.db_path)

            return stats

        except Exception as e:
            raise MemoryStorageError(
                message=f"Failed to get storage stats: {e}", operation="get_storage_stats"
            )
        finally:
            cursor.close()

    def close(self) -> None:
        """Close database connection and cleanup resources."""
        if self._db:
            self._db.close()
            self._db = None
            logger.info("MemoryStorage database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

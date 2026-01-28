"""
SQLite database manager for conversation memory storage.

This module provides SQLite database operations and schema management
for storing conversations, messages, and associated metadata.
"""

import sqlite3
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import logging

# Import from existing models module
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from models.conversation import Message, MessageRole, ConversationMetadata


class SQLiteManager:
    """
    SQLite database manager with connection pooling and thread safety.

    Manages conversations, messages, and metadata with proper indexing
    and migration support for persistent storage.
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLite manager with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection.

        Returns:
            SQLite connection for current thread
        """
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            # Enable foreign key constraints
            self._local.connection.execute("PRAGMA foreign_keys=ON")
            # Optimize for performance
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
        return self._local.connection

    def _initialize_database(self) -> None:
        """
        Initialize database schema with all required tables.

        Creates conversations, messages, and metadata tables with proper
        indexing and relationships for efficient querying.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

            # Create conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    session_id TEXT,
                    total_messages INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    context_window_size INTEGER DEFAULT 4096,
                    model_history TEXT DEFAULT '[]'
                )
            """)

            # Create messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool_call', 'tool_result')),
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.5 CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
                    metadata TEXT DEFAULT '{}',
                    embedding_id TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for efficient querying
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)"
            )

            # Create metadata table for application state
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert initial schema version
            conn.execute("""
                INSERT OR IGNORE INTO app_metadata (key, value) 
                VALUES ('schema_version', '1.0.0')
            """)

            conn.commit()
            self.logger.info(f"Database initialized: {self.db_path}")

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()

    def create_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create a new conversation.

        Args:
            conversation_id: Unique conversation identifier
            title: Optional conversation title
            session_id: Optional session identifier
            metadata: Optional metadata dictionary
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO conversations 
                (id, title, session_id, metadata)
                VALUES (?, ?, ?, ?)
            """,
                (
                    conversation_id,
                    title or conversation_id,
                    session_id or conversation_id,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            self.logger.debug(f"Created conversation: {conversation_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to create conversation {conversation_id}: {e}")
            raise

    def add_message(
        self,
        message_id: str,
        conversation_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        importance_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_id: Optional[str] = None,
    ) -> None:
        """
        Add a message to a conversation.

        Args:
            message_id: Unique message identifier
            conversation_id: Target conversation ID
            role: Message role (user/assistant/system/tool_call/tool_result)
            content: Message content
            token_count: Estimated token count
            importance_score: Importance score 0.0-1.0
            metadata: Optional message metadata
            embedding_id: Optional embedding reference
        """
        conn = self._get_connection()
        try:
            # Add message
            conn.execute(
                """
                INSERT INTO messages 
                (id, conversation_id, role, content, token_count, importance_score, metadata, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    token_count,
                    importance_score,
                    json.dumps(metadata or {}),
                    embedding_id,
                ),
            )

            # Update conversation stats
            conn.execute(
                """
                UPDATE conversations 
                SET 
                    total_messages = total_messages + 1,
                    total_tokens = total_tokens + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (token_count, conversation_id),
            )

            conn.commit()
            self.logger.debug(
                f"Added message {message_id} to conversation {conversation_id}"
            )
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to add message {message_id}: {e}")
            raise

    def get_conversation(
        self, conversation_id: str, include_messages: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation details.

        Args:
            conversation_id: Conversation ID to retrieve
            include_messages: Whether to include messages

        Returns:
            Conversation data or None if not found
        """
        conn = self._get_connection()
        try:
            # Get conversation info
            cursor = conn.execute(
                """
                SELECT * FROM conversations WHERE id = ?
            """,
                (conversation_id,),
            )
            conversation = cursor.fetchone()

            if not conversation:
                return None

            result = {
                "id": conversation["id"],
                "title": conversation["title"],
                "created_at": conversation["created_at"],
                "updated_at": conversation["updated_at"],
                "metadata": json.loads(conversation["metadata"]),
                "session_id": conversation["session_id"],
                "total_messages": conversation["total_messages"],
                "total_tokens": conversation["total_tokens"],
                "context_window_size": conversation["context_window_size"],
                "model_history": json.loads(conversation["model_history"]),
            }

            if include_messages:
                cursor = conn.execute(
                    """
                    SELECT * FROM messages 
                    WHERE conversation_id = ? 
                    ORDER BY timestamp ASC
                """,
                    (conversation_id,),
                )
                messages = []
                for row in cursor:
                    messages.append(
                        {
                            "id": row["id"],
                            "conversation_id": row["conversation_id"],
                            "role": row["role"],
                            "content": row["content"],
                            "timestamp": row["timestamp"],
                            "token_count": row["token_count"],
                            "importance_score": row["importance_score"],
                            "metadata": json.loads(row["metadata"]),
                            "embedding_id": row["embedding_id"],
                        }
                    )
                result["messages"] = messages

            return result
        except Exception as e:
            self.logger.error(f"Failed to get conversation {conversation_id}: {e}")
            raise

    def get_recent_conversations(
        self, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversations.

        Args:
            limit: Maximum number of conversations to return
            offset: Offset for pagination

        Returns:
            List of conversation summaries
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT 
                    id, title, created_at, updated_at, 
                    total_messages, total_tokens, session_id
                FROM conversations 
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

            conversations = []
            for row in cursor:
                conversations.append(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "total_messages": row["total_messages"],
                        "total_tokens": row["total_tokens"],
                        "session_id": row["session_id"],
                    }
                )

            return conversations
        except Exception as e:
            self.logger.error(f"Failed to get recent conversations: {e}")
            raise

    def get_messages_by_role(
        self, conversation_id: str, role: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation filtered by role.

        Args:
            conversation_id: Conversation ID
            role: Message role filter
            limit: Optional message limit

        Returns:
            List of messages
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT * FROM messages 
                WHERE conversation_id = ? AND role = ?
                ORDER BY timestamp ASC
            """
            params = [conversation_id, role]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = conn.execute(query, params)
            messages = []
            for row in cursor:
                messages.append(
                    {
                        "id": row["id"],
                        "conversation_id": row["conversation_id"],
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "token_count": row["token_count"],
                        "importance_score": row["importance_score"],
                        "metadata": json.loads(row["metadata"]),
                        "embedding_id": row["embedding_id"],
                    }
                )

            return messages
        except Exception as e:
            self.logger.error(f"Failed to get messages by role {role}: {e}")
            raise

    def update_conversation_metadata(
        self, conversation_id: str, metadata: Dict[str, Any]
    ) -> None:
        """
        Update conversation metadata.

        Args:
            conversation_id: Conversation ID
            metadata: New metadata dictionary
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                UPDATE conversations 
                SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (json.dumps(metadata), conversation_id),
            )
            conn.commit()
            self.logger.debug(f"Updated metadata for conversation {conversation_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to update conversation metadata: {e}")
            raise

    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation and all its messages.

        Args:
            conversation_id: Conversation ID to delete
        """
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            self.logger.info(f"Deleted conversation {conversation_id}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            raise

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        conn = self._get_connection()
        try:
            stats = {}

            # Conversation stats
            cursor = conn.execute("SELECT COUNT(*) as count FROM conversations")
            stats["total_conversations"] = cursor.fetchone()["count"]

            # Message stats
            cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
            stats["total_messages"] = cursor.fetchone()["count"]

            cursor = conn.execute("SELECT SUM(token_count) as total FROM messages")
            result = cursor.fetchone()
            stats["total_tokens"] = result["total"] or 0

            # Database size
            cursor = conn.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            result = cursor.fetchone()
            stats["database_size_bytes"] = result["size"] if result else 0

            return stats
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            raise

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            delattr(self._local, "connection")
        self.logger.info("SQLite manager closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

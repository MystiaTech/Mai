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
            conn.close()
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

        # Check if tables exist before using them
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        )
        if not cursor.fetchone():
            conn.rollback()
            conn.close()
            raise RuntimeError(
                "Database tables not initialized. Call initialize() first."
            )
        cursor.close()
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

            cursor = conn.execute(query, tuple(params))
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

    def get_recent_messages(
        self, conversation_id: str, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages from a conversation.

        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return
            offset: Offset for pagination

        Returns:
            List of messages ordered by timestamp (newest first)
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """

            cursor = conn.execute(query, (conversation_id, limit, offset))
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
            self.logger.error(f"Failed to get recent messages: {e}")
            raise

    def get_conversation_metadata(
        self, conversation_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive metadata for specified conversations.

        Args:
            conversation_ids: List of conversation IDs to retrieve metadata for

        Returns:
            Dictionary mapping conversation_id to comprehensive metadata
        """
        conn = self._get_connection()
        try:
            metadata = {}

            # Create placeholders for IN clause
            placeholders = ",".join(["?" for _ in conversation_ids])

            # Get basic conversation metadata
            cursor = conn.execute(
                f"""
                SELECT 
                    id, title, created_at, updated_at, metadata,
                    session_id, total_messages, total_tokens, context_window_size,
                    model_history
                FROM conversations 
                WHERE id IN ({placeholders})
                ORDER BY updated_at DESC
                """,
                conversation_ids,
            )

            conversations_data = cursor.fetchall()

            for conv in conversations_data:
                conv_id = conv["id"]

                # Parse JSON metadata fields
                try:
                    conv_metadata = (
                        json.loads(conv["metadata"]) if conv["metadata"] else {}
                    )
                    model_history = (
                        json.loads(conv["model_history"])
                        if conv["model_history"]
                        else []
                    )
                except json.JSONDecodeError:
                    conv_metadata = {}
                    model_history = []

                # Initialize metadata structure
                metadata[conv_id] = {
                    # Basic conversation metadata
                    "conversation_info": {
                        "id": conv_id,
                        "title": conv["title"],
                        "created_at": conv["created_at"],
                        "updated_at": conv["updated_at"],
                        "session_id": conv["session_id"],
                        "total_messages": conv["total_messages"],
                        "total_tokens": conv["total_tokens"],
                        "context_window_size": conv["context_window_size"],
                    },
                    # Topic information from metadata
                    "topic_info": {
                        "main_topics": conv_metadata.get("main_topics", []),
                        "topic_frequency": conv_metadata.get("topic_frequency", {}),
                        "topic_sentiment": conv_metadata.get("topic_sentiment", {}),
                        "primary_topic": conv_metadata.get("primary_topic", "general"),
                    },
                    # Conversation metadata
                    "metadata": conv_metadata,
                    # Model history
                    "model_history": model_history,
                }

            # Calculate engagement metrics for each conversation
            for conv_id in conversation_ids:
                if conv_id in metadata:
                    # Get message statistics
                    cursor = conn.execute(
                        """
                        SELECT 
                            role,
                            COUNT(*) as count,
                            AVG(importance_score) as avg_importance,
                            MIN(timestamp) as first_message,
                            MAX(timestamp) as last_message
                        FROM messages 
                        WHERE conversation_id = ?
                        GROUP BY role
                        """,
                        (conv_id,),
                    )

                    role_stats = cursor.fetchall()

                    # Calculate engagement metrics
                    total_user_messages = 0
                    total_assistant_messages = 0
                    total_importance = 0
                    message_count = 0
                    first_message_time = None
                    last_message_time = None

                    for stat in role_stats:
                        if stat["role"] == "user":
                            total_user_messages = stat["count"]
                        elif stat["role"] == "assistant":
                            total_assistant_messages = stat["count"]

                        total_importance += stat["avg_importance"] or 0
                        message_count += stat["count"]

                        if (
                            not first_message_time
                            or stat["first_message"] < first_message_time
                        ):
                            first_message_time = stat["first_message"]
                        if (
                            not last_message_time
                            or stat["last_message"] > last_message_time
                        ):
                            last_message_time = stat["last_message"]

                    # Calculate user message ratio
                    user_message_ratio = total_user_messages / max(1, message_count)

                    # Add engagement metrics
                    metadata[conv_id]["engagement_metrics"] = {
                        "message_count": message_count,
                        "user_message_count": total_user_messages,
                        "assistant_message_count": total_assistant_messages,
                        "user_message_ratio": user_message_ratio,
                        "avg_importance": total_importance / max(1, len(role_stats)),
                        "conversation_duration_seconds": (
                            (last_message_time - first_message_time).total_seconds()
                            if first_message_time and last_message_time
                            else 0
                        ),
                    }

                    # Calculate temporal patterns
                    if last_message_time:
                        cursor = conn.execute(
                            """
                            SELECT 
                                strftime('%H', timestamp) as hour,
                                strftime('%w', timestamp) as day_of_week,
                                COUNT(*) as count
                            FROM messages 
                            WHERE conversation_id = ?
                            GROUP BY hour, day_of_week
                            """,
                            (conv_id,),
                        )

                        temporal_data = cursor.fetchall()

                        # Analyze temporal patterns
                        hour_counts = {}
                        day_counts = {}
                        for row in temporal_data:
                            hour = row["hour"]
                            day = int(row["day_of_week"])
                            hour_counts[hour] = hour_counts.get(hour, 0) + row["count"]
                            day_counts[day] = day_counts.get(day, 0) + row["count"]

                        # Find most common hour and day
                        most_common_hour = (
                            max(hour_counts.items(), key=lambda x: x[1])[0]
                            if hour_counts
                            else None
                        )
                        most_common_day = (
                            max(day_counts.items(), key=lambda x: x[1])[0]
                            if day_counts
                            else None
                        )

                        metadata[conv_id]["temporal_patterns"] = {
                            "most_common_hour": int(most_common_hour)
                            if most_common_hour
                            else None,
                            "most_common_day": most_common_day,
                            "hour_distribution": hour_counts,
                            "day_distribution": day_counts,
                            "last_activity": last_message_time,
                        }
                    else:
                        metadata[conv_id]["temporal_patterns"] = {
                            "most_common_hour": None,
                            "most_common_day": None,
                            "hour_distribution": {},
                            "day_distribution": {},
                            "last_activity": None,
                        }

                    # Get related conversations (same session or similar topics)
                    if metadata[conv_id]["conversation_info"]["session_id"]:
                        cursor = conn.execute(
                            """
                            SELECT id, title, updated_at
                            FROM conversations 
                            WHERE session_id = ? AND id != ?
                            ORDER BY updated_at DESC
                            LIMIT 5
                            """,
                            (
                                metadata[conv_id]["conversation_info"]["session_id"],
                                conv_id,
                            ),
                        )

                        related = cursor.fetchall()
                        metadata[conv_id]["context_clues"] = {
                            "related_conversations": [
                                {
                                    "id": r["id"],
                                    "title": r["title"],
                                    "updated_at": r["updated_at"],
                                    "relationship": "same_session",
                                }
                                for r in related
                            ]
                        }
                    else:
                        metadata[conv_id]["context_clues"] = {
                            "related_conversations": []
                        }

            return metadata

        except Exception as e:
            self.logger.error(f"Failed to get conversation metadata: {e}")
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

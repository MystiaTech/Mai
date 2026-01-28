"""
Memory module for Mai conversation management.

This module provides persistent storage and retrieval of conversations,
messages, and associated vector embeddings for semantic search capabilities.
"""

from .storage.sqlite_manager import SQLiteManager
# from .storage.vector_store import VectorStore  # Will be added in Task 2

from typing import Optional
import logging


class MemoryManager:
    """
    Main interface for memory operations in Mai.

    Provides unified access to conversation storage and vector search
    capabilities through SQLite with sqlite-vec extension.
    """

    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize memory manager with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._sqlite_manager: Optional[SQLiteManager] = None
        # self._vector_store: Optional[VectorStore] = None  # Will be added in Task 2
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """
        Initialize storage components.

        Creates database schema and vector tables if they don't exist.
        """
        try:
            self._sqlite_manager = SQLiteManager(self.db_path)
            # self._vector_store = VectorStore(self._sqlite_manager)  # Will be added in Task 2
            self.logger.info(
                f"Memory manager initialized with database: {self.db_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {e}")
            raise

    @property
    def sqlite_manager(self) -> SQLiteManager:
        """Get SQLite manager instance."""
        if self._sqlite_manager is None:
            raise RuntimeError(
                "Memory manager not initialized. Call initialize() first."
            )
        return self._sqlite_manager

    # @property
    # def vector_store(self) -> VectorStore:
    #     """Get vector store instance."""
    #     if self._vector_store is None:
    #         raise RuntimeError("Memory manager not initialized. Call initialize() first.")
    #     return self._vector_store

    def close(self) -> None:
        """Close database connections."""
        if self._sqlite_manager:
            self._sqlite_manager.close()
        self.logger.info("Memory manager closed")

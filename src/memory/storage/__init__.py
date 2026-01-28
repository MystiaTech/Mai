"""
Storage module for memory operations.

Provides SQLite database management and vector storage capabilities
for conversation persistence and semantic search.
"""

from .sqlite_manager import SQLiteManager
# from .vector_store import VectorStore  # Will be added in Task 2

__all__ = ["SQLiteManager", "VectorStore"]

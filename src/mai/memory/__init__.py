"""
Mai Memory Module

Provides persistent storage and retrieval of conversations
with semantic search capabilities.

This module serves as the foundation for Mai's memory system,
enabling conversation retention and intelligent context retrieval.
"""

# Version information
__version__ = "0.1.0"
__author__ = "Mai Team"

# Core exports
from .storage import MemoryStorage

# Optional exports (may not be available if dependencies missing)
try:
    from .storage import (
        MemoryStorageError,
        VectorSearchError,
        DatabaseConnectionError,
    )

    __all__ = [
        "MemoryStorage",
        "MemoryStorageError",
        "VectorSearchError",
        "DatabaseConnectionError",
    ]
except ImportError:
    __all__ = ["MemoryStorage"]

# Module metadata
__module_info__ = {
    "name": "Mai Memory Module",
    "description": "Persistent memory storage with semantic search",
    "version": __version__,
    "features": {
        "sqlite_storage": True,
        "vector_search": "sqlite-vec" in globals(),
        "embeddings": "sentence-transformers" in globals(),
        "fallback_search": True,
    },
    "dependencies": {
        "required": ["sqlite3"],
        "optional": {
            "sqlite-vec": "Vector similarity search",
            "sentence-transformers": "Text embeddings",
        },
    },
}


def get_module_info():
    """Get module information and capabilities."""
    return __module_info__


def is_vector_search_available() -> bool:
    """Check if vector search is available."""
    try:
        import sqlite_vec
        from sentence_transformers import SentenceTransformer

        return True
    except ImportError:
        return False


def is_embeddings_available() -> bool:
    """Check if text embeddings are available."""
    try:
        from sentence_transformers import SentenceTransformer

        return True
    except ImportError:
        return False


def get_memory_storage(*args, **kwargs):
    """
    Factory function to create MemoryStorage instances.

    Args:
        *args: Positional arguments to pass to MemoryStorage
        **kwargs: Keyword arguments to pass to MemoryStorage

    Returns:
        Configured MemoryStorage instance
    """
    from .storage import MemoryStorage

    return MemoryStorage(*args, **kwargs)
